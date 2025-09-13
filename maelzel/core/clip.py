from __future__ import annotations

import os
import itertools
import pitchtools as pt
import numpy as np

from maelzel.common import F, asF, F0, F1, asmidi
from maelzel.core import event
from maelzel.core.synthevent import SynthEvent
from maelzel.core.workspace import Workspace
from maelzel import _util


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.common import time_t, pitch_t
    from typing_extensions import Self
    from maelzel.core.synthevent import PlayArgs
    from maelzel import scoring
    import csoundengine
    from maelzel.scorestruct import ScoreStruct
    from maelzel.core.config import CoreConfig
    from maelzel.core.renderer import Renderer
    from maelzel.snd import audiosample


__all__ = (
    'Clip',
)


class Clip(event.MEvent):
    """
    A Clip represent an audio sample in time

    It is possible to select a fragment from the source

    Args:
        source: the source of the clip (a filename, audiosample, samples as numpy array)
        dur: the duration of the clip, in quarternotes. If not given, the duration will
            be the duration of the source. If loop==True, then the duration **needs** to
            be given
        pitch: the pitch representation of this clip. It has no influence in the playback
            itself, it is only for notation purposes
        amp: the playback gain
        offset: the time offset of this clip. Like in a Note, if not given,
            the start time depends on the context (previous events) where this
            clip is evaluated
        startsecs: selection start time (in seconds)
        endsecs: selection end time (in seconds). If 0., play until the end of the source
        loop: if True, playback of this Clip should be looped
        label: a label str to identify this clip
        speed: playback speed of the clip
        dynamic: allows to attach a dynamic expression to this Clip. This is
            only for notation purposes, it does not modify playback
        tied: this clip should be tied to the next one. This is only valid if the clips
            share the same source (same soundfile or samples) and allows to automate
            parameters such as playback speed or amplitude.
        noteheadShape: the notehead shape to use for notation, one of 'normal', 'cross',
            'harmonic', 'triangle', 'xcircle', 'rhombus', 'square', 'rectangle', 'slash', 'cluster'.
            (see :ref:`config['show.clipNoteheadShape'] <config_show_clipnoteheadshape>`)

    """
    _excludedPlayKeys = ('instr', 'args')

    __slots__ = ('amp',
                 'selectionStartSecs',
                 'selectionEndSecs',
                 'source',
                 'soundfile',
                 'numChannels',
                 'channel',
                 'dynamic',
                 'pitch',
                 'sourceDurSecs',
                 'loop',
                 'noteheadShape',
                 '_sr',
                 '_speed',
                 '_playbackMethod',
                 '_engine',
                 '_csoundTable',
                 '_sample',
                 '_durContext',
                 '_explicitDur',
                 '_cache'
                 )

    def __init__(self,
                 source: str | audiosample.Sample | tuple[np.ndarray, int],
                 dur: time_t = None,
                 pitch: pitch_t = None,
                 amp: float = 1.,
                 offset: time_t = None,
                 startsecs: float | F = 0.,
                 endsecs: float | F = 0.,
                 channel: int | None = None,
                 loop=False,
                 speed: F | float = F1,
                 label='',
                 dynamic='',
                 tied=False,
                 noteheadShape=''
                 ):
        if source == '?':
            from maelzel.core import _dialogs
            selection = _dialogs.selectSndfileForOpen()
            if selection is None or not selection:
                raise ValueError("No source selected")
            source = selection

        if loop and dur is None:
            raise ValueError(f"The duration of a looping Clip needs to be given (source: {source})")

        self.soundfile = ''
        """The soundfile holding the audio data (if any)"""

        self.numChannels = 0
        """The number of channels of this Clip"""

        self._sr = 0
        """The samplerate of this Clip"""

        self.sourceDurSecs: F = F0
        """The duration in seconds of the source of this Clip"""

        self.tied = tied
        """Is this clip tied to the next?"""

        self.loop = loop
        """Should this clip loop?"""

        self.channel = channel
        """Which channel to load from soundfile. If None, all channels are loaded"""

        self._playbackMethod = 'diskin'
        """One of 'diskin' | 'table' """

        self._engine: csoundengine.Engine | None = None
        """Will be set to the playback engine when realtime has been initialized"""

        self._csoundTable: int = 0
        """Will be filled during event initialization"""

        self.noteheadShape = noteheadShape
        """The shape to use as notehead"""

        self._explicitDur: F | None = None if dur is None else asF(dur)

        self._cache = {}

        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Soundfile not found: '{source}'")
            self.soundfile = source
            import sndfileio
            info = sndfileio.sndinfo(source)
            self._sr = info.samplerate
            self.sourceDurSecs = F(info.duration)
            self.numChannels = info.channels if self.channel is None else 1
            self._playbackMethod = 'diskin'
        elif isinstance(source, tuple):
            assert len(source) == 2 and isinstance(source[0], np.ndarray)
            samples = source[0]
            self.source = samples
            self._sr = source[1]
            self.sourceDurSecs = F(len(samples) / self._sr)
            self.numChannels = 1 if len(samples.shape) == 1 else samples.shape[-1]
            self._playbackMethod = 'table'
        
        self.source: str | np.ndarray = source
        """The source of this clip"""

        self.amp: float = amp
        """An amplitude (gain) applied to this clip"""

        self.dynamic: str = dynamic
        """A dynamic expression attached to the score representation of this clip"""

        self.selectionStartSecs: F = asF(startsecs)
        """The start time of the selected fragment of the source media (in seconds)"""

        self.selectionEndSecs: F = asF(endsecs if endsecs > 0 else self.sourceDurSecs)
        """The end time of the selected fragment of the source media (in seconds)"""

        self._speed: F = asF(speed)
        """Playback speed"""

        self._sample: audiosample.Sample | None = None
        self._durContext: tuple[ScoreStruct, F] | None = None

        if pitch is None:
            s = self.asSample()
            freq = s.fundamentalFreq()
            if freq:
                pitchrepr = pt.f2m(freq)
            else:
                pitchrepr = 60.
        else:
            pitchrepr = asmidi(pitch)

        self.pitch: float = pitchrepr
        """The pitch representation of this clip.
        This is used for notation purposes, it has no influence on the playback
        of this clip"""

        if offset is not None:
            offset = asF(offset)

        super().__init__(offset=offset, dur=F0, label=label)
        self._calculateDuration()

    @property
    def sr(self) -> float:
        return self._sr

    @property
    def speed(self) -> F:
        """The playback speed"""
        return self._speed

    @speed.setter
    def speed(self, speed: time_t):
        self._speed = asF(speed)

    @property
    def name(self) -> str:
        return f"Clip(source={self.source})"

    def copy(self) -> Self:
        # We do not copy the parent attr
        source = self.source if isinstance(self.source, str) else (self.source, self._sr)
        out = self.__class__(source=source, 
                             pitch=self.pitch,
                             amp=self.amp,
                             offset=self.offset,
                             label=self.label,
                             dynamic=self.dynamic,
                             startsecs=self.selectionStartSecs,
                             endsecs=self.selectionEndSecs,
                             channel=self.channel,
                             speed=self.speed,
                             loop=self.loop,
                             tied=self.tied)
        self._copyAttributesTo(out)
        return out

    def __hash__(self):
        source = self.source if isinstance(self.source, str) else id(self.source)
        parts = (source, self.selectionStartSecs, self.selectionEndSecs, self.speed, self.sr,
                 self.numChannels, self.sourceDurSecs, self.amp,
                 self.dynamic, self.noteheadShape, self.pitch)
        return hash(parts)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = slice.start if slice.start is None else self.selectionStartSecs
            end = slice.stop if slice.stop is None else self.selectionEndSecs
            assert slice.step is None
            source = self.source if isinstance(self.source, str) else (self.source, self._sr)
            return Clip(source, startsecs=start, endsecs=end,
                        speed=self.speed, amp=self.amp, dynamic=self.dynamic)
        else:
            raise ValueError("Only a slice of the form clip[start:end] is supported")

    def asSample(self) -> audiosample.Sample:
        """
        Return a :class:`maelzel.snd.audiosample.Sample` with the audio data of this Clip

        Returns:
            a Sample with the audio data of this Clip. The returned Sample is read-only.

        Example
        ~~~~~~~

        TODO
        """
        if self._sample is not None:
            return self._sample

        from maelzel.snd import audiosample    
        if isinstance(self.source, np.ndarray):
            return audiosample.Sample(self.source, sr=int(self.sr), 
                                      start=float(self.selectionStartSecs), 
                                      end=float(self.selectionEndSecs), 
                                      readonly=True)
        else:
            assert isinstance(self.source, str)
            sample = audiosample.Sample(self.source,
                                        readonly=True,
                                        engine=self._engine,
                                        start=float(self.selectionStartSecs),
                                        end=float(self.selectionEndSecs))
            self._sample = sample
            return sample

    def isRest(self) -> bool:
        return False

    def durSecs(self) -> F:
        return (self.selectionEndSecs - self.selectionStartSecs) / self.speed

    def pitchRange(self) -> tuple[float, float]:
        return (self.pitch, self.pitch)

    @property
    def dur(self) -> F:
        "The duration of this Clip, in quarter notes"
        if self._explicitDur:
            return self._explicitDur

        absoffset = self.absOffset()
        struct = self.activeScorestruct()

        if self._dur and self._durContext is not None:
            cachedstruct, cachedbeat = self._durContext
            if struct is cachedstruct and cachedbeat == absoffset:
                return self._dur

        self._calculateDuration(absoffset=absoffset, struct=struct)
        return self._dur

    def _calculateDuration(self, absoffset: F|None = None, struct: ScoreStruct|None = None
                           ) -> None:
        if absoffset is None:
            absoffset = self.absOffset()
        if struct is None:
            struct = self.activeScorestruct()
        dur = struct.beatDelta(absoffset, absoffset + self.durSecs())
        self._dur = dur
        self._durContext = (struct, absoffset)

    def __repr__(self):
        return (f"Clip(source={self.source}, "
                f"numChannels={self.numChannels}, "
                f"sr={self.sr}, "
                f"dur={_util.showT(self.dur)}, "
                f"sourcedur={_util.showT(self.sourceDurSecs)}s)")

    def _synthEvents(self,
                     playargs: PlayArgs,
                     parentOffset: F,
                     workspace: Workspace,
                     ) -> list[SynthEvent]:
        skip = float(self.selectionStartSecs / self.speed)
        scorestruct = workspace.scorestruct
        reloffset = self.relOffset()
        offset = reloffset + parentOffset
        starttime = float(scorestruct.beatToTime(offset))
        endtime = float(scorestruct.beatToTime(offset + self.dur))
        amp = self.amp if self.amp is not None else 1.0
        bps = [(starttime, self.pitch, amp),
               (endtime, self.pitch, amp)]

        if self.playargs:
            playargs = playargs.updated(self.playargs)

        if self._playbackMethod == 'diskin':
            assert isinstance(self.source, str), f"The diskin playback method needs a path " \
                                                 f"as source, got {self.source}"
            assert os.path.exists(self.soundfile)
            args = {'ipath': self.soundfile,
                    'isndfilechan': -1 if self.channel is None else self.channel,
                    'kspeed': float(self.speed),
                    'iskip': skip,
                    'iwrap': 1 if self.loop else 0}
            playargs = playargs.clone(instr='_clip_diskin', args=args)

        elif self._playbackMethod == 'table':
            args = {'isndtab': 0,  # The table number will be filled later
                    'kspeed': float(self.speed),
                    'istart': skip,
                    'ixfade': -1 if not self.loop else 0.1}
            playargs = playargs.clone(instr='_playtable', args=args)
        else:
            raise RuntimeError(f"Playback method {self._playbackMethod} not supported")
        event = SynthEvent.fromPlayArgs(bps=bps,
                                        playargs=playargs,
                                        numchans=self.numChannels,
                                        initfunc=self._initEvent)

        if playargs.automations:
            event.addAutomationsFromPlayArgs(playargs, scorestruct=scorestruct)
        return [event]

    def _initEvent(self, event: SynthEvent, renderer: Renderer) -> None:
        if self._playbackMethod == 'table':
            if not self._csoundTable:
                if isinstance(self.source, np.ndarray):
                    self._csoundTable = renderer.makeTable(self.source, sr=int(self.sr)).tabnum
                else:
                    assert os.path.exists(self.soundfile)
                    self._csoundTable = renderer.readSoundfile(self.soundfile)
            event._ensureArgs()['isndtab'] = self._csoundTable

    def spectrum(self, resolution=50., windowsize=0., mindb=-90., hoptime=0., minfreq=0, maxfreq=0, start=0., end=0.):
        """
        Analyze this sample to construct a partial tracking spectrum

        Args:
            resolution: analysis resolution in Hz
            windowsize: the window size, in Hz. Normally higher than resolution, defaults
                to the same as resolution
            mindb: min. amplitude for any bin to be used for partial tracking
            hoptime: time offset between analysis windows. For an overlap of n, use
                1/(windowsize*n)
            minfreq: partials with an average freq. lower than this will be discarded
            maxfreq: partials with an average freq. higher than this will be discarded
            start: start time of the analysis, in seconds
            end: end time of the analysis, in seconds

        Returns:
            a :class:`maelzel.partialtracking.spectrum.Spectrum`
        """
        if not windowsize:
            windowsize = resolution
        if hoptime == 0.:
            hoptime = 1/(windowsize * 4)
        if maxfreq == 0:
            maxfreq = self.sr / 2
        if end == 0:
            end = float(self.sourceDurSecs)
        cachekey = (resolution, windowsize, hoptime, minfreq, maxfreq, start, end)
        if (subcache := self._cache.get('.spectrum')) is None:
            self._cache['.spectrum'] = subcache = {}
        elif spec := subcache.get(cachekey):
            return spec
        sample = self.asSample()
        if end == 0:
            end = sample.duration
        elif end < 0:
            end = sample.duration + end
        else:
            end = min(end, sample.duration)
        assert end > start
        if start > 0 or end < sample.duration:
            sample = sample[start:end]
        spec = sample.partialTrackingAnalysis(resolution=resolution, windowsize=windowsize, mindb=mindb, hoptime=hoptime)
        if maxfreq:
            partials = [p for p in spec.partials if minfreq <= p.meanfreq() < maxfreq]
            spec = spec.__class__(partials)
        if len(subcache) == 10:   # Evict oldest cached spectrum
            del subcache[next(iter(subcache.keys()))]
        subcache[cachekey] = spec
        return spec

    def chordsAt(self,
                 times: list[float],
                 resolution: float = 50,
                 windowsize=0.,
                 mindb=-90,
                 durs: list[time_t] | time_t | None = None,
                 maxcount=0,
                 ampfactor=1.,
                 maxfreq=2000,
                 minfreq=0.,
                 averageAmplitude=False
                 ) -> list[event.Chord | event.Note]:
        margin = 1/resolution * 8
        start = max(0., times[0] - margin)
        end = times[-1] + margin
        if durs is None:
            durations = [t1 - t0 for t0, t1 in itertools.pairwise(times)]
            durations.append(durations[-1])
        elif isinstance(durs, (list, tuple)):
            durations = durs
        else:
            durations = [durs] * len(times)
        spectrum = self.spectrum(resolution=resolution, mindb=mindb, windowsize=windowsize, start=start, end=end)
        chords = []
        minamp = pt.db2amp(mindb)
        minfreq = max(minfreq, resolution * 0.8)
        for i, time in enumerate(times):
            eventdur = durations[i]
            partials = spectrum.partialsBetween(start=time, end=time)
            # eventdur = dur or (times[i+1] - time if i < len(times) - 1 else F(1))
            if not partials:
                chords.append(event.Rest(dur=eventdur))
            else:
                bps = [partial.at(time) for partial in partials]
                bps = [bp for bp in bps if minfreq <= bp[0] < maxfreq and bp[1] > minamp]
                if maxcount > 0:
                    bps.sort(key=lambda bp: bp[1], reverse=True)
                    bps = bps[:maxcount]
                if not bps:
                    chords.append(event.Rest(dur=eventdur))
                else:
                    components = [event.Note(pt.f2m(bp[0]), amp=bp[1] * ampfactor, properties={'bandwidth': bp[3]})
                                  for bp in bps]
                    if averageAmplitude:
                        for note in components:
                            note.amp = 1.
                        chordamp = sum(bp[1] for bp in bps) / len(bps)
                    else:
                        chordamp = 1.
                    chord = event.Chord(components, dur=eventdur, amp=chordamp, properties={'time': time})
                    chords.append(chord)
        return chords

    def chordAt(self,
                time: float,
                resolution: float = 50,
                channel=0,
                mindb=-90,
                dur: time_t = None,
                maxcount=0,
                ampfactor=1.0,
                maxfreq=20000,
                minfreq=0
                ) -> event.Chord | None:
        """
        Analyze the spectrum at the given time and extract the most relevant partials

        A small fragment of the clip is analyzed and the most relevant sinusoidal
        components at the given time are extracted and returned in the form of a
        chord

        Args:
            time: the time to analyze. This is a time in seconds within the clip
            resolution: the resolution of the analysis, in Hz
            channel: which channel to analyze, for multichannel clips
            mindb: the min. amplitude (in dB) for a component to be included
            dur: the duration of the returned chord
            maxcount: the max. number of components in the chord
            ampfactor: an amplitude factor to apply to each component.
            maxfreq: the max. frequency for a component to be included
            minfreq: the min. frequency for a component to be included

        Returns:
            a Chord representing the spectrum of the clip at the given time, or None
            if no components were found following the given restraints. None if no
            components found which would satisfy the given conditions at the given time

        """
        sample = self.asSample()
        pairs = sample.spectrumAt(time, resolution=resolution, channel=channel,
                                  mindb=mindb, maxcount=maxcount, maxfreq=maxfreq,
                                  minfreq=minfreq)
        if not pairs:
            return None
        components = [event.Note(pt.f2m(freq), amp=amp * ampfactor)
                      for freq, amp in pairs]
        return event.Chord(components, dur=dur)

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig|None = None,
                      parentOffset: F|None = None
                      ) -> list[scoring.Notation]:
        if not config:
            config = Workspace.active.config
        offset = self.absOffset()
        dur = self.dur
        from maelzel import scoring
        notation = scoring.Notation.makeNote(pitch=self.pitch,
                                             duration=dur,
                                             offset=offset,
                                             dynamic=self.dynamic,
                                             gliss=bool(self.gliss))
        if self.tied:
            notation.tiedNext = True

        if self.label:
            notation.addText(self._scoringAnnotation(config=config))

        shape = self.noteheadShape if self.noteheadShape else config['show.clipNoteheadShape']
        if shape:
            notation.setNotehead(shape)
        if self.symbols:
            for symbol in self.symbols:
                symbol.applyToNotation(notation, parent=self)
        return [notation]
