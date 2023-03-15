from __future__ import annotations
import pitchtools as pt

import sndfileio
import numpy as np
import os

from maelzel.scorestruct import ScoreStruct
from maelzel.common import F, asF, F0, asmidi
from maelzel.core.config import CoreConfig
from maelzel.core.mobj import MContainer
from maelzel.core.event import MEvent
from maelzel.core._typedefs import *
from maelzel.core import _dialogs
from maelzel.core.synthevent import SynthEvent, PlayArgs
from maelzel.core.workspace import Workspace
from maelzel.core import playback
from maelzel.snd import audiosample
from maelzel.core import _util

from maelzel import scoring
from emlib.misc import firstval

import csoundengine


__all__ = (
    'Clip',
)


class Clip(MEvent):
    """
    A Clip represent an audio sample in time

    It is possible to select a fragment from the source

    Args:
        source: the source of the clip (a filename, audiosample, samples as numpy array)
        pitch: the pitch representation of this clip. It has no influence in the playback
            itself, it is only for notation purposes
        dur: the totalDuration of the clip. If not given, the totalDuration of the
            source is used
        offset: the time offset of this clip. Like in a Note, if not given,
            the start time depends on the context (previous events) where this
            clip is evaluated
        label: a label str to identify this clip
        dynamic: allows to attach a dynamic expression to this Clip. This is
            only for notation purposes, it does not modify playback
        startsecs: selection start time (in seconds)
        endsecs: selection end time (in seconds). If 0., play until the end of the source
        speed: playback speed of the clip

    """
    _isDurationRelative = False
    _excludedPlayKeys: tuple[str] = ('instr', 'args')

    __slots__ = ('startSecs', 'endSecs', '_speed', 'source', 'soundfile', 'numChannels',
                 'sr', 'amp', 'dynamic', 'pitch', 'sourceDurSecs')

    def __init__(self,
                 source: str | audiosample.Sample | tuple[np.ndarray, int],
                 pitch: pitch_t = None,
                 amp: float = 1.,
                 offset: time_t = None,
                 label: str = '',
                 dynamic: str = '',
                 startsecs: float | F = 0.,
                 endsecs: float | F = 0.,
                 channel: int = None,
                 speed: F | float =F(1),
                 parent: MContainer | None = None,
                 loop=False,
                 tied=False,
                 noteheadShape: str = None
                 ):
        if source == '?':
            source = _dialogs.selectSndfileForOpen()
            if not source:
                raise ValueError("No source selected")

        self.soundfile = ''
        """The soundfile holding the audio data (if any)"""

        self.numChannels = 0
        """The number of channels of this Clip"""

        self.sr = 0
        """The samplerate of this Clip"""

        self.sourceDurSecs: F = F0
        """The totalDuration in seconds of the source of this Clip"""

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

        if isinstance(source, tuple) and len(source) == 2 and isinstance(source[0], np.ndarray):
            data, sr = source
            assert isinstance(data, np.ndarray)
            source = audiosample.Sample(data, sr=sr)

        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Soundfile not found: '{source}'")
            self.soundfile = source
            info = sndfileio.sndinfo(source)
            self.sr = info.samplerate
            self.sourceDurSecs = info.duration
            self.numChannels = info.channels if self.channel is None else 1
            self._playbackMethod = 'diskin'

        elif isinstance(source, audiosample.Sample):
            self.sr = source.sr
            self.sourceDurSecs = source.duration
            self.numChannels = source.numchannels
            self._playbackMethod = 'table'

        else:
            raise TypeError(f"Expected a soundfile path, a Sample or a tuple (samples, sr), got {source}")

        self.source: str |  audiosample.Sample = source
        """The source of this clip"""

        self.amp = amp
        """An amplitude (gain) applied to this clip"""

        self.dynamic: str = dynamic
        """A dynamic expression attached to the score representation of this clip"""

        self.startSecs: F = asF(startsecs)
        """Start offset of the clip relative to the start of the source (in seconds)"""

        self.endSecs: F = asF(endsecs if endsecs > 0 else self.sourceDurSecs)

        self._speed: F = asF(speed)
        """Playback speed"""

        self._sample: audiosample.Sample|None = None
        self._durContext: tuple[ScoreStruct, F] | None = None

        if pitch is None:
            s = self.asSample()
            freq = s.fundamentalFreq()
            if freq:
                pitch = pt.f2n(freq)
            else:
                pitch = 60
        else:
            pitch = asmidi(pitch)

        self.pitch: float = pitch
        """The pitch representation of this clip.
        This is used for notation purposes, it has no influence on the playback
        of this clip"""

        super().__init__(offset=offset, dur=None, label=label, parent=parent)

    @property
    def speed(self) -> F:
        return self._speed

    @speed.setter
    def speed(self, speed: time_t):
        self._speed = asF(speed)

    @property
    def name(self) -> str:
        return f"Clip(source={self.source})"

    def copy(self) -> Clip:
        # We do not copy the parent attr
        out = Clip(source=self.source,
                   pitch=self.pitch,
                   amp=self.amp,
                   offset=self.offset,
                   label=self.label,
                   dynamic=self.dynamic,
                   startsecs=self.startSecs,
                   endsecs=self.endSecs,
                   channel=self.channel,
                   speed=self.speed,
                   loop=self.loop,
                   tied=self.tied)
        self._copyAttributesTo(out)
        return out

    def __hash__(self):
        source = self.source if isinstance(self.source, str) else id(self.source)
        parts = (source, self.startSecs, self.endSecs, self.speed, self.sr,
                 self.numChannels, self.sourceDurSecs, self.amp,
                 self.dynamic, self.noteheadShape)
        return hash(parts)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = slice.start if slice.start is None else self.startSecs
            end = slice.stop if slice.stop is None else self.endSecs
            assert slice.step is None
            return Clip(self.source, startsecs=start, endsecs=end,
                        speed=self.speed, amp=self.amp, dynamic=self.dynamic)
        else:
            raise ValueError("Only a slice of the form clip[start:end] is supported")

    def asSample(self) -> audiosample.Sample:
        """
        Return a :class:`maelzel.snd.audiosample.Sample` with the audio data of this Clip

        Returns:
            a Sample with the audio data of this Clip

        Example
        ~~~~~~~

        TODO
        """
        if self._sample is not None:
            return self._sample
        if isinstance(self.source, audiosample.Sample):
            return self.source
        else:
            sample = audiosample.Sample(self.source)
            self._sample = sample
            return sample

    def isRest(self) -> bool:
        return False

    def durSecs(self) -> F:
        assert isinstance(self.endSecs, F)
        assert isinstance(self.startSecs, F)
        assert isinstance(self.speed, F)
        return (self.endSecs - self.startSecs) / self.speed

    def _calculateDuration(self,
                           relativeOffset: F | None,
                           parentOffset: F | None,
                           force=False
                           ) -> F | None:
        struct = self.scorestruct() or Workspace.active.scorestruct
        # TODO: use _durContext to validate the cached totalDuration

        if not force and self._resolvedDur is not None:
            return self._resolvedDur

        offset = firstval(relativeOffset, self.offset, self._resolvedOffset, F0)
        startbeat = (parentOffset or F0) + offset
        starttime = struct.beatToTime(startbeat)
        dursecs = self.durSecs()
        endbeat = struct.timeToBeat(starttime + dursecs)
        dur = endbeat - startbeat
        self._resolvedDur = dur
        self._durContext = (struct, startbeat)
        assert isinstance(dur, F)
        return dur

    def resolveDur(self) -> F:
        if not self.parent:
            reloffset = self.offset
            parentoffset = F(0)
        else:
            reloffset = self.resolveOffset()
            parentoffset = self.parent.absoluteOffset()
        return self._calculateDuration(relativeOffset=reloffset, parentOffset=parentoffset, force=True)

    def __repr__(self):
        return f"Clip(source={self.source}, numChannels={self.numChannels}, sr={self.sr}, dur={self.dur}, resolvedDur={self.resolveDur()}, sourcedursecs={_util.showT(self.sourceDurSecs)}secs)"

    def _synthEvents(self,
                     playargs: PlayArgs,
                     parentOffset: F,
                     workspace: Workspace,
                     ) -> list[SynthEvent]:
        skip = self.startSecs / self.speed
        scorestruct = workspace.scorestruct
        reloffset = self.resolveOffset()
        offset = reloffset + parentOffset
        starttime = float(scorestruct.beatToTime(offset))
        endtime = float(starttime + self.durSecs())
        amp = firstval(self.amp, 1.0)
        bps = [[starttime, self.pitch, amp],
               [endtime, self.pitch, amp]]

        if self._playbackMethod == 'diskin':
            assert isinstance(self.source, str), f"The diskin playback method needs a path " \
                                                 f"as source, got {self.source}"
            assert os.path.exists(self.soundfile)
            args = {'ipath': self.soundfile,
                    'isndfilechan': -1 if self.channel is None else self.channel,
                    'kspeed': self.speed,
                    'iskip': skip}
            playargs = playargs.clone(instr='_clip_diskin', args=args)

        elif self._playbackMethod == 'table':
            args = {'isndtab': 0,  # The table number will be filled later
                    'kspeed': self.speed,
                    'istart': skip,
                    'ixfade': -1 if not self.loop else 0.1}
            playargs = playargs.clone(instr='_playtable', args=args)
        else:
            raise RuntimeError(f"Playback method {self._playbackMethod} not supported")
        event = SynthEvent(bps=bps, linkednext=self.tied, numchans=self.numChannels,
                           initfunc=self._initEvent,
                           **playargs.db)
        return [event]

    def _initEvent(self, event: SynthEvent, renderer: playback.Renderer):
        if self._playbackMethod == 'table':
            if not self._csoundTable:
                self._csoundTable = renderer.makeTable(self.source.samples, sr=self.sr)
            event.args['isndtab'] = self._csoundTable

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig = None,
                      parentOffset: F | None = None
                      ) -> list[scoring.Notation]:
        if not config:
            config = Workspace.active.config
        offset = self.absoluteOffset()
        dur = self.resolveDur()
        notation = scoring.makeNote(pitch=self.pitch,
                                    duration=dur,
                                    offset=offset,
                                    dynamic=self.dynamic,
                                    gliss=bool(self.gliss))
        if self.tied:
            notation.tiedNext = True

        if self.label:
            notation.addText(self._scoringAnnotation(config=config))

        shape = self.noteheadShape if self.noteheadShape is not None else config['show.clipNoteheadShape']
        if shape:
            notation.setNotehead(shape)
        if self.symbols:
            for symbol in self.symbols:
                symbol.applyTo(notation)
        return [notation]
