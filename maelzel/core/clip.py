from __future__ import annotations
import pitchtools as pt

import sndfileio
import numpy as np

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
        dur: the duration of the clip. If not given, the duration of the
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

    __slots__ = ('startSecs', 'endSecs', 'speed', 'source', 'soundfile', 'numChannels',
                 'sr', 'amp', 'dynamic', 'pitch', 'sourceDurSecs')

    def __init__(self,
                 source: str | audiosample.Sample | tuple[np.ndarray, int],
                 dur: time_t = None,
                 pitch: pitch_t = None,
                 amp: float = 1.,
                 offset: time_t = None,
                 label: str = '',
                 dynamic: str = '',
                 startsecs=0.,
                 endsecs=0.,
                 channel: int = None,
                 speed: F | float =F(1),
                 parent: MContainer | None = None,
                 loop=False,
                 tied=False
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

        self.sourceDurSecs = 0
        """The duration in seconds of the source of this Clip"""

        self.tied = tied
        """Is this clip tied to the next?"""

        self.loop = loop
        """Should this clip loop?"""

        self._soundfileCsoundStringNum = 0
        self._soundfileCsoundTable = 0
        self._playbackMethod = 'diskin'

        self._engine = playback.playEngine()
        self._session = self._engine.session()

        if isinstance(source, tuple) and len(source) == 2 and isinstance(source[0], np.ndarray):
            data, sr = source
            assert isinstance(data, np.ndarray)
            source = audiosample.Sample(data, sr=sr)

        if isinstance(source, str):
            self.soundfile = source
            info = sndfileio.sndinfo(source)
            self.sr = info.samplerate
            self.sourceDurSecs = info.duration
            self.numChannels = info.channels
            self._soundfileCsoundStringNum = self._engine.strSet(self.soundfile)

        elif isinstance(source, audiosample.Sample):
            self.sr = source.sr
            self.sourceDurSecs = source.duration
            self.numChannels = source.numchannels
            self._soundfileCsoundTable = self._session.makeTable(data=source.samples,
                                                                 sr=source.sr,
                                                                 unique=False)
            self._playbackMethod = 'table'

        else:
            raise TypeError(f"Expected a soundfile path, a Sample or a tuple (samples, sr), got {source}")

        self.source: str |  audiosample.Sample = source
        """The source of this clip"""

        self.amp = amp
        """An amplitude (gain) applied to this clip"""

        self.dynamic: str = dynamic
        """A dynamic expression attached to the score representation of this clip"""

        self.startSecs = startsecs
        """Start offset of the clip relative to the start of the source (in seconds)"""

        self.endSecs = endsecs if endsecs > 0 else self.sourceDurSecs

        self.speed = asF(speed)
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

    def __hash__(self):
        source = self.source if isinstance(self.source, str) else id(self.source)
        parts = (source, self.startSecs, self.endSecs, self.speed, self.sr,
                 self.numChannels, self.sourceDurSecs, self.amp)
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
        return (self.endSecs - self.startSecs) / self.speed

    def _calculateDuration(self,
                           relativeOffset: F | None,
                           parentOffset: F | None,
                           force=False
                           ) -> F | None:
        struct = self.scorestruct() or Workspace.active.scorestruct
        # TODO: use _durContext to validate the cached duration

        if not force and self._resolvedDur is not None:
            return self._resolvedDur

        offset = firstval(relativeOffset, self.offset, self._resolvedOffset, F0)
        startbeat = (parentOffset or F0) + offset
        starttime = struct.beatToTime(startbeat)
        dursecs = self.durSecs()
        endbeat = struct.timeToBeat(starttime + dursecs)
        self._resolvedDur = out = endbeat - startbeat
        self._durContext = (struct, startbeat)
        return out

    def resolveDur(self) -> F:
        if not self.parent:
            reloffset = self.offset
            parentoffset = F(0)
        else:
            reloffset = self.resolveOffset()
            parentoffset = self.parent.absoluteOffset()
        return self._calculateDuration(relativeOffset=reloffset, parentOffset=parentoffset, force=True)

    def __repr__(self):
        return f"Clip(source={self.source}, sr={self.sr}, dur={self.dur}, resolvedDur={self.resolveDur()}, sourcedursecs={_util.showT(self.sourceDurSecs)}secs)"

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
        endtime = starttime + self.durSecs()
        amp = firstval(self.amp, 1)
        bps = [[starttime, self.pitch, amp],
               [endtime, self.pitch, amp]]

        if self._playbackMethod == 'diskin':
            assert isinstance(self.source, str), f"The diskin playback method needs a path " \
                                                 f"as source, got {self.source}"
            args = {'ipath': self._soundfileCsoundStringNum,
                    'kspeed': self.speed,
                    'iskip': skip}
            playargs = playargs.clone(instr='_clip_diskin', args=args)

        elif self._playbackMethod == 'table':
            args = {'isndtab': self._soundfileCsoundTable.tabnum,
                    'kspeed': self.speed,
                    'istart': skip,
                    'ixfade': -1 if not self.loop else 0.1}
            playargs = playargs.clone(instr='_playtable', args=args)
        else:
            raise RuntimeError(f"Playback method {self._playbackMethod} not supported")
        event = SynthEvent(bps=bps, linkednext=self.tied, **playargs.db)
        event.linkednext = self.tied
        return [event]

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
                                    offset=offset)
        notation.setNotehead('square')
        return [notation]
