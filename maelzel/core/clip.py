from __future__ import annotations
import pitchtools as pt
import sndfileio
import numpy as np

from maelzel.scorestruct import ScoreStruct
from maelzel.common import F, asF, asmidi
from maelzel.core.mobj import MContainer
from maelzel.core.event import MEvent
from maelzel.core._typedefs import *
from maelzel.core import _dialogs
from maelzel.core.synthevent import SynthEvent, PlayArgs
from maelzel.core.workspace import Workspace
from maelzel.snd import audiosample

from maelzel import scoring
import emlib.misc


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

    __slots__ = ('startsecs', 'endsecs', 'speed', 'source', 'soundfile', 'numchannels',
                 'samplerate', 'amp', 'dynamic', 'pitch', 'sourcedursecs')

    def __init__(self,
                 source: str | audiosample.Sample | tuple[np.ndarray, int],
                 pitch: pitch_t = None,
                 amp: float = 1.,
                 offset: time_t = None,
                 label: str = '',
                 dynamic: str = '',
                 startsecs=0.,
                 endsecs=0.,
                 speed: F | float =F(1),
                 parent: MContainer | None = None
                 ):
        if source == '?':
            source = _dialogs.selectSndfileForOpen()
            if not source:
                raise ValueError("No source selected")

        self.soundfile = ''
        """The soundfile holding the audio data (if any)"""

        self.numchannels = 0
        """The number of channels of this Clip"""

        self.samplerate = 0
        """The samplerate of this Clip"""

        self.sourcedursecs = 0
        """The duration in seconds of the source of this Clip"""

        if isinstance(source, str):
            self.soundfile = source
            info = sndfileio.sndinfo(source)
            self.samplerate = info.samplerate
            self.sourcedursecs = info.duration
            self.numchannels = info.channels
        elif isinstance(source, audiosample.Sample):
            self.samplerate = source.sr
            self.sourcedursecs = source.duration
            self.numchannels = source.numchannels
        elif isinstance(source, tuple) and len(source) == 2 and isinstance(source[0], np.ndarray):
            samples, sr = source
            self.samplerate = sr
            self.sourcedursecs = len(samples) / sr
            self.numchannels = 1 if len(samples.shape) == 1 else samples.shape[1]

        self.source = source
        """The source of this clip"""

        self.amp = amp
        """An amplitude (gain) applied to this clip"""

        self.dynamic: str = dynamic
        """A dynamic expression attached to the score representation of this clip"""

        self.startsecs = startsecs
        """Start offset of the clip relative to the start of the source (in seconds)"""

        self.endsecs = endsecs if endsecs > 0 else self.sourcedursecs

        self.speed = asF(speed)
        """Playback speed"""

        self._sample: audiosample.Sample|None = None

        if pitch is None:
            s = self.asSample()
            freq = s.fundamentalFreq()
            pitch = pt.f2n(freq)
        else:
            pitch = asmidi(pitch)

        self.pitch: float = pitch
        """The pitch representation of this clip.
        This is used for notation purposes, it has no influence on the playback
        of this clip"""

        super().__init__(offset=offset, dur=None, label=label, parent=parent)

    def __hash__(self):
        source = self.source if isinstance(self.source, str) else id(self.source)
        parts = (source, self.startsecs, self.endsecs, self.speed, self.samplerate,
                 self.numchannels, self.sourcedursecs, self.amp)
        return hash(parts)

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
            sample = self.source
        elif isinstance(self.source, tuple):
            samples, sr = self.source
            sample =  audiosample.Sample(samples, sr=sr)
        else:
            sample = audiosample.Sample(self.source)
        self._sample = sample
        return sample

    def isRest(self) -> bool:
        return False

    def durSecs(self) -> F:
        return (self.endsecs - self.startsecs) / self.speed

    def _calculateDuration(self,
                           relativeOffset: F | None,
                           parentOffset: F | None,
                           force=False
                           ) -> F | None:
        if not force and (dur := self.offset if self.offset is not None else self._resolvedDur) is not None:
            return dur
        struct = self.scorestruct() or Workspace.active.scorestruct

        offset = emlib.misc.firstval(relativeOffset, self.offset, self._resolvedOffset, F(0))
        startbeat = (parentOffset or F(0)) + offset
        starttime = struct.beatToTime(startbeat)
        dursecs = self.durSecs()
        endbeat = struct.timeToBeat(starttime + dursecs)
        self._resolvedDur = out = endbeat - startbeat
        return out

    def resolveDur(self) -> F:
        if not self.parent:
            return self._calculateDuration(relativeOffset=self.offset, parentOffset=F(0))
        return self._calculateDuration(relativeOffset=self.resolveOffset(),
                                       parentOffset=self.parent.absoluteOffset())

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        # todo
        return []

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig = None
                      ) -> list[scoring.Notation]:
        if not config:
            config = getConfig()
        dur = self.resolveDur()
        offset = self.absoluteOffset()
        # offset = self._scoringOffset(config=config)

        notation = scoring.makeNote(pitch=self.pitch,
                                    duration=dur,
                                    offset=offset)
        return [notation]


