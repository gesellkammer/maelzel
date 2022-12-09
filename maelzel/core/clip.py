from __future__ import annotations
from maelzel.scorestruct import ScoreStruct
from maelzel.common import F, asF
from maelzel.core.mobj import MContainer
from maelzel.core.event import MEvent
from maelzel.snd import audiosample
import numpy as np
from maelzel.core._typedefs import *
from maelzel.core import _dialogs
import sndfileio
from .synthevent import SynthEvent, PlayArgs
from .workspace import Workspace
import emlib.misc


class Clip(MEvent):
    """
    A Clip represent an audio sample in time

    Args:
        source: the source of the clip (a filename, audiosample, samples as numpy array)
        dur: the duration of the clip. If not given, the duration of the
            source is used
        offset: the time offset of this clip. Like in a Note, if not given,
            the start time depends on the context (previous events) where this
            clip is evaluated
        label: a label str to identify this clip
        dynamic: allows to attach a dynamic expression to this Clip. This is
            only for notation purposes, it does not modify playback
        skip: skip time of the source. If given, it will affect the duration
        speed: playback speed of the clip

    """
    _isDurationRelative = False

    __slots__ = ('skip', 'speed', 'source', 'soundfile', 'numchannels', 'samplerate',
                 'amp', 'dynamic', 'pitch', 'sourcedursecs')

    def __init__(self,
                 source: str | audiosample.Sample | tuple[np.ndarray, int],
                 pitch: pitch_t = None,
                 amp: float = 1.,
                 offset: time_t = None,
                 label: str = '',
                 dynamic: str = '',
                 skip=0.,
                 speed: F | float =F(1),
                 parent: MContainer | None = None
                 ):
        if source == '?':
            source = _dialogs.selectSndfileForOpen()
            if not source:
                raise ValueError("No source selected")
        self.soundfile = ''
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

        self.skip = skip
        """Start offset of the clip relative to the start of the source"""

        self.speed = asF(speed)
        """Playback speed"""

        self._sample: audiosample.Sample|None = None

        if pitch is None:
            s = self.asSample()
            pitch = s.fundamentalFreq()

        self.pitch = pitch

        super().__init__(offset=offset, dur=None, label=label, parent=parent)

    def asSample(self) -> audiosample.Sample:
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
        return self.sourcedursecs / self.speed

    def _calculateDuration(self,
                           relativeOffset: F | None,
                           parentOffset: F | None,
                           force=False
                           ) -> F | None:
        if not force and (dur := self.offset if self.offset is not None else self._resolvedDur) is not None:
            return dur
        offset = emlib.misc.firstval(relativeOffset, self.offset, self._resolvedOffset, F(0))
        startbeat = (parentOffset or F(0)) + offset
        struct = self.scorestruct() or Workspace.active.scorestruct
        starttime = struct.beatToTime(startbeat)
        dursecs = self.sourcedursecs / self.speed
        endtime = starttime + dursecs
        endbeat = struct.timeToBeat(endtime)
        self._resolvedDur = out = endbeat - startbeat
        return out

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        # todo
        return []