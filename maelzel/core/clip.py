from __future__ import annotations
from maelzel.scorestruct import ScoreStruct
from maelzel.common import F
from maelzel.core.event import Note
from maelzel.snd import audiosample
import numpy as np
from maelzel.core._typedefs import *
from maelzel.core import _dialogs
import sndfileio
from .synthevent import SynthEvent, PlayArgs
from .workspace import getWorkspace

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .workspace import Workspace


class Clip(Note):
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

    __slots__ = ('skip', 'speed', 'source', 'soundfile', 'numchannels')

    def __init__(self,
                 source: str | audiosample.Sample | tuple[np.ndarray, int],
                 dur: time_t = None,
                 pitch: pitch_t = None,
                 amp: float = 1.,
                 offset: time_t = None,
                 label: str = '',
                 dynamic: str = None,
                 skip=0.,
                 speed=1.
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
            self.dur = info.duration
            self.numchannels = info.channels
        elif isinstance(source, audiosample.Sample):
            self.samplerate = source.sr
            self.dur = source.duration
            self.numchannels = source.numchannels
        elif isinstance(source, tuple) and len(source) == 2 and isinstance(source[0], np.ndarray):
            samples, sr = source
            self.samplerate = sr
            self.dur = len(samples) / sr
            self.numchannels = 1 if len(samples.shape) == 1 else samples.shape[1]

        self.source = source
        """The source of this clip"""

        self.skip = skip
        """Start offset of the clip relative to the start of the source"""

        self.speed = speed
        """Playback speed"""

        self._sample: audiosample.Sample|None = None

        if pitch is None:
            s = self.asSample()
            pitch = s.fundamentalFreq()

        super().__init__(pitch=pitch, dur=dur, offset=offset, label=label,
                         tied=False, dynamic=dynamic, amp=amp)

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

    def plot(self):
        self.asSample().plot()

    def isRest(self) -> bool:
        return False

    def resolvedDur(self, start: time_t = None, struct: ScoreStruct = None) -> F:
        if struct is None:
            struct = getWorkspace().scorestruct
        return struct.totalDurationBeats()

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        # todo
        return []