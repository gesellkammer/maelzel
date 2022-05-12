from __future__ import annotations
from maelzel.core import musicobj
from maelzel.snd import audiosample
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Tuple
import numpy as np
from maelzel.core._typedefs import *
from maelzel.core import tools


class Clip(musicobj.Note):
    """
    A Clip represent an audio sample in time

    Args:
        source: the source of the clip
        dur: the duration of the clip. If not given, the duration of the
            source is used
        start: the time offset of this clip. Like in a Note, if not given,
            the start time depends on the context (previous events) where this
            clip is evaluated
        label: a label str to identify this clip
        dynamic: allows to attach a dynamic expression to this Clip. This is
            only for notation purposes, it does not modify playback
        skip: skip time of the source. If given, it will affect the duration
        speed: playback speed of the clip

    Attributes:
        sample: a :class:`maelzel.snd.audiosample.Sample`
    """
    __slots__ = ('skip', 'speed', 'source')

    def __init__(self, source: Union[str, audiosample.Sample, Tuple[np.ndarray, int]],
                 dur: time_t = None,
                 pitch: pitch_t = None,
                 amp: float = None,
                 start: time_t = None,
                 label: str = '',
                 dynamic: str = None,
                 skip=0.,
                 speed=1.
                 ):
        if source == '?':
            source = tools.selectSndfileForOpen()
            if not source:
                raise ValueError("No source selected")
        sample = audiosample.asSample(source)
        musicobj.Note.__init__(self, dur=dur, start=start, label=label, pitch=pitch,
                               tied=False, dynamic=dynamic, amp=amp)
        self.source = audiosample.asSample(source)
        self.skip = skip
        self.speed = speed