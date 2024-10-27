from numbers import Rational
from typing import Union
from maelzel.common import F, pitch_t, num_t


__all__ = (
    'num_t',
    'time_t',
    'pitch_t',
    'fade_t',
    'breakpoint_t',
    'location_t'
)


time_t = Union[int, float, F]
fade_t = Union[float, tuple[float, float]]
breakpoint_t = list[float]
location_t = tuple[int, time_t]
beat_t = Union[F, float, tuple[int, time_t]]
