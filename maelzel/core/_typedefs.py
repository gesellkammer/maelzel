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


time_t = Union[float, Rational, F]
fade_t = Union[float, tuple[float, float]]
# breakpoint_t = list[num_t]
breakpoint_t = list[float]
location_t = tuple[int, time_t]

