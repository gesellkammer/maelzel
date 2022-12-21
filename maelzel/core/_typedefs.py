from numbers import Rational
from typing import Union

num_t = Union[float, Rational]
time_t = Union[float, Rational]
pitch_t = Union[int, float, str]
fade_t = Union[float, tuple[float, float]]
breakpoint_t = list[num_t]
