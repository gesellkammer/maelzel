from numbers import Rational
from typing import Union
from maelzel.common import F

num_t = Union[float, Rational, F]
time_t = Union[float, Rational, F]
pitch_t = Union[int, float, str]
fade_t = Union[float, tuple[float, float]]
breakpoint_t = list[num_t]
