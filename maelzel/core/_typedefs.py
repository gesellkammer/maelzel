from maelzel.rational import Rat
from numbers import Rational
from typing import TYPE_CHECKING, Union, TypeVar

if TYPE_CHECKING:
    # num_t = Union[float, int, Rat]
    num_t = Union[float, Rational]
    # time_t = Union[float, int, Rat]
    time_t = Union[float, Rational]
    pitch_t = Union[int, float, str]
    fade_t = Union[float, tuple[float, float]]
    breakpoint_t = list[num_t]
