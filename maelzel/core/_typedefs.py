from maelzel.rational import Rat
from typing import List, Union, Tuple, TypeVar

num_t = Union[float, int, Rat]
time_t = Union[float, int, Rat]
pitch_t = Union[int, float, str]
fade_t = Union[float, Tuple[float, float]]
breakpoint_t = List[num_t]
T = TypeVar("T")