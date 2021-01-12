"""
Here are things which depend only on external libraries

Here we define:

* types
* constants
* basic conversion functions

"""

import logging as _logging
from fractions import Fraction
from typing import Union as U, Optional as Opt, Tuple, TypeVar, Iterator as Iter, Sequence as Seq, List


num_t = U[float, int, Fraction]
time_t = U[float, int, Fraction]
pitch_t = U['Note', float, str]
fade_t = U[float, Tuple[float, float]]
T = TypeVar("T")

MAXDUR = 99999
UNSET = object()


logger = _logging.getLogger(f"maelzel.music_core")


def F(x: U[Fraction, float, int], den=None, maxden=1000000) -> Fraction:
    if den is not None:
        return Fraction(x, den).limit_denominator(maxden)
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(maxden)


def asTime(x:num_t, maxden=1024) -> Fraction:
    if isinstance(x, Fraction):
        return x.limit_denominator(maxden)
    return F(x, maxden=maxden)


