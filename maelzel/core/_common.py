"""
Here are things which depend only on external libraries

Here we define:

* types
* constants
* basic conversion functions

"""

import logging as _logging
# from fractions import Fraction as Rat
from maelzel.mpqfractions import Rat
from typing import (Union as U, Tuple, TypeVar, Optional as Opt, List, Dict,
                    Iterable as Iter, Callable)


num_t = U[float, int, Rat]
time_t = U[float, int, Rat]
pitch_t = U[int, float, str]
fade_t = U[float, Tuple[float, float]]
breakpoint_t = Tuple[num_t, ...]
T = TypeVar("T")

MAXDUR = 99999
UNSET = object()

logger = _logging.getLogger(f"maelzel.core")


def asRat(x: U[Rat, float, int], den:int=None, maxden:int=None) -> Rat:
    """
    Create a Fraction

    Args:
        x: any number
        den: if given, then x is the numerator and den is the denominator of a fraction
        maxden: the max. denominator when converting a float to a Fraction

    Returns:

    """
    if isinstance(x, Rat):
        return x
    elif isinstance(x, float):
        out = Rat.from_float(x)
        if maxden:
            out = out.limit_denominator(maxden)
        return out
    elif den is not None:
        return Rat(x, den)
    else:
        # x is an int
        return Rat(x)


def isNumber(x) -> bool:
    """ is x builtin number? (int, float or Fraction) """
    return isinstance(x, (int, float, Rat))


def asTuple(obj) -> tuple:
    return obj if isinstance(obj, tuple) else tuple(obj)
