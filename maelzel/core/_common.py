"""
Here are things which depend only on external libraries

Here we define:

* types
* constants
* basic conversion functions

"""

import logging as _logging
# We want to share the same Rat implementation as scorestruct
from maelzel.rational import Rat
from typing import List, Iterable
from ._typedefs import T
import os


MAXDUR = 99999

class _UNSET:
    def __repr__(self):
        return 'UNSET'

    def __bool__(self):
        return False

UNSET = _UNSET()

logger = _logging.getLogger(f"maelzel.core")


def asRat(x, den: int = None, maxden: int = None) -> Rat:
    """
    Create a Fraction

    Args:
        x: any number
        den: if given, then x is the numerator and den is the denominator of a fraction
        maxden: the max. denominator when converting a float to a Fraction

    Returns:
        x as Rational

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
    elif isinstance(x, int):
        # x is an int
        return Rat(x)
    else:
        raise TypeError("Expected a Rational, an int, a float or a tuple (int, den), "
                        f"got {x} of type {type(x)}")


def isNumber(x) -> bool:
    """ is x builtin number? (int, float or Fraction) """
    return isinstance(x, (int, float, Rat))




def getPath(s: str) -> str:
    if s == "?":
        from emlib import dialogs
        return dialogs.selectFile()
    else:
        return os.path.expanduser(s)
