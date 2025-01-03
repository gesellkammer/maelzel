"""
NB: this module cannot import anything from maelzel itself
"""
from __future__ import annotations
import numbers as _numbers
import logging as _logging
import pitchtools as pt
from numbers import Rational

import typing as _t

if not _t.TYPE_CHECKING:
    from quicktions import Fraction as F
else:
    from fractions import Fraction as F


__all__ = (
    'getLogger',
    'F',
    'F0',
    'F1',
    'asF',
    'asmidi',
    'pitch_t',
    'timesig_t',
    'num_t',
)


num_t = _t.Union[int, float, F]
pitch_t = _t.Union[int, float, str]
timesig_t = _t.Tuple[int, int]


F0: F = F(0)
F1: F = F(1)


def asF(t: int | float | str | F) -> F:
    """
    Convert ``t`` to a fraction if needed
    """
    if isinstance(t, F):
        return t
    elif isinstance(t, (int, float, str)):
        return F(t)
    else:
        raise TypeError(f"Could not convert {t} to a rational")


def asmidi(x) -> float:
    """
    Convert x to a midinote

    Args:
        x: a str ("4D", "1000hz") a number (midinote) or anything
           with an attribute .midi

    Returns:
        a midinote

    """
    if isinstance(x, str):
        return pt.str2midi(x)
    elif isinstance(x, (int, float)):
        assert 0 <= x <= 200, f"Expected a midinote (0-127) but got {x}"
        return x
    raise TypeError(f"Expected a str, a Note or a midinote, got {x}")


def getLogger(name: str, fmt='[%(name)s:%(filename)s:%(lineno)s:%(funcName)s:%(levelname)s] %(message)s',
              filelog: str = ''
              ) -> _logging.Logger:
    """
    Construct a logger

    Args:
        name: the name of the logger
        fmt: the format used
        filelog: if given, logging info is **also** output to this file

    Returns:
        the logger
    """
    logger = _logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.propagate = False

    handler = _logging.StreamHandler()
    formatter = _logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if filelog:
        filehandler = _logging.FileHandler(filelog)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    return logger


class Unset:
    def __repr__(self):
        return 'UNSET'

    def __bool__(self):
        return False


UNSET = Unset()
