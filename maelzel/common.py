"""
NB: this module cannot import anything from maelzel itself
"""
from __future__ import annotations
import logging as _logging
import pitchtools as pt
import functools as _functools


import typing as _t
if _t.TYPE_CHECKING:
    from fractions import Fraction as F
else:
    from quicktions import Fraction as F


__all__ = (
    'getLogger',
    'F',
    'F0',
    'F1',
    'asF',
    'asmidi',
    'pitch_t',
    'time_t',
    'timesig_t',
    'num_t',
    'beat_t',
    'location_t'
)


num_t: _t.TypeAlias = _t.Union[int, float, F]
time_t: _t.TypeAlias = _t.Union[int, float, F]
pitch_t: _t.TypeAlias = _t.Union[int, float, str]
timesig_t: _t.TypeAlias = tuple[int, int]
location_t: _t.TypeAlias = tuple[int, time_t]
beat_t: _t.TypeAlias = _t.Union[time_t, location_t]


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


@_functools.cache
def getLogger(name: str,
              fmt='[%(name)s:%(filename)s:%(lineno)s:%(funcName)s:%(levelname)s] %(message)s',
              filelog: str = '',
              force=True
              ) -> _logging.Logger:
    """
    Construct a logger

    Args:
        name: the name of the logger
        fmt: the format used
        filelog: if given, logging info is **also** output to this file
        force: set own handlers, even if the logger already exists

    Returns:
        the logger
    """
    # TODO: move this to _logutils
    logger = _logging.getLogger(name)
    if logger.hasHandlers():
        # an old logger
        if not force:
            return logger
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


class _Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class UnsetType(metaclass=_Singleton):
    """
    A singleton representing an unset value.
    """
    def __repr__(self):
        return 'UNSET'

    def __bool__(self):
        return False




UNSET = UnsetType()
