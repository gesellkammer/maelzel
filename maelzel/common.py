
from pitchtools import n2m
import numbers as _numbers
import logging as _logging


__all__ = (
    'F',
    'asF',
    'asmidi',
    'F0',
    'F1',
    'pitch_t',
    'timesig_t',
    'number_t',
    'getLogger'
)

import typing as t
pitch_t = t.Union[int, float, str]
timesig_t = t.Tuple[int, int]
number_t = t.Union[float, _numbers.Rational]

# Rat is like Fraction with the only difference that its __repr__ is float like
# If quicktions are present it will use that as a base
# from maelzel.rational import Rat as F
from quicktions import Fraction as F


F0 = F(0)
F1 = F(1)


def asF(t) -> F:
    """
    Convert ``t`` to a fraction if needed
    """
    if isinstance(t, F):
        return t
    elif isinstance(t, _numbers.Rational):
        return F(t.numerator, t.denominator)
    elif isinstance(t, (int, float, str)):
        return F(t)
    else:
        raise TypeError(f"Could not convert {t} to a rational")


def asmidi(x: pitch_t) -> float:
    """
    Converts a notename to a midinote.
    """
    if isinstance(x, (int, float)):
        if x > 127:
            raise ValueError("A midinote expected (< 128), but got a value of {x}!")
        return x
    elif isinstance(x, str):
        return n2m(x)
    try:
        return float(x)
    except TypeError:
        raise TypeError(f"could not convert {x} to a midi note")


def getLogger(name: str, fmt='[%(name)s:%(filename)s:%(lineno)s - %(funcName)s] %(message)s',
              filelog: str = ''
               ) -> _logging.Logger:
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
