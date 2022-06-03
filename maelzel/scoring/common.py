"""
Common type definitions and routines
"""
from __future__ import annotations
from maelzel.rational import Rat as F
from dataclasses import dataclass
import enum
import logging
import numbers as _numbers
from typing import NamedTuple, Union, Tuple, List, TypeVar, Optional
import pitchtools as pt

time_t = Union[float, int, F]
pitch_t = Union[int, float, str]
timesig_t = Tuple[int, int]
division_t = List[Union[int, list]]
# division_t = List[Union[int, 'division_t']]
timerange_t = Tuple[F, F]
T = TypeVar("T")

logger = logging.getLogger("maelzel.scoring")

# This module can't import ANYTHING from .


def asF(t: _numbers.Real) -> F:
    """
    Convert ``t`` to a fraction if needed
    """
    if isinstance(t, F):
        return t
    elif isinstance(t, _numbers.Rational):
        return F(t.numerator, t.denominator)
    return F(t)


def asmidi(x) -> float:
    """
    Convert x to midi

    A number is interpreted as midi, never as a frequency

    Args:
        x: a notename as str or a midinote as int/float

    Returns:
        a (possibly fractional) midinote

    """
    if isinstance(x, float):
        assert 0<=x<=128
        return x
    elif isinstance(x, str):
        return pt.n2m(x)
    elif isinstance(x, int):
        assert 0 <= x < 128
        return float(x)
    elif hasattr(x, "pitch"):
        return x.pitch
    raise TypeError(f"Cannot interpret {x} as a midinote")


class TimeSpan(NamedTuple):
    start: F
    end: F

    @property
    def duration(self) -> F:
        return self.end-self.start


@dataclass
class Annotation:
    text: str
    placement: str = 'above'
    fontSize: Optional[int] = None

    def __post_init__(self):
        assert not self.text.isspace()


@dataclass
class NotatedDuration:
    """
    Class representing the duration of a note

    To convert base to quarterDuration: base/4

    Attributes:
        base: 4=quarter note, 8=8th, etc
        dots: number of dots
        tuplets: a list of (num, den). Example: [(3, 2)] for a normal triplet
    """
    base: int
    dots: int=0
    tuplets: Optional[List[Tuple[int, int]]] = None


class GLISS(enum.Enum):
    START = 1
    END = 2
    ENDSTART = 3
    NONE = 4
