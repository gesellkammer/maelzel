"""
Common type definitions and routines
"""
from __future__ import annotations
from dataclasses import dataclass
import enum
import logging
import numbers as _numbers
import pitchtools as pt
from maelzel.common import F, asF
import sys

from typing import NamedTuple, Union

if sys.version_info.minor >= 10:
    from typing import TypeAlias
    time_t: TypeAlias = float | int | F
    pitch_t: TypeAlias = int | float | str
    timesig_t: TypeAlias = tuple[int, int]
    division_t: TypeAlias = tuple[Union[int, 'division_t'], ...]
    timerange_t: TypeAlias = tuple[F, F]
else:
    time_t = Union[float, int, F]
    pitch_t = Union[int, float, str]
    timesig_t = tuple[int, int]
    division_t = tuple[Union[int, 'division_t'], ...]
    timerange_t = tuple[F, F]


logger = logging.getLogger("maelzel.scoring")

# This module can't import ANYTHING from .


__all__ = (
    'F',
    'asF',
    'logger',
    'asmidi',
    'TimeSpan',
    'NotatedDuration',
    'GLISS',
    'time_t',
    'pitch_t',
    'timesig_t',
    'division_t',
    'timerange_t'
)


def _asF(t: _numbers.Real) -> F:
    """
    Convert ``t`` to a fraction if needed
    """
    if isinstance(t, F):
        return t
    elif isinstance(t, _numbers.Rational):
        return F(t.numerator, t.denominator)
    return F(t)


def asmidi(x, maxmidi=130) -> float:
    """
    Convert x to midi

    A number is interpreted as midi, never as a frequency

    Args:
        x: a notename as str or a midinote as int/float
        maxmidi: the max. value to accept as midi. Any value higher will
            generate an error

    Returns:
        a (possibly fractional) midinote

    """
    if isinstance(x, float):
        assert 0<=x<=maxmidi, f"Invalid midinote: {x}"
        return x
    elif isinstance(x, str):
        return pt.n2m(x)
    elif isinstance(x, int):
        assert 0 <= x < maxmidi
        return float(x)
    elif hasattr(x, "pitch"):
        return x.notename
    raise TypeError(f"Cannot interpret {x} as a midinote")


class TimeSpan(NamedTuple):
    start: F
    end: F

    @property
    def duration(self) -> F:
        return self.end - self.start


@dataclass
class NotatedDuration:
    """
    Class representing the totalDuration of a note

    To convert base to quarterDuration: base/4

    Attributes:
        base: 4=quarter note, 8=8th, etc.
        dots: number of dots
        tuplets: a list of (num, den). Example: [(3, 2)] for a normal triplet
    """
    base: int
    dots: int = 0
    tuplets: list[tuple[int, int]] | None = None


class GLISS(enum.Enum):
    START = 1
    END = 2
    ENDSTART = 3
    NONE = 4
