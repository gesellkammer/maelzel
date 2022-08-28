"""
Common type definitions and routines
"""
from __future__ import annotations
from maelzel.rational import Rat as F
from dataclasses import dataclass
import enum
import logging
import numbers as _numbers
from typing import TYPE_CHECKING, NamedTuple
import pitchtools as pt

if TYPE_CHECKING:
    from typing import Union, Tuple, List, TypeVar, Optional
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


class Annotation:
    __slots__ = ('text', 'placement', 'fontsize', 'fontstyles', 'box')

    def __init__(self, text: str, placement='above', fontsize: float = None, fontstyle='',
                 box: str|bool = False):
        assert not text.isspace()
        if fontsize is not None:
            assert isinstance(fontsize, (int, float))
        self.text = text
        self.placement = placement
        self.fontsize = fontsize
        self.box: str = box if isinstance(box, str) else 'square' if box else ''
        if not fontstyle:
            self.fontstyles = None
        else:
            styles = fontstyle.split(',')
            for style in styles:
                assert style in {'italic', 'bold'}, f'Style {style} not supported'
            self.fontstyles = styles

    def isItalic(self):
        return self.fontstyles and 'italic' in self.fontstyles

    def isBold(self):
        return self.fontstyles and 'bold' in self.fontstyles


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
