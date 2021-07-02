from typing import Union as U, Tuple, List, NamedTuple, Optional as Opt, TypeVar
from fractions import Fraction as F
from maelzel.mpqfractions import Rat
from pitchtools import m2n, n2m, split_notename, split_cents, accidental_name
import dataclasses
import enum

# This module can't import ANYTHING from .

time_t = U[float, int, F, Rat]
number_t = U[int, float, F, Rat]
pitch_t  = U[int, float, str]
timesig_t = Tuple[int, int]
division_t = List[U[int, 'division_t']]
timerange_t = Tuple[F, F]
T = TypeVar("T")


def asF(t: number_t) -> F:
    if isinstance(t, F):
        return t
    elif isinstance(t, Rat):
        return F(t.numerator, t.denominator)
    return F(t)


def asFractionOrNone(t: number_t) -> Opt[F]:
    if t is None:
        return None
    return asF(t)


def asmidi(x) -> float:
    if isinstance(x, float):
        assert 0<=x<=128
        return x
    elif isinstance(x, str):
        return n2m(x)
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


@dataclasses.dataclass
class Annotation:
    text: str
    placement: str = 'above'
    fontSize: int = None

    def __post_init__(self):
        assert not self.text.isspace()


@dataclasses.dataclass
class NotatedDuration:
    """
    base: 4=quarter note, 8=8th, etc
    dots: number of dots
    tuplets: a list of (num, den). Example: [(3, 2)] for a normal triplet

    To convert base to quarterDuration: base/4
    """
    base: int
    dots: int=0
    tuplets: List[Tuple[int, int]] = None


class GLISS(enum.Enum):
    START = 1
    END = 2
    ENDSTART = 3
    NONE = 4