from typing import Union as U, Tuple, List, NamedTuple, Optional as Opt, TypeVar
from fractions import Fraction as F
from emlib.pitchtools import m2n, n2m, split_notename, split_cents, accidental_name
import dataclasses

# This module can't import ANYTHING from .

time_t = U[float, int, F]
number_t = U[int, float, F]
pitch_t  = U[int, float, str]
timesig_t = Tuple[int, int]
division_t = List[U[int, 'division_t']]
timerange_t = Tuple[F, F]
T = TypeVar("T")


def asF(t: number_t) -> F:
    if isinstance(t, F):
        return t
    return F(t)


def asFractionOrNone(t: number_t) -> Opt[F]:
    if t is None:
        return None
    if isinstance(t, F):
        return t
    return F(t)


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
class Tempo:
    """
    tempo: the tempo value, relative to the referent
    referent: the value for which tempo is given. 1=quarter note, 1/2=8th note
    """
    tempo: F
    referent: F = F(1)

    @property
    def quarterTempo(self) -> F:
        return F(1)/self.referent*self.tempo

    # def __hash__(self):
    #    return hash((self.tempo, self.referent))


@dataclasses.dataclass
class Annotation:
    text: str
    placement: str = 'above'
    fontSize: int = None


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


dynamicLevels = ["pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "ffff"]
availableDynamics = set(dynamicLevels)

availableNoteheads = {"slash", "triangle", "diamond", "square", "cross", "rectangle", "none"}

availableArticulations = {'accent', 'stacatto', 'tenuto'}