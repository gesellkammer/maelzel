"""
Isorhythmic structures
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from emlib import iterlib

try:
    from quicktions import Fraction
except ImportError:
    from fractions import Fraction

if TYPE_CHECKING:
    from typing import *
    T = TypeVar('T')
    from maelzel.common import num_t
    

class Isorhythm:
    def __init__(self, color: Sequence[T], talea: Sequence[num_t]):
        self.color = color
        self.talea = talea
        self.iter = iter(self)

    def reset(self):
        self.iter = iter(self)

    def __iter__(self) -> Iterable[tuple[T, num_t]]:
        return zip(iterlib.cycle(self.color), iterlib.cycle(self.talea))

    def generate(self, maxdur: num_t) -> list[tuple[T, num_t]]:
        """
        Generate pairs until the given max. duration is reached

        Args:
            maxdur: the max. accumulated duration

        Returns:
            a list of pairs
        """
        partialdur = Fraction(0)
        pairs = []
        for color, talea in self:
            if partialdur + talea > maxdur:
                break
            pairs.append((color, talea))
            partialdur += talea
        return pairs

