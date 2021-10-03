"""
Isorhythmic structures
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from maelzel.rational import Rat
from emlib import iterlib

if TYPE_CHECKING:
    from typing import *
    T = TypeVar('T')
    number_t = Union[int, float, Rat]


class Isorhythm:
    def __init__(self, color: Sequence[T], talea: Sequence[number_t]):
        self.color = color
        self.talea = talea
        self.iter = iter(self)

    def reset(self):
        self.iter = iter(self)

    def __iter__(self) -> Iterable[Tuple[T, number_t]]:
        return zip(iterlib.cycle(self.color), iterlib.cycle(self.talea))

    def generate(self, maxdur: number_t) -> List[Tuple[T, number_t]]:
        """
        Generate pairs until the given max. duration is reached

        Args:
            maxdur: the max. accumulated duration

        Returns:
            a list of pairs
        """
        partialdur = Rat(0)
        pairs = []
        for color, talea in self:
            if partialdur + talea > maxdur:
                break
            pairs.append((color, talea))
            partialdur += talea
        return pairs

