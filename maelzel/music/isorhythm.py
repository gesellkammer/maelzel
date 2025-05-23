"""
Isorhythmic structures
"""
from __future__ import annotations
from emlib import iterlib
import itertools

try:
    from quicktions import Fraction
except ImportError:
    from fractions import Fraction

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Iterable, TypeVar, Generic
    from maelzel.common import num_t
    _T = TypeVar('_T')



class Isorhythm(Generic[_T]):
    """
    Isorhythmic structures

    Isorhythmic structures are a way of organizing musical material
    into a repeating pattern of durations.

    Args:
        color: the sequence of elements to be repeated
        talea: the sequence of durations to be repeated

    Attributes:
        color: the sequence of elements to be repeated
        talea: the sequence of durations to be repeated
        iterator: the iterator for the isorhythm
    """
    def __init__(self, color: Sequence[_T], talea: Sequence[num_t]):
        self.color = color
        self.talea = talea
        self.iterator = self._makeiter()

    def reset(self) -> None:
        """
        Reset the iterator to the beginning of the sequence
        """
        self.iterator = self._makeiter()

    def _makeiter(self) -> Iterable[tuple[_T, num_t]]:
        return zip(iterlib.cycle(self.color), iterlib.cycle(self.talea))

    def __iter__(self) -> Iterable[tuple[_T, num_t]]:
        return iter(self.iterator)

    def take(self, numpairs: int) -> list[tuple[_T, num_t]]:
        """
        Take the next n pairs from the iterator

        Args:
            numpairs: the number of pairs to take

        Returns:
            the generated pairs

        Example
        ~~~~~~~

            >>> isorhythm = Isorhythm(['A', 'B'], [1, 2, 1])
            >>> isorhythm.take(3)
            [('A', 1), ('B', 2), ('A', 1)]
        """

        return list(itertools.islice(self.iterator, numpairs))

    def generate(self, maxdur: num_t) -> list[tuple[_T, num_t]]:
        """
        Generate pairs until the given max. duration is reached

        Args:
            maxdur: the max. accumulated duration

        Returns:
            a list of pairs

        Example
        ~~~~~~~

            >>> isorhythm = Isorhythm(['A', 'B'], [1, 2, 1])
            >>> isorhythm.generate(4)
            [('A', 1), ('B', 2), ('A', 1)]

        """
        partialdur = Fraction(0)
        pairs = []
        for color, talea in self.iterator:
            if partialdur + talea > maxdur:
                break
            pairs.append((color, talea))
            partialdur += talea
        return pairs
