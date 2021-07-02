from __future__ import annotations
from .core import Notation, notationsCanMerge, mergeNotationsIfPossible
from .common import *
import dataclasses
from typing import List, Tuple, Union as U

durratio_t = Tuple[int, int]

@dataclasses.dataclass
class DurationGroup:
    """
    A DurationGroup is a container, grouping together Notations under time modifier (a
    tuple)
    A DurationGroup consists of a sequence of Notations or DurationGroups, allowing to
    define nested tuples or beats. The notations inside a DurationGroup already hold the
    real beat-duration. The durRatio is a ratio by which to multiply a given duration to
    obtain the notated duration.

    durRatio: a tuple (num, den) indication the ratio by which to multiply the duration
    of the items to obtain the notated items: the items inside this group

    In the case of a simple triplet, the items would hold something like::

        >>> from maelzel import scoring
        >>> notations = [scoring.makeNote(60, duration=F(1, 3)),
        ...              scoring.makeNote(61, duration=F(2, 3))]
        >>> DurationGroup(durRatio=(3, 2), items=notations)
    """
    durRatio: durratio_t
    items: List[U[Notation, 'DurationGroup']]

    def symbolicDuration(self) -> F:
        """
        The symbolic duration of this Notation. This represents
        the notated figure (1=quarter, 1/2=eighth note, 1/4=16th note, etc)
        """
        return sum(item.symbolicDuration() for item in self.items)

    def __repr__(self):
        parts = [f"DurationGroup({self.durRatio[0]}/{self.durRatio[1]}, "]
        for item in self.items:
            if isinstance(item, Notation):
                parts.append("  " + str(item))
            else:
                s = str(item)
                for line in s.splitlines():
                    parts.append("  " + line)
        parts.append(")")
        return "\n".join(parts)

    def mergeNotations(self) -> DurationGroup:
        i0 = self.items[0]
        out = [i0 if isinstance(i0, Notation) else i0.mergeNotations()]
        for i1 in self.items[1:]:
            if isinstance(out[-1], Notation) and isinstance(i1, Notation):
                if notationsCanMerge(out[-1], i1):
                    out[-1] = out[-1].mergeWith(i1)
                else:
                    out.append(i1)
            elif isinstance(i1, Notation):
                assert isinstance(out[-1], DurationGroup)
                out.append(i1)
            else:
                assert isinstance(out[-1], Notation) and isinstance(i1, DurationGroup)
                out.append(i1.mergeNotations())
        return DurationGroup(durRatio=self.durRatio, items=out)


