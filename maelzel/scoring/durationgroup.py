from __future__ import annotations
import dataclasses
from .core import Notation
from .common import *
from typing import List, Tuple, Union as U


def _canBeMerged(n0: Notation, n1: Notation) -> bool:
    """
    Returns True if n0 and n1 can me merged into one Notation
    with a regular duration

    NB: a regular duration is one which can be represented via
    one notation (a quarter, a half, a dotted 8th, a double dotted 16th are
    all regular durations, 5/8 of a quarter is not --which is a shame)
    """
    if (not n0.tiedNext or
            not n1.tiedPrev or
            n0.durRatios != n1.durRatios or
            n0.pitches != n1.pitches
            ):
        return False
    # durRatios are the same so check if durations would sum to a regular duration
    dur0 = n0.symbolicDuration()
    dur1 = n1.symbolicDuration()
    sumdur = dur0 + dur1
    num, den = sumdur.numerator, sumdur.denominator
    return den < 64 and num in {1, 2, 3, 7}


def mergeNotationsIfPossible(notations: List[Notation]) -> List[Notation]:
    """
    If two consecutive notations have same .durRatio and merging them
    would result in a regular note, merge them.

    8 + 8 = q
    q + 8 = q·
    q + q = h
    16 + 16 = 8

    In general:

    1/x + 1/x     2/x
    2/x + 1/x     3/x  (and viceversa)
    3/x + 1/x     4/x  (and viceversa)
    6/x + 1/x     7/x  (and viceversa)
    """
    assert len(notations) > 1
    out = [notations[0]]
    for n1 in notations[1:]:
        if _canBeMerged(out[-1], n1):
            out[-1] = out[-1].mergeWith(n1)
        else:
            out.append(n1)
    assert len(out) <= len(notations)
    assert sum(n.duration for n in out) == sum(n.duration for n in notations)
    return out


@dataclasses.dataclass
class DurationGroup:
    """
    A DurationGroup groups together Notations under time modifier (a tuple)
    A DurationGroup consists of a sequence of Notations or DurationGroups,
    allowing to define nested tuples or beats
    """
    durRatio: Tuple[int, int]
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
                if _canBeMerged(out[-1], i1):
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


def _splitGroupAt(group: List[Notation], splitPoints: List[F]
                  ) -> List[List[Notation]]:
    subgroups = defaultdict(lambda:list())
    for n in group:
        for i, splitPoint in enumerate(splitPoints):
            if n.end <= splitPoint:
                subgroups[i-1].append(n)
                break
        else:
            subgroups[len(splitPoints)].append(n)
    return [subgroup for subgroup in subgroups.values() if subgroup]