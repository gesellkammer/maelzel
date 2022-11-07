from __future__ import annotations
from .core import Notation, notationsCannotMerge
from .common import *
from numbers import Rational
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    durratio_t = tuple[int, int]


__all__ = (
    'DurationGroup',
    'asDurationGroupTree'
)


def rationalToTuple(r: Rational) -> tuple[int, int]:
    return r.numerator, r.denominator


class DurationGroup:
    """
    A DurationGroup is a container, grouping Notation under one time modifier

    A DurationGroup consists of a sequence of Notations or DurationGroups, allowing to
    define nested tuples or beats. The notations inside a DurationGroup already hold the
    real beat-duration. The durRatio is a ratio by which to multiply a given duration to
    obtain the notated duration.

    Attributes:
        durRatio: a tuple (num, den) indication the ratio by which to multiply the duration
            of the items to obtain the notated items: the items inside this group
        items: the items in this group

    In the case of a simple triplet, the items would hold something like::

        >>> from maelzel.scoring import *
        >>> notations = [makeNote(60, duration=F(1, 3)),
        ...              makeNote(61, duration=F(2, 3))]
        >>> DurationGroup(durRatio=(3, 2), items=notations)
    """
    def __init__(self, durRatio: durratio_t | Rational, items: list[Notation | 'DurationGroup'] = None):
        self.durRatio: durratio_t = durRatio if isinstance(durRatio, tuple) else rationalToTuple(durRatio)
        self.items: list[Notation | 'DurationGroup'] = items if items else []
        self.properties: dict | None = None

    def setProperty(self, key: str, value) -> None:
        if self.properties is None:
            self.properties = {}
        self.properties[key] = value

    def getProperty(self, key: str, default=None):
        if self.properties is None:
            return default
        return self.properties.get(key, default)

    @property
    def ratio(self) -> F:
        return F(*self.durRatio)

    @property
    def offset(self) -> F:
        return self.items[0].offset

    @property
    def end(self) -> F:
        return self.items[-1].end

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def append(self, item: Notation | DurationGroup) -> None:
        self.items.append(item)

    def duration(self) -> F:
        """
        The actual duration of the items in this group

        """
        return sum((item.duration if isinstance(item, Notation) else item.duration()
                    for item in self.items), F(0))

    def symbolicDuration(self) -> F:
        """
        The symbolic duration of this group.

        This represents the notated figure (1=quarter, 1/2=eighth note, 1/4=16th note, etc)
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

    def _flattenUnnecessarySubgroups(self):
        if self.durRatio != (1, 1):
            return

        items = []
        for item in self.items:
            if isinstance(item, Notation):
                items.append(item)
            elif isinstance(item, DurationGroup):
                if item.durRatio == (1, 1):
                    item._flattenUnnecessarySubgroups()
                    items.extend(item.items)
                else:
                    items.append(item)
        self.items = items

    def mergeWith(self, other: DurationGroup) -> DurationGroup:
        """
        Merge this group with other

        """
        # we don't check here, just merge
        group = DurationGroup(durRatio=self.durRatio, items=self.items + other.items)
        group = group.mergedNotations()
        return group

    def mergedNotations(self) -> DurationGroup:
        """
        Returns a new group with all items merged (recursively)

        Returns:
            a new DurationGroup with merged items (whenever possible)
        """
        self._flattenUnnecessarySubgroups()
        i0 = self.items[0]
        out = [i0 if isinstance(i0, Notation) else i0.mergedNotations()]
        for i1 in self.items[1:]:
            i0 = out[-1]
            if isinstance(i0, Notation) and isinstance(i1, Notation):
                if i0.canMergeWith(i1):
                    out[-1] = i0.mergeWith(i1)
                else:
                    out.append(i1)
            else:
                # n+G, G+n or G+G
                out.append(i1 if isinstance(i1, Notation) else i1.mergedNotations())
        if len(out) == 1 and isinstance(out[0], Notation):
            n = out[0]
            if n.durRatios and n.durRatios[-1] != F(1):
                n.durRatios.pop()
            return DurationGroup(durRatio=F(1), items=[n])
        return DurationGroup(durRatio=self.durRatio, items=out)


def asDurationGroupTree(groups: list[DurationGroup]) -> DurationGroup:
    """
    Transform a list of DurationGroups into a tree structure

    A tree has a root and leaves, where each leave can be the root of a subtree

    Args:
        groups: the groupTree to get/make the root for

    Returns:
        the root of a tree structure
    """
    if len(groups) == 1 and groups[0].durRatio == (1, 1):
        return groups[0]
    return DurationGroup(durRatio=(1, 1), items=groups)
