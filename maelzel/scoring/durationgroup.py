from __future__ import annotations
from emlib import iterlib
import sys

from .core import Notation
from .common import *
from numbers import Rational
from typing import TYPE_CHECKING
import weakref
if TYPE_CHECKING:
    from typing import Iterator
    durratio_t = tuple[int, int]
    from . import enharmonics

__all__ = (
    'DurationGroup',
    'asDurationGroupTree'
)


def _unpackRational(r: Rational) -> tuple[int, int]:
    return r.numerator, r.denominator


def _mergeProperties(a: dict | None, b: dict | None) -> dict | None:
    return a | b if (a and b) else (a or b)


class DurationGroup:
    """
    A DurationGroup is a container, grouping Notation under one time modifier

    A DurationGroup consists of a sequence of Notations or DurationGroups, allowing to
    define nested tuplets or beats. The notations inside a DurationGroup already hold the
    real beat-duration. The durRatio is a ratio by which to multiply a given duration to
    obtain the notated duration.

    Attributes:
        durRatio: a tuple (num, den) indication the ratio by which to multiply the duration
            of the items to obtain the notated items: the items inside this group
            For example, an quarternote triplet would have a durRatio (3, 2) and the items
            inside it would have a duration of 1/3. When multiplied by the durRatio each
            item would have a duration of 1/2
        items: the items in this group

    In the case of a simple triplet, the items would hold something like::

        >>> from maelzel.scoring import *
        >>> notations = [makeNote(60, duration=F(1, 3)),
        ...              makeNote(61, duration=F(2, 3))]
        >>> DurationGroup(durRatio=(3, 2), items=notations)
    """
    def __init__(self,
                 durRatio: durratio_t | Rational,
                 items: list[Notation | 'DurationGroup'] = None,
                 properties: dict | None = None,
                 parent: DurationGroup | None = None):
        assert isinstance(items, list), f"Expected a list of Notation|DurationGroup, got {items}"
        self.durRatio: durratio_t = durRatio if isinstance(durRatio, tuple) else _unpackRational(durRatio)
        self.items: list[Notation | 'DurationGroup'] = items
        self.properties = properties
        self.parent: weakref.ReferenceType[DurationGroup] | None = weakref.ref(parent) if parent else None

    def setParent(self, parent: DurationGroup, recurse=True):
        self.parent = weakref.ref(parent)
        if recurse:
            for item in self.items:
                if isinstance(item, DurationGroup):
                    item.setParent(self, recurse=True)

    def findRoot(self) -> DurationGroup:
        return self if not self.parent else self.parent().findRoot()

    def setProperty(self, key: str, value) -> None:
        """Set a property for this DurationGroup"""
        if self.properties is None:
            self.properties = {key: value}
        else:
            self.properties[key] = value

    def getProperty(self, key: str, default=None):
        """Get the value of a property for this DurationGroup"""
        if self.properties is None:
            return default
        return self.properties.get(key, default)

    @property
    def ratio(self) -> F:
        """The durRatio of this DurationGroup as a fraction"""
        return F(*self.durRatio)

    @property
    def offset(self) -> F:
        """The offset of this DurationGroup within the measure"""
        return self.items[0].offset

    @property
    def end(self) -> F:
        """The end of the last item within this DurationGroup"""
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

    def dump(self, indents=0, indent='  ', stream=None):
        stream = stream or sys.stdout
        print(f"{indent*indents}DurationGroup({self.durRatio[0]}/{self.durRatio[1]})", file=stream)
        for item in self.items:
            if isinstance(item, Notation):
                print(indent*(indents+1), item, sep='', file=stream)
            else:
                item.dump(indents=indents+1, stream=stream)

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
            else:
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
        group = DurationGroup(durRatio=self.durRatio, items=self.items + other.items,
                              properties=_mergeProperties(self.properties, other.properties))
        group = group.mergedNotations()
        return group

    def mergedNotations(self, flatten=True) -> DurationGroup:
        """
        Returns a new group with all items merged (recursively)

        Args:
            flatten: if True, superfluous subgrups are flattened

        Returns:
            a new DurationGroup with merged items (whenever possible)
        """
        if flatten:
            self._flattenUnnecessarySubgroups()
        i0 = self.items[0]
        out = [i0 if isinstance(i0, Notation) else i0.mergedNotations(flatten=False)]
        for i1 in self.items[1:]:
            i0 = out[-1]
            if isinstance(i0, Notation) and isinstance(i1, Notation):
                if i0.canMergeWith(i1):
                    out[-1] = i0.mergeWith(i1)
                else:
                    out.append(i1)
            else:
                # n+G, G+n or G+G
                out.append(i1 if isinstance(i1, Notation) else i1.mergedNotations(flatten=False))
        if len(out) == 1 and isinstance(out[0], Notation):
            n = out[0]
            if n.durRatios and n.durRatios[-1] != F(1):
                n.durRatios.pop()
            return DurationGroup(durRatio=F(1), items=[n])
        return DurationGroup(durRatio=self.durRatio, items=out)

    def recurse(self, reverse=False) -> Iterator[Notation]:
        """
        Iterate over the items in self, recursively

        Args:
            reverse: if True, iterate backwards

        Returns:
            an iterator over all Notations within this DurationGroup
        """
        items = self.items if not reverse else reversed(self.items)
        for item in items:
            if isinstance(item, Notation):
                yield item
            else:
                yield from item.recurse(reverse=reverse)

    def recurseWithGroup(self, reverse=False
                         ) -> Iterator[tuple[Notation, DurationGroup]]:
        """
        Iterate over the items if self, recursively

        The same as :meth:`DurationGroup.recurse` but for each item yields a tuple (notation, group)
        where group is the group to which the notation belongs. This is useful in order to
        modify the group in the case on needs to, for example, remove a notation from its group

        Args:
            reverse: if True, iterate in reverse

        Returns:
            an iterator of tuples (notation, durationgroup)

        """
        items = self.items if not reverse else reversed(self.items)
        for item in items:
            if isinstance(item, Notation):
                yield (item, self)
            else:
                yield from item.recurseWithGroup(reverse=reverse)

    def removeUnmatchedSpanners(self):
        from . import spanner
        spanner.removeUnmatchedSpanners(self.recurse())

    def repairLinks(self) -> int:
        """
        Repair ties and glissandi in place

        Returns:
            the number of modifications
        """
        count = 0
        n0: Notation
        n1: Notation
        ties = list[self.logicalTies()]

        def findTie(n: Notation, ties):
            return next((tie for tie in ties if n in tie), None)

        skip = False
        for (n0, g0), (n1, g1) in iterlib.pairwise(self.recurseWithGroup()):
            if skip:
                skip = False
                continue

            if not n0.tiedNext:
                n1.tiedPrev = False

            if n0.gliss and n0.tiedNext and n0.pitches != n1.pitches:
                count += 1
                n0.tiedNext = False
                n1.tiedPrev = False

            if n0.gliss and not n0.tiedNext and n0.pitches == n1.pitches:
                if n0.isRealnote and n1.isRealnote:
                    count += 1
                    n0.tiedNext = True
                    n1.tiedPrev = True
                    n0.gliss = False
                    if n0.tiedPrev:
                        tie  = findTie(n0, ties)
                        if tie:
                            for tie0, tie1 in iterlib.pairwise(tie):
                                tie0.tiedNext = True
                                tie1.tiedPrev = True
                                tie0.gliss = False
        return count

    def logicalTies(self) -> Iterator[Notation]:
        """
        Iterate over all logical ties within self (recursively)

        Returns: an iterator over the logical ties within self (recursively)

        """
        last = []
        for n in self.recurse():
            if n.tiedPrev:
                last.append(n)
            else:
                if last:
                    yield last
                last = [n]
        if last:
            yield last

    def glissMarkTiedNotesAsHidden(self) -> None:
        """
        Within a glissando, notes tied to previous and next notes can be hidden
        """
        it = self.recurse()
        for n in it:
            if n.gliss and not n.tiedPrev and n.tiedNext:
                # this starts a glissando and has tied notes after
                for n2 in it:
                    if n2.tiedPrev:
                        n2.setNotehead('hidden')
                    if n2.tiedNext:
                        break

    def removeUnnecessaryGracenotes(self) -> int:
        """
        Removes unnecessary gracenotes

        Returns:
            the number of modifications

        An unnecessary gracenote are:

        * has the same pitch as the next real note and starts a glissando. Such gracenotes might
          be created during quantization.
        * has the same pitch as the previous real note and ends a glissando
        * n0/real -- gliss -- n1/grace n2/real and n1.pitches == n2.pitches

        """
        # TODO: make quantization configurable
        count = 0
        skip = False
        n0: Notation
        n1: Notation
        for (n0, group0), (n1, group1) in iterlib.pairwise(self.recurseWithGroup()):
            if skip:
                skip = False
                continue
            if not (n0.tiedNext or n0.gliss) or n0.isRest or n1.isRest:
                continue

            if n0.quantizedPitches() == n1.quantizedPitches():
                if n0.isGracenote and n1.isRealnote:
                    n0.copyAttributesTo(n1)
                    group0.items.remove(n0)
                    count += 1
                    if n0.spanners:
                        for spanner in n0.spanners.copy():
                            n0.transferSpanner(spanner, n1)

                elif n0.isRealnote and n1.isGracenote:
                    n0.gliss = n1.gliss
                    n0.tiedNext = n1.tiedNext
                    n1.copyAttributesTo(n0)
                    group1.items.remove(n1)
                    count += 1
                    skip = True
                    if n1.spanners:
                        for spanner in n1.spanners.copy():
                            n1.transferSpanner(spanner, n0)

        self.removeUnmatchedSpanners()
        return count

    def repair(self):
        modcount = self.repairLinks()
        modcount += self.removeUnnecessaryGracenotes()
        if modcount == 0:
            return
        for i in range(10):
            if self.repairLinks() == 0:
                break
            if self.removeUnnecessaryGracenotes() == 0:
                break

    def fixEnharmonics(self, options: enharmonics.EnharmonicOptions=None):
        if options is None:
            from maelzel.core.workspace import Workspace
            options = Workspace.active.config.makeEnharmonicOptions()
        notations = list(self.recurse())
        from . import enharmonics
        enharmonics.fixEnharmonicsInPlace(notations, options=options)


def asDurationGroupTree(groups: list[DurationGroup]
                        ) -> DurationGroup:
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
