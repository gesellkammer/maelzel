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
    'Node',
    'asTree'
)


def _unpackRational(r: Rational) -> tuple[int, int]:
    return r.numerator, r.denominator


def _mergeProperties(a: dict | None, b: dict | None) -> dict | None:
    return a | b if (a and b) else (a or b)


class Node:
    """
    A Node is a container, grouping Notation under one time modifier

    A Node consists of a sequence of Notations or Nodes, allowing to
    define nested tuplets or beats. The notations inside a Node already hold the
    real beat-totalDuration. The durRatio is a ratio by which to multiply a given totalDuration to
    obtain the notated totalDuration.

    Attributes:
        durRatio: a tuple (num, den) indication the ratio by which to multiply the totalDuration
            of the items to obtain the notated items: the items inside this tree
            For example, an quarternote triplet would have a durRatio (3, 2) and the items
            inside it would have a totalDuration of 1/3. When multiplied by the durRatio each
            item would have a totalDuration of 1/2
        items: the items in this tree

    In the case of a simple triplet, the items would hold something like::

        >>> from maelzel.scoring import *
        >>> notations = [makeNote(60, totalDuration=F(1, 3)),
        ...              makeNote(61, totalDuration=F(2, 3))]
        >>> Node(ratio=(3, 2), items=notations)
    """
    def __init__(self,
                 ratio: tuple[int, int] | Rational,
                 items: list[Notation | 'Node'] = None,
                 properties: dict | None = None,
                 parent: Node | None = None):
        assert isinstance(items, list), f"Expected a list of Notation|Node, got {items}"
        self.durRatio: tuple[int, int] = ratio if isinstance(ratio, tuple) else _unpackRational(ratio)
        self.items: list[Notation | 'Node'] = items
        self.properties = properties
        self.parent: weakref.ReferenceType[Node] | None = weakref.ref(parent) if parent else None

    def setParent(self, parent: Node, recurse=True):
        self.parent = weakref.ref(parent)
        if recurse:
            for item in self.items:
                if isinstance(item, Node):
                    item.setParent(self, recurse=True)

    def findRoot(self) -> Node:
        return self if not self.parent else self.parent().findRoot()

    def setProperty(self, key: str, value) -> None:
        """Set a property for this Node"""
        if self.properties is None:
            self.properties = {key: value}
        else:
            self.properties[key] = value

    def getProperty(self, key: str, default=None):
        """Get the value of a property for this Node"""
        if self.properties is None:
            return default
        return self.properties.get(key, default)

    @property
    def offset(self) -> F:
        """The offset of this Node within the measure"""
        return self.items[0].offset

    @property
    def end(self) -> F:
        """The end of the last item within this Node"""
        return self.items[-1].end

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def append(self, item: Notation | Node) -> None:
        self.items.append(item)

    def totalDuration(self) -> F:
        """
        The actual totalDuration of the items in this tree

        """
        return sum((item.duration if isinstance(item, Notation) else item.totalDuration()
                    for item in self.items), F(0))

    def symbolicDuration(self) -> F:
        """
        The symbolic total duration of this tree.

        This represents the notated figure (1=quarter, 1/2=eighth note, 1/4=16th note, etc)
        """
        return sum(item.symbolicDuration() for item in self.items)

    def dump(self, indents=0, indent='  ', stream=None):
        stream = stream or sys.stdout
        print(f"{indent*indents}Node({self.durRatio[0]}/{self.durRatio[1]})", file=stream)
        for item in self.items:
            if isinstance(item, Notation):
                print(indent*(indents+1), item, sep='', file=stream)
            else:
                item.dump(indents=indents+1, stream=stream)

    def __repr__(self):
        parts = [f"Node({self.durRatio[0]}/{self.durRatio[1]}, "]
        for item in self.items:
            if isinstance(item, Notation):
                parts.append("  " + str(item))
            else:
                s = str(item)
                for line in s.splitlines():
                    parts.append("  " + line)
        parts.append(")")
        return "\n".join(parts)

    def _flattenUnnecessaryChildren(self):
        if self.durRatio != (1, 1):
            return
        items = []
        for item in self.items:
            if isinstance(item, Notation):
                items.append(item)
            else:
                if item.durRatio == (1, 1):
                    item._flattenUnnecessaryChildren()
                    items.extend(item.items)
                else:
                    items.append(item)
        self.items = items

    def mergeWith(self, other: Node) -> Node:
        """
        Merge this tree with other
        """
        # we don't check here, just merge
        node = Node(ratio=self.durRatio, items=self.items + other.items,
                    properties=_mergeProperties(self.properties, other.properties))
        node = node.mergedNotations()
        return node

    def mergedNotations(self, flatten=True) -> Node:
        """
        Returns a new tree with all items merged (recursively)

        Args:
            flatten: if True, superfluous children are flattened

        Returns:
            a new Node with merged items (whenever possible)
        """
        if flatten:
            self._flattenUnnecessaryChildren()
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
            return Node(ratio=F(1), items=[n])
        return Node(ratio=self.durRatio, items=out)

    def lastNotation(self) -> Notation:
        """
        Return the last Notation of this Node (recursively)
        """
        last = self.items[-1]
        if isinstance(last, Notation):
            return last
        else:
            return last.lastNotation()

    def firstNotation(self) -> Notation:
        """
        Return the first Notation of this Node (recursively)
        """
        first = self.items[0]
        return first if isinstance(first, Notation) else first.firstNotation()

    def recurse(self, reverse=False) -> Iterator[Notation]:
        """
        Iterate over the items in self, recursively

        Args:
            reverse: if True, iterate backwards

        Returns:
            an iterator over all Notations within this Node
        """
        items = self.items if not reverse else reversed(self.items)
        for item in items:
            if isinstance(item, Notation):
                yield item
            else:
                yield from item.recurse(reverse=reverse)

    def recurseWithNode(self, reverse=False
                        ) -> Iterator[tuple[Notation, Node]]:
        """
        Iterate over the items if self, recursively

        The same as :meth:`Node.recurse` but for each item yields a tuple (notation, node)
        where node is the node to which the notation belongs. This is useful in order to
        modify the tree in the case one needs to, for example, remove a notation from its tree

        Args:
            reverse: if True, iterate in reverse

        Returns:
            an iterator of tuples (notation, node)

        """
        items = self.items if not reverse else reversed(self.items)
        for item in items:
            if isinstance(item, Notation):
                yield (item, self)
            else:
                yield from item.recurseWithNode(reverse=reverse)

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
        for (n0, g0), (n1, g1) in iterlib.pairwise(self.recurseWithNode()):
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
        for (n0, node0), (n1, node1) in iterlib.pairwise(self.recurseWithNode()):
            if skip:
                skip = False
                continue
            if not (n0.tiedNext or n0.gliss) or n0.isRest or n1.isRest:
                continue

            if n0.quantizedPitches() == n1.quantizedPitches():
                if n0.isGracenote and n1.isRealnote:
                    # n0.copyAttributesTo(n1)
                    n0.copyAttachmentsTo(n1)
                    n0.copyFixedSpellingTo(n1)
                    node0.items.remove(n0)
                    count += 1
                    if n0.spanners:
                        for spanner in n0.spanners.copy():
                            n0.transferSpanner(spanner, n1)

                # elif n0.isRealnote and n1.isGracenote:
                elif n1.isGracenote:
                    n0.gliss = n1.gliss
                    n0.tiedNext = n1.tiedNext
                    n1.copyAttachmentsTo(n0)
                    node1.items.remove(n1)
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

    def fixEnharmonics(self,
                       options: enharmonics.EnharmonicOptions,
                       prevTree: Node = None
                       ):
        notations = list(self.recurse())
        n0 = notations[0]
        if not n0.isRest and n0.tiedPrev and prevTree is not None:
            # get previous note's spelling and fix n0 with it
            last = prevTree.lastNotation()
            if last.tiedNext and last.pitches == n0.pitches:
                spellings = last.notenames
                for i, spelling in enumerate(spellings):
                    n0.fixNotename(spelling, idx=i)

        from . import enharmonics
        enharmonics.fixEnharmonicsInPlace(notations, options=options)


def asTree(nodes: list[Node]
           ) -> Node:
    """
    Transform a list of Nodes into a tree structure

    A tree has a root and leaves, where each leave can be the root of a subtree

    Args:
        nodes: the tree to get/make the root for

    Returns:
        the root of a tree structure
    """
    if len(nodes) == 1 and nodes[0].durRatio == (1, 1):
        root = nodes[0]
    root = Node(ratio=(1, 1), items=nodes)
    assert root.totalDuration() == sum(n.totalDuration() for n in nodes)
    return root
