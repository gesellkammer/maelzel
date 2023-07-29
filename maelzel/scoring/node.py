from __future__ import annotations
from emlib import iterlib
from dataclasses import dataclass
import sys

from maelzel.common import asF, F
from .core import Notation
from . import attachment
from .common import F, logger
from numbers import Rational
import textwrap
from typing import TYPE_CHECKING

import weakref
if TYPE_CHECKING:
    from typing import Iterator
    durratio_t = tuple[int, int]
    from . import enharmonics

__all__ = (
    'Node',
    'SplitError'
)


@dataclass
class TreeLocation:
    notation: Notation
    """The notation"""

    notationIndex: int
    """Notation index within the measure"""

    parent: Node
    """The Node parent"""

    measureIndex: int | None = None
    """The measure index this notation belongs to, if known"""


def _unpackRational(r: Rational) -> tuple[int, int]:
    return r.numerator, r.denominator


def _mergeProperties(a: dict | None, b: dict | None) -> dict | None:
    return a | b if (a and b) else (a or b)


class SplitError(Exception):
    """Raised when a tree cannot be split at a given offset"""


class Node:
    """
    A Node is a recursive container, grouping Notations and other Nodes under one time modifier

    **Notations within a Node are quantized**. When a measure is quantized, its
    contents are grouped in a tree structure made out of Nodes
    (see :meth:`QuantizedMeasure.tree <maelzel.scoring.quant.QuantizedMeasure.tree>`)

    A Node consists of a sequence of Notations or Nodes, allowing to
    define nested tuplets or beats. The notations inside a Node already hold the
    real beat-duration. The durRatio is a ratio by which to multiply a given duration to
    obtain the notated duration.

    A Node is used to represent the result of quantization.

    Quantization happens first
    at the beat level and after that all quantized beats within a measure are merged
    together to create a tree structure spanning along the entire measure

    .. seealso:: :meth:`QuantizedMeasure.tree <maelzel.scoring.quant.QuantizedMeasure.tree>`

    Attributes:
        durRatio: a tuple ``(num, den)` indication the ratio by which to multiply the duration
            of the items to obtain the notated items: the items inside this tree
            For example, a quarternote triplet would have a durRatio (3, 2) and the items
            inside it would have a duration of 1/3. When multiplied by the durRatio each
            item would have a duration of 1/2
        items: the items in this tree (an item can be a Notation or a Node)

    In the case of a simple triplet, the items would hold something like::

        >>> from maelzel.scoring import *
        >>> notations = [makeNote(60, duration=F(1, 3)),
        ...              makeNote(61, duration=F(2, 3))]
        >>> Node(ratio=(3, 2), items=notations)
    """
    def __init__(self,
                 ratio: tuple[int, int] | Rational = (1, 1),
                 items: list[Notation | 'Node'] = None,
                 properties: dict | None = None,
                 parent: Node | None = None):
        if items:
            assert isinstance(items, list), f"Expected a list of Notation|Node, got {items}"
        self.durRatio: tuple[int, int] = ratio if isinstance(ratio, tuple) else (ratio.numerator, ratio.denominator)
        self.items: list[Notation | Node] = items if items is not None else []
        self.properties = properties
        self._parent : weakref.ReferenceType[Node] | None = weakref.ref(parent) if parent else None

    def __hash__(self):
        itemshash = hash(tuple(self.items))
        return hash(("Node", itemshash, self.durRatio, str(self.properties) if self.properties else None))

    @property
    def parent(self) -> Node | None:
        return self._parent() if self._parent is not None else None

    def empty(self) -> bool:
        for item in self.items:
            if isinstance(item, Node):
                if not item.empty():
                    return False
            elif not item.isRest:
                return False
            elif item.hasAttributes():
                return False
        return True

    def __len__(self):
        return len(self.items)

    def _setParentDownstream(self):
        """Set the parent of each Node downstream"""
        for item in self.items:
            if isinstance(item, Node):
                item._parent = weakref.ref(self)
                item._setParentDownstream()

    def findRoot(self) -> Node:
        """
        Find the root of this node

        Nodes are organized in tree structures. This method will climb the tree structure
        until the root of the tree (the node without a parent) is found
        """
        parent = self.parent
        return self if parent is None else parent.findRoot()

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
        """Add an item or Node to this Node"""
        self.items.append(item)

    def totalDuration(self) -> F:
        """
        The duration of the items in this tree, in quarter notes (recursively)

        """
        return sum((item.duration if isinstance(item, Notation) else item.totalDuration()
                    for item in self.items), F(0))

    def symbolicDuration(self) -> F:
        """
        The symbolic total duration of this tree.

        This represents the notated figure (1=quarter, 1/2=eighth note, 1/4=16th note, etc)
        """
        return sum(item.symbolicDuration() for item in self.items)

    def dump(self, numindents=0, indent='  ', stream=None):
        """Dump this node, recursively"""
        stream = stream or sys.stdout
        MAXWIDTH = 90
        print(f"{indent * numindents}Node ratio: {self.durRatio[0]}/{self.durRatio[1]}, offset={self.offset}, end={self.end}", file=stream)
        IND = indent * (numindents + 1)
        for item in self.items:
            if isinstance(item, Notation):
                itemlines = textwrap.wrap(repr(item), width=MAXWIDTH)
                print(IND, itemlines[0], file=stream, sep='')
                for line in itemlines[1:]:
                    print(IND, '  ', line, file=stream, sep='')
            else:
                item.dump(numindents=numindents + 1, stream=stream)

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

        It is assumed that these Nodes can merge

        Args:
            other: the other node

        Returns:
            the merged node / tree
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
            n = out[0].copy()
            if n.durRatios and n.durRatios[-1] != 1:
                n.durRatios.pop()
            return Node(ratio=(1, 1), items=[n])
        return Node(ratio=self.durRatio,
                    items=out,
                    parent=self.parent,
                    properties=self.properties.copy() if self.properties else None)

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
                yield item, self
            else:
                yield from item.recurseWithNode(reverse=reverse)

    def repairLinks(self) -> int:
        """
        Repair ties and glissandi inplace

        Returns:
            the number of modifications
        """
        count = 0
        n0: Notation
        n1: Notation
        ties = list(self.logicalTies())

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
                        tie = findTie(n0, ties)
                        if tie:
                            for tie0, tie1 in iterlib.pairwise(tie):
                                tie0.tiedNext = True
                                tie1.tiedPrev = True
                                tie0.gliss = False
        return count

    def logicalTieLocations(self, measureIndex: int | None = None
                            ) -> list[list[TreeLocation]]:
        """
        Like logicalTies, but for each notation returns a TreeLocation

        Args:
            measureIndex: if given, it is used to fill the .measureIndex attribute
                in the returned TreeLocation

        Returns:
            a list of ties, where each tie is a list of TreeLocations

        """
        lasttie = []
        idx = 0
        ties = []
        for notation, parent in self.recurseWithNode():
            item = TreeLocation(notation=notation, notationIndex=idx, parent=parent,
                                measureIndex=measureIndex)
            if notation.tiedPrev:
                lasttie.append(item)
            else:
                if lasttie:
                    ties.append(lasttie)
                lasttie = [item]
            idx += 1
        if lasttie:
            ties.append(lasttie)
        return ties

    def logicalTies(self) -> list[list[Notation]]:
        """
        Iterate over all logical ties within self (recursively)

        Returns:
            an iterator over the logical ties within self (recursively), where a
            logical tie is a list of Notations which are tied together

        """
        ties = []
        lasttie = []
        for notation in self.recurse():
            if notation.tiedPrev:
                lasttie.append(notation)
            else:
                if lasttie:
                    ties.append(lasttie)
                lasttie = [notation]
        if lasttie:
            ties.append(lasttie)
        return ties

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

        * has the same pitch as the next real note and starts a glissando.
          Such gracenotes might be created during quantization.
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
        return count

    def repair(self):
        if self.totalDuration() >= 2:
            self._splitUnnecessaryNodes(2)

        self.repairLinks()
        self.removeUnnecessaryGracenotes()

    def fixEnharmonics(self,
                       options: enharmonics.EnharmonicOptions,
                       prevTree: Node = None
                       ) -> None:
        """
        Find the best enharmonic spelling for the notations within this tree, inplace

        Args:
            options: the enharmonic options used
            prevTree: the previous tree (the tree corrsponding to the previous measure)

        """
        notations = list(self.recurse())
        n0 = notations[0]
        if not n0.isRest and n0.tiedPrev and prevTree is not None:
            # get previous note's spelling and fix n0 with it
            last = prevTree.lastNotation()
            if last.tiedNext and last.pitches == n0.pitches:
                spellings = last.resolveNotenames()
                for i, spelling in enumerate(spellings):
                    n0.fixNotename(spelling, idx=i)

        from . import enharmonics
        enharmonics.fixEnharmonicsInPlace(notations, options=options)

    def breakBeamsAt(self, offset: F) -> Notation | None:
        """
        Breaks beams at the given offset

        This method will not split a notation if the given offset
        is placed inside a syncopation

        Args:
            offset: the offset to break beams at

        Returns:
            the notation at the right of the split, or None if no split
            was performed

        """
        for item in self.items:
            if offset < item.offset:
                return None
            if offset < item.end:
                if isinstance(item, Notation) and item.offset == offset:
                    item.addAttachment(attachment.Breath(visible=False))
                    return item
                else:
                    item.breakBeamsAt(offset)
        return None

    def _splitUnnecessaryNodesAt(self, breakOffset: F, minDuration: F | int
                                 ) -> bool:
        items = []
        didsplit = False
        for item in self.items:
            if isinstance(item, Notation):
                items.append(item)
            else:
                if item.offset < breakOffset < item.end and item.totalDuration() >= minDuration:
                    for sub1, sub2 in iterlib.pairwise(item.items):
                        if sub2.offset == breakOffset and sub1.durRatios == sub2.durRatios:
                            logger.debug(f"Found unnecessary node at {breakOffset}, splitting")
                            left, right = item._splitAtBoundary(breakOffset)
                            items.append(left)
                            items.append(right)
                            didsplit = True
                            break
                    else:
                        items.append(item)
                else:
                    items.append(item)
        self.items = items
        return didsplit

    def _splitUnnecessaryNodes(self, duration: F | int) -> None:
        """
        Split any _beatNodes which are unnecessarily joined

        Args:
            duration: the duration of the node to split
        """
        if self.totalDuration() < duration:
            return
        items = []
        for item in self.items:
            if isinstance(item, Notation):
                items.append(item)
            else:
                if item.totalDuration() == duration:
                    breakoffset = item.offset + 1
                    for sub1, sub2 in iterlib.pairwise(item.recurse()):
                        if sub2.offset == breakoffset and sub1.durRatios  == sub2.durRatios:
                            left, right = item._splitAtBoundary(breakoffset)
                            logger.debug(f"Splitting node {self} at {breakoffset}")
                            items.append(left)
                            items.append(right)
                            break
                    else:
                        logger.debug(f"Did not split node {self} at {breakoffset}")
                        items.append(item)
                else:
                    items.append(item)
        self.items = items

    def _splitAtBoundary(self, offset) -> tuple[Node, Node]:
        assert self.offset < offset < self.end
        left = Node(ratio=self.durRatio)
        right = Node(ratio=self.durRatio)
        for item in self.items:
            if isinstance(item, Notation):
                if item.offset < offset:
                    assert item.end <= offset
                    left.append(item)
                else:
                    assert item.offset >= offset
                    right.append(item)
            else:
                if item.end < offset:
                    left.append(item)
                elif item.offset > offset:
                    right.append(item)
                else:
                    leftsub, rightsub = item._splitAtBoundary(offset)
                    left.append(leftsub)
                    right.append(rightsub)
        return left, right

    def splitNotationAtBoundary(self, offset: F, key=None) -> Notation | None:
        """
        Split any notation which crosses the given offset, inplace

        A notation will be split if it crosses the given offset. Raises
        SplitError if it cannot split at the given offset

        Args:
            offset: the offset of the desired split. It should be a beat boundary
            key: if given, a function which returns True if the note should be split

        Returns:
            the notation next to the split offset, or None if no split operation
            was performed
        """
        offset = asF(offset)
        n = self._splitNotationAtBoundary(offset=offset, key=key)
        if n is not None:
            logger.debug(f"Split notation at boundary {offset}, did that generate unnecessary _beatNodes?")
            didsplit = self._splitUnnecessaryNodesAt(offset, minDuration=2)
            logger.debug("--- Yes" if didsplit else "--- No...")
        return n

    def _remerge(self):
        """
        Merge notations recursively **in place**
        """
        self.items = self.mergedNotations(flatten=True).items

    def _splitNotationAtBoundary(self, offset: F, key=None) -> Notation | None:
        """
        Split any notation which crosses the given offset, inplace

        A notation will be split if it crosses the given offset. Raises
        SplitError if it cannot split at the given offset

        Args:
            offset: the offset of the desired split. It should be a beat boundary
            key: if given, a function which returns True if the note should be split

        Returns:
            the notation next to the split offset, or None if no notation was broken
        """
        if not (self.offset <= offset < self.end):
            return None

        for i, item in enumerate(self.items):
            if item.offset >= offset:
                return None
            elif offset < item.end:
                if isinstance(item, Node):
                    return item._splitNotationAtBoundary(offset=offset, key=key)
                symdur = item.symbolicDuration()
                if symdur.denominator not in (1, 2, 4, 8, 16):
                    raise SplitError(f"Cannot split {item} at {offset}")
                if symdur.numerator not in (1, 2, 3, 4, 7):
                    raise SplitError(f"Cannot split {item} at {offset}, "
                                     f"Symbolic duration: {symdur}")
                if key and not key(item):
                    logger.debug(f"Found a syncopation but the callback was negative, so "
                                 f"{item} will not be split")
                    break
                logger.debug(f"Splitting node (offset={self.offset}, end={self.end} at {offset=}")
                parts = item.splitAtOffsets([offset])
                if not len(parts) == 2:
                    raise SplitError(f"Expected two parts as a result of the split "
                                     f"operation, got {parts} ({item=})")
                if any(not part.hasRegularDuration() for part in parts):
                    raise SplitError(f"Cannot split {item} at {offset} (parts: {parts}, "
                                     f"symbolic durations: {[p.symbolicDuration() for p in parts]}), "
                                     f"durRatios: {[p.durRatios for p in parts]}")
                parts[1].mergeable = False
                newitems = self.items[:i] + parts + self.items[i+1:]
                self.items = newitems
                self._remerge()
                return parts[1]
        return None

    @staticmethod
    def asTree(nodes: list[Node]) -> Node:
        """
        Transform a list of Nodes into a tree structure

        A tree has a root and leaves, where each leave can be the root of a subtree

        Args:
            nodes: the tree to get/make the root for

        Returns:
            the root of a tree structure
        """
        if len(nodes) == 1 and nodes[0].durRatio == (1, 1):
            return nodes[0]
        root = Node(ratio=(1, 1), items=nodes)
        assert root.totalDuration() == sum(n.totalDuration() for n in nodes)
        return root
