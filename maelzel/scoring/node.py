from __future__ import annotations
from dataclasses import dataclass
import sys
import textwrap
import weakref

from maelzel.common import F, asF, F0
from maelzel.scoring.quantdefs import QuantizedBeatDef
from .common import logger
from maelzel._logutils import LazyStr
from .core import Notation
from . import attachment
from itertools import pairwise
from collections import UserList
from . import quantdata

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterator, Callable, Sequence
    durratio_t = tuple[int, int]
    from . import enharmonics
    from maelzel.scoring import quantdefs
    from maelzel.scoring import quant
    from .common import division_t


__all__ = (
    'Node',
    'SplitError',
    'LogicalTie',
    'TreeLocation'
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


class LogicalTie(UserList[TreeLocation]):

    @property
    def locations(self) -> list[TreeLocation]:
        return self.data

    def duration(self) -> F:
        dur = sum((loc.notation.duration for loc in self.data), start=F(0))
        assert isinstance(dur, F)
        return dur

    def notations(self) -> list[Notation]:
        return [loc.notation for loc in self]

    def __contains__(self, item: Notation) -> bool:
        return any(loc.notation is item for loc in self)

    def index(self, item: TreeLocation | Notation, start=0, stop=sys.maxsize, /) -> int:
        if isinstance(item, Notation):
            return self.notations().index(item, start, stop)
        else:
            raise TypeError(f"item of type {type(item)} not supported")


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
                 items: list[Notation | Node],
                 ratio: tuple[int, int] | F = (1, 1),
                 parent: Node | None = None,
                 properties: dict | None = None,
                 measure: quant.QuantizedMeasure | None = None,
                 readonly=False,
                 _duration=F0
                 ):
        if not all(_.isQuantized() for _ in items if isinstance(_, Notation)):
            raise ValueError(f"A Node accepts only quantized Notations, {items=}")
        assert (isinstance(ratio, tuple) and len(ratio) == 2) or isinstance(ratio, F), f"{ratio=}"
        self.durRatio: tuple[int, int] = ratio if isinstance(ratio, tuple) else (ratio.numerator, ratio.denominator)
        self.items: list[Notation | Node] = items
        self._properties = properties
        self._measureReference: weakref.ReferenceType[quant.QuantizedMeasure] | None = weakref.ref(measure) if measure else None
        self._parent: weakref.ReferenceType[Node] | None = weakref.ref(parent) if parent else None
        self._duration: F = _duration
        self.readonly = readonly

    def copy(self) -> Node:
        items = []
        for item in self.items:
            if isinstance(item, Notation):
                items.append(item.copy(spanners=True))
            else:
                items.append(item.copy())
        return Node(items=items, ratio=self.durRatio, parent=self.parent,
                    properties=self._properties,
                    measure=self._measureReference() if self._measureReference else None,
                    readonly=self.readonly,
                    _duration=self._duration)

    def clone(self,
              items: list[Notation | Node] | None = None,
              ratio: tuple[int, int] | F = (1, 1),
              parent: Node | None = None,
              properties: dict | None = None,
              measure: quant.QuantizedMeasure | None = None,
              readonly: bool | None = None
              ) -> Node:
        out = Node(items=items if items is not None else self.items,
                   ratio=ratio or self.durRatio,
                   parent=parent or self.parent,
                   properties=properties if properties is not None else self._properties,
                   measure=measure if measure is not None else self._measureReference() if self._measureReference else None,
                   readonly=readonly if readonly is not None else self.readonly)
        return out

    def _checkCanModify(self) -> None:
        if self.readonly:
            raise RuntimeError("Cannot modify read-only node")

    def __hash__(self):
        itemshash = hash(tuple(self.items))
        return hash(("Node", itemshash, self.durRatio, str(self._properties) if self._properties else None))

    def root(self) -> Node:
        """
        Find the root of this node

        Nodes are organized in tree structures. This method will climb the tree structure
        until the root of the tree (the node without a parent) is found
        """
        if (parent := self.parent) is None:
            return self
        else:
            return parent.root()

    @property
    def parentMeasure(self) -> quant.QuantizedMeasure | None:
        if self.parent:
            return self.root().parentMeasure
        # We are root
        return self._measureReference() if self._measureReference else None

    @parentMeasure.setter
    def parentMeasure(self, value: quant.QuantizedMeasure):
        self._checkCanModify()
        self._measureReference = weakref.ref(value)

    @property
    def parent(self) -> Node | None:
        return self._parent() if self._parent is not None else None

    @parent.setter
    def parent(self, other: Node | None):
        """Set the parent of this node"""
        self._checkCanModify()
        self._parent = weakref.ref(other) if other else None

    def empty(self) -> bool:
        """Is this node empty? """
        for item in self.items:
            if isinstance(item, Node):
                if not item.empty():
                    return False
            elif not item.isRest or item.hasAttributes():
                return False
        return True

    def __len__(self):
        return len(self.items)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (self.durRatio == other.durRatio and
                self._properties == other._properties and
                all(item0 == item1 for item0, item1 in zip(self.items, other.items)))

    def setReadOnly(self, value: bool, recurse=True) -> None:
        self.readonly = value
        if recurse:
            for n in self.items:
                if isinstance(n, Node):
                    n.setReadOnly(value, recurse=True)

    def setParentRecursively(self):
        """Set the parent of each Node downstream"""
        self._checkCanModify()
        for item in self.items:
            if isinstance(item, Node):
                item.parent = self
                item.setParentRecursively()

    def setProperty(self, key: str, value) -> None:
        """Set a property for this Node"""
        if self._properties is None:
            self._properties = {key: value}
        else:
            self._properties[key] = value

    def getProperty(self, key: str, default=None):
        """Get the value of a property for this Node"""
        if self._properties is None:
            return default
        return self._properties.get(key, default)

    @property
    def qoffset(self) -> F:
        """
        The offset of this Node within the measure

        This is the same as .offset, since all Nodes are quantized,
        but it is present to match quantized Notations, where .offset
        can be None if they are not quantized, and .qoffset is always
        a F, raising an error if the Notation is not quantized
        """
        item0 = self.items[0]
        return item0.qoffset if isinstance(item0, Notation) else item0.offset

    @property
    def offset(self) -> F:
        """The offset of this Node within the measure"""
        item0 = self.items[0]
        return item0.qoffset if isinstance(item0, Notation) else item0.offset

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
        if not self._duration:
            self._duration = sum((item.duration if isinstance(item, Notation) else item.totalDuration()
                                 for item in self.items), F(0))
        return self._duration

    def fusedDurRatio(self) -> F:
        num, den = self.durRatio
        ratio = F(num, den)
        if self.parent:
            ratio *= self.parent.fusedDurRatio()
        return ratio

    def symbolicDuration(self) -> F:
        """
        The symbolic total duration of this tree.

        This represents the notated figure (1=quarter, 1/2=eighth note, 1/4=16th note, etc)
        at the level of this node.
        """
        dur = self.totalDuration()
        if self.parent:
            dur *= self.parent.fusedDurRatio()
        return dur
        # return sum((item.symbolicDuration() for item in self.items), F(0))

    def check(self):
        for child in self.items:
            if isinstance(child, Node):
                if child.parent is not self:
                    raise ValueError(f"Invalid parent for {child=}. Parent should be {self}, found {child.parent}")
                assert child.parent is self
                child.check()
            else:
                assert child.isQuantized(), f"Unquantized notation {child} in node {self}"

        dur = self.totalDuration()
        if dur == 0:
            raise ValueError(f"Node with 0 duration: {self}")

    def dump(self, numindents=0, indent='  ', stream=None) -> None:
        """
        Dump this node, recursively

        Args:
            numindents: the number of indents to use for this node
            indent: indentation string
            stream: the stream to dumo to
        """
        stream = stream or sys.stdout
        MAXWIDTH = 90
        print(f"{indent * numindents}Node ratio: {self.durRatio[0]}/{self.durRatio[1]}, "
              f"offset={self.offset}, end={self.end}, dur={self.totalDuration()}, "
              f"symbolicdur={self.symbolicDuration()}", file=stream)
        IND = indent * (numindents + 1)
        for item in self.items:
            if isinstance(item, Notation):
                itemlines = textwrap.wrap(repr(item), width=MAXWIDTH)
                print(IND, itemlines[0], file=stream, sep='')
                for line in itemlines[1:]:
                    print(IND, '  ', line, file=stream, sep='')
            else:
                item.dump(numindents=numindents + 1, stream=stream)

    def _treeRepr(self, indent=0) -> str:
        indent0 = " " * indent
        num, den= self.durRatio
        header = f"{indent0}Node({num}/{den}, dur={self.totalDuration()} "
        if isinstance(self.items[0], Notation):
            parts = [header + str(self.items[0])]
            indentstr = " " * len(header)
            startidx = 1
        else:
            parts = [header]
            indentstr = " " * (indent + 4)
            startidx = 0
        for item in self.items[startidx:]:
            if isinstance(item, Notation):
                parts.append(indentstr + str(item))
            else:
                s = item._treeRepr(len(indentstr))
                parts.append(s)

        parts[-1] += ")"
        return "\n".join(parts)

    def __repr__(self):
        return self._treeRepr(indent=0)

    def _setitems(self, items: list[Notation | Node]) -> None:
        self._checkCanModify()
        self.items = items

    def _repairDurRatios(self, durRatios: list[tuple[int, int]] | None = None):
        if durRatios is None:
            durRatios = [self.durRatio]
        else:
            durRatios.append(self.durRatio)
        notationDurRatio = tuple(F(num, den) for num, den in durRatios)
        for item in self.items:
            if isinstance(item, Node):
                item._repairDurRatios(durRatios)
            else:
                item.durRatios = notationDurRatio
        durRatios.pop()

    def _flattenUnnecessaryChildren(self) -> None:
        """
        Flattens regular nodes (nodes with ratio (1:1)), in place

        """
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
        self._setitems(items)

    def mergeWith(self, other: Node, readonly: bool | None = None) -> Node:
        """
        Merge this tree with other

        It is assumed that these Nodes can merge

        Args:
            other: the other node

        Returns:
            the merged node / tree
        """
        # we don't check here, just merge
        if readonly is None:
            readonly = self.readonly or other.readonly
        node = Node(ratio=self.durRatio,
                    items=self.items + other.items,
                    properties=_mergeProperties(self._properties, other._properties))
        node = node.mergedNotations()
        node.readonly = readonly
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

        def mergeonce(items: list[Notation | Node]) -> tuple[list[Notation | None], bool]:
            modified = False
            # out = [i0 if isinstance(i0, Notation) else i0.mergedNotations(flatten=False)]
            out = [items[0]]
            for i1 in items[1:]:
                i0 = out[-1]
                if isinstance(i0, Notation) and isinstance(i1, Notation):
                    if i0.canMergeWith(i1):
                        modified = True
                        out[-1] = i0.mergeWith(i1)
                    else:
                        out.append(i1)
                else:
                    # n+G, G+n or G+G
                    out.append(i1)
                    # out.append(i1 if isinstance(i1, Notation) else i1.mergedNotations(flatten=False))
            return out, modified

        items = [item if isinstance(item, Notation) else item.mergedNotations() for item in self.items]

        for i in range(10):
            items, modified = mergeonce(items)
            if not modified:
                if i > 1:
                    logger.debug("Merged notations in %d iterations", i)
                break
        else:
            logger.debug("Merging didn't converge, before: %s, after: %s", LazyStr.str(self.items), LazyStr.str(items))

        if len(items) == 1 and isinstance(items[0], Notation):
            n = items[0].copy()
            n.spanners = items[0].spanners
            if n.durRatios and n.durRatios[-1] != 1:
                n.durRatios = n.durRatios[:-1]
            node = Node(ratio=(1, 1), items=[n], readonly=self.readonly)
        else:
            node = Node(ratio=self.durRatio,
                        items=items,
                        parent=self.parent,
                        properties=self._properties.copy() if self._properties else None,
                        readonly=self.readonly)
            node.setParentRecursively()
        return node

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

    def findNextNotation(self, notation: Notation) -> Notation | None:
        """
        Find the notation next to the given notation

        Args:
            notation: the notation to query. The returned notation, if found, will
                be the notation next to this

        Returns:
            the notation next to the given notation, or None if no notation found.
            The returned notation does not need to be on the same node.

        """
        found = False
        for n, node in self.recurseWithNode():
            if found:
                return n
            elif n is notation:
                found = True
        return None

    def findNodeForNotation(self, notation: Notation) -> Node | None:
        """
        Find the node of the given notation

        Args:
            notation: the notation for which to find its node

        Returns:
            the node parent to the given notation

        """
        if notation in self.items:
            return self
        for item in self.items:
            if isinstance(item, Node):
                subnode = item.findNodeForNotation(notation)
                if subnode:
                    return subnode
        return None

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
        self._checkCanModify()
        count = 0
        n0: Notation
        n1: Notation
        ties = list(self.logicalTies())

        def findTie(n: Notation, ties):
            return next((tie for tie in ties if n in tie), None)

        skip = False
        for (n0, g0), (n1, g1) in pairwise(self.recurseWithNode()):
            if skip:
                skip = False
                continue

            if n0.tiedNext and len(n0.pitches) == len(n1.pitches):
                assert n0.pitches == n1.pitches, f"{n0=}, {n1=}"

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
                            for tie0, tie1 in pairwise(tie):
                                tie0.tiedNext = True
                                tie1.tiedPrev = True
                                tie0.gliss = False
        return count

    def logicalTieLocations(self, measureIndex: int | None = None
                            ) -> list[LogicalTie]:
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
        return [LogicalTie(tie) for tie in ties]

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

    def _repairGracenotesAtBeginning(self) -> None:
        """
        A gracenote or group thereof should not start a tuplet
        Returns:

        """
    def removeUnnecessaryGracenotes(self) -> int:
        """
        Removes unnecessary gracenotes

        Returns:
            the number of modifications

        An unnecessary gracenote:

        * has the same pitch as the next real note and starts a glissando.
          Such gracenotes might be created during quantization.
        * has the same pitch as the previous real note and ends a glissando
        * n0/real -- gliss -- n1/grace n2/real and n1.pitches == n2.pitches
        * has the same pitch as the previous real note and has no attribute itself

        """
        # TODO: make quantization configurable
        count = 0
        skip = False
        n0: Notation
        n1: Notation

        for (n0, node0), (n1, node1) in pairwise(self.recurseWithNode()):
            if skip:
                skip = False
                continue
            if not (n0.tiedNext or n0.gliss) or n0.isRest or n1.isRest:
                continue

            if n0.quantizedPitches() == n1.quantizedPitches():
                if n0.isGracenote and n1.isRealnote:
                    n0.copyAttachmentsTo(n1)
                    n0.copyFixedSpellingTo(n1)
                    logger.debug("Removing gracenote %s from node %s", LazyStr.str(n0), LazyStr.str(node0))
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
                    logger.debug("Removing gracenote %s from node %s", LazyStr.str(n1), LazyStr.str(node1))
                    node1.items.remove(n1)
                    count += 1
                    skip = True
                    if n1.spanners:
                        for spanner in n1.spanners.copy():
                            n1.transferSpanner(spanner, n0)
        return count

    def repair(self):
        self._flattenUnnecessaryChildren()
        self._removeUnnecessaryNodesInPlace()
        self._remerge()
        self.repairLinks()
        self.removeUnnecessaryGracenotes()
        self.setParentRecursively()
        # self._unmergeUnnecessaryNodes(minDur=F(1, 4))

    def fixEnharmonics(self,
                       options: enharmonics.EnharmonicOptions,
                       prevTree: Node | None = None
                       ) -> None:
        """
        Find the best enharmonic spelling for the notations within this tree, inplace

        Args:
            options: the enharmonic options used
            prevTree: the previous tree (the tree corrsponding to the previous measure), if
                applicable.

        """
        self._checkCanModify()
        notations = list(self.recurse())
        n0 = notations[0]
        if prevTree is None:
            measure = self.parentMeasure
            if measure and (prevMeasure := measure.previousMeasure()):
                prevTree = prevMeasure.tree
        if not n0.isRest and n0.tiedPrev and prevTree is not None:
            # get previous note's spelling and fix n0 with it
            last = prevTree.lastNotation()
            if last.tiedNext and last.pitches == n0.pitches:
                spellings = last.resolveNotenames()
                for i, spelling in enumerate(spellings):
                    n0.fixNotename(spelling, index=i)

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
        self._checkCanModify()
        for item in self.items:
            if offset < item.qoffset:
                return None
            if offset < item.end:
                if isinstance(item, Notation):
                    if item.offset == offset:
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
                    for sub1, sub2 in pairwise(item.items):
                        if sub2.offset == breakOffset:
                            if (isinstance(sub1, Notation) and isinstance(sub2, Notation) and
                                    sub1.durRatios == sub2.durRatios):
                                logger.debug("Found unnecessary node at %d/%d, splitting", breakOffset.numerator, breakOffset.denominator)
                                left, right = item._splitAtBoundary(breakOffset)
                                items.append(left)
                                items.append(right)
                                didsplit = True
                                break
                            elif (isinstance(sub1, Node) and isinstance(sub2, Node) and
                                  sub1.durRatio == sub2.durRatio):
                                left, right = item._splitAtBoundary(breakOffset)
                                items.append(left)
                                items.append(right)
                                didsplit = True
                                break
                    else:
                        items.append(item)
                else:
                    items.append(item)
        self._setitems(items)
        self.setParentRecursively()
        return didsplit

    def isFlat(self) -> bool:
        """
        True if self has no Nodes as children
        """
        return all(isinstance(item, Notation) for item in self.items)

    def notationAt(self, offset: F) -> Notation:
        if not (self.offset <= offset < self.end):
            raise ValueError(f"Offset {offset} outside of node ({self.offset=}, {self.end=}")
        return next(n for n in self.recurse() if n.offset <= offset < n.end)

    # def _unmergeUnnecessaryNodes(self, minDur: F = F(1, 4)) -> bool:
    #     """
    #
    #     Returns:
    #         True if changes were performed
    #     """
    #     from . import quantutils
    #     items = []
    #     changes = False
    #     for item in self.items:
    #         if isinstance(item, Notation):
    #             items.append(item)
    #         else:
    #             assert isinstance(item, Node)
    #             if item.durRatio == (1, 1) or item.isFlat():
    #                 changes |= item._unmergeUnnecessaryNodes(minDur=minDur)
    #                 items.append(item)
    #             else:
    #                 assert item.durRatio != (1, 1) and not item.isFlat()
    #                 symdur = item.symbolicDuration()
    #                 partdur = symdur / 2
    #                 if partdur < minDur or not quantutils.isRegularDuration(partdur):
    #                     items.append(item)
    #                 else:
    #                     splitoffset = item.offset + item.totalDuration()/2
    #                     n = item.notationAt(splitoffset)
    #                     if n.offset == splitoffset:
    #                         # Note starts at split point, so no syncopation
    #                         left, right = item._splitAtBoundary(splitoffset)
    #                         changes = True
    #                         items.append(left)
    #                         items.append(right)
    #     if changes:
    #         self.items = items
    #     return changes

    # def _splitUnnecessaryNodes(self, duration: F | int) -> None:
    #     """
    #     Split any nodes which are unnecessarily joined
    #
    #     * For a node to be unnecessary it needs to be divisible in two
    #       and not have a syncopation
    #     * For a node to be split it needs to match the given duration
    #     * 1/1 nodes are never split
    #
    #     Args:
    #         duration: the duration of the node to split. Only nodes with this exact
    #             duration will be considered for splitting. Nodes are always split in half
    #     """
    #     if self.totalDuration() < duration:
    #         return
    #     items = []
    #     for item in self.items:
    #         if isinstance(item, Notation):
    #             items.append(item)
    #         else:
    #             # a Node
    #             if item.totalDuration() == duration:
    #                 splitoffset = item.offset + duration//2
    #                 for sub1, sub2 in pairwise(item.recurse()):
    #                     if sub2.offset == splitoffset and sub1.durRatios == sub2.durRatios:
    #                         assert item.offset < splitoffset < item.end, f"{item=}, {splitoffset=}"
    #                         logger.debug("Splitting node %s at %s", LazyStr.str(self), str(splitoffset))
    #                         # logger.debug(f"Splitting node {self} at {splitoffset}")
    #                         left, right = item._splitAtBoundary(splitoffset)
    #                         items.append(left)
    #                         items.append(right)
    #                         break
    #                 else:
    #                     logger.debug("Did not split node %s at %s", LazyStr.str(self), str(splitoffset))
    #                     # logger.debug(f"Did not split node {self} at {splitoffset}")
    #                     items.append(item)
    #             else:
    #                 items.append(item)
    #     self._setitems(items)
    #     self.setParentRecursively()

    def _splitAtBoundary(self, offset) -> tuple[Node, Node]:
        if not (self.offset < offset < self.end):
            raise ValueError(f"Offset {offset} not within this node. "
                             f"({self.offset=}, {self.end=}), node={self}")
        left, right = [], []
        for item in self.items:
            if isinstance(item, Notation):
                if item.end <= offset:
                    left.append(item)
                elif item.offset >= offset:
                    right.append(item)
                else:
                    raise ValueError(f"Invalid item for node, {item=}, {self=}")
            else:
                if item.end <= offset:
                    left.append(item)
                elif item.offset >= offset:
                    right.append(item)
                elif item.totalDuration() == 0:
                    # a gracenote exactly at the offset
                    left.append(item)
                else:
                    assert item.offset <= offset < item.end
                    leftsub, rightsub = item._splitAtBoundary(offset)
                    left.append(leftsub)
                    right.append(rightsub)
        lnode = Node(left, ratio=self.durRatio, parent=self.parent, readonly=self.readonly)
        rnode = Node(right, ratio=self.durRatio, parent=self.parent, readonly=self.readonly)
        return lnode, rnode

    def splitNotationAtOffset(self, offset: F, tie=True, mergeable=True,
                              beatstruct: Sequence[QuantizedBeatDef]=None
                              ) -> list[Notation] | None:
        """
        Splits a notation present at the given offset, returns a list of parts

        Assumes that the offset divides a notation within this tree. Since
        a tree represents a quantized structure, this split can only be performed
        at offsets which themselves belong to this quantization. An invalid
        offset will raise a ValueError exception2

        Args:
            offset: the offset to split at
            tie: tie the resulting notations, if a notation was in fact split
            mergeable: the split notations should be mergeable. Set it to False to avoid
                any future remerges

        Returns:
            the resulting notations (they can be more than one if any part
            results in an irregular duration). None if no need to split

        """
        if beatstruct is None:
            meas = self.parentMeasure
            if meas is None:
                raise ValueError("Parent measure not set for this node, cannot split")
            beatstruct = meas.beatStructure()
        offset = asF(offset)
        if not self.offset < offset < self.end:
            raise ValueError(f"Offset not within this node: {offset=}, node={self}")
        for i, item in enumerate(self.items):
            if item.qoffset == offset:
                # No need to split
                return None
            elif item.qoffset < offset < item.end:
                if isinstance(item, Node):
                    return item.splitNotationAtOffset(offset, tie=tie, beatstruct=beatstruct)
                left, right = item.splitAtOffset(offset, tie=tie)
                if not mergeable:
                    right.mergeablePrev = False
                # Check that the items have regular durations...
                items = self.items[:i]
                out = []
                if left.hasRegularDuration():
                    items.append(left)
                    out.append(left)
                else:
                    leftparts = Node.breakIrregularDurationInNode(left, beatstruct=beatstruct)
                    items.extend(leftparts)
                    out.extend(leftparts)
                if right.hasRegularDuration():
                    items.append(right)
                    out.append(right)
                else:
                    rightparts = Node.breakIrregularDurationInNode(right, beatstruct=beatstruct)
                    items.extend(rightparts)
                    out.extend(rightparts)
                items.extend(self.items[i+1:])
                self._setitems(items)
                return out

        # This should never happen
        raise ValueError(f"??? Offset not within any notation in this node: {offset=}, items={self.items}")

    def _remerge(self):
        """
        Merge notations recursively **in place**
        """
        self._checkCanModify()
        self._repairDurRatios()
        self._setitems(self.mergedNotations(flatten=True).items)
        self.setParentRecursively()

    def splitNotationAtBeat(self,
                            beats: Sequence[QuantizedBeatDef],
                            beatIndex: int,
                            callback: Callable[[Notation, F], bool] | None = None,
                            repair=True
                            ) -> list[Notation] | None:
        """
        Split any notation which crosses the given beat offset, inplace

        A notation will be split if it crosses the given offset. Raises
        SplitError if it cannot split at the given offset

        Args:
            beats: the sequence of beats in the measure where this node is defined
                (see QuantizedMeasure.beatStructure)
            beatIndex: the index of the beat at which to split
            callback: if given, a function which returns True if the note should be split

        Returns:
            the parts resulting from splitting this notation at the given beat boundary,
            or None if no split operation was performed
        """
        parts = self._splitNotationAtBeat(beats=beats, beatIndex=beatIndex, callback=callback)
        if parts is not None and repair:
            _ = self._splitUnnecessaryNodesAt(beats[beatIndex].offset, minDuration=2)
        return parts

    def _splitNotationAtBeat(self,
                             beats: Sequence[quantdefs.QuantizedBeatDef],
                             beatIndex: int,
                             callback: Callable[[Notation, Node, F], bool] | None = None
                             ) -> list[Notation] | None:
        """
        Split any notation which crosses the given offset, inplace

        A notation will be split if it crosses the given offset. Here we do not evaluate
        if the split is necessary, this method just performs the action. It raises
        SplitError if the split cannot be performed

        Args:
            beats: the beat definitions of this tree
            beatIndex: the index of the beat at which to split
            callback: if given, a function which returns True if the note should be split.
                The callback has the form (notation: Notation, offset: F) -> bool

        Returns:
            the parts resulting of the split operation, or None if no notation was broken
            The notation right to the split offset, or None if no notation was broken. The returned
            notations are part of this Node
        """
        self._checkCanModify()
        beat = beats[beatIndex]
        offset = beat.offset
        if not (self.offset <= offset < self.end):
            return None

        measidx = self.parentMeasure.measureIndex()
        for i, item in enumerate(self.items):
            if item.qoffset >= offset:
                # Past the last item
                return None
            elif item.end <= offset:
                # Still no intersection, continue looking
                continue

            assert item.qoffset < offset < item.end
            if isinstance(item, Node):
                return item._splitNotationAtBeat(beats=beats, beatIndex=beatIndex, callback=callback)

            assert item.isQuantized(), f"Item not quantized: {item}"
            if not item.hasRegularDuration():
                raise SplitError(f"Item does not have a regular duration: {item=}, "
                                 f"symbolic duration={item.symbolicDuration()}")
            if callback:
                if not callback(item, self, offset):
                    logger.debug("Syncopation at %d:%s, %s negative, %s will NOT be split", measidx, str(offset), str(callback), LazyStr.str(item))
                    #logger.debug(f"Syncopation at {measidx}:{offset}, {callback} negative, "
                    #              f"{item} will NOT be split")
                    break
                logger.debug("Syncopation at %d:%s - %s was positive, splitting", measidx, str(offset), str(callback))

            parts = item.splitAtOffsets([offset])
            if not len(parts) == 2:
                raise SplitError(f"Expected two parts as a result of the split "
                                 f"operation, got {parts} ({item=})")
            parts[1].mergeablePrev = False
            regularParts = []
            left, right = parts
            if left.hasRegularDuration():
                regularParts.append(left)
            else:
                leftparts = Node.breakIrregularDurationInNode(left, beats)
                regularParts.extend(leftparts)
            if right.hasRegularDuration():
                regularParts.append(right)
            else:
                rightparts = Node.breakIrregularDurationInNode(right, beats)
                regularParts.extend(rightparts)
            if any(not part.hasRegularDuration() for part in regularParts):
                raise SplitError(f"Cannot split {item} at {offset} (parts: {parts}, "
                                 f"symbolic durations: {[p.symbolicDuration() for p in parts]}), "
                                 f"durRatios: {[p.durRatios for p in parts]}")
            newitems = self.items[:i] + regularParts + self.items[i+1:]
            assert all(a.end == b.offset for a, b in pairwise(newitems)), f"{i=}, {item=}, {offset=}\n{newitems=}\n{self.items=}\n{regularParts=}"
            self._setitems(newitems)
            self._remerge()
            return regularParts
        return None

    @staticmethod
    def asTree(nodes: list[Node], readonly=False) -> Node:
        """
        Transform a list of Nodes into a tree structure

        A tree has a root and leaves, where each leave can be the root of a subtree

        Args:
            nodes: the tree to get/make the root for

        Returns:
            the root of a tree structure
        """

        if len(nodes) == 1 and nodes[0].durRatio == (1, 1):
            n0 = nodes[0]
            root = n0.clone(readonly=readonly)
        else:
            root = Node(ratio=(1, 1), items=nodes, readonly=readonly)  # type: ignore
        assert root.totalDuration() == sum(n.totalDuration() for n in nodes)
        root.repair()
        return root

    def _removeUnnecessaryNodesInPlace(self) -> None:
        def _inner(node: Node) -> Node:
            if len(node.items) == 1 and isinstance(node.items[0], Node) and node.durRatio == node.items[0].durRatio:
                return _inner(node.items[0])
            else:
                return node
        root = _inner(self)
        if root is self:
            return
        self._setitems(root.items)
        self.durRatio = root.durRatio

    @staticmethod
    def breakIrregularDurationInNode(n: Notation, beatstruct: Sequence[QuantizedBeatDef]) -> list[Notation]:
        """
        Break irregular durations in a node

        Args:
            n: the notation to break
            beatstruct: the beatstructure

        Returns:
            the resulting notations.
        """
        # this is called on each part of a notation when split at a beat boundary
        assert n.duration > 0
        assert n.isQuantized() and not n.hasRegularDuration()
        from maelzel.scoring import util
        beatoffsets = [b.offset for b in beatstruct]
        fragments = util.splitInterval(n.qoffset, n.end, beatoffsets)
        N = len(fragments)
        assert N > 0,  f"??? {n=}, {beatoffsets=}"
        if N == 1:
            # does not cross any beats
            beat = next((b for b in beatstruct if b.offset <= n.qoffset and n.end <= b.end), None)
            assert beat is not None, f"Could not find beat for {n}, beats={beatstruct}"
            parts = n._breakIrregularDurationInBeat(beatDur=beat.duration, beatDivision=beat.division, beatOffset=beat.offset)
            assert parts is not None
            return parts
        elif N == 2:
            n0, n1 = n.splitAtOffset(fragments[1][0])
            parts = []
            for part in (n0, n1):
                if part.hasRegularDuration():
                    parts.append(part)
                else:
                    parts.extend(Node.breakIrregularDurationInNode(part, beatstruct=beatstruct))
            Notation.tieNotations(parts)
            return parts
        else:
            parts = []
            offset0, end0 = fragments[0]
            offset1, end1 = fragments[1][0], fragments[-2][1]
            offset2, end2 = fragments[-1]
            n0 = n.clone(offset=offset0, duration=end0-offset0, spanners=False)
            n1 = n.clone(offset=offset1, duration=end1-offset1, spanners=False)
            n2 = n.clone(offset=offset2, duration=end2-offset2, spanners=False)
            for part in (n0, n1, n2):
                if part.hasRegularDuration():
                    parts.append(part)
                else:
                    parts.extend(Node.breakIrregularDurationInNode(part, beatstruct=beatstruct))
            Notation.tieNotations(parts)
            n._copySpannersToSplitNotation(parts)
            return parts

    @staticmethod
    def beatToTree(notations: list[Notation], division: int | division_t,
                   beatOffset: F, beatDur: F) -> Node:
        """
        Create a tree from a quantized beat

        Args:
            notations: the notations in this beat
            division: the division for this beat
            beatOffset: the offset within the measure
            beatDur: the duration of the beat

        Returns:
            a Node representing the tree structure of this beat

        """
        return beatToTree(notations=notations, division=division,
                          beatOffset=beatOffset, beatDur=beatDur)


def beatToTree(notations: list[Notation], division: int | division_t,
               beatOffset: F, beatDur: F
               ) -> Node:
    if isinstance(division, tuple) and len(division) == 1:
        division = division[0]

    if isinstance(division, int):
        durRatio = quantdata.durationRatios[division]
        return Node(notations, ratio=durRatio)  # type: ignore

    # assert isinstance(division, tuple) and len(division) >= 2
    numSubBeats = len(division)
    now = beatOffset
    dt = beatDur/numSubBeats
    durRatio = quantdata.durationRatios[numSubBeats]
    items = []
    for subdiv in division:
        subdivEnd = now + dt
        subdivNotations = [n for n in notations if now <= n.qoffset < subdivEnd and n.end <= subdivEnd]
        if subdiv == 1:
            items.extend(subdivNotations)
        else:
            items.append(beatToTree(notations=subdivNotations, division=subdiv, beatOffset=now, beatDur=dt))
        now += dt
    return Node(items, ratio=durRatio)
