"""
Quantize durations to musical notation

The most important function here is :func:`quantize`, which treturns
a :class:`QuantizedScore`

"""
from __future__ import annotations

from dataclasses import dataclass
import sys
import os
from math import sqrt
import itertools
from bisect import bisect

from maelzel.common import F, F0, asF

from . import core
from . import definitions
from . import util
from . import quantdata
from . import quantutils
from . import clefutils
from . import spanner as _spanner
from . import attachment
from .quantprofile import QuantizationProfile
from .common import logger
from .quantdefs import QuantizedBeatDef
from maelzel._logutils import LazyStr

from .notation import Notation, SnappedNotation
from .node import Node
import maelzel.scorestruct as st


from emlib import misc
from emlib import mathlib
from emlib.result import Result

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.common import beat_t
    from .common import division_t
    from typing import Iterator, Sequence
    import maelzel.core
    from .node import LogicalTie
    from . import enharmonics
    from . import renderoptions
    from . import renderer


__all__ = (
    'quantizeParts',
    'QuantizedScore',
    'QuantizedPart',
    'QuantizedMeasure',
    'QuantizedBeat',
    'quantizeMeasure',
    'quantizePart',
    'splitNotationAtMeasures',
    'QuantizationProfile'
)

_INDENT = "  "


def _fitToGrid(offsets: list[float], grid: list[float]) -> list[int]:
    out = []
    idx = 0
    for offset in offsets:
        relidx = misc.nearest_index(offset, grid[idx:])
        absidx = idx + relidx
        out.append(absidx)
        idx = absidx - 1 if absidx > 0 else 0
    return out


def _fitToGrid2(offsets: list[float], grid: list[float]) -> list[int]:
    out = []
    idx = 0
    gridlen = len(grid)
    for offset in offsets:
        idx = bisect(grid, offset, lo=idx)
        if idx == gridlen:
            idx = gridlen - 1
        elif idx != 0:
            idxleft = idx - 1
            idx = idx if grid[idx] - offset < offset - grid[idxleft] else idxleft
        out.append(idx)
    return out


def _fitEventsToGridNearest(events: list[Notation], grid: list[F]) -> list[int]:
    # We use floats to make this faster. Rounding errors should not pose a problem
    # in this context
    fgrid = [float(g) for g in grid]
    offsets = [event.qoffset for event in events]
    return [misc.nearest_index(float(offset), fgrid) for offset in offsets]


def snapEventsToGrid(notations: list[Notation],
                     grid: list[F],
                     fgrid: list[float] | None = None
                     ) -> tuple[list[int], list[SnappedNotation]]:
    """
    Snap unquantized events to a given grid

    Args:
        notations: a list of unquantized Notation's
        grid: the grid to snap the events to, as returned by generateBeatGrid
        fgrid: the same grid, but as floats.

    Returns:
        tuple (assigned slots, quantized events)
    """
    beatdur = grid[-1]
    # assignedSlots = _fitEventsToGridNearest(events=notations, grid=grid)
    offsets = [float(n.offset) for n in notations]
    if fgrid is None:
        fgrid = [float(tick) for tick in grid]
    assignedSlots = _fitToGrid2(offsets, fgrid)
    snapped: list[SnappedNotation] = []
    lastidx = len(grid) - 1
    for idx in range(len(notations)-1):
        n = notations[idx]
        slot0 = assignedSlots[idx]
        offset0 = grid[slot0]
        # is it the last slot (as grace note?)
        if slot0 == lastidx:
            if not n.isRest:
                snapped.append(SnappedNotation(n, offset0, F0))
        else:
            offset1 = grid[assignedSlots[idx+1]]
            dur = offset1 - offset0
            if dur == 0 and n.isRest:
                continue
            snapped.append(SnappedNotation(n, offset0, dur))

    lastoffset = grid[assignedSlots[-1]]
    last = notations[-1]
    lastdur = beatdur - lastoffset
    if not (lastdur == 0 and last.isRest):
        snapped.append(SnappedNotation(last, lastoffset, duration=lastdur))
    assert not any(s.duration == 0 and s.notation.isRest for s in snapped), snapped
    return assignedSlots, snapped


def _isBeatFilled(events: list[Notation], beatDuration: F, beatOffset: F = F0
                  ) -> bool:
    """
    Check if notations fill the beat exactly

    This will be False if there are any holes in the beat or not all
    durations are set.

    Args:
        events: list of notations inside the beat to check
        beatDuration: the duration of the beat
        beatOffset: the offset of the start of the beat

    Returns:
        True if the notations fill the beat

    """
    return (events[0].offset == beatOffset
            and events[-1].end == beatOffset + beatDuration
            and all(ev0.end == ev1.offset for ev0, ev1 in itertools.pairwise(events)))


def _eventsShow(events: list[Notation]) -> str:
    from maelzel._util import showF
    lines = [""]
    for ev in events:
        back = "←" if ev.tiedPrev else ""
        forth = "→" if ev.tiedNext else ""
        tiedStr = f"tied: {back}{forth}"
        lines.append(f"  {showF(ev.qoffset)} – {showF(ev.end)} "
                     f"dur={showF(ev.duration)} {tiedStr}")
    return "\n".join(lines)


def _checkQuantizedNotations(notations: list[Notation],
                             totalDuration: F,
                             offset=F0
                             ) -> str:
    if any(n.offset is None for n in notations):
        return f"The notations should have an offset, {notations}"

    if not all(n0.qoffset <= n1.qoffset for n0, n1 in itertools.pairwise(notations)):
        return f"Events are not sorted: {_eventsShow(notations)}"

    if not all(n0.end <= n1.qoffset for n0, n1 in itertools.pairwise(notations) if n0.duration is not None):
        return f"Events overlap: {_eventsShow(notations)}"

    if not all(n.qoffset - offset <= totalDuration for n in notations):
        return f"Events outside of time range ({totalDuration=}): {_eventsShow(notations)}"

    if not all(n.end <= offset+totalDuration for n in notations if n.duration is not None):
        return "Events extend over given duration"
    
    # No errors
    return ''





def _evalGridError(profile: QuantizationProfile,
                   snappedEvents: list[SnappedNotation],
                   beatDuration: F) -> float:
    """
    Evaluate the error regarding the deviation of the makeSnappedNotation events from the original offset/duration

    Given a list of events in a beat and these events makeSnappedNotation to a given subdivision of
    the beat, evaluate how good this snapping is in representing the original events.
    This is used to find the best subdivision of a beat.

    Args:
        profile: the quantization preset to use
        snappedEvents: the events after being makeSnappedNotation to a given grid
        beatDuration: the duration of the beat

    Returns:
        a value indicative of how much the quantization diverges from
        the unquantized version. Lower is better
    """
    assert isinstance(beatDuration, F)
    offsetErrorWeight = profile.offsetErrorWeight
    restOffsetErrorWeight = profile.restOffsetErrorWeight
    graceDuration = profile.graceDuration
    graceNoteOffsetErrorFactor = 0.5
    beatdurf = float(beatDuration)
    numGracenotes = 0
    totalOffsetError = 0
    totalDurationError = 0
    for snapped in snappedEvents:
        n = snapped.notation
        offsetError = abs(n.qoffset - snapped.offset) / beatdurf

        if snapped.duration == 0:
            numGracenotes += 1
            offsetError *= graceNoteOffsetErrorFactor
            durationError = abs(n.duration - graceDuration) / beatdurf
        else:
            if n.isRest:
                offsetError *= restOffsetErrorWeight / offsetErrorWeight
            durationError = abs(n.duration - snapped.duration) / beatdurf

        totalOffsetError += offsetError
        totalDurationError += durationError

    gracenoteError = numGracenotes / len(snappedEvents)
    error = mathlib.euclidian_distance(
        (totalOffsetError, totalDurationError, gracenoteError),
        (offsetErrorWeight, profile.durationErrorWeight, profile.graceErrorWeight))
    error = error ** profile.gridErrorExp
    return error


class QuantizedBeat:
    """
    A QuantizedBeat holds notations inside a beat filling the beat

    Args:
        divisions: The division of this beat
        assignedSlots: Which slots are assigned to notations in this beat
        notations: The notations in this beat. They are cropped to fit
        beatDuration: The duration of the beat in quarter notes
        beatOffset: The offset of the beat in relation to the measure
        quantizationError: The error calculated during quantization.
            The higher the error, the less accurate the quantization
        quantizationInfo: Info collected during quantization
        weight: The weight of this beat within the measure. 2=strong, 1=weak, 0=no weight

    """

    __slots__ = ('divisions', 'assignedSlots', 'notations', 'duration',
                 'offset', 'quantizationError', 'quantizationInfo',
                 'weight')

    def __init__(self,
                 divisions: division_t,
                 assignedSlots: list[int],
                 notations: list[Notation],
                 beatDuration: F,
                 beatOffset: F = F0,
                 quantizationError: float = 0.,
                 quantizationInfo: str = '',
                 weight: int = 0):

        notations = [n for n in notations
                     if not (n.isRest and n.isGracenote and not n.hasAttributes())]

        self.divisions: division_t = divisions
        "The division of this beat"

        self.assignedSlots: list[int] = assignedSlots
        "Which slots are assigned to notations in this beat"

        self.notations: list[Notation] = notations
        "The notations in this beat. They are cropped to fit"

        self.duration: F = beatDuration
        "The duration of the beat in quarter notes"

        self.offset: F = beatOffset
        "The offset of the beat in relation to the measure"

        self.quantizationError: float = quantizationError
        "The error calculated during quantization. The higher the error, the less accurate the quantization"

        self.quantizationInfo: str = quantizationInfo
        "Info collected during quantization"

        self.weight: int = weight
        "The weight of this beat within the measure. 2=strong, 1=weak, 0=no weight"

        assert all(not n.durRatios for n in notations), notations

        self._applyDurationRatios()

    def __repr__(self):
        parts = [
            f"divisions: {self.divisions}, assignedSlots={self.assignedSlots}, "
            f"notations={self.notations}, beatDuration={self.duration}, beatOffset={self.offset}, "
            f"quantizationError={self.quantizationError:.5g}, weight={self.weight}"
        ]
        if self.quantizationInfo:
            parts.append(f'quantizationInfo={self.quantizationInfo}')
        return f'QuantizedBeat({", ".join(parts)})'

    @property
    def end(self) -> F:
        """The end of this beat in quarternotes"""
        return self.offset + self.duration

    def dump(self, indents=0, indent='  ', stream=None):
        """Dump this beat"""
        stream = stream or sys.stdout
        print(f"{indent*indents}QuantizedBeat(divisions={self.divisions}, assignedSlots={self.assignedSlots}, "
              f"beatDuration={self.duration}, beatOffset={self.offset}, "
              f"quantizationError={self.quantizationError:.3g})", file=stream)
        ind = indent * (indents + 1)
        for n in self.notations:
            print(f"{ind}{n}", file=stream)

    def _applyDurationRatios(self):
        # After this, all notations are quantized
        quantutils.applyDurationRatio(self.notations, division=self.divisions,
                                      beatOffset=self.offset, beatDur=self.duration)

    def asTree(self) -> Node:
        """
        Returns the notations in this beat as a tree

        Returns:
            a Node which is the root of a tree representing the notations in
            this beat (grouped by their duration ratio)
        """
        return Node.beatToTree(self.notations, division=self.divisions,
                               beatOffset=self.offset, beatDur=self.duration)

    def __hash__(self):
        notationHashes = [hash(n) for n in self.notations]
        data = [hash(self.divisions), hash(self.duration), hash(self.offset)]
        data.extend(notationHashes)
        return hash(tuple(data))


class QuantizedMeasure:
    """
    A QuantizedMeasure holds a quantized tree

    If given a list of QuantizedBeats, these are merged together in a recursive
    structure to generate a tree of Nodes. See :meth:`QuantizedMeasure.asTree`

    Args:
        timesig: the time signature
        quarterTempo: the tempo of the quarter note
        beats: a list of QuantizedBeats
        quantprofile: the quantization profile used to generate this quantized measure.
            This is necessary to create a tree structure
        subdivisions: a list of integers corresponding to the number of units of the
            denominator of the fused time signature. This can override the subdivision
            structure defined in the time signature itself. For example, for
            a 7/8 measure with subdivisions of 2+3+2, this parameter should be (2, 3, 2)
        parent: the QuantizedPart this measure belongs to
    """
    def __init__(self,
                 timesig: st.TimeSignature,
                 quarterTempo: F,
                 beats: list[QuantizedBeat],
                 quantprofile: QuantizationProfile,
                 subdivisions: tuple[int, ...] | None = None,
                 parent: QuantizedPart | None = None,
                 readonly=False):
        assert subdivisions is None or isinstance(subdivisions, tuple)

        self.timesig: st.TimeSignature = timesig
        "The time signature"

        self.quarterTempo = quarterTempo
        "The tempo for the quarter note"

        self.beats = beats
        "A list of QuantizedBeats"

        self.quantprofile = quantprofile
        "The quantization profile used to generate this measure"

        self.parent = parent
        "The parent of this measure (a QuantizedPart)"

        self.subdivisions = subdivisions
        """The subdivision structure of this measure"""

        self._offsets = None
        """Cached offsets"""

        self._beatDefs: list[QuantizedBeatDef] | None = None

        if self.beats:
            self._checkBeats()

        self.tree = self._makeTree()
        """The root of the tree representation"""

        if readonly:
            self.tree.setReadOnly(True, recurse=True)

    def setReadOnly(self, value: bool):
        self.tree.setReadOnly(True, recurse=True)


    def __repr__(self):
        parts = [f"timesig={self.timesig}, quarterTempo={self.quarterTempo}, tree={self.tree}"]
        if self.quantprofile:
            parts.append(f"profile={self.quantprofile.name}")
        return f"QuantizedMeasure({', '.join(parts)})"

    def __hash__(self):
        if self.empty():
            return hash((self.timesig, self.quarterTempo))
        else:
            return hash((self.tree, self.timesig, self.quarterTempo, self.subdivisions))

    def measureIndex(self) -> int | None:
        """Return the measure index of this measure within the QuantizedPart"""
        if not self.parent:
            return None
        return self.parent.measures.index(self)

    def previousMeasure(self) -> QuantizedMeasure | None:
        """
        Returns the previous measure in the part

        Returns:
            the previous measure (a QuantizedMeasure), or None if no previous measure

        Raises:
            ValueError: if this QuantizedMeasure does not have a parent
        """
        if self.parent is None:
            raise ValueError(f"This {type(self)} has no parent")
        idx = self.measureIndex()
        if idx is None:
            return None
        if idx == 0:
            return None
        return self.parent.measures[idx - 1]

    def duration(self) -> F:
        """
        Duration of this measure, in quarter notes

        Returns:
            the duration of the measure in quarter notes
        """
        return self.timesig.quarternoteDuration

    def beatBoundaries(self) -> list[F]:
        """
        The beat offsets, including the end of the measure

        Returns:
            the boundaries of the beats, which is the offsets plus the end of the measure
        """
        boundaries = self.beatOffsets().copy()
        boundaries.append(boundaries[0] + self.duration())
        return boundaries

    def beatStructure(self) -> list[QuantizedBeatDef]:
        if self._beatDefs is None:
            self._beatDefs = [QuantizedBeatDef(offset=beat.offset, duration=beat.duration,
                                               division=beat.divisions, weight=beat.weight)
                              for beat in self.beats]
        return self._beatDefs

    def beatWeights(self) -> list[int]:
        beatstruct = self.beatStructure()
        return [b.weight for b in beatstruct]

    def beatOffsets(self) -> list[F]:
        """
        Returns a list of the offsets of each beat within this measure

        Returns:
            the offset of each beat. The first offset is always 0
        """
        # We cannot call beatStructure because this is called prior to the tree existing
        if self._offsets is None:
            beatdefs = self.beatStructure()
            self._offsets = [beat.offset for beat in beatdefs]
        return self._offsets

    def _pinEnharmonicSpelling(self,
                               options: enharmonics.EnharmonicOptions,
                               prevMeasure: QuantizedMeasure | None = None
                               ) -> None:
        """
        Pin the enharmonic spellings within this measure (inplace)

        Args:
            options: the EnharmonicOptions to use

        """
        if self.empty():
            return
        tree = self.tree
        first = tree.firstNotation()
        if not first.isRest and first.tiedPrev:
            assert prevMeasure is not None
            if prevMeasure.empty():
                logger.info(f"The first note ({first}) of measure {self.measureIndex()} is tied to "
                            f"the previous note, but the previous measure is empty")
                prevTree = None
            else:
                prevTree = prevMeasure.tree
        else:
            prevTree = None
        tree.fixEnharmonics(options=options, prevTree=prevTree)

    def empty(self) -> bool:
        """
        Is this measure empty?

        Returns:
            True if empty
        """
        return self.tree.empty()

    def repair(self) -> bool:
        root = mergeSiblings(self.tree, profile=self.quantprofile, beatOffsets=self.beatOffsets(), beatWeights=self.beatWeights())
        if root != self.tree:
            self.tree = root
            return True
        return False

    def dump(self, numindents=0, indent=_INDENT, tree=True, stream=None) -> None:
        ind = _INDENT * numindents
        stream = stream or sys.stdout
        print(f"{ind}Timesig: {self.timesig}"
              f"(quarter={self.quarterTempo})", file=stream)
        # if self.empty():
        #    print(f"{ind}EMPTY", file=stream)
        if tree:
            self.tree.dump(numindents, indent=indent, stream=stream)
        elif self.beats:
            for beat in self.beats:
                beat.dump(indents=numindents, indent=indent, stream=stream)

    def notations(self) -> list[Notation]:
        """
        Returns a flat list of all notations in this measure

        The notations returned are the actual notations, so any modification
        to them will affect the measure.

        Returns:
            a list of Notations in this measure
        """
        if self.empty():
            return []

        return list(self.tree.recurse())

    def recurseNotationsWithParent(self, reverse=False) -> Iterator[tuple[Notation, Node]]:
        """
        Returns a flat iterator over all notations in this measure, along with their parent nodes

        The notations returned are the actual notations, so any modification
        to them will affect the measure.

        Args:
            reverse: If True, the notations are returned in reverse order

        Returns:
            an iterator over the Notations in this measure
        """
        if self.empty():
            return iter(())
        return self.tree.recurseWithNode(reverse=reverse)

    def _makeTree(self) -> Node:
        """
        Returns the root of a tree of Nodes representing the items in this measure
        """
        if not self.quantprofile:
            raise ValueError("Cannot create tree without a QuantizationProfile")

        if not self.beats:
            raise ValueError(f"This QuantizedMeasure is empty: {self}")

        tree = _makeTreeFromQuantizedBeats(beats=self.beats,
                                           quantprofile=self.quantprofile,
                                           beatOffsets=self.beatOffsets(),
                                           beatWeights=self.beatWeights())
        tree.parentMeasure = self
        return tree

    def logicalTies(self) -> list[LogicalTie]:
        """
        Returns a list of locations for each tie

        A tie is a list of tied notations. For each tie, a list of TreeLocations
        is returned
        """
        return self.tree.logicalTieLocations(measureIndex=self.measureIndex())

    def beatAtOffset(self, offset: F) -> tuple[QuantizedBeatDef, int]:
        """
        The beat which includes the given offset

        Args:
            offset: offset relative to the start of this measure

        Returns:
            a tuple (beatdef: QuantizedBeatDef, beatIndex: int). To access the
            actual QuantizedBeat, use ``measure.beats[beatIndex]``

        Raises:
            ValueError: if the given offset is outside the measure
        """
        beats = self.beatStructure()
        for i, beat in enumerate(beats):
            if beat.offset <= offset < beat.end:
                return beat, i
        raise ValueError(f"The given offset {offset} is not within this measure, "
                         f"measure duration: {self.duration()}, beat structure: {beats}")

    def splitNotationAt(self, offset: F, tie=True, mergeable=False) -> list[Notation] | None:
        """
        Split any notation present at offset in place, returns the resulting parts

        Args:
            offset: the beat offset within the measure
            tie: if True, tie the resulting parts
            mergeable: if False, mark the marks as unmergeable

        Returns:
            the resulting parts or None if no notations present at the given offset.
            Raises ValueError if the given offset is not within the span of this
            measure

        """
        if offset > self.duration():
            raise ValueError(f"The given offset {offset} is not within the span "
                             f"of this measure ({self.duration()=}")
        return self.tree.splitNotationAtOffset(offset=offset, tie=tie, mergeable=mergeable, 
                                               beatstruct=self.beatStructure())

    def findLogicalTie(self, n: Notation) -> LogicalTie | None:
        if not n.tiedPrev and not n.tiedNext:
            return None
        ties = self.logicalTies()
        tie = next((tie for tie in ties if n in tie), None)
        if not tie:
            return None
        # This tie might be part of a bigger tie at the part level
        if tie[0].notation.tiedPrev or tie[-1].notation.tiedNext:
            return self.parent.findLogicalTie(n)
        else:
            return tie

    def _splitStrongBeat(self, n: Notation, node: Node, offset: F):
        """Returns True if n should be split at offset"""
        if self._splitWeakBeat(n, node, offset):
            return True
        beatoffsets = self.beatOffsets()
        noffset = n.qoffset
        nend = noffset + n.duration
        assert noffset < offset < nend
        symdur = n.symbolicDuration()
        beatdef, beatidx = self.beatAtOffset(offset)
        centered = offset - noffset == nend - offset
        maindur, numdots = quantutils.splitDots(symdur)

        if symdur % 1 == 0:
            return not centered
        elif n.duration / beatdef.duration < 1:
            if not centered or numdots > 0:
                logger.debug("duration less than beat but not centered or dotted, splitting")
                return True
        elif numdots > 0 and (noffset not in beatoffsets or nend not in beatoffsets):
            logger.debug("dotted and not aligned, splitting")
            return True
        elif node.durRatio[0] in (3, 5, 7) and centered:
            # A tuplet centered across the beat
            return False
        elif n.isRest:
            return True
        elif noffset not in beatoffsets:
            logger.debug("start not aligned, splitting, symdur=%s, centered=%s", symdur, centered)
            return True
        return False

    def _splitWeakBeat(self, n: Notation, node: Node, offset: F):
        """True if n should be split at offset"""

        if n.duration == self.duration():
            return False

        noffset: F = n.qoffset
        nend: F = noffset + n.duration
        assert noffset < offset < nend

        leftdur: F = offset - noffset
        rightdur: F = nend - offset

        beatoffsets = self.beatOffsets()
        qprofile = self.quantprofile

        if noffset in beatoffsets or nend in beatoffsets:
            return False

        if quantutils.asymettry(leftdur, rightdur) >= qprofile.syncopMaxAsymmetry:
            logger.debug(f"Too much assymetry, splitting {n} at {offset}, {beatoffsets=}")
            return True

        EPS = F(1, 10000)
        leftbeat, idx = self.beatAtOffset(offset - EPS)
        rightbeat = self.beatStructure()[idx+1]

        partdur, beat = (leftdur, leftbeat) if leftdur < rightdur else (rightdur, rightbeat)
        if n.duration < qprofile.syncopMinFraction * beat.duration:
            logger.debug(f"Duration {n.duration} < {(qprofile.syncopMinFraction*beat.duration)=}")
            return True

        if partdur/beat.duration <= qprofile.syncopPartMinFraction:
            logger.debug(f"Part of {n}, {partdur=} too short in relation to the beat, {qprofile.syncopPartMinFraction=}")
            return True

        if n.tiedPrev and n.tiedNext and n.symbolicDuration().numerator in (3, 7, 15):
            logger.debug(f"Too complex for a syncopation, {n.symbolicDuration()=}")
            return True

        ratio = n.fusedDurRatio()
        if qprofile.syncopExcludeSymDurs:
            for d in (leftdur, rightdur):
                if (ratio*d).numerator in qprofile.syncopExcludeSymDurs:
                    return True
        return False

    def breakSyncopations(self, level: str = '') -> None:
        """
        Break notes extended over beat boundaries, **in place**

        The level indicates which syncopations to break. 'all' will split
        any notations extending over any beat; 'weak' will only break
        syncopations over secondary beats (for example, the 3rd quarter-note
        in a 4/4 measure); 'strong' will only break syncopations over strong
        beats (the 4th quarternote in a 6/4 measure with the form 3+3, or the 3rd
        quarternote in a 7/8 measure with the form 2+2+3)

        Args:
            level: one of 'all', 'weak', 'strong'. If not given, the level set
                in the quantization profile is used

        """
        if self.empty() or len(self.beats) == 1:
            return

        minWeight = self.quantprofile.breakSyncopationsMinWeight(level)
        if minWeight is None:
            raise ValueError(f"Expected one of 'all, 'weak', 'strong', got {level}")

        beatstruct = self.beatStructure()
        tree = self.tree
        needsRepair = False
        # First we split strong beats, then check weak ones
        for i, beat in ((i, beat) for i, beat in enumerate(self.beats[1:-1], start=1) if beat.weight >= minWeight):
            if tree._splitNotationAtBeat(beatstruct, beatIndex=i, callback=self._splitStrongBeat):
                needsRepair = True
        for i, beat in ((i, beat) for i, beat in enumerate(self.beats[1:-1], start=1) if beat.weight < minWeight):
            if tree._splitNotationAtBeat(beatstruct, beatIndex=i, callback=self._splitWeakBeat):
                needsRepair = True

        if needsRepair:
            self.tree.repair()

    def removeUnnecessaryGracenotes(self) -> None:
        self.tree.removeUnnecessaryGracenotes()
        tiedGraceMinDur = self.quantprofile.tiedSnappedGracenoteMinRealDuration
        if tiedGraceMinDur == 0:
            return
        for n, node in self.tree.recurseWithNode():
            if (n.isGracenote and (n.tiedNext or n.tiedPrev) and
                    not n.hasAttributes() and n.getProperty('.snappedGracenote') and
                    n.getProperty('.originalDuration', F0) < tiedGraceMinDur):
                nidx = node.items.index(n)
                if nidx < len(node.items) - 2:
                    # not the last of the node, so get next notation in node
                    nextnote = node.findNextNotation(n)
                elif node.parent:
                    # last note in node, next could be in parent or we could be at
                    # the end of the tree
                    nextnote = node.parent.findNextNotation(n)
                else:
                    # last note, so no next note in this measure
                    nextnote = None
                if nextnote:
                    nextnote.tiedPrev = False
                    if n.hasAttributes():
                        n.copyAttributesTo(nextnote)
                del node.items[nidx]

    def check(self):
        self._checkBeats()
        self._checkTree()

    def _checkTree(self):
        measuredur = self.duration()
        treedur = self.tree.totalDuration()
        if measuredur != treedur:
            n = self.measureIndex()
            logger.error(f"Duration mismatch, measure #{n}, should be {measuredur}")
            self.dump()
            raise ValueError(f"Measure #{n} has a duration mismatch between the duration "
                             f"according to the time signature ({measuredur}) and the "
                             f"duration of its tree, {treedur}.")
        for n0, n1 in itertools.pairwise(self.tree.recurse()):
            if n0.tiedNext:
                assert n1.tiedPrev
                tiedpitches = n0.tiedPitches()
                if tiedpitches is None:
                    assert all(p in n1.pitches for p in n0.pitches)
                else:
                    assert all(p in n1.pitches for p in tiedpitches), f"{n0=}, {n1=}"

    def _checkBeats(self):
        if not self.beats:
            return
        # check that the measure is filled
        for i, beat in enumerate(self.beats):
            for n in beat.notations:
                assert n.duration is not None, n
                assert n.durRatios is not None, n

                if n.duration > 0:
                    assert n.isQuantized(), n
                if n.isRest:
                    assert n.duration > 0, n
                else:
                    assert n.duration >= 0, n
            durNotations = sum(n.duration for n in beat.notations)
            if durNotations != beat.duration:
                measnum = self.measureIndex()
                logger.error(f"Duration mismatch, loc: ({measnum}, {i}). Beat dur: {beat.duration}, Notations dur: {durNotations}")
                logger.error(beat.notations)
                self.dump(tree=False)
                self.dump(tree=True)
                raise ValueError(f"Duration mismatch in beat {i}")

    def setBeamSubdivisions(self,
                            beat: F | float | int,
                            minimum: F | int = F0, maximum: F | int = F0,
                            once=True
                            ) -> None:
        """
        Customize beam subdivision

        The beams of consecutive 16th (or shorter) notes are, by default, not subdivided.
        That is, the beams of more than two stems stretch unbroken over entire groups of notes.
        This behavior can be modified to subdivide the beams into sub-groups. Beams will be
        subdivided at intervals to match the metric value of the subdivision.

        .. note:: At the moment this is only supported by the lilypond backend

        Args:
            beat: the beat to customize
            minimum: minimum limit of beam subdivision. A fraction or simply the denominator.
                1/8 indicates an eighth note, 16 indicates a 16th note
            maximum: maximum limit of beam subdivision. Similar to minimum
            once: if True, only apply this customization to the beat starting at ``beat``

        Example
        ~~~~~~~

            >>> measure = part.measureAt(...)
            # Split any beam shorter than 1/8. This will break a group of 4 16th notes
            # into two groups of 2 16th notes
            >>> measure.setBeamSubdivisions(0, F(1, 8))
        """
        if not (minimum > 0 or maximum > 0):
            raise ValueError(f"Either 'minimum' or 'maximum' must be positive, got {minimum=}, {maximum=}.")
        if isinstance(minimum, int):
            minimum = F(1, minimum)
        if isinstance(maximum, int):
            maximum = F(1, maximum)
        if n := self.notationAt(beat):
            n.addAttachment(attachment.BeamSubdivisionHint(minimum=minimum, maximum=maximum, once=once))

    def notationAt(self, beat: F | float | int) -> Notation | None:
        """
        Returns the notation present at the given beat location relative to this measure

        Args:
            beat: the relative location

        Returns:
            the Notation at the given beat location, or None if no Notation found
        """
        if beat >= self.duration():
            return None
        for n in self.tree.recurse():
            assert n.offset is not None
            if n.offset <= beat < n.end:
                return n
        return None


def _crossesSubdivisions(slotStart: int, slotEnd: int, slotsAtSubdivs: list[int]) -> bool:
    if slotStart not in slotsAtSubdivs:
        nextSlot = next(slot for slot in slotsAtSubdivs if slot > slotStart)
        if slotEnd > nextSlot:
            return True
    elif slotEnd not in slotsAtSubdivs:
        prevSlot = next(slot for slot in reversed(slotsAtSubdivs) if slot < slotEnd)
        if slotStart < prevSlot:
            return True
    return False


def _makeTreeFromQuantizedBeats(beats: list[QuantizedBeat],
                                beatOffsets: Sequence[F],
                                quantprofile: QuantizationProfile,
                                beatWeights: Sequence[int]
                                ) -> Node:
    """
    Returns the root of a tree of Nodes representing the items in this measure
    """
    if not beats:
        raise ValueError("No quantized beats were given")

    for beat in beats:
        assert all(n.isQuantized() for n in beat.notations), f"{beat.notations=}"

    nodes = [beat.asTree().mergedNotations() for beat in beats]
    assert sum(node.totalDuration() for node in nodes) == sum(beat.duration for beat in beats)
    root = Node.asTree(nodes)
    root.check()
    root = mergeSiblings(root, profile=quantprofile, beatOffsets=beatOffsets, beatWeights=beatWeights)
    if root.totalDuration() != sum(beat.duration for beat in beats):
        import io
        f = io.StringIO()
        root.dump(stream=f)
        logger.error(f"Tree:\n{f.getvalue()}")
        raise ValueError(f"Duration mismatch in tree, root dur: {root.totalDuration()}, "
                         f"beats dur: {sum(beat.duration for beat in beats)}.\nbeats: {beats}")
    return root


def _evalRhythmComplexity(profile: QuantizationProfile,
                          snapped: list[SnappedNotation],
                          div: division_t,
                          beatDur: F,
                          assignedSlots: list[int]
                          ) -> tuple[float, str]:
    """
    Evaluate the complexity of the rhythm

    Args:
        profile: the quantization profile being used
        snapped: a list of SnappedNotations
        div: the division used for quantization
        beatDur: the duration of the beat
        assignedSlots: the slots assigned to each snapped notation

    Returns:
        a tuple (penalty: float, debugmsg: str) where penalty is higher
        if the complexity is higher. debugmsg will only contain debug info
        if profile.debug is True
    """
    # calculate notes across subdivisions
    if len(div) == 1:
        div0 = div[0]
        if not isinstance(div0, int):
            raise ValueError(f"Deeply nested divisions are not supported, got {div}")
        slots = assignedSlots + [div0]
        # duration in terms of number of slots
        durs = [b - a for a, b in itertools.pairwise(slots)]
        numTies = sum(dur not in quantdata.regularDurations for dur in durs)
        if div0 % 2 == 0:
            numSyncop = sum(dur > 1 and s % 2 == 1 for dur, s in zip(durs, assignedSlots))
        elif div0 == 3:
            numSyncop = 0
        else:
            for mod in (3, 5, 7, 11, 13, 17, 19):
                if div0 % mod == 0:
                    numSyncop = sum(dur > 1 and slot % mod > 0 for dur, slot in zip(durs, assignedSlots))
                    break
            else:
                numSyncop = len(assignedSlots)
                if 0 in assignedSlots:
                    numSyncop -= 1
    else:
        # slotsAtSubdivs: list[int] = [0] + list(itertools.accumulate(div))
        slotsAtSubdivs = quantutils.slotsAtSubdivisions(div)
        numSyncop = 0
        lastslot = quantutils.divisionNumSlots(div)
        for slotStart, slotEnd in itertools.pairwise(assignedSlots + [lastslot]):
            if _crossesSubdivisions(slotStart, slotEnd, slotsAtSubdivs):
                numSyncop += 1

        numTies = sum(not isRegularDuration(dur=n.duration, beatDur=beatDur)
                      for n in snapped
                      if not n.notation.isRest)

    penalty = mathlib.weighted_euclidian_distance([
        (numSyncop / len(snapped), profile.rhythmComplexityNotesAcrossSubdivisionWeight),
        (numTies / len(snapped), profile.rhythmComplexityIrregularDurationsWeight)
    ])
    debugstr = f'{numSyncop=}, {numTies=}' if profile.debug else ''
    return penalty, debugstr


def quantizeBeatBinary(eventsInBeat: list[Notation],
                       quarterTempo: F,
                       profile: QuantizationProfile,
                       beatDuration: F,
                       beatOffset: F,
                       divisionHint: tuple[division_t, float] | None = None,
                       minTieDur=F(1, 10000),
                       prevDivision: division_t | None = None
                       ) -> QuantizedBeat:
    """
    Calculate the best subdivision

    Args:
        eventsInBeat: a list of Notations, where the offset is relative to the start of the
            measure and should not extend outside the beat.
            The duration can be left undefined (as -1) if the event to which this attack
            refers extends to the next attack or to the end of the beat.
        beatDuration: duration of the beat, in quarter notes (1=quarter, 0.5=eigth note)
        beatOffset: offset (start time) of this beat in relation to the beginning of the meaure
        quarterTempo: the tempo corresponding to a quarter note
        minTieDur: min. value of a tied note. Tied notes shorter than this
            are absorved by the previous note. In general these are the result
            of quantization errors
        divisionHint: if given, a tuple (division to prioritize, strength).
        profile: the subdivision preset used

    Returns:
        a QuantizedBeat, where:
        .divisions constains a list of the subdivisions of the beat where:
            * (4,) = subdivide the beat in four equal parts
            * (3, 4) = subdivide the beat in two parts, the first part in 3 and the second in 4 parts
            * (5, 3, 7) = subdivide the beat in three parts, then each of these parts in 5, 3, and 7 slots
        .assignedSlots constains a list of the assigned slot to each attack

    """
    # assert all(not ev.tiedNext for ev in eventsInBeat), eventsInBeat
    assert beatDuration > 0
    beatDuration = asF(beatDuration)

    assert beatDuration in {F(1, 1), F(1, 2), F(1, 4), F(2, 1)}, f"{beatDuration=}"

    if len(eventsInBeat) > 2:
        last = eventsInBeat[-1]
        if 0 < last.duration < minTieDur:
            if not last.isRest and not last.tiedPrev and not last.tiedNext:
                logger.warning(f"Suppressing notation {last}, duration: {last.duration}")
            eventsInBeat = eventsInBeat[:-1]
            eventsInBeat[-1].duration += last.duration

    # If only one event, bypass quantization
    if len(eventsInBeat) == 1:
        assert eventsInBeat[0].offset == beatOffset
        return QuantizedBeat((1,), assignedSlots=[0], notations=eventsInBeat,
                             beatDuration=beatDuration, beatOffset=beatOffset)

    if not _isBeatFilled(eventsInBeat, beatDuration=beatDuration, beatOffset=beatOffset):
        raise ValueError(f"Beat not filled, filling gaps: {eventsInBeat}")

    if len(eventsInBeat) > 1:
        eventsInBeat = _mergeUnquantizedNotations(eventsInBeat)

    tempo = asF(quarterTempo) / beatDuration
    possibleDivisions = profile.possibleBeatDivisionsForTempo(tempo)
    if divisionHint is not None:
        prioritizedDiv, prioritizedDivStrength = divisionHint
        possibleDivisions = [prioritizedDiv] + possibleDivisions
    else:
        prioritizedDiv, prioritizedDivStrength = None, 0.

    # (totalError, div, snappedEvents, assignedSlots, debuginfo)
    rows: list[tuple[float, division_t, list[SnappedNotation], list[int], str]] = []
    seen = set()
    events0 = [ev.clone(offset=ev.qoffset - beatOffset, spanners=ev.spanners) for ev in eventsInBeat]
    minError = 999.

    firstOffset = eventsInBeat[0].duration
    lastOffsetMargin = beatDuration - (eventsInBeat[-1].qoffset - beatOffset)

    optimizeMargins = True

    prevOuterRatio = F(*quantutils.outerTuplet(prevDivision)) if prevDivision else F0

    for div in possibleDivisions:
        if div in seen or div in profile.blacklist:
            continue

        # Exclude divisions which are not worth evaluating at full
        # NB: simplifyDivision is efficient, but it is called a lot,
        # so it is worth to find early if a division does not need to
        # be analyzed in full
        skip = False
        if len(div) > 1:
            if optimizeMargins:
                # Rule out divs with superfluous subdivisions to the left
                leftSkippedSubdivs = firstOffset // F(1, len(div))
                if leftSkippedSubdivs > 0:
                    div2 = (1,) * leftSkippedSubdivs + div[leftSkippedSubdivs:]
                    if div2 in seen:
                        # We don't continue here in order to allow for ruling out
                        # the divisions with superfluous divisions to the right
                        skip = True
                    else:
                        seen.add(div)
                        div = div2

                # Rule out divs with superfluous subdivisions to the right
                rightSkippedSubdivs = lastOffsetMargin // F(1, len(div))
                if rightSkippedSubdivs > 0:
                    div2 = div[:-rightSkippedSubdivs] + (1,) * rightSkippedSubdivs
                    if div2 in seen:
                        continue
                    else:
                        seen.add(div)
                        div = div2

                if skip:
                    continue

        if profile.maxGridDensity and max(div)*len(div) > profile.maxGridDensity:
            continue

        grid0, fgrid0 = quantutils.divisionGrid0Float(beatDuration=beatDuration, division=div)
        assignedSlots, snappedEvents = snapEventsToGrid(events0, grid=grid0, fgrid=fgrid0)
        simplifiedDiv = quantutils.simplifyDivision(div, assignedSlots, reduce=False)

        if simplifiedDiv in seen or simplifiedDiv in profile.blacklist:
            continue

        if len(simplifiedDiv) > 1:
            simplifiedDiv2 = quantutils.reduceDivision(div, newdiv=simplifiedDiv, assignedSlots=assignedSlots)
            if simplifiedDiv2 in seen:
                continue
            elif simplifiedDiv2 != simplifiedDiv:
                seen.add(simplifiedDiv)
            simplifiedDiv = simplifiedDiv2

        if simplifiedDiv != div:
            div = simplifiedDiv
            newgrid = quantutils.divisionGrid0(beatDuration=beatDuration, division=simplifiedDiv)
            assignedSlots = quantutils.resnap(assignedSlots, newgrid=newgrid, oldgrid=grid0)
        seen.add(div)

        divPenalty, divPenaltyInfo = profile.divisionPenalty(div)
        if prevDivision and prevOuterRatio != 1:
            outerRatio = F(*quantutils.outerTuplet(div))
            if prevOuterRatio == outerRatio:
                # logger.debug(f"{div=}, {prevDivision=}, {outerRatio=}, Outer tuplet match, pre. {divPenalty=}, post, {divPenalty*profile.outerTupletMatchFactor}, {divPenaltyInfo=}")
                divPenalty *= profile.outerTupletMatchFactor

        if (divError := divPenalty * sqrt(profile.divisionErrorWeight)) > minError * 1.05:
            if profile.debug and divError / minError < 1.2:
                # Only show near miss divisions, this might help tune the quantization
                logger.debug("Skipping %s, divError: %g, minError: %g", str(div), divError, minError)
            continue

        gridError = _evalGridError(profile=profile,
                                   snappedEvents=snappedEvents,
                                   beatDuration=beatDuration)


        if (weightedGridError := gridError * sqrt(profile.gridErrorWeight)) > minError:
            if profile.debug and weightedGridError / minError < 1.5:
                logger.debug("Skipping %s, weightedGridError: %g, minError: %g", str(div), weightedGridError, minError)
            continue

        rhythmComplexity, rhythmInfo = _evalRhythmComplexity(profile=profile,
                                                             snapped=snappedEvents,
                                                             div=div,
                                                             beatDur=beatDuration,
                                                             assignedSlots=assignedSlots)


        totalError = mathlib.weighted_euclidian_distance([
            (gridError, profile.gridErrorWeight),
            (divPenalty, profile.divisionErrorWeight),
            (rhythmComplexity, profile.rhythmComplexityWeight)   # XXX
        ])

        if div is prioritizedDiv:
            totalError /= prioritizedDivStrength

        if totalError > minError:
            if profile.debug and totalError / minError < 2:
                logger.debug("Skipping %s, totalError: %g, minError: %g", str(div), totalError, minError)
            continue
        else:
            minError = totalError

        debuginfo = ''
        if profile.debug:
            debuginfo = (f"{gridError=:.3g}, {rhythmComplexity=:.3g} ({rhythmInfo}), "
                         f"{divPenalty=:.3g} ({divPenalty*sqrt(profile.divisionErrorWeight):.4g}, "
                         f"{divPenaltyInfo})")

        rows.append((totalError, div, snappedEvents, assignedSlots, debuginfo))

        if totalError == 0:
            break

    # first sort by div length, then by error
    # We make sure that (7,) is better than (7, 1) for the cases where the
    # assigned slots are actually the same
    rows.sort(key=lambda r: len(r[1]))

    if profile.debug:
        rows.sort(key=lambda row: row[0])
        maxrows = min(profile.debugMaxDivisions, len(rows))
        print(f"Beat: {beatOffset} - {beatOffset + beatDuration} (dur: {beatDuration})")
        table = [(f"{r[0]:.5g}",) + r[1:] for r in rows[:maxrows]]
        misc.print_table(table, headers="error div snapped slots info".split(), floatfmt='.4f', showindex=False)

    error, div, snappedEvents, assignedSlots, debuginfo = min(rows, key=lambda row: row[0])
    notations: list[Notation] = [snapped.applySnap(extraOffset=beatOffset)
                                 for snapped in snappedEvents]

    beatNotations: list[Notation] = []
    for n in notations:
        if n.duration == 0:
            beatNotations.append(n)
        else:
            assert beatOffset <= n.qoffset < n.qoffset + n.duration <= beatOffset + beatDuration, f"{n=}, {beatOffset=}, {beatDuration=}"
            eventParts = n._breakIrregularDurationInBeat(beatDivision=div, beatDur=beatDuration,
                                                         beatOffset=beatOffset)
            if eventParts:
                beatNotations.extend(eventParts)
            elif n.duration > 0 or (n.duration == 0 and not n.isRest):
                beatNotations.append(n)
            else:
                assert n.isRest and n.duration == 0
                # Do not add a null-duration rest

    if len(beatNotations) == 1:
        n0 = beatNotations[0]
        beatEnd = beatOffset + beatDuration
        if div != (1,) and len(assignedSlots) == 1 and assignedSlots[0] == 0:
            div = (1,)
        elif n0.isRest and (n0.end == beatEnd or n0.mergeableNext) and (n0.offset == beatOffset or n0.mergeablePrev):
            div = (1,)
            beatNotations = [n0.clone(duration=beatDuration, spanners=n0.spanners)]
            assignedSlots = [0]

    if sum(n.duration for n in beatNotations) != beatDuration:
        raise AssertionError(f"{beatDuration=}, {beatNotations=}")

    return QuantizedBeat(div, assignedSlots=assignedSlots, notations=beatNotations,
                         beatDuration=beatDuration, beatOffset=beatOffset,
                         quantizationError=error, quantizationInfo=debuginfo)


def _mergeUnquantizedNotations(notations: list[Notation]) -> list[Notation]:
    """
    Consolidate notations which can be merged together, prior to quantization

    Args:
        notations: list of notations, normally within a beat

    Returns:
        list of notations where adjacent notations which can be merged
        (rests, tied notes) are merged

    """
    if len(notations) <= 1:
        raise ValueError("Not enough notations to merge")
    assert all(not n.durRatios for n in notations), f"Notations should be unquantized: {notations=}, {[n.durRatios for n in notations]}"
    out = [notations[0]]
    for n in notations[1:]:
        last = out[-1]
        if last.isRest == n.isRest and last.canMergeWith(n):
            out[-1] = last.mergeWith(n, check=False)
        else:
            out.append(n)
    return out


def quantizeBeatTernary(eventsInBeat: list[Notation],
                        quarterTempo: F,
                        profile: QuantizationProfile,
                        beatDuration: F,
                        beatOffset: F
                        ) -> list[QuantizedBeat]:
    """
    Quantize a ternary beat

    This is done by breaking the ternary beat into two
    parts, 1+2 and 2+1 and rating the result to find
    which partition is best for quantization

    Args:
        eventsInBeat: the events in this beat
        quarterTempo: the tempo corresponding to a quarter note
        profile: the quantization profile
        beatDuration: the duration of the beat
        beatOffset: the start time of the beat in quarter notes

    Returns:
        a list of quantized beats corresponding to subdivisions of this ternary
        beat (either 1+2, 2+1 or 1+1+1)
    """
    assert beatDuration.numerator == 3
    subdiv = beatDuration / 3
    possibleDistributions = [
        (beatOffset, beatOffset+subdiv*2, beatOffset+subdiv*3),  # 2 + 1
        (beatOffset, beatOffset+subdiv, beatOffset+subdiv*3),    # 1 + 2
        # (beatOffset, beatOffset+subdiv, beatOffset+subdiv*2, beatOffset+subdiv*3)  # 1+1+1
    ]

    results = []
    for offsets in possibleDistributions:
        eventsInSubbeats = quantutils.breakNotationsByBeat(eventsInBeat, offsets)
        beats = [quantizeBeatBinary([ev.copy(spanners=True) for ev in events], quarterTempo=quarterTempo, profile=profile,
                                    beatDuration=end-start, beatOffset=start)
                 for start, end, events in eventsInSubbeats]
        totalerror = sum(beat.quantizationError * beat.duration for beat in beats)
        results.append((totalerror, beats))
    if profile.debug:
        for result in results:
            error, beats = result
            durations = [beat.duration for beat in beats]
            print(f"Error: {error}, division: {durations}")
    beats = min(results, key=lambda result: result[0])[1]
    return beats


def _notationNeedsBreak(n: Notation, beatDur: F, beatDivision: division_t,
                        beatOffset=F0) -> bool:
    """
    Does this notation need to be broken?

    Args:
        n: the notation. It should already be quantized
        beatDur: the duration of the beat, in quarters
        beatDivision: the division of the beat
        beatOffset: when does this beat start

    Returns:
        True if the notation needs to be split

    """
    assert n.duration is not None and n.duration >= 0
    assert isinstance(beatDivision, tuple), f"Expected a tuple, got {beatDivision}"
    assert isinstance(beatDur, F), f"Expected a fraction, got {beatDur}"
    assert isinstance(beatOffset, F), f"Expected a fraction, got {beatOffset}"

    if n.end > beatOffset + beatDur:
        raise ValueError(f"n extends over the beat. "
                         f"n={n.offset} - {n.end}, beat={beatOffset} - {beatOffset+beatDur}")

    if n.duration == 0:
        return False

    if len(beatDivision) == 1:
        # division of the sort (5,)
        majordiv: int = beatDivision[0]
        slotdur = beatDur / majordiv
        nslots = n.duration / slotdur
        if nslots.denominator != 1:
            raise ValueError(f"n is not quantized with given division.\n  n={n}\n  division={beatDivision}")
        assert isinstance(nslots, F), f"Expected nslots of type F, got {type(nslots).__name__} (nslots={nslots})"
        return nslots.numerator not in quantdata.regularDurations
    else:
        # check if n extends over subdivision
        dt = beatDur / len(beatDivision)
        for tick in util.fractionRange(beatOffset, beatOffset+beatDur, dt):
            if n.qoffset < tick < n.end:
                return True
        # n is confined to one subdivision of the beat, find which
        now = beatOffset
        for i, div in enumerate(beatDivision):
            if now <= n.qoffset < now+dt:
                # found!
                return _notationNeedsBreak(n, beatDur=dt, beatDivision=(div,), beatOffset=now)
        return False


def isRegularDuration(dur: F, beatDur: F) -> bool:
    """
    Is the duration regular?

    This function operates on unquantized notations. Quantized notations
    should call Notation.hasRegularDuration

    Regular durations are those which can be represented
    without ties - either binary units (1, 2, 4, 8, ...) or dotted notes
    (3, 6, 7, ...).

    Args:
        dur: the duration to evaluate
        beatDur: the duration of the beat

    Returns:
        True if this duration is regular

    """
    if dur == 0:  # a gracenote?
        return True
    assert dur < beatDur
    dur2 = dur / beatDur
    if dur2.denominator > 128:
        return False
    if dur2.numerator not in quantdata.regularDurations:
        return False
    return True


def quantizeMeasure(events: list[Notation],
                    timesig: st.TimeSignature,
                    quarterTempo: F,
                    profile: QuantizationProfile,
                    beatStructure: list[st.BeatDef],
                    ) -> QuantizedMeasure:
    """
    Quantize notes in a given measure

    Args:
        events: the events inide the measure. The offset is relative
            to the beginning of the measure. Offset and duration are in
            quarterLengths, i.e. they are not dependent on tempo. The tempo
            is used as a hint to find a suitable quantization
        timesig: the time signature of the measure: a tuple (num, den)
        quarterTempo: the tempo of the measure using a quarter note as refernce
        profile: the quantization preset. Leave it unset to use the default
            preset.


    Returns:
        a QuantizedMeasure

    """
    measureDur = timesig.quarternoteDuration
    assert all(ev0.end == ev1.offset for ev0, ev1 in itertools.pairwise(events)), "Events not stacked"
    assert sum(ev.duration for ev in events) == measureDur, "Measure not filled"

    quantizedBeats: list[QuantizedBeat] = []
    beatOffsets = [beat.offset for beat in beatStructure]
    beatOffsets.append(beatStructure[-1].end)

    idx = 0
    for spanstart, spanend, eventsInBeat in quantutils.breakNotationsByBeat(events, beatOffsets=beatOffsets):
        beatWeight = beatStructure[idx].weight
        beatdur = spanend - spanstart
        ev0 = eventsInBeat[0]
        quanthint = ev0.findAttachment(attachment.QuantHint)
        if beatdur.numerator in (1, 2, 4):
            quantizedBeat = quantizeBeatBinary(eventsInBeat=eventsInBeat,
                                               quarterTempo=quarterTempo,
                                               beatDuration=spanend - spanstart,
                                               beatOffset=spanstart,
                                               profile=profile,
                                               divisionHint=None if not quanthint else (quanthint.division, quanthint.strength),
                                               prevDivision=quantizedBeats[-1].divisions if quantizedBeats else None)
            quantizedBeat.weight = beatWeight
            quantizedBeats.append(quantizedBeat)
        elif beatdur.numerator == 3:
            subBeats = quantizeBeatTernary(eventsInBeat=eventsInBeat,
                                           quarterTempo=quarterTempo,
                                           beatDuration=beatdur,
                                           beatOffset=spanstart,
                                           profile=profile)
            subBeats[0].weight = beatWeight
            quantizedBeats.extend(subBeats)
        else:
            raise ValueError(f"beat duration not supported: {beatdur}")
        idx += 1

    quantizedBeats[0].weight = 2
    return QuantizedMeasure(timesig=timesig, quarterTempo=asF(quarterTempo), beats=quantizedBeats,
                            quantprofile=profile)


def splitNotationAtMeasures(n: Notation, struct: st.ScoreStruct
                            ) -> list[tuple[int, Notation]]:
    """
    Split a Notation at measure boundaries

    Args:
        n: the Notation to split. It should have a set offset
        struct: the ScoreStructure

    Returns:
        a list of tuples (measure number, notation), indicating
        to which measure each part belongs to. The notation in the
        tuple has an offset relative to the beginning of the measure

    """
    assert n.offset is not None and n.offset >= 0 and n.duration >= 0
    measureindex0, beat0 = struct.beatToLocation(n.offset)
    measureindex1, beat1 = struct.beatToLocation(n.end)

    if measureindex0 is None or measureindex1 is None:
        raise ValueError(f"Could not find a score location for this event: {n}")

    if beat1 == F0 and n.duration > 0:
        # Note ends at the barline
        measureindex1 -= 1
        beat1 = struct.getMeasureDef(measureindex1).durationQuarters

    numMeasures = measureindex1 - measureindex0 + 1

    if numMeasures == 1:
        # The note fits within one measure. Make the offset relative to the measure
        event = n.clone(offset=beat0, duration=beat1 - beat0, spanners=n.spanners)
        return [(measureindex0, event)]

    measuredef = struct.getMeasureDef(measureindex0)
    dur = measuredef.durationQuarters - beat0
    # First part
    notation = n.clone(offset=beat0, duration=dur, tiedNext=True)
    pairs = [(measureindex0, notation)]

    # add intermediate measure, if any
    if numMeasures > 2:
        for m in range(measureindex0 + 1, measureindex1):
            measuredef = struct.getMeasureDef(m)
            notation = n.cloneAsTie(duration=measuredef.durationQuarters,
                                    tiedPrev=True,
                                    tiedNext=True,
                                    offset=F0)
            pairs.append((m, notation))

    # add last notation
    if beat1 > F0:
        notation = n.cloneAsTie(offset=F0, duration=beat1, tiedPrev=True, tiedNext=n.tiedNext)
        pairs.append((measureindex1, notation))
        
    parts = [part for measidx, part in pairs]
    n._copySpannersToSplitNotation(parts)

    for idx, part in pairs[:-1]:
        assert part.isRest or part.tiedNext, f"{n=}, {pairs=}"
    for idx, part in pairs[1:]:
        assert part.isRest or part.tiedPrev, f"{n=}, {pairs=}"

    sumdur = sum(struct.beatDelta((i, n.qoffset), (i, n.end)) for i, n in pairs)
    assert sumdur == n.duration, f"{n=}, {sumdur=}, {numMeasures=}\n{pairs=}"
    return pairs


def _mergeNodes(node1: Node,
                node2: Node,
                profile: QuantizationProfile,
                beatOffsets: Sequence[F],
                beatWeights: Sequence[int]
                ) -> Node:
    """
    Merge two nodes into a single node.

    Args:
        node1: The first node to merge.
        node2: The second node to merge.
        profile: The quantization profile.
        beatOffsets: The offsets of the beat subdivisions.

    Returns:
        The merged node.
    """
    # we don't check here, just merge
    assert node1.parent is node2.parent
    node = Node(ratio=node1.durRatio, items=node1.items + node2.items, parent=node1.parent)
    node = node.mergedNotations()
    out = _mergeSiblings(node, profile=profile, beatOffsets=beatOffsets, beatWeights=beatWeights)
    out.parent = node1.parent
    out.setParentRecursively()

    return out


def _nodesCanMerge(g1: Node,
                   g2: Node,
                   profile: QuantizationProfile,
                   beatOffsets: Sequence[F],
                   beatWeights: Sequence[int]
                   ) -> Result:
    """
    Returns Result.Ok() if the given nodes can merge, Result.Fail(errormsg) otherwise

    Args:
        g1: first node
        g2: second node
        profile: the quantization profile
        beatOffsets: the offsets of the beat subdivisions. Any Node is always
            circumscribed to one measure but can excede a beat

    Returns:
        a Result

    """
    if g1.end != g2.offset:
        raise ValueError(f"The nodes are not neighbours: {g1.end=}, {g2.offset=}")

    if g1.parent is None or g2.parent is None:
        raise ValueError("Cannot merge root node")

    if g1.durRatio != g2.durRatio:
        return Result.Fail("not same durRatio")

    if g1.durRatio != (1, 1) and g1.parent.durRatio != (1, 1) and g1.totalDuration() + g2.totalDuration() == g1.parent.totalDuration():
        return Result.Fail("A parent cannot hold a group of the same size of itself")

    for i, offset in enumerate(beatOffsets):
        if g1.end == offset:
            acrossBeat = i
            assert i > 0
            beat1Dur = offset - beatOffsets[i-1]
            break
    else:
        acrossBeat = 0
        beat1Dur = F0

    g1last = g1.lastNotation()
    g2first = g2.firstNotation()
    if not g1last.tiedNext and g1.durRatio != (1, 1):
        return Result.Fail("Nodes do not need to merge")

    if g1.durRatio == (1, 1) and len(g1) == len(g2) == 1:
        if g1last.gliss and g1last.tiedPrev and g1.symbolicDuration() + g2.symbolicDuration() > 1:
            return Result.Fail('A glissando over a beat needs to be broken at the beat')
        if not g1last.canMergeWith(g2first):
            return Result.Fail('Cannot merge notations')
        # Special case: always merge binary beats with single items since there is always
        # a way to notate those
        return Result.Ok()

    mergedSymbolicDur = g1last.symbolicDuration() + g2first.symbolicDuration()
    if g1.durRatio == (3, 2) and mergedSymbolicDur == F(3, 2):
        return Result.Fail("Don't merge 3/2 when the merged Notation results in dotted quarter")

    if g1.durRatio != (1, 1):
        g1dur = g1.totalDuration()
        g2dur = g2.totalDuration()
        if acrossBeat and g1.durRatio[0] not in profile.allowedTupletsAcrossBeat:
            return Result.Fail("tuplet not allowed to merge across beat")
        elif g1dur + g2dur > profile.mergedTupletsMaxDuration:
            return Result.Fail("incompatible duration")
        elif not profile.mergeTupletsDifferentDur and acrossBeat and g1dur != g2dur:
            return Result.Fail("Tuplet nodes of different duration cannot merge across beats")

    item1, item2 = g1.items[-1], g2.items[0]
    syncopated = g1last.tiedNext or (g1last.isRest and g2first.isRest and g1last.durRatios == g2first.durRatios)
    if not g1last.mergeableNext:
        syncopated = False

    if acrossBeat:
        if not syncopated:
            return Result.Fail('no need to extend node over beat')
        beatWeight = beatWeights[acrossBeat]
        if beatWeight > profile.breakSyncopationsMinWeight():
            return Result.Fail(f'Joining these nodes would result in a syncopation'
                               f' across a beat with a weight of {beatWeight}, but '
                               f'the current quantization profile sets a min. level of {beatWeight}')
        # logger.debug(f"{acrossBeat=}, {syncopated=}, {beatWeight=}, {profile.breakSyncopationsMinWeight()=}")

    if type(item1) is not type(item2):
        return Result.Fail("A Node cannot merge with a single item")

    if isinstance(item1, Node):
        assert isinstance(item2, Node)
        if not (r := _nodesCanMerge(item1, item2, profile=profile, beatOffsets=beatOffsets, beatWeights=beatWeights)):
            return Result.Fail(f'nested tuplets cannot merge: {r.info}')
        else:
            nestedtup = (g1.durRatio[0], item1.durRatio[0])
            if acrossBeat and item1.durRatio != (1, 1) and g1.durRatio != (1, 1) and nestedtup not in profile.allowedNestedTupletsAcrossBeat:
                return Result.Fail(f'complex nested tuplets cannot merge: {nestedtup}')
            return Result.Ok()
    else:
        if TYPE_CHECKING: assert isinstance(item2, Notation)
        # Two Notations
        if not acrossBeat and not syncopated and g1.durRatio == g2.durRatio == (3, 2):
            return Result.Fail('Merging these tuplets is not needed')

        if not acrossBeat:
            return Result.Ok()

        symdur: F = item1.symbolicDuration() + item2.symbolicDuration()

        if syncopated and symdur.denominator not in (1, 2, 4, 8, 16):
            return Result.Fail(f'Cannot merge notations resulting in irregular durations. Resulting symbolic duration: {symdur}')

        if item1.gliss and item1.tiedNext and item2.gliss:
            if symdur >= 2 and item1.tiedPrev:
                return Result.Fail("Cannot merge glissandi resulting in long (>= halfnote) notes")

        if not profile.allowNestedTupletsAcrossBeat:
            g1nested = any(isinstance(item, Node) and item.durRatio != g1.durRatio
                           for item in g1.items)
            if g1nested:
                return Result.Fail("Cannot merge nested tuples 1")

            g2nested = any(isinstance(item, Node) and
                           item.durRatio != (1, 1) and
                           item.durRatio != g2.durRatio != (1, 1)
                           for item in g2.items)
            if g2nested:
                return Result.Fail("Cannot merge nested tuples 2")

        if item1.duration > 0 and item2.duration > 0:
            syncopationAsymmetry = item1.duration / item2.duration
            if syncopationAsymmetry < 1:
                syncopationAsymmetry = 1 / syncopationAsymmetry
            if syncopationAsymmetry > profile.syncopMaxAsymmetry:
                return Result.Fail(f'The syncopation asymmetry is too big: {item1=}, {item2=}, '
                                   f'{syncopationAsymmetry=}')

        mergeddur = item1.duration + item2.duration
        minMergedDur = beat1Dur * profile.syncopMinFraction
        if mergeddur < minMergedDur:
            return Result.Fail(f'Relative duration of merged Notations across beat too short: '
                               f'{item1=}, {item2=}, min. merged duration: {float(minMergedDur):g}, beat dur: {beat1Dur}')

        minSyncopationSideDuration = profile.syncopPartMinFraction * beat1Dur
        if item1.duration < minSyncopationSideDuration:
            return Result.Fail(f'Rel. duration of {item1} too short to merge with {item2}. '
                               f'Min side duration: {float(minSyncopationSideDuration):g}')

        if item2.duration < minSyncopationSideDuration:
            return Result.Fail(f'Rel. duration of {item2} too short to merge with {item1}. '
                               f'Min side duration: {float(minSyncopationSideDuration):g}')


        if (asymettry := quantutils.asymettry(item1.duration, item2.duration)) > profile.syncopMaxAsymmetry:
            return Result.Fail(f'Assymetry between parts of a syncopation too big, '
                               f'{asymettry=}, {profile.syncopMaxAsymmetry=}')

        if g1.durRatio == (3, 2) and item1.symbolicDuration() == item2.symbolicDuration() == 1 and item1.tiedNext:
            return Result.Fail('Not needed')

        return Result.Ok()


def mergeSiblings(root: Node,
                  profile: QuantizationProfile,
                  beatOffsets: Sequence[F],
                  beatWeights: Sequence[int]) -> Node:
    newroot = _mergeSiblings(root, profile=profile, beatOffsets=beatOffsets, beatWeights=beatWeights)
    newroot.setParentRecursively()
    newroot.repair()
    return newroot


def _mergeSiblings(root: Node,
                   profile: QuantizationProfile,
                   beatOffsets: Sequence[F],
                   beatWeights: Sequence[int],
                   maxiter=10
                   ) -> Node:
    """
    Merge sibling tree of the same kind, if possible (recursively)

    Args:
        root: the root of a tree of Nodes
        profile: the quantization profile
        beatOffsets: these offsets are used to determine if a merged node
            would cross a beat boundary. The quantization profile has some
            rules regarding merging tuplets across beat boundaries which need
            this information

    Returns:
        a new tree. Caller needs to call .setParentRecursively() on the returned
        root

    """
    for _ in range(maxiter):
        root2 = _mergeSiblings0(root=root, profile=profile,
                                beatOffsets=beatOffsets, beatWeights=beatWeights)
        if root2 == root:
            break
        root = root2
    else:
        logger.debug("Could not converge in %d iterations", maxiter)
    return root


def _mergeSiblings0(root: Node,
                    profile: QuantizationProfile,
                    beatOffsets: Sequence[F],
                    beatWeights: Sequence[int]
                    ) -> Node:

    # merge only tree (not Notations) across tree of same level
    if len(root.items) < 2:
        return root

    items = []
    for item2 in root.items:
        if isinstance(item2, Node):
            item2 = _mergeSiblings(item2, profile=profile, beatOffsets=beatOffsets, beatWeights=beatWeights)
            item2.parent = root
        if not items:
            items.append(item2)
            continue

        item1 = items[-1]
        if isinstance(item1, Node) and isinstance(item2, Node):
            assert item1.parent is item2.parent, f"Invalid parents: {item1.parent=}, {item2.parent=}"
            if item1.durRatio != item2.durRatio:
                items.append(item2)
            else:
                if r := _nodesCanMerge(item1, item2, profile=profile, beatOffsets=beatOffsets, beatWeights=beatWeights):
                    mergednode = _mergeNodes(item1, item2, profile=profile, beatOffsets=beatOffsets, beatWeights=beatWeights)
                    items[-1] = mergednode
                else:
                    if profile.debug:
                        logger.debug("Nodes cannot merge: %s\n%s\n%s", r.info, LazyStr.str(item1), LazyStr.str(item2))
                        # logger.debug(f'Nodes cannot merge ({r.info}): \n{item1}\n{item2}')
                    items.append(item2)
        elif isinstance(item1, Notation) and isinstance(item2, Notation) and item1.canMergeWith(item2):
            items[-1] = item1.mergeWith(item2)
        else:
            items.append(item2)
    newroot = Node(ratio=root.durRatio, items=items, parent=root.parent)
    assert root.totalDuration() == newroot.totalDuration()
    return newroot


# def _maxTupletLength(timesig: timesig_t, subdivision: int):
#     den = timesig[1]
#     if subdivision == 3:
#         return {2: 2, 4: 2, 8: 1}[den]
#     elif subdivision == 5:
#         return 2 if den == 2 else 1
#     else:
#         return 1


@dataclass
class QuantizedPart:
    """
    A UnquantizedPart which has already been quantized following a ScoreStruct

    A QuantizedPart is a part of a :class:`QuantizedScore`
    """
    struct: st.ScoreStruct
    """The scorestructure used for quantization"""

    measures: list[QuantizedMeasure]
    """The measures of this part"""

    quantProfile: QuantizationProfile
    """QuantizationProfile used for quantization"""

    name: str = ''
    """The name of this part, used as staff name"""

    shortName: str = ''
    """The abbreviated staff name"""

    groupid: str = ''
    """A groupid, if applicable"""

    groupName: tuple[str, str] | None = None

    firstClef: str = ''
    """The first clef of this part"""
    
    possibleClefs: tuple[str, ...] = ()
    """Clefs to use when auto clef changes is used"""

    autoClefChanges: bool | None = None
    """If True, add clef changes when rendering this Part; None=use default.
    This corresponds to RenderOptions.autoClefChanges. Any part with manual
    clef changes will not be modified. To modify such a part see
    :meth:`QuantizedPart.addClefChanges`"""

    showName: bool = True
    """If True, show part name when rendered"""

    readonly: bool = False

    def __post_init__(self):
        for measure in self.measures:
            measure.parent = self
        self.repair()
        if self.readonly:
            for measure in self.measures:
                measure.setReadOnly(True)

    def __getitem__(self, index: int) -> QuantizedMeasure:
        return self.measures[index]

    def __len__(self) -> int:
        return len(self.measures)

    def setReadOnly(self, value: bool) -> None:
        for m in self.measures:
            m.setReadOnly(value)

    def repair(self):
        # self._repairGracenotesInBeats()
        firstnote = next(self.measures[0].tree.recurse())
        if (clef := firstnote.findAttachment(attachment.Clef)):
            self.firstClef = clef.kind

        self.removeUnnecessaryGracenotes()
        self.repairLinks(tieSpelling=True)
        self.repairSpanners()

    def check(self):
        for measure in self.measures:
            measure.check()

    def show(self, fmt='png', backend=''):
        """
        Show this quantized part as notation

        Args:
            fmt: the format to show, one of 'png', 'pdf'
            backend: the backend to use. One of 'lilypond', 'musicxml'
        """
        self.render(backend=backend).show(fmt=fmt)

    def render(self, 
               options: renderoptions.RenderOptions | None = None, 
               backend=''
               ) -> renderer.Renderer:
        """
        Render this quantized part

        Args:
            options: the RenderOptions to use
            backend: the backend to use. If not given the backend defined in the
                render options will be used instead

        Returns:
            the Renderer

        """
        score = QuantizedScore(parts=[self])
        return score.render(options=options, backend=backend)

    def __iter__(self) -> Iterator[QuantizedMeasure]:
        return iter(self.measures)

    def __hash__(self):
        measureHashes = tuple(hash(m) for m in self.measures)
        return hash(('QuantizedPart', self.name) + measureHashes)

    def flatNotations(self) -> Iterator[Notation]:
        """Iterate over all notations in this part"""
        for measure in self.measures:
            yield from measure.tree.recurse()

    def averagePitch(self, maxNotations=0) -> float:
        """
        The average pitch of this part

        Args:
            maxNotations: if given, only the first *maxNotations* are considered
                for calculating the average pitch.

        Returns:
            the average pitch of the notations in this part (0 if this part is empty)
        """
        accum, num = 0., 0
        for n in self.flatNotations():
            if not n.isRest:
                accum += n.meanPitch()
                num += 1
                if maxNotations and  num > maxNotations:
                    break
        return accum/num if num > 0 else 0

    def findLogicalTie(self, n: Notation) -> LogicalTie | None:
        """
        Given a Notation which is part of a logical tie (it is tied or tied to), return the logical tie

        Args:
            n: a Notation which is part of a logical tie

        Returns:
            a list of TreeLocation representing the logical tie the notation *n* belongs to

        """
        for tie in self.logicalTies():
            if any(loc.notation is n for loc in tie):
                return tie
        return None

    def logicalTies(self) -> list[LogicalTie]:
        """
        Return a list of logical ties in this part

        A logical tie is a sequence of notations that are tied together.
        """
        # return _logicalTies(self)
        ties = []
        for i, measure in enumerate(self.measures):
            ties.extend(measure.logicalTies())

        if len(ties) < 2:
            return ties

        current = ties[0]
        mergedties = [current]
        for tie in ties[1:]:
            if tie[0].notation.tiedPrev and current[-1].notation.tiedNext:
                current.extend(tie)
            else:
                mergedties.append(current)
                current = tie
        mergedties.append(current)
        return mergedties

    def dump(self, numindents=0, indent=_INDENT, tree=True, stream=None):
        """Dump this part to a stream or stdout"""
        for i, m in enumerate(self.measures):
            ind = _INDENT * numindents
            print(f'{ind}Measure #{i}', file=stream or sys.stdout)
            m.dump(numindents=numindents + 1, indent=indent, tree=tree, stream=stream)

    def bestClef(self) -> str:
        """
        Return the best clef for the notations in this part

        The returned str if one of 'treble', 'treble8',
        'bass' and 'bass8'

        Returns:
            the clef descriptor which best fits this part; one of 'treble',
            'treble8', 'bass', 'bass8', where the 8 indicates an octave
            transposition in the direction of the clef (high for treble,
            low for bass)
        """
        notations = list(n for n in self.flatNotations() if not n.isRest)
        if not notations:
            return "treble"
        return clefutils.bestClefForNotations(notations)

    def findClefChanges(self,
                        apply=True,
                        removeManualClefs=False,
                        window=1,
                        simplificationThreshold=0.,
                        biasFactor=1.5,
                        propertyKey='',
                        minClef='',
                        maxClef='',
                        possibleClefs: Sequence[str] = ()
                        ) -> None:
        """
        Determines the most appropriate clef changes for this part

        The clef changes are added as properties to the notations at which
        the changes are to be made. If called with ``addClefs=True``,
        these clef changes are materialized as clef attachments

        Args:
            apply: if True, clef change directives are actually added to the
                quantized notations. Otherwise, only hints given as properties are added
            removeManualClefs: if True, remove any manual clef
            window: the window size when determining the best clef for a given section
            simplificationThreshold: a simplification threshold. A value of 0. disables simplification
            biasFactor: The higher this value, the more weight is given to the
                previous clef, thus making it more difficult to change clef
                for minor jumps
            minClef: if given, only clefs equal or higher to this can be used
            maxClef: if given, only clefs equal or lower to this can be used
            possibleClefs: if given, a seq. of allowed clefs
            propertyKey: the property key to add to the notation to mark
                a clef change. Setting this property alone will not
                result in a clef change in the notation (see `addClefs`)

        """
        notations = list(self.flatNotations())
        if removeManualClefs:
            for n in notations:
                if n.attachments:
                    n.removeAttachmentsByClass(attachment.Clef)
        if not possibleClefs:
            possibleClefs = self.possibleClefs
        # This adds the clef changes as attachment to the notation prior to which
        # the clef change has effect.
        clefutils.findBestClefs(notations,
                                addClefs=apply,
                                windowSize=window,
                                simplificationThreshold=simplificationThreshold,
                                biasFactor=biasFactor,
                                key=propertyKey,
                                firstClef=self.firstClef,
                                possibleClefs=possibleClefs,
                                minClef=minClef,
                                maxClef=maxClef)
        if (clef := notations[0].findAttachment(attachment.Clef)):
            self.firstClef = clef.kind

    def resolveEnharmonics(self, options: enharmonics.EnharmonicOptions) -> None:
        """
        Resolve enharmonic spelling, in place

        Args:
            options: the enharmonic options to use
        """
        prevMeasure = None
        for i, measure in enumerate(self.measures):
            measure._pinEnharmonicSpelling(options=options, prevMeasure=prevMeasure)
            prevMeasure = measure

    def removeRedundantDynamics(self,
                                resetAfterEmptyMeasure=True,
                                resetTime: int = 0,
                                resetAfterRest: int = 0,
                                resetAfterCustomBarline=True) -> None:
        """
        Remove superfluous dynamics in this part, inplace

        Args:
            resetAfterEmptyMeasure: dynamics are reset after an empty measure
            resetTime: if given, dynamics are reset (forgotten) after this number of quarters
                after last change
        """
        dynamic = ''
        now = F(0)
        lastChange = F(-10000)
        restAccum = F0
        struct = self.struct
        customBarlines = ('double', 'double-thin', 'final', 'solid')
        for i, meas in enumerate(self.measures):
            if resetAfterCustomBarline and i > 0 and struct.getMeasureDef(i - 1).barline in customBarlines:
                dynamic = ''

            if meas.empty():
                if resetAfterEmptyMeasure:
                    dynamic = ''
            else:
                for n in meas.notations():
                    if resetTime and now - lastChange > resetTime:
                        dynamic = ''
                    if n.isRest:
                        restAccum += n.duration
                        if resetAfterRest and restAccum >= resetAfterRest:
                            dynamic = ''
                            restAccum = F0
                    else:
                        restAccum = F0
                        if not n.tiedPrev and n.dynamic and n.dynamic in definitions.dynamicLevels:
                            # Only dynamic levels are ever superfluous (f, ff, mp), other 'dynamics'
                            # like sf should not be removed
                            if n.dynamic == dynamic:
                                logger.debug(f"Removing dynamic for {n} at measure {i} ({now=})")
                                n.dynamic = ''
                            else:
                                dynamic = n.dynamic
                                lastChange = now
            now += meas.duration()

    def removeUnnecessaryGracenotes(self) -> None:
        """
        Removes unnecessary gracenotes, in place

        An unnecessary gracenote fullfills one of the following conditions:

        * has the same pitch as the next real note and starts a glissando.
          Such gracenotes might be created during quantization.
        * has the same pitch as the previous real note and ends a glissando
        * n0/real -- gliss -- n1/grace n2/real and n1.pitches == n2.pitches

        """
        measure: QuantizedMeasure
        for i, measure in enumerate(self.measures):
            if measure.tree.empty():
                continue
            measure.removeUnnecessaryGracenotes()
            if i > 0 and (n1 := measure.tree.firstNotation()).isGracenote:
                n0 = self.measures[i-1].tree.lastNotation()
                if n0.tiedNext and n0.pitches == n1.pitches and not (n1.attachments) and not (n1.spanners):
                    # n1 is an unnecessary gracenote
                    logger.debug("Found unnecessary grace note: %s at measure %d. "
                                 "It is tied to %s from the previous measure but adds"
                                 "nothing to it", n1, i, n0)
                    node = measure.tree.findNodeForNotation(n1)
                    assert node is not None
                    node.items.remove(n1)

    def repairSpanners(self) -> None:
        """
        Match orfan spanners, optionally removing unmatched spanners (in place)

        """
        # _spanner.removeUnmatchedSpanners(self.flatNotations(tree=tree))
        notations = list(self.flatNotations())
        _spanner.solveHairpins(notations)
        _spanner.matchOrfanSpanners(notations=notations, removeUnmatched=False)
        openspanners = _spanner.markSpannerNestingLevel(notations)
        if openspanners:
            for n in notations:
                if not n.spanners:
                    continue
                for sp in openspanners:
                    if sp in n.spanners:
                        n.spanners.remove(sp)
                        
        for n in notations:
            if n.spanners:
                uuids: list[str] = [spanner.uuid for spanner in n.spanners]
                for duplicateuuid in misc.duplicates(uuids):  # type: ignore
                    spanners = [spanner for spanner in n.spanners if spanner.uuid == duplicateuuid]
                    start = next((s for s in spanners if s.kind == 'start'), None)
                    end = next((s for s in spanners if s.kind == 'end'), None)
                    logger.warning(f"Duplicate spanners found: {spanners}, {self=}")
                    if start and end:
                        logger.warning(f"Start/end spanner at the same notation, removing spanner {start}/{end}, {self=}")
                        n.removeSpanner(duplicateuuid)
                    else:
                        # Only start / end spanner, keep only one
                        logger.warning(f"Duplicate spanners with uuid {duplicateuuid}, removing ({self=})")
                        n.removeSpanner(duplicateuuid)
                        n.addSpanner(spanners[0])

    def getMeasure(self, idx: int, extend=True) -> QuantizedMeasure | None:
        """
        Get a measure within this part

        Args:
            idx: the measure index (starts at 0)
            extend: if True and the index is outside the defined measures,
                a new empty QuantizedMeasure will be created and added
                to this part

        Returns:
            The corresponding measure. If outside the defined measures a new
            empty QuantizedMeasure will be created
        """
        numMeasures = len(self.measures)
        if idx > numMeasures - 1:
            if not extend:
                return None
            for i in range(numMeasures - 1, idx+1):
                # We create empty measures as needed
                mdef = self.struct.getMeasureDef(i)
                qmeasure = QuantizedMeasure(timesig=mdef.timesig,
                                            quarterTempo=mdef.quarterTempo,
                                            beats=[],
                                            quantprofile=self.quantProfile,
                                            parent=self)
                self.measures.append(qmeasure)
        return self.measures[idx]

    def repairLinks(self, tieSpelling=True) -> None:
        """
        Repairs ties and glissandi (in place)

        Args:
            tieSpelling: if True, ensures that tied notes share the same spelling
        """
        ties = self.logicalTies()

        for n0, n1 in itertools.pairwise(self.flatNotations()):
            if n0.tiedNext:
                if n0.isRest or n1.isRest or set(n0.pitches).isdisjoint(set(n1.pitches)):
                    # No pitches in common
                    n0.tiedNext = False
                    n1.tiedPrev = False
                else:
                    hints0 = n0.tieHints('forward')
                    hints1 = n1.tieHints('backward')
                    for idx0, pitch0 in enumerate(n0.pitches):
                        idx1 = next((idx for idx, pitch in enumerate(n1.pitches) if pitch == pitch0), None)
                        if idx1 is not None:
                            note0 = n0.notename(idx0)
                            note1 = n1.notename(idx1)
                            if tieSpelling and note1 != note0:
                                n1.fixNotename(note0, index=idx1)
                            hints0.add(idx0)
                            hints1.add(idx1)

            elif n0.gliss:
                if n1.isRest or n0.pitches == n1.pitches:
                    if n0.tiedPrev:
                        logicalTie = next((tie for tie in ties if n0 in tie), None)
                        if logicalTie:
                            for n in logicalTie:
                                n.notation.gliss = False
                    else:
                        n0.gliss = False
                    n0.tiedNext = True
                    n1.tiedPrev = True

    def addEmptyMeasures(self, numMeasures: int) -> None:
        """Add the given number of empty measures at the end"""
        if numMeasures <= 0:
            return
        N = len(self.measures)
        for measureIndex in range(N - 1, N - 1 + numMeasures):
            measuredef = self.struct.getMeasureDef(measureIndex)
            empty = QuantizedMeasure(timesig=measuredef.timesig,
                                     quarterTempo=measuredef.quarterTempo,
                                     beats=[],
                                     quantprofile=self.quantProfile,
                                     parent=self)
            self.measures.append(empty)

    def resolveChordEnharmonics(self, enharmonicOptions: enharmonics.EnharmonicOptions | None = None
                                ) -> None:
        """
        Finds the best enharmonic variant for each chord in this part, individually

        As an alternative for finding the best global spelling it is possible to
        just fix each chord individually

        """
        from . import enharmonics
        for measure in self.measures:
            for n in measure.notations():
                if n.isRest or len(n.pitches) <= 1:
                    continue
                notenames = n.resolveNotenames(addFixedAnnotation=True)
                spellings = enharmonics.bestChordSpelling(notenames, options=enharmonicOptions)
                for i, spelling in enumerate(spellings):
                    n.fixNotename(spelling, i)

    def addSpanner(self, spanner: _spanner.Spanner | str, start: beat_t, end: beat_t
                   ) -> None:
        """
        Adds a spanner between two notations at the given locations

        Args:
            spanner: the spanner to add. A Spanner or the name of the spanner class
                ('slur', 'beam', etc.)
            start: the start location.
            end: the end location. Notice that to match a notation it must
                be present at this location. If the notation ends exactly at this
                location it will not be matched. For example, if a slur is needed
                across all notes within beat 0 and 1, one would call this method
                as ``part.addSpanner('slur', 0, 1-F(1, 100000))

        Returns:

        """
        n0, _ = self.notationAt(start)
        n1, _ = self.notationAt(end)
        n0.addSpanner(spanner, n1)

    def notationAt(self, beat: beat_t) -> tuple[Notation, QuantizedMeasure]:
        """
        Returns the event at the given beat / location

        If the beat/location given is within the boundaries of this part,
        this method should always return a notation.


        Args:
            beat: the beat as absolute offset or location (measureindex, beatinmeasure)

        Returns:
            a tuple (notation, measure), where measure is the measure to which the
            returned event belongs. Bear in mind that within a quantized part
            a notation is always included within the boundaries of the measure.
            The notation itself might be tied to a previous/next notation.

        """
        measure, relbeat = self.measureAt(beat)
        tree = measure.tree
        for n in tree.recurse():
            assert n.offset is not None
            if n.offset <= relbeat < n.end:
                return n, measure
        raise ValueError(f"No event at beat {beat}")

    def measureAt(self, beat: beat_t) -> tuple[QuantizedMeasure, F]:
        """
        Returns the measure at the given location

        Args:
            beat: an absolute beat in quarter notes or a location as
                tuple (measure idx, relative beat)

        Returns:
            a tuple (measure: QuantizedMeasure, relative beat: F)

        """
        if isinstance(beat, tuple):
            measureidx, relbeat = beat
        else:
            measureidx, relbeat = self.struct.beatToLocation(beat)

        if measureidx is None or measureidx >= len(self.measures):
            raise IndexError(f"Location {beat} is outside of this part's boundaries")
        measure = self.measures[measureidx]
        if relbeat > measure.duration():
            raise ValueError(f"The relative beat {relbeat} exceeds the duration of the measure ({measure})")
        return measure, relbeat

    def splitNotationAt(self, offset: beat_t, tie=True, mergeable=False) -> list[Notation] | None:
        """
        Split any notation present at offset in place, returns the resulting parts

        Args:
            offset: the beat offset
            tie: if True, tie the resulting parts
            mergeable: if False, mark the marks as unmergeable

        Returns:
            the resulting parts or None if no notations present at the given offset.
            Raises ValueError if the given offset is not within the span of this
            measure

        """

        measure, relbeat = self.measureAt(offset)
        return measure.splitNotationAt(relbeat, tie=tie, mergeable=mergeable)

    def breakBeam(self, location: beat_t) -> Notation | None:
        """
        Break beams at a given location

        Args:
            location: the beat or location as tuple (measureindex, relative beat) to
                break beams at

        Returns:
            the notation at which beams are broken (the notation at the given offset)
            or None if no break is possible at the given location
        """
        measure, relbeat = self.measureAt(location)
        return measure.tree.breakBeamsAt(relbeat)

    def breakSyncopations(self, level: str = 'weak') -> None:
        """
        Break notes extending over beat boundaries, inplace

        * 'all': break syncopations at any beat boundary
        * 'weak': break syncopations at weak accent beats (for example, the 3rd
          beat in a 4/4 bar)
        * 'strong': break syncopations only at strong beats

        Args:
            level: one of 'all', 'weak', 'strong'

        """
        for m in self.measures:
            m.breakSyncopations(level=level)

    def breakSyncopationAt(self,
                           location: F | tuple[int, F],
                           ) -> list[Notation] | None:
        """
        Break a syncopation/beam at the given beat/location

        This method works **in place** at the tree level

        Args:
            location: an absolute offset in quarter notes, or a location as
                tuple (measure index, relative offset)

        Returns:
            the notations resulting of the split operation, or None
            if no notation was broken
        """
        measure, relbeat = self.measureAt(location)
        return measure.splitNotationAt(relbeat, tie=True, mergeable=False)

    def coreVoice(self) -> maelzel.core.Voice:
        qs = QuantizedScore([self])
        corescore = qs.coreScore()
        return corescore.voices[0]


def quantizePart(part: core.UnquantizedPart,
                 struct: st.ScoreStruct,
                 quantprofile: QuantizationProfile,
                 fillStructure=False,
                 ) -> QuantizedPart:
    """
    Quantizes a sequence of non-overlapping events (a "part")

    Quantize to the score structure defined in `struct`, according to the strategies
    defined in `preset`

    Args:
        struct: the ScoreStruct to use
        part: the events to quantize. Event within a part
            should not overlap
        fillStructure: if True and struct is not endless, the
            generated UnquantizedPart will have as many measures as are defined
            in the struct. Otherwisem only as many measures as needed
            to hold the given events will be created
        quantprofile: the QuantizationProfile used

    Returns:
        a QuantizedPart

    """
    assert isinstance(part, core.UnquantizedPart)
    part.fillGaps()
    notations = part.notations
    core.resolveOffsets(notations)
    quantutils.fixGlissWithinTiesInPlace(notations)
    allpairs = [splitNotationAtMeasures(n=n, struct=struct) for n in notations]
    maxMeasure = max(pairs[-1][0] for pairs in allpairs)
    notationsPerMeasure: list[list[Notation]] = [[] for _ in range(maxMeasure+1)]
    for pairs in allpairs:
        for measureIdx, notation in pairs:
            notationsPerMeasure[measureIdx].append(notation)
    qmeasures = []
    for idx, notations in enumerate(notationsPerMeasure):
        measureDef = struct.getMeasureDef(idx)
        beatStruct = measureDef.beatStructure()
        if not notations:
            qmeasures.append(QuantizedMeasure(timesig=measureDef.timesig,
                                              quarterTempo=measureDef.quarterTempo,
                                              beats=[],
                                              quantprofile=quantprofile))
        else:
            if not misc.issorted(notations, key=lambda n: n.offset):
                raise ValueError(f"Notations are not sorted: {notations}")
            core.removeSmallOverlaps(notations)
            if sum(n.duration for n in notations) != measureDef.durationQuarters:
                notations = quantutils.fillSpan(notations, F0, measureDef.durationQuarters)
            qmeasure = quantizeMeasure(notations,
                                       timesig=measureDef.timesig,
                                       quarterTempo=measureDef.quarterTempo,
                                       profile=quantprofile,
                                       beatStructure=beatStruct)
            qmeasures.append(qmeasure)
    if fillStructure:
        if struct.endless:
            raise ValueError("Cannot fill an endless ScoreStructure")
        for i in range(maxMeasure+1, struct.numMeasures()):
            measureDef = struct.getMeasureDef(i)
            qmeasure = QuantizedMeasure(timesig=measureDef.timesig,
                                        quarterTempo=measureDef.quarterTempo,
                                        beats=[],
                                        quantprofile=quantprofile)
            qmeasures.append(qmeasure)
    qpart = QuantizedPart(struct, 
                          qmeasures, 
                          name=part.name, 
                          shortName=part.shortName,
                          groupid=part.groupid, 
                          quantProfile=quantprofile,
                          groupName=part.groupName,
                          showName=part.showName, 
                          firstClef=part.firstClef,
                          possibleClefs=part.possibleClefs)
    if quantprofile.breakSyncopationsLevel != 'none':
        for measure in qpart:
            measure.breakSyncopations(level=quantprofile.breakSyncopationsLevel)

    for hook in part.hooks:
        if isinstance(hook, attachment.PostPartQuantHook):
            hook(qpart)
        else:
            logger.warning(f"Unknown hook: {hook}")
    return qpart


class QuantizedScore:
    """
    A QuantizedScore represents a list of quantized parts

    See :func:`quantize` for an example

    Args:
        parts: the parts of this QuantizedScore
        title: an optional title for this score
        composer: an optional composer for this score

    Attributes:
        parts: the parts of this score
        title: an optional title
        composer: an optional composer
    """
    __slots__ = ('parts', 'title', 'composer')

    def __init__(self,
                 parts: list[QuantizedPart],
                 title='',
                 composer='',
                 ):
        if not parts:
            raise ValueError("Score must have at least one part")

        self.parts: list[QuantizedPart] = parts
        """A list of QuantizedParts"""

        self.title: str = title
        """Title of the score, used for rendering purposes"""

        self.composer: str = composer
        """Composer of the score, used for rendering"""


    def check(self):
        """Check this QuantizedScore"""

        for pidx, part in enumerate(self.parts):
            part.check()

    def setReadOnly(self, value: bool) -> None:
        for part in self.parts:
            part.setReadOnly(value)

    def resolveEnharmonics(self, enharmonicOptions: enharmonics.EnharmonicOptions) -> None:
        """
        Finds the best spelling for each part in this score, inplace

        Args:
            enharmonicOptions: the enharmonic options to use
        """
        for part in self.parts:
            part.resolveEnharmonics(enharmonicOptions)

    def resolveChordEnharmonics(self, enharmonicOptions: enharmonics.EnharmonicOptions | None = None
                                ) -> None:
        """
        Finds the best enharmonic variant for each chord individually and pins it to it

        As an alternative for finding the best global spelling it is possible to
        just fix each chord individually

        """
        for part in self.parts:
            part.resolveChordEnharmonics(enharmonicOptions=enharmonicOptions)

    def __hash__(self):
        partHashes = [hash(p) for p in self.parts]
        return hash((self.scorestruct, self.title, self.composer) + tuple(partHashes))

    def __getitem__(self, item: int) -> QuantizedPart:
        return self.parts[item]

    def __iter__(self) -> Iterator[QuantizedPart]:
        return iter(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def __repr__(self):
        import io
        stream = io.StringIO()
        self.dump(tree=False, stream=stream)
        return stream.getvalue()

    def dump(self, tree=True, indent=_INDENT, stream=None, numindents: int = 0) -> None:
        """
        Dump this QuantizedScore to a given stream or to stdout

        Args:
            tree: if True, use the tree representation for each measure
            indent: the indentation to use
            stream: the stream to write to
            numindents: the starting indentation

        """
        for i, part in enumerate(self):
            print(f"{indent*numindents}Part #{i}:", file=stream)
            part.dump(tree=tree, numindents=numindents+1, indent=indent, stream=stream)

    @property
    def scorestruct(self) -> st.ScoreStruct:
        """Returns the ScoreStruct of this score"""
        if not self.parts:
            raise IndexError("This QuantizedScore has no parts")
        return self.parts[0].struct

    @property
    def quantprofile(self) -> QuantizationProfile:
        if not self.parts:
            raise IndexError("This QuantizedScore has no parts")
        return self.parts[0].quantProfile

    @scorestruct.setter
    def scorestruct(self, struct: st.ScoreStruct) -> None:
        if self.parts:
            for part in self.parts:
                part.struct = struct

    def numMeasures(self) -> int:
        """Returns the number of measures in this score"""
        return max(len(part.measures)
                   for part in self.parts)

    def padEmptyMeasures(self) -> None:
        """Adds empty measures at the end of each part so that all have the same length"""
        numMeasures = self.numMeasures()
        for part in self.parts:
            part.addEmptyMeasures(numMeasures - len(part.measures))

    def groupParts(self) -> list[list[QuantizedPart]]:
        """
        Group parts which have the same id

        At the moment we do not support subgroups

        Returns:
            A list of groups where a group is a list of parts with the same id
        """
        groups = {}
        out: list[list[QuantizedPart]] = []
        for part in self.parts:
            if part.groupid:
                group = groups.get(part.groupid)
                if group is None:
                    groups[part.groupid] = group = []
                    out.append(group)
                group.append(part)
            else:
                out.append([part])
        return out

    def write(self,
              outfile: str,
              options: renderer.RenderOptions | None = None,
              backend='',
              format=''
              ) -> renderer.Renderer:
        """
        Export this score as pdf, png, lilypond, MIDI or musicxml

        When rendering to pdf or png both the lilypond or the
        musicxml backend can be used.

        Args:
            outfile: the path of the written file. Use 'stdout' to print to stdout
            options: render options used to generate the output
            backend: backend used when writing to png / pdf (one of 'lilypond', 'musicxml')
            format: format used (one of 'pdf', 'png', 'musicxml', 'lilypond'). If not given
                it is inferred from the file extension.

        Returns:
            the Renderer used
        """
        ext = os.path.splitext(outfile)[1].lower()
        if not format:
            format = {'.ly': 'lilypond',
                      '.xml': 'musicxml',
                      '.musicxml': 'musicxml',
                      '.pdf': 'pdf',
                      '.png': 'png'}.get(ext)
        if format == 'lilypond' or format == 'ly':
            r = self.render(options=options, backend='lilypond')
            if outfile == 'stdout':
                print(r.render())
            else:
                r.write(outfile)
            return r
        elif format == 'musicxml':
            r = self.render(options=options, backend='musicxml')
            if outfile == 'stdout':
                print(r.render())
            else:
                r.write(outfile)
            return r
        elif format in ('pdf', 'png'):
            assert outfile != 'stdout'
            r = self.render(options=options, backend=backend)
            r.write(outfile)
            return r
        else:
            raise ValueError(f"Format {format} ({ext=}) not supported, possible formats are 'pdf', 'png', 'musicxml', 'lilypond'")

    def show(self, backend='', fmt='png', external: bool = False) -> None:
        self.render(backend=backend).show(fmt=fmt, external=external)

    def render(self,
               options: renderer.RenderOptions | None = None,
               backend: str = ''
               ) -> renderer.Renderer:
        """
        Render this quantized score

        Args:
            options: the RenderOptions to use
            backend: the backend to use. If not given the backend defined in the
                render options will be used instead

        Returns:
            the Renderer

        """
        from . import render
        if options is None:
            from maelzel.core import workspace
            cfg = workspace.getConfig()
            options = cfg.makeRenderOptions()
            if backend:
                options.backend = backend
        elif backend and backend != options.backend:
            options = options.clone(backend=backend)
        return render.renderQuantizedScore(self, options=options)

    def coreScore(self) -> maelzel.core.Score:
        """
        Convert this QuantizedScore to a :class:`~maelzel.core.score.Score`

        Returns:
            the corresponding maelzel.core.Score

        Example
        -------

            >>> from maelzel.core import *
            >>> chain = Chain([...
        """
        from .notation import notationsToCoreEvents
        import maelzel.core
        voices = []
        for part in self.parts:
            events = []
            for measure in part:
                notations = list(measure.tree.recurse())
                events.extend(notationsToCoreEvents(notations))
            voice = maelzel.core.Voice(events)
            if part.name:
                voice.name = part.name
            if part.shortName:
                voice.shortname = part.shortName
            voices.append(voice)
        return maelzel.core.Score(voices=voices, scorestruct=self.scorestruct, title=self.title)

    def toCoreScore(self) -> maelzel.core.Score:
        """
        DEPRECATED. Convert this QuantizedScore to a :class:`~maelzel.core.score.Score`

        Use .coreScore instead

        Returns:
            the corresponding maelzel.core.Score

        Example
        -------

            >>> from maelzel.core import *
            >>> chain = Chain([...
        """
        import warnings
        warnings.warn("Deprecated, use .coreScore()")
        return self.coreScore()


def quantizeParts(parts: list[core.UnquantizedPart],
                  quantizationProfile: QuantizationProfile,
                  struct: st.ScoreStruct | None = None,
                  enharmonicOptions: enharmonics.EnharmonicOptions | None = None
                  ) -> QuantizedScore:
    """
    Quantize and render unquantized notations organized into parts

    Args:
        parts: a list of Parts, where each part represents a series
            of non-overlapping events which have not yet been quantized
        struct:
            the structure of the resulting score. To create a simple score
            with an anitial time signature and tempo, use something like
            `ScoreStructure.fromTimesig((4, 4), quarterTempo=52)`. If not given,
            defaults to a 4/4 score with tempo 60
        quantizationProfile:
            The quantization preset determines how events are quantized,
            which divisions of the beat are possible, how the best division
            is weighted and selected, etc. Not all options in a preset
            are supported by all backends (for example, the musicxml backend
            does not support nested tuplets).
            See quant.presetQuantizationProfiles, which is a dict with
            some predefined profiles
        enharmonicOptions: if given, these are used to find the most suitable
            enharmonic representation

    Returns:
        a QuantizedScore

    Example
    -------

    .. code::

        >>> from maelzel import scoring
        >>> from maelzel.scorestruct import ScoreStruct
        >>> scorestruct = ScoreStruct('''
        ... 4/4, 80
        ... 3/4
        ... 3/4
        ... 5/8, 60
        ... 7/8
        ... 4/4
        ... ...
        ... ''')
        >>> scorestruct
        =========== ======= ====================  =========
        Meas. Index	Timesig	Tempo (quarter note)	Label
        =========== ======= ====================  =========
        0           4/4     80
        1           3/4
        2           3/4
        3           5/8     60
        4           7/8
        5           4/4
        =========== ======= ====================  =========

        >>> notes = [
        ... (0.5, "4C"),
        ... (1.5, "4C+"),
        ... (1/3, "4D-25"),
        ... (2/3, "4E+25"),
        ... (2+1/5, "4F#+"),
        ... (5.8, "4A-10")
        ... ]
        >>> notations = [scoring.Notation(duration=dur, pitches=[p]) for dur, p in notes]
        >>> part = scoring.UnquantizedPart(notations)
        >>> qscore = scoring.quant.quantize([part], struct=scorestruct)
        >>> qscore.parts[0].dump()

        Timesig: 4/4 (quarter=80)
          Ratio (1, 1)
            «0.000:0.500 4C»
            «0.500:2.000 4C+»
          Ratio (3, 2)
            «2.000:2.333 3/2 4D-25»
            «2.333:3.000 3/2 4E+25»
          Ratio (1, 1)
            «3.000:4.000 tiedNext 4F#+»
        Timesig: 3/4 (quarter=80)
          Ratio (1, 1)
            «0.000:1.000 tiedPrev tiedNext 4F#+»
          Ratio (5, 4)
            «1.000:1.200 5/4 tiedPrev 4F#+»
            «1.200:2.000 5/4 tiedNext 4A-10»
          Ratio (1, 1)
            «2.000:3.000 tiedPrev tiedNext 4A-10»
        Timesig: 3/4 (quarter=80)
          Ratio (1, 1)
            «0.000:3.000 tiedPrev tiedNext 4A-10»
        Timesig: 5/8 (quarter=60)
          Ratio (1, 1)
            «0.000:1.000 tiedPrev 4A-10»
          Ratio (1, 1)
            «1.000:2.500 rest»

        >>> renderopts = scoring.render.RenderOptions(showCents=True)
        >>> renderer = scoring.render.renderQuantizedScore(qscore, options=renderopts,
        ...                                                backend='lilypond')
        >>> renderer.write("~/tmp/foo.pdf")

    .. image:: ../assets/quantize-example.png
    """
    if not parts:
        raise ValueError("No parts provided")
    if struct is None:
        struct = st.ScoreStruct((4, 4), tempo=60)
    qparts = []
    for i, part in enumerate(parts):
        profile = part.quantProfile or quantizationProfile
        try:
            qpart = quantizePart(part, struct=struct, quantprofile=profile)
            qpart.check()
            qparts.append(qpart)
        except Exception as e:
            e.add_note(f"Error while quantizing part {i}")
            raise e
    qscore = QuantizedScore(qparts)
    if enharmonicOptions:
        qscore.resolveEnharmonics(enharmonicOptions)
    else:
        qscore.resolveChordEnharmonics()

    return qscore
