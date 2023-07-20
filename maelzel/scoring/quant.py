"""
Quantize durations to musical notation

The most important function here is :func:`quantize`, which treturns
a :class:`QuantizedScore`

"""
from __future__ import annotations

from dataclasses import dataclass
import sys
import os
import time
from math import sqrt

from maelzel.common import F, F0
from maelzel._util import humanReadableTime

from .common import *

from . import core
from . import definitions
from . import util
from . import quantdata
from . import quantutils
from . import clefutils
from . import enharmonics
from . import renderer
from . import spanner as _spanner
from . import renderoptions
from . import attachment
from .quantprofile import QuantizationProfile

from .notation import Notation, makeRest, SnappedNotation, tieNotations
from .node import Node, asTree, SplitError
from maelzel import scorestruct as st

from emlib import iterlib
from emlib import misc
from emlib.result import Result
from emlib import mathlib

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.scoring.common import timesig_t
    from typing import Iterator, Sequence
    import maelzel.core


__all__ = (
    'quantize',
    'QuantizedScore',
    'QuantizedPart',
    'QuantizedMeasure',
    'QuantizedBeat',
    'quantizeMeasure',
    'quantizePart',
    'splitNotationAtMeasures',
    'splitNotationsAtOffsets',
    'PartLocation',
    'QuantizationProfile'
)

_INDENT = "  "


_regularDurations = {0, 1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32}


def _fitEventsToGridNearest(events: list[Notation], grid: list[F]) -> list[int]:
    # We use floats to make this faster. Rounding errors should not pose a problem
    # in this context
    fgrid = [float(g) for g in grid]
    return [misc.nearest_index(float(event.offset), fgrid) for event in events]


def snapEventsToGrid(notations: list[Notation],
                     grid: list[F],
                     ) -> tuple[list[int], list[SnappedNotation]]:
    """
    Snap unquantized events to a given grid

    Args:
        notations: a list of unquantized Notation's
        grid: the grid to snap the events to, as returned by generateBeatGrid

    Returns:
        tuple (assigned slots, quantized events)
    """
    beatDuration = grid[-1]
    assignedSlots = _fitEventsToGridNearest(events=notations, grid=grid)
    snappedEvents = []
    for idx in range(len(notations)-1):
        n = notations[idx]
        slot0 = assignedSlots[idx]
        offset0 = grid[slot0]
        # is it the last slot (as grace note?)
        if slot0 == len(grid) - 1:
            snappedEvents.append(SnappedNotation(n, offset0, F0))
        else:
            offset1 = grid[assignedSlots[idx+1]]
            snappedEvents.append(SnappedNotation(n, offset0, offset1-offset0))

    lastOffset = grid[assignedSlots[-1]]
    dur = beatDuration - lastOffset
    last = SnappedNotation(notations[-1], lastOffset, duration=dur)
    snappedEvents.append(last)
    return assignedSlots, snappedEvents


def _isBeatFilled(events: list[Notation], beatDuration: F, beatOffset: F = F0) -> bool:
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
            and all(ev0.end == ev1.offset for ev0, ev1 in iterlib.pairwise(events))
            and events[-1].end == beatOffset + beatDuration)


def _eventsShow(events: list[Notation]) -> str:
    lines = [""]
    for ev in events:
        back = "←" if ev.tiedPrev else ""
        forth = "→" if ev.tiedNext else ""
        tiedStr = f"tied: {back}{forth}"
        if ev.duration is None:
            lines.append(f"  {util.showF(ev.offset)} – .. {tiedStr}")
        else:
            lines.append(f"  {util.showF(ev.offset)} – {util.showF(ev.end)} "
                         f"dur={util.showF(ev.duration)} {tiedStr}")
    return "\n".join(lines)


def _fillDuration(notations: list[Notation], duration: F, offset=F0) -> list[Notation]:
    """
    Fill a beat/measure with silences / extend unset durations to next notation

    After calling this, the returned list of notations should fill the given
    duration exactly. This function is normally called prior to quantization

    Args:
        notations: a list of notations inside the beat
        duration: the duration to fill
        offset: the starting time to fill

    Returns:
        a list of notations which fill the beat exactly

    .. note::

        If any notation has an unset duration, this will extend either to
        the next notation or to fill the given duration

    """
    assert all(n.offset is not None for n in notations)
    assert all(n.offset-offset <= duration for n in notations), \
        f"Events start after duration to fill ({duration=}): {_eventsShow(notations)}"
    assert all(n0.offset <= n1.offset for n0, n1 in iterlib.pairwise(notations)), \
        f"events are not sorted: {_eventsShow(notations)}"
    assert all(n0.end <= n1.offset for n0, n1 in iterlib.pairwise(notations) if n0.duration is not None), \
        f"events overlap: {_eventsShow(notations)}"
    assert all(n.end <= offset+duration for n in notations if n.duration is not None), \
        "events extend over beat duration"

    out = []
    now = offset

    if not notations:
        # measure is empty
        out.append(makeRest(duration, offset=now))
        return out

    if notations[0].offset > now:
        out.append(makeRest(notations[0].offset-now, offset=now))
        now = notations[0].offset

    for n0, n1 in iterlib.pairwise(notations):
        if n0.offset > now:
            # there is a gap, fill it with a rest
            out.append(makeRest(offset=now, duration=n0.offset - now))
        if n0.duration is None:
            out.append(n0.clone(duration=n1.offset - n0.offset))
        else:
            out.append(n0)
            if n0.end < n1.offset:
                out.append(makeRest(offset=n0.end, duration=n1.offset - n0.end))
        now = n1.offset

    # last event
    n = notations[-1]
    if n.duration is None:
        out.append(n.clone(duration=duration-n.offset))
    else:
        out.append(n)
        if n.end < offset + duration:
            out.append(makeRest(offset=n.end, duration=duration + offset - n.end))
    end = offset + duration
    assert sum(n.duration for n in out) == duration
    assert all(offset <= n.offset <= end for n in out)
    return out


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
    graceNoteDuration = profile.gracenoteDuration
    graceNoteOffsetErrorFactor = 0.5
    beatdur = float(beatDuration)
    numGracenotes = 0
    totalOffsetError = 0
    totalDurationError = 0
    for snapped in snappedEvents:
        event = snapped.notation
        offsetError = abs(event.offset - snapped.offset) / beatdur
        if event.isRest:
            offsetError *= restOffsetErrorWeight / offsetErrorWeight

        if snapped.duration == 0:
            numGracenotes += 1
            offsetError *= graceNoteOffsetErrorFactor
            durationError = abs(event.duration - graceNoteDuration) / beatdur
        else:
            durationError = abs(event.duration - snapped.duration) / beatdur

        totalOffsetError += offsetError
        totalDurationError += durationError

    gracenoteError = numGracenotes / len(snappedEvents)
    error = mathlib.euclidian_distance(
        [totalOffsetError, totalDurationError, gracenoteError],
        [offsetErrorWeight, profile.durationErrorWeight, profile.gracenoteErrorWeight])
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

    __slots__ = ('divisions', 'assignedSlots', 'notations', 'beatDuration',
                 'beatOffset', 'quantizationError', 'quantizationInfo',
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

        self.divisions: division_t = divisions
        "The division of this beat"

        self.assignedSlots: list[int] = assignedSlots
        "Which slots are assigned to notations in this beat"

        self.notations: list[Notation] = notations
        "The notations in this beat. They are cropped to fit"

        self.beatDuration: F = beatDuration
        "The duration of the beat in quarter notes"

        self.beatOffset: F = beatOffset
        "The offset of the beat in relation to the measure"

        self.quantizationError: float = quantizationError
        "The error calculated during quantization. The higher the error, the less accurate the quantization"

        self.quantizationInfo: str = quantizationInfo
        "Info collected during quantization"

        self.weight: int = weight
        "The weight of this beat within the measure. 2=strong, 1=weak, 0=no weight"

        self.applyDurationRatios()

    def __repr__(self):
        parts = [
            f"divisions: {self.divisions}, assignedSlots={self.assignedSlots}, "
            f"notations={self.notations}, beatDuration={self.beatDuration}, beatOffset={self.beatOffset}, "
            f"quantizationError={self.quantizationError:.5g}, weight={self.weight}"
        ]
        if self.quantizationInfo:
            parts.append(f'quantizationInfo={self.quantizationInfo}')
        return f'QuantizedBeat({", ".join(parts)})'

    @property
    def beatEnd(self) -> F:
        """The end of this beat in quarternotes"""
        return self.beatOffset + self.beatDuration

    def dump(self, indents=0, indent='  ', stream=None):
        """Dump this beat"""
        stream = stream or sys.stdout
        print(f"{indent*indents}QuantizedBeat(divisions={self.divisions}, assignedSlots={self.assignedSlots}, "
              f"beatDuration={self.beatDuration}, beatOffset={self.beatOffset}, "
              f"quantizationError={self.quantizationError:.3g})", file=stream)
        ind = indent * (indents + 1)
        for n in self.notations:
            print(f"{ind}{n}", file=stream)

    def applyDurationRatios(self):
        _applyDurationRatio(self.notations, division=self.divisions,
                            beatOffset=self.beatOffset, beatDur=self.beatDuration)

    def tree(self) -> Node:
        """
        Returns the notations in this beat as a tree

        Returns:
            a Node which is the root of a tree representing the notations in
            this beat (grouped by their duration ratio)
        """
        return _beatToTree(self.notations, division=self.divisions,
                           beatOffset=self.beatOffset, beatDur=self.beatDuration)

    def __hash__(self):
        notationHashes = [hash(n) for n in self.notations]
        data = [self.divisions, self.beatDuration, self.beatOffset]
        data.extend(notationHashes)
        return hash(tuple(data))


class QuantizedMeasure:
    """
    A QuantizedMeasure holds a list of QuantizedBeats

    Those QuantizedBeats are merged together in a recursive structure
    to generate a tree of Nodes. See :meth:`QuantizedMeasure.tree`
    """
    def __init__(self,
                 timesig: timesig_t,
                 quarterTempo: F,
                 beats: list[QuantizedBeat] | None = None,
                 quantprofile: QuantizationProfile | None = None,
                 parent: QuantizedPart | None = None):
        self.timesig = timesig
        "The time signature (a tuple num, den)"

        self.quarterTempo = quarterTempo
        "The tempo for the quarter note"

        self.beats = beats
        "A list of QuantizedBeats"

        self.quantprofile = quantprofile
        "The quantization profile used to generate this measure"


        self.parent = parent
        "The parent of this measure (a QuantizedPart)"

        self._offsets = None
        self._root: Node | None = None

        if self.beats:
            self.check()

    def __repr__(self):
        parts = [f"timesig={self.timesig}, quarterTempo={self.quarterTempo}, beats={self.beats}"]
        if self.quantprofile:
            parts.append(f"profile={self.quantprofile.name}")
        return f"QuantizedMeasure({', '.join(parts)})"

    def __hash__(self):
        if not self.beats:
            return hash((self.timesig, self.quarterTempo, self.getMeasureIndex()))
        else:
            return hash((self.timesig, self.quarterTempo) + tuple(hash(b) for b in self.beats))

    def resetTree(self):
        """Remove any cached tree representation of this measure"""
        self._root = None

    def getMeasureIndex(self) -> int | None:
        """Return the measure index of this measure within the QuantizedPart"""
        if not self.parent:
            return None
        return self.parent.measures.index(self)

    def measureDef(self) -> st.MeasureDef | None:
        if not self.parent or not self.parent.struct:
            return None
        measureindex = self.getMeasureIndex()
        if measureindex is None:
            return None
        return self.parent.struct.getMeasureDef(measureindex)

    def getPreviousMeasure(self) -> QuantizedMeasure | None:
        """Returns the previous measure in the part"""
        idx = self.getMeasureIndex()
        if idx is None:
            return None
        if idx == 0:
            return None
        assert self.parent is not None
        return self.parent.measures[idx - 1]

    def duration(self) -> F:
        """
        Duration of this measure, in quarter notes

        Returns:
            the duration of the measure in quarter notes
        """
        if self.isEmpty():
            return util.measureQuarterDuration(self.timesig)
        return sum(self.beatDurations())

    def beatBoundaries(self) -> list[F]:
        """
        The beat offsets, including the end of the measure

        This method will return the boundaries even if the measure itself
        is empty. In that case these boundaries are the boundaries of
        the beat structure, as defined in the score structure

        Returns:
            the boundaries of the beats, which is the offsets plus the end of the measure
        """
        boundaries = self.beatOffsets().copy()
        boundaries.append(boundaries[0] + self.duration())
        return boundaries

    def beatOffsets(self) -> list[F]:
        """
        Returns a list of the offsets of each beat within this measure

        Returns:
            the offset of each beat
        """
        if self.isEmpty():
            measuredef = self.measureDef()
            if measuredef is None:
                return []
            beatStructure = measuredef.beatStructure()
            beatOffsets = [beat.offset for beat in beatStructure]
            # beatOffsets.append(beatStructure[-1].end)
            self._offsets = beatOffsets
        elif self._offsets is None:
            self._offsets = [beat.beatOffset for beat in self.beats]

        return self._offsets

    def fixEnharmonics(self, options: enharmonics.EnharmonicOptions) -> None:
        """
        Pin the enharmonic spellings within this measure (inplace)

        Args:
            options: the EnharmonicOptions to use

        """
        if self.isEmpty():
            return
        tree = self.tree()
        first = tree.firstNotation()
        if not first.isRest and first.tiedPrev:
            prevMeasure = self.getPreviousMeasure()
            assert prevMeasure is not None
            if prevMeasure.isEmpty():
                raise ValueError("The first note of this measure is tied to the previous"
                                 " note, but the previous measure is empty")
            prevTree = prevMeasure.tree()
        else:
            prevTree = None
        tree.fixEnharmonics(options=options, prevTree=prevTree)

    def isEmpty(self) -> bool:
        """
        Is this measure empty?

        Returns:
            True if empty
        """
        if not self.beats:
            return True
        for beat in self.beats:
            if beat.notations and any(not n.isRest or n.hasAttributes() for n in beat.notations):
                return False
        return True

    def dump(self, numindents=0, indent=_INDENT, tree=True, stream=None):
        ind = _INDENT * numindents
        stream = stream or sys.stdout
        print(f"{ind}Timesig: {self.timesig[0]}/{self.timesig[1]} "
              f"(quarter={self.quarterTempo})", file=stream)
        if self.isEmpty():
            print(f"{ind}EMPTY", file=stream)
        elif tree:
            self.tree().dump(numindents, indent=indent, stream=stream)
        else:
            for beat in self.beats:
                beat.dump(indents=numindents, indent=indent, stream=stream)

    def notations(self, tree: bool) -> list[Notation]:
        """
        Returns a flat list of all notations in this measure

        Args:
            tree: if True, use the tree representation of the measure. Otherwise,
                the beat representation is used

        Returns:
            a list of Notations in this measure
        """
        if self.isEmpty():
            return []
        if tree:
            root = self.tree()
            return list(root.recurse())

        if not self.beats:
            return []
        notations = []
        for beat in self.beats:
            notations.extend(beat.notations)
        assert len(notations) > 0
        for n0, n1 in iterlib.pairwise(notations):
            if n0.end != n1.offset:
                logger.error(f"Error in durations:\n\t{n0=},\t{n1=}")
        return notations

    def nodes(self) -> list[Node]:
        """
        Returns the contents of this measure grouped as a list of Nodes
        """
        if not self.beats:
            return []
        nodes = [beat.tree().mergedNotations() for beat in self.beats]

        def removeUnnecessaryChildrenInplace(node: Node) -> None:
            items = []
            for item in node.items:
                if isinstance(item, Node) and len(item.items) == 1:
                    item = item.items[0]
                items.append(item)
            node.items = items

        for node in nodes:
            removeUnnecessaryChildrenInplace(node)

        dur = sum(node.totalDuration() for node in nodes)
        assert dur == self.duration()
        return nodes

    def tree(self) -> Node:
        """
        Returns the root of a tree of Nodes representing the items in this measure
        """
        if self._root is not None:
            return self._root
        if self.isEmpty():
            raise ValueError("This measure is empty")
        if not self.quantprofile:
            raise ValueError(f"Cannot create tree without a QuantizationProfile")
        self.check()
        root = asTree(self.nodes())
        root = _mergeSiblings(root, profile=self.quantprofile, beatOffsets=self.beatOffsets())
        root.repair()
        if root.totalDuration() != self.duration():
            logger.error(f"Measure index: {self.getMeasureIndex()}")
            self.dump(tree=False)
            self.dump(tree=True)
            raise ValueError(f"Duration mismatch in tree. "
                             f"Tree duration: {root.totalDuration()}, "
                             f"measure duration: {self.duration()}")
        assert root.totalDuration() == self.duration()
        self._root = root
        return root

    def breakSyncopations(self, level='weak') -> None:
        """
        Break notes extended over beat boundaries, **in place**

        The level indicates which syncopations to break. 'all' will split
        any notations extending over any beat; 'weak' will only break
        syncopations over secondary beats (for example, the 3rd quarter-note
        in a 4/4 measure); 'strong' will only break syncopations over strong
        beats (the 4th quarternote in a 6/4 measure with the form 3+3, or the 3rd
        quarternote in a 7/8 measure with the form 2+2+3)

        Args:
            level: one of 'all', 'weak', 'strong'

        """
        minWeight = {
            'all': 0,
            'weak': 1,
            'strong': 2
        }.get(level)
        if minWeight is None:
            raise ValueError(f"Expected one of 'all, 'weak', 'strong', got {level}")

        if self.isEmpty():
            return

        def dosplit(n: Notation) -> bool:
            if n.isRest:
                return False
            if n.duration < 1:
                return True
            for beat in self.beats:
                if beat.beatOffset <= n.offset < beat.beatEnd:
                    return n.offset - beat.beatOffset > 0
            else:
                raise ValueError(f"Notation {n} is not part of this Measure")

        tree = self.tree()
        for beat in self.beats:
            if beat.weight >= minWeight:
                try:
                    tree._splitNotationAtBoundary(beat.beatOffset, key=dosplit)
                except SplitError as e:
                    logger.error(f"Could not split tree at offset {beat.beatOffset}: {e}")

    def beatDurations(self) -> list[F]:
        """
        Returns a list with the durations (in quarterNotes) of the beats in this measure
        """
        if not self.beats:
            return []
        return [beat.beatDuration for beat in self.beats]

    def beatDivisions(self) -> list:
        """
        Returns the divisions of each beat in this measure
        """
        if not self.beats:
            return []
        return [beat.divisions for beat in self.beats]

    def check(self):
        if not self.beats:
            return
        # check that the measure is filled
        for i, beat in enumerate(self.beats):
            for n in beat.notations:
                assert n.duration is not None, n
                if n.isRest:
                    assert n.duration > 0, n
                else:
                    assert n.duration >= 0, n
            durNotations = sum(n.duration for n in beat.notations)
            if durNotations != beat.beatDuration:
                measnum = self.getMeasureIndex()
                logger.error(f"Duration mismatch, loc: ({measnum}, {i}). Beat dur: {beat.beatDuration}, Notations dur: {durNotations}")
                logger.error(beat.notations)
                self.dump(tree=False)
                self.dump(tree=True)
                raise ValueError(f"Duration mismatch in beat {i}")


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


def _evalRhythmComplexity(profile: QuantizationProfile,
                          snappedEvents: list[SnappedNotation],
                          division: division_t,
                          beatDur: F,
                          assignedSlots: list[int]
                          ) -> tuple[float, str]:
    # calculate notes across subdivisions
    if len(division) == 1:
        numNotesAcrossSubdivisions = 0
        slots = assignedSlots + [division[0]]
        durs = [b - a for a, b in iterlib.pairwise(slots)]
        numTies = sum(dur not in _regularDurations for dur in durs)

    else:
        # slotsAtSubdivs = [0] + list(iterlib.partialsum(division[:-1]))
        slotsAtSubdivs = [0] + list(iterlib.partialsum(division))
        numNotesAcrossSubdivisions = 0
        lastslot = sum(iterlib.flatten(division))
        for slotStart, slotEnd in iterlib.pairwise(assignedSlots + [lastslot]):
            if _crossesSubdivisions(slotStart, slotEnd, slotsAtSubdivs):

                numNotesAcrossSubdivisions += 1
        numIrregularNotes = sum(not isRegularDuration(dur=n.duration, beatDur=beatDur)
                                for n in snappedEvents)
        numTies = numIrregularNotes

    penalty = mathlib.weighted_euclidian_distance([
        (numNotesAcrossSubdivisions/len(snappedEvents), profile.rhythmComplexityNotesAcrossSubdivisionWeight),
        (numTies/len(snappedEvents), profile.rhythmComplexityIrregularDurationsWeight)
    ])
    if profile.debug:
        debugstr = f'numNotesAcrossSubdivs={numNotesAcrossSubdivisions}, {numTies=}'
    else:
        debugstr = ''
    return penalty, debugstr


def quantizeBeatBinary(eventsInBeat: list[Notation],
                       quarterTempo: F,
                       profile: QuantizationProfile,
                       beatDuration: F,
                       beatOffset: F,
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
        if 0 < last.duration < F(1, 10000):
            logger.warning(f"Suppressing notation {last}")
            eventsInBeat = eventsInBeat[:-1]
            eventsInBeat[-1].duration += last.duration

    # If only one event, bypass quantization
    if len(eventsInBeat) == 1:
        assert eventsInBeat[0].offset == beatOffset
        return QuantizedBeat((1,), assignedSlots=[0], notations=eventsInBeat,
                             beatDuration=beatDuration, beatOffset=beatOffset)

    if not _isBeatFilled(eventsInBeat, beatDuration=beatDuration, beatOffset=beatOffset):
        raise ValueError(f"Beat not filled, filling gaps: {eventsInBeat}")

    time0 = time.time()

    tempo = asF(quarterTempo) / beatDuration
    possibleDivisions = profile.possibleBeatDivisionsByTempo(tempo)
    rows = []
    seen = set()
    events0 = [ev.clone(offset=ev.offset - beatOffset) for ev in eventsInBeat]
    minError = 999.

    firstOffset = eventsInBeat[0].duration
    lastOffsetMargin = beatDuration - (eventsInBeat[-1].offset - beatOffset)

    optimizeMargins = True

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

        if profile.maxGridDensity and quantutils.divisionDensity(div) > profile.maxGridDensity:
            continue

        grid0 = quantutils.divisionGrid0(beatDuration=beatDuration, division=div)
        assignedSlots, snappedEvents = snapEventsToGrid(events0, grid=grid0)
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

        if divPenalty * sqrt(profile.divisionErrorWeight) > minError:
            continue

        gridError = _evalGridError(profile=profile,
                                   snappedEvents=snappedEvents,
                                   beatDuration=beatDuration)

        if gridError * sqrt(profile.gridErrorWeight) > minError:
            continue

        rhythmComplexity, rhythmInfo = _evalRhythmComplexity(profile=profile,
                                                             snappedEvents=snappedEvents,
                                                             division=div,
                                                             beatDur=beatDuration,
                                                             assignedSlots=assignedSlots)

        totalError = mathlib.weighted_euclidian_distance([
            (gridError, profile.gridErrorWeight),
            (divPenalty, profile.divisionErrorWeight),
            (rhythmComplexity, profile.rhythmComplexityWeight)   # XXX
        ])


        if totalError > minError:
            continue
        else:
            minError = totalError

        debuginfo = ''
        if profile.debug:
            debuginfo = (f"{gridError=:.3g}, {rhythmComplexity=:.3g} ({rhythmInfo}), " 
                         f"{divPenalty=:.3g} ({divPenalty*profile.divisionErrorWeight:.4g}, "
                         f"{divPenaltyInfo})"
                         )
        rows.append((totalError, div, snappedEvents, assignedSlots, debuginfo))

        if totalError == 0:
            break

    # first sort by div length, then by error
    # Like this we make sure that (7,) is better than (7, 1) for the cases where the
    # assigned slots are actually the same
    rows.sort(key=lambda r: len(r[1]))

    if profile.debug:
        rows.sort(key=lambda row: row[0])
        maxrows = min(profile.debugMaxDivisions, len(rows))
        print(f"Beat: {beatOffset} - {beatOffset + beatDuration} (dur: {beatDuration})")
        print(f"Best {maxrows} divisions: (quantized in {humanReadableTime(time.time()-time0)})")
        table = [(f"{r[0]:.5g}",) + r[1:] for r in rows[:maxrows]]
        misc.print_table(table, headers="error div snapped slots info".split(), floatfmt='.4f', showindex=False)

    error, div, snappedEvents, assignedSlots, debuginfo = min(rows, key=lambda row: row[0])
    notations = [snapped.makeSnappedNotation(extraOffset=beatOffset)
                 for snapped in snappedEvents]
    assert sum(n.duration for n in notations) == beatDuration, \
        f"{beatDuration=}, {notations=}"

    beatNotations = []
    for n in notations:
        if n.isGracenote:
            beatNotations.append(n)
        else:
            eventParts = breakIrregularDuration(n, beatDivision=div, beatDur=beatDuration,
                                                beatOffset=beatOffset)
            if eventParts:
                beatNotations.extend(eventParts)
            elif n.duration > 0 or (n.duration == 0 and not n.isRest):
                beatNotations.append(n)
            else:
                assert n.isRest and n.duration == 0
                # Do not add a null-duration rest

    if div != (1,) and len(beatNotations) == 1 and len(assignedSlots) == 1 and assignedSlots[0] == 0:
        div = (1,)
    elif all(n.isRest for n in beatNotations) and len(beatNotations) > 1:
        beatNotations = [beatNotations[0].clone(duration=beatDuration)]
        div = (1,)

    assert sum(n.duration for n in beatNotations) == beatDuration, f"{beatDuration=}, {beatNotations=}"
    return QuantizedBeat(div, assignedSlots=assignedSlots, notations=beatNotations,
                         beatDuration=beatDuration, beatOffset=beatOffset,
                         quantizationError=error, quantizationInfo=debuginfo)


def quantizeBeatTernary(eventsInBeat: list[Notation],
                        quarterTempo: F,
                        profile: QuantizationProfile,
                        beatDuration: F,
                        beatOffset: F
                        ) -> list[QuantizedBeat]:
    """
    Quantize a ternary beat

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
    possibleOffsets = [
        (beatOffset, beatOffset+subdiv, beatOffset+subdiv*3),    # 1 + 2
        (beatOffset, beatOffset+subdiv*2, beatOffset+subdiv*3),  # 2 + 1
        # (beatOffset, beatOffset+subdiv, beatOffset+subdiv*2, beatOffset+subdiv*3)  # 1+1+1
    ]

    results = []
    for offsets in possibleOffsets:
        eventsInSubbeats = splitNotationsAtOffsets(eventsInBeat, offsets)
        beats = [quantizeBeatBinary(events, quarterTempo=quarterTempo, profile=profile,
                                    beatDuration=span.duration, beatOffset=span.start)
                 for span, events in eventsInSubbeats]
        totalerror = sum(beat.quantizationError * beat.beatDuration for beat in beats)
        results.append((totalerror, beats))
    if profile.debug:
        for result in results:
            error, beats = result
            durations = [beat.beatDuration for beat in beats]
            print(f"Error: {error}, division: {durations}")
    beats = min(results, key=lambda result: result[0])[1]
    return beats


def _fillMeasure(eventsInMeasure: list[Notation],
                 timesig: timesig_t,
                 ) -> list[Notation]:
    """
    Helper function, ensures that the measure is filled

    Args:
        eventsInMeasure: this events should fit within the measure but don't necessarily
            fill the measure
        timesig: the time-signature of the measure

    Returns:
        a list of Notations which fill the measure without any gaps

    """
    measureDuration = util.measureQuarterDuration(timesig)
    assert all(0 <= ev.offset and ev.end <= measureDuration for ev in eventsInMeasure)
    return _fillDuration(eventsInMeasure, measureDuration)


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
    assert isinstance(beatDivision, tuple), f"Expected a list, got {type(beatDivision).__name__}"
    assert isinstance(beatDur, F), f"Expected a fraction, got {type(beatDur).__name__}"
    assert isinstance(beatOffset, F), f"Expected a fraction, got {type(beatOffset).__name__}"

    if n.end > beatOffset + beatDur:
        raise ValueError(f"n extends over the beat. "
                         f"n={n.offset} - {n.end}, beat={beatOffset} - {beatOffset+beatDur}")

    if n.duration == 0:
        return False

    if len(beatDivision) == 1:
        div = beatDivision[0]
        slotdur = beatDur/div
        nslots = n.duration / slotdur
        if nslots.denominator != 1:
            raise ValueError(f"n is not quantized with given division.\n  n={n}\n  division={beatDivision}")
        assert isinstance(nslots, F), f"Expected nslots of type {F}, got {type(nslots).__name__} (nslots={nslots})"
        return nslots.numerator not in _regularDurations
    else:
        # check if n extends over subdivision
        dt = beatDur/len(beatDivision)
        ticks = mathlib.fraction_range(beatOffset, beatOffset+beatDur, dt)
        for tick in ticks:
            if n.offset < tick < n.end:
                return True
        # n is confined to one subdivision of the beat, find which
        now = beatOffset
        for i, div in enumerate(beatDivision):
            if now <= n.offset < now+dt:
                # found!
                return _notationNeedsBreak(n, dt, div, beatOffset=now)


def _breakIrregularDuration(n: Notation, beatDur: F, div: int, beatOffset: F = F0,
                            minPartDuration=F(1,64)
                            ) -> list[Notation] | None:
    """
    Split irregular durations within a beat during quantization

    An irregular duration is a duration which cannot be expressed as a quarter/eights/16th/etc.
    For example a beat filled with a sextuplet with durations (1, 5), the second
    note is irregular and must be split. Since it begins in an uneven slot, it is
    split as 1+4

    Args:
        n: the Notation to split
        slotindex: which slot is n assigned to within the beat/subbeat
        slotdur: which is the quarterNote duration of slotDur

    ::

        5  -> 4+1 if n starts in an even slot, 1+4 if it starts in an odd slot
        9  -> 8+1 / 1+8
        10 -> 8+2 / 2+8
        11 -> 8+3 / 3+8
        13 -> 12+1 / 1+12
        15 -> 12+3 / 3+12
        17 -> 16+1 / 1+16
        18 -> 16+2 == 8+1
        19 -> 16+3 / 3+16
        20 -> 16+4 == 4+1
        21 -> 16+4+1 (quarter~16th~64th)
        22 -> 16+6 (quarter~16th·)
        23 -> 16+7 (quarter~16th··)
        25 -> 24+1 (16+9 == q~8th~64th)
        higher -> error

    """
    assert n.duration <= beatDur
    # beat is subdivided regularly
    slotdur = beatDur/div
    nslots = n.duration/slotdur

    if nslots.denominator != 1:
        raise ValueError(f"Duration is not quantized with given division.\n  {n=}, {div=}, {slotdur=}, {nslots=}")

    if nslots.numerator in _regularDurations:
        return None

    slotindex = (n.offset-beatOffset)/slotdur
    assert int(slotindex) == slotindex
    slotindex = int(slotindex)

    if not slotindex.denominator == 1:
        raise ValueError(f"Offset is not quantized with given division. n={n}, division={div}")

    # parts = _splitIrregularDuration(n, slotindex=slotindex, slotdur=slotdur)

    numslots = int(n.duration / slotdur)
    if numslots == 1:
        return [n]
    elif numslots > 25:
        raise ValueError("Division not supported")

    slotDivisions = quantdata.splitIrregularSlots(numslots=numslots, slotindex=slotindex)

    offset = F(n.offset)
    parts: list[Notation] = []
    for slots in slotDivisions:
        partDur = slotdur * slots
        assert partDur > minPartDuration
        parts.append(n.clone(offset=offset, duration=partDur))
        offset += partDur

    tieNotations(parts)
    assert sum(part.duration for part in parts) == n.duration
    assert (p0 := parts[0]).offset == n.offset and p0.tiedPrev == n.tiedPrev and p0.spanners == n.spanners
    assert (p1 := parts[-1]).end == n.end and p1.tiedNext == n.tiedNext
    return parts


def isRegularDuration(dur: F, beatDur: F) -> bool:
    """
    Is the duration regular?

    Regular durations are those which (in priciple) can be represented
    without tied - either binary units (1, 2, 4, 8, ...) or dotted notes
    (3, 6, 7, ...).

    Args:
        dur: the duration to evaluate
        beatDur: the duration of the beat

    Returns:
        True if this duration is regular

    """
    if dur == 0:  # a gracenote?
        return True
    dur2 = dur / beatDur
    if dur2.denominator > 128:
        return False
    if dur2.numerator not in _regularDurations:
        return False
    return True


def breakIrregularDuration(n: Notation,
                           beatDur: F,
                           beatDivision: division_t,
                           beatOffset: F = F0
                           ) -> list[Notation] | None:
    """
    Breaks a notation with irregular duration into its parts

    - a Notations should not extend over a subdivision of the beat if the
      subdivisions in question are coprimes
    - within a subdivision, a Notation should not result in an irregular multiple of the
      subdivision. Irregular multiples are all numbers which have prime factors other than
      2 or can be expressed with a dot
      Regular durations: 2, 3, 4, 6, 7 (double dotted), 8, 12, 16, 24, 32
      Irregular durations: 5, 9, 10, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27,
      28, 29, 30, 31

    Args:
        n: the Notation to break
        beatDur: the duration of the beat
        beatDivision: the division of the beat
        beatOffset: the offset of the beat

    Returns:
        None if the notations has a regular duration, or a list of tied Notations which
        together represent the original notation
    """

    assert beatOffset <= n.offset and n.end <= beatOffset + beatDur
    assert n.duration >= 0

    if n.duration == 0:
        return None

    if isinstance(beatDivision, (tuple, list)) and len(beatDivision) == 1:
        beatDivision = beatDivision[0]

    if isinstance(beatDivision, int):
        return _breakIrregularDuration(n, beatDur=beatDur,
                                       div=beatDivision, beatOffset=beatOffset)

    # beat is not subdivided regularly. check if n extends over subdivision
    numDivisions = len(beatDivision)
    divDuration = beatDur/numDivisions

    ticks = list(mathlib.fraction_range(beatOffset, beatOffset+beatDur+divDuration, divDuration))
    assert len(ticks) == numDivisions + 1

    subdivisionTimespans = list(iterlib.pairwise(ticks))
    subdivisions = list(zip(subdivisionTimespans, beatDivision))
    subns = n.splitAtOffsets(ticks)
    allparts: list[Notation] = []
    for subn in subns:
        # find the subdivision
        for timespan, numslots in subdivisions:
            if mathlib.intersection(timespan[0], timespan[1], subn.offset, subn.end) is not None:
                parts = breakIrregularDuration(n=subn, beatDur=divDuration, beatDivision=numslots,
                                               beatOffset=timespan[0])
                if parts is None:
                    # subn is regular
                    allparts.append(subn)
                else:
                    allparts.extend(parts)
    assert sum(part.duration for part in allparts) == n.duration
    tieNotations(allparts)
    return allparts


def _isMeasureFilled(notations: list[Notation], timesig: timesig_t) -> bool:
    """Do the notations fill the measure?"""
    measureDuration = util.measureQuarterDuration(timesig)
    notationsDuration = sum(n.duration for n in notations)
    if notationsDuration > measureDuration:
        logger.error(f"timesig: {timesig}, Notation: {notations}")
        logger.error(f"Sum duration: {notationsDuration}")
        raise ValueError("notations do not fit in measure")
    return notationsDuration == measureDuration


def splitNotationsAtOffsets(notations: list[Notation],
                            offsets: Sequence[F]
                            ) -> list[tuple[TimeSpan, list[Notation]]]:
    """
    Split the given notations between the given offsets

    **NB**: Any notations starting after the last offset will not be considered!

    Args:
        notations: the notations to split
        offsets: the boundaries.

    Returns:
        a list of tuples (timespan, notation)

    """
    timeSpans = [TimeSpan(beat0, beat1) for beat0, beat1 in iterlib.pairwise(offsets)]
    splittedEvents = []
    for ev in notations:
        if ev.duration > 0:
            splittedEvents.extend(ev.splitAtOffsets(offsets))
        else:
            splittedEvents.append(ev)

    eventsPerBeat = []
    for timeSpan in timeSpans:
        eventsInBeat = []
        for ev in splittedEvents:
            if timeSpan.start <= ev.offset < timeSpan.end:
                assert ev.end <= timeSpan.end
                eventsInBeat.append(ev)
        eventsPerBeat.append(eventsInBeat)
        assert sum(ev.duration for ev in eventsInBeat) == timeSpan.end - timeSpan.start
        assert all(timeSpan.start <= ev.offset <= ev.end <= timeSpan.end
                   for ev in eventsInBeat)
    return list(zip(timeSpans, eventsPerBeat))


def _beatToTree(notations: list[Notation], division: int | division_t,
                beatOffset: F, beatDur: F
                ) -> Node:
    if isinstance(division, tuple) and len(division) == 1:
        division = division[0]
    if isinstance(division, int):
        durRatio = quantdata.durationRatios[division]
        return Node(ratio=durRatio, items=notations)

    # assert isinstance(division, tuple) and len(division) >= 2
    numSubBeats = len(division)
    now = beatOffset
    dt = beatDur/numSubBeats
    durRatio = quantdata.durationRatios[numSubBeats]
    items = []
    for subdiv in division:
        subdivEnd = now+dt
        subdivNotations = [n for n in notations if now <= n.offset < subdivEnd and n.end <= subdivEnd]
        if subdiv == 1:
            items.extend(subdivNotations)
        else:
            items.append(_beatToTree(subdivNotations, division=subdiv, beatOffset=now, beatDur=dt))
        now += dt
    return Node(durRatio, items)


def _applyDurationRatio(notations: list[Notation],
                        division: int | division_t,
                        beatOffset: F,
                        beatDur: F
                        ) -> None:
    """
    Applies a duration ratio to each notation, recursively.

    A duration ratio converts the actual duration of a notation to its
    notated value and is used to render these as tuplets later

    Args:
        notations: the notations inside the period beatOffset:beatOffset+beatDur
        division: the division of the beat/subbeat. Examples: 4, [3, 4], [2, 2, 3], etc
        beatOffset: the start of the beat
        beatDur: the duration of the beat

    """
    if isinstance(division, int) or len(division) == 1:
        if isinstance(division, tuple):
            division = division[0]
        durRatio = F(*quantdata.durationRatios[division])
        if durRatio != 1:
            for n in notations:
                if n.durRatios is None:
                    n.durRatios = []
                # if not(durRatio == 1 and n.durRatios and n.durRatios[-1] == 1):
                # if not (durRatio == 1 and n.durRatios and n.durRatios[-1] == 1):
                n.durRatios.append(durRatio)
    else:
        numSubBeats = len(division)
        now = beatOffset
        dt = beatDur / numSubBeats
        durRatio = F(*quantdata.durationRatios[numSubBeats])
        if durRatio != 1:
            for n in notations:
                if n.durRatios is None:
                    n.durRatios = []
                # if not (durRatio == 1 and n.durRatios and n.durRatios[-1] == 1):

                    n.durRatios.append(durRatio)
        for subdiv in division:
            subdivEnd = now + dt
            subdivNotations = [n for n in notations
                               if now <= n.offset < subdivEnd and n.end <= subdivEnd]
            _applyDurationRatio(subdivNotations, subdiv, now, dt)
            now += dt


def quantizeMeasure(events: list[Notation],
                    timesig: timesig_t,
                    quarterTempo: F,
                    profile: QuantizationProfile,
                    subdivisionStructure: list[int] = None
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
        subdivisionStructure: how this measure is subdivided. If not given
            it will be inferred from the time signature and tempo.

    Returns:
        a QuantizedMeasure

    """
    measureDur = util.measureQuarterDuration(timesig)
    assert all(0 <= ev.offset <= ev.end <= measureDur
               for ev in events), f"{events=}, {measureDur=}"
    for ev0, ev1 in iterlib.pairwise(events):
        if ev0.end != ev1.offset:
            logger.error(f"{ev0} (end={ev0.end}), {ev1} (offset={ev1.offset})")
            raise AssertionError("events are not stacked")

    quantizedBeats: list[QuantizedBeat] = []

    if not _isMeasureFilled(events, timesig):
        logger.debug(f"Measure {timesig} is not filled ({events=}). "
                     f"Filling gaps with silences")
        events = _fillMeasure(events, timesig)

    beatStructure = st.measureBeatStructure(timesig=timesig,
                                            quarterTempo=quarterTempo,
                                            subdivisionStructure=subdivisionStructure)

    beatOffsets = [beat.offset for beat in beatStructure]
    beatOffsets.append(beatStructure[-1].end)

    idx = 0
    for span, eventsInBeat in splitNotationsAtOffsets(events, offsets=beatOffsets):
        beatWeight = beatStructure[idx].weight
        beatdur = span.end - span.start
        if beatdur.numerator in (1, 2, 4):
            quantizedBeat = quantizeBeatBinary(eventsInBeat=eventsInBeat,
                                               quarterTempo=quarterTempo,
                                               beatDuration=span.end - span.start,
                                               beatOffset=span.start,
                                               profile=profile)
            quantizedBeat.weight = beatWeight
            quantizedBeats.append(quantizedBeat)
        elif beatdur.numerator == 3:
            subBeats = quantizeBeatTernary(eventsInBeat=eventsInBeat,
                                           quarterTempo=quarterTempo,
                                           beatDuration=beatdur,
                                           beatOffset=span.start,
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
        struct: the ScoreStructure
        n: the Notation to split

    Returns:
        a list of tuples (measure number, notation), indicating
        to which measure each part belongs to. The notation in the
        tuple has an offset relative to the beginning of the measure

    """
    assert n.offset >= 0 and n.duration >= 0
    measureindex0, beat0 = struct.beatToLocation(n.offset)
    measureindex1, beat1 = struct.beatToLocation(n.end)

    if measureindex0 is None or measureindex1 is None:
        raise ValueError(f"Could not find a score location for this event: {n}")

    if beat1 == 0 and n.duration > 0:
        # Note ends at the barline
        measureindex1 -= 1
        beat1 = struct.getMeasureDef(measureindex1).durationQuarters

    numMeasures = measureindex1 - measureindex0 + 1

    if numMeasures == 1:
        # The note fits within one measure. Make the offset relative to the measure
        event = n.clone(offset=beat0, duration=beat1 - beat0)
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
    if beat1 > 0:
        notation = n.cloneAsTie(offset=F0, duration=beat1, tiedPrev=True, tiedNext=n.tiedNext)
        pairs.append((measureindex1, notation))

    for idx, part in pairs[:-1]:
        assert part.isRest or part.tiedNext, f"{n=}, {pairs=}"
    for idx, part in pairs[1:]:
        assert part.isRest or part.tiedPrev, f"{n=}, {pairs=}"
    # tieNotations(parts)

    sumdur = sum(struct.beatDelta((i, n.offset), (i, n.end)) for i, n in pairs)
    assert sumdur == n.duration, f"{n=}, {sumdur=}, {numMeasures=}\n{pairs=}"
    return pairs


def _mergeNodes(node1: Node,
                node2: Node,
                profile: QuantizationProfile,
                beatOffsets: list[F]
                ) -> Node:
    # we don't check here, just merge
    node = Node(ratio=node1.durRatio, items=node1.items + node2.items)
    node = node.mergedNotations()
    return _mergeSiblings(node, profile=profile, beatOffsets=beatOffsets)


def _nodesCanMerge(g1: Node,
                   g2: Node,
                   profile: QuantizationProfile,
                   beatOffsets: list[F]
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
    assert len(g1.items) > 0 and len(g2.items) > 0
    assert g1.offset < g1.end <= g2.offset
    assert g1.end == g2.offset
    # acrossBeat = next((offset for offset in beatOffsets if g1.end == offset), None)
    acrossBeat = any(g1.end == offset
                     for offset in beatOffsets)

    if g1.durRatio != g2.durRatio:
        return Result.Fail("not same durRatio")

    g1last = g1.lastNotation()
    g2first = g2.firstNotation()

    if g1.durRatio == (1, 1) and len(g1) == len(g2) == 1:
        if g1last.gliss and g1last.tiedPrev and g1.symbolicDuration() + g2.symbolicDuration() > 1:
            return Result.Fail('A glissando over a beat needs to be broken at the beat')
        # Special case: always merge binary nodes with single items since there is always
        # a way to notate those
        return Result.Ok()

    mergedSymbolicDur = g1last.symbolicDuration() + g2first.symbolicDuration()

    if g1.durRatio == (3, 2) and mergedSymbolicDur == F(3, 2):
        return Result.Fail("Don't merge 3/2 when the merged Notation results in dotted quarter")

    if g1.durRatio != (1, 1):

        if acrossBeat and g1.durRatio[0] not in profile.allowedTupletsAcrossBeat:
            return Result.Fail("tuplet not allowed to merge across beat")
        elif g1.totalDuration() + g2.totalDuration() > profile.mergedTupletsMaxDuration:
            return Result.Fail("incompatible duration")
        elif not profile.mergeTupletsOfDifferentDuration and acrossBeat and g1.totalDuration() != g2.totalDuration():
            return Result.Fail("Nodes of different duration cannot merge")

    item1, item2 = g1.items[-1], g2.items[0]
    syncopated = g1last.tiedNext or (g1last.isRest and g2first.isRest and g1last.durRatios == g2first.durRatios)

    if acrossBeat and not syncopated:
        return Result.Fail('no need to extend node over beat')

    if isinstance(item1, Node) and isinstance(item2, Node):
        if not (r := _nodesCanMerge(item1, item2, profile=profile, beatOffsets=beatOffsets)):
            return Result.Fail(f'nested tuplets cannot merge: {r.info}')
        else:
            nestedtup = (g1.durRatio[0], item1.durRatio[0])
            if acrossBeat and item1.durRatio != (1, 1) and g1.durRatio != (1, 1) and nestedtup not in profile.allowedNestedTupletsAcrossBeat:
                return Result.Fail(f'complex nested tuplets cannot merge: {nestedtup}')
            return Result.Ok()
    elif isinstance(item1, Node) or isinstance(item2, Node):
        return Result.Fail('A Node cannot merge with a single item')
    else:
        assert isinstance(item1, Notation) and isinstance(item2, Notation)
        if not acrossBeat:
            return Result.Ok()

        symdur: F = item1.symbolicDuration() + item2.symbolicDuration()

        if syncopated and symdur.denominator not in (1, 2, 4, 8):
            return Result.Fail(f'Cannot merge notations resulting in irregular durations. Resulting symbolic duration: {symdur}')

        if item1.gliss and item1.tiedNext and item2.gliss:
            if symdur >= 2 and item1.tiedPrev:
                return Result.Fail("Cannot merge glissandi resulting in long (>= halfnote) notes")

        if not profile.mergeNestedTupletsAcrossBeats:
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

        if item1.duration + item2.duration < profile.minBeatFractionAcrossBeats:
            return Result.Fail('Absolute duration of merged Notations across beat too short')

        if symdur < profile.minSymbolicDurationAcrossBeat:
            return Result.Fail('Symbolic duration of merged notations across beat too short')

        if g1.durRatio == (3, 2) and item1.symbolicDuration() == item2.symbolicDuration() == 1 and item1.tiedNext:
            return Result.Fail('Not needed')

        return Result.Ok()


def _mergeSiblings(root: Node,
                   profile: QuantizationProfile,
                   beatOffsets: list[F],
                   ) -> Node:
    """
    Merge sibling tree of the same kind, if possible

    Args:
        root: the root of a tree of Nodes
        profile: the quantization profile
        beatOffsets: these offsets are used to determine if a merged node
            would cross a beat boundary. The quantization profile has some
            rules regarding merging tuplets across beat boundaries which need
            this information

    Returns:
        a new tree
    """
    # merge only tree (not Notations) across tree of same level
    if len(root.items) <= 1:
        return root
    items = []
    item1 = root.items[0]
    if isinstance(item1, Node):
        item1 = _mergeSiblings(item1, profile=profile, beatOffsets=beatOffsets)
    items.append(item1)
    for item2 in root.items[1:]:
        item1 = items[-1]
        if isinstance(item2, Node):
            item2 = _mergeSiblings(item2, profile=profile, beatOffsets=beatOffsets)
        if isinstance(item1, Node) and isinstance(item2, Node):
            # check if the tree should merge
            if r := _nodesCanMerge(item1, item2, profile=profile, beatOffsets=beatOffsets):
                logger.debug(f"Nodes can merge: \n    {item1}\n    {item2}")
                mergednode = _mergeNodes(item1, item2, profile=profile, beatOffsets=beatOffsets)
                items[-1] = mergednode
                logger.debug(f"---- Merged node:\n    {mergednode}")
            else:
                logger.debug(f'Nodes cannot merge: \n{item1}\n{item2}\n----> {r.info}')
                items.append(item2)
        elif isinstance(item1, Notation) and isinstance(item2, Notation) and item1.canMergeWith(item2):
            items[-1] = item1.mergeWith(item2)
        else:
            items.append(item2)
    outnode = Node(ratio=root.durRatio, items=items)
    assert root.totalDuration() == outnode.totalDuration()
    return outnode


def _maxTupletLength(timesig: timesig_t, subdivision: int):
    den = timesig[1]
    if subdivision == 3:
        return {2: 2, 4: 2, 8: 1}[den]
    elif subdivision == 5:
        return 2 if den == 2 else 1
    else:
        return 1


@dataclass
class PartLocation:
    """
    Represents a location (a Notation inside a beat, inside a measure, inside a part)
    """
    measureIndex: int
    """The measure index"""

    beatIndex: int
    """The beat index"""

    notationIndex: int
    """The notation index within the beat"""

    beat: QuantizedBeat
    """A reference to the beat"""

    notation: Notation
    """The notation at this location"""


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

    quantprofile: QuantizationProfile
    """QuantizationProfile used for quantization"""

    name: str = ''
    """The name of this part, used as staff name"""

    shortname: str = ''
    """The abbreviated staff name"""

    groupid: str = ''
    """A groupid, if applicable"""

    groupname: tuple[str, str] | None = None

    firstclef: str = ''
    """The first clef of this part"""

    autoClefChanges: bool | None = None
    """If True, add clef changes when rendering this Part; None=use default.
    This corresponds to RenderOptions.autoClefChanges. Any part with manual
    clef changes will not be modified. To modify such a part see
    :meth:`QuantizedPart.addClefChanges`"""

    showName: bool = True
    """If True, show part name when rendered"""

    def __post_init__(self):
        for measure in self.measures:
            measure.parent = self
            measure.quantprofile = self.quantprofile

        self.repair()

    def repair(self):
        self._repairGracenotes()
        self._repairLinks()
        self.repairSpanners(tree=False)

    def render(self, options: renderoptions.RenderOptions = None, backend=''
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

    def resetTrees(self):
        """Remove any cached tree representation of the measures in this part"""
        for measure in self.measures:
            measure.resetTree()

    def flatNotations(self, tree: bool) -> Iterator[Notation]:
        """Iterate over all notations in this part"""
        for measure in self.measures:
            yield from measure.notations(tree=tree)

    def averagePitch(self, maxNotations=0) -> float:
        """
        The average pitch of this part

        Args:
            maxNotations: if given, only the first *maxNotations* are considered
                for calculating the average pitch.

        Returns:
            the average pitch of the notations in this part
        """
        accum = 0
        num = 0
        for m in self.measures:
            if m.isEmpty():
                continue
            for b in m.beats:
                for n in b.notations:
                    if n.isRest:
                        continue
                    accum += n.meanPitch()
                    num += 1
                    if maxNotations and num > maxNotations:
                        break
        return accum/num if num > 0 else 0

    def findLogicalTie(self, n: Notation) -> list[PartLocation] | None:
        """
        Given a Notation which is part of a logical tie (it is tied or tied to), return the logical tie

        Args:
            n: a Notation which is part of a logical tie

        Returns:
            a list of PartLocation representing the logical tie the notation *n* belongs to

        """
        for tie in self.logicalTies():
            if any(loc.notation is n for loc in tie):
                return tie
        return None

    def logicalTies(self) -> Iterator[list[PartLocation]]:
        """
        Returns a list of all logical ties in this part

        A logical tie is a list of notations which are linked together

        """
        last: list[PartLocation] = []
        for loc in self.iterNotations():
            if loc.notation.tiedPrev:
                last.append(loc)
            else:
                if last:
                    yield last
                last = [loc]
        if last:
            yield last

    def mergedTies(self) -> list[Notation]:
        """
        Returns a list of all merged ties in this part
        """
        def mergeTie(notations: list[Notation]) -> Notation:
            n0 = notations[0].copy()
            n0.durRatios = None
            for n in notations[1:]:
                n0.duration += n.duration
            return n0

        out: list[Notation] = []
        for tie in self.logicalTies():
            if len(tie) > 1:
                notations = [loc.notation for loc in tie]
                out.append(mergeTie(notations))
            else:
                out.append(tie[0].notation)
        return out

    def iterNotations(self) -> Iterator[PartLocation]:
        """
        Iterates over all notations giving the location of each notation

        For each notation yields a PartLocation with attributes:

            measureNum: int
            beatNum: int
            beat: QuantizedBeat
            notation: Notation

        """
        for measureidx, m in enumerate(self.measures):
            if m.isEmpty():
                continue
            assert m.beats is not None
            for beatidx, b in enumerate(m.beats):
                for notationidx, n in enumerate(b.notations):
                    yield PartLocation(measureidx, beatIndex=beatidx, beat=b, notation=n, notationIndex=notationidx)

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
        notations = self.flatNotations(tree=True)
        return clefutils.bestclef(list(notations))


    def findClefChanges(self, addClefs=False, removeManualClefs=False, window=1,
                        threshold=0., biasFactor=1.5, property='clef'
                        ) -> None:
        """
        Determines the most appropriate clef changes for this part

        The clef changes are added as properties to the notations at which
        the changes are to be made. If called with ``addClefs==True``,
        these clef changes are materialized as clef attachments

        Args:
            addClefs: if True, clef change directives are actually added to the
                quantized notations. Otherwise, only hints given as properties are added
            removeManualClefs: if True, remove any manual clef
            window: the window size when determining the best clef for a given section
            threshold: a simplification threshold. A value of 0. disables simplification
            biasFactor: The higher this value, the more weight is given to the
                previous clef, thus making it more difficult to change clef
                for minor jumps
            property: the property key to add to the notation to mark
             a clef change. Setting this property alone will not
             result in a clef change in the notation (see `addClefs`)

        """
        notations = list(self.flatNotations(tree=True))
        if removeManualClefs:
            # from . import attachment
            for n in notations:
                n.removeAttachments(lambda attach: isinstance(attach, attachment.Clef))
        clefutils.findBestClefs(notations, addclefs=addClefs, winsize=window,
                                threshold=threshold, biasfactor=biasFactor,
                                key=property)

    def removeUnnecessaryDynamics(self, tree: bool, resetAfterEmptyMeasure=True) -> None:
        """
        Remove superfluous dynamics in this part, inplace
        """
        dynamic = ''
        for meas in self.measures:
            if meas.isEmpty() and resetAfterEmptyMeasure:
                dynamic = ''
                continue
            for n in meas.notations(tree=tree):
                if n.isRest:
                    continue
                if not n.tiedPrev and n.dynamic and n.dynamic in definitions.dynamicLevels:
                    # Only dynamic levels are ever superfluous (f, ff, mp), other 'dynamics'
                    # like sf should not be removed
                    if n.dynamic == dynamic:
                        n.dynamic = ''
                    else:
                        dynamic = n.dynamic

    def removeUnnecessaryGracenotes(self, tree: bool) -> None:
        """
        Removes unnecessary gracenotes, in place

        An unnecessary gracenote are:

        * has the same pitch as the next real note and starts a glissando. Such gracenotes might
          be created during quantization.
        * has the same pitch as the previous real note and ends a glissando
        * n0/real -- gliss -- n1/grace n2/real and n1.pitches == n2.pitches

        Args:
            tree: if True, remove unnecessary gracenotes from the tree representation
                of the measures in this part. Otherwise, modify the quantized beats
        """
        if tree:
            for measure in self.measures:
                root = measure.tree()
                root.removeUnnecessaryGracenotes()
            return

        trash: list[PartLocation] = []
        maxiter = 10
        for _ in range(maxiter):
            skip = False
            for loc0, loc1 in iterlib.pairwise(self.iterNotations()):
                if skip:
                    skip = False
                    continue

                n0: Notation = loc0.notation
                n1: Notation = loc1.notation
                if n0.isGracenote:
                    if n0.pitches == n1.pitches and (n0.tiedNext or n0.gliss):
                        n0.copyAttributesTo(n1)
                        trash.append(loc0)
                        n0.removeSpanners()
                elif n1.isGracenote:
                    if n0.pitches == n1.pitches and (n0.tiedNext or n0.gliss):
                        n0.gliss = n1.gliss
                        n0.tiedNext = n1.tiedNext
                        trash.append(loc1)
                        skip = True
                        n1.removeSpanners()

            numRemoved = len(trash)
            for loc in trash:
                loc.beat.notations.remove(loc.notation)
            trash.clear()
            skip = False
            for l0, l1, l2 in iterlib.window(self.iterNotations(), 3):
                if skip:
                    skip = False
                    continue
                n0, n1, n2 = l0.notation, l1.notation, l2.notation
                if not n0.isGracenote and n1.isGracenote and not n2.isGracenote and n0.gliss and n1.pitches == n2.pitches:
                    trash.append(l1)
                    skip = True
            for loc in trash:
                loc.beat.notations.remove(loc.notation)
            numRemoved += len(trash)
            if numRemoved == 0:
                break
        else:
            logger.warning(f"Reached max. number of iterations ({maxiter}) while "
                           "attempting to remove superfluous gracenotes")

    def repairSpanners(self, tree: bool, removeUnnecessary=True) -> None:
        """
        Match orfan spanners, optionally removing unmatched spanners (in place)

        Args:
            tree: if True, use the tree representation for each measure
            removeUnnecessary: if True, remove any spanner with no matching
                start/end spanner
        """
        # _spanner.removeUnmatchedSpanners(self.flatNotations(tree=tree))
        notations = list(self.flatNotations(tree=tree))
        _spanner.solveHairpins(notations)
        _spanner.matchOrfanSpanners(notations=notations,
                                    removeUnmatched=removeUnnecessary)
        _spanner.markNestingLevels(notations)
        _spanner.moveEndSpannersToEndOfLogicalTie(notations)
        if not tree:
            self.resetTrees()

    def _repairGracenotes(self):
        """
        Repair some corner cases where gracenotes cause rendering problems

        This should be called before creating node trees in each measure,  since
        this works at the beat level.
        """
        for measureidx, measure in enumerate(self.measures):
            if not measure.beats:
                continue
            for beatidx, beat in enumerate(measure.beats):
                last = beat.notations[-1]
                if last.isGracenote and last.offset == beat.beatOffset + beat.beatDuration:
                    if beatidx == len(measure.beats) - 1:
                        nextmeasure = self.getMeasure(measureidx + 1)
                        if nextmeasure.isEmpty():
                            # TODO
                            pass
                        else:
                            # move the gracenote to the next measure
                            beat.notations.pop()
                            nextmeasure.beats[0].notations.insert(0, last.clone(offset=F0))
                    else:
                        # move gracenote to bext beat
                        nextbeat = measure.beats[beatidx + 1]
                        beat.notations.pop()
                        nextbeat.notations.insert(0, last)

    def getMeasure(self, idx: int, extend=True) -> QuantizedMeasure:
        """
        Get a measure within this part

        Args:
            idx: the measure index (starts at 0)
            extend: if True and the index is outside the defined measures,
                a new QuantizedMeasure (empty) will be created and added
                to this part

        Returns:
            The corresponding measure. If outside the defined measures a new
            empty QuantizedMeasure will be created
        """
        if idx > len(self.measures) - 1 and extend:
            for i in range(len(self.measures) - 1, idx+1):
                mdef = self.struct.getMeasureDef(i)
                qmeasure = QuantizedMeasure(timesig=mdef.timesig,
                                            quarterTempo=mdef.quarterTempo,
                                            quantprofile=self.quantprofile,
                                            parent=self)
                self.measures.append(qmeasure)
        return self.measures[idx]

    def _repairLinks(self) -> None:
        """
        Repairs ties and glissandi (in place)
        """
        for n0, n1 in iterlib.pairwise(self.flatNotations(tree=False)):
            if n0.tiedNext:
                if n0.isRest or not n0.pitches or n0.pitches != n1.pitches:
                    n0.tiedNext = False
                    n1.tiedPrev = False
                elif not n0.pitches or not n1.pitches or any(x not in n1.pitches for x in n0.pitches):
                    n0.tiedNext = False
                    n1.tiedPrev = False
                #elif n1.isGracenote:
                #    n0.tiedNext = False
                #    n1.tiedPrev = False
            elif n0.gliss:
                if n1.isRest or n0.pitches == n1.pitches:
                    if n0.tiedPrev:
                        logicalTie = self.findLogicalTie(n0)
                        if logicalTie:
                            for n in logicalTie:
                                n.notation.gliss = False
                    else:
                        n0.gliss = False
                    n0.tiedNext = True
                    n1.tiedPrev = True

    def pad(self, numMeasures: int) -> None:
        """Add the given number of empty measures at the end"""
        if numMeasures <= 0:
            return
        l = len(self.measures)
        for measureIndex in range(l - 1, l - 1 + numMeasures):
            measuredef = self.struct.getMeasureDef(measureIndex)
            empty = QuantizedMeasure(timesig=measuredef.timesig,
                                     quarterTempo=measuredef.quarterTempo,
                                     parent=self)
            self.measures.append(empty)

    def fixChordSpellings(self, tree=True, enharmonicOptions: enharmonics.EnharmonicOptions = None
                          ) -> None:
        """
        Finds the best spelling for each chord individually

        As an alternative for finding the best global spelling it is possible to
        just fix each chord individually

        """
        for measure in self.measures:
            for n in measure.notations(tree=tree):
                if n.isRest or len(n) <= 1:
                    continue
                notenames = n.resolveNotenames(addFixedAnnotation=True)
                spellings = enharmonics.bestChordSpelling(notenames, options=enharmonicOptions)
                for i, spelling in enumerate(spellings):
                    n.fixNotename(spelling, i)

    def eventAt(self, beat: F | tuple[int, F]) -> tuple[Notation, QuantizedMeasure]:
        measure, relbeat = self.measureAt(beat)
        tree = measure.tree()
        for n in tree.recurse():
            assert n.offset is not None
            if n.offset <= relbeat <= n.end:
                return n, measure

    def measureAt(self, beat: F | tuple[int, F]) -> tuple[QuantizedMeasure, F]:
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
        if measureidx >= len(self.measures):
            raise IndexError(f"Location {beat} is outside of this part's boundaries")
        measure = self.measures[measureidx]
        if relbeat > measure.duration():
            raise ValueError(f"The relative beat {relbeat} exceeds the duration of the measure ({measure})")
        return measure, relbeat

    def breakBeam(self, location: F | tuple[int, F]) -> Notation | None:
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
        tree = measure.tree()
        return tree.breakBeamsAt(relbeat)

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
                           ) -> Notation | None:
        """
        Break a syncopation/beam at the given beat/location

        This method works **in place** at the tree level

        Args:
            location: an absolute offset in quarter notes, or a location as
                tuple (measure index, relative offset)
        """
        measure, relbeat = self.measureAt(location)
        root = measure.tree()
        notation = root._splitNotationAtBoundary(relbeat)
        if notation:
            notation.mergeable = False
        return notation


def quantizePart(part: core.UnquantizedPart,
                 struct: st.ScoreStruct,
                 quantprofile: str | QuantizationProfile,
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
    if isinstance(quantprofile, str):
        quantprofile = QuantizationProfile.fromPreset(quantprofile)
    part.fillGaps()
    notations = part.notations
    core.resolveOffsets(notations)
    quantutils.transferAttributesWithinTies(notations)
    allpairs = [splitNotationAtMeasures(n=n, struct=struct) for n in notations]
    maxMeasure = max(pairs[-1][0] for pairs in allpairs)
    notationsPerMeasure: list[list[Notation]] = [[] for _ in range(maxMeasure+1)]
    for pairs in allpairs:
        for measureIdx, notation in pairs:
            notationsPerMeasure[measureIdx].append(notation)
    qmeasures = []
    for idx, notations in enumerate(notationsPerMeasure):
        measureDef = struct.getMeasureDef(idx)
        if not notations:
            qmeasures.append(QuantizedMeasure(timesig=measureDef.timesig,
                                              quarterTempo=measureDef.quarterTempo,
                                              quantprofile=quantprofile))
        else:
            if not misc.issorted(notations, key=lambda n: n.offset):
                raise ValueError(f"Notations are not sorted: {notations}")
            core.removeSmallOverlaps(notations)
            qmeasure = quantizeMeasure(notations,
                                       timesig=measureDef.timesig,
                                       quarterTempo=measureDef.quarterTempo,
                                       profile=quantprofile,
                                       subdivisionStructure=measureDef.subdivisionStructure)
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
    qpart = QuantizedPart(struct, qmeasures, name=part.name, shortname=part.shortname,
                          groupid=part.groupid, quantprofile=quantprofile,
                          groupname=part.groupname,
                          showName=part.showName)
    if quantprofile.breakSyncopationsLevel != 'none':
        for measure in qpart:
            measure.breakSyncopations(level=quantprofile.breakSyncopationsLevel)

    for hook in part.hooks:
        if isinstance(hook, attachment.PostPartQuantHook):
            hook(qpart)
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
        self.parts: list[QuantizedPart] = parts
        """A list of QuantizedParts"""

        self.title: str = title
        """Title of the score, used for rendering purposes"""

        self.composer: str = composer
        """Composer of the score, used for rendering"""

        #if parts:
        #    for part in parts:
        #        part.repair()

    def check(self):
        """Check this QuantizedScore"""

        for pidx, part in enumerate(self.parts):
            for midx, measure in enumerate(part.measures):
                try:
                    measure.check()
                except ValueError as e:
                    logger.error(f"Check error in part {pidx}, measure {midx}")
                    raise e

    def fixEnharmonics(self, enharmonicOptions: enharmonics.EnharmonicOptions) -> None:
        """
        Finds the best spelling for each part in this score, inplace

        Args:
            enharmonicOptions: the enharmonic options to use
        """
        for part in self.parts:
            for measure in part.measures:
                measure.fixEnharmonics(enharmonicOptions)

    def fixChordSpellings(self, tree=True, enharmonicOptions: enharmonics.EnharmonicOptions = None
                          ) -> None:
        """
        Finds the best spelling for each chord individually

        As an alternative for finding the best global spelling it is possible to
        just fix each chord individually

        """
        for part in self.parts:
            part.fixChordSpellings(tree=tree, enharmonicOptions=enharmonicOptions)

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
            print(f"{indent*numindents}UnquantizedPart #{i}:", file=stream)
            part.dump(tree=tree, numindents=numindents+1, indent=indent, stream=stream)

    @property
    def scorestruct(self) -> st.ScoreStruct:
        """Returns the ScoreStruct of this score"""
        if not self.parts:
            raise IndexError("This QuantizedScore has no parts")
        return self.parts[0].struct

    @scorestruct.setter
    def scorestruct(self, struct: st.ScoreStruct) -> None:
        if self.parts:
            for part in self.parts:
                part.struct = struct

    def resetTrees(self):
        """Remove any cached tree representation of the measures in this score"""
        for part in self.parts:
            part.resetTrees()

    def removeUnnecessaryDynamics(self, tree: bool):
        """Removes any unnecessary dynamics in this score

        Args:
            tree: if True, apply the transformation to the tree representation
        """
        for part in self:
            part.removeUnnecessaryDynamics(tree=tree)
        if not tree:
            self.resetTrees()

    def numMeasures(self) -> int:
        """Returns the number of measures in this score"""
        return max(len(part.measures)
                   for part in self.parts)

    def padEmptyMeasures(self) -> None:
        """Adds empty measures at the end of each part so that all have the same length"""
        numMeasures = self.numMeasures()
        for part in self.parts:
            part.pad(numMeasures - len(part.measures))

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
              backend=''
              ) -> renderer.Renderer:
        """
        Export this score as pdf, png, lilypond, MIDI or musicxml

        When rendering to pdf or png both the lilypond or the
        musicxml backend can be used.

        Args:
            outfile: the path of the written file
            options: render options used to generate the output
            backend: backend used when writing to png / pdf (one of 'lilypond', 'musicxml')

        Returns:
            the Renderer used
        """
        ext = os.path.splitext(outfile)[1].lower()
        if ext == '.ly':
            r = self.render(options=options, backend='lilypond')
            r.write(outfile)
            return r
        elif ext == '.xml' or ext == '.musicxml':
            r = self.render(options=options, backend='musicxml')
            r.write(outfile)
            return r
        elif ext == '.pdf' or ext == '.png':
            r = self.render(options=options, backend=backend)
            r.write(outfile)
            return r
        else:
            raise ValueError(f"Format {ext} not supported")

    def show(self, backend='', fmt='png', external: bool = None):
        renderer = self.render(backend=backend)
        renderer.show(fmt=fmt, external=external)

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
            cfg = workspace.Workspace.active.config
            options = cfg.makeRenderOptions()
            if backend:
                options.backend = backend
        elif backend and backend != options.backend:
            options = options.clone(backend=backend)
        return render.renderQuantizedScore(self, options=options)

    def toCoreScore(self) -> maelzel.core.Score:
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
                tree = measure.tree()
                notations = list(tree.recurse())
                events.extend(notationsToCoreEvents(notations))
            voice = maelzel.core.Voice(events)
            voices.append(voice)
        return maelzel.core.Score(voices=voices, scorestruct=self.scorestruct, title=self.title)


def quantize(parts: list[core.UnquantizedPart],
             struct: st.ScoreStruct = None,
             quantizationProfile: QuantizationProfile = None,
             enharmonicOptions: enharmonics.EnharmonicOptions = None
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
    if quantizationProfile is None:
        quantizationProfile = QuantizationProfile()
    if struct is None:
        struct = st.ScoreStruct(timesig=(4, 4), tempo=60)
    qparts = []
    for part in parts:
        profile = part.quantProfile or quantizationProfile
        qpart = quantizePart(part, struct=struct, quantprofile=profile)
        qparts.append(qpart)
    qscore = QuantizedScore(qparts)
    if enharmonicOptions:
        qscore.fixEnharmonics(enharmonicOptions)
    else:
        qscore.fixChordSpellings(tree=True, enharmonicOptions=enharmonicOptions)

    qscore.check()
    return qscore
