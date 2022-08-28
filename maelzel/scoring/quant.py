"""
Quantize durations to musical notation

The most important function here is :func:`quantize`, which treturns
a :class:`QuantizedScore`

Example
-------

.. code-block:: python

    XXX
"""
from __future__ import annotations
from dataclasses import dataclass, field as _field

from .common import *
from . import core
from . import definitions
from . import util
from . import quantdata
from . import enharmonics

from .core import Notation, makeRest
from .durationgroup import DurationGroup, durratio_t
from maelzel.scorestruct import ScoreStruct

from emlib import iterlib
from emlib import misc
from emlib import mathlib
from pitchtools import notated_pitch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numbers import Rational
    from typing import Union, Optional, Iterator
    number_t = Union[int, float, Rational]


__all__ = (
    'quantize',
    'QuantizationProfile',
    'makeQuantizationProfile',
    'QuantizedScore',
    'QuantizedPart',
    'QuantizedMeasure',
    'QuantizedBeat',
    'quantizeBeat',
    'quantizeMeasure',
    'quantizePart',
    'splitNotationAtOffsets',
    'splitNotationByMeasure',
    'PartLocation',
)
    
def _factory(obj) -> _field:
    return _field(default_factory=lambda:obj)


def _presetField(key) -> _field:
    return _factory(quantdata.complexityPresets[key])


class QuantError(Exception):
    pass


@dataclass
class QuantizationProfile:
    """
    Most important parameters:

    - nestedTuples: if True, allow nested tuples. NB: musicxml rendered
      via MuseScore does not support nested tuples
    - gridErrorWeight: a weight to control the overall effect of offset
      and duration errors when fitting events to a grid. A higher
      weight will cause quantization to minimize offset and duration
      errors, at the cost of choosing more complex divisions
    - divisionErrorWeight: also a weight to controll all effect
      dealing with the complexity of a given division/subdivision

    Lower level parameters to calculate grid error:

    - offsetErrorWeight: the importance of offset errors to calculate
      the best subdivision of a beat
    - restOffsetErrorWeight: how relevant should be the offset error in
      the case of rests
    - durationErrorWeight: relevance of duration error when selecting the
      best subdivision
    - graceNoteDuration: if a note is considered a grace note (which have
      no duration per se), should we still account for this duration?
    - minBeatFractionAcrossBeats: when merging durations across beats, a merged
      duration can't be smaller than this duration. This is to prevent joining
      durations across beats which might result in high rhythmic complexity
    - tupletsAllowedAcrossBeats: list of tuplets allowed across a beat
    - mergedTupletsMaxDur: the max quarternote duration for a merged tuplet

    Lower level parameters to calculate division complexity:

    - levelPenaltyWeight: how

    """
    nestedTuples: bool = False
    gridErrorWeight: float = 0.2
    divisionErrorWeight: float = 0.08

    offsetErrorWeight: float = 1.0
    restOffsetErrorWeight: float = 0.25
    durationErrorWeight: float = 0.2
    graceNoteDuration: F = F(1, 32)

    possibleDivisionsByTempo: dict[int, list] = _factory(quantdata.complexityPresets['middle']['divisionsByTempo'])
    divisionPenaltyMap: dict[int, float] = _factory(quantdata.complexityPresets['middle']['divisionPenaltyMap'])
    divisionCardinalityPenaltyMap: dict[int, float] = _factory({1:0.0, 2:0.2, 3:0.4})
    levelPenalty: list[float] = _factory([0., 0.05, 0.4, 0.5, 0.8, 0.8])
    numSubdivsPenaltyMap: dict[int, float] = _factory({1:0.0, 2:0.0, 3:0.0})

    divisionPenaltyWeight: float = 1.0
    cardinalityPenaltyWeight: float = 0.1
    levelPenaltyWeight: float = 0.1
    numSubdivisionsPenaltyWeight: float = 0.2
    minBeatFractionAcrossBeats: F = F(1)
    mergedTupletsMaxDuration: F = F(1)
    allowedTupletsAcrossBeats: list[float] = _factory([1, 3])

defaultQuantizationProfile = QuantizationProfile()


def makeQuantizationProfile(preset='default') -> QuantizationProfile:
    """
    Create a QuantizationProfile from a preset

    Args:
        preset: possible presets: 'defaut', 'music21', 'simple', 'medium', 'complex'

    Returns:
        the quantization profile

    .. note::

        Default equals to medium. The music21 profile is the same as the default profile
        but disables nested tuples, since music21 does not support them
    """
    if preset == 'default':
        return defaultQuantizationProfile
    if preset == 'music21':
        return QuantizationProfile(nestedTuples=False)
    if preset in {'simple', 'medium', 'complex'}:
        complexitypreset = quantdata.complexityPresets[preset]
        return QuantizationProfile(nestedTuples=False,
                                   possibleDivisionsByTempo=complexitypreset['divisionsByTempo'],
                                   divisionPenaltyMap=complexitypreset['divisionPenaltyMap'])
    raise ValueError(f"preset {preset} unknown. Possible values: default, music21,"
                     f" simple, medium, complex")


def _divisionPenalty(div: Union[int, list],
                     profile: QuantizationProfile = defaultQuantizationProfile,
                     nestingLevel=1,
                     ) -> float:
    """
    Evaluate the given division.

    The lower the returned value, the simpler this division is. All things
    being equal, a simpler division should be preferred.

    Args:
        div: division of the beat/subbeat
        nestingLevel: since this is a recursive structure, the nestingLevel
            holds the level of nesting of the division we are analyzing
        profile: the quantization profile to use

    Returns:
        the penalty associated with this division, based on the division
        only (not on how the division fits the notes in the beat).

    """
    if isinstance(div, int):
        return profile.divisionPenaltyMap.get(div, 0.7)
    numSubdivsPenalty = profile.numSubdivsPenaltyMap.get(len(div), 0.7)
    cardinality = len(set(div))
    cardinalityPenalty = profile.divisionCardinalityPenaltyMap.get(cardinality, 0.7)
    divPenalty = sum(_divisionPenalty(subdiv, profile, nestingLevel+1)
                     for subdiv in div) / len(div)
    levelPenalty = profile.levelPenalty[nestingLevel]
    penalty = mathlib.weighted_euclidian_distance([
        (divPenalty, profile.divisionPenaltyWeight),
        (cardinalityPenalty, profile.cardinalityPenaltyWeight),
        (levelPenalty, profile.levelPenaltyWeight),
        (numSubdivsPenalty, profile.numSubdivisionsPenaltyWeight)
    ])
    return min(penalty, 1)


def beatDivisions(tempo: number_t,
                  profile: QuantizationProfile
                  ) -> list[division_t]:
    """
    Given a tempo, return the possible subdivisions of the duration reference to which
    this tempo applies according to the given profile

    **Example 1**: possible subdivisions of a quarter note at tempo 60::

        >>> beatDivisions(60)

    To determine the possible subdivisions of an eigth note, just double the tempo
    """
    assert isinstance(profile, QuantizationProfile)
    quantProfile = profile
    divsByTempo = quantProfile.possibleDivisionsByTempo
    divs = None
    for maxTempo, possibleDivs in divsByTempo.items():
        if tempo < maxTempo:
            divs = possibleDivs
            break
    if not divs:
        raise ValueError("No divisions for the given tempo")
    out = []
    for div in divs:
        if isinstance(div, int):
            out.append(div)
        elif isinstance(div, (list, tuple)) and quantProfile.nestedTuples:
            for perm in iterlib.permutations(div):
                out.append(list(perm))
    return out


def _gridDurations(beatDuration: F, division: division_t) -> list[F]:
    """
    Called to recursively generate a grid corresponding to the given division
    of the beat
    """
    if isinstance(division, int):
        dt = beatDuration/division
        grid = [dt] * division
    elif isinstance(division, (list, tuple)):
        if len(division) == 1:
            grid = _gridDurations(beatDuration, division[0])
        else:
            numDivisions = len(division)
            subdivDur = beatDuration / numDivisions
            grid = [_gridDurations(subdivDur, subdiv) for subdiv in division]
    else:
        raise TypeError(f"Expected an int or a list, got {division} ({type(division)})")
    return grid


def generateBeatGrid(beatDuration: number_t, division: division_t, offset=F(0)
                     ) -> list[F]:
    """
    Generates a grid with the beats for the given duration and division

    The last value of the grid is the offset of the next beat.
    *At the moment* irrational tuples are not supported. The beat must is divided
    regularly and each subdivision can be divided, but there are no tuplets across
    multiple subdivisions of a parent tuplet. This scheme makes tuples like
    ``3:2(8 3:2(8 8 8))`` impossible (a triplet across two/thirds
    of a parent triplet). This however is possible: ``3:2(8 3:2(16 16 16) 8)``
    (``generateBeatGrid(1, [1, 3, 1])``) since each subdivision is encapsulated within
    the duration of its parent.

    Args:
        beatDuration: the duration of the beat
        division: division of the beat. A division is either a simple number indicating
            the division of the beat (3 would indicate a triplet) or a list of divisions

    ::
        >>> generateBeatGrid(1, [4])
        [0, 0.25, 0.5, 0.75, 1.0]

        >>> generateBeatGrid(1, [3, 4])
        [0., 1/6, 2/6, 1/2, 5/8, 3/4, 7/8, 1]

    """
    assert isinstance(division, (list, int))
    assert beatDuration > 0
    beatDuration = asF(beatDuration)

    gridDurations = _gridDurations(beatDuration, division)
    flatDurations = list(iterlib.flatten(gridDurations))
    # flatgrid contains a flat list of the duration of each tick
    # now we need to convert that to offsets
    offsetsGrid = [F(0)] + list(iterlib.partialsum(flatDurations))
    assert offsetsGrid[-1] == beatDuration
    if offset == 0:
        return offsetsGrid
    return [tick + offset for tick in offsetsGrid]


def _fitEventsToGridNearest(events: list[Notation], grid: list[F]) -> list[int]:
    beatDuration = grid[-1]
    assert all(0 <= ev.offset < beatDuration for ev in events)
    assert all(0 <= gridSlot <= beatDuration for gridSlot in grid)
    assignedSlots = [misc.nearest_index(event.offset, grid) for event in events]
    return assignedSlots


def assignSlotsInGrid(events: list[Notation], grid: list[F], method="nearest"
                      ) -> list[int]:
    """
    Fit the notations to the nearest slot in a grid

    Args:
        events: the events to fit to the grid
        grid: a list of offsets within the beat
        method: the method to use. Valid options: "nearest"

    Returns:
        a list of ints of length == len(events), where each int represent the index of
        the slot for the corresponding event.

    .. note::

        Two events can share the same slot, in which case only the last is considered to
        own the slot, the previous events are condsidered to be "grace notes" previous to
        this slot
    """
    if method == "nearest":
        return _fitEventsToGridNearest(events=events, grid=grid)
    else:
        raise ValueError(f"Method {method} not supported. Supported methods: 'nearest'")


def snapEventsToGrid(notations: list[Notation], grid: list[F], method="nearest"
                     ) -> Tuple[list[int], list[Notation]]:
    """
    Snap unquantized events to a given grid

    Args:
        notations: a list of unquantized Notation's
        grid: the grid to snap the events to, as returned by generateBeatGrid
        method: the method to use. Valid values are: "nearest"

    Returns:
        the quantized events
    """
    beatDuration= grid[-1]
    assignedSlots = assignSlotsInGrid(events=notations, grid=grid, method=method)
    snappedEvents = []
    for idx in range(len(notations)-1):
        n = notations[idx]
        slot0 = assignedSlots[idx]
        offset0 = grid[slot0]
        # is it the last slot (as grace note?)
        if slot0 == len(grid) - 1:
            snappedEvents.append(n.clone(offset=offset0, duration=F(0)))
        else:
            offset1 = grid[assignedSlots[idx+1]]
            dur = offset1 - offset0
            snappedEvents.append(n.clone(offset=offset0, duration=dur))

    lastOffset = grid[assignedSlots[-1]]
    dur = beatDuration - lastOffset
    last = notations[-1].clone(offset=lastOffset, duration=dur)
    snappedEvents.append(last)

    return assignedSlots, snappedEvents


def isBeatFilled(events: list[Notation], beatDuration:F, beatOffset:F=F(0)) -> bool:
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
    if any(ev.duration<0 for ev in events):
        raise ValueError(f"Some events have unset durations: "
                         f"{[ev for ev in events if ev.duration<0]}")

    if events[0].offset - beatOffset > 0:
        return False
    if events[-1].end - beatOffset < beatDuration:
        return False
    return all(ev0.end == ev1.offset for ev0, ev1 in iterlib.pairwise(events)) \
           and events[-1].end - beatOffset == beatDuration


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


def _fillDuration(notations: list[Notation], duration: F, offset=F(0)) -> list[Notation]:
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
    assert all(n.offset-offset<duration for n in notations), \
        f"events start after duration ({duration}): {_eventsShow(notations)}"
    assert all(n0.offset <= n1.offset for n0, n1 in iterlib.pairwise(notations)), \
        f"events are not sorted: {_eventsShow(notations)}"
    assert all(n0.end <= n1.offset for n0, n1 in iterlib.pairwise(notations) if n0.duration is not None), \
        f"events overlap: {_eventsShow(notations)}"
    assert all(n.end<=offset+duration for n in notations if n.duration is not None), \
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
        if n.end < duration:
            out.append(makeRest(offset=n.end, duration=duration-n.end))
    assert sum(n.duration for n in out) == duration
    return out


def _evaluateQuantization(profile: QuantizationProfile,
                          eventsInBeat: list[Notation],
                          snappedEvents: list[Notation],
                          beatDuration:F) -> float:
    """
    Evaluate the quantization of the snapped events

    Given a list of events in a beat and these events snapped to a given subdivision of
    the beat, evaluate how good is this snapping in representing the original events.
    This is used to find the best subdivision of a beat.

    Args:
        profile: the quantization profile to use
        eventsInBeat: the unquantized events in the beat
        snappedEvents: the events after being snapped to a given grid
        beatDuration: the duration of the beat

    Returns:
        a value indicative of how much the quantization diverges from
        the unquantized version. Lower is better
    """
    assert isinstance(beatDuration, F)
    offsetErrorWeight = profile.offsetErrorWeight
    restOffsetErrorWeight = profile.restOffsetErrorWeight
    durationErrorWeight = profile.durationErrorWeight
    graceNoteDuration = profile.graceNoteDuration
    graceNoteOffsetErrorFactor = 0.5

    def evaluateEvent(event: Notation, snapped: Notation) -> float:
        offsetError = abs(event.offset - snapped.offset) / beatDuration
        if event.isRest:
            offsetError *= restOffsetErrorWeight / offsetErrorWeight

        if snapped.duration == 0:
            offsetError *= graceNoteOffsetErrorFactor
            durationError = abs(event.duration - graceNoteDuration) / beatDuration
        else:
            durationError = abs(event.duration - snapped.duration) / beatDuration
        error = mathlib.euclidian_distance([float(offsetError), float(durationError)],
                                           [offsetErrorWeight, durationErrorWeight])
        return error

    errors = [evaluateEvent(event, snapped)
              for event, snapped in zip(eventsInBeat, snappedEvents)]
    return sum(errors)


def _notationsFillDurations(ns: list[Notation], beatDuration: F) -> list[Notation]:
    if all(n.duration is not None for n in ns):
        return ns
    out = []
    for i in range(len(ns)-1):
        n = ns[i]
        out.append(n if n.duration >= 0 else n.clone(duration=ns[i+1].offset-n.offset))
    n = ns[-1]
    out.append(n if n.duration >= 0 else n.clone(duration=beatDuration-n.offset))
    return out


def _id2div(divId: str) -> Union[int, list]:
    return eval(divId)


@dataclass
class QuantizedBeat:
    """
    A QuantizedBeat holds notations inside a beat filling the beat
    """
    divisions: division_t
    assignedSlots: list[int]
    notations: list[Notation]  # snapped events
    beatDuration: F
    beatOffset: F = F(0)

    def applyDurationRatios(self):
        _applyDurationRatio(self.notations, division=self.divisions,
                            beatOffset=self.beatOffset, beatDur=self.beatDuration)

    def group(self) -> DurationGroup:
        return _groupByRatio(self.notations, division=self.divisions,
                             beatOffset=self.beatOffset, beatDur=self.beatDuration)

    def __post_init__(self):
        self.applyDurationRatios()


@dataclass
class QuantizedMeasure:
    """
    A QuantizedMeasure holds a list of QuantizedBeats
    """
    timesig: timesig_t
    quarterTempo: F
    beats: Optional[list[QuantizedBeat]] = None
    profile: Optional[QuantizationProfile] = None

    def __post_init__(self):
        if self.beats:
            self.check()

    def isEmpty(self) -> bool:
        if not self.beats:
            return True
        for beat in self.beats:
            if beat.notations and any(not n.isRest or n.spanners for n in beat.notations):
                return False
        return True

    def dump(self, indents=0):
        ind = "  " * indents
        print(f"{ind}Timesig: {self.timesig[0]}/{self.timesig[1]} "
              f"(quarter={self.quarterTempo})")
        if self.isEmpty():
            print(f"{ind}EMPTY")
        else:
            for group in self.groups():
                ind = "  "*(indents+1)
                print(f"{ind}Ratio {group.durRatio}")
                for n in group.items:
                    ind = "  " * (indents+2)
                    print(f"{ind}{n}")

    def iterNotations(self) -> Iterator[Notation]:
        """
        Returns an iterator over the notations in this Measure
        """
        if not self.beats:
            raise StopIteration
        for beat in self.beats:
            for notation in beat.notations:
                yield notation

    def notations(self) -> list[Notation]:
        """
        Returns a flat list of all notations in this measure
        """
        if not self.beats:
            return []
        notations = []
        for beat in self.beats:
            notations.extend(beat.notations)
        assert len(notations) > 0
        assert all(n0.end == n1.offset for n0, n1 in iterlib.pairwise(notations))
        return notations

    def beatGroups(self) -> list[DurationGroup]:
        """
        Returns the contents of this measure grouped as a list of DurationGroups
        """
        if not self.beats:
            return []
        groups = [beat.group().mergeNotations() for beat in self.beats]
        return groups

    def groups(self) -> list[DurationGroup]:
        """
        Returnes a list of DurationGroups representing the items in this measure

        The difference with ``beatGroups()`` is that this method will merge
        notations across beats (for example, when there is a synchopation)
        """
        beatgroups = self.beatGroups()
        minSyncopationDur = self.profile.minBeatFractionAcrossBeats if self.profile else F(1)
        groups = _mergeAcrossBeats(self.timesig, beatgroups,
                                   minBeatFractionAcrossBeats=minSyncopationDur,
                                   mergedTupletsMaxDur=self.profile.mergedTupletsMaxDuration,
                                   quarterTempo=self.quarterTempo)
        return groups

    def beatDurations(self) -> list[F]:
        """
        Returns a list with the durations (in quarterNotes) of the beats in this measure
        """
        if not self.beats:
            return []
        return [beat.beatDuration for beat in self.beats]

    def beatDivisions(self) -> List:
        """
        Returns the divisions of each beat in this measure
        """
        if not self.beats:
            return []
        return [beat.divisions for beat in self.beats]

    def removeUnnecessaryAccidentals(self):
        _removeUnnecessaryAccidentals(self.notations())

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
                logger.error(f"beat dur: {beat.beatDuration}, notations dur: {durNotations}")
                logger.error(beat.notations)
                self.dump()
                raise AssertionError(f"Duration mismatch in beat {i}")

    def fixEnharmonics(self) -> None:
        enharmonics.fixEnharmonicsInPlace(self.notations())


def _removeUnnecessaryAccidentals(ns: list[Notation]) -> None:
    """
    Removes unnecessary accidentals, **in place**

    An accidental is unnecessary if:
    - a note tied to a previous note
    - a note which has already been modified by the same accidental (for example,
        a C# after a C# has been used within the given notations. It stops
        being unnecessary after, for example, C+, in which case a # in C# is a
        necessary accidental

    Args:
        ns: the notations to evaluate

    """
    seen = {}
    for n in ns:
        if n.isRest:
            continue
        if n.tiedPrev:
            n.accidentalHidden = True
            continue
        for pitch in n.pitches:
            notatedPitch = notated_pitch(pitch)
            lastSeen = seen.get(notatedPitch.diatonic_name)
            if lastSeen is None:
                # make accidental necessary only if not diatonic step
                if notatedPitch.diatonic_alteration == 0:
                    # TODO: each note in a chord should have individual attributes (hidden, notehead, etc)
                    n.accidentalHidden = True
            elif lastSeen == notatedPitch.accidental_name:
                n.accidentalHidden = True
            seen[notatedPitch.diatonic_name] = notatedPitch.accidental_name


def _removeInvalidGracenotes(qpart: QuantizedPart) -> None:
    """
    Remove invalid grace notes in this measure, in place
    """
    trash = []
    for loc0, loc1 in iterlib.pairwise(qpart.iterNotations()):
        n0 = loc0[2]
        n1 = loc1[2]
        if n0.tiedNext and n0.pitches == n1.pitches:
            if n0.isGraceNote:
                n0.transferAttributesTo(n1)
                trash.append(loc0)
            elif n1.isGraceNote:
                trash.append(loc1)
    for loc in trash:
        measurenum, beat, n = loc
        beat.notations.remove(n)


def quantizeBeat(eventsInBeat: list[Notation],
                 quarterTempo: number_t,
                 profile: QuantizationProfile,
                 beatDuration: number_t = F(1),
                 beatOffset: F = F(0),
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
        profile: the subdivision profile used

    Returns:
        a BeatQuantisation, where:
        .divisions constains a list of the subdivisions of the beat where:
            * [4] = subdivide the beat in four equal parts
            * [3, 4] = subdivide the beat in two parts, the first part in 3 and the second in 4 parts
            * [8] = subdivide the beat in eigth equal parts
        .assignedSlots constains a list of the assigned slot to each attack

    Example::

        >>> offsets = [0, 0.615]
        >>> attacks = [Notation(F(offset)) for offset in offsets]
        >>> quantizeBeat(attacks, 60, 1)
        BeatDivisionResult(divisions=[8], assignedSlots=[0, 5])

    This indicates that the beat was divided in 8 equal parts, where the first attack was
    assigned the first slot and the second attack was assigned the 5th slot

    # Algorithm

    for each possible division,
        generate a grid of slots
        assign one slot to each event
            rules:
                * the distance between the unquantized offset of the event and the
                  quantized offset of the slot can't excede the duration of one slot
                * if multiple if multiple events are assigned the same slot, the last
                  gets the slot, the previous ones are given the status of a grace note
                  and assigned a duration of 0
                * the solution should minimize the displacement of the attack time and,
                  with less weight, the displacement of the durations

    """
    assert beatDuration > 0
    beatDuration = asF(beatDuration)
    assert beatDuration in {F(1, 1), F(1, 2), F(1, 4), F(2, 1)}
    assert sum(n.duration for n in eventsInBeat) == beatDuration

    if not isBeatFilled(eventsInBeat, beatDuration=beatDuration):
        eventsInBeat = _fillDuration(eventsInBeat, duration=beatDuration, offset=beatOffset)

    assert sum(n.duration for n in eventsInBeat) == beatDuration
    assert all(ev.duration is not None for ev in eventsInBeat)
    assert all(0 <= ev.duration <= beatDuration and
               beatOffset <= ev.offset <= ev.end <= beatOffset+beatDuration
               for ev in eventsInBeat)
    assert sum(ev.duration for ev in eventsInBeat) == beatDuration, _eventsShow(eventsInBeat)

    tempo = asF(quarterTempo) / beatDuration
    possibleDivisions = beatDivisions(tempo, profile=profile)
    beatGrids = [generateBeatGrid(beatDuration=beatDuration, division=div, offset=beatOffset)
                 for div in possibleDivisions]
    possibleGrids = {str(div):beatGrids[i] for i, div in enumerate(possibleDivisions)}
    id2div = {str(div): div for div in possibleDivisions}

    rows = []

    for divId, grid in possibleGrids.items():
        assignedSlots, snappedEvents = snapEventsToGrid(eventsInBeat, grid=grid)
        assert sum(_.duration for _ in snappedEvents) == beatDuration, \
            f"{divId=}, {snappedEvents=}"
        gridError = _evaluateQuantization(profile=profile,
                                          eventsInBeat=eventsInBeat,
                                          snappedEvents=snappedEvents,
                                          beatDuration=beatDuration)
        div = id2div[divId]
        divPenalty = _divisionPenalty(div, profile)

        totalError = mathlib.weighted_euclidian_distance([
            (gridError, profile.gridErrorWeight),
            (divPenalty, profile.divisionErrorWeight)])
        rows.append((totalError, divId, snappedEvents, assignedSlots))

    error, divisionId, snappedEvents, assignedSlots = min(rows, key=lambda row: row[0])
    assert sum(_.duration for _ in snappedEvents) == beatDuration, \
        f"{beatDuration=}, {snappedEvents=}"
    division = _id2div(divisionId)
    if isinstance(division, int):
        division = [division]
    assert isinstance(division, list)
    beatNotations = []
    for ev in snappedEvents:
        if ev.isGraceNote:
            beatNotations.append(ev)
        else:
            eventParts = breakIrregularDuration(ev, beatDivision=division,
                                                beatDur=beatDuration,
                                                beatOffset=beatOffset)
            if not eventParts:
                if ev.duration > 0 or (ev.duration == 0 and not ev.isRest):
                    beatNotations.append(ev)
            else:
                assert sum(_.duration for _ in eventParts) == ev.duration
                beatNotations.extend(eventParts)
    assert sum(ev.duration for ev in beatNotations) == sum(ev.duration for ev in snappedEvents)
    assert sum(ev.duration for ev in beatNotations) == beatDuration, f"{beatDuration=}, {beatNotations=}"
    # assert not any(n.isRest and n.duration == 0 for n in beatNotations), \
    #    f'Invalid notations in beat: {beatNotations}'
    beatNotations = [n for n in beatNotations
                     if not (n.isRest and n.duration == 0)]
    return QuantizedBeat(division, assignedSlots=assignedSlots,
                         notations=beatNotations, beatDuration=beatDuration,
                         beatOffset=beatOffset)


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
    assert all(0<=ev.offset and ev.end<=measureDuration for ev in eventsInMeasure)
    return _fillDuration(eventsInMeasure, measureDuration)


def splitNotationAtOffsets(n: Notation, offsets: list[Rational]) -> list[Notation]:
    """
    Splits a Notation at the given offsets

    Args:
        n: the Notation to split
        offsets: the offsets at which to split n

    Returns:
        the parts after splitting

    Example::

        >>> splitNotationAtOffsets(Notation(F(0.5), duration=F(1)))
        [Notation(0.5, duration=0.5), Notation(1, duration=0.5)]

    """
    if not offsets:
        raise ValueError("offsets is empty")

    assert n.duration is not None and n.duration>=0

    intervals = mathlib.split_interval_at_values(n.offset, n.end, offsets)

    if len(intervals) == 1:
        return [n]

    parts: list[Notation] = [n.clone(offset=start, duration=end-start)
                             for start, end in intervals]
    # Remove superfluous dynamic/articulation
    for part in parts[1:]:
        n.dynamic = ''
        n.articulation = ''

    if not n.isRest:
        parts[0].tiedPrev = n.tiedPrev
        parts[-1].tiedNext = n.tiedNext
        _tieNotationParts(parts)

    assert sum(part.duration for part in parts) == n.duration
    assert parts[0].offset == n.offset
    assert parts[-1].end == n.end

    return parts


_regularSlotNumbers = {1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32}


def notationNeedsBreak(n: Notation, beatDur:F, beatDivision: division_t,
                       beatOffset=F(0)) -> bool:
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
    assert isinstance(beatDivision, list), f"Expected a list, got {type(beatDivision).__name__}"
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
            raise ValueError(f"n is not quantized with given division.\n  n={n}\n  div={beatDivision}")
        assert isinstance(nslots, F), f"Expected nslots of type {F}, got {type(nslots).__name__} (nslots={nslots})"
        return nslots.numerator not in _regularSlotNumbers
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
                return notationNeedsBreak(n, dt, div, beatOffset=now)


def _tieNotationParts(parts: list[Notation]) -> None:
    """ Tie these notations in place """
    for part in parts[:-1]:
        part.tiedNext = True
    for part in parts[1:]:
        part.tiedPrev = True
        part.dynamic = ''
        part.articulation = ''


def _splitIrregularDuration(n: Notation, slotIndex: int, slotDur: F) -> list[Notation]:
    """
    Split irregular durations

    An irregular duration is a duration which cannot be extpressed as a quarter/eights/16th/etc
    For example a beat filled with a sextuplet with durations (1, 5), the second
    note is irregular and must be split. Since it begins in an uneven slot, it is
    split as 1+4

    Args:
        n: the Notation to split
        slotIndex: which slot is n assigned to within the beat/subbeat
        slotDur: which is the quarterNote duration of slotDur

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
    assert isinstance(n, Notation), f"Expected a Notation, got {type(n).__name__}={n}"
    assert isinstance(slotDur, F), f"Expected a F, got {type(slotDur).__name__}={slotDur}"
    assert n.duration > 0

    numSlots = int(n.duration / slotDur)
    if numSlots > 25:
        raise ValueError("Division not supported")

    slotDivisions = quantdata.slotDivisionStrategy[numSlots]
    if slotIndex % 2 == 1 and slotDivisions[-1] % 2 == 1:
        slotDivisions = list(reversed(slotDivisions))

    offset = n.offset
    parts: list[Notation] = []
    for slots in slotDivisions:
        partDur = slotDur * slots
        parts.append(n.clone(offset=offset, duration=partDur))
        offset += partDur

    _tieNotationParts(parts)
    assert sum(part.duration for part in parts) == n.duration
    assert parts[0].offset == n.offset
    assert parts[-1].end == n.end
    return parts


def _breakIrregularDuration(n: Notation, beatDur:F, div: int, beatOffset=F(0)
                            ) -> Optional[list[Notation]]:
    # beat is subdivided regularly
    slotDur = beatDur/div
    nslots = n.duration/slotDur
    assert isinstance(nslots, F), f"Expected type F, got {type(nslots).__name__}={nslots}"

    if nslots.denominator != 1:
        raise ValueError(f"Duration is not quantized with given division.\n  {n=}, {div=}")

    if nslots.numerator in _regularSlotNumbers:
        return None

    slotIndex = (n.offset-beatOffset)/slotDur
    assert int(slotIndex) == slotIndex
    slotIndex = int(slotIndex)

    if not slotIndex.denominator == 1:
        raise ValueError(f"Offset is not quantized with given division. n={n}, div={div}")

    parts = _splitIrregularDuration(n, slotIndex, slotDur)
    assert sum(part.duration for part in parts) == n.duration
    return parts


def breakIrregularDuration(n: Notation, beatDur:Rational, beatDivision: division_t,
                           beatOffset: Rational = F(0)
                           ) -> Optional[list[Notation]]:
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
    assert isinstance(beatDivision, (int, list)), f"Expected type int/list, got {type(beatDivision).__name__}={beatDivision}"
    assert isinstance(beatDur, F), f"Expected type F, got {type(beatDur).__name__}={beatDur}"
    assert isinstance(beatOffset, F), f"Expected type F, got {type(beatOffset).__name__}={beatOffset}"
    assert n.duration>=0

    if n.duration == 0:
        return None

    if n.end > beatOffset + beatDur:
        raise ValueError(f"n extends over the beat. "
                         f"n={n.offset} - {n.end}, beat={beatOffset} - {beatOffset+beatDur}")

    if isinstance(beatDivision, int):
        return _breakIrregularDuration(n, beatDur=beatDur,
                                       div=beatDivision, beatOffset=beatOffset)

    if len(beatDivision) == 1:
        return _breakIrregularDuration(n, beatDur=beatDur,
                                       div=beatDivision[0], beatOffset=beatOffset)

    # beat is not subdivided regularly. check if n extends over subdivision
    numDivisions = len(beatDivision)
    divDuration = beatDur/numDivisions

    ticks = list(mathlib.fraction_range(beatOffset, beatOffset+beatDur+divDuration, divDuration))

    subdivisionTimespans = list(iterlib.pairwise(ticks))
    assert len(subdivisionTimespans) == numDivisions, \
        f"{subdivisionTimespans=}, {beatDivision=}"
    subdivisions = list(zip(subdivisionTimespans, beatDivision))
    subns = splitNotationAtOffsets(n, ticks)
    allparts: list[Notation] = []
    for subn in subns:
        # find the subdivision
        for timespan, numslots in subdivisions:
            if mathlib.intersection(timespan[0], timespan[1], subn.offset, subn.end) is not None:
                parts = breakIrregularDuration(subn, divDuration, numslots, timespan[0])
                if parts is None:
                    # subn is regular
                    allparts.append(subn)
                else:
                    allparts.extend(parts)
    assert sum(part.duration for part in allparts) == n.duration
    _tieNotationParts(allparts)
    assert all(isinstance(part, Notation) for part in allparts)
    assert sum(p.duration for p in allparts) == n.duration
    return allparts


def isMeasureFilled(notations: list[Notation], timesig: timesig_t) -> bool:
    """Do the notations fill the measure?"""
    measureDuration = util.measureQuarterDuration(timesig)
    notationsDuration = sum(n.duration for n in notations)
    if notationsDuration > measureDuration:
        logger.error(f"timesig: {timesig}, Notation: {notations}")
        logger.error(f"Sum duration: {notationsDuration}")
        raise ValueError("notations do not fit in measure")
    return notationsDuration == measureDuration


def measureSplitNotationsAtBeats(eventsInMeasure: list[Notation],
                                 timesig: timesig_t,
                                 quarterTempo: number_t,
                                 ) -> list[Tuple[TimeSpan, list[Notation]]]:
    """
    Split the events in this measure into its individual beats

    Returns a list of tuples (time span of beat, eventsInBeat)
    Used by quantizeMeasures

    The events here have a duration and offset in quarterLength, not in raw
    seconds (``.tempoCorrected`` should be True)

    .. note::

        We ensure that the returned events in beat completely fill the beat

    Args:
        eventsInMeasure: the events within the measure. The offset of each notation
            is relative to the start of the measure. The events should fill the measure
        timesig: the time signature of the measure
        quarterTempo: the tempo (used as hint to divide the measure in beats)

    Returns:
        a list of tuples (timeSpan, eventsInBeat)
    """
    assert isinstance(eventsInMeasure, list) and all(isinstance(_, Notation) for _ in eventsInMeasure)
    assert isinstance(timesig, tuple) and isinstance(timesig[0], int) and isinstance(timesig[1], int) 
    assert isinstance(quarterTempo, (int, F))

    assert isMeasureFilled(eventsInMeasure, timesig), \
        f"Measure is not filled. Timesig {timesig}, tempo: {quarterTempo}\n" \
        f"events: {eventsInMeasure}"

    beatOffsets = util.measureOffsets(timesig=timesig, quarterTempo=quarterTempo)
    timeSpans = [TimeSpan(beat0, beat1) for beat0, beat1 in iterlib.pairwise(beatOffsets)]
    splittedEvents = []
    for ev in eventsInMeasure:
        if ev.duration > 0:
            splittedEvents.extend(splitNotationAtOffsets(ev, beatOffsets))
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
        assert sum(ev.duration for ev in eventsInBeat) == timeSpan.duration
        assert all(timeSpan.start <= ev.offset <= ev.end <= timeSpan.end
                   for ev in eventsInBeat)
    return list(zip(timeSpans, eventsPerBeat))


def _groupByRatio(notations: list[Notation], division:Union[int, division_t],
                  beatOffset:F, beatDur:F
                  ) -> DurationGroup:
    if isinstance(division, int) or len(division) == 1:
        if isinstance(division, list):
            division = division[0]
        durRatio = quantdata.durationRatios[division]
        return DurationGroup(durRatio=durRatio, items=notations)

    assert isinstance(division, list) and len(division) >= 2
    numSubBeats = len(division)
    now = beatOffset
    dt = beatDur/numSubBeats
    durRatio = quantdata.durationRatios[numSubBeats]
    items = []
    for subdiv in division:
        subdivEnd = now+dt
        subdivNotations = [n for n in notations if now<=n.offset<subdivEnd and n.end<=subdivEnd]
        if subdiv == 1:
            items.extend(subdivNotations)
        else:
            items.append(_groupByRatio(subdivNotations, subdiv, now, dt))
        now += dt
    return DurationGroup(durRatio, items)


def _applyDurationRatio(notations:list[Notation], division:Union[int, division_t],
                        beatOffset:F, beatDur:F) -> None:
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
        if isinstance(division, list):
            division = division[0]
        durRatio = quantdata.durationRatios[division]
        for n in notations:
            if n.durRatios is None:
                n.durRatios = []
            n.durRatios.append(F(*durRatio))
    else:
        numSubBeats = len(division)
        now = beatOffset
        dt = beatDur / numSubBeats
        durRatio = F(*quantdata.durationRatios[numSubBeats])
        for n in notations:
            if n.durRatios is None:
                n.durRatios = []
            n.durRatios.append(durRatio)
        for subdiv in division:
            subdivEnd = now + dt
            subdivNotations = [n for n in notations
                               if now <= n.offset < subdivEnd and n.end <= subdivEnd]
            _applyDurationRatio(subdivNotations, subdiv, now, dt)
            now += dt


def quantizeMeasure(events: list[Notation],
                    timesig: timesig_t,
                    quarterTempo: number_t,
                    profile: QuantizationProfile
                    ) -> QuantizedMeasure:
    """
    Quantize notes in a given measure

    Args:
        events: the events inide the measure. The offset is relative
            to the beginning of the measure. Offset and duration are in
            quarterLengths, i.e. they are not dependent on tempo. The tempo
            is used as a hint to find a suitable quantization
        timesig: the time signature of the measure: a touple (num, den)
        quarterTempo: the tempo of the measure using a quarter note as refernce
        profile: the quantization profile. Leave it unset to use the default
            profile.

    Returns:
        a QuantizedMeasure

    """
    measureQuarterLength = util.measureQuarterDuration(timesig)
    assert all(0<=ev.offset<=ev.end<=measureQuarterLength
               for ev in events)
    for ev0, ev1 in iterlib.pairwise(events):
        if ev0.end != ev1.offset:
            logger.error(f"{ev0} (end={ev0.end}), {ev1} (offset={ev1.offset})")
            raise AssertionError("events are not stacked")

    quantBeats: list[QuantizedBeat] = []

    if not isMeasureFilled(events, timesig):
        events = _fillMeasure(events, timesig)

    for span, eventsInBeat in measureSplitNotationsAtBeats(eventsInMeasure=events,
                                                           timesig=timesig,
                                                           quarterTempo=quarterTempo):
        beatDuration = span.end - span.start
        quantizedBeat = quantizeBeat(eventsInBeat=eventsInBeat,
                                     quarterTempo=quarterTempo,
                                     beatDuration=beatDuration,
                                     beatOffset=span.start,
                                     profile=profile)
        quantBeats.append(quantizedBeat)
    return QuantizedMeasure(timesig=timesig, quarterTempo=asF(quarterTempo), beats=quantBeats,
                            profile=profile)


def splitNotationByMeasure(struct: ScoreStruct,
                           event: Notation,
                           ) -> list[Tuple[int, Notation]]:
    """
    Split a Notation if it extends across multiple measures.

    Args:
        struct: the ScoreStructure
        event: the Notation to split

    Returns:
        a list of tuples (measure number, notation), indicating
        to which measure each part belongs to. The notation in the
        tuple has an offset relative to the beginning of the measure

    """
    assert event.offset >= 0 and event.duration >= 0
    loc0 = struct.beatToLocation(event.offset)
    loc1 = struct.beatToLocation(event.end)

    if loc0 is None or loc1 is None:
        raise ValueError(f"Could not find a score location for this event: {event}")

    if loc1.beat == 0 and event.duration > 0:
        # Note ends at the barline
        loc1.measureIndex -= 1
        loc1.beat = struct.getMeasureDef(loc1.measureIndex).durationBeats()

    numMeasures = loc1.measureIndex - loc0.measureIndex + 1
    assert numMeasures >= 1, f"{loc0=}, {loc1=}"

    if numMeasures == 1:
        # The note fits within one measure. Make the offset relative to the measure
        event = event.clone(offset=loc0.beat, duration=loc1.beat-loc0.beat)
        return [(loc0.measureIndex, event)]

    measuredef = struct.getMeasureDef(loc0.measureIndex)
    dur = measuredef.durationBeats() - loc0.beat
    notation = event.clone(offset=loc0.beat, duration=dur, tiedNext=True)
    pairs = [(loc0.measureIndex, notation)]

    # add intermediate measure, if any
    if numMeasures > 2:
        for m in range(loc0.measureIndex + 1, loc1.measureIndex):
            measuredef = struct.getMeasureDef(m)
            notation = event.clone(offset=F(0),
                                   duration=measuredef.durationBeats(),
                                   tiedPrev=True, tiedNext=True, dynamic='',
                                   articulation='')
            pairs.append((m, notation))

    # add last notation
    if loc1.beat > 0:
        notation = event.clone(offset=F(0), duration=loc1.beat, tiedPrev=True,
                               dynamic='', articulation='')
        pairs.append((loc1.measureIndex, notation))

    sumdur = sum(struct.beatDelta((i, n.offset), (i, n.end)) for i, n in pairs)
    assert sumdur == event.duration, f"{event=}, {sumdur=}, {numMeasures=}\n{pairs=}"
    return pairs


def _removeOverlapInplace(notations: list[Notation], threshold=F(1,1000)) -> None:
    """
    Remove overlap between notations.

    This should be only used to remove small overlaps product of rounding errors.
    """
    removed = []
    for n0, n1 in iterlib.pairwise(notations):
        assert n0.offset <= n1.offset
        diff = n0.end-n1.offset
        if diff > 0:
            if diff > threshold:
                raise ValueError(f"Notes overlap by too much: {diff}, {n0}, {n1}")
            duration = n1.offset - n0.offset
            if duration <= 0:
                removed.append(n0)
            else:
                n0.duration = duration
    for n in removed:
        notations.remove(n)


_splitPointsByTimesig: dict[Tuple[int, int], list[number_t]] = {
    (4, 4): [2],
}


def _mergeAcrossBeats(timesig: timesig_t, groups: list[DurationGroup],
                      minBeatFractionAcrossBeats=F(1),
                      mergedTupletsMaxDur=F(2),
                      allowedTupletsAcrossBeats=(1, 2, 3, 4, 8),
                      quarterTempo:number_t=60
                      ) -> list[DurationGroup]:
    """
    Merges quantized notations across beats

    Args:
        timesig: the time signature of the measure
        groups: as returned by quantizedMeasure.groups()
        minBeatFractionAcrossBeats: the min. duration of a note merged across
            a beat (in terms of the beat duration)
        quarterTempo: the tempo of the quarter note

    Returns:
        a list of DurationGroup

    """
    flatNotations: list[Tuple[Notation, durratio_t]] = []
    for group in groups:
        for item in group.items:
            assert isinstance(item, Notation), f"Nested tuples are not supported yet"
            flatNotations.append((item, group.durRatio))
    mergedNotations = [flatNotations[0]]
    beatDur = util.beatDurationForTimesig(timesig, quarterTempo)
    minMergedDuration = minBeatFractionAcrossBeats * beatDur
    for n1, durRatio1 in flatNotations[1:]:
        n0, durRatio0 = mergedNotations[-1]
        assert isinstance(n0, Notation)
        mergedDur = n0.duration+n1.duration
        if durRatio0 == durRatio1 and core.notationsCanMerge(n0, n1):
            # splitPoints = _splitPointsByTimesig.get(timesig)
            if ((mergedDur < minMergedDuration) or
                # (splitPoints and any(n0.offset < p < n1.end for p in splitPoints)) or
                (n0.gliss and n0.duration+n1.duration>=2 and n0.tiedPrev) or
                (durRatio0[1] not in allowedTupletsAcrossBeats) or
                (durRatio0 != (1, 1) and mergedDur>mergedTupletsMaxDur)):
                # can't merge
                mergedNotations.append((n1, durRatio1))
            else:
                merged = core.mergeNotations(n0, n1)
                mergedNotations[-1] = (merged, durRatio0)
        else:
            mergedNotations.append((n1, durRatio1))
    # Rebuild duration groups
    n, durRatio = mergedNotations[0]
    groups = [(durRatio, [n])]
    for n, durRatio in mergedNotations[1:]:
        lastgroup = groups[-1]
        if durRatio == lastgroup[0]:
            # Check if group is full
            groupdur = sum(_.duration for _ in lastgroup[1])
            if durRatio[1] != (1, 1) and groupdur == _maxTupletLength(timesig, durRatio[1]):
                groups.append((durRatio, [n]))
            else:
                lastgroup[1].append(n)
        else:
            groups.append((durRatio, [n]))
    return [DurationGroup(durRatio, items) for durRatio, items in groups]


def _maxTupletLength(timesig: timesig_t, tuplet:int):
    den = timesig[1]
    if tuplet == 3:
        return {2: 2, 4:2, 8: 1}[den]
    elif tuplet == 5:
        return 2 if den == 2 else 1
    else:
        return 1


class PartLocation(NamedTuple):
    """
    Represents a location (a Notation inside a Part) inside a part
    """
    measureNum: int
    beat: QuantizedBeat
    notation: Notation


@dataclass
class QuantizedPart:
    """
    A Part which has already been quantized following a ScoreStruct
    """
    struct: ScoreStruct
    measures: list[QuantizedMeasure]
    label: str = ""

    def __post_init__(self):
        self._fixTies()
        self.removeUnnecessaryGracenotes()

    def flatNotations(self) -> Iterator[Notation]:
        for m in self.measures:
            for n in m.notations():
                yield n

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

    def iterNotations(self) -> Iterator[PartLocation]:
        """
        Iterates over all notations giving the location of each notation
        For each notation yields a tuple:

        (measure number, QuantizedBeat, Notation)
        """
        for i, m in enumerate(self.measures):
            if m.isEmpty():
                continue
            assert m.beats is not None
            for b in m.beats:
                for n in b.notations:
                    yield PartLocation(i, b, n)

    def dump(self, indents=0):
        for m in self.measures:
            m.dump(indents=indents)

    def bestClef(self) -> str:
        """Returns the best clef for this part"""
        return bestClefForPart(self)

    def removeUnnecessaryDynamics(self, resetAfterEmptyMeasure=True) -> None:
        """
        Remove superfluous dynamics in this part, in place
        """
        dynamic = ''
        for meas in self.measures:
            if meas.isEmpty() and resetAfterEmptyMeasure:
                dynamic = ''
                continue
            for n in meas.notations():
                if n.isRest:
                    continue
                if not n.tiedPrev and n.dynamic and n.dynamic in definitions.dynamicLevels:
                    # Only dynamic levels are ever superfluous (f, ff, mp), other 'dynamics'
                    # like sf should not be removed
                    if n.dynamic == dynamic:
                        n.dynamic = ''
                    else:
                        dynamic = n.dynamic

    def removeUnnecessaryGracenotes(self) -> None:
        """
        Removes unnecessary gracenotes

        An unnecessary gracenote is one which has the same pitch as the
        next real note and starts a glissando. Such gracenotes might
        be created during quantization.

        """
        trash: list[PartLocation] = []
        for loc0, loc1 in iterlib.pairwise(self.iterNotations()):
            n0 = loc0.notation
            n1 = loc1.notation
            if n0.tiedNext and n0.pitches == n1.pitches:
                if n0.isGraceNote:
                    n0.transferAttributesTo(n1)
                    trash.append(loc0)
                elif n1.isGraceNote:
                    n0.tiedNext = False
                    trash.append(loc1)
            elif (n0.isGraceNote and n0.gliss and
                    n0.end == n1.offset and
                    len(n0.pitches) == 1 and len(n1.pitches) == 1 and
                    n0.pitches[0] == n1.pitches[0]):
                trash.append(loc0)
        for loc in trash:
            try:
                loc.beat.notations.remove(loc.notation)
            except:
                logger.info(f"Could not remove gracenote: {loc.notation} ({loc}")

    def glissMarkTiedNotesAsHidden(self) -> None:
        """
        Within a glissando, notes tied to previous and next notes can be hidden
        """
        it = self.iterNotations()
        for loc in it:
            n = loc.notation
            if n.gliss and not n.tiedPrev and n.tiedNext:
                # this starts a glissando and has tied notes after
                for loc2 in it:
                    if loc2.notation.tiedPrev:
                        loc2.notation.notehead = "hidden"
                    if not loc2.notation.tiedNext:
                        break

    def _fixTies(self):
        for n0, n1 in iterlib.pairwise(self.flatNotations()):
            if n0.tiedNext and not any(x in n1.pitches for x in n0.pitches):
                n0.tiedNext = False
                n1.tiedPrev = False

    def fixEnharmonics(self):
        """
        Find best enharmonic spelling for notes in this part
        """
        for m in self.measures:
            m.fixEnharmonics()

    def pad(self, numMeasures: int) -> None:
        """Add the given number of empty measures at the end"""
        if numMeasures <= 0:
            return
        l = len(self.measures)
        for measureIndex in range(l - 1, l - 1 + numMeasures):
            measuredef = self.struct.getMeasureDef(measureIndex)
            empty = QuantizedMeasure(timesig=measuredef.timesig,
                                     quarterTempo=measuredef.quarterTempo)
            self.measures.append(empty)


def quantizePart(part: core.Part,
                 struct: ScoreStruct,
                 profile: Union[str, QuantizationProfile] = 'default',
                 fillStructure=False,
                 ) -> QuantizedPart:
    """
    Quantizes a sequence of non-overlapping events (a "part")

    Quantize to the score structure defined in `struct`, according to the strategies
    defined in `profile`

    Args:
        struct: the ScoreStruct to use
        part: the events to quantize. Event within a part
            should not overlap
        fillStructure: if True and struct is not endless, the
            generated Part will have as many measures as are defined
            in the struct. Otherwise only as many measures as needed
            to hold the given events will be created
        profile: the QuantizationProfile used

    Returns:
        a list of QuantizedMeasures. To convert these to a Part,
        call convertQuantizedMeasuresToPart

    """
    assert isinstance(part, core.Part)
    if isinstance(profile, str):
        profile = makeQuantizationProfile(profile)
    part.fillGaps()
    assert not part.hasGaps()
    label = part.label
    part = core.stackNotations(part)
    allpairs = [splitNotationByMeasure(struct, event) for event in part]
    maxMeasure = max(pairs[-1][0] for pairs in allpairs)
    notationsPerMeasure: list[list[Notation]] = [[] for _ in range(maxMeasure+1)]
    for pairs in allpairs:
        for measureIdx, notation in pairs:
            notationsPerMeasure[measureIdx].append(notation)
    qmeasures = []
    for idx, notations in enumerate(notationsPerMeasure):
        measureDef = struct.getMeasureDef(idx)
        if not notations:
            qmeasures.append(QuantizedMeasure(measureDef.timesig,
                                              measureDef.quarterTempo))
        else:
            notations.sort(key=lambda notation:notation.offset)
            _removeOverlapInplace(notations)
            try:
                qmeasure = quantizeMeasure(notations,
                                           timesig=measureDef.timesig,
                                           quarterTempo=measureDef.quarterTempo,
                                           profile=profile)
            except QuantError as e:
                logger.error(f"Error quantizing measure {idx}")
                logger.error(f"   notations in measure: {notationsPerMeasure}")
                raise e
            qmeasures.append(qmeasure)
    if fillStructure:
        if struct.endless:
            raise ValueError("Cannot fill an endless ScoreStructure")
        for i in range(maxMeasure+1, struct.numDefinedMeasures()):
            measureDef = struct.getMeasureDef(i)
            qmeasure = QuantizedMeasure(timesig=measureDef.timesig,
                                        quarterTempo=measureDef.quarterTempo, beats=[])
            qmeasures.append(qmeasure)
    part = QuantizedPart(struct, qmeasures, label=label)
    part.glissMarkTiedNotesAsHidden()
    part.removeUnnecessaryGracenotes()
    return part


def bestClefForPart(part: QuantizedPart, maxNotes=0) -> str:
    """
    Return the best clef for the notations in this part

    The returned str if one of 'treble', 'treble8',
    'bass' and 'bass8'

    Args:
        part: a quantized part

    Returns:
        the clef descriptor which best fits this part; one of 'treble',
        'treble8', 'bass', 'bass8', where the 8 indicates an octave
        transposition in the direction of the clef (high for treble,
        low for bass)
    """
    # Only analyze the first n notes
    avgPitch = part.averagePitch(maxNotations=maxNotes)

    if avgPitch == 0:
        # all rests
        return "treble"

    if avgPitch > 80:
        return "treble8"
    elif avgPitch > 58:
        return "treble"
    elif avgPitch > 36:
        return "bass"
    else:
        return "bass8"


@dataclass
class QuantizedScore:
    """
    A QuantizedScore represents a list of quantized parts

    Args:
        parts: A list of QuantizedParts
        title: Title of the score, used for rendering purposes
        composer: Composer of the score, used for rendering

    See :func:`quantize` for an example
    """
    parts: list[QuantizedPart]
    """A list of QuantizedParts"""

    title: Optional[str] = None
    """Title of the score, used for rendering purposes"""

    composer: Optional[str] = None
    """Composer of the score, used for rendering"""

    def __getitem__(self, item: int) -> QuantizedPart:
        return self.parts[item]

    def __iter__(self) -> Iterator[QuantizedPart]:
        return iter(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def dump(self) -> None:
        for part in self:
            part.dump()

    @property
    def scorestruct(self) -> ScoreStruct:
        """Returns the ScoreStruct of this score"""
        return self.parts[0].struct

    @scorestruct.setter
    def scorestruct(self, struct: ScoreStruct) -> None:
        if self.parts:
            for part in self.parts:
                part.struct = struct


    def removeUnnecessaryDynamics(self):
        """Removes any unnecessary dynamics in this score"""
        for part in self:
            part.removeUnnecessaryDynamics()

    def numMeasures(self) -> int:
        """Returns the number of measures in this score"""
        return max(len(part.measures)
                   for part in self.parts)

    def padEmptyMeasures(self) -> None:
        """Adds empty measures at the end of each part so that all have the same amount"""
        numMeasures = self.numMeasures()
        for part in self.parts:
            part.pad(numMeasures - len(part.measures))


def quantize(parts: list[core.Part],
             struct: ScoreStruct = None,
             quantizationProfile: QuantizationProfile = None,
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
            The quantization profile determines how events are quantized,
            which divisions of the beat are possible, how a best division
            is weighted and selected, etc. Not all options in a profile
            are supported by all backends (for example, music21 backend
            does not support nested tuples).
            See quant.presetQuantizationProfiles, which is a dict with
            some predefined profiles

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
        >>> notations = [scoring.Notation(dur, [p]) for dur, p in notes]
        >>> part = scoring.Part(notations)
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
        struct = ScoreStruct.fromTimesig((4, 4), quarterTempo=60)
    qparts = []
    for part in parts:
        qpart = quantizePart(part, struct=struct, profile=quantizationProfile)
        qpart.label = part.label
        qparts.append(qpart)
    return QuantizedScore(qparts)


# -------------------------------------------

def tests():
    evs = [Notation(F(t)) for t in [0.25, 0.5, 1.5]]
    for dur in [1, 2, 3, 4]:
        filled = _fillDuration([Notation(F(0.5))], F(dur))
        sumdur = sum(ev.duration for ev in filled)
        assert  sumdur == F(dur), f"Expected dur: {dur}, got {sumdur} (filled events={filled})"
    assert sum(ev.duration for ev in _fillMeasure(evs, (3, 4))) == 3
    assert sum(ev.duration for ev in _fillMeasure(evs, (5, 8))) == F(5, 2)


