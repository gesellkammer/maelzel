from __future__ import annotations

from .common import *
from . import core
from . import util

from . import scorestruct
from .scorestruct import Notation

import dataclasses

from typing import List, Union as U, Tuple, Optional as Opt, Dict, Iterator as Iter
from emlib import iterlib
from emlib import misc
from emlib import mathlib
from emlib.pitchtools import notated_pitch


_presets = {
    # Possible divisions of a pulse, depending on the tempo for the given pulse
    # These are always simple pulses (quarter note, 8th note, etc.)
    'default.divisionsByTempo': {
        40: [],
        63: [1, 2, 3, 4, 5, [2, 3], 6, 7, [1, 1, 5], [2, 2, 3], [2, 5], [3, 4], 8, [1, 2, 5], [2, 3, 3], [3, 5],
             9, [1, 3, 5], [4, 5], [2, 2, 5],
             10, [2, 3, 5], [5, 6], [3, 3, 5],
             12, 14, [8, 8] ],
        80: [1, 2, 3, 4, 5, [2, 3], 6, 7, [2, 5], [2, 2, 3], [3, 4], 8, [3, 5], 9, [4, 5], 10, [5, 6], 12],
        100: [1, 2, 3, 4, 5, [2, 3], 6, 7, [3, 4], 8, 9, 10],
        132: [1, 2, 3, 4, 5, 6],
        180: [1, 2, 3, 4],
        200: [1, 2, 3],
        999: [1]
    },
    'simple.divisionsByTempo': {
        40: [],
        63: [1, 2, 3, 4, 5, [2, 3], 6, [3, 4], [4, 4], 9, [4, 5], [5, 5],
             [4, 4, 4], [8, 8]],
        80: [1, 2, 3, 4, 5, [2, 3], 6, [4, 4], [4, 5], [5, 5], [4, 4, 4]],
        100: [1, 2, 3, 4, 5, 6, [4,4]],
        132: [1, 2, 3, 4, 5, 6],
        160: [1, 2, 3, 4],
        200: [1, 2, 3],
        999: [1]
    },
    'default.divisionPenaltyMap': {
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.02,
        5: 0.04,
        6: 0.02,
        7: 0.05,
        8: 0.02,
        9: 0.05,
        10: 0.07,
        11: 0.3,
        12: 0.1,
        13: 0.3,
        14: 0.1,
        16: 0.4
    }
}

# how to divide an irregular duration into regular parts
# Regular durations are those which can be expressed via
# a quarter, eighth, 1/16 note, or any dotted or double
# dotted variation thereof
_slotDivisionStrategy = {
    5: (4, 1),
    9: (8, 1),
    10: (8, 2),
    11: (8, 3),
    13: (12, 1),
    15: (12, 3),
    17: (16, 1),
    18: (16, 2),
    19: (16, 3),
    20: (16, 4),
    21: (16, 4, 1),
    22: (16, 6),
    23: (16, 4, 3),
    25: (16, 8, 1)
}


# these are ratios to convert a duration back to its representation
# if a notation has an effective duration of 1/5 (one 16th of a 5-tuplet),
# applying the ratio 5/4 will convert it to 1/4, i.e, a 16th note
# the ratio can then be used to generate the needed tuplet by the notation
# backend
_durationRatios = {
    1: (1, 1),
    2: (1, 1),
    3: (3, 2),
    4: (1, 1),
    5: (5, 4),
    6: (3, 2),
    7: (7, 4),
    8: (1, 1),
    9: (9, 8),
    10: (5, 4),
    11: (11, 8),
    12: (3, 2),
    13: (13, 8),
    14: (7, 4),
    15: (15, 8),
    16: (1, 1),
    17: (17, 16),
    18: (9, 8),
    19: (19, 16),
    20: (5, 4),
    21: (21, 16),
    22: (11, 8),
    23: (23, 16),
    24: (3, 2),
    25: (5, 4),
    26: (13, 8),
    27: (27, 16),
    28: (7, 4),
    29: (29, 16),
    30: (5, 4),
    31: (31, 16),
    32: (1, 1)
}


def _factory(obj) -> dataclasses.field:
    return dataclasses.field(default_factory=lambda:obj)


def _presetField(key) -> dataclasses.field:
    return _factory(_presets[key])


@dataclasses.dataclass
class QuantizationProfile:
    """
    Most important parameters:

    nestedTuples: if True, allow nested tuples. NB: musicxml rendered
        via MuseScore does not support nested tuples
    gridErrorWeight: a weight to control the overall effect of offset
        and duration errors when fitting events to a grid. A higher
        weight will cause quantization to minimize offset and duration
        errors, at the cost of choosing more complex divisions
    divisionErrorWeight: also a weight to controll all effect
        dealing with the complexity of a given division/subdivision

    Lower level parameters to calculate grid error:

    offsetErrorWeight: the importance of offset errors to calculate
        the best subdivision of a beat
    restOffsetErrorWeight: how relevant should be the offset error in
        the case of rests
    durationErrorWeight: relevance of duration error when selecting the
        best subdivision
    graceNoteDuration: if a note is considered a grace note (which have
        no duration per se), should we still account for this duration?

    Lower level parameters to calculate division complexity:

    levelPenaltyWeight: how 

    """
    nestedTuples: bool = False
    gridErrorWeight: float = 0.2
    divisionErrorWeight: float = 0.08

    offsetErrorWeight: float = 1.0
    restOffsetErrorWeight: float = 0.25
    durationErrorWeight: float = 0.1
    graceNoteDuration: F = F(1, 32)

    possibleDivisionsByTempo: Dict[int, list] = _presetField('default.divisionsByTempo')
    divisionPenaltyMap: Dict[int, float] = _presetField('default.divisionPenaltyMap')
    divisionCardinalityPenaltyMap: Dict[int, float] = _factory({1:0.0, 2:0.2, 3:0.4})
    levelPenalty: List[float] = _factory([0., 0.05, 0.4, 0.5, 0.8, 0.8])
    numSubdivsPenaltyMap: Dict[int, float] = _factory({1:0.0, 2:0.0, 3:0.0})

    divisionPenaltyWeight: float = 1.0
    cardinalityPenaltyWeight: float = 0.1
    levelPenaltyWeight: float = 0.1
    numSubdivisionsPenaltyWeight: float = 0.2


defaultQuantizationProfile = QuantizationProfile()

presetQuantizationProfiles = {
    'default': defaultQuantizationProfile,
    'music21': QuantizationProfile(nestedTuples=False),
    'simple': QuantizationProfile(nestedTuples=False,
                                  possibleDivisionsByTempo=_presetField('simple.divisionsByTempo'),
                                  )
}


def _divisionPenalty(div: U[int, list],
                     profile: QuantizationProfile = defaultQuantizationProfile,
                     nestingLevel=1,
                     ) -> float:
    """
    Evaluate the given division. The lower the returned value, the
    simpler this division is. All things being equal, a simpler division
    should be preferred.

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
    divPenalty = sum(_divisionPenalty(subdiv, profile, nestingLevel+1) for subdiv in div)/len(div)
    levelPenalty = profile.levelPenalty[nestingLevel]
    penalty = mathlib.weighted_euclidian_distance([
        (divPenalty, profile.divisionPenaltyWeight),
        (cardinalityPenalty, profile.cardinalityPenaltyWeight),
        (levelPenalty, profile.levelPenaltyWeight),
        (numSubdivsPenalty, profile.numSubdivisionsPenaltyWeight)
    ])
    return min(penalty, 1)


def beatDivisions(tempo:number_t = 60,
                  profile: U[str, QuantizationProfile]="default"
                  ) -> List[division_t]:
    """
    Given a tempo, return the possible subdivisions of the duration reference to which
    this tempo applies according to the given profile

    Example 1: return the possible subdivisions of a quarter note at tempo 60

    >>> beatDivisions(60)

    To determine the possible subdivisions of an eigth note, just double the tempo
    """
    if isinstance(profile, str):
        quantProfile = presetQuantizationProfiles.get(profile)
        if quantProfile is None:
            raise KeyError(f"Quantization profile {profile} unknown. Known profiles: {presetQuantizationProfiles.keys()}")
    else:
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


def _gridDurations(beatDuration: F, division: division_t) -> List[F]:
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


def generateBeatGrid(beatDuration: number_t, division: division_t, offset=F(0)) -> List[F]:
    """
    >>> generateBeatGrid(1, [4])
    [0, 0.25, 0.5, 0.75, 1.0]

    >>> generateBeatGrid(1, [3, 4])
    [0., 1/6, 2/6, 1/2, 5/8, 3/4, 7/8, 1]
    """
    assert isinstance(division, (list, int))
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


def _fitEventsToGridNearest(events: List[Notation], grid: List[F]) -> List[int]:
    beatDuration = grid[-1]
    assert all(0 <= ev.offset < beatDuration for ev in events)
    assert all(0 <= gridSlot <= beatDuration for gridSlot in grid)
    assignedSlots = [misc.nearest_index(event.offset, grid) for event in events]
    return assignedSlots


def assignSlotsInGrid(events: List[Notation], grid: List[F], method="nearest") -> List[int]:
    """

    Args:
        events: the events to fit to the grid
        grid: a list of offsets within the beat
        method: the method to use. Valid options: "nearest"

    Returns:
        a list of ints of length == len(events), where each int represent the index of the slot
        for the corresponding event.

    NB: two events can share the same slot, in which case only the last is considered to own
    the slot, the previous events are condsidered to be "grace notes" previous to this slot
    """
    if method == "nearest":
        return _fitEventsToGridNearest(events=events, grid=grid)
    else:
        raise ValueError(f"Method {method} not supported. Supported methods: 'nearest'")


def snapEventsToGrid(events: List[Notation], grid: List[F], method="nearest"
                     ) -> Tuple[List[int], List[Notation]]:
    """
    Snap unquantized events to a given grid

    Args:
        events: a list of unquantized Notation's
        grid: the grid to snap the events to, as returned by generateBeatGrid
        method: the method to use. Valid values are: "nearest"

    Returns:
        the quantized events
    """
    beatDuration= grid[-1]
    assignedSlots = assignSlotsInGrid(events=events, grid=grid, method=method)
    snappedEvents = []
    for idx in range(len(events)-1):
        ev = events[idx]
        slot0 = assignedSlots[idx]
        offset0 = grid[slot0]
        # is it the last slot (as grace note?)
        if slot0 == len(grid) - 1:
            snappedEvents.append(dataclasses.replace(ev, offset=offset0, duration=F(0)))
        else:
            offset1 = grid[assignedSlots[idx+1]]
            snappedEvents.append(dataclasses.replace(ev, offset=offset0, duration=offset1-offset0))

    lastOffset = grid[assignedSlots[-1]]
    last = dataclasses.replace(events[-1], offset=lastOffset, duration=beatDuration-lastOffset)
    snappedEvents.append(last)
    return assignedSlots, snappedEvents


def isBeatFilled(events: List[Notation], beatDuration:F, beatOffset:F=F(0)) -> bool:
    """
    Check if notations fill the beat exactly (are there holes in the beat?
    are all durations already set?)

    Args:
        events: list of notations inside the beat to check
        beatDuration: the duration of the beat
        beatOffset: the offset of the start of the beat

    Returns:
        True if the notations fill the beat

    """
    if any(ev.duration<0 for ev in events):
        raise ValueError(f"Some events have unset durations: {[ev for ev in events if ev.duration<0]}")

    if events[0].offset - beatOffset > 0:
        return False
    if events[-1].end - beatOffset < beatDuration:
        return False
    return all(ev0.end == ev1.offset for ev0, ev1 in iterlib.pairwise(events)) \
           and events[-1].end - beatOffset == beatDuration


def _eventsShow(events: List[Notation]) -> str:
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


def fillDuration(events: List[Notation], duration: F, offset=F(0)) -> List[Notation]:
    """
    Fill a beat/measure with silences / extend unset durations to next notation
    After calling this, the returned list of notations should fill the given
    duration exactly. This function is normally called prior to quantization

    Args:
        events: a list of notations inside the beat
        duration: the duration to fill
        offset: the starting time to fill

    Returns:
        a list of notations which fill the beat exactly

    NB: if any notation has an unset duration, this will extend either to
        the next notation or to fill the given duration

    """
    # print("events", _eventsShow(events), "offset", offset)
    assert all(ev.offset is not None for ev in events)
    assert all(ev.offset - offset < duration for ev in events), f"events start after duration ({duration}): {_eventsShow(events)}"
    assert all(ev0.offset <= ev1.offset for ev0, ev1 in iterlib.pairwise(events)), f"events are not sorted: {_eventsShow(events)}"
    assert all(ev0.end <= ev1.offset for ev0, ev1 in iterlib.pairwise(events) if ev0.duration is not None), f"events overlap: {_eventsShow(events)}"
    assert all(ev.end <= offset + duration for ev in events if ev.duration is not None), "events extend over beat duration"

    out = []
    now = offset

    if not events:
        # measure is empty
        out.append(Notation(duration, offset=now, rest=True))
        return out

    if events[0].offset > now:
        out.append(Notation(duration=events[0].offset-now, offset=now, rest=True))
        now = events[0].offset

    for ev0, ev1 in iterlib.pairwise(events):
        if ev0.offset > now:
            # there is a gap, fill it with a rest
            out.append(Notation(offset=now, duration=ev0.offset - now, rest=True))
        if ev0.duration is None:
            out.append(dataclasses.replace(ev0, duration=ev1.offset - ev0.offset))
        else:
            out.append(ev0)
            if ev0.end < ev1.offset:
                out.append(Notation(offset=ev0.end, duration=ev1.offset - ev0.end, rest=True))
        now = ev1.offset

    # last event
    ev = events[-1]
    if ev.duration is None:
        out.append(dataclasses.replace(ev, duration=duration-ev.offset))
    else:
        out.append(ev)
        if ev.end < duration:
            out.append(Notation(offset=ev.end, duration=duration-ev.end, rest=True))
    return out


def evaluateQuantization(profile: QuantizationProfile,
                         eventsInBeat: List[Notation],
                         snappedEvents: List[Notation],
                         beatDuration:F) -> float:
    """
    Given a list of events in a beat and these events snapped to a given subdivision
    of the beat, evaluate how good is this snapping in representing the original
    events. This is used to find the best subdivision of a beat.

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

    def evaluateEvent(event: Notation, snapped) -> float:
        offsetError = abs(event.offset - snapped.offset) / beatDuration
        if snapped.duration == 0:
            offsetError *= graceNoteOffsetErrorFactor
        if event.rest:
            offsetError *= restOffsetErrorWeight / offsetErrorWeight
        if snapped.duration == 0:
            durationError = abs(event.duration - graceNoteDuration) / beatDuration
        else:
            durationError = abs(event.duration - snapped.duration) / beatDuration
        error = mathlib.euclidian_distance([offsetError, durationError],
                                           [offsetErrorWeight, durationErrorWeight])
        return error

    errors = [evaluateEvent(event, snapped) for event, snapped in zip(eventsInBeat, snappedEvents)]
    return sum(errors)


def _notationsFillDurations(ns: List[Notation], beatDuration: F) -> List[Notation]:
    if all(n.duration is not None for n in ns):
        return ns
    out = []
    for i in range(len(ns)-1):
        n = ns[i]
        out.append(n if n.duration >= 0 else dataclasses.replace(n, duration=ns[i+1].offset-n.offset))
    n = ns[-1]
    out.append(n if n.duration >= 0 else    dataclasses.replace(n, duration=beatDuration-n.offset))
    return out


def _id2div(divId: str) -> U[int, list]:
    return eval(divId)


@dataclasses.dataclass
class QuantizedBeat:
    divisions: division_t
    assignedSlots: List[int]
    notations: List[Notation]  # snapped events
    beatDuration: F
    beatOffset: F = F(0)

    def applyDurationRatios(self):
        _applyDurationRatio(self.notations, division=self.divisions,
                            beatOffset=self.beatOffset, beatDur=self.beatDuration)

    def group(self) -> scorestruct.DurationGroup:
        return _groupByRatio(self.notations, division=self.divisions,
                             beatOffset=self.beatOffset, beatDur=self.beatDuration)

    def __post_init__(self):
        self.applyDurationRatios()


@dataclasses.dataclass
class QuantizedMeasure:
    timesig: timesig_t
    quarterTempo: F
    beats: Opt[List[QuantizedBeat]] = None

    def __post_init__(self):
        if self.beats:
            self.check()
            # self.removeUnnecessaryAccidentals()

    def isEmpty(self) -> bool:
        if not self.beats:
            return True
        return self.beats is None or not any(beat.notations for beat in self.beats)

    def dump(self):
        print(f"Timesig: {self.timesig[0]}/{self.timesig[1]} (quarter={self.quarterTempo})")
        if not self.beats:
            print("  EMPTY")
        else:
            for beat in self.beats:
                print(f"  {float(beat.beatOffset):.3f}, division: {beat.divisions}")
                for ev in beat.notations:
                    print("  ", ev)
            assert not self.isEmpty()

    def notations(self, merge=False) -> List[Notation]:
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
        if merge:
            notations = scorestruct.mergeNotationsIfPossible(notations)
        return notations

    def groups(self) -> List[scorestruct.DurationGroup]:
        if not self.beats:
            return []
        return [beat.group().mergeNotations() for beat in self.beats]
        # return [beat.group() for beat in self.beats]

    def beatDurations(self) -> List[F]:
        """
        Returns a list with the durations (in quarterNotes) of the beats
        in this measure
        """
        if not self.beats:
            return []
        return [beat.beatDuration for beat in self.beats]

    def beatDivisions(self) -> List:
        if not self.beats:
            return []
        return [beat.divisions for beat in self.beats]

    def removeUnnecessaryAccidentals(self):
        removeUnnecessaryAccidentals(self.notations())

    def check(self):
        if not self.beats:
            return
        # check that the measure is filled
        quarterDur = util.measureQuarterDuration(self.timesig)
        total = F(0)
        for beat in self.beats:
            assert all(ev.duration is not None and ev.duration >= 0
                       for ev in beat.notations)
            total += sum(ev.duration for ev in beat.notations)
        assert total == quarterDur, f"{total=}, {quarterDur=}"


def removeUnnecessaryAccidentals(ns: List[Notation]) -> None:
    seen = {}
    for n in ns:
        if n.rest:
            continue
        for pitch in n.pitches:
            notatedPitch = notated_pitch(pitch)
            lastSeen = seen.get(notatedPitch.diatonic_step)
            if lastSeen is None:
                # make accidental necessary only if not diatonic step
                if notatedPitch.diatonic_alteration == 0:
                    # TODO: each note in a chord should have individual attributes (hidden, notehead, etc)
                    n.accidentalHidden = True
            elif lastSeen == notatedPitch.accidental_name:
                n.accidentalHidden = True
            seen[notatedPitch.diatonic_step] = notatedPitch.accidental_name


def glissMarkTiedNotesAsHidden(qpart: QuantizedPart) -> None:
    it = qpart.iterNotations()
    for loc in it:
        n = loc.notation
        if n.gliss and not n.tiedPrev and n.tiedNext:
            # this starts a glissando and has tied notes after
            for loc2 in it:
                if loc2.notation.tiedPrev:
                    loc2.notation.noteheadHidden = True
                if not loc2.notation.tiedNext:
                    break


def _removeInvalidGracenotes(qpart: QuantizedPart) -> None:
    """
    Remove invalid grace notes in this measure, in place
    """
    trash = []
    def transferAttributes(source: Notation, dest: Notation) -> None:
        dest.tiedPrev = source.tiedPrev
        dest.gliss = source.gliss
        dest.articulation = source.articulation
        if source.annotations:
            if dest.annotations is None:
                dest.annotations = source.annotations
            else:
                dest.annotations.extend(source.annotations)

    for loc0, loc1 in iterlib.pairwise(qpart.iterNotations()):
        n0 = loc0[2]
        n1 = loc1[2]
        if n0.tiedNext and n0.pitches == n1.pitches:
            if n0.isGraceNote():
                transferAttributes(n0, n1)
                trash.append(loc0)
            elif n1.isGraceNote():
                trash.append(loc1)
    for loc in trash:
        measurenum, beat, n = loc
        beat.notations.remove(n)


def bestSubdivision(eventsInBeat: List[Notation],
                    quarterTempo: number_t,
                    beatDuration: number_t = F(1),
                    beatOffset: F = F(0),
                    profile:U[QuantizationProfile, str]="default"
                    ) -> QuantizedBeat:
    """
    Args:
        eventsInBeat: a list of Notations, where the offset is relative to the start of the
            measure and should not extend outside the beat.
            The duration can be left undefined (as -1) if the event to which this attack
            refers extends to the next attack or to the end of the beat.
        beatDuration: duration of the beat, in quarter notes (1=quarter, 0.5=eigth note)
        beatOffset: offset (start time) of this beat in relation to the beginning of the meaure
        quarterTempo: the tempo corresponding to a quarter note
        profile:
            the subdivision profile used (see beatDivisions). None = default profile

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
        >>> bestSubdivision(attacks, 60, 1)
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
    beatDuration = asF(beatDuration)

    if not isBeatFilled(eventsInBeat, beatDuration=beatDuration):
        eventsInBeat = fillDuration(eventsInBeat, duration=beatDuration, offset=beatOffset)

    assert all(ev.duration is not None for ev in eventsInBeat)
    assert all(0 <= ev.duration <= beatDuration and
               beatOffset <= ev.offset <= ev.end <= beatOffset+beatDuration
               for ev in eventsInBeat)
    assert sum(ev.duration for ev in eventsInBeat) == beatDuration, _eventsShow(eventsInBeat)

    if isinstance(profile, str):
        profile = presetQuantizationProfiles[profile]

    tempo = asF(quarterTempo) / beatDuration
    possibleDivisions = beatDivisions(tempo, profile=profile)
    beatGrids = [generateBeatGrid(beatDuration=beatDuration, division=div, offset=beatOffset)
                 for div in possibleDivisions]
    possibleGrids = {str(div):beatGrids[i] for i, div in enumerate(possibleDivisions)}
    id2div = {str(div): div for div in possibleDivisions}

    rows = []

    for divId, grid in possibleGrids.items():
        assignedSlots, snappedEvents = snapEventsToGrid(eventsInBeat, grid=grid)
        gridError = evaluateQuantization(profile=profile,
                                         eventsInBeat=eventsInBeat,
                                         snappedEvents=snappedEvents,
                                         beatDuration=beatDuration)
        div = id2div[divId]
        divPenalty = _divisionPenalty(div, profile)

        totalError = mathlib.weighted_euclidian_distance([(gridError, profile.gridErrorWeight),
                                                         (divPenalty, profile.divisionErrorWeight)])
        rows.append((totalError, divId, snappedEvents, assignedSlots))

    error, divisionId, snappedEvents, assignedSlots = min(rows, key=lambda row: row[0])
    division = _id2div(divisionId)
    if isinstance(division, int):
        division = [division]
    assert isinstance(division, list)
    beatNotations = []
    for ev in snappedEvents:
        if ev.duration == 0:
            beatNotations.append(ev)
            continue
        eventParts = breakIrregularDuration(ev, beatDivision=division,
                                            beatDuration=beatDuration,
                                            beatOffset=beatOffset)
        if not eventParts:
            beatNotations.append(ev)
        else:
            assert sum(ev.duration for ev in eventParts) == ev.duration
            beatNotations.extend(eventParts)
    assert sum(ev.duration for ev in beatNotations) == sum(ev.duration for ev in snappedEvents)
    return QuantizedBeat(division, assignedSlots=assignedSlots,
                         notations=beatNotations, beatDuration=beatDuration,
                         beatOffset=beatOffset)


def fillMeasure(eventsInMeasure: List[Notation],
                timesig: timesig_t,
                quarterTempo=F(60)
                ) -> List[Notation]:
    """
    Helper function, ensures that the measure is filled

    Args:
        eventsInMeasure: this events should fit within the measure but don't necessarily
            fill the measure
        timesig: the time-signature of the measure
        quarterTempo: the tempo corresponding to a quarter note

    Returns:
        a list of Notations which fill the measure without any gaps

    """
    measureDuration = util.measureTimeDuration(timesig, quarterTempo)
    assert all(0<=ev.offset and ev.end<=measureDuration for ev in eventsInMeasure)
    return fillDuration(eventsInMeasure, measureDuration)


def splitNotationAtOffsets(n: Notation, offsets: List[F]) -> List[Notation]:
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

    parts: List[Notation] = [dataclasses.replace(n, offset=start, duration=end-start)
             for start, end in intervals]
    if not n.rest:
        parts[0].tiedPrev = n.tiedPrev
        parts[-1].tiedNext = n.tiedNext
        _tieNotationParts(parts)

    assert sum(part.duration for part in parts) == n.duration
    assert parts[0].offset == n.offset
    assert parts[-1].end == n.end

    return parts


_regularSlotNumbers = {1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32}


def notationNeedsBreak(n: Notation, beatDuration:F, beatDivision: division_t, beatOffset=F(0)) -> bool:
    assert n.duration is not None and n.duration >= 0
    assert isinstance(beatDivision, list), f"Expected object of type {list}, got {type(beatDivision).__name__}"
    assert isinstance(beatDuration, F), f"Expected object of type {F}, got {type(beatDuration).__name__}"
    assert isinstance(beatOffset, F), f"Expected object of type {F}, got {type(beatOffset).__name__}"

    if n.end > beatOffset + beatDuration:
        raise ValueError(f"n extends over the beat. "
                         f"n={n.offset} - {n.end}, beat={beatOffset} - {beatOffset+beatDuration}")

    if n.duration == 0:
        return False

    if len(beatDivision) == 1:
        div = beatDivision[0]
        slotdur = beatDuration / div
        nslots = n.duration / slotdur
        if nslots.denominator != 1:
            raise ValueError(f"n does is not quantized with given division.\n  n={n}\n  div={beatDivision}")
        assert isinstance(nslots, F), f"Expected nslots of type {F}, got {type(nslots).__name__} (nslots={nslots})"
        return nslots.numerator not in _regularSlotNumbers
    else:
        # check if n extends over subdivision
        dt = beatDuration / len(beatDivision)
        ticks = mathlib.fraction_range(beatOffset, beatOffset + beatDuration, dt)
        for tick in ticks:
            if n.offset < tick < n.end:
                return True
        # n is confined to one subdivision of the beat, find which
        now = beatOffset
        for i, div in enumerate(beatDivision):
            if now <= n.offset < now+dt:
                # found!
                return notationNeedsBreak(n, dt, div, beatOffset=now)


def _tieNotationParts(parts: List[Notation]) -> None:
    """ Tie these notations in place """
    for part in parts[:-1]:
        part.tiedNext = True
    for part in parts[1:]:
        part.tiedPrev = True


def _splitIrregularDuration(n: Notation, slotIndex: int, slotDur: F) -> List[Notation]:
    """
    Args:
        n: the Notation to split
        slotIndex: which slot is n assigned to within the beat/subbeat
        slotDur: which is the quarterNote duration of slotDur

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
    assert isinstance(n, Notation), f"Expected type Notation, got {type(n).__name__}={n}"
    assert isinstance(slotDur, F), f"Expected type F, got {type(slotDur).__name__}={slotDur}"
    assert n.duration > 0

    numSlots = int(n.duration / slotDur)
    if numSlots > 25:
        raise ValueError("Division not supported")

    slotDivisions = _slotDivisionStrategy[numSlots]
    if slotIndex % 2 == 1 and slotDivisions[-1] % 2 == 1:
        slotDivisions = list(reversed(slotDivisions))

    offset = n.offset
    parts: List[Notation] = []
    for slots in slotDivisions:
        partDur = slotDur * slots
        parts.append(n.clone(offset=offset, duration=partDur))
        offset += partDur

    _tieNotationParts(parts)
    assert sum(part.duration for part in parts) == n.duration
    assert parts[0].offset == n.offset
    assert parts[-1].end == n.end
    return parts


def _breakIrregularDuration(n: Notation, beatDuration:F, div: int, beatOffset=F(0)
                            ) -> Opt[List[Notation]]:
    # beat is subdivided regularly
    slotDur = beatDuration/div
    nslots = n.duration/slotDur
    assert isinstance(nslots, F), f"Expected type F, got {type(nslots).__name__}={nslots}"

    if nslots.denominator != 1:
        raise ValueError(f"Duration is not quantized with given division.\n  {n=}, {div=}")

    if nslots.numerator in _regularSlotNumbers:
        return None

    slotIndex = (n.offset-beatOffset)/slotDur

    if not slotIndex.denominator == 1:
        raise ValueError(f"Offset is not quantized with given division. n={n}, div={div}")

    parts = _splitIrregularDuration(n, slotIndex, slotDur)
    assert sum(part.duration for part in parts) == n.duration
    return parts


def breakIrregularDuration(n: Notation, beatDuration:F, beatDivision: division_t, beatOffset=F(0)
                           ) -> Opt[List[Notation]]:
    """
    * a Notations should not extend over a subdivision of the beat if the
      subdivisions in question are coprimes
    * within a subdivision, a Notation should not result in an irregular multiple of the
      subdivision. Irregular multiples are all numbers which have prime factors other than
      2 or can be expressed with a dot
      Regular durations: 2, 3, 4, 6, 7 (double dotted), 8, 12, 16, 24, 32
      Irregular durations: 5, 9, 10, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31

    Args:
        n: the Notation to break
        beatDuration: the duration of the beat
        beatDivision: the division of the beat
        beatOffset: the offset of the beat

    Returns:
        None if the notations has a regular duration, or a list of tied Notations which together
        represent the original notation

    """
    assert isinstance(beatDivision, (int, list)), f"Expected type int/list, got {type(beatDivision).__name__}={beatDivision}"
    assert isinstance(beatDuration, F), f"Expected type F, got {type(beatDuration).__name__}={beatDuration}"
    assert isinstance(beatOffset, F), f"Expected type F, got {type(beatOffset).__name__}={beatOffset}"
    assert n.duration>=0

    if n.duration == 0:
        return None

    if n.end > beatOffset + beatDuration:
        raise ValueError(f"n extends over the beat. "
                         f"n={n.offset} - {n.end}, beat={beatOffset} - {beatOffset+beatDuration}")

    if isinstance(beatDivision, int):
        return _breakIrregularDuration(n, beatDuration=beatDuration,
                                      div=beatDivision, beatOffset=beatOffset)

    if len(beatDivision) == 1:
        return _breakIrregularDuration(n, beatDuration=beatDuration,
                                       div=beatDivision[0], beatOffset=beatOffset)

    # beat is not subdivided regularly. check if n extends over subdivision
    numDivisions = len(beatDivision)
    divDuration = beatDuration / numDivisions

    ticks = list(mathlib.fraction_range(beatOffset, beatOffset+beatDuration+divDuration, divDuration))

    subdivisionTimespans = list(iterlib.pairwise(ticks))
    assert len(subdivisionTimespans) == numDivisions, f"{subdivisionTimespans=}, {beatDivision=}"
    subdivisions = list(zip(subdivisionTimespans, beatDivision))
    subns = splitNotationAtOffsets(n, ticks)
    allparts: List[Notation] = []
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


def isMeasureFilled(notations: List[Notation], timesig: timesig_t) -> bool:
    measureDuration = util.measureQuarterDuration(timesig)
    return sum(n.duration for n in notations) == measureDuration


def measureSplitNotationsAtBeats(eventsInMeasure: List[Notation],
                                 timesig: timesig_t,
                                 quarterTempo: number_t,
                                 ) -> List[Tuple[TimeSpan, List[Notation]]]:
    """
    Split the events in this measure into its individual beats
    Returns a list of tuples (time span of beat, eventsInBeat)
    Used by quantizeMeasures

    NB: we ensure that the returned events in beat completely fill the beat
    NB: the events here have a duration and offset in quarterLength, not in raw
        seconds (.tempoCorrected should be True)

    Args:
        eventsInMeasure: the events within the measure. The offset of each notation
            is relative to the start of the measure. The events should fill the measure
        timesig: the time signature of the measure
        quarterTempo: the tempo (used as hint to divide the measure in beats)

    Returns:
        a list of tuples (timeSpan, eventsInBeat)
    """
    assert misc.assert_type(eventsInMeasure, [Notation])
    assert misc.assert_type(timesig, (int, int))
    assert isinstance(quarterTempo, (int, F))

    assert isMeasureFilled(eventsInMeasure, timesig)

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
        assert all(ev.duration>=0 for ev in eventsInBeat)
        assert all(timeSpan.start <= ev.offset <= ev.end <= timeSpan.end
                   for ev in eventsInBeat)
    return list(zip(timeSpans, eventsPerBeat))


def _groupByRatio(notations: List[Notation], division:U[int, division_t],
                  beatOffset:F, beatDur:F
                  ) -> scorestruct.DurationGroup:
    if isinstance(division, int) or len(division) == 1:
        if isinstance(division, list):
            division = division[0]
        durRatio = _durationRatios[division]
        return scorestruct.DurationGroup(durRatio=durRatio, items=notations)

    assert isinstance(division, list) and len(division) >= 2
    numSubBeats = len(division)
    now = beatOffset
    dt = beatDur/numSubBeats
    durRatio = _durationRatios[numSubBeats]
    items = []
    for subdiv in division:
        subdivEnd = now+dt
        subdivNotations = [n for n in notations if now<=n.offset<subdivEnd and n.end<=subdivEnd]
        if subdiv == 1:
            items.extend(subdivNotations)
        else:
            items.append(_groupByRatio(subdivNotations, subdiv, now, dt))
        now += dt
    return scorestruct.DurationGroup(durRatio, items)


def _applyDurationRatio(notations:List[Notation], division:U[int, division_t], beatOffset:F, beatDur:F) -> None:
    """
    Applies a duration ratio to each notation, recursively. A duration ratio converts the actual
    duration of a notation to its notated value and is used to render these as tuplets later

    Args:
        notations: the notations inside the period beatOffset:beatOffset+beatDur
        division: the division of the beat/subbeat. Examples: 4, [3, 4], [2, 2, 3], etc
        beatOffset: the start of the beat
        beatDur: the duration of the beat

    Returns:

    """
    if isinstance(division, int) or len(division) == 1:
        if isinstance(division, list):
            division = division[0]
        durRatio = _durationRatios[division]
        for n in notations:
            if n.durRatios is None:
                n.durRatios = []
            n.durRatios.append(F(*durRatio))
    else:
        numSubBeats = len(division)
        now = beatOffset
        dt = beatDur / numSubBeats
        durRatio = F(*_durationRatios[numSubBeats])
        for n in notations:
            if n.durRatios is None:
                n.durRatios = []
            n.durRatios.append(durRatio)
        for subdiv in division:
            subdivEnd = now + dt
            subdivNotations = [n for n in notations if now <= n.offset < subdivEnd and n.end <= subdivEnd]
            _applyDurationRatio(subdivNotations, subdiv, now, dt)
            now += dt


def quantizeMeasure(events: List[Notation],
                    timesig: timesig_t,
                    quarterTempo: number_t,
                    profile: U[QuantizationProfile, str]="default"
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
    if isinstance(profile, str):
        profile = presetQuantizationProfiles[profile]

    beats: List[QuantizedBeat] = []

    if not isMeasureFilled(events, timesig):
        events = fillMeasure(events, timesig, quarterTempo)

    for span, eventsInBeat in measureSplitNotationsAtBeats(eventsInMeasure=events,
                                                           timesig=timesig,
                                                           quarterTempo=quarterTempo):
        beatDuration = span.end - span.start
        beatQuantisation = bestSubdivision(eventsInBeat=eventsInBeat,
                                           quarterTempo=quarterTempo,
                                           beatDuration=beatDuration,
                                           beatOffset=span.start,
                                           profile=profile)

        beats.append(beatQuantisation)

    return QuantizedMeasure(timesig=timesig, quarterTempo=quarterTempo, beats=beats)


def splitByMeasure(struct: scorestruct.ScoreStructure,
                   event: Notation,
                   ) -> List[Tuple[int, Notation]]:
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
    loc0 = struct.timeToBeat(event.offset)
    loc1 = struct.timeToBeat(event.end)
    if loc0 is None or loc1 is None:
        raise ValueError("Could not find a score location for this event")

    numMeasures = loc1.measureNum-loc0.measureNum+1
    if numMeasures == 1:
        event = event.clone(offset=loc0.beat, duration=loc1.beat-loc0.beat)
        return [(loc0.measureNum, event)]

    measuredef = struct.measuredefs[loc0.measureNum]
    dur = measuredef.numberOfBeats()-loc0.beat
    notation = event.clone(offset=loc0.beat, duration=dur, tiedNext=True)
    pairs = [(loc0.measureNum, notation)]

    # add intermediate measure, if any
    if numMeasures>2:
        for m in range(loc0.measureNum+1, loc1.measureNum):
            measuredef = struct.measuredefs[m]
            notation = event.clone(offset=F(0),
                                   duration=measuredef.numberOfBeats(),
                                   tiedPrev=True, tiedNext=True)
            pairs.append((m, notation))

    # add last notation
    if loc1.beat>0:
        notation = event.clone(offset=F(0), duration=loc1.beat, tiedPrev=True)
        pairs.append((loc1.measureNum, notation))

    sumdur = sum(struct.elapsedTime((i, n.offset), (i, n.end)) for i, n in pairs)
    assert sumdur == event.duration, f"{event=}, {sumdur=}, {pairs=}"
    return pairs


def splitEventByMeasures(struct: scorestruct.ScoreStructure,
                         event: core.Event,
                         ) -> List[Tuple[int, Notation]]:
    """
    Divide an event in multiple Notations making sure that each Notation
    fits within each measure.
    The offset each resulting notation is relative to the start of the measure,
    its duration is according to the tempo of each measure.
    The generated notations are tied together and each notation holds a reference
    to this original event

    Args:
        event: the event to split across measures
        struct: the ScoreStructure holding the measure definitions

    Returns:
        a list of pairs (measure index, notation)
        This can be used to join together all notations belonging to a specific
        measure for later quantization
    """
    assert event.offset is not None and event.dur is not None
    loc0 = struct.timeToBeat(event.offset)
    loc1 = struct.timeToBeat(event.end)
    if loc0 is None or loc1 is None:
        raise ValueError("Could not find a score location for this event")

    numMeasures = loc1.measureNum-loc0.measureNum + 1
    if numMeasures == 1:
        dur = loc1.beat - loc0.beat
        notation = Notation.fromEvent(event, offset=loc0.beat, duration=dur)
        return [(loc0.measureNum, notation)]

    measuredef = struct.measuredefs[loc0.measureNum]
    dur = measuredef.numberOfBeats() - loc0.beat
    notation = Notation.fromEvent(event, offset=loc0.beat, duration=dur, tiedNext=True)
    #notation = Notation(offset=loc0.beat, duration=dur, sourceEvent=event,
    #                    tiedNext=True)
    pairs = [(loc0.measureNum, notation)]

    # add intermediate measure, if any
    if numMeasures > 2:
        for m in range(loc0.measureNum+1, loc1.measureNum):
            measuredef = struct.measuredefs[m]
            #notation = Notation(offset=F(0), duration=measuredef.numberOfBeats(),
            #                    sourceEvent=event, tiedPrev=True, tiedNext=True)
            notation = Notation.fromEvent(event, offset=F(0),
                                          duration=measuredef.numberOfBeats(),
                                          tiedPrev=True, tiedNext=True)
            pairs.append((m, notation))

    # add last notation
    if loc1.beat > 0:
        notation = Notation.fromEvent(event, offset=F(0), duration=loc1.beat,
                                      tiedPrev=True)
        pairs.append((loc1.measureNum, notation))
    if event.isRest():
        for idx, notation in pairs:
            notation.tiedNext = False
            notation.tiedPrev = False
    sumdur = sum(struct.elapsedTime((i, n.offset), (i, n.end)) for i, n in pairs)
    assert sumdur == event.dur, f"event.dur={event.dur}, notation dur={sumdur}, {pairs=}"
    return pairs


def _removeOverlapInplace(notations: List[Notation], threshold=F(1,1000)) -> None:
    """
    Remove overlap between notations. This should be only used to
    remove small overlaps product of rounding errors.
    """
    for n0, n1 in iterlib.pairwise(notations):
        assert n0.offset <= n1.offset
        diff = n0.end-n1.offset
        if diff > 0:
            if diff > threshold:
                raise ValueError(f"Notes overlap by too much: {diff}, {n0}, {n1}")
            n0.duration = n1.offset - n0.offset


class PartLocation(NamedTuple):
    measureNum: int
    beat: QuantizedBeat
    notation: Notation


@dataclasses.dataclass
class QuantizedPart:
    struct: scorestruct.ScoreStructure
    measures: List[QuantizedMeasure]
    label: str = ""

    def flatNotations(self) -> Iter[Notation]:
        for m in self.measures:
            for n in m.notations():
                yield n

    def iterNotations(self) -> Iter[PartLocation]:
        """
        Iterates over all notations giving the location of each notation
        For each notation yields a tuple:

        (measure number, QuantizedBeat, Notation)
        """
        for i, m in enumerate(self.measures):
            for b in m.beats:
                for n in b.notations:
                    yield PartLocation(i, b, n)

    def dump(self):
        for m in self.measures:
            m.dump()


def quantizePart(struct: scorestruct.ScoreStructure,
                 eventsInPart: List[Notation],
                 fillStructure=False,
                 profile:U[str,QuantizationProfile]='default'
                 ) -> QuantizedPart:
    """
    Quantizes a sequence of non-overlapping events (a "part") to the
    score structure defined in `struct`, according to the strategies
    defined in `profile`

    Args:
        struct: the ScoreStruct to use
        eventsInPart: the events to quantize. Event within a part
            should not overlap
        fillStructure: if True and struct is not endless, the
            generated Part will have as many measures as are defined
            in the struct. Otherwise only as many measures as needed
            to hold the given events will be created
        profile: the QuantizationProfile used

    Returns:
        a list of QuantizedMeasures. To convert these to a scorestruct.Part,
        call convertQuantizedMeasuresToPart

    """
    eventsInPart = core.stackNotations(eventsInPart)
    allpairs = [splitByMeasure(struct, event) for event in eventsInPart]
    maxMeasure = max(pairs[-1][0] for pairs in allpairs)
    notationsPerMeasure: List[List[Notation]] = [[] for _ in range(maxMeasure+1)]
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
            qmeasures.append(quantizeMeasure(notations,
                                             timesig=measureDef.timesig,
                                             quarterTempo=measureDef.quarterTempo,
                                             profile=profile))
    if fillStructure:
        if struct.endless:
            raise ValueError("Cannot fill an endless ScoreStructure")
        for i in range(maxMeasure+1, struct.numMeasures()):
            measureDef = struct.getMeasureDef(i)
            qmeasure = QuantizedMeasure(timesig=measureDef.timesig,
                                        quarterTempo=measureDef.quarterTempo, beats=[])
            qmeasures.append(qmeasure)
    part = QuantizedPart(struct, qmeasures)
    _removeInvalidGracenotes(part)
    glissMarkTiedNotesAsHidden(part)
    return part


def bestClefForPart(part: QuantizedPart) -> str:
    # Only analyze the first n notes
    locations = iterlib.take(part.iterNotations(), 8)
    accum = 0
    numpitches = 0

    for loc in locations:
        if loc.notation.rest:
            continue
        pitches = loc.notation.pitches
        accum += sum(pitches)
        numpitches += len(pitches)

    if numpitches == 0:
        # all rests
        return "treble"

    avgPitch = accum/numpitches

    if avgPitch < 55:
        return "bass"
    elif avgPitch < 80:
        return "treble"
    else:
        return "treble8"


def tests():
    evs = [Notation(F(t)) for t in [0.25, 0.5, 1.5]]
    for dur in [1, 2, 3, 4]:
        filled = fillDuration([Notation(F(0.5))], F(dur))
        sumdur = sum(ev.duration for ev in filled)
        assert  sumdur == F(dur), f"Expected dur: {dur}, got {sumdur} (filled events={filled})"
    assert sum(ev.duration for ev in fillMeasure(evs, (3, 4), F(60))) == 3
    assert sum(ev.duration for ev in fillMeasure(evs, (5, 8), F(60))) == F(5, 2)


