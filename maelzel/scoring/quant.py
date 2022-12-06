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

import copy
from dataclasses import dataclass, field as _field, fields as _fields

from .common import *
from . import core
from . import definitions
from . import util
from . import quantdata
from . import quantutils
from . import enharmonics
from . import attachment

from .notation import Notation, makeRest, SnappedNotation
from .durationgroup import DurationGroup, asDurationGroupTree
from maelzel.scorestruct import ScoreStruct, measureBeatOffsets

from emlib import iterlib
from emlib import misc
from emlib.misc import Result
from emlib import mathlib
from pitchtools import notated_pitch

from typing import TYPE_CHECKING, NamedTuple
if TYPE_CHECKING:
    from numbers import Rational
    from typing import Union, Iterator, Sequence
    import maelzel.core
    number_t = Union[int, float, Rational]


__all__ = (
    'quantize',
    'QuantizationProfile',
    'makeQuantizationProfile',
    'QuantizedScore',
    'QuantizedPart',
    'QuantizedMeasure',
    'QuantizedBeat',
    'quantizeMeasure',
    'quantizePart',
    'splitNotationAtOffsets',
    'splitNotationByMeasure',
    'PartLocation',
)

_INDENT = "  "


_regularDurations = {1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32}


def _factory(obj) -> _field:
    return _field(default_factory=lambda: copy.copy(obj))


def _presetField(key) -> _field:
    return _factory(quantdata.presets[key])


class QuantError(Exception):
    pass


@dataclass
class QuantizationProfile:
    """
    A QuantizationProfile is used to configure quantization

    To construct a QuantiztationProfile based on a preset, use
    :meth:`QuantizationProfile.fromPreset`

    Most important parameters:

    - nestedTuplets: if True, allow nested tuplets. NB: musicxml rendered
      via MuseScore does not support nested tuplets
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
    - tupletMaxDur: the max quarternote duration for a merged subdivision

    Lower level parameters to calculate division complexity:

    - numNestedTupletsPenaltyWeight: how

    """
    nestedTuplets: bool = False
    """Are nested tuplets allowed?"""

    gridErrorWeight: float = 1
    """Weight of the overall effect of offset and duration errors when fitting events to a grid. 
    A higher weight minimizes offset and duration errors at the cost of more complex divisions"""

    gridErrorExp: float = 0.85
    """An exponent applied to the grid error. Since this values is always between 0 and 1,
    an exponent less than 1 makes the effects of grid errors grow faster"""

    divisionErrorWeight: float = 0.002
    """Weight of the division complexity"""

    maxDivPenalty: float = 0.1
    """A max. division penalty, will discard any divisions which have a penalty
    higher than this value. This can be used to further customize the quantization
    process"""

    maxGridDensity: int = 0
    """
    If given (higher than 0) it discards any division of the beat with a higher number
    of slots than this value. For example, a division of (3, 4, 4) has a density of 12,
    since the highest subdivision, 4, applied to the entire beat would result in 
    12 notes per beat
    """

    rhythmComplexityWeight: float = 0.001
    """Weight of the actual quantized rhythm. This includes evaluating synchopes, ties, etc."""

    rhythmComplexityNotesAcrossSubdivisionWeight = 0.1
    """
    When calculating rhythm complexity this weight is applied to the penalty of notes extending
    over subdivisions of the beat (inner-beat syncopes)
    """

    rhythmComplexityIrregularDurationsWeight = 0.9
    """
    When calculating rhythm complexity this weight is applied to the penalty of notes whose
    duration is irregular (durations of 5 or 9 units, which need ties to be represented)
    """

    offsetErrorWeight: float = 1.0
    """Weight of the offset between original start and snapped start"""

    restOffsetErrorWeight: float = 0.5
    """Similar to offsetErrorWeight but for rests"""

    durationErrorWeight: float = 0.2
    """Weight of the difference in duration resulting from quantization"""

    gracenoteDuration: F = F(1, 32)
    """A duration to assume for grace notes"""

    gracenoteErrorWeight: float = 0

    possibleDivisionsByTempo: dict[int, list] = _factory(quantdata.presets['high']['possibleDivisionsByTempo'])
    """A mapping of possible divisions ordered by max. tempo"""

    divisionPenaltyMap: dict[int, float] = _factory(quantdata.presets['high']['divisionPenaltyMap'])
    """A mapping of the penalty of each division"""

    divisionCardinalityPenaltyMap: dict[int, float] = _factory({1:0.0, 2:0.1, 3:0.4})
    """Penalty applied when different divisions are used within a beat (e.g 4 where one 8 is a 3-plet and the other a 5-plet)"""

    numNestedTupletsPenalty: list[float] = _factory([0., 0.1, 0.4, 0.5, 0.8, 0.8])
    """Penalty applied to nested levels by level"""

    complexNestedTupletsFactor: float = 1.5
    """For certain combinations of nested tuplets an extra complexity factor can be applied.
    If this factor is 1.0, then no extra penalty is calculated. Any number above 1 will
    penalize complex nested tuplets (prefer (5, 5, 5) over (3, 3, 3, 3, 3)).
    """

    numSubdivsPenaltyMap: dict[int, float] = _factory({1: 0.0, 2: 0.0, 3: 0.0, 4:0., 5:0., 6:0., 7:0.})
    """Penalty applied to number of subdivisions, by number of subdivision"""

    divisionPenaltyWeight: float = 1.0
    """Weight of division penalty"""

    cardinalityPenaltyWeight: float = 0.1
    """Weight of cardinality"""

    numNestedTupletsPenaltyWeight: float = 1.0
    """Weight of sublevel penalty"""

    numSubdivisionsPenaltyWeight: float = 0.2

    minBeatFractionAcrossBeats: F = F(1, 8)
    """How long can a synchopation be, in terms of the length of the beat"""

    minSymbolicDurationAcrossBeat: F = F(1, 3)

    mergedTupletsMaxDuration: F = F(2)
    """How long can a tuplet over the beat be"""

    mergeTupletsOfDifferentDuration: bool = False
    """Allow merging tuplets which have different total durations?"""

    mergeNestedTupletsAcrossBeats: bool = False
    """Allow merging nested tuplets across the beat"""

    allowedTupletsAcrossBeat: tuple[int, ...] = (1, 2, 3, 4, 5, 8)
    """Which tuplets are allowed to cross the beat"""

    maxPenalty: float = 1.0
    """A max. penalty when quantizing a beat, to limit the search space"""

    debug: bool = False
    """Turns on debugging"""

    debugMaxDivisions: int = 20
    """Max number of quantization possibilities to display when debugging"""

    blacklist: set[division_t] = _field(default_factory=set)

    name: str = ''

    _cachedDivisionsByTempo: dict[tuple[number_t, bool], list[division_t]] = _field(default_factory=dict)
    _cachedDivisionPenalty: dict[tuple[int, ...], tuple[float, str]] = _field(default_factory=dict)

    def __post_init__(self):
        self._cachedDivisionsByTempo = {}
        self._cachedDivisionPenalty = {}

    def possibleBeatDivisionsByTempo(self, tempo: number_t) -> list[division_t]:
        """
        The possible divisions of the pulse for the given tempo

        Args:
            tempo: the tempo to calculate divisions for. A profile can define different
                divisions according to different tempi (simpler divisions if the tempo
                is fast, more complex if the tempo is slow).

        Returns:
            a list of possible divisions for the given tempo.

        """
        #if divs := self._cachedDivisionsByTempo.get((tempo, self.nestedTuplets)):
        #    return divs
        divsByTempo = self.possibleDivisionsByTempo
        divs = None
        for maxTempo, possibleDivs in divsByTempo.items():
            if tempo < maxTempo:
                divs = possibleDivs
                break
        if not divs:
            raise ValueError("No divisions for the given tempo")
        if not self.nestedTuplets:
            divs = [div for div in divs
                    if not _isNestedTupletDivision(div)]
        divs.sort(key=lambda div: len(div))
        self._cachedDivisionsByTempo[(tempo, self.nestedTuplets)] = divs
        return divs

    def divisionPenalty(self, division: division_t
                        ) -> tuple[float, str]:
        """
        A penalty based on the complexity of the division of the pulse alone

        Args:
            division: the division to rate

        Returns:
            a tuple (penalty: float, debuginfo: str), where the penalty is an
            arbitrary number (lower=simpler division, higher=more complex) and
            debuginfo can be used to query how this penalty was calculated
            (debuginfo will only be filled if .debug is True)

        """
        if (cached := self._cachedDivisionPenalty.get(division)) is not None:
            penalty, info = cached
        else:
            division = tuple(sorted(division))
            if (cached := self._cachedDivisionPenalty.get(division)) is not None:
                penalty, info = cached
            else:
                penalty, info = _divisionPenalty(division=division, profile=self,
                                                 maxPenalty=self.maxPenalty, debug=self.debug)
                self._cachedDivisionPenalty[division] = (penalty, info)
        return penalty, info

    @staticmethod
    def fromPreset(preset: str,
                   nestedTuplets: bool = None,
                   blacklist: list[division_t] = None,
                   **kws
                   ) -> QuantizationProfile:
        """
        Create a QuantizationProfile from a preset

        Args:
            preset: the complexity of the quantization as a preset, one of 'low',
                'medium', 'high' or 'highest'. This preset determines all other
                settings.
            nestedTuplets: if given, overrides the preset's own nestedTzplets setting. If True,
                nested tuplets are allowed. NB: a tuplet within a binary subdivision of
                the beat is not considered as nested (for example, the division (1, 3), indicating
                a quarter divided into an eighth-note and three 16ths as a 3:2 tuplet.
            blacklist:
            **kws:

        Returns:
            the QuantizationProfile
        """
        return makeQuantizationProfile(complexity=preset,
                                       nestedTuplets=nestedTuplets,
                                       blacklist=blacklist,
                                       **kws)


def _isNestedTupletDivision(div: division_t) -> bool:
    if isinstance(div, int):
        # A shortcut division, like 3 or 5
        return False
    return not mathlib.ispowerof2(len(div)) and any(not mathlib.ispowerof2(subdiv) for subdiv in div)


defaultQuantizationProfile = QuantizationProfile()


def makeQuantizationProfile(complexity='high',
                            nestedTuplets: bool = None,
                            blacklist: list[division_t] = None,
                            **kws) -> QuantizationProfile:
    """
    Create a QuantizationProfile from a preset

    Args:
        complexity: complexity presets, one of 'low', 'medium', 'high', 'highest'
            (see ``maelzel.scoring.quantdata.presets``)
        nestedTuplets: if True, allow nested tuplets.
        blacklist: if given, a list of divisions to exclude
        kws: any keywords passed to :class:`QuantizationProfile`

    Returns:
        the quantization preset

    """
    def cascade(key: str, kws: dict, preset: dict, default: QuantizationProfile):
        return misc.firstval(kws.pop(key, None), preset.get(key), getattr(default, key))

    if complexity not in quantdata.presets:
        raise ValueError(f"complexity preset {complexity} unknown. Possible values: {quantdata.presets.keys()}")
    preset = quantdata.presets[complexity]
    keys = [field.name for field in _fields(defaultQuantizationProfile)]
    for key in keys:
        value = cascade(key, kws, preset, defaultQuantizationProfile)
        kws[key] = value
    if nestedTuplets is not None:
        kws['nestedTuplets'] = nestedTuplets
    if not kws.get('name'):
        kws['name'] = complexity
    out = QuantizationProfile(**kws)
    if blacklist:
        blacklistset = set(blacklist)
        for maxtempo, divisions  in out.possibleDivisionsByTempo.items():
            divisions = [div for div in divisions if div not in blacklistset]
            out.possibleDivisionsByTempo[maxtempo] = divisions
        out.blacklist = blacklistset
    return out


def _divisionPenalty(division: division_t,
                     profile: QuantizationProfile,
                     nestingLevel=0,
                     maxPenalty=0.7,
                     debug=False
                     ) -> tuple[float, str]:
    """
    Evaluate the given division.

    The lower the returned value, the simpler this division is. All things
    being equal, a simpler division should be preferred.

    Args:
        division: division of the beat/subbeat
        nestingLevel: since this is a recursive structure, the nestingLevel
            holds the level of nesting of the division we are analyzing
        profile: the quantization preset to use

    Returns:
        the penalty associated with this division, based on the division
        only (not on how the division fits the notes in the beat).

    """
    assert isinstance(division, int) or (
                isinstance(division, tuple) and all(isinstance(x, int) for x in division)), f"{division=}"

    if isinstance(division, int):
        divPenalty = profile.divisionPenaltyMap.get(division, maxPenalty)
        numSubdivsPenalty = profile.numSubdivsPenaltyMap[1]
        cardinality = 1
    else:
        divPenalty = sum(_divisionPenalty(subdiv, profile, nestingLevel+1, maxPenalty=maxPenalty)[0]
                         for subdiv in division)  # / len(division)
        numSubdivsPenalty = profile.numSubdivsPenaltyMap.get(len(division), maxPenalty)
        cardinality = max(1, _divisionCardinality(division, excludeBinary=True))

    cardinalityPenalty = profile.divisionCardinalityPenaltyMap.get(cardinality, maxPenalty)
    # We only calculate level penalty on the outmost level
    levelPenalty = profile.numNestedTupletsPenalty[_divisionDepth(division)] if nestingLevel == 0 else 0

    penalty = mathlib.weighted_euclidian_distance([
        (divPenalty, profile.divisionPenaltyWeight),
        (cardinalityPenalty, profile.cardinalityPenaltyWeight),
        (numSubdivsPenalty, profile.numSubdivisionsPenaltyWeight),
        (levelPenalty, profile.numNestedTupletsPenaltyWeight),

    ])

    if nestingLevel == 0 and isinstance(division, tuple):
        l = len(division)
        if l == 5 or l == 7:
            numComplexSubdivs = sum(subdiv in (3, 5, 7) or subdiv > 8
                                    for subdiv in division)
            penalty *= profile.complexNestedTupletsFactor ** numComplexSubdivs
        elif l == 6:
            numComplexSubdivs = sum(subdiv in (5, 7) or subdiv > 8
                                    for subdiv in division)
            penalty *= profile.complexNestedTupletsFactor ** numComplexSubdivs

    if debug and nestingLevel == 0:
        info = f"{divPenalty=:.3g}, {cardinalityPenalty=:.3g}, {numSubdivsPenalty=:.3g}, {levelPenalty=:.3g}"
    else:
        info = ''
    return min(penalty, 1), info


def _divisionCardinality(division, excludeBinary=False):
    # TODO: make general form for deeply nested tuplets
    if isinstance(division, int):
        return 1

    allfactors = quantutils.primeFactors(len(division), excludeBinary=excludeBinary)
    for subdiv in division:
        allfactors.update(quantutils.primeFactors(subdiv, excludeBinary=excludeBinary))
    return len(allfactors)


def _divisionDepth(division):
    # TODO: make general form for deeply nested tuplets
    if isinstance(division, int):
        return 1
    if mathlib.ispowerof2(len(division)):
        return 1
    if all(mathlib.ispowerof2(subdiv) for subdiv in division):
        return 1
    return 2


def _fitEventsToGridNearest(events: list[Notation], grid: list[F]) -> list[int]:
    return [misc.nearest_index(event.offset, grid) for event in events]


def _fitEventsToGrid(events: list[Notation], grid: list[F]) -> list[int]:
    minidx = 1
    maxidx = len(grid)
    if maxidx == 1 and events[0].offset == 0:
        return [0]
    offset1 = grid[0]
    out = []
    for n in events:
        offset = n.offset
        for idx in range(minidx, maxidx):
            offset2 = grid[idx]
            if offset1 <= offset <= offset2:
                if offset - offset1 < offset2 - offset:
                    out.append(idx - 1)
                else:
                    out.append(idx)
                minidx = idx
                break
            offset1 = offset2
    return out


def snapEventsToGrid(notations: list[Notation], grid: list[F],
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
    # assignedSlots = _fitEventsToGrid(events=notations, grid=grid)
    snappedEvents = []
    for idx in range(len(notations)-1):
        n = notations[idx]
        slot0 = assignedSlots[idx]
        offset0 = grid[slot0]
        # is it the last slot (as grace note?)
        if slot0 == len(grid) - 1:
            snappedEvents.append(SnappedNotation(n, offset0, F(0)))
        else:
            offset1 = grid[assignedSlots[idx+1]]
            snappedEvents.append(SnappedNotation(n, offset0, offset1-offset0))

    lastOffset = grid[assignedSlots[-1]]
    dur = beatDuration - lastOffset
    last = SnappedNotation(notations[-1], lastOffset, duration=dur)
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
    assert all(n.offset-offset <= duration for n in notations), \
        f"Events start after duration to fill ({duration=}): {_eventsShow(notations)}"
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


def _evalGridError(profile: QuantizationProfile,
                   snappedEvents: list[SnappedNotation],
                   beatDuration:F) -> float:
    """
    Evaluate the error regarding the deviation of the snapped events from the original offset/duration

    Given a list of events in a beat and these events snapped to a given subdivision of
    the beat, evaluate how good this snapping is in representing the original events.
    This is used to find the best subdivision of a beat.

    Args:
        profile: the quantization preset to use
        snappedEvents: the events after being snapped to a given grid
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
    quantizationError: float = 0
    quantizationInfo: str = ''

    def __post_init__(self):
        self.applyDurationRatios()

    def applyDurationRatios(self):
        _applyDurationRatio(self.notations, division=self.divisions,
                            beatOffset=self.beatOffset, beatDur=self.beatDuration)

    def group(self) -> DurationGroup:
        return _groupByRatio(self.notations, division=self.divisions,
                             beatOffset=self.beatOffset, beatDur=self.beatDuration)

    def __hash__(self):
        notationHashes = [hash(n) for n in self.notations]
        data = [self.divisions, self.beatDuration, self.beatOffset]
        data.extend(notationHashes)
        return hash(tuple(data))


@dataclass
class QuantizedMeasure:
    """
    A QuantizedMeasure holds a list of QuantizedBeats
    """
    timesig: timesig_t
    quarterTempo: F
    beats: list[QuantizedBeat] | None = None
    profile: QuantizationProfile | None = None

    def __post_init__(self):
        if self.beats:
            self.check()

    def __repr__(self):
        return f'QuantizedMeasure(timesig={self.timesig}, quarterTempo={self.quarterTempo}, ' \
               f'beats={self.beats}, profile={self.profile.name})'

    def __hash__(self):
        return hash((self.timesig, self.quarterTempo) + tuple(hash(b) for b in self.beats))

    def beatOffsets(self) -> list[F]:
        return [beat.beatOffset for beat in self.beats]

    def isEmpty(self) -> bool:
        if not self.beats:
            return True
        for beat in self.beats:
            if beat.notations and any(not n.isRest or n.spanners for n in beat.notations):
                return False
        return True

    def dump(self, indents=0):
        ind = _INDENT * indents
        print(f"{ind}Timesig: {self.timesig[0]}/{self.timesig[1]} "
              f"(quarter={self.quarterTempo})")
        if self.isEmpty():
            print(f"{ind}EMPTY")
        else:
            for group in self.groupTree():
                ind = _INDENT * (indents+1)
                print(f"{ind}Ratio {group.durRatio}")
                for n in group.items:
                    ind = _INDENT * (indents+2)
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
        groups = [beat.group().mergedNotations() for beat in self.beats]

        def removeUnnecessarySubgroupsInplace(group: DurationGroup) -> None:
            items = []
            for item in group.items:
                if isinstance(item, DurationGroup) and len(item.items) == 1:
                    item = item.items[0]
                items.append(item)
            group.items = items

        for group in groups:
            removeUnnecessarySubgroupsInplace(group)

        return groups

    def groupTree(self) -> list[DurationGroup]:
        """
        Returnes a list of DurationGroups representing the items in this measure

        The difference with ``beatGroups()`` is that this method will merge
        notations across beats (for example, when there is a synchopation)
        """
        groups = self.beatGroups()
        assert self.profile, f"Cannot create groupTree without a QuantizationProfile"
        root = asDurationGroupTree(groups)
        root = _mergeSiblings(root, profile=self.profile, beatOffsets=self.beatOffsets())
        #if root.durRatio == (1, 1):
        #    return root.items
        return [root]

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


def _removeUnnecessaryDurationRatios(n: Notation) -> None:
    if not n.durRatios:
        return
    for r in reversed(n.durRatios.copy()):
        if r != F(1):
            break
        n.durRatios.pop()


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
            n.accidentalTraits.hidden = True
            continue
        for pitch in n.pitches:
            notatedPitch = notated_pitch(pitch)
            lastSeen = seen.get(notatedPitch.diatonic_name)
            if lastSeen is None:
                # make accidental necessary only if not diatonic step
                if notatedPitch.diatonic_alteration == 0:
                    # TODO: each note in a chord should have individual attributes (hidden, notehead, etc)
                    n.accidentalTraits.hidden = True
                    # n.accidentalHidden = True
            elif lastSeen == notatedPitch.accidental_name:
                n.accidentalTraits.hidden = True
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
        slotsAtSubdivs = [0] + list(iterlib.partialsum(division[:-1]))
        numNotesAcrossSubdivisions = 0
        lastslot = sum(iterlib.flatten(division))
        for slotStart, slotEnd in iterlib.pairwise(assignedSlots + [lastslot]):
            if slotStart not in slotsAtSubdivs or slotEnd not in slotsAtSubdivs:
                numNotesAcrossSubdivisions += 1
        numIrregularNotes = sum(not isRegularDuration(dur=n.duration, beatDur=beatDur)
                                for n in snappedEvents)
        numTies = numIrregularNotes

    penalty = mathlib.weighted_euclidian_distance([
        (numNotesAcrossSubdivisions/len(snappedEvents), profile.rhythmComplexityNotesAcrossSubdivisionWeight),
        (numTies/len(snappedEvents), profile.rhythmComplexityIrregularDurationsWeight)
    ])
    if profile.debug:
        debugstr = f'numNotesAcrossSubdics={numNotesAcrossSubdivisions}, {numTies=}'
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
            eventsInBeat = eventsInBeat[:-1]
            eventsInBeat[-1].duration += last.duration

    if not isBeatFilled(eventsInBeat, beatDuration=beatDuration):
        eventsInBeat = _fillDuration(eventsInBeat, duration=beatDuration, offset=beatOffset)

    if not sum(n.duration for n in eventsInBeat) == beatDuration:
        print(_eventsShow(eventsInBeat))
        raise ValueError("Events in beat do not sum up to the beat duration")

    assert all(0 <= ev.duration <= beatDuration and
               beatOffset <= ev.offset <= ev.end <= beatOffset+beatDuration
               for ev in eventsInBeat)

    # If all rests, bypass quantization
    if all(n.isRest for n in eventsInBeat):
        if len(eventsInBeat) == 1 and eventsInBeat[0].offset == beatOffset:
            return QuantizedBeat((1, ), assignedSlots=[0], notations=eventsInBeat,
                                 beatDuration=beatDuration, beatOffset=beatOffset)

    if len(eventsInBeat) == 1 and eventsInBeat[0] == beatOffset:
        return QuantizedBeat((1, ), assignedSlots=[0], notations=eventsInBeat,
                             beatDuration=beatDuration, beatOffset=beatOffset)

    tempo = asF(quarterTempo) / beatDuration
    possibleDivisions = profile.possibleBeatDivisionsByTempo(tempo)

    rows = []
    seen = set()
    events0 = [ev.clone(offset=ev.offset - beatOffset) for ev in eventsInBeat]
    minError = 999.
    skipped = 0

    for div in possibleDivisions:
        if div in profile.blacklist:
            continue
        divPenalty, divPenaltyInfo = profile.divisionPenalty(div)

        if profile.divisionErrorWeight > 0 and divPenalty > profile.maxDivPenalty:
            skipped += 1
            continue

        if profile.maxGridDensity and quantutils.divisionDensity(div) > profile.maxGridDensity:
            continue

        grid0 = quantutils.divisionGrid0(beatDuration=beatDuration, division=div)
        assignedSlots, snappedEvents = snapEventsToGrid(events0, grid=grid0)
        simplifiedDiv = quantutils.simplifyDivision(div, assignedSlots)
        if simplifiedDiv in seen or simplifiedDiv in profile.blacklist:
            continue
        if simplifiedDiv != div:
            # TODO: optimize the re-snapping to avoid calling  snapeventstogrid again
            div = simplifiedDiv
            newgrid = quantutils.divisionGrid0(beatDuration=beatDuration, division=simplifiedDiv)
            # assignedSlots = quantutils.resnap(assignedSlots, grid0, newgrid)
            assignedSlots, snappedEvents = snapEventsToGrid(events0, grid=newgrid)

        gridError = _evalGridError(profile=profile,
                                   snappedEvents=snappedEvents,
                                   beatDuration=beatDuration)


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

        seen.add(div)

        if totalError > minError and not profile.debug:
            continue

        debuginfo = ''
        if profile.debug:
            debuginfo = f"{gridError=:.3g}, {rhythmComplexity=:.3g} ({rhythmInfo}), " \
                        f"{divPenalty=:.3g} ({divPenalty*profile.divisionErrorWeight:.4g}, {divPenaltyInfo})"
        rows.append((totalError, div, snappedEvents, assignedSlots, debuginfo))

        if totalError == 0:
            break

        minError = totalError

    # first sort by div length, then by error
    # Like this we make sure that (7,) is better than (7, 1) for the cases where the
    # assigned slots are actually the same
    rows.sort(key=lambda r: len(r[1]))

    if profile.debug:
        print(f"Beat: {beatOffset} - {beatOffset + beatDuration} (dur: {beatDuration})")
        print(f"Skipped {skipped} divisions with a div. penalty > {profile.maxDivPenalty}")
        rows.sort(key=lambda row: row[0])
        maxrows = min(profile.debugMaxDivisions, len(rows))
        print(f"Best {maxrows} divisions: ")
        table = [(f"{r[0]:.5g}",) + r[1:] for r in rows[:maxrows]]
        misc.print_table(table, headers="error div snapped slots info".split(), floatfmt='.4f', showindex=False)

    error, div, snappedEvents, assignedSlots, debuginfo = min(rows, key=lambda row: row[0])
    notations = [snapped.notation.clone(offset=snapped.offset+beatOffset, duration=snapped.duration)
                 for snapped in snappedEvents]
    assert sum(_.duration for _ in notations) == beatDuration, \
        f"{beatDuration=}, {notations=}"

    beatNotations = []
    for n in notations:
        if n.isGraceNote:
            beatNotations.append(n)
        else:
            eventParts = breakIrregularDuration(n, beatDivision=div, beatDur=beatDuration, beatOffset=beatOffset)
            if eventParts:
                beatNotations.extend(eventParts)
            elif n.duration > 0 or (n.duration == 0 and not n.isRest):
                beatNotations.append(n)

    if div != (1,) and len(beatNotations) == 1 and len(assignedSlots) == 1 and assignedSlots[0] == 0:
        div = (1,)
    elif all(n.isRest for n in beatNotations) and len(beatNotations) > 1:
        beatNotations = [makeRest(duration=beatDuration, offset=beatNotations[0].offset)]
        div = (1,)

    assert sum(ev.duration for ev in beatNotations) == sum(ev.duration for ev in snappedEvents) == beatDuration, f"{beatDuration=}, {beatNotations=}"

    return QuantizedBeat(div, assignedSlots=assignedSlots, notations=beatNotations,
                         beatDuration=beatDuration, beatOffset=beatOffset,
                         quantizationError=error, quantizationInfo=debuginfo)


def quantizeBeatTernary(eventsInBeat: list[Notation],
                        quarterTempo: F,
                        profile: QuantizationProfile,
                        beatDuration: F,
                        beatOffset: F
                        ) -> list[QuantizedBeat]:
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
    best = min(results, key=lambda result: result[0])
    return best[1]


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


def splitNotationAtOffsets(n: Notation, offsets: Sequence[Rational]) -> list[Notation]:
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
    assert all(isinstance(x0, F) and isinstance(x1, F)
               for x0, x1 in intervals)

    if len(intervals) == 1:
        return [n]

    parts: list[Notation] = [n.clone(offset=start, duration=end-start)
                             for start, end in intervals]

    # Remove superfluous dynamic/articulation
    for part in parts[1:]:
        part.dynamic = ''
        part.removeAttachments(lambda item: isinstance(item, (attachment.Articulation, attachment.Text)))
        if part.spanners:
            part.spanners.clear()

    if not n.isRest:
        _tieNotationParts(parts)
        parts[0].tiedPrev = n.tiedPrev
        parts[-1].tiedNext = n.tiedNext

    assert sum(part.duration for part in parts) == n.duration
    assert parts[0].offset == n.offset
    assert parts[-1].end == n.end
    if not n.isRest:
        assert parts[0].tiedPrev == n.tiedPrev
        assert parts[-1].tiedNext == n.tiedNext, f"{n=}, {parts=}"

    return parts


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
                return notationNeedsBreak(n, dt, div, beatOffset=now)


def _tieNotationParts(parts: list[Notation]) -> None:
    """ Tie these notations in place """

    for part in parts[:-1]:
        part.tiedNext = True

    hasGliss = parts[0].gliss
    for part in parts[1:]:
        part.tiedPrev = True
        part.dynamic = ''
        part.removeAttachments(lambda a: isinstance(a, (attachment.Text, attachment.Articulation)))
        if hasGliss:
            part.gliss = True

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
    if numSlots == 1:
        return [n]
    elif numSlots > 25:
        raise ValueError("Division not supported")

    slotDivisions = quantdata.slotDivisionStrategy[numSlots]
    if slotIndex % 2 == 1 and slotDivisions[-1] % 2 == 1:
        slotDivisions = list(reversed(slotDivisions))

    offset = n.offset
    parts: list[Notation] = []
    for slots in slotDivisions:
        partDur = slotDur * slots
        assert partDur > F(1, 64)
        parts.append(n.clone(offset=offset, duration=partDur))
        offset += partDur

    _tieNotationParts(parts)
    assert sum(part.duration for part in parts) == n.duration
    assert parts[0].offset == n.offset
    assert parts[-1].end == n.end
    assert parts[0].tiedPrev == n.tiedPrev
    assert parts[-1].tiedNext == n.tiedNext
    return parts


def _breakIrregularDuration(n: Notation, beatDur:F, div: int, beatOffset=F(0)
                            ) -> list[Notation] | None:
    # beat is subdivided regularly
    slotdur = beatDur/div
    nslots = n.duration/slotdur
    assert isinstance(nslots, F), f"Expected type F, got {type(nslots).__name__}={nslots}"

    if nslots.denominator != 1:
        raise ValueError(f"Duration is not quantized with given division.\n  {n=}, {div=}, {slotdur=}, {nslots=}")

    if nslots.numerator in _regularDurations:
        return None

    slotIndex = (n.offset-beatOffset)/slotdur
    assert int(slotIndex) == slotIndex
    slotIndex = int(slotIndex)

    if not slotIndex.denominator == 1:
        raise ValueError(f"Offset is not quantized with given division. n={n}, division={div}")

    parts = _splitIrregularDuration(n, slotIndex, slotdur)
    return parts


def isRegularDuration(dur: Rational, beatDur: Rational) -> bool:
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
                           beatOffset: Rational = F(0)
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
    assert isinstance(beatDivision, (int, tuple)), f"Expected type int/tuple, got {type(beatDivision).__name__}={beatDivision}"
    assert isinstance(beatDur, F), f"Expected type F, got {type(beatDur).__name__}={beatDur}"
    assert isinstance(beatOffset, F), f"Expected type F, got {type(beatOffset).__name__}={beatOffset}"
    assert beatOffset <= n.offset and n.end <= beatOffset + beatDur
    assert n.duration >= 0

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
    assert len(ticks) == numDivisions + 1

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
                parts = breakIrregularDuration(n=subn, beatDur=divDuration, beatDivision=numslots,
                                               beatOffset=timespan[0])
                if parts is None:
                    # subn is regular
                    allparts.append(subn)
                else:
                    allparts.extend(parts)
    assert sum(part.duration for part in allparts) == n.duration
    _tieNotationParts(allparts)
    assert all(isinstance(part, Notation) for part in allparts)
    assert sum(p.duration for p in allparts) == n.duration
    assert allparts[0].tiedPrev == n.tiedPrev
    assert allparts[-1].tiedNext == n.tiedNext
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


def splitNotationsAtOffsets(notations: list[Notation],
                            offsets: Sequence[F]
                            ) -> list[tuple[TimeSpan, list[Notation]]]:
    """
    Split the given notations between the given offsets

    Args:
        notations: the notations to split
        offsets: the boundaries.

    Returns:
        a list of of tuples (timespan, notation)

    """
    timeSpans = [TimeSpan(beat0, beat1) for beat0, beat1 in iterlib.pairwise(offsets)]
    splittedEvents = []
    for ev in notations:
        if ev.duration > 0:
            splittedEvents.extend(splitNotationAtOffsets(ev, offsets))
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


def measureSplitNotationsAtBeats(eventsInMeasure: list[Notation],
                                 timesig: timesig_t,
                                 quarterTempo: number_t,
                                 subdivisionStructure: list[int] = None
                                 ) -> list[tuple[TimeSpan, list[Notation]]]:
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
    assert isinstance(quarterTempo, (int, F)), f"Expected {F}, got {quarterTempo}, {type(quarterTempo)}"

    assert isMeasureFilled(eventsInMeasure, timesig), \
        f"Measure is not filled. Timesig {timesig}, tempo: {quarterTempo}\n" \
        f"events: {eventsInMeasure}"

    beatOffsets = measureBeatOffsets(timesig=timesig, quarterTempo=quarterTempo,
                                     subdivisionStructure=subdivisionStructure)
    return splitNotationsAtOffsets(eventsInMeasure, beatOffsets)


def _groupByRatio(notations: list[Notation], division: int | division_t,
                  beatOffset:F, beatDur:F
                  ) -> DurationGroup:
    if isinstance(division, int) or len(division) == 1:
        if isinstance(division, tuple):
            division = division[0]
        durRatio = quantdata.durationRatios[division]
        return DurationGroup(durRatio=durRatio, items=notations)

    assert isinstance(division, tuple) and len(division) >= 2
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


def _applyDurationRatio(notations:list[Notation],
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
        for n in notations:
            if n.durRatios is None:
                n.durRatios = []
            if not(durRatio == 1 and n.durRatios and n.durRatios[-1] == 1):
                n.durRatios.append(durRatio)
    else:
        numSubBeats = len(division)
        now = beatOffset
        dt = beatDur / numSubBeats
        durRatio = F(*quantdata.durationRatios[numSubBeats])
        for n in notations:
            if n.durRatios is None:
                n.durRatios = []
            if not (durRatio == 1 and n.durRatios and n.durRatios[-1] == 1):
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
        timesig: the time signature of the measure: a touple (num, den)
        quarterTempo: the tempo of the measure using a quarter note as refernce
        profile: the quantization preset. Leave it unset to use the default
            preset.

    Returns:
        a QuantizedMeasure

    """
    measureQuarterLength = util.measureQuarterDuration(timesig)
    assert all(0<=ev.offset<=ev.end<=measureQuarterLength
               for ev in events), f"{events=}, {measureQuarterLength=}"
    for ev0, ev1 in iterlib.pairwise(events):
        if ev0.end != ev1.offset:
            logger.error(f"{ev0} (end={ev0.end}), {ev1} (offset={ev1.offset})")
            raise AssertionError("events are not stacked")

    quantizedBeats: list[QuantizedBeat] = []

    if not isMeasureFilled(events, timesig):
        events = _fillMeasure(events, timesig)

    for span, eventsInBeat in measureSplitNotationsAtBeats(eventsInMeasure=events,
                                                           timesig=timesig,
                                                           quarterTempo=quarterTempo,
                                                           subdivisionStructure=subdivisionStructure):
        beatdur = span.end - span.start
        if beatdur.numerator in (1, 2, 4):
            quantizedBeat = quantizeBeatBinary(eventsInBeat=eventsInBeat,
                                               quarterTempo=quarterTempo,
                                               beatDuration=span.end - span.start,
                                               beatOffset=span.start,
                                               profile=profile)
            quantizedBeats.append(quantizedBeat)
        elif beatdur.numerator == 3:
            subBeats = quantizeBeatTernary(eventsInBeat=eventsInBeat,
                                           quarterTempo=quarterTempo,
                                           beatDuration=beatdur,
                                           beatOffset=span.start,
                                           profile=profile)
            quantizedBeats.extend(subBeats)
        else:
            raise ValueError(f"beat duration not supported: {beatdur}")
    return QuantizedMeasure(timesig=timesig, quarterTempo=asF(quarterTempo), beats=quantizedBeats,
                            profile=profile)


def splitNotationByMeasure(n: Notation, struct: ScoreStruct
                           ) -> list[tuple[int, Notation]]:
    """
    Split a Notation if it extends across multiple measures.

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
        beat1 = struct.getMeasureDef(measureindex1).durationBeats

    numMeasures = measureindex1 - measureindex0 + 1

    if numMeasures == 1:
        # The note fits within one measure. Make the offset relative to the measure
        event = n.clone(offset=beat0, duration=beat1 - beat0)
        return [(measureindex0, event)]

    measuredef = struct.getMeasureDef(measureindex0)
    dur = measuredef.durationBeats - beat0
    notation = n.clone(offset=beat0, duration=dur, tiedNext=True)
    pairs = [(measureindex0, notation)]

    # add intermediate measure, if any
    if numMeasures > 2:
        for m in range(measureindex0 + 1, measureindex1):
            measuredef = struct.getMeasureDef(m)
            notation = n.clone(offset=F(0),
                               duration=measuredef.durationBeats,
                               tiedPrev=True, tiedNext=True, dynamic='')
            notation.removeAttachmentsByClass('articulation')
            pairs.append((m, notation))

    # add last notation
    if beat1 > 0:
        notation = n.clone(offset=F(0), duration=beat1, tiedPrev=True,
                           dynamic='')
        notation.removeAttachments(lambda a: isinstance(a, attachment.Articulation))
        pairs.append((measureindex1, notation))

    parts = [part for index, part in pairs]
    _tieNotationParts(parts)

    sumdur = sum(struct.beatDelta((i, n.offset), (i, n.end)) for i, n in pairs)
    assert sumdur == n.duration, f"{n=}, {sumdur=}, {numMeasures=}\n{pairs=}"
    return pairs


def _mergeGroups(group1: DurationGroup, group2: DurationGroup,
                 profile: QuantizationProfile,
                 beatOffsets: list[F]
                 ) -> DurationGroup:
    # we don't check here, just merge
    group = DurationGroup(durRatio=group1.durRatio, items=group1.items + group2.items)
    group = group.mergedNotations()
    group = _mergeSiblings(group, profile=profile, beatOffsets=beatOffsets)
    return group


def _groupsCanMerge(g1: DurationGroup, g2: DurationGroup, profile: QuantizationProfile,
                    beatOffsets: list[F]
                    ) -> Result:
    assert len(g1.items) > 0 and len(g2.items) > 0
    acrossBeat = any(g1.offset < offset < g2.end for offset in beatOffsets)

    if g1.durRatio != g2.durRatio:
        return Result(False, "not same durRatio")
    if g1.durRatio != (1, 1):
        if acrossBeat and g1.durRatio[0] not in profile.allowedTupletsAcrossBeat:
            return Result(False, "tuplet not allowed to merge across beat")
        elif (g1.duration() + g2.duration() > profile.mergedTupletsMaxDuration):
            return Result(False, "incompatible duration")
        elif not profile.mergeTupletsOfDifferentDuration and acrossBeat and g1.duration() != g2.duration():
            return Result(False, "Groups of different duration cannot merge")

    item1, item2 = g1.items[-1], g2.items[0]
    if isinstance(item1, DurationGroup) and isinstance(item2, DurationGroup):
        if not (r := _groupsCanMerge(item1, item2, profile=profile, beatOffsets=beatOffsets)):
            return Result(False, f'nested tuplets cannot merge: {r.info}')
        else:
            return Result(True, '')
    if isinstance(item1, DurationGroup) or isinstance(item2, DurationGroup):
        return Result(False, 'A group cannot merge with a single item')

    if acrossBeat and not profile.mergeNestedTupletsAcrossBeats:
        g1nested = any(isinstance(item, DurationGroup) and item.durRatio != g1.durRatio for item in g1.items)
        if g1nested:
            return Result(False, "Cannot merge nested tuples")
        g2nested = any(isinstance(item, DurationGroup) and item.durRatio != g2.durRatio for item in g2.items)
        if g2nested:
            return Result(False, "Cannot merge nested tuples")

    if isinstance(item1, Notation) and isinstance(item2, Notation) and acrossBeat:
        if not item1.canMergeWith(item2):
            return Result(False, f'{item1} cannot merge with {item2}')
        if item1.duration + item2.duration < profile.minBeatFractionAcrossBeats:
            return Result(False, 'Absolute duration of merged Notations across beat too short')
        if item1.symbolicDuration() + item2.symbolicDuration() < profile.minSymbolicDurationAcrossBeat:
            return Result(False, 'Symbolic duration of merged notations across beat too short')

    #print(f"*************** ok, merging\n{g1}\n with {g2}")
    #print(f"::::: {item1=}\n::::: {item2=}")
    #if isinstance(g1.items[0], DurationGroup):
    #    print("****", g1.items[0].durRatio, g1.durRatio)
    return Result(True, '')


def _mergeSiblings(root: DurationGroup,
                   profile: QuantizationProfile,
                   beatOffsets: list[F]
                   ) -> DurationGroup:
    """
    Merge sibling groupTree of the same kind, if possible

    Args:
        root: the root of a tree of DurationGroups
        profile: the quantization profile
        beatOffsets: these offsets are used to determine if a merged group
            would cross a beat boundary. The quantization profile has some
            rules regarding merging tuplets across beat boundaries which need
            this information

    Returns:
        a new tree
    """
    # merge only groupTree (not Notations) across groupTree of same level
    if len(root.items) <= 1:
        return root
    newroot = DurationGroup(durRatio=root.durRatio)
    item1 = root.items[0]
    if isinstance(item1, DurationGroup):
        item1 = _mergeSiblings(item1, profile=profile, beatOffsets=beatOffsets)
    newroot.append(item1)
    for item2 in root.items[1:]:
        item1 = newroot.items[-1]
        if isinstance(item2, DurationGroup):
            item2 = _mergeSiblings(item2, profile=profile, beatOffsets=beatOffsets)
        if isinstance(item1, DurationGroup) and isinstance(item2, DurationGroup):
            # check if the groupTree should merge
            if (r:=_groupsCanMerge(item1, item2, profile=profile, beatOffsets=beatOffsets)):
                mergedgroup = _mergeGroups(item1, item2, profile=profile, beatOffsets=beatOffsets)
                newroot.items[-1] = mergedgroup
            else:
                newroot.append(item2)
        elif isinstance(item1, Notation) and isinstance(item2, Notation) and item1.canMergeWith(item2):
            newroot.items[-1] = item1.mergeWith(item2)
        else:
            newroot.append(item2)
    return newroot


def _maxTupletLength(timesig: timesig_t, subdivision: int):
    den = timesig[1]
    if subdivision == 3:
        return {2: 2, 4:2, 8: 1}[den]
    elif subdivision == 5:
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
    name: str = ''
    shortname: str = ''

    def __post_init__(self):
        self._fixTies()
        self.removeUnnecessaryGracenotes()

    def __hash__(self):
        measureHashes = tuple(hash(m) for m in self.measures)
        return hash(('QuantizedPart', self.name) + measureHashes)

    def flatNotations(self) -> Iterator[Notation]:
        """Iterate over all notations in this part"""
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

    def logicalTies(self) -> list[list[Notation]]:
        """
        Yields all logical ties in this part

        A logical tie is a list of notations which are linked together

        """
        ties = []
        for measurenum, beatoffset, n in self.iterNotations():
            if n.tiedPrev:
                ties[-1].append(n)
            else:
                ties.append([n])
        return ties

    def mergedTies(self) -> list[Notation]:
        """
        Yields all merged ties in this part
        """
        def mergeTie(notations: list[Notation]) -> Notation:
            n0 = notations[0].copy()
            n0.durRatios = None
            for n in notations[1:]:
                n0.duration += n.duration
            return n0

        return [mergeTie(tie) if len(tie) > 1 else tie[0] for tie in self.logicalTies()]


    def iterNotations(self) -> Iterator[PartLocation]:
        """
        Iterates over all notations giving the location of each notation

        For each notation yields a tuple: ``(measure number, QuantizedBeat, Notation)``
        """
        for i, m in enumerate(self.measures):
            if m.isEmpty():
                continue
            assert m.beats is not None
            for b in m.beats:
                for n in b.notations:
                    yield PartLocation(i, b, n)

    def dump(self, indents=0):
        for i, m in enumerate(self.measures):
            ind = _INDENT * indents
            print(f'{ind}Measure #{i}')
            m.dump(indents=indents)

    def bestClef(self, maxNotes=0) -> str:
        """
        Return the best clef for the notations in this part

        The returned str if one of 'treble', 'treble8',
        'bass' and 'bass8'

        Args:
            maxNotes: if given, only use the first *maxNotes* notes
                to calculate the clef

        Returns:
            the clef descriptor which best fits this part; one of 'treble',
            'treble8', 'bass', 'bass8', where the 8 indicates an octave
            transposition in the direction of the clef (high for treble,
            low for bass)
        """
        avgPitch = self.averagePitch(maxNotations=maxNotes)

        if avgPitch == 0:
            # all rests
            return "treble"

        if avgPitch > 86:
            return "treble8"
        elif avgPitch > 58:
            return "treble"
        elif avgPitch > 36:
            return "bass"
        else:
            return "bass8"

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
                        loc2.notation.setNotehead('hidden')
                    if not loc2.notation.tiedNext:
                        break

    def _fixTies(self):
        notations = list(self.flatNotations())
        for n in notations:
            if n.isRest or not n.pitches:
                n.tiedNext = False
                n.tiedPrev = False

        for n0, n1 in iterlib.pairwise(notations):
            if n0.tiedNext:
                if not n0.pitches or not n1.pitches or any(x not in n1.pitches for x in n0.pitches):
                    n0.tiedNext = False
                    n1.tiedNext = False

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
                 profile: str | QuantizationProfile,
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
        profile = QuantizationProfile.fromPreset(profile)
    part.fillGaps()
    notations = core.stackNotations(part)
    quantutils.transferAttributesWithinTies(notations)
    allpairs = [splitNotationByMeasure(n=n, struct=struct) for n in notations]
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
            core.removeOverlapInplace(notations)
            try:
                qmeasure = quantizeMeasure(notations,
                                           timesig=measureDef.timesig,
                                           quarterTempo=measureDef.quarterTempo,
                                           profile=profile,
                                           subdivisionStructure=measureDef.subdivisionStructure)
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
    qpart = QuantizedPart(struct, qmeasures, name=part.name, shortname=part.shortname)
    qpart.glissMarkTiedNotesAsHidden()
    qpart.removeUnnecessaryGracenotes()
    return qpart


@dataclass
class QuantizedScore:
    """
    A QuantizedScore represents a list of quantized parts

    See :func:`quantize` for an example
    """
    parts: list[QuantizedPart]
    """A list of QuantizedParts"""

    title: str | None = None
    """Title of the score, used for rendering purposes"""

    composer: str | None = None
    """Composer of the score, used for rendering"""

    def __hash__(self):
        partHashes = [hash(p) for p in self.parts]
        return hash((self.title, self.composer) + tuple(partHashes))

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
        if not self.parts:
            raise IndexError("This QuantizedScore has no parts")
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
        """Adds empty measures at the end of each part so that all have the same length"""
        numMeasures = self.numMeasures()
        for part in self.parts:
            part.pad(numMeasures - len(part.measures))

    def toCoreScore(self, mergeTies=True) -> maelzel.core.Score:
        """
        Convert this to a maelzel.core.Score

        Args:
            mergeTies: if True, notes/chords which can be merged into longer events
                are merged

        Returns:
            a maelzel.core.Score representing this QuantizedScore

        """
        from .notation import notationsToCoreEvents
        import maelzel.core
        voices = []
        for part in self.parts:
            if mergeTies:
                notations = part.mergedTies()
            else:
                ties = part.logicalTies()
                notations = []
                for tie in ties:
                    notations.extend(tie)
            events = notationsToCoreEvents(notations)
            voice = maelzel.core.Voice(events)
            voices.append(voice)
        return maelzel.core.Score(voices=voices, scorestruct=self.scorestruct, title=self.title)


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
            The quantization preset determines how events are quantized,
            which divisions of the beat are possible, how a best division
            is weighted and selected, etc. Not all options in a preset
            are supported by all backends (for example, music21 backend
            does not support nested tuplets).
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
        qparts.append(qpart)
    return QuantizedScore(qparts)

