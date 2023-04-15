"""
Quantize durations to musical notation

The most important function here is :func:`quantize`, which treturns
a :class:`QuantizedScore`

"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field as _field, fields as _fields
from functools import cache
import sys
import os

from .common import *
from . import core
from . import definitions
from . import util
from . import quantdata
from . import quantutils
from . import enharmonics
from . import attachment
from . import renderer
from . import spanner as _spanner


from .notation import Notation, makeRest, SnappedNotation, tieNotationParts
from .node import Node, asTree
from maelzel import scorestruct as st

from emlib import iterlib
from emlib import misc
from emlib.misc import Result
from emlib import mathlib
from emlib import logutils

from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from numbers import Rational
    from typing import Iterator, Sequence
    import maelzel.core
    number_t = Union[int, float, Rational, F]


__all__ = (
    'quantize',
    'QuantizationProfile',
    'QuantizedScore',
    'QuantizedPart',
    'QuantizedMeasure',
    'QuantizedBeat',
    'quantizeMeasure',
    'quantizePart',
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
      and totalDuration errors when fitting events to a grid. A higher
      weight will cause quantization to minimize offset and totalDuration
      errors, at the cost of choosing more complex divisions
    - divisionErrorWeight: also a weight to controll all effect
      dealing with the complexity of a given division/subdivision

    Lower level parameters to calculate grid error:

    - offsetErrorWeight: the importance of offset errors to calculate
      the best subdivision of a beat
    - restOffsetErrorWeight: how relevant should be the offset error in
      the case of rests
    - durationErrorWeight: relevance of totalDuration error when selecting the
      best subdivision
    - graceNoteDuration: if a note is considered a grace note (which have
      no totalDuration per se), should we still account for this totalDuration?
    - minBeatFractionAcrossBeats: when merging durations across beats, a merged
      totalDuration can't be smaller than this totalDuration. This is to prevent joining
      durations across beats which might result in high rhythmic complexity
    - tupletsAllowedAcrossBeats: list of tuplets allowed across a beat
    - tupletMaxDur: the max quarternote totalDuration for a merged subdivision

    Lower level parameters to calculate division complexity:

    - numNestedTupletsPenaltyWeight: how

    """
    nestedTuplets: bool = False
    """Are nested tuplets allowed?"""

    gridErrorWeight: float = 1
    """Weight of the overall effect of offset and totalDuration errors when fitting events to a grid. 
    A higher weight minimizes offset and totalDuration errors at the cost of more complex divisions"""

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
    totalDuration is irregular (durations of 5 or 9 units, which need ties to be represented)
    """

    offsetErrorWeight: float = 1.0
    """Weight of the offset between original start and makeSnappedNotation start"""

    restOffsetErrorWeight: float = 0.5
    """Similar to offsetErrorWeight but for rests"""

    durationErrorWeight: float = 0.2
    """Weight of the difference in totalDuration resulting from quantization"""

    gracenoteDuration: F = F(1, 32)
    """A totalDuration to assume for grace notes"""

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

    breakLongGlissandi: bool = True
    """When a glissando extends over a quarternote, break it into quarter notes
    
    If the noteheads are hidden, a glissando over a half-note cannot be differentiated
    from a glissando over a quarternote. If this option is True, such a long glissando
    is broken into quarternotes in order to prevent this misinterpretation"""

    maxPenalty: float = 1.0
    """A max. penalty when quantizing a beat, to limit the search space"""

    debug: bool = False
    """Turns on debugging"""

    debugMaxDivisions: int = 20
    """Max number of quantization possibilities to display when debugging"""

    blacklist: set[division_t] = _field(default_factory=set)
    """A set of divisions which should never be considered"""

    name: str = ''
    """A name for this profile, if needed"""

    breakSyncopationsLevel: str = 'strong'
    """
    Break syncopations at beat boundaries ('none': do not break syncopations, 'all': break at all beats, 
    'strong': only strong beats)
    """

    _cachedDivisionsByTempo: dict[tuple[number_t, bool], list[division_t]] = _field(default_factory=dict)
    _cachedDivisionPenalty: dict[tuple[int, ...], tuple[float, str]] = _field(default_factory=dict)

    def __post_init__(self):
        self._cachedDivisionsByTempo = {}
        self._cachedDivisionPenalty = {}

    @staticmethod
    def makeSimple(maxSubdivisions=3,
                   possibleSubdivisions=(1, 2, 3, 4, 5, 6, 8),
                   maxDensity=16,
                   allegroTempo=132,
                   allegroMaxSubdivisions=1,
                   allegroPossibleSubdivisions=(1, 2, 3, 4, 6),
                   nestedTuplets=False,
                   complexityPreset='medium',
                   mintempo=1) -> QuantizationProfile:
        """
        Static method to create a simple QuantizationProfile based on a preset

        Args:
            maxSubdivisions: the max. subdivisions of a beat for tempi under allegro
            possibleSubdivisions: the kind of subdivisions possible
            maxDensity: the max. number of slots per quarter note
            allegroTempo: tempo used to switch to the allegro subdivision profile. Set
                this very high to always use the slow profile, or set it very low
                to always use the high profile
            allegroMaxSubdivisions: similar to maxSubdivisions, used for tempi which
                are higher than the value given for allegroTempo
            allegroPossibleSubdivisions: similar to possibleSubdivisions, used for
                tempi higher than *allegroTempo*
            nestedTuplets: are nested tuplets allowed? A nested tuple is a non-binary
                subdivision of the beat within a non-binary subdivision of the
                beat (something like (3, 5, 7), which divides the beat in three, each
                subdivision itself divided in 3, 5 and 7 parts)
            complexityPreset: the preset to use to fill the rest of the parameters
                (one of 'lowest', 'low', 'medium', 'high', 'highest')
            mintempo: the min. allowed tempo for the quarter note.

        Returns:
            a QuantizationProfile

        """
        divs = {
            mintempo: [],
            allegroTempo: quantutils.allSubdivisions(maxsubdivs=maxSubdivisions,
                                                     possiblevals=possibleSubdivisions,
                                                     maxdensity=maxDensity),
            999: quantutils.allSubdivisions(maxsubdivs=allegroMaxSubdivisions,
                                            possiblevals=allegroPossibleSubdivisions,
                                            maxdensity=max(int(maxDensity*0.5), 8))
        }
        out = QuantizationProfile.fromPreset(complexity=complexityPreset,
                                             nestedTuplets=nestedTuplets)
        out.possibleDivisionsByTempo = divs
        return out

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
            logger.error("Possible divisions of the beat, by tempo: ")
            logutils.prettylog(logger, self.possibleDivisionsByTempo)
            raise ValueError(f"No divisions for the given tempo (q={int(tempo)})")

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
    def fromPreset(complexity='high',
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
            for maxtempo, divisions in out.possibleDivisionsByTempo.items():
                divisions = [div for div in divisions if div not in blacklistset]
                out.possibleDivisionsByTempo[maxtempo] = divisions
            out.blacklist = blacklistset
        return out


@cache
def _isNestedTupletDivision(div: division_t) -> bool:
    if isinstance(div, int):
        # A shortcut division, like 3 or 5
        return False
    return not mathlib.ispowerof2(len(div)) and any(not mathlib.ispowerof2(subdiv) for subdiv in div)


defaultQuantizationProfile = QuantizationProfile()


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
            numComplexSubdivs = sum(subdiv > 8 or subdiv in (3, 5, 7)
                                    for subdiv in division)
            penalty *= profile.complexNestedTupletsFactor ** numComplexSubdivs
        elif l == 6:
            numComplexSubdivs = sum(subdiv == 5 or subdiv == 7 or subdiv > 8
                                    for subdiv in division)
            penalty *= profile.complexNestedTupletsFactor ** numComplexSubdivs

    if debug and nestingLevel == 0:
        info = f"{divPenalty=:.3g}, {cardinalityPenalty=:.3g}, {numSubdivsPenalty=:.3g}, {levelPenalty=:.3g}"
    else:
        info = ''
    return min(penalty, 1), info


@cache
def _divisionCardinality(division, excludeBinary=False):
    # TODO: make general form for deeply nested tuplets
    if isinstance(division, int):
        return 1

    allfactors = quantutils.primeFactors(len(division), excludeBinary=excludeBinary).copy()
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
    # We use floats to make this faster. Rounding errors should not pose a problem
    # in this context
    fgrid = [float(g) for g in grid]
    return [misc.nearest_index(float(event.offset), fgrid) for event in events]


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
        beatDuration: the totalDuration of the beat
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
    totalDuration exactly. This function is normally called prior to quantization

    Args:
        notations: a list of notations inside the beat
        duration: the totalDuration to fill
        offset: the starting time to fill

    Returns:
        a list of notations which fill the beat exactly

    .. note::

        If any notation has an unset totalDuration, this will extend either to
        the next notation or to fill the given totalDuration

    """
    assert all(n.offset is not None for n in notations)
    assert all(n.offset-offset <= duration for n in notations), \
        f"Events start after totalDuration to fill ({duration=}): {_eventsShow(notations)}"
    assert all(n0.offset <= n1.offset for n0, n1 in iterlib.pairwise(notations)), \
        f"events are not sorted: {_eventsShow(notations)}"
    assert all(n0.end <= n1.offset for n0, n1 in iterlib.pairwise(notations) if n0.duration is not None), \
        f"events overlap: {_eventsShow(notations)}"
    assert all(n.end<=offset+duration for n in notations if n.duration is not None), \
        "events extend over beat totalDuration"

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
    assert sum(n.duration for n in out) == duration
    end = offset + duration
    assert all(offset <= n.offset <= end for n in out)
    return out


def _evalGridError(profile: QuantizationProfile,
                   snappedEvents: list[SnappedNotation],
                   beatDuration:F) -> float:
    """
    Evaluate the error regarding the deviation of the makeSnappedNotation events from the original offset/totalDuration

    Given a list of events in a beat and these events makeSnappedNotation to a given subdivision of
    the beat, evaluate how good this snapping is in representing the original events.
    This is used to find the best subdivision of a beat.

    Args:
        profile: the quantization preset to use
        snappedEvents: the events after being makeSnappedNotation to a given grid
        beatDuration: the totalDuration of the beat

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
                 beatOffset: F = F(0),
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
            "divisions: {self.divisions}, assignedSlots={self.assignedSlots}, "
             f"notations={self.notations}, beatDuration={self.beatDuration}, beatOffset={self.beatOffset}, "
             f"quantizationError={self.quantizationInfo:.5g}, weight={self.weight}"
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
        return hash((self.timesig, self.quarterTempo) + tuple(hash(b) for b in self.beats))

    def resetTree(self):
        """Remove any cached tree representation of this measure"""
        self._root = None

    def getMeasureIndex(self) -> int | None:
        """Return the measure index of this measure within the QUantizedPart"""
        if not self.parent:
            return None
        return self.parent.measures.index(self)

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
            the totalDuration of the measure in quarter notes
        """
        return sum(self.beatDurations())

    def beatBoundaries(self) -> list[F]:
        """The beat offsets, including the end of the measure"""
        boundaries = [F(0)]
        now = F(0)
        for dur in self.beatDurations():
            boundaries.append(now)
            now += dur
        return boundaries

    def beatOffsets(self) -> list[F]:
        """
        Returns a list of the offsets of each beat within this measure

        Returns:
            the offset of each beat
        """
        if self._offsets is None:
            self._offsets = [beat.beatOffset for beat in self.beats]
        return self._offsets

    def fixEnharmonics(self, options: enharmonics.EnharmonicOptions) -> None:
        """
        Pin the enharmonic spellings within this measure (in place)

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
                print(f"Error in durations:\n\t{n0=},\t{n1=}")
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
        nodes = self.nodes()
        root = asTree(nodes)
        self._root = root = _mergeSiblings(root, profile=self.quantprofile, beatOffsets=self.beatOffsets())
        root.repair()
        if root.totalDuration() != self.duration():
            logger.error(f"Measure index: {self.getMeasureIndex()}")
            self.dump(tree=False)
            self.dump(tree=True)
            raise ValueError(f"Duration mismatch in tree. "
                             f"Tree duration: {root.totalDuration()}, "
                             f"measure duration: {self.duration()}")
        assert root.totalDuration() == self.duration()
        return root

    def breakSyncopations(self, level='weak') -> None:
        """
        Break notes extending over beat boundaries

        The level indicates which syncopations to break. 'all' will split
        any notations extending over any beat; 'weak' will only break
        syncopations over secondary beats (for example, the 3rd quarter-note
        in a 4/4 measure); 'strong' will only break syncopations over strong
        beats (the 4th quarternote in a 6/4 measure with the form 3+3, or the 3rd
        quarternote in a 7/8 measure with the form 2+2+3

        Args:
            level: one of 'all', 'weak', 'strong'

        """
        minWeight = {
            'all': 0,
            'weak': 1,
            'strong': 2
        }.get(level)
        if minWeight is None:
            raise KeyError(f"Expected one of 'all, 'weak', 'strong', got {level}")

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
                tree.splitAtBeatBoundary(beat.beatOffset, key=dosplit)

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

    def breakBeamsAtBeats(self) -> None:
        if self.isEmpty():
            return
        _breakBeamsAtOffsets(self.tree(), self.beatOffsets())


def _breakBeamsAtOffsets(root: Node, offsets: list[F]) -> None:
    for item in root:
        if isinstance(item, Notation) and item.offset in offsets:
            item.setProperty('.breakBeam', True)
        else:
            _breakBeamsAtOffsets(item, offsets)

def _removeUnnecessaryDurationRatios(n: Notation) -> None:
    if not n.durRatios:
        return
    for r in reversed(n.durRatios.copy()):
        if r != F(1):
            break
        n.durRatios.pop()


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
            The totalDuration can be left undefined (as -1) if the event to which this attack
            refers extends to the next attack or to the end of the beat.
        beatDuration: totalDuration of the beat, in quarter notes (1=quarter, 0.5=eigth note)
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
    # assert sum(event.duration for event in eventsInBeat) == beatDuration


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
        raise ValueError("Events in beat do not sum up to the beat totalDuration")

    #assert all(0 <= ev.duration <= beatDuration and
    #           beatOffset <= ev.offset <= ev.end <= beatOffset+beatDuration
    #           for ev in eventsInBeat)

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
    firstOffset = eventsInBeat[0].duration

    lastOffset = eventsInBeat[-1].offset - beatOffset
    lastOffsetMargin = beatDuration - (eventsInBeat[-1].offset - beatOffset)
    offsetThreshold = beatDuration / 8

    for div in possibleDivisions:

        if div in seen or div in profile.blacklist:
            continue

        # Exclude divisions which are not worth evaluating at full
        # NB: simplifyDivision is efficient, but it is  called a lot
        # so it is worth to find early if a division does not need to
        # be analyzed in full

        if len(div) > 1:
            numSimplifiedSubdivisions = max(0, firstOffset - offsetThreshold) // F(1, len(div))
            if numSimplifiedSubdivisions > 0:
                div2 = (1,) * numSimplifiedSubdivisions + div[numSimplifiedSubdivisions:]
                if div2 in seen:
                    continue
                else:
                    seen.add(div2)

        if len(div) > 1:
            numSimplifiedSubdivisions = max(0, lastOffsetMargin - offsetThreshold) // F(1, len(div))
            if numSimplifiedSubdivisions > 0:
                rdiv = div[::-1]
                div2 = (1,) * numSimplifiedSubdivisions + rdiv[numSimplifiedSubdivisions:]
                div2 = div2[::-1]
                if div2 in seen:
                    continue
                else:
                    seen.add(div2)

        divPenalty, divPenaltyInfo = profile.divisionPenalty(div)

        if profile.divisionErrorWeight > 0 and divPenalty > profile.maxDivPenalty:
            skipped += 1
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
        misc.print_table(table, headers="error div makeSnappedNotation slots info".split(), floatfmt='.4f', showindex=False)

    assert rows, f"{possibleDivisions=}"
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
            eventParts = breakIrregularDuration(n, beatDivision=div, beatDur=beatDuration, beatOffset=beatOffset)
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
    assert all(0<=ev.offset and ev.end<=measureDuration for ev in eventsInMeasure)
    return _fillDuration(eventsInMeasure, measureDuration)


def _splitNotationAtOffsets(n: Notation, offsets: Sequence[F]) -> list[Notation]:
    """
    Splits a Notation at the given offsets

    Args:
        n: the Notation to split
        offsets: the offsets at which to split n

    Returns:
        the parts after splitting

    Example::

        >>> splitNotationAtOffsets(Notation(F(0.5), totalDuration=F(1)))
        [Notation(0.5, totalDuration=0.5), Notation(1, totalDuration=0.5)]

    """
    return n.splitNotationAtOffsets(offsets)

def _splitNotationAtOffsets_old(n: Notation, offsets: Sequence[Rational]) -> list[Notation]:
    """
    Splits a Notation at the given offsets

    Args:
        n: the Notation to split
        offsets: the offsets at which to split n

    Returns:
        the parts after splitting

    Example::

        >>> splitNotationAtOffsets(Notation(F(0.5), totalDuration=F(1)))
        [Notation(0.5, totalDuration=0.5), Notation(1, totalDuration=0.5)]

    """
    if not offsets:
        raise ValueError("offsets is empty")

    assert n.duration>=0

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
        # part.removeAttachments(lambda item: isinstance(item, (attachment.Articulation, attachment.Text)))
        if part.spanners:
            part.spanners.clear()

    if not n.isRest:
        tieNotationParts(parts)
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
        beatDur: the totalDuration of the beat, in quarters
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
        slotDur: which is the quarterNote totalDuration of slotDur

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

    slotDivisions = quantdata.splitIrregularSlots(numSlots, slotIndex)

    offset = F(n.offset)
    parts: list[Notation] = []
    for slots in slotDivisions:
        partDur = slotDur * slots
        assert partDur > F(1, 64)
        parts.append(n.clone(offset=offset, duration=partDur))
        offset += partDur

    tieNotationParts(parts)
    assert sum(part.duration for part in parts) == n.duration
    assert parts[0].offset == n.offset
    assert parts[-1].end == n.end
    assert parts[0].tiedPrev == n.tiedPrev
    assert parts[-1].tiedNext == n.tiedNext
    assert parts[0].spanners == n.spanners
    return parts


def _breakIrregularDuration(n: Notation, beatDur: Rational, div: int, beatOffset: Rational =F(0)
                            ) -> list[Notation] | None:
    # beat is subdivided regularly
    slotdur = beatDur/div
    nslots = n.duration/slotdur

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
    Is the totalDuration regular?

    Regular durations are those which (in priciple) can be represented
    without tied - either binary units (1, 2, 4, 8, ...) or dotted notes
    (3, 6, 7, ...).

    Args:
        dur: the totalDuration to evaluate
        beatDur: the totalDuration of the beat

    Returns:
        True if this totalDuration is regular

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
        beatDur: the totalDuration of the beat
        beatDivision: the division of the beat
        beatOffset: the offset of the beat

    Returns:
        None if the notations has a regular totalDuration, or a list of tied Notations which
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
    subns = n.splitNotationAtOffsets(ticks)
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
    tieNotationParts(allparts)
    #assert all(isinstance(part, Notation) for part in allparts)
    #assert sum(p.totalDuration for p in allparts) == n.totalDuration
    #assert allparts[0].tiedPrev == n.tiedPrev
    #assert allparts[-1].tiedNext == n.tiedNext
    return allparts


def isMeasureFilled(notations: list[Notation], timesig: timesig_t) -> bool:
    """Do the notations fill the measure?"""
    measureDuration = util.measureQuarterDuration(timesig)
    notationsDuration = sum(n.duration for n in notations)
    if notationsDuration > measureDuration:
        logger.error(f"timesig: {timesig}, Notation: {notations}")
        logger.error(f"Sum totalDuration: {notationsDuration}")
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
        a list of of tuples (timespan, notation)

    """
    timeSpans = [TimeSpan(beat0, beat1) for beat0, beat1 in iterlib.pairwise(offsets)]
    splittedEvents = []
    for ev in notations:
        if ev.duration > 0:
            splittedEvents.extend(ev.splitNotationAtOffsets(offsets))
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
                beatOffset:F, beatDur:F
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
        subdivNotations = [n for n in notations if now<=n.offset<subdivEnd and n.end<=subdivEnd]
        if subdiv == 1:
            items.extend(subdivNotations)
        else:
            items.append(_beatToTree(subdivNotations, division=subdiv, beatOffset=now, beatDur=dt))
        now += dt
    return Node(durRatio, items)


def _applyDurationRatio(notations:list[Notation],
                        division: int | division_t,
                        beatOffset: F,
                        beatDur: F
                        ) -> None:
    """
    Applies a totalDuration ratio to each notation, recursively.

    A totalDuration ratio converts the actual totalDuration of a notation to its
    notated value and is used to render these as tuplets later

    Args:
        notations: the notations inside the period beatOffset:beatOffset+beatDur
        division: the division of the beat/subbeat. Examples: 4, [3, 4], [2, 2, 3], etc
        beatOffset: the start of the beat
        beatDur: the totalDuration of the beat

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
            to the beginning of the measure. Offset and totalDuration are in
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
            raise ValueError(f"beat totalDuration not supported: {beatdur}")
        idx += 1

    quantizedBeats[0].weight = 2
    return QuantizedMeasure(timesig=timesig, quarterTempo=asF(quarterTempo), beats=quantizedBeats,
                            quantprofile=profile)


def splitNotationByMeasure(n: Notation, struct: st.ScoreStruct
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
    tieNotationParts(parts)

    sumdur = sum(struct.beatDelta((i, n.offset), (i, n.end)) for i, n in pairs)
    assert sumdur == n.duration, f"{n=}, {sumdur=}, {numMeasures=}\n{pairs=}"
    return pairs


def _mergeNodes(node1: Node, node2: Node,
                profile: QuantizationProfile,
                beatOffsets: list[F]
                ) -> Node:
    # we don't check here, just merge
    node = Node(ratio=node1.durRatio, items=node1.items + node2.items)
    node = node.mergedNotations()
    return _mergeSiblings(node, profile=profile, beatOffsets=beatOffsets)


def _nodesCanMerge(g1: Node, g2: Node, profile: QuantizationProfile,
                   beatOffsets: list[F]
                   ) -> Result:
    assert len(g1.items) > 0 and len(g2.items) > 0
    assert g1.offset < g1.end <= g2.offset
    assert g1.end == g2.offset
    acrossBeat = next((offset for offset in beatOffsets if g1.end == offset ), None)

    if g1.durRatio != g2.durRatio:
        return Result(False, "not same durRatio")
    if g1.durRatio != (1, 1):
        if acrossBeat and g1.durRatio[0] not in profile.allowedTupletsAcrossBeat:
            return Result(False, "tuplet not allowed to merge across beat")
        elif (g1.totalDuration() + g2.totalDuration() > profile.mergedTupletsMaxDuration):
            return Result(False, "incompatible totalDuration")
        elif not profile.mergeTupletsOfDifferentDuration and acrossBeat and g1.totalDuration() != g2.totalDuration():
            return Result(False, "Nodes of different totalDuration cannot merge")

    item1, item2 = g1.items[-1], g2.items[0]

    if isinstance(item1, Node) and isinstance(item2, Node):
        if not (r := _nodesCanMerge(item1, item2, profile=profile, beatOffsets=beatOffsets)):
            return Result(False, f'nested tuplets cannot merge: {r.info}')
        else:
            return Result(True, '')
    elif isinstance(item1, Node) or isinstance(item2, Node):
        return Result(False, 'A Node cannot merge with a single item')
    else:
        assert isinstance(item1, Notation) and isinstance(item2, Notation)
        if not acrossBeat:
            return Result(True, '')

        symdur = item1.symbolicDuration() + item2.symbolicDuration()

        if item1.gliss and item1.tiedNext and item2.gliss:
            if symdur >= 2 and item1.tiedPrev:
                return Result(False, "Cannot merge glissandi resulting in long (>= halfnote) notes")

        if not profile.mergeNestedTupletsAcrossBeats:
            g1nested = any(isinstance(item, Node) and item.durRatio != g1.durRatio for item in g1.items)
            if g1nested:
                return Result(False, "Cannot merge nested tuples")
            g2nested = any(isinstance(item, Node) and item.durRatio != g2.durRatio for item in g2.items)
            if g2nested:
                return Result(False, "Cannot merge nested tuples")

        #if not item1.canMergeWith(item2):
        #    return Result(False, f'Notation {item1} cannot merge with {item2}')
        if item1.duration + item2.duration < profile.minBeatFractionAcrossBeats:
            return Result(False, 'Absolute totalDuration of merged Notations across beat too short')

        if symdur < profile.minSymbolicDurationAcrossBeat:
            return Result(False, 'Symbolic totalDuration of merged notations across beat too short')

        return Result(True, '')

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
                mergednode = _mergeNodes(item1, item2, profile=profile, beatOffsets=beatOffsets)
                items[-1] = mergednode
            else:
                logger.debug(f'mergeSiblings: {item1} cannot merge with {item2}: {r.info}')
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
        return {2: 2, 4:2, 8: 1}[den]
    elif subdivision == 5:
        return 2 if den == 2 else 1
    else:
        return 1


@dataclass
class PartLocation:
    """
    Represents a location (a Notation inside a beat, inside a measure, inside a part)
    inside a part
    """
    measureIndex: int
    "The measure index"

    beatIndex: int
    "The beat index"

    notationIndex: int
    "The notation index within the beat"

    beat: QuantizedBeat
    "A reference to the beat"

    notation: Notation
    "The notation at this location"


@dataclass
class QuantizedPart:
    """
    A Part which has already been quantized following a ScoreStruct

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

    firstclef: str = ''
    """The first clef of this part"""

    def __post_init__(self):
        for measure in self.measures:
            measure.parent = self
            measure.quantprofile = self.quantprofile
        self._repairGracenotes()
        self._repairLinks()
        self.removeUnnecessarySpanners(tree=False)

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

    def findLogicalTieFromNotation(self, n: Notation) -> list[PartLocation] | None:
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

    def removeUnnecessaryDynamics(self, tree: bool, resetAfterEmptyMeasure=True) -> None:
        """
        Remove superfluous dynamics in this part, in place
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

    def removeUnnecessaryGracenotes(self, tree) -> None:
        """
        Removes unnecessary gracenotes

        An unnecessary gracenote are:

        * has the same pitch as the next real note and starts a glissando. Such gracenotes might
          be created during quantization.
        * has the same pitch as the previous real note and ends a glissando
        * n0/real -- gliss -- n1/grace n2/real and n1.pitches == n2.pitches

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

    def removeUnnecessarySpanners(self, tree: bool) -> None:
        """
        Remove unmatched spanners in this part

        Args:
            tree: if True, use the tree representation for each measure
        """
        _spanner.removeUnmatchedSpanners(self.flatNotations(tree=tree))
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
                            nextmeasure.beats[0].notations.insert(0, last.clone(offset=F(0)))
                    else:
                        # move gracenote to bext beat
                        nextbeat = measure.beats[beatidx + 1]
                        beat.notations.pop()
                        nextbeat.notations.insert(0, last)

    def getMeasure(self, idx: int) -> QuantizedMeasure:
        """
        Get a measure within this part

        Args:
            idx: the measure index (starts at 0)

        Returns:
            The corresponding measure. If outside of the defined measures a new
            empty QuantizedMeasure will be created
        """
        if idx > len(self.measures) - 1:
            for i in range(len(self.measures) - 1, idx+1):
                mdef = self.struct.getMeasureDef(i)
                qmeasure = QuantizedMeasure(timesig=mdef.timesig,
                                            quarterTempo=mdef.quarterTempo,
                                            quantprofile=self.quantprofile,
                                            parent=self)
                self.measures.append(qmeasure)
        return self.measures[idx]

    def _repairLinks(self):
        """
        Repairs ties and glissandi
        """
        for n0, n1 in iterlib.pairwise(self.flatNotations(tree=False)):
            if n0.tiedNext:
                if n0.isRest or not n0.pitches or n0.pitches != n1.pitches:
                    n0.tiedNext = False
                    n1.tiedPrev = False
                elif not n0.pitches or not n1.pitches or any(x not in n1.pitches for x in n0.pitches):
                    n0.tiedNext = False
                    n1.tiedPrev = False
                elif n1.isGracenote:
                    n0.tiedNext = False
                    n1.tiedPrev = False
            elif n0.gliss:
                if n1.isRest or n0.pitches == n1.pitches:
                    if n0.tiedPrev:
                        logicalTie = self.findLogicalTieFromNotation(n0)
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

    def breakSyncopations(self, level: str = 'weak') -> None:
        """
        Break notes extending over beat boundaries, in place

        * 'all': break syncopations at any beat boundary
        * 'weak': break syncopations at weak accent beats (for example, the 3rd
          beat in a 4/4 bar
        * 'strong': break syncopations only at strong beats

        Args:
            which: one of 'all', 'weak', 'strong'

        """
        for m in self.measures:
            m.breakSyncopations(level=level)


def quantizePart(part: core.Part,
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
            generated Part will have as many measures as are defined
            in the struct. Otherwise only as many measures as needed
            to hold the given events will be created
        quantprofile: the QuantizationProfile used

    Returns:
        a list of QuantizedMeasures. To convert these to a Part,
        call convertQuantizedMeasuresToPart

    """
    assert isinstance(part, core.Part)
    if isinstance(quantprofile, str):
        quantprofile = QuantizationProfile.fromPreset(quantprofile)
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
            qmeasures.append(QuantizedMeasure(timesig=measureDef.timesig,
                                              quarterTempo=measureDef.quarterTempo,
                                              quantprofile=quantprofile))
        else:
            notations.sort(key=lambda notation:notation.offset)
            core.removeOverlapInplace(notations)
            try:
                qmeasure = quantizeMeasure(notations,
                                           timesig=measureDef.timesig,
                                           quarterTempo=measureDef.quarterTempo,
                                           profile=quantprofile,
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
                                        quarterTempo=measureDef.quarterTempo,
                                        beats=[],
                                        quantprofile=quantprofile)
            qmeasures.append(qmeasure)
    qpart = QuantizedPart(struct, qmeasures, name=part.name, shortname=part.shortname,
                          groupid=part.groupid, quantprofile=quantprofile,
                          firstclef=part.firstclef)
    if quantprofile.breakSyncopationsLevel != 'none':
        for measure in qpart:
            measure.breakSyncopations(level=quantprofile.breakSyncopationsLevel)
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
        Finds the best spelling for each part in this score, in place

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
        return hash((self.title, self.composer) + tuple(partHashes))

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

    def groupParts(self) -> list[QuantizedPart | list[QuantizedPart]]:
        """
        Group parts which have the same id

        Returns:
            A list of groups where a tree is a list of parts with the same id
        """
        groups = {}
        out: list[QuantizedPart | list[QuantizedPart]] = []
        for part in self.parts:
            if part.groupid:
                if part.groupid not in groups:
                    group = [part]
                    groups[part.groupid] = group
                    out.append(group)
                else:
                    groups[part.groupid].append(part)
            else:
                out.append(part)
        return [item if isinstance(item, QuantizedPart) or len(item) > 1 else item[0]
                for item in out]

    def write(self,
              outfile: str,
              options: renderer.RenderOptions | None = None
              ) -> renderer.Renderer:
        """
        Export this score as pdf, png, lilypond, MIDI or musicxml
        Args:
            outfile:
            options:

        Returns:

        """
        ext = os.path.splitext(outfile)[1]
        if ext == '.ly':
            r = self.render(options=options, backend='lilypond')
            r.write(outfile)
            return r
        elif ext == '.pdf' or ext == '.png':
            r = self.render(options=options)
            r.write(outfile)
            return r
        else:
            raise ValueError(f"Format {ext} not supported")

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
            which divisions of the beat are possible, how a best division
            is weighted and selected, etc. Not all options in a preset
            are supported by all backends (for example, music21 backend
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
        >>> notations = [scoring.Notation(totalDuration=dur, pitches=[p]) for dur, p in notes]
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
