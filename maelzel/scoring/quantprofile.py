from __future__ import annotations
from math import sqrt
from functools import cache
from dataclasses import dataclass, field as _field, fields as _fields
from maelzel.scoring.common import F, division_t, number_t, logger
from maelzel.scoring import quantdata
from maelzel.scoring import quantutils
import copy
import pprint
from emlib import misc
from emlib import mathlib


def _factory(obj) -> _field:
    return _field(default_factory=lambda: copy.copy(obj))


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
    - syncopationMinBeatFraction: when merging durations across beats, a merged
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

    rhythmComplexityNotesAcrossSubdivisionWeight = 0.2
    """
    When calculating rhythm complexity this weight is applied to the penalty of notes extending
    over subdivisions of the beat (inner-beat syncopes)
    """

    rhythmComplexityIrregularDurationsWeight = 0.8
    """
    When calculating rhythm complexity this weight is applied to the penalty of notes whose
    duration is irregular (durations of 5 or 9 units, which need ties to be represented)
    """

    offsetErrorWeight: float = 1.0
    """Weight of the offset between original start and makeSnappedNotation start"""

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

    divisionCardinalityPenaltyMap: dict[int, float] = _factory({1: 0.0, 2: 0.1, 3: 0.4})
    """Penalty applied when different divisions are used within a beat 
    (e.g 4 where one 8 is a 3-plet and the other a 5-plet)"""

    numNestedTupletsPenalty: list[float] = _factory([0., 0.1, 0.4, 0.5, 0.8, 0.8])
    """Penalty applied to nested levels by level"""

    complexNestedTupletsFactor: float = 1.8
    """For certain combinations of nested tuplets an extra complexity factor can be applied.
    If this factor is 1.0, then no extra penalty is calculated. Any number above 1 will
    penalize complex nested tuplets (prefer (5, 5, 5) over (3, 3, 3, 3, 3)).
    """

    numSubdivsPenaltyMap: dict[int, float] = _factory({1: 0.0, 2: 0.0, 3: 0.0, 4: 0., 5: 0., 6: 0., 7: 0.})
    """Penalty applied to number of subdivisions, by number of subdivision"""

    divisionPenaltyWeight: float = 1.0
    """Weight of division penalty"""

    cardinalityPenaltyWeight: float = 0.1
    """Weight of cardinality"""

    numNestedTupletsPenaltyWeight: float = 1.0
    """Weight of sublevel penalty"""

    numSubdivisionsPenaltyWeight: float = 0.2
    """Weight to penalize the number of subdivisions"""

    syncopationMinBeatFraction: F = F(1, 6)
    """How long can a synchopation be, in terms of the length of the beat"""

    syncopationMinSymbolicDuration: F = F(1, 3)
    """Min. symbolic duration of a syncopation"""

    syncopationMaxAsymmetry: float = 2.0
    """The max. ratio between the longer and the shorter parts to be mergeable
    as a syncopation"""

    mergedTupletsMaxDuration: F = F(2)
    """How long can a tuplet over the beat be"""

    mergeTupletsOfDifferentDuration: bool = False
    """Allow merging tuplets which have different total durations?"""

    allowNestedTupletsAcrossBeat: bool = False
    """Allow merging nested tuplets across the beat"""

    allowedTupletsAcrossBeat: tuple[int, ...] = (1, 2, 3, 4, 5, 8)
    """Which tuplets are allowed to cross the beat"""

    allowedNestedTupletsAcrossBeat: list[tuple[int, int]] = _factory([(3, 3)])
    """Which nested tuplets are allowed to cross the beat?

    Nested tuplets are those which are non-binary at more than one level, like
    a triplet with a quintuplet inside. For example a value of (3, 3) indicates
    that a big triplet with a triplet inside which both go across the beat
    (for example the rhythm 2/3, 2/9, 2/9, 2/9, 2/3) would be allowed"""

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

    tiedSnappedGracenoteMinRealDuration: F = F(1, 1000000)
    """
    The min. real duration of a tied snapped gracenote in order for it NOT 
    to be removed
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
                                            maxdensity=max(int(maxDensity * 0.5), 8))
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
        cached = self._cachedDivisionsByTempo.get((tempo, self.nestedTuplets))
        if cached:
            return cached

        divsByTempo = self.possibleDivisionsByTempo
        divs = None
        for maxTempo, possibleDivs in divsByTempo.items():
            if tempo < maxTempo:
                divs = possibleDivs
                break
        if not divs:
            logger.error("Possible divisions of the beat, by tempo: ")
            logger.error(pprint.pformat(self.possibleDivisionsByTempo))
            raise ValueError(f"No divisions for the given tempo (q={int(tempo)})")

        if not self.nestedTuplets:
            divs = [div for div in divs if not quantutils.isNestedTupletDivision(div)]
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
            return cached

        division = tuple(sorted(division))
        if (cached := self._cachedDivisionPenalty.get(division)) is not None:
            return cached

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


defaultQuantizationProfile = QuantizationProfile()


@cache
def _divisionCardinality(division: division_t, excludeBinary=False) -> int:
    # TODO: make general form for deeply nested tuplets
    if isinstance(division, int):
        return 1

    allfactors = quantutils.primeFactors(len(division), excludeBinary=excludeBinary).copy()
    for subdiv in division:
        allfactors.update(quantutils.primeFactors(subdiv, excludeBinary=excludeBinary))
    if 1 in allfactors:
        allfactors.remove(1)
    return len(allfactors)


@cache
def _divisionDepth(division: division_t) -> int:
    # TODO: make general form for deeply nested tuplets
    if isinstance(division, int):
        return 1
    if mathlib.ispowerof2(len(division)):
        return 1
    if all(mathlib.ispowerof2(subdiv) for subdiv in division):
        return 1
    return 2


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
        a tuple (penalty, info), where penalty is the penalty associated with
        this division, based on the division only (not on how the division
        fits the notes in the beat).

    """
    assert isinstance(division, int) or (
                isinstance(division, tuple) and all(isinstance(x, int) for x in division)), f"{division=}"

    if isinstance(division, int):
        divPenalty = profile.divisionPenaltyMap.get(division, maxPenalty)
        numSubdivsPenalty = profile.numSubdivsPenaltyMap[1]
        cardinality = 1
    else:
        internalPenalty = sqrt(sum(_divisionPenalty(subdiv, profile, nestingLevel+1, maxPenalty=maxPenalty)[0]**2
                                   for subdiv in division))
        divPenalty = internalPenalty + profile.divisionPenaltyMap.get(len(division), maxPenalty)
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

