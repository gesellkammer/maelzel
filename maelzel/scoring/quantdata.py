from __future__ import annotations
from functools import cache
from dataclasses import dataclass, replace as _dataclass_replace, fields as _fields
from emlib.common import runonce

from maelzel.scoring.common import division_t
from maelzel.common import F
import itertools
import math

from typing import Sequence


@dataclass
class DivisionDef:
    maxTempo: int
    maxSubdivisions: int
    possibleValues: tuple[int, ...]
    maxDensity: int

    @cache
    def subdivisions(self, blacklist: tuple[division_t, ...] = ()) -> list[division_t]:
        subdivs = allSubdivisions(maxSubdivs=self.maxSubdivisions,
                                  possibleValues=self.possibleValues,
                                  maxDensity=self.maxDensity,
                                  skipSubdivs=())
        if not blacklist:
            return subdivs
        blacklistset = set(blacklist)
        return [subdiv for subdiv in subdivs if subdiv not in blacklistset]

    def __hash__(self):
        return hash((self.maxTempo, self.maxSubdivisions, hash(self.possibleValues), self.maxDensity))


@dataclass
class QuantPreset:
    divisionDefs: tuple[DivisionDef, ...]
    divisionsPenaltyMap: dict[int, float]
    nestedTuplets: bool
    numNestedTupletsPenalty: tuple[float, ...]
    gridErrorWeight: float
    divisionErrorWeight: float
    rhythmComplexityWeight: float
    gridErrorExp: float
    exactGridFactor: float = 1.
    maxDivPenalty: float | None = None
    cardinalityPenaltyWeight: float | None = None
    numSubdivisionsPenaltyWeight: float | None = None
    syncopExcludeSymDurs: tuple[int, ...] | None = None
    maxGraceRatio: float | None = None
    _cachedDivsByTempo: dict[int, tuple[division_t, ...]] | None = None

    def clone(self, **kws) -> QuantPreset:
        return _dataclass_replace(self, **kws)

    def __post_init__(self):
        assert 0 <= self.exactGridFactor <= 1

    @classmethod
    @runonce
    def keys(cls) -> set[str]:
        allkeys = set(field.name for field in _fields(QuantPreset)
                      if not field.name.startswith("_"))
        allkeys ^= {'divisionsPenaltyMap'}
        return allkeys


@cache
def getPresets() -> dict[str, QuantPreset]:
    presets = {
        'exact': QuantPreset(
            divisionDefs = (
                DivisionDef(maxTempo=60,
                    maxSubdivisions=9,
                    possibleValues=(4, 5, 6, 7, 8, 9, 11, 13, 15, 17),
                    maxDensity=42),
                DivisionDef(maxTempo=90,
                    maxSubdivisions=8,
                    possibleValues=(4, 5, 6, 7, 8, 9, 11, 13, 15, 17),
                    maxDensity=36),
                DivisionDef(maxTempo=180,
                    maxSubdivisions=8,
                    possibleValues=(4, 5, 6, 7, 8, 9, 11, 13),
                    maxDensity=28),
                DivisionDef(maxTempo=360,
                    maxSubdivisions=6,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=24),
                DivisionDef(maxTempo=800,
                    maxSubdivisions=3,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=20)),
            divisionsPenaltyMap={
                1:0.0,
                2:0.0,
                3:0.0,
                4:0.01,
                5:0.01,
                6:0.02,
                7:0.01,
                8:0.01,
                9:0.04,
                10:0.04,
                11:0.1,
                12:0.1,
                13:0.2,
                14:0.1,
                15:0.2,
                16:0.4,
            },
            # divisionsPenaltyMap=defaultDivisionPenaltyMap,
            nestedTuplets=True,
            numNestedTupletsPenalty=(0., 0., 0., 0.1, 0.4, 0.8),
            gridErrorWeight=2.0,
            divisionErrorWeight=0.001,
            rhythmComplexityWeight=0.005,
            cardinalityPenaltyWeight=0.001,
            numSubdivisionsPenaltyWeight=0.00,
            gridErrorExp=0.7,
            maxDivPenalty=0.1,
            exactGridFactor=0.01,
            maxGraceRatio=2.0,
        ),
        'highest': QuantPreset(
            divisionDefs = (
                DivisionDef(maxTempo=48,
                    maxSubdivisions=9,
                    possibleValues=(4, 5, 6, 7, 8, 9, 11, 13, 15, 17),
                    maxDensity=42),
                DivisionDef(maxTempo=63,
                    maxSubdivisions=8,
                    possibleValues=(4, 5, 6, 7, 8, 9, 11, 13, 15, 17),
                    maxDensity=30),
                DivisionDef(maxTempo=88,
                    maxSubdivisions=8,
                    possibleValues=(4, 5, 6, 7, 8, 9, 11, 13),
                    maxDensity=28),
                DivisionDef(maxTempo=120,
                    maxSubdivisions=8,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9, 11, 13),
                    maxDensity=26),
                DivisionDef(maxTempo=240,
                    maxSubdivisions=6,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=24),
                DivisionDef(maxTempo=800,
                    maxSubdivisions=3,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=20)),
            divisionsPenaltyMap={
                1:0.0,
                2:0.0,
                3:0.0,
                4:0.01,
                5:0.01,
                6:0.02,
                7:0.01,
                8:0.01,
                9:0.04,
                10:0.04,
                11:0.1,
                12:0.1,
                13:0.2,
                14:0.1,
                15:0.2,
                16:0.4,
            },
            # divisionsPenaltyMap=defaultDivisionPenaltyMap,
            nestedTuplets=True,
            numNestedTupletsPenalty=(0., 0., 0., 0.1, 0.4, 0.8),
            gridErrorWeight=2.0,
            divisionErrorWeight=0.001,
            rhythmComplexityWeight=0.005,
            cardinalityPenaltyWeight=0.001,
            numSubdivisionsPenaltyWeight=0.00,
            gridErrorExp=0.7,
            maxDivPenalty=0.1,
            exactGridFactor=0.1,
            maxGraceRatio=2.0,
        ),
        'high': QuantPreset(
            divisionDefs = (
                DivisionDef(maxTempo=48,
                    maxSubdivisions=8,
                    possibleValues=(3, 5, 6, 7, 8, 9, 11, 13, 15),
                    maxDensity=30),
                DivisionDef(maxTempo=56,
                    maxSubdivisions=8,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9, 11, 13, 15),
                    maxDensity=28),
                DivisionDef(maxTempo=66,
                    maxSubdivisions=6,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9, 11, 15),
                    maxDensity=26),
                DivisionDef(maxTempo=80,
                    maxSubdivisions=6,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=22),
                DivisionDef(maxTempo=100,
                    maxSubdivisions=4,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9),
                    maxDensity=20),
                DivisionDef(maxTempo=132,
                    maxSubdivisions=4,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9),
                    maxDensity=16),
                DivisionDef(maxTempo=180,
                    maxSubdivisions=3,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9),
                    maxDensity=10),
                DivisionDef(maxTempo=400,
                    maxSubdivisions=3,
                    possibleValues=(2, 3, 4, 5, 6),
                    maxDensity=8),
                DivisionDef(maxTempo=800,
                    maxSubdivisions=1,
                    possibleValues=(2, 3, 4, 5, 6),
                    maxDensity=6)),
            divisionsPenaltyMap=defaultDivisionPenaltyMap,
            nestedTuplets=True,
            numNestedTupletsPenalty=(0., 0., 0.03, 0.4, 0.5, 0.8),
            gridErrorWeight=1.0,
            divisionErrorWeight=0.0001,
            rhythmComplexityWeight=0.1,
            cardinalityPenaltyWeight=0,
            gridErrorExp=0.75,
            maxDivPenalty=0.2,
            exactGridFactor=0.25,
            maxGraceRatio=2.0),
        'medium': QuantPreset(
            divisionDefs = (
                DivisionDef(maxTempo=60,
                    maxSubdivisions=4,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=24),
                DivisionDef(maxTempo=80,
                    maxSubdivisions=4,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=18),
                DivisionDef(maxTempo=100,
                    maxSubdivisions=3,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                    maxDensity=12),
                DivisionDef(maxTempo=132,
                    maxSubdivisions=3,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8),
                    maxDensity=8),
                DivisionDef(maxTempo=180,
                    maxSubdivisions=2,
                    possibleValues=(1, 2, 3, 4, 5, 6, 8),
                    maxDensity=6),
                DivisionDef(maxTempo=400,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 3, 4),
                    maxDensity=4),
                DivisionDef(800,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 3, 4),
                    maxDensity=4)),
            divisionsPenaltyMap=defaultDivisionPenaltyMap,
            nestedTuplets=False,
            numNestedTupletsPenalty=(0, 0., 0.05, 0.4, 0.5, 0.8),
            gridErrorWeight=1.0,
            divisionErrorWeight=0.01,
            rhythmComplexityWeight=0.01,
            gridErrorExp=0.9,
            maxDivPenalty=0.2),
        'low': QuantPreset(
            divisionDefs = (
                DivisionDef(maxTempo=60,
                    maxSubdivisions=3,
                    possibleValues=(1, 2, 3, 4, 5, 6, 8),
                    maxDensity=20),
                DivisionDef(maxTempo=80,
                    maxSubdivisions=3,
                    possibleValues=(1, 2, 3, 4, 5, 6, 8),
                    maxDensity=16),
                DivisionDef(maxTempo=100,
                    maxSubdivisions=2,
                    possibleValues=(1, 2, 3, 4, 5, 6, 8),
                    maxDensity=12),
                DivisionDef(maxTempo=132,
                    maxSubdivisions=2,
                    possibleValues=(1, 2, 3, 4, 6, 8),
                    maxDensity=8),
                DivisionDef(maxTempo=180,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 3, 4, 6, 8),
                    maxDensity=6),
                DivisionDef(maxTempo=400,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 3, 4),
                    maxDensity=4),
                DivisionDef(800,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 4),
                    maxDensity=4)),
            divisionsPenaltyMap=defaultDivisionPenaltyMap,
            nestedTuplets=False,
            numNestedTupletsPenalty=(0, 0., 0.05, 0.4, 0.5, 0.8),
            gridErrorWeight=1.0,
            divisionErrorWeight=0.5,
            rhythmComplexityWeight=0.1,
            syncopExcludeSymDurs=(3, 7, 15),
            gridErrorExp=1.),
        'speech': QuantPreset(
            divisionDefs = (
                DivisionDef(maxTempo=60,
                    maxSubdivisions=2,
                    possibleValues=(2, 3, 4, 5, 6, 8),
                    maxDensity=16),
                DivisionDef(maxTempo=80,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 3, 4, 5, 6, 8),
                    maxDensity=14),
                DivisionDef(maxTempo=100,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 3, 4, 5, 6, 8),
                    maxDensity=12),
                DivisionDef(maxTempo=132,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 3, 4, 6, 8),
                    maxDensity=8),
                DivisionDef(maxTempo=180,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 3, 4, 6, 8),
                    maxDensity=6),
                DivisionDef(maxTempo=400,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 3, 4),
                    maxDensity=4),
                DivisionDef(800,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 4),
                    maxDensity=4)),
            divisionsPenaltyMap=defaultDivisionPenaltyMap,
            nestedTuplets=False,
            numNestedTupletsPenalty=(0, 0., 0.05, 0.4, 0.5, 0.8),
            gridErrorWeight=1.0,
            divisionErrorWeight=0.01,
            rhythmComplexityWeight=0.4,
            syncopExcludeSymDurs=(3, 7, 15),
            gridErrorExp=1.,
            maxGraceRatio=4.),
    }
    return presets


def subdivisionsWithMaxDensity(numSubdivisions: int,
                               maxDensity: int,
                               possibleValues: Sequence[int]):
    return [comb for comb in itertools.combinations_with_replacement(possibleValues, numSubdivisions)
            if sum(comb) <= maxDensity]


def _subdivisionsWithMaxDensity(numSubdivisions: int,
                                maxDensity: int,
                                possibleValues: Sequence[int]):
    # C(n, r) where n=len(possibleValues) and r=numSubdivisions
    # and the sum of possibleValues[indexes] is <= maxDensity
    # Order is not important. Repetition is allowed
    # so C(n+r-1, n-1). We use itertools, which is the same
    maxLocalDensity = maxDensity * 2
    minval = min(possibleValues)
    maxval = int(round(maxLocalDensity / numSubdivisions))
    maxval = min(maxval, maxDensity - (numSubdivisions - 1) * minval)
    values = list(possibleValues)
    if max(values) > maxval:
        values = [_ for _ in possibleValues if _ <= maxval]
    values.sort()
    return [comb for comb in itertools.combinations_with_replacement(values, numSubdivisions)
            if sum(comb) <= maxDensity]


def subdivisions(numSubdivisions: int,
                 maxValue: int,
                 possibleValues: set[int] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14},
                 ) -> list[tuple[int, ...]]:
    """
    Generate all possible subdivisions.

    To obtain all possible grids these need to be permutated (the generated
    divisions are always sorted from high to low)
    
    Args:
        numSubdivisions: number of divisions
        possibleValues: possible value for each division
        maxValue: the max. allowed value from possiblevals

    Returns:
        a list of divisions
    """
    minValue = 1
    used = set()
    out = []
    for i in range(maxValue, minValue - 1, -1):
        if i not in possibleValues or any(x % i == 0 for x in used):
            continue
        used.add(i)
        if numSubdivisions == 1:
            out.append((i,))
        else:
            subdivs = subdivisions(numSubdivisions - 1, possibleValues=possibleValues, maxValue=i)
            for subdiv in subdivs:
                out.append((i,) + subdiv)
    return out


def permutateDivisions(divs: Sequence[division_t]) -> list[division_t]:
    out = []
    for p in divs:
        if len(p) == 1:
            out.append(p)
        else:
            out.extend(set(itertools.permutations(p)))
    return out


def isSuperfluousDivision(div: division_t, possiblevals: set[int]) -> bool:
    lendiv = len(div)
    if lendiv == 1:
        return False
    p0 = div[0]
    sumdiv = sum(div)
    if sumdiv % 2 == 0 and sumdiv in possiblevals and all(x == p0 for x in div):
        # (4, 4) == (2, 2, 2, 2) == 8
        # We exclude cases like (3, 3, 3) == 9 and (5, 5, 5) == 15, but include (3, 3) == 6
        return True
    if lendiv in (2, 4, 8) and all(x in (1, 2, 4, 8) for x in div) and max(div)*lendiv in possiblevals:
        # (2, 4) == (4, 4) == 8, (2, 2, 2, 4) == 16, (4, 4) == 8
        return True
    return False


def allSubdivisions(maxSubdivs: int,
                    maxDensity: int,
                    possibleValues: Sequence[int],
                    blacklist: Sequence[division_t] = (),
                    method='average',
                    skipSubdivs=()
                    ) -> list[division_t]:

    allSubdivs: list[division_t] = []
    possibleValsSet = set(possibleValues)
    for numSubdivs in range(maxSubdivs, 0, -1):
        if numSubdivs in skipSubdivs:
            continue
        if method == 'average':
            allSubdivs.extend(subdivisionsWithMaxDensity(numSubdivisions=numSubdivs, maxDensity=maxDensity, possibleValues=possibleValues))
        else:
            maxval = int(round(maxDensity / numSubdivs))
            allSubdivs.extend(subdivisions(numSubdivisions=numSubdivs, maxValue=maxval, possibleValues=possibleValsSet))

    allSubdivs = [s for s in allSubdivs if not isSuperfluousDivision(s, possibleValsSet)]
    allSubdivs = list(set(permutateDivisions(allSubdivs)))

    if blacklist:
        blacklist = permutateDivisions(blacklist)
        blacklistset = set(blacklist)
        allSubdivs = [div for div in allSubdivs if div not in blacklistset]

    allSubdivs.sort()
    allSubdivs.sort(key=lambda p: len(p))
    # allSubdivs.sort(key=lambda p: sum(p))
    return allSubdivs


regularDurations = {0, 1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32}


defaultDivisionPenaltyMap = {
    1:0.0,
    2:0.0,
    3:0.0,
    4:0.01,
    5:0.01,
    6:0.01,
    7:0.01,
    8:0.01,
    9:0.04,
    10:0.04,
    11:0.1,
    12:0.1,
    13:0.2,
    14:0.1,
    15:0.1,
    16:0.4,
}


@cache
def divisionsByTempo(divisionDefs: tuple[DivisionDef, ...], 
                     blacklist: tuple[division_t, ...] = ()
                     ) -> dict[int, tuple[division_t, ...]]:
    return {d.maxTempo: d.subdivisions(blacklist=blacklist) for d in divisionDefs}




# how to divide an irregular duration into regular parts
# Regular durations are those which can be expressed via
# a quarter, eighth, 1/16 note, or any dotted or double
# dotted variation thereof
slotDivisionStrategy = {
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


@cache
def splitIrregularSlots(numslots: int, slotindex: int) -> tuple[int, ...]:
    slotDivisions = slotDivisionStrategy[numslots]
    if slotindex % 2 == 1 and slotDivisions[-1] % 2 == 1:
        slotDivisions = tuple(reversed(slotDivisions))
    return slotDivisions


def divisionToRatio(den: int) -> tuple[int, int]:
    """
    Calculate the duration ratio corresponding to the given numerator

    The returned ratio converts a duration back to its representation
    If a notation has an effective duration of 1/5 (one 16th of a 5-subdivision),
    # applying the ratio 5/4 will convert it to 1/4, i.e, a 16th note
    # the ratio can then be used to generate the needed subdivision by the notation
    # backend    
    
    Args:
        den: The denominator 

    Returns:
        a tuple (num, den) representing the ratio to convert a duration
        back to its representation.

    """
    if (den & (den - 1)) == 0:  # is it a power of 2?
        return (1, 1)
    # prev power of two
    den = 2 ** int(math.log2(den))
    ratio = F(den, den)
    return ratio.numerator, ratio.denominator
    
    
# these are ratios to convert a duration back to its representation
# if a notation has an effective duration of 1/5 (one 16th of a 5-subdivision),
# applying the ratio 5/4 will convert it to 1/4, i.e, a 16th note
# the ratio can then be used to generate the needed subdivision by the notation
# backend
durationRatios = {
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
