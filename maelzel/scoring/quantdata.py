from __future__ import annotations
from functools import cache
from dataclasses import dataclass, replace as _dataclass_replace

from maelzel.scoring.common import division_t
from maelzel.common import F
import itertools
import math

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence


def subdivisions(numdivs: int,
                 maxval: int,
                 possiblevals: set[int] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14},
                 ) -> list[tuple[int, ...]]:
    """
    Generate all possible subdivisions.

    To obtain all possible grids these need to be permutated (the generated
    divisions are always sorted from high to low)
    
    Args:
        numdivs: number of divisions 
        possiblevals: possible value for each division
        maxval: the max. allowed value from possiblevals

    Returns:
        a list of divisions
    """
    minval = 1
    used = set()
    out = []
    for i in range(maxval, minval-1, -1):
        if i not in possiblevals or any(x % i == 0 for x in used):
            continue
        used.add(i)
        if numdivs == 1:
            out.append((i,))
        else:
            subdivs = subdivisions(numdivs - 1, possiblevals=possiblevals, maxval=i)
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


def allSubdivisions(maxsubdivs=5,
                    maxdensity=20,
                    possiblevals: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14),
                    blacklist: Sequence[division_t] = ()
                    ) -> list[division_t]:

    allsubdivs: list[division_t] = []
    possibleValsSet = set(possiblevals)
    for numsubdivs in range(maxsubdivs, 0, -1):
        maxval = int(round(maxdensity / numsubdivs))
        allsubdivs.extend(subdivisions(numdivs=numsubdivs, maxval=maxval, possiblevals=possibleValsSet))

    allsubdivs = [s for s in allsubdivs if not isSuperfluousDivision(s, possibleValsSet)]
    allsubdivs = permutateDivisions(allsubdivs)

    if blacklist:
        blacklist = permutateDivisions(blacklist)
        blacklistset = set(blacklist)
        allsubdivs = [div for div in allsubdivs if div not in blacklistset]

    allsubdivs.sort(key=lambda p: sum(p))
    return allsubdivs


# A quantization preset consists of presetname.key, where needed keys are:
# divisionsByTempo and divisionPenaltyMap


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


@dataclass
class DivisionDef:
    maxTempo: int
    maxSubdivisions: int
    possibleValues: tuple[int, ...]
    maxDensity: int

    @cache
    def subdivisions(self, blacklist: tuple[division_t, ...] = ()) -> tuple[division_t, ...]:
        subdivs = allSubdivisions(maxsubdivs=self.maxSubdivisions,
                                  possiblevals=self.possibleValues,
                                  maxdensity=self.maxDensity)
        if not blacklist:
            return tuple(subdivs)
        else:
            blacklistset = set(blacklist)
            return tuple(subdiv for subdiv in subdivs if subdiv not in blacklistset)

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
    maxDivPenalty: float | None = None
    cardinalityPenaltyWeight: float | None = None
    numSubdivisionsPenaltyWeight: float | None = None
    _cachedDivsByTempo: dict[int, tuple[division_t, ...]] | None = None

    def clone(self, **kws) -> QuantPreset:
        return _dataclass_replace(self, **kws)


@cache
def divisionsByTempo(divisionDefs: tuple[DivisionDef, ...], 
                     blacklist: tuple[division_t, ...] = ()
                     ) -> dict[int, tuple[division_t, ...]]:
    return {d.maxTempo: d.subdivisions(blacklist=blacklist) for d in divisionDefs}


@cache
def getPresets() -> dict[str, QuantPreset]:
    presets = {
        'highest': QuantPreset(
            divisionDefs = (
                DivisionDef(maxTempo=62,
                    maxSubdivisions=7,
                    possibleValues=(3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=40),
                DivisionDef(maxTempo=120,
                    maxSubdivisions=7,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=32),
                DivisionDef(maxTempo=300,
                    maxSubdivisions=6,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=50),
                DivisionDef(maxTempo=800,
                    maxSubdivisions=3,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=28)),
            divisionsPenaltyMap={
                1:0.0,  
                2:0.0,  
                3:0.0,  
                4:0.01, 
                5:0.01,
                6:0.02, 7:0.01, 8:0.01, 9:0.04, 10:0.04,
                11:0.1, 12:0.1, 13:0.2, 14:0.1, 15:0.2,
                16:0.4,
            },
            nestedTuplets=True,
            numNestedTupletsPenalty=(0., 0., 0., 0.1, 0.4, 0.8),
            gridErrorWeight=2.0,
            divisionErrorWeight=0.0,
            rhythmComplexityWeight=0.001,
            numSubdivisionsPenaltyWeight=0.,
            gridErrorExp=0.7,
            maxDivPenalty=0.4),
        'high': QuantPreset(
            divisionDefs = (
                DivisionDef(maxTempo=48,
                    maxSubdivisions=7,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13),
                    maxDensity=32),
                DivisionDef(maxTempo=63,
                    maxSubdivisions=7,
                    possibleValues=(3, 5, 6, 7, 8, 9, 10, 11, 13),
                    maxDensity=28),
                DivisionDef(maxTempo=80,
                    maxSubdivisions=5,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=24),
                DivisionDef(maxTempo=100,
                    maxSubdivisions=4,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                    maxDensity=16),
                DivisionDef(maxTempo=132,
                    maxSubdivisions=4,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                    maxDensity=12),
                DivisionDef(maxTempo=180,
                    maxSubdivisions=3,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                    maxDensity=10),
                DivisionDef(maxTempo=400,
                    maxSubdivisions=3,
                    possibleValues=(1, 2, 3, 4, 5, 6),
                    maxDensity=8),
                DivisionDef(maxTempo=800,
                    maxSubdivisions=1,
                    possibleValues=(1, 2, 3, 4, 5, 6),
                    maxDensity=6)),
            divisionsPenaltyMap=defaultDivisionPenaltyMap,
            nestedTuplets=True,
            numNestedTupletsPenalty=(0., 0., 0.03, 0.4, 0.5, 0.8),
            gridErrorWeight=1.0,
            divisionErrorWeight=0.005,
            rhythmComplexityWeight=0.005,
            cardinalityPenaltyWeight=0,
            gridErrorExp=0.75,
            maxDivPenalty=0.2),
        'medium': QuantPreset(
            divisionDefs = (
                DivisionDef(maxTempo=60,
                    maxSubdivisions=4,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=20),
                DivisionDef(maxTempo=80,
                    maxSubdivisions=4,
                    possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                    maxDensity=16),
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
                    maxSubdivisions=4,
                    possibleValues=(1, 2, 3, 4, 5, 6, 8),
                    maxDensity=16),
                DivisionDef(maxTempo=80,
                    maxSubdivisions=3,
                    possibleValues=(1, 2, 3, 4, 5, 6, 8),
                    maxDensity=14),
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
            gridErrorExp=1.),
    }
    return presets


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
