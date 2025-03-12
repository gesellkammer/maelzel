from __future__ import annotations
from . import quantutils
from functools import cache
from dataclasses import dataclass
from maelzel.scoring.common import division_t


# A quantization preset consists of presetname.key, where needed keys are:
# divisionsByTempo and divisionPenaltyMap


regularDurations = {0, 1, 2, 3, 4, 6, 7, 8, 12, 16, 24, 32}


defaultDivisionPenaltyMap = {
    1:0.0,
    2:0.0,
    3:0.0,
    4:0.01,
    5:0.02,
    6:0.01,
    7:0.02,
    8:0.01,
    9:0.05,
    10:0.06,
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
    def subdivisions(self, blacklist: tuple[division_t] | None = None) -> tuple[division_t, ...]:
        subdivs = quantutils.allSubdivisions(maxsubdivs=self.maxSubdivisions,
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


@cache
def divisionsByTempo(divisionDefs: tuple[DivisionDef, ...], blacklist: tuple[division_t, ...] = None
                     ) -> dict[int, tuple[division_t, ...]]:
    return {d.maxTempo: d.subdivisions(blacklist=blacklist) for d in divisionDefs}


quantpresets = {
    'highest': QuantPreset(
        divisionDefs = (
            DivisionDef(maxTempo=62,
                maxSubdivisions=6,
                possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15),
                maxDensity=30),
            DivisionDef(maxTempo=300,
                maxSubdivisions=5,
                possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15),
                maxDensity=29),
            DivisionDef(maxTempo=800,
                maxSubdivisions=3,
                possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                maxDensity=28)),
        divisionsPenaltyMap={
            1:0.0,  2:0.0,  3:0.0,  4:0.01, 5:0.02,
            6:0.02, 7:0.02, 8:0.01, 9:0.04, 10:0.04,
            11:0.1, 12:0.1, 13:0.2, 14:0.1, 15:0.2,
            16:0.4,
        },
        nestedTuplets=True,
        numNestedTupletsPenalty=(0., 0., 0., 0.1, 0.4, 0.8),
        gridErrorWeight=1.0,
        divisionErrorWeight=0.002,
        rhythmComplexityWeight=0.0001,
        numSubdivisionsPenaltyWeight=0.,
        gridErrorExp=0.7,
        maxDivPenalty=0.4),
    'high': QuantPreset(
        divisionDefs = (
            DivisionDef(maxTempo=48,
                maxSubdivisions=6,
                possibleValues=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                maxDensity=30),
            DivisionDef(maxTempo=63,
                maxSubdivisions=6,
                possibleValues=(3, 5, 6, 7, 8, 9, 11),
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
        divisionErrorWeight=0.01,
        rhythmComplexityWeight=0.001,
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
        gridErrorExp=1.)
}



# Presets used to create a QuantizationProfile
_oldpresets = {
    'highest': {
        'possibleDivisionsByTempo': {
            10: [],
            62: quantutils.allSubdivisions(maxsubdivs=6,
                                           possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15),
                                           maxdensity=30),

            300: quantutils.allSubdivisions(maxsubdivs=5,
                                            possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15),
                                            maxdensity=29),
            800: quantutils.allSubdivisions(maxsubdivs=3,
                                            possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                                            maxdensity=28),
        },
        'divisionPenaltyMap': {
            1:0.0,  2:0.0,  3:0.0,  4:0.01, 5:0.02,
            6:0.02, 7:0.02, 8:0.01, 9:0.04, 10:0.04,
            11:0.1, 12:0.1, 13:0.2, 14:0.1, 15:0.2,
            16:0.4,
       },
        'nestedTuplets': True,
        'numNestedTupletsPenalty': (0.0, 0.0, 0.0, 0.1, 0.4, 0.8),
        'gridErrorWeight': 1.0,
        'divisionErrorWeight': 0.002,
        'rhythmComplexityWeight': 0.0001,
        'numSubdivisionsPenaltyWeight': 0.,
        'gridErrorExp': 0.7,
        'maxDivPenalty': 0.4,
    },
    'high': {
        'possibleDivisionsByTempo': {
            10: [],
            48: quantutils.allSubdivisions(maxsubdivs=6,
                                           possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                                           maxdensity=30),
            63: quantutils.allSubdivisions(maxsubdivs=6,
                                           # possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                                           possiblevals=(3, 5, 6, 7, 8, 9, 11),
                                           maxdensity=28),
            80: quantutils.allSubdivisions(maxsubdivs=5,
                                           possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                                           maxdensity=24),
            100: quantutils.allSubdivisions(maxsubdivs=4,
                                            possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                                            maxdensity=16),
            132: quantutils.allSubdivisions(maxsubdivs=4,
                                            possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                                            maxdensity=12),
            180: quantutils.allSubdivisions(maxsubdivs=3,
                                            possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                                            maxdensity=10),
            # 100 for 3/16
            400: quantutils.allSubdivisions(maxsubdivs=3,
                                            possiblevals=(1, 2, 3, 4, 5, 6),
                                            maxdensity=8),
            # 200 for 3/16
            800: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 3, 4, 5, 6),
                                            maxdensity=6),
        },
        'divisionPenaltyMap': defaultDivisionPenaltyMap,
        'nestedTuplets': True,
        'numNestedTupletsPenalty': (0., 0., 0.03, 0.4, 0.5, 0.8),
        'gridErrorWeight': 1.0,
        'divisionErrorWeight': 0.01,
        'rhythmComplexityWeight': 0.001,
        'cardinalityPenaltyWeight': 0,
        'gridErrorExp': 0.75,
        'maxDivPenalty': 0.2,
    },
    'medium': {
        'possibleDivisionsByTempo': {
            10: [],
            60: quantutils.allSubdivisions(maxsubdivs=4,
                                           possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                                           maxdensity=20),
            80: quantutils.allSubdivisions(maxsubdivs=4,
                                           possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9, 11),
                                           maxdensity=16),
            100: quantutils.allSubdivisions(maxsubdivs=3,
                                            possiblevals=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                                            maxdensity=12),
            132: quantutils.allSubdivisions(maxsubdivs=3,
                                            possiblevals=(1, 2, 3, 4, 5, 6, 7, 8),
                                            maxdensity=8),
            180: quantutils.allSubdivisions(maxsubdivs=2,
                                            possiblevals=(1, 2, 3, 4, 5, 6, 8),
                                            maxdensity=6),
            400: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 3, 4),
                                            maxdensity=4),
            800: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 3, 4),
                                            maxdensity=4),

        },
        'divisionPenaltyMap': defaultDivisionPenaltyMap,
        'nestedTuplets': False,
        'numNestedTupletsPenalty': (0, 0.0, 0.05, 0.4, 0.5, 0.8),
        'gridErrorWeight': 1.0,
        'divisionErrorWeight': 0.01,
        'rhythmComplexityWeight': 0.01,
        'gridErrorExp': 0.9,
        'maxDivPenalty': 0.2,
    },
    'low': {
        'possibleDivisionsByTempo': {
            10: [],
            60: quantutils.allSubdivisions(maxsubdivs=4,
                                           possiblevals=(1, 2, 3, 4, 5, 6, 8),
                                           maxdensity=16),
            80: quantutils.allSubdivisions(maxsubdivs=3,
                                           possiblevals=(1, 2, 3, 4, 5, 6, 8),
                                           maxdensity=14),
            100: quantutils.allSubdivisions(maxsubdivs=2,
                                            possiblevals=(1, 2, 3, 4, 5, 6, 8),
                                            maxdensity=12),
            132: quantutils.allSubdivisions(maxsubdivs=2,
                                            possiblevals=(1, 2, 3, 4, 6, 8),
                                            maxdensity=8),
            180: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 3, 4, 6, 8),
                                            maxdensity=6),
            400: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 3, 4),
                                            maxdensity=4),
            800: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 4),
                                            maxdensity=4),

        },
        'divisionPenaltyMap': defaultDivisionPenaltyMap,
        'nestedTuplets': False,
        'numNestedTupletsPenalty': [0, 0.0, 0.05, 0.4, 0.5, 0.8],
        'gridErrorWeight': 1.0,
        'divisionErrorWeight': 0.5,
        'rhythmComplexityWeight': 0.1,
        'gridErrorExp': 1,
    },
    'lowest': {
        'possibleDivisionsByTempo': {
            10: [],
            60: quantutils.allSubdivisions(maxsubdivs=2,
                                           possiblevals=(1, 2, 3, 4, 5, 6, 8),
                                           maxdensity=16),
            80: quantutils.allSubdivisions(maxsubdivs=2,
                                           possiblevals=(1, 2, 3, 4, 5, 6, 8),
                                           maxdensity=14),
            100: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 3, 4, 5, 6, 8, 12),
                                            maxdensity=12),
            132: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 3, 4, 6, 8),
                                            maxdensity=8),
            180: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 3, 4, 6, 8),
                                            maxdensity=6),
            400: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 3, 4),
                                            maxdensity=4),
            800: quantutils.allSubdivisions(maxsubdivs=1,
                                            possiblevals=(1, 2, 4),
                                            maxdensity=4),

        },
        'divisionPenaltyMap': defaultDivisionPenaltyMap,
        'nestedTuplets': False,
        'numNestedTupletsPenalty': (0, 0.0, 0.05, 0.4, 0.5, 0.8),
        'gridErrorWeight': 1.0,
        'divisionErrorWeight': 0.5,
        'rhythmComplexityWeight': 0.1,
        'gridErrorExp': 1,
    }
}


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
