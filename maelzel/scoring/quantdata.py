from __future__ import annotations
from . import quantutils
from functools import cache

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

# Presets used to create a QuantizationProfile
presets = {
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
        'numNestedTupletsPenalty': [0.0, 0.0, 0.0, 0.1, 0.4, 0.8],
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
        'numNestedTupletsPenalty': [0., 0., 0.03, 0.4, 0.5, 0.8],
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
        'numNestedTupletsPenalty': [0, 0.0, 0.05, 0.4, 0.5, 0.8],
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
        'numNestedTupletsPenalty': [0, 0.0, 0.05, 0.4, 0.5, 0.8],
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
