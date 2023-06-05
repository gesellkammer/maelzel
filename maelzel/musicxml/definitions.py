from __future__ import absolute_import
from dataclasses import dataclass


STR2CLASS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 8, 'B': 10}

POSSIBLE_DYNAMICS = {'pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff'}

# Noteheadsize as defined by lilypond
DYNAMIC_TO_RELATIVESIZE = {
    'pppp': -5,
    'ppp' : -4,
    'pp'  : -3,
    'p'   : -2,
    'mp'  : -1,
    'mf'  : 0,
    'f'   : 1,
    'ff'  : 2,
    'fff' : 3,
    'ffff': 4
}

# musicxml font-size attribute
DYNAMIC_TO_FONTSIZE = {
    'pppp': 10,
    'ppp':  12,
    'pp':   16,
    'p':    18,
    'mp':   20,
    'mf':   24,
    'f':    28,
    'ff':   32,
    'fff':  36,
    'ffff': 40
}


REGULAR_NOTETYPES = {
    4:   'quarter',
    8:   'eighth',
    16:  '16th',
    32:  '32nd',
    64:  '64th',
    128: '128th'
}


NOTETYPE_TO_NUMERIC_DURATION = {
    'whole': 1,
    'half': 2,
    'quarter': 4,
    'eighth': 8,
    '16th': 16,
    '32nd': 32,
    '64th': 64,
    '128th': 128
}


IRREGULAR_NOTETYPES = {
    # note_type, dots
    (7,4): ('quarter', 2),
    (3,2): ('quarter', 1),
    # (5,4): ('quarter', 0),  #
    (1,1): ('quarter', 0),
    (7,8): ('eighth', 2),
    (3,4): ('eighth', 1),
    # (5,8): ('eighth', 0),    #
    (1,2): ('eighth', 0),
    (7,16):('16th', 2),
    (3,8): ('16th', 1),
    (5,16):('16th', 0),
    (1,4): ('16th', 0),
    (7,32):('32nd', 2),
    (3,16):('32nd', 1),
    (1, 8):('32nd', 0),
    (7,64):('64th', 2),
    (3,32):('64th', 1),
    (1,16):('64th', 0),

    (1,3):('eighth', 0),
    (2,3):('quarter', 0),

    (1,5):('16th', 0),
    (2,5):('eighth', 0),
    (3,5):('eighth', 1),
    (4,5):('quarter', 0),

    (1,6):('16th', 0),
    (2,6):('eighth', 0),
    (3,6):('eighth', 1),
    (4,6):('quarter', 0),
    # (5,6):('quarter', 0),   #

    (1,7):('16th', 0),
    (2,7):('eighth', 0),
    (3,7):('eighth', 1),
    (4,7):('quarter', 0),
    # (5,7):('quarter', 0),   #
    (6,7):('quarter', 1),

    (1,9):('32nd', 0),
    (2,9):('16th', 0),
    (3,9):('16th', 1),
    (4,9):('eighth', 0),
    # (5,9):('eighth', 0),    #
    (6,9):('eighth', 1),
    (7,9):('eighth', 2),
    (8,9):('quarter', 0),

    (1,10):('32nd', 0),
    (3,10):('16th', 1),
    (7,10):('eighth', 2),

    (1,11):('32nd', 0),
    (2,11):('16th', 0),
    (3,11):('16th', 1),
    (4,11):('eight', 0),
    # (5,11):('eighth', 0),
    (6,11):('eighth', 1),
    (7,11):('eighth', 2),
    (8,11):('quarter', 0),
    (9,11):('quarter', 0),
    (10,11):('quarter', 1),


    (1,12):('32nd', 0),
    (2,12):('16th', 0),
    (3,12):('16th', 1),
    (4,12):('eighth', 0),
    # (5,12):('eighth', 0),    #
    (6,12):('eighth', 1),
    (7,12):('eighth', 2),    #
    (8,12):('quarter', 0),
    (9,12):('quarter', 0),   #

    (11,12):('quarter', 1),  #

    (1, 15):('32nd', 0),
    (2, 15):('16th', 0),
    (3, 15):('16th', 1),
    (4, 15):('eighth', 0),
    (5, 15):('eighth', 0),
    (6, 15):('eighth', 1),
    (7, 15):('eighth', 2),
    (8, 15):('quarter', 0),
    (9, 15):('quarter', 0),
    (10, 15):('quarter', 0),
    (11, 15):('quarter', 0),
    (12, 15):('quarter', 1),
    (13, 15):('quarter', 1),
    (14, 15):('quarter', 1),

    (1, 16):('64th', 0),
    (2, 16):('32nd', 0),
    (3, 16):('32nd', 1),
    (4, 16):('16th', 0),
    (6, 16):('16th', 1),
    (7, 16):('16th', 2),
    (8,16) :('eighth', 0),
    (11,16):('eighth', 0),
    (12,16):('eighth', 1),
    (15,16):('eighth', 3),

    (1, 32) :('128th', 0),
    (2, 32) :('64th', 0),
    (3, 32) :('64th', 1),
    (7, 32) :('32th', 2),
    (11,32) :('16th', 0),
    (15, 32):('16th', 3),
    (31, 32):('eighth', 4),
}

NOTETYPES = list(REGULAR_NOTETYPES.values())

MUSICXML_ACCIDENTALS = {
   -1.50: 'three-quarters-flat',
   -1.25: 'flat-down',
   -1.00: 'flat',
   -0.75: 'flat-up',
   -0.50: 'quarter-flat',
   -0.25: 'natural-down',
    0.00: 'natural',
    0.25: 'natural-up',
    0.50: 'quarter-sharp',
    0.75: 'sharp-down',
    1.00: 'sharp',
    1.25: 'sharp-up',
    1.50: 'three-quarters-sharp',
}


@dataclass
class XmlNotehead:
    shape: str
    filled: bool


MUSICXML_NOTEHEADS: list[XmlNotehead] = [
    XmlNotehead("normal", filled=False),
    XmlNotehead("square", filled=False),
    XmlNotehead("diamond", filled=False),
    XmlNotehead("harmonic", filled=False),
    XmlNotehead("x", filled=False),
    XmlNotehead("circle-x", filled=False)
]


# bw_to_noteheadindex = bpf.linear(0, 0, 1, len(MUSICXML_NOTEHEADS)-1)


# Grid resolution, resulting from
# lcm(3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 32)
# Can't represent 11, 13 or 17 tuples

MUSICXML_DIVISIONS = 10080
MUSICXML_TENTHS = 40


NOTE_DICT = dict(((0, 'C0'), (1, 'C1'), (2, 'D0'), (3, 'D1'), (4, 'E0'),
                  (5, 'F0'), (6, 'F1'), (7, 'G0'), (8, 'G1'), (9, 'A0'),
                  (10, 'A1'), (11, 'B0')))


OCTAVE_TO_LILYPOND_OCTAVE = {
   -1:",,,,",
    0:",,,",
    1:",,",
    2:",",
    3:"",
    4:"'",
    5:"''",
    6:"'''",
    7:"''''",
    8:"'''''"

}


ARTICULATIONS = {
    "accent",
    "marcato"
}