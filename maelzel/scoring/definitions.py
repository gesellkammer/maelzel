from __future__ import annotations
from dataclasses import dataclass
from functools import cache


stemTypes = {
    'normal',
    'hidden'
}

# Articulations are actually any symbol which can be attached to a note in a non-exclusive manner
articulations = {
    'accent',  # >
    'staccato',  # .
    'tenuto',  # -
    'marcato',  # ^
    'staccatissimo',  # ' spiccato wedge
    'espressivo',  # <>
    'portato',  # - + .
    'arpeggio',
    'upbow',
    'downbow',
    'flageolet',
    'open',
    'closed',
    'stopped',
    'openstring',
    'snappizz',
    'laissezvibrer',
}

"""
musicxml articulations
(https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/articulations/)

    <accent>
    <strong-accent>
    <staccato>
    <tenuto>
    <detached-legato>
    <staccatissimo>
    <spiccato>
    <scoop>
    <plop>
    <doit>
    <falloff>
    <breath-mark>
    <caesura>
    <stress>
    <unstress>
    <soft-accent>
    <other-articulation>
"""


articulationMappings = {
    '>': 'accent',
    '.': 'staccato',
    '-': 'tenuto',
    '^': 'marcato',
    "'": 'staccatissimo',
    '<>': 'espressivo',
    'harmonic': 'flageolet',

    'strong-accent': 'marcato',
    'soft-accent': 'espressivo',
    'detached-legato': 'portato',
    'spiccato': 'staccatissimo',
    'snap-pizzicato': 'snappizz',
    'snappizzicato': 'snappizz',
    'l.v.': 'laissezvibrer',
    'lv': 'laissezvibrer'
}


@cache
def allArticulations() -> set[str]:
    """Returns all possible articulations including alternatives

    This will return both 'accent' and '>', staccato and '.'
    """
    return articulations | articulationMappings.keys()


def normalizeArticulation(articulation: str, default='') -> str:
    if articulation in articulations:
        return articulation
    elif (mapped := articulationMappings.get(articulation)) is not None:
        return mapped
    else:
        return default


clefs = {
    'treble15': 'treble15',
    'treble15a': 'treble15',
    'treble8': 'treble8',
    'treble8a': 'treble8',
    'treble': 'treble',
    'violin': 'treble',
    'g': 'treble',
    'f': 'bass',
    'bass': 'bass',
    'alto': 'alto',
    'viola': 'alto',
    'bass8': 'bass8',
    'bass8b': 'bass8',
    'bass15': 'bass15',
}


clefsByOrder = ("bass15", "bass8", "bass", "alto", "treble", "treble8", "treble15")

clefSortOrder = {clef: i for i, clef in enumerate(clefsByOrder)}


noteheadShapes = {
    'normal',
    'hidden',
    'cross',
    'harmonic',
    'triangleup',
    'xcircle',
    'triangle',
    'rhombus',
    'square',
    'rectangle',
    'slash',
    'diamond',
    'do',
    're',
    'mi',
    'fa',
    'sol',
    'la',
    'ti',
    'cluster'
}

_noteheadShapesMapping = {
    'x': 'cross',
    'circle-x': 'xcircle',
    'unpitched': 'cross',
    'flageolet': 'harmonic',
    'inverted triangle': 'triangleup',
    'slashed': 'diamond',
    'so': 'sol'
}


@cache
def allNoteheadShapes() -> set[str]:
    return noteheadShapes | _noteheadShapesMapping.keys()


def normalizeNoteheadShape(shape: str, default='') -> str:
    if shape in noteheadShapes:
        return shape
    return _noteheadShapesMapping.get(shape, default)


# These dynamics are supported in both lilypond and musicxml
dynamicLevelsSeq = ["ppppp", "pppp", "ppp", "pp", "p", "mp", "mf",
                    "f", "ff", "fff", "ffff", "fffff"]


dynamicLevels = set(dynamicLevelsSeq)


# These expressions are supported by lilypond
dynamicExpressions = {
    'fp', 'sf', 'sff', 'sp', 'spp', 'sfz', 'rfz', 'n'
}

# Dynamics which are not supported should be converted to the nearest
# supported dynamic. This is particularly important when parsing some
# external source, like muscxml

dynamicMappings = {
    'pppppp': 'ppppp',
    'ffffff': 'fffff',
    'sfp': 'sp',
    'sfpp': 'spp',
    'rf': 'rfz',
    'sffz': 'sfz',
    'fz': 'sf',
    'pf': 'f',
    'sfzp': 'fp'
}


alterations = {
    'natural': 0,
    'natural-up': 25,
    'quarter-sharp': 50,
    'sharp-down': 75,
    'sharp': 100,
    'sharp-up': 125,
    'three-quarters-sharp': 150,
    'natural-down': -25,
    'quarter-flat': -50,
    'flat-up': -75,
    'flat': -100,
    'flat-down': -125,
    'three-quarters-flat': -150
}


@cache
def validDynamics() -> set[str]:
    out = dynamicLevels.copy()
    out.update(dynamicExpressions)
    return out


def normalizeDynamic(dynamic: str, default='') -> str:
    """
    Normalize a dynamic, returns *default* if the dynamic is invalid

    Args:
        dynamic: the dynamic to normalize
        default: fallback dynamic

    Returns:
        the normalized dynamic
    """
    if dynamic in dynamicLevels or dynamic in dynamicExpressions:
        return dynamic
    elif (mapped := dynamicMappings.get(dynamic)) is not None:
        assert isinstance(mapped, str)
        return mapped
    return default


availableOrnaments = {'trill', 'mordent', 'prall', 'turn'}


ornamentMappings = {
    'trill-mark': 'trill',
    'inverted-mordent': 'prall',
    'shake': 'prall'
}


def normalizeOrnament(ornament: str, default='') -> str:
    if ornament in availableOrnaments:
        return ornament
    elif (_:=ornamentMappings.get(ornament)) is not None:
        return _
    return default


availableFermatas = {
    'normal',
    'square',
    'angled',
    'double-angled',
    'double-square'
}


def normalizeFermata(fermata: str) -> str:
    return fermata if fermata in availableFermatas else 'normal'


enclosureBoxes = {
    'square',
    'circle',
    'rounded',
    ''
}

boxMappings = {
    'none': '',
    'rounded-box': 'square',
    'box': 'square',
    True: 'square'
}


def normalizeEnclosure(enclosure: str | bool) -> str:
    if isinstance(enclosure, bool):
        return 'square' if enclosure else ''
    if enclosure in enclosureBoxes:
        return enclosure
    if (mapped := boxMappings.get(enclosure)) is not None:
        return mapped
    raise ValueError(f"Enclosure '{enclosure}' not known, possible values are {enclosureBoxes} or"
                     f" {boxMappings.keys()}")


barstyles = {
    'single',
    'double',
    'final',
    'dashed',
    'dotted',
    'tick',
    'short',
    'double-thin',
    'none'
}


barstyleMapping = {
    'regular': 'single',
    'heavy': 'solid',
    'light-light': 'double-thin',
    'light-heavy': 'final',
    'heavy-light': 'final',
    'heavy-heavy': 'double',
    'hidden': 'none'
}

def normalizeBarstyle(barstyle: str, default='') -> str:
    if barstyle in barstyles:
        return barstyle
    if (_:=barstyleMapping.get(barstyle.lower())) is not None:
        return _
    return default


# These are taken mostly from lilypond and are not supported in musicxml as breah-marks
# https://lilypond.org/doc/v2.23/Documentation/notation/list-of-breath-marks
breathMarks = {
    'comma',
    'varcomma',
    'upbow',
    'outsidecomma',
    'caesura',
    'chant'
}


@dataclass(unsafe_hash=True)
class Notehead:
    shape: str = ''
    color: str = ''
    size: int | float | None = None
    parenthesis: bool = False
    hidden: bool = False

    def __post_init__(self):
        assert isinstance(self.shape, str)
        assert isinstance(self.color, str)

        if self.shape:
            if self.shape[-1] == '?':
                self.shape = self.shape[:-1]
                self.parenthesis = True
            if self.shape == 'hidden':
                self.shape = ''
                self.hidden = True

        assert not self.shape or self.shape in noteheadShapes, \
            f'shape "{self.shape}" not in {noteheadShapes}'

    def __copy__(self):
        return Notehead(shape=self.shape, color=self.color, size=self.size,
                        parenthesis=self.parenthesis, hidden=self.hidden)

    def copy(self):
        return self.__copy__()

    def update(self, other: Notehead):
        if other.shape:
            self.shape = other.shape
        if other.color:
            self.color = other.color
        if other.size is not None:
            self.size = other.size

        self.parenthesis = other.parenthesis
        self.hidden = other.hidden

    def description(self) -> str:
        shape = self.shape
        if self.parenthesis:
            shape += '?'
        parts = []
        if shape:
            parts.append(shape)
        if self.color:
            parts.append(f'color={self.color}')
        if self.size is not None:
            parts.append(f'size={self.size}')
        if self.hidden:
            parts.append('hidden=True')
        return ';'.join(parts)

    @staticmethod
    def parseDescription(descr: str):
        parts = descr.split(';')
        shape = ''
        color = ''
        size = None
        parenthesis = False
        for part in parts:
            if '=' in part:
                key, value = part.split('=')
                if key == 'color':
                    color = value
                elif key == 'size':
                    size = float(value)
                    if int(size) == size:
                        size = int(size)
                else:
                    raise ValueError(f"Key not known {key}")
            else:
                shape = part
                if shape[-1] == '?':
                    shape = shape[:-1]
                    parenthesis = True
        return Notehead(shape=shape, color=color, size=size, parenthesis=parenthesis)
