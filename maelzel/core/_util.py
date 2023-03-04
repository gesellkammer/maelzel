"""
Internal utilities
"""
from __future__ import annotations

import logging
from maelzel.scoring import definitions
from typing import TYPE_CHECKING
from functools import cache
import sys
import os
import bpf4 as bpf
import pitchtools as pt
from dataclasses import dataclass
from maelzel.common import F
from maelzel.colortheory import safeColors
from . import environment

if TYPE_CHECKING:
    from typing import Union, Optional
    from ._typedefs import *


@cache
def buildingDocumentation() -> bool:
    return "sphinx" in sys.modules


def checkBuildingDocumentation(logger=None) -> bool:
    """
    Check if we are running because of a documentation build

    Args:
        logger: if given, it is used to log messages

    Returns:
        True if currently building documentation

    """
    building = buildingDocumentation()
    if building:
        msg = "Not available while building documentation"
        if logger:
            logger.error(msg)
        else:
            print(msg)
    return building


def pngShow(pngpath: str, forceExternal=False, app: str = '') -> None:
    """
    Show a png either with an external app or inside jupyter

    Args:
        pngpath: the path to a png file
        forceExternal: if True, it will show in an external app even
            inside jupyter. Otherwise it will show inside an external
            app if running a normal session and show an embedded
            image if running inside a notebook
        app: used if a specific external app is needed. Otherwise the os
            defined app is used
    """
    if environment.insideJupyter and not forceExternal:
        from . import jupytertools
        jupytertools.showPng(pngpath)
    else:
        environment.openPngWithExternalApplication(pngpath, app=app)


def carryColumns(rows: list, sentinel=None) -> list:
    """
    Carries values from one row to the next, if needed

    Converts a series of rows with possibly unequal number of elements per row
    so that all rows have the same length, filling each new row with elements
    from the previous, if they do not have enough elements (elements are "carried"
    to the next row)
    """
    maxlen = max(len(row) for row in rows)
    initrow = [0] * maxlen
    outrows = [initrow]
    for row in rows:
        lenrow = len(row)
        if lenrow < maxlen:
            row = row + outrows[-1][lenrow:]
        if sentinel in row:
            row = row.__class__(x if x is not sentinel else lastx for x, lastx in zip(row, outrows[-1]))
        outrows.append(row)
    # we need to discard the initial row
    return outrows[1:]


def as2dlist(rows: list[list | tuple]) -> list[list]:
    """
    Ensure that all rows are lists

    Args:
        rows: a list of sequences

    Returns:
        a list of lists
    """
    return [row if isinstance(row, list) else list(row)
            for row in rows]


def normalizeFade(fade: fade_t,
                  defaultfade: float
                  ) -> tuple[float, float]:
    """ Returns (fadein, fadeout) """
    if fade is None:
        fadein, fadeout = defaultfade, defaultfade
    elif isinstance(fade, tuple):
        assert len(fade) == 2, f"fade: expected a tuple or list of len=2, got {fade}"
        fadein, fadeout = fade
    elif isinstance(fade, (int, float)):
        fadein = fadeout = fade
    else:
        raise TypeError(f"fade: expected a fadetime or a tuple of (fadein, fadeout), got {fade}")
    return fadein, fadeout


def normalizeFilename(path: str) -> str:
    return os.path.expanduser(path)


def midinotesNeedSplit(midinotes: list[float], splitpoint=60, margin=4
                       ) -> bool:
    if len(midinotes) == 0:
        return False
    numabove = sum(int(m > splitpoint - margin) for m in midinotes)
    numbelow = sum(int(m < splitpoint + margin) for m in midinotes)
    return bool(numabove and numbelow)


_enharmonic_sharp_to_flat = {
    'C#': 'Db',
    'D#': 'Eb',
    'E#': 'F',
    'F#': 'Gb',
    'G#': 'Ab',
    'A#': 'Bb',
    'H#': 'C'
}
_enharmonic_flat_to_sharp = {
    'Cb': 'H',
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#',
    'Hb': 'A#'
}


dbToAmpCurve: bpf.BpfInterface = bpf.expon(
    -120, 0,
    -60, 0.0,
    -40, 0.1,
    -30, 0.4,
    -18, 0.9,
    -6, 1,
    0, 1,
    exp=0.333)


def enharmonic(n: str) -> str:
    n = n.capitalize()
    if "#" in n:
        return _enharmonic_sharp_to_flat[n]
    elif "x" in n:
        return enharmonic(n.replace("x", "#"))
    elif "is" in n:
        return enharmonic(n.replace("is", "#"))
    elif "b" in n:
        return _enharmonic_flat_to_sharp[n]
    elif "s" in n:
        return enharmonic(n.replace("s", "b"))
    elif "es" in n:
        return enharmonic(n.replace("es", "b"))
    else:
        return n

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Helper functions for Note, Chord, ...
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def midicents(midinote: float) -> int:
    """
    Returns the cents to next chromatic pitch

    Args:
        midinote: a (fractional) midinote

    Returns:
        cents to next chromatic pitch
    """
    return int(round((midinote - round(midinote)) * 100))


def quantizeMidi(midinote: float, step=1.0) -> float:
    return round(midinote / step) * step


def centsshown(centsdev: int, divsPerSemitone: int) -> str:
    """
    Given a cents deviation from a chromatic pitch, return
    a string to be shown along the notation, to indicate the
    true tuning of the note. If we are very close to a notated
    pitch (depending on divsPerSemitone), then we don't show
    anything. Otherwise, the deviation is always the deviation
    from the chromatic pitch

    Args:
        centsdev: the deviation from the chromatic pitch
        divsPerSemitone: 4 means 1/8 tones

    Returns:
        the string to be shown alongside the notated pitch
    """
    # cents can be also negative (see self.cents)
    pivot = int(round(100 / divsPerSemitone))
    dist = min(centsdev % pivot, -centsdev % pivot)
    if dist <= 2:
        return ""
    if centsdev < 0:
        # NB: this is not a normal - sign! We do this to avoid it being confused
        # with a syllable separator during rendering (this is currently the case
        # in musescore
        return f"–{-centsdev}"
    return str(int(centsdev))


def asmidi(x) -> float:
    """
    Convert x to a midinote

    Args:
        x: a str ("4D", "1000hz") a number (midinote) or anything
           with an attribute .midi

    Returns:
        a midinote

    """
    if isinstance(x, str):
        return pt.str2midi(x)
    elif isinstance(x, (int, float)):
        assert 0 <= x <= 200, f"Expected a midinote (0-127) but got {x}"
        return x
    raise TypeError(f"Expected a str, a Note or a midinote, got {x}")


def asfreq(n) -> float:
    """
    Convert a midinote, notename of Note to a freq.

    NB: a float value is interpreted as a midinote

    Args:
        n: a note as midinote, notename or Note

    Returns:
        the corresponding frequency
    """
    if isinstance(n, str):
        return pt.n2f(n)
    elif isinstance(n, (int, float)):
        return pt.m2f(n)
    elif hasattr(n, "freq"):
        return n.freq
    else:
        raise ValueError(f"cannot convert {n} to a frequency")


@dataclass
class NoteProperties:
    """
    Represents the parsed properties of a note, as returned by :func:`parseNote`

    The format to parse is Pitch[:dur][:property1][...]

    .. seealso:: :func:`parseNote`
    """
    notename: str | list[str]
    """A pitch or a list of pitches"""

    dur: F | None
    """An optional totalDuration"""

    keywords: dict[str, str] | None = None
    """Any other properties (gliss, tied, ...)"""

    def classifyKeywords(self) -> tuple[dict, list[tuple[str, str]]]:
        """
        Splits keywords into properties and symbols

        Properties are any keyword passed to the constructor (gliss, tied, offset, dynamic, etc)
        Symbols are articulations, notehead shapes, etc.

        Returns:
            a tuple (properties, symbols)
        """
        props = {}
        symbols = []
        for k, v in self.keywords.items():
            if k in _knownSymbols:
                symbols.append((k, v))
            else:
                props[k] = v
        return props, symbols



_dotRatios = [1, F(3, 2), F(7, 4), F(15, 8), F(31, 16)]


def _parseSymbolicDuration(s: str) -> F:
    if not s.endswith("."):
        return F(4, int(s))
    dots = s.count(".")
    s = s[:-dots]
    ratio = _dotRatios[dots]
    return F(4, int(s)) * ratio


def parseDuration(s: str) -> F:
    """
    Parse a totalDuration given a str

    Possible expressions include '3/4', '1+3/4', '4+1/3+2/5'
    Raises ValueError if the expression cannot be parsed

    Args:
        s: the totalDuration as string

    Returns:
        the parsed totalDuration as fraction
    """
    terms = s.split('+')
    fterms = [F(term) for term in terms]
    return sum(fterms)


def parsePitch(s: str) -> tuple[str, bool, bool]:
    """
    Parse a pitch like 4a~ or 4C#+15!

    Args:
        s: the notename to parse

    Returns:
        a tuple (notename, tied, fixed)

    """
    tied = None
    fixed = None
    if s.endswith('~'):
        tied = True
        s = s[:-1]
    if s.endswith('!'):
        fixed = True
        s = s[:-1]
    return s, tied, fixed


def parseNote(s: str) -> NoteProperties:
    """
    Parse a note definition string with optional totalDuration and other properties

    Pitch specific modifiers, like ! or ~ are not parsed

    ================================== ============= ====  ===========
    Note                               Pitch         Dur   Properties
    ================================== ============= ====  ===========
    4c#                                4C#           None  None
    4F+:0.5                            4F+           0.5   None
    4G:1/3                             4G            1/3   None
    4Bb-:mf                            4B-           None  {'dynamic':'mf'}
    4G-:0.4:ff:articulation=accent     4G-           0.4   {'dynamic':'ff', 'articulation':'accent'}
    4F#,4A                             [4F#, 4A]     None  None
    4G:^                               4G            None  {'articulation': 'accent'}
    4A/8                               4A            0.5
    4Gb/4.:pp                          4Gb           1.5   {dynamic: 'pp'}
    ================================== ============= ====  ===========


    Args:
        s: the note definition to parse

    Returns:
        a NoteProperties object with the result

    4C#~
    """
    dur, properties = None, {}

    if ":" not in s:
        note = s
    else:
        note, rest = s.split(":", maxsplit=1)
        parts = rest.split(":")
        try:
            dur = parseDuration(parts[0])
            parts = parts[1:]
        except ValueError:
            pass
        for part in parts:
            if part in _knownDynamics:
                properties['dynamic'] = part
            elif part in _knownArticulations:
                properties['articulation'] = part
            elif part == 'gliss':
                properties['gliss'] = True
            elif part == 'tied':
                properties['tied'] = True
            elif "=" in part:
                key, value = part.split("=", maxsplit=1)
                if key == 'offset':
                    value = F(value)
                elif key == 'gliss':
                    if "," in value:
                        value = [pt.str2midi(pitch) for pitch in value.split(",")]
                    else:
                        value = pt.str2midi(value)
                properties[key] = value

    if "/" in note:
        note, symbolicdur = note.split("/")
        dur = _parseSymbolicDuration(symbolicdur)
    notename = [p.strip() for p in note.split(",")] if "," in note else note
    return NoteProperties(notename=notename, dur=dur, keywords=properties)


_knownDynamics = {
    'pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff', 'n'
}

_knownArticulations_ = {
    '-', '.', '^', 'accent', 'tenuto', 'staccato'
}


_knownArticulations = definitions.allArticulations()


_knownSymbols = {
    'articulation', 'notehead', 'size', 'color'
}


def _highlightLilypond(s: str) -> str:
    # TODO
    return s


def showLilypondScore(score: str) -> None:
    """
    Display a lilypond score, either at the terminal or within a notebook

    Args:
        score: the score as text
    """
    # TODO: add highlighting, check if inside jupyter, etc.
    print(score)
    return


def dictRemoveNoneKeys(d: dict):
    keysToRemove = [k for k, v in d.items() if v is None]
    for k in keysToRemove:
        del d[k]


def htmlSpan(text, color: str = '', fontsize: str = '', italic=False, bold=False) -> str:
    if color.startswith(':'):
        color = safeColors[color[1:]]
    styleitems = {}
    if color:
        styleitems['color'] = color
    if fontsize:
        styleitems['font-size'] = fontsize
    stylestr = ";".join(f"{k}:{v}" for k, v in styleitems.items())
    text = str(text)
    if italic:
        text = f'<i>{text}</i>'
    if bold:
        text = f'<strong>{text}</strong>'
    return f'<span style="{stylestr}">{text}</span>'


def showF(f: F, maxdenom=1000) -> str:
    """
    Show a fraction, limit den to *maxdenom*

    Args:
        f: the fraction to show

    Returns:
        a readable string representation

    """
    if f.denominator > maxdenom:
        f2 = f.limit_denominator(maxdenom)
        return "*%d/%d" % (f2.numerator, f2.denominator)
    return "%d/%d" % (f.numerator, f.denominator)


def showT(f: F | float | None) -> str:
    """Show *f* as time"""
    if f is None:
        return "None"
    if not isinstance(f, float):
        f = float(f)
    return f"{f:.3f}".rstrip('0').rstrip('.')

