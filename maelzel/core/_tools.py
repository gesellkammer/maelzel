"""
Internal utilities
"""
from __future__ import annotations

import bisect
import re
import sys
from functools import cache
from dataclasses import dataclass
import bpf4 as bpf
import pitchtools as pt
from emlib import misc

from maelzel import _util
from maelzel.scoring import definitions
from maelzel.common import F
from maelzel.colortheory import safeColors
from maelzel.core import environment
from maelzel.core import symbols as _symbols

from typing import TYPE_CHECKING, Any, Sequence
if TYPE_CHECKING:
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


def pngShow(pngpath: str, forceExternal=False, app: str = '', scalefactor=1.0) -> None:
    """
    Show a png either with an external app or inside jupyter

    Args:
        pngpath: the path to a png file
        forceExternal: if True, it will show in an external app even
            inside jupyter. Otherwise, it will show inside an external
            app if running a normal session and show an embedded
            image if running inside a notebook
        app: used if a specific external app is needed. Otherwise, the os
            defined app is used
        scalefactor: used to scale the image when shown within jupyter
    """
    if environment.insideJupyter and not forceExternal:
        from . import jupytertools
        jupytertools.showPng(pngpath, scalefactor=scalefactor)
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


@dataclass
class NoteProperties:
    """
    Represents the parsed properties of a note, as returned by :func:`parseNote`

    The format to parse is Pitch[:dur][:property1][...]

    .. seealso:: :func:`parseNote`
    """
    notename: str | list[str]
    """A pitch or a list of pitches. The string 'Rest' indicates a rest"""

    dur: F | None
    """An optional duration"""

    keywords: dict[str, Any] | None = None
    """Any other properties (gliss, tied, ...)"""

    symbols: list[_symbols.Symbol] | None = None
    """Symbols attached to this note"""

    spanners: list[_symbols.Spanner] | None = None
    """Spanners attached to this note"""


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
    Parse a duration given a str

    Possible expressions include '3/4', '1+3/4', '4+1/3+2/5'
    Raises ValueError if the expression cannot be parsed

    Args:
        s: the duration as string

    Returns:
        the parsed duration as fraction
    """
    try:
        return F(s)
    except ValueError:
        pass

    if "*" not in s or "(" not in s:
        # simple form
        terms = s.split('+')
        fterms = [F(term) for term in terms]
        return sum(fterms)
    parts = []
    cursor = 0
    for match in re.finditer(r"\d+\/\d+", s):
        parts.append(s[cursor:match.start()])
        num, den = match.group().split("/")
        parts.append(f"F({num}, {den}")
        cursor = match.end()
    if cursor < len(s) - 1:
        parts.append(s[cursor:])
    s2 = "".join(parts)
    return eval(s2)


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


def _evalArgs(args: list[str]) -> dict[str, Any]:
    def evalArg(arg: str) -> tuple[str, Any]:
        if "=" in arg:
            k, v = [_.strip() for _ in arg.split("=")]
            if (n := misc.asnumber(v)) is not None:
                return (k, n)
            elif v in ("True", "False", "None"):
                return (k, eval(v))
            else:
                return k, v
        else:
            # a flag
            return (arg, True)
    return dict(evalArg(arg) for arg in args)


def stripNoteComments(s: str) -> str:
    """
    Strip comments from notes defined as strings

    Args:
        s: the note definition

    Returns:
        the definition without comments

    Example
    -------

        >>> stripNoteComments("    4F#:3/4:ff:dim    # comment ")
        4F#:3/4:ff:dim

    """
    parts = s.split(" #")
    return parts[0].strip()


def parseNote(s: str) -> NoteProperties:
    """
    Parse a note definition string with optional duration and other properties

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
    4A/4                               4A            0.5
    4Gb/8.:pp                          4Gb           3/4   {dynamic: 'pp'}
    r:.5                               rest          0.5
    ================================== ============= ====  ===========


    Args:
        s: the note definition to parse

    Returns:
        a NoteProperties object with the result

    4C#~
    """
    dur = None
    properties: dict[str, str | float | bool | F] = {}
    symbols: list[_symbols.Symbol] = []
    spanners: list[_symbols.Spanner] = []

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
            if match := re.match(r"(\w+)\((\w.+)\)", part):
                # example: stem(hidden=True)
                clsname = match.group(1)
                argstr = match.group(2)
                args = [_.strip() for _ in argstr.split(",")]
                kws = _evalArgs(args)
                if clsname in _knownSymbols:
                    symbols.append(_symbols.makeSymbol(clsname, **kws))
                else:
                    raise ValueError(f"Unknown class {clsname}, possible symbols: {_knownSymbols}")

            elif '=' in part:
                key, value = part.split("=", maxsplit=1)
                if key == 'offset':
                    properties['offset'] = F(value)
                elif key == 'gliss':
                    if "," in value:
                        value = [pt.str2midi(pitch) for pitch in value.split(",")]
                    else:
                        value = pt.str2midi(value)
                    properties['gliss'] = value
                elif key in _knownSymbols:
                    symbols.append(_symbols.makeSymbol(key, value))
                else:
                    properties[key] = value
            elif part in _knownDynamics:
                properties['dynamic'] = part
            elif part in _knownArticulations:
                # properties['articulation'] = part
                symbols.append(_symbols.Articulation(part))
            elif part in ('gliss', 'tied'):
                properties[part] = True
            elif part == 'stemless':
                symbols.append(_symbols.Stem(hidden=True))
            elif part == '()':
                symbols.append(_symbols.Notehead(parenthesis=True))
            elif part in definitions.allNoteheadShapes():
                symbols.append(_symbols.Notehead(part))
            elif part in _knownSpanners or (part[0] == "~" and part[1:] in _knownSpanners):
                spanners.append(_symbols.makeSpanner(part))
            else:
                allkeys = _allkeys()
                _util.checkChoice(s, part, allkeys)

    if "/" in note:
        note, symbolicdur = note.split("/")
        dur = _parseSymbolicDuration(symbolicdur)
    if "," in note:
        notename = [p.strip() for p in note.split(",")]
    elif note.lower() in ('r', 'rest'):
        notename = 'rest'
    else:
        notename = note
    return NoteProperties(notename=notename, dur=dur, keywords=properties,
                          symbols=symbols, spanners=spanners)


@cache
def _allkeys():
    allkeys = []
    allkeys.extend(_knownSymbols)
    allkeys.extend(_knownDynamics)
    allkeys.extend(_knownArticulations)
    allkeys.extend(_knownSpanners)
    return allkeys


_knownDynamics = {
    'pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff', 'n'
}


_knownArticulations = definitions.allArticulations()


_knownSymbols = {
    'notehead',
    'articulation',
    'text',
    'color',
    'accidental',
    'ornament',
    'fermata',
    'stem'
}


_knownSpanners = {
    'slur',
    'cresc',
    'decresc',
    'line',
    'trill',
    'tr',
    'bracket',
    'dim',
    'beam'
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


def htmlSpan(text, color='', fontsize='', italic=False, bold=False) -> str:
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


def cropBreakpoints(bps: list[Sequence[num_t]], t0: float, t1: float
                    ) -> list[Sequence[num_t]]:
    """
    Crop a sequence of breakpoints

    A breakpoint is a list or tuple of the form (time, value1, ...). The values
    following time can be anything, the first value must be the time

    Args:
        bps: the breakpoints to crop
        t: the time to crop at

    Returns:
        the new breakpoints. Breakpoints between the cropping edges will be
        shared between the old and new breakpoints so if they are mutable, any
        modification will be visible
    """
    def interpolate(t, times, bps):
        idx = bisect.bisect(times, t)
        assert 0 < idx < len(times), f"{t=}, out of range, {times=}"
        if times[idx - 1] < t:
            return _interpolateBreakpoints(t, bps[idx - 1], bps[idx]), idx
        else:
            return bps[idx-1], idx

    if t0 <= bps[0][0] and t1 >= bps[-1][0]:
        return bps
    times = [bp[0] for bp in bps]
    out = []
    if t0 <= times[0]:
        chunkstart = 0
    else:
        firstbreakpoint, chunkstart = interpolate(t0, times, bps)
        out.append(firstbreakpoint)

    if t1 >= times[-1]:
        chunkend = len(bps)
        lastbreakpoint = None
    else:
        lastbreakpoint, chunkend = interpolate(t1, times, bps)

    out.extend(bps[chunkstart:chunkend])
    if lastbreakpoint is not None:
        out.append(lastbreakpoint)
    return out


def _interpolateBreakpoints(t: float, bp0: Sequence[num_t], bp1: Sequence[num_t]
                            ) -> list[num_t]:
    t0, t1 = bp0[0], bp1[0]
    assert t0 <= t <= t1, f"{t0=}, {t=}, {t1=}"
    delta = (t - t0) / (t1 - t0)
    bp = [t]
    for v0, v1 in zip(bp0[1:], bp1[1:]):
        bp.append(v0 + (v1-v0)*delta)
    return bp


