"""
Internal utilities
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from functools import cache
import PIL
import sys
import os
import bpf4 as bpf
import pitchtools as pt
from dataclasses import dataclass
from . import symbols as _symbols

if TYPE_CHECKING:
    from typing import *
    from ._typedefs import *
    T = TypeVar("T")
    from maelzel import scoring


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


def imgSize(path:str) -> Tuple[int, int]:
    """ returns (width, height) """
    im = PIL.Image.open(path)
    return im.size


def showTime(f) -> str:
    if f is None:
        return "None"
    return f"{float(f):.3g}"


def addColumn(mtx: Union[List[List[T]], List[Tuple[T]]], col: List[T], inplace=False) -> List[List[T]]:
    """
    Add a column to a list of lists/tuples

    Args:
        mtx: a matrix (a list of lists)
        col: a list of elements to add as a new column to mtx
        inplace: add the elements in place or create a new matrix

    Returns:
        if inplace, returns the old matrix, otherwise a new matrix

    Example::

        mtx = [[1,   2,  3],
               [11, 12, 13],
               [21, 22, 23]]

        addColumn(mtx, [4, 14, 24])

        [[1,   2,  3,  4],
          11, 12, 13, 14],
          21, 22, 23, 24]]

    """
    if isinstance(mtx[0], list):
        if not inplace:
            return [row + [elem] for row, elem in zip(mtx, col)]
        else:
            for row, elem in zip(mtx, col):
                row.append(elem)
            return mtx
    elif isinstance(mtx[0], tuple):
        if inplace:
            raise ValueError("Can't add a column in place, since each row is tuple"
                             " and tuples are not mutable")
        return [row+(item,) for row, item in zip(mtx, col)]
    else:
        raise TypeError(f"mtx should be a seq. of lists, or tuples, "
                        f"got {mtx} ({type(mtx[0])})")


def carryColumns(rows: list, sentinel=None) -> list:
    """
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


def as2dlist(rows) -> List[list]:
    return [row if isinstance(row, list) else list(row)
            for row in rows]


def normalizeFade(fade: fade_t,
                  defaultfade: float
                  ) -> Tuple[float, float]:
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


def midinotesNeedSplit(midinotes: List[float], splitpoint=60, margin=4
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


def enharmonic(n:str) -> str:
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


def quantizeMidi(midinote:float, step=1.0) -> float:
    return round(midinote / step) * step


def centsshown(centsdev:int, divsPerSemitone:int) -> str:
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
    dist = min(centsdev%pivot, -centsdev%pivot)
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
        assert 0<=x<=200, f"Expected a midinote (0-127) but got {x}"
        return x
    elif hasattr(x, 'midi'):
        return x.pitch
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
class NoteComponent:
    notename: str
    midi: float
    freq: float
    ampdb: float
    ampgroup: int


def splitByAmp(midis: List[float], amps:List[float], numGroups=8, maxNotesPerGroup=8
               ) -> List[List[NoteComponent]]:
    """
    split the notes by amp into groups (similar to a histogram based on amplitude)

    Args:
        midis: a seq of midinotes
        amps: a seq of amplitudes in dB (same length as midinotes)
        numGroups: the number of groups to divide the notes into
        maxNotesPerGroup: the maximum of included notes per group, picked by loudness

    Returns:
        a list of chords with length=numgroups
    """
    step = (dbToAmpCurve*numGroups).floor()
    notes = []
    for midi, amp in zip(midis, amps):
        db = pt.amp2db(amp)
        notes.append(NoteComponent(pt.m2n(midi), midi, pt.m2f(midi), db, int(step(db))))
    chords: List[List[NoteComponent]] = [[] for _ in range(numGroups)]
    notes2 = sorted(notes, key=lambda n: n[3], reverse=True)
    for note in notes2:
        chord = chords[note[4]]
        if len(chord) <= maxNotesPerGroup:
            chord.append(note)
    for chord in chords:
        chord.sort(key=lambda n: n[3], reverse=True)
    return chords


def applySymbols(symbols: List[_symbols.Symbol],
                 notations: Union[scoring.Notation, List[scoring.Notation]]) -> None:
    for symbol in symbols:
        if isinstance(symbol, _symbols.Dynamic):

            notations[0].dynamic = symbol.kind
        elif isinstance(symbol, _symbols.Notehead):
            for n in notations:
                if symbol.kind:
                    n.notehead = symbol.kind
                if symbol.color:
                    n.color = symbol.color
                n.noteheadParenthesis = symbol.parenthesis
        elif isinstance(symbol, _symbols.Articulation):
            notations[0].articulation = symbol.kind
        elif isinstance(symbol, _symbols.Expression):
            notations[0].addAnnotation(symbol.text)





