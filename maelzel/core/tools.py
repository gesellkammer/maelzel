from __future__ import annotations

import re

import emlib.img
import music21 as m21
from emlib import misc
import bpf4 as bpf
from ._common import *
from . import environment
from fractions import Fraction
from pitchtools import str2midi, n2f, m2f, m2n, amp2db
import math
from typing import NamedTuple
from .workspace import currentConfig, currentWorkspace
from . import symbols as _symbols
from maelzel import scoring
import configdict
import PIL


class AudiogenError(Exception): pass


breakpoint_t = Tuple[float,...]


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


if misc.inside_jupyter():
    from IPython.core.display import (display as jupyterDisplay, 
                                      Image as JupyterImage)


def setJupyterHookForClass(cls, func, fmt='image/png'):
    """
    Register func as a displayhook for class `cls`
    """
    if not misc.inside_jupyter():
        logger.debug("_setJupyterHookForClass: not inside IPython/jupyter, skipping")
        return
    import IPython
    ip = IPython.get_ipython()
    formatter = ip.display_formatter.formatters[fmt]
    return formatter.for_type(cls, func)


def imgSize(path:str) -> Tuple[int, int]:
    """ returns (width, height) """
    im = PIL.Image.open(path)
    return im.size


def jupyterMakeImage(path: str) -> JupyterImage:
    """
    Makes a jupyter Image, which can be displayed inline inside
    a notebook

    Args:
        path: the path to the image file

    Returns:
        an IPython.core.display.Image

    """
    if not misc.inside_jupyter():
        raise RuntimeError("Not inside a Jupyter session")

    scalefactor = currentConfig()['show.scaleFactor']
    if scalefactor != 1.0:
        imgwidth, imgheight = imgSize(path)
        width = imgwidth*scalefactor
    else:
        width = None
    return JupyterImage(filename=path, embed=True, width=width)


def jupyterShowImage(path: str):
    """
    Show an image inside (inline) of a jupyter notebook

    Args:
        path: the path to the image file

    """
    if not misc.inside_jupyter():
        logger.error("jupyter is not available")
        return

    img = jupyterMakeImage(path)
    return jupyterDisplay(img)


def m21JupyterHook(enable=True) -> None:
    """
    Set an ipython-hook to display music21 objects inline on the
    ipython notebook

    Args:
        enable: if True, the hook will be set up and enabled
            if False, the hook is removed
    """
    if not misc.inside_jupyter():
        logger.debug("m21JupyterHook: not inside ipython/jupyter, skipping")
        return
    from IPython.core.getipython import get_ipython
    from IPython.core import display
    from IPython.display import Image, display
    ip = get_ipython()
    formatter = ip.display_formatter.formatters['image/png']
    if enable:
        def showm21(stream: m21.stream.Stream):
            cfg = currentConfig()
            fmt = cfg['m21.displayhook.format']
            filename = str(stream.write(fmt))
            if fmt.endswith(".png") and cfg['html.theme'] == 'dark':
                emlib.img.pngRemoveTransparency(filename)
            return display(Image(filename=filename))
            # return display.Image(filename=filename)._repr_png_()

        dpi = formatter.for_type(m21.Music21Object, showm21)
        return dpi
    else:
        logger.debug("disabling display hook")
        formatter.for_type(m21.Music21Object, None)


def pngShow(pngpath:str, forceExternal=False, app:str='') -> None:
    """
    Show a png either inside jupyter or with an external app

    Args:
        pngpath: the path to a png file
        forceExternal: if True, it will show in an external app even
            inside jupyter. Otherwise it will show inside an external
            app if running a normal session and show an embedded
            image if running inside a notebook

    """
    if misc.inside_jupyter() and not forceExternal:
        jupyterShowImage(pngpath)
    else:
        environment.viewPng(pngpath, app=app)


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
        return str2midi(x)
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
        return n2f(n)
    elif isinstance(n, (int, float)):
        return m2f(n)
    elif hasattr(n, "freq"):
        return n.freq
    else:
        raise ValueError(f"cannot convert {n} to a frequency")


def notes2ratio(n1, n2, maxdenominator=16) -> Fraction:
    """
    find the ratio between n1 and n2

    n1, n2: notes -> "C4", or midinote (do not use frequencies)

    Returns: a Fraction with the ratio between the two notes

    NB: to obtain the ratios of the harmonic series, the second note
        should match the intonation of the corresponding overtone of
        the first note

    C4 : D4       --> 8/9
    C4 : Eb4+20   --> 5/6
    C4 : E4       --> 4/5
    C4 : F#4-30   --> 5/7
    C4 : G4       --> 2/3
    C4 : A4       --> 3/5
    C4 : Bb4-30   --> 4/7
    C4 : B4       --> 8/15
    """
    f1, f2 = asfreq(n1), asfreq(n2)
    return Fraction.from_float(f1/f2).limit_denominator(maxdenominator)


def midinotesNeedSplit(midinotes: List[float], splitpoint=60, margin=4
                       ) -> bool:
    if len(midinotes) == 0:
        return False
    numabove = sum(int(m > splitpoint - margin) for m in midinotes)
    numbelow = sum(int(m < splitpoint + margin) for m in midinotes)
    return bool(numabove and numbelow)


class NoteComponent(NamedTuple):
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
        db = amp2db(amp)
        notes.append(NoteComponent(m2n(midi), midi, m2f(midi), db, int(step(db))))
    chords: List[List[NoteComponent]] = [[] for _ in range(numGroups)]
    notes2 = sorted(notes, key=lambda n: n[3], reverse=True)
    for note in notes2:
        chord = chords[note[4]]
        if len(chord) <= maxNotesPerGroup:
            chord.append(note)
    for chord in chords:
        chord.sort(key=lambda n: n[3], reverse=True)
    return chords


def analyzeAudiogen(audiogen:str, check=True) -> dict:
    """
    Analyzes the audio generating part of an instrument definition,
    returns the analysis results as a dictionary

    Args:
        audiogen: as passed to play.defPreset
        check: if True, will check that audiogen is well formed

    Returns:
        a dict with keys:
            numSignals (int): number of a_ variables
            minSignal: min. index of a_ variables
            maxSignal: max. index of a_ variables (minSignal+numSignals=maxSignal)
            numOutchs: number of
                (normally minsignal+numsignals = maxsignal)
    """
    audiovarRx = re.compile(r"\ba[0-9]\b")
    outOpcodeRx = re.compile(r"\boutch\b")
    audiovarsList = []
    numOutchs = 0
    for line in audiogen.splitlines():
        foundAudiovars = audiovarRx.findall(line)
        audiovarsList.extend(foundAudiovars)
        outOpcode = outOpcodeRx.fullmatch(line)
        if outOpcode is not None:
            opcode = outOpcode.group(0)
            args = line.split(opcode)[1].split(",")
            assert len(args)%2 == 0
            numOutchs = len(args) // 2

    audiovars = set(audiovarsList)
    chans = [int(v[1:]) for v in audiovars]
    maxchan = max(chans)
    # check that there is an audiovar for each channel
    if check:
        if len(audiovars) == 0:
            raise AudiogenError("audiogen defines no output signals (a_ variables)")

        for i in range(maxchan+1):
            if f"a{i}" not in audiovars:
                raise AudiogenError("Not all channels are defined", i, audiovars)

    needsRouting = numOutchs == 0
    numSignals = len(audiovars)
    if needsRouting:
        numOuts = int(math.ceil(numSignals/2))*2
    else:
        numOuts = numOutchs

    out = {
        'signals': audiovars,
        # number of signals defined in the audiogen
        'numSignals': numSignals,
        'minSignal': min(chans),
        'maxSignal': max(chans),
        # if numOutchs is 0, the audiogen does not implement routing
        'numOutchs': numOutchs,
        'needsRouting': needsRouting,
        'numOutputs': numOuts
    }
    return out


def showTime(f) -> str:
    if f is None:
        return "None"
    return f"{float(f):.3g}"


def addColumn(mtx: List[List[T]], col: List[T], inplace=False) -> List[List[T]]:
    """
    Add a column to a list of lists

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

    raise TypeError(f"mtx should be a seq. of lists, "
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


def applySymbols(symbols: List[_symbols.Symbol],
                 notations: U[scoring.Notation, List[scoring.Notation]]) -> None:
    for symbol in symbols:
        if isinstance(symbol, _symbols.Dynamic):
            notations[0].dynamic = symbol.kind
        elif isinstance(symbol, _symbols.Notehead):
            for n in notations:
                n.notehead = symbol.kind
        elif isinstance(symbol, _symbols.Articulation):
            notations[0].articulation = symbol.kind