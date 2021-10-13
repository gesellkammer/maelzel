from __future__ import annotations

import os
from dataclasses import dataclass
import music21 as m21
import PIL
import bpf4 as bpf
import pitchtools as pt

import emlib.img
import emlib.misc
import emlib.dialogs

from ._common import *
from . import environment
from . import symbols as _symbols
from .state import appstate as _appstate
from .workspace import activeConfig
from .environment import insideJupyter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from ._typedefs import *
    from maelzel import scoring
    T = TypeVar("T")


class AudiogenError(Exception): pass


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


if insideJupyter:
    from IPython.core.display import (display as jupyterDisplay, HTML as JupyterHTML,
                                      Image as JupyterImage)


def setJupyterHookForClass(cls, func, fmt='image/png'):
    """
    Register func as a displayhook for class `cls`
    """
    if not insideJupyter:
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
    if not insideJupyter:
        raise RuntimeError("Not inside a Jupyter session")

    scalefactor = activeConfig()['show.scaleFactor']
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
    if not insideJupyter:
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
    if not insideJupyter:
        logger.debug("m21JupyterHook: not inside ipython/jupyter, skipping")
        return
    from IPython.core.getipython import get_ipython
    from IPython.core import display
    from IPython.display import Image, display
    ip = get_ipython()
    formatter = ip.display_formatter.formatters['image/png']
    if enable:
        def showm21(stream: m21.stream.Stream):
            cfg = activeConfig()
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
        app: used if a specific external app is needed. Otherwise the os
            defined app is used
    """
    if insideJupyter and not forceExternal:
        jupyterShowImage(pngpath)
    else:
        environment.openPngWithExternalApplication(pngpath, app=app)


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


def midinotesNeedSplit(midinotes: List[float], splitpoint=60, margin=4
                       ) -> bool:
    if len(midinotes) == 0:
        return False
    numabove = sum(int(m > splitpoint - margin) for m in midinotes)
    numbelow = sum(int(m < splitpoint + margin) for m in midinotes)
    return bool(numabove and numbelow)


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


def applySymbols(symbols: List[_symbols.Symbol],
                 notations: Union[scoring.Notation, List[scoring.Notation]]) -> None:
    for symbol in symbols:
        if isinstance(symbol, _symbols.Dynamic):
            notations[0].dynamic = symbol.kind
        elif isinstance(symbol, _symbols.Notehead):
            for n in notations:
                n.notehead = symbol.kind
        elif isinstance(symbol, _symbols.Articulation):
            notations[0].articulation = symbol.kind
        elif isinstance(symbol, _symbols.Expression):
            notations[0].addAnnotation(symbol.text)


def _highlightLilypond(s: str) -> str:
    # TODO
    return s


def showLilypondScore(score: str) -> None:
    # TODO
    print(score)
    return

    if not environment.insideJupyter:
        print(score)
    else:
        html = _highlightLilypond(score)
        jupyterDisplay(JupyterHTML(html))


def amplitudeToDynamics(amp: float) -> str:
    from . import workspace
    w = workspace.activeWorkspace()
    dyncurve = w.dynamicsCurve
    return dyncurve.amp2dyn(amp)


def selectFromList(options: Sequence[str], title="", default=None) -> Optional[str]:
    if environment.insideJupyter():
        return emlib.dialogs.selectItem(options, title=title) or default
    else:
        # TODO: use tty tools, like fzf
        return emlib.dialogs.selectItem(options, title=title) or default


def selectFileForSave(key:str, filter="All (*.*)", prompt="Save File") -> Optional[str]:
    """
    Select a file for open via a gui dialog, remember the last directory

    Args:
        key: the key to use to remember the last directory
        filter: for example "Images (*.png, *.jpg);; Videos (*.mp4)"
        prompt: title of the dialog

    Returns:
        the selected file, or None if the operation was cancelled
    """

    lastdir = _appstate[key]
    outfile = emlib.dialogs.saveDialog(filter=filter, directory=lastdir, title=prompt)
    if outfile:
        _appstate[key] = os.path.split(outfile)[0]
    return outfile


def normalizeFilename(path: str) -> str:
    return os.path.expanduser(path)


def selectFileForOpen(key: str, filter="All (*.*)", prompt="Open", ifcancel:str=None
                      ) -> Optional[str]:
    """
    Select a file for open via a gui dialog, remember the last directory

    Args:
        key: the key to use to remember the last directory
        filter: for example "Images (*.png, *.jpg);; Videos (*.mp4)"
        prompt: title of the dialog
        ifcancel: if given and the operation is cancelled a ValueError
            with this as message is raised

    Returns:
        the selected file, or None if the operation was cancelled
    """
    lastdir = _appstate[key]
    selected = emlib.dialogs.selectFile(filter=filter, directory=lastdir, title=prompt)
    if selected:
        _appstate[key] = os.path.split(selected)[0]
    elif ifcancel is not None:
        raise ValueError(ifcancel)
    return selected


def saveRecordingDialog(prompt="Save Recording") -> Optional[str]:
    return selectFileForSave("recLastDir", "Audio (*.wav, *.aif, *.flac)",
                             prompt=prompt)



