from __future__ import annotations

import os
import re
import music21 as m21
from emlib import misc
from .common import *
from . import environment


import textwrap
from fractions import Fraction
from ._base import Opt, Seq, List

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Helper functions for Note, Chord, ...
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def midicents(midinote: float) -> int:
    """
    Returns the cents to next chromatic pitch

    :param midinote: a (fractional) midinote
    :return: cents to next chromatic pitch
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

    :param centsdev: the deviation from the chromatic pitch
    :param divsPerSemitone: 4 means 1/8 tones
    :return: the string to be shown alongside the notated pitch
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
    import PIL
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

    scalefactor = config.get('show.scalefactor', 1.0)
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
            fmt = config['m21.displayhook.format']
            filename = str(stream.write(fmt))
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
        return x.midi
    raise TypeError(f"Expected a str, a Note or a midinote, got {x}")


def asfreq(n) -> float:
    """
    Convert a midinote, notename of Note to a freq.
    NB: a float value is interpreted as a midinote

    Args:
        n: a note as midinote, notename or Note

    Returns:
        a frequency taking into account the A4 defined in emlib.pitch
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


def midinotesNeedSplit(midinotes, splitpoint=60, margin=4) -> bool:
    if len(midinotes) == 0:
        return False
    numabove = sum(int(m > splitpoint - margin) for m in midinotes)
    numbelow = sum(int(m < splitpoint + margin) for m in midinotes)
    return bool(numabove and numbelow)


def splitByAmp(midis: List[float], amps:List[float], numGroups=8, maxNotesPerGroup=8
               ) -> List[List[float]]:
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
    # 0         1     2     3   4
    # notename, note, freq, db, step
    for note, amp in zip(midis, amps):
        db = amp2db(amp)
        notes.append((m2n(note), note, m2f(note), db, int(step(db))))
    chords = [[] for _ in range(numGroups)]
    notes2 = sorted(notes, key=lambda n: n[3], reverse=True)
    for note in notes2:
        chord = chords[note[4]]
        if len(chord) <= maxNotesPerGroup:
            chord.append(note)
    for chord in chords:
        chord.sort(key=lambda n: n[3], reverse=True)
    return [ch for ch in chords]


def analyzeAudiogen(audiogen:str, check=True) -> dict:
    """
    Args:
        audiogen: as passed to play.defPreset
        check: if True, will check that audiogen is well formed

    Returns:
        a dict with keys:
            numSignals (int): number of a_ variables
            minSignal: min. index of a_ variables
            maxSignal: max. index of a_ variables
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
            numOutchs = len(args) / 2

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


def reindent(text, prefix="", stripEmptyLines=True):
    if stripEmptyLines:
        text = misc.strip_lines(text)
    text = textwrap.dedent(text)
    if prefix:
        text = textwrap.indent(text, prefix=prefix)
    return text


def getIndentation(code:str) -> int:
    """ get the number of spaces used to indent code """
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped:
            spaces = len(line) - len(stripped)
            return spaces
    return 0


def joinCode(codes: Seq[str]) -> str:
    """
    Like join, but preserving indentation

    Args:
        codes: a list of code strings

    Returns:

    """
    codes2 = [textwrap.dedent(code) for code in codes if code]
    code = "\n".join(codes2)
    numspaces = getIndentation(codes[0])
    if numspaces:
        code = textwrap.indent(code, prefix=" "*numspaces)
    return code


def showTime(f:Opt[F]) -> str:
    if f is None:
        return "None"
    return f"{float(f):.3f}"


