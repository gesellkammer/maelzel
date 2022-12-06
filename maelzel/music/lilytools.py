from __future__ import annotations
import os
import sys
import logging
import subprocess
import tempfile
import re
import textwrap

import pitchtools as pt
from emlib import filetools
from emlib import misc
from functools import cache
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    pitch_t = Union[int, float, str]


logger = logging.getLogger("maelzel")


class PlatformNotSupported(Exception):
    pass


@dataclass
class _CallResult:
    returnCode: int
    stdout: str
    stderr: str


def _addLineNumbers(s: str, start=1) -> Iterator[str]:
    lines = s.splitlines()
    numZeros = len(str(len(lines)))
    fmt = f"%0{numZeros}d %s"
    for i, l in enumerate(lines, start=start):
        yield fmt % (i, l)


def callWithCapturedOutput(args: Union[str, List[str]], shell=False) -> _CallResult:
    """
    Call a subprocess with params

    Returns output, return code, error message
    """
    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=shell)
    return _CallResult(proc.wait(),
                       proc.stdout.read().decode('utf-8'),
                       proc.stderr.read().decode('utf-8'))


def _checkOutput(args: List[str], encoding="utf-8") -> Optional[str]:
    """
    Like subprocess.check_output, but returns None if failed instead
    of throwing an exeception
    """
    try:
        out = subprocess.check_output(args)
        return out.decode(encoding)
    except subprocess.CalledProcessError:
        return None


def findLilypond() -> Optional[str]:
    """
    Find lilypond binary, or None if not found
    """
    platform = os.uname()[0].lower()
    if platform == 'linux':
        path = _checkOutput(["which", "lilypond"])
        if path is not None:
            path = path.strip()
            assert os.path.exists(path)
            return path
        paths = ("/usr/bin/lilypond", "/usr/local/bin/lilypond",
                 "~/.local/bin/lilypond")
        paths = [os.path.expanduser(p) for p in paths]
        for path in paths:
            if os.path.exists(path):
                return path
        return None
    elif platform == 'darwin':
        paths = ['/Applications/LilyPond.app/Contents/Resources/bin/lilypond']
        paths = [os.path.expanduser(p) for p in paths]
        for path in paths:
            if os.path.exists(path):
                return path
        return None
    else:
        raise PlatformNotSupported(f"Platform {platform} is not supported")


def musicxml2ly(xmlfile: str, outfile: str = None) -> str:
    if outfile is None:
        outfile = os.path.splitext(xmlfile)[0] + '.ly'
    subprocess.call(['musicxml2ly', '-o', outfile, xmlfile])
    return outfile


def saveScore(score:str, outfile:str, book=False, microtonal=False) -> None:
    if book or microtonal:
        score = postProcessLilyScore(score, book=book, microtonal=microtonal)
    open(outfile, "w").write(score)


def renderScore(score:str, outfile:str=None,
                book=False, microtonal=False,
                openWhenFinished=False) -> str:
    lilyfile = tempfile.mktemp(suffix=".ly")
    saveScore(score, lilyfile, book=book, microtonal=microtonal)
    if outfile is None:
        outfile = tempfile.mktemp(suffix=".pdf")
    out = renderLily(lilyfile, outfile, openWhenFinished=openWhenFinished)
    os.remove(lilyfile)
    return out


def renderLily(lilyfile:str, outfile:str=None,
               removeHeader:bool=None, book:bool=None,
               imageResolution:int=None,
               openWhenFinished=False) -> Optional[str]:
    assert os.path.exists(lilyfile)
    assert imageResolution is None or imageResolution in {150, 200, 300, 600, 1200}
    if outfile is None:
        outfile = filetools.withExtension(lilyfile, 'pdf')
    fmt = os.path.splitext(outfile)[1][1:]
    assert fmt in ('pdf', 'png', 'ps')

    if fmt == "png" and removeHeader is None and book is None:
        removeHeader = True
        book = True

    if removeHeader or book:
        tmply = tempfile.mktemp(suffix=".ly")
        postProcessFile(lilyfile, tmply, removeHeader=removeHeader, book=book)
        lilyfile = tmply

    basefile = os.path.splitext(outfile)[0]
    if sys.platform == "win32":
        lilybinary = 'lilypond'
        shell = True
    else:
        lilybinary = findLilypond()
        shell = False
    args = [lilybinary, f'--{fmt}', '-o', basefile]
    if fmt == 'png' and imageResolution:
        args.append(f'-dresolution={imageResolution}')
    args.append(lilyfile)
    if shell:
        cmd = " ".join(args)
        logger.debug("Calling liliypond with shell:", cmd)
        result = callWithCapturedOutput(cmd, shell)
    else:
        logger.debug("Calling liliypond subprocess:", args)
        result = callWithCapturedOutput(args, shell)

    txt = open(lilyfile).read()
    hasMidiBlock = re.search(r'\\midi\b', txt)

    if (not hasMidiBlock and not os.path.exists(outfile)) or result.returnCode != 0:
        logger.error(f"Error while running lilypond, failed to produce a {fmt} file: {outfile}")
        logger.error(f"Return code: {result.returnCode}")
        logger.error("stdout: \n" + textwrap.indent(result.stdout, "!! "))
        logger.error("stderr: \n" + textwrap.indent(result.stderr, "!! "))
        logger.info("Contents of the lilypond file: ")
        lilysource = open(lilyfile).read()
        lilysource = "\n".join(_addLineNumbers(lilysource))
        logger.info(textwrap.indent(lilysource, " "))
        return None
    elif result.stderr or result.stdout:
        logger.debug("lilypond executed OK")
        if result.stdout.strip():
            logger.debug("stdout: ")
            logger.debug(textwrap.indent(result.stdout, " "))
        elif result.stderr:
            logger.debug("stderr: ")
            logger.debug(textwrap.indent(result.stderr, " "))

    if openWhenFinished:
        misc.open_with_app(outfile)

    return outfile


_microtonePrelude = r"""

% \version "2.19.22"

% adapted from http://lsr.di.unimi.it/LSR/Item?id=784

% Define the alterations as fraction of the equal-tempered whole tone.
#(define-public SEVEN-E-SHARP  7/8)
#(define-public SHARP-RAISE    5/8)
#(define-public SHARP-LOWER    3/8)
#(define-public NATURAL-RAISE  1/8)
#(define-public NATURAL-LOWER -1/8)
#(define-public FLAT-RAISE    -3/8)
#(define-public FLAT-LOWER    -5/8)
#(define-public SEVEN-E-FLAT  -7/8)

% Note names can now be defined to represent these pitches in our
% Lilypond input.  We extend the list of Dutch note names:
arrowedPitchNames =  #`(
                   (ceses . ,(ly:make-pitch -1 0 DOUBLE-FLAT))
                   (cesqq . ,(ly:make-pitch -1 0 SEVEN-E-FLAT))
                   (ceseh . ,(ly:make-pitch -1 0 THREE-Q-FLAT))
                   (ceseq . ,(ly:make-pitch -1 0 FLAT-LOWER))
                   (ces   . ,(ly:make-pitch -1 0 FLAT))
                   (cesiq . ,(ly:make-pitch -1 0 FLAT-RAISE))
                   (ceh   . ,(ly:make-pitch -1 0 SEMI-FLAT))
                   (ceq   . ,(ly:make-pitch -1 0 NATURAL-LOWER))
                   (c     . ,(ly:make-pitch -1 0 NATURAL))
                   (ciq   . ,(ly:make-pitch -1 0 NATURAL-RAISE))
                   (cih   . ,(ly:make-pitch -1 0 SEMI-SHARP))
                   (ciseq . ,(ly:make-pitch -1 0 SHARP-LOWER))
                   (cis   . ,(ly:make-pitch -1 0 SHARP))
                   (cisiq . ,(ly:make-pitch -1 0 SHARP-RAISE))
                   (cisih . ,(ly:make-pitch -1 0 THREE-Q-SHARP))
                   (cisqq . ,(ly:make-pitch -1 0 SEVEN-E-SHARP))
                   (cisis . ,(ly:make-pitch -1 0 DOUBLE-SHARP))

                   (deses . ,(ly:make-pitch -1 1 DOUBLE-FLAT))
                   (desqq . ,(ly:make-pitch -1 1 SEVEN-E-FLAT))
                   (deseh . ,(ly:make-pitch -1 1 THREE-Q-FLAT))
                   (deseq . ,(ly:make-pitch -1 1 FLAT-LOWER))
                   (des   . ,(ly:make-pitch -1 1 FLAT))
                   (desiq . ,(ly:make-pitch -1 1 FLAT-RAISE))
                   (deh   . ,(ly:make-pitch -1 1 SEMI-FLAT))
                   (deq   . ,(ly:make-pitch -1 1 NATURAL-LOWER))
                   (d     . ,(ly:make-pitch -1 1 NATURAL))
                   (diq   . ,(ly:make-pitch -1 1 NATURAL-RAISE))
                   (dih   . ,(ly:make-pitch -1 1 SEMI-SHARP))
                   (diseq . ,(ly:make-pitch -1 1 SHARP-LOWER))
                   (dis   . ,(ly:make-pitch -1 1 SHARP))
                   (disiq . ,(ly:make-pitch -1 1 SHARP-RAISE))
                   (disih . ,(ly:make-pitch -1 1 THREE-Q-SHARP))
                   (disqq . ,(ly:make-pitch -1 1 SEVEN-E-SHARP))
                   (disis . ,(ly:make-pitch -1 1 DOUBLE-SHARP))

                   (eeses . ,(ly:make-pitch -1 2 DOUBLE-FLAT))
                   (eesqq . ,(ly:make-pitch -1 2 SEVEN-E-FLAT))
                   (eeseh . ,(ly:make-pitch -1 2 THREE-Q-FLAT))
                   (eeseq . ,(ly:make-pitch -1 2 FLAT-LOWER))
                   (ees   . ,(ly:make-pitch -1 2 FLAT))
                   (eesiq . ,(ly:make-pitch -1 2 FLAT-RAISE))
                   (eeh   . ,(ly:make-pitch -1 2 SEMI-FLAT))
                   (eeq   . ,(ly:make-pitch -1 2 NATURAL-LOWER))
                   (e     . ,(ly:make-pitch -1 2 NATURAL))
                   (eiq   . ,(ly:make-pitch -1 2 NATURAL-RAISE))
                   (eih   . ,(ly:make-pitch -1 2 SEMI-SHARP))
                   (eiseq . ,(ly:make-pitch -1 2 SHARP-LOWER))
                   (eis   . ,(ly:make-pitch -1 2 SHARP))
                   (eisiq . ,(ly:make-pitch -1 2 SHARP-RAISE))
                   (eisih . ,(ly:make-pitch -1 2 THREE-Q-SHARP))
                   (eisqq . ,(ly:make-pitch -1 2 SEVEN-E-SHARP))
                   (eisis . ,(ly:make-pitch -1 2 DOUBLE-SHARP))

                   (feses . ,(ly:make-pitch -1 3 DOUBLE-FLAT))
                   (fesqq . ,(ly:make-pitch -1 3 SEVEN-E-FLAT))
                   (feseh . ,(ly:make-pitch -1 3 THREE-Q-FLAT))
                   (feseq . ,(ly:make-pitch -1 3 FLAT-LOWER))
                   (fes   . ,(ly:make-pitch -1 3 FLAT))
                   (fesiq . ,(ly:make-pitch -1 3 FLAT-RAISE))
                   (feh   . ,(ly:make-pitch -1 3 SEMI-FLAT))
                   (feq   . ,(ly:make-pitch -1 3 NATURAL-LOWER))
                   (f     . ,(ly:make-pitch -1 3 NATURAL))
                   (fiq   . ,(ly:make-pitch -1 3 NATURAL-RAISE))
                   (fih   . ,(ly:make-pitch -1 3 SEMI-SHARP))
                   (fiseq . ,(ly:make-pitch -1 3 SHARP-LOWER))
                   (fis   . ,(ly:make-pitch -1 3 SHARP))
                   (fisiq . ,(ly:make-pitch -1 3 SHARP-RAISE))
                   (fisih . ,(ly:make-pitch -1 3 THREE-Q-SHARP))
                   (fisqq . ,(ly:make-pitch -1 3 SEVEN-E-SHARP))
                   (fisis . ,(ly:make-pitch -1 3 DOUBLE-SHARP))

                   (geses . ,(ly:make-pitch -1 4 DOUBLE-FLAT))
                   (gesqq . ,(ly:make-pitch -1 4 SEVEN-E-FLAT))
                   (geseh . ,(ly:make-pitch -1 4 THREE-Q-FLAT))
                   (geseq . ,(ly:make-pitch -1 4 FLAT-LOWER))
                   (ges   . ,(ly:make-pitch -1 4 FLAT))
                   (gesiq . ,(ly:make-pitch -1 4 FLAT-RAISE))
                   (geh   . ,(ly:make-pitch -1 4 SEMI-FLAT))
                   (geq   . ,(ly:make-pitch -1 4 NATURAL-LOWER))
                   (g     . ,(ly:make-pitch -1 4 NATURAL))
                   (giq   . ,(ly:make-pitch -1 4 NATURAL-RAISE))
                   (gih   . ,(ly:make-pitch -1 4 SEMI-SHARP))
                   (giseq . ,(ly:make-pitch -1 4 SHARP-LOWER))
                   (gis   . ,(ly:make-pitch -1 4 SHARP))
                   (gisiq . ,(ly:make-pitch -1 4 SHARP-RAISE))
                   (gisih . ,(ly:make-pitch -1 4 THREE-Q-SHARP))
                   (gisqq . ,(ly:make-pitch -1 4 SEVEN-E-SHARP))
                   (gisis . ,(ly:make-pitch -1 4 DOUBLE-SHARP))

                   (aeses . ,(ly:make-pitch -1 5 DOUBLE-FLAT))
                   (aesqq . ,(ly:make-pitch -1 5 SEVEN-E-FLAT))
                   (aeseh . ,(ly:make-pitch -1 5 THREE-Q-FLAT))
                   (aeseq . ,(ly:make-pitch -1 5 FLAT-LOWER))
                   (aes   . ,(ly:make-pitch -1 5 FLAT))
                   (aesiq . ,(ly:make-pitch -1 5 FLAT-RAISE))
                   (aeh   . ,(ly:make-pitch -1 5 SEMI-FLAT))
                   (aeq   . ,(ly:make-pitch -1 5 NATURAL-LOWER))
                   (a     . ,(ly:make-pitch -1 5 NATURAL))
                   (aiq   . ,(ly:make-pitch -1 5 NATURAL-RAISE))
                   (aih   . ,(ly:make-pitch -1 5 SEMI-SHARP))
                   (aiseq . ,(ly:make-pitch -1 5 SHARP-LOWER))
                   (ais   . ,(ly:make-pitch -1 5 SHARP))
                   (aisiq . ,(ly:make-pitch -1 5 SHARP-RAISE))
                   (aisih . ,(ly:make-pitch -1 5 THREE-Q-SHARP))
                   (aisqq . ,(ly:make-pitch -1 5 SEVEN-E-SHARP))
                   (aisis . ,(ly:make-pitch -1 5 DOUBLE-SHARP))

                   (beses . ,(ly:make-pitch -1 6 DOUBLE-FLAT))
                   (besqq . ,(ly:make-pitch -1 6 SEVEN-E-FLAT))
                   (beseh . ,(ly:make-pitch -1 6 THREE-Q-FLAT))
                   (beseq . ,(ly:make-pitch -1 6 FLAT-LOWER))
                   (bes   . ,(ly:make-pitch -1 6 FLAT))
                   (besiq . ,(ly:make-pitch -1 6 FLAT-RAISE))
                   (beh   . ,(ly:make-pitch -1 6 SEMI-FLAT))
                   (beq   . ,(ly:make-pitch -1 6 NATURAL-LOWER))
                   (b     . ,(ly:make-pitch -1 6 NATURAL))
                   (biq   . ,(ly:make-pitch -1 6 NATURAL-RAISE))
                   (bih   . ,(ly:make-pitch -1 6 SEMI-SHARP))
                   (biseq . ,(ly:make-pitch -1 6 SHARP-LOWER))
                   (bis   . ,(ly:make-pitch -1 6 SHARP))
                   (bisiq . ,(ly:make-pitch -1 6 SHARP-RAISE))
                   (bisih . ,(ly:make-pitch -1 6 THREE-Q-SHARP))
                   (bisqq . ,(ly:make-pitch -1 6 SEVEN-E-SHARP))
                   (bisis . ,(ly:make-pitch -1 6 DOUBLE-SHARP)))
pitchnames = \arrowedPitchNames
#(ly:parser-set-note-names pitchnames)

% The symbols for each alteration
arrowGlyphs = #`(
        ( 1                     . "accidentals.doublesharp")
        (,SEVEN-E-SHARP         . "accidentals.sharp.slashslashslash.stemstem")
        ( 3/4                   . "accidentals.sharp.slashslash.stemstemstem")
        (,SHARP-RAISE           . "accidentals.sharp.arrowup")
        ( 1/2                   . "accidentals.sharp")
        (,SHARP-LOWER           . "accidentals.sharp.arrowdown")
        ( 1/4                   . "accidentals.sharp.slashslash.stem")
        (,NATURAL-RAISE         . "accidentals.natural.arrowup")
        ( 0                     . "accidentals.natural")
        (,NATURAL-LOWER         . "accidentals.natural.arrowdown")
        (-1/4                   . "accidentals.mirroredflat")
        (,FLAT-RAISE            . "accidentals.flat.arrowup")
        (-1/2                   . "accidentals.flat")
        (,FLAT-LOWER            . "accidentals.flat.arrowdown")
        (-3/4                   . "accidentals.mirroredflat.flat")
        (,SEVEN-E-FLAT          . "accidentals.flatflat.slash")
        (-1                     . "accidentals.flatflat")
)

% The glyph-list needs to be loaded into each object that
%  draws accidentals.
\layout {
  \context {
    \Score
    \override KeySignature.glyph-name-alist = \arrowGlyphs
    \override Accidental.glyph-name-alist = \arrowGlyphs
    \override AccidentalCautionary.glyph-name-alist = \arrowGlyphs
    \override TrillPitchAccidental.glyph-name-alist = \arrowGlyphs
    \override AmbitusAccidental.glyph-name-alist = \arrowGlyphs
  }
  \context {
    \Staff
    extraNatural = ##f % this is a workaround for bug #1701
  }
}

"""


def postProcessLilyScore(score:str, removeHeader=False, book=False,
                         microtonal=False) -> str:
    """
    Apply some post-processing options to a lilypond score

    Args:
        score: the lilypond score, as string
        removeHeader: remove the header section
        book: apply the book-preamble to this score
        microtonal: if True, a microtonal prelude is added which enables
            to use eighth tones via the suffixes -iq and -eq

    Returns:
        the modified score
    """

    def _remove_header(s):
        header = re.search(r"\\header\s?\{[^\}]+\}", s)
        if header:
            s = s[:header.span()[0]]+'\n\\header {}\n'+s[header.span()[1]:]
        return s

    def _add_preamble(s, book=False, microtonal=False):
        version = re.search(r"\\version.+", s)
        preamble = []
        if book:
            preamble.append('\n\\include "lilypond-book-preamble.ly"\n')
        if microtonal:
            preamble.append(_microtonePrelude)
        preamblestr = "\n".join(preamble)
        if version:
            s = s[:version.span()[1]]+preamblestr+s[version.span()[1]:]
        else:
            s = "\n".join([preamblestr, s])
        return s

    if removeHeader:
        score = _remove_header(score)
    if book or microtonal:
        score = _add_preamble(score, book=book, microtonal=microtonal)
    return score


def postProcessFile(lilyfile: str, outfile: str=None, removeHeader=True,
                    book=True) -> None:
    s = open(lilyfile).read()
    s = postProcessLilyScore(s, removeHeader=removeHeader, book=book)
    if outfile is None:
        outfile = lilyfile
    open(outfile, "w").write(s)


_octaveMapping = {
    -1: ",,,,",
    0: ",,,",
    1: ",,",
    2: ",",
    3: "",
    4: "'",
    5: "''",
    6: "'''",
    7: "''''",
    8: "'''''"
}

_centsToSuffix = {
    0  : '',
    25 : 'iq',
    50 : 'ih',
    75 : 'iseq',
    100: 'is',
    125: 'isiq',
    150: 'isih',

    -25 : 'eq',
    -50 : 'eh',
    -75 : 'esiq',
    -100: 'es',
    -125: 'eseq',
    -150: 'eseh'
}


def lilyOctave(octave:int) -> str:
    """
    Convert an octave number to its lilypond representation

    ...
    2 -> ,,
    3 -> ,
    4 -> '
    5 -> ''
    ...
    """
    return _octaveMapping[octave]


def pitchName(pitchclass: str, cents: int) -> str:
    """
    Convert a note and a cents deviation from it to its
    lilypond representation

    Args:
        pitchclass: the basenote without any alteration (a, b, c, ...)
        cents: the cents deviation. 100=sharp, -100=flat, etc. The cents
            deviation should be quantized to either 1/4 tones (0, 50, 100, ...)
            or eighth tones (0, 25, 50, 75, ...)
    """
    suffix = _centsToSuffix.get(cents)
    if suffix is None:
        raise ValueError(f"Invalid cents value: {cents}, expected one of -150, -125,..., 0, 25, 50,..., 150")
    return pitchclass.lower() + _centsToSuffix[cents]


def notenameToLily(notename: str, divsPerSemitone=4) -> str:
    """
    Convert a notename to its lilypond representation.
    A notename is a string as understood by pitchtools.n2m
    (for example "4C#+20", "Db4-25", etc.). It will be quantized
    to the nearest microtone, determined by divsPerSemitone

    Args:
        notename: the note to convert to lilypond
        divsPerSemitone: the number of divisions of the semitone
            (use 2 for a 1/4 tone resolution,4 for a 1/8 tone resolution)

    Returns:
        the corresponding lilypond representation.
    """

    notename = pt.quantize_notename(notename, divisions_per_semitone=divsPerSemitone)
    octave, letter, alteration, cents = pt.split_notename(notename, default_octave=-2)
    if alteration:
        cents += pt.alteration_to_cents(alteration)
    lilyoctave = lilyOctave(octave) if octave >= -1 else ''
    pitchname = pitchName(letter, cents)
    return pitchname + lilyoctave


_durationToLily = {
    'whole': '1',
    'half': '2',
    'quarter': '4',
    'eighth': '8',
    '16th': '16',
    '32nd': '32',
    '64th': '64',
    0.03125: '128',
    0.046875: '128.',
    0.0625:  '64',
    0.09375: '64.',
    0.109375:'64..',
    0.125:  '32',
    0.1875: '32.',
    0.21875:'32..',
    0.25:  '16',
    0.375: '16.',
    0.4375:'16..',
    0.5:  '8',
    0.75: '8.',
    0.875:'8..',
    1:   '4',
    1.5: '4.',
    1.75:'4..',
    2:'2',
    3:'2.',
    3.5:'2..',
    3.75:'2...',
    4:'1',
    6:'1.',
    7:'1..'
}


def isValidLilypondDuration(s: str) -> bool:
    """
    is this a valid lilypond duration
    """
    if "." in s:
        basedur, extradots = s.split(".", maxsplit=1)
        # numdots = len(extradots) + 1
    else:
        basedur = s
    if basedur not in {'1', '2', '4', '8', '16', '32', '64', '128'}:
        return False
    return True


def makeDuration(quarterLength: Union[int, float, str], dots=0) -> str:
    """
    Args:
        quarterLength: the duration as a fraction of a quarter-note. Possible string
            values: 'quarter', 'eighth', '16th', etc
        dots: the number of dots
    """
    if isinstance(quarterLength, str):
        # is it a lilypond duration already?
        if isValidLilypondDuration(quarterLength):
            return quarterLength
        lilydur = _durationToLily[quarterLength]
    elif isinstance(quarterLength, int):
        assert quarterLength in {1, 2, 4}, f"quarterLength: {quarterLength}"
        lilydur = _durationToLily[quarterLength]
    elif isinstance(quarterLength, float):
        if int(quarterLength) == quarterLength:
            lilydur = _durationToLily[int(quarterLength)]
        else:
            lilydur = _durationToLily[quarterLength]
            if dots > 0:
                raise ValueError("Dots can't be used when giving a duration as a float")
    else:
        raise TypeError(f"Expected a str, int or float, got {quarterLength} ({type(quarterLength)})")
    return lilydur + "." * dots


noteheadStyles = {
    'default',
    'harmonic',
    'cross',
    'xcircle',
    'triangle',
    'harmonic-black',
    'do',
    're',
    'mi',
    'fa',
    'la',
    'diamond',
    'slash',
}

def customNotehead(notehead: str = 'default', parenthesis: bool = False, color: str = None,
                   sizeFactor: float = None
                   ) -> str:
    """

    Args:
        notehead: one of 'cross', 'harmonic', 'triangleup', 'xcircle', 'triangle',
            'rhombus', 'square', 'rectangle'
        parenthesis: if True, enclose the notehead in a parenthesis
        color: the color of the notehead. Must be a css color color or #RRGGBB
            (see https://lilypond.org/doc/v2.23/Documentation/notation/inside-the-staff#coloring-objects)
        sizeFactor: a size factor applied to the notehead (1.0 indicates the default size)

    Returns:
        the lilypond text to be placed **before** the note

    """
    # TODO: take account of parenthesis
    parts = []
    if notehead is not None and notehead != "default":
        assert notehead in noteheadStyles, f"{notehead=}, {noteheadStyles=}"
        parts.append(rf"\once \override NoteHead.style = #'{notehead}")
    if color:
        parts.append(rf'\once \override NoteHead.color = "{color}"')
    if sizeFactor is not None and sizeFactor != 1.0:
        relsize = fontSizeFactorToRelativeSize(sizeFactor)
        parts.append(rf'\once \override NoteHead.font-size =#{relsize}')
    if parenthesis:
        parts.append(r'\parenthesize')

    return " ".join(parts) if parts else ''


def fontSizeFactorToRelativeSize(factor: float) -> int:
    """
    Convert a fontsize factor to lilypond's relative size

    From the manual: "The fontSize value is a number indicating the
    size relative to the standard size for the current staff height.
    The default fontSize is 0; adding 6 to any fontSize value doubles
    the printed size of the glyphs, and subtracting 6 halves the size.
    Each step increases the size by approximately 12%."

    This is in fact a decibel scale

    Args:
        factor: the size factor, where 1.0 means default size and 2.0
            indicates a doubling of the size

    Returns:
        the relative size, where 0 means default and 6 indicates a
        doubling of the size. The returned value is rounded to the
        nearest int
    """
    relsize = pt.amp2db(factor)
    return round(relsize)


def makePitch(pitch: pitch_t,
              divsPerSemitone:int=4,
              parenthesizeAccidental=False,
              forceAccidental=False,
              ) -> str:
    """
    Create the liylpond text to render the given pitch

    Args:
        pitch: a fractional midinote or a notename
        divsPerSemitone: the resolution of the pitch (num. divisions per semitone)
        parenthesizeAccidental: should the accidental, if any, be within parenthesis?
        forceAccidental: if True, force the given accidental

    Returns:
        the lilypond text to render the given pitch (needs a duration suffix)

    """
    if isinstance(pitch, (int, float)):
        assert pitch >= 12, f"Pitch too low: {pitch}"
        notename = pt.m2n(pitch)
    elif isinstance(pitch, str):
        notename = pitch
        assert pt.is_valid_notename(notename), f"Invalid notename: {notename}"
    else:
        raise TypeError(f"Expected a midinote or a notename, got {pitch} (type: {type(pitch)})")
    lilypitch = notenameToLily(notename, divsPerSemitone=divsPerSemitone)
    if forceAccidental:
        lilypitch += '!'
    if parenthesizeAccidental:
        lilypitch += '?'
    return lilypitch


_clefToLilypondClef = {
    'g': 'treble',
    'treble': 'treble',
    'violin': 'treble',
    'treble8': 'treble^8',
    'treble8a': 'treble^8',
    'treble15': 'treble^15',
    'treble15a': 'treble^15',
    'f': 'bass',
    'bass': 'bass',
    'bass8': 'bass_8',
    'bass8b': 'bass_8',
    'bass15': 'bass_15',
    'bass15b': 'bass_15',
    'alto': 'alto',
    'viola': 'alto',
    'tenor': 'tenor'
}


def keySignature(fifths: int, mode='major') -> str:
    """
    Traditional (circle of fifths) key signature

    Args:
        fifths: the number of fifths, > 0 indicates sharps, < 0 indicates flats
        mode: 'major' or 'minor'

    Returns:
        the corresponding lilypond code (for example '\key a major')

    Example
    ~~~~~~~

        >>> keySignature(3, 'major')
        \key a major

    """
    # \key f \major
    keys = {
        ('sharp', 'major'): ('c', 'g', 'd', 'a',  'e',  'b',  'fis', 'cis', 'gis', 'dis'),
        ('sharp', 'minor'): ('a', 'e', 'b', 'fis', 'cis', 'gis', 'dis', 'ais'),
        ('flat', 'major'): ('c', 'f', 'bes', 'ees', 'aes', 'des', 'ges'),
        ('flat', 'minor'): ('a', 'd', 'g', 'c', 'f', 'bes', 'ees')
    }
    direction = 'sharp' if fifths >= 0 else 'flat'
    key = keys[(direction, mode)][abs(fifths)]
    return fr'\key {key} \{mode}'


def makeClef(clef: str) -> str:
    """
    Create a lilypond clef indication from the clef given
    
    .. note::
    
        clef can be one of treble, bass or alto. To indicate octave displacement
        add an '8' and 'a' for 8va alta, and 'b' for octava bassa. If not specified
        'treble8' indicates *8va bassa* and 'bass8' indicates *8va bassa*. Also
        possible is the '15' modifier for two octaves (higher or lower). 
    
    Args:
        clef: one of treble, bass, treble8, bass8, alto, treble15, bass15

    Returns:
        the lilypond clef representation
    """
    lilyclef = _clefToLilypondClef.get(clef.lower())
    if lilyclef is None:
        raise ValueError(f"Unknown clef {clef}. "
                         f"Possible values: {_clefToLilypondClef.keys()}")
    if "^" in lilyclef or "_" in lilyclef:
        lilyclef = '"' + lilyclef + '"'
    return r"\clef " + lilyclef


def colorFlag(color: str) -> str:
    return rf'\override Flag.color = "{color}"'

def colorStem(color: str) -> str:
    return rf'\override Stem.color = "{color}"'


def makeNote(pitch: pitch_t, duration: Union[float, str], dots=0, tied=False,
             divsPerSemitone=4, noteheadcolor: str = None,
             notehead: str = None, parenthesis=False,
             cautionary=False) -> str:
    """
    Returns the lilypond representation of the given note
    
    **NB**: Tuplets should be created independently

    Args:
        pitch: pitch as midinote or notename
        duration: duration as quarter length
        dots: number of dots
        tied: is this note tied?
        divsPerSemitone: pitch resolution
        stemcolor: the color of the stem (as css color)
        noteheadcolor: color of the notehead (as css color)
        parenthesis: should the notehead be within parenthesis?
        cautiniory: if True, put the accidental within parenthesis

    Returns:
        The lilypond text
    """
    parts = []
    if notehead or parenthesis or noteheadcolor:
        parts.append(customNotehead(notehead=notehead, color=noteheadcolor, parenthesis=parenthesis))
        parts.append(' ')
    parts.append(makePitch(pitch, divsPerSemitone=divsPerSemitone, parenthesizeAccidental=cautionary))
    parts.append(makeDuration(duration, dots=dots))
    if tied:
        parts.append("~")
    return "".join(parts)


@cache
def getLilypondVersion() -> Optional[str]:
    """
    Return the lilypond version as string

    **NB**: The result is cached
    """
    lilybin = findLilypond()
    if not lilybin:
        logger.error("Could not find lilypond")
        return None
    output = _checkOutput([lilybin, "--version"])
    match = re.search(r"GNU LilyPond \d+\.\d+\.\d+", output)
    if not match:
        logger.error(f"Could not parse lilypond's output: {output}")
        return None
    return match[0][13:]


def millimetersToPoints(mm:float) -> float:
    return mm * 2.87


def pointsToMillimeters(points:float) -> float:
    return points / 2.87


def paperBlock(paperWidth:float=None,
               margin:float=None,
               leftMargin:float=None,
               rightMargin:float=None,
               lineWidth:float=None,
               topMargin:float=None,
               bottomMargin:float=None,
               indent=2,
               unit="mm"
               ) -> str:
    lines = ["\\paper {"]
    indentStr = " " * indent

    if margin is not None:
        leftMargin = rightMargin = topMargin = bottomMargin = margin

    if paperWidth is not None:
        lines.append(fr"{indentStr}paper-width = {paperWidth}\{unit}")
    if leftMargin is not None:
        lines.append(fr"{indentStr}left-margin = {leftMargin}\{unit}")
    if rightMargin is not None:
        lines.append(fr"{indentStr}right-margin = {rightMargin}\{unit}")
    if topMargin is not None:
        lines.append(fr"{indentStr}top-margin = {topMargin}\{unit}")
    if bottomMargin is not None:
        lines.append(fr"{indentStr}bottom-margin = {bottomMargin}\{unit}")
    if lineWidth is not None:
        lines.append(fr"{indentStr}line-width = {lineWidth}\{unit}")
    lines.append("}")
    return "\n".join(lines)


def makeTextMark(text: str, fontsize:int=None, fontrelative=True,
                 box='', italic=False, bold=False
                 ) -> str:
    """
    Creates a system text mark - a text above all measures

    Args:
        text: the text
        fontsize: the size of the text.
        fontrelative: font size is relative or absolute?
        box: one of 'square', 'circle', 'rounded' or '' for no box
        italic: italic?
        bold: bold?

    Returns:
        the text to add to a .ly script. Normally to be added at the beginning
        of the measure in the uppermost part
    """
    markups = []
    if fontsize:
        if fontrelative:
            markups.append(fr"\fontsize #{int(fontsize)}")
        else:
            markups.append(fr"\abs-fontsize #{int(fontsize)}")
    if box:
        if (markup := _boxMarkup.get(box)) is None:
            raise KeyError(f"Box shape {box} not supported, possible shapes are {_boxMarkup.keys()}")
        markups.append(markup)
    if italic:
        markups.append(r"\italic")
    if bold:
        markups.append(r"\bold")
    if markups:
        markupstr = " ".join(markups)
        return fr'\mark \markup {{ {markupstr} "{text}" }}'
    else:
        return fr'\mark "{text}"'


_boxMarkup = {
    'square': r'\box',
    'box': r'\box',
    'circle': r'\circle',
    'rounded': r'\rounded-box',
    'rounded-box': r'\rounded-box'
}


def makeText(text: str, fontsize:int=None, fontrelative=False,
             placement='above', italic=False, bold=False,
             box='') -> str:
    """
    Creates a lilypond text annotation to be attached to a note/rest

    **NB**: this needs to be added **AFTER** a note

    Args:
        text: the text
        fontsize: a font size, or None to use lilypond's default
        fontrelative: if True, the fontsize is relative to the default fontsize
        placement: 'above' or 'below'
        italic: if True, the text should be italic
        bold: if True, the text should be bold
        box: one of 'square', 'circle', 'rounded' or '' for no box

    Returns:
        the lilypond markup to generate the given annotation
    """
    placementchr = "^" if placement == "above" else "_"
    markups = []
    if fontsize:
        if fontrelative:
            markups.append(fr"\fontsize #{int(fontsize)}")
        else:
            markups.append(fr"\abs-fontsize #{int(fontsize)}")
    if italic:
        markups.append(r'\italic')
    if bold:
        markups.append(r'\bold')
    if box:
        if (markup := _boxMarkup.get(box)) is None:
            raise KeyError(f"Box shape {box} not supported, possible shapes are {_boxMarkup.keys()}")
        markups.append(markup)
    if markups:
        markupstr = " ".join(markups)
        return fr'{placementchr}\markup {{ {markupstr} "{text}" }}'
    return fr'{placementchr}\markup "{text}"'