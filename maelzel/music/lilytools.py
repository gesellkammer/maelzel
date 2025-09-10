from __future__ import annotations
import os
import shutil
import sys
import subprocess
import tempfile
import re
import textwrap

import pitchtools as pt
from dataclasses import dataclass
from maelzel.common import F, getLogger
from numbers import Rational

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterator
    from maelzel.common import pitch_t


_cache = {'lilypath': ''}


class PlatformNotSupportedError(Exception):
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
    for i, line in enumerate(lines, start=start):
        yield fmt % (i, line)


def callWithCapturedOutput(args: str | list[str], shell=False) -> _CallResult:
    """
    Call a subprocess with params

    Returns output, return code, error message
    """
    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=shell)
    assert proc.stdout is not None and proc.stderr is not None
    stdout = proc.stdout.read().decode('utf-8')
    stderr = proc.stderr.read().decode('utf-8')
    return _CallResult(proc.wait(), stdout, stderr)


def _checkOutput(args: list[str], encoding="utf-8") -> str | None:
    """
    Like subprocess.check_output, but returns None if failed instead
    of throwing an exeception

    Returns:
        the output or None if there was an error
    """
    try:
        out = subprocess.check_output(args)
        return out.decode(encoding)
    except subprocess.CalledProcessError:
        return None


def _testLilypond(lilybin: str, fmt='.pdf') -> bool:
    lilytxt = r'''
\version "2.20.0"
{
    <c' d'' b''>8. ~ <c' d'' b''>8
}
    '''
    assert fmt in ('.pdf', '.png')
    lyfile = tempfile.mktemp(suffix='.ly')
    open(lyfile, 'w').write(lilytxt)
    assert os.path.exists(lyfile)
    outfile = os.path.splitext(lyfile)[0] + fmt
    try:
        outfile2 = renderLily(lyfile, outfile=outfile, removeHeader=False, lilypondBinary=lilybin)
        return outfile2 is not None and os.path.exists(outfile2)
    except Exception as e:
        getLogger(__file__).error(f"Test lilypond raised an error: {e}")
        return False


def _installLilypondMacosHomebrew() -> str | None:
    """
    install lilypond via homebrew in macos

    Assumes that homebrew is installed

    Returns:
        the path to lilypond, or None if failed
    """
    logger = getLogger(__file__)
    logger.info("Installing lilypond via homebrew: `brew install lilypond`")
    retcode = subprocess.call('brew install lilypond', shell=True)
    if retcode != 0:
        # did we fail?
        lilybin = shutil.which('lilypond')
        if lilybin and _testLilypond(lilybin):
            logger.warning(f"brew install lilypond returned code {retcode}, but "
                           f"lilypond seems installed")
            return lilybin
        logger.error(f"Lilypond could not be installed, return code {retcode}")
        return None
    if lilybin := shutil.which('lilypond'):
        logger.info("lilypond installed ok")
        return lilybin
    logger.error("brew install lilypond returned successfully but lilypond is "
                 "not in the path")
    return None


def installLilypond() -> str:
    """
    Install lilypond, returns the binary

    .. note::

        The installed lilypond will only be available for usage within python
        see pypi/lilyponddist
    """
    import lilyponddist
    lilybin = lilyponddist.lilypondbin()
    if not lilybin or not lilybin.exists():
        raise RuntimeError("Could not install lilypond")
    lilypath = lilybin.as_posix()
    _cache['lilypath'] = lilypath
    return lilypath


def findLilypond(install=True) -> str | None:
    """
    Find lilypond binary, or None if not found

    Returns:
        the path to a working lilypond binary, or None if
        the path was not found
    """
    logger = getLogger(__file__)
    if (lilypath := _cache.get('lilypath', '')):
        if os.path.exists(lilypath):
            return lilypath
        else:
            logger.warning(f"lilypond path was cached but it has become invalid. "
                           f"Previously cached path: {lilypath}")

    # try which
    logger.debug("findLilypond: searching via shutil.which")
    lilypond = shutil.which('lilypond')
    if lilypond:
        logger.debug(f"... found! lilypond path: {lilypond}")
        _cache['lilypath'] = lilypond
        return lilypond

    logger.debug("findLilypond: Lilypond is not in the path, trying lilyponddist")
    import lilyponddist
    if not lilyponddist.is_lilypond_installed() and not install:
        return None
    lilypath = lilyponddist.lilypondbin().as_posix()
    _cache['lilypath'] = lilypath
    return lilypath


def saveScore(score: str, outfile: str, book=False, microtonal=False, cropToContent=False
              ) -> None:
    if book or microtonal or cropToContent:
        score = postProcessLilyScore(score, book=book, microtonal=microtonal,
                                     cropToContent=cropToContent)

    open(outfile, "w").write(score)


def renderScore(score: str,
                outfile='',
                removeHeader=True,
                cropToContent=False,
                book=False,
                microtonal=False,
                openWhenFinished=False,
                removeTempfiles=False
                ) -> list[str]:
    """
    Render a lilypond score

    Args:
        score: the lilypond text
        outfile: the outfile (a .png or .pdf file). If not given, the lilypond
            score is rendered to a temp file
        removeHeader: remove the default header
        book: if True, add book preamble to the code
        microtonal: if True, add microtonal prelude
        openWhenFinished: if True, open the rendered file when finished
        cropToContent: crop the generated image to the content
        removeTempfiles: if True, remove tempfile

    Returns:
        the path of the generated files. In the case of pdf output, only one files is 
        generated.
        
    Raises:
        RuntimeError: if rendering failes
        
    """
    lilyfile = tempfile.mktemp(suffix=".ly")
    if removeHeader:
        score = postProcessLilyScore(score, removeHeader=True)
    saveScore(score, lilyfile, book=book, microtonal=microtonal, cropToContent=cropToContent)
    if outfile is None:
        outfile = tempfile.mktemp(suffix=".pdf")
    outfiles = renderLily(lilyfile, outfile=outfile, openWhenFinished=openWhenFinished)
    if removeTempfiles:
        os.remove(lilyfile)
    return outfiles


def show(text: str, fmt='png', external=False, maxwidth=0, snippet: bool | None = None, crop=True) -> None:
    """
    Render the given lilypond text and show it as an image

    Args:
        text: the lilypond text to render
        fmt: format, one of 'png' or 'pdf'
        external: if True, show the image using an external app, even if
            run within jupyter.
        maxwidth: a maximum width applied when showing the image
            embedded within jupyter
        snippet: if True, the text is just a snippet (what is placed inside a staff)
            and will be converted to a full score via snippetToScore
        crop: crop image to the actual music

    """
    assert fmt in ('png', 'pdf')

    if snippet:
        text = snippetToScore(text)
    elif snippet is None:
        if "\\score" not in text:
            text = snippetToScore(text)

    outfile = tempfile.mktemp(suffix='.' + fmt)
    outfiles = renderScore(text, outfile=outfile, cropToContent=crop)
    if fmt == 'png':
        if crop:
            from maelzel._imgtools import imagefileAutocrop
            for f in outfiles:
                croppedok = imagefileAutocrop(f, f, bgcolor="#FFFFFF")
                assert croppedok
        from maelzel.core import jupytertools
        # TODO: show all pages if many pages were generated!!
        jupytertools.showPng(pngpath=outfiles[0], forceExternal=external, maxwidth=maxwidth)
    else:
        import emlib.misc
        emlib.misc.open_with_app(path=outfiles[0], wait=True, min_wait=0.1)


def snippetToScore(snippet: str) -> str:
    return fr"""
\score {{
<<
  \new Staff {{
      {snippet}
  }}
>>
}}
    """


def renderLily(lilyfile: str,
               outfile='',
               removeHeader=True,
               book=False,
               imageResolution: int = None,
               openWhenFinished=False,
               lilypondBinary='',
               ) -> list[str]:
    """
    Call lilypond to render the given file

    Args:
        lilyfile: the .ly file to render
        outfile: the output file to generate (pdf, png)
        removeHeader: if True, remove the default header
        book: if True, use book formatting
        imageResolution: the image resolution in dpi when rendering to png
        openWhenFinished: if True, open the generated file when finished
        lilypondBinary: if given, use this binary for rendering

    Returns:
        the generated outfiles. Raises RuntimeError if failed to render
    """
    assert os.path.exists(lilyfile)
    assert imageResolution is None or imageResolution in {150, 200, 300, 600, 1200}
    if not outfile:
        from emlib.filetools import withExtension
        outfile = withExtension(lilyfile, 'pdf')
    fmt = os.path.splitext(outfile)[1][1:]
    assert fmt in ('pdf', 'png', 'ps')
    logger = getLogger(__file__)
    logger.debug(f"Rendering lilypond '{lilyfile}' to '{outfile}'")

    if removeHeader or book:
        tmply = tempfile.mktemp(suffix=".ly")
        postProcessFile(lilyfile, tmply, removeHeader=removeHeader, book=book)
        lilyfile = tmply

    basefile = os.path.splitext(outfile)[0]
    shell = True if sys.platform == 'win32' else False
    if not lilypondBinary:
        lilypondBinary = findLilypond()
        if not lilypondBinary:
            raise RuntimeError("lilypond binary not found")
    args = [lilypondBinary, f'--{fmt}', '-o', basefile]
    if fmt == 'png' and imageResolution:
        args.append(f'-dresolution={imageResolution}')
    args.append(lilyfile)
    if shell:
        cmd = " ".join(args)
        logger.debug(f"Calling lilypond with shell: {cmd}")
        result = callWithCapturedOutput(cmd, shell)
    else:
        logger.debug(f"Calling lilypond subprocess: {args}")
        result = callWithCapturedOutput(args, shell)

    txt = open(lilyfile).read()
    hasMidiBlock = re.search(r'\\midi\b', txt)

    if "#(ly:set-option 'crop #t)" in txt:
        # A cropped file should have been generated
        # TODO: add
        ...

    if result.returnCode != 0:
        logger.error("stdout: \n" + textwrap.indent(result.stdout, "!! "))
        logger.error("stderr: \n" + textwrap.indent(result.stderr, "!! "))
        raise RuntimeError(f"Could not render {args}. Lilypond returned error code {result.returnCode}")
    elif fmt == 'png' and not os.path.exists(outfile):
        # maybe multiple pages?
        import glob
        pages = glob.glob(f"{basefile}-page*.png")
        if pages:
            outfiles = pages
        else:
            # Ok, so no files found
            logger.error("stdout: \n" + textwrap.indent(result.stdout, "!! "))
            logger.error("stderr: \n" + textwrap.indent(result.stderr, "!! "))
            raise RuntimeError(f"Error while rendering {args}: file {outfile} not found")
    elif not os.path.exists(outfile):
        logger.error(f"Error while running lilypond (path={lilyfile}), "
                     f"failed to produce a {fmt} file: {outfile}")
        logger.error(f"Called lilypond with args: {args}")
        logger.error(f"Return code: {result.returnCode}")
        logger.error(f"Outfile: '{outfile}', exists: {os.path.exists(outfile)}")
        logger.error("stdout: \n" + textwrap.indent(result.stdout, "!! "))
        logger.error("stderr: \n" + textwrap.indent(result.stderr, "!! "))
        logger.info("Contents of the lilypond file: ")
        lilysource = open(lilyfile).read()
        lilysource = "\n".join(_addLineNumbers(lilysource))
        logger.info(textwrap.indent(lilysource, " "))
        raise RuntimeError(f"Error while rendering {args}: file {outfile} not found")
    else:
        logger.debug("lilypond executed OK")
        if result.stdout.strip():
            logger.debug("stdout: ")
            logger.debug(textwrap.indent(result.stdout, " "))
        elif result.stderr:
            logger.debug("stderr: ")
            logger.debug(textwrap.indent(result.stderr, " "))
        outfiles = [outfile]
    
    if openWhenFinished:
        from emlib import misc
        misc.open_with_app(outfile[0])

    return outfiles


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


def postProcessLilyScore(score: str,
                         removeHeader=False,
                         book=False,
                         microtonal=False,
                         cropToContent=False
                         ) -> str:
    """
    Apply some post-processing options to a lilypond score

    Args:
        score: the lilypond score, as string
        removeHeader: remove the header section and add an empty header instead (otherwise
            the default watermark is rendered)
        book: apply the book-preamble to this score
        microtonal: if True, a microtonal prelude is added which enables
            to use eighth tones via the suffixes -iq and -eq
        cropToContent: add option to crop image to content. This affects the
            outfile, so beware

    Returns:
        the modified score
    """

    def _removeHeader(s):
        header = re.search(r"\\header\s?\{[^\}]+\}", s)
        if header:
            s = s[:header.span()[0]]+'\n\\header { tagline = "" }\n'+s[header.span()[1]:]
        else:
            s = '\\header { tagline = "" }\n' + s
        return s

    def _addPreamble(s, book=False, microtonal=False, crop=False):
        version = re.search(r"\\version.+", s)
        preamble = []
        if book:
            preamble.append('\n\\include "lilypond-book-preamble.ly"\n')
        if microtonal:
            preamble.append(_microtonePrelude)
        if crop:
            preamble.append(r"#(ly:set-option 'crop #t)")
        preamblestr = "\n".join(preamble)
        if version:
            s = s[:version.span()[1]]+preamblestr+s[version.span()[1]:]
        else:
            s = "\n".join([preamblestr, s])
        return s

    if removeHeader:
        score = _removeHeader(score)
    if book or microtonal or cropToContent:
        score = _addPreamble(score, book=book, microtonal=microtonal, crop=cropToContent)

    return score


def postProcessFile(lilyfile: str, outfile='', removeHeader=True,
                    book=True) -> None:
    s = open(lilyfile).read()
    s = postProcessLilyScore(s, removeHeader=removeHeader, book=book)
    if outfile is None:
        outfile = lilyfile
    open(outfile, "w").write(s)


_octaveMapping = {
    -3: ",,,,,,",
    -2: ",,,,,",
    -1: ",,,,",
    0: ",,,",
    1: ",,",
    2: ",",
    3: "",
    4: "'",
    5: "''",
    6: "'''",
    7: "''''",
    8: "'''''",
    9: "''''''",
    10: "'''''''",
    11: "''''''''",
    12: "'''''''''"
}

_centsToSuffix = {
    0: '',
    25: 'iq',
    50: 'ih',
    75: 'iseq',
    100: 'is',
    125: 'isiq',
    150: 'isih',

    -25: 'eq',
    -50: 'eh',
    -75: 'esiq',
    -100: 'es',
    -125: 'eseq',
    -150: 'eseh'
}


def lilyOctave(octave: int) -> str:
    """
    Convert an octave number to its lilypond representation

    ...
    2 -> ,,
    3 -> ,
    4 -> '
    5 -> ''
    ...
    """
    assert isinstance(octave, int), f"Expected an int, got {octave}"
    lilyoctave = _octaveMapping.get(octave)
    if lilyoctave is None:
        octaves = _octaveMapping.keys()
        o0 = min(octaves)
        o1 = max(octaves)
        raise ValueError(f"Invalid octave. Octave should be between {o0} and {o1}, "
                         f"got {octave}")
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
    pitchclass = pitchclass.lower()
    assert pitchclass in 'abcdefg'
    suffix = _centsToSuffix.get(cents)
    if suffix is None:
        raise ValueError(f"Invalid cents value: {cents}, valid cents: {_centsToSuffix.keys()}")
    return pitchclass + suffix


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
    noteparts = pt.split_notename(notename)
    octave = noteparts.octave
    lilyoctave = lilyOctave(octave) if octave >= -1 else ''
    pitchname = pitchName(noteparts.diatonic_name, noteparts.alteration_cents + noteparts.cents_deviation)
    return pitchname + lilyoctave


_durationToLily = {
    'whole': '1',
    'half': '2',
    'quarter': '4',
    'eighth': '8',
    '16th': '16',
    '32nd': '32',
    '64th': '64',

    F(1, 32): '128',
    F(3, 64): '128.',
    F(7, 128): '128..',
    F(1, 16): '64',
    F(3, 32): '64.',
    F(7, 64): '64..',
    F(1, 8): '32',
    F(3, 16): '32.',
    F(7, 32): '32..',
    F(1, 4): '16',
    F(3, 8): '16.',
    F(7, 16): '16..',
    F(1, 2): '8',
    F(3, 4): '8.',
    F(7, 8): '8..',
    F(1, 1): '4',
    F(3, 2): '4.',
    F(7, 4): '4..',
    F(2, 1): '2',
    F(3, 1): '2.',
    F(7, 2): '2..',
    F(15, 4): '2...',

    0.03125: '128',
    0.046875: '128.',
    0.0625:  '64',
    0.09375: '64.',
    0.109375: '64..',
    0.125: '32',
    0.1875: '32.',
    0.21875: '32..',
    0.25: '16',
    0.375: '16.',
    0.4375: '16..',
    0.5: '8',
    0.75: '8.',
    0.875: '8..',
    1: '4',
    1.5: '4.',
    1.75: '4..',
    2: '2',
    3: '2.',
    3.5: '2..',
    3.75: '2...',
    4: '1',
    6: '1.',
    7: '1..'
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


def makeDuration(quarterLength: int | float | str | F, dots=0) -> str:
    """
    Args:
        quarterLength: the duration as a fraction of a quarter-note. Possible string
            values: 'quarter', 'eighth', '16th', etc
        dots: the number of dots

    Returns:
        the lilypond text corresponding to the duration
    """
    if isinstance(quarterLength, str):
        # is it a lilypond duration already?
        if isValidLilypondDuration(quarterLength):
            return quarterLength
        lilydur = _durationToLily[quarterLength]
    elif isinstance(quarterLength, int):
        assert quarterLength in {1, 2, 4}, f"quarterLength: {quarterLength}"
        lilydur = _durationToLily[quarterLength]
    elif isinstance(quarterLength, Rational):
        if quarterLength.denominator == 1:
            lilydur = _durationToLily[quarterLength.numerator]
        else:
            lilydur = _durationToLily[quarterLength]
            if dots > 0:
                raise ValueError("Dots can't be used when giving a duration as a fraction")
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


def customNotehead(notehead: str = 'default', parenthesis: bool = False, color='',
                   sizeFactor: float = None
                   ) -> str:
    """
    Creates a custom notehead in LilyPond.

    Args:
        notehead: one of 'cross', 'harmonic', 'triangleup', 'xcircle', 'triangle',
            'rhombus', 'square', 'rectangle'
        parenthesis: if True, enclose the notehead in a parenthesis
        color: the color of the notehead. Must be a css color or #RRGGBB
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
              divsPerSemitone: int = 4,
              parenthesizeAccidental=False,
              forceAccidental=False,
              ) -> str:
    """
    Create the liylpond text to render the given pitch

    Args:
        pitch: a fractional midinote or a notename. If a notename is given, the exact
            spelling of the note will be used.
        divsPerSemitone: the resolution of the pitch (num. divisions per semitone)
        parenthesizeAccidental: should the accidental, if any, be within parenthesis?
        forceAccidental: if True, force the given accidental. This adds a ! sign
            to the pitch

    Returns:
        the lilypond text to render the given pitch (needs a duration suffix)

    """
    if isinstance(pitch, (int, float)):
        assert pitch >= 12, f"Pitch too low: {pitch}"
        notename = pt.m2n(pitch)
    elif isinstance(pitch, str):
        notename = pitch
        assert pt.is_valid_notename(notename, minpitch=1), f"Invalid notename: {notename}"
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
        the corresponding lilypond code

    """
    # \key f \major
    keys = {
        ('sharp', 'major'): ('c', 'g', 'd', 'a', 'e', 'b', 'fis', 'cis', 'gis', 'dis'),
        ('sharp', 'minor'): ('a', 'e', 'b', 'fis', 'cis', 'gis', 'dis', 'ais'),
        ('flat', 'major'): ('c', 'f', 'bes', 'ees', 'aes', 'des', 'ges'),
        ('flat', 'minor'): ('a', 'd', 'g', 'c', 'f', 'bes', 'ees')
    }
    direction = 'sharp' if fifths >= 0 else 'flat'
    key = keys[(direction, mode)][abs(fifths)]
    return fr'\key {key} \{mode}'


def makeClef(clef: str, color='') -> str:
    """
    Create a lilypond clef indication from the clef given

    .. note::

        clef can be one of treble, bass or alto. To indicate octave displacement
        add an '8' and 'a' for 8va alta, and 'b' for octava bassa. If not specified
        'treble8' indicates *8va bassa* and 'bass8' indicates *8va bassa*. Also
        possible is the '15' modifier for two octaves (higher or lower).

    Args:
        clef: one of treble, bass, treble8, bass8, alto, treble15, bass15
        color: if given, color of the clef

    Returns:
        the lilypond clef representation
    """
    lilyclef = _clefToLilypondClef.get(clef.lower())
    if lilyclef is None:
        raise ValueError(f"Unknown clef {clef}. "
                         f"Possible values: {_clefToLilypondClef.keys()}")
    if "^" in lilyclef or "_" in lilyclef:
        lilyclef = '"' + lilyclef + '"'
    out = r"\clef " + lilyclef
    if color:
        out = rf'\once \override Clef.color = #"{color}" ' + out
    return out


def colorFlag(color: str) -> str:
    return rf'\override Flag.color = "{color}"'


def colorStem(color: str) -> str:
    return rf'\override Stem.color = "{color}"'


def makeNote(pitch: pitch_t,
             duration: float | str,
             dots=0,
             tied=False,
             divsPerSemitone=4,
             noteheadcolor='',
             notehead='',
             parenthesis=False,
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
        notehead: the notehead shape
        noteheadcolor: color of the notehead (as css color)
        parenthesis: should the notehead be within parenthesis?
        cautionary: if True, put the accidental within parenthesis

    Returns:
        The lilypond text
    """
    parts = []
    if notehead or parenthesis or noteheadcolor:
        parts.append(customNotehead(notehead=notehead, color=noteheadcolor,
                                    parenthesis=parenthesis))
        parts.append(' ')
    parts.append(makePitch(pitch, divsPerSemitone=divsPerSemitone,
                           parenthesizeAccidental=cautionary))
    parts.append(makeDuration(duration, dots=dots))
    if tied:
        parts.append("~")
    return "".join(parts)


def getLilypondVersion() -> str | None:
    """
    Return the lilypond version as string

    **NB**: The result is not cached
    """
    lilybin = findLilypond()
    logger = getLogger(__file__)
    if not lilybin:
        logger.error("Could not find lilypond")
        return None
    output = _checkOutput([lilybin, "--version"])
    if output is None:
        raise RuntimeError(f"Could not call lilypond to get the version, "
                           f"lilypond binary: '{lilybin}'")
    match = re.search(r"GNU LilyPond \d+\.\d+\.\d+", output)
    if not match:
        logger.error(f"Could not parse lilypond's output: {output}")
        return None
    return match[0][13:]


def millimetersToPoints(mm: float) -> float:
    return mm * 2.87


def pointsToMillimeters(points: float) -> float:
    return points / 2.87


def paperBlock(paperWidth: float = None,
               margin: float = None,
               leftMargin: float = None,
               rightMargin: float = None,
               lineWidth: float = None,
               topMargin: float = None,
               bottomMargin: float = None,
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


def makeTextMark(text: str,
                 fontsize: int | float = None,
                 fontrelative=True,
                 box='',
                 italic=False,
                 bold=False
                 ) -> str:
    """
    Creates a system text mark - a text above all measures

    Args:
        text: the text
        fontsize: the size of the text.
        fontrelative: font size is relative or absolute?
        box: one of 'square', 'circle', 'rounded' or ('' or 'none') for no box
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
    if box and box != 'none':
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
    'rectangle': r'\box',
    'circle': r'\circle',
    'rounded': r'\rounded-box',
    'rounded-box': r'\rounded-box'
}


def makeText(text: str,
             fontsize: int | float | None = None,
             fontrelative=False,
             placement='above',
             italic=False, 
             bold=False,
             box=''
             ) -> str:
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
        box: one of 'square', 'circle', 'rectangle', 'rounded' or '' for no box

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
