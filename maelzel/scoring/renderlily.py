"""
This module implements a lilypond renderer, converts our own
intermediate representation as defined after quantization
into a .ly file and renders that via lilypond to pdf or png
"""

import os
import logging
import tempfile

from maelzel.music import lilytools
from .common import *
from .core import Notation
from .render import Renderer, RenderOptions
from .quant import DurationGroup
from . import quant


logger = logging.getLogger("maelzel.scoring")


def _lilyNote(pitch: pitch_t, baseduration:int, dots:int=0, tied=False) -> str:
    assert baseduration in {0, 1, 2, 4, 8, 16, 32, 64, 128}, \
        f'baseduration should be a power of two, got {baseduration}'
    pitch = lilytools.makePitch(pitch)
    if baseduration == 0:
        # a grace note
        return fr"\grace {pitch}8"
    return fr"{pitch}{baseduration}{'.'*dots}{'~' if tied else ''}"


_articulations = {
    'staccato': r'\staccato',
    'accent': r'\accent',
    'marcato': r'\marcato',
    'tenuto': r'\tenuto',
    'staccatissimo': r'\staccatissimo',
}


_noteheads = {
    'cross': r'\xNote',
    'harmonic': r"\once \override NoteHead.style = #'harmonic",
    'triangleup': r'\palmMute',
    'xcircle': r"\once \override NoteHead.style = #'xcircle",
    'triangle': r"\once \override NoteHead.style = #'triangle",
    'rhombus': r"\once \override NoteHead.style = #'harmonic-black",
    'square': r"\once \override NoteHead.style = #'la",
    'rectangle': r"\once \override NoteHead.style = #'la"
}


def _lilyArticulation(articulation:str) -> str:
    return _articulations[articulation]


def _lilyNotehead(notehead:str, parenthesis:bool=False) -> str:
    # TODO: take account of parenthesis
    return _noteheads[notehead]


def notationToLily(n: Notation) -> str:
    """
    Converts a Notation to its lilypond representation
    NB: we do not take tuplets into consideration here,
    since they should be taken care of at a higher level
    (see renderGroup)

    Args:
        n: the notation

    Returns:
        the lilypond notation corresponding to n, as a string

    """
    notatedDur = n.notatedDuration()
    base, dots = notatedDur.base, notatedDur.dots
    event = ""
    if n.isRest:
        rest = "r" + str(base) + "."*dots
        return rest

    # notehead modifiers precede the note
    if n.noteheadHidden:
        event += r" \once \hide NoteHead "
    elif n.notehead:
        event += _lilyNotehead(n.notehead, n.noteheadParenthesis)
        event += " "

    if n.stem == 'hidden':
        event += r" \once \hide Stem "

    if len(n.pitches) == 1:
        pitch = n.pitches[0]
        notename = n.getFixedNotenameByIndex(0, m2n(pitch))
        event += _lilyNote(notename, baseduration=base, dots=dots, tied=n.tiedNext)
    else:
        lilypitches = []
        for i, pitch in enumerate(n.pitches):
            notename = n.getFixedNotenameByIndex(i, m2n(pitch))
            lilypitches.append(lilytools.makePitch(notename))
        event += f"<{' '.join(lilypitches)}>{base}{'.'*dots}"
        if n.tiedNext:
            event += "~"
    if not n.tiedPrev:
        if n.articulation:
            event += _lilyArticulation(n.articulation)
        if n.dynamic:
            event += fr" \{n.dynamic}"
        if n.gliss:
            event += r" \glissando"
        if n.annotations:
            for annotation in n.annotations:
                fontSize = fr"\abs-fontsize #{int(annotation.fontSize)}"
                if annotation.placement == 'above':
                    event += fr"^\markup {{ {fontSize} {{ {annotation.text} }} }} "
                else:
                    event += fr"_\markup {{ {fontSize} {{ {annotation.text} }} }} "
    return event


_spaces = " " * 1000


def _renderGroup(seq: List[str],
                 group: DurationGroup,
                 durRatios:List[F],
                 options: RenderOptions,
                 state: dict,
                 numIndents: int = 0,
                 ) -> None:
    """
    A DurationGroup is a sequence of notes which share (and fill) a time modifier.
    It can be understood as a "tuplet", whereas "normal" durations are interpreted
    as a 1:1 tuplet. A group can consist of Notations or other DurationGroups

    Args:
        group: the group to render
        durRatios: a seq. of duration ratios OUTSIDE this group. Can be
        an empty list
    """
    indentSize = 2
    # \tuplet 3/2 { b4 b b }
    if group.durRatio != (1, 1):
        durRatios.append(F(*group.durRatio))
        tupletStarted = True
        num, den = group.durRatio
        seq.append(_spaces[:numIndents*indentSize])
        seq.append(f"\\tuplet {num}/{den} {{\n")
        numIndents += 1
    else:
        tupletStarted = False
    seq.append(_spaces[:numIndents*indentSize])
    for i, item in enumerate(group.items):
        if isinstance(item, DurationGroup):
            _renderGroup(seq, item, durRatios, options=options, numIndents=numIndents+1)
        else:
            if not item.gliss and state['glissSkip']:
                seq.append(r"\glissandoSkipOff ")
                state['glissSkip'] = False
            lilyItem = notationToLily(item)
            seq.append(lilyItem)
            seq.append(" ")
            if item.gliss and not item.tiedPrev and item.tiedNext:
                seq.append(r"\glissandoSkipOn ")
                assert not state['glissSkip']
                state['glissSkip'] = True
            elif item.gliss and item.tiedPrev and not item.tiedNext:
                seq.append(r"\glissandoSkipOff ")
                assert state['glissSkip']
                state['glissSkip'] = False

    seq.append("\n")
    if tupletStarted:
        numIndents -= 1
        seq.append(_spaces[:numIndents*indentSize])
        seq.append("}\n")


def quantizedPartToLily(part: quant.QuantizedPart,
                        addMeasureMarks=True,
                        clef:str=None,
                        addTempoMarks=True,
                        options:RenderOptions=None,
                        indents=0,
                        indentSize=2) -> str:
    """
    Convert a QuantizedPart to lilypond

    Args:
        part: the QuantizedPart
        addMeasureMarks: if True, this part will include all markings which are global
            to all parts (metronome marks, any measure labels). This should be True
            for the uppermost part and be set to False for the rest.
        clef: if given the part will be forced to start with this clef, otherwise
            the most suitable clef is picked
        options: the RenderOptions used
        indents: how many indents to use as a base
        indentSize: the number of spaces to indent per indent number

    Returns:
        the rendered lilypond code

    """
    options = options if options is not None else RenderOptions()
    quarterTempo = 60

    seq = []

    def _(t:str, indents:int=0, preln=False, postln=False):
        if preln and seq and seq[-1][-1] != "\n":
            seq.append("\n")
        if indents:
            seq.append(_spaces[:indents*indentSize])
        seq.append(t)
        if postln:
            seq.append("\n")

    def ownline(t: str, indents:int=0):
        _(t, indents, preln=True, postln=True)

    # ownline("<<", indents)
    # indents += 1

    ownline(r"\new Staff \with {", indents)
    if part.label:
        ownline(f'    instrumentName = #"{part.label}"', indents)
    ownline("}", indents)
    ownline("{", indents)
    indents += 1

    if clef is not None:
        ownline(fr"\clef {clef}", indents)
    else:
        clef = quant.bestClefForPart(part)
        ownline(lilytools.makeClef(clef), indents)

    lastTimesig = None
    state = {
        'glissSkip': False
    }
    for i, measure in enumerate(part.measures):
        ownline(f"% measure {i}", indents)
        indents += 1
        measureDef = part.struct.getMeasureDef(i)
        if addTempoMarks and measureDef.timesig != lastTimesig:
            lastTimesig = measureDef.timesig
            num, den = measureDef.timesig
            ownline(r"\numericTimeSignature", indents)
            ownline(fr"\time {num}/{den}", indents)
        # TODO: add barlinetype
        if addTempoMarks and measure.quarterTempo != quarterTempo:
            quarterTempo = measure.quarterTempo
            ownline(fr"\tempo 4 = {quarterTempo}")
        if measureDef.annotation and addMeasureMarks:
            # TODO: add measure annotation
            pass
        if measure.isEmpty():
            _(f"R1*{num}/{den}")
        else:
            for group in measure.groups():
                _renderGroup(seq, group, durRatios=[], options=options,
                             numIndents=indents, state=state)
        indents -= 1
        _(f"|   % end measure {i}", indents)

    indents -= 1
    ownline("}", indents)
    # indents -= 1
    # ownline(">>", indents)
    return "".join(seq)


_prelude = r"""

glissandoSkipOn = {
  \override NoteColumn.glissando-skip = ##t
  \hide NoteHead
  \override NoteHead.no-ledgers = ##t
}

glissandoSkipOff = {
  \revert NoteColumn.glissando-skip
  \undo \hide NoteHead
  \revert NoteHead.no-ledgers
}

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

_horizontalSpacingMedium = r"""
  \layout {
    \context {
      \Score
      \override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/16)
    }
  }
"""
_horizontalSpacingLarge = r"""
  \layout {
    \context {
      \Score
      \override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/8)
    }
  }
"""


def makeScore(parts: List[quant.QuantizedPart],
              options: RenderOptions=None,
              lilypondVersion:U[str, bool]=True
              ) -> str:
    """
    Convert a list of QuantizedParts to a lilypond score (as str)

    Args:
        parts: the list of QuantizedParts to convert
        options: RenderOptions used to render the parts
        lilypondVersion: if given, use it to tag the rendered file

    Returns:
        the generated score as str
    """
    indentSize = 2
    strs = []
    if lilypondVersion:
        if isinstance(lilypondVersion, bool):
            lilypondVersion = lilytools.getLilypondVersion()
        strs.append(f'\\version "{lilypondVersion}"\n')

    # anchor for book preamble insert
    strs.append(r'% <book>')

    if options.staffSize is not None:
        staffSizePoints = lilytools.millimetersToPoints(options.staffSize)
        strs.append(f'#(set-global-staff-size {staffSizePoints})')

    strs.append(r'% <paper>')

    strs.append(_prelude)

    strs.append(r"\score {")
    strs.append(r"<<")
    for i, part in enumerate(parts):
        partstr = quantizedPartToLily(part,
                                      addMeasureMarks=i==0,
                                      addTempoMarks=i==0,
                                      options=options,
                                      indents=1,
                                      indentSize=indentSize)
        strs.append(partstr)
    strs.append(r">>")

    #if options.lilypondHorizontalSpacing == 'medium':
    #    strs.append(_horizontalSpacingMedium)
    #elif options.lilypondHorizontalSpacing == 'large':
    #    strs.append(_horizontalSpacingLarge)

    strs.append(r"}")
    out = "\n".join(strs)
    return out


class LilypondRenderer(Renderer):
    def __init__(self, parts: List[quant.QuantizedPart], options: RenderOptions=None):
        super().__init__(parts, options=options)
        self._lilyscore: Opt[str] = None

    def render(self) -> None:
        if self._rendered:
            return
        self._lilyscore = makeScore(self.parts, options=self.options)
        self._rendered = True

    def writeFormats(self) -> List[str]:
        return ['pdf', 'ly', 'png']

    def write(self, outfile: str, removeTemporaryFiles=False) -> None:
        lilytxt = self.nativeScore()
        base, ext = os.path.splitext(outfile)
        tempfiles = []
        if ext == '.ly':
            open(outfile, "w").write(lilytxt)
        elif ext == '.png':
            # Adding he book-preamble crops the png output to the actual contents
            if self.options.lilypondPngBookPreamble:
                if not 'lilypond-book-preamble.ly' in lilytxt and '% <book>' in lilytxt:
                    lilytxt = lilytxt.replace('% <book>',
                                              r'\include "lilypond-book-preamble.ly"')
            lilytxt = lilytxt.replace("% <paper>", lilytools.paperBlock(margin=20))
            lilyfile = tempfile.mktemp(suffix=".ly")
            tempfiles.append(lilyfile)
            open(lilyfile, "w").write(lilytxt)
            print(lilyfile)
            lilytools.renderLily(lilyfile, outfile)
        elif ext == '.pdf':
            lilyfile = tempfile.mktemp(suffix=".ly")
            open(lilyfile, "w").write(lilytxt)
            lilytools.renderLily(lilyfile, outfile)
            tempfiles.append(lilyfile)
        else:
            raise ValueError(f"Extension should be .pdf, .png or .ly, got {ext}")
        if removeTemporaryFiles:
            for f in tempfiles:
                os.remove(f)

    def asMusic21(self) -> None:
        return None

    def nativeScore(self) -> str:
        self.render()
        return self._lilyscore