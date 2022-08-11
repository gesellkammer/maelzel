"""
This module implements a lilypond renderer

It converts our own intermediate representation as defined after quantization
into a .ly file and renders that via lilypond to pdf or png
"""
from __future__ import annotations
import os
import tempfile
import textwrap
import shutil
from dataclasses import dataclass

import emlib.textlib
import emlib.filetools

from maelzel.music import lilytools
from .common import *
from .core import Notation
from .render import Renderer, RenderOptions
from .durationgroup import DurationGroup
from . import quant, util
from . import spanner as _spanner
import logging

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union
    from .common import pitch_t


logger = logging.getLogger("maelzel.scoring")


def _lilyNote(pitch: pitch_t, baseduration:int, dots:int=0, tied=False, cautionary=False
              ) -> str:
    assert baseduration in {0, 1, 2, 4, 8, 16, 32, 64, 128}, \
        f'baseduration should be a power of two, got {baseduration}'
    pitch = lilytools.makePitch(pitch, accidentalParenthesis=cautionary)
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

_noteheadToLilypond = {
    'cross': 'cross',
    'harmonic': 'harmonic',
    'xcircle': 'xcircle',
    'triangle': 'triangle',
    'rhombus': 'harmonic-black',
    'square': 'la',
    'rectangle': 'la',
}


def _lilyArticulation(articulation:str) -> str:
    return _articulations[articulation]


@dataclass
class _Notehead:
    kind: str = ''
    parenthesized: bool = False
    color: str = ''
    sizefactor: Optional[float] = None


def _parseNotehead(notehead: str) -> _Notehead:
    kind, *attrs = notehead.split(";")
    if kind and kind[-1] == '?':
        parenthesized = True
        kind = kind[:-1]
    else:
        parenthesized = False

    color = ''
    sizefactor = None

    for attr in attrs:
        k, v = attr.split("=")
        if k == 'color':
            color = v
        elif k == 'size':
            sizefactor = float(v)

    return _Notehead(kind=kind, parenthesized=parenthesized, color=color, sizefactor=sizefactor)


def _lilyNoteheadInsideChord(notehead: str) -> str:
    if not notehead:
        return ''
    parts = []
    parsedNotehead = _parseNotehead(notehead)
    if parsedNotehead.color:
        parts.append(fr'\tweak NoteHead.color "{parsedNotehead.color}"')
    if parsedNotehead.sizefactor:
        relsize = lilytools.fontSizeFactorToRelativeSize(parsedNotehead.sizefactor)
        parts.append(fr'\tweak NoteHead.font-size #{relsize}')
    if parsedNotehead.kind:
        lilynotehead = _noteheadToLilypond.get(parsedNotehead.kind)
        if not lilynotehead:
            raise ValueError(f'Unknown notehead: {notehead}, '
                             f'possible noteheads: {_noteheadToLilypond.keys()}')
        parts.append(fr"\tweak NoteHead.style #'{lilynotehead}")
    if parsedNotehead.parenthesized:
        parts.append(r'\parenthesize')
    return " ".join(parts)


def _lilyNotehead(notehead: str) -> str:
    """
    Convert a noteshape to its lilypond representation

    This uses \override so it can't be placed inside a chord

    Args:
        notehead: the noteshape. It can end with '?', in which case it will be
            parenthesized

    Returns:
        the lilypond code representing this notehead. This needs to be placed **before**
        the note/chord it will modify.

    """
    assert isinstance(notehead, str), f"Expected a str, got {notehead}"

    if not notehead:
        return ''

    if notehead == 'hidden':
        return r"\once \hide NoteHead"

    parts = []
    parsedNotehead = _parseNotehead(notehead)
    if parsedNotehead.color:
        parts.append(fr'\once \override NoteHead.color = "{parsedNotehead.color}"')
    if parsedNotehead.sizefactor:
        relsize = lilytools.fontSizeFactorToRelativeSize(parsedNotehead.sizefactor)
        parts.append(fr'\once \override NoteHead.font-size = #{relsize}')
    if parsedNotehead.kind:
        lilynotehead = _noteheadToLilypond.get(parsedNotehead.kind)
        if not lilynotehead:
            raise ValueError(f'Unknown notehead: {notehead}, '
                             f'possible noteheads: {_noteheadToLilypond.keys()}')
        parts.append(fr"\once \override NoteHead.style = #'{lilynotehead}")
    if parsedNotehead.parenthesized:
        parts.append(r'\parenthesize')
    return " ".join(parts)


def notationToLily(n: Notation, options: RenderOptions) -> str:
    """
    Converts a Notation to its lilypond representation

    .. note::

        We do not take tuplets into consideration here,
        since they should be taken care of at a higher level
        (see renderGroup)

    Args:
        n: the notation
        options: render options

    Returns:
        the lilypond notation corresponding to n, as a string

    """
    notatedDur = n.notatedDuration()
    base, dots = notatedDur.base, notatedDur.dots
    if n.isRest or len(n.pitches) == 1 and n.pitches[0] == 0:
        rest = "r" + str(base) + "."*dots
        return rest

    parts = []
    _ = parts.append

    if n.color:
        # apply color to notehead, stem and ard accidental
        _(fr'\once \override Beam.color = "{n.color}" '
          fr'\once \override Stem.color = "{n.color}" '
          fr'\once \override Accidental.color = "{n.color}" '
          fr'\once \override Flag.color = "{n.color}"')
        if n.properties and 'noteheadColor' not in n.properties:
            _(fr'\once \override NoteHead.color = "{n.color}"')

    if n.properties:
        if color := n.properties.get('noteheadColor'):
            _(fr'\once \override NoteHead.color = "{color}"')
        if sizeFactor := n.properties.get('noteheadSizeFactor'):
            relsize = lilytools.fontSizeFactorToRelativeSize(sizeFactor)
            _(fr'\once \override NoteHead.font-size =#{relsize}')

    if n.sizeFactor is not None and n.sizeFactor != 1:
        _(rf"\once \magnifyMusic {n.sizeFactor}")

    if n.stem == 'hidden':
        _(r"\once \override Stem.transparent = ##t")

    graceGroup = n.getProperty("graceGroup")
    if graceGroup is not None or n.isGraceNote:
        base = 8
        dots = 0
        _(r"\grace")
        if graceGroup == "start":
            _("{")

    if len(n.pitches) == 1:
        if n.notehead:
            _(_lilyNotehead(n.notehead if isinstance(n.notehead, str) else n.notehead[0]))
        print("creating note", n.notename(), f"tied={n.tiedNext}")
        _(_lilyNote(n.notename(), baseduration=base, dots=dots, tied=n.tiedNext,
                    cautionary=n.getProperty('accidentalParenthesis', False)))
    else:
        if not n.notehead:
            noteheads = None
        elif isinstance(n.notehead, str):
            _(_lilyNotehead(n.notehead))
            noteheads = None   # No individual noteheads
        elif isinstance(n.notehead, list):
            noteheads = n.notehead
            assert len(noteheads) == len(n.pitches), f"noteheads: {noteheads}, pitches: {n.pitches}"
        else:
            raise TypeError(f'Notation.notehead can be a str or list[str], got {n.notehead}')
        _("<")
        for i, pitch in enumerate(n.pitches):
            if noteheads and noteheads[i]:
                _(_lilyNoteheadInsideChord(noteheads[i]))
            _(lilytools.makePitch(n.notename(i),
                                  accidentalParenthesis=n.getProperty('accidentalParenthesis', False)))
        _(f">{base}{'.'*dots}{'~' if n.tiedNext else ''}")

    if (not n.tiedPrev or options.articulationInsideTie) and n.articulation:
        _(_lilyArticulation(n.articulation))

    if (not n.tiedPrev or options.articulationInsideTie) and n.dynamic:
        dyn = n.dynamic if not n.dynamic.endswith('!') else n.dynamic[:-1]
        _(fr"\{dyn}")

    if n.gliss:
        _(r"\glissando")

    if n.annotations:
        for annotation in n.annotations:
            _(lilytools.makeTextAnnotation(annotation.text, fontsize=annotation.fontSize))

    if options.showCents:
        # TODO: cents annotation should follow options (below/above, fontsize)
        centsText = util.centsAnnotation(n.pitches, divsPerSemitone=options.divsPerSemitone)
        if centsText:
            fontrelsize = options.centsFontSize - options.staffSize
            _(lilytools.makeTextAnnotation(centsText, fontsize=fontrelsize,
                                           fontrelative=True, placement='below'))

    if graceGroup == "stop":
        _("}")
    return " ".join(parts)


_spaces = " " * 1000


def _renderGroup(seq: list[str],
                 group: DurationGroup,
                 durRatios:list[F],
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
            _renderGroup(seq, item, durRatios, options=options, numIndents=numIndents+1,
                         state=state)
        else:
            assert isinstance(item, Notation)
            if not item.gliss and state['glissSkip']:
                seq.append(r"\glissandoSkipOff ")
                state['glissSkip'] = False

            if item.isRest:
                state['dynamic'] = ''

            if item.dynamic:
                dynamic = item.dynamic
                if (options.removeSuperfluousDynamics and
                        not item.dynamic.endswith('!') and
                        item.dynamic == state['dynamic']):
                    item.dynamic = ''
                state['dynamic'] = dynamic

            seq.append(notationToLily(item, options=options))

            if item.spanners:
                for spanner in item.spanners:
                    if ((spanner.endingAtTie == 'last' and item.tiedNext) or
                            (spanner.endingAtTie == 'first' and item.tiedPrev)):
                        continue
                    if isinstance(spanner, _spanner.Slur):
                        if spanner.kind == 'start':
                            seq.append(r"\(")
                        elif spanner.kind == 'end':
                            seq.append(r"\)")
                    elif isinstance(spanner, _spanner.Hairpin):
                        if spanner.kind == 'start':
                            seq.append(fr" \{spanner.direction}")
                        elif spanner.kind  == 'end':
                            seq.append(r" \!")

            seq.append(" ")
            if item.gliss and not item.tiedPrev and item.tiedNext:
                seq.append(r"\glissandoSkipOn ")
                assert not state['glissSkip']
                state['glissSkip'] = True
            elif item.gliss and item.tiedPrev and not item.tiedNext:
                seq.append(r"\glissandoSkipOff ")
                # assert state['glissSkip']
                state['glissSkip'] = False

    seq.append("\n")
    if tupletStarted:
        numIndents -= 1
        seq.append(_spaces[:numIndents*indentSize])
        seq.append("}\n")


def quantizedPartToLily(part: quant.QuantizedPart,
                        options: RenderOptions,
                        addMeasureMarks=True,
                        clef:str=None,
                        addTempoMarks=True,
                        indents=0,
                        indentSize=2,
                        numMeasures:int = 0) -> str:
    """
    Convert a QuantizedPart to lilypond

    Args:
        part: the QuantizedPart
        options: the RenderOptions used
        addMeasureMarks: if True, this part will include all markings which are global
            to all parts (metronome marks, any measure labels). This should be True
            for the uppermost part and be set to False for the rest.
        clef: if given the part will be forced to start with this clef, otherwise
            the most suitable clef is picked
        addTempoMarks: if True, add any tempo marks to this Part
        indents: how many indents to use as a base
        indentSize: the number of spaces to indent per indent number
        numMeasures: if given, indicates the number of measures in this score. This will
            be used to render the final barline if this part reaches the end

    Returns:
        the rendered lilypond code

    """
    quarterTempo = 60
    scorestruct = part.struct

    seq = []

    def _(t: str, indents: int = 0, preln=False, postln=False):
        if preln and seq and seq[-1][-1] != "\n":
            seq.append("\n")
        if indents:
            seq.append(_spaces[:indents*indentSize])
        seq.append(t)
        if postln:
            seq.append("\n")

    def line(t: str, indents: int = 0):
        _(t, indents, preln=True, postln=True)

    if part.label:
        line(r"\new Staff \with {", indents)
        line(f'    instrumentName = #"{part.label}"', indents)
        line("}", indents)
        line("{", indents)
    else:
        line(r"\new Staff {", indents)

    indents += 1
    line(r"\numericTimeSignature", indents)

    if clef is not None:
        line(fr'\clef "{clef}"', indents)
    else:
        clef = quant.bestClefForPart(part)
        line(lilytools.makeClef(clef), indents)

    lastTimesig = None

    state = {
        'glissSkip': False,
        'dynamic': ''
    }

    for i, measure in enumerate(part.measures):
        line(f"% measure {i}", indents)
        indents += 1
        measureDef = scorestruct.getMeasureDef(i)

        if addTempoMarks and measureDef.timesig != lastTimesig:
            lastTimesig = measureDef.timesig
            num, den = measureDef.timesig
            line(fr"\time {num}/{den}", indents)

        if addTempoMarks and measure.quarterTempo != quarterTempo:
            quarterTempo = measure.quarterTempo
            line(fr"\tempo 4 = {quarterTempo}", indents)

        if measureDef.annotation and addMeasureMarks:
            relfontsize = options.measureAnnotationFontSize - options.staffSize
            _(lilytools.makeTextMark(measureDef.annotation,
                                     fontsize=relfontsize, fontrelative=True,
                                     boxed=options.measureAnnotationBoxed))
        if measure.isEmpty():
            num, den = measure.timesig
            measureDur = float(util.measureQuarterDuration(measure.timesig))
            if measureDur in {1., 2., 3., 4., 6., 7., 8.}:
                lilydur = lilytools.makeDuration(measureDur)
                _(f"R{lilydur}")
            else:
                _(f"R1*{num}/{den}")
            state['dynamic'] = ''
        else:
            for group in measure.groups():
                _renderGroup(seq, group, durRatios=[], options=options,
                             numIndents=indents, state=state)
        indents -= 1

        if not measureDef.barline or measureDef.barline == 'single':
            _(f"|   % end measure {i}", indents)
        elif measureDef.barline == 'final' or numMeasures and i == numMeasures - 1:
            _(r'\bar "|."    % final bar', indents)
        elif measureDef.barline == 'double':
            _(fr'\bar "||"   % end measure {i}', indents)
        elif measureDef.barline == 'solid':
            _(fr'\bar "."    % end measure {i}', indents)
        elif measureDef.barline == 'dashed':
            _(fr'\bar "!"    % end measure {i}', indents)
        else:
            raise ValueError(f"Barline type {measureDef.barline} not known")

    indents -= 1

    line("}", indents)
    return "".join(seq)


# --------------------------------------------------------------

_prelude = r"""

glissandoSkipOn = {
  \override NoteColumn.glissando-skip = ##t
  % \hide NoteHead
  \override NoteHead.no-ledgers = ##t
}

glissandoSkipOff = {
  \revert NoteColumn.glissando-skip
  % \undo \hide NoteHead
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
    
    % overrides to allow glissandi across systembreaks
    \override Glissando.breakable = ##t
    \override Glissando.after-line-breaking = ##t
    
    % <score-overrides>
    
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
    \override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/32)
  }
}  
"""

_horizontalSpacingXL = r"""
\layout {
  \context {
    \Score
    \override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/64)
  }
}
"""


def makeScore(score: quant.QuantizedScore,
              options: RenderOptions,
              midi=False
              ) -> str:
    """
    Convert a list of QuantizedParts to a lilypond score (as str)

    Args:
        score: the list of QuantizedParts to convert
        options: RenderOptions used to render the parts
        midi: if True, include a ``\midi`` block so that lilypond
            generates a midi file together with the rendered image file

    Returns:
        the generated score as str
    """
    indentSize = 2
    numMeasures = max(len(part.measures)
                      for part in score.parts)
    struct = score.scorestruct.copy()
    struct.setBarline(numMeasures - 1, 'final')
    score.scorestruct = struct

    strs = []
    _ = strs.append
    lilypondVersion = lilytools.getLilypondVersion()
    _(f'\\version "{lilypondVersion}"\n')

    if options.title or options.composer:
        _(textwrap.dedent(fr'''
        \header {{
            title = "{options.title}"
            composer = "{options.composer}"
            tagline = ##f
        }}
        '''))
    else:
        _(r"\header { tagline = ##f }")

    # Global settings
    # staffSizePoints = lilytools.millimetersToPoints(options.staffSize)
    staffSizePoints = options.staffSize
    if options.renderFormat == 'png':
        staffSizePoints *= options.lilypondPngStaffsizeScale
    if not midi:
        assert options.renderFormat in ('png', 'pdf'), f"Render format unknown: '{options.renderFormat}'"
    _(f'#(set-global-staff-size {staffSizePoints})')

    if options.cropToContent:
        _(r'\include "lilypond-book-preamble.ly"')
        # TODO: add configuration for this
        _(lilytools.paperBlock(margin=20, unit="mm"))
    else:
        # We only set the paper size if rendering to pdf
        _(f"#(set-default-paper-size \"{options.pageSize}\" '{options.orientation})")
        _(lilytools.paperBlock(margin=options.pageMarginMillimeters, unit="mm"))

    _(_prelude)

    if options.glissandoLineThickness != 1:
        _(r"""
        \layout {
          \context { 
            \Voice
            \override Glissando #'thickness = #%d
            \override Glissando #'gap = #0.05
          }
        }
        """ % options.glissandoLineThickness)

    if options.horizontalSpacing == 'medium':
        _(_horizontalSpacingMedium)
    elif options.horizontalSpacing == 'large':
        _(_horizontalSpacingLarge)
    elif options.horizontalSpacing == 'xlarge':
        _(_horizontalSpacingXL)

    _(r"\score {")
    _(r"<<")
    for i, part in enumerate(score):
        partstr = quantizedPartToLily(part,
                                      addMeasureMarks=i==0,
                                      addTempoMarks=i==0,
                                      options=options,
                                      indents=1,
                                      indentSize=indentSize,
                                      numMeasures=numMeasures)
        _(partstr)
    _(r">>")
    if midi:
        _(" "*indentSize + "\midi { }")
    _(r"}")  # end \score
    return emlib.textlib.joinPreservingIndentation(strs)


class LilypondRenderer(Renderer):
    def __init__(self, score: quant.QuantizedScore, options: RenderOptions):
        super().__init__(score, options=options)
        self._lilyscore = ''
        self._withMidi = False

    def render(self) -> None:
        if self._rendered:
            return
        self._lilyscore = makeScore(self.score, options=self.options, midi=self._withMidi)
        self._rendered = True

    def writeFormats(self) -> list[str]:
        return ['pdf', 'ly', 'png']

    def write(self, outfile: str, fmt: str=None, removeTemporaryFiles=False) -> None:
        outfile = emlib.filetools.normalizePath(outfile)
        base, ext = os.path.splitext(outfile)
        if fmt is None:
            fmt = ext[1:]
        if fmt == 'png':
            if self.options.renderFormat != 'png':
                self.reset()
            self.options.cropToContent = True
            self.options.renderFormat = 'png'
        elif fmt == 'pdf':
            if self.options.renderFormat != 'pdf':
                self.reset()
            self.options.cropToContent = False
            self.options.renderFormat = 'pdf'
        elif fmt == 'mid':
            if not self._withMidi:
                self._rendered = False
                self._withMidi = True
        elif fmt == 'ly':
            pass
        else:
            raise ValueError(f"Format {fmt} unknown. Possible formats: png, pdf, mid")
        self.render()
        lilytxt = self._lilyscore
        tempfiles = []
        if fmt == 'png' or fmt == 'pdf':
            lilyfile = tempfile.mktemp(suffix=".ly")
            open(lilyfile, "w").write(lilytxt)
            if fmt != ext[1:]:
                outfile2 = f"{outfile}.{fmt}"
                lilytools.renderLily(lilyfile, outfile2,
                                     imageResolution=self.options.pngResolution)
                shutil.move(outfile2, outfile)
            else:
                lilytools.renderLily(lilyfile, outfile)
            tempfiles.append(lilyfile)
        elif fmt == 'mid' or fmt == 'midi':
            lilyfile = tempfile.mktemp(suffix='.ly')
            open(lilyfile, "w").write(lilytxt)
            lilytools.renderLily(lilyfile)
            midifile = emlib.filetools.withExtension(lilyfile, "midi")
            assert os.path.exists(midifile), f"Failed to generate MIDI file. Expected path: {midifile}"
            tempfiles.append(lilyfile)
            shutil.move(midifile, outfile)
        elif fmt == 'ly':
            open(outfile, "w").write(lilytxt)
        else:
            raise ValueError(f"Format should be pdf, png or ly, got {fmt}")
        if removeTemporaryFiles:
            for f in tempfiles:
                os.remove(f)

    def asMusic21(self) -> None:
        """
        This renderer does not implement conversion to music21
        """
        return None

    def nativeScore(self) -> str:
        """
        Returns the lilypond score (as text)

        Returns:
            the string representing the rendered lilypond score
        """
        self.render()
        return self._lilyscore