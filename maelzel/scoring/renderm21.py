"""
This module implements a music21 renderer, converts our own
intermediate representation as defined after quantization
into musicxml and renders that musicxml via musescore.
"""

import os
import logging

import music21 as m21
from maelzel.music import m21tools, m21fix

from .common import *
from .core import Notation
from .render import Renderer, RenderOptions
from .scorestruct import DurationGroup
from . import quant
from . import util


logger = logging.getLogger("maelzel.scoring")


_noteheadToMusic21 = {
    'diamond': 'mi'
}


def notationToMusic21(n: Notation, durRatios: List[F], tupleType: Opt[str],
                      options:RenderOptions
                      ) -> m21.note.GeneralNote:
    """
    Converts a Notation to a music21 Rest/Note/Chord/GraceNote

    Args:
        n: the notation
        durRatios: the duration ratios of the context in which the notation
            is. A Notation has already duration ratios,
        tupleType:
        options:

    Returns:

    """
    if n.isGraceNote():
        duration = m21tools.makeDuration(n.getProperty("graceNoteType", "eighth"))

    else:
        notatedDur = n.notatedDuration()
        durType = 4 / notatedDur.base
        duration = m21tools.makeDuration(durType, dots=notatedDur.dots,
                                         durRatios=durRatios, tupleType=tupleType)

    if n.rest:
        return m21.note.Rest(duration=duration)

    pitches = n.pitches
    if n.notehead:
        notehead, fill = util.parseNotehead(n.notehead)
        notehead = _noteheadToMusic21.get(notehead, notehead)
    else:
        notehead, fill = None, None

    if len(pitches) == 1:
        out, centsdev = m21tools.makeNote(pitch=pitches[0], duration=duration,
                                          divsPerSemitone=options.divsPerSemitone,
                                          showcents=False,
                                          notehead=notehead,
                                          noteheadFill=fill,
                                          tiedToPrevious=n.tiedPrev,
                                          hideAccidental=n.accidentalHidden)
    else:
        out, centsdev = m21tools.makeChord(pitches=pitches, duration=duration,
                                           divsPerSemitone=options.divsPerSemitone,
                                           notehead=notehead,
                                           noteheadFill=fill,
                                           showcents=options.showCents,
                                           tiedToPrevious=n.tiedPrev,
                                           hideAccidental=n.accidentalHidden)

    out.noteheadParenthesis = n.noteheadParenthesis
    if n.isGraceNote():
        out = out.getGrace()
        out.duration.slash = n.getProperty("graceNoteSlash", True)
    else:
        m21tools.makeTie(out, tiedPrev=n.tiedPrev, tiedNext=n.tiedNext)
    if options.glissHideTiedNotes and n.noteheadHidden:
        m21tools.hideNotehead(out)
    # annotations need to be added later, when the music21 object has already
    # been added to a stream
    return out


def _m21RenderGroup(measure: m21.stream.Measure,
                    group: DurationGroup,
                    durRatios:List[F],
                    options: RenderOptions) -> None:
    """
    A DurationGroup is a sequence of notes which share (and fill) a time modifier.
    It can be understood as a "tuplet", whereas "normal" durations are interpreted
    as a 1:1 tuplet. A group can consist of Notations or other DurationGroups

    Args:
        measure: the measure being rendered to
        group: the group to render
        durRatios: a seq. of duration ratios OUTSIDE this group. Can be
        an empty list
    """
    if group.durRatio != 1:
        durRatios.append(F(*group.durRatio))
    for i, item in enumerate(group.items):
        if isinstance(item, DurationGroup):
            _m21RenderGroup(measure, item, durRatios, options=options)
            continue
        assert isinstance(item, Notation)
        if group.durRatio == 1:
            tupleType = None
        else:
            if i == 0:
                tupleType = 'start'
            elif i == len(group.items) - 1:
                tupleType = 'stop'
            else:
                tupleType = None
        m21obj = notationToMusic21(item, durRatios, tupleType=tupleType, options=options)

        if not item.rest and item.gliss is not None:
            m21obj.editorial.gliss = item.gliss

        measure.append(m21obj)
        if not item.rest and options.showCents and not item.tiedPrev:
            centsStr = util.centsAnnotation(item.pitches,
                                            divsPerSemitone=options.divsPerSemitone)
            if centsStr:
                m21tools.addTextExpression(m21obj, text=centsStr,
                                           placement=options.centsPlacement,
                                           fontSize=options.centsFontSize)
        if item.annotations and not item.tiedPrev:
            for annotation in item.annotations:
                m21tools.addTextExpression(m21obj, text=annotation.text,
                                           placement=annotation.placement,
                                           fontSize=annotation.fontSize)
        if item.dynamic and not item.tiedPrev:
            m21dyn = m21.dynamics.Dynamic(item.dynamic)
            measure.insert(m21obj.offset, m21dyn)

        if item.articulation and not item.tiedPrev:
            m21tools.addArticulation(m21obj, item.articulation)
    if group.durRatio != 1:
        durRatios.pop()


def _m21ApplyGlissandi(part: m21.stream.Part, options:RenderOptions) -> None:
    """
    Render glissandi in part. Notes with glissando should have been annotated
    via `note.editorial.gliss = True`. The glissando is understood to be
    rendered between the start of a glissando and the next attack. Notes
    tied to the note starting a glissando are part of the original note
    and are skipped. The end note of a glissando should normally be
    contiguous to the note starting a glissando (no rests in between)

    Args:
        part: the part where glissandi should be rendered
        options: the render options

    """
    for m in part.getElementsByClass('Measure'):
        for note in m.getElementsByClass(m21.note.NotRest):
            if not note.editorial.get('gliss', False):
                continue
            if note.tie and note.tie.type != 'start':
                continue
            endnote, isContiguous = m21tools.findNextAttack(note)
            if not isContiguous and not options.glissAllowNonContiguous:
                logger.info(f"Can't render glissando between non"
                            f"contiguous notes: {note}")
                continue
            m21tools.addGliss(note, endnote,
                              hideTiedNotes=options.glissHideTiedNotes)


def quantizedPartToMusic21(part: quant.QuantizedPart,
                           addMeasureMarks=True,
                           clef=None,
                           options:RenderOptions=None) -> m21.stream.Part:
    """
    Convert a QuantizedPart to a music21 Part

    Args:
        part: the QuantizedPart
        addMeasureMarks: if True, this part will include all markings which are global
            to all parts (metronome marks, any measure labels). This should be True
            for the uppermost part and be set to False for the rest.
        clef: if given the part will be forced to start with this clef, otherwise
            the most suitable clef is picked
        options: the RenderOptions used

    Returns:
        the rendered music21 Part

    """
    options = options if options is not None else RenderOptions()
    m21part = m21tools.makePart(clef=clef, partName=part.label)
    quarterTempo = 60
    timesig = None
    for i, measure in enumerate(part.measures):
        measureDef = part.struct.getMeasureDef(i)
        m21measure = m21tools.makeMeasure(measure.timesig,
                                          timesigIsNew=measure.timesig != timesig,
                                          barline=measureDef.barline)
        timesig = measure.timesig
        if addMeasureMarks:
            if measure.quarterTempo != quarterTempo:
                quarterTempo = measure.quarterTempo
                tempoMark = m21tools.makeMetronomeMark(util.asSimplestNumberType(quarterTempo))
                m21measure.insert(0, tempoMark)
            if measureDef.annotation:
                m21measure.insert(0, m21tools.makeTextExpression(
                    text=measureDef.annotation, placement="above",
                    fontSize=options.measureAnnotationFontSize))

        if measure.isEmpty():
            dur = measureDef.numberOfBeats()
            rest = m21.note.Rest(duration=m21.duration.Duration(dur))
            m21measure.append(rest)
            m21part.append(m21measure)
        else:
            for group in measure.groups():
                _m21RenderGroup(m21measure, group, [], options=options)
        m21tools.measureFixAccidentals(m21measure)
        m21part.append(m21measure)

    _m21ApplyGlissandi(m21part, options)
    m21tools.fixNachschlaege(m21part)
    return m21part


def renderScore(parts: List[quant.QuantizedPart], options: RenderOptions=None
                ) -> m21.stream.Score:
    """
    Convert a list of QuantizedParts to a music21 Score

    Args:
        parts: the list of QuantizedParts to convert
        options: RenderOptions used to render the parts

    Returns:
        a music21 Score
    """
    options = options if options is not None else RenderOptions()
    m21parts = [quantizedPartToMusic21(part, addMeasureMarks=i==0, options=options)
                for i, part in enumerate(parts)]
    m21score = m21tools.stackParts(m21parts)
    m21tools.scoreSetMetadata(m21score, title=options.title, composer=options.composer)
    return m21score


class Music21Renderer(Renderer):
    def __init__(self, parts: List[quant.QuantizedPart], options: RenderOptions=None):
        super().__init__(parts, options=options)
        self._m21score: Opt[m21.stream.Score] = None

    def render(self) -> None:
        if self._rendered:
            return
        self._m21score = renderScore(self.parts, options=self.options)
        self._rendered = True

    def writeFormats(self) -> List[str]:
        return ['pdf', 'xml']

    def write(self, outfile: str) -> None:
        m21score = self.nativeScore()
        base, ext = os.path.splitext(outfile)
        if ext == ".xml":
            m21score.write('xml', outfile)
        elif ext == ".pdf":
            xmlfile = base + ".xml"
            m21score.write('musicxml.pdf', xmlfile)
        elif ext == ".png":
            pngfile = base + ".png"
            m21score.write('musicxml.png', pngfile)
        else:
            raise ValueError("Format not supported")

    def show(self) -> None:
        self.render()
        m21fix.show(self._m21score, 'xml.pdf')

    def musicxml(self) -> str:
        return m21tools.getXml(self.asMusic21())

    def asMusic21(self) -> m21.stream.Stream:
        return self.nativeScore()

    def nativeScore(self) -> m21.stream.Score:
        self.render()
        return self._m21score