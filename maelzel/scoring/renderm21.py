"""
This module implements a music21 renderer, converts our own
intermediate representation as defined after quantization
into musicxml and renders that musicxml via musescore.
"""
from __future__ import annotations
import os
import tempfile
from emlib.iterlib import pairwise
import emlib.img

import music21 as m21
from maelzel.music import m21tools, m21fix
from maelzel import musicxml as mxml

from .common import *
from .core import Notation
from .render import Renderer, RenderOptions
from . import quant
from . import util
from . import definitions
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


# See https://www.w3.org/2021/06/musicxml40/musicxml-reference/data-types/notehead-value/
_noteheadToMusic21 = {
    'harmonic': 'mi',
    'cross': 'x',
    'triangleup': 'triangle',
    'xcircle': 'circle-x',
    'triangle': 'do',
    'rhombus': 'mi',    # 'diamond',
    'square': 'la', # 'la',
    'rectangle': 'rectangle'
}

assert all(shape in definitions.noteheadShapes for shape in _noteheadToMusic21.keys())


def noteToMusic21(n: Notation, divsPerSemitone=4, durRatios=None, tupleType=None
                  ) -> Tuple[m21.note.Note, int]:
    assert len(n.pitches) == 1

    if n.isGraceNote():
        duration = m21tools.makeDuration(n.getProperty("graceNoteType", "eighth"))
    else:
        notatedDur = n.notatedDuration()
        durType = 4 / notatedDur.base
        duration = m21tools.makeDuration(durType, dots=notatedDur.dots,
                                         durRatios=durRatios, tupleType=tupleType)

    pitch = n.pitches[0]
    if n.notehead:
        notehead, fill = util.parseNotehead(n.notehead)
        m21notehead = _noteheadToMusic21.get(notehead)
        if m21notehead is None:
            logger.error(f"Unknown notehead {notehead}, using default")
            m21notehead = None
    else:
        m21notehead, fill = None, None

    out, centsdev = m21tools.makeNote(pitch=pitch, duration=duration,
                                      divsPerSemitone=divsPerSemitone,
                                      showcents=False,
                                      notehead=m21notehead,
                                      noteheadFill=fill,
                                      tiedToPrevious=n.tiedPrev,
                                      hideAccidental=n.accidentalHidden)
    return out, centsdev


def notationToMusic21(n: Notation, durRatios: List[F], tupleType: Opt[str],
                      options:RenderOptions
                      ) -> m21.note.GeneralNote:
    """
    Converts a Notation to a music21 Rest/Note/Chord/GraceNote

    Args:
        n: the notation
        durRatios: the duration ratios of the context in which the notation
            is. A Notation has already duration ratios,
        tupleType: either "start", "stop", or None
        options: the render options

    Returns:
        a m21 GeneralNote (a Note, a Chord, a GraceNote or a Rest)

    """
    if n.isGraceNote():
        duration = m21tools.makeDuration(n.getProperty("graceNoteType", "eighth"))

    else:
        notatedDur = n.notatedDuration()
        durType = 4 / notatedDur.base
        duration = m21tools.makeDuration(durType, dots=notatedDur.dots,
                                         durRatios=durRatios, tupleType=tupleType)

    if n.isRest:
        return m21.note.Rest(duration=duration)

    pitches = n.pitches
    if n.notehead:
        notehead, fill = util.parseNotehead(n.notehead)
        m21notehead = _noteheadToMusic21.get(notehead)
        if m21notehead is None:
            logger.error(f"notationToMusic21: Unknown notehead {notehead}, using default")
            m21notehead = None
    else:
        m21notehead, fill = None, None

    if len(pitches) == 1:
        out, centsdev = m21tools.makeNote(pitch=pitches[0], duration=duration,
                                          divsPerSemitone=options.divsPerSemitone,
                                          showcents=False,
                                          notehead=m21notehead,
                                          noteheadFill=fill,
                                          tiedToPrevious=n.tiedPrev,
                                          hideAccidental=n.accidentalHidden)
    else:
        out, centsdev = m21tools.makeChord(pitches=pitches, duration=duration,
                                           divsPerSemitone=options.divsPerSemitone,
                                           notehead=m21notehead,
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

    if n.stem == 'hidden':
        m21tools.hideStem(out)

    if n.color:
        out.style.color = n.color

    # annotations need to be added later, when the music21 object has already
    # been added to a stream
    return out


def _m21RenderGroup(measure: m21.stream.Measure,
                    group: quant.DurationGroup,
                    durRatios:List[F],
                    options: RenderOptions) -> None:
    """
    A quant.DurationGroup is a sequence of notes which share (and fill) a time modifier.
    It can be understood as a "tuplet", whereas "normal" durations are interpreted
    as a 1:1 tuplet. A group can consist of Notations or other quant.DurationGroups

    Args:
        measure: the measure being rendered to
        group: the group to render
        durRatios: a seq. of duration ratios OUTSIDE this group. Can be
        an empty list
    """
    if group.durRatio != 1:
        durRatios.append(F(*group.durRatio))
    centsLabelOrder = 'ascending' if options.centsPlacement == 'below' else 'descending'

    for i, item in enumerate(group.items):
        if isinstance(item, quant.DurationGroup):
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

        if not item.isRest and item.gliss is not None:
            m21obj.editorial.makeGliss = item.gliss

        measure.append(m21obj)
        if not item.isRest and options.showCents and not item.tiedPrev:
            centsStr = util.centsAnnotation(item.pitches,
                                            divsPerSemitone=options.divsPerSemitone,
                                            order=centsLabelOrder)
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
            m21tools.addGliss(note, endnote, linetype='solid', continuous=True,
                              hideTiedNotes=options.glissHideTiedNotes)


def _fixGracenoteAtBeginning(part: quant.QuantizedPart) -> None:
    if len(part.measures) < 2:
        return
    for m0, m1 in pairwise(part.measures):
        if not m1.beats:
            continue
        m1b0 = m1.beats[0]
        if len(m1b0.notations) <= 1:
            continue
        if m1b0.notations[0].duration == 0 and m1b0.notations[1].isRest:
            m0last = m0.beats[-1]
            m0last.notations.append(m1b0.notations[0])
            m1b0.notations = m1b0.notations[1:]


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
    if clef is None:
        midinotesInPart = [n.notation.meanPitch() for n in part.iterNotations()
                           if not n.notation.isRest]
        clef = util.clefNameFromMidinotes(midinotesInPart)
    m21part = m21tools.makePart(clef=clef, partName=part.label)
    quarterTempo = 60
    timesig = None
    _fixGracenoteAtBeginning(part)
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
                m21measure.append(tempoMark)
                # m21measure.insert(0, tempoMark)
            if measureDef.annotation:
                m21measure.insert(0, m21tools.makeTextExpression(
                    text=measureDef.annotation, placement="above",
                    fontSize=options.measureAnnotationFontSize))

        if measure.isEmpty():
            dur = measureDef.numberOfBeats()
            rest = m21.note.Rest(duration=m21.duration.Duration(dur))
            m21measure.append(rest)
        else:
            measure.removeUnnecessaryAccidentals()
            for group in measure.groups():
                _m21RenderGroup(m21measure, group, [], options=options)
            m21tools.measureFixAccidentals(m21measure)
        m21part.append(m21measure)

    m21tools.fixNachschlaege(m21part)
    _m21ApplyGlissandi(m21part, options)
    return m21part


def dinSizeToMM(dinsize: str, orientation='portrait') -> Tuple[int, int]:
    dinsize = dinsize.lower()
    if dinsize == "a3":
        x, y = 297, 420
    elif dinsize == "a4":
        x, y = 210, 297
    else:
        raise ValueError(f"dinsize should be one of 'a3', 'a4', got {dinsize}")
    if orientation == "portrait":
        return x, y
    else:
        return y, x


def renderScore(score: quant.QuantizedScore, options: RenderOptions=None
                ) -> m21.stream.Score:
    """
    Convert a list of QuantizedParts to a music21 Score

    Args:
        score: the list of QuantizedParts to convert
        options: RenderOptions used to render the parts

    Returns:
        a music21 Score
    """
    if options is None:
        raise ValueError("options should not be None")
    options = options if options is not None else RenderOptions()
    cnv = mxml.LayoutUnitConverter.fromStaffsize(options.staffSize)
    heightmm, widthmm = dinSizeToMM(options.pageSize, orientation=options.orientation)
    scalingmm = mxml.pointsToMillimeters(options.staffSize)
    scoreLayout = m21.layout.ScoreLayout(scalingMillimeters=scalingmm,
                                         scalingTenths=mxml.MUSICXML_TENTHS,
                                         pageHeight=cnv.toTenths(heightmm),
                                         pageWidth=cnv.toTenths(widthmm))
    m21parts = []
    for i, part in enumerate(score):
        m21part = quantizedPartToMusic21(part, addMeasureMarks=i==0, options=options)
        m21parts.append(m21part)
    m21score = m21tools.stackParts(m21parts)
    m21score.insert(-1, scoreLayout)
    m21tools.scoreSetMetadata(m21score,
                              title=score.title or '',
                              composer=score.composer or '')
    m21fix.fixStream(m21score, inPlace=True)
    return m21score


class Music21Renderer(Renderer):
    def __init__(self, score: quant.QuantizedScore, options: RenderOptions=None):
        super().__init__(score, options=options)
        self._m21score: Opt[m21.stream.Score] = None

    def render(self, fmt:str = '') -> None:
        if self._rendered:
            return
        self._m21score = renderScore(self.score, options=self.options)
        self._rendered = True

    def writeFormats(self) -> List[str]:
        return ['pdf', 'xml', 'png']

    def write(self, outfile: str) -> None:
        base, ext = os.path.splitext(outfile)
        fmt = ext[1:]
        if fmt == 'musicxml':
            fmt = 'xml'
        self.render(fmt)
        m21score = self._m21score
        if fmt == "xml":
            m21score.write('xml', outfile)
        elif fmt == "pdf":
            m21score.write('musicxml.pdf', outfile)
        elif fmt == "png":
            xmlfile = tempfile.mktemp(suffix=".xml")
            m21score.write('xml', xmlfile)
            m21tools.renderMusicxml(xmlfile, outfile)
            if ext == '.png' and self.options.opaque:
                emlib.img.pngRemoveTransparency(outfile)
            os.remove(xmlfile)
        else:
            raise ValueError("Format not supported")
        if not os.path.exists(outfile):
            logger.error(f"failed to write {outfile}")
            xmlscore = m21tools.getXml(m21score)
            logger.debug("musicxml score:")
            logger.debug(xmlscore)
            raise RuntimeError(f"failed to write {outfile}")

    def musicxml(self) -> str:
        return m21tools.getXml(self.asMusic21())

    def asMusic21(self) -> m21.stream.Score:
        return self.nativeScore()

    def nativeScore(self) -> m21.stream.Score:
        self.render()
        return self._m21score