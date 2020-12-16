from __future__ import annotations
from .scorestruct import *
from . import core
from . import quant
from dataclasses import dataclass

from emlib.music import m21tools
from emlib.music import m21fix
from typing import List
import music21 as m21
import os
import logging


logger = logging.getLogger("emlib.scoring")


def _simplestNumber(f: F) -> U[int, float]:
    fl = float(f)
    i = int(fl)
    return i if i == fl else fl


@dataclass
class RenderOptions:
    """
    orientation: one of "portrait" or "landscape"
    staffSize: the size of each staff in point
    pageSize: one of "a4", "a3"

    divsPerSemitone: the number of divisions of the semitone
    showCents: should each note/chord have a text label attached
    indicating the cents deviation from the nearest semitone?
    centsPlacement: where to put the cents annotation
    centsFontSize: the font size of the cents annotation

    measureAnnotationFontSize: font size for measure annotations

    glissAllowNonContiguous: if True, allow glissandi between notes which
        have rests between them
    glissHideTiedNotes: if True, hide tied notes which are part of a gliss.

    title: the title of the score
    composer: the composer of the score
    """
    orientation: str = "portrait"
    staffSize: int = 12
    pageSize: str = 'a4'

    divsPerSemitone: int = 4
    showCents: bool = False
    centsPlacement: str = "above"
    centsFontSize: int = 10

    measureAnnotationFontSize: int = 12

    glissAllowNonContiguous: bool = False
    glissHideTiedNotes: bool = False

    title: str = ''
    composer: str = ''


class Renderer:
    def __init__(self, parts: List[quant.QuantizedPart], options:RenderOptions=None):
        assert parts
        assert parts[0].struct is not None
        self.parts = parts
        self.struct = parts[0].struct
        if options is None:
            options = RenderOptions()
        self.options = options
        self._rendered = False

    def render(self) -> None:
        """
        This method should be implemented by the backend
        """
        raise NotImplementedError("Please Implement this method")

    def writeFormats(self) -> List[str]:
        """
        Returns: a list of possible write formats (pdf, xml, musicxml, etc)
        """
        raise NotImplementedError("Please Implement this method")

    def write(self, outfile:str) -> None:
        raise NotImplementedError("Please Implement this method")

    def show(self) -> None:
        raise NotImplementedError("Please Implement this method")

    def musicxml(self) -> Opt[str]:
        m21stream = self.asMusic21()
        return m21tools.getXml(m21stream) if m21stream else None

    def asMusic21(self) -> Opt[m21.stream.Stream]:
        """
        If the renderer can return a music21 stream version of the render,
        return it here, otherwise return None
        """
        return None



_noteheadToMusic21 = {
    'diamond': 'mi'
}

def _notationToMusic21(n: Notation, durRatios: List[F], tupleType: Opt[str],
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
        m21obj = _notationToMusic21(item, durRatios, tupleType=tupleType, options=options)

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
                tempoMark = m21tools.makeMetronomeMark(_simplestNumber(quarterTempo))
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


def partsToMusic21(parts: List[quant.QuantizedPart], options: RenderOptions=None
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
        self._m21score = partsToMusic21(self.parts, options=self.options)
        self._rendered = True

    def writeFormats(self) -> List[str]:
        return ['pdf', 'xml']

    def write(self, outfile) -> None:
        m21score = self.nativeScore()
        base, ext = os.path.splitext(outfile)
        if ext == ".xml":
            m21score.write('xml', outfile)
        elif ext == ".pdf":
            xmlfile = base + ".xml"
            m21score.write('musicxml.pdf', xmlfile)
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


def renderQuantizedParts(parts: List[quant.QuantizedPart],
                         options:RenderOptions=None,
                         backend='music21') -> Renderer:
    """
    Render the already quantized parts as notation.

    Args:
        parts: the already quantized parts
        options: the RenderOptions used. A value of None will use default options
        backend: one of {'music21'}

    Returns:
        a Renderer
    """
    if backend == 'music21':
        renderer = Music21Renderer(parts, options=options)
        return renderer
    else:
        raise ValueError(f"Supported backends: 'music21'. Got {backend}")


def renderParts(parts: List[core.Part],
                struct: ScoreStructure=None,
                options: RenderOptions=None,
                backend='music21',
                quantizationProfile:quant.QuantizationProfile=None
                ) -> Renderer:
    """
    Quantize and render unquantized events organized into parts

    Args:
        parts: a list of Parts, where each part represents a series
            of non-overlapping events which have not yet been quantized
        struct:
            the structure of the resulting score. To create a simple score
            with an anitial time signature and tempo, use something like
            `ScoreStructure.fromTimesig((4, 4), quarterTempo=52)`. If not given,
            defaults to a 4/4 score with tempo 60
        options:
            leave as None to use default render options, or create a
            RenderOptions object to specify things like page size, title,
            pitch resolution, etc.
        backend:
            The backend used for rendering. Supported backends at the
            moment: 'music21'
        quantizationProfile:
            The quantization profile determines how events are quantized,
            which divisions of the beat are possible, how a best division
            is weighted and selected, etc. Not all options in a profile
            are supported by all backends (for example, music21 backend
            does not support nested tuples).
            See quant.presetQuantizationProfiles, which is a dict with
            some predefined profiles

    Returns:
        a Renderer. To produce a pdf, call .write('out.pdf') on
        the returned Renderer.

    """
    if quantizationProfile is None:
        quantizationProfile = quant.QuantizationProfile()
    if backend == 'music21':
        quantizationProfile.nestedTuples = False
    if struct is None:
        struct = ScoreStructure.fromTimesig((4,4), quarterTempo=60)
    qparts = []
    for part in parts:
        qpart = quant.quantizePart(struct,
                                   eventsInPart=part,
                                   profile=quantizationProfile)
        qpart.label = part.label
        qparts.append(qpart)
    renderer = renderQuantizedParts(parts=qparts, options=options, backend=backend)
    renderer.render()
    return renderer


def render(obj,
           struct:ScoreStructure=None,
           options: RenderOptions = None,
           backend='music21',
           quantizationProfile: quant.QuantizationProfile = None
           ) -> Renderer:
    if isinstance(obj, core.Part):
        parts = [obj]
    elif isinstance(obj, list):
        if all(isinstance(item, core.Part) for item in obj):
            parts = obj
        elif all(isinstance(item, core.Notation) for item in obj):
            parts = [core.Part(obj)]
        else:
            raise TypeError(f"Can't show {obj}")
    elif isinstance(obj, core.Notation):
        parts = [core.Part([obj])]
    else:
        raise TypeError(f"Can't show {obj}")
    return renderParts(parts, struct=struct, options=options, backend=backend,
                       quantizationProfile=quantizationProfile)


def show(obj, struct:ScoreStructure = None, options:RenderOptions = None) -> None:
    """
    Args:
        obj: a Part, a list of Parts, a Notation, a list of Notations
        struct: the scorestructure to use when rendering
        options: the render options to use, or None to use defaults
    """
    render(obj, struct=struct).show()