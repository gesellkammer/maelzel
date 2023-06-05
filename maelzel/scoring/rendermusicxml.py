from __future__ import annotations
from xml.dom import minidom as md
from functools import cache
from dataclasses import dataclass, field
import math

import pitchtools as pt

from .common import *
from . import attachment
from . import definitions
from .core import Notation
from .render import Renderer, RenderOptions
from .node import Node
from . import quant, util
from maelzel.scoring.tempo import inferMetronomeMark
from . import spanner as _spanner

from .render import Renderer, RenderOptions


def _elem(doc: md.Document, parent: md.Element, name: str, attrs: dict = None
          ) -> md.Element:
    """Create a child Element"""
    elem: md.Element = doc.createElement(name)
    parent.appendChild(elem)
    if attrs:
        for name, value in attrs.items():
            elem.setAttribute(name, str(value))
    return elem


def _elemText(doc: md.Document, parent: md.Element, child: str, text) -> None:
    """Create a child with text, <child>text</child>"""
    childelem = _elem(doc, parent, child)
    _text(doc, childelem, str(text))


def _text(doc: md.Document, parent: md.Element, text: str
          ) -> md.Text:
    """Create a text for parent"""
    textelem = doc.createTextNode(text)
    parent.appendChild(textelem)
    return textelem


_alterToAccidental = {
    0.0: 'natural',
    0.25: 'natural-up',
    0.5: 'quarter-sharp',
    0.75: 'sharp-down',
    1.0: 'sharp',
    1.25: 'sharp-up',
    1.5: 'three-quarters-sharp',
    -0.25: 'natural-down',
    -0.5: 'quarter-flat',
    -0.75: 'flat-up',
    -1.0: 'flat',
    -1.25: 'flat-down',
    -1.5: 'three-quarters-flat'
}


def _measureMaxDivisions(measure: quant.QuantizedMeasure, mindivs=840
                         ) -> int:
    """
    Calculate the max. divisions to represent all durations in measure

    The minimum value is lcm(5, 6, 7, 8)
    """
    if measure.isEmpty():
        return mindivs
    denominators = {n.duration.denominator
                    for n in measure.tree().recurse()}
    return max(math.lcm(*denominators), mindivs)


def _mxmlClef(clef: str) -> tuple[str, int, int]:
    """
    Return the corresponding mxml clef

    Args:
        clef: the clef name, one of '

    Returns:
        a tuple (sign, line, octavechange)
    """
    return {
        'treble': ('G', 2, 0),
        'treble8': ('G', 2, 1),
        'bass': ('F', 4, 0),
        'bass8': ('F', 4, -1),
        'alto': ('C', 3, 0)
    }[clef]


def renderMusicxml(score: quant.QuantizedScore,
                   options: RenderOptions
                   ) -> str:
    xmldoc = _makeMusicxmlDocument(score=score, options=options)
    return xmldoc.toprettyxml()


def _makeMusicxmlDocument(score: quant.QuantizedScore,
                          options: RenderOptions
                          ) -> md.Document:
    impl = md.getDOMImplementation('')
    dt = impl.createDocumentType('score-partwise',
                                 "-//Recordare//DTD MusicXML 4.0 Partwise//EN",
                                 "http://www.musicxml.org/dtds/partwise.dtd")
    doc: md.Document = impl.createDocument('http://www.w3.org/1999/xhtml', 'score-partwise', dt)
    root: md.Element  = doc.documentElement
    root.setAttribute('version', '4.0')
    part_list = _elem(doc, root, 'part-list')
    partnum = 1
    partids = {}
    partGroups = score.groupParts()

    for partGroup in partGroups:
        for part in partGroup:
            partid = f"P{partnum}"
            partids[id(part)] = partid
            score_part = _elem(doc, part_list, 'score-part', {'id': partid})
            part_name = _elem(doc, score_part, 'part-name')
            _text(doc, part_name, part.name or partid)
            partnum += 1

    for partGroup in partGroups:
        for part in partGroup:
            partid = partids[id(part)]
            part_ = _elem(doc, root, 'part', {'id': partid})
            _renderPart(part, doc, part_, renderOptions=options)

    return doc


@dataclass
class _RenderState:
    divisions: int = 0
    measure: quant.QuantizedMeasure | None = None
    insideSlide: bool = False
    glissando: bool = False
    tupletStack: list[tuple[int, int]] = field(default_factory=list)
    dynamic: str = ''
    insideGraceGroup: bool = False
    openSpanners: dict[str, _spanner.Spanner] = field(default_factory=dict)


def _renderNotation(n: Notation,
                    doc: md.Document,
                    parent: md.Element,
                    options: RenderOptions,
                    state: _RenderState
                    ) -> None:
    notatedDur = n.notatedDuration()

    durationDivisions: F = n.duration * state.divisions
    assert durationDivisions.denominator == 1

    if n.isRest or (len(n.pitches) == 1 and n.pitches[0] == 0):
        # A rest
        note_ = _elem(doc, parent, 'note')
        _elem(doc, note_, 'rest')
        _elemText(doc, note_, 'duration', int(durationDivisions))

        # TODO: Add attachments
        return

    # A note / Chord
    notenames = n.resolveNotenames()
    notatedPitches = [pt.notated_pitch(notename) for notename in notenames]
    ischord = len(n.pitches) > 1
    for i, pitch in enumerate(n.pitches):
        notatedPitch = notatedPitches[i]
        assert abs(notatedPitch.cents_deviation) in (0, 25, 50), f"{notatedPitch=}"
        note_ = _elem(doc, parent, 'note')
        if ischord and i > 0:
            _elem(doc, note_, 'chord')
        pitch_ = _elem(doc, note_, 'pitch')
        _elemText(doc, pitch_, 'step', notatedPitch.diatonic_name)
        if notatedPitch.diatonic_alteration and not n.tiedPrev:
            _elemText(doc, pitch_, 'alter', f'{notatedPitch.diatonic_alteration:g}')
        _elemText(doc, pitch_, 'octave', notatedPitch.octave)

        # For some musicxml parsers, the order of these elements matter
        # For example, musescore needs: duration, tie, type, accidental
        # (elements can be omitted but, if present, need to appear in
        # this order)
        _elemText(doc, note_, 'duration', int(durationDivisions))
        if n.tiedPrev:
            _elem(doc, note_, 'tie', {'type': 'stop'})

        if n.tiedNext:
            _elem(doc, note_, 'tie', {'type': 'start'})
            # TODO: add the tie in <notations>

        _elemText(doc, note_, 'type', notatedDur.baseName())

        if not n.tiedPrev and notatedPitch.cents_deviation != 0:
            _elemText(doc, note_, 'accidental', _alterToAccidental[notatedPitch.diatonic_alteration])

        durratio = notatedDur.timeModification()
        if durratio is not None:
            timemod_ = _elem(doc, note_, 'time-modification')
            _elemText(doc, timemod_, 'actual-notes', durratio.numerator)
            _elemText(doc, timemod_, 'normal-notes', durratio.denominator)








def _renderNode(node: Node,
                durRatios: list[F],
                doc: md.Document,
                parent: md.Element,
                options: RenderOptions,
                state: _RenderState
                ) -> None:
    # A tuplet follows the first note
    if node.durRatio != (1, 1):
        durRatios.append(F(*node.durRatio))
        state.tupletStack.append(node.durRatio)
        tupletStarted = True
    else:
        tupletStarted = False

    for i, item in enumerate(node.items):
        if isinstance(item, Node):
            _renderNode(node=item, durRatios=durRatios, options=options,
                        doc=doc, parent=parent,
                        state=state)
        else:
            assert isinstance(item, Notation)
            if not item.gliss and state.glissando:
                # Stop glissando skip
                state.glissando = False

            if item.isRest:
                state.dynamic = ''

            if item.dynamic:
                dynamic = item.dynamic
                if (options.removeSuperfluousDynamics and
                        not item.dynamic.endswith('!') and
                        item.dynamic == state.dynamic and
                        item.dynamic in definitions.dynamicLevels):
                    item.dynamic = ''
                state.dynamic = dynamic

            _renderNotation(item, doc=doc, parent=parent,
                            options=options, state=state)


def _renderPart(part: quant.QuantizedPart,
                doc: md.Document,
                root: md.Element,
                renderOptions: RenderOptions,
                addMeasureMarks=True,
                addTempoMarks=True,
                ) -> None:
    lastDivisions = 0
    lastTimesig = (0, 0)
    firstclef = part.firstclef or part.bestClef()
    lastTempo = 0
    scorestruct = part.struct

    # TODO: key signature

    state = _RenderState()

    for measureidx, measure in enumerate(part.measures):
        state.measure = measure

        measureDef = scorestruct.getMeasureDef(measureidx)
        measure_ = _elem(doc, root, 'measure', {'number': measureidx + 1})
        attributes_ = _elem(doc, measure_, 'attributes')
        divisions = _measureMaxDivisions(measure)
        state.divisions = divisions

        if divisions != lastDivisions:
            _elemText(doc, attributes_, 'divisions', str(divisions))
            lastDivisions = divisions
        if measure.timesig != lastTimesig:
            lastTimesig = measure.timesig
            time_ = _elem(doc, attributes_, "time")
            _elemText(doc, time_, "beats", measure.timesig[0])
            _elemText(doc, time_, "beat-type", measure.timesig[1])
        if measureidx == 0:
            clef_ = _elem(doc, attributes_, "clef")
            clefsign, clefline, clefoctave = _mxmlClef(firstclef)
            _elemText(doc, clef_, "sign", clefsign)
            _elemText(doc, clef_, "line", clefline)
            if clefoctave:
                _elemText(doc, clef_, "clef-octave-change", clefoctave)
        # End <attributes>
        if addTempoMarks and measure.quarterTempo != lastTempo:
            metro = inferMetronomeMark(measure.quarterTempo, timesig=measure.timesig)
            _addMetronome(doc, measure_, unit=metro.unitstr, bpm=metro.bpm, numdots=metro.dots)
            lastTempo = measure.quarterTempo

        # Measure Marks
        if addMeasureMarks:
            if measureDef.annotation:
                # TODO: add measure annotation
                pass
            if measureDef.rehearsalMark:
                # TODO: add measure rehearsal mark
                pass

        if measure.isEmpty():
            note_ = _elem(doc, measure_, "note")
            _elem(doc, note_, "rest", {'measure': 'yes'})
            _elemText(doc, note_, "duration", int(measure.duration() * divisions))
        else:
            root = measure.tree()
            _renderNode(root, doc=doc, parent=measure_,
                        durRatios=[], options=renderOptions, state=state)
        # Notes


def _addMetronome(doc: md.Document,
                  parent: md.Element,
                  unit: str,
                  bpm: float,
                  numdots: int=0,
                  bpmdecimals=1,
                  placement='above') -> None:
    direction_ = _elem(doc, parent, "direction", {'placement': placement})
    dirtype_ = _elem(doc, direction_, "direction-type")
    metro_ = _elem(doc, dirtype_, "metronome")
    _elemText(doc, metro_, 'beat-unit', unit)
    if numdots > 0:
        for _ in range(numdots):
            _elem(doc, metro_, 'beat-unit-dot')
    _elemText(doc, metro_, 'per-minute', f'{round(bpm, bpmdecimals):g}')


class MusicxmlRenderer(Renderer):
    def __init__(self,
                 score: quant.QuantizedScore,
                 options: RenderOptions):
        super().__init__(score=score, options=options)
        self._lastrender: str = ''

    def writeFormats(self) -> list[str]:
        return ['pdf', 'musicxml', 'png']

    def render(self, options: RenderOptions | None = None) -> str:
        self._lastrender = self._render(options=options if options is not None else self.options)
        return self._lastrender

    @cache
    def _render(self, options: RenderOptions) -> str:
        assert isinstance(options, RenderOptions)
        return renderMusicxml(self.quantizedScore, options=options)


