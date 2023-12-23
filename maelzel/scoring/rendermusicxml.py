from __future__ import annotations

from xml.dom import minidom as md
from functools import cache
from dataclasses import dataclass, field
import os
import re
import glob
import math
import subprocess

import pitchtools as pt

from emlib import iterlib
import emlib.colordata
import emlib.filetools

from .common import *
from . import attachment as _attachment
from . import definitions
from . import quant
from . import spanner as _spanner
from .core import Notation
from .node import Node
from .render import Renderer, RenderOptions
from . import util
from maelzel.scoring.tempo import inferMetronomeMark


_articulationAttachments = {
    'staccato': 'staccato',
    'tenuto': 'tenuto',
    'accent': 'accent',
    'marcato': 'strong-accent',
    'staccatissimo': 'staccatissimo',
    'espressive': 'soft-accent',
    'portato': 'detached-legato',

}


_xmlEnclosures = {
    'rectangle',
    'square',
    'oval',
    'circle',
    'bracket',
    'triangle',
    'diamond',
    'hexagon',
    'none'
}

# availableOrnaments = {'trill', 'mordent', 'prall', 'turn', 'tremolo'}
_xmlOrnaments = {
    'prall': 'inverted-mordent',
    'mordent': 'mordent',
    'turn': 'turn',
    'trill': 'trill-mark',
    'tremolo': 'tremolo'
}


_notationsAttachments = {
    'laissezvibrer': 'tied',
    'arpeggio': 'arpeggiate'
}


_technicalAttachments = {
    'upbow': 'up-bow',
    'downbow': 'down-bow',
    'snappizz': 'snap-pizzicato',
    'flageolet': 'harmonic',
    'open': 'open',
    'closed': 'stopped',
    'stopped': 'stopped',
    'openstring': 'open-string',
}


def _xmlAttributes(obj, attributes: tuple[str, ...]) -> dict:
    return {attr: value for attr in attributes
            if (value:=getattr(obj, attr))}


def _elem(doc: md.Document, parent: md.Element, name: str, attrs: dict = None,
          **kws
          ) -> md.Element:
    """Create a child Element"""
    elem: md.Element = doc.createElement(name)
    parent.appendChild(elem)
    if kws:
        attrs = kws if not attrs else attrs | kws

    if attrs:
        for name, value in attrs.items():
            if value is None:
                continue
            if name == 'color':
                if not _isXmlColor(value):
                    xmlcolor = _asXmlColor(value)
                    if not xmlcolor:
                        raise ValueError(f"Color {value} cannot be converted to a valid XML color")
                    value = xmlcolor
            elem.setAttribute(name, str(value))
    return elem


def _elemText(doc: md.Document, parent: md.Element, child: str, text, attrs: dict = None,
              **kws
              ) -> md.Element:
    """Create a child with text, <child>text</child>, returns <child>"""
    childelem = _elem(doc, parent, child, attrs=attrs, **kws)
    _text(doc, childelem, str(text))
    return childelem


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


def _measureMaxDivisions(measure: quant.QuantizedMeasure
                         ) -> int:
    """
    Calculate the max. divisions to represent all durations in measure

    The minimum value is lcm(5, 6, 7, 8)
    """
    if measure.empty():
        return 1
    denominators = {n.duration.denominator
                    for n in measure.tree.recurse()}
    return max(math.lcm(*denominators), 1)


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
        'treble15': ('G', 2, 2),
        'bass': ('F', 4, 0),
        'bass8': ('F', 4, -1),
        'bass15': ('F', 4, -2),
        'alto': ('C', 3, 0)
    }[clef]


def renderMusicxml(score: quant.QuantizedScore,
                   options: RenderOptions,
                   indent='\t'
                   ) -> str:
    xmldoc = makeMusicxmlDocument(score=score, options=options)
    return xmldoc.toprettyxml(indent=indent)


def makeMusicxmlDocument(score: quant.QuantizedScore,
                         options: RenderOptions
                         ) -> md.Document:
    impl = md.getDOMImplementation('')
    assert impl is not None
    dt = impl.createDocumentType('score-partwise',
                                 "-//Recordare//DTD MusicXML 4.0 Partwise//EN",
                                 "http://www.musicxml.org/dtds/partwise.dtd")
    doc: md.Document = impl.createDocument('http://www.w3.org/1999/xhtml', 'score-partwise', dt)
    root: md.Element  = doc.documentElement
    root.setAttribute('version', '4.0')

    defaults_ = _elem(doc, root, 'defaults')
    scaling_ = _elem(doc, defaults_, 'scaling')
    mm, tenths = options.musicxmlTenthsScaling()
    _elemText(doc, scaling_, 'millimeters', mm)
    _elemText(doc, scaling_, 'tenths', tenths)

    pagelayout_ = _elem(doc, defaults_, 'page-layout')
    heightmm, widthmm = options.pageSizeMillimeters()
    _elemText(doc, pagelayout_, 'page-height', int(heightmm * tenths))
    _elemText(doc, pagelayout_, 'page-width', int(widthmm * tenths))

    partlist_ = _elem(doc, root, 'part-list')
    partnum = 1
    partids = {}
    partGroups = score.groupParts()

    for partGroup in partGroups:
        for part in partGroup:
            partid = f"P{partnum}"
            partids[id(part)] = partid
            scorepart_ = _elem(doc, partlist_, 'score-part', {'id': partid})
            partname_ = _elem(doc, scorepart_, 'part-name')
            _text(doc, partname_, part.name or partid)
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
    clef: str = ''
    insideGraceGroup: bool = False
    openSpanners: dict[str, _spanner.Spanner] = field(default_factory=dict)


def _timeModification(notatedDur: NotatedDuration, doc: md.Document, parent: md.Element):
    durratio = notatedDur.timeModification()
    if durratio is not None:
        timemod_ = _elem(doc, parent, 'time-modification')
        _elemText(doc, timemod_, 'actual-notes', durratio.numerator)
        _elemText(doc, timemod_, 'normal-notes', durratio.denominator)


def _needsNotationsElement(n: Notation):
    # TODO
    return False


_unfilledShapes = {
    'harmonic': 'diamond'
}

# 'square' and 'rectangle' are actually defined in the musicxml standard but are
# not imported into musescore
_filledShapes: dict[str, str] = {
    'cross': 'x',
    'triangleup': 'triangle',
    'xcircle': 'circle-x',
    'triangle': 'triangle',
    'rhombus': 'diamond',
    'square': 'la',
    'rectangle': 'la',
    'slash': 'slash',
    'diamond': 'diamond',
    'do': 'do',
    're': 're',
    'mi': 'mi',
    'fa': 'fa',
    'sol': 'sol',
    'la': 'la',
    'ti': 'ti',
    'cluster': 'cluster'
}


@dataclass
class XmlNotehead:
    shape: str
    filled: bool
    color: str = ''
    parentheses: bool = False
    hidden: bool = False
    size: int = 0

    def __post_init__(self):
        assert self.shape
        assert not self.color or _isXmlColor(self.color), f"Color {self.color} is not a valid XML color"


def scoringNoteheadToMusicxml(notehead: definitions.Notehead) -> XmlNotehead:
    hidden = notehead.hidden
    filled = True

    if notehead.shape == 'hidden' or notehead.hidden:
        hidden = True
        shape = 'none'
    elif notehead.shape == 'normal' or notehead.shape == '':
        shape = 'normal'
    elif notehead.shape == 'hidden':
        hidden = True
        shape = 'none'
    elif notehead.shape in _filledShapes:
        shape = _filledShapes[notehead.shape]
        filled = True
    elif notehead.shape in _unfilledShapes:
        shape = _unfilledShapes[notehead.shape]
        filled = False
    else:
        raise ValueError(f"Notehead shape '{notehead.shape}' not supported")
    color = _asXmlColor(notehead.color) if notehead.color else ''
    size = int(notehead.size) if notehead.size is not None else 0
    return XmlNotehead(shape=shape, hidden=hidden, filled=filled, color=color,
                       parentheses=notehead.parenthesis, size=size)


def _xmlAccidentalSize(relativesize: int | float) -> str:
    intsize = int(round(relativesize))
    if intsize == 0:
        return ''
    elif intsize == 1:
        return 'full'
    elif intsize >= 2:
        return 'large'
    elif intsize <= -1:
        return 'cue'
    else:
        logger.warning(f"Size {intsize} not supported")
        return ''


def _notePitch(doc: md.Document,
               parent: md.Element,
               pitch: pt.NotatedPitch | None,
               durationDivisions: int,
               numdots: int = 0,
               chord=False,
               tiedprev=False,
               tiednext=False,
               grace=False,
               accidentalTraits: _attachment.AccidentalTraits | None = None,
               durtype: str = '',
               stemless=False
               ):
    if pitch is None:
        _elem(doc, parent, 'rest')
    else:
        if grace:
            _elem(doc, parent, 'grace')
        if chord:
            _elem(doc, parent, 'chord')
        pitch_ = _elem(doc, parent, 'pitch')
        _elemText(doc, pitch_, 'step', pitch.diatonic_name)
        if not pitch.chromatic_alteration or not tiedprev:
            _elemText(doc, pitch_, 'alter', f'{pitch.diatonic_alteration:g}')
        _elemText(doc, pitch_, 'octave', pitch.octave)

    # Common attributes for rest/note
    if durationDivisions > 0:
        _elemText(doc, parent, 'duration', durationDivisions)

    # TODO: implement selective ties
    if tiedprev:
        _elem(doc, parent, 'tie', {'type': 'stop'})
    if tiednext:
        _elem(doc, parent, 'tie', {'type': 'start'})

    if durtype:
        _elemText(doc, parent, 'type', durtype)

    if numdots:
        for _ in range(numdots):
            _elem(doc, parent, 'dot')

    if pitch and ((not tiedprev and pitch.cents_deviation != 0) or accidentalTraits):
        attrs = {}
        if accidentalTraits:
            if accidentalTraits.parenthesis:
                attrs['parentheses'] = 'yes'
            if accidentalTraits.color and (xmlcolor := _asXmlColor(accidentalTraits.color)):
                attrs['color'] = xmlcolor
            if accidentalTraits.brackets:
                attrs['bracket'] = 'yes'
            if accidentalTraits.size:
                xmlsize = _xmlAccidentalSize(accidentalTraits.size)
                if xmlsize:
                    attrs['size'] = xmlsize

        _elemText(doc, parent, 'accidental',
                  text=_alterToAccidental[pitch.diatonic_alteration],
                  attrs=attrs)

    if stemless:
        _elemText(doc, parent, 'stem', 'none')



def _tupletNotation(doc: md.Document,
                    notationsElem: md.Element,
                    tuplet: tuple[int, int],
                    tupletnumber: int,
                    tuplettype='',
                    bracket: bool | None = None):
    """
     <tuplet bracket="yes" number="2" type="start">
            <tuplet-actual>
              <tuplet-number>5</tuplet-number>
              <tuplet-type>16th</tuplet-type>
            </tuplet-actual>
            <tuplet-normal>
              <tuplet-number>4</tuplet-number>
              <tuplet-type>16th</tuplet-type>
            </tuplet-normal>
          </tuplet>
    """
    attrs = dict(number=tupletnumber, type='start')
    if bracket is not None:
        attrs['bracket'] = 'yes' if bracket else 'no'
    tuplet_ = _elem(doc, notationsElem, 'tuplet', attrs)
    tupletactual_ = _elem(doc, tuplet_, 'tuplet-actual')
    _elemText(doc, tupletactual_, 'tuplet-number', tuplet[0])
    if tuplettype:
        _elemText(doc, tupletactual_, 'tuplet-type', tuplettype)
    tupletnormal_ = _elem(doc, tuplet_, 'tuplet-normal')
    _elemText(doc, tupletnormal_, 'tuplet-number', tuplet[1])
    if tuplettype:
        _elemText(doc, tupletnormal_, 'tuplet-type', tuplettype)


def _direction(doc: md.Document,
               parent: md.Element,
               direction: str,
               attrs: dict | None = None,
               placement: str = ''):
    dirattrs = {'placement': placement} if placement else None
    direction_ = _elem(doc, parent, 'direction', dirattrs)
    dirtype_ = _elem(doc, direction_, 'direction-type')
    inner_ = _elem(doc, dirtype_, direction, attrs)
    return direction_, inner_


def _hairpinDirection(doc: md.Document,
                      parent: md.Element,
                      kind='crescendo',
                      placement='below',
                      niente=False
                      ):
    _direction(doc, parent, direction='wedge', placement=placement,
               attrs={'type': kind, 'niente': "yes" if niente else "no"})


def _dynamicDirection(doc: md.Document,
                      parent: md.Element,
                      dynamic: str,
                      placement='below',
                      velocity: int=0):
    """
    <direction placement="below">
      <direction-type>
         <dynamics default-x="84" default-y="-73" halign="center">
            <f/>
         </dynamics>
      </direction-type>
      <sound dynamics="98"/>
   </direction>
    Returns:

    """
    direction_, dynamics_ = _direction(doc, parent, direction='dynamics', placement=placement)
    _elem(doc, dynamics_, dynamic)
    if velocity:
        _elem(doc, direction_, 'sound', dynamics=velocity)


def _renderPitch(doc: md.Document,
                 parent: md.Element,
                 notation: Notation,
                 notename: str,
                 idx: int,
                 state: _RenderState,
                 ) -> None:
    accidentalTraits = notation.findAttachment(cls=_attachment.AccidentalTraits, anchor=idx)
    if accidentalTraits:
        assert isinstance(accidentalTraits, _attachment.AccidentalTraits)
    pitch = None if notation.isRest else pt.notated_pitch(notename)
    durationDivisions: F = notation.duration * state.divisions
    # #_elemText(doc, note0_, 'type', )
    notatedDur = notation.notatedDuration()

    _notePitch(doc, parent=parent,
               pitch=pitch,
               durationDivisions=int(durationDivisions),
               chord=not notation.isRest and len(notation.pitches) > 1 and idx > 0,
               tiedprev=notation.tiedPrev,
               tiednext=notation.tiedNext,
               grace=notation.isGracenote,
               accidentalTraits=accidentalTraits,
               durtype=notatedDur.baseName() if notation.duration > 0 else 'eighth',
               numdots=notatedDur.dots,
               stemless=notation.isStemless)


def _renderNotehead(doc: md.Document, parent: md.Element, notehead: definitions.Notehead):
    # <notehead filled="no">diamond</notehead>
    try:
        xmlnotehead = scoringNoteheadToMusicxml(notehead)
        attrs = {}
        if xmlnotehead.shape and not xmlnotehead.hidden:
            attrs['filled'] = 'yes' if xmlnotehead.filled else 'no'
        if xmlnotehead.color:
            attrs['color'] = xmlnotehead.color
        if xmlnotehead.parentheses:
            attrs['parentheses'] = 'yes'
        if xmlnotehead.size:
            attrs['font-size'] = xmlnotehead.size
        notehead_ = _elem(doc, parent, 'notehead', attrs=attrs)
        if xmlnotehead.shape:
            _text(doc, notehead_, xmlnotehead.shape)
    except ValueError as e:
        logger.error(f"Error converting notehead to musicxml (error='{e}'), skipping: {notehead}")



def _renderNotation(n: Notation,
                    nindex: int,
                    node: Node,
                    doc: md.Document,
                    parent: md.Element,
                    options: RenderOptions,
                    state: _RenderState,
                    ) -> None:

    ornaments_: md.Element | None = None
    attributes_: md.Element | None = None
    # Preprocess
    if n.attachments:
        for attach in n.attachments:
            if isinstance(attach, _attachment.Harmonic):
                n = n.resolveHarmonic()
            elif isinstance(attach, _attachment.Clef) and attach.kind != state.clef:
                if attributes_ is None:
                    attributes_ = _elem(doc, parent, 'attributes')
                clef_ = _elem(doc, attributes_, 'clef')

                sign, line, octave = _mxmlClef(attach.kind)
                _elemText(doc, clef_, 'sign', sign)
                _elemText(doc, clef_, 'line', line)
                if octave != 0:
                    _elemText(doc,clef_, 'clef-octave-change', octave)
                state.clef = attach.kind

    notatedDur = n.notatedDuration()

    durationDivisions: F = n.duration * state.divisions
    assert durationDivisions.denominator == 1, f"{durationDivisions=}, {n.duration=}, {state.divisions=}"

    if n.dynamic:
        _dynamicDirection(doc, parent, n.dynamic)

    numpitches = len(n.pitches) if not n.isRest else 1
    notes_ = [_elem(doc, parent, 'note') for i in range(numpitches)]
    note0_ = notes_[0]

    notenames = n.resolveNotenames()
    qnotenames = [pt.quantize_notename(notename, divisions_per_semitone=options.divsPerSemitone)
                  for notename in notenames]

    _renderPitch(doc, parent=note0_, notation=n,
                 notename=qnotenames[0] if qnotenames else '',
                 idx=0, state=state)

    if not n.isRest:
        if numpitches > 1:
            # The rest pitches if a chord
            for i, notename in enumerate(qnotenames[1:], 1):
                parent = notes_[i]
                _renderPitch(doc, parent,
                             notation=n,
                             idx=i,
                             notename=notename,
                             state=state)

    # Noteheads
        for i in range(numpitches):
            notehead = n.getNotehead(i)
            if notehead:
                _renderNotehead(doc, notes_[i], notehead)

    # TODO: add cents deviation annotation

    if notatedDur.tuplets:
        _timeModification(notatedDur, doc, note0_)

    _notations_: dict[int, md.Element] = {}

    def notations(noteindex: int) -> md.Element:
        if (elem := _notations_.get(noteindex)) is None:
            _notations_[noteindex] = elem = _elem(doc, notes_[noteindex], 'notations')
        return elem

    if (starttuplets := n.getProperty('__starttuplets__', None)):
        for tupindex, tupratio in enumerate(starttuplets):
            _tupletNotation(doc, notations(0), tuplet=tupratio, tupletnumber=tupindex+1,
                            bracket=tupratio[0] != 3)

    elif (tuplets := n.getProperty('__stoptuplets__', None)):
        for tupindex, tuplet in iterlib.reversed_enumerate(tuplets):
            # assert state.tupletStack[-1] == tuplet
            # <notations><tuplet number="2" type="stop"/></notations>
            _elem(doc, notations(0), 'tuplet', number=tupindex+1, type='stop')

    spannerkind = 'slide' if options.musicxmlSolidSlideGliss and options.glissLineType == 'solid' else 'glissando'
    linetype = options.glissLineType

    if state.glissando and (not n.tiedPrev or not n.gliss):
        for i in range(numpitches):
            _notationsSpanner(doc, notations(i), spannerkind, number=i+1, type='stop')

    # continue gliss: n.tiedPrev and n.gliss and state.glissando
    elif n.tiedPrev and n.gliss and state.glissando:
        if options.glissHideTiedNotes:
            _elemText(doc, note0_, 'notehead', text='none')
        # TODO: continue glissando
        pass

    # start gliss: n.gliss and (not state.glissando or not n.tiedPrev)
    if n.gliss and (not state.glissando or not n.tiedPrev):
        for i in range(numpitches):
            _notationsSpanner(doc, notations(i), spannerkind, number=i+1,
                              type='start', linetype=linetype)

    state.glissando = n.gliss

    # Attachments
    if n.attachments:
        articulations_: md.Element | None = None
        articulations = [attach for attach in n.attachments
                         if isinstance(attach, _attachment.Articulation) and attach.kind in _articulationAttachments]
        if articulations:
            articulations_ = _elem(doc, notations(0), 'articulations')
            for articulation in articulations:
                attrs = _xmlAttributes(articulation, ('color', 'placement'))
                xmlname = _articulationAttachments[articulation.kind]
                _elem(doc, articulations_, xmlname, attrs=attrs)

        for attach in n.attachments:
            if isinstance(attach, _attachment.Articulation):
                if attach.kind in _notationsAttachments:
                    # All notes in chord need to get the attachment
                    for i in range(len(n.pitches)):
                        notations_ = notations(i)
                        attrs = _xmlAttributes(attach, ('color', 'placement'))
                        _elem(doc, notations_, _notationsAttachments[attach.kind], attrs)
                        if attach.kind == 'laissezvibrer':
                            attrs['type'] = 'let-ring'

                elif attach.kind in _technicalAttachments:
                    notations_ = notations(0)
                    technical_ = _elem(doc, notations_, 'technical')
                    attrs = _xmlAttributes(attach, ('color', 'placement'))
                    symbol_ = _elem(doc, technical_, _technicalAttachments[attach.kind], attrs)
                    if attach.kind == 'flageolet':
                        _elem(doc, symbol_, 'natural')
            elif isinstance(attach, _attachment.Ornament):
                # TODO: ornaments
                if ornaments_ is None:
                    ornaments_ = _elem(doc, notations(0), 'ornaments')
                xmlornament = _xmlOrnaments.get(attach.kind)
                if xmlornament:
                    _elem(doc, ornaments_, xmlornament)
                else:
                    logger.error(f"Ornament {attach.kind} cannot be converted to musicxml. "
                                 f"Possible musicxml ornaments: {_xmlOrnaments.keys()}")
            elif isinstance(attach, _attachment.Tremolo):
                if ornaments_ is None:
                    ornaments_ = _elem(doc, notations(0), 'ornaments')
                tremtype = 'stop' if attach.tremtype == 'end' else  attach.tremtype
                _elemText(doc, ornaments_, "tremolo", attach.nummarks, type=tremtype)

            elif isinstance(attach, _attachment.Breath) and attach.visible:
                # TODO: breath marks
                if articulations_ is None:
                    articulations_ = _elem(doc, notations(0), 'articulations')
                # <breath-mark default-x="41" default-y="11" placement="above"/>
                _elem(doc, articulations_, 'breath-mark',
                      placement=attach.placement, color=attach.color or None)




    # Notations spanners
    if n.spanners:
        for spanner in n.spanners:
            if isinstance(spanner, _spanner.Slur):
                if spanner.kind == 'start':
                    # TODO: fix multiple simultaneous slurs
                    _notationsSpanner(doc, notations(0), 'slur', placement=spanner.placement,
                                      number=spanner.nestingLevel, linetype=spanner.linetype)
                elif spanner.kind == 'end':
                    _notationsSpanner(doc, notations(0), 'slur', type='stop',
                                      number=spanner.nestingLevel, linetype=spanner.linetype)
            elif isinstance(spanner, _spanner.TrillLine):
                ornaments_ = _elem(doc, notations(0), 'ornaments')
                if spanner.kind == 'start':
                    _elem(doc, ornaments_, 'trill-mark')
                    _elem(doc, ornaments_, 'wavy-line', type='start', number=1)
                else:
                    _elem(doc, ornaments_, 'wavy-line', type='stop', number=1)


    return


def _notationsSpanner(doc, notationsElem, spanner: str, placement='', type='start',
                      linetype: str = '',
                      attrs: dict | None = None, number=1):
    allattrs = dict(number=number, type=type)
    if placement:
        allattrs['placement'] = placement
    if linetype:
        allattrs['line-type'] = linetype
    if attrs:
        allattrs.update(attrs)
    _elem(doc, notationsElem, spanner, allattrs)


def _hasNotations(n: Notation) -> bool:
    """True if n needs a <notations> tag"""
    if n.gliss:
        return True

    _notationSpanners = (
        _spanner.Slur,
        _spanner.TrillLine
    )
    if n.spanners  and any (isinstance(sp, _notationSpanners) for sp in n.spanners):
        return True
    return False


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
        logger.debug(f"Starting tuplet {node.durRatio} at {node.firstNotation()}")
        node.firstNotation().getProperty('__starttuplets__', setdefault=[]).append(node.durRatio)
        state.tupletStack.append(node.durRatio)
        lastNotation = node.lastNotation()
        lastNotation.getProperty('__stoptuplets__', setdefault=[]).append(node.durRatio)

    for i, item in enumerate(node.items):

        if isinstance(item, Node):
            _renderNode(node=item, durRatios=durRatios, options=options,
                        doc=doc, parent=parent,
                        state=state)
        else:
            assert isinstance(item, Notation)

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

            # Spanner pre, as direction
            if item.spanners:
                for spanner in item.spanners:
                    if isinstance(spanner, _spanner.Hairpin):
                        if spanner.kind == 'start':
                            _direction(doc, parent, direction='wedge', placement='below',
                                       attrs={'type': 'crescendo' if spanner.direction == '<' else 'diminuendo',
                                              'niente': "yes" if spanner.niente else "no"})
                        elif spanner.kind == 'end':
                            _direction(doc, parent, direction='wedge', attrs={'type': 'stop'})
                    elif isinstance(spanner, _spanner.Bracket):
                        if spanner.kind == 'start':
                            placement = spanner.placement or 'above'
                            lineend = spanner.lineend or ('down' if placement == 'above' else 'up')
                            _direction(doc, parent, direction='bracket', placement=placement,
                                       attrs={'type': 'start',
                                              'line-type': spanner.linetype,
                                              'number': 1,
                                              'line-end': lineend})

            # Text attachments are translated as directions, which are placed before a note
            if item.attachments:
                for attach in item.attachments:
                    if isinstance(attach, _attachment.Text):
                        fontsize = 0 if attach.fontsize is None else attach.fontsize * options.musicxmlFontScaling
                        _words(doc, parent,
                               text=attach.text,
                               italic=attach.italic,
                               fontsize=fontsize,
                               enclosure=attach.box,
                               bold=attach.weight=='bold',
                               placement=attach.placement)

            if not item.isRest and options.showCents and not item.tiedPrev:
                if text := util.centsAnnotation(item.pitches,
                                                divsPerSemitone=options.divsPerSemitone,
                                                addplus=options.centsAnnotationPlusSign,
                                                separator=options.centsAnnotationSeparator):
                    _words(doc, parent=parent,
                           text=text,
                           placement=options.centsAnnotationPlacement,
                           fontsize=options.centsAnnotationFontsize * options.musicxmlFontScaling)

            _renderNotation(item, nindex=i, node=node, doc=doc, parent=parent,
                            options=options, state=state)
            # Spanner post, as direction
            if item.spanners:
                for spanner in item.spanners:
                    if isinstance(spanner, _spanner.Bracket):
                        if spanner.kind == 'end':
                            placement = spanner.placement or 'above'
                            lineend = spanner.lineend or ('down' if placement == 'above' else 'up')
                            _direction(doc, parent, direction='bracket',
                                       attrs={'type': 'stop', 'line-end': lineend})

    #if node.durRatio != (1, 1):
    #    state.tupletStack.pop()


_scoringBarstyleToXmlBarstyle = {
    'single': 'regular',
    'final': 'light-heavy',
    'double': 'light-light',
    'solid': 'heavy',
    'dotted': 'dotted',
    'dashed': 'dashed',
    'tick': 'tick',
    'short': 'short',
    'double-thin': 'light-light',
    'double-heavy': 'heavy-heavy',
    'none': 'none'
}


def _isXmlColor(color: str) -> bool:
    return re.match(r"^#(?:[0-9A-F]{3}){1,2}$", color) is not None


def _asXmlColor(color: str, default='') -> str:
    """
    Within the xml standard, a color needs to adjust to '#[\dA-F]{6}([\dA-F][\dA-F])?'

    Returns:
        the corresponding hex color as needed in musicxml  (a hex rgs color
        in uppercase), or None if the given color is not supported
    """
    if re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$", color):
        return color.upper()
    if hexcolor := emlib.colordata.CSS4_COLORS.get(color):
        return hexcolor
    return default



def _words(doc: md.Document, parent: md.Element, text: str, placement='', color='',
           italic=False, bold=False, enclosure='', fontsize: int | float = 0, fontfamily=''):
    attrs = {}
    if color:
        xmlcolor = _asXmlColor(color)
        if not xmlcolor:
            logger.error(f"_words: Color {color} not supported ({text=})")
        else:
            attrs['color'] = color
    if enclosure:
        assert enclosure in _xmlEnclosures, f"Enclosure {enclosure} not known, possible enclosures: {_xmlEnclosures}"
        attrs['enclosure'] = enclosure
    if bold:
        attrs['font-weight'] = 'bold'
    if italic:
        attrs['font-style'] = 'italic'
    if fontsize:
        attrs['font-size'] = fontsize
    if fontfamily:
        attrs['font-family'] = fontfamily
    direction_ , words_ = _direction(doc, parent, direction='words', placement=placement, attrs=attrs)
    _text(doc, words_, text)


def _prepareRender(part: quant.QuantizedPart) -> None:
    # 1. In musicxml, breath-mark notations appear AFTER the note. But in
    #    scoring, the breath mark indicates a breath before the note. So in
    #    order to render correctly we move all such attachments to the previous
    #    notation
    for n0, n1 in iterlib.pairwise(part.flatNotations()):
        assert isinstance(n0, Notation) and isinstance(n1, Notation)
        if not n1.attachments:
            continue
        breath = n1.findAttachment(_attachment.Breath)
        if breath:
            n0.addAttachment(breath)
            n1.removeAttachmentsByClass(_attachment.Breath)


def _renderPart(part: quant.QuantizedPart,
                doc: md.Document,
                root: md.Element,
                renderOptions: RenderOptions,
                addMeasureMarks=True,
                addTempoMarks=True,
                ) -> None:
    assert isinstance(doc, md.Document)
    assert isinstance(root, md.Element)
    lastDivisions = 0
    lastTimesig = (0, 0)
    firstclef = part.firstclef or part.bestClef()
    lastTempo = 0
    scorestruct = part.struct

    # TODO: key signature

    state = _RenderState(clef=firstclef)

    _prepareRender(part)

    for measureidx, measure in enumerate(part.measures):
        state.measure = measure

        measuredef = scorestruct.getMeasureDef(measureidx)
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
            for (num, den) in measure.timesig.parts:
                _elemText(doc, time_, "beats", num)
                _elemText(doc, time_, "beat-type", den)
            # TODO: add support for subdivision structure for non-compound time signatures
        if measureidx == 0:
            clef_ = _elem(doc, attributes_, "clef")
            clefsign, clefline, clefoctave = _mxmlClef(firstclef)
            _elemText(doc, clef_, "sign", clefsign)
            _elemText(doc, clef_, "line", clefline)
            if clefoctave:
                _elemText(doc, clef_, "clef-octave-change", clefoctave)

        if renderOptions.staffSize:
            # Musicxml uses a relative size
            scalingFactor = int(renderOptions.staffSize / renderOptions.referenceStaffsize * 100)
            staffdetails_ = _elem(doc, attributes_, 'staff-details')
            _elemText(doc, staffdetails_, 'staff-size', text=scalingFactor, scaling='100')

        # End <attributes>
        if addTempoMarks and measure.quarterTempo != lastTempo:
            metro = inferMetronomeMark(measure.quarterTempo, timesig=measure.timesig.parts[0])
            _addMetronome(doc, measure_, unit=metro.unitstr, bpm=metro.bpm, numdots=metro.dots)
            lastTempo = measure.quarterTempo

        # Measure Marks
        if addMeasureMarks:
            if measuredef.annotation:
                style = renderOptions.parsedMeasureAnnotationStyle
                _words(doc, parent=measure_,
                       text=measuredef.annotation,
                       placement='above',
                       enclosure=style.box,
                       fontsize=style.fontsize * renderOptions.musicxmlFontScaling if style.fontsize else 0.,
                       bold=style.bold,
                       italic=style.italic)

            if measuredef.rehearsalMark:
                style = renderOptions.parsedRehearsalMarkStyle
                _words(doc, parent=measure_,
                       text=measuredef.rehearsalMark.text,
                       placement='above',
                       fontsize=style.fontsize * renderOptions.musicxmlFontScaling if style.fontsize else 0.,
                       enclosure=measuredef.rehearsalMark.box or style.box,
                       bold=style.bold,
                       italic=style.italic)

        if measure.empty():
            note_ = _elem(doc, measure_, "note")
            _elem(doc, note_, "rest", {'measure': 'yes'})
            _elemText(doc, note_, "duration", int(measure.duration() * divisions))
            _elemText(doc, note_, "voice", 1)
        else:
            _renderNode(measure.tree, doc=doc, parent=measure_,
                        durRatios=[], options=renderOptions, state=state)

        if measuredef.barline:
            barline_ = _elem(doc, measure_, 'barline', location='right')
            xmlbarstyle = _scoringBarstyleToXmlBarstyle.get(measuredef.barline)
            if not xmlbarstyle:
                logger.error(f"Barstyle {measuredef.barline} not known. "
                             f"Possible barlines: {_scoringBarstyleToXmlBarstyle.keys()}")
            else:
                _elemText(doc, barline_, 'bar-style', xmlbarstyle)


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


def _findMuseScore():
    import maelzel.core.environment as envir
    return envir.findMusescore()


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
        return renderMusicxml(self.quantizedScore, options=options, indent=options.musicxmlIndent)

    def musicxml(self) -> str | None:
        xmltext = self.render()
        return xmltext

    def write(self, outfile: str, fmt: str = None, removeTemporaryFiles=False
              ) -> None:
        outfile = emlib.filetools.normalizePath(outfile)
        tempbase, ext = os.path.splitext(outfile)
        options = self.options

        if fmt is None:
            fmt = ext[1:]
            if fmt == 'xml':
                fmt = 'musicxml'

        if fmt not in ('musicxml', 'png', 'pdf'):
            raise ValueError(f"Format {fmt} unknown. Possible formats: musicxml, png, pdf")

        xmltext = self.render(options=options)
        trim = options.cropToContent
        if trim is None:
            trim = True if fmt == 'png' else False

        if fmt == 'musicxml':
            open(outfile, "w").write(xmltext)
        elif fmt == 'png' or fmt == 'pdf':
            musescorebin = options.musescoreBinary or _findMuseScore()
            if not musescorebin or not os.path.exists(musescorebin):
                msg = ("Could not find MuseScore. To solve this you can either make"
                       " sure that a binary 'musescore' is found in the PATH, or"
                       " you can customize the path within maelzel.core via"
                       " `getConfig()['musescorepath'] = '/path/to/musescore'`. To"
                       " install MuseScore follow the instructions here: "
                       "https://musescore.org/en/download")
                raise RuntimeError(msg)
            import tempfile
            musicxmlfile = tempfile.mktemp(suffix='.musicxml')
            open(musicxmlfile, "w").write(xmltext)
            if not outfile.endswith(fmt):
                raise ValueError(f"The outfile {outfile} does not match the format {fmt}")
            callMuseScore(musicxmlfile, outfile=outfile, musescorepath=musescorebin,
                          dpi=options.pngResolution, trim=trim)
            if removeTemporaryFiles:
                os.remove(musicxmlfile)
        else:
            raise ValueError(f"Format {fmt} not supported")


def callMuseScore(musicxmlfile: str,
                  outfile: str,
                  musescorepath: str,
                  dpi: int = 0,
                  trim=True,
                  forcetrim=True,
                  captureStderr=True
                  ) -> None:
    """
    Call MuseScore to render musicxml as pdf, png or svg

    Raises RuntimeError if MuseScore failed to generate the output file

    Args:
        musicxmlfile: the file to render
        outfile: the output file. The extension must match the output format
        musescorepath: the path to the MuseScore exe
        dpi: image resolution when generating PNG
        trim: ask MuseScreo to trim the image to its contents
        forcetrim: force trimming by doing it ourselves (uses pillow)

    """
    fmt = os.path.splitext(outfile)[1][1:]
    assert fmt in ('pdf', 'png', 'svg')
    args = [musescorepath, '-o', outfile, '--force']
    if dpi:
        args.extend(['--image-resolution', str(dpi)])
    if fmt == 'png' and trim:
        args.extend(['--trim-image', '20'])
    args.append(musicxmlfile)
    logger.debug(f"Rendering musicxml via MuseScore. Args: {args}")
    if os.path.exists(outfile):
        os.remove(outfile)
    subprocess.call(args, stderr=subprocess.PIPE if captureStderr else None)
    base = os.path.splitext(outfile)[0]
    if fmt == 'png':
        pattern = base + '-*.png'
        pagefiles = glob.glob(pattern)
        if not pagefiles:
            logger.error(f"callMusescore: output files not found. Pattern: {pattern}.\n"
                         f"Subprocess called with {args}")
            raise RuntimeError("MuseScore was called, but no files were generated as expected. "
                               f"Search pattern: {pattern}")
        pagefiles.sort()
        os.rename(pagefiles[0], outfile)
        for f in pagefiles[1:]:
            os.remove(f)
        if trim and forcetrim:
            from emlib import img
            img.cropToBoundingBox(outfile, margin=10)
    elif fmt == 'pdf' or fmt == 'svg':
        if not os.path.exists(outfile):
            logger.error(f"callMusescore: output files not found: '{outfile}'.\n"
                         f"Subprocess called with {args}")
            raise RuntimeError(f"Failed to generate output file ({outfile}) via MuseScore")
    else:
        raise ValueError(f"Format {fmt} not supported")
