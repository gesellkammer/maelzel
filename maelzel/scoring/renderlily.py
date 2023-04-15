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
from dataclasses import dataclass, field
import pitchtools as pt

import emlib.textlib
import emlib.filetools
from emlib import iterlib

from maelzel.music import lilytools
from .common import *
from . import attachment
# from .attachment import *
from . import definitions
from .core import Notation
from .render import Renderer, RenderOptions
from .node import Node
from . import quant, util
from . import spanner as _spanner
from . import lilypondsnippets

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .common import pitch_t


__all__ = (
    'LilypondRenderer',
    'quantizedPartToLily',
    'makeScore'
)


def lyNote(pitch: pitch_t, baseduration: int, dots: int = 0, tied=False, cautionary=False,
           fingering: str = ''
           ) -> str:
    assert baseduration in {0, 1, 2, 4, 8, 16, 32, 64, 128}, \
        f'baseduration should be a power of two, got {baseduration}'
    pitch = lilytools.makePitch(pitch, parenthesizeAccidental=cautionary)
    out = fr"{pitch}{baseduration}{'.'*dots}"
    if tied:
        out += '~'

    # This should be last
    if fingering:
        out += f'-{fingering}'

    return out


_articulationToLily = {
    'staccato': r'\staccato',
    'accent': r'\accent',
    'marcato': r'\marcato',
    'tenuto': r'\tenuto',
    'staccatissimo': r'\staccatissimo',
    'portato': r'\portato',
    'arpeggio': r'\arpeggio',
    'upbow': r'\upbow',
    'downbow': r'\downbow',
    'flageolet': r'\flageolet',
    'openstring': r'\open',
    'open': r'\open',
    'closed': r'\stopped',
    'stopped': r'\stopped',
    'snappizz': r'\snappizzicato',
    'snappizzicato': r'\snappizzicato',
    'laissezvibrer': r'\laissezVibrer'
}


_noteheadToLily = {
    'normal': '',
    'cross': 'cross',
    'harmonic': 'harmonic',
    'xcircle': 'xcircle',
    'triangle': 'triangle',
    'rhombus': 'harmonic-black',
    'square': 'la',
    'rectangle': 'la',
    'slash': 'slash',
    'diamond': 'diamond',
    'triangleup': 'ti',
    'do': 'do',
    're': 're',
    'mi': 'mi',
    'fa': 'fa',
    'sol': 'sol',
    'la': 'la',
    'ti': 'ti',
    'cluster': 'slash'

}


_fermataToLily = {
    'normal': r'\fermata',
    'square': r'\longfermata',
    'angled': r'\shortfermata',
    'double-square': r'\verylongfermata',
    'double-angled': r'\veryshortfermata'
}


_placementToLily = {
    '': '',
    'above': '^',
    'below': '_'
}


_linetypeToLily = {
    'solid': 'solid-line',
    'dashed': 'dashed-line',
    'dotted': 'dotted-line',
    'zigzag': 'zigzag',
    'wavy': 'trill',
    'trill': 'trill'
}

_lilyBarlines = {
    'single': r'|',
    'double': r'||',
    'final': r'|.',
    'solid': r'.',
    'dashed': r'!',
    'dotted': r';',
    'tick': "'",
    'short': ",",
    'double-thin': "=",
    'double-heavy': '||',
    'none': '',
    'hidden': ''
}


def markConsecutiveGracenotes(root: Node) -> None:
    """
    Marks consecutive gracenotes by setting their 'graceGroup' attribute inplace

    This is needed in lilypond since groups of gracenotes need to
    be placed within curly brackets, like ``\grace { a8 e8 }``,
    but when we are rendering we are iterating recursively,
    so we need to look ahead before

    Args:
        root: the root of the tree to work on

    """
    graceGroupOpen = False
    for n0, n1 in iterlib.pairwise(root.recurse()):
        if n0.isRest or not n0.isGracenote:
            assert not graceGroupOpen
            continue
        if n1.isGracenote:
            if not graceGroupOpen:
                n0.setProperty(".graceGroup", "start")
                graceGroupOpen = True
            else:
                n0.setProperty(".graceGroup", "continue")
        else:
            if graceGroupOpen:
                n0.setProperty(".graceGroup", "stop")
                graceGroupOpen = False
    if graceGroupOpen:
        lastn = iterlib.first(root.recurse(reverse=True))
        assert lastn.isGracenote
        lastn.setProperty('.graceGroup', 'stop')


def lyArticulation(articulation: attachment.Articulation) -> str:
    # TODO: render articulation color if present
    return _articulationToLily[articulation.kind]


def lyNotehead(notehead: definitions.Notehead, insideChord=False) -> str:
    """
    Convert a scoring Notehead to its lilypond representation

    This uses ``\override`` so it can't be placed inside a chord

    Args:
        notehead: the noteshape. It can end with '?', in which case it will be
            parenthesized
        insideChord: is this notehead inside a chord?

    Returns:
        the lilypond code representing this notehead. This needs to be placed **before**
        the note/chord it will modify.

    """
    assert isinstance(notehead, definitions.Notehead), f"Expected a Notehead, got {notehead}"

    if notehead.hidden:
        return r"\once \hide NoteHead"

    parts = []

    if notehead.color:
        if insideChord:
            parts.append(fr'\tweak NoteHead.color "{notehead.color}"')
        else:
            parts.append(fr'\once \override NoteHead.color = "{notehead.color}"')

    if notehead.size is not None:
        relsize = lilytools.fontSizeFactorToRelativeSize(notehead.size)
        if insideChord:
            parts.append(fr'\tweak NoteHead.font-size #{relsize}')
        else:
            parts.append(fr'\once \override NoteHead.font-size = #{relsize}')

    if notehead.shape and notehead.shape != 'normal':
        lilynotehead = _noteheadToLily.get(notehead.shape)
        if not lilynotehead:
            raise ValueError(f'Unknown notehead shape: {notehead.shape}, '
                             f'possible noteheads: {_noteheadToLily.keys()}')
        #if insideChord:
        #    parts.append(fr"\tweak NoteHead.style #'{lilynotehead}")
        #else:
        #    parts.append(fr"\once \override NoteHead.style = #'{lilynotehead}")
        parts.append(fr"\tweak NoteHead.style #'{lilynotehead}")

    if notehead.parenthesis:
        parts.append(r'\parenthesize')

    return " ".join(parts)


@dataclass
class RenderState:
    measure: quant.QuantizedMeasure | None = None
    insideSlide: bool = False
    glissando: bool = False
    dynamic: str = ''
    insideGraceGroup: bool = False
    openSpanners: dict[str, _spanner.Spanner] = field(default_factory=dict)


def notationToLily(n: Notation, options: RenderOptions, state: RenderState) -> str:
    """
    Converts a Notation to its lilypond representation

    .. note::

        We do not take tuplets into consideration here,
        since they should be taken care of at a higher level
        (see renderNode)

    Args:
        n: the notation
        options: render options
        state: the render state

    Returns:
        the lilypond notation corresponding to n, as a string

    """
    assert n.noteheads is None or isinstance(n.noteheads, dict), f"{n=}"

    notatedDur = n.notatedDuration()
    base, dots = notatedDur.base, notatedDur.dots
    parts = []
    _ = parts.append

    if n.isRest or (len(n.pitches) == 1 and n.pitches[0] == 0):
        _("r" + str(base) + "."*dots)
        # A rest can have: dynamics, fermatas, ..
        if (not n.tiedPrev or options.articulationInsideTie) and n.dynamic:
            dyn = n.dynamic if not n.dynamic.endswith('!') else n.dynamic[:-1]
            _(fr"\{dyn}")
        for attach in n.attachments:
            if isinstance(attach, attachment.Text):
                _(lilytools.makeText(attach.text, placement=attach.placement,
                                     fontsize=attach.fontsize,
                                     italic=attach.isItalic(), bold=attach.isBold(),
                                     box=attach.box))
            elif isinstance(attach, attachment.Fermata):
                _(_fermataToLily.get(attach.kind, r'\fermata'))
            elif isinstance(attach, attachment.Clef):
                _(lilytools.makeClef(attach.kind))
            else:
                logger.warning(f"Attachment {attach} not supported for rests")

        return ' '.join(parts)

    if n.color:
        # apply color to notehead, stem and ard accidental
        _(fr'\once \override Beam.color = "{n.color}" '
          fr'\once \override Stem.color = "{n.color}" '
          fr'\once \override Accidental.color = "{n.color}" '
          fr'\once \override Flag.color = "{n.color}" ' 
          fr'\once \override NoteHead.color = "{n.color}"')

    #if n.properties:
    #    if color := n.properties.get('noteheadColor'):
    #        _(fr'\once \override NoteHead.color = "{color}"')
    #    if sizeFactor := n.properties.get('noteheadSizeFactor'):
    #        relsize = lilytools.fontSizeFactorToRelativeSize(sizeFactor)
    #        _(fr'\once \override NoteHead.font-size =#{relsize}')

    if n.sizeFactor is not None and n.sizeFactor != 1:
        _(rf"\once \magnifyMusic {n.sizeFactor}")

    if n.stem == 'hidden':
        _(r"\once \override Stem.transparent = ##t")

    if n.isGracenote:
        base, dots = 8, 0
        graceGroupProp = n.getProperty('.graceGroup')
        if graceGroupProp == 'start':
            assert not state.insideGraceGroup
            _(r"\grace {")
            state.insideGraceGroup = True
        elif graceGroupProp == 'continue' or graceGroupProp == 'stop':

            assert state.insideGraceGroup, f"{n=}"
            pass
        else:
            assert not graceGroupProp and not state.insideGraceGroup
            _(r"\grace")

    # Attachments PRE pitch
    for attach in n.attachments:

        if isinstance(attach, attachment.Harmonic):
            if attach.interval == 0:
                n = n.copy()
                n.setNotehead('harmonic')
            elif len(n.pitches) > 1:
                logger.error("Cannot set a chord as artificial harmonic yet")
            else:
                fund = n.notename(0)
                touched = pt.transpose(fund, attach.interval)
                n = n.clone(pitches=(fund, touched))
                n.fixNotename(touched, idx=1)
                n.setNotehead('harmonic', idx=1)
        elif isinstance(attach, attachment.Clef):
            _(lilytools.makeClef(attach.kind))
        elif isinstance(attach, attachment.Breath):
            if attach.visible:
                if attach.kind:
                    logger.info("Setting breath type is not supported yet")
                    # _(fr"\once \set breathMarkType = #'{attach.kind}")
                _(r"\breathe")
            else:
                _(r'\beamBreak')
        elif isinstance(attach, attachment.Property) and attach.key == 'hidden':
            _("\single \hideNotes")


    if len(n.pitches) == 1:
        # ***************************
        # ********** Note ***********
        # ***************************
        if n.noteheads and (notehead:=n.noteheads.get(0)) is not None:
            _(lyNotehead(notehead))
        elif n.tiedPrev and n.gliss and state.glissando and options.glissHideTiedNotes:
            _(lyNotehead(definitions.Notehead(hidden=True)))

        fingering = next((a for a in n.attachments if isinstance(a, attachment.Fingering)), None)
        accidentalTraits = n.findAttachment(attachment.AccidentalTraits)
        if accidentalTraits:
            assert isinstance(accidentalTraits, attachment.AccidentalTraits)
        else:
            accidentalTraits = attachment.AccidentalTraits.default()

        if accidentalTraits.color:
            _(fr'\once \override Accidental.color "{accidentalTraits.color}"')

        _(lyNote(n.notename(),
                 baseduration=base,
                 dots=dots,
                 tied=n.tiedNext,
                 cautionary=accidentalTraits.parenthesis,
                 fingering=fingering.fingering if fingering else ''))
    else:
        # ***************************
        # ********** Chord **********
        # ***************************
        if n.tiedPrev and n.gliss and state.glissando and options.glissHideTiedNotes:
            _(lyNotehead(definitions.Notehead(hidden=True)))
            noteheads = None
        elif not n.noteheads:
            noteheads = None
        elif n.noteheads and set(n.noteheads.keys()) == set(range(len(n.pitches))) and len(set(n.noteheads.values())) == 1:
            # All the same noteheads, place it outside the chord
            _(lyNotehead(n.noteheads[0]))
            noteheads = None
        else:
            noteheads = n.noteheads
        _("<")
        notenames = n.resolveNotenames()
        notatedpitches = [pt.notated_pitch(notename) for notename in notenames]
        chordAccidentalTraits = n.findAttachment(cls=attachment.AccidentalTraits, anchor=None) or attachment.AccidentalTraits.default()
        for i, pitch in enumerate(n.pitches):
            if noteheads and (notehead := noteheads.get(i)) is not None:
                _(lyNotehead(notehead, insideChord=True))
            notename = notenames[i]
            notatedpitch = notatedpitches[i]
            accidentalTraits = n.findAttachment(cls=attachment.AccidentalTraits, anchor=i) or chordAccidentalTraits

            if any(otherpitch.diatonic_index == notatedpitch.diatonic_index for otherpitch in notatedpitches
                   if otherpitch is not notatedpitch):
                forceAccidental = True
            else:
                forceAccidental = accidentalTraits.force

            if accidentalTraits.hidden:
                _("\once\omit Accidental")
            if accidentalTraits.color:
                _(fr'\tweak Accidental.color "{accidentalTraits.color}"')

            _(lilytools.makePitch(notename,
                                  parenthesizeAccidental=accidentalTraits.parenthesis,
                                  forceAccidental=forceAccidental))
        _(f">{base}{'.'*dots}{'~' if n.tiedNext else ''}")

    if trem := next((a for a in n.attachments if isinstance(a, attachment.Tremolo)), None):
        if trem.tremtype == 'single':
            _(f":{trem.singleDuration()}")
        else:
            # TODO: render this correctly as two note tremolo
            _(f":{trem.singleDuration()}")

    if (not n.tiedPrev or options.articulationInsideTie) and n.dynamic:
        dyn = n.dynamic if not n.dynamic.endswith('!') else n.dynamic[:-1]
        _(fr"\{dyn}")

    if n.attachments:
        n.attachments.sort(key=lambda a: a.getPriority())
    for attach in n.attachments:
        if isinstance(attach, attachment.Text):
            _(lilytools.makeText(attach.text, placement=attach.placement,
                                 fontsize=attach.fontsize,
                                 italic=attach.isItalic(), bold=attach.isBold(),
                                 box=attach.box))
        elif isinstance(attach, attachment.Articulation):
            if not n.tiedPrev or options.articulationInsideTie:
                _(lyArticulation(attach))
        elif isinstance(attach, attachment.Fermata):
            _(_fermataToLily.get(attach.kind, r'\fermata'))
        elif isinstance(attach, attachment.Ornament):
            _(fr'\{attach.kind}')
        elif isinstance(attach, attachment.Bend):
            interval = ('+' if attach.interval > 0 else '')+str(round(attach.interval, 1))
            _(fr'\bendAfter #{interval}')


    if options.showCents:
        # TODO: cents annotation should follow options (below/above, fontsize)
        centsText = util.centsAnnotation(n.pitches, divsPerSemitone=options.divsPerSemitone)
        if centsText:
            fontrelsize = options.centsFontSize - options.staffSize
            _(lilytools.makeText(centsText, fontsize=fontrelsize,
                                 fontrelative=True, placement='below'))

    #if n.isGracenote:
    #    if state.insideGraceGroup and n.getProperty('graceGroup') == 'stop':
    #        _("}")
    #        state.insideGraceGroup = False
    #else:
    #    state.insideGraceGroup = False

    return " ".join(parts)


_spaces = " " * 1000


def _handleSpannerPre(spanner: _spanner.Spanner, state: RenderState) -> str | None:
    """
    Generates lilypond text for spanners which either need to be placed before the note or
    whose customizations need to be placed before the note

    Args:
        spanner: the spanner to generate code for

    Returns:
        the generated lilypond text or None
    """
    out = []
    _ = out.append
    if isinstance(spanner, _spanner.Slur):
        if spanner.kind == 'start':
            _(fr' \slur{spanner.linetype.capitalize()} ')

    elif isinstance(spanner, _spanner.OctaveShift):
        if spanner.kind == 'start':
            _(rf"\ottava #{spanner.octaves} ")
        else:
            _(rf"\ottava #0 ")

    elif isinstance(spanner, _spanner.Bracket):
        if spanner.kind == 'start' and spanner.linetype != 'solid':
            style = _linetypeToLily[spanner.linetype]
            _(rf" \once \override HorizontalBracket #'style = #'{style} ")

    elif isinstance(spanner, _spanner.LineSpan) and spanner.kind == 'start':
        y = 1 if spanner.placement == 'below' else -1
        markup = ''
        if spanner.starthook:
            markup += f"\\draw-line  #'(0 . {y}) "
        if spanner.starttext:
            if spanner.verticalAlign:
                _(rf'\once \override TextSpanner.bound-details.left.stencil-align-dir-y = #{spanner.verticalAlign.upper()} ')
            markup += f' \\upright "{spanner.starttext}"'
        if markup:
            _(rf'\once \override TextSpanner.bound-details.left.text = \markup {{ {markup} }} ')
        if spanner.endtext:
            if spanner.verticalAlign:
                _(rf'\once \override TextSpanner.bound-details.right.stencil-align-dir-y = #{spanner.verticalAlign.upper()} ')
            _(rf'\once \override TextSpanner.bound-details.right.text = \markup {{ \upright "{spanner.endtext}" }} ')
        elif spanner.endhook:
            _(rf"\once \override TextSpanner.bound-details.right.text = \markup {{ \draw-line #'(0 . {y}) }} ")
        _(rf"\once \override TextSpanner.style = #'{_linetypeToLily[spanner.linetype]} ")
        # TODO: endtext, middletext

    elif isinstance(spanner, _spanner.TrillLine):
        if spanner.kind == 'start':
            if not (spanner.startmark or spanner.alteration or spanner.trillpitch):
                # It will be just a wavy line, so we use a textspanner
                _(rf"\once \override TextSpanner.style = #'trill ")
            elif spanner.trillpitch:
                _(r'\pitchedTrill ')

    elif isinstance(spanner, _spanner.Slide):
        if spanner.kind == 'start':
            if spanner.linetype != 'solid':
                _(rf"\once \override Glissando.style = #'{_linetypeToLily[spanner.linetype]} ")
        else:
            if state.insideSlide and not state.glissando:
                _(r"\glissandoSkipOff ")
                state.insideSlide = False

    return ''.join(out)


def _handleSpannerPost(spanner: _spanner.Spanner, state: RenderState) -> str | None:
    out = []
    _ = out.append

    if spanner.lilyPlacementPost and spanner.kind == 'start' and spanner.placement:
        _(_placementToLily.get(spanner.placement))

    if isinstance(spanner, _spanner.Slur):
        _("(" if spanner.kind == 'start' else ")")

    elif isinstance(spanner, _spanner.Beam):
        _("[" if spanner.kind == 'start' else "]")

    elif isinstance(spanner, _spanner.Hairpin):
        if spanner.kind == 'start':
            if spanner.niente:
                _(r"\once \override Hairpin.circled-tip = ##t ")
            _(fr" \{spanner.direction} ")
        elif spanner.kind == 'end':
            _(r" \! ")

    elif isinstance(spanner, _spanner.Bracket):
        if spanner.kind == 'start':
            if spanner.text:
                _(rf' -\tweak HorizontalBracketText.text "{spanner.text}" ')
            _(r"\startGroup ")
        else:
            _(r"\stopGroup ")

    elif isinstance(spanner, _spanner.TrillLine):
        # If it has a start mark we use a trill line, otherwise we use a textspan
        if spanner.startmark == 'trill':
            if spanner.kind == 'start':
                _(r'\startTrillSpan ')
                if spanner.trillpitch:
                    _(lilytools.makePitch(spanner.trillpitch) + " ")
            else:
                _(r'\stopTrillSpan ')
        else:
            # just a wavy line, the line type should have been customized
            # in the pre-phase
            _(r'\startTextSpan ' if spanner.kind == 'start' else r'\stopTextSpan ')

    elif isinstance(spanner, _spanner.LineSpan):
        _(r'\startTextSpan ' if spanner.kind == 'start' else r'\stopTextSpan ')

    elif isinstance(spanner, _spanner.Slide):
        if not state.glissando and spanner.kind == 'start':
            _(r"\glissando \glissandoSkipOn ")
            state.insideSlide = True

    return ''.join(out)


def _forceBracketsForNestedTuplets(node: Node):
    if node.durRatio != (1, 1):
        for item in node.items:
            if isinstance(item, Node) and item.durRatio != (1, 1):
                item.setProperty('.forceTupletBracket', True)
                node.setProperty('.forceTupletBracket', True)
    for item in node.items:
        if isinstance(item, Node):
            _forceBracketsForNestedTuplets(item)


def renderNode(node: Node,
               durRatios: list[F],
               options: RenderOptions,
               state: RenderState,
               numIndents: int = 0,
               ) -> str:
    """
    A node is a sequence of notes which share (and fill) a time modifier.
    It can be understood as a "subdivision", whereas "normal" durations are interpreted
    as a 1:1 subdivision. A node can consist of Notations or other Nodes

    Args:
        node: the node to render
        durRatios: a seq. of duration ratios OUTSIDE this node. Can be
            an empty list
        options: the render options to use
        state: context of the ongoing render
        numIndents: number of indents for the generated code.
    """
    seq = []
    _ = seq.append
    indentSize = 2

    if node.durRatio != (1, 1):
        durRatios.append(F(*node.durRatio))
        tupletStarted = True
        num, den = node.durRatio
        _(_spaces[:numIndents*indentSize])
        _(f"\\tuplet {num}/{den} {{\n")
        numIndents += 1
        if node.getProperty('.forceTupletBracket'):
            _(_spaces[:numIndents * indentSize])
            _(r"\once \override TupletBracket.bracket-visibility = ##t")

    else:
        tupletStarted = False
    _(_spaces[:numIndents*indentSize])
    for i, item in enumerate(node.items):
        if isinstance(item, Node):
            _(renderNode(item, durRatios, options=options, numIndents=numIndents + 1,
                         state=state))
        else:
            assert isinstance(item, Notation)
            if not item.gliss and state.glissando:
                _(r"\glissandoSkipOff ")
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

            #if item.offset in beatOffsets[1:-1]:
            #    _(r'\beamBreak ')

            # Slur modifiers (line type, etc.) need to go before the start of
            # the first note of the spanner :-(
            # Some spanners have customizations which need to be declared
            # before the note to which the spanner is attached to
            if item.spanners:
                item.spanners.sort(key=lambda spanner: spanner.priority())

                for spanner in item.spanners:
                    if lilytext := _handleSpannerPre(spanner, state=state):
                        _(lilytext)

            _(notationToLily(item, options=options, state=state))

            if item.isGracenote:
                if state.insideGraceGroup and item.getProperty('.graceGroup') == 'stop':
                    _("}")
                    state.insideGraceGroup = False
            else:
                state.insideGraceGroup = False


            # * If the item has a glissando, add \glissando
            #   * If it is tied, add glissandoSkipOn IF not already on
            #   * If not tied, turn off skip if already on
            # * else (no gliss): turn off skip if on

            if item.gliss:
                if not state.glissando:
                    _(r"\glissando ")
                if item.tiedNext:
                    if not state.glissando:
                        state.glissando = True
                        _(r"\glissandoSkipOn ")
                else:
                    if state.glissando:
                        state.glissando = False
                        _(r"\glissandoSkipOff ")
            else:
                if state.glissando:
                    _(r"\glissandoSkipOff ")
                    state.glissando = False

            if item.spanners:
                for spanner in item.spanners:
                    if lilytext := _handleSpannerPost(spanner, state=state):
                        _(lilytext)


            _(" ")



    _("\n")
    if tupletStarted:
        numIndents -= 1
        _(_spaces[:numIndents*indentSize])
        _("}\n")
    return ''.join(seq)


def quantizedPartToLily(part: quant.QuantizedPart,
                        options: RenderOptions,
                        addMeasureMarks=True,
                        clef: str = None,
                        addTempoMarks=True,
                        indents=0,
                        indentSize=2,
                        ) -> str:
    """
    Convert a QuantizedPart to lilypond code

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

    Returns:
        the rendered lilypond code

    """
    quarterTempo = 0
    scorestruct = part.struct

    seq = []

    def _(t: str, indents: int = 0, preln=False, postln=False):
        if preln and seq and seq[-1][-1] != "\n":
            seq.append("\n")
        if indents:
            t = textwrap.dedent(t)
            t = textwrap.indent(t, prefix=_spaces[:indents*indentSize])
            # seq.append(_spaces[:indents*indentSize])
        seq.append(t)
        if postln:
            seq.append("\n")

    def line(t: str, indents: int = 0):
        _(t, indents, preln=True, postln=True)

    if part.name:
        line(r"\new Staff \with {", indents)
        line(f'    instrumentName = #"{part.name}"', indents)
        if part.shortname:
            line(f'    shortInstrumentName = "{part.shortname}"', indents)
        line("}", indents)
        line("{", indents)
    else:
        line(r"\new Staff {", indents)

    indents += 1
    line(r"\numericTimeSignature", indents)

    if not clef:
        clef = part.bestClef()
    line(lilytools.makeClef(clef), indents)

    lastTimesig = None

    state = RenderState()

    for i, measure in enumerate(part.measures):
        # Start measure
        # Reset state
        state.measure = measure

        line(f"% measure {i+1}", indents)
        indents += 1
        measureDef = scorestruct.getMeasureDef(i)

        if addTempoMarks and measureDef.timesig != lastTimesig:
            lastTimesig = measureDef.timesig
            num, den = measureDef.timesig
            if symbol := measureDef.properties.get('symbol'):
                if symbol == 'single-number':
                    line(r"\once \override Staff.TimeSignature.style = #'single-digit")
            if measureDef.subdivisionStructure:
                # compound meter, like 3+2 / 8
                # \compoundMeter #'((2 2 2 8))
                parts = ' '.join(str(part) for part in measureDef.subdivisionStructure)
                line(fr"\compoundMeter #'(({parts} {den}))")
            else:
                line(fr"\time {num}/{den}", indents)

        if addTempoMarks and measure.quarterTempo != quarterTempo:
            quarterTempo = measure.quarterTempo
            # lilypond only support integer tempi
            # TODO: convert to a different base if the tempo is too slow/fast for
            #       the quarter, or convert according to the time signature
            line(fr"\tempo 4 = {int(quarterTempo)}", indents)

        if measureDef.keySignature:
            line(lilytools.keySignature(fifths=measureDef.keySignature.fifths,
                                        mode=measureDef.keySignature.mode))

        if addMeasureMarks:
            if measureDef.annotation:
                relfontsize = options.measureAnnotationFontSize - options.staffSize
                _(lilytools.makeTextMark(measureDef.annotation,
                                         fontsize=relfontsize, fontrelative=True,
                                         box=options.measureAnnotationBox))
            if measureDef.rehearsalMark:
                relfontsize = options.rehearsalMarkFontSize - options.staffSize
                box = measureDef.rehearsalMark.box if options.rehearsalMarkBoxed else ''
                _(lilytools.makeTextMark(measureDef.rehearsalMark.text,
                                         fontsize=relfontsize, fontrelative=True,
                                         box=box))

        if measure.isEmpty():
            num, den = measure.timesig
            measureDur = float(util.measureQuarterDuration(measure.timesig))
            if measureDur in {1., 2., 3., 4., 6., 7., 8.}:
                lilydur = lilytools.makeDuration(measureDur)
                _(f"R{lilydur}")
            else:
                _(f"R1*{num}/{den}")
            state.dynamic = ''
        else:
            root = measure.tree()
            _forceBracketsForNestedTuplets(root)
            markConsecutiveGracenotes(root)
            lilytext = renderNode(root, durRatios=[], options=options,
                                  numIndents=0, state=state)
            _(lilytext, indents=indents)
        indents -= 1

        if not measureDef.barline or measureDef.barline == 'single':
            line(f"|   % end measure {i+1}", indents)
        else:
            if (barstyle := _lilyBarlines.get(measureDef.barline)) is None:
                logger.error(f"Barstile '{measureDef.barline}' unknown. "
                             f"Supported styles: {_lilyBarlines.keys()}")
                barstyle = '|'
            line(rf'\bar "{barstyle}"    |  % end measure {i+1}', indents)

    indents -= 1

    line(f"}}   % end staff {part.name}", indents)
    return "".join(seq)


# --------------------------------------------------------------

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
    IND = " " * indentSize
    numMeasures = max(len(part.measures)
                      for part in score.parts)
    struct = score.scorestruct.copy()
    struct.setBarline(numMeasures - 1, 'final')
    score.scorestruct = struct

    strs = []

    def _(s, dedent=False, indent=0):
        if dedent:
            s = textwrap.dedent(s)
        elif indent:
            s = textwrap.dedent(s)
            s = textwrap.indent(s, prefix=IND * indent)
        strs.append(s)

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

    _(f'#(set-global-staff-size {staffSizePoints})')

    if options.cropToContent:
        _(r'\include "lilypond-book-preamble.ly"')
        # TODO: add configuration for this
        _(lilytools.paperBlock(margin=20, unit="mm"))
    else:
        # We only set the paper size if rendering to pdf
        _(f"#(set-default-paper-size \"{options.pageSize}\" '{options.orientation})")
        _(lilytools.paperBlock(margin=options.pageMarginMillimeters, unit="mm"))

    _(lilypondsnippets.prelude)

    if options.glissLineThickness != 1:
        _(r"""
        \layout {
          \context { 
            \Voice
            \override Glissando #'thickness = #%d
            \override Glissando #'gap = #0.05
          }
        }
        """ % options.glissLineThickness, dedent=True)

    if options.horizontalSpacing:
        spacingPreset = lilypondsnippets.horizontalSpacingPresets[options.horizontalSpacing]
        if spacingPreset:
            _(spacingPreset)

    if options.lilypondGlissandoMinimumLength:
        _(lilypondsnippets.glissandoMinimumLength(options.lilypondGlissandoMinimumLength))

    _(r"\score {")
    _(r"<<")
    indents = 1
    groups = score.groupParts()
    partindex = 0
    for group in groups:
        if isinstance(group, quant.QuantizedPart):
            group = [group]
        if len(group) > 1:
            _(r"\new StaffGroup <<", indent=1)
            indents += 1
        for part in group:
            partstr = quantizedPartToLily(part,
                                          addMeasureMarks=partindex == 0,
                                          addTempoMarks=partindex == 0,
                                          options=options,
                                          indents=indents,
                                          indentSize=indentSize,
                                          clef=part.firstclef)
            _(partstr)
            partindex += 1
        if len(group) > 1:
            _(rf">>", indent=1)
            indents -= 1

    _(r">>")
    if midi:
        _(" "*indentSize + r"\midi { }")
    _(r"}   % end score")  # end \score
    return "\n".join(strs)


class LilypondRenderer(Renderer):
    def __init__(self, score: quant.QuantizedScore, options: RenderOptions):
        super().__init__(score, options=options)
        self._lilyscore = ''
        self._withMidi = False

    def render(self) -> None:
        if self._rendered:
            return
        self._lilyscore = makeScore(self.quantizedScore, options=self.options, midi=self._withMidi)
        self._rendered = True

    def writeFormats(self) -> list[str]:
        return ['pdf', 'ly', 'png']

    def write(self, outfile: str, fmt: str = None, removeTemporaryFiles=False) -> None:
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
            raise ValueError(f"Format {fmt} unknown. Possible formats: png, pdf, mid, ly")
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
