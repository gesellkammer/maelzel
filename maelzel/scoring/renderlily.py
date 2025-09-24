"""
This module implements a lilypond renderer

It converts our own intermediate representation as defined after quantization
into a .ly file and renders that via lilypond to pdf or png
"""
from __future__ import annotations

import os
import shutil
import textwrap
import math
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING
from itertools import pairwise

import emlib.filetools
import emlib.mathlib
import emlib.textlib
import pitchtools as pt

from emlib.iterlib import first

from maelzel import _imgtools, _util
from maelzel._indentedwriter import IndentedWriter
from maelzel.common import F, asF
from maelzel.music import lilytools
from maelzel.scoring.common import logger
from maelzel.textstyle import TextStyle

from . import attachment, definitions, lilypondsnippets, quant, util
from . import spanner as _spanner
from .core import Notation
from .node import Node
from .render import Renderer, RenderOptions

if TYPE_CHECKING:
    from maelzel.common import pitch_t


__all__ = (
    'LilypondRenderer',
    'renderPart',
    'makeScore'
)


def lyNote(pitch: pitch_t,
           baseduration: int,
           dots: int = 0,
           tied=False,
           cautionary=False,
           fingering: str = '',
           forceAccidental=False
           ) -> str:
    if not emlib.mathlib.ispowerof2(baseduration):
        raise ValueError(f'baseduration should be a power of two, got {baseduration}')
    pitch = lilytools.makePitch(pitch, parenthesizeAccidental=cautionary, forceAccidental=forceAccidental)
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
    r"""
    Marks consecutive gracenotes by setting their 'graceGroup' attribute inplace

    This is needed in lilypond since groups of gracenotes need to
    be placed within curly brackets, like ``\grace { a8 e8 }``,
    but when we are rendering we are iterating recursively,
    so we need to look ahead before

    Args:
        root: the root of the tree to work on

    """
    graceGroupOpen = False
    for n0, n1 in pairwise(root.recurse()):
        if n0.isRest or not n0.isGracenote:
            assert not graceGroupOpen
            continue
        assert n0.isGracenote
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
        if lastn := next(root.recurse(reverse=True), None):
            assert lastn.isGracenote
            lastn.setProperty('.graceGroup', 'stop')


def lyArticulation(articulation: attachment.Articulation) -> str:
    """
    Convert a scoring Articulation to its lilypond representation

    Args:
        articulation: the articulation to convert

    Returns:
        the lilypond code representing this articulation. This needs to be placed **before**
        the note/chord it will modify.

    """
    # TODO: render articulation color if present
    name = _articulationToLily[articulation.kind]
    parts = []
    if articulation.color:
        parts.append(rf'\tweak color "{articulation.color}"')
    parts.append(name)
    return " ".join(parts)


def lyNotehead(notehead: definitions.Notehead, insideChord=False) -> str:
    r"""
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


def _renderTextAttachment(attach: attachment.Text,
                          options: RenderOptions,
                          ) -> str:
    if not attach.role:
        return lilytools.makeText(text=attach.text,
                                  fontrelative=attach.relativeSize,
                                  fontsize=attach.fontsize,
                                  placement=attach.placement or 'above',
                                  italic=attach.italic,
                                  bold=attach.weight=='bold',
                                  box=attach.box)
    elif attach.role == 'label':
        style = TextStyle.parse(options.noteLabelStyle)
        return lilytools.makeText(text=attach.text,
                                  fontrelative=attach.relativeSize,
                                  fontsize=attach.fontsize or   style.fontsize or None,
                                  placement=attach.placement or style.placement or 'above',
                                  italic=attach.italic or style.italic,
                                  bold=attach.weight=='bold' or style.bold,
                                  box=attach.box or style.box)
    else:
        raise ValueError(f"Text role {attach.role} not implemented")


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

    # Attachments pre everything
    if n.attachments:
        for attach in n.attachments:
            if isinstance(attach, attachment.Clef):
                _(lilytools.makeClef(attach.kind))

    if n.isRest or (len(n.pitches) == 1 and n.pitches[0] == 0):
        # ******************************
        # **          Rest            **
        # ******************************
        _("r" + str(base) + "."*dots)
        # A rest can have: dynamics, fermatas, ..
        if (not n.tiedPrev or options.articulationInsideTie) and n.dynamic:
            dyn = n.dynamic if not n.dynamic.endswith('!') else n.dynamic[:-1]
            _(fr"\{dyn}")
        if n.attachments:
            for attach in n.attachments:
                if isinstance(attach, attachment.Text):
                    _(_renderTextAttachment(attach, options=options))
                elif isinstance(attach, attachment.Fermata):
                    _(_fermataToLily.get(attach.kind, r'\fermata'))
                elif isinstance(attach, attachment.Clef):
                    _(lilytools.makeClef(attach.kind))
                elif isinstance(attach, attachment.Breath):
                    if attach.horizontalPlacement == 'post':
                        if attach.visible:
                            if attach.kind:
                                logger.info("Setting breath type is not supported yet")
                                # _(fr"\once \set breathMarkType = #'{attach.kind}")
                            _(r"\breathe")
                        else:
                            _(r'\beamBreak')
                else:
                    logger.warning(f"Attachment {attach} not supported for rests")
        return ' '.join(parts).strip()

    if n.attachments:
        for attach in n.attachments:
            if isinstance(attach, attachment.Color):
                color = attach.color
                # apply color to notehead, stem and ard accidental
                _(fr'\once \override Beam.color = "{color}" '
                  fr'\once \override Stem.color = "{color}" '
                  fr'\once \override Accidental.color = "{color}" '
                  fr'\once \override Flag.color = "{color}" '
                  fr'\once \override NoteHead.color = "{color}"')
            elif isinstance(attach, attachment.SizeFactor):
                size = attach.size
                if size != 1:
                    _(rf"\once \magnifyMusic {size}")
            elif isinstance(attach, attachment.StemTraits):
                if attach.hidden:
                    _(r"\once \override Stem.transparent = ##t")
                elif attach.color:
                    _(rf'\once \override Stem.color = "{attach.color}" ')

    # Attachments PRE pitch
    if n.attachments:
        for attach in n.attachments:
            if isinstance(attach, attachment.Harmonic):
                n = n.resolveHarmonic()
            elif isinstance(attach, attachment.Breath) and attach.horizontalPlacement == 'pre':
                if attach.visible:
                    if attach.kind:
                        logger.info("Setting breath type is not supported yet")
                        # _(fr"\once \set breathMarkType = #'{attach.kind}")
                    _(r"\breathe")
                else:
                    _(r'\beamBreak')
            elif isinstance(attach, attachment.Hidden):
                _(r"\single \hideNotes")

    if n.isGracenote:
        dots = 0
        if n.attachments and (props:=n.findAttachment(attachment.GracenoteProperties)) is not None:
            base = 4 // props.value
            slashed = props.slash
        else:
            base = 8
            slashed = False
        lilytoken = r"\grace" if not slashed else r"\slashedGrace"
        graceGroupProp = n.getProperty('.graceGroup')
        if not graceGroupProp:
            assert not state.insideGraceGroup
            _(lilytoken)
        if graceGroupProp == 'start':
            assert not state.insideGraceGroup
            _(lilytoken + "{" )
            state.insideGraceGroup = True
        elif graceGroupProp == 'continue' or graceGroupProp == 'stop':
            assert state.insideGraceGroup, f"{n=}"
            pass


    if len(n.pitches) == 1:
        # ***************************
        # **         Note          **
        # ***************************
        if notehead := n.getNotehead(0):
            _(lyNotehead(notehead))
        elif n.tiedPrev and n.gliss and state.glissando and options.glissHideTiedNotes:
            _(lyNotehead(definitions.Notehead(hidden=True)))

        fingering = first(a for a in n.attachments if isinstance(a, attachment.Fingering)) if n.attachments else None
        parenthesis = False
        if not n.tiedPrev:
            if accidentalTraits := n.findAttachment(attachment.AccidentalTraits):
                parenthesis = accidentalTraits.parenthesis
                if accidentalTraits.color:
                    _(fr'\once \override Accidental.color = "{accidentalTraits.color}"')

        _(lyNote(n.notename(),
                 baseduration=base,
                 dots=dots,
                 tied=n.tiedNext,
                 cautionary=parenthesis,
                 fingering=fingering.fingering if fingering else ''))
    else:
        # ***************************
        # **         Chord         **
        # ***************************

        # TODO: accidentals
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
        notenames = n.resolveNotenames(removeFixedAnnotation=True)
        notatedpitches = [pt.notated_pitch(notename) for notename in notenames]
        chordAccidentalTraits = n.findAttachment(cls=attachment.AccidentalTraits, pitchanchor=None) or attachment.AccidentalTraits.default()
        backties = n.tieHints('backward') if n.tiedPrev else None
        for i, pitch in enumerate(n.pitches):
            if noteheads and (notehead := noteheads.get(i)) is not None:
                _(lyNotehead(notehead, insideChord=True))
            notename = notenames[i]
            notatedpitch = notatedpitches[i]

            accidentalTraits = n.findAttachment(cls=attachment.AccidentalTraits, pitchanchor=i) or chordAccidentalTraits

            if accidentalTraits.hidden:
                _(r"\once\omit Accidental")
            if accidentalTraits.color:
                _(fr'\tweak Accidental.color "{accidentalTraits.color}"')

            forceAccidental = accidentalTraits.force
            if n.tiedPrev:
                assert backties is not None
                if i in backties:
                    forceAccidental = False
            elif any(otherpitch.chromatic_index == notatedpitch.chromatic_index and
                     otherpitch.diatonic_name != notatedpitch.diatonic_name
                     for otherpitch in notatedpitches):
                forceAccidental = True

            _(lilytools.makePitch(notename,
                                  parenthesizeAccidental=accidentalTraits.parenthesis,
                                  forceAccidental=forceAccidental))

        _(f">{base}{'.'*dots}{'~' if n.tiedNext else ''}")

    if trem := n.findAttachment(attachment.Tremolo):
        if trem.color:
            _(rf'\once \override StemTremolo.color = #"{trem.color}"')
        if trem.tremtype == 'single':
            if trem.relative:
                _(f":{trem.singleDuration()}")
            else:
                nummarksbase = int(math.log(base, 2) - 2)
                assert nummarksbase > 0
                # 8: 1, 16: 2, 32: 3, 64: 4, ...
                tremdur = 2 ** (2 + nummarksbase + trem.nummarks)
                _(f":{tremdur}")
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
                _(_renderTextAttachment(attach, options=options))
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
            elif isinstance(attach, attachment.Breath) and attach.horizontalPlacement == 'post':
                if attach.visible:
                    if attach.kind:
                        logger.info("Setting breath type is not supported yet")
                        # _(fr"\once \set breathMarkType = #'{attach.kind}")
                    _(r"\breathe")
                else:
                    _(r'\beamBreak')

    if options.showCents and not n.tiedPrev:
        # TODO: cents annotation should follow options (below/above, fontsize)
        if text := util.centsAnnotation(n.pitches,
                                        divsPerSemitone=options.divsPerSemitone,
                                        addplus=options.centsTextPlusSign,
                                        separator=options.centsAnnotationSeparator,
                                        snap=options.centsTextSnap):
            fontrelsize = options.centsAnnotationFontsize - options.staffSize
            _(lilytools.makeText(text,
                                 fontsize=fontrelsize,
                                 fontrelative=True,
                                 placement=options.centsAnnotationPlacement))

    out = " ".join(parts)
    return out.strip()


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
    # This handles only the linetype
    if isinstance(spanner, _spanner.Slur) and spanner.linetype != 'solid':
        if spanner.kind == 'start':
            if spanner.nestingLevel == 1:
                _(fr' \slur{spanner.linetype.capitalize()} ')
            elif spanner.nestingLevel == 2:
                _(fr' \phrasingSlur{spanner.linetype.capitalize()} ')

    elif isinstance(spanner, _spanner.OctaveShift):
        if spanner.kind == 'start':
            _(rf"\ottava #{spanner.octaves} ")
        else:
            _(r"\ottava #0 ")

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
                _(r"\once \override TextSpanner.style = #'trill ")
            elif spanner.trillpitch:
                _(r'\pitchedTrill ')

    elif isinstance(spanner, _spanner.Slide):
        if spanner.kind == 'start':
            if spanner.linetype != 'solid':
                _(rf"\once \override Glissando.style = #'{_linetypeToLily[spanner.linetype]} ")
            if spanner.color:
                _(rf'\once \override Glissando.color = "{spanner.color}"')
            # TODO: support the .text attribute of .Slide
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
        assert spanner.nestingLevel < 4
        if spanner.kind == 'start':
            t = "_(" if spanner.placement == 'below' else "("
        else:
            t = ")"
        _(f"\\={spanner.nestingLevel}{t}")

    elif isinstance(spanner, _spanner.Beam):
        if spanner.kind == 'start':
            t = "_[" if spanner.placement == 'below' else "["
        else:
            t = "]"
        _(t)

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
               numIndents=0,
               indentSize=2
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
        indentSize: the number of spaces per indent
    """
    w = IndentedWriter(indentsize=indentSize, indents=numIndents)

    if node.durRatio != (1, 1):
        # A new tuplet. Check if the node has any leading gracenotes, which need to
        # be rendered before the tuplet

        gracenotes: list[tuple[Notation, Node]] = []
        for n, parent in node.recurseWithNode():
            if n.isGracenote:
                gracenotes.append((n, parent))
            else:
                break
        if gracenotes:
            for n, parent in gracenotes:
                parent.items.remove(n)
        tempnode = Node([n for n, parent in gracenotes], ratio=(1, 1))
        txt = renderNode(tempnode, durRatios=durRatios.copy(), options=options, state=state,
                             numIndents=numIndents, indentSize=indentSize)
        w.add(txt)

        durRatios.append(F(*node.durRatio))
        tupletStarted = True
        num, den = node.durRatio

        w.line(f"\\tuplet {num}/{den} {{")
        w.indents += 1
        if node.getProperty('.forceTupletBracket'):
            w.line(r"\once \override TupletBracket.bracket-visibility = ##t")

    else:
        tupletStarted = False

    for i, item in enumerate(node.items):
        if isinstance(item, Node):
            nodetxt = renderNode(item, durRatios, options=options, numIndents=0,
                                 state=state, indentSize=w.indentsize)
            w.line(nodetxt)
            continue

        assert isinstance(item, Notation)
        item.checkIntegrity(fix=True)

        if att := item.findAttachment(attachment.BeamSubdivisionHint):
            s = r"\once " if att.once else ''
            w.line(fr"{s}\set subdivideBeams = ##t")
            if att.minimum:
                num, den = att.minimum.numerator, att.minimum.denominator
                w.line(rf"{s}\set beamMinimumSubdivision = #{num}/{den}")
            if att.maximum:
                num, den = att.maximum.numerator, att.maximum.denominator
                w.line(rf"{s}\set beamMaximumSubdivision = #{num}/{den}")

        if not item.gliss and state.glissando:
            w.add(r"\glissandoSkipOff ")
            state.glissando = False

        if item.isRest:
            state.dynamic = ''

        if item.dynamic:
            dynamic = item.dynamic
            if (options.removeRedundantDynamics and
                    not item.dynamic.endswith('!') and
                    item.dynamic == state.dynamic and
                    item.dynamic in definitions.dynamicLevels):
                item.dynamic = ''
            state.dynamic = dynamic

        # Slur modifiers (line type, etc.) need to go before the start of
        # the first note of the spanner :-(
        # Some spanners have customizations which need to be declared
        # before the note to which the spanner is attached to
        if item.spanners:
            item.spanners.sort(key=lambda spanner: spanner.priority())

            for spanner in item.spanners:
                if lilytext := _handleSpannerPre(spanner, state=state):
                    w.add(lilytext)

        w.add(notationToLily(item, options=options, state=state))

        if item.gliss:
            if not state.glissando:
                if props := item.findAttachment(attachment.GlissProperties):
                    assert isinstance(props, attachment.GlissProperties)
                    if props.linetype != 'solid':
                        w.line(rf"\tweak Glissando.style #'{_linetypeToLily[props.linetype]}")
                    if props.color:
                        w.add(rf'\tweak Glissando.color "{props.color}"')

                w.add(r"\glissando ")
            if item.tiedNext:
                if not state.glissando:
                    state.glissando = True
                    w.add(r"\glissandoSkipOn")
            else:
                if state.glissando:
                    state.glissando = False
                    w.add(r"\glissandoSkipOff")
        else:
            if state.glissando:
                w.add(r"\glissandoSkipOff")
                state.glissando = False

        if item.isGracenote:
            if state.insideGraceGroup and item.getProperty('.graceGroup') == 'stop':
                w.add("}")
                state.insideGraceGroup = False
        else:
            state.insideGraceGroup = False

        # * If the item has a glissando, add \glissando
        #   * If it is tied, add glissandoSkipOn IF not already on
        #   * If not tied, turn off skip if already on
        # * else (no gliss): turn off skip if on

        if item.spanners:
            for spanner in item.spanners:
                if lilytext := _handleSpannerPost(spanner, state=state):
                    w.add(lilytext)

    w.block()
    if tupletStarted:
        w.indents -= 1
        w.line("}")
    return w.join()


def _isSmallDenominator(den: int, quarterTempo: F, eighthNoteThreshold=50) -> bool:
    # 7/8 has small denominator at tempo q=50, but not at tempo q=25,
    if den <= 4:
        return False
    mintempo = eighthNoteThreshold * 8 / den
    return quarterTempo > mintempo


def renderPart(part: quant.QuantizedPart,
               options: RenderOptions,
               addMeasureMarks=True,
               clef='',
               addTempoMarks=True,
               indents=0,
               indentSize=2,
               ) -> str:
    """
    Render a QuantizedPart as lilypond code

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
    quarterTempo = F(0)
    scorestruct = part.struct
    w = IndentedWriter(indentsize=indentSize, indents=indents)

    if part.name and part.showName:
        w.line(r"\new Staff \with {")
        w.line(f'    instrumentName = #"{part.name}"')
        if part.shortName:
            w.line(f'    shortInstrumentName = "{part.shortName}"')
        if clef or part.firstClef:
            w.line('    ' + lilytools.makeClef(part.firstClef))

        w.line("}")
        w.line("{")
    else:
        w.line(r"\new Staff {")

    w.indents += 1
    w.line(r"\numericTimeSignature")

    if not clef:
        clef = part.firstClef or part.bestClef()
    w.line(lilytools.makeClef(clef))

    timesig = None

    state = RenderState()

    for i, measure in enumerate(part.measures):
        assert isinstance(measure, quant.QuantizedMeasure)
        # Start measure
        # Reset state
        state.measure = measure

        w.line(f"% measure {i+1}")
        w.indents += 1
        measureDef = scorestruct.getMeasureDef(i)

        if addTempoMarks and measureDef.timesig != timesig:
            timesig = measureDef.timesig
            if len(timesig.parts) == 1:
                if not timesig.subdivisionStruct:
                    num, den = timesig.fusedSignature
                    if options.addSubdivisionsForSmallDenominators and _isSmallDenominator(den, quarterTempo):
                        den, subdivs = measureDef.subdivisionStructure()
                        num, den2 = measureDef.timesig.fusedSignature
                        assert den == den2
                        w.line(fr"\time {','.join(map(str, subdivs))} {num}/{den}")
                    else:
                        # common case, simple timesig num/den
                        w.line(fr"\time {timesig.numerator}/{timesig.denominator}")
                else:
                    subdivs = ",".join(map(str, timesig.subdivisionStruct))
                    # \time 2,2,3 7/8
                    num, den = timesig.fusedSignature
                    w.line(fr"\time {subdivs} {num}/{den}")
            else:
                # 3/8 -> (3 8)
                pairs = ' '.join(f"({num} {den})" for num, den in timesig.parts)
                w.line(fr"\compoundMeter #'({pairs})")
                # Add subdivisions if needed.
                if (options.compoundMeterSubdivision == 'all' or
                        (options.compoundMeterSubdivision == 'heterogeneous' and
                         measureDef.timesig.isHeterogeneous()) or
                        any(denom == 4 for num, denom in measureDef.timesig.parts)):
                    den, multiples = measureDef.subdivisionStructure()
                    num = measureDef.timesig.fusedSignature[0]
                    subdivs = ",".join(map(str, multiples))
                    w.line(fr"\time {subdivs} {num}/{den}")

        if addTempoMarks and measure.quarterTempo != quarterTempo:
            quarterTempo = measure.quarterTempo
            # lilypond only support integer tempi
            # TODO: convert to a different base if the tempo is too slow/fast for
            #       the quarter, or convert according to the time signature
            w.line(fr"\tempo 4 = {int(quarterTempo)}")

        if measureDef.keySignature:
            w.line(lilytools.keySignature(fifths=measureDef.keySignature.fifths,
                                          mode=measureDef.keySignature.mode))

        if addMeasureMarks:
            if measureDef.annotation:
                style = options.parsedmeasureLabelStyle
                relfontsize = style.fontsize - options.staffSize if style.fontsize else 0
                w.line(lilytools.makeTextMark(measureDef.annotation,
                                              fontsize=relfontsize,
                                              fontrelative=True,
                                              box=style.box))
            if measureDef.rehearsalMark and measureDef.rehearsalMark.text:
                style = options.parsedRehearsalMarkStyle
                relfontsize = style.fontsize - options.staffSize if style.fontsize else 0
                box = measureDef.rehearsalMark.box or style.box
                w.line(lilytools.makeTextMark(measureDef.rehearsalMark.text,
                                              fontsize=relfontsize, fontrelative=True,
                                              box=box))

        if measure.empty():
            measureDur = measure.duration()
            if measureDur.denominator == 1 and measureDur.numerator in (1, 2, 3, 4, 6, 7, 8):
                lilydur = lilytools.makeDuration(measureDur)
                w.line(f"R{lilydur}")
            else:
                num, den = measure.timesig.fusedSignature
                w.line(f"R1*{num}/{den}")
            state.dynamic = ''
        else:
            root = measure.tree
            _forceBracketsForNestedTuplets(root)
            markConsecutiveGracenotes(root)
            lilytext = renderNode(root, durRatios=[], options=options,
                                  numIndents=0, indentSize=w.indentsize,
                                  state=state)
            w.line(lilytext)
        w.indents -= 1

        if not measureDef.barline or measureDef.barline == 'single':
            w.line(f"|   % end measure {i+1}")
        else:
            if (barstyle := _lilyBarlines.get(measureDef.barline)) is None:
                logger.error(f"Barstile '{measureDef.barline}' unknown. "
                             f"Supported styles: {_lilyBarlines.keys()}")
                barstyle = '|'
            w.line(rf'\bar "{barstyle}"    |  % end measure {i+1}')

    w.indents -= 1

    w.line(f"}}   % end staff {part.name}")
    return w.join()


# --------------------------------------------------------------

def makeScore(score: quant.QuantizedScore,
              options: RenderOptions,
              midi=False
              ) -> str:
    r"""
    Render a QuantizedScore as a lilypond score (as str)

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
        if indent:
            s = textwrap.indent(s, prefix=IND * indent)
        strs.append(s)

    lilypondVersion = lilytools.getLilypondVersion()
    if not lilypondVersion:
        raise RuntimeError("Could not determine lilypond version")

    _(f'\\version "{lilypondVersion}"\n')

    if options.title or options.composer:
        _(
fr'''
\header {{
    title = "{options.title}"
    composer = "{options.composer}"
    tagline = ##f
}}
''')
    else:
        _(r"\header { tagline = ##f }")

    # Global settings
    # staffSizePoints = lilytools.millimetersToPoints(options.staffSize)
    staffSizePoints = options.staffSize
    if options.renderFormat == 'png':
        staffSizePoints *= options.lilypondPngStaffsizeScale

    _(f'#(set-global-staff-size {staffSizePoints})')

    if options.preview or options.cropToContent:
        _(lilytools.paperBlock(margin=20, unit="mm"))
        if options.preview:
            _(r"#(ly:set-option 'preview #t)")
        else:
            _(r"#(ly:set-option 'crop #t)")
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
            \override Glissando.thickness = #%d
            \override Glissando.gap = #0.05
          }
        }
        """ % options.glissLineThickness, dedent=True)

    if options.flagStyle != 'normal':
        lilyFlagStyle = {'straight': 'modern-straight-flag',
                         'flat': 'flat-flag',
                         'old-straight': 'old-straight-flag'}[options.flagStyle]
        _(lilypondsnippets.flagStyleLayout(lilyFlagStyle))

    if options.horizontalSpace:
        spacingPreset = lilypondsnippets.horizontalSpacePresets[options.horizontalSpace]
        if spacingPreset:
            _(spacingPreset)

    if options.lilypondGlissMinLength:
        _(lilypondsnippets.glissandoMinimumLength(options.lilypondGlissMinLength))

    if options.useStemlets:
        _(lilypondsnippets.stemletLength(length=options.stemletLength, context='Score'))

    _(r"\score {")

    _(r"<<")
    indents = 1
    groups = score.groupParts()
    partindex = 0
    for group in groups:
        if len(group) > 1:
            if group[0].groupName is not None:
                name, shortname = group[0].groupName
            else:
                name, shortname = '', ''
            _(fr'\new StaffGroup \with {{ instrumentName = "{name}" shortInstrumentName = "{shortname}" }} <<', indent=1)
            indents += 1
        for part in group:
            partstr = renderPart(part,
                                 addMeasureMarks=partindex == 0,
                                 addTempoMarks=partindex == 0,
                                 options=options,
                                 indents=indents,
                                 indentSize=indentSize,
                                 clef=part.firstClef)
            _(partstr)
            partindex += 1
        if len(group) > 1:
            _(r">>", indent=1)
            indents -= 1

    _(r">>")
    if options.proportionalSpacing:
        dur = asF(options.proportionalNotationDuration)
        if options.proportionalSpacingKind == 'strict':
            strict, uniform = True, True
        elif options.proportionalSpacingKind == 'uniform':
            strict, uniform = False, True
        else:
            strict, uniform = False, False
        _(lilypondsnippets.proportionalSpacing(num=dur.numerator, den=dur.denominator,
                                               strict=strict, uniform=uniform))

    if midi:
        _(" "*indentSize + r"\midi { }")
    _(r"}   % end score")  # end \score
    return "\n".join(strs)


class LilypondRenderer(Renderer):
    def __init__(self,
                 score: quant.QuantizedScore,
                 options: RenderOptions):
        super().__init__(score, options=options)
        self._withMidi = False

    def render(self, options: RenderOptions | None = None) -> str:
        return self._render(options=options if options is not None else self.options)

    @cache
    def _render(self, options: RenderOptions) -> str:
        assert isinstance(options, RenderOptions)
        return makeScore(self.quantizedScore, options=options, midi=self._withMidi)

    def writeFormats(self) -> list[str]:
        return ['pdf', 'ly', 'png']

    def write(self, outfile: str, fmt='', removeTemporaryFiles=False) -> None:
        # for png files, renders only the first page
        outfile = emlib.filetools.normalizePath(outfile)
        tempbase, ext = os.path.splitext(outfile)
        options = self.options.copy()
        if not fmt:
            fmt = ext[1:]

        if fmt not in ('png', 'pdf', 'ly', 'mid'):
            raise ValueError(f"Format {fmt} unknown. Possible formats: png, pdf, mid, ly")

        # Modify render options according to format, if needed
        if fmt == 'png':
            if options.cropToContent is None:
                options.cropToContent = True
            options.renderFormat = 'png'
        elif fmt == 'pdf':
            if options.cropToContent is None:
                options.cropToContent = False
            options.renderFormat = 'pdf'
        elif fmt == 'mid':
            if not self._withMidi:
                self._withMidi = True

        lilytxt = self.render(options=options)
        tempfiles = []
        lilypondBinary = self.options.lilypondBinary

        if fmt == 'ly':
            open(outfile, "w").write(lilytxt)

        elif fmt == 'png' or fmt == 'pdf':
            lilyfile = _util.mktemp(suffix=".ly")
            tempbase = os.path.splitext(lilyfile)[0]
            tempout = f"{tempbase}.{fmt}"
            open(lilyfile, "w").write(lilytxt)
            logger.debug(f"Rendering lilypond '{lilyfile}' to '{tempout}'")
            outfiles = lilytools.renderLily(lilyfile=lilyfile,
                                            outfile=tempout,
                                            imageResolution=options.pngResolution,
                                            lilypondBinary=lilypondBinary)
            if options.preview:
                previewfile = f"{tempbase}.preview.{fmt}"
                if os.path.exists(previewfile):
                    logger.debug(f"Found preview file {previewfile}, using that as output")
                    tempout = previewfile
            elif options.cropToContent:
                cropfile = f"{tempbase}.cropped.{fmt}"
                if os.path.exists(cropfile):
                    logger.debug(f"Found crop file '{cropfile}', using that as output")
                    tempout = cropfile
                else:
                    logger.debug(f"Asked to generate a crop file, but the file '{cropfile}' "
                                 f"was not found. Image file: {tempout}")
                    if fmt == 'png':
                        tempout = outfiles[0]
                        logger.debug("Trying to generate cropped file via pillow")
                        _imgtools.imagefileAutocrop(tempout, cropfile, bgcolor="#ffffff")
                        if not os.path.exists(cropfile):
                            logger.debug("Failed to generate crop file, aborting cropping")
                        else:
                            tempout = cropfile
            logger.debug(f"Moving {tempout} to {outfile}")
            shutil.move(tempout, outfile)
            tempfiles.append(lilyfile)
            # Cascade: if preview: base.preview.fmt, if crop: base.crop.fmt else base.fmt

        elif fmt == 'mid' or fmt == 'midi':
            lilyfile = _util.mktemp(suffix='.ly')
            open(lilyfile, "w").write(lilytxt)
            lilytools.renderLily(lilyfile=lilyfile, lilypondBinary=lilypondBinary)
            midifile = emlib.filetools.withExtension(lilyfile, "midi")
            if not os.path.exists(midifile):
                raise RuntimeError(f"Failed to generate MIDI file. Expected path: {midifile}")
            shutil.move(midifile, outfile)
            tempfiles.append(lilyfile)

        if removeTemporaryFiles:
            for f in tempfiles:
                os.remove(f)
