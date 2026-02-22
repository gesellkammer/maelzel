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
from itertools import pairwise

import emlib.filetools
import emlib.mathlib
import emlib.textlib
import pitchtools as pt

from maelzel import _imgtools
from maelzel import _util
from maelzel._indentedwriter import IndentedWriter
from maelzel.common import F, asF
from maelzel.music import lilytools
from maelzel.scoring.common import logger
from maelzel.textstyle import TextStyle

from . import attachment
from . import definitions
from . import lilypondsnippets
from . import quant
from . import util
from . import spanner as _spanner
from .core import Notation
from .node import Node
from .render import Renderer, RenderOptions

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.common import pitch_t
    from maelzel.scorestruct import MeasureDef, ScoreStruct
    from typing import Sequence


__all__ = (
    'LilypondRenderer',
    'renderPart',
    'renderScore'
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
    options: RenderOptions
    measure: quant.QuantizedMeasure | None = None
    insideSlide: bool = False
    glissando: bool = False
    dynamic: str = ''
    insideGraceGroup: bool = False
    openSpanners: dict[str, _spanner.Spanner] = field(default_factory=dict)
    pedalStyle: str = ''

    def __post_init__(self):
        if not self.pedalStyle:
            self.pedalStyle = self.options.pedalStyle or 'mixed'


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
                                  box=attach.box,
                                  family=attach.fontfamily)
    elif attach.role == 'label':
        style = TextStyle.parse(options.noteLabelStyle)
        return lilytools.makeText(text=attach.text,
                                  fontrelative=attach.relativeSize,
                                  fontsize=attach.fontsize or style.fontsize or None,
                                  placement=attach.placement or style.placement or 'above',
                                  italic=attach.italic or style.italic,
                                  bold=attach.weight=='bold' or style.bold,
                                  box=attach.box or style.box,
                                  family=attach.fontfamily)
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

        fingering = n.findAttachment(attachment.Fingering)
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

        notenames = n.resolveNotenames()
        notatedpitches = [pt.notated_pitch(notename) for notename in notenames]
        chordAccidentalTraits = n.findAttachment(cls=attachment.AccidentalTraits, pitchanchor=None) or attachment.AccidentalTraits.default()
        backTies = n.tieHints('backward') if n.tiedPrev else None
        chordparts = []
        if n.tiedNext:
            forwardTies = n.tieHints('forward')
            needsCustomTies = forwardTies and len(forwardTies) != len(n.pitches)
        else:
            forwardTies, needsCustomTies = None, False

        for i, pitch in enumerate(n.pitches):
            if noteheads and (notehead := noteheads.get(i)) is not None:
                chordparts.append(lyNotehead(notehead, insideChord=True))
            notename = notenames[i]
            notatedpitch = notatedpitches[i]

            accidentalTraits = n.findAttachment(cls=attachment.AccidentalTraits, pitchanchor=i) or chordAccidentalTraits

            if accidentalTraits.hidden:
                chordparts.append(r"\once\omit Accidental")
            if accidentalTraits.color:
                chordparts.append(fr'\tweak Accidental.color "{accidentalTraits.color}"')

            forceAccidental = accidentalTraits.force
            if n.tiedPrev:
                if backTies and i in backTies:
                    forceAccidental = False
            elif any(otherpitch.chromatic_index == notatedpitch.chromatic_index and
                     otherpitch.diatonic_name != notatedpitch.diatonic_name
                     for otherpitch in notatedpitches):
                forceAccidental = True

            lilypitch = lilytools.makePitch(notename,
                                            parenthesizeAccidental=accidentalTraits.parenthesis,
                                            forceAccidental=forceAccidental)
            if n.tiedNext and needsCustomTies and i in forwardTies:
                lilypitch += "~"
            chordparts.append(lilypitch)

        endtag = f">{base}{'.'*dots}"
        if n.tiedNext and not needsCustomTies:
            endtag += "~"
        _(f"<{' '.join(chordparts)}{endtag}")

    if trem := n.findAttachment(attachment.Tremolo):
        if trem.color:
            _(rf'\once \override StemTremolo.color = #"{trem.color}"')
        if trem.tremtype == 'single':
            if trem.relative:
                _(f":{trem.singleDuration()}")
            else:
                nummarksbase = int(math.log(base, 2) - 2)
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

    elif isinstance(spanner, _spanner.Pedal):
        if spanner.kind == 'start':
            if spanner.style and spanner.style != state.pedalStyle:
                _(rf"\set Staff.pedalSustainStyle = #'{spanner.style} ")
                state.pedalStyle = spanner.style

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

    elif isinstance(spanner, _spanner.Pedal):
        if spanner.kind == 'start':
            _(r"\sustainOn ")
        else:
            _(r"\sustainOff ")

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
               durRatios: tuple[F, ...],
               options: RenderOptions,
               state: RenderState,
               indents=0,
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
        indents: number of indents for the generated code.
        indentSize: the number of spaces per indent
    """
    w = IndentedWriter(indentSize=indentSize, indents=indents)
    tupletStarted = False
    if node.durRatio != (1, 1):
        tupletStarted = True
        # A new tuplet. Check if the node has any leading gracenotes, which need to
        # be rendered before the tuplet
        gracenotes: list[tuple[Notation, Node]] = node.initialGracenotes()
        if gracenotes:
            for n, parent in gracenotes:
                parent.items.remove(n)
            tempnode = Node([n for n, parent in gracenotes], ratio=(1, 1))
            w.add(renderNode(tempnode, durRatios=durRatios, options=options, state=state,
                             indents=indents, indentSize=indentSize))
        durRatios = durRatios + (F(*node.durRatio),)
        num, den = node.durRatio
        w.line(f"\\tuplet {num}/{den} {{", postindent=1)
        if node.getProperty('.forceTupletBracket'):
            w.line(r"\once \override TupletBracket.bracket-visibility = ##t")

    for i, item in enumerate(node.items):
        if isinstance(item, Node):
            w.line(renderNode(item, durRatios, options=options, indents=0,
                              state=state, indentSize=w.indentSize))
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
            if (options.removeRedundantDynamics and
                    not item.dynamic.endswith('!') and
                    item.dynamic == state.dynamic and
                    item.dynamic in definitions.dynamicLevels):
                item.dynamic = ''
            state.dynamic = item.dynamic

        # Slur modifiers (line type, etc.) need to go before the start of
        # the first note of the spanner. Some spanners need to declare customizations
        # before the note to which they are attached to
        if item.spanners:
            item.spanners.sort(key=lambda spanner: spanner.priority())
            for spanner in item.spanners:
                if lilytext := _handleSpannerPre(spanner, state=state):
                    w.add(lilytext)

        if item.gliss:
            if glissmap := item.findAttachment(attachment.GlissMap):
                if (nextev := node.nextNotation(item)) is not None:
                    idxs0 = [item.index(pair[0], tolerance=0.01) for pair in glissmap.pairs]
                    idxs1 = [nextev.index(pair[1], tolerance=0.01) for pair in glissmap.pairs]
                    pairstrs = [f"({source} . {dest})"
                                for source, dest in zip(idxs0, idxs1)
                                if source is not None and dest is not None]
                    if pairstrs:
                        pairstr = " ".join(pairstrs)
                        w.line(rf"\once \set glissandoMap = #'({pairstr})")

        # <<<<<<<<<<<<<< The pitches of the event >>>>>>>>>>>>>>
        w.add(notationToLily(item, options=options, state=state))

        # <<<<<<<<<<<<<< Post pitch >>>>>>>>>>>>>>
        if item.gliss:
            if not state.glissando:
                if props := item.findAttachment(attachment.GlissProperties):
                    # TODO: implement GlissProperties.index
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
                state.glissando = False
                w.add(r"\glissandoSkipOff")

        if item.isGracenote:
            if state.insideGraceGroup and item.getProperty('.graceGroup') == 'stop':
                state.insideGraceGroup = False
                w.add("}")
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


def _numStartGracenotes(part: quant.QuantizedPart) -> int:
    gracenotes = 0
    for n in part.flatNotations():
        if not n.isGracenote:
            break
        gracenotes += 1
    return gracenotes


def addTimeSignature(w: IndentedWriter, measuredef: MeasureDef, options: RenderOptions):
    timesig = measuredef.timesig
    if len(timesig.parts) == 1:
        if not timesig.subdivisionStruct:
            num, den = timesig.fusedSignature
            if options.addSubdivisionsForSmallDenominators and _isSmallDenominator(den, measuredef.quarterTempo):
                den, subdivs = measuredef.subdivisionStructure()
                num, den2 = measuredef.timesig.fusedSignature
                # assert den == den2
                line = fr"\time {','.join(map(str, subdivs))} {num}/{den2}"
                print(line)
                w.line(line)
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
                 measuredef.timesig.isHeterogeneous()) or
                any(denom == 4 for num, denom in measuredef.timesig.parts)):
            den, multiples = measuredef.subdivisionStructure()
            num = measuredef.timesig.fusedSignature[0]
            subdivs = ",".join(map(str, multiples))
            w.line(fr"\time {subdivs} {num}/{den}")


def _renderMeasures(w: IndentedWriter,
                    part: quant.QuantizedPart,
                    addTempoMarks: bool,
                    options: RenderOptions,
                    addMeasureMarks: bool,
                    startGracenotes=0) -> None:
    scorestruct = part.struct
    partGracenotes = _numStartGracenotes(part)
    timesig = scorestruct.measure(0).timesig
    quarterTempo = F(0)
    state = RenderState(options)

    if any(n.spanners and any(isinstance(s, _spanner.Pedal) for s in n.spanners)
           for n in part.flatNotations()):
        # Pedals found, set default pedal style
        w.line(fr"\set Staff.pedalSustainStyle = #'{state.pedalStyle}")

    for i, measure in enumerate(part.measures):
        # Start measure
        state.measure = measure
        w.line(f"% measure {i+1}", postindent=1)
        measuredef = scorestruct.measure(i)
        if i > 0 and measuredef.timesig != timesig:
            addTimeSignature(w, measuredef=measuredef, options=options)
            timesig = measuredef.timesig

        if addTempoMarks and measure.quarterTempo != quarterTempo:
            quarterTempo = measure.quarterTempo
            refvalue, numdots = measuredef.tempoRef
            w.line(fr"\tempo {str(refvalue) + "." * numdots} = {int(measuredef.tempo)}")

        if measuredef.key:
            w.line(lilytools.keySignature(fifths=measuredef.key.fifths, mode=measuredef.key.mode))

        if i == 0 and partGracenotes < startGracenotes:
            s = "s8 " * (startGracenotes - partGracenotes)
            w.line(r"\grace{ " + s + "}")

        if addMeasureMarks:
            if measuredef.annotation:
                style = options.parsedmeasureLabelStyle
                w.line(lilytools.makeTextMark(measuredef.annotation,
                                              fontsize=style.fontsize-options.staffSize if style.fontsize else 0,
                                              fontrelative=True,
                                              box=style.box))
            if measuredef.mark and measuredef.mark.text:
                style = options.parsedRehearsalMarkStyle
                w.line(lilytools.makeTextMark(measuredef.mark.text,
                                              fontsize=style.fontsize-options.staffSize if style.fontsize else 0,
                                              fontrelative=True,
                                              box=measuredef.mark.box or style.box))

        if measure.isEmpty():
            measureDur = measure.duration()
            if measureDur.denominator == 1 and measureDur.numerator in (1, 2, 3, 4, 6, 7, 8):
                w.line(f"R{lilytools.makeDuration(measureDur)}")
            else:
                num, den = measure.timesig.fusedSignature
                w.line(f"R1*{num}/{den}")
            state.dynamic = ''
        else:
            root = measure.tree
            _forceBracketsForNestedTuplets(root)
            markConsecutiveGracenotes(root)
            w.line(renderNode(root, durRatios=(), options=options,
                              indentSize=w.indentSize, state=state))

        if measuredef.barline and measuredef.barline != 'single':
            w.line(rf'\bar "{_lilyBarlines[measuredef.barline]}"')

        w.indents -= 1
        w.line(f"|   % end measure {i+1}")


def _numToName(num: int) -> str:
    assert 0 <= num <= 6
    return ("zero", "one", "two", "three", "four", "five", "six")[num]


def _renderPartHeader(w: IndentedWriter,
                      options: RenderOptions,
                      measuredef: MeasureDef,
                      clef='',
                      name='',
                      abbrev='',
                      showName=True,
                      ) -> None:
    """
    Renders the header, advances one indentation

    Args:
        w:
        options:
        measuredef:
        clef:
        name:
        abbrev:
        showName:

    Returns:

    """
    w.line(r"\new Staff \with {")
    with w.indent():
        if name and showName:
            w.line(f'instrumentName = #"{name}"')
            if abbrev:
                w.line(f'shortInstrumentName = "{abbrev}"')
        if clef:
            w.line(lilytools.makeClef(clef))
    w.line("}{", postindent=1)
    w.line(r"\numericTimeSignature")
    addTimeSignature(w, measuredef, options=options)
    w.line(lilytools.makeClef(clef))


def renderPart(part: quant.QuantizedPart,
               options: RenderOptions,
               addMeasureMarks=True,
               clef='',
               addTempoMarks=True,
               indents=0,
               indentSize=2,
               startGracenotes=0,
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
    w = IndentedWriter(indentSize=indentSize, indents=indents)
    _renderPartHeader(w=w, options=options, measuredef=part.struct[0], clef=clef)
    _renderMeasures(w=w, part=part,
                    addTempoMarks=addTempoMarks,
                    options=options,
                    addMeasureMarks=addMeasureMarks,
                    startGracenotes=startGracenotes)

    w.indents -= 1
    w.line(f"}}   % end staff {part.name}")
    return w.join()

# --------------------------------------------------------------

def renderScore(score: quant.QuantizedScore,
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
    SPACES = "                                                                                "
    numMeasures = max(len(part.measures)
                      for part in score.parts)
    struct = score.struct.copy()
    struct.setBarline(numMeasures - 1, 'final')
    score.struct = struct

    strs = []

    def _(s, dedent=False, indent=0):
        if dedent:
            s = textwrap.dedent(s)
        if indent:
            prefix = SPACES[:indentSize*indent]
            s = prefix+s if "\n" not in s else textwrap.indent(s, prefix=prefix)
        strs.append(s)

    lilyversion = lilytools.lilypondVersion(options.lilypondBinary)
    if not lilyversion:
        raise RuntimeError("Could not determine lilypond version")

    _(f'\\version "{lilyversion}"\n')
    _(r"\header {")
    if options.title:
        _(f'{IND}title = "{options.title}"')
    if options.composer:
        _(f'{IND}composer = "{options.composer}"')
    _(f'{IND}tagline = ##f')
    _("}")

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

    if score.isPolymetric():
        _(lilypondsnippets.polymetricScore)

    # There is a "bug" in lilypond where, if a part has gracenotes at the beginning
    # of the part, parts without gracenotes end up unaligned. For this, any part
    # without gracenotes needs to have "silent" gracenotes.
    maxStartGracenotes = max(_numStartGracenotes(part) for part in score.parts)
    w = IndentedWriter(indentSize=indentSize, indents=0)
    w(r"\score {")
    w(r"<<")
    w.indents += 1
    partindex = 0
    voiceindex = 0
    root = score.partTree()
    stack: list[dict] = []
    for item in root.serialize():
        if isinstance(item, dict):
            if item['kind'] == 'group':
                if item['open']:
                    stack.append(item)
                    w(fr'\new StaffGroup \with {{ instrumentName = "{item['name']}" '
                      f'shortInstrumentName = "{item['abbrev']}" }} <<')
                    w.indents += 1

                else:
                    group = stack.pop()
                    assert group['kind'] == 'group' and group['id'] == item['id']
                    w.indents -= 1
                    w.line(r">>    % end group")

            elif item['kind'] == 'part':
                if item['open']:
                    _renderPartHeader(w, options=options,
                                      measuredef=item['struct'][0],
                                      clef=item['clef'],
                                      name=item['name'],
                                      abbrev=item['abbrev'])
                    w("<<")
                    w.indents += 1
                    voiceindex = 0
                    stack.append(item)
                else:
                    w.indents -= 1
                    w(">>    % end part")
                    w.indents -= 1
                    w(f"}}    % end staff {item['name']}")

                    partdict = stack.pop()
                    assert partdict['kind'] == 'part' and partdict['id'] == item['id']
        else:
            assert isinstance(item, quant.QuantizedPart)
            part = item
            if stack and stack[-1]['kind'] == 'part':
                # A multivoice part
                voicename = _numToName(voiceindex+1)
                w(f'\\new Voice = "{voicename}" {{')
                w.indents += 1
                w(f"\\voice{voicename.capitalize()}")
                _renderMeasures(w=w,
                                part=part,
                                addTempoMarks=partindex==0,
                                options=options,
                                addMeasureMarks=partindex==0,
                                startGracenotes=maxStartGracenotes)
                w.indents -= 1
                w("}")
                voiceindex += 1
            else:
                name, abbrev = part.name, part.abbrev
                # If the group has a name, do not show name of part
                if stack:
                    parent = stack[-1]
                    if parent['kind'] == 'group' and parent['name']:
                        name, abbrev = '', ''

                _renderPartHeader(w,
                                  options=options,
                                  measuredef=part.struct[0],
                                  clef=part.firstClef,
                                  name=name,
                                  abbrev=abbrev)
                _renderMeasures(w, part=part,
                                options=options,
                                addTempoMarks=partindex==0,
                                addMeasureMarks=partindex==0,
                                startGracenotes=maxStartGracenotes)
                w.indents -= 1
                w(f"}}    % end staff {part.name}")
            partindex += 1
    w.indents -= 1
    w(r">>   % end of voices")
    if options.proportionalSpacing:
        dur = asF(options.proportionalNotationDuration)
        if options.proportionalSpacingKind == 'strict':
            strict, uniform = True, True
        elif options.proportionalSpacingKind == 'uniform':
            strict, uniform = False, True
        else:
            strict, uniform = False, False
        w(lilypondsnippets.proportionalSpacing(num=dur.numerator, den=dur.denominator,
                                               strict=strict, uniform=uniform))

    if midi:
        w(" "*indentSize + r"\midi { }")
    w(r"}   % end score")  # end \score
    strs.append(w.join())
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
        return renderScore(self.score, options=options, midi=self._withMidi)

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
            logger.debug("Rendering lilypond '%s' to '%s'", lilyfile, tempout)
            outfiles = lilytools.renderLily(lilyfile=lilyfile,
                                            outfile=tempout,
                                            imageResolution=options.pngResolution,
                                            lilypondBinary=lilypondBinary,
                                            backend=options.lilypondBackend)
            if options.preview:
                previewfile = f"{tempbase}.preview.{fmt}"
                if os.path.exists(previewfile):
                    logger.debug("Found preview file %s, using that as output", previewfile)
                    tempout = previewfile
            elif options.cropToContent:
                cropfile = f"{tempbase}.cropped.{fmt}"
                if os.path.exists(cropfile):
                    logger.debug("Found crop file '%s', using that as output", cropfile)
                    tempout = cropfile
                else:
                    logger.debug("Asked to generate a crop file, but the file '%s' "
                                 "was not found. Image file: %s", cropfile, tempout)
                    if fmt == 'png':
                        tempout = outfiles[0]
                        logger.debug("Trying to generate cropped file via pillow")
                        _imgtools.imagefileAutocrop(tempout, cropfile, bgcolor="#ffffff")
                        if not os.path.exists(cropfile):
                            logger.debug("Failed to generate crop file, aborting cropping")
                        else:
                            tempout = cropfile
            logger.debug("Moving %s to %s", tempout, outfile)
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
