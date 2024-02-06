"""
Plotting using notation

Unicode encoding for musical symbols is based on Standard Music Font
Layout (SMuFL) (see https://w3c.github.io/smufl/gitbook/).

"""

from __future__ import annotations
from math import *
import pitchtools as pt
from maelzel import colortheory
import matplotlib.pyplot as plt
import emlib.mathlib
from dataclasses import dataclass
from matplotlib.artist import Artist
from matplotlib.textpath import TextPath
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Rectangle, Circle, Polygon, PathPatch
from matplotlib.text import Text
from matplotlib import transforms
from pathlib import Path
import itertools

from maelzel.core import Score, Voice, Note, Chord, MEvent
from maelzel.scorestruct import ScoreStruct


def _verticalPosToCleffPos(pos: int) -> int:
    """

    5C    35     7
    4C    28     0
    3C    21
    """
    return pos - 28


def _bravuraPath() -> Path:
    from maelzel import dependencies
    path = dependencies.dataPath() / 'Bravura.otf'
    if not path.exists():
        raise OSError(f"Did not find bravura font, searched path: {path}")
    return path


_BRAVURA_ACCIDENTALS = {
    'natural': '\uE261',
    'natural-up': '\uE272',
    'quarter-sharp': '\uE282',
    'sharp-down': '\uE275',
    'sharp': '\uE262',
    'sharp-up': '\uE274',
    'three-quarters-sharp': '\uE283',
    'natural-down': '\uE273',
    'quarter-flat': '\uE284',
    'flat-up': '\uE270',
    'flat': '\uE260',
    'flat-down': '\uE271',
    'three-quarters-flat': '\uE281'
}


@dataclass
class ClefDef:
    name: str
    pitchrange: tuple[int, int]
    lines: tuple[str, str, str, str, str]
    ledgerlines: list[str]
    color: tuple[float, float, float]

    def verticalRange(self) -> tuple[int, int]:
        return (pt.vertical_position(self.lines[0]), pt.vertical_position(self.lines[-1]))


_clefdefs = [
    ClefDef(name='treble',
            pitchrange=(60, 84),
            lines=("4E", "4G", "4B", "5D", "5F"),
            ledgerlines=['4C', '5A', '6C'],
            color=(0., 0., 0.4)),
    ClefDef(name='bass',
            pitchrange=(36, 60),
            lines=("2G", "2B", "3D", "3F", "3A"),
            ledgerlines=['2C', '2E', '4C'],
            color=(0.4, 0.0, 0.0)),

]


_ledgerlines = [
    ('4C', ('4C',)),
    ('5A', ('5A', '5B')),
    ('6C', ('6C', '6D'))
]


_voiceColors = [
    (0.8, 0.1, 0.1),
    (0.5, 0.0, 0.6),
    (0.0, 0.1, 0.7),
    (0.0, 0.6, 0.5)
]


def _measureOffsetsIncluding(scorestruct: ScoreStruct, end: F, realtime: bool) -> list[float]:
    """
    Returns all measure offsets including the given `end`
    """
    offsets = []
    offset = 0
    for mdef in scorestruct.iterMeasureDefs():
        offsets.append(offset)
        if offset > end:
            break
        offset += mdef.durationQuarters

    if realtime:
        offsets = [scorestruct.beatToTime(offset) for offset in offsets]
    return [float(offset) for offset in offsets]


def makeAxes(tightlayout=True, hideyaxis=True) -> tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(1, 1)
    if hideyaxis:
        axes.get_yaxis().set_visible(False)
    if tightlayout:
        fig.tight_layout()
    return fig, axes


def plotVoices(voices: list[Voice],
               axes: plt.Axes = None,
               realtime=False,
               colors: list[tuple[float, float, float]] = None,
               eventHeadAlpha=0.3,
               eventLineAlpha=0.8,
               eventStartAlpha=1.0,
               eventStartLuminosityFactor=0.8,
               eventHeight=1.3,
               eventLineHeight=0.3,
               maxwidth=0.1,
               accidentalColor=(0., 0., 0., 1.0),
               accidentalScale=(0.5, 1),
               linkedEventDesaturation=1.0,
               accidentalSize=22,
               accidentalShift=2,
               eventStartLineWidth=2,
               accidentalFixedScale=True,
               staffLineWidth=2,
               drawHeadForTiedEvents=False,
               ledgerLineColor=(0., 0., 0., 0.3),
               barlines=True,
               barlineColor=(0., 0., 0., 0.2),
               barlineWidth=1,
               barlineAcrossAllStaffs=True,
               scorestruct: ScoreStruct = None,
               timeSignatures=True,
               chordLink=False,
               setLimits=True):
    if axes is None:
        fig, axes = makeAxes()
    else:
        fig = axes.get_figure()

    minpitch, maxpitch = 127, 0
    for voice in voices:
        voicemin, voicemax = voice.pitchRange()
        minpitch = min(minpitch, voicemin)
        maxpitch = max(maxpitch, voicemax)
    drawnclefs = drawStaffs(axes, int(minpitch), int(maxpitch), linewidth=staffLineWidth,
                            ledgerLineColor=ledgerLineColor)

    if not colors:
        colors = _voiceColors
    voiceColors = itertools.islice(itertools.cycle(colors), len(voices))

    accidentalFontProperties = FontProperties(fname=_bravuraPath().as_posix(), size=accidentalSize)

    if not scorestruct:
        scorestruct = voices[0].scorestruct(resolve=True)

    if barlines:
        end = max(voice.dur for voice in voices)
        offsets = _measureOffsetsIncluding(scorestruct, end, realtime=realtime)

        if barlineAcrossAllStaffs:
            miny = min(clef.verticalRange()[0] for clef in drawnclefs)
            maxy = max(clef.verticalRange()[1] for clef in drawnclefs)
            minpos = _verticalPosToCleffPos(miny)
            maxpos = _verticalPosToCleffPos(maxy)
            for x in offsets:
                axes.add_line(Line2D([x, x], [minpos, maxpos], linewidth=barlineWidth, color=barlineColor))
        else:
            for clef in drawnclefs:
                miny, maxy = clef.verticalRange()
                minpos, maxpos = _verticalPosToCleffPos(miny), _verticalPosToCleffPos(maxy)
                for x in offsets:
                    axes.add_line(Line2D([x, x], [minpos, maxpos], linewidth=barlineWidth, color=barlineColor))

    for voice, baseColor in zip(voices, voiceColors):
        eventHeadColor = baseColor + (eventHeadAlpha,)
        eventLineColor = baseColor + (eventLineAlpha,)
        eventStartColor = colortheory.luminosityFactor(baseColor, eventStartLuminosityFactor) + (eventStartAlpha,)
        eventHeadLinkedColor = colortheory.desaturate(baseColor, linkedEventDesaturation) + (eventHeadAlpha,)
        eventLineLinkedColor = colortheory.desaturate(baseColor, linkedEventDesaturation) + (eventLineAlpha,)

        linked = False
        tied = False

        for event, offset in voice.eventsWithOffset():
            if isinstance(event, Note):
                pitches = [event.name]
                targets = [''] if not event.gliss else [event.glissTarget()]
            elif isinstance(event, Chord):
                pitches = [note.name for note in event]
                targets = [''] * len(event) if not event.gliss else event.glissTarget()
            else:
                logger.warning(f"{type(event)} not supported yet: {event}")
                continue

            if not realtime:
                x0 = float(offset)
                x1 = x0 + float(event.dur)
            else:
                x0 = float(scorestruct.beatToTime(offset))
                x1 = x0 + scorestruct.timeDelta(offset, offset + event.dur)

            _chordLinkColor = eventStartColor[:3] + (eventHeadAlpha,)
            if not linked:
                _eventcolor = eventHeadColor
                _eventlinecolor = eventLineColor
            else:
                _eventcolor = eventHeadLinkedColor
                _eventlinecolor = eventLineLinkedColor

            if len(pitches) > 1:
                if chordLink and (not tied or drawHeadForTiedEvents):
                    # Draw chord link
                    minpitch = min(pitches, key=pt.n2m)
                    maxpitch = max(pitches, key=pt.n2m)
                    minpos = _verticalPosToCleffPos(pt.vertical_position(minpitch))
                    maxpos = _verticalPosToCleffPos(pt.vertical_position(maxpitch))
                    axes.add_line(Line2D([x0, x0], [minpos-eventHeight/2, maxpos+eventHeight/2],
                                         linewidth=eventStartLineWidth*2, color=_chordLinkColor,
                                         ))

                    # chordHead = Rectangle((x0, minpos - eventHeight / 2),
                    #                       width=min(float(event.dur), maxwidth / 4),
                    #                       height=maxpos - minpos + eventHeight,
                    #                       color=_eventcolor, edgecolor=None, linewidth=0)
                    # axes.add_patch(chordHead)

            for pitch, target in zip(pitches, targets):
                npitch = pt.notated_pitch(pitch)
                cleffpos = _verticalPosToCleffPos(npitch.vertical_position)

                yoffsetFactor = npitch.diatonic_alteration * 0.5
                if not linked:
                    axes.add_line(Line2D([x0, x0], [cleffpos-eventHeight/2, cleffpos+eventHeight/2],
                                         linewidth=eventStartLineWidth, color=eventStartColor))
                if not event.gliss:
                    axes.add_patch(Rectangle((x0, cleffpos+yoffsetFactor-eventLineHeight/2),
                                             width=x1-x0, height=eventLineHeight,
                                             color=_eventlinecolor, linewidth=0))
                else:
                    assert target
                    targetpitch = pt.notated_pitch(target)
                    targetpos = _verticalPosToCleffPos(targetpitch.vertical_position)
                    targetoffset = targetpitch.diatonic_alteration * 0.5
                    y0 = cleffpos + yoffsetFactor - eventLineHeight / 2
                    y1 = targetpos + targetoffset - eventLineHeight / 2
                    points = [[x0, y0], [x1, y1], [x1, y1+eventLineHeight], [x0, y0+eventLineHeight]]
                    axes.add_patch(Polygon(points, closed=True, color=_eventlinecolor))

                if not tied or drawHeadForTiedEvents:
                    eventHead = Rectangle((x0, cleffpos-eventHeight/2),
                                          width=min(float(event.dur), maxwidth), height=eventHeight,
                                          color=_eventcolor, edgecolor=None, linewidth=0)
                    axes.add_patch(eventHead)
                    accidentalCode = _BRAVURA_ACCIDENTALS[npitch.accidental_name]
                    if accidentalFixedScale:
                        trans = axes.transData + transforms.ScaledTranslation(xt=accidentalShift/72., yt=0,
                                                                              scale_trans=fig.dpi_scale_trans)
                        axes.text(x0, cleffpos, s=accidentalCode, fontproperties=accidentalFontProperties,
                                  color=accidentalColor, transform=trans)
                    else:
                        textpath = TextPath(xy=(0.01, 0), s=accidentalCode, prop=accidentalFontProperties)
                        offset = transforms.Affine2D().scale(*accidentalScale).translate(x0,
                                                                                         cleffpos) + axes.transData
                        pathpatch = PathPatch(textpath, color=accidentalColor, transform=offset)
                        axes.add_patch(pathpatch)

            linked = event.linkedNext()
            tied = event.tied
    if setLimits:
        if realtime:
            axes.set_xlim(-0.5, max(voice.durSecs() for voice in voices))
        else:
            axes.set_xlim(-0.5, max(float(voice.dur) for voice in voices))


def drawStaffs(axes: plt.Axes, minpitch: int, maxpitch: int,
               linewidth=2, ledgerLineColor=(0., 0., 0., 0.3)
               ) -> list[ClefDef]:
    drawn = []
    for clefdef in _clefdefs:
        intersect0, intersect1 = emlib.mathlib.intersection(minpitch, maxpitch, *clefdef.pitchrange)
        if intersect0 is not None:
            for pitch in clefdef.lines:
                npitch = pt.notated_pitch(pitch)
                cleffpos = _verticalPosToCleffPos(npitch.vertical_position)
                axes.axhline(cleffpos, xmin=0, color=clefdef.color, linewidth=linewidth)
            drawn.append(clefdef)

        for ledgerline in clefdef.ledgerlines:
            if minpitch <= pt.n2m(ledgerline) <= maxpitch:
                npitch = pt.notated_pitch(ledgerline)
                cleffpos = _verticalPosToCleffPos(npitch.vertical_position)
                axes.axhline(cleffpos, xmin=0, color=ledgerLineColor, linewidth=1)
    return drawn

