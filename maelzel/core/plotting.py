"""
Plotting using notation

Unicode encoding for musical symbols is based on Standard Music Font
Layout (SMuFL) (see https://w3c.github.io/smufl/gitbook/).

"""

from __future__ import annotations
import pitchtools as pt
from maelzel.common import F
from maelzel import colortheory
import matplotlib.pyplot as plt
import emlib.mathlib
from dataclasses import dataclass
from matplotlib.textpath import TextPath
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon, PathPatch
from matplotlib import transforms
from pathlib import Path
import itertools
from functools import cache

from maelzel.core import Voice, Note, Chord
from maelzel.core import logger
from maelzel.scorestruct import ScoreStruct

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.core.mobj import MObj


def _verticalPosToClefPos(pos: int) -> int:
    """

    5C    35     7
    4C    28     0
    3C    21
    """
    return pos - 28


@cache
def _bravuraPath() -> Path:
    import maelzel.dependencies
    path = maelzel.dependencies.dataPath() / 'Bravura.otf'
    if not path.exists():
        raise OSError(f"Did not find bravura font, searched path: {path}")
    return path


_SMUFL_ACCIDENTALS = {
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


_ACCIDENTAL_SHIFT = {
    'natural': (2, 0),
    'natural-up': (2, 0),
    'quarter-sharp': (2, 0),
    'sharp-down': (2, 0),
    'sharp': (2, 0),
    'sharp-up': (2, 0),
    'three-quarters-sharp': (1, 0),
    'natural-down': (2, 0),
    'quarter-flat': (2, -2),
    'flat-up': (2, -2),
    'flat': (2, -2),
    'flat-down': (2, -2),
    'three-quarters-flat': (1, -2)
}

_DYNAMIC_SHIFT = {
    'pppp': -4,
    'ppp': -4,
    'pp': -3,
    'p': -1,
    'mp': -2,
    'mf': -2,
    'f': -1,
    'ff': -2,
    'fff': -3,
    'ffff': -4,
}

_SMUFL_TIMESIG_DIGITS = {
    '0': '\uE080',
    '1': '\uE081',
    '2': '\uE082',
    '3': '\uE083',
    '4': '\uE084',
    '5': '\uE085',
    '6': '\uE086',
    '7': '\uE087',
    '8': '\uE088',
    '9': '\uE089',
    '/': '\uE090'
}


_SMUFL_DYNAMICS = {
    'pppp': '\uE529',
    'ppp': '\uE52A',
    'pp': '\uE52B',
    'p': '\uE520',
    'mp': '\uE52C',
    'mf': '\uE52D',
    'f': '\uE522',
    'ff': '\uE52F',
    'fff': '\uE530',
    'ffff': '\uE531',
}


@dataclass
class LineStyle:
    color: tuple[float, float, float, float] | tuple[float, float, float]
    width: int = 1
    style: str = ':'


@dataclass
class ClefDef:
    name: str
    shortname: str
    pitchrange: tuple[int, int]
    lines: tuple[str, str, str, str, str]
    ledgerlines: list[str]
    color: tuple[float, float, float] | tuple[float, float, float, float]
    refline: int
    linestyle: str = '-'
    linewidth: int = 2

    def verticalRange(self) -> tuple[int, int]:
        return (pt.vertical_position(self.lines[0]), pt.vertical_position(self.lines[-1]))

    def referenceNote(self) -> str:
        return self.lines[self.refline]


@cache
def _clefDefinitions() -> list[ClefDef]:
    return [
        ClefDef(name='15a',
                shortname='G6:',
                pitchrange=(84, 108),
                lines=("6E", "6G", "6B", "7D", "7F"),
                refline=1,
                ledgerlines=["7A", "7C"],
                color=(0., 0., 0.4, 0.7),
                linestyle='--',
                linewidth=1),
        ClefDef(name='treble',
                shortname='G:',
                pitchrange=(60, 84),
                lines=("4E", "4G", "4B", "5D", "5F"),
                refline=1,
                ledgerlines=['4C', '5A', '6C'],
                color=(0., 0., 0.3),
                linewidth=1),
        ClefDef(name='bass',
                shortname='F:',
                pitchrange=(36, 60),
                lines=("2G", "2B", "3D", "3F", "3A"),
                refline=3,
                ledgerlines=['2C', '2E', '4C'],
                color=(0.3, 0.0, 0.0),
                linewidth=1),
        ClefDef(name='15b',
                shortname='F2:',
                pitchrange=(12, 36),
                lines=("0G", "0B", "1D", "1F", "1A"),
                refline=3,
                ledgerlines=['0C', '0E', '2C', '2E'],
                color=(0.4, 0.0, 0.0, 0.7),
                linestyle='--',
                linewidth=1),
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


def makeAxes(tightlayout=True, hideyaxis=False, figsize: tuple[int, int] | None = None
             ) -> tuple[Figure, Axes]:
    """
    Shortcut to construct a pyplot Axes object within a new figure

    Args:
        tightlayout: use tight layout
        hideyaxis: hide the y axis
        figsize: the size of the figure as a tuple (width, height). If not given
            the default figure size is used

    Returns:
        a tuple (figure, axes)

    """
    if tightlayout:
        fig, axes = plt.subplots(1, 1, figsize=figsize, layout="constrianed")
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
    if hideyaxis:
        axes.get_yaxis().set_visible(False)
    # if tightlayout:
    #     fig.tight_layout()
    return fig, axes


def _collectVoices(*objs: MObj | list[MObj]) -> list[Voice]:
    voices = []
    for obj in objs:
        if isinstance(obj, (list, tuple)):
            voices.extend(_collectVoices(*obj))
        else:
            voices.extend(obj._asVoices())
    return voices


def plot(*objs: MObj | list[MObj], **kws):
    voices = _collectVoices(*objs)
    return plotVoices(voices=voices, **kws)


def plotVoices(voices: list[Voice],
               axes: Axes | None = None,
               realtime=False,
               colors: list[tuple[float, float, float]] | None = None,
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
               eventStartLineWidth=2,
               accidentalFixedScale=True,
               staffLineWidth=1,
               drawHeadForTiedEvents=False,
               ledgerLineColor=(0., 0., 0., 0.3),
               barlines=True,
               barlineColor=(0., 0., 0., 0.2),
               barlineWidth=1,
               barlineAcrossAllStaffs=True,
               scorestruct: ScoreStruct | None = None,
               timeSignatures=True,
               chordLink=False,
               setLimits=True,
               timesigSize=18,
               dynamicSize=20,
               dynamicColor=(0.6, 0.6, 0.6),
               figsize: tuple[int, int] = (15, 5),
               grid=True) -> Axes:
    """
    Plot the given voices

    Args:
        voices: the voices to plot
        axes: use this Axes, if given
        realtime: use seconds, otherwise uses beats
        colors: if given, a list of tuples (r, g, b) for each voice
        eventHeadAlpha: alpha value for the head of each event
        eventLineAlpha: alpha value for the event line
        eventStartAlpha: alpha value for the start of each event
        eventStartLuminosityFactor: luminosity factor for the start of each event
        eventHeight: the height of the events box
        eventLineHeight:
        maxwidth: the max. width of the events box
        accidentalColor: color for the events accidental
        accidentalScale: scaling factor for the accidental's size
        linkedEventDesaturation: desaturation for linked events
        accidentalSize: accidental size
        eventStartLineWidth: line width for the head of each event
        accidentalFixedScale: use fixed scaling for the accidental's size
        staffLineWidth: linewidth for staff lines
        drawHeadForTiedEvents: draw the event head when an event is tied to a previous event
        ledgerLineColor: color for ledger lines
        barlines: draw barlines
        barlineColor: barline color
        barlineWidth: barline width
        barlineAcrossAllStaffs: draw barlines across all staffs
        scorestruct: scorestruct used
        timeSignatures: draw time signatures
        chordLink: draw a line to link the pitches from a chord
        setLimits: limit the display range
        timesigSize: size of the time signature
        dynamicSize: size for dynamics
        dynamicColor: color for dynamics
        figsize: figure size of the plot
        grid: draw a grid

    Returns:
        the Axes object used, will be the axes given if passed, or a new axes otherwise

    """
    if axes is None:
        fig, axes = makeAxes(figsize=figsize)
    else:
        fig = axes.get_figure()
        assert fig is not None
    z0 = 3
    smuflpath = _bravuraPath().as_posix()
    minpitch, maxpitch = 127, 0
    for voice in voices:
        pitchrange = voice.pitchRange()
        if pitchrange is not None:
            voicemin, voicemax = pitchrange
            minpitch = min(minpitch, voicemin)
            maxpitch = max(maxpitch, voicemax)
    drawnstaffs = drawStaffs(axes, minpitch=int(minpitch), maxpitch=int(maxpitch),
                             ledgerLineColor=ledgerLineColor,
                             linewidth=staffLineWidth)

    if not scorestruct:
        scorestruct = voices[0].activeScorestruct()

    scoreend = max(voice.dur for voice in voices)
    measureOffsets = _measureOffsetsIncluding(scorestruct, scoreend, realtime=realtime)

    if timeSignatures:
        timesigFontProperties = FontProperties(fname=smuflpath,
                                               size=timesigSize)

        highestline = drawnstaffs[0].lines[-1]
        clefpos = _pitchToPosition(highestline) + 0.8
        timesig = None
        for i, measureOffset in enumerate(measureOffsets):
            measuredef = scorestruct.getMeasureDef(i)
            if measuredef.timesig != timesig:
                timesig = measuredef.timesig
                smufltimesig = (f"{_SMUFL_TIMESIG_DIGITS[str(timesig.numerator)]}\n"
                                f"{_SMUFL_TIMESIG_DIGITS[str(timesig.denominator)]}")
                trans = axes.transData + transforms.ScaledTranslation(xt=-5/72., yt=0,
                                                                      scale_trans=fig.dpi_scale_trans)
                axes.text(measureOffset, clefpos, s=smufltimesig, transform=trans,
                          fontproperties=timesigFontProperties,
                          color=(0.3, .3, .3))

    if not colors:
        colors = _voiceColors
    voiceColors = itertools.islice(itertools.cycle(colors), len(voices))

    accidentalFontProps = FontProperties(fname=smuflpath, size=accidentalSize)
    dynamicFontProps = FontProperties(fname=smuflpath, size=dynamicSize)

    if barlines:

        if barlineAcrossAllStaffs:
            miny = min(clef.verticalRange()[0] for clef in drawnstaffs)
            maxy = max(clef.verticalRange()[1] for clef in drawnstaffs)
            minpos = _verticalPosToClefPos(miny)
            maxpos = _verticalPosToClefPos(maxy)
            for x in measureOffsets:
                axes.add_line(Line2D([x, x], [minpos, maxpos], linewidth=barlineWidth,
                                     color=barlineColor, zorder=z0+5))
        else:
            for clef in drawnstaffs:
                miny, maxy = clef.verticalRange()
                minpos, maxpos = _verticalPosToClefPos(miny), _verticalPosToClefPos(maxy)
                for x in measureOffsets:
                    axes.add_line(Line2D([x, x], [minpos, maxpos], linewidth=barlineWidth,
                                         color=barlineColor, zorder=z0+5))

    for voice, baseColor in zip(voices, voiceColors):
        eventHeadColor = baseColor + (eventHeadAlpha,)
        eventLineColor = baseColor + (eventLineAlpha,)
        eventStartColor = colortheory.luminosityFactor(baseColor, eventStartLuminosityFactor) + (eventStartAlpha,)
        eventHeadLinkedColor = colortheory.desaturate(baseColor, linkedEventDesaturation) + (eventHeadAlpha,)
        eventLineLinkedColor = colortheory.desaturate(baseColor, linkedEventDesaturation) + (eventLineAlpha,)

        linked = False
        tied = False

        for event, offset in voice.eventsWithOffset():
            if event.isRest():
                # TODO: find a way to show rests in an unobtrussive way
                continue
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
                    minpos = _pitchToPosition(minpitch)
                    maxpos = _pitchToPosition(maxpitch)
                    axes.add_line(Line2D([x0, x0], [minpos-eventHeight/2, maxpos+eventHeight/2],
                                         linewidth=eventStartLineWidth*2,
                                         color=_chordLinkColor,
                                         zorder=z0+10
                                         ))

                    # chordHead = Rectangle((x0, minpos - eventHeight / 2),
                    #                       width=min(float(event.dur), maxwidth / 4),
                    #                       height=maxpos - minpos + eventHeight,
                    #                       color=_eventcolor, edgecolor=None, linewidth=0)
                    # axes.add_patch(chordHead)
            if event.dynamic:
                dynamicCode = _SMUFL_DYNAMICS.get(event.dynamic, '')
                if dynamicCode:
                    clefpos = _pitchToPosition(pitches[0]) - 0.5
                    xt = _DYNAMIC_SHIFT.get(event.dynamic, 0) / 72.
                    yt = -12/72.
                    trans = axes.transData + transforms.ScaledTranslation(xt=xt, yt=yt, scale_trans=fig.dpi_scale_trans)
                    axes.text(x0, clefpos, s=dynamicCode, fontproperties=dynamicFontProps,
                              color=dynamicColor, transform=trans, zorder=z0 + 5)

            for pitch, target in zip(pitches, targets):
                npitch = pt.notated_pitch(pitch)
                clefpos = _verticalPosToClefPos(npitch.vertical_position)

                yoffsetFactor = npitch.diatonic_alteration * 0.5
                if not linked:
                    axes.add_line(Line2D([x0, x0], [clefpos-eventHeight/2, clefpos+eventHeight/2],
                                         linewidth=eventStartLineWidth,
                                         color=eventStartColor,
                                         zorder=z0+10))
                if not event.gliss:
                    axes.add_patch(Rectangle((x0, clefpos+yoffsetFactor-eventLineHeight/2),
                                             width=x1-x0, height=eventLineHeight,
                                             color=_eventlinecolor, linewidth=0, zorder=z0+10))
                else:
                    assert target
                    targetpitch = pt.notated_pitch(target)
                    targetpos = _verticalPosToClefPos(targetpitch.vertical_position)
                    targetoffset = targetpitch.diatonic_alteration * 0.5
                    y0 = clefpos + yoffsetFactor - eventLineHeight / 2
                    y1 = targetpos + targetoffset - eventLineHeight / 2
                    points = [[x0, y0], [x1, y1], [x1, y1+eventLineHeight], [x0, y0+eventLineHeight]]
                    axes.add_patch(Polygon(points, closed=True, color=_eventlinecolor, zorder=z0+10))

                if not tied or drawHeadForTiedEvents:
                    eventHead = Rectangle((x0, clefpos-eventHeight/2),
                                          width=min(float(event.dur), maxwidth), height=eventHeight,
                                          color=_eventcolor, edgecolor=None, linewidth=0,
                                          zorder=z0+10)
                    axes.add_patch(eventHead)
                    accidentalCode = _SMUFL_ACCIDENTALS[npitch.accidental_name]
                    if accidentalFixedScale:
                        xshift, yshift = _ACCIDENTAL_SHIFT[npitch.accidental_name]
                        trans = axes.transData + transforms.ScaledTranslation(xt=xshift/72.,
                                                                              yt=yshift/72.,
                                                                              scale_trans=fig.dpi_scale_trans)
                        axes.text(x0, clefpos, s=accidentalCode, fontproperties=accidentalFontProps,
                                  color=accidentalColor, transform=trans, zorder=z0+10)
                    else:
                        textpath = TextPath(xy=(0.01, 0), s=accidentalCode, prop=accidentalFontProps)
                        offset = transforms.Affine2D().scale(*accidentalScale).translate(x0,
                                                                                         clefpos) + axes.transData
                        pathpatch = PathPatch(textpath, color=accidentalColor, transform=offset)
                        axes.add_patch(pathpatch)

            linked = event.isLinkedNext()
            tied = event.tied
    if setLimits:
        if realtime:
            axes.set_xlim(-0.5, max(float(voice.durSecs()) for voice in voices))
        else:
            axes.set_xlim(-0.5, max(float(voice.dur) for voice in voices))
    if grid:
        axes.grid(axis='x', color=(0.95, 0.95, 0.95, 0.3))
    return axes


def _pitchToPosition(pitch: str) -> float:
    return _verticalPosToClefPos(pt.notated_pitch(pitch).vertical_position)


def drawStaffs(axes: Axes, minpitch: int, maxpitch: int,
               ledgerLineColor=(0., 0., 0., 0.3),
               linestyles={
                   '4C': LineStyle(color=(0., 0., 0., 0.6), style=':'),
               },
               ledgerLinesLabels=True,
               linewidth: int | None = None
               ) -> list[ClefDef]:
    # '-', '--', '-.', ':', '', (offset, on-off-seq), ...}
    z0 = 3
    drawn = []
    drawnlines = set()
    defaultLedgetLineStyle = LineStyle(color=ledgerLineColor, width=1)
    for clefdef in _clefDefinitions():
        clefmin, clefmax = clefdef.pitchrange
        intersect0, intersect1 = emlib.mathlib.intersection(minpitch, maxpitch, clefmin, clefmax)
        if intersect0 is not None:
            for pitch in clefdef.lines:
                clefpos = _pitchToPosition(pitch)
                # npitch = pt.notated_pitch(pitch)
                # clefpos = _verticalPosToClefPos(npitch.vertical_position)
                axes.axhline(clefpos, xmin=0, color=clefdef.color,
                             linewidth=linewidth or clefdef.linewidth, linestyle=clefdef.linestyle,
                             zorder=z0)
            drawn.append(clefdef)

        for ledgerlinePitch in clefdef.ledgerlines:
            if minpitch <= pt.n2m(ledgerlinePitch) <= maxpitch and ledgerlinePitch not in drawnlines:
                linestyle = linestyles.get(ledgerlinePitch, defaultLedgetLineStyle)
                clefpos = _pitchToPosition(ledgerlinePitch)
                # npitch = pt.notated_pitch(ledgerlinePitch)
                # clefpos = _verticalPosToClefPos(npitch.vertical_position)
                axes.axhline(clefpos, xmin=0, color=linestyle.color,
                             linewidth=linestyle.width, linestyle=linestyle.style,
                             zorder=z0)
                drawnlines.add(ledgerlinePitch)

    yticks = [_pitchToPosition(clefdef.lines[clefdef.refline])
              for clefdef in drawn]
    ylabels = [clefdef.shortName for clefdef in drawn]
    # ylabels = [clefdef.referenceNote() for clefdef in drawn]
    if ledgerLinesLabels:
        for line in drawnlines:
            if line.endswith('C'):
                ylabels.append(line)
                yticks.append(_pitchToPosition(line))
    axes.set_yticks(yticks)
    axes.set_yticklabels(ylabels)

    return drawn
