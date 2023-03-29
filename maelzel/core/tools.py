from __future__ import annotations

from maelzel import scorestruct
from maelzel import core
from maelzel.common import F
from .workspace import Workspace, getWorkspace

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._typedefs import time_t
    from . import score


__all__ = (
    'amplitudeToDynamics',
    'makeClickTrack',
    'packInVoices'
)


def amplitudeToDynamics(amp: float, workspace: Workspace = None) -> str:
    """
    Convert an amplitude 0-1 to a dynamic according to the current dynamics curve

    Args:
        amp: an amplitude between 0-1
        workspace: if given, use the dynamic curve defined in this workspace to
            calculate dynamics

    Returns:
        a dynamic ('pp', 'f', etc)

    Example
    ~~~~~~~

        TODO

    """
    w = workspace or getWorkspace()
    return w.dynamicCurve.amp2dyn(amp)


def makeClickTrack(struct: scorestruct.ScoreStruct,
                   minMeasures: int = 0,
                   clickdur: time_t = None,
                   strongBeatPitch="5C",
                   weakBeatPitch="5G",
                   playpreset: str = '_click',
                   playargs: dict[str, float] = None,
                   fade=0) -> score.Score:
    """
    Creates a score representing a clicktrack of the given ScoreStruct

    Args:
        struct: the ScoreStruct
        minMeasures: if given, the minimum number of measures. This might be needed
            in the case of an endless scorestruct
        clickdur: the length of each tick. Use None to use the totalDuration of the beat.
            **NB**: the totalDuration of the playback can be set individually from the totalDuration
            of the displayed pitch
        strongBeatPitch: the pitch to use as a strong tick
        weakBeatPitch: the pitch to use as a weak tick
        playpreset: the preset instr to use for playback. The default plays the given
            pitches two octaves higher as very short clicks
        playargs: parameters passed to the play preset, if needed
        fade: a fadetime for the clicks

    Returns:
        a maelzel.core Score

    Example
    ~~~~~~~

    .. code-block:: python

        from maelzel.core import *
        scorestruct = ScoreStruct(r'''
        4/4, 72
        .
        5/8
        3/8
        2/4, 96
        .
        5/4
        3/4
        '')
        clicktrack = tools.makeClickTrack(scorestruct)
        clicktrack.show()
        clicktrack.rec('clicktrack.flac')

    .. image:: ../assets/clicktrack.png

    .. seealso:: :ref:`clicktrack_notebook`

    """
    now = 0
    events = []
    if minMeasures and minMeasures > struct.numDefinedMeasures():
        struct = struct.copy()
        struct.addMeasure(numMeasures=minMeasures - struct.numDefinedMeasures())

    for m in struct.measuredefs:
        num, den = m.timesig
        if den == 4:
            for i, n in enumerate(range(m.timesig[0])):
                pitch = strongBeatPitch if i == 0 else weakBeatPitch
                ev = core.Note(pitch, offset=now, dur=clickdur or 1).setPlay(fade=(0, 0.1))
                events.append(ev)
                now += 1
        elif den == 8:
            for i, n in enumerate(range(m.timesig[0])):
                pitch = strongBeatPitch if i == 0 else weakBeatPitch
                ev = core.Note(pitch, offset=now, dur=clickdur or 0.5).setPlay(fade=(0, 0.1))
                events.append(ev)
                now += 0.5
        elif den == 16:
            if m.quarterTempo > 80:
                dur = clickdur or m.durationBeats
                ev = core.Note(strongBeatPitch, dur=dur, offset=now)
                events.append(ev)
                now += m.durationBeats
            else:
                beats = m.subdivisions()
                for i, beat in enumerate(beats):
                    pitch = strongBeatPitch if i == 0 else weakBeatPitch
                    ev = core.Note(pitch, dur=clickdur or beat, offset=now)
                    events.append(ev)
                    now += beat
        else:
            raise ValueError(f"Timesig {m.timesig} not supported")
    voice = core.Voice(events)
    voice.setPlay(fade=fade)
    if playpreset:
        from .presetmanager import presetManager
        presetdef = presetManager.getPreset(playpreset)
        if playargs:
            if arg := next((arg for arg in playargs if arg not in presetdef.args), None):
                raise KeyError(f"arg {arg} not known for preset {playpreset}. Possible args: {presetdef.db}")
        voice.setPlay(instr=playpreset, args=playargs)
    from . import score
    return score.Score([voice], scorestruct=struct)


def packInVoices(objs: list[core.MEvent | core.Chain]) -> list[core.Voice]:
    """
    Distribute the items across voices
    """
    from maelzel import packing
    items = []
    unpitched = []
    for obj in objs:
        if obj.offset is None:
            raise ValueError(f"Only objects with an explict offset can be packed, got {obj}")
        r = obj.pitchRange()
        if r is None:
            unpitched.append(r)
        else:
            pitch = (r[0] + r[1]) / 2
            dur = obj.dur
            item = packing.Item(obj,
                                offset=F(obj.offset.numerator, obj.offset.denominator),
                                dur=F(dur.numerator, dur.denominator),
                                step=pitch)
            items.append(item)
    tracks = packing.packInTracks(items)
    return [core.Voice(track.unwrap()) for track in tracks]


