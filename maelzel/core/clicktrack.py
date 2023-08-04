from __future__ import annotations

from maelzel import scorestruct
from maelzel.core._typedefs import time_t
from maelzel.core import Note, Voice, Score


def makeClickTrack(struct: scorestruct.ScoreStruct,
                   minMeasures: int = 0,
                   clickdur: time_t = None,
                   strongBeatPitch="5C",
                   weakBeatPitch="5G",
                   playpreset: str = '_click',
                   playargs: dict[str, float] = None,
                   fade=0) -> Score:
    """
    Creates a score representing a clicktrack of the given ScoreStruct

    Args:
        struct: the ScoreStruct
        minMeasures: if given, the minimum number of measures. This might be needed
            in the case of an endless scorestruct
        clickdur: the length of each tick. Use None to use the duration of the beat.
            **NB**: the duration of the playback can be set individually from the duration
            of the displayed pitch
        strongBeatPitch: the pitch to use as a strong tick
        weakBeatPitch: the pitch to use as a weak tick
        playpreset: the preset instr to use for playback. The default plays the given
            pitches two octaves higher as very short clicks
        playargs: parameters passed to the play preset, if needed
        fade: a fadetime for the clicks

    Returns:
        a :class:`~maelzel.core.score.Score`

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
    if minMeasures and minMeasures > struct.numMeasures():
        struct = struct.copy()
        struct.addMeasure(numMeasures=minMeasures - struct.numMeasures())

    for m in struct.measuredefs:
        num, den = m.timesig
        if den == 4:
            for i, n in enumerate(range(m.timesig[0])):
                pitch = strongBeatPitch if i == 0 else weakBeatPitch
                ev = Note(pitch, offset=now, dur=clickdur or 1).setPlay(fade=(0, 0.1))
                events.append(ev)
                now += 1
        elif den == 8:
            for i, n in enumerate(range(m.timesig[0])):
                pitch = strongBeatPitch if i == 0 else weakBeatPitch
                ev = Note(pitch, offset=now, dur=clickdur or 0.5).setPlay(fade=(0, 0.1))
                events.append(ev)
                now += 0.5
        elif den == 16:
            if m.quarterTempo > 80:
                dur = clickdur or m.durationQuarters
                ev = Note(strongBeatPitch, dur=dur, offset=now)
                events.append(ev)
                now += m.durationQuarters
            else:
                beats = m.subdivisions()
                for i, beat in enumerate(beats):
                    pitch = strongBeatPitch if i == 0 else weakBeatPitch
                    ev = Note(pitch, dur=clickdur or beat, offset=now)
                    events.append(ev)
                    now += beat
        else:
            raise ValueError(f"Timesig {m.timesig} not supported")
    voice = Voice(events)
    voice.setPlay(fade=fade)
    if playpreset:
        from .presetmanager import presetManager
        presetdef = presetManager.getPreset(playpreset)
        if playargs:
            if arg := next((arg for arg in playargs if arg not in presetdef.args), None):
                raise KeyError(f"arg {arg} not known for preset {playpreset}. Possible args: {presetdef.db}")
        voice.setPlay(instr=playpreset, args=playargs)
    return Score([voice], scorestruct=struct)