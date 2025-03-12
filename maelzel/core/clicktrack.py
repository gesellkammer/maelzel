from __future__ import annotations

from maelzel import scorestruct
from maelzel.common import F
from maelzel.core import Note, Voice, Score
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from maelzel.common import time_t


def makeClickTrack(struct: scorestruct.ScoreStruct,
                   minMeasures: int = 0,
                   clickdur: time_t | None = None,
                   strongBeatPitch="5C",
                   middleBeatPitch="5E",
                   weakBeatPitch="5G",
                   playpreset: str = '.click',
                   playbackTransposition=24,
                   fade=0
                   ) -> Score:
    """
    Creates a score representing a clicktrack of the given ScoreStruct

    Args:
        struct: the ScoreStruct
        minMeasures: if given, the minimum number of measures. This might be needed
            in the case of an endless scorestruct
        clickdur: the length of each tick, None to use the duration of the beat. The
            duration of the playback can be set individually from the duration of the
            displayed note.
        strongBeatPitch: the pitch to use as a strong tick
        weakBeatPitch: the pitch to use as a weak tick
        middleBeatPitch: pitch used for beginning of secondary groups for
            compound time signatures (signatures of the form 4/4+3/8)
        playpreset: the preset instr to use for playback. The default plays the given
            pitches two octaves higher as very short clicks
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
    now = F(0)
    events = []
    if minMeasures and minMeasures > struct.numMeasures():
        struct = struct.copy()
        struct.addMeasure(numMeasures=minMeasures - struct.numMeasures())

    def _processPart(num: int,
                     den: int,
                     quarterTempo: F,
                     now: F,
                     strongPitch: float | str,
                     weakPitch: float | str,
                     clickdur: time_t | None,
                     subdivisions: Sequence[F] | None = None
                     ) -> tuple[list[Note], F]:
        events = []
        if den == 4:
            for i, n in enumerate(range(num)):
                pitch = strongPitch if i == 0 else weakPitch
                ev = Note(pitch, offset=now, dur=clickdur or 1).setPlay(fade=(0, 0.1))
                events.append(ev)
                now += F(1)
        elif den == 8:
            for i, n in enumerate(range(num)):
                pitch = strongPitch if i == 0 else weakPitch
                ev = Note(pitch, offset=now, dur=clickdur or 0.5).setPlay(fade=(0, 0.1))
                events.append(ev)
                now += F(1, 2)
        elif den == 16:
            if quarterTempo > 80:
                durationQuarters = F(num) / 4
                dur = clickdur or durationQuarters
                ev = Note(strongPitch, dur=dur, offset=now)
                events.append(ev)
                now += durationQuarters
            else:
                assert subdivisions is not None
                for i, dur in enumerate(subdivisions):
                    pitch = strongPitch if i == 0 else weakPitch
                    ev = Note(pitch, dur=clickdur or dur, offset=now)
                    events.append(ev)
                    now += dur
        else:
            raise ValueError(f"Timesig {num}/{den} not supported")
        return events, now

    for m in struct.measuredefs:
        measureSubdivisions = m.subdivisions()
        for i, part in enumerate(m.timesig.parts):
            num, den = part
            subdivs = None if len(m.timesig.parts) > 1 else measureSubdivisions
            eventsInPart, now = _processPart(num=num, den=den, now=now,
                                             quarterTempo=m.quarterTempo,
                                             strongPitch=strongBeatPitch if i == 0 else middleBeatPitch,
                                             weakPitch=weakBeatPitch,
                                             clickdur=clickdur,
                                             subdivisions=subdivs)
            events.extend(eventsInPart)

    voice = Voice(events)
    voice.setPlay(fade=fade)
    voice.setPlay(instr=playpreset)
    if playpreset == '.click':
        voice.setPlay(itransp=playbackTransposition)
    return Score([voice], scorestruct=struct)
