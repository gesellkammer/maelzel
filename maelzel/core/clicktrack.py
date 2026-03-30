from __future__ import annotations

from maelzel import scorestruct
from maelzel.common import F
from maelzel.core import Note, Voice, Score
from maelzel.scoring import quantutils
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from maelzel.common import time_t


def clickTrack(struct: scorestruct.ScoreStruct,
               minMeasures: int = 0,
               clickdur: time_t | None = None,
               strongPitch="5C",
               compoundPitch="5E",
               weakPitch="5G",
               preset: str = '.click',
               transposition=24,
               fade=0.
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
        strongPitch: the pitch to use as a strong tick
        weakPitch: the pitch to use as a weak tick
        compoundPitch: pitch used for beginning of secondary groups for
            compound time signatures (signatures of the form 4/4+3/8)
        preset: the preset instr to use for playback. The default plays the given
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
        struct.addMeasure(count=minMeasures - struct.numMeasures())

    def _processTimesigPart(
            num: int,
            den: int,
            quarterTempo: F,
            now: F,
            strongPitch: float | str,
            weakPitch: float | str,
            clickdur: time_t | None,
            subdivisions: Sequence[F] | None = None
        ) -> tuple[list[Note], F]:
        # This processes a part within a time signature. If the time signature
        # is simple (3/4), this is called once, but for compound signatures this
        # is called for every part
        # TODO: use the time signature's subdivision structure
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
            if quarterTempo > 80 and num in (1, 2, 3, 4, 6, 7, 8, 12, 15):
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
            events0, _ = _processTimesigPart(
                num, den//2,
                quarterTempo=quarterTempo*2,
                now=now,
                strongPitch=strongPitch,
                weakPitch=weakPitch,
                clickdur=clickdur)
            for ev in events0:
                dur = ev.dur / 2
                events.append(ev.clone(dur=clickdur or dur, offset=now))
                now += dur
        return events, now

    for m in struct.measures:
        if m.ticks:
            for i, tick in enumerate(m.ticks):
                if i == 0:
                    pitch = strongPitch
                else:
                    pitch = compoundPitch if i in m.compoundTicks else weakPitch
                events.append(Note(pitch, dur=tick or clickdur, offset=now))
                now += tick
        else:
            measureSubdivs = m.subdivisions()
            subdivGroups = scorestruct.groupSubdivisions(m.timesig.parts, measureSubdivs)
            for i, part in enumerate(m.timesig.parts):
                numerator, denom = part
                # Subdivisions corresponding to this part
                partSubdivs = subdivGroups[i]
                eventsInPart, now = _processTimesigPart(
                    num=numerator, den=denom,
                    now=now,
                    quarterTempo=m.quarterTempo,
                    strongPitch=strongPitch if i == 0 else compoundPitch,
                    weakPitch=weakPitch,
                    clickdur=clickdur,
                    subdivisions=partSubdivs)
                events.extend(eventsInPart)
    voice = Voice(events)
    voice.setPlay(fade=fade)
    voice.setPlay(instr=preset)
    if transposition:
        voice.setPlay(transpose=transposition)
    return Score([voice], scorestruct=struct)
