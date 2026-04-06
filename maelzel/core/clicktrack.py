from __future__ import annotations

from maelzel import scorestruct
from maelzel.core import Workspace
from maelzel.common import F
from maelzel.core import Note, Voice, Score
import pitchtools as pt
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from maelzel.common import time_t
    import csoundengine.synth


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
    if minMeasures and minMeasures > struct.numMeasures():
        struct = struct.copy()
        struct.addMeasure(count=minMeasures - struct.numMeasures())

    weight2pitch = {
        3: strongPitch,
        2: compoundPitch,
        1: weakPitch,
        0: weakPitch
    }
    events = []
    now = F(0)
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
            beats = m.beatStructure()

            for beat in beats:
                pitch = weight2pitch[beat.weight]
                event = Note(pitch, dur=clickdur or beat.duration, offset=now)
                events.append(event)
                now += beat.duration

    voice = Voice(events)
    voice.setPlay(fade=fade, instr=preset)
    if transposition:
        voice.setPlay(transpose=transposition)
    return ClickTrack([voice], scorestruct=struct)


class ClickTrack(Score):

    def play(self,
             /,
             countin: int | Sequence[int | F | float] = 0,
             countinPitch: str | float = "7F#",
             **kws) -> csoundengine.synth.SynthGroup:
        if not countin:
            return super().play(**kws)

        struct = self.activeScorestruct()
        if isinstance(countin, int):
            # number of measures
            firstMeasureDur = float(struct.locationToBeat(1, 0))
            countEvents = self.synthEvents(end=firstMeasureDur, **kws)
            pitch = countinPitch if isinstance(countinPitch, float) else pt.n2m(countinPitch)
            countEvents = [ev.replacePitch(pitch) for ev in countEvents]
            print(countEvents)
            restEvents = self.synthEvents(delay=float(struct.locationToTime(1, 0)), **kws)
            allEvents = countEvents + restEvents
        else:
            assert isinstance(countin, (list, tuple))
            # These are durations
            # TODO
            allEvents = []

        return self._playEvents(allEvents, workspace=Workspace.active)
