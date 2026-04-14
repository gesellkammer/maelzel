from __future__ import annotations

from maelzel import scorestruct
from maelzel.core import Workspace
from maelzel.common import F
from maelzel.core import Note, Voice, Score
import pitchtools as pt
from maelzel.core import synthevent
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from maelzel.common import time_t
    import csoundengine.synth
    from typing_extensions import Self

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
    return ClickTrack(voice, scorestruct=struct, preset=preset)


class ClickTrack(Score):

    def __init__(self,
                 voice: Voice,
                 scorestruct: scorestruct.ScoreStruct,
                 preset='.click',
                 countinPitch: str | float = "7F#"):
        super().__init__([voice], scorestruct=scorestruct)
        self.preset = preset
        self.countinPitch: float = countinPitch if isinstance(countinPitch, float) else pt.n2m(countinPitch)
        self.voices[0].setScoreStruct(scorestruct)

    @property
    def voice(self) -> Voice:
        return self.voices[0]

    def clone(self,
              voice: Voice = None,
              scorestruct: scorestruct.ScoreStruct | None = None,
              preset: str = '') -> Self:
        if scorestruct is None:
            scorestruct = self.scorestruct()
        assert scorestruct is not None
        return self.__class__(voice=voice or self.voices[0],
                              scorestruct = scorestruct,
                              preset = preset or self.preset)

    def countinEvents(self,
                      countin: int,
                      measure: int,
                      countinPitch: str | float = ""):
        if countinPitch:
            pitch: float = countinPitch if isinstance(countinPitch, float) else pt.n2m(countinPitch)
        else:
            pitch: float = self.countinPitch

        if countin == 0:
            events = self.synthEvents(skip=(measure, 0), end=(measure+1, 0))
            events = [ev.replacePitch(pitch) for ev in events]
        else:
            # number of beats at the tempo of the given measure
            measdef = self.activeScorestruct().measure(measure)
            qtempo = measdef.quarterTempo
            dur = float(60 / qtempo)
            events = []
            gain = 0.5
            for i in range(countin):
                ev = synthevent.SynthEvent(bps=[(i*dur, pitch, gain), ((i+1)*dur, pitch, gain)],
                                           instr=self.preset)
                events.append(ev)
        return events

    def play(self,
             /,
             countin: bool | int = False,  # True: countin of one measure, otherwise number of beats
             countinPitch: str | float = "",
             **kws) -> csoundengine.synth.SynthGroup:
        if not countin:
            return super().play(**kws)

        struct = self.activeScorestruct()
        skip = kws.get('skip', 0)
        skipMeas = struct.asLocation(skip)[0] if skip else 0

        # number of measures
        countEvents = self.countinEvents(countin=0 if countin is True else countin,
                                         measure=skipMeas,
                                         countinPitch=countinPitch)
        delay = countEvents[-1].end
        restEvents = self.synthEvents(delay=delay, **kws)
        allEvents = countEvents + restEvents
        return self._playEvents(allEvents, workspace=Workspace.active)