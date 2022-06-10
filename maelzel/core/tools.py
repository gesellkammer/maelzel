from __future__ import annotations

from maelzel import scorestruct
# from ._common import *
from maelzel.rational import Rat

from .workspace import getWorkspace
from . import musicobj

from typing import TYPE_CHECKING, NamedTuple
if TYPE_CHECKING:
    from typing import Optional, Union
    from ._typedefs import time_t
    # T = TypeVar("T")


__all__ = (
    'showLilypondScore',
    'amplitudeToDynamics',
    'makeClickTrack',
    'parseNote',
    'NoteProperties'
)



def _highlightLilypond(s: str) -> str:
    # TODO
    return s


def showLilypondScore(score: str) -> None:
    """
    Display a lilypond score, either at the terminal or within a notebook

    Args:
        score: the score as text
    """
    # TODO: add highlighting, check if inside jupyter, etc.
    print(score)
    return


def amplitudeToDynamics(amp: float) -> str:
    """
    Convert an amplitude 0-1 to a dynamic according to the current dynamics curve

    Args:
        amp: an amplitude between 0-1

    Returns:
        a dynamic ('pp', 'f', etc)

    Example
    ~~~~~~~

        TODO

    """
    w = getWorkspace()
    dyncurve = w.dynamicsCurve
    return dyncurve.amp2dyn(amp)


def makeClickTrack(struct: scorestruct.ScoreStruct,
                   clickdur: time_t = None,
                   strongBeatPitch="5C",
                   weakBeatPitch="5G",
                   playpreset: str = '.click',
                   playparams: dict[str, float] = None,
                   fade=0) -> musicobj.Score:
    """
    Creates a score representing a clicktrack of the given ScoreStruct

    Args:
        struct: the ScoreStruct
        clickdur: the length of each tick. Use None to use the duration of the beat.
            **NB**: the duration of the playback can be set individually from the duration
            of the displayed pitch
        strongBeatPitch: the pitch to use as a strong tick
        weakBeatPitch: the pitch to use as a weak tick
        playpreset: the preset instr to use for playback. The default plays the given
            pitches two octaves higher as very short clicks
        playparams: parameters passed to the play preset, if needed
        fade: a fadetime for the clicks

    Returns:
        a maelzel.core Score

    Example
    ~~~~~~~

        TODO
    """
    now = 0
    events = []
    for m in struct.measuredefs:
        num, den = m.timesig
        if den  == 4:
            for i, n in enumerate(range(m.timesig[0])):
                pitch = strongBeatPitch if i == 0 else weakBeatPitch
                ev = musicobj.Note(pitch, start=now, dur=clickdur or 1).setPlay(fade=(0, 0.1))
                events.append(ev)
                now += 1
        elif den == 8:
            for i, n in enumerate(range(m.timesig[0])):
                pitch = strongBeatPitch if i == 0 else weakBeatPitch
                ev = musicobj.Note(pitch, start=now, dur=clickdur or 0.5).setPlay(fade=(0, 0.1))
                events.append(ev)
                now += 0.5
        elif den == 16:
            if m.quarterTempo > 80:
                dur = clickdur or m.durationBeats()
                ev = musicobj.Note(strongBeatPitch, dur=dur, start=now)
                events.append(ev)
                now += m.durationBeats()
            else:
                beats = m.subdivisions()
                for i, beat in enumerate(beats):
                    pitch = strongBeatPitch if i == 0 else weakBeatPitch
                    ev = musicobj.Note(pitch, dur=clickdur or beat, start=now)
                    events.append(ev)
                    now += beat
        else:
            raise ValueError(f"Timesig {m.timesig} not supported")
    voice = musicobj.Voice(events)
    voice.setPlay(fade=fade)
    if playpreset:
        voice.setPlay(instr=playpreset, params=playparams)
    return musicobj.Score([voice], scorestruct=struct)


class NoteProperties(NamedTuple):
    """
    Represents the parsed properties of a note, as returned by :func:`parseNote`

    The format to parse is Pitch[:dur][:property1][...]

    .. seealso:: :func:`parseNote`
    """
    pitch: Union[str, list[str]]
    """A pitch or a list of pitches"""

    dur: Optional[Rat]
    """An optional duration"""

    properties: Optional[dict[str, str]]
    """Any other properties"""


def parseNote(s: str) -> NoteProperties:
    """
    Parse a note definition string with optional duration and other properties

    ============================== ========= ====  ===========
    Note                           Pitch     Dur   Properties
    ============================== ========= ====  ===========
    4c#                            4C#       None  None
    4F+:0.5                        4F+       0.5   None
    4G:1/3                         4G        1/3   None
    4Bb-:mf                        4B-       None  {'dynamic':'mf'}
    4G-:0.4:ff:articulation=accent 4G-       0.4   {'dynamic':'ff',
                                                    'articulation':'accent'}
    4F#,4A                         [4F#, 4A] None  None
    4G:^                           4G        None  {'articulation': 'accent'}
    ============================== ========= ====  ===========


    Args:
        s: the note definition to parse

    Returns:
        a NoteProperties object with the result
    """
    dur, properties = None, None
    if ":" not in s:
        pitch = s
    else:
        pitch, rest = s.split(":", maxsplit=1)
        parts = rest.split(":")
        properties = {}
        for part in parts:
            try:
                dur = Rat(part)
            except ValueError:
                if part in _knownDynamics:
                    properties['dynamic'] = part
                elif part == 'gliss':
                    properties['gliss'] = True
                elif part == 'tied':
                    properties['tied'] = True
                elif "=" in part:
                    key, value = part.split("=", maxsplit=1)
                    properties[key] = value
        if not properties:
            properties = None
    notename = [p.strip() for p in pitch.split(",",)] if "," in pitch else pitch
    return NoteProperties(pitch=notename, dur=dur, properties=properties)


_knownDynamics = {
    'pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff', 'n'
}


