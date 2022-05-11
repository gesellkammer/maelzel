from __future__ import annotations

import os
import pitchtools as pt

import emlib.img
import emlib.misc
import emlib.dialogs

from maelzel import scorestruct
from maelzel.rational import Rat
from ._common import *
from . import _util
from . import environment
from .state import appstate as _appstate
from .workspace import getWorkspace
from . import musicobj

from typing import TYPE_CHECKING, NamedTuple
if TYPE_CHECKING:
    from typing import *
    from ._typedefs import *
    T = TypeVar("T")


def _highlightLilypond(s: str) -> str:
    # TODO
    return s


def showLilypondScore(score: str) -> None:
    # TODO
    print(score)
    return


def amplitudeToDynamics(amp: float) -> str:
    w = getWorkspace()
    dyncurve = w.dynamicsCurve
    return dyncurve.amp2dyn(amp)


def selectFromList(options: Sequence[str], title="", default=None) -> Optional[str]:
    if environment.insideJupyter():
        return emlib.dialogs.selectItem(options, title=title) or default
    else:
        # TODO: use tty tools, like fzf
        return emlib.dialogs.selectItem(options, title=title) or default


def selectFileForSave(key:str, filter="All (*.*)", prompt="Save File") -> Optional[str]:
    """
    Select a file for open via a gui dialog, remember the last directory

    Args:
        key: the key to use to remember the last directory
        filter: for example "Images (*.png, *.jpg);; Videos (*.mp4)"
        prompt: title of the dialog

    Returns:
        the selected file, or None if the operation was cancelled
    """

    lastdir = _appstate[key]
    outfile = emlib.dialogs.saveDialog(filter=filter, directory=lastdir, title=prompt)
    if outfile:
        _appstate[key] = os.path.split(outfile)[0]
    return outfile


def selectFileForOpen(key: str, filter="All (*.*)", prompt="Open", ifcancel:str=None
                      ) -> Optional[str]:
    """
    Select a file for open via a gui dialog, remember the last directory

    Args:
        key: the key to use to remember the last directory
        filter: for example "Images (*.png, *.jpg);; Videos (*.mp4)"
        prompt: title of the dialog
        ifcancel: if given and the operation is cancelled a ValueError
            with this as message is raised

    Returns:
        the selected file, or None if the operation was cancelled
    """
    if _util.checkBuildingDocumentation(logger):
        return None
    lastdir = _appstate.get(key)
    selected = emlib.dialogs.selectFile(filter=filter, directory=lastdir, title=prompt)
    if selected:
        _appstate[key] = os.path.split(selected)[0]
    elif ifcancel is not None:
        raise ValueError(ifcancel)
    return selected


def selectSndfileForOpen(prompt="Open Soundfile",
                         filter='Audio (*.wav, *.aif, *.flac, *.mp3)',
                         ifcancel: str = None
                         ) -> Optional[str]:
    """
    Select a soundfile for open via a gui dialog, remember the last directory

    Args:
        prompt: title of the dialog
        filter: the file types to accept
        ifcancel: if given and the operation is cacelled a ValueError with this message
            is raised

    Returns:
        the selected file, or None if the operation was cancelled

    .. seealso:: :func:`~maelzel.core.tools.selectFileForOpen`
    """
    return selectFileForOpen(key='loadSndfileLastDir', filter=filter, ifcancel=ifcancel)


def saveRecordingDialog(prompt="Save Recording") -> Optional[str]:
    return selectFileForSave("recLastDir", "Audio (*.wav, *.aif, *.flac)",
                             prompt=prompt)


def makeClickTrack(struct: scorestruct.ScoreStruct,
                   clickdur: time_t = None,
                   strongBeatPitch="5C",
                   weakBeatPitch="5G",
                   playpreset: str = '.click',
                   playparams: Dict[str, float] = None,
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

    Returns:
        a Voice
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
    pitch: Union[str, List[str]]
    dur: Optional[Rat]
    properties: Optional[Dict[str, str]]


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
    if not ":" in s:
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


def pngShow(pngpath:str, forceExternal=False, app:str='') -> None:
    """
    Show a png either with an external app or inside jupyter

    Args:
        pngpath: the path to a png file
        forceExternal: if True, it will show in an external app even
            inside jupyter. Otherwise it will show inside an external
            app if running a normal session and show an embedded
            image if running inside a notebook
        app: used if a specific external app is needed. Otherwise the os
            defined app is used
    """
    if environment.insideJupyter and not forceExternal:
        from . import jupytertools
        jupytertools.showPng(pngpath)
    else:
        environment.openPngWithExternalApplication(pngpath, app=app)
