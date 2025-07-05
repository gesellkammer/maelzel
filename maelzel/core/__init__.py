# API
from .config import CoreConfig
from .workspace import Workspace, getConfig, setScoreStruct, getWorkspace
from .event import Note, Chord, asEvent, Rest
from .chain import Chain, Voice
from .clip import Clip
from .score import Score
from .playback import play, getSession
from . import symbols
from .presetmanager import presetManager, defPreset
from ._common import logger
from . import _appstate

from maelzel.scorestruct import ScoreStruct
from maelzel.common import F


from ._lazyapi import render


__all__ = [
    'CoreConfig',
    'Workspace',
    'getWorkspace',
    'Note',
    'Chord',
    'asEvent',
    'Rest',
    'Chain',
    'Voice',
    'Score',
    'Clip',
    'play',
    'getSession',
    'render',
    'presetManager',
    'defPreset',
    'ScoreStruct',
    'setScoreStruct',
    'F',
    'getConfig',
    'logger',
    'symbols'
]


def _onFirstRun():
    import sys
    if "sphinx" in sys.modules:
        print("Building documentation...")
        return

    print("*** maelzel.core: first run")
    from maelzel.core.presetmanager import presetManager
    if '.piano' in presetManager.presetdefs:
        print("*** maelzel.core: found builtin piano soundfont; setting default instrument to '.piano'")
        rootconfig = CoreConfig.root()
        rootconfig['play.instr'] = '.piano'
        try:
            rootconfig.save()
        except FileNotFoundError:
            print(f"*** maelzel.core: Could not save config: {rootconfig.getPath()}")

    _appstate.appstate['firstRun'] = False   # state is persistent so no need to save


if _appstate.appstate['firstRun']:
    _onFirstRun()
