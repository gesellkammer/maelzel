# API
from .config import CoreConfig
from .workspace import Workspace, getConfig, setScoreStruct
from .event import Note, Chord, asEvent, Rest
from .chain import Chain, Voice
from .score import Score
from .playback import play, getSession
from .offline import render
from . import symbols
from .presetmanager import presetManager, defPreset
from ._common import logger
from . import _appstate

from maelzel.scorestruct import ScoreStruct
from maelzel.common import F

# from . import synthevent


__all__ = [
    'CoreConfig',
    'Workspace',
    'Note',
    'Chord',
    'asEvent',
    'Rest',
    'Chain',
    'Voice',
    'Score',
    'play',
    'getSession',
    'render',
    'presetManager',
    'defPreset',
    'ScoreStruct',
    'setScoreStruct',
    'F',
    # 'synthevent',
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
        assert CoreConfig.root is not None
        CoreConfig.root['play.instr'] = '.piano'
        try:
            CoreConfig.root.save()
        except FileNotFoundError:
            print(f"*** maelzel.core: Could not save config: {CoreConfig.root.getPath()}")

    _appstate.appstate['firstRun'] = False   # state is persistent so no need to save


if _appstate.appstate['firstRun']:
    _onFirstRun()
