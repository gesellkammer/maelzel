# API
from .config import CoreConfig
from .workspace import Workspace, getConfig
from .event import Note, Chord, asEvent
from .chain import Chain, Voice
from .score import Score
from .playback import play, getSession
from .offline import render
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
    'Chain',
    'Voice',
    'Score',
    'play',
    'getSession',
    'render',
    'presetManager',
    'defPreset',
    'ScoreStruct',
    'F',
    # 'synthevent',
    'getConfig',
    'logger'
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
