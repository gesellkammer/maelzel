# API
from .config import CoreConfig
from .workspace import *
from .mobj import *
from .event import *
from .clip import *
from .chain import *
from .score import *
from .playback import play, playSession
from .offline import render
from .presetmanager import *
from . import _appstate

from maelzel.scorestruct import ScoreStruct

from maelzel.common import F
from . import synthevent


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

