# API
from .mobj import *
from .event import *
from .clip import *
from .chain import *
from .score import *
from .playback import *
from .presetmanager import *
from .workspace import *
from .config import CoreConfig
from . import _appstate

from maelzel.scorestruct import ScoreStruct

from maelzel.common import F
from . import synthevent


def _onFirstRun():
    print("*** maelzel.core: first run")
    from maelzel.core.presetmanager import presetManager
    if '_piano' in presetManager.presetdefs:
        print("*** maelzel.core: found builtin piano soundfont; setting default instrument to '_piano'")
        assert CoreConfig.root is not None
        CoreConfig.root['play.instr'] = '_piano'
        CoreConfig.root.save()
    _appstate.appstate['firstRun'] = False   # state is persistent so no need to save


rootConfig = CoreConfig(source='load')


if _appstate.appstate['firstRun']:
    _onFirstRun()

