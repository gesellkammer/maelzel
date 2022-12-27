from __future__ import annotations
from configdict import ConfigDict
import emlib.misc
import sys
import os
from . import workspace


__all__ = ('appstate')


_home = os.path.expanduser("~")


appstate = ConfigDict(
    'maelzel.core.state',
    persistent=True,
    default={
        'saveCsdLastDict': _home,
        'writeLastDir': _home,
        'recLastDir': workspace.Workspace.active.recordPath(),
        'loadSndfileLastDir': _home,
        'firstRun': True,
        'soundfontLastDir': _home
    }
)

