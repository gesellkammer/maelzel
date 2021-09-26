from __future__ import annotations
from configdict import ConfigDict
import emlib.misc
import sys
import os
from . import workspace


__all__ = ('appstate')


with ConfigDict("maelzel.core.state") as appstate:
    home = os.path.expanduser("~")
    appstate.addKey('saveCsdLastDir', home)
    appstate.addKey('writeLastDir', home)
    appstate.addKey('recLastDir', workspace.recordPath())
    if sys.platform == 'linux':
        appstate.addKey('soundfontLastDirectory',
                        emlib.misc.first_existing_path("/usr/share/sounds/sf2",
                                                       "~/Documents"))
    else:
        appstate.addKey('soundfontLastDirectory', home)
