from __future__ import annotations
from configdict import ConfigDict
import os
import appdirs


__all__ = ('appstate', )


_home = os.path.expanduser("~")
_recdir = appdirs.user_data_dir(appname="maelzel", version="recordings")


appstate = ConfigDict(
    'maelzel.core.state',
    persistent=True,
    default={
        'saveCsdLastDict': _home,
        'writeLastDir': _home,
        'recLastDir': _recdir,
        'loadSndfileLastDir': _home,
        'firstRun': True,
        'soundfontLastDir': _home
    }
)
