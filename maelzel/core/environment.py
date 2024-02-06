"""
Functionality to deal with the environment in which `maelzel.core` is running

"""

# ************************************************************************************
# **** NB: This module should not import anything from the maelzel.core namespace ****
# ************************************************************************************


from __future__ import annotations
import sys
import os
import shutil
import emlib.misc
import logging
from functools import cache
from maelzel._util import pythonSessionType


insideJupyter = pythonSessionType() == 'jupyter'
_logger = logging.getLogger()


def hasBinary(binary: str) -> bool:
    if shutil.which(binary):
        return True
    return False


def openPngWithExternalApplication(path: str, wait=False, app: str = '') -> None:
    """
    Open the given png file
    """
    if app:
        return emlib.misc.open_with_app(path, app, wait=wait)
    emlib.misc.open_with_app(path, wait=wait)


def findMusescore() -> str | None:
    """
    Tries to find musescore, returns the path to the executable or None

    We rely on the active config (key: ``musescorepath``) or a binary
    ``musescore`` being present in the path.

    If musescore is not found, set the correct path via::

        from maelzel.core import *
        conf = getConfig()
        conf['musescorepath'] = '/path/to/musescore'
        conf.save()

    .. note::

        On macOS the path to the binary should be used, not the path to the .app
        (which is actually a directory).
    """
    from maelzel.core import workspace
    cfg = workspace.getConfig()
    musescorepath = cfg.get('musescorepath')
    if musescorepath:
        if os.path.exists(musescorepath):
            return musescorepath
        else:
            _logger.warning(f"musescorepath set to {musescorepath} in the active config, but the path does"
                            f"not exist")

    if (path := shutil.which('musescore')) is not None:
        return path
    if (path := shutil.which('MuseScore')) is not None:
        return path
        
    _logger.warning("MuseScore not found. Tried to find 'musescore' or 'MuseScore' in the path, "
                    "without success. To fix this issue, make sure MuseScore is installed. "
                    "Then set the path: \n"
                    ">>> from maelzel.core import *\n"
                    ">>> conf = getConfig()\n"
                    ">>> conf['musescorepath'] = '/path/to/musescore'\n"
                    ">>> conf.save()  # Save the config for future sessions")
    return None
