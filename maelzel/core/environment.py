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
from typing import Optional as Opt
import emlib.envir
import emlib.misc
import logging
from functools import cache
from typing import Optional


insideJupyter = emlib.envir.inside_jupyter()
_logger = logging.getLogger()

_linuxImageViewers = [
    ('feh', 'feh --image-bg white'),
    ('imv', 'imv -b "#ffffff"')
]


def hasBinary(binary: str) -> bool:
    if shutil.which(binary):
        return True
    return False


@cache
def preferredImageViewer() -> Opt[str]:
    """
    Returns a command string or None if no default was found.

    For that case, use emlib.misc.open_with_standard_app

    We try to find installed viewers which might work best for displaying
    a single image, possibly as fast as possible and without any added
    functionallity. If no such app is found, we return None and let the
    os decide which app to use.
    """
    if sys.platform == 'linux':
        for binary, cmd in _linuxImageViewers:
            if hasBinary(binary):
                return cmd
    return None


def openPngWithExternalApplication(path: str, wait=False, app: str = '') -> None:
    """
    Open the given png file
    """
    if app:
        return emlib.misc.open_with_app(path, app, wait=wait)

    cmd = preferredImageViewer()
    if cmd:
        emlib.misc.open_with_app(path, cmd, wait=wait)
    else:
        emlib.misc.open_with_app(path, wait=wait)


def findMusescore() -> Optional[str]:
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
                    "Then set the path via: \n"
                    ">>> from maelzel.core import *\n"
                    ">>> conf = getConfig()\n"
                    ">>> conf['musescorepath'] = '/path/to/musescore'\n"
                    ">>> conf.save()  # Save the config for future sessions")
    return None
