"""
Functionality to deal with the environment in which maelzel.core is running

By environment we understand the platform, os and external applications
needed to perform certain tasks
"""
from __future__ import annotations
import sys
import os
import shutil
from typing import Optional as Opt
import emlib.misc
import logging
from typing import Optional


insideJupyter = emlib.misc.inside_jupyter()
logger = logging.getLogger("maelzel")


_linuxImageViewers = [
    ('feh', 'feh --image-bg white'),
    ('imv', 'imv -b "#ffffff"')
]


def hasBinary(binary:str) -> bool:
    if shutil.which(binary):
        return True
    return False


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


def openPngWithExternalApplication(path:str, wait=False, app:str= '') -> None:
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
    ``musescore``being present in the path. Also if music21 is installed
    and the user has set 'musescoreDirectPNGPath' pointing to an existing
    binary, that will be tried also. If musescore is not found, set
    the correct path via::

        from maelzel.core import *
        conf = getConfig()
        conf['musescorepath'] = '/path/to/musescore'
        conf.save()

    .. note::

        On macOS the path to the binary should be used, not the path to the .app
        (which is actually a directory).
    """
    from maelzel.core import workspace
    import music21 as m21
    cfg = workspace.getConfig()
    musescorepath = cfg.get('musescorepath')
    if musescorepath:
        if os.path.exists(musescorepath):
            return musescorepath
        else:
            logger.warning(f"musescorepath set to {musescorepath} in the active config, but the path does"
                           f"not exist")
    us = m21.environment.UserSettings()
    musescorepath = us['musescoreDirectPNGPath']
    if os.path.exists(musescorepath):
        return str(musescorepath)
    path = shutil.which('musescore')
    if path is not None:
        return path
    logger.warning("MuseScore not found. To fix this issue, make sure musescore is installed. "
                   "Then set the path via: \n"
                   ">>> from maelzel.core import *\n"
                   ">>> conf = getConfig()\n"
                   ">>> conf['musescorepath'] = '/path/to/musescore'\n"
                   ">>> conf.save()")
    return None
