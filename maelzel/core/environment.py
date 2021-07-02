from __future__ import annotations
import sys
import shutil
from typing import List, Optional as Opt
from emlib import misc
import logging


insideJupyter = misc.inside_jupyter()
logger = logging.getLogger("maelzel")


def hasBinary(binary:str) -> bool:
    if shutil.which(binary):
        return True
    return False


def defaultImageViewer() -> Opt[str]:
    """
    Returns a command string or None if no default was found.
    For that case, use emlib.misc.open_with_standard_app
    """
    if sys.platform == 'linux':
        if hasBinary('feh'):
            return 'feh --image-bg white'
        elif hasBinary('imv'):
            return 'imv -b "#ffffff"'
    return None


def viewPng(path:str, wait=False, app:str='') -> None:
    if app:
        return misc.open_with(path, app, wait=wait)

    cmd = defaultImageViewer()
    if cmd:
        misc.open_with(path, cmd, wait=wait)
    else:
        misc.open_with_standard_app(path, wait=wait)



