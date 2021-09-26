"""
Functionality to deal with the environment in which maelzel.core is running

By environment we understand the platform, os and external applications
needed to perform certain tasks
"""
from __future__ import annotations
import sys
import shutil
from typing import Optional as Opt
import emlib.misc
import logging


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



