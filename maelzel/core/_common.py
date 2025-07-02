"""
Here are things which depend only on external libraries

Here we define:

* types
* constants
* basic conversion functions

"""

from maelzel.common import getLogger
import os


__all__ = (
    'MAXDUR',
    'getPath',
    'logger',
)


MAXDUR = 99999


def getPath(s: str) -> str:
    if s == "?":
        from emlib import dialogs
        return dialogs.selectFile()
    else:
        return os.path.expanduser(s)


logger = getLogger("maelzel.core")
