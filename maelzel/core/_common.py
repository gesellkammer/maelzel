"""
Here are things which depend only on external libraries

Here we define:

* types
* constants
* basic conversion functions

"""
import logging as _logging
import os


__all__ = (
    'UNSET',
    'MAXDUR',
    'isNumber',
    'getPath',
    'logger',
)


MAXDUR = 99999


class _UNSET:
    def __repr__(self):
        return 'UNSET'

    def __bool__(self):
        return False


UNSET = _UNSET()

logger = _logging.getLogger(f"maelzel.core")


def isNumber(x) -> bool:
    """ is x builtin number? (int, float or Fraction) """
    return isinstance(x, (int, float, F))


def getPath(s: str) -> str:
    if s == "?":
        from emlib import dialogs
        return dialogs.selectFile()
    else:
        return os.path.expanduser(s)
