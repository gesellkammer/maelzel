"""
Here are things which depend only on external libraries

Here we define:

* types
* constants
* basic conversion functions

"""
import logging as _logging
import textwrap as _textwrap
import os


__all__ = (
    'UNSET',
    'MAXDUR',
    'isNumber',
    'getPath',
    'logger',
    'prettylog'
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


def prettylog(level: str, msg: str, width=80, indent=4) -> None:
    levelint = _logging.getLevelName(level)
    lines = _textwrap.wrap(msg, width=width,
                           initial_indent='\n' + ' '*(indent-1),
                           subsequent_indent=' '*indent,
                           replace_whitespace=False)

    msg = '\n'.join(lines)
    logger.log(level=levelint, msg=msg)
