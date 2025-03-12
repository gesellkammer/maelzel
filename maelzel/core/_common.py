"""
Here are things which depend only on external libraries

Here we define:

* types
* constants
* basic conversion functions

"""
import logging as _logging
import textwrap as _textwrap

from maelzel.common import getLogger, F
import os


__all__ = (
    'MAXDUR',
    'isNumber',
    'getPath',
    'logger',
    'prettylog',
)


MAXDUR = 99999


def isNumber(x) -> bool:
    """ is x builtin number? (int, float or Fraction) """
    return isinstance(x, (int, float, F))


def getPath(s: str) -> str:
    if s == "?":
        from emlib import dialogs
        return dialogs.selectFile()
    else:
        return os.path.expanduser(s)


def prettylog(level: str, msg: str, logger: str|_logging.Logger='maelzel.core', width=80, indent=4) -> None:
    loggerinstance = logger if isinstance(logger, _logging.Logger) else _logging.getLogger(logger)
    levelint = _logging.getLevelName(level)
    lines = _textwrap.wrap(msg, width=width,
                           initial_indent='\n' + ' '*(indent-1),
                           subsequent_indent=' '*indent,
                           replace_whitespace=False)

    msg = '\n'.join(lines)
    loggerinstance.log(level=levelint, msg=msg)


# _logdir = appdirs.user_log_dir('maelzel-core')
# os.makedirs(_logdir, exist_ok=True)
# filelog = os.path.join(_logdir, 'maelzel-core.log')

logger = getLogger("maelzel.core")
