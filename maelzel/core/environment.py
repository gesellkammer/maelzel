"""
Functionality to deal with the environment in which `maelzel.core` is running

"""

# ************************************************************************************
# **** NB: This module should not import anything from the maelzel.core namespace ****
# ************************************************************************************


from __future__ import annotations
import os
import shutil
from maelzel import _util


insideJupyter = _util.pythonSessionType() == 'jupyter'


def hasBinary(binary: str) -> bool:
    """
    Check if the given binary is available in the system's PATH.

    Args:
        binary (str): The name of the binary to check.

    Returns:
        bool: True if the binary is found, False otherwise.
    """
    return bool(shutil.which(binary))


def openPngWithExternalApplication(path: str, wait=False, app='') -> None:
    """
    Open the given png file
    """
    import emlib.misc
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
    import logging
    cfg = workspace.getConfig()
    musescorepath = cfg.get('musescorepath')
    _logger = logging.getLogger(__file__)
    if musescorepath:
        if os.path.exists(musescorepath):
            return musescorepath
        else:
            _logger.warning(f"musescorepath set to {musescorepath} in the active config, but the path does"
                            f"not exist")

    possibleNames = ('musescore', 'MuseScore')
    for name in possibleNames:
        if (path := shutil.which(name)) is not None:
            return path

    _logger.warning("MuseScore not found. Tried to find 'musescore' or 'MuseScore' in the path, "
                    "without success. To fix this issue, make sure MuseScore is installed. "
                    "Then set the path: \n"
                    ">>> from maelzel.core import *\n"
                    ">>> conf = getConfig()\n"
                    ">>> conf['musescorepath'] = '/path/to/musescore'\n"
                    ">>> conf.save()  # Save the config for future sessions")
    return None
