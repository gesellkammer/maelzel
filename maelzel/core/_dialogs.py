from __future__ import annotations
import os
from . import environment
from ._appstate import appstate as _appstate
from ._common import logger
from maelzel.core import _util

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Optional


def selectFromList(options: Sequence[str], title="", default=None) -> Optional[str]:
    import emlib.dialogs
    if environment.insideJupyter:
        return emlib.dialogs.selectItem(options, title=title) or default
    else:
        # TODO: use tty tools, like fzf
        return emlib.dialogs.selectItem(options, title=title) or default


def selectFileForSave(key:str, filter="All (*.*)", prompt="Save File") -> Optional[str]:
    """
    Select a file for open via a gui dialog, remember the last directory

    Args:
        key: the key to use to remember the last directory
        filter: for example "Images (*.png, *.jpg);; Videos (*.mp4)"
        prompt: title of the dialog

    Returns:
        the selected file, or None if the operation was cancelled
    """
    lastdir = _appstate[key]
    import emlib.dialogs
    outfile = emlib.dialogs.saveDialog(filter=filter, directory=lastdir, title=prompt)
    if outfile:
        _appstate[key] = os.path.split(outfile)[0]
    return outfile


def selectFileForOpen(key: str, filter="All (*.*)", prompt="Open", ifcancel:str=None
                      ) -> Optional[str]:
    """
    Select a file for open via a gui dialog, remember the last directory

    Args:
        key: the key to use to remember the last directory
        filter: for example "Images (*.png, *.jpg);; Videos (*.mp4)"
        prompt: title of the dialog
        ifcancel: if given and the operation is cancelled a ValueError
            with this as message is raised

    Returns:
        the selected file, or None if the operation was cancelled
    """
    if _util.checkBuildingDocumentation(logger):
        return None
    import emlib.dialogs
    lastdir = _appstate.get(key)
    selected = emlib.dialogs.selectFile(filter=filter, directory=lastdir, title=prompt)
    if selected:
        _appstate[key] = os.path.split(selected)[0]
    elif ifcancel is not None:
        raise ValueError(ifcancel)
    return selected


def selectSndfileForOpen(prompt="Open Soundfile",
                         filter='Audio (*.wav, *.aif, *.flac, *.mp3)',
                         ifcancel: str = None
                         ) -> Optional[str]:
    """
    Select a soundfile for open via a gui dialog, remember the last directory

    Args:
        prompt: title of the dialog
        filter: the file types to accept
        ifcancel: if given and the operation is cacelled a ValueError with this message
            is raised

    Returns:
        the selected file, or None if the operation was cancelled

    .. seealso:: :func:`~maelzel.core.tools.selectFileForOpen`
    """
    return selectFileForOpen(key='loadSndfileLastDir', filter=filter, ifcancel=ifcancel,
                             prompt=prompt)


def saveRecordingDialog(prompt="Save Recording") -> Optional[str]:
    """
    Open a native dialog to save a soundfile

    Args:
        prompt: the text to use as prompt

    Returns:
        The selected path, or None if selection was aborted

    """
    return selectFileForSave("recLastDir", "Audio (*.wav, *.aif, *.flac)",
                             prompt=prompt)
