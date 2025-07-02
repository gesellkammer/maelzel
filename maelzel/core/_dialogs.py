from __future__ import annotations
import os
from ._appstate import appstate as _appstate
from ._common import logger
from maelzel.core import _tools
from emlib import envir

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional


def selectFromList(options: list[str],
                   title="",
                   default=None,
                   gui: bool | None = None,
                   ) -> str | None:
    """
    Select an option from a list of options

    None is returned if no selection was done

    Args:
        options: the options as a list of str
        title: the title of the selection widget (might not be used in text mode)
        default: the value returned if no selection was made
        gui: if None, detect the environment. Otherwise, if True a gui dialog is forced
            and if False a terminal dialog is forced

    Returns:
        the option selected, or *default* if not selection was done
    """
    if gui is None:
        if envir.running_inside_terminal() and envir.is_interactive_session():
            gui = False
        else:
            gui = True

    if gui:
        import emlib.dialogs
        return emlib.dialogs.selectItem(options, title=title) or default
    else:
        import maelzel.tui
        idx = maelzel.tui.menu(options)
        return default if idx is None else options[idx]


def selectFileForSave(key: str, filter="All (*.*)", prompt="Save File"
                      ) -> str | None:
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


def selectFileForOpen(key: str, filter="All (*.*)", prompt="Open"
                      ) -> str:
    """
    Select a file for open via a gui dialog, remember the last directory

    Args:
        key: the key to use to remember the last directory
        filter: for example "Images (*.png, *.jpg);; Videos (*.mp4)"
        prompt: title of the dialog


    Returns:
        the selected file, or None if the operation was cancelled
    """
    if _tools.checkBuildingDocumentation(logger):
        return ''
    import emlib.dialogs
    lastdir = _appstate.get(key, '')
    selected = emlib.dialogs.selectFile(filter=filter, directory=lastdir, title=prompt)
    if selected:
        _appstate[key] = os.path.split(selected)[0]
    return selected


def selectSndfileForOpen(prompt="Open Soundfile",
                         filter='Audio (*.wav, *.aif, *.flac, *.mp3)',
                         ) -> str:
    """
    Select a soundfile for open via a gui dialog, remember the last directory

    Args:
        prompt: title of the dialog
        filter: the file types to accept

    Returns:
        the selected file, or an empty string if the operation was cancelled

    .. seealso:: :func:`~maelzel.core.tools.selectFileForOpen`
    """
    return selectFileForOpen(key='loadSndfileLastDir', filter=filter, prompt=prompt)


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
