from __future__ import annotations
import os
from ._appstate import appstate as _appstate
from ._common import logger
from maelzel.core import _tools
from emlib import envir

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Optional


def selectFromList(options: list[str],
                   title="",
                   default=None,
                   gui: bool = None,
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
        ensure: if True, an exception is raised if the selection is cancelled

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
        from maelzel import tui
        idx = tui.menu(options)
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
                         ) -> str | None:
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
