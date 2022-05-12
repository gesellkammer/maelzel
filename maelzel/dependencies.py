import shutil
from pathlib import Path
import os
import sys
import logging
from datetime import datetime
from maelzel._state import state

logger = logging.getLogger('maelzel')


def checkCsound() -> str:
    """ Returns True if csound is installed """
    if shutil.which("csound") is not None:
        return ""
    return "Could not find csound in the path"


def vampPluginsInstalled(cached=True) -> bool:
    from maelzel.snd import vamptools
    return 'pyin:pyin' in vamptools.list_plugins(cached=cached)


def checkVampPlugins(fix=True) -> str:
    if vampPluginsInstalled():
        return ""
    if not fix:
        return ("Vamp plugin 'pyin' not found. Install it from "
                "https://code.soundsoftware.ac.uk/projects/pyin.")
    try:
        installVampPlugins()
    except RuntimeError as err:
        logger.error(f"Error while trying to install vamp plugins: {err}")
        return str(err)


def checkLilypond() -> str:
    if shutil.which("lilypond") is not None:
        return ""
    return "Could not find lilypond in the path"


def _copyFiles(files: list[str], dest: str, verbose=False) -> None:
    assert os.path.isdir(dest)
    for f in files:
        if verbose:
            print(f"Copying file {f} to {dest}")
        shutil.copy(f, dest)


def installVampPlugins() -> None:
    """
    Install needed vamp plugins in the user folder

    Raises RuntimeError if there was an error during the installation
    """
    from maelzel.snd import vamptools
    rootfolder = Path(os.path.split(__file__)[0]).parent
    assert rootfolder.exists()
    subfolder = {
        'darwin': 'macos',
        'windows': 'windows',
        'linux': 'linux'
    }.get(sys.platform, None)
    if subfolder is None:
        raise RuntimeError(f"Platform {sys.platform} not supported")
    pluginspath = rootfolder/'data/vamp'/subfolder
    if not pluginspath.exists():
        raise RuntimeError(f"Could not find own vamp plugins. Folder: {pluginspath}")
    components = list(pluginspath.glob("*"))
    if not components:
        raise RuntimeError(f"Plugins not found in out distribution. "
                           f"Plugins folder: {pluginspath}")
    pluginsDest = vamptools.vamp_folder()
    os.makedirs(pluginsDest, exist_ok=True)
    logger.debug(f"Installing vamp plugins from {pluginspath} to {pluginsDest}")
    logger.debug(f"Plugins found: {pluginspath.glob('*.n3')}")
    _copyFiles([component.as_posix() for component in components], pluginsDest, verbose=True)
    # This step will always fail since vampyhost cached the pluginloader. We need
    # to reload the module, which we cannot do here
    vampplugins = vamptools.list_plugins(cached=False)
    if 'pyin:pyin' not in vampplugins:
        print(f"VAMP plugins were installed to the user folder: {pluginsDest}")
        print("Components installed: ")
        for comp in components:
            print(f"    {comp.name}")
        msg = ("You need to restart the python session in order for the installed "
               "plugins to be available to maelzel. These plugins provide "
               "feature extraction functions, like pyin pitch tracking")
        from maelzel import tui
        tui.panel(msg, margin=(1, 1), padding=(1, 2), bordercolor='red', title="VAMP plugins",
                  width=80)


def checkDependencies(abortIfErrors=False, tryfix=True) -> list[str]:
    """
    Checks the dependencies of all maelzel subpackages

    Returns a list of errors as strings

    Args:
        abortIfErrors: if True, abort the check if any errors are found
            Otherwise all checks are performed even if some checks return
            errors
        tryfix: if True, try to fix errors along the way, if possible. Some
            fixes might require to restart the current python session

    Returns:
        a list of errors (or an empty list if no errors found)
    """
    steps = [
        checkCsound,
        checkLilypond,
        lambda: checkVampPlugins(fix=tryfix)
    ]
    errors = []
    for step in steps:
        err = step()
        if err:
            errors.append(err)
            if abortIfErrors:
                return errors
    if not errors:
        logger.info("Check Dependencies: everything OK!")
    else:
        logger.error("Errors while checking dependencies:")
        for err in errors:
            logger.error(f"    {err}")

    if not errors:
        state['last_dependency_check'] = datetime.now().isoformat()

    return errors


def checkDependenciesIfNeeded(daysSinceLastCheck=1) -> bool:
    """
    Checks dependencies if needed

    A check is needed at the first run and after a given interval

    Returns:
        True if dependencies are installed
    """
    timeSincelast_run = datetime.now() - datetime.fromisoformat(state['last_dependency_check'])
    if timeSincelast_run.days < daysSinceLastCheck:
        logger.debug("Dependency check not needed")
        return True

    logger.warning("Maelzel - checking dependencies")
    errors = checkDependencies(abortIfErrors=False, tryfix=True)
    if not errors:
        return True
    logger.error(f"Error while checking dependencies: ")
    for err in errors:
        logger.error(f"    {err}")
    return False