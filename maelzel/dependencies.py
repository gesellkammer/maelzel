from __future__ import annotations
import shutil
from pathlib import Path
import os
import sys
import logging
from datetime import datetime
from maelzel import _state


logger = logging.getLogger('maelzel')


def checkCsound() -> str:
    """ Returns True if csound is installed """
    if shutil.which("csound") is not None:
        return ""
    return "Could not find csound in the path (checked via shutil.which)"


def vampPluginsInstalled(cached=True) -> bool:
    """
    Are the needed VAMP plugins installed?

    Args:
        cached: if True, results are cached

    Returns:
        True if the needed VAMP plugins are installed

    """
    from maelzel.snd import vamptools
    return 'pyin:pyin' in vamptools.listPlugins(cached=cached)


def checkVampPlugins(fix=True) -> str:
    """
    Check if the needed VAMP plugins are installed

    Args:
        fix: if True, VAMP plugins are installed if they are not present

    Returns:
        an error string or an empty string if everything is ok

    """
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
    """
    Check if lilypond is installed

    Returns:
        an error string if something went wrong, an empty string if lilypond is
        installed
    """
    if shutil.which("lilypond") is not None:
        return ""
    return "Could not find lilypond in the path"


def _copyFiles(files: list[str], dest: str, verbose=False) -> None:
    assert os.path.isdir(dest)
    for f in files:
        if verbose:
            print(f"Copying file {f} to {dest}")
        shutil.copy(f, dest)


def maelzelRootFolder() -> Path:
    """
    Returns the root folder of the maelzel installation
    """
    return Path(os.path.split(__file__)[0]).parent


def checkCsoundPlugins(fix=True) -> str:
    """
    Checks that the needed csound plugins are installed

    Returns:
        an error string if failed, an empty string if OK
    """
    logger.debug("Checking dependencies for csound plugins")
    import risset
    logger.debug("Reading risset's main index")
    idx = risset.MainIndex(update=True)
    neededopcodes = {
        'presetinterp': 'else',
        'weightedsum': 'else',
        'poly': 'poly'
    }
    errors = []
    installedopcodes = {opcode.name for opcode in idx.defined_opcodes()
                        if opcode.installed}
    for opcodename, pluginname in neededopcodes.items():
        if opcodename not in installedopcodes:
            if fix:
                logger.info(f"Opcode {opcodename} (from plugin {pluginname}) not found, I will"
                            f"try to install it now")
                plugin = idx.plugins[pluginname]
                errmsg = idx.install_plugin(plugin)
                if errmsg:
                    logger.error(f"Could not install plugin {pluginname}")
                    errors.append(errmsg)
                else:
                    logger.info(f"Installed {pluginname} OK")
            else:
                logger.error(f"Opcode {opcodename} (from plugin {pluginname}) not found!")
                errors.append(f"Opcode {opcodename} (plugin: {pluginname}) not found")
    if errors:
        return "\n".join(errors)
    return ''


def installVampPlugins() -> None:
    """
    Install needed vamp plugins in the user folder

    Raises RuntimeError if there was an error during the installation

    To ease the installation maelzel is actually shipped with the needed
    plugins at the moment. Installation is thus reduced to copying the
    plugins to the correct folder for the platform
    """
    from maelzel.snd import vamptools
    rootfolder = maelzelRootFolder()
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
    pluginsDest = vamptools.vampFolder()
    os.makedirs(pluginsDest, exist_ok=True)
    logger.debug(f"Installing vamp plugins from {pluginspath} to {pluginsDest}")
    logger.debug(f"Plugins found: {pluginspath.glob('*.n3')}")
    _copyFiles([component.as_posix() for component in components], pluginsDest, verbose=True)
    # This step will always fail since vampyhost cached the pluginloader. We need
    # to reload the module, which we cannot do here
    vampplugins = vamptools.listPlugins(cached=False)
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
        a list of errors (or an empty list if no errors where found)
    """
    steps = [
        checkCsound,
        checkLilypond,
        lambda: checkCsoundPlugins(fix=tryfix),
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
        _state.state['last_dependency_check'] = datetime.now().isoformat()

    return errors


def checkDependenciesIfNeeded(daysSinceLastCheck=0) -> bool:
    """
    Checks dependencies if needed

    A check is needed at the first run and after a given interval

    Returns:
        True if dependencies are installed
    """
    if daysSinceLastCheck == 0 and not _state.state['first_run']:
        logger.debug('Skipping dependency check - not first run')
        return True

    lastcheck = datetime.fromisoformat(_state.state['last_dependency_check'])
    if (datetime.now() - lastcheck).days < daysSinceLastCheck:
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