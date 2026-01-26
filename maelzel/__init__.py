import sys

from maelzel import _state
from maelzel import _util


def _firstRun():
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format='[%(name)s:%(filename)s:%(lineno)s - %(funcName)s] %(message)s')

    parent = logging.getLogger('maelzel')
    parent.setLevel('DEBUG')
    parentHandler = logging.StreamHandler()
    parent.addHandler(parentHandler)

    msg = '''
    This is the first run. Checking external dependencies...
    '''

    from maelzel import tui
    tui.panel(title='Welcome to maelzel!', titlealign='left',
              text=msg, padding=(1, 1))

    from maelzel import dependencies
    errors = dependencies.checkDependencies()
    if not errors:
        print("\n*** All dependencies are installed! ***")


if "sphinx" not in sys.modules:
    # Only check if not building docs
    if _state.isFirstSession():
        _firstRun()
    else:
        lastVersion = _state.state['last_version']
        lastVersionTup = _util.splitVersion(lastVersion) if lastVersion else (0, 0, 0)
        from importlib.metadata import version
        currVersion = version("maelzel")
        currVersionTup = _util.splitVersion(currVersion)
        if currVersionTup > lastVersionTup:
            from maelzel import dependencies
            dependencies.checkDependencies()  # <-- this updates _state.state




