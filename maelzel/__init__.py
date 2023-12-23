from maelzel import _state
import sys


# Only check dependencies on first run
if _state.isFirstSession() and "sphinx" not in sys.modules:

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

    # logging.basicConfig(level=logging.DEBUG)
    from maelzel import dependencies
    errors = dependencies.checkDependencies()
    if not errors:
        print("\n*** All dependencies are installed! ***")
