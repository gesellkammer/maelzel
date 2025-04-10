from configdict import ConfigDict
from functools import cache


_defaultState = {
    'last_dependency_check': '1900-01-01T00:00:00',
    'first_run': True
}


state = ConfigDict("maelzel.state", _defaultState, persistent=True)


@cache
def isFirstSession() -> bool:
    """
    Returns True if this is the first run

    This value remains the same for the first session
    """
    if state['first_run']:
        state['first_run'] = False
        return True
    return False
