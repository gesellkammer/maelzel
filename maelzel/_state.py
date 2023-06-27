from configdict import ConfigDict
from datetime import datetime
from functools import cache

_defaultState = {
    'last_dependency_check': datetime(1900, 1, 1).isoformat(),
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

