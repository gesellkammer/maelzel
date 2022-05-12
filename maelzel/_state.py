from configdict import ConfigDict
from datetime import datetime


_defaultState = {
    'last_dependency_check': datetime(1900, 1, 1).isoformat(),
    'last_run': datetime(1900, 1, 1).isoformat()
}


state = ConfigDict("maelzel.state", _defaultState, persistent=True)
