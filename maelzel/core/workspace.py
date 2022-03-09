"""

Workspace
=========

maelzel.core is organized on the idea of a workspace. A workspace contains the current state
(an active scorestrucutre, an active config). Many actions, like note playback, notation rendering,
etc., use the active workspace to determine tempo, score structure, default playback instrument, etc.

At any moment there is always an active workspace. This can be accessed via :func:`activeWorkspace`.
At the start of a session a default workspace (the 'root' workspace) is created, based on the default
config and a default score structure.

"""
from __future__ import annotations
import os
import pitchtools
import appdirs as _appdirs

from ._common import logger, UNSET, Rat
from ._typedefs import time_t
from .config import rootConfig
from maelzel.music.dynamics import DynamicCurve
from maelzel.scorestruct import ScoreStruct

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Any
    import configdict


def _resetCache() -> None:
    from .musicobj import resetImageCache
    resetImageCache()


___all__ = (
    'Workspace',
    'activeWorkspace',
    'activeConfig',
    'newConfig',
    'setScoreStruct',
    'setTempo',
)


class Workspace:
    """
    Create a new Workspace / get an existing Workspace.

    If the name refers to an existing Workspace, the already existing
    Workspace is returned

    Args:
        name: the name of the workspace, or nothing to create an unique name.
            The name 'root' refers to the root Workspace
        scorestruct: the ScoreStruct.
        config: the active config for this workspace
        renderer: will be set to the active offline renderer while rendering offline
        dynamicCurve: a DynamicCurve used to map amplitude to dynamic expressions
        activate: if True, make this Workpsace active

    Attributes:
        name: the name of this Workspace
        renderer: if not None, the active offline renderer
        config: the active config for this Workspace
        scorestruct: the active ScoreStruct
        dynamicsCurve: a DynamicCurve, mapping amplitude to dynamic expression
    """
    root: Workspace = None

    _initDone: bool = False
    _active: Workspace
    _counter: int = 0

    def __init__(self,
                 name: str = '',
                 scorestruct:Optional[ScoreStruct]=None,
                 config:configdict.ConfigDict=None,
                 renderer: Any = None,
                 dynamicCurve: DynamicCurve = None,
                 activate=True):

        if not name:
            name = Workspace._createUniqueName()
        self.name = name
        self.renderer = renderer
        self.config = config or activeConfig()
        if scorestruct is None:
            scorestruct = ScoreStruct.fromTimesig((4, 4), quarterTempo=60)
        self._scorestruct = scorestruct
        self.dynamicsCurve = dynamicCurve or DynamicCurve.fromdescr(shape=self.config.get('dynamicsCurve.shape', 'expon(3.0)'))
        if activate:
            self.activate()

    def __repr__(self):
        parts = [f"name={self.name}, scorestruct={self.scorestruct}"]
        return f"Workspace({', '.join(parts)})"
            
    @property
    def scorestruct(self) -> ScoreStruct:
        """Returns the current ScoreSctruct"""
        return self._scorestruct
    
    @scorestruct.setter
    def scorestruct(self, s: ScoreStruct):
        _resetCache()
        self._scorestruct = s

    @classmethod
    def _createUniqueName(cls) -> str:
        cls._counter += 1
        return f"Workspace-{cls._counter}"

    @property
    def a4(self) -> float:
        return self.config['A4']

    @a4.setter
    def a4(self, value:float):
        self.config['A4'] = value
        if self.isActive():
            pitchtools.set_reference_freq(value)

    @classmethod
    def getActive(cls) -> Workspace:
        """Get the active Workspace

        To set the active workspace, call `.activate()` on a previously
        created Workspace
        """
        return cls._active

    def getTempo(self, measureNum=0) -> float:
        """Get the quarter-note tempo at the given measure"""
        return float(self.scorestruct.getMeasureDef(measureNum).quarterTempo)

    def activate(self) -> None:
        """Make this the active Workspace"""
        pitchtools.set_reference_freq(self.a4)
        Workspace._active = self

    def isActive(self) -> bool:
        """Is this the active Workspace?"""
        return activeWorkspace() is self
    
    def clone(self, name:str=None, config:configdict.ConfigDict=UNSET,
              scorestruct:ScoreStruct=UNSET, activate=True
              ) -> Workspace:
        """
        Clone this Workspace
        """
        if config is UNSET:
            config = self.config.clone(persistent=False, cloneCallbacks=True)
        if scorestruct is UNSET:
            scorestruct = self.scorestruct.copy()
        if not name or name is UNSET:
            name = Workspace._createUniqueName()
        return Workspace(name,
                         config=config,
                         scorestruct=scorestruct or self.scorestruct,
                         activate=activate)


def _init() -> None:
    if Workspace._initDone:
        logger.debug("init was already done")
        return
    Workspace._initDone = True
    w = Workspace("root", config=rootConfig)
    Workspace.root = w
    w.activate()


def activeWorkspace() -> Workspace:
    """
    Get the active workspace
    """
    return Workspace.getActive()


def setTempo(quarterTempo:float, measureNum=0) -> None:
    """
    Set the current tempo. 
    
    This is only possible if the currently active ScoreStruct has only 
    one initial tempo
    """
    w = activeWorkspace()
    w.scorestruct.setTempo(quarterTempo, measureNum=measureNum)


def activeConfig() -> configdict.ConfigDict:
    """
    Return the active config.
    """
    return activeWorkspace().config


def newConfig(updates:dict=None, cloneCurrent=True, name:str=None) -> configdict.ConfigDict:
    """
    Clone the current Workspace with a new config and set it as active

    The returned config is a `configdict.ConfigDict <https://configdict.readthedocs.io/en/latest/api/configdict.configdict.ConfigDict.html>`_,
    which behaves like a dictionary but allows only a specific set of keys.

    This new config will not be persistent. To make persistent changes, modify
    the root config::

        >>> from maelzel.core import *
        >>> rootConfig.edit()

    **maelzel.core** is organized around the idea of an active workspace (see
    :func:`~maelzel.core.workspace.activeWorkspace`: at any moment there is one active workspace.
    Each workspace has a config associated with it. By calling `newConfig`, a new workspace
    is created with, either a clone of the previous config (if called with
    `cloneCurrent=True`) or of the root config. All other attributes of
    a workspace are inherited from the previous workspace. To access the newly
    created workspace, call :func:`~maelzel.core.workspace.activeWorkspace`.

    Args:
        cloneCurrent: if True, clones the current config otherwise clones
            the default
        name: name of the workspace created with this config. If no name is given,
            a new unique name is created (this name can be found
            via ``activeConfig().name``)
        updates: if given, the new config is updated with this dict

    Returns:
        the new config

    For more information on configuration, see :ref:`Configuration<config>`

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> cfg = newConfig()
        >>> cfg['play.numChannels'] = 4
        # This is the same as
        >>> Workspace(config=activeConfig().clone(updates={'play.numChannels': 4}))
    """
    w = activeWorkspace()
    if cloneCurrent:
        config = w.config.clone(updates=updates, cloneCallbacks=True)
    else:
        rootWorkspace = Workspace.root
        config = rootWorkspace.config.clone(updates=updates, cloneCallbacks=True)
    newWorkspace = w.clone(name=name, config=config, scorestruct=w.scorestruct)
    return newWorkspace.config


def activeScoreStruct() -> ScoreStruct:
    """
    Returns the active ScoreStruct (which defines tempo and time signatures)

    If no ScoreStruct has been set explicitely, a default is always active which
    creates an endless 4/4 score with tempo q=60

    .. note::
        To modify the current structure a new structure can be set via
        ``setScoreStruct(newscore)``
        Alternatively, if the current score structure has no multiple tempos,
        the tempo can be modified via `setTempo`.

    """
    s = activeWorkspace().scorestruct
    assert s is not None
    return s


def setScoreStruct(s: ScoreStruct) -> None:
    """
    Sets the current score structure
    """
    activeWorkspace().scorestruct = s


def _presetsPath() -> str:
    datadirbase = _appdirs.user_data_dir("maelzel")
    path = os.path.join(datadirbase, "core", "presets")
    return path


def presetsPath() -> str:
    """ Returns the path were instrument presets are read/written"""
    userpath = activeConfig()['play.presetsPath']
    if userpath:
        return userpath
    return _presetsPath()


def recordPath() -> str:
    """
    The path where temporary recordings are saved

    We do not use the temporary folder because it is wiped regularly
    and the user might want to access a recording after rebooting.
    The returned folder is guaranteed to exist

    The default record path can be customized by modifying the config
    'rec.path'
    """
    userpath = activeConfig()['rec.path']
    if userpath:
        path = userpath
    else:
        path = _appdirs.user_data_dir(appname="maelzel", version="recordings")
    if not os.path.exists(path):
        os.makedirs(path)
    return path



_init()