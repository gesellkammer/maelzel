"""
Module text
"""
from __future__ import annotations
from ._common import logger, UNSET
import appdirs as _appdirs
import weakref as _weakref
import os
import pitchtools as pt
from .config import rootConfig
from maelzel.music.dynamics import DynamicCurve
from maelzel.scorestruct import ScoreStruct

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    import configdict


def _resetCache() -> None:
    from .musicobj import resetImageCache
    resetImageCache()


class Workspace:
    _initDone: bool = False
    _workspaces: Dict[str, _weakref.ReferenceType[Workspace]] = {}
    _current: Workspace
    _root: Workspace

    def __new__(cls, name: str,
                scorestruct: Optional[ScoreStruct]=None,
                config:configdict.ConfigDict=None,
                renderer: Any = None,
                activate=True):
        if name in Workspace._workspaces:
            logger.debug(f"Workspace {name} already exists. Returning the old"
                         f"workspace")
            assert scorestruct is None and config is None and renderer is None
            out = Workspace._workspaces[name]()
            if activate:
                out.activate()
        return super().__new__(cls)


    def __init__(self,
                 name: str,
                 scorestruct:Optional[ScoreStruct]=None,
                 config:configdict.ConfigDict=None,
                 renderer: Any = None,
                 dynamicCurve: DynamicCurve = None,
                 activate=True
                 ):
        self.name = name
        self.renderer = renderer
        self.config = config or getConfig()
        if scorestruct is None:
            scorestruct = ScoreStruct.fromTimesig((4, 4), quarterTempo=60)
        self._scorestruct = scorestruct
        self.dynamicsCurve = dynamicCurve or DynamicCurve.getDefault()
        Workspace._workspaces[name] = _weakref.ref(self)
        if activate:
            self.activate()

    def __repr__(self):
        parts = [f"name={self.name}, scorestruct={self.scorestruct}"]
        return f"Workspace({', '.join(parts)})"
            
    @property
    def scorestruct(self) -> ScoreStruct:
        return self._scorestruct
    
    @scorestruct.setter
    def scorestruct(self, s: ScoreStruct):
        _resetCache()
        self._scorestruct = s

    def __del__(self):
        if self.name == "root":
            logger.error("Cannot delete 'root' workspace")
            return
        if self.name in Workspace._workspaces:
            Workspace._workspaces.pop(self.name)

    @staticmethod
    def _createUniqueName() -> str:
        for n in range(1, 9999):
            name = f"Workspace-{n}"
            if name not in Workspace._workspaces:
                return name
        raise RuntimeError("Too many workspaces")

    @property
    def a4(self) -> float:
        return self.config['A4']

    @a4.setter
    def a4(self, value:float):
        self.config['A4'] = value
        if self.isActive():
            pt.set_reference_freq(value)

    def getTempo(self, measureNum=0) -> float:
        return float(self.scorestruct.getMeasureDef(measureNum).quarterTempo)

    def activate(self) -> None:
        pt.set_reference_freq(self.a4)
        Workspace._current = self

    def isActive(self) -> bool:
        return currentWorkspace() is self
    
    def clone(self, name:str=None, config:configdict.ConfigDict=UNSET,
              scorestruct:ScoreStruct=UNSET, activate=True):
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
    
    @classmethod
    def workspaces(cls) -> List[str]:
        """Returns the names of all created workspaces
        
        To access the actual Workspace, do Workspace(<workspacename>)
        """
        return list(cls._workspaces.keys())


def _init() -> None:
    if Workspace._initDone:
        logger.debug("init was already done")
        return
    Workspace._initDone = True
    w = Workspace("root", config=rootConfig)
    Workspace._root = w
    w.activate()


def currentWorkspace() -> Workspace:
    """
    Get current workspace
    """
    w = Workspace._current
    assert w is not None
    return w


def setTempo(quarterTempo:float, measureNum=0) -> None:
    """
    Set the current tempo. 
    
    This is only possible if the currently active ScoreStruct has only 
    one initial tempo
    """
    w = currentWorkspace()
    w.scorestruct.setTempo(quarterTempo, measureNum=measureNum)


def getConfig() -> configdict.ConfigDict:
    """
    Return the current config.
    """
    return currentWorkspace().config


def newConfig(cloneCurrent=True, name:str=None,     updates:dict=None
              ) -> configdict.ConfigDict:
    """
    Create a new config and set it as the active one.

    This will clone the current workspace with this new config

    maelzel.core is organizad around the idea of a current workspace.
    Each workspace has a valid config. By calling `newConfig`, a new workspace
    is created with, either a clone of the previous config (if called with
    `cloneCurrent=True`) or of the root config. All other attributes of
    a workspace are inherited from the previous workspace. To access the newly
    created workspace, call :func:`~maelzel.workspace.currentWorkspace`.

    Args:
        cloneCurrent: if True, clones the current config otherwise clones
            the default
        name: name of the workspace created with this config. If no name is given,
            a new unique name is created (this name can be found
            via ``currentConfig().name``)

    Returns:
        the new config

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> cfg = newConfig()
        >>> cfg['play.numChannels'] = 4
        >>> cfg['']
    """
    w = currentWorkspace()
    if cloneCurrent:
        config = w.config.clone(updates=updates, cloneCallbacks=True)
    else:
        rootWorkspace = Workspace._workspaces['root']()
        assert rootWorkspace is not None
        config = rootWorkspace.config.clone(updates=updates, cloneCallbacks=True)
    newWorkspace = currentWorkspace().clone(name=name, config=config, 
                                            scorestruct=w.scorestruct)
    return newWorkspace.config


def currentScoreStruct() -> ScoreStruct:
    """
    Returns the current ScoreStruct (which defines tempo and time signatures)

    If no ScoreStruct has been set explicitely, a default is always active which
    creates an endless 4/4 score with tempo q=60

    .. note::
        To modify the current structure a new structure can be set via
        ``setScoreStruct(newscore)``
        Alternatively, if the current score structure has no multiple tempos,
        the tempo can be modified via `setTempo`.

    """
    s = currentWorkspace().scorestruct
    assert s is not None
    return s


def setScoreStruct(s: ScoreStruct) -> None:
    """
    Sets the current score structure
    """
    currentWorkspace().scorestruct = s


def _presetsPath() -> str:
    datadirbase = _appdirs.user_data_dir("maelzel")
    path = os.path.join(datadirbase, "core", "presets")
    return path


def presetsPath() -> str:
    """ Returns the path were instrument presets are read/written"""
    userpath = getConfig()['play.presetsPath']
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
    userpath = getConfig()['rec.path']
    if userpath:
        path = userpath
    else:
        path = _appdirs.user_data_dir(appname="maelzel", version="recordings")
    if not os.path.exists(path):
        os.makedirs(path)
    return path


_init()