from __future__ import annotations
from ._common import *
import configdict
import appdirs
import os
from .config import rootConfig
from pitchtools import set_reference_freq
from typing import Any
from maelzel.scorestruct import ScoreStruct
from functools import lru_cache
import weakref


def _resetCache():
    from .musicobj import resetImageCache
    resetImageCache()


class Workspace:
    initdone: bool = False
    workspaces: Dict[str, weakref.ReferenceType[Workspace]] = {}
    current: Workspace
    root: Workspace

    def __new__(cls, name: str,
                scorestruct: Opt[ScoreStruct]=None,
                config:configdict.ConfigDict=None,
                renderer: Any = None,
                activate=True):
        if name in Workspace.workspaces:
            return Workspace.workspaces[name]
        return super().__new__(cls)


    def __init__(self,
                 name: str,
                 scorestruct:Opt[ScoreStruct]=None,
                 config:configdict.ConfigDict=None,
                 renderer: Any = None,
                 activate=True
                 ):
        self.name = name
        self.renderer = renderer
        self.config = config or getConfig()
        if scorestruct is None:
            scorestruct = ScoreStruct.fromTimesig((4, 4), quarterTempo=60)
        self._scorestruct = scorestruct
        Workspace.workspaces[name] = weakref.ref(self)
        if activate:
            self.activate()
            
    @property
    def scorestruct(self) -> ScoreStruct:
        return self._scorestruct
    
    @scorestruct.setter
    def scorestruct(self, s: ScoreStruct):
        _resetCache()
        self._scorestruct = s

    def __del__(self):
        if self.name == "root":
            logger.error("Can't delete 'root' workspace")
            return
        Workspace.workspaces.pop(self.name)

    @staticmethod
    def _createUniqueName() -> str:
        for n in range(1, 9999):
            name = f"Workspace-{n}"
            if name not in Workspace.workspaces:
                return name
        raise RuntimeError("Too many workspaces")

    @property
    def a4(self) -> float:
        return self.config['A4']

    @a4.setter
    def a4(self, value:float):
        self.config['A4'] = value
        if self is getConfig():
            set_reference_freq(value)

    @property
    def tempo(self) -> float:
        if not self.scorestruct.hasUniqueTempo():
            raise ValueError("The current ScoreStructure has multiple tempi")
        return float(self.scorestruct.measuredefs[0].quarterTempo)

    @tempo.setter
    def tempo(self, quarterTempo: float):
        if not self.canSetTempo():
            raise ValueError("The current ScoreStructure has multiple tempi. "
                             "It is not possible to set the tempo globally")
        scorestruct = self.scorestruct
        assert scorestruct is not None
        scorestruct.measuredefs[0].quarterTempo = quarterTempo
        scorestruct.markAsModified()
        # _resetCache()

    def activate(self) -> None:
        set_reference_freq(self.a4)
        Workspace.current = self
        # _resetCache()

    def canSetTempo(self) -> bool:
        """ Returns True if it is possible to
        set the tempo for the current state's ScoreStructure"""
        return self.scorestruct.hasUniqueTempo()

    def isActive(self) -> bool:
        return currentWorkspace() is self


def _init():
    if Workspace.initdone:
        logger.debug("init was already done")
        return
    Workspace.initdone = True
    w = Workspace("root", config=rootConfig)
    Workspace.root = w
    w.activate()


def cloneWorkspace(name: str='', config:configdict.ConfigDict=UNSET,
                   scorestruct:ScoreStruct=UNSET, activate=True
                   ) -> Workspace:
    """
    Clone the current workspace.

    The current config will be cloned also, so any modification to it
    will be indendent from the config it is based upon. The scorestruct
    is also cloned.

    Args:
        name: the name of this workspace. If not given, a unique name is generated
        config: a config, as returned, for example, via getConfig().clone(...).
            If left unfilled, a copy of the current config is used
        scorestruct: a new scorestruct, or unset to use a copy of the current scorestruct
        activate: if True, this new workspace is set to be the active one

    Returns:
        the cloned Workspace

    """
    w = currentWorkspace()
    if config is UNSET:
        config = w.config.clone(persistent=False, cloneCallbacks=True)
    if scorestruct is UNSET:
        scorestruct = w.scorestruct.copy()
    if not name:
        name = Workspace._createUniqueName()
    return Workspace(name,
                     config=config,
                     scorestruct=scorestruct or w.scorestruct,
                     activate=activate)


def currentWorkspace() -> Workspace:
    """
    Get current state
    """
    w = Workspace.current
    assert w is not None
    return w


def setTempo(quarterTempo:float) -> None:
    """
    Set the current tempo. This is only possible if the currently active
    ScoreStructure has only one initial tempo
    """
    w = currentWorkspace()
    if w.canSetTempo():
        w.tempo = quarterTempo


def getConfig() -> configdict.ConfigDict:
    """
    Return the current config. If some new state has been pushed this will
    be a non-persistent config.
    """
    return currentWorkspace().config


def newConfig(cloneCurrent=True, **kws) -> configdict.ConfigDict:
    """
    This will clone the current workspace with a copy of the current config

    The current score struct is kept

    Args:
        cloneCurrent: if True, clones the current config otherwise clones
            the default

    Returns:
        the new config
    """
    w = currentWorkspace()
    if cloneCurrent:
        config = w.config.clone(updates=kws, cloneCallbacks=True)
    else:
        rootWorkspace = Workspace.workspaces['root']()
        assert rootWorkspace is not None
        config = rootWorkspace.config.clone(updates=kws, cloneCallbacks=True)
    newWorkspace = cloneWorkspace(config=config, scorestruct=w.scorestruct)
    return newWorkspace.config


def currentScoreStructure() -> ScoreStruct:
    """
    Returns the current ScoreStructure (which defines tempo and time signatures)

    If no ScoreStructure has been set explicitely, a default is always active which
    creates an endless 4/4 score with tempo q=60

    .. note::
        To modify the current structure a new structure can be set via
        ``setScoreStructure(newscore)``
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
    _resetCache()
    currentWorkspace().scorestruct = s


@lru_cache(maxsize=1)
def _presetsPath() -> str:
    datadirbase = appdirs.user_data_dir("maelzel")
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
        path = appdirs.user_data_dir(appname="maelzel", version="recordings")
    if not os.path.exists(path):
        os.makedirs(path)
    return path


_init()