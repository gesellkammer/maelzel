from __future__ import annotations
from ._common import *
import configdict
import appdirs
import os
from .config import mainConfig
from pitchtools import set_reference_freq
from typing import Any
from maelzel.scorestruct import ScoreStructure
from functools import lru_cache


def _resetCache():
    from .musicobj import resetImageCache
    resetImageCache()


class Workspace:
    initdone: bool = False
    workspaces: Dict[str, Workspace] = {}
    current: Workspace

    def __new__(cls, name: str,
                scorestruct: Opt[ScoreStructure]=None,
                config:configdict.CheckedDict=None,
                renderer: Any = None,
                activate=True):
        if name in Workspace.workspaces:
            return Workspace.workspaces[name]
        return super().__new__(cls)


    def __init__(self,
                 name: str,
                 scorestruct:Opt[ScoreStructure]=None,
                 config:configdict.CheckedDict=None,
                 renderer: Any = None,
                 activate=True
                 ):
        self.name = name
        self.renderer = renderer
        self.config = config or currentConfig()
        if scorestruct is None:
            scorestruct = ScoreStructure.fromTimesig((4, 4), quarterTempo=60)
        self.scorestruct = scorestruct
        Workspace.workspaces[name] = self
        if activate:
            self.activate()

    @property
    def a4(self) -> float:
        return self.config['A4']

    @a4.setter
    def a4(self, value:float):
        self.config['A4'] = value
        if self is currentConfig():
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


def _init():
    if Workspace.initdone:
        logger.debug("init was already done")
        return
    Workspace.initdone = True
    w = Workspace("main", config=mainConfig)
    w.activate()


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


def setA4(a4:float) -> None:
    """ Set the A4 value for the current state """
    currentWorkspace().a4 = a4


def currentConfig() -> configdict.CheckedDict:
    """
    Return the current config. If some new state has been pushed this will
    be a non-persistent config.
    """
    return currentWorkspace().config


def currentScoreStructure() -> ScoreStructure:
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


def setScoreStructure(s: ScoreStructure) -> None:
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
    userpath = currentConfig()['play.presetsPath']
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
    userpath = currentConfig()['rec.path']
    if userpath:
        path = userpath
    else:
        path = appdirs.user_data_dir(appname="maelzel", version="recordings")
    if not os.path.exists(path):
        os.makedirs(path)
    return path


_init()