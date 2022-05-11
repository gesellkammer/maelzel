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

from ._common import logger, UNSET
from ._typedefs import time_t
from .config import rootConfig
from maelzel.music.dynamics import DynamicCurve
from maelzel.scorestruct import ScoreStruct
from maelzel.rational import Rat

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Any, Union, Tuple
    import configdict


def _resetCache() -> None:
    from .musicobj import resetImageCache
    resetImageCache()


___all__ = (
    'Workspace',
    'getWorkspace',
    'getConfig',
    'setConfig',
    'getScoreStruct'
    'setScoreStruct',
    'setTempo',
)


class Workspace:
    """
    Create a new Workspace

    Args:
        name: the name of the workspace, or nothing to create an unique name.
            The name 'root' refers to the root Workspace
        scorestruct: the ScoreStruct. If None, a default scorestruct (4/4, q=60) is used
        config: the active config for this workspace. If None, a copy of the root config
            is used
        updates: if given, these are applied to the config
        renderer: will be set to the active offline renderer while rendering offline
        dynamicCurve: a DynamicCurve used to map amplitude to dynamic expressions
        active: if True, make this Workpsace active

    Attributes:
        name: the name of this Workspace
        renderer: if not None, the active offline renderer
        config: the active config for this Workspace
        scorestruct: the active ScoreStruct
        dynamicsCurve: a DynamicCurve, mapping amplitude to dynamic expression

    A Workspace can also be used as a context manager, in which case it will be
    activated when entering the context and  and deactivated at exit

    .. code::

        from maelzel.core import *
        scorestruct = ScoreStruct.fromString(r'''
        4/4, 60
        .
        3/4
        5/8, 72
        ''')
        notes = Chain([Note(m, start=i) for i, m in enumerate(range(60, 72))])
        # Create a temporary Workspace with the given scorestruct and a clone
        # of the active config
        with Workspace(scorestruct=scorestruct, config=getConfig()) as w:
            notes.show()

    """
    root: Workspace = None

    _initDone: bool = False
    _active: Workspace = None
    _counter: int = 0

    def __init__(self,
                 config: configdict.ConfigDict = None,
                 scorestruct: Optional[ScoreStruct] = None,
                 renderer: Any = None,
                 dynamicCurve: DynamicCurve = None,
                 name: str = '',
                 updates: dict = None,
                 active=False):

        if not name:
            name = Workspace._createUniqueName()
        self.name = name
        self.renderer = renderer
        if config:
            self.config = config if not updates else config.clone(updates=updates)
        else:
            self.config = rootConfig.clone(updates=updates)
        if scorestruct is None:
            scorestruct = ScoreStruct.fromTimesig((4, 4), quarterTempo=60)
        self._scorestruct = scorestruct
        self.dynamicsCurve = dynamicCurve or DynamicCurve.fromdescr(shape=self.config.get('dynamicsCurve.shape', 'expon(3.0)'))
        self._previousWorkspace: Optional[Workspace] = getWorkspace()
        if active:
            self.activate()

    def deactivate(self) -> None:
        """
        Deactivates this Workspace and sets the previous Workspace as active

        .. note::

            There should always be an active Workspace

        """
        if not getWorkspace() is self:
            logger.warning("Cannot deactivate this Workspace since it is not active")
        elif self is Workspace.root:
            logger.warning("Cannot deactivate the root Workspace")
        elif self._previousWorkspace is None:
            logger.warning("This Workspace has not previous workspace, activating the root"
                           " Workspace instead")
            Workspace.root.activate()
        else:
            self._previousWorkspace.activate()

    def __del__(self):
        if self.isActive() and self is not Workspace.root:
            self.deactivate()

    def __enter__(self):
        if getWorkspace() is self:
            return
        self._previousWorkspace = getWorkspace()
        self.activate()

    def __exit__(self, *args, **kws):
        if self._previousWorkspace is not None:
            self._previousWorkspace.activate()
            self._previousWorkspace = None

    def __repr__(self):
        parts = [f"name={self.name}, scorestruct={repr(self.scorestruct)}"]
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

    def activate(self) -> Workspace:
        """Make this the active Workspace

        This method returns self in order to allow chaining:

        Example
        -------

            >>> from maelzel.core import *
            >>> from pitchtools import *
            >>> w = Workspace(updates={'A4': 432}).activate()
            >>> n2f("A4")
            432
            >>> w.deactivate()
            >>> n2f("A4")
            442
        """
        pitchtools.set_reference_freq(self.a4)
        Workspace._active = self
        return self

    def isActive(self) -> bool:
        """Is this the active Workspace?"""
        return getWorkspace() is self
    
    def clone(self, name: str = None,
              config: configdict.ConfigDict = UNSET,
              scorestruct: ScoreStruct = UNSET,
              active=False
              ) -> Workspace:
        """
        Clone this Workspace

        Args:
            name: the name of the newly created Workspace. None will generate a
                unique name
            config: the config to use. **Leave unset** to clone the currently active
                config, use ``rootConfig`` or use ``rootConfig.makeDefault()`` to
                create a config with all values set to default
            scorestruct: if unset, use this Workspace's scorestruct
            active: if True, activate the cloned Workspace

        Returns:
            the cloned Workspace


        Example
        -------

            >>> from maelzel.core import *
            >>> myworkspace = getWorkspace().clone()
            >>> myworkspace.config['A4'] = 432
            >>> with myworkspace as w:
            ...     # This will activate the workspace and deactivate it at exit
            ...     # Now do something baroque
        """
        if config is UNSET:
            config = self.config.clone(persistent=False, cloneCallbacks=True)
        if scorestruct is UNSET:
            scorestruct = self.scorestruct.copy()
        if not name or name is UNSET:
            name = Workspace._createUniqueName()
        return Workspace(config=config,
                         scorestruct=scorestruct or self.scorestruct,
                         active=active,
                         name=name
                         )


def _init() -> None:
    if Workspace._initDone:
        logger.debug("init was already done")
        return
    Workspace._initDone = True
    w = Workspace(name="root", config=rootConfig, active=True)
    Workspace.root = w


def getWorkspace() -> Workspace:
    """
    Get the active workspace

    To create a new Workspace and set it as the active Workspace use::

        >>> from maelzel.core import *
        >>> w = getWorkspace().clone(active=True)
    """
    return Workspace.getActive()


def cloneWorkspace(workspace: Workspace = None,
                   config=None,
                   scorestruct: ScoreStruct = None,
                   updates: dict = None,
                   active=False) -> Workspace:
    """
    Clone the active or the given Workspace

    This is just a shortcut for ``getWorkspace().clone(...)``

    Args:
        workspace: the Workspace to clone, or None to use the active
        config: if given, use this config for this Workspace
        scorestruct: if given, use this ScoreStruct for the cloned Workspace
        updates: any updates for the config
        active: if True, set the cloned Workspace as active

    Returns:
        the cloned Workspace

    Example
    -------

        >>> from maelzel.core import *
        >>> scostruct = ScoreStruct.fromString()
    """
    w = workspace or getWorkspace()
    return Workspace(config=config or w.config,
                     scorestruct=scorestruct or w.scorestruct,
                     active=active,
                     updates=updates)


def setTempo(quarterTempo:float, measureNum=0) -> None:
    """
    Set the current tempo. 
    
    This is only allowed if the currently active ScoreStruct has only
    one initial tempo
    """
    w = getWorkspace()
    w.scorestruct.setTempo(quarterTempo, measureNum=measureNum)


def getConfig() -> configdict.ConfigDict:
    """
    Return the active config.

    .. seealso:: :func:`newConfig`
    """
    return getWorkspace().config


def setConfig(config: configdict.ConfigDict) -> None:
    """
    Activate this config

    This is the same as ``getWorkspace().config = config``. It is put here
    for visibility

    Args:
        config: the new config
    """
    getWorkspace().config = config


def Config(updates: dict = None, source: Union[configdict.ConfigDict, str] = 'root',
           active=False
           ) -> configdict.ConfigDict:
    """
    Create a new config

    The returned config is a clone of either a given source dict, the root config,
    the active or default config. This function is simply a shortcut to clone a
    config and is placed here for discoverability

    Args:
        updates: a dict of updates to the new config
        source: the dict to use as source. Either another config or one of 'root' (to clone
            the root config), 'active' or 'default'
        active: if True, set this new config as the active config in the current
            Workspace

    Returns:
        the cloned config (a configdict.ConfigDict)

    =========== =====================
    Name        Description
    =========== =====================
    root        The root config. This is the initial config of the session and includes
                any saved cu
    """
    if isinstance(source, str):
        if source == 'root':
            baseconfig = rootConfig
        elif source == 'active':
            baseconfig = getConfig()
        elif source == 'default':
            baseconfig = rootConfig.makeDefault()
        else:
            raise KeyError(f"Source {source} unknown. Valid values: 'root', 'default', 'active'")
    else:
        baseconfig = source
        assert baseconfig.default == rootConfig.default, "The given config is not valid"

    out = baseconfig.clone(updates=updates)
    if active:
        setConfig(out)
    return out


def getScoreStruct() -> ScoreStruct:
    """
    Returns the active ScoreStruct (which defines tempo and time signatures)

    If no ScoreStruct has been set explicitely, a default is always active which
    creates an endless 4/4 score with tempo q=60

    .. note::
        To modify the active scorestrict a new scorestruct can be set via
        ``setScoreStruct(newscore)``

        If the current score structure has no multiple tempos,
        the tempo can be modified via `setTempo`.

    .. seealso:: :func:`~maelzel.core.workpsace.setScoreStruct`, :func:`~maelzel.core.workpsace.setTempo`
    """
    s = getWorkspace().scorestruct
    assert s is not None
    return s


def setScoreStruct(s: ScoreStruct) -> None:
    """
    Sets the current score structure
    """
    getWorkspace().scorestruct = s


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


def toBeat(x: Union[time_t, Tuple[int, time_t]]) -> Rat:
    """
    Convert a time in secs or a location (measure, beat) to a quarter-note beat

    Args:
        x: the time/location to convert

    Returns:
        the corresponding quarter note beat according to the active ScoreStruct

    """
    return getWorkspace().scorestruct.toBeat(x)


def toTime(x: Union[time_t, Tuple[int, time_t]]) -> Rat:
    """
    Convert a quarter-note beat or a location (measure, beat) to an absolute time in secs

    Args:
        x: the beat/location to convert

    Returns:
        the corresponding time according to the active ScoreStruct

    """
    return getWorkspace().scorestruct.toTime(x)


_init()
