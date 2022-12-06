from __future__ import annotations
import os
import pitchtools
import appdirs as _appdirs
import warnings

from ._common import logger, UNSET
from .config import CoreConfig
from maelzel.music.dynamics import DynamicCurve
from maelzel.scorestruct import ScoreStruct

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any


def _resetCache() -> None:
    from .mobj import resetImageCache
    resetImageCache()


__all__ = (
    'Workspace',
    'getWorkspace',

    'getConfig',
    'setConfig',

    'getScoreStruct',
    'setScoreStruct',

    'setTempo',
    'logger'
)


class Workspace:
    """
    Create a new Workspace

    Args:
        scorestruct: the ScoreStruct. If None, a default scorestruct (4/4, q=60) is used
        config: the active config for this workspace. If None, a copy of the root config
            is used
        updates: if given, these are applied to the config
        renderer: will be set to the active offline renderer while rendering offline
        dynamicCurve: a DynamicCurve used to map amplitude to dynamic expressions
        active: if True, make this Workpsace active

    Attributes:
        renderer: if not None, the active offline renderer
        config: the active config for this Workspace
        scorestruct: the active ScoreStruct
        dynamicCurve: a DynamicCurve, mapping amplitude to dynamic expression

    A Workspace can also be used as a context manager, in which case it will be
    activated when entering the context and  and deactivated at exit

    .. code::

        from maelzel.core import *
        scorestruct = ScoreStruct(r'''
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
    _counter: int = 0

    active: Workspace = None

    def __init__(self,
                 config: CoreConfig = None,
                 scorestruct: ScoreStruct | None = None,
                 renderer: Any = None,
                 dynamicCurve: DynamicCurve = None,
                 updates: dict = None,
                 active=False):

        self.renderer = renderer

        if config is None or isinstance(config, str):
            config = CoreConfig(updates=updates, source=config)
        elif updates:
            config = config.clone(updates=updates)
        else:
            assert isinstance(config, CoreConfig)
        self._config: CoreConfig = config

        if dynamicCurve is None:
            mindb = config['dynamicCurveMindb']
            maxdb = config['dynamicCurveMaxdb']
            dynamics = config['dynamicCurveDynamics'].split()
            dynamicCurve = DynamicCurve.fromdescr(shape=config['dynamicCurveShape'],
                                                  mindb=mindb, maxdb=maxdb,
                                                  dynamics=dynamics)

        self.dynamicCurve = dynamicCurve

        if scorestruct is None:
            scorestruct = ScoreStruct.fromTimesig((4, 4), quarterTempo=60)
        self._scorestruct = scorestruct
        self._previousWorkspace: Workspace | None = Workspace.active
        if active:
            self.activate()

    @property
    def config(self) -> CoreConfig:
        return self._config

    @config.setter
    def config(self, config: CoreConfig) -> None:
        self._config = config
        if self.isActive():
            self._activateConfig()

    def _activateConfig(self) -> None:
        config = self._config
        pitchtools.set_reference_freq(config['A4'])

    def deactivate(self) -> Workspace:
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
            return Workspace.root
        else:
            self._previousWorkspace.activate()
            return self._previousWorkspace

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
        return (f"Workspace(scorestruct={repr(self.scorestruct)}, "
                f"dynamicCurve={self.dynamicCurve})")

    @property
    def scorestruct(self) -> ScoreStruct:
        """Returns the current ScoreSctruct"""
        return self._scorestruct
    
    @scorestruct.setter
    def scorestruct(self, s: ScoreStruct):
        _resetCache()
        self._scorestruct = s

    @property
    def a4(self) -> float:
        return self.config['A4']

    @a4.setter
    def a4(self, value: float):
        self.config.bypassCallbacks = True
        self.config['A4'] = value
        self.config.bypassCallbacks = False
        if self.isActive():
            pitchtools.set_reference_freq(value)

    def getTempo(self, measureNum=0) -> float:
        """Get the quarter-note tempo at the given measure"""
        return float(self.scorestruct.getMeasureDef(measureNum).quarterTempo)

    def activate(self) -> Workspace:
        """Make this the active Workspace

        This method returns self in order to allow chaining

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
        Workspace.active = self
        return self

    def isActive(self) -> bool:
        """Is this the active Workspace?"""
        return getWorkspace() is self
    
    def clone(self,
              config: CoreConfig = UNSET,
              scorestruct: ScoreStruct = UNSET,
              active=False,

    ) -> Workspace:
        """
        Clone this Workspace

        Args:
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
            assert isinstance(self.config, CoreConfig)
            config = self.config.copy()
        if scorestruct is UNSET:
            scorestruct = self.scorestruct.copy()
        return Workspace(config=config,
                         scorestruct=scorestruct or self.scorestruct,
                         active=active)

    def presetsPath(self) -> str:
        """
        Returns the path were instrument presets are read/written

        The path can be configured with the core configuration::

            getConfig()['play.presetsPath'] = '/my/custom/path'

        Otherwise the default for the current platform is returned.

        Example
        ~~~~~~~

        Running in linux using ipython

        .. code-block:: python

            >>> from maelzel.core import *
            >>> path = getWorkspace().presetsPath()
            >>> path
            '/home/XXX/.local/share/maelzel/core/presets'
            >>> os.listdir(path)
            ['.click.yaml',
             'click.yaml',
             'noise.yaml',
             'accordion.yaml',
             'piano.yaml',
             'voiceclick.yaml']

        """
        return self.config.get('play.presetsPath') or _presetsPath()

    def recordPath(self) -> str:
        """
        The path where temporary recordings are saved

        We do not use the temporary folder because it is wiped regularly
        and the user might want to access a recording after rebooting.
        The returned folder is guaranteed to exist

        The default record path can be customized by modifying the config
        'rec.path'
        """
        userpath = self.config['rec.path']
        if userpath:
            path = userpath
        else:
            path = _appdirs.user_data_dir(appname="maelzel", version="recordings")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def setDynamicsCurve(self, shape='expon(0.5)', mindb=-80, maxdb=0) -> Workspace:
        """
        Set a new dynamics curve for this Workspace

        Args:
            shape: the shape of the curve
            mindb: the db value mapped to the softest dynamic
            maxdb: the db value mapped to the loudest dynamic

        Returns:

        """
        self.dynamicCurve = DynamicCurve.fromdescr(shape=shape, mindb=mindb, maxdb=maxdb)
        return self


def _init() -> None:
    if Workspace._initDone:
        logger.debug("init was already done")
        return
    Workspace._initDone = True
    w = Workspace(config=CoreConfig.root, active=True)
    Workspace.root = w


def getWorkspace() -> Workspace:
    """
    Get the active workspace

    To create a new Workspace and set it as the active Workspace use::

        >>> from maelzel.core import *
        >>> w = getWorkspace().clone(active=True)
    """
    return Workspace.active


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

    """
    w = workspace or getWorkspace()
    return Workspace(config=config or w.config,
                     scorestruct=scorestruct or w.scorestruct,
                     active=active,
                     updates=updates)


def setTempo(quarterTempo: float, measureIndex=0) -> None:
    """
    Set the current tempo. 
    
    Args:
        quarterTempo: the new tempo
        measureIndex: the measure number to modify. The scorestruct's tempo is modified
            until the next tempo

    See Also
    ~~~~~~~~

    * :meth:`ScoreStruct.setTempo <maelzel.scorestruct.ScoreStruct.setTempo>`
    * :ref:`setTempo notebook <setTempo_notebook>`


    Example
    ~~~~~~~

    .. code-block:: python

        from maelzel.core import *
        # A chromatic scale of eighth notes
        scale = Chain(Note(m, dur=0.5)
                      for m in range(60, 72))

        # Will play 8th notes at 60
        scale.play()

        setTempo(120)
        # Will play at twice the speed
        scale.play()

        # setTempo is a shortcut to ScoreStruct's setTempo method
        struct = getScoreStruct()
        struct.setTempo(40)

    .. code-block:: python

        >>> setScoreStruct(ScoreStruct(r'''
        ... 3/4, 120
        ... 4/4, 66
        ... 5/8, 132
        ... '''))
        >>> setTempo(40)
        >>> getScoreStruct().dump()
        0, 3/4, 40
        1, 4/4, 66
        2, 5/8, 132

    """
    w = getWorkspace()
    w.scorestruct.setTempo(quarterTempo, measureIndex=measureIndex)


def getConfig() -> CoreConfig:
    """
    Return the active config.

    """
    return Workspace.active.config


def setConfig(config: CoreConfig) -> None:
    """
    Activate this config

    This is the same as ``getWorkspace().config = config``.

    Args:
        config: the new config
    """
    getWorkspace().config = config


def makeConfig(updates: dict = None,
               proto: str | CoreConfig = 'root',
               active=False
               ) -> CoreConfig:
    """
    Create a new config

    By default the returned config is a clone of the root config[1]_ with any updates
    applied to it. This function is simply a shortcut and is placed here for discoverability

    This is the same as::

        # Using root as prototype
        setConfig(rootConfig.clone({...}))

        # Using the active config as prototype
        setConfig(getConfig().clone({...})

    Args:
        updates: a dict of updates to the new config
        proto: the dict to use as source. If ``None`` or ``'root'`` is given, the root config is used
            Other possible sources are the *active config* (as returned by :func:`getConfig`)
            or the *default config* [2]_ (use ``source='default'``)
        active: if True, set the newly created Config as active within the current
            Workspace

    Returns:
        the cloned config

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        # Do something with a modified config
        >>> config = makeConfig({'play.instr': 'piano',
        ...                      'play.useDynamics': True})
        >>> voice = Chain("4C:.5:ff 4E-:1:p 4F:0.5:f".split())
        # Using the config as context manager makes it active for the
        # lifetime of the context
        >>> with config:
        ...     voice.play()

    .. [1] The root config is the config created at the start of a session, which includes any
        changes persisted via :meth:`CoreConfig.save() <maelzel.core.config.CoreConfig.save>`)
    .. [2] The default config is the config without any user customizations
    """
    warnings.warn("Decreated, use the constructor  CoreConfig instead")
    if proto is None or proto == 'root':
        proto = CoreConfig.root
    elif proto == 'default':
        proto = CoreConfig.root.makeDefault()
    elif proto == 'active':
        proto = getConfig()
    out = proto.clone(updates=updates)
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

    This is a shortcut to ``getWorkspace().scorestruct = s``
    """
    getWorkspace().scorestruct = s


def _presetsPath() -> str:
    datadirbase = _appdirs.user_data_dir("maelzel")
    path = os.path.join(datadirbase, "core", "presets")
    return path


_init()

