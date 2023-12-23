from __future__ import annotations
import os
import appdirs as _appdirs
from functools import cache
import pitchtools

from ._common import logger, UNSET, _Unset
from .config import CoreConfig

from maelzel.dynamiccurve import DynamicCurve
from maelzel.scorestruct import ScoreStruct

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import playback


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
        dynamicCurve: a DynamicCurve used to map amplitude to dynamic expressions
        active: if True, make this Workpsace active

    A Workspace can also be used as a context manager, in which case it will be
    activated when entering the context and deactivated at exit

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
    root: Workspace | None = None
    """The root workspace. This is the workspace active at the start of a session
    and is always kept alive since it holds a reference to the root config. It should
    actually never be None"""

    active: Workspace | None = None
    """The currently active workspace. Never None after the class has been initialized"""

    _initdone: bool = False

    def __init__(self,
                 config: CoreConfig = None,
                 scorestruct: ScoreStruct | None = None,
                 dynamicCurve: DynamicCurve = None,
                 updates: dict = None,
                 active=False):

        assert self._initdone

        if config is None or isinstance(config, str):
            config = CoreConfig(updates=updates, source=config)
        elif updates:
            config = config.clone(updates=updates)
        else:
            assert isinstance(config, CoreConfig)

        if dynamicCurve is None:
            mindb = config['dynamicCurveMindb']
            maxdb = config['dynamicCurveMaxdb']
            dynamics = config['dynamicCurveDynamics'].split()
            dynamicCurve = DynamicCurve.fromdescr(shape=config['dynamicCurveShape'],
                                                  mindb=mindb, maxdb=maxdb,
                                                  dynamics=dynamics)

        if scorestruct is None:
            scorestruct = ScoreStruct(timesig=(4, 4), tempo=60)

        self._config: CoreConfig = config
        """The CoreConfig attached to this Workspace"""

        self.renderer: playback.Renderer | None = None
        """The active renderer, if any"""

        self.dynamicCurve = dynamicCurve
        """The dynamic curve used to convert dynamics to amplitudes"""

        self._scorestruct = scorestruct
        """The scorestruct attached to this workspace"""

        self._previousWorkspace: Workspace | None = Workspace.active
        """The previous workspace. Will be activated when this one is desactivated"""

        if active:
            self.activate()

    @property
    def config(self) -> CoreConfig:
        """The CoreConfig for this workspace"""
        return self._config

    @config.setter
    def config(self, config: CoreConfig) -> None:
        self._config = config
        if self.isActive():
            self._activateConfig()

    def _activateConfig(self) -> None:
        config = self._config
        pitchtools.set_reference_freq(config['A4'])
        _resetCache()

    @staticmethod
    def getActive() -> Workspace:
        active = Workspace.active
        assert active is not None
        return active

    @staticmethod
    def _initclass() -> None:
        if Workspace._initdone:
            logger.debug("init was already done")
            return
        Workspace._initdone = True
        if CoreConfig.root is None:
            CoreConfig.root = root = CoreConfig(source='load')
        else:
            root = CoreConfig.root
        # The root config itself should never be active since it is read-only
        w = Workspace(config=root.copy(), active=True)
        Workspace.root = w

    def deactivate(self) -> None:
        """
        Deactivates this Workspace and sets the previous Workspace as active

        .. note::

            There is always an active Workspace. An attempt to deactivate the
            root Workspace will be ignored

        Returns:
            the now active workspace
        """
        if Workspace.active is not self:
            logger.warning("Cannot deactivate this Workspace since it is not active")
        elif self is Workspace.root:
            logger.warning("Cannot deactivate the root Workspace")
        elif self._previousWorkspace is None:
            logger.warning("This Workspace has not previous workspace, activating the root"
                           " Workspace instead")
            assert Workspace.root is not None
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
        return (f"Workspace(scorestruct={repr(self.scorestruct)}, "
                f"config={self.config.diff()}, "
                f"dynamicCurve={self.dynamicCurve})")

    @property
    def scorestruct(self) -> ScoreStruct:
        """The default ScoreSctruct for this Workspace"""
        return self._scorestruct
    
    @scorestruct.setter
    def scorestruct(self, s: ScoreStruct):
        self._scorestruct = s

    @property
    def a4(self) -> float:
        """The reference frequency in this Workspace"""
        return self.config['A4']

    @a4.setter
    def a4(self, value: float):
        self.config.bypassCallbacks = True
        self.config['A4'] = value
        self.config.bypassCallbacks = False
        if self.isActive():
            pitchtools.set_reference_freq(value)

    def getTempo(self, measureNum=0) -> float:
        """Get the quarternote tempo at the given measure"""
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
        return Workspace.active is self
    
    def clone(self,
              config: CoreConfig = None,
              scorestruct: ScoreStruct = None,
              active=False
              ) -> Workspace:
        """
        Clone this Workspace

        Args:
            config: the config to use. **Leave unset** to clone this Workspace's config.
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
        if config is None:
            assert isinstance(self.config, CoreConfig)
            config = self.config.copy()
        if scorestruct is None:
            scorestruct = self.scorestruct.copy()
        return Workspace(config=config,
                         scorestruct=scorestruct or self.scorestruct,
                         active=active)

    @staticmethod
    def presetsPath() -> str:
        """
        Returns the path where instrument presets are read/written

        Example
        ~~~~~~~

        Running in linux

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
        return presetsPath()

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
            self
        """
        self.dynamicCurve = DynamicCurve.fromdescr(shape=shape, mindb=mindb, maxdb=maxdb)
        return self

    @staticmethod
    def rootConfig():
        return CoreConfig.root
        # return Workspace.root.config

    def amp2dyn(self, amp: float) -> str:
        return self.dynamicCurve.amp2dyn(amp)


def getWorkspace() -> Workspace:
    """
    Get the active workspace

    The active Workspace can be accessed via ``Workspace.active``. This function
    is simply a shortcut, placed here for visibility

    Example
    ~~~~~~~

    Create a new Workspace based on the active Workspace and activate it

        >>> from maelzel.core import *
        >>> w = getWorkspace().clone(active=True)

    The active workspace can always be accessed directly:

        >>> w = Workspace.active
        >>> w is getWorkspace()
        True

    """
    assert Workspace.active is not None
    return Workspace.active


def setTempo(tempo: float, reference=1, measureIndex=0) -> None:
    """
    Set the current tempo. 
    
    Args:
        tempo: the new tempo.
        reference: the reference value (1=quarternote, 2=halfnote, 0.5: 8th note)
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
    active = Workspace.active
    assert active is not None
    active.scorestruct.setTempo(tempo, reference=reference, measureIndex=measureIndex)


def getConfig() -> CoreConfig:
    """
    Return the active config.
    """
    active = Workspace.active
    assert active is not None
    return active.config


def setConfig(config: CoreConfig) -> None:
    """
    Activate this config

    This is the same as ``getWorkspace().config = config``.

    Args:
        config: the new config
    """
    active = Workspace.active
    assert active is not None
    active.config = config


def getScoreStruct() -> ScoreStruct:
    """
    Returns the active ScoreStruct (which defines tempo and time signatures)

    If no ScoreStruct has been set explicitely, a default struct is always active


    .. note::
        The active scorestruct can be set via ``setScoreStruct(newscore)`` (:func:`setScoreStruct`)

        If the current score structure has no multiple tempos,
        the tempo can be modified via :func:`setTempo`.

    .. seealso::
        * :func:`~maelzel.core.workpsace.setScoreStruct`
        * :func:`~maelzel.core.workpsace.setTempo`
        * :class:`~maelzel.scorestruct.ScoreStruct`
    """
    active = Workspace.active
    assert active is not None
    return active.scorestruct


def setScoreStruct(score: str | ScoreStruct | None = None,
                   timesig: tuple[int, int] | str = None,
                   tempo: int = None) -> None:
    """
    Sets the current score structure

    If given a ScoreStruct, this is simply a shortcut to ``getWorkspace().scorestruct = s``
    If given a score as string or simply a time signature and/or tempo, it creates
    a ScoreStruct and sets it as active

    Args:
        score: the scorestruct as a ScoreStruct or a string score (see ScoreStruct for more
            information about the format). If None, a simple ScoreStruct using the given
            time-signature and/or tempo will be created
        timesig: only used if no score is given.
        tempo: the quarter-note tempo. Only used if no score is given

    .. seealso::
        * :func:`~maelzel.core.workpsace.getScoreStruct`
        * :func:`~maelzel.core.workpsace.setTempo` (modifies the tempo of the active scorestruct)
        * :class:`~maelzel.scorestruct.ScoreStruct`
        * :func:`~maelzel.core.workpsace.getWorkspace`

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> setScoreStruct(ScoreStruct(tempo=72))
        >>> setScoreStruct(r'''
        ... 4/4, 72
        ... 3/8
        ... 5/4
        ... .         # Same time-signature and tempo
        ... , 112     # Same time-signature, faster tempo
        ... 20, 3/4, 60   # At measure index 20, set the time-signature to 3/4 and tempo to 60
        ... ...       # Endless score
        ... ''')
        
    """
    if isinstance(score, str):
        s = ScoreStruct(score)
    elif isinstance(score, ScoreStruct):
        s = score
    else:
        assert score is None
        if timesig is not None or tempo is not None:
            s = ScoreStruct(timesig=timesig, tempo=tempo)
        else:
            s = ScoreStruct(timesig=(4, 4), tempo=60)
    w = Workspace.active
    assert w is not None
    w.scorestruct = s


@cache
def presetsPath() -> str:
    datadirbase = _appdirs.user_data_dir("maelzel")
    path = os.path.join(datadirbase, "core", "presets")
    return path


Workspace._initclass()