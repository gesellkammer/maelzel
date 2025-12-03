from __future__ import annotations
import os
import appdirs as _appdirs
from functools import cache
import pitchtools

from ._common import logger
from .config import CoreConfig

from maelzel.common import F
from maelzel.dynamiccurve import DynamicCurve
from maelzel.scorestruct import ScoreStruct

import typing as _t
if _t.TYPE_CHECKING:
    from maelzel.core.renderer import Renderer
    from typing import Any
    import csoundengine.session
    import csoundengine.synth
    from . import presetmanager


def _clearCache() -> None:
    from .mobj import clearImageCache
    clearImageCache()


__all__ = (
    'Workspace',
    'getWorkspace',
    'logger',
    'ws'
)


class Workspace:
    """
    Create a new Workspace

    Args:
        scorestruct: the ScoreStruct. If not given, a default scorestruct (4/4, q=60) is used
        config: the active config for this workspace. If None, a copy of the root config
            is used
        updates: updates the config
        dynamicCurve: a DynamicCurve used to map amplitude to dynamic expressions
        active: make this Workpsace active

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
        # of the root config
        with Workspace(scorestruct=scorestruct, updates={'quant.complexity': 'low'}):
            notes.show()

    """
    root: _t.ClassVar[Workspace]
    """The root workspace. This is the workspace active at the start of a session
    and is always kept alive since it holds a reference to the root config. It should
    actually never be None"""

    active: _t.ClassVar[Workspace]
    """The currently active workspace"""

    _initdone: _t.ClassVar[bool] = False

    def __init__(self,
                 config: CoreConfig | None = None,
                 scorestruct: ScoreStruct | None = None,
                 dynamicCurve: DynamicCurve | None = None,
                 updates: dict[str, Any] | None = None,
                 active=False):

        assert self._initdone

        if config is None:
            config = CoreConfig(updates=updates or None)
        elif updates:
            config = config.clone(updates=updates)

        if dynamicCurve is None:
            mindb = config['dynamicCurveMindb']
            maxdb = config['dynamicCurveMaxdb']
            dynamics = config['dynamicCurveDynamics'].split()
            dynamicCurve = DynamicCurve.fromdescr(shape=config['dynamicCurveShape'],
                                                  mindb=mindb, maxdb=maxdb,
                                                  dynamics=dynamics)

        if scorestruct is None:
            scorestruct = ScoreStruct((4, 4), tempo=60)

        self._config: CoreConfig = config
        """The CoreConfig attached to this Workspace"""

        self.renderer: Renderer | None = None
        """The active renderer, if any"""

        self.dynamicCurve: DynamicCurve = dynamicCurve
        """The dynamic curve used to convert dynamics to amplitudes"""

        self._scorestruct: ScoreStruct = scorestruct
        """The scorestruct attached to this workspace"""

        self._previousWorkspace: Workspace | None = None
        """The previous workspace. Will be activated when this one is desactivated"""

        self._presetManager: presetmanager.PresetManager | None = None

        self._reverbSettings: dict[str, float] = {}
        self._reverbKeys: tuple[str, ...] = ('gaindb', 'delayms', 'decay', 'damp')

        self._reverbSynth: csoundengine.synth.Synth | None = None

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
        _clearCache()

    @staticmethod
    def clearCache() -> None:
        """
        Cleat the Workspace cache.

        At the moment this cache includes only the image generated via .show

        """
        _clearCache()

    @staticmethod
    def _initclass() -> None:
        if Workspace._initdone:
            logger.debug("init was already done")
            return
        Workspace._initdone = True
        CoreConfig._root = rootconfig = CoreConfig(source='load')
        # The root config itself should never be active since it is read-only
        Workspace.root = Workspace(config=rootconfig.copy(), active=True)
        assert Workspace.active is Workspace.root

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
            root = Workspace.root
            assert root is not None
            root.activate()
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
    def scorestruct(self, s: ScoreStruct | str):
        self.setScoreStruct(s)

    def setScoreStruct(self,
                       score: str | ScoreStruct | tuple[int, int] = (4, 4),
                       tempo: F | int | float = 60
                       ) -> None:
        """
        Sets the score structure for the current Workspace

        This is the same as `ScoreStruct(...).activate()`

        If given a ScoreStruct, it sets it as the active score structure.
        As an alternative a score structure as string can be given, or simply
        a time signature and/or tempo, in which case it will create the ScoreStruct
        and set it as active

        Args:
            score: the scorestruct as a ScoreStruct, a string score (see ScoreStruct for more
                information about the format) or simply a time signature.
            tempo: the quarter-note tempo. Only used if no score is given

        .. seealso::
            * :func:`~maelzel.core.workpsace.getScoreStruct`
            * :func:`~maelzel.core.workpsace.setTempo` (modifies the tempo of the active scorestruct)
            * :class:`~maelzel.scorestruct.ScoreStruct`
            * :func:`~maelzel.core.workpsace.getWorkspace`

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> w = getWorkspace()
            >>> w.scorestruct = ScoreStruct(tempo=72)
            >>> w.setScoreStruct(r'''
            ... 4/4, 72
            ... 3/8
            ... 5/4
            ... .         # Same time-signature and tempo
            ... , 112     # presetSame time-signature, faster tempo
            ... 20, 3/4, 60   # At measure index 20, set the time-signature to 3/4 and tempo to 60
            ... ...       # Endless score
            ... ''')
        """
        if isinstance(score, ScoreStruct):
            self._scorestruct = score
        else:
            config = self.config
            struct = ScoreStruct(score,
                                 tempo=tempo,
                                 breakTempo=config['quant.beatWeightTempoThresh'],
                                 subdivTempo=config['quant.subdivTempoThresh'])
            self._scorestruct = struct
        self.clearCache()

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
        return float(self.scorestruct.measure(measureNum).quarterTempo)

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
        if hasattr(Workspace, "active"):
            self._previousWorkspace = Workspace.active
        Workspace.active = self
        return self

    def isActive(self) -> bool:
        """Is this the active Workspace?"""
        return Workspace.active is self

    def clone(self,
              config: CoreConfig | None = None,
              scorestruct: ScoreStruct | None = None,
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
        The path where instrument presets are read/written

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
        return _presetsPath()

    def recordPath(self) -> str:
        """
        The path where temporary recordings are saved

        We do not use the temporary folder because it is wiped regularly
        and the user might want to access a recording after rebooting.
        The returned folder is guaranteed to exist

        The default record path can be customized by modifying the config
        'rec.path'

        .. seealso:: :meth:`Workspace.setRecordPath`
        """
        userpath = self.config['rec.path']
        if userpath:
            path = userpath
        else:
            path = _appdirs.user_data_dir(appname="maelzel", version="recordings")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def setRecordPath(self, path: str) -> None:
        """
        Set the path where temporary recordings are saved
        
        Args:
            path: the new path. It must be an existing path 

        """
        if not os.path.exists(path):
            raise OSError(f"Path {path} does not exist. Create it first: `os.makedirs('{path}')`")
        self.config['rec.path'] = path

    def setDynamicCurve(self, shape='expon(0.5)', mindb=-80, maxdb=0) -> None:
        """
        Set a new dynamic curve for this Workspace

        Args:
            shape: the shape of the curve
            mindb: the db value mapped to the softest dynamic
            maxdb: the db value mapped to the loudest dynamic

        """
        self.dynamicCurve = DynamicCurve.fromdescr(shape=shape, mindb=mindb, maxdb=maxdb)
        
    def amp2dyn(self, amp: float) -> str:
        return self.dynamicCurve.amp2dyn(amp)

    def setTempo(self,
                 tempo: float,
                 reference: tuple[int, int] | F | float = (4, 0),
                 measure=0) -> None:
        """
        Set the current tempo for the active scorestruct

        Args:
            tempo: the new tempo.
            reference: the reference value (1=quarternote, 2=halfnote, 0.5: 8th note)
            measure: the measure index  to modify. The scorestruct's tempo is modified
                until the next tempo. Measures start at 0

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

            w = getWorkspace()
            w.setTempo(120)
            # Will play at twice the speed
            scale.play()

            # setTempo is a shortcut to ScoreStruct's setTempo method
            w.setTempo(40)

        .. code-block:: python

            >>> setScoreStruct(ScoreStruct(r'''
            ... 3/4, 120
            ... 4/4, 66
            ... 5/8, 132
            ... '''))
            >>> w = getWorkspace()
            >>> w.setTempo(40)
            >>> w.scorestruct.dump()
            0, 3/4, 40
            1, 4/4, 66
            2, 5/8, 132

        """
        self.scorestruct.setTempo(measureidx=measure, tempo=tempo, reference=reference)

    def audioSession(self,
                     outdev='',
                     backend='',
                     numchannels: int | None = None,
                     buffersize: int = 0,
                     numbuffers: int = 0,
                     **kws) -> csoundengine.session.Session:
        """
        Get the audio Session used for playback

        Arguments are ignored if a session is already active

        Args:
            outdev: output device used. Depends on the backend used. List all devices
                via maelzel.core.playback.getAudioDevices, use '?' to select from a list of devices.
            backend: backend, depends on your platform. Use '?' to interactively select one
            numchannels: number of channels used. Defaults to the number of channels
                of the audio device used
            buffersize: buffer size to use, depends on the backend and device used
            numbuffers: number of buffers, determines the blocksize depending on the backend
            **kws: any keyword argument is passed to maelzel.core.playback.getSession

        Returns:
            the csoundengine.Session active for this workspace. If not already created,
            a new session is started


        """
        from maelzel.core import playback
        return playback.audioSession(name=self.config['play.engineName'],
                                     outdev=outdev,
                                     backend=backend,
                                     numchannels=numchannels,
                                     buffersize=buffersize,
                                     numbuffers=numbuffers,
                                     **kws)

    def reverbSynth(self) -> csoundengine.synth.Synth | None:
        """
        The reverb synth used for live playback

        Returns:
            the reverb synth or None if no synth using reverb has been scheduled
        """
        if not self.isAudioSessionActive():
            return None
        return self._reverbSynth
        # return _reverbEvent(session=self.audioSession(), instrname=self.config['reverbInstr'])

    def reverbInfo(self) -> dict[str, _t.Any]:
        """
        Returns information about this workspace's reverb

        Returns:
            a dict with reverb settings corresponding to the values which
            can be set via :meth:`~Workspace.setReverb`. Modifications
            to this dictionary have no side effects.
        """
        d: dict[str, _t.Any] = self._reverbSettings.copy()
        config = self.config
        for key in self._reverbKeys:
            if key not in d:
                d[key] = config[f'reverb{key.capitalize()}']
        synth = self.reverbSynth()
        if synth:
            d['synth'] = synth
            d['active'] = True
        else:
            d['active'] = False
        return d

    def setReverb(self,
                  gaindb: float | None = None,
                  delayms: int | None = None,
                  decay: float | None = None,
                  damp: float | None = None,
                  init=False
                  ) -> None:
        """
        Set parameters for reverb synth

        This modifies the running reverb and/or sets any values
        for reverb if used in the future or used for offline rendering.
        Only values passed are modified, None indicates to use the
        current value/default value. Default values can be modified
        via the config (ie. ``getWorkspace().config['reverbGaindb'] = -3``).
        To check the current values, see ???

        Args:
            gaindb: gain of reverb, in dB. Default: -6 (config: ``reverbGaindb``)
            delayms: predelay, in milliseconds. Default: 60 (config: ``reverbDelayms``)
            decay: decay time in seconds. Default: 3 (config: ``reverbDecay``)
            damp: damping factor between 0-1 (lower=less damping).
                Default: 0.2 (config: ``reverbDamp``)
            init: if True, start the reverb synth if not started already. This will
                start the audio session if needed

        """
        # Default(-6.), Default(60), Default(3), Default(0.2)
        if gaindb is not None:
            self._reverbSettings['gaindb'] = gaindb
        if delayms is not None:
            self._reverbSettings['delayms'] = delayms
        if decay is not None:
            self._reverbSettings['decay'] = decay
        if damp is not None:
            self._reverbSettings['damp'] = damp

        if init and (not self._reverbSynth or not self._reverbSynth.playing()):
            self._schedReverb()
        elif self._reverbSynth:
            # session = self.audioSession()
            # revsynth = _reverbEvent(session=session, instrname=instr)
            revsynth = self._reverbSynth
            if revsynth:
                if gaindb is not None:
                    revsynth.set(kgaindb=gaindb)
                if delayms is not None:
                    revsynth.set(kdelayms=delayms)
                if decay is not None:
                    revsynth.set(kdecay=decay)
                if damp is not None:
                    revsynth.set(kdamp=damp)

    def _schedReverb(self,
                     session: csoundengine.session.AbstractRenderer | None = None,
                     delay=0.
                     ) -> csoundengine.synth.Synth:
        if self._reverbSynth is not None and self._reverbSynth.playing():
            assert self.isAudioSessionActive()
            return self._reverbSynth
        config = self.config
        instr = config['reverbInstr']
        if session is None:
            session = self.audioSession()
        # if prevsynth := _reverbEvent(session=session, instrname=instr):
        #    return prevsynth
        def whenfinished(*args):
            assert isinstance(session, csoundengine.session.Session)
            logger.warning(f"Reverb synth stopped, args: %s", args)
            oldsynth = self._reverbSynth
            assert oldsynth is not None
            self._reverbSynth = None

            if self.config['reverbRestart'] and session.engine.elapsedTime() - oldsynth.start > 0.5:
                logger.warning("... restarting reverb")
                self._schedReverb(session=session, delay=0.5)

        def setfunc(synth, key: str, value: float, delay: float) -> bool:
            key2 = key[1:]
            if key2 in self._reverbKeys:
                self._reverbSettings[key2] = value
            return True

        synth = session.sched(instr,
                              name=instr,
                              delay=delay,
                              priority=-1,
                              kwet=1,
                              kgaindb=self._reverbSettings.get('gaindb', config['reverbGaindb']),
                              kdelayms=self._reverbSettings.get('delayms', config['reverbDelayms']),
                              kdecay=self._reverbSettings.get('decay', config['reverbDecay']),
                              kdamp=self._reverbSettings.get('damp', config['reverbDamp']),
                              whenfinished=whenfinished)
        synth._setCallback = setfunc
        self._reverbSynth = synth
        return synth

    def isAudioSessionActive(self) -> bool:
        """
        Returns True if the sound engine is active
        """
        name = self.config['play.engineName']
        import csoundengine
        return name in csoundengine.Engine.activeEngines

    @property
    def presetManager(self) -> presetmanager.PresetManager:
        """
        Returns the preset manager for this Workspace
        
        At the time there is one preset manager shared by all workspaces
        
        Returns:
            The preset manager
        """
        if self._presetManager is None:
            from . import presetmanager
            self._presetManager = presetmanager.presetManager
        return self._presetManager


def getWorkspace() -> Workspace:
    """
    Get the active workspace

    The active Workspace can also be accessed via ``Workspace.active``
    (a class variable)
    
    Example
    ~~~~~~~

    Modify the currect scorestructure

        >>> from maelzel.core import *
        >>> w = getWorkspace()
        >>> w.scorestruct = '''3/4, 4=72'''

    The active workspace can always be accessed directly:

        >>> w = Workspace.active
        >>> w is getWorkspace()
        True

    This function is so often needed that there is a shorthand: ``ws``

        >>> w = ws()
        >>> w is Workspace.active
        True

    Create a clone of the current workspace and set it as active:

        >>> w = getWorkspace().clone(active=True)
        >>> w.config = CoreConfig(...)

    """
    assert Workspace.active is not None
    return Workspace.active


def _reverbEvent(session: csoundengine.session.Session, instrname='', kind='mainreverb') -> csoundengine.synth.Synth | None:
    for event in session.namedEvents.values():
        instr = session.getInstr(event.instrname)
        if instr and instr.properties.get('kind') == kind:
            if not instrname or instrname == instr.name:
                return event
    return None


@cache
def _presetsPath() -> str:
    datadirbase = _appdirs.user_data_dir("maelzel")
    path = os.path.join(datadirbase, "core", "presets")
    return path


Workspace._initclass()
ws = getWorkspace