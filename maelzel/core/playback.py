"""
This module handles playing of events

"""
from __future__ import annotations

from functools import cache
import numpy as np
import csoundengine
from csoundengine.sessionhandler import SessionHandler

from maelzel.core._common import logger
from maelzel.core.presetdef import PresetDef
from maelzel.core import presetmanager
from maelzel.core.workspace import getConfig, Workspace
from maelzel.core import environment
from maelzel.core import _playbacktools
from maelzel.core.synthevent import SynthEvent
import maelzel.core.renderer as _renderer
import maelzel.core.realtimerenderer as _realtimerenderer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import csoundengine.engine
    import csoundengine.session
    import csoundengine.baseschedevent
    import csoundengine.schedevent
    import csoundengine.tableproxy
    import csoundengine.instr
    import csoundengine.busproxy
    import csoundengine.csoundlib
    import csoundengine.synth
    import csoundengine.event
    from typing import Sequence, Callable
    from .mobj import MObj
    from maelzel.snd import audiosample
    from maelzel.core.config import CoreConfig


__all__ = (
    'play',
    'testAudio',
    'getSession',
)


class _SyncSessionHandler(SessionHandler):
    def __init__(self, renderer: _SynchronizedContext):
        self.renderer = renderer

    def schedEvent(self, event: csoundengine.event.Event):
        return self.renderer._schedSessionEvent(event)


def testAudio(duration=4, period=0.5, numChannels: int | None = None, delay=0.5,
              backend=''
              ) -> None:
    """
    Test the audio engine by sending pink to each channel

    Args:
        duration: the duration of the test
        period: how long to test each channel. Channels are cycled
        numChannels: the number of channels to use if starting the engine
        delay: how long to wait before starting the test.

    """
    engine = _playEngine(numchannels=numChannels, backend=backend)
    if not engine:
        engine = _playEngine(numchannels=numChannels, backend=backend)
        logger.info(f"Started engine, backend={engine.backend}...")
    engine.testAudio(dur=duration, period=period, delay=delay)


def getAudioDevices(backend=''
                    ) -> tuple[list[csoundengine.csoundlib.AudioDevice],
                               list[csoundengine.csoundlib.AudioDevice]]:
    """
    Returns (indevices, outdevices), where each of these lists is an AudioDevice.

    Args:
        backend: specify a backend supported by your installation of csound.
            None to use a default for you OS. Use '?' to interactively select
            a backend from a list of available options

    Returns:
        a tuple of (input devices, output devices)

    .. note::

        For jack an audio device is a client

    An AudioDevice is defined as::

        AudioDevice:
            id: str
            name: str
            kind: str
            index: int = -1
            numchannels: Optional[int] = 0

    Of is the ``.name`` attribute: this is the value that needs to be
    passed to :func:`playEngine` to select a specific device.

    .. seealso:: :func:`playEngine`
    """
    import csoundengine.csoundlib
    if backend == '?':
        backend = _selectBackend()
    return csoundengine.csoundlib.getAudioDevices(backend=backend)


def _playEngine(numchannels: int | None = None,
                backend='',
                outdev='',
                verbose: bool | None = None,
                buffersize=0,
                latency: float | None = None,
                numbuffers=0,
                config: CoreConfig | None = None,
                name=''
                ) -> csoundengine.Engine:
    """
    Get the play engine; start it if needed

    **maelzel** used csound for any sound related task (synthesis, sample
    playback, etc). To interact with csound it relies on the
    :class:`Engine <https://csoundengine.readthedocs.io/en/latest/api/csoundengine.engine.Engine.html>` class

    If an Engine is already active, that Engine is returned, even if the configuration
    is different. **To start the play engine with a different configuration, stop the engine first**

    Args:
        numchannels: the number of output channels, overrides config 'play.numChannels'
        backend: the audio backend used, overrides config 'play.backend'. Use '?' to
            interactively select a backend from a list of available options
        outdev: leave as None to use the backend's default, use '?' to select
            from a list of available devices. To list all available devices
            see :func:`getAudioDevices`
        verbose: if True, output debugging information
        buffersize: if given, use this as the buffer size. None to use a sensible
            default for the backend
        latency: an added latency

    Returns:
        the play Engine


    .. seealso:: :func:`getAudioDevices`
    """
    if config is None:
        config = Workspace.active.config
    engineName = name or config['play.engineName']
    if engine := csoundengine.Engine.activeEngines.get(engineName):
        return engine
    numchannels = numchannels or config['play.numChannels']
    if backend == "?":
        backend = _selectBackend()
        if not backend:
            raise KeyboardInterrupt
    elif not backend:
        backend = config['play.backend']
    verbose = verbose if verbose is not None else config['play.verbose']
    logger.debug(f"Starting engine {engineName} (nchnls={numchannels})")
    latency = latency if latency is not None else config['play.schedLatency']
    engine = csoundengine.Engine(name=engineName,
                                 nchnls=numchannels,
                                 backend=backend,
                                 outdev=outdev,
                                 globalcode=presetmanager.presetManager.csoundPrelude,
                                 verbose=verbose,
                                 latency=latency,
                                 buffersize=buffersize,
                                 a4=config['A4'],
                                 numbuffers=numbuffers)
    # We create the session as soon as possible, to configure the engine for
    # the session's reserved instrument ranges / tables
    session = engine.session()
    for instr in _builtinInstrs():
        session.registerInstr(instr)
    return engine


def _selectBackend() -> str:
    """
    Select a backend to use

    Returns:
        the name of the backend, or an empty string if no selection was made
    """
    import csoundengine.csoundlib
    backends = list(set(b.name for b in csoundengine.csoundlib.audioBackends()))
    backends.sort()
    from maelzel.core import _dialogs
    selectedbackend = _dialogs.selectFromList(backends, title="Select Backend")
    return selectedbackend or ''


def stopSynths():
    """
    Stops all synths (notes, chords, etc.) being played

    If stopengine is True, the play engine itself is stopped
    """
    getSession().unschedAll(future=True)


@cache
def _builtinInstrs() -> list[csoundengine.instr.Instr]:
    from csoundengine.instr import Instr
    return [
        Instr('.reverbstereo', r'''\
            |kfeedback=0.85, kwet=0.8, ichan=1, kcutoff=12000|
            a1, a2 monitor
            aL, aR  reverbsc a1, a2, kfeedback, kcutoff, sr, 0.5, 1
            aL = aL * kwet - a1 * (1 - kwet) 
            aR = aR * kwet - a2 * (1 - kwet)
            outch ichan, aL - a1, ichan+1, aR - a2
            ''')
    ]


def getSession(numchannels: int | None = None,
               backend='',
               outdev='',
               verbose: bool | None = None,
               buffersize: int = 0,
               latency: float | None = None,
               numbuffers: int = 0,
               ensure: bool = False,
               name=''
               ) -> csoundengine.session.Session:
    """
    Returns / creates the audio Session 

    If no Session has been created already, a Session is initialized
    with the given parameters and returned. Otherwise the active
    Session is returned and any parameters passed are ignored

    .. note::
        There is one audio session, shared by all workspaces. Only the
        first call to this function will initialize the session to
        specific parameters. If you need to initialize the session
        to specific values, call this function before any playback
        related functionality is used. If any playback related
        function/method is called before, the session is created
        from default values. To configure these default values see
        the configuration

    Args:
        numchannels: the number of output channels, overrides config 'play.numChannels'
        backend: the audio backend used, overrides config 'play.backend'
        outdev: leave as None to use the backend's default, use '?' to select
            from a list of available devices. To list all available devices
            see :func:`getAudioDevices`
        verbose: if True, output debugging information
        buffersize: if given, use this as the buffer size. None to use a sensible
            default for the backend
        latency: an added latency
        numbuffers: the number of buffers used by the csound engine
        ensure: if True, an exception is raised if a Session already existed
            with parameters differing from the given

    Returns:
        the active Session

    Raises:
        SessionParametersMismatchError: if ensure was True and the given parameters
            do not match the existing session

    .. seealso:: :class:`csoundengine.Session <https://csoundengine.readthedocs.io/en/latest/api/csoundengine.session.Session.html>`
    """
    if not isSessionActive(name=name):
        engine = _playEngine(name=name,
                             numchannels=numchannels,
                             backend=backend,
                             outdev=outdev,
                             verbose=verbose,
                             buffersize=buffersize,
                             latency=latency,
                             numbuffers=numbuffers)
        return engine.session()

    # Session is already active, check params
    engine = _playEngine(name=name)
    if ensure:
        for paramname, value in [('nchnls', numchannels), 
                                 ('backend', backend), 
                                 ('outdev', outdev), 
                                 ('extraLatency', latency),
                                 ('numBuffers', numbuffers),
                                 ('bufferSize', buffersize)]:
            if value is not None and (old := getattr(engine, paramname)) != value:
                raise ValueError(f"A Session already exists with {paramname}={old}, user asked {value}")
            
    return engine.session()


def isSessionActive(name='') -> bool:
    """
    Returns True if the sound engine is active
    """
    if not name:
        name = getConfig()['play.engineName']
    return name in csoundengine.Engine.activeEngines


def _dummySynth(dur=0.001, engine: csoundengine.Engine | None = None) -> csoundengine.synth.Synth:
    if not engine:
        engine = _playEngine()
    session = engine.session()
    return session.sched('.dummy', 0, dur)


def play(*sources: MObj | Sequence[SynthEvent] | csoundengine.event.Event,
         whenfinished: Callable | None = None,
         display=False,
         **eventparams
         ) -> csoundengine.synth.SynthGroup | _SynchronizedContext:
    """
    Play a sequence of objects / events in sync.  Can be used as a context manager

    When playing multiple objects via their respective .play method, initialization
    (loading soundfiles, soundfonts, etc.) might result in events getting out of sync
    with each other. This function first collects all events; any initialization is 
    done beforehand as to ensure that events keep in sync. After initialization all 
    events are scheduled and their synths are gathered in a SynthGroup

    To customize playback, use this function as a context manager or call ``.synthEvents`` 
    method on each object instead of ``.play``. ``.synthEvents`` has the same signature
    but returns the data so that it can be played later.

    Args:
        sources: a possibly nested sequence of MObjs or events as returned from
            :meth:`MObj.events`. Empty when used as a context manager.
        whenfinished: a callback taking no arguments and returning None. It will be called
            when the last event is finished
        display: if called as a context manager, the result of playback is displayed
        eventparams: any keyword arguments will be passed to :meth:`MObj.events` if
            events need to be generated

    Returns:
        A SynthGroup holding all scheduled synths

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> from csoundengine.session import SessionEvent
        >>> import csoundengine as ce
        >>> session = getSession()
        >>> session.defInstr('reverb', r'''
        >>> |kfeedback=0.85|
        ... a1, a2 monitor
        ... aL, aR  reverbsc a1, a2, kfeedback, 12000, sr, 0.5, 1
        ... outch 1, aL - a1, 2, aR - a2
        ... ''')
        >>> session.defInstr('sin', r'''
        ... |imidi=60, iamp=0.01|
        ... a1 oscili iamp, mtof(imidi)
        ... a1 *= linsegr(0, 0.5, 1, 2, 0)
        ... outch 1, a1
        ... ''')
        >>> play(
        >>>     Chord("4C 4E", 7, start=1).synthEvents(position=0.5),
        >>>     Note("4C#", 6, offset=1.5),  # No customization,
        >>>     SessionEvent('reverb', dur=10, args={'kfeedback': 0.8}, priority=2),
        >>>     SessionEvent('sin', delay=0.1, dur=3, args={'imidi': 61.33, 'iamp':0.02})
        >>> )

    As context manager:

        >>> note = Note("4C#", 6, offset=1.5)
        >>> chord = Chord("4C 4E", 7, start=1)
        >>> clip = Clip(...)
        >>> with play() as s:  # returns the audio Session used
        ...     note.play(instr='.piano')
        ...     chord.play(position=0.5)
        ...     clip.play(speed=0.5, delay=1)
        ...     s.sched('reverb, priority=2')
        ...     s.sched('sin', ...)

    .. seealso::

        :func:`render`, :meth:`MObj.play() <maelzel.core.mobj.MObj.play>`,
        :meth:`MObj.synthEvents() <maelzel.core.mobj.MObj.synthEvents>`

    """
    if not sources:
        # Used as context manager
        return _SynchronizedContext(whenfinished=whenfinished, display=display)

    coreevents, sessionevents = _playbacktools.collectEvents(events=sources,
                                                             eventparams=eventparams,
                                                             workspace=Workspace.active)
    numChannels = _playbacktools.nchnlsForEvents(coreevents)
    if not isSessionActive():
        engine = _playEngine(numchannels=numChannels)
    else:
        engine = _playEngine()
        assert engine.nchnls is not None
        if engine.nchnls < numChannels:
            logger.error("Some events output to channels outside of the engine's range")

    rtrenderer = _realtimerenderer.RealtimeRenderer(engine=engine)
    return rtrenderer.schedEvents(coreevents=coreevents, sessionevents=sessionevents, whenfinished=whenfinished)


class _SynchronizedContext(_renderer.Renderer):
    """
    Context manager to group realtime events to ensure synched playback

    **NB**: A user does not normally create a SynchronizedContext instance.
    This context manager is created when :func:`play` is called as a context
    manager (see example)

    When playing multiple objects via their respective .play method, initialization
    (loading soundfiles, soundfonts, etc.) might result in events getting out of sync
    with each other.

    Within this context all ``.play`` calls are collected and all events are
    scheduled at the end of the context. Any initialization is done beforehand
    as to ensure that events keep in sync. Pure csound events can also be
    scheduled in sync during this context, using the ``_sched`` method
    of the context manager or simply calling the .sched method of
    the active session.

    After exiting the context all scheduled synths can be
    accessed via the ``synthgroup`` attribute.

    .. note::

        Use this context manager whenever you are mixing multiple objects with
        customized play arguments, and external csoundengine instruments. As an alternative
        it is possible to use :func:`play` and wrap any pure csound event into a
        :class:`csoundengine.event.Event` (see https://csoundengine.readthedocs.io/en/latest/session.html#sessionevent-class)

    Args:
        whenfinished: call this function when the last event is finished. A function taking
            no arguments and returning None

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> session = getSession()  # returns a csoundengine.session.Session
        >>> session.defInstr('reverb', r'''
        ... |kfeedback=0.85|
        ... a1, a2 monitor
        ... aL, aR  reverbsc a1, a2, kfeedback, 12000, sr, 0.5, 1
        ... outch 1, aL - a1, 2, aR - a2
        ... ''')
        >>> chain = Chain([Note(m, 0.5) for m in range(60, 72)])
        >>> with play() as ctx:   # <--- this creates a SynchronizedContext
        ...     chain.play(instr='piano', gain=0.5)
        ...     session.sched('reverb', 0, dur=10, priority=2)
        # Within a jupyter session placing the context after exit will display the
        # html output by the SynthGroup generated during playback. This shows
        # information about the synths and creates a Stop button to cancel
        # playback at any moment
        >>> ctx
    """
    def __init__(self,
                 whenfinished: Callable | None = None,
                 display=False):

        super().__init__(presetManager=presetmanager.presetManager)

        self.session: csoundengine.session.Session = getSession()
        """The corresponding Session, can be used to access the session during the context"""

        self.engine: csoundengine.engine.Engine = self.session.engine
        """The play engine, can be used during the context"""

        self.synthgroup: csoundengine.synth.SynthGroup | None = None
        """A SynthGroup holding all scheduled synths during the context"""

        self.workspace: Workspace = Workspace.active
        """The workspace active as the context manager is created"""

        self._instrDefs: dict[str, csoundengine.instr.Instr] = {}
        """An index of registered Instrs"""

        self._futureSynths: list[_FutureSynth] = []
        """A list of all synths scheduled"""

        self._tokenToFuture: dict[int, _FutureSynth] = {}
        """Maps token to FutureSynth"""

        self._prevRenderer = None
        """The previous active renderer, if any"""

        self._prevSessionHandler: SessionHandler | None = None

        self._tokenToSynth: dict[int, csoundengine.synth.Synth] = {}
        """Maps token to actual Synth"""

        self._synthCount = 0
        """A counter to generate tokens"""

        self._prevSessionSchedCallback = None
        self._finishedCallback = whenfinished
        self._displaySynthAtExit = display
        self._enterActions: list[Callable[[_SynchronizedContext], None]] = []
        self._exitActions: list[Callable[[_SynchronizedContext], None]] = []
        self._coresynths: list[csoundengine.synth.Synth] = []
        self._sessionsynths: list[csoundengine.synth.Synth] = []
        self._insideContext = False

    def isRealtime(self) -> bool:
        return True

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int | tuple[int, int] = 0,
                  sr: int = 0,
                  tabnum: int = 0
                  ) -> csoundengine.tableproxy.TableProxy:
        if self.session._handler:
            raise NotImplementedError
        return self.session.makeTable(data=data, size=size, sr=sr, tabnum=tabnum)

    def readSoundfile(self, path: str, chan=0, skiptime=0.) -> csoundengine.tableproxy.TableProxy:
        return self.session.readSoundfile(path=path, chan=chan, skiptime=skiptime)

    def includeFile(self, path: str) -> None:
        self.session.includeFile(path)

    def _sched(self, event: csoundengine.event.Event | SynthEvent
               ) -> _FutureSynth:
        token = self._getSynthToken()
        future = _FutureSynth(parent=self, event=event, token=token)
        self._futureSynths.append(future)
        self._tokenToFuture[token] = future
        return future

    def _schedSessionEvent(self, event: csoundengine.event.Event
                           ) -> _FutureSynth:
        return self._sched(event)

    def assignBus(self, kind='', value: float | None = None, persist=False
                  ) -> csoundengine.busproxy.Bus:
        return self.session.assignBus(kind=kind, value=value, persist=persist)

    def prepareSessionEvent(self, event: csoundengine.event.Event) -> bool:
        _, needssync = self.session.prepareSched(instr=event.instrname,
                                                 priority=event.priority)
        return needssync

    def getInstr(self, instrname: str) -> csoundengine.instr.Instr:
        instr = self.session.getInstr(instrname)
        if instr is None:
            raise ValueError(f"Instrument name '{instrname}' unknown. "
                             f"Possible instruments: {self.session.registeredInstrs().keys()}")
        return instr

    def prepareInstr(self, instr: csoundengine.instr.Instr, priority: int
                     ) -> bool:
        instrname = instr if isinstance(instr, str) else instr.name
        reifiedinstr, needssync = self.session.prepareSched(instrname, priority=priority)
        return needssync

    def releaseBus(self, bus: int | csoundengine.busproxy.Bus) -> None:
        def action(self, bus=bus):
            if isinstance(bus, int):
                self.session.engine.releaseBus(int)
            else:
                bus.release()
        self._exitActions.append(action)

    def _repr_html_(self):
        if self.synthgroup is not None:
            return self.synthgroup._repr_html_()
        return repr(self)

    def show(self):
        if environment.insideJupyter:
            from IPython.display import display
            display(self)
        else:
            print(repr(self))

    def _schedDone(self) -> bool:
        return self.synthgroup is not None

    def unsched(self, token: int) -> None:
        # TODO: implement delay
        if self._schedDone():
            synth = self.getSynth(token)
            if synth is not None:
                synth.stop()
            else:
                raise ValueError(f"No synth with the given token: {token}")
        else:
            future = self._tokenToFuture.get(token)
            if future is None:
                raise ValueError(f"Token {token} unknown")
            self._removeFuture(future)

    def _removeFuture(self, future: _FutureSynth):
        token = future.token
        del self._tokenToFuture[token]
        self._futureSynths.remove(future)

    def _getSynthToken(self) -> int:
        token = self._synthCount
        self._synthCount += 1
        return token

    def schedEvent(self, event: SynthEvent | csoundengine.event.Event
                   ) -> _FutureSynth:
        """
        Schedule one event to be played when we exit the context

        This method is called internally when an object calls its
        .play method while this class is active as a context manager

        Args:
            event: the event to schedule

        Returns:
            a FutureSynth. It can be used to schedule automation
            or control the synth itself

        """
        return self._sched(event)

    def schedEvents(self,
                    coreevents: list[SynthEvent],
                    sessionevents: list[csoundengine.event.Event] | None = None,
                    whenfinished: Callable | None = None
                    ) -> _FutureSynthGroup:
        """
        Schedule multiple events at once

        Args:
            coreevents: the events to schedule
            sessionevents: an optional list of SessionEvents
            whenfinished: optional callback to call when the scheduled events finish

        Returns:
            a FutureSynthGroup, which can be used as a SynthGroup to control
            and automate the events as a group

        """
        allevents = []
        if coreevents:
            allevents.extend(coreevents)
        if sessionevents:
            allevents.extend(sessionevents)
        futures = [self._sched(event) for event in allevents]
        return _FutureSynthGroup(futures)

    def registerPreset(self, presetdef: PresetDef) -> bool:
        return self.session.registerInstr(presetdef.getInstr())

    def _presetFromToken(self, token: int) -> PresetDef | None:
        future = self._tokenToFuture.get(token)
        if not future:
            return None
        event = future.event
        presetdef = self.presetManager.getPreset(event.instr)
        return presetdef

    def getSynth(self, token: int) -> csoundengine.synth.Synth | None:
        """
        Get the scheduled Synth which the given token is associated with

        Args:
            token: the synth id

        Returns:
            the actual Synth or None if no synth was associated with this token

        """
        if not self.synthgroup:
            raise RuntimeError("Synths are only accessible after render")
        return self._tokenToSynth.get(token)

    def __enter__(self):
        """
        Performs initialization of the context

        """
        if self._insideContext:
            raise RuntimeError("Alread inside this context")
        self.workspace = workspace = Workspace.active
        for action in self._enterActions:
            action(self)

        self._prevRenderer = workspace.renderer
        workspace.renderer = self
        # self._prevSessionSchedCallback = self.session.setSchedCallback(self._schedSessionEvent)
        self._prevSessionHandler = self.session.setHandler(_SyncSessionHandler(self))
        self._insideContext = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Executes the operations at context end

        This includes preparing all resources and then actually
        scheduling all events
        """
        # first, restore the state prior to the context
        # self.session.setSchedCallback(self._prevSessionSchedCallback)
        # self._prevSessionSchedCallback = None
        self._insideContext = False
        self.session.setHandler(self._prevSessionHandler)
        self._prevSessionHandler = None

        self.workspace.renderer = self._prevRenderer
        self._prevRenderer = None

        if exc_type is not None:
            # There was an exception since entering
            logger.warning("Playing aborted")
            return

        if not self._futureSynths:
            logger.debug("No events scheduled, exiting context")
            self.synthgroup = None
            return

        corefutures = [f for f in self._futureSynths if f.kind == 'synthevent']
        sessionfutures = [f for f in self._futureSynths if f.kind == 'sessionevent']

        renderer = _realtimerenderer.RealtimeRenderer(engine=self.engine)
        coreevents = [f._synthevent() for f in corefutures]
        sessionevents = [f._csoundevent() for f in sessionfutures]
        synths, sessionsynths = renderer._schedEvents(presetManager=self.presetManager,
                                                      coreevents=coreevents,
                                                      sessionevents=sessionevents,
                                                      whenfinished=self._finishedCallback)

        for idx, synth in enumerate(synths):
            token = corefutures[idx].token
            self._tokenToSynth[token] = synth

        for idx, synth in enumerate(sessionsynths):
            token = sessionfutures[idx].token
            self._tokenToSynth[token] = synth

        for func in self._exitActions:
            func(self)

        self._coresynths = synths
        self._sessionsynths = sessionsynths

        self.synthgroup = csoundengine.synth.SynthGroup(synths + sessionsynths)

        if self._displaySynthAtExit:
            self.show()

    def playSample(self,
                   source: int | str | csoundengine.tableproxy.TableProxy | tuple[np.ndarray, int] | audiosample.Sample,
                   delay=0.,
                   dur=-1,
                   chan=1,
                   gain=1.,
                   speed=1.,
                   loop=False,
                   pan=0.5,
                   skip=0.,
                   fade: float | tuple[float, float] | None = None,
                   crossfade=0.02,
                   ) -> _FutureSynth:
        """
        Play a sample through this renderer

        Args:
            source: a soundfile, a TableProxy, a tuple (samples, sr) or a maelzel.snd.audiosample.Sample
            delay: when to play
            dur: the duration. -1 to play until the end (will detect the end of the sample)
            chan: the channel to output to
            gain: a gain applied
            speed: playback speed
            loop: should the sample be looped?
            pan: the panning position
            skip: time to skip from the audio sample
            fade: a fade applied to the playback
            crossfade: a crossfade time when looping

        Returns:
            a :class:`_FutureSynth`
        """
        # TODO: make a FutureSynth event instead
        from maelzel.snd import audiosample
        if isinstance(source, tuple):
            data, sr = source
            source = self.session.makeTable(data=data, sr=sr)
        elif isinstance(source, audiosample.Sample):
            source = self.session.makeTable(data=source.samples, sr=source.sr)
        event = self.session.makeSampleEvent(source=source, delay=delay, dur=dur, chan=chan, gain=gain,
                                             speed=speed, loop=loop, pan=pan, skip=skip, fade=fade,
                                             crossfade=crossfade)
        return self._schedSessionEvent(event=event)

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              args: list[float] | dict[str, float] | None = None,
              whenfinished: Callable | None = None,
              **kws: float | str) -> _FutureSynth:
        """
        Schedule a csound event in the active Session

        This is an internal method. The user can simply call .sched on the session
        itself and it will be redirected here via its schedule callback.

        Args:
            instrname: the instr. name
            delay: start time
            dur: duration
            priority: priority of the event
            args: any parameters passed to the instr. Either a list of values
                (starting with p5) or a dict mapping parameter name to its value.
            whenfinished: dummy arg, just included to keep the signature of Session.sched
            **kws: named pfields

        Returns:
            a csoundengine's Event

        Example
        ~~~~~~~

        Schedule a reverb at a higher priority to affect all notes played. Notice
        that the reverb instrument is declared at the Session (see
        :func:`playback.getSession <maelzel.core.playback.getSession>`). All instruments
        registered at this Session are immediately available for offline rendering.

        >>> from maelzel.core import *
        >>> session = getSession()
        >>> session.defInstr('reverb', r'''
        >>> |kfeedback=0.85|
        ... a1, a2 monitor
        ... aL, aR  reverbsc a1, a2, kfeedback, 12000, sr, 0.5, 1
        ... outch 1, aL - a1, 2, aR - a2
        ... ''')
        >>> chain = Chain([Note(m, 0.5) for m in range(60, 72)])
        >>> with play():
        >>>     synth = chain.play(position=1, instr='piano')
        >>>     synth.automate(...)
        >>>     session.sched('reverb', 0, dur=10, priority=2, args={'kfeedback':0.9})
        """
        assert isinstance(instrname, str), f"{instrname=}"
        if instrname not in self.session.instrs:
            logger.error(f"Unknown instrument {instrname}. "
                         f"Defined instruments: {self.session.registeredInstrs().keys()}")
            raise ValueError(f"Instrument {instrname} unknown")
        event = csoundengine.event.Event(instrname=instrname,
                                         delay=delay,
                                         dur=dur,
                                         priority=priority,
                                         args=args,
                                         kws=kws)
        return self._schedSessionEvent(event)


class _FutureSynth(csoundengine.baseschedevent.BaseSchedEvent, csoundengine.synth.ISynth):
    """
    A FutureSynth is a handle to a future synth within a SynchronizedContext

    Whenever a synthevent is scheduled, a FutureSynth is created with a token
    mapped to the future event. After all synthevents have been gathered, they are
    initialized and scheduled in one operation, thus making synchronisation
    as tight and efficient as if they had been scheduled within a builtin
    structure like a voice or score.

    Args:
        parent: the parent of this synth
        event: the event this synth is wrapping
        token: an integer to map this synth to the real Synth when it is
            scheduled
        kind: ??
    """

    def __init__(self,
                 parent: _SynchronizedContext,
                 event: SynthEvent | csoundengine.event.Event,
                 token: int):
        assert isinstance(parent, _SynchronizedContext)
        assert isinstance(event, (SynthEvent, csoundengine.event.Event))
        assert isinstance(token, int) and token >= 0
        if isinstance(event, SynthEvent):
            start, dur = event.start, event.dur
        else:
            start, dur = event.delay, event.dur
        super().__init__(start=start, dur=dur)
        self.parent: _SynchronizedContext = parent
        self.event: SynthEvent | csoundengine.event.Event = event
        self.token: int = token
        self.kind = 'synthevent' if isinstance(event, SynthEvent) else 'sessionevent'
        self.session = parent.session

    def _synthevent(self) -> SynthEvent:
        if isinstance(self.event, SynthEvent):
            return self.event
        raise ValueError(f"This FutureSynth has an event of type {type(self.event)}")

    def _csoundevent(self) -> csoundengine.event.Event:
        if isinstance(self.event, csoundengine.event.Event):
            return self.event
        raise ValueError(f"This FutureSynth has an event of type {type(self.event)}")


    def aliases(self) -> dict[str, str]:
        return self.instr.aliases

    def controlNames(self, aliases=True, aliased=False) -> frozenset[str]:
        return self.instr.controlNames(aliases=aliases, aliased=aliased)

    def pfieldNames(self, aliases=True, aliased=False) -> frozenset[str]:
        return self.instr.pfieldNames(aliases=aliases, aliased=aliased)

    def paramValue(self, param: str) -> float | str | None:
        event = self.event
        if isinstance(event, csoundengine.event.Event):
            if event.kws and param in event.kws:
                return event.kws[param]
            instr = self.session.getInstr(event.instrname)
            assert instr is not None
            return instr.paramValue(param)
        else:
            return event.paramValue(param)

    def synth(self) -> csoundengine.synth.Synth:
        """
        Access the associated Synth

        Notice that this method can only be called after the synth has actually
        been scheduled. Normally this is after exiting the synchronized context

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> notes = [Note(...), Note(...)]
            >>> with play():
            >>>     futures = [note.play() for note in notes]
            >>> for future in futures:
            >>>     future.synth().set(...)
        """
        synth = self.parent.getSynth(self.token)
        if synth is None:
            raise RuntimeError(f"Synth for this FutureSynth ({self}) has not been scheduled")
        return synth

    def scheduled(self) -> bool:
        return self.parent.synthgroup is not None

    def __repr__(self):
        scheduled = self.scheduled()
        if not scheduled:
            return f"FutureSynth(scheduled=False, token={self.token}, event={self.event})"
        else:
            return f"FutureSynth(scheduled=True, token={self.token}, event={self.event}, synth={self.synth()})"

    def _repr_html_(self):
        if self.scheduled():
            synth = self.synth()
            return synth._repr_html_()
        else:
            return repr(self)

    def getPreset(self) -> PresetDef:
        """Get the preset definition for the instr used in this event"""
        if isinstance(self.event, SynthEvent):
            return self.event.getPreset()
        else:
            raise ValueError(f"This _FutureSynth wraps a session event and "
                             f"has no preset, event={self.event}")

    def dynamicParamNames(self, aliases=True, aliased=False) -> frozenset[str]:
        return self.instr.dynamicParamNames(aliases=aliases, aliased=aliased)

    @property
    def instr(self) -> csoundengine.instr.Instr:
        """Get the Instr associated with the event's preset"""
        if isinstance(self.event, SynthEvent):
            return self.event.getPreset().getInstr()
        else:
            instr = self.session.getInstr(self.event.instrname)
            if instr is None:
                raise RuntimeError(f"Could not find this event's instr. {self=}")
            return instr

    def set(self, param='', value: float = 0., delay=0., **kws) -> None:
        """
        Modify a named argument
        """
        if kws:
            for k, v in kws.items():
                self.set(param=k, value=v, delay=delay)

        if param:
            dynparams = self.dynamicParamNames(aliased=True)
            if param not in dynparams:
                raise KeyError(f"Parameter {param} not known for instr "
                               f"{self.instr.name}. Possible parameters: {dynparams}")
            else:
                self.event.set(param=param, value=value, delay=delay)
                # self.parent._set(token=self.token, param=param, value=value, delay=delay)

    def _setPfield(self, param: str, value: float, delay=0.) -> None:
        self.set(param=param, value=value, delay=delay)

    def _setTable(self, param: str, value: float, delay=0.) -> None:
        self.set(param=param, value=value, delay=delay)

    def automate(self,
                 param: int | str,
                 pairs: Sequence[float] | np.ndarray,
                 mode="linear",
                 delay=0.,
                 overtake=False,
                 ) -> None:
        # TODO: implement overtake and mode
        params = self.instr.dynamicParams(aliases=True, aliased=True)
        if param not in params:
            raise ValueError(f"Parameter {param} unknown for instr {self.instr}. "
                             f"Possible parameters: {params}")
        if isinstance(self.event, SynthEvent):
            self.event.automate(param=param, pairs=pairs, delay=delay, interpolation=mode, overtake=overtake)
        else:
            # A Session event
            self.event.automate(param=param, pairs=pairs, delay=delay, interpolation=mode, overtake=overtake)  # type: ignore

    def stop(self, delay=0.) -> None:
        """ Stop this synth """
        if not self.scheduled():
            self.parent.unsched(self.token)
        else:
            self.synth().stop(delay=delay)

    def playing(self) -> bool:
        """ Is this synth playing? """
        return self.scheduled() and self.synth().playing()

    def finished(self) -> bool:
        """ Has this synth ceased to play? """
        return self.scheduled() and self.synth().finished()

    def ui(self, **specs: tuple[float, float]) -> None:
        if self.scheduled():
            return self.synth().ui(**specs)
        else:
            raise RuntimeError("This synth has not been scheduled yet")


class _FutureSynthGroup(csoundengine.baseschedevent.BaseSchedEvent):

    def __init__(self, synths: list[_FutureSynth]):
        self.synths: list[_FutureSynth] = synths
        self.parent: _SynchronizedContext = synths[0].parent
        start = min(synth.start for synth in synths)
        end = max(synth.end for synth in synths)
        dur = end - start
        super().__init__(start=start, dur=dur)
        self.session = synths[0].session
        self.engine = self.session.engine

    def paramValue(self, param: str) -> float | str | None:
        if param not in self.paramNames():
            raise KeyError(f"Unknown parameter '{param}'. Possible parameters: {self.paramNames()}")
        for synth in self.synths:
            value = synth.paramValue(param)
            if value is not None:
                return value
        return None

    def dynamicParamNames(self, aliases=True, aliased=False) -> frozenset[str]:
        params = set()
        for synth in self.synths:
            params.update(synth.dynamicParamNames(aliases=aliases, aliased=aliased))
        return frozenset(params)

    def automate(self,
                 param: str | int,
                 pairs: Sequence[float] | np.ndarray,
                 mode='linear',
                 delay=0.,
                 overtake=False,
                 ) -> float:
        count = 0
        for synth in self.synths:
            if param in synth.dynamicParamNames(aliased=True):
                synth.automate(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
                count += 1
        if not count:
            possibleparams = self.dynamicParamNames(aliased=True)
            raise ValueError(f"Parameter '{param}' not known by any synth in this "
                             f"group. Possible parameters: {possibleparams}.\n "
                             f"Synths: {self.synths}")
        return 0.

    def _setPfield(self, param: str, value: float, delay=0.) -> None:
        for synth in self.synths:
            if param in synth.pfieldNames(aliased=True):
                synth._setPfield(param=param, value=value, delay=delay)

    def _setTable(self, param: str, value: float, delay=0.) -> None:
        for synth in self.synths:
            if param in synth.controlNames(aliased=True):
                synth._setTable(param=param, value=value, delay=delay)

    def set(self, param='', value: float = 0., delay=0., **kws) -> None:
        count = 0
        for synth in self.synths:
            try:
                synth.set(param=param, value=value, delay=delay, **kws)
                count += 1
            except KeyError:
                pass
        if not count:
            raise KeyError(f"Parameter '{param}' unknown. "
                           f"Possible parameters: {self.dynamicParamNames(aliased=True)}")

    def synthgroup(self) -> csoundengine.synth.SynthGroup:
        if self.parent.synthgroup is None:
            raise RuntimeError("The synths in this group have not been scheduled yet")
        synths = [futuresynth.synth() for futuresynth in self.synths]
        assert all(synth is not None for synth in synths)
        return csoundengine.synth.SynthGroup(synths)

    def __getitem__(self, idx: int):
        return self.synths[idx]

    def scheduled(self) -> bool:
        return all(synth.scheduled() for synth in self.synths)

    def stop(self, delay=0.) -> None:
        """ Stop this synthgroup """
        for synth in self.synths:
            synth.stop(delay=delay)

    def __repr__(self):
        scheduled = self.scheduled()
        return f"FutureSynthGroup(scheduled={scheduled}, synths={self.synths})"

    def _repr_html_(self):
        if self.scheduled():
            synthgroup = self.synthgroup()
            return synthgroup._repr_html_()
        else:
            return repr(self)

