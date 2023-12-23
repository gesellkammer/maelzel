"""
This module handles playing of events

"""
from __future__ import annotations
from functools import cache

import numpy as np
import csoundengine

from maelzel.core._common import logger, prettylog
from maelzel.core.presetdef import PresetDef
from maelzel.core import presetmanager
from maelzel.core.workspace import getConfig, Workspace
from maelzel.core import environment
from maelzel.core import _dialogs
from maelzel.core import _playbacktools
from maelzel.core.synthevent import SynthEvent
from maelzel.core.renderer import Renderer
from maelzel.core.automation import SynthAutomation
from maelzel import _util


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import csoundengine.baseschedevent
    from typing import Sequence, Callable
    from .mobj import MObj


__all__ = (
    'play',
    'testAudio',
    'playSession',
)


class RealtimeRenderer(Renderer):
    """
    A RealtimeRenderer is created whenever realtime playback is initiated.

    Normally a user does not create a RealtimeRenderer. It is created during
    playback. There are three gateways to initiate a playback process:

    * :meth:`maelzel.core.MObj.play`. The play method of an object
      (a :class:`~maelzel.core.event.Note`, a :class:`~maelzel.core.chain.Chain`, a
      :class:`~maelzel.core.score.Score`)
      is called, this creates a :class:`RealtimeRenderer` which immediately calls its
      :meth:`RealtimeRenderer.schedEvents` method
    * :func:`maelzel.core.playback.play`. This initiates playback for multiple
      objects / events and syncs the playback as if all the objects where part
      of a group.
    * :func:`maelzel.core.playback.synchedplay`. This context manager acts very similar
      to the `play` function, ensureing that playback is synched.

    """

    def __init__(self, engine: csoundengine.Engine = None):
        super().__init__(presetManager=presetmanager.presetManager)
        if engine is None:
            engine = _playEngine()
        self.engine: csoundengine.Engine = engine
        self.session: csoundengine.session.Session = engine.session()

    def isRealtime(self) -> bool:
        return True

    def assignBus(self, kind='', value: float = None, persist=False
                  ) -> csoundengine.busproxy.Bus:
        return self.session.assignBus(kind=kind, value=value, persist=persist)

    def releaseBus(self, bus: int | csoundengine.busproxy.Bus):
        if isinstance(bus, int):
            self.engine.releaseBus(bus)
        else:
            bus.release()

    def registerPreset(self, presetdef: PresetDef) -> bool:
        instr = presetdef.getInstr()
        # The Session itself caches instrs and checks definitions
        isnew = self.session.registerInstr(instr)
        if isnew:
            logger.debug(f"*********** Session registered new instr: '{instr.name}'")
        self.registeredPresets[presetdef.name] = presetdef
        return isnew

    def isInstrDefined(self, instrname: str) -> bool:
        return self.session.getInstr(instrname) is not None

    def getInstr(self, instrname: str) -> csoundengine.instr.Instr:
        instr = self.session.getInstr(instrname)
        if instr is None:
            raise ValueError(f"Instrument '{instrname}' unknown. Possible instruments: "
                             f"{self.session.registeredInstrs().keys()}")
        return instr

    def prepareInstr(self, instr: str | csoundengine.instr.Instr, priority: int
                     ) -> bool:
        """
        Prepare the given Instr for scheduling at the given priority

        Args:
            instr: the Instr or the instr name
            priority: the priority to schedule the instr at

        Returns:
            True if the audio engine needs sync

        """
        instrname = instr if isinstance(instr, str) else instr.name
        reifiedinstr, needssync = self.session.prepareSched(instrname, priority=priority)
        return needssync

    def prepareSessionEvent(self, event: csoundengine.event.Event
                            ) -> bool:

        _, needssync = self.session.prepareSched(instr=event.instrname,
                                                 priority=event.priority)
        return needssync

    def _schedSessionEvent(self, event: csoundengine.event.Event
                           ) -> csoundengine.synth.Synth:
        assert event.instrname in self.session.instrs
        return self.session.schedEvent(event)

    def schedDummyEvent(self, dur=0.001) -> csoundengine.synth.Synth:
        """
        Schedule a dummy synth

        Args:
            dur: the duration of the synth

        Returns:
            a Synth
        """
        return _dummySynth(dur=dur, engine=self.engine)

    def getSynth(self, token: int) -> csoundengine.synth.Synth | None:
        return self.session.getSynthById(token)

    def schedEvent(self, event: SynthEvent | csoundengine.event.Event
                   ) -> csoundengine.synth.Synth:
        if isinstance(event, SynthEvent):
            if event.initfunc:
                event.initfunc(event, self)
            instr = self.presetManager.getInstr(event.instr)
            pfields5, dynargs = event._resolveParams(instr)
            return self.sched(instrname=event.instr,
                              delay=event.delay,
                              dur=event.dur,
                              args=pfields5,
                              priority=event.priority,
                              whenfinished=event.whenfinished,
                              **dynargs)
        else:
            return self._schedSessionEvent(event)

    def schedEvents(self,
                    coreevents: list[SynthEvent],
                    sessionevents: list[csoundengine.event.Event] = None,
                    whenfinished: Callable = None
                    ) -> csoundengine.synth.SynthGroup:
        """
        Schedule core and session events in sync

        All initialization is done beforehand so that it can be ensured
        that the events are scheduled in sync without the need to set a fixed
        extra latency. This is important when events include playing sound fonts,
        louding large amounts of data/samples, etc

        Args:
            coreevents: the core events to schedule
            sessionevents: the csound events to schedule
            whenfinished: a callable to be fired when the synths have finished

        Returns:
            a :class:`csoundengine.synth.Synthgroup` with all the scheduled synths
        """
        synths, sessionsynths = _schedEvents(self,
                                             presetManager=self.presetManager,
                                             coreevents=coreevents,
                                             sessionevents=sessionevents,
                                             whenfinished=whenfinished)
        numevents = len(coreevents) + (len(sessionevents) if sessionevents else 0)
        assert len(synths) == numevents, f"{len(synths)=}, {numevents=}"
        synths.extend(sessionsynths)
        return csoundengine.synth.SynthGroup(synths)

    def includeFile(self, path: str) -> None:
        self.session.includeFile(path)

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int = 0,
                  sr: int = 0,
                  tabnum: int = 0
                  ) -> int:
        table = self.session.makeTable(data=data, size=size, sr=sr, tabnum=tabnum)
        return table.tabnum

    def readSoundfile(self,
                      soundfile: str,
                      chan=0,
                      skiptime=0.
                      ) -> int:
        table = self.session.readSoundfile(path=soundfile, chan=chan, skiptime=skiptime)
        return table.tabnum

    def sched(self,
              instrname: str,
              delay: float = 0.,
              dur: float = -1,
              priority: int = 1,
              args: list[float | str] | dict[str, float] = None,
              whenfinished: Callable = None,
              relative=True,
              **kws: dict[str, float | str],
              ):
        return self.session.sched(instrname=instrname,
                                  delay=delay,
                                  dur=dur,
                                  args=args,
                                  priority=priority,
                                  whenfinished=whenfinished,
                                  relative=relative,
                                  **kws)

    def sync(self):
        self.engine.sync()

    def pushLock(self):
        """Lock the sound engine's clock, to schedule a number of synths in sync.

        This is mostly used internally
        """
        # self.engine.sync()
        self.engine.pushLock()

    def popLock(self):
        """Pop a previously pushed clock lock

        This method is mostly used internally
        """
        self.engine.popLock()

# <------------------- end RealtimeRenderer


def testAudio(duration=4, period=0.5, numChannels: int = None, delay=0.5,
              backend: str = None
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
        logger.info("Starting engine...")
        engine = _playEngine(numchannels=numChannels, backend=backend)
    engine.testAudio(dur=duration, period=period, delay=delay)


def getAudioDevices(backend=''
                    ) -> tuple[list[csoundengine.csoundlib.AudioDevice],
                               list[csoundengine.csoundlib.AudioDevice]]:
    """
    Returns (indevices, outdevices), where each of these lists is an AudioDevice.

    Args:
        backend: specify a backend supported by your installation of csound.
            None to use a default for you OS

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
    return csoundengine.csoundlib.getAudioDevices(backend=backend)


def _playEngine(numchannels: int = None,
                backend: str = None,
                outdev: str = None,
                verbose: bool = None,
                buffersize: int = None,
                latency: float = None,
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
        backend: the audio backend used, overrides config 'play.backend'
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
    config = Workspace.getActive().config
    engineName = config['play.engineName']
    if engine := csoundengine.getEngine(engineName):
        if any(_ is not None for _ in (numchannels, backend, outdev, verbose, buffersize, latency)):
            prettylog('WARNING',
                      "\nThe sound engine has been started already. Any configuration passed "
                      f"will have no effect. To modify the configuration of the engine first "
                      f"stop the engine (`playEngine().stop()`) and call `playEngine(...)` "
                      f"with the desired configuration. "
                      f"\nCurrent sound engine: {engine}")
        return engine
    numchannels = numchannels or config['play.numChannels']
    if backend == "?":
        backends = [b.name for b in csoundengine.csoundlib.audioBackends(available=True)]
        backend = _dialogs.selectFromList(backends, title="Select Backend")
    backend = backend or config['play.backend']
    verbose = verbose if verbose is not None else config['play.verbose']
    logger.debug(f"Starting engine {engineName} (nchnls={numchannels})")
    latency = latency if latency is not None else config['play.schedLatency']
    engine = csoundengine.Engine(name=engineName,
                                 nchnls=numchannels,
                                 backend=backend,
                                 outdev=outdev,
                                 globalcode=presetmanager.presetManager.csoundPrelude,
                                 quiet=not verbose,
                                 latency=latency,
                                 buffersize=buffersize,
                                 a4=config['A4'])
    waitAfterStart = config['play.waitAfterStart']
    if waitAfterStart > 0:
        import time
        time.sleep(waitAfterStart)
    # We create the session as soon as possible, to configure the engine for
    # the session's reserved instrument ranges / tables
    _ = engine.session()
    return engine


def stopSynths():
    """
    Stops all synths (notes, chords, etc.) being played

    If stopengine is True, the play engine itself is stopped
    """
    playSession().unschedAll(future=True)


def playSession(numchannels: int = None,
                backend: str = None,
                outdev: str = None,
                verbose: bool = None,
                buffersize: int = None,
                latency: float = None
                ) -> csoundengine.session.Session:
    """
    Returns the csoundengine Session

    If a Session is already present, the active session is returned. In this case,
    any arguments passed are ignored.

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

    Returns:
        the active Session

    .. seealso:: :class:`csoundengine.Session <https://csoundengine.readthedocs.io/en/latest/api/csoundengine.session.Session.html>`
    """
    if isSessionActive():
        return _playEngine().session()
    engine = _playEngine(numchannels=numchannels, backend=backend, outdev=outdev,
                         verbose=verbose, buffersize=buffersize, latency=latency)
    return engine.session()


def isSessionActive() -> bool:
    """
    Returns True if the sound engine is active
    """
    name = getConfig()['play.engineName']
    return csoundengine.getEngine(name) is not None


def _dummySynth(dur=0.001, engine: csoundengine.Engine = None) -> csoundengine.synth.Synth:
    if not engine:
        engine = _playEngine()
    session = engine.session()
    return session.sched('.dummy', 0, dur)


def play(*sources: MObj | Sequence[SynthEvent] | csoundengine.event.Event,
         whenfinished: Callable = None,
         **eventparams
         ) -> csoundengine.synth.SynthGroup | SynchronizedContext:
    """
    Play a sequence of objects / events

    When playing multiple objects via their respective .play method, initialization
    (loading soundfiles, soundfonts, etc.) might result in events getting out of sync
    with each other.

    This function first collects all events; any initialization is done beforehand
    as to ensure that events keep in sync. After initialization all events are scheduled
    and their synths are gathered in a SynthGroup

    This function can also be used as context manager if not given any sources (see
    example below)

    .. note::

        To customize playback use the ``.events`` method, which works exactly like
        ``.play`` but returns the data so that it can be played later.

    Args:
        sources: a possibly nested sequence of MObjs or events as returned from
            :meth:`MObj.events`. Empty when used as a context manager.
        whenfinished: a callback taking no arguments and returning None. It will be called
            when the last event is finished
        eventparams: any keyword arguments will be passed to :meth:`MObj.events` if
            events need to be generated

    Returns:
        A SynthGroup holding all scheduled synths

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> from csoundengine.session import SessionEvent
        >>> import csoundengine as ce
        >>> session = playSession()
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
        >>>     Chord("4C 4E", 7, start=1).events(position=0.5),
        >>>     Note("4C#", 6, offset=1.5),  # No customization,
        >>>     SessionEvent('reverb', dur=10, args={'kfeedback': 0.8}, priority=2),
        >>>     SessionEvent('sin', delay=0.1, dur=3, args={'imidi': 61.33, 'iamp':0.02})
        >>> )

    As context manager

        >>> note = Note(...)
        >>> clip = Clip(...)
        >>> with play() as p:
        ...     note.play(...)
        ...     clip.play(...)

    .. seealso::

        :class:`Synched`, :func:`render`, :meth:`MObj.play() <maelzel.core.mobj.MObj.play>`,
        :meth:`MObj.events() <maelzel.core.mobj.MObj.events>`

    """
    if not sources:
        return SynchronizedContext(whenfinished=whenfinished)

    coreevents, sessionevents = _playbacktools.collectEvents(events=sources,
                                                             eventparams=eventparams,
                                                             workspace=Workspace.active)
    numChannels = _playbacktools.nchnlsForEvents(coreevents)
    if not isSessionActive():
        _playEngine(numchannels=numChannels)
    else:
        engine = _playEngine()
        assert engine.nchnls is not None
        if engine.nchnls < numChannels:
            logger.error("Some events output to channels outside of the engine's range")

    rtrenderer = RealtimeRenderer()
    return rtrenderer.schedEvents(coreevents=coreevents, sessionevents=sessionevents, whenfinished=whenfinished)


def _schedEvents(renderer: RealtimeRenderer,
                 coreevents: list[SynthEvent],
                 presetManager: presetmanager.PresetManager,
                 sessionevents: list[csoundengine.event.Event] = None,
                 posthook: Callable[[list[csoundengine.synth.Synth]], None] | None = None,
                 whenfinished: Callable = None,
                 locked=True
                 ) -> tuple[list[csoundengine.synth.Synth], list[csoundengine.synth.Synth]]:
    """
    Schedule events in synch

    Args:
        renderer: the renderer
        coreevents: all SynthEvents generated by maelzel.core objects
        presetManager: the preset manager to translate presets in real csoundengine instrs
        sessionevents: pure csoundengine events
        posthook: a callback to be called after events have been scheduled
        whenfinished: a callback to be fired when all events here are finished
        locked: lock the renderer's clock

    Returns:
        a tuple (coresynths, sessionsynths) where coresynths is a list of Synths generated
        by the core events (one synth per SynthEvent) and sessionsynths are the synths
        generated by the sessionevents (one synth per session event)
    """
    needssync = renderer.prepareEvents(events=coreevents, sessionevents=sessionevents)
    resolvedParams = [ev._resolveParams(instr=presetManager.getInstr(ev.instr))
                      for ev in coreevents]

    if whenfinished and renderer.isRealtime():
        lastevent = max(coreevents, key=lambda ev: ev.end if ev.end > 0 else float('inf'))
        lastevent.whenfinished = lambda id: whenfinished() if not lastevent.whenfinished else lambda id, ev=lastevent: ev.whenfinished(id) or whenfinished()

    if needssync:
        renderer.sync()

    if len(coreevents) + (0 if not sessionevents else len(sessionevents)) < 2:
        locked = False

    if locked:
        renderer.pushLock()  # <---------------- Lock

    synths: list[csoundengine.synth.Synth] = []
    for coreevent, (pfields5, dynargs) in zip(coreevents, resolvedParams):
        if coreevent.gain == 0:
            synths.append(renderer.schedDummyEvent())
            continue

        synth = renderer.sched(PresetDef.presetNameToInstrName(coreevent.instr),
                               delay=coreevent.delay,
                               dur=coreevent.dur,
                               args=pfields5,
                               priority=coreevent.priority,
                               whenfinished=coreevent.whenfinished,
                               **dynargs)
        synths.append(synth)
        if coreevent.automationSegments:
            instr = presetManager.getInstr(coreevent.instr)
            for segment in coreevent.automationSegments:
                if segment.pretime is None:
                    # a point
                    synth.set(segment.param, segment.value, delay=segment.time)
                else:
                    # a segment
                    if (prevalue := segment.prevalue) is None:
                        prevalue = instr.dynamicParams().get(segment.param)
                        if prevalue is None:
                            raise ValueError(f"Default value for {segment.param} not known, "
                                             f"default values: {instr.dynamicParams()} (instr={instr})")
                    pairs = [0, prevalue,
                             segment.time - segment.pretime, segment.value]
                    synth.automate(param=segment.param, pairs=pairs, delay=segment.pretime)
        if coreevent.automations:
            for automation in coreevent.automations.values():
                synth.automate(param=automation.param, pairs=automation.data, delay=automation.delay)

    if posthook:
        posthook(synths)

    if sessionevents:
        sessionsynths = [renderer._schedSessionEvent(ev) for ev in sessionevents]
        # synths.extend(sessionsynths)
    else:
        sessionsynths = []

    if locked:
        renderer.popLock()  # <----------------- Unlock
    return synths, sessionsynths


class SynchronizedContext(Renderer):
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
        >>> session = playSession()  # returns a csoundengine.session.Session
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
                 whenfinished: Callable = None,
                 display=False):

        super().__init__(presetManager=presetmanager.presetManager)

        self.session: csoundengine.session.Session = playSession()
        """The corresponding Session, can be used to access the session during the context"""

        self.engine: csoundengine.Engine = self.session.engine
        """The play engine, can be used during the context"""

        self.synthgroup: csoundengine.synth.SynthGroup | None = None
        """A SynthGroup holding all scheduled synths during the context"""

        self.workspace: Workspace = Workspace.getActive()
        """The workspace active as the context manager is created"""

        self._instrDefs: dict[str, csoundengine.instr.Instr] = {}
        """An index of registered Instrs"""

        self._automationEvents: list[SynthAutomation] = []
        """A list of all the automation events scheduled"""

        self._futureSynths: list[_FutureSynth] = []
        """A list of all synths scheduled"""

        self._tokenToFuture: dict[int, _FutureSynth] = {}
        """Maps token to FutureSynth"""

        self._prevRenderer = None
        """The previous active renderer, if any"""

        self._tokenToSynth: dict[int, csoundengine.synth.Synth] = {}
        """Maps token to actual Synth"""

        self._synthCount = 0
        """A counter to generate tokens"""

        self._prevSessionSchedCallback = None
        self._finishedCallback = whenfinished
        self._displaySynthAtExit = display
        self._enterActions: list[Callable[[SynchronizedContext], None]] = []
        self._exitActions: list[Callable[[SynchronizedContext], None]] = []
        self._coresynths: list[csoundengine.synth.Synth] = []
        self._sessionsynths: list[csoundengine.synth.Synth] = []

        # self._eventToToken: dict[csoundengine.event.Event | SynthEvent, int] = {}

    def isRealtime(self) -> bool:
        return True

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int = 0,
                  sr: int = 0,
                  tabnum: int = 0
                  ) -> int:
        return self.session.makeTable(data=data, size=size, sr=sr, tabnum=tabnum).tabnum

    def readSoundfile(self, soundfile: str, chan=0, skiptime=0.) -> int:
        return self.session.readSoundfile(path=soundfile, chan=chan, skiptime=skiptime).tabnum

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
                    sessionevents: list[csoundengine.event.Event] = None,
                    whenfinished: Callable = None
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

    def _automate(self,
                  token: int,
                  param: str,
                  pairs: Sequence[float] | np.ndarray,
                  delay=0.,
                  overtake=False):
        event = SynthAutomation(token=token, param=param, data=pairs, delay=delay, overtake=overtake)
        self._automationEvents.append(event)

    def _presetFromToken(self, token: int) -> PresetDef | None:
        future = self._tokenToFuture.get(token)
        if not future:
            return None
        event = future.event
        if not isinstance(event, SynthEvent):
            raise ValueError("The token {token} corresponds to event {event}, which is not"
                             "a maelzel.core")
        presetdef = self.presetManager.getPreset(event.instr)
        return presetdef

    def _set(self, token: int, param: str, value: float, delay: float):
        presetdef = self._presetFromToken(token)
        if presetdef is None:
            raise RuntimeError(f"Unknown token {token}")
        params = presetdef.dynamicParams(aliases=True, aliased=True)
        if param not in params:
            raise KeyError(f"Parameter {param} not known. Possible parameters: {params}")
        event = SynthAutomation(token=token, param=param, data=[0, value], delay=delay)
        self._automationEvents.append(event)

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

    def _scheduleAutomations(self, synths: list[csoundengine.synth.Synth]
                             ) -> None:
        """
        This is called as callback with the synths generated from the future events

        There should be a 1:1 correspondence between the scheduled core events,
        the tokens and the synths.

        """
        for event in self._automationEvents:
            if event.token is None:
                logger.error(f"Automation event {event} has no valid token (token is None)")
                continue
            if event.token < 0 or event.token >= len(synths):
                logger.error(f"Token out of range in automation event {event}, "
                             f"token={event.token}, number of synths: {len(synths)}")
                continue
            synth = synths[event.token]
            if isinstance(event.data, float):
                synth.set(param=event.param, value=event.data, delay=event.delay)
            elif len(event.data) == 2:
                t, v = event.data
                delay = event.delay + t
                synth.set(param=event.param, value=v, delay=delay)
            else:
                synth.automate(param=event.param, pairs=event.data, delay=event.delay)

    def __enter__(self):
        """
        Performs initialization of the context

        If not called as a context manager, this method together with `exitContext`
        can be called manually to produce the same effect.

        """
        self.workspace = workspace = Workspace.getActive()
        for action in self._enterActions:
            action(self)

        self._prevRenderer = workspace.renderer
        workspace.renderer = self
        self._prevSessionSchedCallback = self.session.setSchedCallback(self._schedSessionEvent)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Executes the operations at context end

        This includes preparing all resources and then actually
        scheduling all events
        """
        # first, restore the state prior to the context
        self.session.setSchedCallback(self._prevSessionSchedCallback)
        self._prevSessionSchedCallback = None
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

        renderer = RealtimeRenderer()
        synths, sessionsynths = _schedEvents(renderer,
                                             presetManager=self.presetManager,
                                             coreevents=[f.event for f in corefutures],
                                             sessionevents=[f.event for f in sessionfutures],
                                             posthook=self._scheduleAutomations,
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

    def sched(self,
              instrname: str,
              delay=0.,
              dur=-1.,
              priority=1,
              args: list[float] | dict[str, float] = None,
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
        :func:`playback.playSession <maelzel.core.playback.playSession>`). All instruments
        registered at this Session are immediately available for offline rendering.

        >>> from maelzel.core import *
        >>> session = playSession()
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
        kind:
    """

    def __init__(self,
                 parent: SynchronizedContext,
                 event: SynthEvent | csoundengine.event.Event,
                 token: int):
        assert isinstance(parent, SynchronizedContext)
        assert isinstance(event, (SynthEvent, csoundengine.event.Event))
        assert isinstance(token, int) and token >= 0
        if isinstance(event, SynthEvent):
            start, dur = event.start, event.dur
        else:
            start, dur = event.delay, event.dur
        super().__init__(start=start, dur=dur)
        self.parent: SynchronizedContext = parent
        self.event: SynthEvent | csoundengine.event.Event = event
        self.token: int = token
        self.kind = 'synthevent' if isinstance(event, SynthEvent) else 'sessionevent'
        self.session = parent.session

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
            return f"FutureSynth(scheduled=False, event={self.event}, token={self.token})"
        else:
            return f"FutureSynth(scheduled=True, event={self.event}, synth={self.synth()})"

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

    @property
    def instr(self) -> csoundengine.instr.Instr:
        """Get the Instr associated with the event's preset"""
        if isinstance(self.event, SynthEvent):
            return self.getPreset().getInstr()
        else:
            return self.session.getInstr(self.event.instrname)

    def set(self, param='', value: float = 0., delay=0., **kws) -> None:
        """
        Modify a named argument
        """
        if kws:
            for k, v in kws.items():
                self.set(param=k, value=v, delay=delay)

        if param:
            self.parent._set(token=self.token, param=param, value=value, delay=delay)

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
        self.parent._automate(token=self.token, param=param, pairs=pairs, delay=delay, overtake=overtake)

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
        self.parent: SynchronizedContext = synths[0].parent
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
            if param in synth.getPreset().dynamicParams(aliased=True):
                synth.set(param=param, value=value, delay=delay)
                count += 1
        if not count:
            raise KeyError(f"Parameter '{param}' unknown. "
                           f"Possible parameters: {self.dynamicParamNames(aliased=True)}")

    def automate(self,
                 param: str | int,
                 pairs: Sequence[float] | np.ndarray,
                 mode='linear',
                 delay=0.,
                 overtake=False,
                 strict=True
                 ) -> float:
        count = 0
        for futuresynth in self.synths:
            if param in futuresynth.dynamicParamNames(aliased=True):
                futuresynth.automate(param=param, pairs=pairs, delay=delay)
                count += 1
        if count == 0:
            raise KeyError(f"Unknown parameter '{param}'. "
                           f"Possible parameters: {self.dynamicParamNames(aliased=True)}")
        return 0

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
