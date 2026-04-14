from __future__ import annotations

from maelzel.core import renderer as _renderer
from maelzel.core.workspace import Workspace
from maelzel.core.synthevent import SynthEvent
from maelzel.core import presetmanager
from maelzel.core import environment
from maelzel.core._common import logger
import maelzel.core.realtimerenderer as _rtrenderer


import csoundengine
import csoundengine.baseschedevent
from csoundengine.sessionhandler import SessionHandler

import typing as _t
if _t.TYPE_CHECKING:
    import csoundengine.engine
    import csoundengine.session
    import csoundengine.synth
    import csoundengine.event
    import csoundengine.instr
    import csoundengine.busproxy
    import csoundengine.tableproxy
    from maelzel.snd import audiosample
    from maelzel.core.presetdef import PresetDef
    import numpy as np


class _SyncSessionHandler(SessionHandler):
    def __init__(self, renderer: SynchronizedContext):
        self.renderer = renderer

    def schedEvent(self, event: csoundengine.event.Event):
        return self.renderer._schedSessionEvent(event)


class FutureSynth(csoundengine.baseschedevent.BaseSchedEvent, csoundengine.synth.ISynth):
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
        self._instr: csoundengine.instr.Instr | None = None

    @property
    def instrName(self) -> str:
        """Name of the instrument template (not to be confused with p1)"""
        return self.event.instr if isinstance(self.event, SynthEvent) else self.event.instrname

    def _synthevent(self) -> SynthEvent:
        if isinstance(self.event, SynthEvent):
            return self.event
        raise ValueError(f"This FutureSynth has an event of type {type(self.event)}, {self.kind=}")

    def _csoundevent(self) -> csoundengine.event.Event:
        if isinstance(self.event, csoundengine.event.Event):
            return self.event
        raise ValueError(f"This FutureSynth has an event of type {type(self.event)}, {self.kind=}")

    def aliases(self) -> dict[str, str]:
        return self.instr.aliases

    def controlNames(self, aliases=True, aliased=False) -> frozenset[str]:
        return self.instr.controlNames(aliases=aliases, aliased=aliased)

    def pfieldNames(self, aliases=True, aliased=False) -> frozenset[str]:
        return self.instr.pfieldNames(aliases=aliases, aliased=aliased)

    def paramValue(self, param: str) -> float | str | None:
        return self.instr.paramValue(param)

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
            return presetmanager.presetManager.getPreset(self.event.instr)
        else:
            raise ValueError(f"This _FutureSynth wraps a session event and "
                             f"has no preset, event={self.event}")

    def dynamicParamNames(self, aliases=True, aliased=False) -> frozenset[str]:
        return self.instr.dynamicParamNames(aliases=aliases, aliased=aliased)

    @property
    def instr(self) -> csoundengine.instr.Instr:
        """Get the csoundengine's Instr associated with the event's preset"""
        if self._instr is not None:
            return self._instr
        if isinstance(self.event, SynthEvent):
            instr = presetmanager.presetManager.getInstr(self.event.instr)
        else:
            instr = self.session.getInstr(self.event.instrname)
            assert instr is not None, f"Could not find this event's instr. {self=}"
        self._instr = instr
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

    def _setPfield(self, param: str, value: float, delay=0.) -> None:
        self.set(param=param, value=value, delay=delay)

    def _setTable(self, param: str, value: float, delay=0.) -> None:
        self.set(param=param, value=value, delay=delay)

    def automate(self,
                 param: int | str,
                 pairs: _t.Sequence[float] | np.ndarray,
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
            self.synth().ui(**specs)
        else:
            raise RuntimeError("This synth has not been scheduled yet")


class FutureSynthGroup(csoundengine.baseschedevent.BaseSchedEvent):

    def __init__(self, synths: list[FutureSynth]):
        self.synths: list[FutureSynth] = synths
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

    def dynamicParamNames(self, aliases=True, aliased=False) -> frozenset[str]:
        params = set()
        for synth in self.synths:
            params.update(synth.dynamicParamNames(aliases=aliases, aliased=aliased))
        return frozenset(params)

    def automate(self,
                 param: str | int,
                 pairs: _t.Sequence[float] | np.ndarray,
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


class SynchronizedContext(_renderer.Renderer):
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
                 session: csoundengine.session.Session,
                 whenfinished: _t.Callable | None = None,
                 display=False):

        super().__init__(presetManager=Workspace.active.presetManager)

        self.session: csoundengine.session.Session = session
        """The corresponding Session, can be used to access the session during the context"""

        self.engine: csoundengine.engine.Engine = self.session.engine
        """The play engine, can be used during the context"""

        self.synthgroup: csoundengine.synth.SynthGroup | None = None
        """A SynthGroup holding all scheduled synths during the context"""

        self.workspace: Workspace = Workspace.active
        """The workspace active as the context manager is created"""

        self._instrDefs: dict[str, csoundengine.instr.Instr] = {}
        """An index of registered Instrs"""

        self._futureSynths: list[FutureSynth] = []
        """A list of all synths scheduled"""

        self._tokenToFuture: dict[int, FutureSynth] = {}
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
        self._enterActions: list[_t.Callable[[SynchronizedContext], None]] = []
        self._exitActions: list[_t.Callable[[SynchronizedContext], None]] = []
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
        assert tabnum == 0, f"Setting the table number is not supported at the moment"
        if self.session._handler:
            raise NotImplementedError
        return self.session.makeTable(data=data, size=size, sr=sr)

    def readSoundfile(self, path: str, chan=0, skiptime=0.) -> csoundengine.tableproxy.TableProxy:
        return self.session.readSoundfile(path=path, chan=chan, skiptime=skiptime)

    def includeFile(self, path: str) -> None:
        self.session.includeFile(path)

    def _sched(self, event: csoundengine.event.Event | SynthEvent
               ) -> FutureSynth:
        token = self._getSynthToken()
        future = FutureSynth(parent=self, event=event, token=token)
        self._futureSynths.append(future)
        self._tokenToFuture[token] = future
        return future

    def _schedSessionEvent(self, event: csoundengine.event.Event
                           ) -> FutureSynth:
        return self._sched(event)

    def assignBus(self, kind='', value: float | None = None, persist=False
                  ) -> csoundengine.busproxy.Bus:
        return self.session.assignBus(kind=kind, value=value, persist=persist)

    def prepareSessionEvent(self, event: csoundengine.event.Event) -> bool:
        _, needssync = self.session.prepareInstr(instr=event.instrname,
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
        _, needssync = self.session.prepareInstr(instr=instrname, priority=priority)
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

    def _removeFuture(self, future: FutureSynth):
        token = future.token
        del self._tokenToFuture[token]
        self._futureSynths.remove(future)

    def _getSynthToken(self) -> int:
        token = self._synthCount
        self._synthCount += 1
        return token

    def schedEvent(self, event: SynthEvent | csoundengine.event.Event
                   ) -> FutureSynth:
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
                    whenfinished: _t.Callable | None = None
                    ) -> FutureSynthGroup:
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
        return FutureSynthGroup(futures)

    def registerPreset(self, presetdef: PresetDef) -> bool:
        return self.session.registerInstr(presetdef.getInstr())

    def _presetFromToken(self, token: int) -> PresetDef | None:
        future = self._tokenToFuture.get(token)
        if not future:
            return None

        return self.presetManager.getPreset(future.instrName)

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

        renderer = _rtrenderer.RealtimeRenderer(engine=self.engine,
                                                presetManager=self.workspace.presetManager)
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
                   ) -> FutureSynth:
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
            assert isinstance(sr, int)
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
              whenfinished: _t.Callable | None = None,
              **kws: float | str) -> FutureSynth:
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


# class FutureSynthGroup(csoundengine.baseschedevent.BaseSchedEvent):
#
#     def __init__(self, synths: list[FutureSynth]):
#         self.synths: list[FutureSynth] = synths
#         self.parent: SynchronizedContext = synths[0].parent
#         start = min(synth.start for synth in synths)
#         end = max(synth.end for synth in synths)
#         dur = end - start
#         super().__init__(start=start, dur=dur)
#         self.session = synths[0].session
#         self.engine = self.session.engine
#
#     def paramValue(self, param: str) -> float | str | None:
#         if param not in self.paramNames():
#             raise KeyError(f"Unknown parameter '{param}'. Possible parameters: {self.paramNames()}")
#         for synth in self.synths:
#             value = synth.paramValue(param)
#             if value is not None:
#                 return value
#         return None
#
#     def dynamicParamNames(self, aliases=True, aliased=False) -> frozenset[str]:
#         params = set()
#         for synth in self.synths:
#             params.update(synth.dynamicParamNames(aliases=aliases, aliased=aliased))
#         return frozenset(params)
#
#     def automate(self,
#                  param: str | int,
#                  pairs: _t.Sequence[float] | np.ndarray,
#                  mode='linear',
#                  delay=0.,
#                  overtake=False,
#                  ) -> float:
#         count = 0
#         for synth in self.synths:
#             if param in synth.dynamicParamNames(aliased=True):
#                 synth.automate(param=param, pairs=pairs, mode=mode, delay=delay, overtake=overtake)
#                 count += 1
#         if not count:
#             possibleparams = self.dynamicParamNames(aliased=True)
#             raise ValueError(f"Parameter '{param}' not known by any synth in this "
#                              f"group. Possible parameters: {possibleparams}.\n "
#                              f"Synths: {self.synths}")
#         return 0.
#
#     def _setPfield(self, param: str, value: float, delay=0.) -> None:
#         for synth in self.synths:
#             if param in synth.pfieldNames(aliased=True):
#                 synth._setPfield(param=param, value=value, delay=delay)
#
#     def _setTable(self, param: str, value: float, delay=0.) -> None:
#         for synth in self.synths:
#             if param in synth.controlNames(aliased=True):
#                 synth._setTable(param=param, value=value, delay=delay)
#
#     def set(self, param='', value: float = 0., delay=0., **kws) -> None:
#         count = 0
#         for synth in self.synths:
#             try:
#                 synth.set(param=param, value=value, delay=delay, **kws)
#                 count += 1
#             except KeyError:
#                 pass
#         if not count:
#             raise KeyError(f"Parameter '{param}' unknown. "
#                            f"Possible parameters: {self.dynamicParamNames(aliased=True)}")
#
#     def synthgroup(self) -> csoundengine.synth.SynthGroup:
#         if self.parent.synthgroup is None:
#             raise RuntimeError("The synths in this group have not been scheduled yet")
#         synths = [futuresynth.synth() for futuresynth in self.synths]
#         assert all(synth is not None for synth in synths)
#         return csoundengine.synth.SynthGroup(synths)
#
#     def __getitem__(self, idx: int):
#         return self.synths[idx]
#
#     def scheduled(self) -> bool:
#         return all(synth.scheduled() for synth in self.synths)
#
#     def stop(self, delay=0.) -> None:
#         """ Stop this synthgroup """
#         for synth in self.synths:
#             synth.stop(delay=delay)
#
#     def __repr__(self):
#         scheduled = self.scheduled()
#         return f"FutureSynthGroup(scheduled={scheduled}, synths={self.synths})"
#
#     def _repr_html_(self):
#         if self.scheduled():
#             synthgroup = self.synthgroup()
#             return synthgroup._repr_html_()
#         else:
#             return repr(self)

