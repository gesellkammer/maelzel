from __future__ import annotations

import numpy as np
import maelzel.core.renderer as _renderer
from maelzel.core import synthevent
from maelzel.core.workspace import Workspace
from maelzel.core.presetdef import PresetDef

import csoundengine

import typing

if typing.TYPE_CHECKING:
    import csoundengine.busproxy
    import csoundengine.instr
    import csoundengine.tableproxy
    import csoundengine.event
    import csoundengine.synth
    import csoundengine.session
    from maelzel.core import presetmanager


class RealtimeRenderer(_renderer.Renderer):
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

    def __init__(self, engine: csoundengine.Engine | None = None,
                 presetManager: presetmanager.PresetManager | None = None):
        if presetManager is None:
            from . import presetmanager
            presetManager = presetmanager.presetManager
        super().__init__(presetManager=presetManager)
        if engine is None:
            from maelzel.core import playback
            engine = playback._playEngine(config=Workspace.active.config)
        self.engine: csoundengine.Engine = engine
        self.session: csoundengine.session.Session = engine.session()

    def isRealtime(self) -> bool:
        return True

    def assignBus(self, kind='', value: float | None = None, persist=False
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

    def _schedDummyEvent(self, dur=0.001) -> csoundengine.synth.Synth:
        """
        Schedule a dummy synth

        Args:
            dur: the duration of the synth

        Returns:
            a Synth
        """
        return self.session.sched('.dummy', 0, dur)

    def getSynth(self, token: int) -> csoundengine.synth.Synth | None:
        return self.session.getSynthById(token)

    def schedEvent(self, event: synthevent.SynthEvent | csoundengine.event.Event
                   ) -> csoundengine.synth.Synth:
        if isinstance(event, synthevent.SynthEvent):
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
                              **dynargs)  # type: ignore
        else:
            return self._schedSessionEvent(event)

    def schedEvents(self,
                    coreevents: list[synthevent.SynthEvent],
                    sessionevents: list[csoundengine.event.Event] | None = None,
                    whenfinished: typing.Callable | None = None
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
        synths, sessionsynths = self._schedEvents(presetManager=self.presetManager,
                                                  coreevents=coreevents,
                                                  sessionevents=sessionevents,
                                                  whenfinished=whenfinished)
        numevents = len(coreevents) + (len(sessionevents) if sessionevents else 0)
        assert len(synths) + len(sessionsynths) == numevents, f"{len(synths)=}, {numevents=}"
        synths.extend(sessionsynths)
        return csoundengine.synth.SynthGroup(synths)

    def _schedEvents(self,
                     coreevents: list[synthevent.SynthEvent],
                     presetManager: presetmanager.PresetManager,
                     sessionevents: list[csoundengine.event.Event] | None = None,
                     posthook: typing.Callable[[list[csoundengine.synth.Synth]], None] | None = None,
                     whenfinished: typing.Callable | None = None,
                     locked=True
                     ) -> tuple[list[csoundengine.synth.Synth], list[csoundengine.synth.Synth]]:
        """
        Schedule events in synch

        Args:
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
        needssync = self.prepareEvents(events=coreevents, sessionevents=sessionevents)
        resolvedParams = [ev._resolveParams(instr=presetManager.getInstr(ev.instr))
                          for ev in coreevents]

        if whenfinished and self.isRealtime():
            lastevent = max(coreevents, key=lambda ev: ev.end if ev.end > 0 else float('inf'))
            lastevent.whenfinished = (lambda id: whenfinished() if not lastevent.whenfinished else 
                lambda id, ev=lastevent: ev.whenfinished(id) or whenfinished())

        if needssync:
            self.sync()

        if len(coreevents) + (0 if not sessionevents else len(sessionevents)) < 2:
            locked = False

        if locked:
            self.pushLock()  # <---------------- Lock

        synths: list[csoundengine.synth.Synth] = []
        for coreevent, (pfields5, dynargs) in zip(coreevents, resolvedParams):
            if coreevent.gain == 0:
                synths.append(self._schedDummyEvent())
                continue

            synth = self.sched(PresetDef.presetNameToInstrName(coreevent.instr),
                               delay=coreevent.delay,
                               dur=coreevent.dur,
                               args=pfields5,
                               priority=coreevent.priority,
                               whenfinished=coreevent.whenfinished,
                               **dynargs)  # type: ignore
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
                for automation in coreevent.automations:
                    synth.automate(param=automation.param, pairs=automation.data, delay=automation.delay)

        if sessionevents:
            sessionsynths = []
            for ev in sessionevents:
                synth = self._schedSessionEvent(ev)
                sessionsynths.append(synth)
                if ev.automations:
                    for automation in ev.automations:
                        synth.automate(param=automation.param,
                                       pairs=automation.pairs,
                                       delay=automation.delay,
                                       mode=automation.interpolation,
                                       overtake=automation.overtake)

            # sessionsynths = [renderer._schedSessionEvent(ev) for ev in sessionevents]

            # synths.extend(sessionsynths)
        else:
            sessionsynths = []

        if posthook:
            posthook(synths)

        if locked:
            self.popLock()  # <----------------- Unlock
        return synths, sessionsynths

    def includeFile(self, path: str) -> None:
        self.session.includeFile(path)

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int | tuple[int, int] = 0,
                  sr: int = 0,
                  tabnum: int = 0
                  ) -> csoundengine.tableproxy.TableProxy:
        return self.session.makeTable(data=data, size=size, sr=sr, tabnum=tabnum)

    def readSoundfile(self,
                      soundfile: str,
                      chan=0,
                      skiptime=0.
                      ) -> csoundengine.tableproxy.TableProxy:
        return self.session.readSoundfile(path=soundfile, chan=chan, skiptime=skiptime)

    def sched(self,
              instrname: str,
              delay: float = 0.,
              dur: float = -1,
              priority: int = 1,
              args: list[float | str] | dict[str, float] | None = None,
              whenfinished: typing.Callable | None = None,
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
                                  **kws)  # type: ignore

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
