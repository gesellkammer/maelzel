"""
Interface for audio rendering (base class for realtime and offline)
"""
from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
import csoundengine

import csoundengine.event

from maelzel.core import presetdef
from maelzel.core import synthevent
from maelzel.core.presetmanager import PresetManager
from . import environment

from typing import Callable, Sequence


class Renderer(ABC):

    def __init__(self, presetManager: PresetManager):
        self.registeredPresets: dict[str, presetdef.PresetDef] = {}
        self.presetManager = presetManager

    def show(self) -> None:
        """
        If inside jupyter, force a display of this OfflineRenderer

        """
        if environment.insideJupyter:
            from IPython.display import display
            display(self)

    @abstractmethod
    def isRealtime(self) -> bool:
        """Is this renderer working in real-time?"""
        raise NotImplementedError

    @abstractmethod
    def assignBus(self, kind='', value: float | None = None, persist=False
                  ) -> csoundengine.busproxy.Bus:
        """Assign a bus"""
        raise NotImplementedError

    @abstractmethod
    def releaseBus(self, bus: int | csoundengine.busproxy.Bus) -> None:
        """
        Release a previously assigned bus

        The bus must have been created with ``persist=True``
        """
        raise NotImplementedError

    def prepareEvents(self,
                      events: list[synthevent.SynthEvent],
                      sessionevents: list[csoundengine.event.Event] = None
                      ) -> bool:
        """
        Prepare a series of events for scheduling

        All init codes which need to be compiled are compiled and synched,
        then all presets are prepared for the given priority.
        This method minimizes the number of syncs needed.

        Args:
            events: the core events to prepare
            sessionevents: any session events

        Returns:
            True if the renderer needs to be synched. The called is resposible
            for calling the :meth:`Renderer.sync` method
        """
        needssync = False
        presetManager = self.presetManager
        for event in events:
            presetname = event.instr
            if presetname not in self.registeredPresets:
                presetdef = presetManager.getPreset(presetname)
                if presetdef is None:
                    raise ValueError(f"Preset {presetname} does not exist (event={event})")
                needssync |= self.registerPreset(presetdef)
        if needssync:
            self.sync()
        for event in events:
            presetdef = self.registeredPresets[event.instr]
            instr = presetdef.getInstr()
            needssync |= self.prepareInstr(instr, event.priority)
            if event.initfunc:
                event.initialize(self)
        if sessionevents:
            for ev in sessionevents:
                instr = self.getInstr(ev.instrname)
                if instr is None:
                    raise ValueError(f"Instrument {ev.instrname} is not defined")
                needssync |= self.prepareInstr(instr, priority=ev.priority)
        return needssync

    def prepareEvent(self, event: synthevent.SynthEvent) -> bool:
        """
        Prepare an event to be scheduled

        Args:
            event: the event to schedule

        Returns:
            True if the operation performed some action on the audio engine

        """
        presetname = event.instr
        needssync = False
        if presetname not in self.registeredPresets:
            presetdef = self.presetManager.getPreset(presetname)
            if presetdef is None:
                raise ValueError(f"Preset {presetname} does not exist (event={event})")
            needssync |= self.preparePreset(presetdef, priority=event.priority)
        if event.initfunc:
            event.initialize(self)
        return needssync

    @abstractmethod
    def prepareSessionEvent(self, event: csoundengine.event.Event) -> bool:
        """
        Prepare a csound event for scheduling

        Args:
            event: the event to preare

        Returns:
            True if the backend needs sync
        """
        raise NotImplementedError

    def preparePreset(self, presetdef: presetdef.PresetDef, priority: int
                      ) -> bool:
        """
        Prepare a preset to be used

        Args:
            presetdef: the preset definition
            priority: the priority to use

        Returns:
            True if this operation performed some action on the audio engine. This can
            be used to sync the audio engine if needed.

        """
        presetname = presetdef.name
        needssync = False
        if presetname not in self.registeredPresets:
            presetdef = self.presetManager.getPreset(presetname)
            assert presetdef is not None, f"Preset {presetname} does not exist"
            needssync |= self.registerPreset(presetdef)
        instr = presetdef.getInstr()
        needssync |= self.prepareInstr(instr, priority)
        return needssync

    @abstractmethod
    def prepareInstr(self, instr: csoundengine.instr.Instr, priority: int
                     ) -> bool:
        """
        Prepare an Instr instance for the given priority

        Args:
            instr: the
            priority:

        Returns:
            True if the audio engine performaed an action, False if the instrument
            was already prepared at the given priority
        """
        raise NotImplementedError

    def sync(self) -> None:
        """
        Block until the audio engine has processed its immediate events
        """
        return

    @abstractmethod
    def getInstr(self, instrname: str) -> csoundengine.instr.Instr:
        """Get the Instr corresponding to the instr name"""
        raise NotImplementedError

    @abstractmethod
    def registerPreset(self, presetdef: presetdef.PresetDef) -> bool:
        """Register a Preset at this renderer"""
        raise NotImplementedError

    @abstractmethod
    def _schedSessionEvent(self, event: csoundengine.event.Event):
        """Schedule a session event"""
        raise NotImplementedError

    @abstractmethod
    def getSynth(self, token: int) -> csoundengine.synth.Synth | None:
        raise NotImplementedError

    @abstractmethod
    def schedEvent(self, event: synthevent.SynthEvent | csoundengine.event.Event):
        """Schedule a synthevent"""
        raise NotImplementedError

    @abstractmethod
    def schedEvents(self,
                    coreevents: list[synthevent.SynthEvent],
                    sessionevents: list[csoundengine.event.Event] = None,
                    whenfinished: Callable = None):
        """
        Schedule multiple events simultanously

        Args:
            coreevents: the synth events to schedule
            sessionevents: the csound events to schedule
            whenfinished: a function to call when scheduling is finished

        Returns:
            the returned value depends on the kind of renderer
        """
        raise NotImplementedError

    @abstractmethod
    def includeFile(self, path: str) -> None:
        """Add an #include clause to this renderer"""
        raise NotImplementedError

    @abstractmethod
    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int = 0,
                  sr: int = 0,
                  tabnum: int = 0
                  ) -> int:
        """Create a Table to be used within this renderer"""
        raise NotImplementedError

    @abstractmethod
    def sched(self,
              instrname: str,
              delay: float,
              dur: float,
              priority: int,
              args: dict[str, float] | list[float|str] = None,
              whenfinished: Callable = None,
              **kws: dict[str, float],
              ):
        """
        Schedule an instrument

        This method schedules a csound event

        Args:
            instrname: the name of the instrument
            delay: start time
            dur: duration of the event. -1 to not set a duration
            args: a list of positional arguments, or a dict of named arguments
            priority: the priority of this event
            whenfinished: a callback to be fired when the event finishes (only
                valid for online rendering)
            kws: when args is passed as a list of positional arguments, any
                named argument can be given here

        Returns:

        """
        raise NotImplementedError

    def schedDummyEvent(self, dur=0.001):
        """
        Schedule a dummy synth
        """
        pass

    def pushLock(self):
        """Lock the audio engine"""
        pass

    def popLock(self):
        """Pop the lock of the audio engine (must be preceded by a pushLock)"""
        pass
