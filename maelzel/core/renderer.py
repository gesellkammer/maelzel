"""
Interface for online and offline rendering
"""
from __future__ import annotations
from maelzel.core import presetdef
from maelzel.core import synthevent
import csoundengine
from csoundengine.session import SessionEvent
from typing import Callable
import numpy as np
from maelzel.core.presetmanager import PresetManager



class Renderer:

    def __init__(self, presetManager: PresetManager):
        self.registeredPresets: dict[str, presetdef.PresetDef] = {}
        self.presetManager = presetManager

    def isRealtime(self) -> bool:
        """Is this renderer working in real-time?"""
        raise NotImplementedError

    def assignBus(self, kind='audio') -> int:
        """Assign a bus"""
        raise NotImplementedError

    def releaseBus(self, busnum: int):
        """Release a previously assigned bus"""
        raise NotImplementedError

    def prepareEvents(self,
                      events: list[synthevent.SynthEvent],
                      sessionevents: list[csoundengine.session.SessionEvent] = None
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
            if not presetname in self.registeredPresets:
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
        if not presetname in self.registeredPresets:
            presetdef = self.presetManager.getPreset(presetname)
            if presetdef is None:
                raise ValueError(f"Preset {presetname} does not exist (event={event})")
            needssync |= self.preparePreset(presetdef, priority=event.priority)
        if event.initfunc:
            event.initialize(self)
        return needssync

    def prepareSessionEvent(self, sessionevent: SessionEvent) -> bool:
        """
        Prepare a SessionEvent for scheduling

        Args:
            sessionevent: the event to preare

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
        if not presetname in self.registeredPresets:
            presetdef = self.presetManager.getPreset(presetname)
            assert presetdef is not None, f"Preset {presetname} does not exist"
            needssync |= self.registerPreset(presetdef)
        instr = presetdef.getInstr()
        needssync |= self.prepareInstr(instr, priority)
        return needssync

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
        pass

    def sync(self) -> None:
        """
        Block until the audio engine has processed its immediate events
        """
        return

    def getInstr(self, instrname: str):
        """Get the Instr corresponding to the instr name"""
        raise NotImplementedError

    def registerPreset(self, presetdef: presetdef.PresetDef) -> bool:
        """Register a Preset at this renderer"""
        raise NotImplementedError

    def schedSessionEvent(self, event: SessionEvent):
        """Schedule a session event"""
        raise NotImplementedError

    def schedEvent(self, event: synthevent.SynthEvent):
        """Schedule a synthevent"""
        raise NotImplementedError

    def schedEvents(self,
                    coreevents: list[synthevent.SynthEvent],
                    sessionevents: list[SessionEvent] = None,
                    whenfinished: Callable = None):
        """
        Schedule multiple events simultanously

        Args:
            coreevents: the synth events to schedule
            sessionevents: the session events to schedule
            whenfinished: a function to call when scheduling is finished

        Returns:
            the returned value depends on the kind of renderer
        """
        raise NotImplementedError

    def includeFile(self, path: str) -> None:
        """Add an #include clause to this renderer"""
        raise NotImplementedError

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int = 0,
                  sr: int = 0,
                  tabnum: int = 0
                  ) -> int:
        """Create a Table to be used within this renderer"""
        raise NotImplementedError

    def sched(self,
             instrname: str,
             delay: float,
             dur: float,
             args: list[float|str],
             priority: int,
             whenfinished: Callable = None):
        """Schedule an event"""
        raise NotImplementedError

    def pushLock(self):
        """Lock the audio engine"""
        pass

    def popLock(self):
        """Pop the lock of the audio engine (must be preceded by a pushLock)"""
        pass
