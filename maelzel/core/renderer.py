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
        raise NotImplementedError

    def assignBus(self, kind='audio') -> int:
        raise NotImplementedError

    def releaseBus(self, busnum: int):
        raise NotImplementedError

    def prepareEvent(self, event: synthevent.SynthEvent) -> None:
        presetname = event.instr
        if not presetname in self.registeredPresets:
            presetdef = self.presetManager.getPreset(presetname)
            assert presetdef is not None, f"Preset {presetname} does not exist"
            self.preparePreset(presetdef, priority=event.priority)
        if event.initfunc:
            event.initialize(self)

    def prepareSessionEvent(self, sessionevent: SessionEvent):
        raise NotImplementedError

    def preparePreset(self, presetdef: presetdef.PresetDef, priority: int) -> None:
        presetname = presetdef.name
        if not presetname in self.registeredPresets:
            presetdef = self.presetManager.getPreset(presetname)
            assert presetdef is not None, f"Preset {presetname} does not exist"
            self.registerPreset(presetdef)
        instr = presetdef.getInstr()
        self.prepareInstr(instr, priority)

    def prepareInstr(self, instr: csoundengine.instr.Instr, priority: int):
        pass

    def getInstr(self, instrname: str):
        raise NotImplementedError

    def registerPreset(self, presetdef: presetdef.PresetDef) -> None:
        raise NotImplementedError

    def schedSessionEvent(self, event: SessionEvent):
        raise NotImplementedError

    def schedEvent(self, event: synthevent.SynthEvent):
        raise NotImplementedError

    def schedEvents(self,
                    coreevents: list[synthevent.SynthEvent],
                    sessionevents: list[SessionEvent] = None,
                    whenfinished: Callable = None):
        raise NotImplementedError

    def includeFile(self, path: str) -> None:
        raise NotImplementedError

    def makeTable(self,
                  data: np.ndarray | list[float] | None = None,
                  size: int = 0,
                  sr: int = 0,
                  tabnum: int = 0
                  ) -> int:
        raise NotImplementedError

    def sched(self,
             instrname: str,
             delay: float,
             dur: float,
             args: list[float|str],
             priority: int,
             whenfinished: Callable = None):
        raise NotImplementedError

    def pushLock(self):
        pass

    def popLock(self):
        pass
