from __future__ import annotations
from maelzel.common import F, F0

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.common import num_t
    from maelzel.core import presetdef
    from maelzel.scorestruct import ScoreStruct
    import csoundengine.synth
    import csoundengine.instr


class ProxySynthBase:

    def __init__(self,
                 offset: F,
                 scorestruct: ScoreStruct):
        self.offset = offset
        self.scorestruct = scorestruct
        self.offsetSecs = scorestruct.beatToTime(offset)

    def scheduled(self) -> bool:
        return True

    def source(self) -> csoundengine.synth.ISynth:
        raise NotImplementedError

    def __call__(self) -> csoundengine.synth.ISynth:
        return self.source()

    def automate(self,
                 param: int|str,
                 pairs: list[tuple[float|int|F, float]],
                 time='beats',
                 relative=True
                 ) -> None:
        assert relative, 'Absolute mode not supported'
        if time == 'beats':
            s = self.scorestruct
            pairs = [(float(s.beatToTime(t + self.offset) - self.offsetSecs), v) for t, v in pairs]
        flatpairs = []
        for pair in pairs:
            flatpairs.extend(pair)
        self._automate(param=param, data=flatpairs)

    def set(self, param: str, value: float, delay: num_t = F0, time='beats'):
        return self.automate(param=param, pairs=[(delay, value)], relative=True, time=time)

    def _automate(self, param: int | str, data: list[float]):
        raise NotImplementedError

    def namedParams(self) -> set[str]:
        raise NotImplementedError

    def stop(self, delay: num_t = F0, time='beats'):
        raise NotImplementedError

    def playing(self) -> bool:
        raise NotImplementedError

    def finished(self) -> bool:
        raise NotImplementedError


class ProxySynth(ProxySynthBase):
    """
    This class wraps a backend synth
    """

    def __init__(self,
                 offset: F,
                 scorestruct: ScoreStruct,
                 synth: csoundengine.synth.Synth,
                 preset: presetdef.PresetDef
                 ):
        super().__init__(offset=offset, scorestruct=scorestruct)
        self.preset = preset
        self._synth = synth

    def __call__(self):
        return self.synth()

    def synth(self) -> csoundengine.synth.Synth:
        return self._synth

    def scheduled(self) -> bool:
        return True

    def _automate(self, param: str, data: list[float], delay=0.):
        # times must be relative seconds
        synth = self.synth()
        assert synth is not None
        synth.automate(param=param, pairs=data, delay=delay)

    def __repr__(self):
        cls = str(type(self))
        return f"{cls}(synth={self.synth()})"

    def _repr_html_(self) -> str:
        if self.scheduled() and (synth := self.synth()) is not None:
            synth = self.synth()
            return synth._repr_html_()
        else:
            return repr(self)

    def instr(self) -> csoundengine.instr.Instr:
        return self.preset.getInstr()

    def namedParams(self) -> set[str]:
        args = self.preset.args
        return set(args.keys()) if args is not None else set()

    def stop(self, delay: num_t = F0, time='beats'):
        if time == 'beats':
            delay = self.scorestruct.timeDelta(self.offset, self.offset+delay)
        if synth := self.synth():
            synth.stop(delay=float(delay))

    def playing(self) -> bool:
        return self.scheduled() and self.synth().playing()

    def finished(self) -> bool:
        return self.scheduled() and self.synth().finished()


class ProxySynthGroup(ProxySynthBase):

    def __init__(self,
                 group: csoundengine.synth.SynthGroup,
                 offset: F,
                 scorestruct: ScoreStruct):
        super().__init__(offset=offset, scorestruct=scorestruct)
        self._group = group

    def namedParams(self) -> set[str]:
        return self._group.dynamicParamNames()

    def _automate(self, param: int | str, data: list[float]):
        self._group.automate(param=param, pairs=data)

    def playing(self) -> bool:
        return self._group.playing()

    def finished(self) -> bool:
        return self._group.finished()

    def stop(self, delay: num_t = F0, time='beats'):
        if time == 'beats':
            delay = self.scorestruct.timeDelta(start=self.offset, end=self.offset + delay)
        self._group.stop(delay=float(delay))

    def _repr_html_(self):
        return self._group._repr_html_()



# class DeferredSynth(ProxySynthBase):

#     def __init__(self,
#                  offset: F,
#                  scorestruct: ScoreStruct,
#                  token: int,
#                  event: synthevent.SynthEvent,
#                  renderer: playback.SynchronizedContext,
#                  ):
#         super().__init__(offset=offset, scorestruct=scorestruct)
#         self.token = token
#         self.event = event
#         self.renderer = renderer
#         self._synth: csoundengine.synth.Synth | None = None

#     def _fetchSynth(self) -> csoundengine.synth.Synth | None:
#         self._synth = synth = self.renderer.getSynth(self.token)
#         return synth

#     def synth(self) -> csoundengine.synth.Synth:
#         synth = self._synth or self._fetchSynth()
#         if synth is None:
#             raise RuntimeError(f"Synth for this {type(self)} ({self}) has not been scheduled")
#         return synth

#     def scheduled(self) -> bool:
#         # return self.renderer.synthgroup is not None
#         return self._synth is not None or self._fetchSynth() is not None

#     def __repr__(self):
#         scheduled = self.scheduled()
#         cls = str(type(self))
#         if not scheduled:
#             return f"{cls}(scheduled=False, event={self.event}, token={self.token})"
#         else:
#             return f"{cls}(scheduled=True, event={self.event}, synth={self.synth()})"

#     def _automate(self, param: int | str, data: list[float], delay=0.):
#         # times must be relative seconds
#         self.renderer._automate(token=self.token, param=param, pairs=data)

#     @property
#     def preset(self) -> presetdef.PresetDef:
#         return self.renderer.presetManager.getPreset(self.event.instr)

#     def namedParams(self) -> set[str]:
#         return set(self.preset.getInstr().paramNames())

#     def playing(self) -> bool:
#         return self.scheduled() and self.synth().playing()

#     def finished(self) -> bool:
#         return self.scheduled() and self.synth().finished()

#     def stop(self, delay: num_t = F0, time='beats'):
#         if time == 'beats':
#             delay = self.scorestruct.timeDelta(start=self.offset, end=self.offset + delay)

#         if not self.scheduled():
#             self.renderer.unsched(self.token)
#         else:
#             self.synth().stop(delay=delay)

# class DeferredGroup(ProxySynthBase):
#     def __init__(self, synths: list[DeferredSynth]):
#         self.synths: list[DeferredSynth] = synths
#         self.renderer: playback.SynchronizedContext = synths[0].renderer
#         self._group: csoundengine.synth.SynthGroup | None = None
#         super().__init__(offset=synths[0].offset,
#                          scorestruct=synths[0].scorestruct)

#     def synthgroup(self) -> csoundengine.synth.SynthGroup:
#         if not self.scheduled():
#             raise RuntimeError("The synths in this group have not been scheduled yet")
#         if self._group is None:
#             synths = [synth.synth() for synth in self.synths]
#             self._group = csoundengine.synth.SynthGroup(synths)
#         return self._group

#     def scheduled(self) -> bool:
#         return self.renderer.synthgroup is not None

#     def _automate(self, param: int | str, data: list[float], delay=0.):
#         for synth in self.synths:
#             synth._automate(param, data=data)

#     def namedParams(self) -> set[str]:
#         allparams = set()
#         for synth in self.synths:
#             allparams.update(synth.namedParams())
#         return allparams

#     def __repr__(self):
#         scheduled = self.scheduled()
#         cls = str(type(self))
#         return f"{cls}(scheduled={scheduled}, synths={self.synths})"

#     def _repr_html_(self) -> str:
#         if self.scheduled():
#             return self.synthgroup()._repr_html_()
#         return repr(self)

#     def stop(self, delay: num_t = F0, time='beats'):
#         for synth in self.synths:
#             synth.stop(delay=delay, time=time)

#     def playing(self) -> bool:
#         return any(synth.playing() for synth in self.synths)

#     def finished(self) -> bool:
#         return all(synth.finished() for synth in self.synths)
