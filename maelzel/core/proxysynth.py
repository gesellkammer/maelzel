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
