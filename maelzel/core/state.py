from ._base import *
import configdict
from .config import config
from emlib.pitchtools import m2f, set_reference_freq
from typing import Any
from maelzel.scorestruct import ScoreStructure


class _State:
    def __init__(self,
                 a4:float=None,
                 tempo:Opt[float]=None,
                 scorestruct:Opt[ScoreStructure]=None,
                 renderer:Any=None,
                 config:configdict.CheckedDict=None):
        if tempo is not None and scorestruct is not None:
            oldtempo = tempo
            tempo = tempo, scorestruct.getMeasureDef(0).quarterTempo
            logger.error(f"both tempo (q={oldtempo}) and scorestruct are not None, "
                         f"scorestruct's tempo will have priority (q={tempo})")
        self.config = config or currentConfig()
        self._a4 = a4 or self.config['A4']
        self.scorestruct = scorestruct
        self.renderer = renderer

    @property
    def a4(self) -> float:
        return self._a4

    @a4.setter
    def a4(self, value: float):
        self._a4 = value
        if self is currentConfig():
            set_reference_freq(value)

    @property
    def tempo(self) -> float:
        return float(self.scorestruct.getMeasureDef(0).quarterTempo)

    @tempo.setter
    def tempo(self, quarterTempo: float):
        pass

    def activate(self) -> None:
        set_reference_freq(self.a4)


_statestack = [_State(a4=config.get('A4', m2f(69)), config=config)]


def pushState(a4:float=None,
              tempo:time_t=None,
              renderer=None,
              config:configdict.CheckedDict=None,
              scorestruct:ScoreStructure=None
              ) -> _State:
    """
    Push a new state to the global state stack. A new state inherits values
    not set from the earlier state

    Args:
        a4: the reference frequency
        tempo: a tempo reference
        renderer: a play.OfflineRenderer
        config: a configuration dict, as created by config.clone()
        scorestruct: the current ScoreStructure.
    """
    assert len(_statestack) >= 1
    assert tempo is None or scorestruct is None
    currState = _statestack[-1]
    if a4 is None:
        a4 = currState.a4
    tempo = F(tempo) if tempo is not None else currState.tempo
    if renderer is None:
        renderer = currState.renderer
    if config is None:
        config = currState.config.copy()
    else:
        assert config is not currentConfig() and \
               not isinstance(config, configdict.ConfigDict)
    state = _State(a4=a4, tempo=tempo, renderer=renderer, config=config)

    _statestack.append(state)
    if a4 != m2f(69):
        set_reference_freq(a4)
    return state


def getState() -> _State:
    """
    Get current state
    """
    return _statestack[-1]


def popState() -> _State:
    """
    Pop a global state from the stack, return the now invalid state.
    The last state can't be popped.
    """
    if len(_statestack) == 1:
        return _statestack[-1]
    laststate = _statestack.pop()
    return laststate


def setTempo(quarterTempo:float) -> None:
    state = getState()
    state.tempo = quarterTempo
    getState().tempo = quarterTempo


def setA4(a4:float) -> None:
    state = getState()
    state.a4 = a4
    state.activate()


def currentConfig() -> configdict.CheckedDict:
    return _statestack[-1].config
