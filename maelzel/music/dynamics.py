"""
functions to convert between dB and musical dynamics
also makes a representation of the amplitude in terms of musical dynamics
"""
from __future__ import annotations
from bisect import bisect as _bisect
import bpf4
import emlib.img
import tempfile
from pitchtools import db2amp, amp2db
from emlib import misc
from typing import List, Sequence as Seq, Union as U, Dict, Tuple, Callable, NamedTuple


_DYNAMICS = ('pppp', 'ppp', 'pp', 'p', 'mp',
             'mf', 'f', 'ff', 'fff', 'ffff')


class DynamicDescr(NamedTuple):
    shape: str
    mindb: float
    maxdb: float = 0.
    dynamics: str = "ppp pp p mp mf f ff fff"

    def makeCurve(self) -> DynamicCurve:
        return DynamicCurve.fromdescr(self.shape, mindb=self.mindb, maxdb=self.maxdb,
                                      dynamics=self.dynamics)


class DynamicCurve(object):
    
    def __init__(self, curve: Callable[[float], float], dynamics:Seq[str] = None):
        """
        Args:
            curve: a bpf mapping 0-1 to amplitude(0-1)
            dynamics: a list of possible dynamics, or None to use the default

        NB: see .fromdescr
        """
        self.dynamics = misc.astype(tuple, dynamics if dynamics else _DYNAMICS)
        bpf = bpf4.asbpf(curve, bounds=(0, 1)).fit_between(0, len(self.dynamics)-1)
        self._amps2dyns, self._dyns2amps = _create_dynamics_mapping(bpf, self.dynamics)
        assert len(self._amps2dyns) == len(self.dynamics)

    @classmethod
    def getDefault(cls) -> DynamicCurve:
        return _default

    @classmethod
    def fromdescr(cls, shape:str='expon(4.0)', mindb=-80.0, maxdb=0.0,
                  dynamics:U[str, Seq[str]] = None) -> DynamicCurve:
        """
        Args:
            shape: the shape of the mapping ('linear', 'expon(2)', etc)
            mindb: min. db value
            maxdb: max. db value
            dynamics: the list of possible dynamics, ordered from soft to loud

        Returns:
            a DynamicCurve

        Example
        ~~~~~~~

        >>> DynamicCurve.fromdescr('expon(3)', mindb=-80, dynamics='ppp pp p mf f ff'.split())

        """
        if isinstance(dynamics, str):
            dynamics = str.split()
        bpf = create_shape(shape, mindb, maxdb)
        return cls(bpf, dynamics)

    def amp2dyn(self, amp:float, nearest=True) -> str:
        """
        Convert amplitude to dynamic

        Args:
            amp: the amplitude (0-1)
            nearest: if True, it searches for the nearest dynamic. Otherwise it gives
                     the dynamic exactly inferior

        Returns:
            the dynamic
        """
        curve = self._amps2dyns
        if amp < curve[0][0]:
            return curve[0][1]
        if amp > curve[-1][0]:
            return curve[-1][1]
        insert_point = _bisect(curve, (amp, None))
        if not nearest:
            idx = max(0, insert_point-1)
            return curve[idx][1]
        amp0, dyn0 = curve[insert_point - 1]
        amp1, dyn1 = curve[insert_point]
        db = amp2db(amp)
        return dyn0 if abs(db-amp2db(amp0)) < abs(db-amp2db(amp1)) else dyn1

    def dyn2amp(self, dyn:str) -> float:
        """
        convert a dynamic expressed as a string to its corresponding amplitude
        """
        amp = self._dyns2amps.get(dyn.lower())
        if amp is None:
            raise ValueError("dynamic %s not known" % dyn)
        return amp

    def dyn2db(self, dyn:str) -> float:
        return amp2db(self.dyn2amp(dyn))

    def db2dyn(self, db:float) -> str:
        return self.amp2dyn(db2amp(db))

    def dyn2index(self, dyn:str) -> int:
        """
        Convert the given dynamic to an integer index
        """
        try:
            return self.dynamics.index(dyn)
        except ValueError:
            raise ValueError("Dynamic not defined, should be one of %s" % self.dynamics)

    def index2dyn(self, idx:int) -> str:        
        return self.dynamics[idx]

    def amp2index(self, amp:float) -> int:
        return self.dyn2index(self.amp2dyn(amp))

    def index2amp(self, index:int) -> float:
        return self.dyn2amp(self.index2dyn(index))

    def asdbs(self, step=1) -> List[float]:
        """
        Convert the dynamics defined in this curve to dBs
        """
        indices = range(0, len(self.dynamics), step)
        dbs = [self.dyn2db(self.index2dyn(index)) for index in indices]
        assert dbs 
        return dbs

    def plot(self, usedB=True):
        import matplotlib.pyplot as plt
        xs = list(range(len(self.dynamics)))
        fig, ax = plt.subplots()
        if usedB:
            ys = [self.dyn2db(dyn) for dyn in self.dynamics]
            ax.set_ylabel("dB")
        else:
            ys = [self.dyn2amp(dyn) for dyn in self.dynamics]
            ax.set_ylabel("amp")
        ax.plot(xs, ys)
        ax.set_xticks(xs)
        ax.set_xticklabels(self.dynamics)
        plt.show()

def _validate_dynamics(dynamics: Seq[str]) -> None:
    assert not set(dynamics).difference(_DYNAMICS), \
        "Dynamics not understood"


def _create_dynamics_mapping(bpf: bpf4.BpfInterface, dynamics:Seq[str] = None
                             ) -> Tuple[List[Tuple[float, str]], Dict[str, float]]:
    """
    Calculate the global dynamics table according to the bpf given

    Args:
        bpf: a bpf from dynamic-index to amp
        dynamics: a list of dynamics

    Returns:
        a tuple (amps2dyns, dyns2amps), where amps2dyns is a List of (amp, dyn)
        and dyns2amps is a dict mapping dyn -> amp
    """
    if dynamics is None:
        dynamics = _DYNAMICS
    assert isinstance(bpf, bpf4.core.BpfInterface)
    _validate_dynamics(dynamics)
    dynamics_table = [(bpf(i), dyn) for i, dyn in enumerate(dynamics)]
    dynamics_dict = {dyn: ampdb for ampdb, dyn, in dynamics_table}
    return dynamics_table, dynamics_dict


def create_shape(shape='expon(3)', mindb:U[int,float]=-90, maxdb:U[int, float]=0
                 ) -> bpf4.BpfInterface:
    """
    Return a bpf mapping 0-1 to amplitudes, as needed by DynamicCurve

    Args:
        shape: a descriptor of the curve to use to map amplitude to dynamics
        mindb: the min. representable amplitude (in dB)
        maxdb: the max. representable amplitude (in dB)

    If X is dynamic and Y is amplitude, an exponential curve with exp > 1
    will allocate more dynamics to the soft amplitude range, resulting in more
    resolution for small amplitudes.
    A curve with exp < 1 will result in more resolution for high dynamics
    """
    minamp, maxamp = db2amp(mindb), db2amp(maxdb)
    return bpf4.util.makebpf(shape, [0, 1], [minamp, maxamp])
    

_default = DynamicCurve(create_shape("expon(4.0)", -80, 0))


def amp2dyn(amp:float, nearest=True) -> str:
    return _default.amp2dyn(amp, nearest)


def dyn2amp(dyn:str) -> float:
    return _default.dyn2amp(dyn)


def dyn2db(dyn:str) -> float:
    return _default.dyn2db(dyn)


def db2dyn(db:float, nearest=True) -> str:
    amp = db2amp(db)
    return _default.amp2dyn(amp, nearest)
   

def dyn2index(dyn:str) -> int:
    return _default.dyn2index(dyn)


def index2dyn(idx:int) -> str:
    return _default.index2dyn(idx)


def setDefaultCurve(shape:str, mindb=-90, maxdb=0, possible_dynamics=None) -> None:
    global _default
    _default = DynamicCurve.fromdescr(shape, mindb=mindb, maxdb=maxdb, dynamics=possible_dynamics)


def getDefaultCurve() -> DynamicCurve:
    return _default
