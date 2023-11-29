"""
functions to convert between dB and musical dynamics
also makes a representation of the amplitude in terms of musical dynamics
"""
from __future__ import annotations
from bisect import bisect as _bisect
from dataclasses import dataclass

from pitchtools import db2amp, amp2db
from emlib import misc
import bpf4

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Callable, Sequence


dynamicSteps = ('pppp', 'ppp', 'pp', 'p', 'mp',
                'mf', 'f', 'ff', 'fff', 'ffff')


@dataclass
class DynamicDescr:
    shape: str
    mindb: float
    maxdb: float = 0.
    dynamics: str = "ppp pp p mp mf f ff fff"

    def makeCurve(self) -> DynamicCurve:
        return DynamicCurve.fromdescr(self.shape, mindb=self.mindb, maxdb=self.maxdb,
                                      dynamics=self.dynamics)


class DynamicCurve:
    """
    A DynamicCurve maps an amplitude to a dynamic expression

    Attributes:
        dynamics: a list of possible dynamic expressions, ordered from
            soft to loud

    .. seealso:: :meth:`DynamicCurve.fromdescr`

    """
    
    def __init__(self, curve: Callable[[float], float], dynamics: Sequence[str] = None):
        """
        Args:
            curve: a bpf mapping 0-1 to amplitude(0-1)
            dynamics: a list of possible dynamics, or None to use the default

        See Also
        ~~~~~~~~

        * :meth:`DynamicCurve.fromdescr`

        """
        self.dynamics: tuple[str, ...] = tuple(dynamics) if dynamics else dynamicSteps
        bpf = bpf4.asbpf(curve, bounds=(0, 1)).fit_between(0, len(self.dynamics)-1)
        self._amps2dyns, self._dyns2amps = _makeDynamicsMapping(bpf, self.dynamics)
        self._shape: str = ''
        assert len(self._amps2dyns) == len(self.dynamics)

    @classmethod
    def fromdescr(cls, shape='expon(0.5)', mindb=-80.0, maxdb=0.0,
                  dynamics:Union[str, Sequence[str]] = None) -> DynamicCurve:
        """
        Creates a DynamicCurve from a shape description

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
            dynamics = dynamics.split()
        bpf = createShape(shape, mindb, maxdb)
        out = cls(bpf, dynamics)
        out._shape = shape
        return out

    def __repr__(self) -> str:
        cls = type(self).__name__
        if self._shape:
            mindb, maxdb = self.decibelRange()
            return f'{cls}(shape={self._shape}, mindb={mindb}, maxdb={maxdb})'
        else:
            return f'{cls}(curve={self._dyns2amps}, dynamics={self.dynamics})'

    def decibelRange(self) -> tuple[float, float]:
        """
        Return the decibel range of this curve

        Returns:
            a tuple (mindb, maxdb)

        Example
        ~~~~~~~

            >>> from maelzel.music import dynamics
            >>> curve = dynamics.DynamicCurve.fromdescr("expon(0.5)", mindb=-60, maxdb=0)
            >>> curve.decibelRange()
            (-60, 0)

        """
        mindyn = self.dynamics[0]
        maxdyn = self.dynamics[-1]
        return self.dyn2db(mindyn), self.dyn2db(maxdyn)

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
        insert_point = _bisect(curve, (amp, ''))
        if not nearest:
            idx = max(0, insert_point-1)
            return curve[idx][1]
        amp0, dyn0 = curve[insert_point - 1]
        amp1, dyn1 = curve[insert_point]
        db = amp2db(amp)
        return dyn0 if abs(db-amp2db(amp0)) < abs(db-amp2db(amp1)) else dyn1

    def dyn2amp(self, dyn:str) -> float:
        """
        Convert a dynamic expressed as a string to a corresponding amplitude
        """
        amp = self._dyns2amps.get(dyn.lower())
        if amp is None:
            raise ValueError(f"dynamic {dyn} not known")
        return amp

    def dyn2db(self, dyn:str) -> float:
        """Convert a dynamic expression to an amplitude in dB"""
        return amp2db(self.dyn2amp(dyn))

    def db2dyn(self, db:float) -> str:
        """Convert an amp in dB to a dynamic expression"""
        return self.amp2dyn(db2amp(db))

    def dyn2index(self, dyn:str) -> int:
        """
        Convert the given dynamic to an integer index
        """
        try:
            return self.dynamics.index(dyn)
        except ValueError:
            raise ValueError(f"Dynamic not defined, should be one of {self.dynamics}")

    def index2dyn(self, idx:int) -> str:
        """
        Convert a dynamic index to a dynamic

        Args:
            idx: the dynamic index

        Returns:
            the corresponding dynamic as string

        Example
        ~~~~~~~

            >>> from maelzel.music.dynamics import DynamicCurve
            >>> curve = DynamicCurve.fromdescr('expon(0.5', dynamics='pp p mf f ff'.split())
            >>> for i in range(len(curve.dynamics)):
            ...     print(i, curve.index2dyn(i), curve.index2amp(i)
            0 	 pp 	 0.0001
            1 	 p 	     0.01
            2 	 mf 	 0.06736388483950367
            3 	 f 	     0.2911398240065722
            4 	 ff 	 1.0
        """
        return self.dynamics[idx]

    def amp2index(self, amp:float) -> int:
        """
        Converts an amplitude (in the range 0-1) to a dynamic index
        
        The dynamic index corresponds to the dynamics given when this
        DynamicCurve was created
        
        Args:
            amp: an amplitude in the range 0-1 

        Returns:
            the corresponding dynamic index
            
        Example
        ~~~~~~~
        
            >>> from maelzel.music.dynamics import DynamicCurve
            >>> curve = DynamicCurve.fromdescr('expon(0.5)', dynamics='pp p mf f ff'.split())
            >>> curve.amp2index(0.5)
            3
            >>> curve.dynamics[3]
            'f'

        """
        return self.dyn2index(self.amp2dyn(amp))

    def index2amp(self, index:int) -> float:
        """
        Convert a dynamic index to an amplitude in the range 0-1

        Args:
            index: the dynamic index

        Returns:
            the corresponding amplitude, in a range 0-1

        Example
        ~~~~~~~

            >>> from maelzel.music.dynamics import DynamicCurve
            >>> curve = DynamicCurve.fromdescr('expon(0.5', dynamics='pp p mf f ff'.split())
            >>> for i in range(len(curve.dynamics)):
            ...     print(i, curve.index2dyn(i), curve.index2amp(i)
            0 	 pp 	 0.0001
            1 	 p 	     0.01
            2 	 mf 	 0.06736388483950367
            3 	 f 	     0.2911398240065722
            4 	 ff 	 1.0


        """
        return self.dyn2amp(self.index2dyn(index))

    def asdbs(self, step=1) -> list[float]:
        """
        Convert the dynamics defined in this curve to dBs
        """
        indices = range(0, len(self.dynamics), step)
        dbs = [self.dyn2db(self.index2dyn(index)) for index in indices]
        assert dbs 
        return dbs

    def plot(self, usedB=True):
        """Plot this dynamic curve"""
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


def _validateDynamics(dynamics: Sequence[str]) -> None:
    assert not set(dynamics).difference(dynamicSteps), \
        "Dynamics not understood"


def _makeDynamicsMapping(bpf: bpf4.BpfInterface,
                         dynamics:Sequence[str] = None
                         ) -> tuple[list[tuple[float, str]], dict[str, float]]:
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
        dynamics = dynamicSteps
    assert isinstance(bpf, bpf4.core.BpfInterface)
    _validateDynamics(dynamics)
    dynamics_table = [(bpf(i), dyn) for i, dyn in enumerate(dynamics)]
    dynamics_dict = {dyn: ampdb for ampdb, dyn, in dynamics_table}
    return dynamics_table, dynamics_dict


def createShape(shape='expon(3)',
                mindb: int | float = -90,
                maxdb: int | float = 0
                ) -> bpf4.BpfInterface:
    """
    Return a bpf mapping 0-1 to amplitudes, as needed by DynamicCurve

    Args:
        shape: a descriptor of the curve to use to map amplitude to dynamics
        mindb: the min. representable amplitude (in dB)
        maxdb: the max. representable amplitude (in dB)

    Returns:
        a bpf mapping the range 0-1 to amplitude following the given shape

    If *x* is dynamic and *y* is amplitude, an exponential curve with exp > 1
    will allocate more dynamics to the soft amplitude range, resulting in more
    resolution for small amplitudes.
    A curve with exp < 1 will result in more resolution for high amplitudes

    .. note::

        If a curve has high resolution for soft amplitudes (meaning that a small variation
        in amplitude results in high variation in dynamic), soft dynamics will
        have low resolution (meaning that a high variation in dynamic will have low variation
        in amplitude) and viceversa
    """
    minamp, maxamp = db2amp(mindb), db2amp(maxdb)
    return bpf4.util.makebpf(shape, [0, 1], [mindb, maxdb]).db2amp()

