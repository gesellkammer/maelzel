from __future__ import annotations
import dataclasses

import emlib.mathlib
import emlib.misc

from ._common import *
from ._typedefs import *
from . import tools
from .workspace import activeConfig
import copy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *

_pitchinterpolToInt = {
    'linear':0,
    'cos':1,
    'freqlinear':2,
    'freqcos':3
}


_fadeshapeToInt = {
    'linear': 0,
    'cos': 1
}


@dataclasses.dataclass
class PlayArgs:
    delay: float = None
    "extra delay"

    chan: int = None
    gain: float = None
    fade: Union[None, float, Tuple[float, float]] = None
    instr: str = None
    pitchinterpol: str = None
    fadeshape: str = None
    params: Dict[str, float] = None
    priority: int = 1
    position: float = None

    def __repr__(self) -> str:
        parts = []
        if self.delay is not None:
            parts.append(f'delay={self.delay}')
        if self.gain is not None:
            parts.append(f'gain={self.gain}')
        if self.fade is not None:
            parts.append(f'fade={self.fade}')
        if self.instr is not None:
            parts.append(f'instr={self.instr}')
        if self.fadeshape is not None:
            parts.append(f'fadeshape={self.fadeshape}')
        if self.params is not None:
            parts.append(f'params={self.params}')
        if self.position is not None:
            parts.append(f'position={self.position}')
        if self.priority != 1:
            parts.append(f'priority={self.priority}')
        return f"PlayArgs({', '.join(parts)})"

    @staticmethod
    def keys() -> Set[str]:
        return {field.name for field in dataclasses.fields(PlayArgs)}

    def values(self) -> Iterable:
        return (getattr(self, k) for k in self.keys())

    def checkValues(self) -> None:
        """ Check own values for validity """
        assert self.pitchinterpol is None or self.pitchinterpol in _pitchinterpolToInt
        assert self.fadeshape is None or self.fadeshape in _fadeshapeToInt
        assert self.chan is None or self.chan

    def hasUndefinedValues(self) -> bool:
        """ Are there any unfilled values in this PlayArgs instance? """
        return not any(v is None for v in self.asdict().values())

    def clone(self, **kws) -> PlayArgs:
        """ Create a new PlayArgs instance with the attributes
        in kws modified/filled in"""
        return dataclasses.replace(self, **kws)

    def copy(self) -> PlayArgs:
        return copy.copy(self)

    def asdict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def filledWith(self, other: PlayArgs) -> PlayArgs:
        """
        Return a new PlayArgs with unset values filled with corresponding
        values from `other`
        """
        a = self.asdict()
        b = other.asdict()
        for k, v in a.items():
            if v is None:
                a[k] = b[k]
        return PlayArgs(**a)

    def fillWith(self, other: PlayArgs) -> None:
        """
        Fill unset values with values from `other`, inplace
        """
        a = self.asdict()
        b = other.asdict()
        for k, v in a.items():
            if v is None:
                setattr(self, k, b[k])

    def fillWithConfig(self, cfg: dict):
        """
        Fill unset values with config
        """
        if self.delay is None:
            self.delay = 0
        if self.gain is None:
            self.gain = cfg['play.gain']
        if not self.instr:
            self.instr = cfg['play.instr']
        if self.fade is None:
            self.fade = cfg['play.fade']
        if not self.pitchinterpol:
            self.pitchinterpol = cfg['play.pitchInterpolation']
        if not self.fadeshape:
            self.fadeshape = cfg['play.fadeShape']
        if self.priority is None:
            self.priority = 1
        if self.position is None:
            self.position = -1


def _interpolateBreakpoints(t: float, bp0: List[float], bp1: List[float]
                            ) -> List[float]:
    t0, t1 = bp0[0], bp1[0]
    assert t0 <= t <= t1, f"{t0=}, {t=}, {t1=}"
    delta = (t - t0) / (t1 - t0)
    bp = [t]
    for v0, v1 in zip(bp0[1:], bp1[1:]):
        bp.append(v0 + (v1-v0)*delta)
    return bp


class CsoundEvent:
    """
    Represents a standard event (a line of variable breakpoints)

    init: delay=0, chan=0, fadein=None, fadeout=None
    bp: delay, midi, amp, ...
    (all breakpoints should have the same length)

    protocol:

    i inum, delay, dur, p4, p5, ...
    i inum, delay, dur, chan, fadein, fadeout, bplen, *flat(bps)

    inum is given by the manager
    dur is calculated based on bps
    bplen is calculated based on bps (it is the length of one breakpoint)

    Attributes:
        bps: breakpoints, where each breakpoint is a tuple of (timeoffset, midi, amp, [...])
        delay: time delay. The effective time of bp[n] will be delay + bp[n][0]
        chan: output channel
        fade: fade time (either a single value or a tuple (fadein, fadeout)
        gain: a gain to be applied to this event
        instr: the instr preset
        pitchinterpol: which pitchinterpolation to use ('linear', 'cos')
        fadeshape: shape of the fade ('linear', 'cos')
        namedArgs: params used to initialize a parameter table
        priority: schedule the corresponding instr at this priority
        userpargs: user pargs
    """
    __slots__ = ("bps", "delay", "chan", "fadein", "fadeout", "gain",
                 "instr", "pitchInterpolMethod", "fadeShape", "stereo", "namedArgs",
                 "priority", "position", "userpargs", "_namedArgsMethod", "tiednext")

    def __init__(self,
                 bps: List[Tuple[float, ...]],
                 delay:float=0.0,
                 chan:int = None,
                 fade:Union[float, Tuple[float, float]]=None,
                 gain:float = 1.0,
                 instr:str=None,
                 pitchinterpol:str=None,
                 fadeshape:str=None,
                 params: Dict[str, float] = None,
                 priority:int=1,
                 position:float = None,
                 userpargs: Optional[List[float]]=None,
                 tiednext=False):
        """
        bps (breakpoints): a seq of (delay, midi, amp, ...) of len >= 1.

        Args:
            bps: breakpoints, where each breakpoint is a tuple of (timeoffset, midi, amp,
            [...])
            delay: time delay. The effective time of bp[n] will be delay + bp[n][0]
            chan: output channel
            fade: fade time (either a single value or a tuple (fadein, fadeout)
            gain: a gain to be applied to this event
            instr: the instr preset
            pitchinterpol: which pitchinterpolation to use ('linear', 'cos')
            fadeshape: shape of the fade ('linear', 'cos')
            params: named parameters
            priority: schedule the corresponding instr at this priority
            userpargs: ???
            tiednext: a hint to merge multiple events into longer lines.
        """
        cfg = activeConfig()

        if len(bps[0]) < 2:
            raise ValueError(f"A breakpoint should have at least (delay, pitch), "
                             f"but got {bps}")

        bps = tools.carryColumns(bps)
        # assert all(isinstance(bp, list) for bp in bps)
        if len(bps[0]) < 3:
            column = [1] * len(bps)
            bps = tools.addColumn(bps, column)
        assert len(bps[0])>= 3
        assert all(isinstance(bp, (list, tuple)) and len(bp) == len(bps[0]) for bp in bps)
        self.bps = bps
        dur = self.bps[-1][0] - self.bps[0][0]
        fadein, fadeout = tools.normalizeFade(fade, cfg['play.fade'])
        self.delay = delay
        self.chan = chan or cfg['play.chan'] or 1
        self.gain = gain or cfg['play.gain']
        self.fadein = max(fadein, 0.0001)
        self.fadeout = fadeout if dur < 0 else min(fadeout, dur)
        self.instr = instr
        self.pitchInterpolMethod = pitchinterpol or cfg['play.pitchInterpolation']
        self.fadeShape = fadeshape or cfg['play.fadeShape']
        self.priority = priority
        self.position = position
        self.userpargs = userpargs
        self.namedArgs = params
        self.tiednext = tiednext
        self._consolidateDelay()
        self._namedArgsMethod = cfg['play.namedArgsMethod']

    @property
    def dur(self) -> float:
        if not self.bps:
            return 0
        return float(self.bps[-1][0] - self.bps[0][0])

    def clone(self, **kws) -> CsoundEvent:
        out = copy.deepcopy(self)
        for k, v in kws.items():
            setattr(out, k, v)
        if out.bps[0][0] != 0:
            out._consolidateDelay()
        return out

    @property
    def start(self) -> float:
        return self.bps[0][0]

    @property
    def end(self) -> float:
        return self.bps[-1][0]

    @property
    def fade(self) -> Tuple[float, float]:
        return (self.fadein, self.fadeout)

    @fade.setter
    def fade(self, value: Tuple[float, float]):
        self.fadein, self.fadeout = value

    @classmethod
    def fromPlayArgs(cls, bps:List[breakpoint_t], playargs: PlayArgs, **kws
                     ) -> CsoundEvent:
        """
        Construct a CsoundEvent from breakpoints and playargs

        Args:
            bps: the breakpoints
            playargs: playargs

        Returns:
            a new CsoundEvent
        """
        d = dataclasses.asdict(playargs)
        if kws:
            d.update(kws)
        return cls(bps=bps, **d)

    def _consolidateDelay(self) -> None:
        delay0 = self.bps[0][0]
        if delay0 > 0:
            self.delay += delay0
            for bp in self.bps:
                bp[0] -= delay0

    def _applyTimeFactor(self, timefactor: float) -> None:
        if timefactor == 1:
            return
        self.delay *= timefactor
        self.bps = [(bp[0]*timefactor,)+bp[1:] for bp in self.bps]

    def cropped(self, start:float, end:float) -> CsoundEvent:
        """
        Return a cropped version of this CsoundEvent
        """
        start -= self.delay
        end -= self.delay
        out = []
        for i in range(len(self.bps)):
            bp: List[float] = self.bps[i]
            if bp[0] < start:
                if i < len(self.bps)-1 and start < self.bps[i+1][0]:
                    bp = _interpolateBreakpoints(start, bp, self.bps[i+1])
                    out.append(bp)
            elif start <= bp[0] < end:
                out.append(bp.copy())
                if i < len(self.bps) - 1 and end <= self.bps[i+1][0]:
                    bp2 = _interpolateBreakpoints(end, bp, self.bps[i+1])
                    out.append(bp2)
            elif bp[0] > end:
                break
        return self.clone(bps=out)


    def breakpointSize(self) -> int:
        """ Returns the number of breakpoints in this CsoundEvent """
        return len(self.bps[0])

    def getPfields(self, numchans=1) -> List[float]:
        """
        returns pfields, **beginning with p2**.

        ==== =====  ======
        idx  parg    desc
        ==== =====  ======
        0    2       delay
        1    3       duration
        2    4       tabnum
        3    5       bpsoffset (pfield index, starting with 1)
        4    6       bpsrows
        5    7       bpscols
        6    8       gain
        7    9       chan
        8    0       position
        9    1       fade0
        0    2       fade1
        1    3       pitchinterpol
        2    4       fadeshape
        .
        . reserved space for user pargs
        .
        ==== =====  ======

        breakpoint data

        tabnum: if 0 it is discarded and filled with a valid number later
        """
        pitchInterpolMethod = _pitchinterpolToInt[self.pitchInterpolMethod]
        fadeshape = _fadeshapeToInt[self.fadeShape]
        # if no userpargs, bpsoffset is 15
        bpsoffset = 15 + (len(self.userpargs) if self.userpargs else 0)
        bpsrows = len(self.bps)
        bpscols = self.breakpointSize()
        pos = self.position
        if pos == -1:
            pos = 0. if numchans == 1 else 0.5

        pfields = [float(self.delay),
                   self.dur,
                   0,  # table index, to be filled later
                   bpsoffset,
                   bpsrows,
                   bpscols,
                   self.gain,
                   self.chan,
                   pos,
                   self.fadein,
                   self.fadeout,
                   pitchInterpolMethod,
                   fadeshape,
                   ]

        for bp in self.bps:
            pfields.extend(bp)
        assert all(isinstance(p, (int, float)) for p in pfields), pfields
        return pfields

    def _repr_html_(self) -> str:
        rows = [[f"{bp[0] + self.delay:.3f}", f"{bp[0]:.3f}"] + ["%.6g"%x for x in bp[1:]] for bp in self.bps]
        headers = ["Abs time", "0. Rel. time", "1. Pitch", "2. Amp"]
        l = len(self.bps[0])
        if l > 3:
            headers += [str(i) for i in range(4, l+1)]
        htmltab = emlib.misc.html_table(rows, headers=headers)
        return f"{self._reprHeader()}<br>" + htmltab

    def _reprHeader(self) -> str:
        return (f"CsoundEvent(delay={float(self.delay):.3g}, dur={self.dur:.3g}, "
                f"gain={self.gain:.4g}, chan={self.chan}"
                f", fade=({self.fadein}, {self.fadeout}), instr={self.instr})")

    def __repr__(self) -> str:
        lines = [self._reprHeader()]

        def bpline(bp):
            rest = " ".join(("%.6g"%b).ljust(8) if isinstance(b, float) else str(b) for b in bp[1:])
            return f"{float(bp[0]):.3f}s: {rest}"

        for i, bp in enumerate(self.bps):
            if i == 0:
                lines.append(f"bps {bpline(bp)}")
            else:
                lines.append(f"    {bpline(bp)}")
        lines.append("")
        return "\n".join(lines)


def cropEvents(events: List[CsoundEvent], start: Optional[float], end: Optional[float]
               ) -> List[CsoundEvent]:
    if start is None:
        start = min(ev.start for ev in events)
    if end is None:
        end = max(ev.end for ev in events)
        assert start < end
    return [event.cropped(start, end) for event in events]
