from __future__ import annotations
from dataclasses import dataclass
import dataclasses as _dataclasses

import emlib.mathlib
import emlib.misc

from ._common import *
from ._typedefs import *
from . import _util
from .workspace import getConfig
import copy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    import csoundengine.instr


@dataclass
class PlayArgs:
    """
    Structure used to set playback options for any given MusicObj
    """
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
        return {field.name for field in _dataclasses.fields(PlayArgs)}

    def values(self) -> Iterable:
        return (getattr(self, k) for k in self.keys())

    def checkValues(self) -> None:
        """ Check own values for validity """
        assert self.pitchinterpol is None or self.pitchinterpol in CsoundEvent.pitchinterpolToInt
        assert self.fadeshape is None or self.fadeshape in CsoundEvent.fadeshapeToInt
        assert self.chan is None or self.chan

    def hasUndefinedValues(self) -> bool:
        """ Are there any unfilled values in this PlayArgs instance? """
        return not any(v is None for v in self.asdict().values())

    def clone(self, **kws) -> PlayArgs:
        """ Create a new PlayArgs instance with the attributes
        in kws modified/filled in"""
        return _dataclasses.replace(self, **kws)

    def copy(self) -> PlayArgs:
        return copy.copy(self)

    def asdict(self) -> Dict[str, Any]:
        return _dataclasses.asdict(self)

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

        Removes any None values
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
        namedArgs: params used to initialize named parameters
        priority: schedule the corresponding instr at this priority
    """
    __slots__ = ("bps", "delay", "chan", "fadein", "fadeout", "gain",
                 "instr", "pitchInterpolMethod", "fadeShape", "stereo", "namedArgs",
                 "priority", "position", "_namedArgsMethod", "tiednext",
                 "numchans", "whenfinished")

    pitchinterpolToInt = {
        'linear': 0,
        'cos': 1,
        'freqlinear': 2,
        'freqcos': 3
    }

    fadeshapeToInt = {
        'linear': 0,
        'cos': 1
    }

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
                 numchans: int = None,
                 tiednext=False,
                 whenfinished: Callable = None):
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
            numchans: the number of channels this event outputs
            tiednext: a hint to merge multiple events into longer lines.
        """
        cfg = getConfig()

        if len(bps[0]) < 2:
            raise ValueError(f"A breakpoint should have at least (delay, pitch), "
                             f"but got {bps}")

        bps = _util.carryColumns(bps)
        # assert all(isinstance(bp, list) for bp in bps)
        if len(bps[0]) < 3:
            column = [1] * len(bps)
            bps = _util.addColumn(bps, column)
        assert len(bps[0])>= 3
        assert all(isinstance(bp, (list, tuple)) and len(bp) == len(bps[0]) for bp in bps)
        self.bps = bps
        dur = self.bps[-1][0] - self.bps[0][0]
        fadein, fadeout = _util.normalizeFade(fade, cfg['play.fade'])
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
        self.namedArgs = params
        self.tiednext = tiednext
        self.numchans = numchans
        self.whenfinished = whenfinished
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
        """time of first breakpoint

        This should be normally 0 since we consolidate the delay and start time
        """
        return self.bps[0][0]

    @property
    def end(self) -> float:
        """The last time of the breakpoints (does not take delay into account)"""
        return self.bps[-1][0]

    @property
    def endtime(self) -> float:
        """The time this event ends (delay + duration)"""
        return self.delay + self.dur

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
        d = _dataclasses.asdict(playargs)
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

    def timeShifted(self, offset: float) -> CsoundEvent:
        return self.clone(delay=self.delay+offset)

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

    def _repr_html_(self) -> str:
        rows = [[f"{bp[0] + self.delay:.3f}", f"{bp[0]:.3f}"] + ["%.6g"%x for x in bp[1:]] for bp in self.bps]
        headers = ["Abs time", "0. Rel. time", "1. Pitch", "2. Amp"]
        l = len(self.bps[0])
        if l > 3:
            headers += [str(i) for i in range(4, l+1)]
        htmltab = emlib.misc.html_table(rows, headers=headers)
        return f"{self._reprHeader()}<br>" + htmltab

    def _reprHeader(self) -> str:
        info = [f"delay={float(self.delay):.3g}, dur={self.dur:.3g}, "
                f"gain={self.gain:.4g}, chan={self.chan}"
                f", fade=({self.fadein}, {self.fadeout}), instr={self.instr}"]
        if self.namedArgs:
            info.append(f"namedArgs={self.namedArgs}")
        infostr = ", ".join(info)
        return f"CsoundEvent({infostr})"

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

    def resolvePfields(self: CsoundEvent, instr: csoundengine.instr.Instr
                        ) -> List[float]:
        """
        Returns pfields, **beginning with p2**.

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
        . named arguments, if anyreserved space for user pargs
        .
        ==== =====  ======

        breakpoint data is appended

        """
        pitchInterpolMethod = CsoundEvent.pitchinterpolToInt[self.pitchInterpolMethod]
        fadeshape = CsoundEvent.fadeshapeToInt[self.fadeShape]
        # if no userpargs, bpsoffset is 15
        numPargs5 = len(instr.pargsIndexToName)
        numBuiltinPargs = 10
        numUserArgs = numPargs5 - numBuiltinPargs
        bpsoffset = 15 + numUserArgs
        bpsrows = len(self.bps)
        bpscols = self.breakpointSize()
        pfields = [
            float(self.delay),
            self.dur,
            0,  # table index, to be filled later
        ]
        pfields5 = [
            bpsoffset,  # p5, idx: 4
            bpsrows,
            bpscols,
            self.gain,
            self.chan,
            self.position,
            self.fadein,
            self.fadeout,
            pitchInterpolMethod,
            fadeshape
        ]
        if self._namedArgsMethod == 'pargs' and numUserArgs > 0:
            pfields5 = instr.pargsTranslate(args=pfields5, kws=self.namedArgs)
        pfields.extend(pfields5)
        for bp in self.bps:
            pfields.extend(bp)

        assert all(isinstance(p, (int, float)) for p in pfields), [(p, type(p)) for p in pfields if
                                                                   not isinstance(p, (int, float))]
        return pfields


def cropEvents(events: List[CsoundEvent], start: float = None, end: float = None,
               rewind=False
               ) -> List[CsoundEvent]:
    """
    Crop the events at the given time slice

    Removes any event / part of an event outside the time slice start:end

    Args:
        events: the events to crop
        start: start of the time slice (None will only crop at the end)
        end: end of the time slice (None will only crop at the beginning)
        rewind: if True, events are timeshifted to the given start

    Returns:
        the cropped events

    """
    assert start is not None or end is not None

    if start is None:
        start = min(ev.delay for ev in events)
    else:
        start = float(start)
    if end is None:
        end = max(ev.endtime for ev in events)
        assert start < end, f"{start=}, {end=}"
    else:
        end = float(end)
    from emlib.mathlib import intersection
    events = [event.cropped(start, end) for event in events
              if intersection(start, end, event.delay, event.endtime) is not None]
    if rewind:
        events = [event.timeShifted(-start)
                  for event in events]
    return events


