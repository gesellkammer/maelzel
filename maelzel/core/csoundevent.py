from __future__ import annotations
from dataclasses import dataclass
import dataclasses as _dataclasses

import emlib.mathlib
import emlib.misc

from ._typedefs import *
from . import _util
from .workspace import getConfig
import copy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Any, Callable
    import csoundengine.instr
    from .config import CoreConfig


__all__ = (
    'PlayArgs',
    'CsoundEvent',
    'cropEvents'
)


@dataclass
class PlayArgs:
    """
    Structure used to set playback options for any given MusicObj
    """
    delay: float = None
    "extra delay"

    chan: int = None
    """The channel to send output to"""

    gain: float = None
    """Gain factor"""

    fade: Union[None, float, tuple[float, float]] = None
    """Fade in / Fade out"""

    instr: str = None
    """Instrument template"""

    pitchinterpol: str = None
    """Pitch interpolation, one of 'linear', 'cos'"""

    fadeshape: str = None
    """The shape of the fade, one of 'linear', 'cos'"""

    params: dict[str, float] = None
    """Preset parameters"""

    priority: int = None
    """Priority to run this event"""

    position: float = None
    """Pan position (between 0-1)"""

    sustain: float = None
    """Sustain time, an extra time added at the end of the event"""

    def __repr__(self) -> str:
        parts = []
        for attr in self.keys():
            val = getattr(self, attr)
            if val is not None:
                parts.append(f'{attr}={val}')
        return f"PlayArgs({', '.join(parts)})"

    @staticmethod
    def keys() -> set[str]:
        return PlayArgs.__dataclass_fields__.keys()
        # return {field.name for field in _dataclasses.fields(PlayArgs)}

    def values(self) -> Iterable:
        return (getattr(self, k) for k in self.keys())

    def checkValues(self) -> None:
        """ Check own values for validity """
        assert self.pitchinterpol is None or self.pitchinterpol in CsoundEvent.pitchinterpolToInt
        assert self.fadeshape is None or self.fadeshape in CsoundEvent.fadeshapeToInt
        assert self.chan is None or self.chan

    def hasUndefinedValues(self) -> bool:
        """ Are there any unfilled values in this PlayArgs instance? """
        return any(getattr(self, attr) is not None for attr in self.keys())
        # return not any(v is None for v in self.asdict().values())

    def clone(self, **kws) -> PlayArgs:
        """ Create a new PlayArgs instance with the attributes
        in kws modified/filled in"""
        out = self.copy()
        for k, v in kws.items():
            setattr(out, k, v)
        return out
        # return _dataclasses.replace(self, **kws)

    def copy(self) -> PlayArgs:
        return PlayArgs(delay=self.delay,
                        chan=self.chan,
                        gain=self.gain,
                        fade=self.fade,
                        instr=self.instr,
                        pitchinterpol=self.pitchinterpol,
                        fadeshape=self.fadeshape,
                        params=self.params,
                        priority=self.priority,
                        position=self.position,
                        sustain=self.sustain
                        )
        # return copy.copy(self)

    def asdict(self) -> dict[str, Any]:
        return _dataclasses.asdict(self)

    def _filledWith(self, other: PlayArgs) -> PlayArgs:
        out = self.copy()
        out.fillWith(other)
        return out

    def filledWith(self, other: PlayArgs) -> PlayArgs:
        return PlayArgs(
            delay = _ if (_:=self.delay) is not None else other.delay,
            chan = _ if (_:=self.chan) is not None else other.chan,
            gain = _ if (_:=self.gain) is not None else other.gain,
            fade = _ if (_:=self.fade) is not None else other.fade,
            instr = self.instr or other.instr,
            pitchinterpol = self.pitchinterpol or other.pitchinterpol,
            fadeshape = self.fadeshape or other.fadeshape,
            params = self.params or other.params,
            priority = self.priority or other.priority,
            position = _ if (_:=self.position) is not None else other.position,
            sustain = _ if (_:=self.sustain) is not None else other.sustain
        )

    def _fillWith(self, other: PlayArgs) -> None:
        for k in PlayArgs.keys():
            if getattr(self, k) is None:
                setattr(self, k, getattr(other, k))

    def fillWith(self, other: PlayArgs) -> None:
        """
        Fill unset values with values from `other`, inplace
        """
        if self.delay is None:
            self.delay = other.delay
        if self.gain is None:
            self.gain = other.gain
        if not self.instr:
            self.instr = other.instr
        if self.fade is None:
            self.fade = other.fade
        if not self.pitchinterpol:
            self.pitchinterpol= other.pitchinterpol
        if not self.fadeshape:
            self.fadeshape = other.fadeshape
        if not self.params:
            self.params = other.params
        if self.priority is None:
            self.priority = other.priority
        if self.position is None:
            self.position = other.position
        if self.sustain is None:
            self.sustain = other.sustain

    def fillWithConfig(self, cfg: CoreConfig):
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
        if self.sustain is None:
            self.sustain = 0.


def _interpolateBreakpoints(t: float, bp0: list[float], bp1: list[float]
                            ) -> list[float]:
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

    A User does not normally create a ``CsoundEvent``: ``CsoundEvent``s are
    created by a :class:`Note` or a :class:`Voice` and are used internally
    to generate a set of events to be played by the csound engine.

    Attributes:
        bps: breakpoints, where each breakpoint is a tuple of (timeoffset, midi, amp, [...])
        delay: time delay. The effective time of bp[n] will be delay + bp[n][0]
        chan: output channel
        fade: fade time (either a single value or a tuple (fadein, fadeout)
        gain: a gain to be applied to this event
        instr: the instr preset
        pitchinterpol: which pitchinterpolation to use ('linear', 'cos')
        fadeShape: shape of the fade ('linear', 'cos')
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
        'cos': 1,
        'scurve': 2,
    }

    def __init__(self,
                 bps: list[tuple[float, ...]],
                 delay:float=0.0,
                 chan:int = None,
                 fade:Union[float, tuple[float, float]]=None,
                 gain:float = 1.0,
                 instr:str=None,
                 pitchinterpol:str=None,
                 fadeshape:str=None,
                 params: dict[str, float] = None,
                 priority:int=1,
                 position:float = None,
                 numchans: int = None,
                 tiednext=False,
                 whenfinished: Callable = None,
                 sustain: float = 0.):
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

        bpslen = len(bps[0])
        if any (len(bp) != bpslen for bp in bps):
            bps = _util.carryColumns(bps)
        # assert all(isinstance(bp, list) for bp in bps)
        if len(bps[0]) < 3:
            raise ValueError("A breakpoint needs to have at least (time, pitch, amp)")

        self.bps = bps
        dur = self.bps[-1][0] - self.bps[0][0]

        if fade is None:
            defaultfade = cfg['play.fade']
            fadein, fadeout = defaultfade, defaultfade
        elif isinstance(fade, tuple):
            fadein, fadeout = fade
        else:
            fadein = fadeout = fade

        self.delay = delay
        self.chan = chan or cfg['play.chan'] or 1
        self.gain = gain or cfg['play.gain']
        self.fadein = fadein
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
    def fade(self) -> tuple[float, float]:
        return (self.fadein, self.fadeout)

    @fade.setter
    def fade(self, value: tuple[float, float]):
        self.fadein, self.fadeout = value

    @classmethod
    def fromPlayArgs(cls, bps:list[breakpoint_t], playargs: PlayArgs, **kws
                     ) -> CsoundEvent:
        """
        Construct a CsoundEvent from breakpoints and playargs

        Args:
            bps: the breakpoints
            playargs: playargs

        Returns:
            a new CsoundEvent
        """
        d = CsoundEvent(bps=bps,
                        delay=playargs.delay,
                        chan=playargs.chan,
                        fade=playargs.fade,
                        gain=playargs.gain,
                        instr=playargs.instr,
                        pitchinterpol=playargs.pitchinterpol,
                        fadeshape=playargs.fadeshape,
                        params=playargs.params,
                        priority=playargs.priority,
                        position=playargs.position,
                        sustain=playargs.sustain)
        if kws:
            for k, v in kws.items():
                setattr(d, k, v)
        return d

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
            bp: list[float] = self.bps[i]
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
                        ) -> list[float]:
        """
        Returns pfields, **beginning with p2**.

        ==== =====  ============================================
        idx  parg    desc
        ==== =====  ============================================
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
        .            reserved space for user pargs
        .
        ==== =====  ============================================

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


def cropEvents(events: list[CsoundEvent], start: float = None, end: float = None,
               rewind=False
               ) -> list[CsoundEvent]:
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


