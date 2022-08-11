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
    from typing import Iterable, Any, Callable, Optional
    import csoundengine.instr
    from .config import CoreConfig


__all__ = (
    'PlayArgs',
    'SynthEvent',
    'cropEvents'
)


class PlayArgs:
    playkeys = {'delay', 'chan', 'gain', 'fade', 'instr', 'pitchinterpol',
                'fadeshape', 'params', 'priority', 'position', 'sustain'}

    def __init__(self, d: dict[str, Any] = None):
        if d is None:
            d = {}
        self.args: dict[str, Any] = d
        assert not(d.keys() - self.playkeys)
        assert all(v is not None for v in d.values())

    def __bool__(self):
        return bool(self.args)

    def keys(self) -> set[str]:
        return self.playkeys

    def values(self):
        args = self.args
        return (args.get(k) for k in self.playkeys)

    def items(self) -> dict[str, Any]:
        args = self.args
        return {k: args.get(k) for k in self.playkeys}

    def get(self, key: str, default=None):
        return self.args.get(key, default)

    def __getitem__(self, item: str):
        return self.args[item]

    def __setitem__(self, key: str, value) -> None:
        if value is None:
            del self.args[key]
        else:
            self.args[key] = value

    def overwriteWith(self, p: PlayArgs) -> None:
        self.args.update(p.args)

    def copy(self) -> PlayArgs:
        return PlayArgs(self.args.copy())

    def clone(self, **kws) -> PlayArgs:
        outargs = self.args.copy()
        outargs.update(kws)
        return PlayArgs(outargs)

    def __repr__(self):
        args = ', '.join(f'{k}={v}' for k, v in self.args.items())
        return f"PlayArgs({args})"

    def asdict(self) -> dict[str, Any]:
        return self.args

    @staticmethod
    def makeDefault(conf: CoreConfig) -> PlayArgs:
        d = dict(delay=0,
                 chan=1,
                 gain=conf['play.gain'],
                 fade=conf['play.fade'],
                 instr=conf['play.instr'],
                 pitchinterpol=conf['play.pitchInterpolation'],
                 fadeshape=conf['play.fadeShape'],
                 priority=1,
                 position=-1,
                 sustain=0)
        return PlayArgs(d)

    def filledWith(self, other: PlayArgs) -> PlayArgs:
        args = self.args.copy()
        for k, v in other.args.items():
            if v is not None:
                args[k] = args.get(k, v)
        return PlayArgs(args)

    def fillWith(self, other: PlayArgs) -> None:
        args = self.args
        for k, v in other.args.items():
            if v is not None:
                args[k] = args.get(k, v)

    def fillWithConfig(self, cfg: CoreConfig) -> None:
        args = self.args
        args.setdefault('delay', 0.)
        args.setdefault('gain', cfg['play.gain'])
        args.setdefault('instr', cfg['play.instr'])
        args.setdefault('fade', cfg['play.fade'])
        args.setdefault('pitchinterpol', cfg['play.pitchInterpolation'])
        args.setdefault('priority', 1)
        args.setdefault('position', -1)
        args.setdefault('sustain', 0)
        args.setdefault('chan', 1)


@dataclass
class _PlayArgs:
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
        assert self.pitchinterpol is None or self.pitchinterpol in SynthEvent.pitchinterpolToInt
        assert self.fadeshape is None or self.fadeshape in SynthEvent.fadeshapeToInt
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
                        sustain=self.sustain,
                        )

    @staticmethod
    def makeDefault(conf: CoreConfig) -> PlayArgs:
        return PlayArgs(delay=0,
                        chan=1,
                        gain=conf['play.gain'],
                        fade=conf['play.fade'],
                        instr=conf['play.instr'],
                        pitchinterpol=conf['play.pitchInterpolation'],
                        fadeshape=conf['play.fadeShape'],
                        priority=1,
                        position=-1,
                        sustain=0)

    def asdict(self) -> dict[str, Any]:
        return _dataclasses.asdict(self)

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

    def overwriteWith(self, p: PlayArgs):
        """
        Overwrite self with p

        Overwrites any value of self where p is not None. This would be equivalent to
        copying p and filling this copy with any set values in self
        """
        if p.delay is not None:
            self.delay = p.delay
        if p.gain is not None:
            self.gain = p.gain
        if p.instr:
            self.instr = p.instr
        if p.fade is not None:
            self.fade = p.fade
        if p.pitchinterpol:
            self.pitchinterpol = p.pitchinterpol
        if p.fadeshape:
            self.fadeshape = p.fadeshape
        if p.priority is not None:
            self.priority = p.priority
        if p.params:
            # TODO: Should these get merged??
            self.params = p.params
        if p.position is not None:
            self.position = p.position
        if p.sustain is not None:
            self.sustain = p.sustain
        if p.chan is not None:
            self.chan = p.chan

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
        if self.chan is None:
            self.chan = other.chan

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
        if self.chan is None:
            self.chan = 1


def _interpolateBreakpoints(t: float, bp0: list[float], bp1: list[float]
                            ) -> list[float]:
    t0, t1 = bp0[0], bp1[0]
    assert t0 <= t <= t1, f"{t0=}, {t=}, {t1=}"
    delta = (t - t0) / (t1 - t0)
    bp = [t]
    for v0, v1 in zip(bp0[1:], bp1[1:]):
        bp.append(v0 + (v1-v0)*delta)
    return bp


class SynthEvent:
    """
    Represents a standard event (a line of variable breakpoints)

    A User never creates a ``SynthEvent``: ``SynthEvent``s are
    created by a :class:`Note` or a :class:`Voice` and are used internally
    to generate a set of events to be played by the playback engine.

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
        sustain: any extra duration given to the event, recorded here for information
            only (it should actually be accounted for by the breakpoints)
    """
    __slots__ = ("bps", "delay", "chan", "fadein", "fadeout", "gain",
                 "instr", "pitchInterpolMethod", "fadeShape", "stereo", "namedArgs",
                 "priority", "position", "_namedArgsMethod", "tiednext",
                 "numchans", "whenfinished", "properties", 'sustain')

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
                 bps: list[list[float, ...]],
                 delay:float=0.0,
                 chan:int = 1,
                 fade:Union[float, tuple[float, float]]=None,
                 gain:float = 1.0,
                 instr: str=None,
                 pitchinterpol:str=None,
                 fadeshape:str=None,
                 params: dict[str, float] = None,
                 priority:int=1,
                 position:float = None,
                 numchans: int = None,
                 tiednext=False,
                 whenfinished: Callable = None,
                 properties: Optional[dict[str, Any]] = None,
                 sustain: float = 0.,
                 **kws):
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
            kws: ignored at the moment
        """
        cfg = getConfig()

        if len(bps[0]) < 2:
            raise ValueError(f"A breakpoint should have at least (delay, pitch), "
                             f"but got {bps}")

        bpslen = len(bps[0])
        if any (len(bp) != bpslen for bp in bps):
            raise ValueError("Not all breakpoints have the same length")

        if len(bps[0]) < 3:
            raise ValueError("A breakpoint needs to have at least (time, pitch, amp)")

        assert isinstance(delay, (int, float)) and delay >= 0

        self.bps = bps
        dur = self.bps[-1][0] - self.bps[0][0]

        if isinstance(fade, tuple):
            fadein, fadeout = fade
        else:
            fadein = fadeout = fade

        self.delay = delay
        self.chan = chan
        self.gain = gain
        self.fadein = fadein
        self.fadeout = fadeout if dur < 0 else min(fadeout, dur)
        self.instr = instr
        self.pitchInterpolMethod = pitchinterpol
        self.fadeShape = fadeshape
        self.priority = priority
        self.position = position
        self.namedArgs = params
        self.tiednext = tiednext
        self.numchans = numchans
        self.whenfinished = whenfinished
        self.properties = properties
        self.sustain = sustain
        self._namedArgsMethod = cfg['play.namedArgsMethod']
        self._consolidateDelay()

    @property
    def dur(self) -> float:
        if not self.bps:
            return 0
        return float(self.bps[-1][0] - self.bps[0][0])

    def clone(self, **kws) -> SynthEvent:
        out = self.copy()
        for k, v in kws.items():
            setattr(out, k, v)
        if out.bps[0][0] != 0:
            out._consolidateDelay()
        return out

    def copy(self) -> SynthEvent:
        return SynthEvent(bps=[bp.copy() for bp in self.bps],
                          delay=self.delay,
                          chan=self.chan,
                          fade=self.fade,
                          gain=self.gain,
                          instr=self.instr,
                          pitchinterpol=self.pitchInterpolMethod,
                          fadeshape=self.fadeShape,
                          params=self.namedArgs,
                          priority=self.priority,
                          position=self.position,
                          numchans=self.numchans,
                          tiednext=self.tiednext,
                          whenfinished=self.whenfinished,
                          properties=self.properties.copy() if self.properties else None)

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
    def fromPlayArgs(cls,
                     bps: list[breakpoint_t],
                     playargs: PlayArgs,
                     properties: Optional[dict[str, Any]] = None,
                     **kws
                     ) -> SynthEvent:
        """
        Construct a SynthEvent from breakpoints and playargs

        Args:
            bps: the breakpoints
            playargs: playargs
            kws: any argument passed to SynthEvent's constructor

        Returns:
            a new SynthEvent
        """
        d = SynthEvent(bps=bps,
                       properties=properties,
                       **playargs.args)
        if kws:
            for k, v in kws.items():
                setattr(d, k, v)
        return d

    def _consolidateDelay(self) -> None:
        delay0 = self.bps[0][0]
        assert delay0 is not None
        assert self.delay is not None
        if delay0 > 0:
            self.delay += delay0
            for bp in self.bps:
                bp[0] -= delay0

    def _applyTimeFactor(self, timefactor: float) -> None:
        if timefactor == 1:
            return
        self.delay *= timefactor
        for bp in self.bps:
            bp[0] *= timefactor

    def timeShifted(self, offset: float) -> SynthEvent:
        return self.clone(delay=self.delay+offset)

    def cropped(self, start:float, end:float) -> SynthEvent:
        """
        Return a cropped version of this SynthEvent
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
        """ Returns the number of breakpoints in this SynthEvent """
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
        return f"SynthEvent({infostr})"

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

    def resolvePfields(self: SynthEvent,
                       instr: csoundengine.instr.Instr
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
        pitchInterpolMethod = SynthEvent.pitchinterpolToInt[self.pitchInterpolMethod]
        fadeshape = SynthEvent.fadeshapeToInt[self.fadeShape]
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

        #assert all(isinstance(p, (int, float)) for p in pfields), [(p, type(p)) for p in pfields if
        #                                                           not isinstance(p, (int, float))]
        return pfields


def cropEvents(events: list[SynthEvent], start: float = None, end: float = None,
               rewind=False
               ) -> list[SynthEvent]:
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


