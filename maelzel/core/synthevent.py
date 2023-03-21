from __future__ import annotations

import emlib.mathlib
import emlib.misc
import pitchtools as pt

from ._common import logger
from typing import TYPE_CHECKING
from maelzel.core import renderer

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Sequence
    import csoundengine.instr
    from .config import CoreConfig
    from ._typedefs import *
    import matplotlib.pyplot as plt


__all__ = (
    'PlayArgs',
    'SynthEvent',
    'cropEvents',
    'plotEvents'
)


_MAX_NUM_PFIELDS = 1900


class PlayArgs:
    """
    Playback customizations for an event or a set of events

    Each :class:`~maelzel.core.mobj.MObj` has a :attr:`~maelzel.core.mobj.MObj.playargs` attribute, which is an
    instance of :class:`PlayArgs` and allows each object to set playback attributes like
    the instrument used, fade, pan position, etc. Each attribute set is added to the :attr:`PlayArgs.args` dict.

    PlayArgs cascade similarly to css. If a note sets a specific attribute, like 'instr' (the instrument used), then
    this value is used since it is the most specific. If the note leaves that unset but the note is contained
    within a :class:`~maelzel.core.mobj.Chain` and this ``Chain`` sets the 'instr' key within its own
    :attr:`~maelzel.core.mobj.MObj.playargs`, then this value is used. If that chain is contained within a
    :class:`~maelzel.core.score.Score` and the score itself has the 'instr' key set, then that value is used, etc.
    Fallback defaults are often defined in the :ref:`configuration <config>`

    Keys
    ~~~~

    * delay: when to schedule this synth. This time is added to the absolute offset of an object
    * chan: the channel to output to. If the synth is multichannel this is the first of many
        adjacent channels (TODO: implement channel mappings or similar strategies for spatialization)
    * gain: an overall gain of the synth
    * fade: a fade value or a tuple (fadein, fadeout), in seconds
    * instr: the instrument preset to use
    * pitchinterpol: interpolation mode for pitch ('linear', 'cos', 'freqlinear', 'freqcos')
    * fadeshape: shape of the fade curve (TODO)
    * args: any args passed to the instr preset (a dict {'pname': value}
    * priority: the priority of this synth. Priorities start with 1, low priorities are evaluated
        first. An instr with a higher priority is used to receive audio from an instr with
        a lower priority
    * position: the horizontal placement in place. 0=left, 1=right. For multichannel (> 2)
        presets this value is interpreted freely by the instrument, which does its own spatialization
    * sustain: if positive the last breakpoint is extended by this totalDuration. This is used mainly for
        sample based instruments (soundfont) to extend the playback. It can be used to implement
        one-shot sample playback
    * transpose: add an extra transposition to all breakpoints
    * linkednext: is this event linked to a next one? This normally depends on the .gliss attribute
        of a note/chord, but can be used to create a glissando which is only valid for playback
        and not displayed as notation
    """
    playkeys = {'delay', 'chan', 'gain', 'fade', 'instr', 'pitchinterpol',
                'fadeshape', 'args', 'priority', 'position', 'sustain', 'transpose',
                'linkednext'}

    """Available keys for playback customization"""

    __slots__ = ('db',)

    def __init__(self, db: dict[str, Any] = None):
        if db is None:
            db = {}
        self.db: dict[str, Any] = db
        """A dictionary holding the arguments explicitely specified"""

        assert not(db.keys() - self.playkeys)
        assert all(v is not None for v in db.values())

    def __bool__(self):
        return bool(self.db)

    def keys(self) -> set[str]:
        """All possible keys for a PlayArgs instance

        This is not the equivalent of the actual set keys
        (see ``playargs.args.keys()``)"""
        return self.playkeys

    def values(self) -> Iterable:
        """
        The values corresponding to all possible keys

        This might contain unset values. For only the actually set
         values, use ``playargs.args.values()``"""
        args = self.db
        return (args.get(k) for k in self.playkeys)

    def items(self) -> dict[str, Any]:
        """Like dict.items()"""
        args = self.db
        return {k: args.get(k) for k in self.playkeys}

    def get(self, key: str, default=None):
        """Like dict.get()"""
        if not key in self.playkeys:
            raise KeyError(f"Unknown key {key}. Possible keys are: {self.playkeys}")
        return self.db.get(key, default)

    def __getitem__(self, item: str):
        return self.db[item]

    def __setitem__(self, key: str, value) -> None:
        assert key in self.playkeys, f'PlayArgs: unknown key "{key}", possible keys: {self.playkeys}'
        if value is None:
            del self.db[key]
        else:
            self.db[key] = value

    def overwriteWith(self, p: PlayArgs) -> None:
        """
        Overwrites this with set values in *p*

        This is actually the same as merging self's dict with
        *p*'s dict as long as *self* or *p* do not have any
        value set to None

        Args:
            p: another PlayArgs instance

        """
        self.db.update(p.db)

    def overwrittenWith(self, p: PlayArgs) -> PlayArgs:
        out = self.copy()
        out.db.update(p.db)
        return out

    def copy(self) -> PlayArgs:
        """
        Returns a copy of self
        """
        return PlayArgs(self.db.copy())

    def clone(self, **kws) -> PlayArgs:
        """
        Clone self with modifications

        Args:
            **kws: one of the possible playkeys

        Returns:
            the cloned PlayArgs

        """
        outargs = self.db.copy()
        outargs.update(kws)
        return PlayArgs(outargs)

    def __repr__(self):
        args = ', '.join(f'{k}={v}' for k, v in self.db.items())
        return f"PlayArgs({args})"

    def asdict(self) -> dict[str, Any]:
        """
        This PlayArgs as dict

        Only set key:value pairs are returned

        Returns:
            the set key:value pairs, as dict
        """
        return self.db

    @staticmethod
    def makeDefault(conf: CoreConfig) -> PlayArgs:
        """
        Create a PlayArgs with defaults from a CoreConfig

        Args:
            conf: a CoreConfig

        Returns:
            the created PlayArgs

        """
        d = dict(delay=0,
                 chan=1,
                 gain=conf['play.gain'],
                 fade=conf['play.fade'],
                 instr=conf['play.instr'],
                 pitchinterpol=conf['play.pitchInterpolation'],
                 fadeshape=conf['play.fadeShape'],
                 priority=1,
                 position=-1,
                 sustain=0,
                 transpose=0)
        return PlayArgs(d)

    def filledWith(self, other: PlayArgs) -> PlayArgs:
        """
        Clone of self with any unset value in self filled with other

        Args:
            other: another PlayArgs

        Returns:
            a clone of self with unset values set from *other*

        """
        db = self.db.copy()
        for k, v in other.db.items():
            if v is not None:
                db[k] = db.get(k, v)
        return PlayArgs(db)

    def fillWith(self, other: PlayArgs) -> None:
        """
        Fill any unset value in self with the value in other **inplace**

        Args:
            other: another PlayArgs

        """
        assert isinstance(other, PlayArgs)
        args = self.db
        for k, v in other.db.items():
            if v is not None:
                args[k] = args.get(k, v)

    def update(self, d: dict[str, Any]) -> None:
        self.db.update(d)

    def fillDefaults(self, cfg: CoreConfig) -> None:
        """
        Fill this PlayArgs with defaults (in place)

        Only unset keys are set.

        Args:
            cfg: a CoreConfig

        """
        args = self.db
        args.setdefault('delay', 0.)
        args.setdefault('gain', cfg['play.gain'])
        args.setdefault('instr', cfg['play.instr'])
        args.setdefault('fade', cfg['play.fade'])
        args.setdefault('pitchinterpol', cfg['play.pitchInterpolation'])
        args.setdefault('priority', 1)
        args.setdefault('position', -1)
        args.setdefault('sustain', 0)
        args.setdefault('chan', 1)
        args.setdefault('fadeshape', cfg['play.fadeShape'])
        args.setdefault('transpose', 0)


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

    A User never creates a :class:`SynthEvent`: a :class:`SynthEvent` is
    created by a :class:`Note` or a :class:`Voice`. They are used internally
    to generate a set of events to be played/recorded by the playback engine.

    """
    __slots__ = ("bps", "delay", "chan", "fadein", "fadeout", "gain",
                 "instr", "pitchinterpol", "fadeShape", "args",
                 "priority", "position", "_namedArgsMethod", "linkednext",
                 "numchans", "whenfinished", "properties", 'sustain',
                 'initfunc', '_initdone')

    pitchinterpolToInt = {
        'linear': 0,
        'cos': 1,
        'freqlinear': 2,
        'freqcos': 3
    }
    """Map an interpolation shape to an identifier used inside csound"""

    fadeshapeToInt = {
        'linear': 0,
        'cos': 1,
        'scurve': 2,
    }
    """Map a fadeshape to an identifier used inside csound"""

    def __init__(self,
                 bps: list[list[float, ...]],
                 instr: str,
                 delay: float = 0.0,
                 chan: int = 1,
                 fade: float | tuple[float, float] = 0,
                 gain: float = 1.0,
                 pitchinterpol: str = 'linear',
                 fadeshape: str = 'cos',
                 args: dict[str, float|str] = None,
                 priority: int = 1,
                 position: float = -1,
                 numchans: int = 2,
                 linkednext=None,
                 whenfinished: Callable = None,
                 properties: dict[str, Any] | None = None,
                 sustain: float = 0.,
                 initfunc: Callable[[SynthEvent, renderer.Renderer], None]=None,
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
            pitchinterpol: which interpolation to use for pitch ('linear', 'cos', 'freqlinear', 'freqcos')
            fadeshape: shape of the fade ('linear', 'cos', 'scurve')
            args: named parameters
            priority: schedule the corresponding instr at this priority
            numchans: the number of channels this event outputs
            linkednext: a hint to merge multiple events into longer lines.
            kws: ignored at the moment
        """
        if len(bps[0]) < 2:
            raise ValueError(f"A breakpoint should have at least (delay, pitch), "
                             f"but got {bps}")

        bpslen = len(bps[0])
        if any(len(bp) != bpslen for bp in bps):
            raise ValueError("Not all breakpoints have the same length")

        if len(bps[0]) < 3:
            raise ValueError("A breakpoint needs to have at least (time, pitch, amp)")

        delay = float(delay)
        # assert isinstance(delay, (int, float)) and delay >= 0, f"Expected int|float >= 0, got {delay} ({type(delay)})"

        if isinstance(fade, tuple):
            fadein, fadeout = fade
        else:
            fadein = fadeout = fade

        self.bps = bps
        """breakpoints, where each breakpoint is a tuple of (timeoffset, midi, amp, [...])"""

        dur = self.bps[-1][0] - self.bps[0][0]

        self.delay = delay
        """time delay - The effective time of bp[n] will be delay + bp[n][0]"""

        self.chan = chan
        """output channel"""

        self.gain = gain
        """a gain to be applied to this event"""

        self.fadein = fadein
        """fade in time"""

        self.fadeout = fadeout if dur < 0 else min(fadeout, dur)
        """fade out time"""

        self.instr = instr
        """Instrument preset used"""

        self.pitchinterpol = pitchinterpol
        """Pitch interpolation"""

        self.fadeShape = fadeshape
        """Shape of the fades"""

        self.priority = priority
        """Schedule priority (priorities start with 1)"""

        self.position = position
        """Panning position (between 0-1)"""

        self.args: dict[str, float|str] = args
        """Any parameters passed to the instrument"""

        self.linkednext = linkednext
        """Is this event linked to the next? 
        A linked synthevent is a tied note or a note with a glissando followed by 
        some continuation. In any case, the last breakpoint of this synthevent and the 
        first breakpoint of the following event should be equal for a two events to 
        be linked. NB: since we are dealing with floats, code should always check that
        the numbers are near instead of using ==
        """

        self.numchans = numchans
        """The number of signals produced by the event"""

        self.whenfinished = whenfinished
        """A function to call when this event has finished"""

        self.properties = properties
        """User defined properties for an event"""

        self.sustain = sustain
        """Sustain time after the actual totalDuration"""

        self.initfunc = initfunc
        """A function called when the event is being scheduled. 
        It has the form (synthevent, renderer) -> None, where synthevent is 
        the event being rendered and renderer is the renderer performing the render 
        (either a maelzel.core.playback.RealtimeRenderer or a 
        maelzel.core.playback.OfflineRenderer). It can be used to initialize any 
        resources needed by the event (load/make tables, add includes, global code, etc)"""

        self._initdone = False

        self._namedArgsMethod = 'pargs'
        self._consolidateDelay()
        assert self.dur > 0, f"Duration of a synth event must be possitive: {self}"
        assert self.bps[0][0] == 0

    def initialize(self, renderer):
        if not self._initdone and self.initfunc:
            self.initfunc(self, renderer)

    def _applySustain(self) -> None:
        assert not self.linkednext and self.sustain > 0
        last = self.bps[-1]
        bp = last.copy()
        bp[0] = last[0] + self.sustain
        self.bps.append(bp)

    @property
    def end(self) -> float:
        """Absolute end time of this event, in seconds"""
        return self.delay + self.bps[-1][0]

    @property
    def dur(self) -> float:
        """Duration of this event, in seconds"""
        if not self.bps:
            return 0
        return float(self.bps[-1][0] - self.bps[0][0])

    def resolvedPosition(self) -> float:
        if self.position >= 0:
            return self.position
        if self.numchans == 1:
            return 0.
        else:
            return 0.5


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
                          pitchinterpol=self.pitchinterpol,
                          fadeshape=self.fadeShape,
                          args=self.args,
                          priority=self.priority,
                          position=self.position,
                          numchans=self.numchans,
                          linkednext=self.linkednext,
                          whenfinished=self.whenfinished,
                          properties=self.properties.copy() if self.properties else None)

    @property
    def fade(self) -> tuple[float, float]:
        """A tuple (fadein, fadeout)"""
        return self.fadein, self.fadeout

    @fade.setter
    def fade(self, value: tuple[float, float]):
        self.fadein, self.fadeout = value

    @classmethod
    def fromPlayArgs(cls,
                     bps: list[breakpoint_t],
                     playargs: PlayArgs,
                     properties: dict[str, Any] | None = None,
                     **kws
                     ) -> SynthEvent:
        """
        Construct a SynthEvent from breakpoints and playargs

        Args:
            bps: the breakpoints
            playargs: playargs
            properties: any properties passed to the constructor
            kws: any argument passed to SynthEvent's constructor

        Returns:
            a new SynthEvent
        """
        args = playargs.db
        if kws:
            args = args.copy()
            args.update(kws)
        return SynthEvent(bps=bps,
                          properties=properties,
                          **args)

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
        """A clone of this event, shifted in time by the given offset"""
        return self.clone(delay=self.delay+offset)

    def cropped(self, start: float, end: float) -> SynthEvent:
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
        if self.linkednext:
            info.append(f'linkednext=True')
        if self.args:
            info.append(f"args={self.args}")
        if self.sustain:
            info.append(f"sustain={self.sustain}")
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

    @staticmethod
    def mergeTiedEvents(events: Sequence[SynthEvent]) -> SynthEvent:
        return mergeLinkedEvents(events)

    def resolvePfields(self: SynthEvent,
                       instr: csoundengine.instr.Instr
                       ) -> list[float|str]:
        """
        Returns pfields, **beginning with p2**.

        ==== =====  ============================================
        idx  parg    desc
        ==== =====  ============================================
        0    2       delay
        1    3       totalDuration
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
        if not self.linkednext and self.sustain > 0:
            self._applySustain()

        pitchInterpolMethod = SynthEvent.pitchinterpolToInt[self.pitchinterpol]
        fadeshape = SynthEvent.fadeshapeToInt[self.fadeShape]
        # if no userpargs, bpsoffset is 15
        numPargs5 = len(instr.pargsIndexToName)
        bpsrows = len(self.bps)
        bpscols = self.breakpointSize()
        pfields = [
            float(self.delay),
            float(self.dur),
            0,  # table index, to be filled later
        ]

        pfields5 = [
            0,            # p5, idx: 4 (bpsoffset)
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
        numBuiltinPargs = len(pfields5)  # 10
        numUserArgs = numPargs5 - numBuiltinPargs
        bpsoffset = 15 + numUserArgs
        pfields5[0] = bpsoffset

        if self._namedArgsMethod == 'pargs' and numUserArgs > 0:
            pfields5 = instr.pargsTranslate(args=pfields5, kws=self.args)
        pfields.extend(pfields5)
        for bp in self.bps:
            pfields.extend(bp)
        pfields = [x if isinstance(x, str) else float(x) for x in pfields]
        if len(pfields) > _MAX_NUM_PFIELDS:
            logger.error(f"This SynthEvent has too many pfields: {len(pfields)}")
        return pfields

    def plot(self, axes: plt.Axes = None, notenames=False) -> plt.Axes:
        """
        Plot the trajectory of this synthevent

        Args:
            axes: a matplotlib.pyplot.Axes, will be used if given
            notenames: if True, use notenames for the y axes

        Returns:
            the axes used
        """
        import matplotlib.pyplot as plt
        import matplotlib
        ownaxes = axes is None
        if axes is None:
            # f: plt.Figure = plt.figure(figsize=figsize)
            f: plt.Figure = plt.figure()
            axes: plt.Axes = f.add_subplot(1, 1, 1)
        times = [bp[0] for bp in self.bps]
        midis = [bp[1] for bp in self.bps]
        if notenames:
            noteformatter = matplotlib.ticker.FuncFormatter(lambda s, y: f'{str(s).ljust(3)}: {pt.m2n(s)}')
            axes.yaxis.set_major_formatter(noteformatter)
            axes.tick_params(axis='y', labelsize=8)

        if ownaxes:
            axes.grid()
        axes.plot(times, midis)
        return axes


def mergeLinkedEvents(events: Sequence[SynthEvent]) -> SynthEvent:
    """
    Merge linked events

    Two events are linked if the first event has its `.linkednext` attribute set as True
    and the last breakpoint of the first event is equal to the first breakpoint of the
    second. This is used within a Chain or Voice to join the playback events of
    multiple chords/notes to single synthevents

    .. note::

        raises ValueError if the events cannot be merged

    Args:
        events: the events to merge

    Returns:
        the merged event

    """
    assert len(events) >= 2
    assert all(ev.linkednext for ev in events[:-1]), f"Cannot merge events not marked as linked: {events}"
    assert all(ev.bps[0][0] == 0 for ev in events)
    firstevent = events[0]
    bps = []
    eps = 1.e-10
    firstdelay = firstevent.delay
    now = firstevent.delay
    for event in events:
        if event.delay < now:
            raise ValueError(f"Trying to merge {events=}\nEvent {event} starts before the end "
                             f"of last event ({now=})")
        elif event.delay - now > eps:
            raise ValueError(f"Trying to merge {events=}\nEvent {event} is not aligned with the "
                             f"end of last event ({now=}, gap = {event.delay-now})")
        assert event.bps[0][0] == 0
        now = event.bps[-1][0] + event.delay

        for bp in event.bps[:-1]:
            bp = bp.copy()
            bp[0] += event.delay - firstdelay
            bps.append(bp)

    # Add the last breakpoint of the last event
    lastevent = events[-1]
    lastbp = lastevent.bps[-1]
    lastbp[0] += lastevent.delay - firstdelay
    bps.append(lastbp)

    mergedevent = firstevent.clone(bps=bps, linkednext=events[-1].linkednext)
    return mergedevent


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
        end = max(ev.end for ev in events)
        assert start < end, f"{start=}, {end=}"
    else:
        end = float(end)
    from emlib.mathlib import intersection
    events = [event.cropped(start, end) for event in events
              if intersection(start, end, event.delay, event.end) is not None]
    if rewind:
        events = [event.timeShifted(-start)
                  for event in events]
    return events


def plotEvents(events: list[SynthEvent], axes: plt.Axes = None, notenames=False
               ) -> plt.Axes:
    """
    Plot all given events within the same axes (static method)

    Args:
        events: the events to plot
        axes: the axes to use, if given
        notenames: if True, use notenames for the y axes

    Returns:
        the axes used

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> from maelzel.core import synthevent
        >>> chord = Chord("4E 4G# 4B", 2, gliss="4Eb 4F 4G")
        >>> synthevent.plotEvents(chord.events(), notenames=True)
    """
    import matplotlib.pyplot as plt
    import matplotlib

    if axes is None:
        # f: plt.Figure = plt.figure(figsize=figsize)
        f: plt.Figure = plt.figure()
        axes: plt.Axes = f.add_subplot(1, 1, 1)

    for event in events:
        event.plot(axes=axes, notenames=False)

    axes.grid()

    if notenames:
        noteformatter = matplotlib.ticker.FuncFormatter(lambda s, y: f'{str(s).ljust(3)}: {pt.m2n(s)}')
        axes.yaxis.set_major_formatter(noteformatter)
        axes.tick_params(axis='y', labelsize=8)

    return axes