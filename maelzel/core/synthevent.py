from __future__ import annotations

import copy
import math
from functools import cache
from emlib import mathlib
import emlib.misc
import pitchtools as pt
from dataclasses import dataclass

from ._common import logger, F
from typing import TYPE_CHECKING, cast as _cast
from maelzel.core import renderer
from maelzel.core.automation import Automation, SynthAutomation
from maelzel.core import presetmanager
from maelzel.core import presetdef
import csoundengine

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Sequence
    from .config import CoreConfig
    from ._typedefs import *
    from matplotlib.axes import Axes
    # import matplotlib.pyplot as plt
    from maelzel.scorestruct import ScoreStruct


__all__ = (
    'PlayArgs',
    'SynthEvent',
)


_MAX_NUM_PFIELDS = 1900


def _unique(d: dict | None, deep: bool) -> dict | None:
    if d is None:
        return d
    return copy.deepcopy(d) if deep else d.copy()


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
    * position: the horizontal placement inplace. 0=left, 1=right. For multichannel (> 2)
        presets this value is interpreted freely by the instrument, which does its own spatialization
    * sustain: if positive the last breakpoint is extended by this duration. This is used mainly for
        sample based instruments (soundfont) to extend the playback. It can be used to implement
        one-shot sample playback
    * transpose: add an extra transposition to all breakpoints
    * glisstime: slide time to next event. This allows to add glissando lines for events
      even if their gliss attr is not set, or to generate legato lines
    """
    playkeys = {'delay', 'chan', 'gain', 'fade', 'instr', 'pitchinterpol',
                'fadeshape', 'args', 'priority', 'position', 'sustain', 'transpose',
                'glisstime', 'skip', 'end', 'linkednext'}

    """Available keys for playback customization"""

    __slots__ = ('db', 'automations')

    def __init__(self,
                 db: dict[str, Any] = None,
                 automations: dict[str, Automation] = None):
        if db is None:
            db = {}
        else:
            assert not (db.keys() - self.playkeys), f"diff={db.keys() - self.playkeys}"
            if any(v is None for v in db.values()):
                raise ValueError(f"Values passed should not be None: "
                                 f"{[k for k, v in db.items() if v is None]}")

        self.db: dict[str, Any] = db
        """A dictionary holding the arguments explicitely specified"""

        self.automations: dict[str, Automation] | None = automations
        """A dict holding Automations
        
        There is a maximum of one Automation per parameter. A new Automation
        for a given parameter will replace the old one
        """

    def setArgs(self, **kws: float | str) -> None:
        """
        Set one or multiple values for the parameters passed to a preset

        Args:
            **kws: any keyword parameter passed to an instrument preset. They are
                not checked
        """
        args: dict | None = self.db.get('args')
        if args is None:
            self.db['args'] = kws
        else:
            args.update(kws)

    def _checkArgs(self) -> None:
        args = self.args
        if args:
            assert all(arg not in PlayArgs.playkeys for arg in args), f"{self=}"

    @property
    def args(self) -> dict | None:
        return self.db.get('args')

    def __bool__(self):
        return bool(self.db) or bool(self.automations)

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
        db = self.db
        return (db.get(k) for k in self.playkeys)

    def items(self) -> dict[str, Any]:
        """Like dict.items()"""
        db = self.db
        return {k: db.get(k) for k in self.playkeys}

    def get(self, key: str, default=None):
        """Like dict.get()"""
        if key not in self.playkeys:
            raise KeyError(f"Unknown key {key}. Possible keys are: {self.playkeys}")
        return self.db.get(key, default)

    def __getitem__(self, item: str):
        return self.db[item]

    def __setitem__(self, key: str, value) -> None:
        if key not in self.playkeys:
            raise KeyError(f'PlayArgs: unknown key "{key}", possible keys: {self.playkeys}')
        if value is None:
            del self.db[key]
        else:
            self.db[key] = value

    def linkedNext(self) -> bool:
        return self.db.get('glisstime') is not None

    def addAutomation(self,
                      param: str,
                      breakpoints: list[tuple[time_t | location_t, float]] | list[tuple[time_t|location_t, float, str]] | list[num_t],
                      interpolation='linear',
                      relative=True) -> None:
        breakpoints = Automation.normalizeBreakpoints(breakpoints, interpolation=interpolation)
        if self.automations is None:
            self.automations = {}
        self.automations[param] = Automation(param=param, breakpoints=breakpoints,
                                             relative=relative)

    @staticmethod
    def _updatedb(db: dict, other: dict) -> None:
        args = db.get('args')
        otherargs = other.get('args')
        db.update(other)
        if args:
            db['args'] = args if not otherargs else args | otherargs
        elif otherargs:
            db['args'] = otherargs

    def _updateAutomations(self, automations: dict[str, Automation]) -> None:
        if self.automations is None:
            self.automations = automations.copy()
        else:
            self.automations |= automations

    def updated(self, other: PlayArgs, automations=True) -> PlayArgs:
        """
        A copy of self overwritten by other

        Args:
            other: the playargs to update self with
            automations: if True, include automations in the update

        Returns:
            a copy of self updated by other

        """
        out = self.copy()
        PlayArgs._updatedb(out.db, other.db)
        if automations and other.automations:
            out._updateAutomations(other.automations)
        return out

    def copy(self) -> PlayArgs:
        """
        Returns a copy of self
        """
        if not self.db and not self.automations:
            return PlayArgs({})
        db = self.db.copy()
        if (args := self.db.get('args')) is not None:
            assert isinstance(args, dict)
            db['args'] = args.copy()
        return PlayArgs(db=db,
                        automations=_unique(self.automations, deep=False))

    def clone(self, **kws) -> PlayArgs:
        """
        Clone self with modifications

        Args:
            **kws: one of the possible playkeys

        Returns:
            the cloned PlayArgs

        """
        out = self.copy()
        PlayArgs._updatedb(out.db, kws)
        return out

    def __repr__(self):
        items = [f'{k}={v}' for k, v in self.db.items()]
        if self.automations:
            items.append(f'automations={self.automations}')
        args = ', '.join(items)
        return f"PlayArgs({args})"

    @staticmethod
    def makeDefault(conf: CoreConfig, copy=True) -> PlayArgs:
        """
        Create a PlayArgs with defaults from a CoreConfig

        Args:
            conf: a CoreConfig
            copy: if T

        Returns:
            the created PlayArgs

        """
        d = conf._makeDefaultPlayArgsDict(copy=copy)
        return PlayArgs(d)

        # d = dict(delay=0,
        #          chan=1,
        #          gain=conf['play.gain'],
        #          fade=conf['play.fade'],
        #          instr=conf['play.instr'],
        #          pitchinterpol=conf['play.pitchInterpolation'],
        #          fadeshape=conf['play.fadeShape'],
        #          priority=1,
        #          position=-1,
        #          sustain=0,
        #          transpose=0)
        # return PlayArgs(d)

    def fillWith(self, other: PlayArgs) -> None:
        """
        Fill any unset value in self with the value in other **inplace**

        Args:
            other: another PlayArgs

        """
        otherdb = other.db
        db = self.db
        for k, v in otherdb.items():
            if v is not None and k != 'args':
                db[k] = db.get(k, v)
        otherargs = otherdb.get('args')
        if otherargs:
            if args := db.get('args'):
                args.update(otherargs)
            else:
                db['args'] = otherargs.copy()

    def update(self, d: dict[str, Any]) -> None:
        PlayArgs._updatedb(self.db, d)

    def makeSynthAutomations(self,
                             scorestruct: ScoreStruct,
                             parentOffset: F,
                             ) -> list[SynthAutomation]:
        if not self.automations:
            return []
        return [automation.makeSynthAutomation(scorestruct=scorestruct, parentOffset=parentOffset)
                for automation in self.automations.values()]


def cropBreakpoints(bps: list[breakpoint_t], start: float, end: float
                    ) -> list[breakpoint_t]:
    """
    Crop the breakpoints at time t

    Args:
        bps: the breakpoints
        start: the time to start cropping
        end: the time to end cropping

    Returns:
        the cropped breakpoints
    """
    assert bps[0][0] == 0
    if start <= bps[0][0] and (end == 0 or end > bps[-1][0]):
        return bps
    newbps = []
    for i in range(len(bps)):
        bp = bps[i]
        bptime = bp[0]
        if bptime < start:
            continue
        elif bptime > start and i > 0:
            # there is part of a breakpoint before, need to interpolate
            bp2 = _interpolateBreakpoints(start, bps[i-1], bp)
            newbps.append(bp2)

        if bptime <= end:
            newbps.append(bp)
        else:
            bp2 = _interpolateBreakpoints(end, bps[i - 1], bp)
            newbps.append(bp2)
            break
    return newbps


def _interpolateBreakpoints(t: num_t, bp0: Sequence[num_t], bp1: Sequence[num_t]
                            ) -> list[float]:
    t0, t1 = bp0[0], bp1[0]
    assert t0 <= t <= t1, f"{t0=}, {t=}, {t1=}"
    delta = (t - t0) / (t1 - t0)
    bp = [float(t)]
    for v0, v1 in zip(bp0[1:], bp1[1:]):
        bp.append(float(v0 + (v1-v0)*delta))
    return bp


@dataclass
class _AutomationSegment:
    """
    Instances of this class are used to gather changes in dynamic parameters
    when merging multiple SynthEvents. Dynamic parameters are either
    builtin-in playback arguments, like position, or instrument defined
    parameters.

    A user never creates automation segments: these are created when
    multiple events are merged within a chain/voice. Each event within
    a chain/voice generates synth events; these are sorted into lines
    of subsequent synthevents, which are merged into one synthevent.
    Changes to pitch and amplitude are represented as breakpoints and
    modulations of any dynamic parameter are collected as automation
    segments
    """

    param: str
    """The parameter to automate (either a builtin synth param or an instr param)"""

    time: float
    """The time (in seconds), relative to the beginning of the event"""

    value: float
    """The new value of the parameter"""

    pretime: float | None = None
    """The previous time, relative to the beginning of the event"""

    prevalue: float | None = None
    """The previous value, None for a """

    kind: str = 'normal'
    """One of 'normal', 'arg' """


class SynthEvent:
    """
    Represents a standard event (a line of variable breakpoints)

    A User never creates a :class:`SynthEvent`: a :class:`SynthEvent` is
    created by a :class:`Note` or a :class:`Voice`. They are used internally
    to generate a set of events to be played/recorded by the playback engine.

    """
    __slots__ = ("bps", "delay", "chan", "fadein", "fadeout", "gain",
                 "instr", "pitchinterpol", "fadeshape", "args", "kws",
                 "priority", "position", "linkednext",
                 "numchans", "whenfinished", "properties", 'sustain',
                 'automationSegments', 'automations',
                 'initfunc', '_initdone')

    dynamicAttributes = (
        'position',
    )
    """Attributes which can change within merged events"""

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
                 bps: list[breakpoint_t],
                 instr: str,
                 delay: float = 0.0,
                 chan: int = 1,
                 fade: float | tuple[float, float] = 0,
                 gain: float = 1.0,
                 pitchinterpol: str = 'linear',
                 fadeshape: str = 'cos',
                 args: dict[str, float | str] = None,
                 priority: int = 1,
                 position: float = -1,
                 numchans: int = 2,
                 linkednext=False,
                 whenfinished: Callable = None,
                 properties: dict[str, Any] | None = None,
                 sustain: float = 0.,
                 initfunc: Callable[[SynthEvent, renderer.Renderer], None] = None,
                 # **kws
                 ):
        """
        bps (breakpoints): a seq of (delay, midi, amp, ...) of len >= 1.

        Args:
            bps: breakpoints, where each breakpoint is a tuple of (timeoffset, midi, amp,
            [...]).
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

        if pitchinterpol not in self.pitchinterpolToInt:
            raise ValueError(f"pitchinterpol should be one of {list(self.pitchinterpolToInt.keys())}, "
                             f"got {pitchinterpol}")

        if fadeshape not in self.fadeshapeToInt:
            raise ValueError(f"fadeshape should be one of {list(self.fadeshapeToInt.keys())}")

        if position < 0:
            position = 0 if numchans == 1 else 0.5

        delay = float(delay)

        if isinstance(fade, tuple):
            fadein, fadeout = fade
        else:
            fadein = fadeout = fade

        self.bps: list[breakpoint_t] = bps
        """breakpoints, where each breakpoint is a list of [timeoffset, midi, amp, [...]]"""

        dur = self.bps[-1][0] - self.bps[0][0]

        self.delay: float = delay
        """time delay - The effective time of bp[n] will be delay + bp[n][0]"""

        self.chan: int = chan
        """output channel"""

        self.gain: float = gain
        """a gain to be applied to this event"""

        self.fadein: float = fadein
        """fade in time"""

        self.fadeout: float = fadeout if dur < 0 else min(fadeout, dur)
        """fade out time"""

        self.instr: str = instr
        """Instrument preset used"""

        self.pitchinterpol: str = pitchinterpol
        """Pitch interpolation"""

        self.fadeshape: str = fadeshape
        """Shape of the fades"""

        self.priority: int = priority
        """Schedule priority (priorities start with 1)"""

        self.position: float = position
        """Panning position (between 0-1)"""

        self.args: dict[str, float | str] | None = args
        """Any parameters passed to the instrument. Can be None"""

        # self.kws: dict[str, float | str] | None = kws
        # """Ignored at the moment"""

        self.linkednext: bool = linkednext
        """Is this event linked to the next? 
        A linked synthevent is a tied note or a note with a glissando followed by 
        some continuation. In any case, the last breakpoint of this synthevent and the 
        first breakpoint of the following event should be equal for a two events to 
        be linked. NB: since we are dealing with floats, code should always check that
        the numbers are near instead of using ==
        """

        self.numchans: int = numchans
        """The number of signals produced by the event"""

        self.whenfinished: Callable | None = whenfinished
        """A function to call when this event has finished"""

        self.properties: dict[str, Any] | None = properties
        """User defined properties for an event"""

        self.sustain: float = sustain
        """Sustain time after the actual duration"""

        self.automations: dict[str, SynthAutomation] | None = None

        self.automationSegments: list[_AutomationSegment] | None = None
        """List of automation points
        
        These are created when multiple events are merged into one.
        The dynamic parameters of the subsequent events are 
        gathered as automation points."""

        self.initfunc = initfunc
        """A function called when the event is being scheduled. 
        It has the form (synthevent, renderer) -> None, where synthevent is 
        the event being rendered and renderer is the renderer performing the render 
        (either a maelzel.core.playback.RealtimeRenderer or a 
        maelzel.core.playback.OfflineRenderer). It can be used to initialize any 
        resources needed by the event (load/make tables, add includes, global code, etc)"""

        self._initdone = False

        self._consolidateDelay()

        if self.dur <= 0:
            raise ValueError(f"Duration of a synth event must be possitive: {self}")

    def getPreset(self) -> presetdef.PresetDef:
        return presetmanager.presetManager.getPreset(self.instr)

    def _ensureArgs(self) -> dict[str, float | str]:
        if self.args is None:
            self.args = {}
        return self.args

    def paramValue(self, param: str):
        instr = self.getInstr()
        param2 = instr.unaliasParam(param, param)
        if self.args and param2 in self.args:
            return self.args[param2]
        defaults = instr.paramDefaultValues()
        value = defaults.get(param)
        if value is None:
            raise KeyError(f"Unknown parameter '{param}', "
                           f"possible parameters: {defaults.keys()}")
        return value

    def getInstr(self) -> csoundengine.instr.Instr:
        return self.getPreset().getInstr()

    def initialize(self, renderer) -> None:
        if not self._initdone and self.initfunc:
            self.initfunc(self, renderer)

    def _applySustain(self) -> None:
        if self.linkednext and self.sustain:
            logger.debug(f"A linked event cannot have sustain ({self=}")
            return
        if self.sustain > 0:
            last = self.bps[-1]
            assert isinstance(last, list)
            bp = last.copy()
            bp[0] = last[0] + self.sustain
            self.bps.append(bp)
        elif self.sustain < 0:
            self.crop(self.dur + self.sustain)

    @property
    def start(self) -> float:
        return self.delay

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
                          fadeshape=self.fadeshape,
                          args=None if not self.args else self.args.copy(),
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

    def addAutomationsFromPlayArgs(self, playargs: PlayArgs, scorestruct: ScoreStruct) -> None:
        if not playargs.automations:
            return
        offset = scorestruct.timeToBeat(self.delay + self.bps[0][0])
        automations = playargs.makeSynthAutomations(scorestruct=scorestruct, parentOffset=offset)
        if self.automations is None:
            self.automations = {automation.param: automation
                                for automation in automations}
        else:
            for automation in automations:
                self.automations[automation.param] = automation

    @classmethod
    def fromPlayArgs(cls,
                     bps: list[breakpoint_t],
                     playargs: PlayArgs,
                     properties: dict[str, Any] | None = None,
                     **kws
                     ) -> SynthEvent:
        """
        Construct a SynthEvent from breakpoints and playargs

        .. note::

            This method does not transfer any automations from the
            playargs to the created SynthEvent. Automations can
            be transferred via :meth:`SynthEvent.addAutomationsFromPlayArgs`

        Args:
            bps: the breakpoints
            playargs: playargs
            properties: any properties passed to the constructor
            kws: any keyword accepted by SynthEvent

        Returns:
            a new SynthEvent
        """
        if not playargs.db:
            db = {}
        elif kws:
            db = playargs.db | kws
        else:
            db = playargs.db.copy()
        db.pop('transpose', None)
        linkednext = db.pop('linkednext', False) or db.get('glisstime') is not None
        return SynthEvent(bps=bps,
                          properties=properties,
                          linkednext=linkednext,
                          **db)

    def _consolidateDelay(self) -> None:
        delay0 = self.bps[0][0]
        if delay0 > 0:
            self.delay += delay0
            for bp in self.bps:
                bp[0] -= delay0
        assert self.bps[0][0] == 0

    def _applyTimeFactor(self, timefactor: float) -> None:
        if timefactor == 1:
            return
        self.delay *= timefactor
        for bp in self.bps:
            bp[0] *= timefactor

    def timeShifted(self, offset: float) -> SynthEvent:
        """A clone of this event, shifted in time by the given offset"""
        return self.clone(delay=self.delay+offset)

    def crop(self, dur: float) -> None:
        self.bps = cropBreakpoints(self.bps, 0, dur)

    def cropped(self, start: float, end: float) -> SynthEvent:
        """
        Return a cropped version of this SynthEvent
        """
        start = max(start - self.delay, 0)
        end -= self.delay
        if end - start <= 0:
            raise ValueError(f"Invalid crop: the end time ({end}) should lie before "
                             f"the start time ({start})")
        out = []
        for i in range(len(self.bps)):
            bp: list[float] = self.bps[i]
            t = bp[0]
            if t < start:
                if i < len(self.bps)-1 and start < self.bps[i + 1][0]:
                    bpi = _interpolateBreakpoints(start, bp, self.bps[i + 1])
                    out.append(bpi)
            elif start <= t < end:
                out.append(bp.copy())
                if i < len(self.bps) - 1 and end <= self.bps[i+1][0]:
                    bp2 = _interpolateBreakpoints(end, bp, self.bps[i+1])
                    out.append(bp2)
            elif t > end:
                break
        return self.clone(bps=out)

    def breakpointSize(self) -> int:
        """ Returns the number of breakpoints in this SynthEvent """
        return len(self.bps[0])

    def _repr_html_(self) -> str:
        rows = [[f"{bp[0] + self.delay:.3f}", f"{bp[0]:.3f}"] + ["%.6g" % x for x in bp[1:]] for bp in self.bps]
        headers = ["Abs time", "0. Rel. time", "1. Pitch", "2. Amp"]
        l = len(self.bps[0])
        if l > 3:
            headers += [str(i) for i in range(4, l+1)]
        htmltab = emlib.misc.html_table(rows, headers=headers)
        return f"{self._reprHeader()}<br>" + htmltab

    def _reprHeader(self) -> str:
        info = [f"delay={float(self.delay):.3g}, dur={self.dur:.3g}, "
                f"instr={self.instr}, "
                f"gain={self.gain:.4g}, chan={self.chan}"
                f", fade=({self.fadein}, {self.fadeout})"]
        if self.linkednext:
            info.append('linkednext=True')
        if self.args:
            info.append(f"args={self.args}")
        #if self.kws:
        #    info.append(f"kws={self.kws}")
        if self.sustain:
            info.append(f"sustain={self.sustain}")
        if self.position is not None and self.position >= 0:
            info.append(f"position={self.position}")
        if self.automationSegments:
            info.append(f'automationSegments={self.automationSegments}')
        if self.automations:
            info.append(f'automations={self.automations}')
        infostr = ", ".join(info)
        return f"SynthEvent({infostr})"

    def __repr__(self) -> str:
        lines = [self._reprHeader()]

        def bpline(bp):
            rest = " ".join(("%.6g" % b).ljust(8) if isinstance(b, float) else str(b) for b in bp[1:])
            return f"{float(bp[0]):7.6g}s: {rest}"

        for i, bp in enumerate(self.bps):
            if i == 0:
                lines.append(f"bps {bpline(bp)}")
            else:
                lines.append(f"    {bpline(bp)}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def cropEvents(events: list[SynthEvent], skip=0., end=math.inf
                   ) -> list[SynthEvent]:
        """
        Crop the events at the given time slice (staticmethod)

        Removes any event / part of an event outside the time slice start:end

        Args:
            events: the events to crop
            skip: start of the time slice (None will only crop at the end)
            end: end of the time slice (None will only crop at the beginning)

        Returns:
            the cropped events

        """
        return [event.cropped(skip, end) for event in events
                if mathlib.hasintersect(skip, end, event.delay, event.end)]

    @staticmethod
    def plotEvents(events: list[SynthEvent], axes: Axes = None, notenames=False
                   ) -> Axes:
        """
        Plot all given events within the same axes (static method)

        Args:
            events: the events to plot
            axes: the matplotlib Axes to use, if given
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
        import matplotlib.ticker

        if axes is None:
            # f: plt.Figure = plt.figure(figsize=figsize)
            f = plt.figure()
            axes: Axes = f.add_subplot(1, 1, 1)

        for event in events:
            event.plot(axes=axes, notenames=False)

        axes.grid()

        if notenames:
            noteformatter = matplotlib.ticker.FuncFormatter(lambda s, y: f'{str(s).ljust(3)}: {pt.m2n(s)}')
            axes.yaxis.set_major_formatter(noteformatter)
            axes.tick_params(axis='y', labelsize=8)

        return axes

    @staticmethod
    def mergeEvents(events: Sequence[SynthEvent]) -> SynthEvent:
        """
        Static method to merge events which are linked (tied, gliss)

        Args:
            events: the events to merge

        Returns:
            the merged event

        """
        return mergeEvents(events)

    def _flatBreakpoints(self) -> list[float]:
        out = []
        for bp in self.bps:
            out.extend(bp)
        return out

    def _resolveParams(self: SynthEvent,
                       instr: csoundengine.instr.Instr
                       ) -> tuple[list[float | str], dict[str, float]]:
        """
        Resolves the values for pfields and dynamic params

        Does the same as _resolveParamsGeneric, but ~8 times faster, at
        the cost of having the arguments hard-coded

        Args:
            instr: the actual Instr, corresponding to the name in self.instr

        Returns:
            a tuple (pfields5, dynargs), where pfields5 are the pfields starting
            at p5 (a list of float|str) and dynargs is a dict of dynamic parameters
        """
        numNamedPfields = instr.numPfields()

        # if not self.linkednext and self.sustain > 0:
        if self.sustain > 0:
            self._applySustain()

        # |kpos, kgain, idataidx_, inumbps, ibplen, ichan, ifadein, ifadeout, ipchintrp_, ifadekind|
        if csoundengine.config['dynamic_pfields']:
            # pfields are also used for dynamic (k) arguments
            pfields5: list[float | str] = [
                self.position,          # kpos
                self.gain,              # kgain
                numNamedPfields + 5,    # idataidx_
                len(self.bps),          # inumbps
                self.breakpointSize(),  # ibplen
                self.chan,              # ichan
                self.fadein,            # ifadein
                self.fadeout,           # ifadeout
                SynthEvent.pitchinterpolToInt[self.pitchinterpol],  # ipchintrp_
                SynthEvent.fadeshapeToInt[self.fadeshape]  # ifadekind
            ]
            if self.args:
                if any(not isinstance(value, (int, float, str)) for value in self.args.values()):
                    raise TypeError(f"Invalid args: {self.args.values()}")
                dynargs = _cast(dict[str, float], self.args)
            else:
                dynargs = {}
        else:
            pfields5: list[float | str] = [
                numNamedPfields + 5,    # idataidx_
                len(self.bps),          # inumbps
                self.breakpointSize(),  # ibplen
                self.chan,              # ichan
                self.fadein,            # ifadein
                self.fadeout,           # ifadeout
                SynthEvent.pitchinterpolToInt[self.pitchinterpol],  # ipchintrp_
                SynthEvent.fadeshapeToInt[self.fadeshape]           # ifadekind
            ]
            dynargs: dict[str, float] = {'kpos': self.position, 'kgain': self.gain}
            if self.args:
                assert all(isinstance(value, float) for value in self.args.values())
                dynargs |= self.args
        instrdefaults = instr.defaultPfieldValues()
        pfields5.extend(instrdefaults[len(pfields5):])
        pfields5.extend(self._flatBreakpoints())
        return pfields5, dynargs

    def _resolveParamsGeneric(self: SynthEvent,
                              instr: csoundengine.instr.Instr):
        """
        Resolves the values for pfields and dynamic params

        This is not used. It is here as a reference, since _resolveParams is used
        instead, which is faster.

        Args:
            instr: the actual Instr, corresponding to the name in self.instr

        Returns:
            a tuple (pfields5, dynargs), where pfields5 are the pfields starting
            at p5 (a list of float|str) and dynargs is a dict of dynamic parameters
        """
        numNamedPfields = instr.numPfields()

        if not self.linkednext and self.sustain > 0:
            self._applySustain()

        #     |kpos, idataidx_, inumbps, ibplen, igain, ichan, ifadein, ifadeout, ipchintrp_, ifadekind|
        pfields = {
            'idataidx_': numNamedPfields + 5,
            'inumbps': len(self.bps),
            'ibplen': self.breakpointSize(),
            'igain': self.gain,
            'ichan': self.chan,
            'ifadein': self.fadein,
            'ifadeout': self.fadeout,
            'ipchintrp_': SynthEvent.pitchinterpolToInt[self.pitchinterpol],
            'ifadekind': SynthEvent.fadeshapeToInt[self.fadeshape]
        }
        dynargs = {'kpos': self.position, 'kgain': self.gain}
        if self.args:
            dynargs |= self.args
        pfields5, dynargs = instr.parseSchedArgs(args=pfields, kws=dynargs)
        pfields5.extend(self._flatBreakpoints())
        return pfields5, dynargs

    # def resolvePfields(self: SynthEvent,
    #                    instr: csoundengine.instr.Instr
    #                    ) -> list[float | str]:
    #     """
    #     Returns pfields, **beginning with p2**.
    #
    #     ==== =====  ============================================
    #     idx  parg    desc
    #     ==== =====  ============================================
    #     0    2       delay
    #     1    3       duration
    #     2    4       tabnum
    #     3    5       bpsoffset (pfield index, starting with 1)
    #     4    6       bpsrows
    #     5    7       bpscols
    #     6    8       gain
    #     7    9       chan
    #     8    0       position
    #     9    1       fade0
    #     0    2       fade1
    #     1    3       pitchinterpol
    #     2    4       fadeshape
    #     .
    #     .            reserved space for user pargs
    #     .
    #     ==== =====  ============================================
    #
    #     breakpoint data is appended
    #
    #     """
    #     if not self.linkednext and self.sustain > 0:
    #         self._applySustain()
    #
    #     pitchInterpolMethod = SynthEvent.pitchinterpolToInt[self.pitchinterpol]
    #     fadeshape = SynthEvent.fadeshapeToInt[self.fadeshape]
    #     # if no userpargs, bpsoffset is 15
    #     numPargs5 = len(instr.pfieldIndexToName)
    #     bpsrows = len(self.bps)
    #     bpscols = self.breakpointSize()
    #     pfields = [
    #         float(self.delay),
    #         float(self.dur),
    #         0,  # table index, to be filled later
    #     ]
    #
    #     pfields5 = [
    #         0,            # p5, idx: 4 (bpsoffset)
    #         bpsrows,
    #         bpscols,
    #         self.gain,
    #         self.chan,
    #         self.position,
    #         self.fadein,
    #         self.fadeout,
    #         pitchInterpolMethod,
    #         fadeshape
    #     ]
    #     numBuiltinPargs = len(pfields5)  # 10
    #     numUserArgs = numPargs5 - numBuiltinPargs
    #     bpsoffset = 15 + numUserArgs
    #     pfields5[0] = bpsoffset
    #
    #     # if self._namedArgsMethod == 'pargs' and numUserArgs > 0:
    #     #     pfields5 = instr.pfieldsTranslate(args=pfields5, kws=self.args)
    #     pfields.extend(pfields5)
    #     for bp in self.bps:
    #         pfields.extend(bp)
    #     pfields = [x if isinstance(x, str) else float(x) for x in pfields]
    #     if len(pfields) > _MAX_NUM_PFIELDS:
    #         logger.error(f"This SynthEvent has too many pfields: {len(pfields)}")
    #     return pfields

    def plot(self, axes: Axes = None, notenames=False) -> Axes:
        """
        Plot the trajectory of this synthevent

        Args:
            axes: a matplotlib.pyplot.Axes, will be used if given
            notenames: if True, use notenames for the y axes

        Returns:
            the axes used
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker
        ownaxes = axes is None
        if axes is None:
            # f: plt.Figure = plt.figure(figsize=figsize)
            f = plt.figure()
            axes: Axes = f.add_subplot(1, 1, 1)
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


def mergeEvents(events: Sequence[SynthEvent]) -> SynthEvent:
    """
    Merge linked events

    Two events are linked if the first event has its `.linkednext` attribute set as True
    and the last breakpoint of the first event is equal to the first breakpoint of the
    second. This is used within a Chain or Voice to join the playback events of
    multiple chords/notes to single synthevents.

    Since all breakpoints are merged into one event, any values regarding
    dynamic parameters (position, instr parameters, …) would be lost. These
    are kept as automation points.

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

    mergedevent = firstevent.clone(bps=bps,
                                   linkednext=lastevent.linkednext,
                                   sustain=lastevent.sustain)
    restevents = events[1:]
    if mergedevent.args is None and any(ev.args for ev in restevents):
        mergedevent.args = {}
    argstate = mergedevent.args if mergedevent.args is not None else {}
    lastoffset = 0.
    automationPoints = []
    state = {attr: getattr(firstevent, attr) for attr in SynthEvent.dynamicAttributes}
    for event in restevents:
        offset = event.delay - mergedevent.delay
        if event.args is not None and event.args:
            assert event.instr
            instr = presetmanager.presetManager.getInstr(event.instr)
            dynparams = instr.dynamicParams()
            for k, v in event.args.items():
                if k in dynparams:
                    assert isinstance(v, (float, int)), f"Expected a float/int, got {v}"
                    automation = _AutomationSegment(param=k,
                                                    time=offset,
                                                    value=float(v),
                                                    prevalue=_cast(float, argstate.get(k, dynparams[k])),
                                                    pretime=lastoffset,
                                                    kind='arg')
                    automationPoints.append(automation)
            argstate = _mergeOptionalDicts(mergedevent.args, event.args)
        for attr in SynthEvent.dynamicAttributes:
            value = getattr(event, attr, None)
            prevalue = state.get(attr)
            if value is not None and value != prevalue:
                automation = _AutomationSegment(param=attr,
                                                time=offset,
                                                value=value,
                                                prevalue=prevalue,
                                                pretime=lastoffset,
                                                kind='normal')
                automationPoints.append(automation)
                state[attr] = value

        lastoffset = offset
    mergedevent.automationSegments = automationPoints
    return mergedevent


def _mergeOptionalDicts(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict:
    if a and b:
        return a | b
    elif a:
        return a
    elif b:
        return b
    return {}


