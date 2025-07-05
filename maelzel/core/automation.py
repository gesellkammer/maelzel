from __future__ import annotations
from dataclasses import dataclass
from emlib import iterlib
from maelzel.common import F
from maelzel.core.workspace import Workspace

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
    from maelzel.common import location_t, num_t
    from maelzel.scorestruct import ScoreStruct
    import numpy as np


class Automation:
    """
    Represent an abstract automation

    When using relative time, the time is relative to the parent. Since the automation
    does not hold a reference to the parent, the parent needs to be given to convert
    the automation to a synthesis automation. In this mode, a time given as a location
    (measureindex, beat), the location is a delta, so if the parent has a given location
    (parentidx, parentbeat), the absolute location will be (parentidx + measureindex,
    parentbeat + beat).

    """
    def __init__(self, param: str, breakpoints: list[tuple[F | location_t, float, str]], relative=True):
        self.param = param
        """The parameter to automate"""

        self.breakpoints = breakpoints
        """A list of breakpoints

        Each breakpoint has the form (time, value, interpolation)
        where time is the time in quarternotes, value is the value of
        the param at the time and interpolation is one of 'linear', 'cos',
        expon(x), etc.
        """

        self.relative = relative
        """Are the breakpoints relative to the parent?"""

    def __repr__(self) -> str:
        return f"Automation({self.param}, breakpoints={self.breakpoints})"

    def _abstime(self, t: F | location_t, parentOffset: F, scorestruct: ScoreStruct) -> F:
        if self.relative:
            parentMeas, parentBeat = scorestruct.beatToLocation(parentOffset)
            if isinstance(t, tuple):
                t0abs = scorestruct.locationToTime(parentMeas + t[0], parentBeat + t[1])
            else:
                t0abs = scorestruct.time(parentOffset + t)
            return t0abs
        else:
            return scorestruct.time(t)

    def absTimeRange(self, parentOffset: F, scorestruct: ScoreStruct|None = None
                     ) -> tuple[F, F]:
        """
        Returns the absolute start and end of this Automation, in seconds

        Args:
            parentOffset: the offset (in quarterbeats) of the parent of this automation
            scorestruct: the struct to use to convert quarterbeats to seconds

        Returns:
            a tuple (start, end), both expressed in seconds
        """
        if scorestruct is None:
            scorestruct = Workspace.active.scorestruct
        start = self._abstime(self.breakpoints[0][0], parentOffset=parentOffset, scorestruct=scorestruct)
        end = self._abstime(self.breakpoints[-1][0], parentOffset=parentOffset, scorestruct=scorestruct)
        return start, end

    def absoluteTimes(self, parentOffset: F, scorestruct: ScoreStruct|None = None
                      ) -> tuple[list[F], F]:
        """
        The times of all the breakpoints in this automations as seconds

        Args:
            parentOffset: the offset (in quarterbeats) of the parent of this automation
            scorestruct: the struct to use to convert quarterbeats to seconds

        Returns:
            a tuple (times: list[F], delay: F), all expressed in seconds
        """
        if scorestruct is None:
            scorestruct = Workspace.active.scorestruct

        times: list[F] = []
        parentAbstime = scorestruct.beatToTime(parentOffset)

        if self.relative:
            parentMeas, parentBeat = scorestruct.beatToLocation(parentOffset)
            t0 = self.breakpoints[0][0]
            if isinstance(t0, tuple):
                t0abs = scorestruct.locationToTime(parentMeas + t0[0], parentBeat + t0[1])
            else:
                t0abs = scorestruct.time(parentOffset + t0)
            delay = t0abs - parentAbstime
            for bp in self.breakpoints:
                t = bp[0]
                if isinstance(t, tuple):
                    abstime = scorestruct.locationToTime(parentMeas + t[0], parentBeat + t[1])
                else:
                    abstime = scorestruct.beatToTime(parentOffset + t)
                times.append(abstime - delay)
        else:
            t0 = scorestruct.time(self.breakpoints[0][0])
            delay = t0 - parentAbstime
            for bp in self.breakpoints:
                abstime = scorestruct.time(bp[0])
                times.append(abstime - t0)
        return times, delay

    def makeSynthAutomation(self,
                            scorestruct: ScoreStruct,
                            parentOffset: F,
                            ) -> SynthAutomation:
        """
        Convert this Automation to a SynthesisAutomationEvent

        Args:
            scorestruct: the active scorestruct
            parentOffset: the absolute offset of the parent, in quarternotes

        Returns:
            the corresponding synthesis automation
        """
        abstimes, delay = self.absoluteTimes(scorestruct=scorestruct, parentOffset=parentOffset)
        data = []
        for abstime, bp in zip(abstimes, self.breakpoints):
            data.append(float(abstime))
            data.append(bp[1])
        return SynthAutomation(param=self.param, delay=float(delay), data=data)

    @staticmethod
    def normalizeBreakpoints(breakpoints: list[tuple] | list[num_t],
                             interpolation='linear'
                             ) -> list[tuple[F | location_t, float, str]]:
        """
        Normalize breakpoints, ensuring that all have the form (time, value, interpolation)

        Args:
            breakpoints: a list of tuples of the form (time, value) or
                (time, value, interpolation) or a flat list of the form
                [time0, value0, time1, value1, ...]. In this case all breakpoints
                use the interpolation given as fallback
            interpolation: the default/fallback interpolation

        Returns:
            a list of tuples of the form (time, value, interpolation). The interpolation
            value is valid for the interpolation between the breakpoint and the next
            breakpoint. The last interpolation is irrelevant.

        """
        if not interpolation:
            interpolation = 'linear'

        if not isinstance(breakpoints[0], tuple):
            # a flat list
            assert len(breakpoints) % 2 == 0, "A flat list of breakpoints needs to be even"
            assert all(isinstance(x, (int, float, F)) for x in breakpoints)
            breakpoints = [(breakpoints[i], breakpoints[i+1]) for i in range(0, len(breakpoints), 2)]
            # breakpoints = list(iterlib.window(breakpoints, 2, 2))

        normalized: list[tuple[F | location_t, float, str]] = []
        for bp in breakpoints:
            assert isinstance(bp, tuple)
            bplen = len(bp)
            if bplen == 3:
                normalized.append(bp)  # type: ignore
            elif bplen == 2:
                normalized.append(bp + (interpolation,))  # type: ignore
            else:
                raise ValueError(f"A breakpoint can have 2 or 3 items, got {bp}")
        return normalized


@dataclass
class SynthAutomation:
    """
    Represents an automation in the realm of synthesis

    It is used to automate a running synthesis operation. In this
    context all times are given in seconds since we are past the
    abstract time

    TODO: allow per breakpoint interpolation kind
    """
    param: str
    """The parameter to automate"""

    data: list[float] | np.ndarray
    """The automation data

    A flat list of (time, value) pairs. For a single event, the data should be [0, value]"""

    delay: float = 0.
    """An additional delay to all time values"""

    interpolation: str = 'linear'
    """One of linear, cos, expon(XX), ... (see opcode interp1d)"""

    overtake: bool = False
    """Overtake current value"""

    token: int | None = None
    """Corresponding synth token for future synths"""

    def copy(self) -> SynthAutomation:
        return SynthAutomation(param=self.param,
                               data=self.data.copy(),
                               delay=self.delay,
                               interpolation=self.interpolation,
                               token=self.token,
                               overtake=self.overtake)

    @property
    def start(self) -> float:
        return self.delay + self.data[0]

    @property
    def end(self) -> float:
        return self.delay + self.data[-2]

    def cropped(self, 
                start: float, 
                end: float, 
                overtake: bool | None = None
                ) -> SynthAutomation | None:
        """
        Copy of this synth automation cropped between (start, end)

        Args:
            start: absolute time to start cropping
            end: absolute time to end cropping
            overtake: if True, the cropped automation should overtake
                from the running value

        Returns:
            a copy of self cropped between start and end
        """
        bps = list(iterlib.window(self.data, 2, 2))
        from maelzel.core import _tools
        bps = _tools.cropBreakpoints(bps, start - self.delay, end - self.delay)  # type: ignore
        data = [float(bp) for bp in _flattenBreakpoints(bps)]
        return SynthAutomation(param=self.param,
                               data=data,
                               delay=start,
                               interpolation=self.interpolation,
                               token=self.token,
                               overtake=overtake if overtake is not None else self.overtake)


def _flattenBreakpoints(bps: list[Sequence[num_t]]) -> list[num_t]:
    """
    Converts breakpoints in the form [(t0, x0, ...), (t1, x1, ...), ...]
    to [t0, x0, ..., t1, x1, ..., ...]

    Example
    ~~~~~~~

        >>> _flattenBreakpoints([(0, 1), (2, 3)])
        [0, 1, 2, 3]
    """
    out = []
    for bp in bps:
        out.extend(bp)
    return out
