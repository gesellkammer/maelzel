from __future__ import annotations
from dataclasses import dataclass, fields, astuple
import pitchtools as pt
from emlib import iterlib
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes


__all__ = (
    'Breakpoint',
    'BreakpointGroup',
    'simplifyBreakpoints',
    'simplifyBreakpointsByDensity',
)


@dataclass(unsafe_hash=True)
class Breakpoint:
    time: float
    "The time of this breakpoint"

    freq: float
    "Frequency of this breakpoint"

    amp: float
    "amplitude (0-1)"

    voiced: bool
    "is the sound voiced?"

    linked: bool
    "is this breakpoint linked to the next? (this should be false for any offset)"

    onsetStrength: float = 0.
    "the onset strength of this breakpoint"

    freqConfidence: float = 1.
    "The confidence of the frequency value (0-1)"

    kind: str = ''     # one of '', 'onset', 'offset'
    "the breakpoint type, one of 'onset', 'offset' or '' for a contiuation breakpoint"

    isaccent: bool = False
    "is this breakpoint an accent, indicating a sudden change within its context?"

    duration: float = 0.
    "gap between this breakpoint and the next"

    properties: dict | None = None

    def __post_init__(self):
        assert self.freq >= 0

    @classmethod
    def fields(cls) -> list[str]:
        return [field.name for field in fields(cls)]

    def setProperty(self, key: str, value) -> None:
        if self.properties is None:
            self.properties = {key: value}
        else:
            self.properties[key] = value

    def getProperty(self, key, default=None):
        if not self.properties:
            return default
        return self.properties.get(key, default)


def simplifyBreakpointsByDensity(breakpoints: list[Breakpoint] | BreakpointGroup,
                                 maxdensity=0.05,
                                 pitchconv: pt.PitchConverter | None = None
                                 ) -> list[Breakpoint]:
    """
    Similar to simplifyBreakpoints but optimizes the max. density

    Args:
        breakpoints: the breakpoints to simplify
        maxdensity: the breakpoints need to be simplified so that the density
            (the number of breakpoints per second) does not exceed this value
        pitchconv: a pitch converter. Use None to use default

    Returns:
        the simplified breakpoints. Notice that an group with <= 2 breakpoints will
        be returned as is

    """
    if isinstance(breakpoints, BreakpointGroup):
        breakpoints = breakpoints.breakpoints

    if len(breakpoints) <= 2:
        return breakpoints

    groupdur = breakpoints[-1].time - breakpoints[0].time
    if len(breakpoints) / groupdur < maxdensity:
        return breakpoints

    if pitchconv is None:
        pitchconv = pt.PitchConverter.default()

    import visvalingamwyatt
    from scipy import optimize

    points = [(b.time, pitchconv.f2m(b.freq)) for b in breakpoints]
    simplified = visvalingamwyatt.Simplifier(points)

    def func(threshold) -> float:
        simplifiedpoints = simplified.simplify(threshold=threshold)
        return len(simplifiedpoints) / groupdur

    res = optimize.minimize_scalar(lambda thresh: abs(func(thresh) - maxdensity),
                                   bracket=(0.0001, 0.99),
                                   tol=0.1)
    threshold = float(res['x'])  # type: ignore
    return simplifyBreakpoints(breakpoints, param=threshold, pitchconv=pitchconv)


def simplifyBreakpoints(breakpoints: list[Breakpoint] | BreakpointGroup,
                        method='visvalingam',
                        param=0.1,
                        pitchconv: pt.PitchConverter | None = None
                        ) -> list[Breakpoint]:
    # TODO: instead of simplifying only based on pitch construct a vector using
    #       both pitch and amplitude (or other extra features) and simplify on that
    #       It must be reduced to one dimension
    #       example: feature = sqrt(b.pitch**2 + b.amp**2)

    if isinstance(breakpoints, BreakpointGroup):
        breakpoints = breakpoints.breakpoints
    elif not isinstance(breakpoints, list):
        raise TypeError(f"Expected list or BreakpointGroup, got {type(breakpoints)}")

    if len(breakpoints) <= 2:
        return breakpoints

    if pitchconv is None:
        pitchconv = pt.PitchConverter.default()

    points = [(b.time, pitchconv.f2m(b.freq)) for b in breakpoints]
    if method == 'visvalingam':
        import visvalingamwyatt
        simplifier = visvalingamwyatt.Simplifier(points)
        simplified = simplifier.simplify(threshold=param)
    else:
        raise ValueError(f"Method {method} not supported")

    def matchBreakpoint(t: float, breakpoints: list[Breakpoint] | BreakpointGroup, eps=1e-10
                        ) -> Breakpoint:
        bp = next((bp for bp in breakpoints if abs(bp.time - t) < eps), None)
        if bp is None:
            raise ValueError(f"Breakpoint not found for t = {t}")
        return bp

    return [matchBreakpoint(t, breakpoints) for t, p in simplified]


class BreakpointGroup:
    """
    A list of breakpoints representing a note

    Args:
        breakpoint: a list of breakpoints
    """
    def __init__(self, breakpoints: list[Breakpoint]):
        self.breakpoints = breakpoints

    def __iter__(self) -> Iterator[Breakpoint]:
        return iter(self.breakpoints)

    def __len__(self) -> int:
        return len(self.breakpoints)

    def __getitem__(self, item):
        return self.breakpoints.__getitem__(item)

    def __repr__(self):
        return (f"Group(start={self.start():.6g}, end={self.end():.6g}, meanfreq={self.meanfreq():.5g}, "
                f"breakpoints={self.breakpoints}")

    def start(self) -> float:
        """Start time of this group"""
        return self.breakpoints[0].time

    def end(self) -> float:
        """End time of this group"""
        return self.breakpoints[-1].time

    def duration(self) -> float:
        """The duration of this group"""
        return self.breakpoints[-1].time - self.breakpoints[0].time

    def meanfreq(self, weighted=True) -> float:
        """Mean frequency of this group

        Args:
            weighted: weight the frequency by the breakpoints energy

        Returns:
              the average frequency
        """
        if len(self.breakpoints) == 1:
            return self.breakpoints[0].freq
        elif not weighted:
            return sum(b.freq for b in self.breakpoints) / len(self.breakpoints)
        else:
            weights, freqs = 0, 0
            for b0, b1 in iterlib.pairwise(self.breakpoints):
                dur = b1.time - b0.time
                weight = (b0.amp + b1.amp) / 2 * dur
                freqs += b0.freq * weight
                weights += weight
            return freqs / weights

    def meanamp(self, weighted=True) -> float:
        """
        Average amplitude of this group

        Args:
            weighted: weight the amplitude by the duration of the breakpoint

        Returns:
            the average amplitude, optionally weighted
        """
        if not weighted:
            return sum(b.amp for b in self.breakpoints)
        else:
            weights, amps = 0., 0.
            for b0, b1 in iterlib.pairwise(self.breakpoints):
                weight = b0.amp * (b1.time - b0.time)
                amps += b0.amp * weight
                weights += weight
            return amps / weights

    def times(self) -> list[float]:
        return [bp.time for bp in self.breakpoints]

    def freqs(self) -> list[float]:
        return [bp.freq for bp in self.breakpoints]

    def _repr_html_(self):
        import tabulate
        columnnames = Breakpoint.fields()
        rows = [astuple(bp) for bp in self.breakpoints]
        html = tabulate.tabulate(rows, tablefmt='html', headers=columnnames, floatfmt=".4f")
        return html

    def plot(self, ax: Axes, spanAlpha=0.2, linewidth=2, onsetAlpha=0.4, spanColor='red') -> None:
        """
        Plot this group

        Args:
            ax: the axes to plot onto
            spanColor: color used for axvspan and onset marks
            spanAlpha: alpha used for axvspan used to mark onset-offset regions
            linewidth: line width for the breakpoints
            onsetAlpha: alpha for onset marks

        """
        times = self.times()
        freqs = self.freqs()
        ax.plot(times, freqs, linewidth=linewidth)
        t0 = self.start()
        if len(self) > 1:
            t1 = self.end()
            ax.axvspan(t0, t1, alpha=spanAlpha, color=spanColor)
        ax.axvline(t0, color=spanColor, alpha=onsetAlpha)
        ax.plot(times, freqs)
