from __future__ import annotations
from dataclasses import dataclass, fields
import pitchtools as pt
from typing import Callable

__all__ = (
    'Breakpoint',
    'simplifyBreakpoints',
    'simplifyBreakpointsByDensity',
    'TranscribeOptions'
)


@dataclass
class TranscribeOptions:

    addGliss: bool = True
    """if True, add a gliss. symbol between parts of a same note group"""

    addAccents: bool = True
    """add an accent symbol to breakpoints with a detected accent"""

    addSlurs: bool = True
    """add a slur encompasing notes within a group"""

    unvoicedNotehead: str = 'x'
    """Notehead used for unpitched notes or an empty string to leave such notes unmodified"""

    unvoicedPitch: str | int = "5C"
    """
    The pitch used for note groups where no pitch was detected. For
    unpitched breakpoints within a group where other pitched breakpoints were
    found the next pitched found will be used.
    """

    unvoicedMinAmpDb: float = -80.
    """
    Breakpoints within an unvoiced group with amp. less than this will not be transcribed.
    """

    a4: float = 442.
    """Reference frequency"""

    simplify: float = 0.
    """
    Simplify breakpoints. 0 disables simplification
    """

    maxDensity: float = 0.
    """
    max. breakpoint density. 0 disables simplification 
    """



    def __post_init__(self):
        assert isinstance(self.addGliss, bool)
        assert isinstance(self.addAccents, bool)
        assert isinstance(self.unvoicedNotehead, str)
        assert isinstance(self.unvoicedPitch, (str, int))
        assert isinstance(self.unvoicedMinAmpDb, (int, float))
        assert isinstance(self.a4, (int, float)) and 432 < self.a4 < 460


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

    def fields(self) -> list[str]:
        return [field.name for field in fields(self)]

    def setProperty(self, key: str, value) -> None:
        if self.properties is None:
            self.properties = {key: value}
        else:
            self.properties[key] = value

    def getProperty(self, key, default=None):
        if not self.properties:
            return default
        return self.properties.get(key, default)


def simplifyBreakpointsByDensity(breakpoints: list[Breakpoint],
                                 maxdensity=0.05,
                                 pitchconv: pt.PitchConverter | None = None
                                 ) -> list[Breakpoint]:
    """
    Similar to simplifyBreakpoints but optimizes the max. density

    Args:
        breakpoints: the breakpoints to simplify
        maxdensity: the breakpoints need to be simplified so that the density
            (the number of breakpoints per second) does not exceed this value

    Returns:
        the simplified breakpoints. Notice that an group with <= 2 breakpoints will
        be returned as is

    """
    if len(breakpoints) <= 2:
        return breakpoints

    groupdur = breakpoints[-1].time - breakpoints[0].time
    if len(breakpoints) / groupdur < maxdensity:
        return breakpoints

    if pitchconv is None:
        pitchconv = pt.PitchConverter.default

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
    threshold = float(res['x'])
    return simplifyBreakpoints(breakpoints, param=threshold, pitchconv=pitchconv)


def simplifyBreakpoints(breakpoints: list[Breakpoint],
                        method='visvalingam',
                        param=0.1,
                        pitchconv: pt.PitchConverter | None = None
                        ) -> list[Breakpoint]:
    # TODO: instead of simplifying only based on pitch construct a vector using
    #       both pitch and amplitude (or other extra features) and simplify on that
    #       It must be reduced to one dimension
    #       example: feature = sqrt(b.pitch**2 + b.amp**2)
    if len(breakpoints) <= 2:
        return breakpoints

    if pitchconv is None:
        pitchconv = pt.PitchConverter.default

    points = [(b.time, pitchconv.f2m(b.freq)) for b in breakpoints]
    if method == 'visvalingam':
        import visvalingamwyatt
        simplifier = visvalingamwyatt.Simplifier(points)
        simplified = simplifier.simplify(threshold=param)
    else:
        raise ValueError(f"Method {method} not supported")

    def matchBreakpoint(t: float, breakpoints: list[Breakpoint], eps=1e-10) -> Breakpoint:
        bp = next((bp for bp in breakpoints if abs(bp.time - t) < eps), None)
        if bp is None:
            raise ValueError(f"Breakpoint not found for t = {t}")
        return bp

    return [matchBreakpoint(t, breakpoints) for t, p in simplified]


