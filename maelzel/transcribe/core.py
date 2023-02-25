from __future__ import annotations
from dataclasses import dataclass, fields


__all__ = (
    'Breakpoint'
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

    strength: float = 0.
    "the onset strength of this breakpoint"

    strengthPercentile: float = 0.
    "the onset strength in terms of its percentile within the entire analysis"

    freqConfidence: float = 1.
    "The confidence of the frequency value (0-1)"

    ampPercentile: float = 0.
    "the amplitude of this breakpoint in termsof its percentile within the entire analysis"

    kind: str = ''     # one of '', 'onset', 'offset'
    "the breakpoint type, one of 'onset', 'offset' or '' for a contiuation breakpoint"

    isaccent: bool = False
    "is this breakpoint an accent, indicating a sudden change within its context?"

    duration: float | None = None
    "gap between this breakpoint and the next"

    def __post_init__(self):
        assert self.freq >= 0

    def fields(self) -> list[str]:
        return [field.name for field in fields(self)]

