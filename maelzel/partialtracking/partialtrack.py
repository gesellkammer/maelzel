from __future__ import annotations

from maelzel._util import hasoverlap
from emlib import iterlib
import pitchtools as pt
import bisect

import typing as _t
if _t.TYPE_CHECKING:
    from .partial import Partial


__all__ = (
    'PartialTrack',
)


class PartialTrack:
    """
    A Track is a list of non-overlapping Partials

    Args:
        partials: the partials in this Track
        maxrange: the max. range in semitones
    """
    __slots__ = ('partials', 'maxrange', 'minnote', 'maxnote', 'start', 'end', '_starts')

    def __init__(self, partials: list[Partial] | None = None, maxrange=36):
        assert isinstance(maxrange, (int, float))
        self.partials: list[Partial] = partials or []
        self.maxrange: int = int(maxrange)
        self.minnote = pt.f2m(min(p.meanfreq() for p in self.partials)) if partials else 0.
        self.maxnote = pt.f2m(max(p.meanfreq() for p in self.partials)) if partials else 0.
        self.start = self.partials[0].start if partials else 0.
        self.end = self.partials[-1].end if partials else 0.
        self._starts: list[float] = [p.start for p in partials] if partials else []

    def __repr__(self):
        return f"Track(partials={len(self.partials)}, range={pt.m2n(round(self.minnote))}-{pt.m2n(round(self.maxnote))}, " \
               f"start={self.start:.3f}, end={self.end:.3f})"

    def __len__(self):
        return len(self.partials)

    def __iter__(self):
        return iter(self.partials)

    def __getitem__(self, item):
        return self.partials[item]

    def append(self, partial: Partial):
        # We assume that there is no overlap
        if partial.start >= self.end:
            self.partials.append(partial)
            self._starts.append(partial.start)
            self.end = partial.end
        else:
            start, end = partial.start, partial.end
            for p in self.partials:
                if p.start > end:
                    break
                if hasoverlap(p.start, p.end, start, end):
                    raise ValueError(f"Partial {partial} does not fit, partials in track: {self.partials}")
            idx = bisect.bisect(self._starts, partial.start)
            self.partials.insert(idx, partial)
            self._starts.insert(idx, partial.start)
            self.start = self._starts[0]
        meanpitch = partial.meanpitch()
        if meanpitch < self.minnote:
            self.minnote = meanpitch
        elif meanpitch > self.maxnote:
            self.maxnote = meanpitch

    def meanpitch(self) -> float:
        """
        The mean pitch of this Track
        """
        return pt.f2m(sum(p.meanfreq() for p in self.partials) / len(self.partials))

    def isTimerangeEmpty(self, start: float, end: float) -> bool:
        """
        Is this Track empty within the given times?
        """
        partials = self.partials
        if partials or partials[-1].end < start or partials[0].start >= end:
            return True
        p0index = bisect.bisect_left(self._starts, end) - 1
        p0 = self.partials[p0index]
        return not hasoverlap(p0.start, p0.end, start, end)

    def partialBefore(self, t: float) -> int | None:
        """
        Returns the index of the partial starting before time t or None

        Args:
            t: the time

        Returns:
            the partial index or None
        """
        if not self.partials or t < self.start:
            return None
        idx = bisect.bisect_left(self._starts, t) - 1
        assert idx >= 0
        assert self.partials[idx].start <= t, f"{t=:.4f}, {idx=}, partial={self.partials[idx]}, partials={self.partials}"
        return idx

    def dump(self):
        from emlib import misc
        rows = [(p.start, p.end, p.numbreakpoints, p.meanfreq())
                for p in self.partials]
        misc.print_table(rows, headers=('start', 'end', 'len', 'freq'), floatfmt=".4f")

    def check(self) -> bool:
        if not self.partials:
            return True
        return all(p0.end <= p1.start
                   for p0, p1 in iterlib.pairwise(self.partials))
