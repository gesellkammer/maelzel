"""
This module implements spectral transcription

Example
~~~~~~~

    import maelzel.transcribe.spectral as sp

    spectrum = sp.analyze(samples, resolution=50, windowsize=70, overlap=4)
    spectrum.simplify(...)


"""
from __future__ import annotations
import numpy as np
import numpyx
import visvalingamwyatt
import loristrck as lt
import bpf4


class Partial:

    def __init__(self, data: np.ndarray, label=0):
        if not len(data.shape) == 2:
            raise ValueError("Expected a 2D numpy array")
        if data.shape[1] < 3:
            raise ValueError(f"Expected a 2D numpy array with at least 3 columns (times, freqs, amps), got {data.shape[1]}")

        self.data = data
        self.start = data[0, 0]
        self.end = data[-1, 0]
        self.numbreakpoints = len(data)
        self.label = label
        self._freqbpf: bpf4.core.Linear | None = None
        self._ampbpf: bpf4.core.Linear | None = None
        self._bwbpf: bpf4.core.Linear | None = None

    def freqbpf(self) -> bpf4.core.Linear:
        if self._freqbpf is None:
            self._freqbpf = bpf4.core.Linear(self.times, self.freqs)
        return self._freqbpf

    def ampbpf(self) -> bpf4.core.Linear:
        if self._ampbpf is None:
            self._ampbpf = bpf4.core.Linear(self.times, self.amps)
        return self._ampbpf

    def meanfreq(self, weighted=True) -> float:
        if weighted:
            return numpyx.weightedavg(self.freqs, self.times, self.amps)
        else:
            freqs = self.freqs
            return numpyx.weightedavg(freqs, self.times, np.ones_like(freqs))

    def meanamp(self) -> float:
        amps = self.amps
        return numpyx.weightedavg(amps, self.times, np.ones_like(amps))

    def meanbw(self, weighted=True) -> float:
        if weighted:
            return numpyx.weightedavg(self.bws, self.times, self.amps)
        else:
            bws = self.bws
            return numpyx.weightedavg(bws, self.times, np.ones_like(bws))

    @property
    def times(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def freqs(self) -> np.ndarray:
        return self.data[:, 1]

    @property
    def amps(self) -> np.ndarray:
        return self.data[:, 2]

    @property
    def phases(self) -> np.ndarray | None:
        if self.data.shape[1] >= 3:
            return self.data[:, 3]
        else:
            return None

    @property
    def bws(self) -> np.ndarray | None:
        if self.data.shape[1] == 5:
            return self.data[:, 4]
        else:
            return None

    def __len__(self):
        return len(self.data)

    def simplified(self, freqthreshold: float = 1) -> Partial:
        points = [(t, f) for t, f in self.data[:, 0:2]]
        simplifier = visvalingamwyatt.Simplifier(points)
        simplifiedpoints = simplifier.simplify(threshold=freqthreshold)
        indexes = [numpyx.searchsorted1(self.data, t) for t, f in simplifiedpoints]
        data = np.vstack(self.data[indexes])
        return Partial(data=data)


def _firstPartialAfter(partials: list[Partial], t0: float) -> int:
    for i, p in enumerate(partials):
        if p.end > t0:
            return i
    return -1


def _partialsBetween(partials: list[Partial], t0=0., t1=0.) -> list[Partial]:
    """
    Return the partials present between t0 and t1

    Partials should be sorted

    This function is not optimized and performs a linear search over the
    partials. If this function is to be called repeatedly or within a
    performance relevant section, use `PartialIndex` instead

    Args:
        partials: a list of partials
        t0: start time in secs
        t1: end time in secs

    Returns:
        the partials within the time range (t0, t1)


    """
    if t1 == 0:
        t1 = max(p.end for p in partials)
    out = []
    for p in partials:
        if p.start > t1:
            break
        if p.end > t0:
            out.append(p)
    return out


class _PartialIndex:
    """
    Create an index to accelerate finding partials

    After creating the PartialIndex, each call to `partialindex.partials_between`
    should be faster than simply calling `partials_index` since the unoptimized
    function needs to always start a linear search from the beginning of the
    partials list.

    !!! note

        The index is only valid as long as the original partial list is not
        modified

    """
    def __init__(self, partials: list[Partial], end: float, dt=1.0):
        """
        Args:
            partials: the partials to index
            dt: the time resolution of the index. The lower this value the faster
                each query will be but the slower the creation of the index itself
        """
        self.start = partials[0].start
        self.end = end
        self.dt = dt
        self.partials = partials
        firstpartials = []
        startidx = 0
        for t in np.arange(self.start, end, dt):
            relidx = _firstPartialAfter(partials[max(0, startidx - 1):], float(t))
            absidx = startidx + relidx
            firstpartials.append(absidx)
            startidx = absidx
        self.firstpartials = firstpartials

    def partialsBetween(self, start: float, end: float) -> list[Partial]:
        """
        Returns the partials which are defined within the given time range

        Args:
            t0: the start of the time interval
            t1: the end of the time interval

        Returns:
            a list of partials present during the given time range
        """
        if start > end:
            raise ValueError(f"The start time should not be later than the end time, got {start=}, {end=}")
        idx = int((start - self.start) / self.dt)
        firstpartial = self.firstpartials[idx]
        if firstpartial < 0:
            return []
        return _partialsBetween(self.partials[firstpartial:], start, end)


class Spectrum:

    def __init__(self, partials: list[Partial], indexdt=1.0):
        self.partials = partials
        self.partials.sort(key=lambda p: p.start)
        self._index: _PartialIndex | None = None
        self.indexdt = indexdt

    @staticmethod
    def read(path: str) -> Spectrum:
        return readsdif(path)

    @property
    def start(self) -> float:
        return float(self.partials[0].start)

    @property
    def end(self) -> float:
        return float(max(p.end for p in self.partials))

    def __repr__(self):
        return f'Spectrum(numpartials={len(self)}, start={self.start}, end={self.end})'

    @property
    def index(self) -> _PartialIndex:
        if self._index is None:
            self._index = _PartialIndex(self.partials, end=self.end, dt=self.indexdt)
        return self._index

    def __len__(self):
        return len(self.partials)

    def partialsBetween(self, start: float, end: float) -> list[Partial]:
        return self.index.partialsBetween(start, end)

    def writesdif(self, outfile: str, rbep=True, ) -> None:
        arrays = [p.data for p in self.partials]
        labels = [p.label for p in self.partials]
        lt.write_sdif(arrays, outfile=outfile, fmt='RBEP' if rbep else '1TRC', labels=labels)


def readsdif(path: str) -> Spectrum:
    arrays, labels = lt.read_sdif(path)
    return Spectrum(partials=[Partial(array, label=label) for array, label in zip(arrays, labels)])


def analyze(samples: np.ndarray,
            sr: int,
            resolution: float,
            windowsize: float = None,
            hoptime: float = None,
            freqdrift: float = None
            ) -> Spectrum:
    partialarrays = lt.analyze(samples,
                               sr=sr,
                               resolution=resolution,
                               windowsize=windowsize or -1,
                               hoptime=hoptime or -1,
                               freqdrift=freqdrift or -1)

    return Spectrum(partials=[Partial(data) for data in partialarrays])