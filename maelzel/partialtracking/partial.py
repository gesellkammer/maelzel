from __future__ import annotations
import numpy as np
import pitchtools as pt
import numpyx
from functools import cache
import bpf4
from maelzel.snd import amplitudesensitivity
import visvalingamwyatt

from typing import Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel import transcribe



class Partial:
    """
    A Partial represents an overtone within a spectrum

    A Partial consists of multiple breakpoints, where each breakpoint
    is defined by its time, frequency, amplitude, phase and bandwidth

    Args:
        data: the breakpoints, a numpy array with columns (time, freq, amp, phase, bandwidth), where
            each row is a breakpoint
        label: an optional integer id
    """

    def __init__(self, data: np.ndarray, label=0):
        if not len(data.shape) == 2:
            raise ValueError("Expected a 2D numpy array")
        if data.shape[1] < 3:
            raise ValueError(f"Expected a 2D numpy array with at least 3 columns (times, freqs, amps), got {data.shape[1]}")

        self.data = data
        """The breakpoints of this Partial, as a 2D array with columns time, frequency, amplitude, phase and bandwidth"""

        self.start = data[0, 0]
        """Start time of this Partial"""

        self.end = data[-1, 0]
        """End time of this partial"""

        self.numbreakpoints = len(data)
        """The number of breakpoints of this partial"""

        self.label = label
        """A Partial can have an optional integer id called a label"""


    def __repr__(self):
        ampdb = pt.amp2db(self.meanamp())
        return f"Partial(start={self.start:.4f}, end={self.end:.4f}, numbreakpoints={len(self.data)}, " \
               f"meanfreq={self.meanfreq():.1f}, meanamp={ampdb:.1f}dB"


    @property
    def duration(self) -> float:
        """The duration of this partial"""
        return self.end - self.start

    def at(self, time: float) -> tuple[float, ...]:
        """
        Evaluate this partial at the given time, using linear interpolation

        If the Partial's data has columns (time, freq, amp, phase, bandwidth) then the returned tuple
        will be (freq,amp, phase, bandwidth). The values at the given time are interpolated

        Args:
            time: the time to evaluate this partial at. Raises ValueError if the partial is not defined
                at the given time

        Returns:
            the interpolated value of this partial at the given time.

        Example
        ~~~~~~~

            >>> from maelzel.partialtracking import spectrum
            >>> sp = spectrum.analyze(...)
            >>> breakpoints = [partial.at(0.5) for partial in sp if partial.start <= 0.5 <= partial.end]
        """
        if time < self.start or time > self.end:
            raise ValueError(f"This partial is not defined at time {time} (start={self.start}, end={self.end})")
        bp = numpyx.table_interpol_linear(self.data, np.array([time], dtype=float))
        return tuple(float(_) for _ in bp[1:])

    @cache
    def freqbpf(self) -> bpf4.core.Linear:
        """
        Create a bpf curve from this partial's frequency

        Returns:
            a bpf representing this partial's frequency
        """
        return bpf4.core.Linear(self.times, self.freqs)


    @cache
    def ampbpf(self) -> bpf4.core.Linear:
        """
        Create a bpf curve from this partial's amplitude

        Returns:
            a bpf representing this partial's amplitude
        """
        return bpf4.core.Linear(self.times, self.amps)

    @cache
    def meanfreq(self, weighted=True) -> float:
        """
        The average frequency of this partial

        Args:
            weighted: if True, the frequency is weighted by the amplitude of the breakpoint

        Returns:
            the average frequency of this partial, in Hz
        """
        if self.numbreakpoints == 1:
            return self.data[0, 1]

        if weighted:
            return numpyx.weightedavg(self.freqs, self.times, self.amps)
        else:
            freqs = self.freqs
            return numpyx.weightedavg(freqs, self.times, np.ones_like(freqs))

    @cache
    def meanpitch(self) -> float:
        freq = self.meanfreq()
        return pt.f2m(freq)

    @cache
    def meanamp(self) -> float:
        """
        The average amplitude of this partial

        Returns:
            the average amplitude
        """
        if self.numbreakpoints == 1:
            return self.data[0, 2]

        amps = self.amps
        return numpyx.weightedavg(amps, self.times, np.ones_like(amps))

    def audibility(self, ampcurve: Callable[[float], float] = None, curvefactor=1.0) -> float:
        """
        The audibility is the Partial's energy scaled by its frequency dependent audibility

        By default an ANSI A-Weighting Curve is used to assign more audibility to frequences
        which will be more audible

        Args:
            ampcurve: a function mapping a frequency to its sensitivity. If not given, an
                ANSI A-Weighting Curve is used
            curvefactor: the actual incidence of the amplitude curve. 1=the amplitude curve
                has full incidence; 0=the amplitude curve is not used, the energy value is
                used. Any values in between will result in an interpolation.

        Returns:
            the audibility

        """
        energy = self.energy()
        ampcurve = ampcurve or amplitudesensitivity.defaultCurve
        factor = ampcurve(self.meanfreq())
        factor2 = (factor - 1) * curvefactor + 1
        return energy * factor2

    @cache
    def energy(self, mindur=0.002) -> float:
        """
        Integrates the amplitude over time to obtain a measurement of this partial's enery

        Args:
            mindur: duration applied to partials with only one breakpoint

        Returns:
            the total enery contributed by this partial

        .. seealso:: :meth:`Partial.audibility`
        """
        if self.numbreakpoints == 1:
            return self.data[0, 2] * mindur

        amps = self.amps
        times = self.times
        return numpyx.trapz(amps, times)
        # ampbpf = self.ampbpf()
        # return ampbpf.integrate_between(self.start, self.end)

    def meanbw(self, weighted=True) -> float:
        """
        The average bandwidth of this partial

        Args:
            weighted: if True, weight the bandwidth by the partial's amplitude at each breakpoint

        Returns:
            the mean bandwidth
        """
        if weighted:
            return numpyx.weightedavg(self.bws, self.times, self.amps)
        else:
            bws = self.bws
            return numpyx.weightedavg(bws, self.times, np.ones_like(bws))

    @property
    def times(self) -> np.ndarray:
        """The times of all breakpoints"""
        return self.data[:, 0]

    @property
    def freqs(self) -> np.ndarray:
        """The frequencies of all breakpoints"""
        return self.data[:, 1]

    @property
    def amps(self) -> np.ndarray:
        """The amplitudes of all breakpoints"""
        return self.data[:, 2]

    @property
    def phases(self) -> np.ndarray | None:
        """The phases of all breakpoints"""
        if self.data.shape[1] >= 3:
            return self.data[:, 3]
        else:
            return None

    @property
    def bws(self) -> np.ndarray | None:
        """The bandwidths of all breakpoints"""
        if self.data.shape[1] == 5:
            return self.data[:, 4]
        else:
            return None

    def __len__(self):
        return len(self.data)

    def clone(self, *,
              data: np.ndarray | None = None,
              label: int = None
              ) -> Partial:
        return Partial(data=data if data is not None else self.data,
                       label=label if label is not None else self.label)

    def freqTransform(self, transform: Callable[[np.ndarray], np.ndarray]) -> Partial:
        """
        Apply a frequency transformation to this Partial

        Args:
            transform: a function receiving an array of frequencies and
                returning the modified frequencies as array

        Returns:
            the transformed Partial

        Example
        ~~~~~~~

        Transpose a partial a 4th up

            >>> from maelzel.pitchtracking import Spectrum
            >>> from maelzel import pitchtoolsnp as ptnp
            >>> sp = Spectrum.analyze(...)
            >>> def transpose(freqs, interval):
            ...     pitches = ptnp.f2m(freqs)
            ...     return ptnp.m2f(pitches + interval)
            >>> newpartial = sp[0].freqTransform(lambda freqs: transpose(freqs, 5))

        Tune to a scale without removing vibrato

            >>>
        """

        data = self.data.copy()
        data[:, 0] = transform(self.freqs)
        return self.clone(data=data)

    def timeTransform(self, transform: Callable[[np.ndarray], np.ndarray]) -> Partial:
        """
        Apply a time transformation to this Partial

        Args:
            transform: a function receiving a list of  times as numpy array and
                returning the modified times as a numpy array

        Returns:
            the transformed Partial

        Example
        ~~~~~~~

            >>> import bpf4
            >>> from maelzel.pitchtracking import Spectrum
            >>> sp = Spectrum.analyze(...)
            >>> partial = sp[0]
            >>> transform = bpf4.linear(0, 0, 1, 2, 2, 10, 3, 100).keep_slope()
            >>> newpartial = partial.timeTransform(transform.map)
        """
        data = self.data.copy()
        data[:, 0] = transform(self.times)
        return Partial(data, label=self.label)

    def simplified(self,
                   freqthreshold: float | None = None,
                   ratio: float | None = None) -> Partial:
        """
        Simplify the breakpoints of this partial

        Any returned partial will at least contain 2 breakpoints

        Args:
            freqthreshold: the frequency threshold. The higher, the simpler the returned partial
            ratio: the ratio (between 0-1) of the points to simplify. A ratio of 0.1 will simplify
                the shape to contain only 10% of the original points.

        Returns:
            the simplified Partial

        """
        if len(self.data) < 2:
            return self

        points = [(t, f) for t, f in self.data[:, 0:2]]
        simplifier = visvalingamwyatt.Simplifier(points)
        if freqthreshold is not None:
            simplifiedpoints = simplifier.simplify(threshold=freqthreshold)
        elif ratio is not None:
            simplifiedpoints = simplifier.simplify(ratio=ratio)
        else:
            raise ValueError("Either freqthreshold or ratio must be given")

        if len(simplifiedpoints) < 2:
            indexes = [0, len(self.data) - 1]
        else:
            # selectedtimes = np.array([t for t, f in simplifiedpoints], dtype=float)
            times = self.times
            indexes = [numpyx.nearestidx(times, t, sorted=True) for t, f in simplifiedpoints]
            # indexes = [numpyx.searchsorted1(times, t) for t, f in simplifiedpoints]
        data = np.vstack(self.data[indexes])
        return Partial(data=data)

    def __copy__(self) -> Partial:
        return Partial(self.data, label=self.label)

    def __deepcopy__(self, memodict={}) -> Partial:
        # We don't copy the data since we considere Partials to be immutable
        return self.__copy__()

    def copy(self) -> Partial:
        """Copy this Partial"""
        return self.__copy__()

    def crop(self, start: float, end: float) -> Partial | None:
        """
        Crop this partial between the given time interval

        If the partial is included in the time interval, returns self. If the
        partial is not defined within the given interval, returns None. Otherwise,
        returns a copy of self cropped to the given interval.

        Args:
            start: start time of the interval
            end: end time of the interval

        Returns:
            the cropped partial or None if the partial is not defined within the given
            interval

        """
        if self.start >= start and self.end <= end:
            return self
        data = _partialDataCrop(self.data, start, end)
        if data is None:
            return None
        return Partial(data, label=self.label)


def _partialDataCrop(p: np.ndarray, start: float, end: float) -> np.ndarray:
    """
    Crop partial at times t0, t1

    Args:
        p: the partial
        start: the start time
        end: the end time

    Returns:
        the cropped partial (raises `ValueError`) if the partial is not defined
        within the given time constraints)

    !!! note

        * Returns p if p is included in the interval t0-t1
        * Returns None if partial is not defined between t0-t1
        * Otherwise crops the partial at t0 and t1, places a breakpoint
          at that time with the interpolated value

    """
    times = p[:, 0]
    pt0 = times[0]
    pt1 = times[-1]
    if pt0 > start and pt1 < end:
        return p
    if start > pt1 or end < pt0:
        raise ValueError(f"Partial is not defined between {start} and {end}")
    idxs = times > start
    idxs *= times < end
    databetween = p[idxs]
    arrays = []
    if start < databetween[0, 0] and start > pt0:
        arrays.append(_sampleAt(p, np.array([start], dtype=np.float64)))
    arrays.append(databetween)
    if end > databetween[-1, 0] and end < pt1:
        arrays.append(_sampleAt(p, np.array([end], dtype=np.float64)))
    return np.vstack(arrays)


def _sampleAt(data: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Sample a partial's data at given times

    Args:
        data: a partial represented as a 2D-array with columns
           times, freqs, amps, phases, bws
        times: the times to evaluate partial at

    Returns:
        a partial (2D-array with columns times, freqs, amps, phases, bws)
    """
    assert isinstance(times, np.ndarray)
    t0 = data[0, 0]
    t1 = data[-1, 0]
    index0 = numpyx.searchsorted1(times, t0)
    index1 = numpyx.searchsorted1(times, t1)-1
    times = times[index0:index1]
    data = numpyx.table_interpol_linear(data, times)
    timescol = times.reshape((times.shape[0], 1))
    return np.hstack((timescol, data))
