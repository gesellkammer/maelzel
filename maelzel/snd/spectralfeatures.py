"""
pYIN (Probabilistic YIN) Fundamental Frequency Estimator
=========================================================

Ref.: Mauch & Dixon (2014), "pYIN: A Fundamental Frequency Estimator
      Using Probabilistic Threshold Distributions"

Usage
-----

.. code-block:: python

    times, f0, voiced = pyin(samples, sr=22050)

Tuning for noisy speech
-----------------------

.. code-block:: python

    # Too many NaN gaps:
    times, f0, voiced = pyin(audio, sr, p_v2u=0.05)

    # Slow to recover after silence / consonant:
    times, f0, voiced = pyin(audio, sr, p_u2v=0.6)

    # Both (continuous speech with consonants):
    times, f0, voiced = pyin(audio, sr, p_v2u=0.05, p_u2v=0.6)
"""
from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import numpy as np
from maelzel.snd.numpysnd import makeFrames
from functools import cache
import typing as _t


def _sumLargestN(arr: np.ndarray, n: int) -> float:
    """
    Returns the sum of the largest n values of arr

    This is used to calculate peakyness of a signal,
    by dividing the sum of magnitudes of the largest
    bins by the sum of all magnitudes, gives a measure
    of how concentrated the signal is

    Args:
        arr: the array
        n: the number of values to sum

    Returns:
        the sum of the largest

    """
    nlargest = arr[arr.argpartition(len(arr) - n)[-n:]]
    return nlargest.sum()


def _rolloffBin(spectrum: np.ndarray, rolloff: float) -> int:
    assert 0 <= rolloff <= 1
    cumenergy = np.cumsum(spectrum)
    total = cumenergy[-1]
    threshold = rolloff * total
    return int(np.argmax(cumenergy > threshold))


def _spectralFlatness(powerspec: np.ndarray) -> float:
    """
    Expects a power spectrum

    Args:
        powerspec: the power spectrum

    Returns:
        the flatness, this gives a measure which ranges from approx 0
        for a pure sinusoid, to approx 1 for white noise.
        The measure is calculated linearly. For some applications you
        may wish to convert the value to a decibel scale

    """
    zeros = powerspec == 0
    if np.any(zeros):
        powerspec = np.where(zeros, 1e-12, powerspec)

    geom_mean = np.exp(np.mean(np.log(powerspec)))
    return geom_mean / np.mean(powerspec)


def _spectralCentroid(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    magsum = magnitude.sum()
    if magsum == 0:
        return 0.0
    return float((magnitude * freqs).sum() / magsum)


def _fftDiff(rfft: np.ndarray, N: int) -> np.ndarray:
    """
    Sum of squared differences at each lag, using FFT autocorrelation

    Args:
        rfft: an rfft frame
        N: number of bins

    Returns:
        the difference function, an array of length N containing the
        correlation difference per fft bin

    """
    acf = np.fft.irfft(rfft * np.conj(rfft))[:N]
    return 2.0 * (acf[0] - acf)


# def _diffFunc(frame: np.ndarray) -> np.ndarray:
#     """
#     difference function
#
#     Args:
#         frame: Audio data
#
#     Returns:
#         an array containing the correlation difference per fft bin
#
#     """
#     N = len(frame)
#     # next power of 2 > than N
#     fftsize = 1 << (2 * N - 1).bit_length()
#     F = np.fft.rfft(frame, n=fftsize)
#     return _fftDiff(F, N)


class CMNDF:
    def __init__(self, size: int):
        self.size = size
        self.factor = np.arange(1, self.size)
        # Preallocated array
        self.cumsum = np.empty_like(self.factor)

    def __call__(self, df: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        assert len(df) == self.size
        if out is None:
            out = np.empty_like(df)
        out[0] = 1.0
        df1 = df[1:]
        cs = self.cumsum
        np.cumsum(df1, out=cs)
        np.multiply(df1, self.factor, out=out[1:])
        # out[1:] = df1 * self.factor
        out[1:] /= np.where(cs > 0, cs, 1.0)
        return out


def _cmndf(df: np.ndarray) -> np.ndarray:
    out = np.empty_like(df)
    out[0] = 1.0
    cs = np.cumsum(df[1:])
    out[1:] = df[1:] * np.arange(1, len(df)) / np.where(cs > 0, cs, 1.0)
    return out


def _parabolicInterp(arr: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(arr) - 1:
        return float(idx)
    # denom = arr[idx - 1] - 2.0 * arr[idx] + arr[idx + 1]
    # return idx if denom == 0.0 else idx - 0.5 * (arr[idx - 1] - arr[idx + 1]) / denom
    val0 = arr[idx - 1]
    val1 = arr[idx]
    val2 = arr[idx+1]
    denom = val0 - 2.0 * val1 + val2
    return idx if denom == 0.0 else idx - 0.5 * (val0 - val2) / denom


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def _analyzeFrame(cmndf: np.ndarray, tau_min: int, tau_max: int
                  ) -> tuple[float, float]:
    """
    Return (best_tau, cmndf_min) for one CMNDF frame.

    Args:
        cmndf: one cmndf frame holding tau values
        tau_min: index to first tau value to use
        tau_max: index to last tau value to use

    Returns:
        a tuple (best_tau, min_tau) within the given values, where
        best_tau represents a fractional index to the cmndf frame
        and min_tau is the actual min. tau value within the given
        selection
    """
    cmndf_selection = cmndf[tau_min: tau_max + 1]
    best_rel = int(np.argmin(cmndf_selection))
    best_tau = _parabolicInterp(cmndf, best_rel + tau_min)
    return best_tau, float(cmndf_selection[best_rel])


@dataclass(slots=True)
class AnalysisFrame:
    index: int
    time: float
    magnitude: np.ndarray
    frameSize: int
    _power: np.ndarray | None = None
    _diff: np.ndarray | None = None
    _cmndf: np.ndarray | None = None

    def power(self) -> np.ndarray:
        """
        Returns the power spectrum (cached)
        """
        if self._power is None:
            self._power = self.magnitude ** 2
        return self._power

    def diff(self) -> np.ndarray:
        if self._diff is None:
            self._diff = _fftDiff(self.magnitude, self.frameSize)
        assert self._diff is not None
        return self._diff

    def cmndf(self, cmndf: CMNDF) -> np.ndarray:
        if self._cmndf is None:
            self._cmndf = cmndf(self.diff())
        return self._cmndf


class AnalysisFeature(ABC):
    name = ''

    def prepare(self, analysis: Analysis) -> None:
        pass

    def consume(self, frame: AnalysisFrame, analysis: Analysis) -> None:
        pass

    def finalize(self, analysis: Analysis) -> dict:
        return {}


class Analysis:
    """
    Shared analysis context for frame-based feature extraction.
    """

    def __init__(self,
                 samples: np.ndarray,
                 sr: int,
                 frameSize: int = 2048,
                 hopSize: int = 0,
                 minFreq: float = 80.0,
                 maxFreq: float = 1000.0,
                 window: str = "hann",
                 normalize: bool = True):
        self.samples = np.asarray(samples, dtype=np.float64)
        if self.samples.ndim != 1:
            raise ValueError("Only mono arrays are supported")

        if normalize:
            peak = np.abs(self.samples).max()
            if peak > 0:
                self.samples = self.samples / peak

        self.sr = int(sr)
        self.frameSize = int(frameSize)
        self.hopSize = int(hopSize) or self.frameSize // 4
        self.minFreq = float(minFreq)
        self.maxFreq = float(maxFreq)
        self.windowName = window
        from scipy.signal import get_window
        self.window = get_window(window, self.frameSize)
        self.frames, self.times = makeFrames(self.samples, sr=self.sr,
                                             frameSize=self.frameSize,
                                             hopSize=self.hopSize)
        self.fftSize = 1 << (2 * self.frameSize - 1).bit_length()
        self.fftFreqs = np.fft.rfftfreq(self.fftSize, d=1. / self.sr)
        self.cmndf = CMNDF(self.frameSize)
        self._features: list[AnalysisFeature] = []

    @property
    def numFrames(self) -> int:
        return len(self.frames)

    def addFeature(self, feature: str | AnalysisFeature) -> Analysis:
        if isinstance(feature, str):
            feature = _featureFactory(feature)
        self._features.append(feature)
        return self

    def addFeatures(self, *features: str | AnalysisFeature) -> Analysis:
        for feature in features:
            self.addFeature(feature)
        return self

    def iterFrames(self) -> _t.Iterator[AnalysisFrame]:
        for index, frame in enumerate(self.frames):
            windowed = frame * self.window
            spectrum = np.fft.rfft(windowed, n=self.fftSize)
            magnitude = np.abs(spectrum)
            power = magnitude ** 2
            yield AnalysisFrame(index=index, time=float(self.times[index]),
                                magnitude=magnitude, power=power,
                                frameSize=self.frameSize)

    def run(self) -> dict:
        for feature in self._features:
            feature.prepare(self)

        for frame in self.iterFrames():
            for feature in self._features:
                feature.consume(frame, self)

        out = {'times': self.times}
        for feature in self._features:
            out.update(feature.finalize(self))
        return out


class PyinFeature(AnalysisFeature):
    name = 'pyin'

    def __init__(self,
                 minFreq: float | None = None,
                 maxFreq: float | None = None,
                 bins: int = 360,
                 voicedProb: float = 0.5,
                 v2u: float = 0.1,
                 u2v: float = 0.4,
                 pitchDrift: float = 100.0):
        self.minFreq = minFreq
        self.maxFreq = maxFreq
        self.bins = bins
        self.voicedProb = voicedProb
        self.v2u = v2u
        self.u2v = u2v
        self.pitchDrift = pitchDrift
        self._frameData: list[tuple[float, float]] = []

    def prepare(self, analysis: Analysis) -> None:
        self.minFreq = analysis.minFreq if self.minFreq is None else self.minFreq
        self.maxFreq = analysis.maxFreq if self.maxFreq is None else self.maxFreq
        self.tau_min = max(1, int(np.floor(analysis.sr / self.maxFreq)))
        self.tau_max = min(analysis.frameSize - 1, int(np.ceil(analysis.sr / self.minFreq)))
        self._frameData = []

    def consume(self, frame: AnalysisFrame, analysis: Analysis) -> None:
        self._frameData.append(_analyzeFrame(frame.cmndf(analysis.cmndf),
                                             self.tau_min, self.tau_max))

    def finalize(self, analysis: Analysis) -> dict:
        f0, _, confidence = _viterbiDecode3(
            self._frameData,
            fmin=self.minFreq,
            fmax=self.maxFreq,
            sr=analysis.sr,
            n_pitch_bins=self.bins,
            voiced_prob=self.voicedProb,
            v2u=self.v2u,
            u2v=self.u2v,
            transition_width=self.pitchDrift)
        return {
            'pyin.f0': f0,
            'pyin.confidence': confidence,
        }


class FlatnessFeature(AnalysisFeature):
    name = 'flatness'

    def __init__(self):
        self.values: np.ndarray | None = None

    def prepare(self, analysis: Analysis) -> None:
        self.values = np.zeros(analysis.numFrames, dtype=float)

    def consume(self, frame: AnalysisFrame, analysis: Analysis) -> None:
        assert self.values is not None
        self.values[frame.index] = _spectralFlatness(frame.power())

    def finalize(self, analysis: Analysis) -> dict:
        assert self.values is not None
        return {'flatness': self.values}


class CentroidFeature(AnalysisFeature):
    name = 'centroid'

    def __init__(self):
        self.values: np.ndarray | None = None

    def prepare(self, analysis: Analysis) -> None:
        self.values = np.zeros(analysis.numFrames, dtype=float)

    def consume(self, frame: AnalysisFrame, analysis: Analysis) -> None:
        assert self.values is not None
        self.values[frame.index] = _spectralCentroid(frame.magnitude, analysis.fftFreqs)

    def finalize(self, analysis: Analysis) -> dict:
        assert self.values is not None
        return {'centroid': self.values}


class RolloffFeature(AnalysisFeature):
    name = 'rolloff'

    def __init__(self, rolloff: float = 0.85):
        self.rolloff = rolloff
        self.values: np.ndarray | None = None

    def prepare(self, analysis: Analysis) -> None:
        self.values = np.zeros(analysis.numFrames, dtype=float)

    def consume(self, frame: AnalysisFrame, analysis: Analysis) -> None:
        assert self.values is not None
        rollbin = _rolloffBin(frame.power(), rolloff=self.rolloff)
        self.values[frame.index] = analysis.fftFreqs[int(rollbin)]

    def finalize(self, analysis: Analysis) -> dict:
        assert self.values is not None
        return {'rolloff': self.values}


class FocusFeature(AnalysisFeature):
    name = 'focus'

    def __init__(self,
                 focusBins: int = 0,
                 focusMinFreq: float = 0.,
                 focusMaxFreq: float = 0.):
        self.focusBins = focusBins
        self.focusMinFreq = focusMinFreq
        self.focusMaxFreq = focusMaxFreq
        self.values: np.ndarray | None = None
        self.focusMinBin = 0
        self.focusMaxBin = 0

    def prepare(self, analysis: Analysis) -> None:
        self.focusBins = self.focusBins or analysis.frameSize // 64
        focusMinFreq = self.focusMinFreq or analysis.minFreq * 2.5
        focusMaxFreq = self.focusMaxFreq or analysis.sr * 0.4
        self.focusMinBin = int(np.searchsorted(analysis.fftFreqs, focusMinFreq, side='left'))
        self.focusMaxBin = int(np.searchsorted(analysis.fftFreqs, focusMaxFreq, side='left'))
        self.values = np.zeros(analysis.numFrames, dtype=float)

    def consume(self, frame: AnalysisFrame, analysis: Analysis) -> None:
        assert self.values is not None
        freal_sel = frame.magnitude[self.focusMinBin:self.focusMaxBin]
        freal_sel_sum = freal_sel.sum()
        self.values[frame.index] = 0 if freal_sel_sum == 0 else _sumLargestN(freal_sel, n=self.focusBins) / freal_sel_sum

    def finalize(self, analysis: Analysis) -> dict:
        assert self.values is not None
        return {
            'focus': self.values,
            'focus.binRange': (self.focusMinBin, self.focusMaxBin),
        }


def _featureFactory(name: str) -> AnalysisFeature:
    if name == 'pyin':
        return PyinFeature()
    elif name == 'flatness':
        return FlatnessFeature()
    elif name == 'centroid':
        return CentroidFeature()
    elif name == 'rolloff':
        return RolloffFeature()
    elif name == 'focus':
        return FocusFeature()
    raise ValueError(f"Unknown feature {name!r}")


# -----------------------------------------------ww----------------------------
# Transition log-cost matrix
# ---------------------------------------------------------------------------

def _transitionCosts(
        n: int,
        log_p_v2v: float,
        transition_width_cents: float,
        cents_per_bin: float
    ) -> np.ndarray:
    """
    Build an (n, n) matrix of log-domain voiced->voiced transition costs.

    Each entry is::

        cost[i, j] = log_p_v2v − 0.5 · ((i−j) · cents_per_bin / sigma_cents)²

    This is an **un-normalised Gaussian log-penalty** — intentionally not a
    valid probability distribution.  The Viterbi algorithm only needs ratios
    of path scores, so the normalisation constant cancels and can be omitted.

    Why un-normalised?
    ------------------

    Any proper row-normalised Gaussian spreads its mass over all n bins,
    making each off-diagonal entry scale as 1/n times the Gaussian value.
    With n = 360 bins, even a 3-bin neighbour gets only ~1/360 of the mass
    after normalisation, costing ≈ log(1/360) ≈ −5.9 nats — far more than
    the unvoiced->voiced cost of log(p_u2v) ≈ −0.9 nats.  The Viterbi then
    always prefers the cheap unvoiced->voiced path, causing U/V flickering.

    With the un-normalised cost the self-transition costs only log(p_v2v)
    and a neighbour k bins away costs log(p_v2v) − k²/(2σ²).  Small pitch
    changes are almost free; the Gaussian penalty only matters for large jumps.
    """
    sigma_bins = transition_width_cents / cents_per_bin
    offsets = np.arange(n)[:, None] - np.arange(n)[None, :]   # (n, n)
    return log_p_v2v - 0.5 * (offsets / sigma_bins) ** 2


# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------

def _emissions(
        frame_data: list[tuple[float, float]],
        log_freq: np.ndarray,
        f_min: float,
        f_max: float,
        sr: int,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-frame log-probability emissions.

    P(voiced | frame) = 1 − cmndf_min   (direct periodicity measure).
    All voiced mass goes on the single best-pitch bin.
    """
    n_frames     = len(frame_data)
    n_bins       = len(log_freq)
    obs_voiced   = np.full((n_frames, n_bins), -np.inf)
    obs_unvoiced = np.empty(n_frames)

    for t, (best_tau, cmndf_min) in enumerate(frame_data):
        if best_tau <= 0:
            obs_unvoiced[t] = 0.0
        else:
            f = sr / best_tau
            if not (f_min <= f <= f_max):
                obs_unvoiced[t] = 0.0
            else:
                p_v = float(np.clip(1.0 - cmndf_min, 1e-9, 1.0 - 1e-9))
                bidx = int(np.argmin(np.abs(log_freq - np.log2(f))))
                obs_voiced[t, bidx] = np.log(p_v)
                obs_unvoiced[t]     = np.log(1.0 - p_v)

    return obs_voiced, obs_unvoiced


# ---------------------------------------------------------------------------
# Viterbi decoder
# ---------------------------------------------------------------------------

def _viterbiDecode1(
        frame_data: list[tuple[float, float]],
        f_min: float,
        f_max: float,
        sr: int,
        n_pitch_bins: int,
        voiced_prob: float,
        v2u: float,
        u2v: float,
        transition_width: float,
    ) -> np.ndarray:
    """
    Viterbi over n_pitch_bins log-spaced pitch states + 1 unvoiced state.
    
    Args:
        frame_data: 
        f_min: min frequency
        f_max: max frequency
        sr: sampling rate
        n_pitch_bins: number of pitch bins
        voiced_prob: voiced probability
        v2u: transition voiced->unvoiced probability
        u2v: transition unvoiced->unvoiced probability
        transition_width: transition width, in cents
        
    Returns:
        the smooth frequency series (numpy array)
        

    Transition design
    -----------------
    
    voiced[i] -> voiced[j]:  log_p_v2v − 0.5·((i−j)/σ)²   (Gaussian log-cost, unnormalised)
    voiced    -> unvoiced:   log_p_v2u
    unvoiced  -> voiced[j]:  log_p_u2v   (emission picks the bin; no per-bin division)
    unvoiced  -> unvoiced:   log_p_u2u
    """
    logfreq = np.linspace(np.log2(f_min), np.log2(f_max), n_pitch_bins)
    cents_per_bin = (logfreq[1] - logfreq[0]) * 1200.0
    UNVOICED = n_pitch_bins
    n_frames = len(frame_data)

    log_p_v2v = np.log(1.0 - v2u)
    log_p_v2u = np.log(v2u)
    log_p_u2v = np.log(u2v)
    log_p_u2u = np.log(1.0 - u2v)

    log_trans = _transitionCosts(n_pitch_bins, log_p_v2v, transition_width, cents_per_bin)

    obs_voiced, obs_unvoiced = _emissions(frame_data, logfreq, f_min, f_max, sr)

    dp = np.full((n_frames, n_pitch_bins + 1), -np.inf)
    bp = np.zeros((n_frames, n_pitch_bins + 1), dtype=np.int32)

    dp[0, :n_pitch_bins] = obs_voiced[0]   + np.log(np.clip(voiced_prob,       1e-9, 1 - 1e-9))
    dp[0, UNVOICED]      = obs_unvoiced[0] + np.log(np.clip(1.0 - voiced_prob, 1e-9, 1 - 1e-9))

    for t in range(1, n_frames):
        prev_v = dp[t - 1, :n_pitch_bins]
        prev_u = dp[t - 1, UNVOICED]

        # voiced -> voiced  (Gaussian log-cost; prev_v[j] + log_trans[j, b] for each b)
        trans_vv    = prev_v[:, None] + log_trans   # (n, n)
        best_from_v = np.max(trans_vv, axis=0)      # (n,)
        bp_from_v   = np.argmax(trans_vv, axis=0)   # (n,)

        # unvoiced -> voiced
        from_u = prev_u + log_p_u2v
        take_u = from_u > best_from_v

        dp[t, :n_pitch_bins] = obs_voiced[t] + np.where(take_u, from_u, best_from_v)
        bp[t, :n_pitch_bins] = np.where(take_u, UNVOICED, bp_from_v)

        # unvoiced state
        argmax_prev_v = int(np.argmax(prev_v))
        max_prev_v    = prev_v[argmax_prev_v]
        # from_v_to_u = np.max(prev_v) + log_p_v2u
        from_v_to_u = max_prev_v + log_p_v2u
        from_u_to_u = prev_u + log_p_u2u
        if from_v_to_u > from_u_to_u:
            dp[t, UNVOICED] = obs_unvoiced[t] + from_v_to_u
            bp[t, UNVOICED] = argmax_prev_v
        else:
            dp[t, UNVOICED] = obs_unvoiced[t] + from_u_to_u
            bp[t, UNVOICED] = UNVOICED

    path = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    return np.where(
        path < n_pitch_bins,
        2.0 ** logfreq[np.clip(path, 0, n_pitch_bins - 1)],
        np.nan)


def _gaussian_max_filter_vec(values: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised O(n) Gaussian max-filter using scipy grey_dilation.
    For each j: best_val[j] = max_i [ values[i] - 0.5*((i-j)/sigma)^2 ]
    """
    from scipy.ndimage import grey_dilation

    # Only need to consider bins within ~4 sigma (negligible prob beyond)
    half_w = int(np.ceil(4.0 * sigma))
    n = len(values)
    size = 2 * half_w + 1

    offsets = np.arange(-half_w, half_w + 1, dtype=float)
    # Parabolic structuring element: the "shape" added to each value
    # grey_dilation computes max_i [ values[i] + se[j-i] ]
    # We want max_i [ values[i] - 0.5*((i-j)/sigma)^2 ]
    # So se[k] = -0.5*(k/sigma)^2
    se = -0.5 * (offsets / sigma) ** 2

    dilated = grey_dilation(values, size=size, structure=se, mode='nearest')

    # Recover argmax via argmax over the explicit windowed view
    # Pad values for boundary handling
    pad = np.pad(values, half_w, mode='edge')
    # Strided windows: shape (n, size)
    strides = (pad.strides[0], pad.strides[0])
    windows = np.lib.stride_tricks.as_strided(pad, shape=(n, size), strides=strides)
    scores = windows + se[np.newaxis, :]   # (n, size)
    local_argmax = np.argmax(scores, axis=1)  # index within window
    best_idx = (np.arange(n) - half_w + local_argmax).clip(0, n - 1).astype(np.int32)

    return dilated, best_idx


def _viterbiDecode2(
        framedata: list[tuple[float, float]],
        fmin: float,
        fmax: float,
        sr: int,
        n_pitch_bins: int,
        voiced_prob: float,
        v2u: float,
        u2v: float,
        transition_width: float,
    ) -> np.ndarray:
    """
    Viterbi over n_pitch_bins log-spaced pitch states + 1 unvoiced state.

    Args:
        framedata:
        fmin: min frequency
        fmax: max frequency
        sr: sampling rate
        n_pitch_bins: number of pitch bins
        voiced_prob: voiced probability
        v2u: transition voiced->unvoiced probability
        u2v: transition unvoiced->voiced probability
        transition_width: transition width, in cents

    Returns:
        the smooth frequency series (numpy array)

    Transition design
    -----------------

    voiced[i] -> voiced[j]:  log_p_v2v − 0.5·((i−j)/σ)²   (Gaussian log-cost)
    voiced    -> unvoiced:   log_p_v2u
    unvoiced  -> voiced[j]:  log_p_u2v
    unvoiced  -> unvoiced:   log_p_u2u

    Complexity
    ----------
    O(T · n) via parabolic max-filter replacing the naive O(T · n²) matrix op.
    """
    logfreq = np.linspace(np.log2(fmin), np.log2(fmax), n_pitch_bins)
    cents_per_bin = (logfreq[1] - logfreq[0]) * 1200.0
    sigma = transition_width / cents_per_bin   # transition width in bins

    UNVOICED = n_pitch_bins
    n_frames = len(framedata)

    log_p_v2v = np.log(1.0 - v2u)
    log_p_v2u = np.log(v2u)
    log_p_u2v = np.log(u2v)
    log_p_u2u = np.log(1.0 - u2v)

    obs_voiced, obs_unvoiced = _emissions(framedata, logfreq, fmin, fmax, sr)

    dp = np.full((n_frames, n_pitch_bins + 1), -np.inf)
    bp = np.zeros((n_frames, n_pitch_bins + 1), dtype=np.int32)

    dp[0, :n_pitch_bins] = obs_voiced[0]   + np.log(np.clip(voiced_prob,       1e-9, 1 - 1e-9))
    dp[0, UNVOICED]      = obs_unvoiced[0] + np.log(np.clip(1.0 - voiced_prob, 1e-9, 1 - 1e-9))

    for t in range(1, n_frames):
        prev_v = dp[t - 1, :n_pitch_bins]  # (n,)
        prev_u = dp[t - 1, UNVOICED]

        # voiced -> voiced: O(n) Gaussian max-filter instead of O(n²) matrix
        gauss_val, gauss_idx = _gaussian_max_filter_vec(prev_v, sigma)
        best_from_v = log_p_v2v + gauss_val   # (n,)
        bp_from_v   = gauss_idx               # (n,)  int indices

        # unvoiced -> voiced: scalar broadcast
        from_u  = prev_u + log_p_u2v
        take_u  = from_u > best_from_v        # (n,) bool

        dp[t, :n_pitch_bins] = obs_voiced[t] + np.where(take_u, from_u, best_from_v)
        bp[t, :n_pitch_bins] = np.where(take_u, UNVOICED, bp_from_v)

        # --- unvoiced state ---
        # Best voiced -> unvoiced predecessor (reuse gauss bookkeeping is not
        # helpful here; we just need the global max of prev_v, computed once)
        argmax_v    = int(np.argmax(prev_v))
        max_prev_v  = prev_v[argmax_v]

        from_v_to_u = max_prev_v + log_p_v2u
        from_u_to_u = prev_u     + log_p_u2u

        if from_v_to_u > from_u_to_u:
            dp[t, UNVOICED] = obs_unvoiced[t] + from_v_to_u
            bp[t, UNVOICED] = argmax_v
        else:
            dp[t, UNVOICED] = obs_unvoiced[t] + from_u_to_u
            bp[t, UNVOICED] = UNVOICED

    # Viterbi back-track
    path     = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    return np.where(
        path < n_pitch_bins,
        2.0 ** logfreq[np.clip(path, 0, n_pitch_bins - 1)],
        np.nan,
    )


@cache
def _logFreq(fmin: float, fmax: float, numbins: int) -> np.ndarray:
    return np.linspace(np.log2(fmin), np.log2(fmax), numbins)


def _gauss_kernel(v2u: float, sigma: float, half_w: int):
    offsets = np.arange(-half_w, half_w + 1, dtype=float)
    log_p_v2v = np.log(1.0 - v2u)
    return log_p_v2v - 0.5 * (offsets / sigma) ** 2


def _viterbiDecode3(
        framedata: list[tuple[float, float]],
        fmin: float,
        fmax: float,
        sr: int,
        n_pitch_bins: int,
        voiced_prob: float,
        v2u: float,
        u2v: float,
        transition_width: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
        framedata:
        fmin:
        fmax:
        sr:
        n_pitch_bins:
        voiced_prob:
        v2u:
        u2v:
        transition_width:

    Returns:
        a tuple (f0: np.ndarray[float], path: np.ndarray[int], confidence: np.ndarray[float])

    """
    # This is currently the fastest version, 2 is next and 1 is slowest, but for
    # little margin
    logfreq = _logFreq(fmin, fmax, n_pitch_bins)
    cents_per_bin = (logfreq[1] - logfreq[0]) * 1200.0
    sigma = transition_width / cents_per_bin
    half_w = int(np.ceil(4.0 * sigma))

    UNVOICED = n_pitch_bins
    n_frames = len(framedata)

    log_p_v2v = np.log(1.0 - v2u)
    log_p_v2u = np.log(v2u)
    log_p_u2v = np.log(u2v)
    log_p_u2u = np.log(1.0 - u2v)

    obs_voiced, obs_unvoiced = _emissions(framedata, logfreq, fmin, fmax, sr)

    # --- Banded transition kernel (2*half_w+1 wide instead of n_pitch_bins wide) ---
    offsets = np.arange(-half_w, half_w + 1, dtype=float)
    gauss_kernel = log_p_v2v - 0.5 * (offsets / sigma) ** 2  # (band_size,)
    band_size = len(offsets)

    # Padded prev_v array (pad with -inf so out-of-range bins never win)
    pad_width = half_w

    dp = np.full((n_frames, n_pitch_bins + 1), -np.inf)
    bp = np.zeros((n_frames, n_pitch_bins + 1), dtype=np.int32)

    dp[0, :n_pitch_bins] = obs_voiced[0]   + np.log(np.clip(voiced_prob,       1e-9, 1 - 1e-9))
    dp[0, UNVOICED]      = obs_unvoiced[0] + np.log(np.clip(1.0 - voiced_prob, 1e-9, 1 - 1e-9))

    padded = np.full(n_pitch_bins + 2 * pad_width, -np.inf)
    # Only allocate once

    for t in range(1, n_frames):
        prev_v = dp[t - 1, :n_pitch_bins]
        prev_u = dp[t - 1, UNVOICED]

        # Pad prev_v with -inf for boundary bins
        # padded = np.full(n_pitch_bins + 2 * pad_width, -np.inf)
        padded[:] = -np.inf
        padded[pad_width:pad_width + n_pitch_bins] = prev_v

        # Strided windows: each row j is the slice of prev_v that can transition to bin j
        strides = (padded.strides[0], padded.strides[0])
        windows = np.lib.stride_tricks.as_strided(
            padded,
            shape=(n_pitch_bins, band_size),
            strides=strides,
        )  # (n, band_size)

        scores = windows + gauss_kernel[np.newaxis, :]  # (n, band_size)
        local_argmax = np.argmax(scores, axis=1)        # (n,)
        best_from_v  = scores[np.arange(n_pitch_bins), local_argmax]  # (n,)
        bp_from_v    = (np.arange(n_pitch_bins) - half_w + local_argmax).clip(0, n_pitch_bins - 1)

        # unvoiced -> voiced
        from_u = prev_u + log_p_u2v
        take_u = from_u > best_from_v

        dp[t, :n_pitch_bins] = obs_voiced[t] + np.where(take_u, from_u, best_from_v)
        bp[t, :n_pitch_bins] = np.where(take_u, UNVOICED, bp_from_v)

        # unvoiced state
        argmax_v   = int(np.argmax(prev_v))
        max_prev_v = prev_v[argmax_v]

        from_v_to_u = max_prev_v + log_p_v2u
        from_u_to_u = prev_u     + log_p_u2u

        if from_v_to_u > from_u_to_u:
            dp[t, UNVOICED] = obs_unvoiced[t] + from_v_to_u
            bp[t, UNVOICED] = argmax_v
        else:
            dp[t, UNVOICED] = obs_unvoiced[t] + from_u_to_u
            bp[t, UNVOICED] = UNVOICED

    path     = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    # --- Confidence: voiced prob relative to total mass at each frame ---
    # log-sum-exp over all voiced bins, then normalise against unvoiced
    import scipy.special
    log_voiced_total = scipy.special.logsumexp(dp[:, :n_pitch_bins], axis=1)  # (n_frames,)
    log_unvoiced     = dp[:, UNVOICED]                                         # (n_frames,)
    log_total        = np.logaddexp(log_voiced_total, log_unvoiced)            # (n_frames,)
    confidence       = np.exp(log_voiced_total - log_total)                    # (n_frames,) in [0,1]

    f0 = np.where(
        path < n_pitch_bins,
        2.0 ** logfreq[np.clip(path, 0, n_pitch_bins - 1)],
        np.nan,
    )
    return f0, path, confidence


def pyin(
        samples: np.ndarray,
        sr: int,
        minFreq: float = 80.0,
        maxFreq: float = 1000.0,
        frameSize: int = 2048,
        hopSize: int = 0,
        window: str = "hann",
        bins: int = 360,
        voicedProb: float = 0.5,
        v2u: float = 0.1,
        u2v: float = 0.4,
        pitchDrift: float = 100.0,
        normalize=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the fundamental frequency (F0) using pYIN.

    =====  ======================================================
    v2u    Description
    =====  ======================================================
    0.3    Drops out fast; sensitive to noisy frames
    0.1    Balanced default
    0.05   Sticky – survives short noise bursts
    0.01   Very sticky – for clean continuous speech
    =====  ======================================================

    =====  ======================================================
    u2v    Description
    =====  ======================================================
    0.1    Slow – may miss phrase onsets
    0.4    Balanced default – ≈ 1–2 frame latency
    0.7    Fast – snaps back within one frame
    =====  ======================================================

    Args:
        samples: Mono audio array (float32 or float64, any amplitude).
        sr : Sample rate in Hz.
        minFreq: min. frequency allowed for f0
        maxFreq: max. frequency allowed for f0
        frameSize: FFT frame size
        hopSize: frame hop in samples; defaults to frameSize // 4.
        window : Window function name (default "hann").
        bins: Log-spaced F0 bins (default 360 -> ~12 cents/bin over 80–1000 Hz).
        v2u: voiced->unvoiced probability per frame. Lower = fewer gaps; signal can
            stay voiced through brief noise.
        u2v: Unvoiced->voiced probability per frame (default 0.4).
            Increase to fix slow re-entry after silence (higher=faster recovery after
            silences or consonants)
        pitchDrift: Gaussian log-cost std-dev in cents. Controls frame-to-frame pitch agility.
        normalize: if True, normalize data before analysis. Data should be normalized or
            compressed for effective analysis

    Returns:
        a tuple of (times, f0, confidence, voiced), where:
        * times: frame centre times in seconds,
        * f0: f0 in hz at each frame
        * voiced: True where pitch was detected, for each frame.

    The key design decision is the **un-normalised Gaussian log-cost** for
    pitch transitions.  Any row-normalised Gaussian over n ≈ 360 bins
    assigns only ~1/n of its mass to each off-diagonal entry, making a
    3-bin (~37 cent) move cost ≈ −5.9 nats — far more than the
    unvoiced->voiced transition cost of log(p_u2v) ≈ −0.9 nats.  The
    Viterbi then prefers the cheap U->V path at every pitch fluctuation,
    creating U/V flickering regardless of how p_v2u and p_u2v are tuned.

    With the un-normalised cost::

        cost[i, j] = log(p_v2v) − 0.5 · ((i−j) · Δ / σ)²

    The self-transition costs log(p_v2v) ≈ −0.1 nats and a 3-bin move
    costs only −0.1 − 0.05 = −0.15 nats — always cheaper than U->V at −0.9.
    """
    analysis = Analysis(samples=samples, sr=sr, frameSize=frameSize, hopSize=hopSize,
                        minFreq=minFreq, maxFreq=maxFreq, window=window, normalize=normalize)
    analysis.addFeature(PyinFeature(minFreq=minFreq, maxFreq=maxFreq, bins=bins,
                                    voicedProb=voicedProb, v2u=v2u, u2v=u2v,
                                    pitchDrift=pitchDrift))
    out = analysis.run()
    times = out['times']
    f0 = out['pyin.f0']
    confidence = out['pyin.confidence']
    voiced = ~np.isnan(f0)
    return times, f0, confidence, voiced


def analyze(
        samples: np.ndarray,
        sr: int,
        frameSize: int = 2048,
        hopSize: int = 0,
        features = ('pyin', 'flatness', 'rolloff', 'focus'),
        minFreq: float = 80.0,
        maxFreq: float = 1000.0,
        window: str = "hann",
        bins: int = 360,
        voicedProb: float = 0.5,
        v2u: float = 0.1,
        u2v: float = 0.4,
        pitchDrift: float = 100.0,
        focusBins: int = 0,
        focusMinFreq: float = 0.,
        focusMaxFreq: float = 0.,
        rolloff: float = 0.85,
        normalize=True
) -> dict:
    """
    Analyzes multiple features

    Args:
        samples: the samples to analyze (1D audio array)
        sr: sample rate
            'flatness', 'centroid', 'rolloff' and 'focus'
        frameSize: window size
        hopSize: hop size in samples
        features: the features to analyze. Available features:
            'pyin', 'flatness', 'centroid', 'rolloff', 'focus'
        minFreq: min. frequency to considere for fundamental tracking
        maxFreq: max. frequency to considere for fundamental tracking
        window: kind of window to use
        bins: number of bins used for fundamental tracking
        voicedProb: initial probability when determining voiced vs unvoiced state
        v2u: voice -> unvoiced probability (pyin)
        u2v: unvoiced -> voiced probability (pyin)
        pitchDrift: drift in cents between frames (pyin)
        focusBins: number of bins to use when calculating the ratio between the loudest
            bins and the total energy (focus)
        focusMinFreq: min. frequency used for spectral focus, defaults to minFreq * 2 (focus)
        focusMaxFreq: max. frequency used for spectral focus, defaults to sr * 0.9 (focus)
        rolloff: rolloff ratio.

    Returns:
        a dict with all given features, plus the times corresponding to each frame. Each feature
        is a key: value pair in the form `str`: `np.ndarray` where the array holds the value of
        the given feature corresponding to the time of the frame. All arrays have the same size

    """
    analysis = Analysis(samples=samples, sr=sr, frameSize=frameSize, hopSize=hopSize,
                        minFreq=minFreq, maxFreq=maxFreq, window=window, normalize=normalize)
    for feature in features:
        if isinstance(feature, AnalysisFeature):
            analysis.addFeature(feature)
        elif feature == 'pyin':
            analysis.addFeature(PyinFeature(minFreq=minFreq, maxFreq=maxFreq, bins=bins,
                                            voicedProb=voicedProb, v2u=v2u, u2v=u2v,
                                            pitchDrift=pitchDrift))
        elif feature == 'flatness':
            analysis.addFeature(FlatnessFeature())
        elif feature == 'centroid':
            analysis.addFeature(CentroidFeature())
        elif feature == 'rolloff':
            analysis.addFeature(RolloffFeature(rolloff=rolloff))
        elif feature == 'focus':
            analysis.addFeature(FocusFeature(focusBins=focusBins,
                                             focusMinFreq=focusMinFreq,
                                             focusMaxFreq=focusMaxFreq))
        else:
            raise ValueError(f"Unknown feature {feature!r}")
    return analysis.run()
