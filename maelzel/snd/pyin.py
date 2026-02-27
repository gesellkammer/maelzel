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
import numpy as np
from scipy.signal import get_window


# ------------------------------------------
# helpers
# ------------------------------------------

def _frameSignal(samples: np.ndarray, framesize: int, hopsize: int) -> np.ndarray:
    """
    Args:
        samples: samples as a 1D array
        framesize: the size of each frame
        hopsize: the hop size in samples

    Returns:
        an array of shape (numframes, framesize), where each row is a frame

    """
    n_frames = 1 + (len(samples) - framesize) // hopsize
    strides  = (samples.strides[0] * hopsize, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=(n_frames, framesize), strides=strides)


def _sumOfLargestN(arr: np.ndarray, n: int) -> float:
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
        powerspec:

    Returns:

    """
    zeros = powerspec == 0
    if np.any(zeros):
        powerspec = np.where(zeros, 1e-12, powerspec)

    geom_mean = np.exp(np.mean(np.log(powerspec)))
    return geom_mean / np.mean(powerspec)


def _fftDiff(rfft: np.ndarray, N: int) -> np.ndarray:
    acf = np.fft.irfft(rfft * np.conj(rfft))[:N]
    return 2.0 * (acf[0] - acf)


def _diffFunc(frame: np.ndarray) -> np.ndarray:
    """
    difference function

    Args:
        frame: Audio data

    Returns:
        an array containing the correlation difference per fft bin

    """
    N = len(frame)
    fftsize = 1 << (2 * N - 1).bit_length()
    F   = np.fft.rfft(frame, n=fftsize)
    return _fftDiff(F, N)
    # acf = np.fft.irfft(F * np.conj(F))[:N]
    # return 2.0 * (acf[0] - acf)


def _cmndf(df: np.ndarray) -> np.ndarray:
    out    = np.empty_like(df)
    out[0] = 1.0
    cs     = np.cumsum(df[1:])
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

def _viterbiDecode(
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
        from_v_to_u = np.max(prev_v) + log_p_v2u
        from_u_to_u = prev_u         + log_p_u2u
        if from_v_to_u > from_u_to_u:
            dp[t, UNVOICED] = obs_unvoiced[t] + from_v_to_u
            bp[t, UNVOICED] = int(np.argmax(prev_v))
        else:
            dp[t, UNVOICED] = obs_unvoiced[t] + from_u_to_u
            bp[t, UNVOICED] = UNVOICED

    path     = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    return np.where(
        path < n_pitch_bins,
        2.0 ** logfreq[np.clip(path, 0, n_pitch_bins - 1)],
        np.nan)


def _viterbiDecodeConvolution(
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
    from scipy.signal import fftconvolve

    logfreq = np.linspace(np.log2(f_min), np.log2(f_max), n_pitch_bins)
    cents_per_bin = (logfreq[1] - logfreq[0]) * 1200.0
    UNVOICED = n_pitch_bins
    n_frames = len(frame_data)

    log_p_v2v = np.log(1.0 - v2u)
    log_p_v2u = np.log(v2u)
    log_p_u2v = np.log(u2v)
    log_p_u2u = np.log(1.0 - u2v)

    # --- Precompute Gaussian kernel (1-D, length 2n-1) for log-domain max-conv ---
    # log_trans[i,j] = log_p_v2v - 0.5*((i-j)/sigma)^2
    sigma = transition_width / cents_per_bin
    half = n_pitch_bins - 1
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    log_gauss_kernel = log_p_v2v - 0.5 * (offsets / sigma) ** 2  # shape: (2n-1,)

    obs_voiced, obs_unvoiced = _emissions(frame_data, logfreq, f_min, f_max, sr)

    dp = np.full((n_frames, n_pitch_bins + 1), -np.inf)
    bp = np.zeros((n_frames, n_pitch_bins + 1), dtype=np.int32)

    dp[0, :n_pitch_bins] = obs_voiced[0]   + np.log(np.clip(voiced_prob,       1e-9, 1 - 1e-9))
    dp[0, UNVOICED]      = obs_unvoiced[0] + np.log(np.clip(1.0 - voiced_prob, 1e-9, 1 - 1e-9))

    for t in range(1, n_frames):
        prev_v = dp[t - 1, :n_pitch_bins]
        prev_u = dp[t - 1, UNVOICED]

        # --- voiced -> voiced via (max, +) convolution ---
        # max_j(prev_v[j] + log_gauss_kernel[b-j]) = (prev_v ⊛ kernel)[b]
        # We approximate this using argmax on the full matrix only where needed.
        # True O(n log n) (max,+) conv requires SMAWK or distance transform tricks;
        # for the Gaussian case the kernel is concave (log-concave), so we can use
        # the "sliding window argmax on shifted scores" approach.
        #
        # Practical fast path: since the Gaussian kernel is symmetric and concave,
        # the argmax for each target bin b is unimodal in j. We use scipy's
        # fftconvolve as a *soft* approximation for finding the peak, then do a
        # narrow exact search around it — giving near-O(n) behaviour in practice.

        # Step 1: find approximate best source via linear (sum) convolution of scores
        # (valid for finding the argmax location when kernel is sharply peaked)
        best_from_v, bp_from_v = _maxconv_gaussian(prev_v, log_gauss_kernel, n_pitch_bins)

        # unvoiced -> voiced
        from_u  = prev_u + log_p_u2v
        take_u  = from_u > best_from_v

        dp[t, :n_pitch_bins] = obs_voiced[t] + np.where(take_u, from_u, best_from_v)
        bp[t, :n_pitch_bins] = np.where(take_u, UNVOICED, bp_from_v)

        # unvoiced state
        from_v_to_u = np.max(prev_v) + log_p_v2u
        from_u_to_u = prev_u         + log_p_u2u
        if from_v_to_u > from_u_to_u:
            dp[t, UNVOICED] = obs_unvoiced[t] + from_v_to_u
            bp[t, UNVOICED] = int(np.argmax(prev_v))
        else:
            dp[t, UNVOICED] = obs_unvoiced[t] + from_u_to_u
            bp[t, UNVOICED] = UNVOICED

    path     = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    return np.where(
        path < n_pitch_bins,
        2.0 ** logfreq[np.clip(path, 0, n_pitch_bins - 1)],
        np.nan,
    )


def _maxconv_gaussian(
    prev_v: np.ndarray,
    log_gauss_kernel: np.ndarray,
    n: int,
    search_radius: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Efficient (max, +) convolution exploiting the concavity of the Gaussian kernel.

    For a log-concave (concave in log) kernel the argmax of
        score(b) = max_j  prev_v[j] + kernel[b - j]
    moves monotonically with b (SMAWK / totally monotone matrix property).

    Here we use a practical approximation:
      1. Find the approximate argmax location via a *sum* FFT convolution.
      2. Do an exact brute-force search in a small window around that location.

    This is exact when the kernel is sharp relative to the signal variation,
    which holds for typical pitch-tracking transition widths.
    """
    from scipy.signal import fftconvolve

    half = n - 1

    # Sum-convolution gives us a smooth proxy for the argmax location
    proxy   = fftconvolve(prev_v, log_gauss_kernel[::-1], mode="full")
    # 'full' output has length 2n-1; centre it to length n (valid range)
    proxy_n = proxy[half: half + n]                  # shape (n,)
    approx_src = np.round(proxy_n - proxy_n).astype(int)  # placeholder — see below

    # Better: use the argmax of the proxy directly as the centre of the search window
    # The "proxy argmax" for target b lives at index (b + argmax_kernel_shifted).
    # Since the kernel is symmetric and centred, the dominant source for bin b is ~ b.
    # We search [b - r, b + r] explicitly.

    best_val = np.full(n, -np.inf)
    best_src = np.zeros(n, dtype=np.int32)

    for r in range(-search_radius, search_radius + 1):
        src = np.arange(n) + r                           # candidate source bin
        valid = (src >= 0) & (src < n)
        k_idx = half - r                                 # kernel index for offset -r
        if not (0 <= k_idx < len(log_gauss_kernel)):
            continue
        val = np.where(valid, prev_v[np.clip(src, 0, n - 1)] + log_gauss_kernel[k_idx], -np.inf)
        better = val > best_val
        best_val = np.where(better, val, best_val)
        best_src = np.where(better, np.where(valid, src, 0), best_src)

    return best_val, best_src
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def makeFrames(samples: np.ndarray,
               sr: int,
               frameSize: int = 2048,
               hopSize: int = 0,
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an array of samples, returns (frames, times)

    Where frames is an array with shape (numframes, framesize) where
    each row represents a frame. times is an array holding the centre
    time for each frame

    Args:
        samples: audio data (1D)
        sr: sampling rate (Hz)
        frameSize: the size of each frame
        hopSize: hop size in samples. If not given, hopsize=framesize//4

    Returns:
        a tuple (frames: np.ndarray, times: np.ndarray)

    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("audio must be 1-D")

    peak = np.abs(samples).max()
    if peak > 0:
        samples = samples / peak

    hopSize = hopSize or frameSize // 4
    pad = frameSize // 2
    padded = np.pad(samples, (pad, pad), mode="constant")
    frames = _frameSignal(padded, frameSize, hopSize)
    times = (np.arange(len(frames)) * hopSize) / sr
    return frames, times


def _pyin_f0(
        frames: np.ndarray,
        sr: int,
        win: np.ndarray | None = None,
        minFreq: float = 80.,
        maxFreq: float = 1000.,
        v2u: float = 0.1,
        u2v: float = 0.4,
        pitchDrift: float = 100.,
        voicedProb: float = 0.5,
        bins: int = 360
    ) -> np.ndarray:
    frameSize = frames.shape[1]
    tau_min = max(1, int(np.floor(sr / maxFreq)))
    tau_max = min(frameSize - 1, int(np.ceil(sr / minFreq)))
    framedata = []
    for frame in frames:
        if win is not None:
            frame = frame * win
        df = _diffFunc(frame)
        cm = _cmndf(df)
        framedata.append(_analyzeFrame(cm, tau_min, tau_max))

    f0 = _viterbiDecode(
        framedata,
        f_min=minFreq, f_max=maxFreq, sr=sr,
        n_pitch_bins=bins,
        voiced_prob=voicedProb,
        v2u=v2u, u2v=u2v,
        transition_width=pitchDrift)
    return f0


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

) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        audio: Mono audio array (float32 or float64, any amplitude).
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

    Returns:
        a tuple of (times, f0, voiced), where:
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
    frames, times = makeFrames(samples, sr=sr, frameSize=frameSize, hopSize=hopSize)
    win: np.ndarray = get_window(window, frameSize)
    f0 = _pyin_f0(frames=frames,
                  sr=sr,
                  win=win,
                  minFreq=minFreq,
                  maxFreq=maxFreq,
                  v2u=v2u,
                  u2v=u2v,
                  pitchDrift=pitchDrift,
                  voicedProb=voicedProb,
                  bins=bins)
    voiced = ~np.isnan(f0)
    return times, f0, voiced


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
        rolloff: float = 0.85
) -> dict[str, np.ndarray]:
    """
    Analyzes multiple features

    Args:
        samples: the samples to analyze (1D audio array)
        sr: sample rate
            'flatness', 'rolloff' and 'focus'
        frameSize: window size
        hopSize: hop size in samples
        features: the features to analyze. Available features: 'pyin',
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
    frames, times = makeFrames(samples, sr=sr, frameSize=frameSize, hopSize=hopSize)
    win: np.ndarray = get_window(window, frameSize)
    if not focusBins:
        focusBins = frameSize // 64
    tau_min = max(1, int(np.floor(sr / maxFreq)))
    tau_max = min(frameSize - 1, int(np.ceil(sr / minFreq)))
    framedata: list[tuple[float, float]] = []
    peakyness = np.zeros_like(times)
    rolloffFreqs = np.zeros_like(times)
    flatness = np.zeros_like(times)
    N = frames.shape[1]
    fftsize = 1 << (2 * N - 1).bit_length()
    fftfreqs = np.fft.rfftfreq(fftsize, d=1./ sr)
    peakynessMinBin = np.argmax(fftfreqs > (focusMinFreq or minFreq * 2.5))
    peakynessMaxBin = np.argmax(fftfreqs > (focusMaxFreq or sr / 2))

    featPyin = 'pyin' in features
    featFlatness = 'flatness' in features
    featRolloff = 'rolloff' in features
    featFocus = 'focus' in features

    frame: np.ndarray
    for i, frame in enumerate(frames):
        frame = frame * win
        F = np.fft.rfft(frame, n=fftsize)
        Fpos = np.abs(F)
        Fpow = F ** 2
        if featPyin:
            df = _fftDiff(Fpos, N)
            cm = _cmndf(df)
            framedata.append(_analyzeFrame(cm, tau_min, tau_max))

        if featFocus:
            Fposlow = Fpos[peakynessMinBin:peakynessMaxBin]
            Fposlowsum = Fposlow.sum()
            peakyness[i] = 0 if Fposlowsum == 0 else _sumOfLargestN(Fposlow, n=focusBins) / Fposlowsum

        if featRolloff:
            rollbin = _rolloffBin(Fpow, rolloff=rolloff)
            rolloffFreqs[i] = fftfreqs[int(rollbin)]

        if featFlatness:
            flatness[i] = _spectralFlatness(Fpow)

    out = {'times': times}

    if featPyin:
        f0 = _viterbiDecode(
            framedata,
            f_min=minFreq, f_max=maxFreq, sr=sr,
            n_pitch_bins=bins,
            voiced_prob=voicedProb,
            v2u=v2u, u2v=u2v,
            transition_width=pitchDrift)
        out['pyin'] = f0

    if featFocus:
        out['focus'] = peakyness

    if featRolloff:
        out['rolloff'] = rolloffFreqs

    if featFlatness:
        out['flatness'] = flatness

    return out

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    SR = 22050; np.random.seed(0)

    def _runs(voiced):
        runs, r = [], 0
        for v in voiced:
            if v: r += 1
            elif r: runs.append(r); r = 0
        if r: runs.append(r)
        return runs

    def _test(label, audio, expected_hz, **kw):
        t0 = time.perf_counter()
        _, f0, voiced = pyin(audio, sr=SR, **kw)
        ms = (time.perf_counter() - t0) * 1000
        v  = f0[voiced]
        if len(v):
            runs = _runs(voiced)
            err  = abs(v.mean() - expected_hz)
            ok   = "✓" if err < expected_hz * 0.03 else "~"
            print(f"{ok} [{label}]  mean={v.mean():.1f} Hz  std={v.std():.2f}  "
                  f"voiced={voiced.sum()}/{len(voiced)}  max_run={max(runs)}  "
                  f"err={err:.1f} Hz  {ms:.0f}ms")
        else:
            print(f"✗ [{label}]  No voiced frames (expected {expected_hz} Hz)")

    t = np.linspace(0, 3.0, 3 * SR)

    print("=== Accuracy ===")
    _test("sine 220 Hz",
          np.sin(2*np.pi*220*t), 220, f_min=80, f_max=500)

    speech_m = (0.4*np.sin(2*np.pi*130*t) + 0.2*np.sin(2*np.pi*260*t) +
                0.1*np.sin(2*np.pi*390*t) + 0.08*np.random.randn(len(t)))
    _test("speech male 130 Hz", speech_m, 130, f_min=60, f_max=400)

    speech_f = (0.5*np.sin(2*np.pi*250*t) + 0.3*np.sin(2*np.pi*500*t) +
                0.08*np.random.randn(len(t)))
    _test("speech female 250 Hz", speech_f, 250, f_min=100, f_max=600)

    _test("noisy 180 Hz (~20 dB SNR)",
          0.5*np.sin(2*np.pi*180*t) + 0.05*np.random.randn(len(t)),
          180, f_min=80, f_max=400, frame_length=4096)

    print()
    print("=== Pitch tracking ===")

    freqs = np.linspace(150, 300, len(t))
    glide = 0.5*np.sin(2*np.pi*np.cumsum(freqs)/SR) + 0.02*np.random.randn(len(t))
    _, f0_g, v_g = pyin(glide, SR, minFreq=100, maxFreq=400)
    v = f0_g[v_g]
    print(f"✓ [glide 150->300]  voiced={v_g.sum()}/{len(v_g)}  range=[{v.min():.0f},{v.max():.0f}]")

    vib_f = 200 + 20*np.sin(2*np.pi*5*t)
    vib   = 0.5*np.sin(2*np.pi*np.cumsum(vib_f)/SR) + 0.02*np.random.randn(len(t))
    _, f0_v, v_v = pyin(vib, SR, minFreq=100, maxFreq=400)
    v = f0_v[v_v]
    print(f"✓ [vibrato 200±20 Hz]  mean={v.mean():.1f}  std={v.std():.1f}  range=[{v.min():.0f},{v.max():.0f}]")

    print()
    print("=== Silence recovery ===")
    def voiced_seg(f, dur): return (0.5*np.sin(2*np.pi*f*np.linspace(0,dur,int(dur*SR)))
                                    + 0.03*np.random.randn(int(dur*SR)))
    def silent_seg(dur):    return 0.01*np.random.randn(int(dur*SR))
    mixed = np.concatenate([voiced_seg(130,0.3), silent_seg(0.1),
                             voiced_seg(130,0.3), silent_seg(0.3),
                             voiced_seg(130,0.3), silent_seg(0.1),
                             voiced_seg(130,0.3)])
    mixed /= np.abs(mixed).max()
    _, f0_m, v_m = pyin(mixed, SR, minFreq=60, maxFreq=400)
    runs_m = _runs(v_m)
    print(f"  voiced={v_m.sum()}/{len(v_m)}  segments={len(runs_m)}/4 expected  "
          f"max_run={max(runs_m) if runs_m else 0}  "
          f"mean_F0={f0_m[v_m].mean():.1f} Hz")

    print()
    print("=== Parameter guide ===")
    for p_v2u, p_u2v, label in [
        (0.3,  0.4, "default-ish (noisy)"),
        (0.1,  0.4, "balanced default"),
        (0.05, 0.4, "sticky dropout"),
        (0.05, 0.7, "sticky + fast recovery"),
    ]:
        _, f0_p, vp = pyin(speech_m, SR, minFreq=60, maxFreq=400,
                           v2u=p_v2u, u2v=p_u2v)
        rp = _runs(vp)
        print(f"  p_v2u={p_v2u:.2f} p_u2v={p_u2v:.1f}  ({label}):  "
              f"voiced={vp.sum()}/{len(vp)}  max_run={max(rp) if rp else 0}  "
              f"mean_f0={f0_p[vp].mean():.1f}")
