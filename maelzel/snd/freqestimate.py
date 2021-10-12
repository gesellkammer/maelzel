"""
Frequency estimation of a signal with different algorithms
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from scipy.signal.windows import blackmanharris
import logging
from typing import Tuple
import bpf4


logger = logging.getLogger(__name__)


def _nextPow2(x: int) -> int:
    return 1 if x == 0 else 2**(x-1).bit_length()


def parabolic(f:np.ndarray, x:int) -> Tuple[float, float]:
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    Args:
        f: a vector (an array of sampled values over a regular grid)
        x: index for that vector

    Returns:
        (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
   
    Example
    =======

    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
   
    >>> f = [2, 3, 1, 6, 4, 2, 3, 1]
   
    >>> parabolic(f, 3)
    Out[4]: (3.2142857142857144, 6.1607142857142856)
   
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return xv, yv


def find(condition) -> np.ndarray:
    """Return the indices where ravel(condition) is true"""
    res, = np.nonzero(np.ravel(condition))
    return res


def f0ViaZeroCrossings(sig:np.ndarray, sr:int) -> Tuple[float, float]:
    """Estimate frequency by counting zero crossings

    Args:
        sig: a sampled signal
        sr : sample rate

    Returns:
        the frequency of the signal
    """
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")

    # Find all indices right before a rising-edge zero crossing
    indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
    
    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    # crossings = indices
    
    # More accurate, using linear interpolation to find intersample 
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    
    # Some other interpolation based on neighboring points might be better. 
    # Spline, cubic, whatever
    return sr / np.mean(np.diff(crossings)), 1


def f0ViaFFT(sig:np.ndarray, sr:int) -> Tuple[float, float]:
    """Estimate frequency from peak of FFT

    Args:
        sig: a sampled signal (mono)
        sr : sample rate

    Returns:
        the frequency of the signal, the probability that the sound
        frequency is valid
    """
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = np.fft.rfft(windowed)
    
    # Find the peak and interpolate to get a more accurate peak
    i = int(np.argmax(abs(f)))    # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(abs(f)), i)[0]
    
    # Convert to equivalent frequency
    return sr * true_i / len(windowed), 1


def f0ViaAutocorrelation(sig:np.ndarray, sr:int) -> Tuple[float, float]:
    """
    Estimate frequency using autocorrelation

    Args:
        sig: a sampled signal
        sr : sample rate

    Returns:
        the frequency of the signal, the probability that the sound
        frequency is valid
    """
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")

    # Calculate autocorrelation (same thing as convolution, but with 
    # one input reversed in time), and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[int(len(corr)/2):]
    
    # Find the first low point
    d = np.diff(corr)
    start = find(d > 0)[0]
    
    # Find the next peak after the low point (other than 0 lag).  This bit is 
    # not reliable for long signals, due to the desired peak occurring between 
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak) 
    return sr / px, 1


def f0ViaHPS(sig:np.ndarray, sr:int, maxharms=5) -> Tuple[float, float]:
    """
    Estimate frequency using harmonic product spectrum (HPS)
    
    """
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")

    windowed = sig * blackmanharris(len(sig))
    c = abs(np.fft.rfft(windowed))
    freq = 0
    for x in range(2,maxharms):
        a = c[::x]  # Should average or maximum instead of decimating
        # a = max(c[::x],c[1::x],c[2::x])
        c = c[:len(a)]
        i = int(np.argmax(abs(c)))
        try:
            true_i = parabolic(abs(c), i)[0]
        except IndexError:
            return freq, 0
        freq = sr * true_i / len(windowed)
        c *= a
    return freq, 1


def f0curvePyin(sig: np.ndarray, sr:int, minfreq=50, maxfreq=5000,
                framelength=2048, winlength=None, hoplength=512
                ) -> Tuple[bpf4.BpfInterface, bpf4.BpfInterface]:
    """
    Calculate the fundamental based on the pyin method

    This routine is implemented in pure python and is VERY slow

    Args:
        sig: the array representing the soud
        sr: sample rate
        minfreq: lowest frequency to considere for f0
        maxfreq: highest frequecy to considere for f0
        framelength: the fft size
        winlength: the window length (must be smaller than framelength)
        hoplength: interval between measurements (in samples)

    Returns:
        a tuple (freq curve, voiced probability curve)

    """
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")

    import librosa
    totaldur = len(sig)/sr
    f0, voiced_flag, voiced_probs = librosa.pyin(sig, sr=sr,
                                                 fmin=minfreq, fmax=maxfreq,
                                                 frame_length=framelength,
                                                 win_length=winlength,
                                                 hop_length=hoplength)
    times = np.linspace(0, totaldur, len(f0))
    return bpf4.core.Linear(times, f0), bpf4.core.Linear(times, voiced_probs)


def _fftWinlength(sr: int, minfreq: int) -> int:
    return _nextPow2(int(sr/minfreq))

def f0curvePyinVamp(sig: np.ndarray, sr:int, fftsize=2048, overlap=4,
                    lowAmpSupression=0.1, onsetSensitivity=0.7,
                    pruneThreshold=0.1,
                    threshDistr='beta15'

                    ) -> Tuple[bpf4.BpfInterface, bpf4.BpfInterface]:
    """
    Calculate the fundamental using 'vamptools' and the pyin vamp plugin

    Args:
        sig: the signal as numpy array
        sr: the sr
        fftsize: with sizes greater than 2048 the result might be unstable
        overlap: hop size as fftsize//overlap
        lowAmpSupression: ??
        onsetSensitivity: ??
        pruneThreshold: ??
        threshDistr: one of beta10, beta15, beta20, beta30

    Returns:
        a tuple (f0 bpf, probability bpf), where f0 is a bpf with the
        detected fundamental. Whenver the algorithms detects unvoiced
        (noise) or absence of a fundamental, the result is negative.
        
    """
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")

    from maelzel.snd import vamptools
    data = vamptools.pyin_pitchtrack(sig, sr=sr, fft_size=fftsize, overlap=overlap,
                                     low_amp_suppression=lowAmpSupression,
                                     onset_sensitivity=onsetSensitivity,
                                     prune_thresh=pruneThreshold,
                                     thresh_distr=threshDistr)
    times = data[:, 0]
    f0 = data[:, 1]
    probs = data[:, 2]
    return bpf4.core.Linear(times, f0), bpf4.core.Linear(times, probs)


def _sigNumChannels(sig: np.ndarray):
    return sig.shape[1] if len(sig.shape) > 1 else 1


def f0curve(sig:np.ndarray, sr:int, minfreq=100, steptime=0.01,
            method='pyin-librosa'
            ) -> Tuple[bpf4.BpfInterface, bpf4.BpfInterface]:
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")

    if method == 'pyin-librosa':
        hoplength = int(steptime * sr)
        winlength = _fftWinlength(sr, minfreq)
        return f0curvePyin(sig, sr, minfreq=minfreq,
                           framelength=winlength*2,
                           winlength=winlength, hoplength=hoplength)
    elif method == 'pyin-vamp':
        for f0, f1, fftsize in [(0,   30, 8192),
                                (30,  60, 4096),
                                (60, 600, 2048),
                                (600, 99999, 1024)]:
            if f0 <= minfreq < f1:
                break
        return f0curvePyinVamp(sig, sr, fftsize=fftsize)
    stepsize = int(steptime*sr)
    windowsize = _fftWinlength(sr, minfreq)
    maxidx = len(sig) - windowsize
    maxn = int(maxidx / stepsize)
    func = {
        'autocorrelation': f0ViaAutocorrelation,
        'fft': f0ViaFFT,
        'hps': f0ViaHPS
    }[method]
    freqs, times, probs = [], [], []
    for n in range(maxn):
        idx = n * stepsize
        arr = sig[idx: idx+windowsize]
        freq, prob = func(arr, sr)
        freqs.append(freq)
        times.append(idx/sr)
        probs.append(prob)
    return bpf4.core.Linear(times, freqs), bpf4.core.Linear(times, probs)
