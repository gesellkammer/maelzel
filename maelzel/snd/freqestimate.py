"""
Frequency estimation of a signal with different algorithms

The most important entry point is :func:`~maelzel.snd.freqestimate.f0curve`, which estimates
the fundamental frequency of an audio signal together with its voicedness (the reliability
of the measurement)

"""
from __future__ import annotations
import numpy as np
import scipy.signal
import logging
import bpf4
from maelzel.snd import vamptools


logger = logging.getLogger(__name__)


def _nextPow2(x: int) -> int:
    return 1 if x == 0 else 2**(x-1).bit_length()


def parabolic(f: np.ndarray, x: int) -> tuple[float, float]:
    """
    Estimates inter-sample maximum

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


def _find(condition) -> np.ndarray:
    """Return the indices where ravel(condition) is true"""
    res, = np.nonzero(np.ravel(condition))
    return res


def f0ZeroCross(sig: np.ndarray, sr: int) -> tuple[float, float]:
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
    indices = _find((sig[1:] >= 0) & (sig[:-1] < 0))
    
    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    # crossings = indices
    
    # More accurate, using linear interpolation to find intersample 
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    
    # Some other interpolation based on neighboring points might be better. 
    # Spline, cubic, whatever
    return sr / np.mean(np.diff(crossings)), 1


def f0FFT(sig: np.ndarray, sr: int) -> tuple[float, float]:
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
    windowed = sig * scipy.signal.windows.blackmanharris(len(sig))
    f = np.fft.rfft(windowed)
    
    # Find the peak and interpolate to get a more accurate peak
    i = int(np.argmax(abs(f)))    # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(abs(f)), i)[0]
    
    # Convert to equivalent frequency
    return sr * true_i / len(windowed), 1


def f0Autocorr(sig: np.ndarray, sr: int) -> tuple[float, float]:
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
    corr = scipy.signal.fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[int(len(corr)/2):]
    
    # Find the first low point
    d = np.diff(corr)
    start = _find(d > 0)[0]
    
    # Find the next peak after the low point (other than 0 lag).  This bit is 
    # not reliable for long signals, due to the desired peak occurring between 
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak) 
    return sr / px, 1


def f0HPS(sig: np.ndarray, sr: int, maxharms=5) -> tuple[float, float]:
    """
    Estimate frequency using harmonic product spectrum (HPS)
    
    """
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")

    windowed = sig * scipy.signal.windows.blackmanharris(len(sig))
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


def f0curvePyin(sig: np.ndarray, sr: int, minfreq=50, maxfreq=5000,
                framelength=2048, winlength=None, hoplength=512
                ) -> tuple[bpf4.BpfInterface, bpf4.BpfInterface]:
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

    totaldur = len(sig) / sr

    #import librosa
    from maelzel.snd import rosita
    # f0, voiced_flag, voiced_probs = librosa.pyin(sig, sr=sr,
    f0, voiceg_flag, voiced_probs = rosita.pyin(sig, sr=sr,
                                                fmin=minfreq, fmax=maxfreq,
                                                frame_length=framelength,
                                                win_length=winlength,
                                                hop_length=hoplength)
    times = np.linspace(0, totaldur, len(f0))
    return bpf4.core.Linear(times, f0), bpf4.core.Linear(times, voiced_probs)


def _fftWinlength(sr: int, minfreq: int) -> int:
    return _nextPow2(int(sr/minfreq))


def f0curvePyinVamp(sig: np.ndarray, sr: int, fftsize=2048, overlap=4,
                    lowAmpSupression=0.1, onsetSensitivity=0.7,
                    pruneThreshold=0.1,
                    threshDistr='beta15',
                    unvoicedFreqs='nan'

                    ) -> tuple[bpf4.BpfInterface, bpf4.BpfInterface]:
    """
    Calculate the fundamental using the pyin vamp plugin

    Args:
        sig: the signal as numpy array
        sr: the sr
        fftsize: with sizes greater than 2048 the result might be unstable
        overlap: hop size as fftsize//overlap
        lowAmpSupression: supress low amplitude pitch estimates
        onsetSensitivity: onset sensitivity
        pruneThreshold: duration pruning threshold
        threshDistr: yin threshold distribution (see table below) - One of
            uniform, beta10, beta15, beta30, single10, single20-
        unvoicedFreqs: one of 'nan', 'negative'. If 'nan' unvoiced  frequencies
            (frequencies for segments where the f0 confidence is too low) are given
            as `nan`. If 'negative' unvoiced freqs are given as negative.

    Returns:
        a tuple (f0 bpf, probability bpf), where f0 is a bpf with the
        detected fundamental. Whenver the algorithms detects unvoiced
        (noise) or absence of a fundamental, the result is negative.

    ============   ============
    thresh_distr   Description
    ============   ============
    uniform        Uniform
    beta10         Beta (mean 0.10)
    beta15         Beta (mean 0.15)
    beta30         Beta (mean 0.30)
    single10       Single value 0.10
    single15       Single value 0.15
    single20       Single value 0.20
    ============   ============
        
    """
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")

    from maelzel.snd import vamptools
    data = vamptools.pyinPitchTrack(sig, sr=sr, fftSize=fftsize, overlap=overlap,
                                    lowAmpSuppression=lowAmpSupression,
                                    onsetSensitivity=onsetSensitivity,
                                    pruneThresh=pruneThreshold,
                                    threshDistr=threshDistr)
    times = np.ascontiguousarray(data[:, 0])
    f0 = np.ascontiguousarray(data[:, 1])
    probs = np.ascontiguousarray(data[:, 2])
    # Unvoiced estimates are given a negative frequency. Convert to nan
    # if needed
    if unvoicedFreqs == 'nan':
        f0[f0 <= 0] = np.nan

    return bpf4.core.Linear(times, f0), bpf4.core.Linear(times, probs)


def _sigNumChannels(sig: np.ndarray):
    return sig.shape[1] if len(sig.shape) > 1 else 1


def f0curve(sig: np.ndarray, sr: int, minfreq=100, steptime=0.01,
            method='pyin'
            ) -> tuple[bpf4.BpfInterface, bpf4.BpfInterface]:
    """
    Estimate the fundamental and its voicedness

    The voicedness curve indicates how accurate the frequency reading is. The
    value should drop whenever the signal is either too low (during silences)
    or it turns noisy (during consonants in the case of speech)

    This function combines all frequency estimation algorithms into one interface.
    For more detailed access to parameters of each method use the specific
    function (pyin vamp: :func:`f0curvePyinVamp`; pyin native: :func:`f0curvePyin`, etc)

    Both the f0 curve and the voicedness curve are returned as bpfs.
    See https://bpf4.readthedocs.io for more information about bpfs

    Args:
        sig: the signal, a 1D float numpy array with samples between -1:1
        sr: the sr
        minfreq: the min. frequency of the fundamental. Using this hint
            we estimate the best value for fft size / window size
        steptime: the time step between measurements
        method: the method used. One of 'pyin', 'fft', 'hps', 'autocorrelation'.
            For pyin make sure that you are using the vamp plugin, since the
            fallback version (using an implementation based on librosa's version
            of the algorithm) is very slow at the moment. **If the vamp plugin
            is not available you should see a warning**

    Returns:
        a tuple (f0curve, f0voicedness)

    Example
    ~~~~~~~

    .. code-block:: python

        from maelzel.snd.audiosample import Sample
        from maelzel.snd import freqestimate
        samp = Sample("snd/finneganswake-fragm01.flac").getChannel(0, contiguous=True)
        freq, voiced = freqestimate.f0curve(sig=samp.samples, sr=samp.sr, method='pyin')
        freq.plot(figsize=(18, 6))

    .. image:: ../assets/snd-freqestimate-f0curve.png

    For a more in-depth example see ``notebooks/analysis-pyin.ipynb``
    """
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")

    if method == 'pyin':
        if vamptools.pyinAvailable():
            method = 'pyin-vamp'
        else:
            method = 'pyin-native'
            logger.warning("The pyin vamp plugin was not found. Falling back to"
                           "the python version, this might be very slow")

    if method == 'pyin-native':
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
        'autocorrelation': f0Autocorr,
        'fft': f0FFT,
        'hps': f0HPS
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
