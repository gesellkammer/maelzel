"""
Frequency estimation of a signal with different algorithms

The most important entry point is :func:`~maelzel.snd.freqestimate.f0curve`, which estimates
the fundamental frequency of an audio signal together with its voicedness (the reliability
of the measurement)

"""
from __future__ import annotations
import numpy as np
import bpf4
from math import ceil
from maelzel._logutils import getLogger


def _nextPow2(x: int) -> int:
    return 1 if x == 0 else 2**(x-1).bit_length()


def f0curvePyin(sig: np.ndarray, sr: int, minfreq=50, maxfreq=2000,
                framesize=2048, winsize=None, hop=512,
                normalize=True
                ) -> tuple[bpf4.Linear, bpf4.Linear]:
    """
    Calculate the fundamental based on the pyin method

    This routine is implemented in pure python and is VERY slow

    Args:
        sig: the array representing the soud
        sr: sample rate
        minfreq: lowest frequency to considere for f0
        maxfreq: highest frequecy to considere for f0
        framesize: the fft size
        winsize: the window length (must be smaller than framelength)
        hop: interval between measurements (in samples)

    Returns:
        a tuple (freq curve, voiced probability curve)

    """
    if _sigNumChannels(sig) > 1:
        raise ValueError("sig should be a mono signal")

    totaldur = len(sig) / sr

    from maelzel.snd import pyin
    times, f0, confidence, voiced = pyin.pyin(sig, sr=sr,
                                              minFreq=minfreq, maxFreq=maxfreq,
                                              frameSize=framesize, hopSize=hop,
                                              normalize=normalize)

    times = np.linspace(0, totaldur, len(f0))
    return bpf4.Linear(times, f0), bpf4.Linear(times, confidence)


def f0curvePyinVamp(sig: np.ndarray,
                    sr: int,
                    fftsize=2048,
                    overlap=4,
                    lowAmpSuppression=0.01,
                    onsetSensitivity=0.7,
                    pruneThreshold=0.1,
                    threshDistr='beta15',
                    unvoicedFreqs='nan'
                    ) -> tuple[bpf4.Linear, bpf4.Linear]:
    """
    Calculate the fundamental using the pyin vamp plugin

    Args:
        sig: the signal as numpy array
        sr: the sr
        fftsize: with sizes lower than 2048 the result might be unstable
        overlap: hop size as fftsize//overlap
        lowAmpSuppression: supress low amplitude pitch estimates, 0.01=-40dB, 0.001=-60dB
        onsetSensitivity: onset sensitivity
        pruneThreshold: totalDuration pruning threshold
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

    if fftsize < 2048:
        getLogger(__file__).warning(f"The fft size ({fftsize}) is too small for f0 tracking, it needs to be"
                                    f" at least 2048. Using 2048 instead")
        fftsize = 2048

    from maelzel.snd import vamptools
    data = vamptools.pyinPitchTrack(sig, sr=sr, fftSize=fftsize, overlap=overlap,
                                    lowAmpSuppression=lowAmpSuppression,
                                    onsetSensitivity=onsetSensitivity,
                                    pruneThresh=pruneThreshold,
                                    threshDistr=threshDistr,
                                    outputUnvoiced=unvoicedFreqs)
    times = np.ascontiguousarray(data[:, 0])
    f0 = np.ascontiguousarray(data[:, 1])
    probs = np.ascontiguousarray(data[:, 2])
    return bpf4.Linear(times, f0), bpf4.Linear(times, probs)


def _sigNumChannels(sig: np.ndarray):
    return sig.shape[1] if len(sig.shape) > 1 else 1


def frequencyToWindowSize(freq: int, sr: int, powerof2=False, factor=2.0) -> int:
    """
    Return the size of a window in samples which can fit the given frequency

    Args:
        freq: the lowest frequency to fit in the window
        sr: the sampling rate
        powerof2: if True, force the size to the next power of two
        factor: the factor used to fit to account for phase shifts. Without this
            the window will not fit an entire cycle of the given frequency unless
            the signal is in perfect sync with the window.

    Returns:
        the size of the window in samples
    """
    winsize = int(ceil(1/freq  * sr * factor))
    if powerof2:
        from emlib.mathlib import nextpowerof2
        winsize = nextpowerof2(winsize)
    return winsize


def f0curve(sig: np.ndarray, sr: int, minfreq=60, overlap=4,
            method='pyin', unvoicedFreqs='nan'
            ) -> tuple[bpf4.Linear, bpf4.Linear]:
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
        overlap: the amount of overlap between analysis windows
        method: the method used. One of 'pyin', 'fft', 'hps', 'autocorrelation'.
            For pyin make sure that you are using the vamp plugin, since the
            fallback version (using an implementation based on librosa's version
            of the algorithm) is very slow at the moment. **If the vamp plugin
            is not available you should see a warning**
        unvoicedFreqs: (only relevant for method 'pyin'). One of 'nan', 'negative'.
            If 'nan' unvoiced  frequencies (frequencies for segments where the
            f0 confidence is too low) are given as `nan`. If 'negative' unvoiced
            freqs are given as negative.


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

    from maelzel.snd import vamptools
    if method == 'pyin':
        if vamptools.pyinAvailable():
            method = 'pyin-vamp'
        else:
            method = 'pyin-native'
            getLogger(__file__).warning("The pyin vamp plugin was not found. Falling back to"
                                        "the python version, this might be very slow")


    if method == 'pyin-native':
        winlength = frequencyToWindowSize(minfreq, sr=sr, powerof2=False)
        hoplength = winlength // overlap
        from emlib.mathlib import nextpowerof2
        fftlength = nextpowerof2(winlength)
        return f0curvePyin(sig, sr, minfreq=minfreq,
                           framesize=fftlength,
                           winsize=winlength, hop=hoplength)
    elif method == 'pyin-vamp':
        fftsize = frequencyToWindowSize(minfreq, sr=sr, powerof2=True)
        return f0curvePyinVamp(sig, sr, fftsize=fftsize, overlap=overlap,
                               unvoicedFreqs=unvoicedFreqs)
    else:
        raise ValueError(f"Invalid method {method}")


def _detectMinFrequency(samples: np.ndarray, sr: int, minFreq=30, overlap=4,
                        minAmp=0.1,
                        refine=True
                        ) -> tuple[float, float]:
    """
    Detect the min. frequency in this audio sample, using vamp's pyin

    Args:
        samples: the samples to analyze
        sr: the sample rate
        minFreq: the lowest freq. to considere
        overlap: the amount of overlap analysis per window
        minAmp: how much to supress low amplitudes
        refine: if True, refine the freq and time data. This allows to
            performe a more coarse pass with a low overlap (for example, 4)
            and then a short pass with a higher overlap to obtain a more
            accurate value

    Returns:
        a tuple (min. freq, corresponding time) or (0., 0.) if no pitched sound was detected

    """
    from maelzel.snd import vamptools
    fftsize = frequencyToWindowSize(minFreq, sr=sr)
    f0data = vamptools.pyinPitchTrack(samples, sr=sr, fftSize=fftsize, overlap=overlap,
                                      lowAmpSuppression=minAmp,
                                      outputUnvoiced='nan')
    freqs = f0data[:,1]
    mask = freqs > minFreq
    selected = freqs[mask]
    if len(selected) == 0:
        return 0., 0.
    idx = selected.argmin()
    time = float(f0data[:,0][mask][idx])
    if refine:
        margin = 0.2
        start = int((time - margin)*sr)
        end = int((time + margin)*sr)
        f, t = detectMinFrequency(samples[start:end], sr=sr, minFreq=minFreq,
                                  overlap=overlap*4, lowAmpSuppression=minAmp,
                                  refine=False)
        return f, time - margin + t
    return float(selected[idx]), time


def detectMinFrequency(samples: np.ndarray, sr: int, frameSize=2048, overlap=2,
                       lowAmpSuppression=0.1,
                       maxFreq=2000,
                       minFreq=30,
                       confidence=0.4,
                       refine=True
                       ) -> tuple[float, float]:
    """
    Detect the min. frequency in this audio sample

    Args:
        samples: the samples to analyze
        sr: the sample rate
        minFreq: the lowest freq. to considere
        overlap: the amount of overlap analysis per window
        confidence: min. confidence to consider
        lowAmpSuppression: how much to supress low amplitudes
        refine: if True, refine the freq and time data. This allows to
            performe a more coarse pass with a low overlap (for example, 4)
            and then a short pass with a higher overlap to obtain a more
            accurate value

    Returns:
        a tuple (min. freq, corresponding time) or (0., 0.) if no pitched sound was detected

    """
    from maelzel.snd import pyin
    from maelzel.snd.numpysnd import schmitt

    times, f0, confs, voiced = pyin.pyin(samples, sr=sr, minFreq=minFreq, maxFreq=maxFreq,
                                              frameSize=frameSize, hopSize=frameSize//overlap)
    mask = (confs > confidence) * (f0 > minFreq)
    selected = f0[mask]
    if len(selected) == 0:
        return 0., 0.
    idx = np.nanargmin(selected)
    time = float(times[mask][idx])
    freq = float(selected[idx])
    if refine:
        margin = 0.2
        start = int((time - margin)*sr)
        end = int((time + margin)*sr)
        f, t = detectMinFrequency(samples[start:end], sr=sr, minFreq=minFreq, maxFreq=maxFreq,
                                  overlap=overlap*4, lowAmpSuppression=lowAmpSuppression,
                                  refine=False)
        if f > 0:
            return f, time - margin + t
    return freq, time

