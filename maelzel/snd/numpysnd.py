"""
numpy utilities for audio arrays
"""
from __future__ import annotations
import numpy as np
import bpf4
from math import sqrt
from pitchtools import db2amp, amp2db
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


def asmono(samples: np.ndarray) -> np.ndarray:
    """
    Mix down a sample to mono. If it is already a mono sample, the sample itself is
    returned. Otherwise the channels are summed together. If the channels are very
    correlated this operation might result in samples exceeding 0dB

    Args:
        samples (np.ndarray): the samples

    Returns:
        a numpy array of shape (numframes,)

    """
    if numChannels(samples) == 1:
        return samples
    return samples.sum(0)


def rms(arr: np.ndarray) -> float:
    """
    Calculate the RMS for the whole array
    """
    arr = np.abs(arr)
    arr **= 2
    return sqrt(np.sum(arr) / len(arr))


def rmsbpf(samples: np.ndarray, sr:int, dt=0.01, overlap=1) -> bpf4.core.Sampled:
    """
    Return a bpf representing the rms of this sample as a function of time

    Args:
        samples: the audio samples
        sr: the sample rate
        dt: analysis time period
        overlap: overlap of analysis frames

    Returns:
        a samples bpf
    """
    s = samples
    period = int(sr * dt + 0.5)
    hopsamps = period // overlap
    dt2 = hopsamps / sr
    numperiods = len(s) // hopsamps
    data = np.empty((numperiods,), dtype=float)
    for i in range(numperiods):
        idx0 = i * hopsamps
        chunk = s[idx0:idx0+period]
        data[i] = rms(chunk)
    return bpf4.core.Sampled(data, x0=0, dx=dt2)


def peak(samples:np.ndarray) -> float:
    """return the highest sample value (dB)"""
    return amp2db(np.abs(samples).max())


def peaksbpf(samples:np.ndarray, sr:int, res=0.01, overlap=2, channel=0
             ) -> bpf4.core.Sampled:
    """
    Return a BPF representing the peaks envelope of the source with the
    resolution given

    Args:
        samples: the sound samples
        sr: sample rate
        res: resolution in seconds
        overlap: how much do windows overlap
        channel: which channel to analyze

    Returns:
        a samples bpf
    """
    samples = getChannel(samples, channel)
    period = int(sr*res+0.5)
    # dt = period/sr
    hopsamps = period//overlap
    numperiods = int(len(samples)/hopsamps)
    data = np.empty((numperiods,), dtype=float)
    for i in range(numperiods):
        idx0 = i * hopsamps
        chunk = samples[idx0:idx0+period]
        data[i] = np.abs(chunk).max()
    return bpf4.core.Sampled(data, x0=0, dx=hopsamps/sr)


def makeRamp(desc:str, numsamples:int) -> np.ndarray:
    """
    Args:
        desc: A string descriptor of a function. Possible descriptors are
            "linear", "expon(x)", "halfcos"
        numsamples: The number of samples to generate a ramp from 0 to 1

    Returns:
        a numpy array of shape (numsamples,), ramping from 0 to 1
    """
    assert isinstance(desc, str)
    assert isinstance(numsamples, int)
    return bpf4.util.makebpf(desc, [0, 1], [0, 1]).map(numsamples)


def numChannels(array: np.ndarray) -> int:
    """ Returns the number of channels represented by the given array"""
    return 1 if len(array.shape) == 1 else array.shape[1]


def getChannel(array: np.ndarray, channel:int) -> np.ndarray:
    """ Get a view into a channel of array. If array is mono, array
    itself is returned """
    return array if len(array.shape) == 1 else array[:, channel]


def arrayFade(samples: np.ndarray, sr: int, fadetime: float,
              mode='inout', shape: Union[str, Callable[[float], float]] = 'linear',
              margin=0
              ) -> None:
    """
    Fade samples **in place**

    Args:
        samples: numpy array
        sr: sr
        fadetime: fade time in seconds
        mode: in, out, or inout
        shape: either a string describing the shape (one of 'linear', 'expon(x)', halfcos)
            or a callable ``(t) -> gain``, defined between 0:1
        margin: if given, all fade operations are performed after/before the given
            margin (in samples) and the margin is zeroed. This is to enforce that
            sound starts/ends in silence
    """
    assert isinstance(samples, np.ndarray)
    assert isinstance(sr, int)
    assert isinstance(fadetime, (int, float))
    assert isinstance(mode, str)
    assert isinstance(shape, str)

    def mult(samples, ramp):
        numch = numChannels(samples)
        if numch == 1:
            samples *= ramp
        else:
            for ch in range(numch):
                samples[:, ch] *= ramp

    fadeframes = int(fadetime*sr)
    numframes = len(samples)
    ramp = makeRamp(shape, fadeframes)
    if mode in ('in', 'inout'):
        mult(samples[margin:fadeframes+margin], ramp)
        if margin>0:
            samples[:margin] = 0
    if mode in ('out', 'inout'):
        frame0 = max(0, len(samples)-fadeframes-margin)
        frame1 = min(frame0+fadeframes, numframes)
        mult(samples[frame0:frame1], ramp[::-1])
        if margin>0:
            samples[-margin:] = 0


def chunks(start:int, stop:int=None, step:int=None) -> Iterable[Tuple[int, int]]:
    """
    Like xrange, but returns a Tuplet (position, distance form last position)

    returns integers
    """
    if stop is None:
        stop = int(start)
        start = 0
    if step is None:
        step = 1
    cur = int(start)
    while cur + step < stop:
        yield cur, step
        cur += step
    yield cur, stop - cur


def firstSound(samples: np.ndarray, threshold=-120.0, periodsamps=256, overlap=2,
               skip=0
               ) -> Optional[int]:
    """
    Find the first sample in samples whith a rms
    exceeding the given threshold

    Args:
        samples: the numpy array holding the samples
        threshold: threshold in dB. An rms with a dB value higher than this will
            be considered a sound
        periodsamps: number of samples to analyze per period
        overlap: determines the hop size between each measurement.
        skip: number of samples to skip at the beginning

    Returns:
        sample index of the first sample holding sound or None if no sound found
    """
    threshold_amp = db2amp(threshold)
    hopsamples = periodsamps//overlap
    i0 = skip
    while True:
        i1 = i0+periodsamps
        if i1>len(samples):
            break
        if rms(samples[i0:i1])>threshold_amp:
            return i0
        i0 += hopsamples
    return None


def firstSilence(samples: np.ndarray, threshold=-100, period=256,
                 overlap=2, soundthreshold=-60, startidx=0
                 ) -> Optional[int]:
    """
    Return the sample where rms decays below threshold

    Args:
        samples: the samples data. Should be one channel
        threshold: the threshold in dBs (rms)
        period: how many samples to use for rms calculation
        overlap: how many samples to skip before taking the next
            measurement
        soundthreshold: the threshold to considere that the sound
            started (rms)
        startidx: the sample to start looking for silence (0 to start
            from beginning)

    Returns:
        the index where the first silence is found (None if no silence found)

    """
    assert numChannels(samples) == 1
    soundstarted = False
    hopsamples = period//overlap
    thresholdamp = db2amp(threshold)
    soundthreshamp = db2amp(soundthreshold)
    lastrms = rms(samples[startidx:startidx+period])
    idx = hopsamples+startidx
    while idx<len(samples)-period:
        win = samples[idx:idx+period]
        rmsnow = rms(win)
        if rmsnow>=soundthreshamp:
            soundstarted = True
        elif rmsnow<=thresholdamp and lastrms>thresholdamp and soundstarted:
            return idx
        lastrms = rmsnow
        idx += hopsamples
    return None


def lastSound(samples: np.ndarray, threshold=-120.0, period=256, overlap=2
              ) -> Optional[int]:
    """
    Find the end of the last sound in the samples.

    (the last time where the rms is lower than the given threshold)

    Args:
        samples: the samples to query
        threshold: the silence threshold, in dB
        period: the frame size (for rms calculation)
        overlap: determines the hop size (``hop size in samples = period/overlap``)

    Returns:
        the idx of the last sound, or None if no sound is found
    """
    assert numChannels(samples) == 1
    samples1 = samples[::-1]
    i = firstSound(samples1, threshold=threshold, periodsamps=period, overlap=overlap)
    return len(samples)-(i+period) if i is not None else None


def _lastSound(samples: np.ndarray, samplerate:int, threshold:float=-120, resolution=0.01):
    """
    Find the time when the last sound fades into silence
    (or, what is the same, the length of any silence at the
    end of the soundfile)

    Args:
        samples: The sample data
        samplerate: the sr of the data
        threshold: The volume threshold defining silence, in dB
        resolution: the time resolution, in seconds

    Returns:
        the time of the last sound (or 0 if no sound found)
    """
    numframes = len(samples)
    chunksize = int(resolution * samplerate)
    minamp = db2amp(threshold)
    for nframes, pos in chunks(0, numframes, chunksize):
        frame1 = numframes - pos
        frame0 = frame1 - nframes
        chunk = samples[frame0:frame1]
        if rms(chunk) > minamp:
            return frame0 / samplerate
    return 0


def normalizationRatio(samples: np.ndarray, maxdb=0.) -> float:
    """
    Return the factor needed to apply the given normalization.

    To normalize the array just multiply it by this ratio

    Args:
        samples: the samples to normalize
        maxdb: maximum peak in dB

    Returns:
        the normalization ratio

    Example
    =======

    >>> a = np.ndarray([-0.5, 0.3, 0.4])
    >>> normalizationRatio(a)
    2.0
    """
    max_peak_possible = db2amp(maxdb)
    peak = np.abs(samples).max()
    ratio = max_peak_possible / peak
    return ratio


