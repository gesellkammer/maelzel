"""
numpy utilities for audio arrays
"""
from __future__ import annotations
import numpy as np
import math
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterator, Callable
    import bpf4


def db2amp(db: float) -> float:
    """
    convert dB to amplitude (0, 1)

    Args:
        db: a value in dB

    .. seealso:: :func:`amp2db`
    """
    return 10.0 ** (0.05 * db)


def amp2db(amp: float) -> float:
    """
    convert amp (0, 1) to dB

    ``20.0 * log10(amplitude)``

    Args:
        amp: the amplitude between 0 and 1

    Returns:
        the corresponding amplitude in dB

    .. seealso:: :func:`db2amp`
    """
    amp = max(amp, sys.float_info.epsilon)
    return math.log10(amp) * 20


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
    return math.sqrt(np.sum(arr) / len(arr))


def rmsBpf(samples: np.ndarray, sr: int, dt=0.01, overlap=1) -> bpf4.Sampled:
    """
    Create a bpf representing the rms of this sample as a function of time

    Args:
        samples: the audio samples
        sr: the sample rate
        dt: analysis time period
        overlap: overlap of analysis frames. The step time is ``dt / overlap``

    Returns:
        a Sampled bpf mapping time to rms
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
    import bpf4
    return bpf4.Sampled(data, x0=0, dx=dt2)


def peak(samples: np.ndarray) -> float:
    """return the highest sample value (dB)"""
    return amp2db(np.abs(samples).max())


def peaksBpf(samples:np.ndarray, sr:int, res=0.01, overlap=2, channel=0
             ) -> bpf4.Sampled:
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
    import bpf4
    return bpf4.Sampled(data, x0=0, dx=hopsamps/sr)


def ampBpf(samples: np.ndarray, sr: int, attack=0.01, release=0.01, chunktime=0.05, overlap=2) -> bpf4.Sampled:
    """
    Constructs a sampled amplitude envelope from a sound signal.

    Args:
        samples: the sound samples
        sr: sample rate
        attack: attack time in seconds
        decay: decay time in seconds
        chunktime: chunk time in seconds
        overlap: how much do windows overlap

    Returns:
        a sampled bpf
    """
    assert numChannels(samples) == 1, "Only mono samples are supported"
    import numpyx
    env = numpyx.amp_follow(samples, sr, attack, release)
    chunksize = int(sr*chunktime+0.5)
    step = chunksize // overlap
    amps = [np.mean(frame) for frame in frames(env, chunksize, step)]
    dt = step / sr
    import bpf4
    return bpf4.Sampled(np.array(amps), x0=0, dx=dt)


def makeRamp(desc: str, numsamples: int) -> np.ndarray:
    """
    Create a ramp from 0 to 1 following the given descriptor

    Args:
        desc: A string descriptor of a function. Possible descriptors are
            "linear", "expon(x)", "halfcos"
        numsamples: The number of samples to generate a ramp from 0 to 1

    Returns:
        a numpy array of shape (numsamples,), ramping from 0 to 1
    """
    assert isinstance(desc, str)
    assert isinstance(numsamples, int)
    import bpf4.util
    return bpf4.util.makebpf(desc, [0, 1], [0, 1]).map(numsamples)


def numChannels(array: np.ndarray) -> int:
    """ Returns the number of channels represented by the given array"""
    return 1 if len(array.shape) == 1 else array.shape[1]


def getChannel(array: np.ndarray, channel: int, ensureContiguous=False) -> np.ndarray:
    """
    Get a view into a channel of array.

    If array is mono, array itself is returned

    Args:
        array: the original array
        channel: which channel to get
        ensureContiguous: if True, ensure that the returned array is contiguous,
            creating a new array if necessary

    Returns:
        the given channel from array
    """
    if len(array.shape) == 1:
        if channel > 0:
            raise IndexError("The given array is 1D")
        return array

    out = array[:, channel]
    if ensureContiguous:
        out = np.ascontiguousarray(out)
    return out


def applyFade(samples: np.ndarray, sr: int, fadetime: float,
              mode='inout', shape: str | Callable[[float], float] = 'linear',
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
        if margin > 0:
            samples[:margin] = 0
    if mode in ('out', 'inout'):
        frame0 = max(0, len(samples)-fadeframes-margin)
        frame1 = min(frame0+fadeframes, numframes)
        mult(samples[frame0:frame1], ramp[::-1])
        if margin > 0:
            samples[-margin:] = 0


def frames(arr: np.ndarray, chunksize: int, step=0) -> Iterator[np.ndarray]:
    """
    Split an array into overlapping frames

    Args:
        arr: The input array to be split.
        chunksize: The size of each frame.
        step: num of items to skip (use 0 for no overlap)

    Returns:
        A generator yielding frames as numpy arrays.

    .. note::

        The last frame may be smaller than chunksize if the array length
        is not a multiple of chunksize.

    Examples
    ~~~~~~~~

        >>> arr = np.arange(30)
        >>> list(framesiter(arr, 8, 4))
        [array([0, 1, 2, 3, 4, 5, 6, 7]),
        array([ 4,  5,  6,  7,  8,  9, 10, 11]),
        array([ 8,  9, 10, 11, 12, 13, 14, 15]),
        array([12, 13, 14, 15, 16, 17, 18, 19]),
        array([16, 17, 18, 19, 20, 21, 22, 23]),
        array([20, 21, 22, 23, 24, 25, 26, 27]),
        array([24, 25, 26, 27, 28, 29])]
    """
    for i in range(0, len(arr) - chunksize + step, step):
        yield arr[i:i + chunksize]


def chunks(start: int, stop: int, step: int
           ) -> Iterator[tuple[int, int]]:
    """
    Like xrange, but returns a tuple (position, chunk size)

    returns integers

    Example
    ~~~~~~~

        >>> list(chunks(0, 10, 3))
        [(0, 3), (3, 3), (6, 3), (9, 1)]
    """
    for i in range(start, stop + 1 - step, step):
        yield i, step
    rest = (stop - start) % step
    if rest > 0:
        yield stop - rest, rest


def firstSound(samples: np.ndarray, threshold=-120.0, periodsamps=256, overlap=2,
               skip=0
               ) -> int | None:
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
                 ) -> int | None:
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
              ) -> int | None:
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


def normalizeByMaxPeak(samples: np.ndarray, maxdb=0., out: np.ndarray | None = None
                       ) -> np.ndarray:
    """
    Normalize samples by maximum peak

    Args:
        samples: the samples to normalize
        maxdb: maximum peak in dB
        out: where to place the results. Can be the input samples
            itself, it which case the operation is performed in place

    Returns:
        the normalized samples

    """
    ratio = normalizationRatio(samples, maxdb=maxdb)
    if out is not None:
        out *= ratio
        return out
    else:
        return samples * ratio


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
    return float(max_peak_possible / peak)
    

def panStereo(samples: np.ndarray, pan: float) -> np.ndarray:
    """
    Apply panning to samples

    Args:
        samples: the samples to pan, a numpy array holding one or two channels
        pan: the panning value, between 0 and 1

    Returns:
        the panned samples, always a stereo array

    Example
    =======

    >>> a = np.ndarray([-0.5, 0.3, 0.4])
    >>> applyPanning(a, 0.5)
    array([-0.25,  0.15,  0.2 ])
    """
    leftamp = math.sin(math.pi * 2 * (1-pan))
    rightamp = math.sin(math.pi * 2 * pan)
    nchnls = numChannels(samples)
    if nchnls == 1:
        out = np.zeros((), dtype=samples.dtype)
        out[:,0] = samples * leftamp
        out[:,1] = samples * rightamp
        return out
    elif nchnls == 2:
        left = samples[:,0] * (1 - pan)
        right = samples[:,1] * (1 + pan)
        return np.stack((left, right), axis=-1)
    elif nchnls == 3:
        left = samples[:,0] * (1 - pan)
        right = samples[:,1] * (1 + pan)
        center = samples[:,2] * (1 - pan)
        return np.stack((left, right, center), axis=-1)
    else:
        raise ValueError(f"Unsupported number of channels: {nchnls}")


def applyPanning(samples: np.ndarray, pan: float) -> None:
    """
    Apply panning to samples, in place

    Args:
        samples: the samples to pan, a stereo numpy array
        pan: the panning value, between 0 and 1

    Returns:
        None

    """
    leftamp = math.sin(math.pi * 2 * (1-pan))
    rightamp = math.sin(math.pi * 2 * pan)
    assert numChannels(samples) == 2, f"Unsupported number of channels: {numChannels(samples)}"
    samples[:,0] *= leftamp
    samples[:,1] *= rightamp
