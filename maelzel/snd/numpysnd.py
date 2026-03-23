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


def rmsAt(signal: np.ndarray, sr: int, times: np.ndarray, winsize: int = 0) -> np.ndarray:
    """
    Calculate RMS of a signal at specified times using a centered window.

    Args:
        signal: 1D numpy array of audio samples
        sr: sample rate in Hz
        times: array of times (in seconds) at which to calculate RMS
        winsize: window size in samples (defaults to sr//100, i.e. 10ms)

    Returns:
        Array of RMS values at each requested time

    """
    if winsize <= 0:
        winsize = sr // 100

    half = winsize // 2
    n = len(signal)
    rmsvals = np.empty(len(times))

    for i, t in enumerate(times):
        center = int(round(t * sr))
        start = center - half
        end = start + winsize

        # How many samples to pad on each side
        padleft  = max(0, -start)
        padright = max(0, end - n)

        # Clamp indices to valid signal range
        sigstart = max(0, start)
        sigend   = min(n, end)

        window = signal[sigstart:sigend]

        if padleft > 0 or padright > 0:
            window = np.pad(window, (padleft, padright), mode='constant')

        rmsvals[i] = np.sqrt(np.mean(window ** 2))

    return rmsvals


def rmsBpf(samples: np.ndarray, sr: int, framedur=0.01, overlap=1) -> bpf4.Sampled:
    """
    Create a bpf representing the rms of this sample as a function of time

    Args:
        samples: the audio samples
        sr: the sample rate
        framedur: analysis time in seconds
        overlap: overlap of analysis frames. The step time is ``dt / overlap``

    Returns:
        a Sampled bpf mapping time to rms
    """
    s = samples
    period = int(sr * framedur + 0.5)
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


def peaksBpf(samples: np.ndarray, sr: int, dt=0.01, overlap=2, channel=0
             ) -> bpf4.Sampled:
    """
    Return a BPF representing the peaks envelope of the source

    Args:
        samples: the sound samples
        sr: sample rate
        dt: resolution in seconds
        overlap: how much do windows overlap
        channel: which channel to analyze

    Returns:
        a samples bpf
    """
    samples = getChannel(samples, channel)
    period = int(sr * dt + 0.5)
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


def ampBpf(samples: np.ndarray, sr: int, attack=0.01, release=0.01, framedur=0.05, overlap=2) -> bpf4.Sampled:
    """
    Constructs a sampled amplitude envelope from a sound signal.

    Args:
        samples: the sound samples
        sr: sample rate
        attack: attack time in seconds
        release: decay time in seconds
        framedur: chunk time in seconds
        overlap: how much do windows overlap

    Returns:
        a sampled bpf
    """
    assert numChannels(samples) == 1, "Only mono samples are supported"
    import numpyx
    env = numpyx.amp_follow(samples, sr, attack, release)
    chunksize = int(sr * framedur + 0.5)
    step = chunksize // overlap
    amps = [np.mean(frame) for frame in _iterFramesPaddedCentered(env, chunksize, step)]
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


def iterFrames(arr: np.ndarray, size: int, hop: int = 0, pad=False, center=True
               ) -> Iterator[np.ndarray]:
    """
    Split an array into overlapping frames

    Args:
        arr: The input array to be split.
        size: The size of each frame.
        hop: num of items to skip (use 0 for no overlap)
        pad: if True, the last frame will be zero padded if needed
        center: pad the beginning

    Returns:
        A generator yielding frames as numpy arrays.

    Examples
    ~~~~~~~~

        >>> arr = np.arange(30)
        >>> list(iterFrames(arr, 8, 4, padded=False))
        [array([0, 1, 2, 3, 4, 5, 6, 7]),
        array([ 4,  5,  6,  7,  8,  9, 10, 11]),
        array([ 8,  9, 10, 11, 12, 13, 14, 15]),
        array([12, 13, 14, 15, 16, 17, 18, 19]),
        array([16, 17, 18, 19, 20, 21, 22, 23]),
        array([20, 21, 22, 23, 24, 25, 26, 27]),
        array([24, 25, 26, 27, 28, 29])]
    """
    if not pad:
        for i in range(0, len(arr) - size + hop, hop):
            yield arr[i:i + size]
    elif center:
        yield from _iterFramesPaddedCentered(arr, size=size, hop=hop, marginleft=0 if not center else size//2)
    else:
        yield from _iterFramesPadded(arr, size=size, hop=hop)


def _iterFramesPadded(arr: np.ndarray, size: int, hop: int) -> Iterator[np.ndarray]:
    """
    Iterate over frames of an audio sample array.

    Args:
        arr: 1-D numpy array
        size: number of items per frame.
        hop: number of items to advance between frames.

    Returns:
        List of numpy array views (no copy) of shape (size,).
        The last frame is zero-padded if it doesn't fill a complete frame.
    """
    n_samples = len(arr)
    n_frames = 1 + (n_samples - 1) // hop  # ensures the last sample appears in at least one frame

    for i in range(n_frames):
        start = i * hop
        end = start + size
        if end <= n_samples:
            yield arr[start:end]          # full frame — pure view, no copy
        else:
            frame = np.zeros(size, dtype=arr.dtype)
            frame[:n_samples - start] = arr[start:]  # copy only for the last partial frame
            yield frame


def _iterFramesPaddedCentered(arr: np.ndarray, size: int, hop: int, marginleft: int = 0
                              ) -> Iterator[np.ndarray]:
    """
    Iterate over frames of an audio sample array.

    Args:
        arr: 1-D numpy array
        size: number of items per frame.
        hop: number of items to advance between frames.
        marginleft: zero pad this number of items at the beginning

    Returns:
        List of numpy array views (no copy) of shape (size,).
        The last frame is zero-padded if it doesn't fill a complete frame.
    """
    n_samples = len(arr)
    n_frames = 1 + (n_samples - 1) // hop  # ensures the last sample appears in at least one frame
    for i in range(n_frames):
        start = i * hop - marginleft
        end = start + size
        if start < 0:
            frame = np.zeros(size, dtype=arr.dtype)
            frame[-start:] = arr[0:size+start]
            yield frame
        elif end <= n_samples:
            yield arr[start:end]          # full frame — pure view, no copy
        else:
            frame = np.zeros(size, dtype=arr.dtype)
            frame[:n_samples - start] = arr[start:]  # copy only for the last partial frame
            yield frame


# ------------------------------------------
# helpers
# ------------------------------------------

def _frameSignal(samples: np.ndarray, framesize: int, hopsize: int) -> np.ndarray:
    """
    Construct an array representing a framed signal

    Each row is a frame (or window) into the given samples, of the given framesize,
    jumping `hopsize` items between frames. Data is not copied (this operation
    uses stride "tricks" and is very efficient).makeFrames

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


def makeFrames(samples: np.ndarray,
               sr: int,
               frameSize: int = 2048,
               hopSize: int = 0,
               normalize=False
               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an array of samples, returns (frames, times)

    Where frames is an array with shape (numframes, framesize) where
    each row represents a frame. times is an array holding the centre
    time for each frame. No windowing is applied at this stage

    Args:
        samples: audio data (1D)
        sr: sampling rate (Hz), needed to calculate the times array
        frameSize: the size of each frame
        hopSize: hop size in samples. If not given, hopsize=framesize//4

    Returns:
        a tuple (frames: np.ndarray, times: np.ndarray) where frames has
        a dimension more than samples, so if samples is a 1D array,
        the returned array has a shape of (numrows, frameSize)

    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("audio must be 1-D")

    if normalize:
        peak = np.abs(samples).max()
        if peak > 0:
            samples = samples / peak

    hopSize = hopSize or frameSize // 4
    pad = frameSize // 2
    padded = np.pad(samples, (pad, pad), mode="constant")
    frames = _frameSignal(padded, frameSize, hopSize)
    times = (np.arange(len(frames)) * hopSize) / sr
    assert frames.shape[1] == frameSize
    return frames, times


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
    for i in range(skip, len(samples), hopsamples):
        frag = samples[i: i+periodsamps]
        if rms(frag) > threshold_amp:
            return i
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


def normalizeByPeak(samples: np.ndarray, headroom=0., out: np.ndarray | None = None
                    ) -> np.ndarray:
    """
    Normalize samples by maximum peak

    Args:
        samples: the samples to normalize
        headroom: maximum peak in dB
        out: where to place the results. Can be the input samples
            itself, it which case the operation is performed in place

    Returns:
        the normalized samples

    """
    if headroom > 0.:
        raise ValueError(f"headroom {headroom} is out of range")
    ratio = normalizationRatio(samples, headroom=headroom)
    if out is None:
        return samples * ratio
    out *= ratio
    return out


def normalizationRatio(samples: np.ndarray, headroom=0.) -> float:
    """
    Return the factor needed to apply the given normalization.

    To normalize the array just multiply it by this ratio

    Args:
        samples: the samples to normalize
        headroom: maximum peak in dB

    Returns:
        the normalization ratio

    Example
    =======

    >>> a = np.ndarray([-0.5, 0.3, 0.4])
    >>> normalizationRatio(a)
    2.0
    """
    maxpeak = db2amp(headroom)
    peak = np.abs(samples).max()
    return float(maxpeak/peak)
    

def panStereo(samples: np.ndarray, pan: float) -> np.ndarray:
    """
    Apply panning to samples

    Args:
        samples: the samples to pan, a numpy array holding one or two channels
        pan: the panning value, between 0 (left) and 1 (right)

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
    assert numChannels(samples) == 2, f"Unsupported number of channels: {numChannels(samples)}"
    leftamp = math.sin(math.pi * 2 * (1-pan))
    rightamp = math.sin(math.pi * 2 * pan)
    samples[:,0] *= leftamp
    samples[:,1] *= rightamp




def _schmittLoop(signal: np.ndarray, high: float, low: float, initial: int = 0, minhold: int = 0
                 ) -> np.ndarray:
    """
    Apply a Schmitt trigger to a signal.

    Args:
        signal: 1D numpy array
        high: upper threshold - output switches to 1 when signal crosses above this
        low: lower threshold - output switches to 0 when signal drops below this
        initial: initial state of the trigger (0 or 1)

    Returns:
        1D integer array of 0s and 1s
    """
    if low >= high:
        raise ValueError(f"low threshold ({low}) must be less than high threshold ({high})")

    out = np.empty(len(signal), dtype=np.int8)
    state = initial
    holdRemaining = 0

    for i, x in enumerate(signal):
        if holdRemaining > 0:
            # Still within the refractory period — no transition allowed
            holdRemaining -= 1
        elif state == 0:
            if x >= high:
                state = 1
                holdRemaining = minhold
        else:
            if x <= low:
                state = 0
                holdRemaining = minhold
        out[i] = state
    return out


def _schmittVect(signal: np.ndarray, high: float, low: float, initial: int = 0, minhold: int = 0
                 ) -> np.ndarray:
    """
    Apply a Schmitt trigger to a signal (vectorized, no Python loop).

    Args:
        signal: 1D numpy array
        high: upper threshold - output switches to 1 when signal crosses above this
        low: lower threshold - output switches to 0 when signal drops below this
        initial: initial state of the trigger (0 or 1)

    Returns:
        1D integer array of 0s and 1s
    """
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    if minhold < 0:
        raise ValueError(f"minhold must be >= 0, got {minhold}")

    rise_idx = np.where(signal >= high)[0]
    fall_idx = np.where(signal <= low)[0]
    out = np.full(len(signal), initial, dtype=np.int8)

    if len(rise_idx) == 0 and len(fall_idx) == 0:
        return out

    # Merge and sort all raw threshold crossings
    events = np.concatenate([rise_idx, fall_idx])
    tags   = np.concatenate([
        np.ones (len(rise_idx), dtype=np.int8),
        np.full (len(fall_idx), -1, dtype=np.int8),
    ])
    order        = np.argsort(events, kind='stable')
    events, tags = events[order], tags[order]

    # Remove same-type runs (keep first of each), respecting initial state
    first_tag    = 1 if initial == 0 else -1
    mask         = np.empty(len(tags), dtype=bool)
    mask[0]      = tags[0] == first_tag
    mask[1:]     = tags[1:] != tags[:-1]
    events, tags = events[mask], tags[mask]

    # Drop a leading event that matches the current state
    if len(tags) > 0:
        if initial == 0 and tags[0] == -1:
            events, tags = events[1:], tags[1:]
        elif initial == 1 and tags[0] == 1:
            events, tags = events[1:], tags[1:]

    # Apply minhold: sequentially walk kept events, enforcing the refractory
    # period from each *kept* event (not from dropped ones).
    # This cannot be fully vectorized because each decision depends on the
    # outcome of the previous one.
    if minhold > 0 and len(events) > 0:
        keep         = np.zeros(len(events), dtype=bool)
        last_kept_at = events[0] - minhold - 1   # sentinel: first event always kept
        for i in range(len(events)):
            if events[i] - last_kept_at > minhold:
                keep[i]      = True
                last_kept_at = events[i]
        events, tags = events[keep], tags[keep]

    # Fill output in slices between consecutive kept events
    for i, (idx, tag) in enumerate(zip(events, tags)):
        end = events[i + 1] if i + 1 < len(events) else len(signal)
        out[idx:end] = 1 if tag == 1 else 0

    return out
    

def schmitt(signal: np.ndarray, high: float, low: float, initial: int = 0, minhold: int = 0
            ) -> np.ndarray:
    """
    Apply a Schmitt trigger to a signal (vectorized, no Python loop).

    Args:
        signal: 1D numpy array
        high: upper threshold - output switches to 1 when signal crosses above this
        low: lower threshold - output switches to 0 when signal drops below this
        initial: initial state of the trigger (0 or 1)
        minhold: hold the value for this number of samples before any change
            can take effect

    Returns:
        1D integer array of 0s and 1s
    """
    if len(signal) < 500:
        return _schmittLoop(signal, high=high, low=low, initial=initial, minhold=minhold)
    else:
        return _schmittVect(signal, high=high, low=low, initial=initial, minhold=minhold)


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
    return float(xv), float(yv)