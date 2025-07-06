"""
Utilities to edit sound-files in chunks
"""
from __future__ import annotations
import tempfile
import os
from math import sqrt
import numpy as np
import bpf4
from emlib.iterlib import flatten
from pitchtools import db2amp, amp2db
import sndfileio

import numpyx
from maelzel.snd import numpysnd as npsnd
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Callable, Iterator, TypeAlias
    Func1: TypeAlias = Callable[[float], float]
    sample_t: TypeAlias = tuple[np.ndarray, int]
    processfunc_t: TypeAlias = Callable[[np.ndarray, int, float], None]
    _FloatFunc: TypeAlias = Callable[[float], float]


def _chunks(start: int, stop: int | None = None, step: int | None = None
            ) -> Iterator[tuple[int, int]]:
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


def fadeSndfile(sndfile: str, outfile: str, fadetime: float,
                mode: str = 'inout', shape: str | Func1 = 'halfcos'
                ) -> None:
    """
    Generate a new file `outfile` with the faded source

    Args:
        sndfile: infile
        outfile: outfile. Use None to generate a name based on the infile
        fadetime: the fade time, in seconds
        mode: one of 'in', 'out', 'inout'
        shape: a string describing the shape (one of 'linear', 'expon(x)', halfcos)
                 or a callable ``(t) -> gain``, defined between 0:1

    Returns:
        the path of the generated outfile
    """
    samples, sr = sndfileio.sndread(sndfile)
    npsnd.applyFade(samples, sr, fadetime=fadetime, mode=mode, shape=shape)
    sndfileio.sndwrite_like(outfile, samples, likefile=sndfile)


def copyFragment(path: str, start: float, end: float, outfile: str
                 ) -> None:
    """
    Read the data between begin and end (in seconds) and write it to outfile

    Args:
        path: the file to copy from
        start: the start time
        end: end time. 0=end of file, negative numbers are calculated
            as a margin from the end of the soundfile
        outfile: path of the saved fragment.
    """
    if end <= 0:
        end = sndfileio.sndinfo(path).duration - end
    samples, sr = sndfileio.sndread(path, start=start, end=end)
    sndfileio.sndwrite_like(outfile, samples, likefile=path)


def process(sourcefile: str,
            outfile: str,
            callback: processfunc_t,
            timerange: tuple[float, float] | None = None,
            bufsize=4096) -> None:
    """
    Process samples of sourcefile in fragments and write them to outfile

    Args:
        sourcefile: the file to read
        outfile: the file to write to
        callback: a function ``(data, sampleindex, now) -> None``. Data should be
            processed in place
        timerange: if given, only the samples within the times given will be processed
        bufsize: the size of each fragment
    """
    if timerange:
        start, end = timerange
    else:
        start, end = 0, 0
    writer = sndfileio.sndwrite_chunked_like(sourcefile, outfile)
    sr = writer.sr
    pos = 0
    for chunk in sndfileio.sndread_chunked(sourcefile, bufsize, start, end):
        if callback:
            callback(chunk, pos, (pos / sr) + start)
        writer.write(chunk)
        pos += len(chunk)


def gain(filename: str, factor: float | _FloatFunc, outfile: str) -> None:
    """
    Change the volume of a audiofile.

    Args:
        filename: the soundfile to read
        factor: a number between 0-1 or a callable (\t -> gain_at_t)
        outfile: the output filename
    """
    if callable(factor):
        import bpf4.util
        factorfunc = bpf4.util.asbpf(factor)
        return _dynamicGain(filename, factorfunc, outfile)

    def callback(data: np.ndarray, sampleidx: int, now: float):
        data *= factor

    process(filename, outfile, callback)


def _dynamicGain(sndfile: str, curve: bpf4.BpfInterface, outfile='inplace') -> None:
    """
    Apply a dynamic gain to source

    Args:
        sndfile (str): the path to the soundfile
        curve (bpf): a bpf mapping time to gain
        outfile (str): the path to save the results, or 'inplace' if the results
            should be saved to the original file

    """
    if outfile == 'inplace':
        inplace = True
        outfile = tempfile.mktemp()
    else:
        inplace = False
    chunkdur = 0.1

    def callback(data: np.ndarray, sampleidx: int, now: float):
        factor = curve.mapn_between(len(data), now, now + chunkdur)
        data *= factor

    process(sndfile, outfile, callback)
    if inplace:
        os.rename(outfile, sndfile)


def mixArrays(sources: list[np.ndarray], offsets: list[int] | None = None) -> np.ndarray:
    """
    Mix the sources together.

    All sources should have the same amount of channels. It is assumed that they share
    the same sr

    Args:
        sources: a list of arrays. They should all have the same amount of channels
        offsets: a list of offsets, in samples (optional)

    Returns:
        a numpy array holding the result of mixing all samples.
    """
    if not offsets:
        offsets = [0] * len(sources)
    else:
        assert len(offsets) == len(sources)
    nchannels = npsnd.numChannels(sources[0])
    assert all(npsnd.numChannels(s) == nchannels for s in sources), \
        "Sources should have the same amount of channels"
    dtype = sources[0].dtype
    assert all(s.dtype == dtype for s in sources), "Arrays should have the same dtype"
    end = max(((len(source) + offset)
               for source, offset in zip(sources, offsets)))
    out = np.zeros((end, nchannels), dtype=dtype)
    for source, t in zip(sources, offsets):
        if nchannels > 1:
            for channel in range(nchannels):
                out[t:len(source), channel] += source[:, channel]
        else:
            out[t:len(source)] += source
    return out


def readChunks(sndfile: str,
               chunksize: int,
               start=0.0, end=0.0
               ) -> Iterator[tuple[np.ndarray, int]]:
    """
    Read chunks of data from source.

    Each chunk has a size of `chunksize` but can have less
    if there are not enough samples to read (at the end of the file)

    .. note::
        Either chunksize or chunkdur can be given, not both

    Args:
        sndfile: the soundfile to read
        chunksize: size of the chunk in samples
        chunkdur: dur. of the chunk (in secs). NB: either chunksize or chunkdur can be given
        start: start time to read (in seconds)
        end: and time to read (in seconds). 0 = end of file

    Returns:
        an iterator of tuple(datachunk, position_in_frames)

    """
    info = sndfileio.sndinfo(sndfile)
    sampleidx = int(start * info.samplerate)
    for chunk in sndfileio.sndread_chunked(sndfile, chunksize=chunksize,
                                           start=start, stop=end):
        yield chunk, sampleidx
        sampleidx += len(chunk)


def equalPowerPan(pan: float) -> tuple[float, float]:
    """
    Calculate the factors to apply an equal-power-pan

    Args:
        pan: a float from 0 to 1

    Returns:
        the left and right factors
    """
    return sqrt(1 - pan), sqrt(pan)


def peakbpf(filename: str, resolution=0.01, method='peak', channel: int | str = 'mix',
            normalize=False) -> bpf4.BpfInterface:
    """
    Build a bpf representing the peaks envelope of the source

    Args:
        filename: the file to analyze
        resolution: resolution of the bpf
        method: one of 'peak', 'rms' or 'mean'
        channel: either a channel number (starting with 0) or 'mix'
        normalize: if True, peaks are normalized before constructing the bpf

    Returns:
        a bpf holding the peaks curve
    """

    info = sndfileio.sndinfo(filename)
    srate = info.samplerate
    chunksize = int(srate * resolution)
    assert chunksize > 10, f"Resolution is too low: {resolution}"

    funcs = {
        'peak': lambda arr: max(numpyx.minmax1d(arr)),
        'rms': npsnd.rms,
        'mean': lambda arr: np.abs(arr).mean()
    }
    func = funcs.get(method)
    if func is None:
        raise ValueError(f"Unknown {method} method")

    times: list[float] = []
    peaks: list[float] = []
    channum = channel if isinstance(channel, int) else -1

    for chunk, pos in readChunks(filename, chunksize):
        if info.channels > 1:
            if channum == -1:
                chunk = npsnd.asmono(chunk)
            else:
                chunk = npsnd.getChannel(chunk, channum)
        times.append(pos / info.samplerate)
        peaks.append(func(chunk))

    timesarr = np.array(times)
    peaksarr = np.array(peaks)
    if normalize:
        peaksarr /= peaksarr.max()
    return bpf4.Linear(timesarr, peaksarr)


def maxPeak(filename: str, start: float = 0, end: float = 0, resolution=0.01
            ) -> tuple[float, float]:
    """
    Return the time of the max. peak and the value of the max. peak

    Args:
        filename: the filename to process
        start: start time
        end: end time
        resolution: resolution in seconds

    Returns:
        a tuple (time of peak, peak value)
    """
    maximum_peak = 0
    max_pos = 0
    info = sndfileio.sndinfo(filename)
    chunksize = int(info.samplerate * resolution)
    for data, pos in readChunks(filename, start=start, end=end, chunksize=chunksize):
        np.abs(data, data)
        peak = np.max(data)
        if peak > maximum_peak:
            maximum_peak = peak
            max_pos = pos
    return max_pos / info.samplerate, maximum_peak


def findFirstSound(sndfile: str, threshold=-120,
                   resolution=0.01, start=0.
                   ) -> float | None:
    """
    Find the time of the first sound in the soundfile

    (or, what is the same, the length of any initial silence at the
    beginning of the soundfile)

    Args:
        sndfile: The path to the soundfile
        threshold: The volume threshold defining silence, in dB
        resolution: the time resolution, in seconds
        start: where to start searching (in seconds)

    Returns:
        time of the first sound (in seconds), or None if no first sound found
    """
    minamp = db2amp(threshold)
    info = sndfileio.sndinfo(sndfile)
    chunksize = int(resolution * info.samplerate)
    for chunk, pos in readChunks(sndfile, chunksize=chunksize, start=start):
        if npsnd.rms(chunk) > minamp:
            return pos / info.samplerate
    return None


def findLastSound(sndfile: str, threshold=-120, resolution=0.01) -> float | None:
    """
    Find the time when the last sound fades into silence

    (or, what is the same, the length of any silence at the
    end of the soundfile)

    Args:
        sndfile: The path to the soundfile
        threshold: The volume threshold defining silence, in dB
        resolution: the time resolution, in seconds

    Returns:
        the time of the last sound (or None if no sound found)
    """
    frames, sr = sndfileio.sndread(sndfile)
    period = int(resolution * sr)
    frame = npsnd.lastSound(frames, threshold=threshold, period=period)
    return frame / sr if frame is not None else None


def stripSilence(sndfile: str, outfile: str, threshold=-100, margin=0.050,
                 fadetime=0.1, mode='both') -> None:
    """
    Remove silence at the beginning and at the end of the source, which fall
    below threshold (dB)

    Args:
        sndfile: the soundfile
        outfile: the output soundfile
        threshold: the amplitude in dB which needs to be crossed to be considered
            as sound
        margin: Determines how much silence is left at the edges (in seconds)
        fadetime: Indicates of a fade in/out are performed
        mode: one of 'left', 'right', 'both' (default=both)

    """
    data, samplerate = sndfileio.sndread(sndfile)
    if mode == 'both' or mode == 'left':
        firstFrame = npsnd.firstSound(data, threshold=threshold)
        if firstFrame is None:
            t0 = 0
        else:
            t0 = firstFrame / samplerate - margin
    else:
        t0 = 0
    if mode == 'both' or mode == 'right':
        lastFrame = npsnd.lastSound(data, threshold=threshold)
        if lastFrame is None:
            t1 = len(data) / samplerate
        else:
            t1 = lastFrame / samplerate + margin
    else:
        t1 = len(data) / samplerate
    frame0 = max(0, int(t0 * samplerate))
    frame1 = min(len(data), int(t1 * samplerate))
    new_data = data[frame0:frame1]

    if fadetime > 0:
        npsnd.applyFade(data, samplerate, fadetime=fadetime, mode='inout',
                        shape='linear', margin=32)

    sndfileio.sndwrite_like(outfile, new_data, likefile=sndfile)


def normalize(path: str, outfile: str, headroom=0.) -> None:
    """
    Normalize the given soundfile.

    Returns the path of the normalized file. If outfile is not given, a new file with the
    suffix "-N" is generated based on the input

    Args:
        path: the path to the soundfile
        outfile: the path to the outfile
        headroom: the peak to normalize to, in dB

    """
    data, sr = sndfileio.sndread(path)
    maxamp = np.abs(data).max()
    headroomamp = db2amp(headroom)
    ratio = headroomamp / maxamp
    data *= ratio
    sndfileio.sndwrite_like(outfile, data, likefile=path)


def detectRegions(sndfile: str, attackthresh: float, decaythresh: float,
                  mindur=0.020, func='rms', resolution=0.004, mingap: float = 0,
                  normalize=False
                  ) -> tuple[list[tuple[float, float]], bpf4.NoInterpol]:
    """
    Detect fragments inside a soundfile.

    Args:
        sndfile: the soundfile to analyze
        attackthresh (dB): the amplitude necessary to start a region
        decaythresh (dB) : the amplitude under which to stop a region
            (should be lower than attackthresh)
        mindur (sec): minimal duration of a region
        func: the function to use to calculate the envelope. One of
            'rms', 'peak', 'mean'
        resolution (sec): the resolution of the analysis
        mingap (sec): the minimal gap between regions
        normalize: wether to normalize the soundfile before analysis

    Returns:
        a list of regions and a bpf mapping time -> sound_present
        each region is a tuple (region start, region end)
    """
    b = peakbpf(sndfile, resolution=resolution, method=func, normalize=normalize)
    import bpf4.util
    bsmooth = bpf4.util.smoothen((b + db2amp(-160)).apply(amp2db), window=int(mindur/8))
    regions: list[tuple[float, float]] = []
    Y = bsmooth.sample_between(bsmooth.x0, bsmooth.x1, resolution)
    X = np.linspace(b.x0, b.x1, len(Y))
    regionopen = False
    regionx0 = 0
    for x, y, in zip(X, Y):
        if not regionopen and y > attackthresh:
            regionopen = True
            regionx0 = x
        elif regionopen and y < decaythresh and x - regionx0 > mindur:
            regionopen = False
            avgamp = amp2db(b.integrate_between(regionx0, x) / (x - regionx0))
            if avgamp > decaythresh:
                regions.append((regionx0, x))
                regionx0 = x
    # merge regions with gap smaller than minimumgap
    if mingap > 0 and len(regions) > 1:
        mergedregions: list[tuple[float, float]] = [regions[0]]
        for region in regions[1:]:
            lastregion = mergedregions[-1]
            gap = region[0] - lastregion[1]
            if gap >= mingap:
                mergedregions.append(lastregion)
            else:
                # add the new region to the last region and hold it,
                # do not append yet to new regions
                mergedregions[-1] = (lastregion[0], region[1])
        regions = [region for region in mergedregions
                   if region[0] >= b.x0 and region[1] <= b.x1]
    import bpf4.api
    mask = bpf4.api.nointerpol(*flatten([((x0, 1), (x1, 0)) for x0, x1 in regions]))
    return regions, mask


def extractRegions(sndfile: str,
                   times: list[tuple[float, float]],
                   outfile: str,
                   mode='seq',
                   fadetimes=(0.005, 0.1),
                   fadeshape='linear'
                   ) -> None:
    """
    Read regions defined by `times`, write them in sequence to outfile

    Args:

        sndfile: a path to a soundfile
        times: a seq. of times (start, end)
        outfile: a path to the outfile
        mode: 'seq' or 'original'. With 'seq', extract regions and stack them sequentially.
            The end duration = sum(duration of each region). With 'original', generates
            a file where the regions stay in their original place and everything else
            is erased
        fadetimes: fade applied to the samples before writing them to avoid clicks
        fadeshape: shape of the fades

    .. note::
        This is useful when you have extracted markers from a soundfile,
        to extract the fragments themselves to a new file
    """
    info = sndfileio.sndinfo(sndfile)
    regions, sr = readRegions(sndfile, times)
    for region in regions:
        npsnd.applyFade(region, sr, fadetimes[0], mode='in', shape=fadeshape)
        npsnd.applyFade(region, sr, fadetimes[1], mode='out', shape=fadeshape)
    writer = sndfileio.sndwrite_chunked_like(outfile, likefile=sndfile)
    if mode == 'original':
        shape = (info.nframes, info.channels) if info.channels > 1 else (info.nframes,)
        samples_out = np.zeros(shape, dtype=float)
        for region, (t0, t1) in zip(regions, times):
            offset = int(t0 * sr)
            samples_out[offset:offset + len(region)] += region
        writer.write(samples_out)
    elif mode == 'seq':
        for region, (t0, t1) in zip(regions, times):
            writer.write(region)
    else:
        raise ValueError(f"Expected 'seq', or 'original', got {mode}")


def readRegions(sndfile: str, times: list[tuple[float, float]]
                ) -> tuple[list[np.ndarray], int]:
    """
    Extract regions from a soundfile

    Args:
        sndfile: the path to a sound-file
        times: a list of tuples (start, end).

    Returns:
        a tuple (list of arrays, sr)
    """
    # TODO: implement this at the backend level (in sndfileio) using seek
    data, sr = sndfileio.sndread(sndfile)
    out = []
    for t0, t1 in times:
        fr0 = int(t0 * sr)
        fr1 = int(t1 * sr)
        new_samples = data[fr0:fr1]
        out.append(new_samples)
    return out, sr


def addSilentChannel(source: str, outfile: str, numchannels=1) -> np.ndarray:
    """
    Given a source, return a new source with an added channel of silence

    Args:
        source: the source soundfile
        outfile: the path to save the resulting soundfile
        numchannels: the number of silent channels to add

    Returns:
        the samples with the added silent channel
        
    .. note::
        the use-case for this is to add a silent channel to a mono file, to make
        clear that the right channel should be silent and the left channel should
        not be played also through the right channel
    """
    samples, sr = sndfileio.sndread(source)
    numsamples = len(samples)
    silence = np.zeros((numsamples, numchannels), dtype=float)
    data = np.column_stack((samples, silence))
    sndfileio.sndwrite_like(outfile=outfile, samples=data, likefile=source)
    return data
    

def _getsamples(source: str | sample_t) -> sample_t:
    """
    source can be: "/path/to/source" or (samples, sr)
    """
    if isinstance(source, str):
        samples, sr = sndfileio.sndread(source)
    elif isinstance(source, tuple) and len(source) == 2:
        samples, sr = source
        assert isinstance(samples, np.ndarray)
        assert isinstance(sr, int)
    else:
        raise TypeError("source can be a path to a soundfile or a Tuple (samples, sr)")
    return (samples, sr)


def scrub(source: str | tuple[np.ndarray, int], curve: bpf4.BpfInterface,
          rewind=False, outfile='') -> sample_t:
    """
    Scrub soundfile with curve

    Args:
        source: the path to a sndfile, or a Tuple (samples, sr)
        curve: a bpf representing real_time:time_in_soundfile
        rewind: if True, do not include silence at the beginning if
                       the bpf does not start at 0
        outfile: if given, samples are written to disk

    Returns:
        a tuple (samples: np.ndarray, sr: int)
    """
    samples, sr = _getsamples(source)
    samplebpf = bpf4.Sampled(samples, 1.0 / sr)
    warped = curve | samplebpf
    newsamples = warped.sample_between(curve.x0, curve.x1, 1./sr)
    if not rewind and curve.x0 > 0:
        out = np.zeros((int(sr * curve.x1),), dtype=float)
        out[-len(newsamples):] = newsamples
        newsamples = out
    if outfile:
        if isinstance(source, str):
            sndfileio.sndwrite_like(outfile, likefile=source, samples=newsamples)
        else:
            sndfileio.sndwrite(outfile=outfile, samples=newsamples, sr=sr)
    return (newsamples, sr)
