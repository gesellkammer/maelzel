"""
Utilities to edit sound-files in chunks
"""
from __future__ import annotations
import tempfile
import os
from math import sqrt
from dataclasses import dataclass
import pysndfile
import numpy as np
import bpf4 as bpf
from emlib.iterlib import flatten
from pitchtools import db2amp, amp2db
from typing import Callable, Tuple, Iterator as Iter, Union as U, \
    List, Optional as Opt
import sndfileio
import logging
import numpyx
from maelzel.snd import numpysnd as npsnd

logger = logging.Logger("maelzel.sndfiletools")
Func1 = Callable[[float], float]
sample_t = Tuple[np.ndarray, int]
processfunc_t = Callable[[np.ndarray, int, int], np.ndarray]


@dataclass
class SndInfo:
    """
    Structure holding information about a soundfile

    * samplerate (int): samplerate
    * numchannels (int): number of channels
    * numframes (int): number of frames. samples = frames * channels
    * encoding (int): one of 'pcm16', 'pcm24', 'pcm32', 'float32', 'float64'

    Attributes:
        samplerate (int): samplerate
        numchannels (int): number of channels
        numframes (int): number of frames. samples = frames * channels
        encoding (int): one of 'pcm16', 'pcm24', 'pcm32', 'float32', 'float64'
    """
    samplerate: int
    numchannels: int
    numframes: int
    encoding: str

    @property
    def duration(self) -> float:
        return self.numframes/self.samplerate


_FloatFunc = Callable[[float], float]


def chunks(start:int, stop:int=None, step:int=None) -> Iter[Tuple[int, int]]:
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


def fadeSndfile(sndfile:str, outfile:str, fadetime:float,
                mode:str, shape:U[str, Func1]):
    """
    Generate a new file `outfile` with the faded sndfile

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
    samples, sr = readSndfile(sndfile)
    npsnd.arrayFade(samples, sr, fadetime=fadetime, mode=mode, shape=shape)
    sndwriteLike(sndfile, outfile, samples)


def copyFragment(path:str, start:float, end:float, outfile:str):
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
        end = sndinfo(path).duration - end
    samples, sr = readSndfile(path, start=start, end=end)
    sndwriteLike(path, outfile, samples)


def process(sourcefile: str,
            outfile:str,
            callback: processfunc_t,
            timerange:Tuple[float, float]=None,
            bufsize=4096) -> None:
    """
    Process samples of sourcefile in chunks of `bufsize` by calling
    `callback` on each chunk. Write them to outfile.

    Args:
        sourcefile: the file to read
        outfile: the file to write to
        callback: a function taking ``(data, pos, now)`` and returning processed data
            or None if data is processed inplace
        timerange: if given, only the samples within the times given will be processed
    """
    ifile = pysndfile.PySndfile(sourcefile)
    ofile = cloneWrite(sourcefile, outfile)
    sr = ifile.samplerate()
    if timerange is None:
        timerange = (0, ifile.frames()/sr)
    frame0 = int(timerange[0] * ifile.samplerate())
    ifile.seek(frame0)
    begin, end = timerange
    frames_to_read = int((end - begin) * sr)
    for pos, size in chunks(0, frames_to_read, bufsize):
        now = pos/sr
        data = ifile.read_frames(size)
        data2 = callback(data, pos, now) if callback else data
        ofile.write_frames(data2)


def gain(filename:str, factor:U[float, _FloatFunc], outfile:str) -> None:
    """
    Change the volume of a audiofile.

    Args:
        filename: the soundfile to read
        factor: a number between 0-1 or a callable (\t -> gain_at_t)
        outfile: the output filename
    """
    if callable(factor):
        factorfunc = bpf.asbpf(factor)
        return _dynamic_gain(filename, factorfunc, outfile)

    def callback(data, pos, now):
        data *= factor
        return data
    process(filename, outfile, callback)


def asSndfile(snd: U[str, pysndfile.PySndfile], rewind=True) -> pysndfile.PySndfile:
    """
    If snd is a path, open that to read; if it is an opened file, return it

    Args:
        snd: either a path as string, or a PySndfile.
        rewind: in the case of receiving a PySndfile, should we rewind it?

    Returns:
        a PySndfile
    """
    if isinstance(snd, pysndfile.PySndfile):
        if rewind:
            snd.seek(0)
        return snd
    if not isinstance(snd, str):
        raise TypeError("path should be a string")
    return pysndfile.PySndfile(snd)


def encodingFromDtype(dtype:str, filetype:str) -> str:
    """
    Return the pysndfile encoding corresponding to the given dtype & filetype

    Args:
        dtype: the dtype of a samples array
        filetype: the file format ('wav', 'aif', 'flac', etc.)

    Returns:
        the encoding as str.

    """
    if dtype in (np.float64, np.float32):
        encoding = 'float32'
    elif dtype in (np.int16,):
        encoding = 'pcm16'
    elif dtype in (np.int32, np.int64):
        encoding = 'pcm24'
    else:
        encoding = 'pcm16'
    if filetype == 'flac':
        encoding = 'pcm24'
    return encoding


def _dynamic_gain(sndfile: str, curve: bpf.BpfInterface, outfile='inplace') -> None:
    """
    Apply a dynamic gain to sndfile

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

    def callback(data, pos, now):
        factor = curve.mapn_between(len(data), now, now+chunkdur)
        data *= factor
        return data
    
    process(sndfile, outfile, callback)
    if inplace:
        os.rename(outfile, sndfile)


def mix(sources: List[np.ndarray], offsets:List[int]=None) -> np.ndarray:
    """
    Mix the sources together. All sources should have the same amount of channels.
    It is assumed that they share the same samplerate

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


def _read_chunks(s: pysndfile.PySndfile,
                 chunksize:int,
                 start=0., end=0.) -> Iter[Tuple[np.ndarray, int]]:
    sr = s.samplerate()
    start_frame = int(start*sr)
    if start_frame>0:
        s.seek(start_frame)
    if end<=0:
        end_frame = s.frames()
    else:
        end_frame = int(end*sr)
    for pos, length in chunks(start_frame, end_frame, chunksize):
        data = s.read_frames(length)
        yield (data, pos)


def readChunks(sndfile:str,
               chunksize:int=None, chunkdur:float=None,
               start=0.0, end=0.0
               ) -> Iter[Tuple[np.ndarray, int]]:
    """
    Read chunks of data from sndfile. Each chunk has aduration of `chunksize`
    in seconds but can have less if there are not enough samples to read
    (at the end of the file)

    .. note::
        Either chunksize or chunkdur can be given, not both

    Args:
        sndfile: the soundfile to read (a str or an already open PySndfile)
        chunksize: size of the chunk in samples
        chunkdur: dur. of the chunk (in secs). NB: either chunksize or chunkdur can be given
        start: start time to read (in seconds)
        end: and time to read (in seconds). 0 = end of file

    Returns:
        a Tuple (datachunk, position_in_frames)

    """
    assert not (chunksize is not None and chunkdur is not None)
    if chunksize is None and chunkdur is None:
        chunksize = 4096
    s = pysndfile.PySndfile(sndfile)
    chunksize = chunksize if chunksize is not None else int(chunkdur*s.samplerate())
    return _read_chunks(s, chunksize=chunksize, start=start, end=end)


def equalPowerPan(pan:float) -> Tuple[float, float]:
    """
    pan is a float from 0 to 1

    returns 2 floats which correspond to the multiplication
    factor for two signals so that an equal-power stereo pan results

    """
    return sqrt(1-pan), sqrt(pan)


def sndinfo(sndfile:str) -> SndInfo:
    """
    Read information about a soundfile

    Returns:
        a :class:`SndInfo` structure with attributes (samplerate: int,
        numchannels: int, numframes: int, encoding: str)
    """
    snd = pysndfile.PySndfile(sndfile)
    return SndInfo(snd.samplerate(), snd.channels(), snd.frames(),
                   snd.encoding_str())


def peakbpf(filename:str, resolution=0.01, method='peak', channel:U[int, str]= 'mix',
            normalize=False) -> bpf.BpfInterface:
    """
    return a BPF representing the peaks envelope of the sndfile with the
    resolution given

    Args:
        filename: the file to analyze
        resolution: resolution of the bpf
        method: one of 'peak', 'rms' or 'mean'
        channel: either a channel number (starting with 0) or 'mix'
        normalize: if True, peaks are normalized before constructing the bpf

    Returns:
        a bpf holding the peaks curve
    """
    info = sndinfo(filename)
    srate = info.samplerate
    chunk_step = int(srate*resolution)
    assert chunk_step > 0
    funcs = {
        'peak': lambda arr: max(numpyx.minmax1d(arr)),
        'rms': npsnd.rms,
        'mean': lambda arr: np.abs(arr).mean()
    }
    func = funcs.get(method)
    if func is None:
        raise ValueError(f"Unknown {method} method")

    times: List[float] = []
    peaks: List[float] = []
    channum = channel if isinstance(channel, int) else -1

    for chunk, pos in readChunks(filename):
        if info.numchannels > 1:
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
    return bpf.core.Linear(timesarr, peaksarr)


def maxPeak(filename: str, start:float=0, end:float=0, resolution=0.01
            ) -> Tuple[float, float]:
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
    info = sndinfo(filename)
    for data, pos in readChunks(filename, start=start, end=end, chunkdur=resolution):
        np.abs(data, data)
        peak = np.max(data)
        if peak > maximum_peak:
            maximum_peak = peak
            max_pos = pos
    return max_pos / info.samplerate, maximum_peak


def _find_first_sound(sndfile:pysndfile.PySndfile, threshold=-120,
                      resolution=0.01, start=0.
                      ) -> Opt[float]:
    """
    Find the time when the first sound appears in the soundfile
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
    chunksize = int(resolution*sndfile.samplerate())
    for chunk, pos in _read_chunks(sndfile, chunksize=chunksize, start=start):
        if npsnd.rms(chunk) > minamp:
            return pos/sndfile.samplerate()
    return None



def findFirstSound(sndfile:str, threshold=-120,
                   resolution=0.01, start=0.
                   ) -> Opt[float]:
    """
    Find the time when the first sound appears in the soundfile
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
    return _find_first_sound(pysndfile.PySndfile(sndfile), threshold=threshold,
                             resolution=resolution, start=start)


def findLastSound(sndfile:str, threshold=-120, resolution=0.01) -> Opt[float]:
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
    s = asSndfile(sndfile)
    frames = s.read_frames()
    period = int(resolution * s.samplerate())
    frame = npsnd.lastSound(frames, threshold=threshold, period=period)
    return frame/s.samplerate() if frame is not None else None


def stripSilence(sndfile:str, outfile:str, threshold=-100, margin=0.050,
                 fadetime=0.1, mode='both') -> None:
    """
    Remove silence at the beginning and at the end of the sndfile, which fall 
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
    f = pysndfile.PySndfile(sndfile)
    if mode == 'both' or mode == 'left':
        t0 = _find_first_sound(f)
        f.seek(0, 0)
        t0 -= margin
    else:
        t0 = 0
    data = f.read_frames()
    samplerate = f.samplerate()
    if mode == 'both' or mode == 'right':
        lastFrame = npsnd.lastSound(data, threshold=threshold)
        if lastFrame is None:
            t1 = len(data) / samplerate
        else:
            t1 = lastFrame / samplerate
            t1 += margin
    else:
        t1 = f.frames()/samplerate
    frame0 = max(0, int(t0 * samplerate))
    frame1 = min(len(data), int(t1*samplerate))
    new_data = data[frame0:frame1]

    if fadetime > 0:
        npsnd.arrayFade(data, samplerate, fadetime=fadetime, mode='inout',
                        shape='linear', margin=32)

    out = openWrite(outfile, npsnd.numChannels(new_data),
                    samplerate=f.samplerate(), encoding=f.encoding_str())
    out.write_frames(new_data)


def normalize(path:str, outfile:str, peak=0.) -> None:
    """
    Normalize the given soundfile. Returns the path of the normalized file
    If outfile is not given, a new file with the suffix "-N" is generated
    based on the input
    
    Args:
        path: the path to the soundfile
        outfile: the path to the outfile
        peak: the peak to normalize to, in dB
    
    """
    sndfile = pysndfile.PySndfile(path)
    sndfile.seek(0, 0)
    peak = db2amp(peak)
    chunksize = 8096
    max_amp = max(np.abs(chunk).max()
                  for chunk, pos in _read_chunks(sndfile, chunksize))
    ratio = peak / max_amp
    sndfile.seek(0, 0)     
    out = _cloneWrite(sndfile, outfile)
    for chunk, pos in _read_chunks(sndfile, chunksize=chunksize):
        chunk *= ratio
        out.write_frames(chunk)
  

def _pysndfileGetFormat(extension:str, encoding:str):
    """
    Return the numeric format corresponding to the given
    extension + encoding

    Args:
        extension: One of the supported filetypes
        encoding : One of 'pcm16', 'pcm24', 'pcm32', 'float32', 'float64'
    
    Returns:
        the format
    """
    fmt, bits = encoding[:3], int(encoding[3:])
    if fmt == 'flt':
        fmt = 'float'
    assert fmt in ('pcm', 'float') and bits in (8, 16, 24, 32, 64)
    if extension[0] == '.':
        extension = extension[1:]
    if extension == 'aif':
        extension = 'aiff'
    fmt = "%s%d" % (
        {'pcm': 'pcm', 
         'flt': 'float'}[fmt],
        bits
    )
    return pysndfile.construct_format(extension, fmt)


def _cloneWrite(reference: pysndfile.PySndfile, outfile:str, sr:int=None, channels:int=None,
                encoding:str=None
                ) -> pysndfile.PySndfile:
    chan = channels or reference.channels()
    encoding = encoding or reference.encoding_str()
    sr = sr or reference.samplerate()
    return openWrite(outfile, channels=chan, samplerate=sr, encoding=encoding)


def cloneWrite(likefile: str, outfile:str, sr:int=None, channels:int=None,
               encoding:str=None
               ) -> pysndfile.PySndfile:
    """
    Given an existing soundfile, open a new soundfile to write to with
    the same characteristics (sr, encoding, channels) as the original file, 
    possibly modifying some of these characteristics

    Args:
        likefile: reference file
        outfile: the outfile to create for writing
        sr: if given, overrides the sr in likefile
        channels: if given, overrides the num. of channels in reference
        encoding: if given, overrides encoding in reference

    Returns:
        the pysndfile.PySndfile instance to use for writing. Write frames
        by calling :meth:`write_frames`

    See also: open_to_write
    """
    s = pysndfile.PySndfile(likefile)
    return _cloneWrite(s, outfile=outfile, sr=sr, channels=channels, encoding=encoding)


def openWrite(filename:str, channels:int=1, samplerate:int=44100, encoding:str= 'float32'
              ) -> pysndfile.PySndfile:
    """
    Open a soundfile to write data to it. The currently used backend is pysndfile.

    Example::

        >>> outfile = openWrite("out.wav", 1, 44100)
        >>> outfile.write_frames(numpyarray)

    .. note::
        the number of channels in numpy array must match the channels of the file
        For mono, the shape must be (numsamples,)
        For stereo, the shape must be (numsamples, 2)
        To stack two channels, use ``numpy.column_stack((chan1, chan2))``

    .. seealso::
        :meth:`sndwrite`


    Not all formats support all encodings.

    ========     ====== ====== ====== ====== ======
    format       pcm16  pcm24  pcm32  flt32  flt64
    ========     ====== ====== ====== ====== ======
    wav/aiff     OK     OK     OK     OK     OK
    wavpack      OK     OK     OK     OK     OK
    flac         OK     OK     --     --     --
    ========     ====== ====== ====== ====== ======
    """
    ext = os.path.splitext(filename)[1]
    fmt = _pysndfileGetFormat(ext, encoding)
    return pysndfile.PySndfile(filename, 'w', format=fmt,
                               channels=channels, samplerate=samplerate)

def guessEncoding(data: np.ndarray, fmt:str) -> str:
    if fmt in {'wav', 'aif', 'aiff', 'wv'}:
        return 'float32'
    elif fmt == 'flac':
        return 'pcm24'
    else:
        raise ValueError(f"Format {fmt} not supported")

def sndwrite(data: np.ndarray, sr:int, filename:str, encoding='auto') -> None:
    """
    Writes all samples at once

    Args:
        data: sample data to wrte
        sr: samplerate
        filename: path to the output file
        encoding: encoding to use, one of pcm16, pcm24, pcm32, float32, float64

    Not all formats support all encodings.

    ========     ====== ====== ====== ====== ======
    format       pcm16  pcm24  pcm32  flt32  flt64
    ========     ====== ====== ====== ====== ======
    wav/aiff     OK     OK     OK     OK     OK
    wavpack      OK     OK     OK     OK     OK
    flac         OK     OK     --     --     --
    ========     ====== ====== ====== ====== ======

    """
    if encoding == 'auto':
        encoding = guessEncoding(data, os.path.splitext(filename)[1][1:])
    channels = npsnd.numChannels(data)
    f = openWrite(filename, channels, sr, encoding=encoding)
    f.write_frames(data)


def sndwriteLike(reference:str, oufile:str, data:np.ndarray, sr:int=None) -> None:
    """
    Write samples to outfile using ``reference`` as reference for sr/encoding

    Args:
        reference (str): the path to the soundfile used as reference
        oufile (str): the outfile
        data (np.ndarray): the samples
        sr (int, optional): needed if samplerate is different from reference

    """
    info = sndinfo(reference)
    sr = sr or info.samplerate
    return sndwrite(data, sr=sr, filename=oufile, encoding=info.encoding)


def detectRegions(sndfile:str, attackthresh:float, decaythresh:float,
                  mindur=0.020, func='rms', resolution=0.004, mingap:float=0,
                  normalize=False
                  ) -> Tuple[List[Tuple[float, float]], bpf.BpfInterface]:
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
    bsmooth = bpf.util.smooth((b+db2amp(-160)).apply(amp2db), mindur/8)
    regions = []
    Y = bsmooth.sample(resolution)
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
    if mingap > 0:
        mergedregions = []
        last_region = (-mingap*2, -mingap*2)
        for region in regions:
            gap = region[0] - last_region[1]
            if gap >= mingap:
                mergedregions.append(last_region)
                last_region = region
            else:
                # add the new region to the last region and hold it,
                # do not append yet to new regions
                last_region = (last_region[0], region[1])
        if mergedregions and last_region[-1] != mergedregions[-1][-1]:
            mergedregions.append(last_region)
        regions = [region for region in mergedregions
                   if region[0] >= b.x0 and region[1] <= b.x1]
    mask = bpf.nointerpol(*flatten([((x0, 1), (x1, 0)) for x0, x1 in regions]))
    return regions, mask


def extractRegions(sndfile:str,
                   times:List[Tuple[float, float]],
                   outfile:str,
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
    s = pysndfile.PySndfile(sndfile)
    nframes, channels = s.frames(), s.channels()
    o = _cloneWrite(s, outfile)
    regions, sr = readRegions(sndfile, times)
    for region in regions:
        npsnd.arrayFade(region, sr, fadetimes[0], mode='in', shape=fadeshape)
        npsnd.arrayFade(region, sr, fadetimes[1], mode='out', shape=fadeshape)
    if mode == 'original':
        shape = (nframes, channels) if channels > 1 else (nframes,)
        samples_out = np.zeros(shape, dtype=float)
        for region, (t0, t1) in zip(regions, times):
            offset = int(t0 * sr)
            samples_out[offset:offset + len(region)] += region
        o.write_frames(samples_out)
    elif mode == 'seq':
        for region, (t0, t1) in zip(regions, times):
            o.write_frames(region)
    else:
        raise ValueError(f"Expected 'seq', or 'original', got {mode}")


def readRegions(sndfile:str, times:List[Tuple[float, float]]
                ) -> Tuple[List[np.ndarray], int]:
    """
    Extract regions from a soundfile

    Args:
        sndfile: the path to a sound-file
        times: a list of tuples (start, end).

    Returns:
        a tuple (list of arrays, samplerate)
    """
    s = pysndfile.PySndfile(sndfile)
    out = []
    sr = s.samplerate()
    for t0, t1 in times:
        fr0 = int(t0 * sr)
        fr1 = int(t1 * sr)
        s.seek(fr0)
        new_samples = s.read_frames(fr1 - fr0)
        out.append(new_samples)
    return out, sr


def readSndfile(sndfile:str, start=0., end=-1.) -> sample_t:
    """
    Read a soundfile, or a fraction of it

    Args:
        sndfile: a path to a soundfile
        start: the starting time
        end: the end time, or None to read until the end

    Returns:
        a tuple (frames, samplerate)
    """
    f = pysndfile.PySndfile(sndfile)
    sr = f.samplerate()
    frame0 = int(start * sr)
    frame1 = int(end * sr) if end >= 0 else f.frames()
    if start > 0:
        f.seek(frame0)
    samples = f.read_frames(frame1 - frame0)
    return (samples, sr)


def addSilentChannel(monofile:str, outfile:str) -> None:
    """
    Given a sndfile, return a new sndfile with an added channel of silence 

    .. note::
        the use-case for this is to add a silent channel to a mono file, to make
        clear that the right channel should be silent and the left channel should
        not be played also through the right channel
    """

    samples, sr = sndfileio.sndread(monofile)
    numchannels = npsnd.numChannels(samples)
    if numchannels != 1:
        logger.warning(f"{monofile} expected to be mono, but contains {numchannels}!")
    numsamples = len(samples)
    silence = np.zeros((numsamples,), dtype=float)
    data = np.column_stack((samples, silence))
    sndfileio.sndwrite_like(outfile=outfile, samples=data, likefile=monofile)


def _getsamples(source: U[str, sample_t]) -> sample_t:
    """
    source can be: "/path/to/sndfile" or (samples, sr) 
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


def scrub(sndfile: U[str, Tuple[np.ndarray, int]], curve: bpf.BpfInterface,
          rewind=False, outfile:str=None) -> sample_t:
    """
    Scrub soundfile with curve

    Args:
        sndfile: the path to a sndfile, or a Tuple (samples, samplerate)
        curve: a bpf representing real_time:time_in_soundfile
        rewind: if True, do not include silence at the beginning if
                       the bpf does not start at 0
        outfile: if given, samples are written to disk

    Returns:
        a tuple (samples, samplerate)
    """
    samples, sr = _getsamples(sndfile)
    samplebpf = curve.core.Sampled(samples, 1.0/sr)
    warped = curve|samplebpf
    newsamples = warped[curve.x0:curve.x1:1.0/sr].ys
    if not rewind and curve.x0 > 0:
        out = np.zeros((sr*curve.x1,), dtype=float)
        out[-len(newsamples):] = newsamples
        newsamples = out
    if outfile is not None:
        if isinstance(sndfile, str):
            sndwriteLike(sndfile, outfile, newsamples)
        else:
            sndwrite(newsamples, sr, outfile)
    return (newsamples, sr)