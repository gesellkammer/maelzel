"""
Utilities to edit sound-files in chunks
"""
from __future__ import annotations
import tempfile
import os
from math import sqrt
from collections import namedtuple as _namedtuple
from dataclasses import dataclass
import pysndfile
import numpy as np
import bpf4 as bpf
from emlib.iterlib import flatten
from emlib.lib import returns_tuple
from emlib.pitchtools import db2amp, amp2db

from emlib import typehints as t
import logging


logger = logging.Logger("emlib.sndfiletools")


@dataclass
class Chunk:
    data: np.ndarray
    position: int


@dataclass
class Samples:
    samples: np.ndarray
    samplerate: int


@dataclass
class SndInfo:
    samplerate: int
    numchannels: int
    numframes: int
    encoding: str

    @property
    def duration(self) -> float:
        return self.numframes/self.samplerate


_Bpf = bpf.BpfInterface
_FloatFunc = t.Callable[[float], float]


def chunks(start:int, stop:int=None, step:int=None) -> t.Iter[tuple[int, int]]:
    """
    Like xrange, but returns a tuplet (position, distance form last position)

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


def add_suffix(filename:str, suffix:str) -> str:
    """
    add a suffix between the name and the extension

    add_suffix("test.txt", "-OLD") == "test-OLD.txt"
    """
    name, ext = os.path.splitext(filename)
    return ''.join((name, suffix, ext))


def fade_array(samples:np.ndarray, srate:int, fadetime:float,
               mode='inout', shape='linear') -> np.ndarray:
    """
    fade samples in place

    samples: numpy array
    srate: samplerate
    fadetime: fade time in seconds
    mode: in, out, or inout
    fadeshape: either a string describing the shape (one of 'linear', 'expon(x)', halfcos)
               or a callable (t) -> gain, defined between 0 - 1
    margin: number of samples to set to 0 at the edge of a fade
    """
    assert isinstance(samples, np.ndarray)
    assert isinstance(srate, int)
    assert isinstance(fadetime, float)
    assert isinstance(mode, str)
    assert isinstance(shape, str)
    margin=0
    
    def mult(samples, ramp):
        numch = array_numchannels(samples)
        if numch == 1:
            samples *= ramp
        else:
            for ch in range(numch):
                samples[:, ch] *= ramp
                
    fadeframes = int(fadetime * srate)
    numframes = len(samples)
    ramp = _mkramp(shape, fadeframes)
    if mode in ('in', 'inout'):
        mult(samples[margin:fadeframes+margin], ramp)
        if margin > 0:
            samples[:margin] = 0
    if mode in ('out', 'inout'):
        frame0 = max(0, len(samples) - fadeframes - margin)
        frame1 = min(frame0 + fadeframes, numframes)
        mult(samples[frame0:frame1], ramp[::-1])
        if margin > 0:
            samples[-margin:] = 0


def fade_sndfile(sndfile:str, outfile:str, fadetime:float, mode:str, shape) -> str:
    """
    Generate a new file `outfile` with the faded sndfile

    Args:
        sndfile: infile
        outfile: outfile. Use None to generate a name based on the infile
        fadetime: the fade time, in seconds
        mode: one of 'in', 'out', 'inout'
        shape: a string describing the shape (one of 'linear', 'expon(x)', halfcos)
                 or a callable (t) -> gain, defined between 0 - 1

    Returns:
        the path of the generated outfile
    """
    samples = read_sndfile(sndfile)
    fade_array(samples.samples, samples.samplerate,
               fadetime=fadetime, mode=mode, shape=shape)
    if outfile is None:
        base, ext = os.path.splitext(sndfile)
        outfile = f"{base}-fade{ext}"

    clone_to_write(sndfile, outfile).write_frames(samples)
    return outfile


def array_numchannels(A: np.ndarray) -> int:
    """ Returns the number of channels represented by the given array"""
    return 1 if len(A.shape) == 1 else A.shape[1]
    

def is_samplesource(source) -> bool:
    """
    a sample source is defined by (samples, sr)
    """
    return (isinstance(source, tuple) and 
            len(source) == 2 and 
            isinstance(source[0], np.ndarray) and
            isinstance(source[1], int))


def is_sndfile(filename: str) -> bool:
    """
    determine if filename can be interpreted as a soundfile
    """
    base, ext = os.path.splitext(filename)
    return ext.lower() in (".wav", ".aif", ".aiff", ".flac", ".wv")


def copy_fragment(path:str, begin:float, end:float,
                  outfile:str=None, suffix='-CROP') -> str:
    """
    Read the data between begin and end (in seconds) and write it to outfile

    Args:
        path: the file to copy from
        begin: the start time
        end: end time
        outfile: path of the saved fragment. If None, it will be constructed
            with suffix
        suffix: the suffix to add to path if no outfile is given

    Returns:
        the outfile

    """
    if outfile is None:
        outfile = add_suffix(path, suffix)
    if end is None or end == -1:
        end = get_info(path).duration

    process(path, outfile, callback=None, timerange=(begin, end))
    return outfile

def process(sourcefile, outfile, callback, timerange=None, bufsize=4096):
    # type: (str, str, t.Callable[[np.ndarray, int, int], np.ndarray], t.Opt[t.Tup[float, float]], int) -> None
    """
    Process samples of sourcefile in chunks of `bufsize` by calling
    `callback` on each chunk.
    Write them to outfile.

    callback: a function taking (data, pos, now) and returning processed data
              Or None if data is processed inplace
    timerange: if given, onle the samples within the times given
               will be processed
    """
    ifile = pysndfile.PySndfile(sourcefile)
    ofile = clone_to_write(sourcefile, outfile)
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


def gain(filename:str, factor:t.U[float, _FloatFunc], outfile:str) -> None:
    """
    Change the volume of a audiofile.

    factor: a number between 0-1 or a callable (\t -> gain_at_t)
    """
    if callable(factor):
        return _dynamic_gain(filename, factor, outfile)

    def callback(data, pos, now):
        data *= factor
        return data
    process(filename, outfile, callback)


def as_sndfile(snd: t.U[str, pysndfile.PySndfile]) -> pysndfile.PySndfile:
    if isinstance(snd, pysndfile.PySndfile):
        return snd
    if not isinstance(snd, str):
        raise TypeError("path should be a string")
    return pysndfile.PySndfile(snd)


def encoding_from_dtype(dtype:str, filetype:str) -> str:
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


def _dynamic_gain(sndfile: str, curve, outfile='inplace') -> None:
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


def _mkramp(desc:str, numsamples:int) -> np.ndarray:
    """
    desc: A string descriptor of a function. Possible descriptors are:
          - "linear"
          - "expon(x)"
          - "halfcos"
    numsamples: The number of samples to generate a ramp from 0 to 1

    Returns: a numpy array of shape (numsamples,), ramping from 0 to 1
    """
    assert isinstance(desc, str)
    assert isinstance(numsamples, int)
    return bpf.util.makebpf(desc, [0, 1], [0, 1]).map(numsamples)


def mix(sources, offsets=None):
    # type: (t.List[np.ndarray], t.Opt[float]) -> np.ndarray
    """
    Mix the sources together

    sources: list of arrays
             Each source can have an time offset (in seconds)
    offsets    : a list of time offsets (optional)
    """
    offsets = [0] * len(sources) if offsets is None else offsets
    nchannels = sources[0].shape[1] if len(sources[0].shape) > 1 else 1
    end = max(((len(source) + offset)
               for source, offset in zip(sources, offsets)))
    out = np.zeros((end, nchannels), dtype=np.float32)
    for source, t in zip(sources, offsets):
        if nchannels > 1:
            for channel in range(nchannels):
                out[t:len(source), channel] += source[:, channel]
        else:
            out[t:len(source)] += source
    return out


def read_chunks(sndfile:str, chunksize:int=None,
                chunkdur:float=None, start=0.0, end=0.0
                ) -> t.Iter[tuple[np.ndarray, int]]:
    """
    read chunks of data from sndfile. each chunk has a
    duration of `chunksize` in seconds but can have less
    if there are not enough samples to read (for instance
    at the end of the file)
    
    Returns: a tuple (datachunk, position_in_frames)

    sndfile        : the soundfile to read
    chunksize      : size of the chunk in samples
    chunkdur       : dur. of the chunk (in secs). NB: either chunksize or chunkdur can be given
    start, end     : start and end time to read (in seconds)

    """
    s = as_sndfile(sndfile)
    srate = s.samplerate()
    start_frame = int(start * srate)
    if start_frame > 0:
        s.seek(start_frame)
    if chunksize is None and chunkdur is None:
        chunksize = 4096
    chunksize = chunksize if chunksize is not None else int(chunkdur * srate)
    if end <= 0:
        end_frame = s.frames()
    else:
        end_frame = int(end * srate)
    for pos, length in chunks(start_frame, end_frame, chunksize):
        data = s.read_frames(length)
        yield (data, pos)


def equal_power(pan:float) -> tuple[float, float]:
    """
    pan is a float from 0 to 1

    returns 2 floats which correspond to the multiplication
    factor for two signals so that an equal-power stereo pan results

    """
    return sqrt(1-pan), sqrt(pan)


def get_info(filename:str) -> SndInfo:
    """
    return a SndInfo
    (sample_rate, channels, size_in_samples)
    """
    snd = pysndfile.PySndfile(filename)
    return SndInfo(snd.samplerate(), snd.channels(), snd.frames(),
                   snd.encoding_str())


def _process(sndfile, func, *args):
    # type: (str, t.Callable[[np.ndarray, int, *Any], None]) -> None
    """
    helper func, will call func with chunks of the soundfile

    func is of the form: def func(data, pos, *args)
    (args is optional)
    """
    for data, pos in read_chunks(sndfile):
        assert len(data) > 0
        func(data, pos, *args)


def _process_fragment(sndfile, start, end, func, *args):
    # type: (str, float, float, t.Callable[[np.ndarray, int, *Any], None], *Any) -> None
    source = as_sndfile(sndfile)
    sr = source.samplerate
    frame0 = int(start * sr)
    frame1 = int(end * sr)
    if frame1 > source.nframes:
        frame1 = source.nframes
    if frame0 >= frame1:
        raise ValueError('the fragment should at least be 1 sample long')
    read_pointer = source.seek(0, 1)
    if frame0 >= read_pointer:
        source.seek(frame0 - read_pointer, 1)
    else:
        source.seek(frame0, 0)
    for pos, chunklen in chunks(0, frame1-frame0, 1024):
        data = source.read_frames(chunklen)
        func(data, pos+frame0, *args)


def peakbpf(filename, res=0.01, func='peak', normalize=False):
    # type: (str, float, str, bool) -> _Bpf
    """
    return a BPF representing the peaks envelope of the sndfile with the
    resolution given

    res: resolution of the bpf
    func: a function taking a numpy array or one of 'peak', 'rms' or 'mean'
    """
    sndinfo = get_info(filename)
    srate = sndinfo.samplerate
    chunk_step = int(srate * res)
    peaks = []
    assert chunk_step > 0
    funcs = {
        'peak': np.max,
        'rms': rms,
        'mean': lambda arr: np.abs(arr).mean()
    }
    if isinstance(func, str):
        func = funcs.get(func)

    def process_mult(data, pos, peaks, func):
        data = abs(data[:, 0])
        for chunk_pos, chunk_length in chunks(0, len(data), chunk_step):
            peaktime = (pos+chunk_pos) / srate
            peak = func(data[chunk_pos:chunk_pos+chunk_length])
            peaks.append((peaktime, peak))

    def process_mono(data, pos, peaks, func):
        data = abs(data)
        for chunk_pos, chunk_length in chunks(0, len(data), chunk_step):
            peaktime = (pos+chunk_pos) / srate
            peak = func(data[chunk_pos:chunk_pos+chunk_length])
            peaks.append((peaktime, peak))
        
    loop = process_mono if sndinfo.numchannels == 1 else process_mult
    _process(filename, loop, peaks, func)
    X, Y = list(zip(*peaks))
    maxpeak = max(Y)
    if normalize:
        Y = Y / maxpeak
    return bpf.core.Linear.fromxy(X, Y)


def maximum_peak(filename, start=None, end=None):
    # type: (str, t.Opt[float], t.Opt[float]) -> float
    """
    return the maximum value of any sample at the given filename
    the return value if a float between 0.0 and 1.0
    """
    maximum_peak = 0
    for data, pos in read_chunks(filename, start=start, end=end):
        np.abs(data, data)
        peak = np.max(data)
        if peak > maximum_peak:
            maximum_peak = peak
    return maximum_peak


def rms(arr: np.ndarray) -> float:
    """
    arr: a numpy array
    """
    arr = np.abs(arr)
    arr **= 2
    return sqrt(np.sum(arr) / len(arr))


def find_first_sound(sndfile:str, threshold=-120, resolution=0.01, start=0.
                     ) -> float:
    """
    Find the time when the first sound appears in the soundfile
    (or, what is the same, the length of any initial silence at the
    beginning of the soundfile)

    Args:
        sndfile: The path to the soundfile
        threshold: The volume threshold defining silence, in dB
        resolution: the time resolution, in seconds
        start: where to start searching (in seconds)

    Returns: time of the first sound (in seconds)
    """
    f = as_sndfile(sndfile)
    minamp = db2amp(threshold)
    pos = 0
    for chunk, pos in read_chunks(f, chunkdur=resolution, start=start):
        if rms(chunk) > minamp:
            break
    return pos/f.samplerate()


def _find_last_sound(samples, samplerate, threshold, resolution):
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


def find_last_sound(sndfile, threshold=-120, resolution=0.01):
    # type: (str, float) -> float
    """
    Find the time when the last sound fades into silence
    (or, what is the same, the length of any silence at the
    end of the soundfile)

    sndfile: The path to the soundfile 
    threshold: The volume threshold defining silence, in dB
    resolution: the time resolution, in seconds

    Returns the time of the last sound
    """
    s = as_sndfile(sndfile)
    frames = s.read_frames()
    return _find_last_sound(frames, s.samplerate(), threshold=threshold, resolution=resolution)
    

def strip_silence(sndfile, threshold=-100, margin=0.050,
                  fadetime=0.1, outfile=None, mode='both'):
    # type: (str, float, float, float, t.Opt[str]) -> t.Opt[np.ndarray]
    """
    Remove silence at the beginning and at the end of
    the sndfile, which fall below threshold (dB)

    margin: Determines how much silence is left at the edges (in seconds)
    fadetime: Indicates of a fade in/out are performed
    outfile: if given, writes the result there.
             Otherwise, the samples are returned as a numpy array
    mode: one of 'left', 'right', 'both' (default=both)
    """
    sndfile = as_sndfile(sndfile)
    if mode == 'both' or mode == 'left':
        t0 = find_first_sound(sndfile)
        sndfile.seek(0, 0)
        t0 -= margin
    else:
        t0 = 0
    data = sndfile.read_frames()
    samplerate = sndfile.samplerate()
    if mode == 'both' or mode == 'right':
        t1 = _find_last_sound(data, samplerate, threshold=threshold, resolution=0.01)
        t1 += margin
    else:
        t1 = sndfile.frames() / samplerate
    frame0 = max(0, int(t0 * samplerate))
    frame1 = min(len(data), int(t1*samplerate))
    new_data = data[frame0:frame1]

    if fadetime > 0:
        fade_array(data, samplerate, fadetime=fadetime, mode='inout', shape='linear', margin=32)

    if outfile is not None:
        clone_to_write(sndfile, outfile).write_frames(new_data)
    else:
        return new_data


def _clip(x, x0, x1):
    # type: (float, float, float) -> float
    return x0 if x < x0 else (x1 if x > x1 else x)


def _calculate_chunk_duration(sndfile):
    # type: (str) -> float
    srate = sndfile.samplerate()
    return _clip(sndfile.nframes / 10., 64, srate * 10) / srate


def normalize(path, peak=-1.5, outfile=None):
    # type: (str, float, t.Opt[str]) -> t.U[str, np.ndarray]
    """
    Normalize the given soundfile. Returns the path of the normalized file
    If outfile is not given, a new file with the suffix "-N" is generated
    based on the input
    
    :param path: the path to the soundfile
    :param peak: the peak to normalize to, in dB
    :param outfile: the path to the outfile, or None to just return the samples
    :return: the path of the generated soundfile 
    """
    sndfile = as_sndfile(path)
    sndfile.seek(0, 0)
    peak = db2amp(peak)
    chunksize = 8096
    max_amp = max(np.abs(chunk).max()
                  for chunk, pos in read_chunks(sndfile, chunksize))
    ratio = peak / max_amp
    sndfile.seek(0, 0)     # goto beginning of file
    if outfile is None:
        outfile = add_suffix(path, '-N')
    new_sndfile = clone_to_write(sndfile, outfile)
    write_frames = new_sndfile.write_frames
    for chunk, pos in read_chunks(sndfile, chunksize=chunksize):
        write_frames(chunk * ratio)
    return outfile


def clone_to_write(sndfile: str, outfile: str) -> pysndfile.PySndfile:
    """
    Return an sndfile open to write with
    the same format and channels of `sndfile`
    """
    sndfile = as_sndfile(sndfile)
    return pysndfile.PySndfile(filename=outfile, mode='w',
                               format=sndfile.format(),
                               channels=sndfile.channels(),
                               samplerate=sndfile.samplerate())


def _pysndfile_get_sndfile_format(extension, encoding):
    """
    Return the numeric format corresponding to the given
    extension + encoding

    extension: One of the supported filetypes
    encoding : One of 'pcm16', 'pcm24', 'pcm32', 'float32', 'float64'
    """
    fmt, bits = encoding[:3], int(encoding[3:])
    if fmt == 'flt':
        fmt == 'float'
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


def open_to_write_like(likefile: str, outfile:str, sr:int=None, channels:int=None, encoding:str=None
                       ) -> 'pysndfile.PySndfile':
    """
    Given an existing soundfile, open a new soundfile to write to with
    the same characteristics (sr, encoding, channels) as the original file, 
    possibly modifying some of these characteristics
    
    See also: open_to_write
    """
    info = get_info(likefile)
    chan = channels or info.numchannels
    encoding = encoding or info.encoding
    sr = sr or info.samplerate
    return open_to_write(outfile, channels=chan, samplerate=sr, encoding=encoding)


def open_to_write(filename:str, channels:int=1, samplerate:int=44100, encoding:str='float32'
                  ) -> 'pysndfile.PySndfile':
    """
    Open a soundfile to write data to it. The currently supported backend
    is pysndfile. 

    Example:

    outfile = open_to_write("out.wav", 1, 44100)
    outfile.write_frames(numpyarray)

    NB: the number of channels in numpyarray must match the channels of the file
        For mono, the shape must be (numsamples,)
        For stereo, the shape must be (numsamples, 2)
            To stack two channels, use numpy.column_stack: numpy.column_stack((chan1, chan2))

    See also: sndwrite
    ^^^^^^^^

    Not all formats support all encodings.
    
               | pcm16  pcm24  pcm32   flt32   flt64
    -----------+---------------------------------
    wav, aiff  | x      x      x       x       x
    flac       | x      x      -       -       -
    
    """
    ext = os.path.splitext(filename)[1]
    fmt = _pysndfile_get_sndfile_format(ext, encoding)
    return pysndfile.PySndfile(filename, 'w', format=fmt,
                               channels=channels, samplerate=samplerate)


def sndwrite(data, sr, filename, encoding='float32'):
    # type: (np.ndarray, int, str, int) -> None
    """
    Writes all samples at once

    Encoding: one of pcm16, pcm24, pcm32, float32, float64
    
    See also: open_to_write
    ^^^^^^^^

    Not all formats support all encodings.
    
               | pcm16  pcm24  pcm32   flt32   flt64
    -----------+---------------------------------
    wav, aiff  | x      x      x       x       x
    flac       | x      x      -       -       -
    
    """
    channels = array_numchannels(data)
    f = open_to_write(filename, channels, sr, encoding=encoding)
    f.write_frames(data)


def sndwrite_like(likefile:str, filename:str, data:np.ndarray, sr:int=None, encoding:str=None
                  ) -> None:
    info = get_info(likefile)
    sr = sr or info.samplerate
    encoding = encoding or info.encoding
    return sndwrite(data, sr=sr, filename=filename, encoding=encoding)


def read_fragment(sndfile, t0, t1):
    """
    source: a path to a soundfile
    t0: start time
    t1: end time

    Returns (samples, samplerate)
    """
    source = as_sndfile(sndfile)
    sr = source.samplerate()
    frame0 = int(t0 * sr)
    frame1 = int(t1 * sr)
    if frame1 > source.frames():
        frame1 = source.frames()
    if frame0 >= frame1:
        raise ValueError('the fragment should at least be 1 sample long')
    read_pointer = source.seek(0, 1)
    if frame0 >= read_pointer:
        source.seek(frame0 - read_pointer, 1)
    else:
        source.seek(frame0, 0)
    out = source.read_frames(frame1 - frame0)
    return Samples(out, sr)


@returns_tuple("regions mask")
def detect_regions(sndfile, attackthresh, decaythresh, mindur=0.020,
                   func='rms', resolution=0.004, mingap=0, normalize=False):
    # type: (str, float, float, float, str, float, float, bool) -> t.Tup[t.List[t.Tup[float, float]], _Bpf]
    """
    Detect fragments inside a soudnfile.

    attackthresh (dB): the amplitude necessary to start a region
    decaythresh (dB) : the amplitude under which to stop a region
                       (should be lower than attackthresh)
    mindur (sec): minimal duration of a region
    func: the function to use to calculate the envelope.
          One of 'rms', 'peak', 'mean'
    resolution (sec): the resolution of the analysis
    mingap (sec): the minimal gap between regions
    normalize: wether to normalize the soundfile before analysis

    Returns: a list of regions and a bpf of those regions
    """
    b = peakbpf(sndfile, res=resolution, func=func, normalize=normalize)
    bsmooth = bpf.util.smooth((b+db2amp(-160)).apply(amp2db), mindur/8)
    # bsmooth = bpf.util.smooth(b.apply(amp2db), resolution*1)
    regions = []
    Y = bsmooth.sample(resolution)
    X = np.linspace(b.x0, b.x1, len(Y))
    regionopen = False
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


def extract_regions(sndfile, times, outfile=None, mode='seq',
                    fadetimes=(0.005, 0.1), fadeshape='linear'):
    # type: (str, t.Seq[t.Tup[float, float]], t.Opt[str], str, t.Tup[float, float], str) -> str
    """
    read regions defined by `times`, write them in sequence to outfile

    sndfile   : a path to a soundfile
    times     : a seq. of times (start, end)
    outfile   : a path to the outfile, or None to append a suffix to `sndfile`
    mode      : 'seq' or 'inplace'
                seq --> extract regions and stack them sequentially.
                        end duration = sum(duration of each region)
                original --> generates a file where the regions stay in
                             their original place and everything else is erased
    fadetimes : fade applied to the samples before writing them to avoid clicks
    fadeshape : shape of the fades

    Returns --> the outfile

    Usage
    ~~~~~

    This is useful when you have extracted markers from a soundfile,
    to extract the fragments themselves to a new file
    """
    s = pysndfile.PySndfile(sndfile)
    nframes, channels = s.frames(), s.channels()
    if outfile is None:
        outfile = add_suffix(sndfile, '-OUT')
    o = clone_to_write(s, outfile)
    regions, sr = read_regions(sndfile, times)
    if s.channels > 1:
        samples_out = np.zeros((s.frames(), s.channels()), dtype=float)
    else:
        samples_out = np.zeros((s.nframes,), dtype=float)
    for region in regions:
        fade_array(region, sr, fadetimes[0], mode='in', fadeshape=fadeshape)
        fade_array(region, sr, fadetimes[1], mode='out', fadeshape=fadeshape)
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
    return outfile


@returns_tuple("arrays samplerate")
def read_regions(sndfile, times):
    # type: (str, t.Seq[t.Tup[float, float]]) -> t.Tup[t.List[np.ndarray], int]
    """
    Extract regions from a soundfile

    sndfile: the path to a sound-file
    times  : a seq of (start, end)

    Returns --> (list of arrays, samplerate)
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


def read_sndfile(sndfile:str, start=0., end=-1.) -> Samples:
    """
    Read a soundfile, or a fraction of it

    sndfile: a path to a soundfile
    start  : the starting time
    end    : the end time, or None to read until the end

    Returns --> (frames, samplerate)
    """
    f = pysndfile.PySndfile(sndfile)
    sr = f.samplerate()
    frame0 = int(start * sr)
    frame1 = int(end * sr) if end >= 0 else f.frames()
    if start > 0:
        f.seek(frame0)
    samples = f.read_frames(frame1 - frame0)
    return Samples(samples, sr)


def silent_samples(duration, sr=44100, channels=1):
    # type: (float, int, int) -> np.ndarray
    """
    Generate samples of silence with the given duration
    """
    silence = np.zeros((int(duration * sr), channels), dtype=float)
    return silence


def add_silent_channel(sourcefile, outfile):
    """
    Given a sndfile, return a new sndfile with an added channel of silence 
    
    NB: the use-case for this is to add a silent channel to a mono file, to make
        clear that the right channel should be silent and the left channel should
        not be played also through the right channel
    """
    samples, sr = read_sndfile(sourcefile)
    if array_numchannels(samples) != 1:
        logger.warn(f"{sourcefile} expected to be mono, but contains {numchannels}!")
    numsamples = len(samples)
    silence = np.zeros((numsamples,), dtype=float)
    data = np.column_stack((samples, silence))
    sndwrite_like(sourcefile, outfile, data=data)


def _getsamples(source):
    """
    source can be: "/path/to/sndfile" or (samples, sr) 
    """
    if isinstance(source, str):
        samples, sr = read_sndfile(source)
    elif isinstance(source, tuple) and len(source) == 2:
        samples, sr = source
        assert isinstance(samples, np.ndarray)
        assert isinstance(sr, (int, float))
    else:
        raise TypeError("source can be a path to a soundfile or a tuple (samples, sr)")
    return Samples(samples, sr)


def scrub(sndfile, curve, rewind=False, outfile=None):
    # type: (t.U[str, t.Tup[np.ndarray, int]], _Bpf, bool, t.Opt[str]) -> t.Tup[np.ndarray, int]
    """
    :param sndfile: the path to a sndfile, or a tuple (samples, samplerate)
    :param curve: a curve representing time:pointer (needs bpf4)
    :param rewind: if True, do not include silence at the beginning if
                   the bpf does not start at 0

    If outfile is given, writes the samples to disk

    Returns --> (samples, samplerate)
    """
    samples, srate = _getsamples(sndfile)
    sample_bpf = bpf.core.Sampled(samples, 1.0/srate)
    warped = curve | sample_bpf
    newsamples = warped[curve.x0:curve.x1:1.0/srate].ys
    if not rewind and curve.x0 > 0:
        out = np.zeros((srate*curve.x1,), dtype=float)
        out[-len(newsamples):] = newsamples
        newsamples = out
    if outfile is not None and isinstance(sndfile, str):
        clone_to_write(sndfile, outfile).write_frames(newsamples)
    return Samples(newsamples, srate)


def scrubmany(sndfile, curves, outfile=None, normalize=True, rewind=False, 
              ampcurves=None):
    """
    scrubs sndfile with multiple curves, mixes the results

    normalize: if True or a value is given, audio is normalized.
               This is useful if writing to outfile, otherwise, you can
               normalize it yourself by calling
               samples *= 1.0/(np.abs(samples).max())

    NB: reults are not normalized. After mixing all the scrubs,
        the mixed result might clip. 
    """
    samples, sr = _getsamples(sndfile)
    layers = [scrub((samples, sr), curve, rewind=True).samples for curve in curves]
    maxtime = max(curve.x1 for curve in curves)
    maxsamp = maxtime * sr + 1
    # get a buffer to hold everything
    buf = np.zeros((maxsamp,), dtype=float)
    i = 0
    for layer, curve in zip(layers, curves):
        offset = int(curve.x0 * sr)
        if ampcurves is not None:
            ampcurve = ampcurves[i]
            env = ampcurve.map(len(layer))
            layer *= env
        buf[offset:offset+len(layer)] += layer
        # buf[:len(layer)] += layer
        i += 1
    if rewind:
        firstsample = int(min(curve.x0 for curve in curves) * sr)
        buf = buf[firstsample:]
    if normalize:
        maxpeak = float(normalize)
        maxval = np.abs(buf).max()
        ratio = maxpeak/float(maxval)
        buf *= ratio
    if outfile:
        sndwrite(buf, sr, outfile)
    return Samples(buf, sr)
