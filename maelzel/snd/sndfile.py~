import pysndfile
import numpy as np
import os
from typing import Tuple, NamedTuple


class SndInfo(NamedTuple):
    sr: int
    channels: int
    frames: int
    duration: float 
    encoding: str
    majorFormat: str


def sndread(sndfile:str, start=0.0, end=0.0) -> Tuple[np.ndarray, int]:
    """
    Reads samples between start and end, returns (samples, samplerate)
    """
    ext = os.path.splitext(sndfile)[1]
    if ext == '.mp3':
        return _sndread_mp3(sndfile, start=start, end=end)
    sf = pysndfile.PySndfile(sndfile)
    sr = sf.samplerate()
    duration = sf.frames() / sr
    if end <= 0:
        end = duration - end
    if start >= end:
        raise ValueError(f"Asked to read 0 frames: start={start}, end={end}")
    if start > 0:
        if start > duration:
            raise ValueError(f"Asked to read after end of file (start={start}, duration={duration}")
        sf.seek(int(start * sr))
    frames = sf.read_frames(int((end - start)*sr))
    return frames, sr

def _convert_mp3_wav(mp3, wav, start=0, end=0):
    import subprocess
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Can't read mp3: ffmpeg is not present")
    cmd = [ffmpeg]
    if start > 0:
        cmd.extend(["-ss", str(start)])
    if end > 0:
        cmd.extend(["-t", str(end - start)])
    cmd.extend(["-i", mp3])
    cmd.append(wav)
    subprocess.call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _sndread_mp3(mp3, start=0, end=0):
    import tempfile
    wav = tempfile.mktemp(suffix=".wav")
    _convert_mp3_wav(mp3, wav, start=start, end=end)
    out = sndread(wav)
    os.remove(wav)
    return out


def sndinfo(sndfile:str) -> SndInfo:
    sf = pysndfile.PySndfile(sndfile)
    sr = sf.samplerate()
    numframes = sf.frames()
    return SndInfo(sr=sr,
                   channels=sf.channels(),
                   frames=numframes,
                   duration=numframes/sr,
                   encoding=sf.encoding_str(),
                   majorFormat=sf.major_format_str())


def sndreadchunks(sndfile, start=0.0, end=0.0, size=4096):
    """
    Returns chunks of frames between start and end. Call sndinfo to 
    query the samplerate
    """
    sf = pysndfile.PySndfile(sndfile)
    sr = sf.samplerate()
    duration = sf.frames() / sr
    if end <= 0:
        end = duration - end
    if start >= end:
        raise ValueError(f"Asked to read 0 frames: start={start}, end={end}")
    if start > 0:
        if start > duration:
            raise ValueError(f"Asked to read after end of file (start={start}, duration={duration}")
        sf.seek(int(start * sr))
    numframes = int((end - start)*sr)
    while numframes > size:
        frames = sf.read_frames(size)
        yield frames
        numframes -= size
    if numframes > 0:
        yield sf.read_frames(numframes)


def sndwrite(samples, sr, sndfile, encoding=None):
    """
    encoding: 'pcm8', 'pcm16', 'pcm24', 'pcm32', 'flt32'. 
              None to use a default based on the given extension
    """
    ext = os.path.splitext(sndfile)[1].lower()
    if encoding is None:
        encoding = _defaultEncodingForExtension(ext)
    fmt = _getFormat(ext, encoding)
    snd = pysndfile.PySndfile(sndfile, mode='w', format=fmt,
                              channels=_numchannels(samples), samplerate=sr)
    snd.write_frames(samples)
    snd.writeSync()


def _getFormat(extension:str, encoding:str):
    assert extension[0] == "."
    fmt, bits = encoding[:3], int(encoding[3:])
    assert fmt in ('pcm', 'flt') and bits in (8, 16, 24, 32)
    extension = extension[1:]
    if extension == 'aif':
        extension = 'aiff'
    fmt = "%s%d" % (
        {'pcm': 'pcm', 
         'flt': 'float'}[fmt],
        bits
    )
    return pysndfile.construct_format(extension, fmt)


def _numchannels(samples:np.ndarray) -> int:
    """
    return the number of channels present in samples

    samples: a numpy array as returned by sndread

    """
    return 1 if len(samples.shape) == 1 else samples.shape[1]


def _defaultEncodingForExtension(ext):
    if ext == ".wav" or ext == ".aif" or ext == ".aiff":
        return "flt32"
    elif ext == ".flac":
        return "pcm24"
    else:
        raise KeyError(f"extension {ext} not known")