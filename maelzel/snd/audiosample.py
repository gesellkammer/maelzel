"""
audiosample
~~~~~~~~~~~

a Sample containts the audio of a soundfile as a
numpy and it knows about its samplerate
It can also perform simple actions (fade-in/out,
cut, insert, reverse, normalize, etc)
on its own audio destructively or return a new Sample.
"""

from __future__ import annotations
import numpy as np
import os
from math import sqrt
import tempfile
import shutil
import subprocess
import bpf4 as _bpf
import pysndfile
import logging
import sys
from pathlib import Path

from configdict import ConfigDict
import numpyx
from emlib.pitchtools import amp2db, db2amp
from emlib import numpytools

from maelzel.snd import csoundengine
from maelzel.snd import sndfiletools
from maelzel.snd.resample import resample as _resample

from typing import List, Tuple, Optional as Opt, Union as U, Iterator as Iter, Sequence as Seq

logger = logging.getLogger("emlib:audiosample")

_initWasDone = False


def _configCheck(config, key, oldvalue, newvalue):
    if key == 'editor':
        if os.path.exists(newvalue) or shutil.which(newvalue):
            return oldvalue
        logger.error(f"Trying to set editor to {newvalue}."
                     " File not found")
        return oldvalue


config = ConfigDict(name='emlib:audiosample',
                    default={
                        'editor': 'audacity',
                        'fade.shape': 'linear',
                    },
                    precallback=_configCheck)


def _increase_suffix(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    tokens = name.split("-")
    newname = None
    if len(tokens) > 1:
        suffix = tokens[-1]
        try:
            suffixnum = int(suffix)
            newname = "{}-{}".format(name[:-len(suffix)], suffixnum + 1)
        except ValueError:
            pass
    if newname is None:
        newname = name + '-1'
    return newname + ext


def _arrays_match_length(a: np.ndarray, b: np.ndarray, mode='longer', pad=0
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    match the lengths of arrays a and b

    if mode is 'longer', then the shorter array is padded with 'pad'
    if mode is 'shorter', the longer array is truncated
    """
    assert isinstance(a, np.ndarray), ("a: Expected Array, got " +
                                       str(a.__class__))
    assert isinstance(b, np.ndarray), ("b: Expected Array, got " +
                                       str(b.__class__))
    lena = len(a)
    lenb = len(b)
    if lena == lenb:
        return a, b
    if mode == 'longer':
        maxlen = max(lena, lenb)
        if pad == 0:
            func = np.zeros_like
        else:
            func = lambda arr: np.ones_like(arr) * pad
        if lena < maxlen:
            tmp = func(b)
            tmp[:lena] = a
            return tmp, b
        else:
            tmp = func(a)
            tmp[:lenb] = b
            return a, tmp
    elif mode == 'shorter':
        minlen = min(lena, lenb)
        if lena > minlen:
            return a[:minlen], b
        else:
            return a, b[:minlen]

    else:
        raise ValueError("Mode not understood"
                         "It must be either 'shorter' or 'longer'")


def split_channels(sndfile:str, labels:List[str]=None, basename='') -> List[Sample]:
    """
    split a multichannel sound-file and name the individual files
    with a suffix specified by labels

    sndfile: the path to a soundfile
    labels: a list of labels (strings)
    basename: if given, used as the base name for the new files

    """
    if isinstance(sndfile, str):
        if basename is None:
            basename = sndfile
    else:
        raise TypeError("sndfile must be the path to a soundfile")
    s = Sample(sndfile)
    if labels is None:
        labels = ["%0d" % (ch + 1) for ch in range(s.channels)]
    assert s.channels == len(labels)
    base, ext = os.path.splitext(basename)
    sndfiles = []
    for ch, label in enumerate(labels):
        filename = "%s-%s%s" % (base, label, ext)
        s_ch = s.get_channel(ch)
        s_ch.write(filename)
        sndfiles.append(s_ch)
    return sndfiles


def _normalize_path(path:str) -> str:
    path = os.path.expanduser(path)
    return os.path.abspath(path)


def open_in_editor(filename:str, wait=False, app=None) -> None:
    editor = app or config['editor']
    filename = _normalize_path(filename)
    if sys.platform == 'darwin':
        os.system(f'open -a "{editor}" "{os.path.abspath(filename)}"')
    elif sys.platform == 'linux':
        proc = subprocess.Popen(args=[editor, filename])
        if wait:
            logger.debug("open_in_editor: waiting until finished")
            proc.wait()
    else:
        logger.debug("open_in_editor: using windows routine")
        proc = subprocess.Popen(f'"{editor}" "{filename}"', shell=True)
        if wait:
            proc.wait()


class Sample:
    def __init__(self, sound:U[str, Path, np.ndarray], samplerate:int=None, start=0., end=0.) -> None:
        """
        sound: str, a Path or np.array
            either sample data or a path to a soundfile
        samplerate: only needed if passed an array
        start, end: sec
            if a path is given, it is possible to read a fragment of the data
            end can be negative, in which case it starts counting from the end
        """
        if isinstance(sound, (str, Path)):
            path = str(sound)
            tmp = Sample.read(path, start=start, end=end)
            samples = tmp.samples
            samplerate = tmp.samplerate
        elif isinstance(sound, np.ndarray):
            assert samplerate is not None
            samples = sound
        else:
            raise TypeError(
                "sound should be a path to a sndfile or a seq. of samples")
        self.samples: np.ndarray = samples
        self.samplerate: int = samplerate
        self.channels = numchannels(self.samples)
        self._asbpf: Opt[_bpf.BpfInterface] = None
        self._csound_table: Opt[csoundengine.TableProxy] = None   # A cached csound table, for playback

    @property
    def nframes(self) -> int:
        return self.samples.shape[0]

    def __repr__(self):
        s = "Sample: dur=%f sr=%d ch=%d" % (self.duration, self.samplerate,
                                            self.channels)
        return s

    @property
    def duration(self) -> float:
        return len(self.samples) / self.samplerate

    @classmethod
    def read(cls, filename:str, start=0., end=0.) -> Sample:
        """
        Read samples from `filename`.

        Args:
            filename: the filename of the soundfile
            start: the time to start reading
            end: the end time to stop reading (0 to read until the end)

        Returns:
            a Sample
        """
        sndfile = pysndfile.PySndfile(filename)
        sr = sndfile.samplerate()
        if end == 0:
            endsample = sndfile.frames()
        else:
            endsample = int(sr * end)
            assert endsample < sndfile.frames()
        if start > 0:
            startsample = int(start * sr)
            sndfile.seek(startsample)
        else:
            startsample = 0
        samples = sndfile.read_frames(endsample - startsample)
        return cls(samples, samplerate=sr)

    def _make_csound_table(self) -> csoundengine.TableProxy:
        if self._csound_table is not None:
            return self._csound_table
        manager = csoundengine.getManager()
        engine = manager.getEngine()
        tabnum = engine.makeEmptyTable(len(self.samples)*self.channels,
                                       numchannels=self.channels, sr=self.samplerate)
        engine.fillTable(tabnum, self.samples.flatten(), block=True)
        self._csound_table = csoundengine.TableProxy(tabnum, freeself=True, manager=manager)
        return self._csound_table

    def _play_csound(self, loop=False, chan:int=1, gain=1., delay=0., pan=None
                     ) -> csoundengine.AbstrSynth:
        """
        Play the given sample
        """
        table = self._make_csound_table()
        assert table.manager is not None
        if pan is None:
            pan = -1
        synth = table.manager.playSample(table.tabnum, chan=chan, gain=gain,
                                         loop=loop, delay=delay, pan=pan)
        return synth

    def play(self, loop=False, chan=1, delay=0., gain=1., pan=None) -> csoundengine.AbstrSynth:
        """
        Play is asynchronouse.

        Args:
            loop: if True, loop the sound
            chan: output channel.
            pan: a value between 0-1 (0=left, 1=right) or None to use a default
            delay: if > 0, schedule playback in the future
            gain: adjust the gain of playback

        Returns:
            a csoundengine.AbstrSynth. You can call .stop on this
            to stop playback.

        """
        return self._play_csound(loop=loop, chan=chan, delay=delay, gain=gain, pan=pan)

    def asbpf(self) -> _bpf.BpfInterface:
        if self._asbpf not in (None, False):
            return self._asbpf
        else:
            self._asbpf = _bpf.core.Sampled(self.samples, 1 / self.samplerate)
            return self._asbpf

    def plot(self, profile='auto') -> None:
        """
        plot the sample data

        profile: one of 'low', 'medium', 'high'
        """
        from . import plotting
        plotting.plot_samples(self.samples, self.samplerate, profile=profile)

    def plot_spectrograph(self, framesize=2048, window='hamming', at=0., dur=0.) -> None:
        """
        window: As passed to scipy.signal.get_window
                `blackman`, `hamming`, `hann`, `bartlett`, `flattop`, `parzen`, `bohman`,
                `blackmanharris`, `nuttall`, `barthann`, `kaiser` (needs beta),
                `gaussian` (needs standard deviation)

        Plots the spectrograph of the entire sample (slice before to use only
        a fraction)

        See Also: .spectrum
        """
        from . import plotting
        if self.channels > 1:
            samples = self.samples[:, 0]
        else:
            samples = self.samples
        s0 = 0 if at == 0 else int(at * self.samplerate)
        s1 = self.nframes if dur == 0 else min(self.nframes,
                                               int(dur * self.samplerate) - s0)
        if s0 > 0 or s1 != self.nframes:
            samples = samples[s0:s1]
        plotting.plot_power_spectrum(samples,
                                     self.samplerate,
                                     framesize=framesize,
                                     window=window)

    def plot_spectrogram(self,
                         fftsize=2048,
                         window='hamming',
                         overlap=4,
                         mindb=-120) -> None:
        """
        fftsize: the size of the fft
        window: window type. One of 'hamming', 'hanning', 'blackman', ...
                (see scipy.signal.get_window)
        mindb: the min. amplitude to plot
        """
        from . import plotting
        if self.channels > 1:
            samples = self.samples[:, 0]
        else:
            samples = self.samples
        return plotting.spectrogram(samples,
                                    self.samplerate,
                                    window=window,
                                    fftsize=fftsize,
                                    overlap=overlap,
                                    mindb=mindb)

    def open_in_editor(self, wait=True, app=None,
                       fmt='wav') -> Opt[Sample]:
        """
        Open the sample in an external editor. The original
        is not changed.

        Args:
            wait: if wait, the editor is opened in blocking mode,
                the results of the edit are returned as a new Sample
            app: if given, this application is used to open the sample.
                Otherwise, the application configured via the key 'editor'
                is used
            fmt: the format to write the samples to

        Returns:
            if wait, it returns the sample after closing editor

        """
        assert fmt in {'wav', 'aiff', 'aif', 'flac'}
        sndfile = tempfile.mktemp(suffix="." + fmt)
        self.write(sndfile)
        logger.debug(f"open_in_editor: opening {sndfile}")
        open_in_editor(sndfile, wait=wait, app=app)
        if wait:
            return Sample.read(sndfile)
        return None

    def write(self, outfile: str, bits: int = None, check=True, **metadata) -> str:
        """
        write the samples to outfile

        outfile: the name of the soundfile. The extension
                 determines the file format
        bits: the number of bits. 32 bits and 64 bits are
              floats, if the format allows it.
              If None, the best resolution is chosen
        """
        ext = (os.path.splitext(outfile)[1][1:]).lower()
        if bits is None:
            if ext in ('wav', 'aif', 'aiff'):
                bits = 32
            elif ext == 'flac':
                bits = 24
            else:
                raise ValueError("extension should be wav, aif or flac")
        if check and bits<=24:
            minval, maxval = numpyx.minmax1d(self.get_channel(0).samples)
            if minval<-1 or maxval>1:
                raise ValueError(f"Trying to save as pcm data, but range is higher than -1, 1 ({minval}, {maxval}")
        o = open_sndfile_to_write(outfile,
                                  channels=self.channels,
                                  samplerate=self.samplerate,
                                  bits=bits)
        o.write_frames(self.samples)
        if metadata:
            _modify_metadata(outfile, metadata)
        return outfile

    def copy(self) -> Sample:
        """
        return a copy of this Sample
        """
        return Sample(self.samples.copy(), self.samplerate)

    def _changed(self) -> None:
        self._csound_table = None

    def __add__(self, other: U[float, Sample]) -> Sample:
        if isinstance(other, Sample):
            s0, s1, sr = broadcast_samplerate(self, other)
            assert isinstance(s0, np.ndarray)
            assert isinstance(s1, np.ndarray)
            s0, s1 = _arrays_match_length(s0, s1, mode='longer')
            return Sample(s0 + s1, samplerate=sr)
        else:
            # not a Sample
            return Sample(self.samples + other, self.samplerate)

    def __iadd__(self, other: U[float, Sample]) -> None:
        if isinstance(other, Sample):
            s0, s1, sr = broadcast_samplerate(self, other)
            s0, s1 = _arrays_match_length(s0, s1, mode='longer')
            s0 += s1
            self.samples = s0
            self.samplerate = sr
        else:
            self.samples += other
        self._changed()

    def __mul__(self, other: U[float, Sample]) -> Sample:
        if isinstance(other, Sample):
            s0, s1, sr = broadcast_samplerate(self, other)
            s0, s1 = _arrays_match_length(s0, s1)
            return Sample(s0 * s1, sr)
        elif callable(other):
            other = _mapn_between(other, len(self.samples), 0, self.duration)
        return Sample(self.samples * other, self.samplerate)

    def __pow__(self, other: float) -> Sample:
        return Sample(self.samples**other, self.samplerate)

    def __imul__(self, other: U[float, Sample]) -> None:
        if isinstance(other, Sample):
            s0, s1, sr = broadcast_samplerate(self, other)
            s0, s1 = _arrays_match_length(s0, s1)
            s0 *= s1
            self.samples = s0
            self.samplerate = sr
            self._changed()
        elif callable(other):
            other = _mapn_between(other, len(self.samples), 0, self.duration)
        self.samples *= other
        self._changed()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: slice) -> Sample:
        """
        samples support slicing

        sample[start:stop] will return a new Sample consisting of a slice
        of this sample between the times start and stop. As it is a slice
        of this Sample, any changes inplace will be reflected in the original
        samples. To avoid this, use .copy:

        # Copy the fragment between seconds 1 and 3
        fragment = original[1:3].copy()

        To slice at the sample level, access .samples directly

        s = Sample("path")

        # This is a Sample
        sliced_by_time = s[fromtime:totime]

        # This is a numpy array
        sliced_by_samples = s.samples[fromsample:tosample]
        """
        if not isinstance(item, slice):
            raise ValueError(
                "We only support sample[start:end]."
                "To access individual samples, use sample.samples[index]")
        start, stop, step = item.start, item.stop, item.step
        if stop is None:
            stop = self.duration
        if start is None:
            start = 0.
        if step is not None:
            return self.resample(step)[start:stop]
        stop = min(stop, self.duration)
        start = min(start, self.duration)
        assert 0 <= start <= stop
        frame0 = int(start * self.samplerate)
        frame1 = int(stop * self.samplerate)
        return Sample(self.samples[frame0:frame1], self.samplerate)

    def fade(self, fadetime:float, mode='inout', shape:str=None) -> Sample:
        """
        Fade this Sample **in place**, returns self

        Args:
            fadetime: the duration of the fade
            mode: 'in', 'out', 'inout'
            shape: the shape of the fade  TODO!!
        """
        if shape is None:
            shape = config['fade.shape']
        sndfiletools.fade_array(self.samples,
                                self.samplerate,
                                fadetime=fadetime,
                                mode=mode,
                                shape=shape)
        return self

    def prepend_silence(self, dur:float) -> Sample:
        """
        Return a new Sample with silence of given dur at the beginning
        """
        silence = make_silence(dur, self.channels, self.samplerate)
        return concat([silence, self])

    def normalize(self, headroom=0.) -> Sample:
        """Normalize in place, returns self

        headroom: maximum peak in dB
        """
        max_peak_possible = db2amp(headroom)
        peak = np.abs(self.samples).max()
        ratio = max_peak_possible / peak
        self.samples *= ratio
        return self

    def peak(self) -> float:
        """return the highest sample value (dB)"""
        return amp2db(np.abs(self.samples).max())

    def peaksbpf(self, res=0.01) -> _bpf.core.Linear:
        """
        Return a BPF representing the peaks envelope of the sndfile with the
        resolution given

        res: resolution in seconds
        """
        samplerate = self.samplerate
        chunksize = int(samplerate * res)
        X, Y = [], []
        data = self.samples if self.channels == 1 else self.samples[:, 0]
        data = np.abs(data)
        for pos in np.arange(0, self.nframes, chunksize):
            maximum = np.max(data[pos:pos + chunksize])
            X.append(pos / samplerate)
            Y.append(maximum)
        return _bpf.core.Linear(X, Y)

    def reverse(self) -> Sample:
        """ reverse the sample in-place """
        self.samples[:] = self.samples[-1::-1]
        self._changed()
        return self

    def rmsbpf(self, dt=0.01, hop=1) -> _bpf.core.Sampled:
        """
        Return a bpf representing the rms of this sample as a function of time
        """
        s = self.samples
        period = int(self.samplerate * dt + 0.5)
        dt = period / self.samplerate
        hopsamps = int(period * hop)
        numperiods = int(len(s) / hopsamps)
        values: List[float] = []
        data = np.empty((numperiods,), dtype=float)
        for i in range(numperiods):
            idx0 = i * hopsamps
            chunk = s[idx0:idx0+period]
            data[i] = rms(chunk)
        return _bpf.core.Sampled(values, x0=0, dx=dt)

    def rms(self) -> float:
        return rms(self.samples)

    def mono(self) -> Sample:
        """
        Return a new Sample with this sample downmixed to mono
        Returns self if already mono
        """
        if self.channels == 1:
            return self
        return Sample(mono(self.samples), samplerate=self.samplerate)

    def remove_silence_left(self, threshold=-120.0, margin=0.01, window=0.02) -> Sample:
        """
        See remove_silence
        """
        period = int(window * self.samplerate)
        first_sound_sample = first_sound(self.samples, threshold, period)
        if first_sound_sample >= 0:
            time = max(first_sound_sample / self.samplerate - margin, 0)
            return self[time:]
        return self

    def remove_silence_right(self, threshold=-120.0, margin=0.01, window=0.02) -> Sample:
        """
        See remove_silence
        """
        period = int(window * self.samplerate)
        lastsample = last_sound(self.samples, threshold, period)
        if lastsample >= 0:
            time = min(lastsample / self.samplerate + margin, self.duration)
            return self[:time]
        return self

    def remove_silence(self, threshold=-120.0, margin=0.01, window=0.02) -> Sample:
        """
        Remove silence from the sides. Returns a new Sample

        threshold: dynamic of silence, in dB
        margin: leave at list this amount of time between the first/last sample
                and the beginning of silence or
        window: the duration of the analysis window, in seconds
        """
        out = self.remove_silence_left(threshold, margin, window)
        out = out.remove_silence_right(threshold, margin, window)
        return out

    def resample(self, samplerate: int) -> Sample:
        """
        Return a new Sample with the given samplerate
        """
        if samplerate == self.samplerate:
            return self
        samples = _resample(self.samples, self.samplerate, samplerate)
        return Sample(samples, samplerate=samplerate)

    def scrub(self, bpf: _bpf.BpfInterface) -> Sample:
        """
        bpf: a bpf mapping time -> time

        Example 1: Read sample at half speed

        dur = sample1.duration
        sample2 = sample1.scrub(bpf.linear((0, 0),
                                           (dur*2, dur)
                                           ))
        """
        samples, sr = sndfiletools.scrub((self.samples, self.samplerate), bpf,
                                         rewind=False)
        return Sample(samples, self.samplerate)

    def get_channel(self, n:int, contiguous=True) -> Sample:
        """
        return a new mono Sample with the given channel
        """
        if self.channels == 1 and n == 0:
            return self
        if n > (self.channels - 1):
            raise ValueError("this sample has only %d channel(s)!" %
                             self.channels)
        newsamples = self.samples[:, n]
        if contiguous and not newsamples.flags.c_contiguous:
            newsamples = np.ascontiguousarray(newsamples)
        return Sample(newsamples, self.samplerate)

    def estimate_freq(self, start=0.2, dur=0.15, strategy='autocorr') -> float:
        """
        estimate the frequency of the sample (in Hz)

        start: where to start the analysis (the beginning
               of a sample is often not very clear)
        dur: duration of the fragment to analyze
        strategy: one of 'autocorr' or 'fft'
        """
        t0 = start
        t1 = min(self.duration, t0 + dur)
        s = self.get_channel(0)[t0:t1]
        from .freqestimate import freq_from_autocorr, freq_from_fft
        func = {
            'autocorr': freq_from_autocorr,
            'fft': freq_from_fft
        }.get(strategy, freq_from_autocorr)
        return func(s.samples, s.samplerate)

    def fundamental_bpf(self, fftsize=2048, stepsize=512, method="pyin") -> _bpf.core.Linear:
        if method == "pyin":
            from maelzel.ext import sonicannotator
            tmpwav = tempfile.mktemp(suffix=".wav")
            self.write(tmpwav)
            bpf = sonicannotator.pyin_smooth_pitch(tmpwav, fftsize=fftsize, stepsize=stepsize, threshdistr=1.5)
            os.remove(tmpwav)
            return bpf
        elif method == "autocorrelation":
            from maelzel.snd import freqestimate
            steptime = stepsize/self.samplerate
            bpf = freqestimate.frequency_bpf(self.samples, sr=self.samplerate, steptime=steptime,
                                             method='autocorrelation')
            return bpf
        else:
            raise ValueError(f"method should be one of 'pyin', 'autocorrelation', but got {method}")

    def chord_at(self, t: float, resolution=30, method='sndtrck', **kws):
        if method == 'sndtrck':
            try:
                import sndtrck
            except ImportError:
                raise ImportError("sndtrck is needed (https://github.com/gesellkammer/sndtrck)")
            margin = 0.15
            t0 = max(0., t - margin)
            t1 = min(self.duration, t + margin)
            s = sndtrck.analyze_samples(self[t0:t1].samples,
                                        self.samplerate,
                                        resolution=resolution,
                                        hop=2)
            chord = s.chord_at(t - t0)
            return chord
        else:
            raise ValueError(f"method {method} not supported. Possible methods: 'sndtrck'")

    def chunks(self, chunksize:int, hop:int=None, pad=False) -> Iter[np.ndarray]:
        """
        Iterate over the samples in chunks of chunksize. If pad is True,
        the last chunk will be zeropadded, if necessary
        """
        return numpytools.chunks(self.samples,
                                 chunksize=chunksize,
                                 hop=hop,
                                 padwith=(0 if pad else None))

    def first_sound(self, threshold=-120.0, period=0.04, hopratio=0.5, start=0.) -> float:
        idx = first_sound(self.samples,
                          threshold=threshold,
                          periodsamps=int(period * self.samplerate),
                          hopratio=hopratio,
                          skip=int(start*self.samplerate))
        return idx / self.samplerate if idx >= 0 else -1

    def first_silence(self, threshold=-80, period=0.04, hopratio=0.5,
                      soundthreshold=-50, start=0.):
        """

        Args:
            threshold: rms value which counts as silence
            period: the time period to calculate the rms
            hopratio: how much to slide between rms calculations, as a fraction of the period
            soundthreshold: rms value which counts as sound
            start: start time (0=start of sample)

        Returns:

        """
        idx = first_silence(samples=self.samples,
                            threshold=threshold,
                            period=int(period*self.samplerate),
                            hopratio=hopratio,
                            soundthreshold=soundthreshold,
                            startidx=int(start*self.samplerate))
        return idx/self.samplerate if idx > 0 else -1


def first_sound(samples, threshold=-120.0, periodsamps=256, hopratio=0.5, skip=0) -> int:
    """
    Find the first sample in samples whith a rms
    exceeding the given threshold

    Returns: time of the first sample holding sound or -1 if
             no sound found
    """
    threshold_amp = db2amp(threshold)
    hopsamples = int(periodsamps * hopratio)
    i = 0
    while True:
        i0 = skip + i * hopsamples
        i1 = i0 + periodsamps
        if i1 > len(samples):
            break
        chunk = samples[i0:i1]
        rms_now = rms(chunk)
        if rms_now > threshold_amp:
            return i0
    return -1


def first_silence(samples: np.ndarray, threshold=-100, period=256,
                  hopratio=0.5, soundthreshold=-60, startidx=0) -> int:
    """
    Return the sample where rms decays below threshold

    Args:
        samples: the samples data
        threshold: the threshold in dBs (rms)
        period: how many samples to use for rms calculation
        hopratio: how many samples to skip before taking the next
            measurement
        soundthreshold: the threshold to considere that the sound
            started (rms)
        startidx: the sample to start looking for silence (0 to start
            from beginning)

    Returns:
        the index where the first silence is found (-1 if no silence found)

    """
    soundstarted = False
    hopsamples = int(period * hopratio)
    thresholdamp = db2amp(threshold)
    soundthreshamp = db2amp(soundthreshold)
    lastrms = rms(samples[startidx:startidx+period])
    idx = hopsamples + startidx
    while idx < len(samples) - period:
        win = samples[idx:idx+period]
        rmsnow = rms(win)
        if rmsnow >= soundthreshamp:
            soundstarted = True
        elif rmsnow <= thresholdamp and lastrms > thresholdamp and soundstarted:
            return idx
        lastrms = rmsnow
        idx += hopsamples
    return -1


def last_sound(samples: np.ndarray, threshold=-120.0, period=256, hopratio=1.0) -> int:
    """
    Find the end of the last sound in the samples.
    (the last time where the rms is lower than the given threshold)

    Returns -1 if no sound is found
    """
    samples1 = samples[::-1]
    i = first_sound(samples1,
                    threshold=threshold,
                    periodsamps=period,
                    hopratio=hopratio)
    if i < 0:
        return i
    return len(samples) - (i + period)


def rms(arr: np.ndarray) -> float:
    """
    calculate the root-mean-square of the array
    """
    return sqrt((arr**2).sum() / len(arr))


def broadcast_samplerate(a: Sample, b: Sample) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Match the samplerates of audio samples a and b to the highest one
    the audio sample with the lowest samplerate is resampled to the
    higher one.

    a, b: Sample instances

    Returns: (resampled_a, resampled_b, new_samplerate)
    """
    assert isinstance(a, Sample)
    assert isinstance(b, Sample)
    if a.samplerate == b.samplerate:
        return a.samples, b.samples, a.samplerate
    sr = max(a.samplerate, b.samplerate)
    if sr == a.samplerate:
        samples0, samples1 = a.samples, _resample(b.samples, b.samplerate, sr)
    else:
        samples0, samples1 = _resample(a.samples, a.samplerate, sr), b.samples
    assert isinstance(samples0, np.ndarray)
    assert isinstance(samples1, np.ndarray)
    return samples0, samples1, sr


def numchannels(samples: np.ndarray) -> int:
    """
    Returns the number of channels held by the `samples` array
    """
    assert isinstance(samples, np.ndarray)
    return 1 if len(samples.shape) == 1 else samples.shape[1]


def _as_numpy_samples(samples: U[Sample, np.ndarray]) -> np.ndarray:
    if isinstance(samples, Sample):
        return samples.samples
    elif isinstance(samples, np.ndarray):
        return samples
    else:
        return np.asarray(samples, dtype=float)


def as_sample(source: U[str, Sample, Tuple[np.ndarray, int]]) -> Sample:
    """
    return a Sample instance

    input can be a filename, a Sample or a tuple (samples, samplerate)
    """
    if isinstance(source, str):
        return Sample.read(source)
    if isinstance(source, Sample):
        return source
    if isinstance(source, tuple) and isinstance(source[0], np.ndarray):
        samples, sr = source
        return Sample(samples, sr)
    else:
        raise TypeError("can't convert source to Sample")


def mono(samples: U[Sample, np.ndarray]) -> np.ndarray:
    """
    If samples are multichannel, it mixes down the samples
    to one channel.
    """
    samples = _as_numpy_samples(samples)
    channels = numchannels(samples)
    if channels == 1:
        return samples
    return np.sum(samples, axis=1) / channels


def concat(sampleseq: Seq[Sample]) -> Sample:
    """
    sampleseq: a seq. of Samples

    concat the given Samples into one Sample.
    Samples should share samplingrate and numchannels
    """
    s = np.concatenate([s.samples for s in sampleseq])
    return Sample(s, sampleseq[0].samplerate)


def _mapn_between(func, n:int, t0:float, t1:float) -> np.ndarray:
    """
    Returns: a numpy array of n-size, mapping func between t0-t1
             at a rate of n/(t1-t0)

    Args:
        func: a callable of the form func(float) -> float, can be a bpf
    """
    if hasattr(func, 'mapn_between'):
        ys = func.mapn_between(n, t0, t1)  # is it a Bpf?
    else:
        X = np.linspace(t0, t1, n)
        ufunc = np.vectorize(func)
        Y = ufunc(X)
        return Y
    return ys


def _modify_metadata(path: str, metadata: dict) -> None:
    """
    possible keys:

    description     \
    originator       |
    orig-ref         |
    umid             | bext
    orig-date        |
    orig-time        |
    coding-hist     /

    title
    copyright
    artist
    comment
    date
    album
    license
    """
    possible_keys = {
        "description": "bext-description",
        "originator": "bext-originator",
        "orig-ref": "bext-orig-ref",
        "umid": "bext-umid",
        "orig-date": "bext-orig-time",
        "coding-hist": "bext-coding-hist",
        "title": "str-title",
        "copyright": "str-copyright",
        "artist": "str-artist",
        "comment": "str-comment",
        "date": "str-date",
        "album": "str-album"
    }
    args = []
    for key, value in metadata.items():
        key2 = possible_keys.get(key)
        if key2 is not None:
            args.append(' --%s "%s"' % (key2, str(value)))
    os.system('sndfile-metadata-set %s "%s"' % (" ".join(args), path))


def open_sndfile_to_write(filename:str, channels=1, samplerate=48000, bits:int=None) -> pysndfile.PySndfile:
    """
    The format is inferred from the extension (wav, aiff, flac, etc.)

    if bits is given, it is used. otherwise it is inferred from the format
    """
    encodings = {
        'wav': {
            16: "pcm16",
            24: "pcm24",
            32: "float32"
        },
        'aif': {
            16: "pcm16",
            24: "pcm24",
            32: "float32",
        },
        'flac': {
            16: "pcm16",
            24: "pcm24",
            32: "pcm24"
        }
    }
    base, ext = os.path.splitext(filename)
    ext = ext[1:].lower()
    if not ext or ext not in encodings:
        raise ValueError(f"The extension ({ext}) is not supported")

    encoding = encodings[ext].get(bits)
    if encoding is None:
        raise ValueError(f"no format possible for {ext} with {bits} bits")
    fmt = pysndfile.construct_format(ext, encoding)
    return pysndfile.PySndfile(filename,
                               'w',
                               format=fmt,
                               channels=channels,
                               samplerate=samplerate)


def make_silence(dur:float, channels:int, sr:int) -> Sample:
    """
    Generate a silent Sample with the given characteristics
    """
    if channels == 1:
        samples = np.zeros((int(dur * sr), ), dtype=float)
    else:
        samples = np.zeros((int(dur * sr), channels), dtype=float)
    return Sample(samples, sr)
