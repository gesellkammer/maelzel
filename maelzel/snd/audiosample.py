"""
audiosample
~~~~~~~~~~~

This module is based on the :class:`Sample` class, which contains the
audio of a soundfile as a numpy array and it knows about its samplerate,
original format and encoding, etc. It can also perform simple actions
(fade-in/out, cut, insert, reverse, normalize, etc) on its own audio
destructively or return a new Sample.

External dependencies
~~~~~~~~~~~~~~~~~~~~~

For some functionality (fundamental pitch analysis) we rely at the moment
on some external dependencies

* pyin vamp plugin: https://code.soundsoftware.ac.uk/projects/pyin/files

Examples
~~~~~~~~

.. code-block:: python

    # load a Sample, fade it, play and write
    from maelzel.snd.audiosample import *
    import time
    s = Sample.read("sound.wav")
    s.fade(0.5)
    synth = s.play(speed=0.5)
    while synth.isPlaying():
        time.sleep(0.1)
    print("finished playing!")
    s.write("sound-faded.flac")
    s.plot_spectrogram()


.. code-block:: python
    samples = [
        Sample.read("soundA.wav"),
        Sample.read("soundB.aif"),
        Sample.read("soundC.flac")]
    samples = broadcast_samplerate(samples)
    a, b, c = samples
    # mix them down
    out = a.prepend_silence(2) + b + c
    out.write("mixed.wav")




"""

from __future__ import annotations
import numpy as np
import os
import tempfile
import subprocess
import bpf4 as _bpf
import pysndfile
import logging
from pathlib import Path
import shutil

import numpyx
from pitchtools import amp2db, db2amp
from emlib import numpytools
import emlib.misc

from maelzel.snd import numpysnd as npsnd
import csoundengine
import csoundengine.tableproxy
import csoundengine.synth

from maelzel.snd import sndfiletools
from maelzel.snd.resample import resample

from typing import List, Tuple, Optional as Opt, Union as U, Iterator as Iter, Sequence as Seq

logger = logging.getLogger("maelzel.audiosample")


def _normalize_path(path:str) -> str:
    path = os.path.expanduser(path)
    return os.path.abspath(path)


def openInEditor(soundfile:str, wait=False, app=None) -> None:
    """
    Open soundfile in an external sound editing appp
    
    Args:
        soundfile: the file to open
        wait: if wait, wait until editing is finished
        app: the app to use. If None is given, a default app is used

    """
    soundfile = _normalize_path(soundfile)
    emlib.misc.open_with(soundfile, app, wait=wait)


def readSoundfile(sndfile: str, start:float=0., end:float=0.) -> Tuple[np.ndarray, int]:
    """
    Read a soundfile, returns a tuple ``(samples:np.ndarray, sr:int)``

    Args:
        sndfile (str): The path of the soundfile
        start (float): The time to start reading. A negative value will seek from the end
        end (float): The time to stop reading (0=end of file). A negative value will
            seek from the end

    Returns:
        a tuple (samples:np.ndarray, samplerate:int)

    Example::

        # Read the first two seconds
        >>> samples, sr = readSoundfile("sound.flac", end=2)

        # Read the last two seconds
        >>> samples, sr = readSoundfile("sound.aif", start=-2)
    """
    sndfile = pysndfile.PySndfile(sndfile)
    sr = sndfile.samplerate()
    numframes = sndfile.frames()
    if end == 0:
        endsample = numframes
    else:
        endsample = int(sr * end)
        assert endsample < numframes
    if start < 0:
        start = numframes / sr + start
    startsample = int(start*sr)
    assert startsample < endsample
    if startsample > 0:
        sndfile.seek(startsample)
    samples = sndfile.read_frames(endsample - startsample)
    return samples, sr


def _get_csoundengine() -> csoundengine.Engine:
    engine = csoundengine.getEngine('audiosample')
    if engine:
        return engine
    return csoundengine.Engine('audiosample')


class Sample:
    """
    A class to hold audio data

    Attributes:
        samples (np.ndarray): the audio data
        samplerate (int): the samplerate of the audio data
        numchannels (int): the number of channels
        duration (float): the duration of the sample
    """
    def __init__(self, sound:U[str, Path, np.ndarray], samplerate:int=None,
                 start=0., end=0.) -> None:
        """
        Args:
            sound: str, a Path or a numpy array
                either sample data or a path to a soundfile
            samplerate: only needed if passed an array
            start: the start time (only valid when reading from a soundfile
            end: the end time (only valid when reading from a soundfile)

        .. note::
            Both start and end can be negative, in which case the frame
            is sought from the end
        """
        if isinstance(sound, (str, Path)):
            samples, samplerate = readSoundfile(sound, start=start, end=end)
        elif isinstance(sound, np.ndarray):
            assert samplerate is not None
            samples = sound
        else:
            raise TypeError("sound should be a path or an array of samples")
        self.samples: np.ndarray = samples
        self.samplerate: int = samplerate
        self.numchannels = npsnd.numChannels(self.samples)
        self._asbpf: Opt[_bpf.BpfInterface] = None
        # A cached csound table, for playback
        self._csound_table: Opt[csoundengine.tableproxy.TableProxy] = None

    @property
    def numframes(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return len(self.samples)/self.samplerate

    def __repr__(self):
        s = "Sample: dur=%f sr=%d ch=%d" % (self.duration, self.samplerate,
                                            self.numchannels)
        return s

    @classmethod
    def read(cls, filename:str, start=0., end=0.) -> Sample:
        """
        Read samples from `filename`.

        Both start and end can be negative, in which case the frame
        is sought from the end

        Args:
            filename: the filename of the soundfile
            start: the time to start reading
            end: the end time to stop reading (0 to read until the end)

        Returns:
            a Sample
        """
        samples, sr = readSoundfile(filename, start=start, end=end)
        return cls(samples, samplerate=sr)

    @classmethod
    def silent(cls, dur: float, channels: int, sr: int) -> Sample:
        """
        Generate a silent Sample with the given characteristics
        """
        if channels == 1:
            samples = np.zeros((int(dur*sr),), dtype=float)
        else:
            samples = np.zeros((int(dur*sr), channels), dtype=float)
        return cls(samples, sr)

    def _makeCsoundTable(self) -> csoundengine.tableproxy.TableProxy:
        if self._csound_table is not None:
            return self._csound_table
        engine = _get_csoundengine()
        tabnum = engine.makeEmptyTable(len(self.samples)*self.numchannels,
                                       numchannels=self.numchannels, sr=self.samplerate)
        engine.fillTable(tabnum, self.samples.flatten(), block=True)
        self._csound_table = csoundengine.tableproxy.TableProxy(tabnum,
                                                                sr=self.samplerate,
                                                                freeself=True,
                                                                engine=engine,
                                                                nchnls=self.numchannels,
                                                                numframes=self.numframes)
        return self._csound_table

    def play(self, loop=False, chan:int=1, gain=1., delay=0.,
             pan=-1, speed=1.0
             ) -> csoundengine.synth.Synth:
        """
        Play the given table

        Args:
            loop (bool): should playback be looped?
            chan (int): first channel to play to. For stereo samples, output
                is routed to consecutive channels starting with this channel
            gain (float): a gain modifier
            delay (float): start delay in seconds
            pan (float): a value between 0 (left) and 1 (right). Use -1
                for a default value, which is 0 for mono samples and 0.5
                for stereo. For 3 or more channels pan is currently ignored
            speed(float): the playback speed. A variation in speed will change
                the pitch accordingly.

        Returns:
            a :class:`csoundengine.synth.Synth`. This synth can be used to
            control playback.

        """
        table = self._makeCsoundTable()
        assert table.engine.started
        assert isinstance(pan, float)
        synth = table.engine.session().playSample(table.tabnum, chan=chan, gain=gain,
                                                  loop=loop, delay=delay, pan=pan,
                                                  speed=speed)
        return synth

    def asbpf(self) -> _bpf.BpfInterface:
        """
        Convert this sample to a bpf4.core.Sampled bpf
        """
        if self._asbpf not in (None, False):
            return self._asbpf
        else:
            self._asbpf = _bpf.core.Sampled(self.samples, 1 / self.samplerate)
            return self._asbpf

    def plot(self, profile='auto') -> None:
        """
        plot the sample data

        Args:
            profile: one of 'low', 'medium', 'high'
        """
        from . import plotting
        plotting.plot_samples(self.samples, self.samplerate, profile=profile)

    def plotSpetrograph(self, framesize=2048, window='hamming', start=0.,
                        dur=0.) -> None:
        """
        Plot the spectrograph of this sample or a fragment thereof

        Args:
            framesize: the size of each analysis, in samples
            window: As passed to scipy.signal.get_window
                    `blackman`, `hamming`, `hann`, `bartlett`, `flattop`, `parzen`,
                    `bohman`,
                    `blackmanharris`, `nuttall`, `barthann`, `kaiser` (needs beta),
                    `gaussian` (needs standard deviation)
            start: if given, plot the spectrograph at this time
            dur: if given, use this fragment of the sample (0=from start to end of
                sample)

        Plots the spectrograph of the entire sample (slice before to use only
        a fraction)
        """
        from . import plotting
        if self.numchannels > 1:
            samples = self.samples[:, 0]
        else:
            samples = self.samples
        s0 = 0 if start == 0 else int(start*self.samplerate)
        s1 = self.numframes if dur == 0 else min(self.numframes,
                                                 int(dur * self.samplerate)-s0)
        if s0 > 0 or s1 != self.numframes:
            samples = samples[s0:s1]
        plotting.plot_power_spectrum(samples,
                                     self.samplerate,
                                     framesize=framesize,
                                     window=window)

    def plotSpectrogram(self,
                        fftsize=2048,
                        window='hamming',
                        overlap=4,
                        mindb=-120) -> None:
        """
        Plot the spectrogram of this sound using matplotlib

        Args:
            fftsize: the size of the fft
            window: window type. One of 'hamming', 'hanning', 'blackman', ...
                    (see scipy.signal.get_window)
            mindb: the min. amplitude to plot
            overlap: determines the hop size (hop size in samples = fftsize/overlap)

        """
        from . import plotting
        if self.numchannels > 1:
            samples = self.samples[:, 0]
        else:
            samples = self.samples
        return plotting.spectrogram(samples,
                                    self.samplerate,
                                    window=window,
                                    fftsize=fftsize,
                                    overlap=overlap,
                                    mindb=mindb)

    def openInEditor(self, wait=True, app=None, fmt='wav'
                     ) -> Opt[Sample]:
        """
        Open the sample in an external editor.

        The original is not changed.

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
        openInEditor(sndfile, wait=wait, app=app)
        if wait:
            return Sample.read(sndfile)
        return None

    def write(self, outfile: str, bits: int = None, checkOverflow=True, **metadata
              ) -> None:
        """
        Write the samples to outfile

        Args:
            outfile: the name of the soundfile. The extension determines the 
                file format
            bits: the number of bits. 32 bits and 64 bits are floats, if the 
                format allows it. If None, the best resolution for the given
                format is used.
            checkOverflow: when saving to pcm (wav, aif), if this is True
                it is ensured that no sample would overflow (all samples are
                within the range -1, 1).
            metadata:
        """
        ext = (os.path.splitext(outfile)[1][1:]).lower()
        if bits is None:
            if ext in ('wav', 'aif', 'aiff'):
                bits = 32
            elif ext == 'flac':
                bits = 24
            else:
                raise ValueError("extension should be wav, aif or flac")
        if checkOverflow and bits<=24:
            minval, maxval = numpyx.minmax1d(self.getChannel(0).samples)
            if minval<-1 or maxval>1:
                raise ValueError(f"Trying to save as pcm data, but range is higher "
                                 f"than -1, 1 ({minval}, {maxval}")
        o = PySndFileToWrite(outfile,
                             channels=self.numchannels,
                             samplerate=self.samplerate,
                             bits=bits)
        o.write_frames(self.samples)
        if metadata:
            _modify_metadata(outfile, metadata)

    def copy(self) -> Sample:
        """
        return a copy of this Sample
        """
        return Sample(self.samples.copy(), self.samplerate)

    def _changed(self) -> None:
        self._csound_table = None

    def __add__(self, other: U[float, Sample]) -> Sample:
        if isinstance(other, (int, float)):
            return Sample(self.samples + other, self.samplerate)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.samplerate == other.samplerate
            if len(self) == len(other):
                return Sample(self.samples + other.samples, self.samplerate)
            elif len(self) > len(other):
                return Sample(self.samples[:len(other)] + other.samples, self.samplerate)
            else:
                return Sample(self.samples + other.samples[:len(self)], self.samplerate)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __iadd__(self, other: U[float, Sample]) -> None:
        if isinstance(other, (int, float)):
            self.samples += other
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.samplerate == other.samplerate
            if len(self) == len(other):
                self.samples += other.samples
            elif len(other) < len(self):
                self.samples[:len(other)] += other.samples
            else:
                self.samples += other.samples[:len(self)]
            self._changed()
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __sub__(self, other: U[float, Sample]) -> Sample:
        if isinstance(other, (int, float)):
            return Sample(self.samples - other, self.samplerate)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.samplerate == other.samplerate
            if len(self) == len(other):
                return Sample(self.samples - other.samples, self.samplerate)
            elif len(self) > len(other):
                return Sample(self.samples[:len(other)] - other.samples, self.samplerate)
            else:
                return Sample(self.samples - other.samples[:len(self)], self.samplerate)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __isub__(self, other: U[float, Sample]) -> None:
        if isinstance(other, (int, float)):
            self.samples -= other
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.samplerate == other.samplerate
            if len(self) == len(other):
                self.samples -= other.samples
            elif len(self) > len(other):
                self.samples[:len(other)] -= other.samples
            else:
                self.samples -= other.samples[:len(self)]
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")
        self._changed()

    def __mul__(self, other: U[float, Sample]) -> Sample:
        if isinstance(other, (int, float)):
            return Sample(self.samples*other, self.samplerate)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.samplerate == other.samplerate
            if len(self) == len(other):
                return Sample(self.samples * other.samples, self.samplerate)
            elif len(self) > len(other):
                return Sample(self.samples[:len(other)] * other.samples, self.samplerate)
            else:
                return Sample(self.samples * other.samples[:len(self)], self.samplerate)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __imul__(self, other: U[float, Sample]) -> None:
        if isinstance(other, (int, float)):
            self.samples *= other
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.samplerate == other.samplerate
            if len(self) == len(other):
                self.samples *= other.samples
            elif len(self) > len(other):
                self.samples[:len(other)] *= other.samples
            else:
                self.samples *= other.samples[:len(self)]
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")
        self._changed()

    def __pow__(self, other: float) -> Sample:
        return Sample(self.samples**other, self.samplerate)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: slice) -> Sample:
        """
        Samples support slicing

        ``sample[start:stop]`` will return a new Sample consisting of a slice
        of this sample between the times start and stop. As it is a slice
        of this Sample, any changes inplace will be reflected in the original
        samples. To avoid this, use :meth:`Sample.copy`.

        Example::

            # Get a slice between seconds 1.5 and 3. Any change to view will be
            # reflected in original
            >>> source = Sample.read("sound.wav")
            >>> view = source[1.5:3.0]

            # To slice at the sample level, access .samples directly
            >>> newsample = Sample(source.samples[1024:2048], source.samplerate).copy()

        """
        if not isinstance(item, slice):
            raise ValueError("Samples only support the form sample[start:end]."
                             "To access individual samples, use sample.samples[index]")
        start, stop, step = item.start, item.stop, item.step
        if stop is None:
            stop = self.duration
        if start is None:
            start = 0.
        if step is not None:
            raise ValueError("Samples do not support a step for slicing")
        stop = min(stop, self.duration)
        start = min(start, self.duration)
        assert 0 <= start <= stop
        frame0 = int(start * self.samplerate)
        frame1 = int(stop * self.samplerate)
        return Sample(self.samples[frame0:frame1], self.samplerate)

    def fade(self, fadetime:U[float, Tuple[float, float]], shape:str='linear') -> Sample:
        """
        Fade this Sample **in place**, returns self.

        If only value is given as fadetime a fade-in and fade-out is performed with
        this fadetime. A tuple can be used to apply a different fadetime for in and out.

        Args:
            fadetime: the duration of the fade.
            shape: the shape of the fade. One of 'linear', 'expon(x)', 'halfcos'

        Returns:
            self

        .. note::
            To generate a faded sample without modifying the original sample,
            use ``sample = sample.copy().fade(...)``

        Example::

            >>> sample1= Sample.read("sound.wav")
            # Fade-in and out
            >>> sample1.fade(0.2)

            >>> sample2 = Sample.read("another.wav")
            # Create a copy with a fade-out of 200 ms
            >>> sample3 = sample2.copy().fade((0, 0.2))

        """
        if isinstance(fadetime, tuple):
            fadein, fadeout = fadetime
            if fadein:
                npsnd.arrayFade(self.samples, self.samplerate, fadetime=fadein,
                                mode='in', shape=shape)
            if fadeout:
                npsnd.arrayFade(self.samples, self.samplerate, fadetime=fadeout,
                                mode='out', shape=shape)
        else:
            npsnd.arrayFade(self.samples, self.samplerate, fadetime=fadetime,
                            mode='inout', shape=shape)
        return self

    def prependSilence(self, dur:float) -> Sample:
        """
        Return a new Sample with silence of given dur at the beginning
        """
        silence = Sample.silent(dur, self.numchannels, self.samplerate)
        return concat([silence, self])

    def normalize(self, headroom=0.) -> Sample:
        """Normalize in place, returns self

        Args:
            headroom: maximum peak in dB

        Returns:
            self
        """
        ratio = npsnd.normalizationRatio(self.samples, headroom)
        self.samples *= ratio
        return self

    def peak(self) -> float:
        """return the highest sample value (dB)"""
        return amp2db(np.abs(self.samples).max())

    def peaksbpf(self, framedur=0.01, overlap=2) -> _bpf.core.Sampled:
        """
        Return a BPF representing the peaks envelope of the sndfile

        The resolution of the returned bpf will be ``framedur/overlap``

        Args:
            framedur: the duration of an analysis frame
            overlap: determines the hop time between analysis frames.
        """
        return npsnd.peaksbpf(self.samples, self.samplerate, res=framedur, overlap=overlap)

    def reverse(self) -> Sample:
        """ reverse the sample in-place, returns self """
        self.samples[:] = self.samples[-1::-1]
        self._changed()
        return self

    def rmsbpf(self, dt=0.01, overlap=1) -> _bpf.core.Sampled:
        """
        Return a bpf representing the rms of this sample over time
        """
        return npsnd.rmsbpf(self.samples, self.samplerate, dt=dt, overlap=overlap)

    def rms(self) -> float:
        """ Returns the rms of the samples (see also: :meth:`rmsbpf`) """
        return npsnd.rms(self.samples)

    def mono(self) -> Sample:
        """
        Return a new Sample with this sample downmixed to mono

        Returns self if already mono
        """
        if self.numchannels == 1:
            return self
        return Sample(npsnd.asmono(self.samples), samplerate=self.samplerate)

    def stripSilenceLeft(self, threshold=-120.0, margin=0.01, window=0.02) -> Sample:
        """
        Remove silence from the left. Returns a new Sample

        Args:
            threshold: dynamic of silence, in dB
            margin: leave at list this amount of time between the first sample
                    and the beginning of silence
            window: the duration of the analysis window, in seconds

        Returns:
            a new Sample with silence removed
        """
        period = int(window * self.samplerate)
        first_sound_sample = npsnd.firstSound(self.samples, threshold, period)
        if first_sound_sample >= 0:
            time = max(first_sound_sample / self.samplerate - margin, 0)
            return self[time:]
        return self

    def stripSilenceRight(self, threshold=-120.0, margin=0.01, window=0.02) -> Sample:
        """
        Remove silence from the right. Returns a new Sample

        Args:
            threshold: dynamic of silence, in dB
            margin: leave at list this amount of time between the first/last sample
                    and the beginning of silence or
            window: the duration of the analysis window, in seconds

        Returns:
            a new Sample with silence removed
        """
        period = int(window * self.samplerate)
        lastsample = npsnd.lastSound(self.samples, threshold, period)
        if lastsample >= 0:
            time = min(lastsample / self.samplerate + margin, self.duration)
            return self[:time]
        return self

    def stripSilence(self, threshold=-120.0, margin=0.01, window=0.02) -> Sample:
        """
        Remove silence from the sides. Returns a new Sample

        Args:
            threshold: dynamic of silence, in dB
            margin: leave at list this amount of time between the first/last sample
                    and the beginning of silence or
            window: the duration of the analysis window, in seconds

        Returns:
            a new Sample with silence at the sides removed
        """
        out = self.stripSilenceLeft(threshold, margin, window)
        out = out.stripSilenceRight(threshold, margin, window)
        return out

    def resample(self, samplerate: int) -> Sample:
        """
        Return a new Sample with the given samplerate
        """
        if samplerate == self.samplerate:
            return self
        samples = resample(self.samples, self.samplerate, samplerate)
        return Sample(samples, samplerate=samplerate)

    def scrub(self, bpf: _bpf.BpfInterface) -> Sample:
        """
        Args:
            bpf: a bpf mapping time -> time

        Example::

            >>> # Read sample at half speed
            >>> sample = Sample("path.wav")
            >>> dur = sample.duration
            >>> sample2 = sample.scrub(bpf.linear([(0, 0), (dur*2, dur)]))

        """
        samples, sr = sndfiletools.scrub((self.samples, self.samplerate), bpf,
                                         rewind=False)
        return Sample(samples, self.samplerate)

    def getChannel(self, n:int, contiguous=False) -> Sample:
        """
        return a new mono Sample with the given channel

        Args:
            n: the channel index (starting with 0)
            contiguous: if True, ensure that the samples are represented as
                contiguous in memory
        """
        if self.numchannels == 1 and n == 0:
            return self
        if n > (self.numchannels-1):
            raise ValueError("this sample has only %d channel(s)!"%
                             self.numchannels)
        newsamples = self.samples[:, n]
        if contiguous and not newsamples.flags.c_contiguous:
            newsamples = np.ascontiguousarray(newsamples)
        return Sample(newsamples, self.samplerate)

    def contiguous(self) -> Sample:
        """
        Return a Sample ensuring that the samples are contiguous in memory
        If self is already contiguous, self is returned
        """
        if self.samples.flags.c_contiguous:
            return self
        return Sample(np.ascontiguousarray(self.samples), self.samplerate)

    def estimateFreq(self, start=0.2, dur=0.15, strategy='autocorr') -> float:
        """
        estimate the frequency of the sample (in Hz)

        Args:
            start: where to start the analysis (the beginning
                   of a sample is often not very clear)
            dur: duration of the fragment to analyze
            strategy: one of 'autocorr' or 'fft'

        Returns:
            the estimated fundamental freq.
        """
        t0 = start
        t1 = min(self.duration, t0 + dur)
        s = self.getChannel(0)[t0:t1]
        from .freqestimate import freq_from_autocorr, freq_from_fft
        func = {
            'autocorr': freq_from_autocorr,
            'fft': freq_from_fft
        }.get(strategy, freq_from_autocorr)
        return func(s.samples, s.samplerate)

    def fundamentalBpf(self, fftsize=2048, overlap=4, method="pyin"
                       ) -> _bpf.BpfInterface:
        """
        Construct a bpf which follows the fundamental of this sample in time

        .. note::
            sonicannotator: https://code.soundsoftware.ac.uk/projects/sonic-annotator/files
            pyin plugin: https://code.soundsoftware.ac.uk/projects/pyin/files
        Args:
            fftsize: the size of the fft, in samples
            overlap: determines the hop size
            method: one of 'pyin', 'autocorrelation'. To be able to use 'pyin'
                sonnicannotator with the pyin plugin needs to be installed. 'pyin'
                is the recommended method at the moment

        Returns:
            a bpf representing the fundamental freq. of this sample

        """
        stepsize = int(fftsize//overlap)
        if method == "pyin-annotator":
            from maelzel.ext import sonicannotator
            tmpwav = tempfile.mktemp(suffix=".wav")
            self.write(tmpwav)
            bpf = sonicannotator.pyin_smooth_pitch(tmpwav, fftsize=fftsize,
                                                   stepsize=stepsize, threshdistr=1.5)
            os.remove(tmpwav)
            return bpf
        elif method == "pyin":
            from maelzel.snd import vamptools
            dt, freqs = vamptools.pyin_smoothpitch(self.samples, self.samplerate,
                                                   fft_size=fftsize,
                                                   step_size=fftsize//overlap)
            return _bpf.core.Sampled(freqs, dt)

        elif method == "autocorrelation":
            from maelzel.snd import freqestimate
            steptime = stepsize/self.samplerate
            bpf = freqestimate.frequency_bpf(self.samples, sr=self.samplerate,
                                             steptime=steptime, method='autocorrelation')
            return bpf
        else:
            raise ValueError(f"method should be one of 'pyin', 'autocorrelation', "
                             f"but got {method}")

    def chordAt(self, t: float, resolution=30, method='sndtrck', **kws):
        if method == 'sndtrck':
            try:
                import sndtrck
            except ImportError:
                raise ImportError("sndtrck is needed "
                                  "(https://github.com/gesellkammer/sndtrck)")
            margin = 0.15
            t0 = max(0., t - margin)
            t1 = min(self.duration, t + margin)
            s = sndtrck.analyze_samples(self[t0:t1].samples,
                                        self.samplerate,
                                        resolution=resolution,
                                        hop=2)
            chord = s.chordAt(t-t0)
            return chord
        else:
            raise ValueError(f"method {method} not supported. Possible "
                             f"methods: 'sndtrck'")

    def chunks(self, chunksize:int, hop:int=None, pad=False) -> Iter[np.ndarray]:
        """
        Iterate over the samples in chunks of chunksize. If pad is True,
        the last chunk will be zeropadded, if necessary

        Args:
            chunksize: the size of each chunk
            hop: the number of samples to skip
            pad: if True, pad the last chunk with 0 to fill chunksize

        Returns:
            an iterator over the chunks
        """
        return numpytools.chunks(self.samples,
                                 chunksize=chunksize,
                                 hop=hop,
                                 padwith=(0 if pad else None))

    def firstSound(self, threshold=-120.0, period=0.04, overlap=2, start=0.
                   ) -> Opt[float]:
        """
        Find the time of the first sound within this sample

        Args:
            threshold: the sound threshold in dB.
            period: the time period to calculate the rms
            overlap: determines the step size between rms calculations
            start: start time (0=start of sample)

        Returns:
            the time of the first sound, or None if no sound found

        """
        idx = npsnd.firstSound(self.samples,
                               threshold=threshold,
                               periodsamps=int(period * self.samplerate),
                               overlap=overlap,
                               skip=int(start*self.samplerate))
        return idx / self.samplerate if idx >= 0 else None

    def firstSilence(self, threshold=-80, period=0.04, overlap=2,
                     soundthreshold=-50, start=0.) -> Opt[float]:
        """
        Find the first silence in this sample

        Args:
            threshold: rms value which counts as silence, in dB
            period: the time period to calculate the rms
            overlap: determines the step size between rms calculations
            soundthreshold: rms value which counts as sound, in dB
            start: start time (0=start of sample)

        Returns:
            the time of the first silence, or None if no silence found

        """
        idx = npsnd.firstSilence(samples=self.samples,
                                 threshold=threshold,
                                 period=int(period*self.samplerate),
                                 overlap=overlap,
                                 soundthreshold=soundthreshold,
                                 startidx=int(start*self.samplerate))
        return idx/self.samplerate if idx is not None else None


def broadcastSamplerate(samples: List[Sample]) -> List[Sample]:
    """
    Match the samplerates audio samples to the highest one.
    The audio sample with the lowest samplerate is resampled to the
    higher one.
    
    """
    assert all(isinstance(s, Sample) for s in samples)
    sr = max(s.samplerate for s in samples)
    return [s.resample(sr) for s in samples]


def _as_numpy_samples(samples: U[Sample, np.ndarray]) -> np.ndarray:
    if isinstance(samples, Sample):
        return samples.samples
    elif isinstance(samples, np.ndarray):
        return samples
    else:
        return np.asarray(samples, dtype=float)


def asSample(source: U[str, Sample, Tuple[np.ndarray, int]]) -> Sample:
    """
    Return a Sample instance

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
    Returns a mono version of samples - mixes down the samples if necessary
    """
    samples = _as_numpy_samples(samples)
    channels = npsnd.numChannels(samples)
    if channels == 1:
        return samples
    return np.sum(samples, axis=1) / channels


def concat(sampleseq: Seq[Sample]) -> Sample:
    """
    Concatenate a sequence of Samples

    Samples should share samplingrate and numchannels

    Args:
        sampleseq: a seq. of Samples

    Returns:
        the concatenated samples as one Sample
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
    possible keys::


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
    sndfile_metadata_set = shutil.which("sndfile-metadata-set")
    if sndfile_metadata_set is None:
        raise RuntimeError("Could not find sndfile-metadata-set. Metadata can'be be set "
                           "via this method")

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
    args = [sndfile_metadata_set]
    for key, value in metadata.items():
        key2 = possible_keys.get(key)
        if key2 is not None:
            args.append('--' + key2)
            args.append(str(value))
        else:
            logger.error(f"Unknown key: {key}, skipping")
    args.append(path)
    subprocess.call(args)


_encodingsByFormat = {
    'wav': {
        16: "pcm16",
        24: "pcm24",
        32: "float32",
        0: "float32"
    },
    'aif': {
        16: "pcm16",
        24: "pcm24",
        32: "float32",
        0: "float32"
    },
    'flac': {
        16: "pcm16",
        24: "pcm24",
        0: "pcm24"
    }
}


def PySndFileToWrite(filename:str, channels=1, samplerate=48000,
                     bits:int=None) -> pysndfile.PySndfile:
    """
    Open a PySndfile in write modus

    The format is inferred from the extension (wav, aiff, flac, etc.)
    If bits is given, it is used. Otherwise it is inferred from the format
    """

    base, ext = os.path.splitext(filename)
    ext = ext[1:].lower()
    if not ext or ext not in _encodingsByFormat:
        raise ValueError(f"The extension ({ext}) is not supported")

    encoding = _encodingsByFormat[ext].get(bits or 0)
    if encoding is None:
        raise ValueError(f"no format possible for {ext} with {bits} bits")
    fmt = pysndfile.construct_format(ext, encoding)
    return pysndfile.PySndfile(filename,
                               'w',
                               format=fmt,
                               channels=channels,
                               samplerate=samplerate)


def mix(samples: List[Sample], offsets:List[float]=None, gains:List[float]=None
        ) -> Sample:
    """
    Mix the given samples down, optionally with a time offset

    All samples should share the same number of channels and samplerate

    Args:
        samples: the Samples to mix
        offsets: if given, an offset in seconds for each sample
        gains: if given, a gain for each sample

    Returns:
        the resulting Sample

    Example::

        >>> a = Sample.read("stereo-2seconds.wav")
        >>> b = Sample.read("stereo-3seconds.wav")
        >>> m = mix([a, b], offsets=[2, 0])
        >>> m.duration
        4.0
    """
    nchannels = samples[0].numchannels
    sr = samples[0].samplerate
    assert all(s.numchannels == nchannels and s.samplerate == sr for s in samples)
    if offsets is None:
        offsets = [0.] * len(samples)
    if gains is None:
        gains = [1.] * len(samples)
    dur = max(s.duration + offset for s, offset in zip(samples, offsets))
    numframes = int(dur * sr)
    if nchannels == 1:
        buf = np.zeros((numframes,), dtype=float)
    else:
        buf = np.zeros((numframes, nchannels), dtype=float)
    for s, offset, gain in zip(samples, offsets, gains):
        startframe = int(offset*sr)
        endframe = startframe+len(s)
        buf[startframe:endframe] += s.samples
        if gain != 1.0:
            buf[startframe:endframe] *= gain
    return Sample(buf, samplerate=sr)
