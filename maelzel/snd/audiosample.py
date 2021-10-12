"""
audiosample
~~~~~~~~~~~

This module is based on the :class:`Sample` class, which contains the
audio of a soundfile as a numpy array and it knows about its sr,
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
    s = Sample("sound.wav")
    s.fade(0.5)
    synth = s.play(speed=0.5)
    while synth.isPlaying():
        time.sleep(0.1)
    print("finished playing!")
    s.write("sound-faded.flac")
    s.plot_spectrogram()


.. code-block:: python
    samples = [
        Sample("soundA.wav"),
        Sample("soundB.aif"),
        Sample("soundC.flac")]
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
import shutil
import bpf4 as _bpf
import sndfileio
import logging
from pathlib import Path
import atexit as _atexit

from pitchtools import amp2db
import numpyx
from emlib import numpytools as _nptools
import emlib.misc

from maelzel.snd import numpysnd as _npsnd
from maelzel.snd import sndfiletools as _sndfiletools

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import csoundengine
    from typing import List, Tuple, Optional as Opt, Union as U, Iterator as Iter, \
        Sequence as Seq

logger = logging.getLogger("maelzel.audiosample")


config = {
    'reprhtml_include_audiotag': True,
    'reprhtml_audiotag_maxduration_minutes': 10,
    'reprhtml_audiotag_width': '100%',
    'reprhtml_audiotag_maxwidth': '1200px',
    'reprhtml_audio_format': 'wav'
}


_sessionTempfiles = []


@_atexit.register
def _cleanup():
    for f in _sessionTempfiles:
        if os.path.exists(f):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)


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
    emlib.misc.open_with_app(soundfile, app, wait=wait)


def readSoundfile(sndfile: str, start:float=0., end:float=0.) -> Tuple[np.ndarray, int]:
    """
    Read a soundfile, returns a tuple ``(samples:np.ndarray, sr:int)``

    Args:
        sndfile (str): The path of the soundfile
        start (float): The time to start reading. A negative value will seek from the end
        end (float): The time to stop reading (0=end of file). A negative value will
            seek from the end

    Returns:
        a tuple (samples:np.ndarray, sr:int)

    Example::

        # Read the first two seconds
        >>> samples, sr = readSoundfile("sound.flac", end=2)

        # Read the last two seconds
        >>> samples, sr = readSoundfile("sound.aif", start=-2)
    """
    if sndfile == "?":
        import emlib.dialogs
        sndfile = emlib.dialogs.selectFile(directory=Path.cwd().as_posix(),
                                           filter=emlib.dialogs.filters['Sound'],
                                           title='Select soundfile')
        if not sndfile:
            raise RuntimeError("No soundfile selected")
    sndfile = _normalize_path(sndfile)
    return sndfileio.sndread(sndfile, start=start, end=end)


_csoundEngine = None


def getPlayEngine(**kws) -> csoundengine.Engine:
    """
    Returns the csound Engine used for playback

    If no playback has been performed up to this point, a new Engine
    is created. Keywords are passed directly to csoundengine.Engine
    and will only take effect if this function is called before any
    playback has been performed.

    An already existing Engine can be set as the playback engine via
    `setPlayEngine`

    See Also
    ~~~~~~~~

    * :func:`setPlayEngine`
    """
    global _csoundEngine
    if _csoundEngine:
        return _csoundEngine
    import csoundengine
    _csoundEngine = csoundengine.Engine()
    return _csoundEngine


def setPlayEngine(engine: csoundengine.Engine) -> None:
    """
    Sets an external Engine as the playback engine.

    Normally a csound engine is created ad-hoc to take care
    of all playback operations. For some niche cases where
    interaction is needed between playback and other operations
    performed on an already existing engine, it is possible to
    set an external engine as the playback engine. This will only
    affect Sample objects created after the external playback engine has
    been set.

    See Also
    ~~~~~~~~

    * :func:`getPlayEngine`

    """
    global _csoundEngine
    _csoundEngine = engine



def _vampPyinAvailable() -> bool:
    try:
        import vamp
    except ImportError:
        return False
    return "pyin:pyin" in vamp.list_plugins()


class Sample:
    """
    A class to hold audio data

    Attributes:
        samples (np.ndarray): the audio data
        samplerate (int): the sr of the audio data
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
        self._csoundTabnum = 0
        self._reprHtml: str = ''
        self._asbpf: Opt[_bpf.BpfInterface] = None

        if isinstance(sound, (str, Path)):
            samples, samplerate = readSoundfile(sound, start=start, end=end)
        elif isinstance(sound, np.ndarray):
            assert samplerate is not None
            samples = sound
        else:
            raise TypeError("sound should be a path or an array of samples")
        self.samples: np.ndarray = samples
        self.sr: int = samplerate
        self.numchannels = _npsnd.numChannels(self.samples)

    def __del__(self):
        if self._csoundTabnum:
            getPlayEngine().freeTable(self._csoundTabnum)

    @property
    def numframes(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return len(self.samples)/self.sr

    def __repr__(self):
        s = "Sample: dur=%f sr=%d ch=%d" % (self.duration, self.sr,
                                            self.numchannels)
        return s

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

    def _makeCsoundTable(self) -> int:
        if self._csoundTabnum:
            return self._csoundTabnum
        engine = getPlayEngine()
        #tabnum = engine.makeEmptyTable(len(self.samples)*self.numchannels,
        #                               numchannels=self.numchannels, sr=self.sr)
        tabnum = engine.makeTable(self.samples, sr=self.sr, block=True)
        #engine.fillTable(tabnum, self.samples.flatten(), block=True)
        self._csoundTabnum= tabnum
        return tabnum

    def play(self, loop=False, chan:int=1, gain=1., delay=0.,
             pan=-1, speed=1.0
             ) -> csoundengine.synth.Synth:
        """
        Play the given sample

        If no playback has taken place, a new playback engine is created
        To create an Engine with specific characteristics use :func:`getPlayEngine`.
        To use an already existing Engine, see :func:`setPlayEngine`

        .. note::

            To play a section of the sample, slice it first, then play the
            resulting sample. To slice a sample, use ``sample[starttime:endtime]``

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

        See Also
        ~~~~~~~~

        * :func:`getPlayEngine`
        * :func:`setPlayEngine`

        """
        engine = getPlayEngine()
        tabnum = self._makeCsoundTable()
        synth = engine.session().playSample(tabnum, chan=chan, gain=gain,
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
            self._asbpf = _bpf.core.Sampled(self.samples, 1/self.sr)
            return self._asbpf

    def plot(self, profile='auto') -> None:
        """
        plot the sample data

        Args:
            profile: one of 'low', 'medium', 'high'
        """
        from . import plotting
        plotting.plotWaveform(self.samples, self.sr, profile=profile)

    def _repr_html_(self) -> str:
        if self._reprHtml:
            return self._reprHtml
        import IPython.display
        import emlib.img
        from . import plotting
        pngfile = tempfile.mktemp(suffix=".png", prefix="plot")
        if self.duration < 20:
            profile = 'highest'
        elif self.duration < 40:
            profile = 'high'
        elif self.duration < 180:
            profile = 'medium'
        else:
            profile = 'low'
        plotting.plotWaveform(self.samples, self.sr, profile=profile,
                              saveas=pngfile)
        img = emlib.img.htmlImgBase64(pngfile) # , maxwidth='800px')
        if self.duration > 60:
            durstr = emlib.misc.sec2str(self.duration)
        else:
            durstr = f"{self.duration:.3g}"
        s = f"<b>Sample</b>(duration=<code>{durstr}</code>, " \
            f"sr=<code>{self.sr}</code>, " \
            f"numchannels=<code>{self.numchannels}</code>)"
        s += "<br>" + img
        if config['reprhtml_include_audiotag'] and \
                self.duration/60 < config['reprhtml_audiotag_maxduration_minutes']:
            audiotag_width = config['reprhtml_audiotag_width']
            maxwidth = config['reprhtml_audiotag_maxwidth']
            # embed short audiofiles, the longer ones are written to disk and read
            # from there
            if self.duration < 60:
                audioobj = IPython.display.Audio(self.samples.T, rate=self.sr)
                audiotag = audioobj._repr_html_()
            else:
                os.makedirs('tmp', exist_ok=True)
                outfile = tempfile.mktemp(dir="tmp", suffix='.mp3')
                self.write(outfile)
                _sessionTempfiles.append(outfile)
                audioobj = IPython.display.Audio(outfile)
                audiotag = audioobj._repr_html_()
            audiotag = audiotag.replace('audio  controls="controls"',
                                        fr'audio controls style="width: {audiotag_width}; max-width: {maxwidth};"')

            #_audiotag = rf"""
            #<br>
            #<audio controls style="width: {audiotag_width}; max-width: {maxwidth};">
            #  <source src="{relname}" type="{mimetype}">
            #  audio tag not supported
            #</audio>
            #"""
            s += "<br>"
            s += audiotag
        self._reprHtml = s
        return s

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
        s0 = 0 if start == 0 else int(start*self.sr)
        s1 = self.numframes if dur == 0 else min(self.numframes,
                                                 int(dur*self.sr)-s0)
        if s0 > 0 or s1 != self.numframes:
            samples = samples[s0:s1]
        plotting.plotPowerSpectrum(samples,
                                   self.sr,
                                   framesize=framesize,
                                   window=window,
                                   )

    def plotSpectrogram(self,
                        fftsize=2048,
                        window='hamming',
                        overlap:int=None,
                        mindb=-120,
                        minfreq:int=40,
                        maxfreq:int=12000,
                        axes=None):
        """
        Plot the spectrogram of this sound using matplotlib

        Args:
            fftsize: the size of the fft.
            window: window type. One of 'hamming', 'hanning', 'blackman', ...
                    (see scipy.signal.get_window)
            mindb: the min. amplitude to plot
            overlap: determines the hop size (hop size in samples = fftsize/overlap).
                None to infer a sensible default from the other parameters
            minfreq: the min. freq to plot
            maxfreq: the highes freq. to plot. If None, a default is estimated
                (check maelzel.snd.plotting.config)
            axes: a matplotlib Axes object. If passed, plotting is done using this
                Axes; otherwise a new Axes object is created and returned

        Returns:
            the matplotlib Axes
        """
        from . import plotting
        if self.numchannels > 1:
            samples = self.samples[:, 0]
        else:
            samples = self.samples
        return plotting.plotSpectrogram(samples,
                                        self.sr,
                                        window=window,
                                        fftsize=fftsize,
                                        overlap=overlap,
                                        mindb=mindb,
                                        minfreq=minfreq,
                                        maxfreq=maxfreq,
                                        axes=axes)

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
        assert fmt in {'wav', 'aiff', 'aif', 'flac', 'mp3', 'ogg'}
        sndfile = tempfile.mktemp(suffix="." + fmt)
        self.write(sndfile)
        logger.debug(f"open_in_editor: opening {sndfile}")
        openInEditor(sndfile, wait=wait, app=app)
        if wait:
            return Sample(sndfile)
        return None

    def write(self, outfile: str, encoding:str=None, overflow='fail',
              fmt:str='', bitrate=224, **metadata
              ) -> None:
        """
        Write the samples to outfile

        Args:
            outfile: the name of the soundfile. The extension determines the 
                file format
            encoding: the encoding to use. One of pcm16, pcm24, pcm32, float32,
                float64 or, in the case of mp3 or ogg, the frame rate as integer
                (160 = 160Kb)
            fmt: if not given, it is inferred from the extension. One of 'wav',
                'aiff', 'flac'.
            overflow: one of 'fail', 'normalize', 'nothing'. This applies only to
                pcm formats (wav, aif, mp3)
            bitrate: bitrate used when writing to mp3
            metadata: XXX
        """
        if outfile == "?":
            import emlib.dialogs
            outfile = emlib.dialogs.saveDialog(filter=emlib.dialogs.filters['Sound'],
                                               title="Save soundfile",
                                               directory=os.getcwd())
            if not outfile:
                logger.warning("No outfile selected, aborting")
                return
        outfile = _normalize_path(outfile)
        samples = self.samples
        if not fmt:
            fmt = os.path.splitext(outfile)[1][1:].lower()
            assert fmt in {'wav', 'aif', 'aiff', 'flac', 'mp3', 'ogg'}
        if not encoding:
            encoding = sndfileio.util.default_encoding(fmt)
        if overflow != 'nothing' and encoding.startswith('pcm'):
            minval, maxval = numpyx.minmax1d(self.getChannel(0).samples)
            if minval < -1 or maxval > 1:
                if overflow == 'fail':
                    raise ValueError("Samples would overflow when written")
                elif overflow == 'normalize':
                    maxpeak = max(maxval, abs(minval))
                    samples = samples / maxpeak
        sndfileio.sndwrite(outfile, samples=samples, sr=self.sr,
                           encoding=encoding, fileformat=fmt,
                           bitrate=bitrate,
                           metadata=metadata)

    def copy(self) -> Sample:
        """
        return a copy of this Sample
        """
        return Sample(self.samples.copy(), self.sr)

    def _changed(self) -> None:
        self._csoundTabnum = 0
        self._reprHtml = ''

    def __add__(self, other: U[float, Sample]) -> Sample:
        if isinstance(other, (int, float)):
            return Sample(self.samples+other, self.sr)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
            if len(self) == len(other):
                return Sample(self.samples+other.samples, self.sr)
            elif len(self) > len(other):
                return Sample(self.samples[:len(other)]+other.samples, self.sr)
            else:
                return Sample(self.samples+ other.samples[:len(self)], self.sr)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __iadd__(self, other: U[float, Sample]) -> None:
        if isinstance(other, (int, float)):
            self.samples += other
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
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
            return Sample(self.samples-other, self.sr)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
            if len(self) == len(other):
                return Sample(self.samples-other.samples, self.sr)
            elif len(self) > len(other):
                return Sample(self.samples[:len(other)]-other.samples, self.sr)
            else:
                return Sample(self.samples- other.samples[:len(self)], self.sr)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __isub__(self, other: U[float, Sample]) -> None:
        if isinstance(other, (int, float)):
            self.samples -= other
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
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
            return Sample(self.samples*other, self.sr)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
            if len(self) == len(other):
                return Sample(self.samples*other.samples, self.sr)
            elif len(self) > len(other):
                return Sample(self.samples[:len(other)]*other.samples, self.sr)
            else:
                return Sample(self.samples* other.samples[:len(self)], self.sr)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __imul__(self, other: U[float, Sample]) -> Sample:
        if isinstance(other, (int, float)):
            self.samples *= other
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
            if len(self) == len(other):
                self.samples *= other.samples
            elif len(self) > len(other):
                self.samples[:len(other)] *= other.samples
            else:
                self.samples *= other.samples[:len(self)]
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")
        self._changed()
        return self

    def __pow__(self, other: float) -> Sample:
        return Sample(self.samples**other, self.sr)

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
            >>> source = Sample("sound.wav")
            >>> view = source[1.5:3.0]

            # To slice at the sample level, access .samples directly
            # NB: this will be a 'view' over the existing samples, any modification
            # will be reflected in the source array. Use .copy to produce an independent
            # sample
            >>> newsample = Sample(source.samples[1024:2048], source.sr)
        """
        if not isinstance(item, slice):
            raise ValueError("Samples only support the form sample[start:end]. "
                             "To access individual samples, use sample.samples[index]")
        start, stop, step = item.start, item.stop, item.step
        if stop is None:
            stop = self.duration
        if start is None:
            start = 0.
        if step is not None:
            raise ValueError("Samples do not support a step for slicing. NB: "
                             "To resample a Sample, use the .resample method")
        stop = min(stop, self.duration)
        start = min(start, self.duration)
        assert 0 <= start <= stop
        frame0 = int(start*self.sr)
        frame1 = int(stop*self.sr)
        return Sample(self.samples[frame0:frame1], self.sr)

    def fade(self, fadetime:U[float, Tuple[float, float]], shape:str='linear'
             ) -> Sample:
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

            >>> sample1= Sample("sound.wav")
            # Fade-in and out
            >>> sample1.fade(0.2)

            >>> sample2 = Sample("another.wav")
            # Create a copy with a fade-out of 200 ms
            >>> sample3 = sample2.copy().fade((0, 0.2))

        """
        if isinstance(fadetime, tuple):
            fadein, fadeout = fadetime
            if fadein:
                _npsnd.arrayFade(self.samples, self.sr, fadetime=fadein,
                                 mode='in', shape=shape)
            if fadeout:
                _npsnd.arrayFade(self.samples, self.sr, fadetime=fadeout,
                                 mode='out', shape=shape)
        else:
            _npsnd.arrayFade(self.samples, self.sr, fadetime=fadetime,
                             mode='inout', shape=shape)
        return self

    def prependSilence(self, dur:float) -> Sample:
        """
        Return a new Sample with silence of given dur at the beginning
        """
        silence = Sample.silent(dur, self.numchannels, self.sr)
        return concat([silence, self])

    def concat(self, *other: Sample) -> Sample:
        """
        Concatenate this Sample with other

        Args:
            *other: one or more Samples to join together

        Returns:
            the resulting Sample
        """
        samples = [self]
        samples.extend(other)
        return concat(samples)

    def normalize(self, headroom=0.) -> Sample:
        """Normalize in place, returns self

        Args:
            headroom: maximum peak in dB

        Returns:
            self
        """
        ratio = _npsnd.normalizationRatio(self.samples, headroom)
        self.samples *= ratio
        return self

    def peak(self) -> float:
        """return the highest sample value (dB)"""
        return amp2db(np.abs(self.samples).max())

    def peaksbpf(self, framedur=0.01, overlap=2) -> _bpf.core.Sampled:
        """
        Return a BPF representing the peaks envelope of the source

        The resolution of the returned bpf will be ``framedur/overlap``

        Args:
            framedur: the duration of an analysis frame
            overlap: determines the hop time between analysis frames.
        """
        return _npsnd.peaksbpf(self.samples, self.sr, res=framedur, overlap=overlap)

    def reverse(self) -> Sample:
        """ reverse the sample in-place, returns self """
        self.samples[:] = self.samples[-1::-1]
        self._changed()
        return self

    def rmsbpf(self, dt=0.01, overlap=1) -> _bpf.core.Sampled:
        """
        Return a bpf representing the rms of this sample over time
        """
        return _npsnd.rmsbpf(self.samples, self.sr, dt=dt, overlap=overlap)

    def rms(self) -> float:
        """ Returns the rms of the samples (see also: :meth:`rmsbpf`) """
        return _npsnd.rms(self.samples)

    def mono(self) -> Sample:
        """
        Return a new Sample with this sample downmixed to mono

        Returns self if already mono
        """
        if self.numchannels == 1:
            return self
        return Sample(_npsnd.asmono(self.samples), samplerate=self.sr)

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
        period = int(window*self.sr)
        first_sound_sample = _npsnd.firstSound(self.samples, threshold, period)
        if first_sound_sample >= 0:
            time = max(first_sound_sample/self.sr-margin, 0)
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
        period = int(window*self.sr)
        lastsample = _npsnd.lastSound(self.samples, threshold, period)
        if lastsample >= 0:
            time = min(lastsample/self.sr+margin, self.duration)
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
        Return a new Sample with the given sr
        """
        if samplerate == self.sr:
            return self
        from maelzel.snd.resample import resample
        samples = resample(self.samples, self.sr, samplerate)
        return Sample(samples, samplerate=samplerate)

    def scrub(self, bpf: _bpf.BpfInterface) -> Sample:
        """
        Scrub the samples with the given curve

        Args:
            bpf: a bpf mapping time -> time

        Example::

            >>> # Read sample at half speed
            >>> sample = Sample("path.wav")
            >>> dur = sample.duration
            >>> sample2 = sample.scrub(bpf.linear([(0, 0), (dur*2, dur)]))

        """
        samples, sr = _sndfiletools.scrub((self.samples, self.sr), bpf,
                                          rewind=False)
        return Sample(samples, self.sr)

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
        return Sample(newsamples, self.sr)

    def contiguous(self) -> Sample:
        """
        Return a Sample ensuring that the samples are contiguous in memory

        If self is already contiguous, self is returned
        """
        if self.samples.flags.c_contiguous:
            return self
        return Sample(np.ascontiguousarray(self.samples), self.sr)

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
        from .freqestimate import f0ViaAutocorrelation, f0ViaFFT
        func = {
            'autocorr': f0ViaAutocorrelation,
            'fft': f0ViaFFT
        }.get(strategy, f0ViaAutocorrelation)
        freq, prob = func(s.samples, s.sr)
        return freq

    def fundamentalBpf(self, fftsize=2048, overlap=4, method:str=None
                       ) -> _bpf.BpfInterface:
        """
        Construct a bpf which follows the fundamental of this sample in time

        .. note::
            The method 'pyin-annotator' depends on both sonicannotator and the
            pyin vamp plugin being installed.
            The method 'pyin' depends on the python module 'vamphost' and the
            pyin vamp plugin being installed

            sonicannotator: https://code.soundsoftware.ac.uk/projects/sonic-annotator/files
            vamp host: original code: https://code.soundsoftware.ac.uk/projects/vampy-host,
                install via 'pip install vamphost'
            pyin plugin: https://code.soundsoftware.ac.uk/projects/pyin/files

        Args:
            fftsize: the size of the fft, in samples
            overlap: determines the hop size
            method: one of 'pyin', 'pyin-annotator', 'fft'.
                To be able to use 'pyin', the 'vamphost' package and the 'pyin' plugin
                must be installed. 'pyin' is the recommended method at the moment
                Use None to autodetect a method based on the installed software
        Returns:
            a bpf representing the fundamental freq. of this sample

        """
        stepsize = int(fftsize//overlap)
        if method is None:
            # auto detect
            if _vampPyinAvailable():
                method = 'pyin'
            elif shutil.which('sonic-annotator') is not None:
                method = 'pyin-annotator'
            else:
                method = 'pyin-librosa'

        if method == "pyin-annotator":
            from maelzel.ext import sonicannotator
            tmpwav = tempfile.mktemp(suffix=".wav")
            s = self.getChannel(0)
            s.write(tmpwav)
            bpf = sonicannotator.pyin_smoothpitch(tmpwav, fftsize=fftsize,
                                                  stepsize=stepsize)
            os.remove(tmpwav)
            return bpf
        elif method == "pyin":
            from maelzel.snd import vamptools
            samples = self.getChannel(0).samples
            dt, freqs = vamptools.pyin_smoothpitch(samples, self.sr,
                                                   fft_size=fftsize,
                                                   step_size=fftsize//overlap)
            return _bpf.core.Sampled(freqs, dt)
        elif method == 'pyin-librosa':
            from maelzel.snd import freqestimate
            samples = self.getChannel(0).samples
            hoplength = fftsize // overlap
            f0curve, probcurve = freqestimate.f0curvePyin(samples, sr=self.sr,
                                                          framelength=fftsize,
                                                          hoplength=hoplength)
            return f0curve

        elif method == 'fft':
            from maelzel.snd import freqestimate
            steptime = stepsize/self.sr
            samples = self.getChannel(0).samples
            f0curve, probcurve = freqestimate.f0curve(samples, sr=self.sr,
                                                      steptime=steptime, method='fft')
            return f0curve
        else:
            raise ValueError(f"method should be one of 'pyin', 'pyin-annotator', "
                             f"'fft'"
                             f"but got {method}")

    def chunks(self, chunksize:int, hop:int=None, pad=False) -> Iter[np.ndarray]:
        """
        Iterate over the samples in chunks of chunksize.

        If pad is True, the last chunk will be zeropadded, if necessary

        Args:
            chunksize: the size of each chunk
            hop: the number of samples to skip
            pad: if True, pad the last chunk with 0 to fill chunksize

        Returns:
            an iterator over the chunks
        """
        return _nptools.chunks(self.samples,
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
        idx = _npsnd.firstSound(self.samples,
                                threshold=threshold,
                                periodsamps=int(period*self.sr),
                                overlap=overlap,
                                skip=int(start*self.sr))
        return idx / self.sr if idx>=0 else None

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
        idx = _npsnd.firstSilence(samples=self.samples,
                                  threshold=threshold,
                                  period=int(period*self.sr),
                                  overlap=overlap,
                                  soundthreshold=soundthreshold,
                                  startidx=int(start*self.sr))
        return idx/self.sr if idx is not None else None


def broadcastSamplerate(samples: List[Sample]) -> List[Sample]:
    """
    Match the samplerates audio samples to the highest one.
    The audio sample with the lowest sr is resampled to the
    higher one.
    
    """
    assert all(isinstance(s, Sample) for s in samples)
    sr = max(s.sr for s in samples)
    return [s.resample(sr) for s in samples]


def _asNumpySamples(samples: U[Sample, np.ndarray]) -> np.ndarray:
    if isinstance(samples, Sample):
        return samples.samples
    elif isinstance(samples, np.ndarray):
        return samples
    else:
        return np.asarray(samples, dtype=float)


def asSample(source: U[str, Sample, Tuple[np.ndarray, int]]) -> Sample:
    """
    Return a Sample instance

    Args:
        source: a filename, a Sample or a tuple (samples, sr)

    Returns:
        a Sample. If already a Sample, it just returns it
    """
    if isinstance(source, Sample):
        return source
    if isinstance(source, str):
        return Sample(source)
    if isinstance(source, tuple) and isinstance(source[0], np.ndarray):
        samples, sr = source
        return Sample(samples, sr)
    else:
        raise TypeError("can't convert source to Sample")


def mono(samples: U[Sample, np.ndarray]) -> np.ndarray:
    """
    Returns a mono version of samples

    Mixes down the samples if necessary
    """
    samples = _asNumpySamples(samples)
    channels = _npsnd.numChannels(samples)
    if channels == 1:
        return samples
    return np.sum(samples, axis=1) / channels


def concat(sampleseq: Seq[Sample]) -> Sample:
    """
    Concatenate a sequence of Samples

    Samples should share numchannels. If mismatching samplerates are found,
    all samples are upsampled to the highest sr

    Args:
        sampleseq: a seq. of Samples

    Returns:
        the concatenated samples as one Sample
    """
    sr = max(s.sr for s in sampleseq)
    if any(s.sr != sr for s in sampleseq):
        logger.info(f"concat: Mismatching samplerates. Samples will be upsampled to {sr}")
        sampleseq = [s if s.sr == sr else s.resample(sr) for s in sampleseq]
    s = np.concatenate([s.samples for s in sampleseq])
    return Sample(s, sr)


def _mapn_between(func, n:int, t0:float, t1:float) -> np.ndarray:
    """
    Returns a numpy array of n-size, mapping func between t0-t1 at a rate of n/(t1-t0)

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


def mix(samples: List[Sample], offsets:List[float]=None, gains:List[float]=None
        ) -> Sample:
    """
    Mix the given samples down, optionally with a time offset

    All samples should share the same number of channels and sr

    Args:
        samples: the Samples to mix
        offsets: if given, an offset in seconds for each sample
        gains: if given, a gain for each sample

    Returns:
        the resulting Sample

    Example::

        >>> a = Sample("stereo-2seconds.wav")
        >>> b = Sample("stereo-3seconds.wav")
        >>> m = mix([a, b], offsets=[2, 0])
        >>> m.duration
        4.0
    """
    nchannels = samples[0].numchannels
    sr = samples[0].sr
    assert all(s.numchannels == nchannels and s.sr == sr for s in samples)
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