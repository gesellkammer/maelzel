"""
audiosample
~~~~~~~~~~~

This module is based on the :class:`~maelzel.snd.audiosample.Sample` class,
which contains the audio of a soundfile as a numpy array and it aware of its sr,
original format and encoding, etc. It can also perform simple actions
(fade-in/out, cut, insert, reverse, normalize, etc) on its own audio
destructively or return a new Sample. It implements most math operations
valid for audio data (``+``, ``-``, ``*``, ``/``)

.. note::

    All operations are samplerate-aware: any operation involving
    multiple :class:`Sample` instances will broadcast these to the highest samplerate used


Examples
~~~~~~~~

.. code-block:: python

    # load a Sample, fade it, play and write
    from maelzel.snd.audiosample import *
    s = Sample("snd/Numbers_EnglishFemale.flac")
    s.fade(0.5)
    s.play(speed=0.5, block=True)
    # Plot a 6 second fragment startine at time=1
    s[1:7].plotSpectrogram(fftsize=4096, overlap=8, mindb=-100, maxfreq=8000)

.. image:: assets/audiosample-plot-spectrogram.png

.. code-block:: python

    samples = [Sample("soundA.wav"),
               Sample("soundB.aif"),
               Sample("soundC.flac")]
    a, b, c = broadcastSamplerate(samples)
    # mix them down
    out = a.prependSilence(2) + b + c
    out.write("mixed.wav")

"""
from __future__ import annotations
import numpy as np
import os
import math
import tempfile
import shutil
import bpf4
import sndfileio
import logging
from pathlib import Path
import atexit as _atexit
import configdict

from pitchtools import amp2db
import emlib.numpytools as _nptools
import emlib.misc

from maelzel.snd import _common
from maelzel.snd import numpysnd as _npsnd
from maelzel.snd import sndfiletools as _sndfiletools

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import csoundengine
    from typing import Iterator, Sequence
    import matplotlib.pyplot as plt
    from maelzel.partialtracking import spectrum as _spectrum

__all__ = (
    'Sample',
)


logger = logging.getLogger("maelzel.snd")


_config = {
    'reprhtml_include_audiotag': True,
    'reprhtml_audiotag_maxduration_minutes': 10,
    'reprhtml_audiotag_width': '100%',
    'reprhtml_audiotag_maxwidth': '1200px',
    'reprhtml_audio_format': 'wav',
    'csoundengine': _common.CSOUNDENGINE,
}


config = configdict.ConfigDict(name='maelzel.snd.audiosample',
                               default=_config)


_sessionTempfiles = []


@_atexit.register
def _cleanup():
    for f in _sessionTempfiles:
        if os.path.exists(f):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)


def _normalizePath(path: str) -> str:
    path = os.path.expanduser(path)
    return os.path.abspath(path)


def _openInEditor(soundfile: str, wait=False, app=None) -> None:
    """
    Open soundfile in an external app
    
    Args:
        soundfile: the file to open
        wait: if True, wait until editing is finished
        app: the app to use. If None is given, a default app is used

    """
    soundfile = _normalizePath(soundfile)
    emlib.misc.open_with_app(soundfile, app, wait=wait, min_wait=5)


def readSoundfile(sndfile: str | Path, start: float = 0., end: float = 0.
                  ) -> tuple[np.ndarray, int]:
    """
    Read a soundfile, returns a tuple ``(samples:np.ndarray, sr:int)``

    Args:
        sndfile: The path of the soundfile
        start: The time to start reading. A negative value will seek from the end.
        end: The time to stop reading (0=end of file). A negative value will
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
    sndfilestr = _normalizePath(str(sndfile))
    return sndfileio.sndread(sndfilestr, start=start, end=end)


def _vampPyinAvailable() -> bool:
    try:
        import vamp
    except ImportError:
        return False
    return "pyin:pyin" in vamp.list_plugins()


class Sample:
    """
    A class representing audio data

    Args:
        sound: str, a Path or a numpy array
            either sample data or a path to a soundfile
        sr: only needed if passed an array
        start: the start time (only valid when reading from a soundfile). Can be
            negative, in which case the frame is sought from the end.
        end: the end time (only valid when reading from a soundfile). Can be
            negative, in which case the frame is sought from the end
        readonly: is this Sample readonly?
        engine: the sound engine (`csoundengine.Engine`) used for playback

    """

    _csoundEngine: csoundengine.Engine | None = None

    def __init__(self,
                 sound: str | Path | np.ndarray,
                 sr: int = 0,
                 start=0.,
                 end=0.,
                 readonly=False,
                 engine: csoundengine.Engine | None = None):
        self._csoundTable: tuple[str, int] | None = None
        """Keeps track of any table created in csound for playback"""

        self._reprHtml: str = ''
        """Caches html representation"""

        self._asbpf: bpf4.BpfInterface | None = None
        """Caches bpf representation"""

        self._f0: bpf4.BpfInterface | None = None

        self.path = ''
        """If non-empty, the path from which the audio data was loaded"""

        self.readonly = readonly

        if isinstance(sound, (str, Path)):
            samples, sr = readSoundfile(sound, start=start, end=end)
            self.path = str(sound)
        elif isinstance(sound, np.ndarray):
            assert sr
            samples = sound
        else:
            raise TypeError("sound should be a path or an array of samples")

        self.samples: np.ndarray = samples
        """The actual audio samples as a numpy array. Can be multidimensional"""

        self.sr: int = sr
        """The sr"""

        self.numchannels = 1 if len(self.samples.shape) == 1 else self.samples.shape[1]
        """The number of channels of each frame"""

        self.engine: csoundengine.Engine | None = engine
        """The sound engine used for playback"""

    def __del__(self):
        if not self._csoundTable:
            return
        enginename, tabnum = self._csoundTable
        if enginename == Sample.getEngine().name:
            Sample.getEngine().freeTable(tabnum)

    @property
    def numframes(self) -> int:
        """The number of frames"""
        return len(self.samples)

    @property
    def duration(self) -> float:
        """The duration in seconds"""
        return len(self.samples)/self.sr

    def __repr__(self):
        return (f"Sample(dur={self.duration}, sr={self.sr:d}, "
                f"ch={self.numchannels})")

    @staticmethod
    def getEngine(**kws) -> csoundengine.Engine:
        """
        Returns the csound Engine used for playback, starts the Engine if needed

        If no playback has been performed up to this point, a new Engine
        is created. Keywords are passed directly to :class:`csoundengine.Engine`
        (https://csoundengine.readthedocs.io/en/latest/api/csoundengine.engine.Engine.html#csoundengine.engine.Engine)
        and will only take effect if this function is called before any
        playback has been performed.

        An already existing Engine can be set as the playback engine via
        :meth:`Sample.setEngine`

        See Also
        ~~~~~~~~

        * :meth:`Sample.setEngine`
        """
        if Sample._csoundEngine:
            return Sample._csoundEngine
        else:
            import csoundengine as ce
            name = _config['csoundengine']
            if (engine := ce.getEngine(name)) is not None:
                Sample._csoundEngine = engine
            else:
                Sample._csoundEngine = ce.Engine(name=name, **kws)
            return Sample._csoundEngine

    @staticmethod
    def setEngine(engine: csoundengine.Engine | str):
        """
        Sets an external Engine as the playback engine.

        Normally a csound engine is created ad-hoc to take care
        of all playback operations. For some niche cases where
        interaction is needed between playback and other operations
        performed on an already existing engine, it is possible to
        set an external engine as the playback engine. This will only
        affect Sample objects created after the external playback engine has
        been set.

        Args:
            engine: the Engine to use for playback

        See Also
        ~~~~~~~~

        * :meth:`Sample.getEngine`

        """
        if isinstance(engine, str):
            import csoundengine as ce
            engineinstance = ce.getEngine(engine)
            if engineinstance:
                Sample._csoundEngine = engineinstance
            else:
                active = ce.activeEngines()
                raise KeyError(f"Engine {engine} does not exist. Active engines: {active}")
        else:
            Sample._csoundEngine = engine

    @classmethod
    def createSilent(cls, dur: float, channels: int, sr: int) -> Sample:
        """
        Generate a silent Sample with the given characteristics

        Args:
            dur: the duration of the new Sample
            channels: the number of channels
            sr: the sample rate

        Returns:
            a new Sample with all samples set to 0
        """
        numframes = int(dur * sr)
        return cls(_silentFrames(numframes, channels), sr)

    def _makeCsoundTable(self, engine: csoundengine.Engine) -> int:
        if self._csoundTable:
            usedengine, table = self._csoundTable
            if usedengine == engine.name:
                return table
            else:
                logger.warning(f"Engine changed, was {usedengine}, now {engine.name}")
        tabproxy = engine.session().makeTable(self.samples, sr=self.sr, block=True)
        tabnum = tabproxy.tabnum
        self._csoundTable = (engine.name, tabnum)
        return tabnum

    def preparePlay(self, engine=None):
        """Send audio data to the audio engine (blocking)"""
        if engine is None:
            engine = Sample.getEngine()
        self._makeCsoundTable(engine)
        engine.session().prepareSched('.playSample')

    def play(self,
             loop=False,
             chan: int = 1,
             gain=1.,
             delay=0.,
             pan: float = None,
             speed=1.0,
             skip=0.,
             dur=0,
             engine: csoundengine.Engine = None,
             block=False
             ) -> csoundengine.synth.Synth:
        """
        Play the given sample

        csoundengine is used for playback. If no playback has taken place, a new
        playback engine is created. To create an Engine with specific characteristics
        use :meth:`Sample.getEngine`. To use an already existing Engine, see
        :func:`setPlayEngine` or pass the engine explicitely as an argument

        Args:
            loop: should playback be looped?
            chan: first channel to play to. For stereo samples, output
                is routed to consecutive channels starting with this channel
            gain: a gain modifier
            delay: start delay in seconds
            pan: a value between 0 (left) and 1 (right). Use -1
                for a default value, which is 0 for mono samples and 0.5
                for stereo. For 3 or more channels pan is currently ignored
            speed: the playback speed. A variation in speed will change
                the pitch accordingly.
            skip: start playback at a given point in time
            dur: duration of playback. 0 indicates to play until the end of the sample
            engine: the name of a csoundengine.Engine, or the Engine instance
                itself. If given, playback will be performed using this engine,
                otherwise a default Engine will be used.
            block: if True, block execution until playback is finished

        Returns:
            a :class:`csoundengine.synth.Synth`. This synth can be used to
            control or stop playback.

        See Also
        ~~~~~~~~

        * :meth:`Sample.getEngine`
        * :meth:`Sample.setEngine`

        """
        if not engine:
            engine = Sample.getEngine()

        tabnum = self._makeCsoundTable(engine)
        if pan is None:
            if self.numchannels == 1:
                pan = 0
            else:
                pan = 0.5

        if dur == 0:
            dur = -1
        synth = engine.session().playSample(tabnum, chan=chan, gain=gain, loop=loop,
                                            skip=skip, dur=dur, delay=delay, pan=pan,
                                            speed=speed)
        if block:
            import time
            while synth.playing():
                time.sleep(0.02)
        return synth

    def asbpf(self) -> bpf4.BpfInterface:
        """
        Convert this sample to a bpf4.core.Sampled bpf

        .. seealso:: `bpf <https://bpf4.readthedocs.io>`_
        """
        if self._asbpf not in (None, False):
            return self._asbpf
        self._asbpf = bpf4.core.Sampled(self.samples, 1/self.sr)
        return self._asbpf

    def plot(self, profile='auto') -> list[plt.Axes]:
        """
        plot the sample data

        Args:
            profile: one of 'low', 'medium', 'high'

        Returns:
            a list of pyplot Axes
        """
        from . import plotting
        return plotting.plotWaveform(self.samples, self.sr, profile=profile)

    def _repr_html_(self) -> str:
        return self.reprHtml()

    def reprHtml(self, withHeader=True, withAudiotag: bool = None) -> str:
        """
        Returns an HTML representation of this Sample

        This can be used within a Jupyter notebook to force the
        html display. It is useful inside a block were it would
        not be possible to put this Sample as the last element
        of the cell to force the html representation

        Args:
            withHeader: include a header line with repr text ('Sample(...)')
            withAudiotag: include html for audio playback. If None, this
                defaults to config['reprhtml_include_audiotag']

        Returns:
            the HTML repr as str

        Example
        -------

            >>> from maelzel.snd.audiosample import Sample
            >>> sample = Sample("snd/Numbers_EnglishFemale.flac")
            >>> sample.reprHtml()

        .. image:: ../assets/audiosample-reprhtml.png

        """
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
        img = emlib.img.htmlImgBase64(pngfile)   # , maxwidth='800px')
        if self.duration > 60:
            durstr = emlib.misc.sec2str(self.duration)
        else:
            durstr = f"{self.duration:.3g}"
        if withHeader:
            s = (f"<b>Sample</b>(duration=<code>{durstr}</code>, "
                 f"sr=<code>{self.sr}</code>, "
                 f"numchannels=<code>{self.numchannels}</code>)<br>")
        else:
            s = ''
        s += img
        if withAudiotag is None:
            withAudiotag = config['reprhtml_include_audiotag']
        if withAudiotag and self.duration/60 < config['reprhtml_audiotag_maxduration_minutes']:
            audiotag_width = config['reprhtml_audiotag_width']
            maxwidth = config['reprhtml_audiotag_maxwidth']
            # embed short audiofiles, the longer ones are written to disk and read
            # from there
            if self.duration < 2:
                audioobj = IPython.display.Audio(self.samples, rate=self.sr)
                audiotag = audioobj._repr_html_()
            else:
                os.makedirs('tmp', exist_ok=True)
                outfile = tempfile.mktemp(dir="tmp", suffix='.mp3')
                self.write(outfile, overflow='normalize')
                _sessionTempfiles.append(outfile)
                audioobj = IPython.display.Audio(outfile)
                audiotag = audioobj._repr_html_()
            audiotag = audiotag.replace('audio  controls="controls"',
                                        fr'audio controls style="width: {audiotag_width}; max-width: {maxwidth};"')
            s += "<br>" + audiotag
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
                `bohman`, `blackmanharris`, `nuttall`, `barthann`, `kaiser` (needs beta),
                `gaussian` (needs standard deviation)
            start: if given, plot the spectrograph at this time
            dur: if given, use this fragment of the sample (0=from start to end of
                sample)

        Plots the spectrograph of the entire sample (slice before to use only
        a fraction)
        """
        from . import plotting
        samples = self.samples if self.numchannels == 1 else self.samples[:, 0]
        s0 = 0 if start == 0 else int(start*self.sr)
        s1 = self.numframes if dur == 0 else min(self.numframes,
                                                 int(dur*self.sr)-s0)
        if s0 > 0 or s1 != self.numframes:
            samples = samples[s0:s1]
        plotting.plotPowerSpectrum(samples, self.sr, framesize=framesize,
                                   window=window)

    def plotSpectrogram(self,
                        fftsize=2048,
                        window='hamming',
                        overlap: int = None,
                        mindb=-120,
                        minfreq: int = 40,
                        maxfreq: int = 12000,
                        axes=None) -> plt.Axes:
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
        return plotting.plotSpectrogram(samples, self.sr, window=window, fftsize=fftsize,
                                        overlap=overlap, mindb=mindb, minfreq=minfreq,
                                        maxfreq=maxfreq, axes=axes)

    def plotMelSpectrogram(self, fftsize=2048, overlap=4, winlength: int = None,
                           nmels=128, axes=None, axislabels=False, cmap: str = 'magma'
                           ) -> plt.Axes:
        """
        Plot a mel-scale spectrogram

        Args:
            fftsize: the fftsize in samples
            overlap: the amount of overlap. An overlap of 4 will result in a hop-size of
                winlength samples // overlap
            winlength: the window length in samples. If None, fftsize is used. If given,
                winlength <= fftsize
            nmels: number of mel bins
            axes: if given, plot on these Axes
            axislabels: if True, include labels on the axes
            cmap: a color map byname

        Returns:
            the Axes used
        """
        from . import plotting
        return plotting.plotMelSpectrogram(self.samples, sr=self.sr, fftsize=fftsize,
                                           overlap=overlap, winlength=winlength, axes=axes,
                                           setlabel=axislabels, nmels=nmels, cmap=cmap)

    def openInEditor(self, wait=True, app=None, fmt='wav'
                     ) -> Sample | None:
        """
        Open the sample in an external editor.

        The original is not changed.

        Args:
            wait: if True, the editor is opened in blocking mode,
                the results of the edit are returned as a new Sample
            app: if given, this application is used to open the sample.
                Otherwise, the application configured via the key 'editor'
                is used
            fmt: the format to write the samples to

        Returns:
            if wait is True, returns the sample after closing editor
        """
        assert fmt in {'wav', 'aiff', 'aif', 'flac', 'mp3', 'ogg'}
        sndfile = tempfile.mktemp(suffix="." + fmt)
        self.write(sndfile)
        logger.debug(f"open_in_editor: opening {sndfile}")
        _openInEditor(sndfile, wait=wait, app=app)
        if wait:
            return Sample(sndfile)
        return None

    def write(self, outfile: str, encoding='', overflow='fail',
              fmt='', bitrate=224, **metadata
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
        outfile = _normalizePath(outfile)
        samples = self.samples
        if not fmt:
            fmt = os.path.splitext(outfile)[1][1:].lower()
            assert fmt in {'wav', 'aif', 'aiff', 'flac', 'mp3', 'ogg'}
        if not encoding:
            encoding = sndfileio.util.default_encoding(fmt)
            if not encoding:
                raise ValueError(f"Format {fmt} is not supported")
        if overflow != 'nothing' and encoding.startswith('pcm'):
            import numpyx
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
        Return a copy of this Sample

        .. note::

            if self is readonly, the copied Sample will not be readonly.
        """
        return Sample(self.samples.copy(), self.sr)

    def _changed(self) -> None:
        self._csoundTable = None
        self._reprHtml = ''

    def __add__(self, other: float | Sample) -> Sample:
        if isinstance(other, (int, float)):
            return Sample(self.samples+other, self.sr)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
            if len(self) == len(other):
                return Sample(self.samples+other.samples, self.sr)
            elif len(self) > len(other):
                return Sample(self.samples[:len(other)]+other.samples, self.sr)
            else:
                return Sample(self.samples + other.samples[:len(self)], self.sr)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __iadd__(self, other: float | Sample) -> None:
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

    def __sub__(self, other: float | Sample) -> Sample:
        if isinstance(other, (int, float)):
            return Sample(self.samples-other, self.sr)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
            if len(self) == len(other):
                return Sample(self.samples-other.samples, self.sr)
            elif len(self) > len(other):
                return Sample(self.samples[:len(other)]-other.samples, self.sr)
            else:
                return Sample(self.samples - other.samples[:len(self)], self.sr)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __isub__(self, other: float | Sample) -> None:
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

    def __mul__(self, other: float | Sample) -> Sample:
        if isinstance(other, (int, float)):
            return Sample(self.samples*other, self.sr)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
            if len(self) == len(other):
                return Sample(self.samples*other.samples, self.sr)
            elif len(self) > len(other):
                return Sample(self.samples[:len(other)]*other.samples, self.sr)
            else:
                return Sample(self.samples * other.samples[:len(self)], self.sr)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __imul__(self, other: float | Sample) -> Sample:
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

    def fade(self, fadetime: float | tuple[float, float], shape: str = 'linear'
             ) -> Sample:
        """
        Fade this Sample **inplace**, returns self.

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
        self._checkWrite()
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
        self._changed()
        return self

    def prependSilence(self, dur: float) -> Sample:
        """
        Return a new Sample with silence of given dur at the beginning
        """
        silence = Sample.createSilent(dur, self.numchannels, self.sr)
        return joinsamples([silence, self])

    def appendSilence(self, dur: float) -> Sample:
        """
        Return a new Sample with added silence at the end

        Args:
            dur: the duration of the added silence

        Returns:
            a new Sample

        .. seealso:: :meth:`Sample.prependSilence`, :meth:`Sample.join`, :meth:`Sample.append`

        """
        silence = _silentFrames(numframes=int(self.sr*dur), channels=self.numchannels)
        samples = np.concatenate([self.samples, silence])
        return Sample(samples, sr=self.sr)

    def concat(self, *other: Sample) -> Sample:
        """
        Join (concatenate) this Sample with other(s)

        Args:
            *other: one or more Samples to join together

        Returns:
            the resulting Sample

        .. seealso:: :meth:`Sample.join`
        """
        samples = [self, *other]
        return joinsamples(samples)

    def _checkWrite(self) -> None:
        if self.readonly:
            raise RuntimeError("This Sample is readonly. Create a copy (which will"
                               " be writable) and operate on that copy")

    def normalize(self, headroom=0.) -> Sample:
        """Normalize inplace, returns self

        Args:
            headroom: maximum peak in dB

        Returns:
            self
        """

        self._checkWrite()
        ratio = _npsnd.normalizationRatio(self.samples, headroom)
        self.samples *= ratio
        self._changed()
        return self

    def peak(self) -> float:
        """return the highest sample value in dB"""
        return amp2db(np.abs(self.samples).max())

    def peaksbpf(self, framedur=0.01, overlap=2) -> bpf4.core.Sampled:
        """
        Create a bpf representing the peaks envelope of the source

        Args:
            framedur: the duration of an analysis frame (in seconds)
            overlap: determines the hop time between analysis frames.
                ``hoptime = framedur / overlap``

        A peak is the absolute maximum value of a sample over a window
        of time (the *framedur* in this case). To use another metric
        for tracking amplitude see :meth:`Sample.rmsbpf` which uses
        rms.

        The resolution of the returned bpf will be ``framedur/overlap``

        .. seealso::

            https://bpf4.readthedocs.io/en/latest/

        """
        return _npsnd.peaksbpf(self.samples, self.sr, res=framedur, overlap=overlap)

    def reverse(self) -> Sample:
        """ reverse the sample **in-place**, returns self """
        self._checkWrite()
        self.samples[:] = self.samples[-1::-1]
        self._changed()
        return self

    def rmsbpf(self, dt=0.01, overlap=1) -> bpf4.core.Sampled:
        """
        Return a bpf representing the rms of this sample over time

        .. seealso:: https://bpf4.readthedocs.io/en/latest/
        """
        return _npsnd.rmsbpf(self.samples, self.sr, dt=dt, overlap=overlap)

    def rms(self) -> float:
        """ Returns the rms of the samples

        This method returns the rms for **all** the frames at once. As such
        it is only of use for short samples. The use case is as follows:

            >>> from maelzel.snd.audiosample import Sample
            >>> from pitchtools import amp2db
            >>> s = Sample("/path/to/sample.flac")
            >>> amp2db(s[0.5:0.7].rms())
            -12.05


        .. seealso:: :meth:`Sample.rmsbpf`
        """
        return _npsnd.rms(self.samples)

    def mixdown(self) -> Sample:
        """
        Return a new Sample with this sample downmixed to mono

        Returns self if already mono
        """
        if self.numchannels == 1:
            return self
        return Sample(_npsnd.asmono(self.samples), sr=self.sr)

    def stripLeft(self, threshold=-120.0, margin=0.01, window=0.02) -> Sample:
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
        if first_sound_sample is not None and first_sound_sample >= 0:
            time = max(first_sound_sample/self.sr-margin, 0)
            return self[time:]
        return self

    def stripRight(self, threshold=-120.0, margin=0.01, window=0.02) -> Sample:
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
        if lastsample is not None and lastsample >= 0:
            time = min(lastsample/self.sr+margin, self.duration)
            return self[:time]
        return self

    def strip(self, threshold=-120.0, margin=0.01, window=0.02) -> Sample:
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
        out = self.stripLeft(threshold, margin, window)
        out = out.stripRight(threshold, margin, window)
        return out

    def resample(self, sr: int) -> Sample:
        """
        Return a new Sample with the given sr
        """
        if sr == self.sr:
            return self
        from maelzel.snd.resample import resample
        samples = resample(self.samples, self.sr, sr)
        return Sample(samples, sr=sr)

    def scrub(self, bpf: bpf4.BpfInterface) -> Sample:
        """
        Scrub the samples with the given curve

        Args:
            bpf: a bpf mapping time -> time (see `bpf <https://bpf4.readthedocs.io>`)


        Example::

            Read sample at half speed
            >>> import bpf4
            >>> sample = Sample("path.wav")
            >>> dur = sample.duration
            >>> sample2 = sample.scrub(bpf4.linear([(0, 0), (dur*2, dur)]))

        """
        samples, sr = _sndfiletools.scrub((self.samples, self.sr), bpf,
                                          rewind=False)
        return Sample(samples, self.sr)

    def getChannel(self, n: int, contiguous=False) -> Sample:
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
            raise ValueError(f"this sample has only {self.numchannels} channel(s)!")
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

    def onsets(self, fftsize=2048, overlap=4, method: str = 'rosita',
               threshold: float = None, mingap=0.03) -> np.ndarray:
        """
        Detect onsets

        Depending on the implementation, onsets can be "possitive"
        onsets, similar to an attack, or just sudden changes in the spectrum; this
        includes "negative" onsets, which would be detected at the sudden end
        of a note. To accurately track onsets it might be useful to use other
        features, like peak amplitude, rms, or voicedness to check the kind
        of onset.

        For an in-depth demonstration of these concepts see
        https://github.com/gesellkammer/maelzel/blob/master/notebooks/onsets.ipynb

        Args:
            fftsize: the size of the window
            overlap: a hop size as a fraction of the fftsize
            method: one of 'rosita' (using a lightweight version of librosa's onset
                detection algorithm) or 'aubio' (needs aubio to be installed)
            threshold: the onset sensitivity. This is a value specific for a given
                method (rosita has a default of 0.07, while aubio has a default of 0.03)
            mingap: the min. time between two onsets

        Returns:
            a list of onset times, as a numpy array

        Example
        ~~~~~~~

        .. code-block:: python

            from maelzel.snd import audiosample
            from maelzel.core import *
            from pitchtools import *

            samp = audiosample.Sample("snd/finneganswake-fragm01.flac").getChannel(0, contiguous=True)[0:10]
            onsets = samp.onsets(threshold=0.1, mingap=0.05)
            ax = samp.plotSpectrogram()
            # Plot each onset as a vertical line
            ax.vlines(onsets, ymin=0, ymax=10000, color='white', alpha=0.4, linewidth=2)

        .. image:: ../assets/audiosample-onsets.png


        See Also
        ~~~~~~~~

        * maelzel.snd.features.onsetsAubio
        * maelzel.snd.features.onsets

        """
        if method == 'rosita':
            if threshold is None:
                threshold = 0.07
            from maelzel.snd import features
            onsets, onsetstrength = features.onsets(self.samples, sr=self.sr,
                                                    winsize=fftsize,
                                                    hopsize=fftsize // overlap,
                                                    threshold=threshold,
                                                    mingap=mingap)
            return onsets
        elif method == 'aubio':
            if threshold is None:
                threshold = 0.03
            from maelzel.snd import features
            onsets = features.onsetsAubio(samples=self.samples, sr=self.sr, winsize=fftsize,
                                          hopsize=fftsize//overlap, threshold=threshold,
                                          mingap=mingap)
            return np.asarray(onsets, dtype=float)
        else:
            raise ValueError(f"method {method} not known. Possible methods: 'rosita', 'aubio'")

    def partialTrackingAnalysis(self,
                                resolution: float = 50.,
                                channel=0,
                                windowsize: float | None = None,
                                freqdrift: float = None,
                                hoptime: float = None,
                                mindb=-90) -> _spectrum.Spectrum:
        """
        Analyze this audiosample using partial tracking

        Args:
            resolution: the resolution of the analysis, in Hz
            channel: which channel to analyze
            windowsize: The window size in hz. This value needs to be higher than the
                resolution since the window in samples needs to be smaller than the fft analysis
            mindb: the amplitude floor.
            hoptime: the time to move the window after each analysis. For overlap==1, this is 1/windowsize.
                For overlap==2, 1/(windowsize*2)
            freqdrift: the max. variation in frequency between two breakpoints (by default, 1/2 resolution)

        Returns:
            a :class:`maelzel.partialtracking.spectrum.Spectrum`

        .. seealso::

            :meth:`~Sample.spectrumAt`, :meth:`maelzel.partialtracking.spectrum.Spectrum.analyze`


        """
        from maelzel.partialtracking.spectrum import Spectrum
        samples = self.getChannel(channel).samples
        return Spectrum.analyze(samples=samples, sr=self.sr,
                                resolution=resolution, windowsize=windowsize,
                                hoptime=hoptime, freqdrift=freqdrift,
                                mindb=mindb)

    def spectrumAt(self,
                   time: float,
                   resolution: float = 50.,
                   channel=0,
                   windowsize: float = -1,
                   mindb=-90,
                   minfreq: int | None = None,
                   maxfreq=12000,
                   maxcount=0
                   ) -> list[tuple[float, float]]:
        """
        Analyze sinusoidal components of this Sample at the given time

        Args:
            time: the time to analyze
            resolution: the resolution of the analysis, in hz
            channel: if this sample has multiple channels, which channel to analyze
            windowsize: the window size in hz
            mindb: the min. amplitude in dB for a component to be included
            minfreq: the min. frequency of a component to be included
            maxfreq: the max. frequency of a component to be included
            maxcount: the max. number of components to include (0 to include all)

        Returns:
            a list of pairs (frequency, amplitude) where each pair represents a sinusoidal
            component of this sample at the given time. Amplitudes are in the range 0-1
        """
        return spectrumAt(self.samples, sr=self.sr, time=time, resolution=resolution,
                          channel=channel, windowsize=windowsize, mindb=mindb,
                          minfreq=minfreq, maxfreq=maxfreq, maxcount=maxcount)

    def fundamentalFreq(self, time: float = None, dur=0.2, fftsize=2048, overlap=4,
                        fallbackfreq=0
                        ) -> float | None:
        """
        Calculate the fundamental freq. at a given time

        The returned frequency is averaged over the given duration period
        At the moment the smooth pyin method is used

        Args:
            time: the time to start sampling the fundamental frequency. If None is given,
                the first actual sound within this Sample is used
            dur: the duration of the estimation period. The returned frequency will be the
                average frequency over this period of time.
            fftsize: the fftsize used
            fallbackfreq: frequency to use when no fundamental frequency was detected
            overlap: amount of overlaps per fftsize, determines the hop time

        Returns:
            the average frequency within the given period of time, or None if no fundamental
            was found

        """
        if time is None:
            time, freq = self.firstPitch()
            return freq if freq else None

        from maelzel.snd import vamptools
        import scipy.stats
        samples = self.samples
        if len(samples.shape) > 1:
            samples = samples[:, 0]
        startsamp = int(time * self.sr)
        endsamp = min(int((time+dur)*self.sr), len(samples))
        samples = samples[startsamp:endsamp]
        dt, freqs = vamptools.pyinSmoothPitch(samples, self.sr, fftSize=fftsize,
                                              stepSize=fftsize//overlap)
        freqs = freqs[~np.isnan(freqs)]
        if len(freqs) == 0:
            avgfreq = fallbackfreq
        else:
            minfreq = self.sr / fftsize * 2
            avgfreq = float(scipy.stats.trim_mean(freqs[freqs > minfreq], proportiontocut=0.1))
        assert not math.isnan(avgfreq)
        return avgfreq

    def fundamentalBpf(self, fftsize=2048, overlap=4, method: str = None
                       ) -> bpf4.BpfInterface:
        """
        Construct a bpf which follows the fundamental of this sample in time

        .. note::

            The method 'pyin-vamp' depends on the python module 'vamphost' and the
            pyin vamp plugin being installed

            - vamp host: original code: https://code.soundsoftware.ac.uk/projects/vampy-host
              (install via ``pip install vamphost``)
            - pyin plugin can be downloaded from https://code.soundsoftware.ac.uk/projects/pyin/files.
              More information about installing VAMP plugins: https://www.vamp-plugins.org/download.html#install

        Args:
            fftsize: the size of the fft, in samples
            overlap: determines the hop size
            method: one of 'pyin-native', 'pyin-vamp', 'fft'.
                To be able to use 'pyin-vamp', the 'vamphost' package and the 'pyin' plugin
                must be installed. 'pyin-vamp' is the recommended method at the moment
                Use None to autodetect a method based on the installed software

        Returns:
            a `bpf <https://bpf4.readthedocs.io>`_ representing the fundamental freq. of this sample
        """
        stepsize = int(fftsize//overlap)
        if method is None:
            # auto detect
            if _vampPyinAvailable():
                method = 'pyin-vamp'
            else:
                method = 'pyin-native'

        if method == "pyin" or method == "pyin-vamp":
            from maelzel.snd import vamptools
            samples = self.getChannel(0).samples
            dt, freqs = vamptools.pyinSmoothPitch(samples, self.sr,
                                                  fftSize=fftsize,
                                                  stepSize=fftsize // overlap)
            return bpf4.core.Sampled(freqs, dt)
        elif method == 'pyin-native':
            from maelzel.snd import freqestimate
            samples = self.getChannel(0).samples
            hoplength = fftsize // overlap
            f0curve, probcurve = freqestimate.f0curvePyin(samples, sr=self.sr,
                                                          framelength=fftsize,
                                                          hoplength=hoplength)
            return f0curve

        elif method == 'fft':
            from maelzel.snd import freqestimate
            samples = self.getChannel(0).samples
            f0curve, probcurve = freqestimate.f0curve(samples, sr=self.sr,
                                                      overlap=overlap, method='fft')
            return f0curve
        else:
            raise ValueError(f"method should be one of 'pyin', 'pyin-annotator', "
                             f"'fft'"
                             f"but got {method}")

    def chunks(self, chunksize: int, hop: int = None, pad=False) -> Iterator[np.ndarray]:
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

    def firstPitch(self, threshold=-120, minfreq=60, overlap=4, channel=0
                   ) -> tuple[float, float]:
        """
        Returns the first (monophonic) pitch found

        Args:
            threshold: the silence threhsold
            minfreq: the min. frequency to considere valid
            overlap: pitch analysis overlap
            channel: for multichannel audio, which channel to use

        Returns:
            a tuple (time, freq) of the first pitched sound found.
            If no pitched sound found, returns (0, 0)

        """
        samples = self.samples if self.numchannels == 1 else self.samples[:,channel]
        firstidx = _npsnd.firstSound(samples, threshold=threshold)
        lastidx = _npsnd.lastSound(samples, threshold=threshold)
        if firstidx is None or lastidx is None:
            return (0, 0)

        from maelzel.snd import freqestimate
        chunksize = self.sr // 4
        for idx in range(firstidx, lastidx, chunksize):
            fragm = samples[idx:idx+chunksize]
            f0, prob = freqestimate.f0curve(fragm, sr=self.sr, minfreq=minfreq,
                                            overlap=overlap, unvoicedFreqs='negative')
            times, freqs = f0.points()
            mask = freqs > minfreq
            if not mask.any():
                continue
            selfreqs = freqs[mask]
            seltimes = times[mask]
            idx = min(len(selfreqs), 3)
            return seltimes[idx] + idx / self.sr, selfreqs[idx]
        return 0, 0

    def firstSound(self, threshold=-120.0, period=0.04, overlap=2, start=0.,
                   ) -> float | None:
        """
        Find the time of the first sound within this sample

        This does not make any difference between background noise or pitched/voiced sound

        Args:
            threshold: the sound threshold in dB.
            period: the time period to calculate the rms
            overlap: determines the step size between rms calculations
            start: start time (0=start of sample)

        Returns:
            the time of the first sound, or None if no sound found

        .. seealso:: :meth:`Sample.firstPitch`

        """
        idx = _npsnd.firstSound(self.samples,
                                threshold=threshold,
                                periodsamps=int(period * self.sr),
                                overlap=overlap,
                                skip=int(start * self.sr))
        if idx is None:
            return None
        return idx / self.sr if idx >= 0 else None

    def firstSilence(self, threshold=-80, period=0.04, overlap=2,
                     soundthreshold=-50, start=0.) -> float | None:
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

    def addChannels(self, channels: np.ndarray | int) -> Sample:
        """
        Create a new Sample with added channels

        Args:
            channels: the audiodata of the new channels or the number
                of empty channels to add (as integer). In the case of
                passing audio data this new samples should have  the
                exact same number of frames as self

        Returns:
            a new Sample with the added channels. The returned Sample
            will have the same duration as self

        """
        if isinstance(channels, int):
            channels = _silentFrames(self.numframes, channels)
        else:
            assert len(channels) == len(self)
        frames = np.column_stack((self.samples, channels))
        return Sample(frames, sr=self.sr)

    @staticmethod
    def mix(samples: list[Sample], offsets: list[float] = None, gains: list[float] = None
            ) -> Sample:
        """
        Mix the given samples down, optionally with a time offset

        This is a static method. All samples should share the same
        number of channels and sr

        Args:
            samples: the Samples to mix
            offsets: if given, an offset in seconds for each sample
            gains: if given, a gain for each sample

        Returns:
            the resulting Sample

        Example::

            >>> from maelzel.snd.audiosample import Sample
            >>> a = Sample("stereo-2seconds.wav")
            >>> b = Sample("stereo-3seconds.wav")
            >>> m = Sample.mix([a, b], offsets=[2, 0])
            >>> m.duration
            4.0
        """
        return mixsamples(samples, offsets=offsets, gains=gains)

    @staticmethod
    def join(samples: Sequence[Sample]) -> Sample:
        """
        Concatenate a sequence of Samples

        Samples should share numchannels. If mismatching samplerates are found,
        all samples are upsampled to the highest sr

        Args:
            samples: a seq. of Samples

        Returns:
            the concatenated samples as one Sample
        """
        return joinsamples(samples)


def broadcastSamplerate(samples: list[Sample]) -> list[Sample]:
    """
    Match the samplerates audio samples to the highest one.

    The audio sample with the lowest sr is resampled to the
    higher one.
    
    """
    assert all(isinstance(s, Sample) for s in samples)
    sr = max(s.sr for s in samples)
    return [s.resample(sr) for s in samples]


def _asNumpySamples(samples: Sample | np.ndarray) -> np.ndarray:
    if isinstance(samples, Sample):
        return samples.samples
    elif isinstance(samples, np.ndarray):
        return samples
    else:
        return np.asarray(samples, dtype=float)


def asSample(source: str | Sample | tuple[np.ndarray, int]) -> Sample:
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


def joinsamples(sampleseq: Sequence[Sample]) -> Sample:
    """
    Concatenate a sequence of Samples

    Samples should share numchannels. If mismatching samplerates are found,
    all samples are upsampled to the highest sr

    Args:
        sampleseq: a seq. of Samples

    Returns:
        the concatenated samples as one Sample
    """
    numchannels = sampleseq[0].numchannels
    if any(s.numchannels != numchannels for s in sampleseq):
        s = next(s for s in sampleseq if s.numchannels != numchannels)
        raise ValueError(f"All samples should have {numchannels} channels, "
                         f"but one Sample has {s.numchannels} channels")
    sr = max(s.sr for s in sampleseq)
    if any(s.sr != sr for s in sampleseq):
        logger.info(f"concat: Mismatching samplerates. Samples will be upsampled to {sr}")
        sampleseq = [s if s.sr == sr else s.resample(sr) for s in sampleseq]
    s = np.concatenate([s.samples for s in sampleseq])
    return Sample(s, sr)


def _mapn_between(func, n: int, t0: float, t1: float) -> np.ndarray:
    """
    Returns a numpy array of n-size, mapping func between t0-t1 at a rate of n/(t1-t0)

    Args:
        func: a callable of the form func(float) -> float, can be a bpf
            (see https://bpf4.readthedocs.io)
    """
    if hasattr(func, 'mapn_between'):
        ys = func.mapn_between(n, t0, t1)  # is it a Bpf?
    else:
        X = np.linspace(t0, t1, n)
        ufunc = np.vectorize(func)
        Y = ufunc(X)
        return Y
    return ys


def _silentFrames(numframes: int, channels: int) -> np.ndarray:
    """
    Generate silent frames

    Args:
        numframes: the number of frames
        channels: the number of channels

    Returns:
        a new numpy array with zeroed frames
    """
    if channels == 1:
        samples = np.zeros((numframes,), dtype=float)
    else:
        samples = np.zeros((numframes, channels), dtype=float)
    return samples


def mixsamples(samples: list[Sample], offsets: list[float] = None, gains: list[float] = None
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

        >>> from maelzel.snd.audiosample import Sample
        >>> a = Sample("stereo-2seconds.wav")
        >>> b = Sample("stereo-3seconds.wav")
        >>> m = Sample.mix([a, b], offsets=[2, 0])
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
        startframe = int(offset * sr)
        endframe = startframe + len(s)
        buf[startframe:endframe] += s.samples
        if gain != 1.0:
            buf[startframe:endframe] *= gain
    return Sample(buf, sr=sr)


def spectrumAt(samples: np.ndarray,
               sr: int,
               time: float,
               resolution: float,
               channel=0,
               windowsize: float = -1,
               mindb=-90,
               minfreq: int | None = None,
               maxfreq=12000,
               maxcount=0
               ) -> list[tuple[float, float]]:
    """
    Analyze sinusoidal components of this Sample at the given time

    Args:
        samples: the samples, a 1D numpy array. If it is not contiguous it will
            be made contiguous.
        sr: the sample rate
        time: the time to analyze
        resolution: the resolution of the analysis, in hz
        channel: if this sample has multiple channels, which channel to analyze
        windowsize: the window size in hz
        mindb: the min. amplitude in dB for a component to be included
        minfreq: the min. frequency of a component to be included
        maxfreq: the max. frequency of a component to be included
        maxcount: the max. number of components to include (0 to include all)

    Returns:
        a list of pairs (frequency, amplitude) where each pair represents a sinusoidal
        component of this sample at the given time. Amplitudes are in the range 0-1

    """
    if _npsnd.numChannels(samples) > 1:
        samples = _npsnd.getChannel(samples, channel)
    resolutionperiod = 1 / resolution
    margin = resolutionperiod * 4
    starttime = max(0., time - margin)
    duration = len(samples) / sr
    endtime = min(time + margin, duration)
    startsample = int(starttime * sr)
    endsample = int(endtime * sr)
    samples = samples[startsample:endsample]
    samples = np.ascontiguousarray(samples)
    
    try:
        import loristrck
    except ImportError:
        raise ImportError("loristrck is needed to perform this operation. Install it via "
                          "'pip install loristrck'")
    partials = loristrck.analyze(samples, sr=sr, resolution=resolution, windowsize=windowsize)
    if minfreq is None:
        minfreq = int(resolution * 1.3)
    validpartials, rest = loristrck.util.select(partials, mindur=margin, minamp=mindb,
                                                maxfreq=maxfreq, minfreq=minfreq)
    breakpoints = loristrck.util.partials_at(validpartials, t=margin, maxcount=maxcount)
    pairs = [(float(bp[0]), float(bp[1])) for bp in breakpoints]
    pairs.sort(key=lambda pair: pair[0])
    return pairs
