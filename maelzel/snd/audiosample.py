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
import abc
import numpy as np
import os
import math
from pathlib import Path

import pitchtools as pt
import emlib.misc

from maelzel import _util
from maelzel.snd import _common
from maelzel.snd import numpysnd as _npsnd
import maelzel.common

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpf4
    import sounddevice
    import csoundengine
    import csoundengine.synth
    from typing import Iterator, Sequence
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from maelzel.partialtracking import spectrum as _spectrum
    from maelzel.transcribe import mono
    from typing_extensions import Self

__all__ = (
    'Sample',
)


config = {
    'reprhtml_include_audiotag': True,
    'reprhtml_audiotag_maxduration_seconds': 600,
    'reprhtml_audiotag_width': '100%',
    'reprhtml_audiotag_maxwidth': '1200px',
    'reprhtml_audiotag_embed_maxduration_seconds': 8,
    'reprhtml_audio_format': 'mp3',
    'csoundengine': _common.CSOUNDENGINE,
}


class PlaybackStream(abc.ABC):
    """
    A class to abstract the playback engine
    """

    @abc.abstractmethod
    def stop(self) -> None:
        """
        Stop this stream
        """
        raise NotImplementedError

    def active(self) -> bool:
        """
        True if playback is active
        """
        raise NotImplementedError

    def _repr_html_(self) -> str:
        from maelzel.core import jupytertools
        jupytertools.displayButton("Stop", self.stop)
        return repr(self)


class _PortaudioPlayback(PlaybackStream):
    """
    Portaudio based playback for audiosamples (based on sounddevice)
    """
    def __init__(self, stream: sounddevice.OutputStream):
        self.stream = stream

    def active(self) -> bool:
        return self.stream.active

    def stop(self):
        self.stream.stop()


class _CsoundenginePlayback(PlaybackStream):
    """
    Csoundengine playback for audiosamples
    """
    def __init__(self, synth: csoundengine.synth.Synth):
        self.synth = synth

    def stop(self):
        self.synth.stop()

    def active(self) -> bool:
        return self.synth.playStatus() != 'playing'


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
    import sndfileio
    return sndfileio.sndread(sndfilestr, start=start, end=end)


def _vampPyinAvailable() -> bool:
    try:
        import vamp
    except ImportError:
        return False
    return "pyin:pyin" in vamp.list_plugins()


def _playSamples(samples: np.ndarray, sr: int, mapping: list[int], gain=1., speed=1., loop=False, block=False
                 ) -> PlaybackStream:
    import sounddevice
    sr = int(sr * speed)
    ctx = sounddevice._CallbackContext(loop=loop)
    ctx.frames = ctx.check_data(data=samples, mapping=mapping, device=None)  # type: ignore

    def callback(outdata, numframes, time, status, gain=gain):
        assert len(outdata) == numframes
        ctx.callback_enter(status, outdata)
        if gain != 1:
            outdata *= gain
        ctx.write_outdata(outdata)
        ctx.callback_exit()

    ctx.start_stream(
        sounddevice.OutputStream,
        samplerate=sr,
        channels=ctx.output_channels,
        dtype=ctx.output_dtype,
        callback=callback,
        blocking=block,
        prime_output_buffers_using_stream_callback=False)
    return _PortaudioPlayback(stream=ctx.stream)



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
        """If non-empty, the audio was loaded from this path and has not changed"""

        self.originalpath = ''
        """The original path from which the sample data was loaded, if applicable"""

        self.readonly = readonly

        if isinstance(sound, (str, Path)):
            samples, sr = readSoundfile(sound, start=start, end=end)
            self.path = str(sound)
            self.originalpath = self.path
        elif isinstance(sound, np.ndarray):
            assert sr
            samples = sound
        else:
            raise TypeError(f"sound should be a path or an array of samples, got {type(sound)}")

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
        import csoundengine as ce
        name = config['csoundengine']
        engine = ce.Engine.activeEngines.get(name) or ce.Engine(name=name, **kws)
        Sample._csoundEngine = engine
        return engine

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
                maelzel.common.getLogger(__file__).warning(f"Engine changed, was {usedengine}, now {engine.name}")
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

    def _playPortaudio(self,
                       loop=False,
                       chan=1,
                       gain=1.,
                       speed=1.,
                       skip=0.,
                       dur=0.,
                       block=False) -> PlaybackStream:
        mapping = list(range(chan, self.numchannels + chan))
        samples = self.samples
        if skip:
            samples = samples[int(self.sr * skip):]

        if dur:
            samples = samples[:int(self.sr*dur)]

        return _playSamples(samples=samples, sr=self.sr, mapping=mapping, loop=loop,
                            block=block, gain=gain, speed=speed)

    def play(self,
             loop=False,
             chan: int = 1,
             gain=1.,
             delay=0.,
             pan: float | None = None,
             speed=1.0,
             skip=0.,
             dur=0,
             engine: csoundengine.Engine | None = None,
             block=False,
             backend=''
             ) -> PlaybackStream:
        """
        Play the given sample

        At the moment two playback backends are available, portaudio and csound.

        If no engine is given and playback is immediate (no delay), playback is
        performed directly via portaudio. This has the advantage that no data
        must be copied to the playback engine (which is the case when using csound)

        If backend is 'csound' or a csoundengine's Engine is passed, csound
        is used as playback backend. The csound backend is recommended if sync
        is needed between this playback and other events.

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
            engine: the Engine instance to use for playback. If not given, playback
                is performed via portaudio
            block: if True, block execution until playback is finished
            backend: one of 'portaudio', 'csound'

        Returns:
            a :class:`PlaybackStream`. This can be used to stop playback

        See Also
        ~~~~~~~~

        * :meth:`Sample.getEngine`
        * :meth:`Sample.setEngine`

        """
        if not backend:
            if engine is None and delay == 0 and speed <= 8:
                backend = 'portaudio'
            else:
                backend = 'csound'

        if backend == 'portaudio':
            maelzel.common.getLogger(__file__).debug("Playback using portaudio (sounddevice)")
            return self._playPortaudio(loop=loop, chan=chan, gain=gain,
                                       speed=speed, skip=skip, dur=dur, block=block)
        elif backend == 'csound':
            # Use csoundengine
            maelzel.common.getLogger(__file__).debug("Playback using csoundengine")
            if not engine:
                engine = Sample.getEngine()

            if self.path:
                source = self.path
            else:
                source = self._makeCsoundTable(engine)

            if pan is None:
                pan = 0 if self.numchannels == 1 else 0.5

            if dur == 0:
                dur = -1
            synth = engine.session().playSample(source, chan=chan, gain=gain, loop=loop,
                                                skip=skip, dur=dur, delay=delay, pan=pan,
                                                speed=speed)
            if block:
                import time
                while synth.playing():
                    time.sleep(0.02)
            return _CsoundenginePlayback(synth=synth)
        else:
            raise ValueError(f"backend should be one of 'csound', 'portaudio', got {backend}")

    def asbpf(self) -> bpf4.BpfInterface:
        """
        Convert this sample to a ``bpf4.core.Sampled`` bpf

        .. seealso:: `bpf <https://bpf4.readthedocs.io>`_
        """
        if self._asbpf not in (None, False):
            return self._asbpf
        import bpf4
        bpf = bpf4.Sampled(self.samples, 1/self.sr)
        self._asbpf = bpf
        return bpf

    def plot(self, profile='auto') -> Figure:
        """
        plot the sample data

        Args:
            profile: one of 'low', 'medium', 'high' or 'auto'

        Returns:
            the Figure used
        """
        from . import plotting
        return plotting.plotWaveform(self.samples, self.sr, profile=profile)

    def _repr_html_(self) -> str:
        return self.reprHtml()

    def show(self, withAudiotag=True, figsize=(24, 4), external=False, profile=''):
        if external:
            raise ValueError("External editor not supported")
        if _util.pythonSessionType() == 'jupyter':
            from IPython.display import display_html
            display_html(self.reprHtml(withAudiotag=withAudiotag, figsize=figsize, profile=profile), raw=True)
        else:
            self.plot()

    def reprHtml(self,
                 withHeader=True,
                 withAudiotag=True,
                 figsize=(24, 4),
                 profile=''
                 ) -> str:
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
        from csoundengine.internal import plotSamplesAsHtml
        if withAudiotag is None:
            withAudiotag = config['reprhtml_include_audiotag']
        if withHeader:
            from emlib.misc import sec2str
            dur = self.duration
            durstr = sec2str(dur) if dur > 60 else f"{dur:.3g}"
            header = (f"<b>Sample</b>(duration=<code>{durstr}</code>, "
                      f"sr=<code>{self.sr}</code>, "
                      f"numchannels=<code>{self.numchannels}</code>)<br>")
        else:
            header = ''
        audiotagMaxDur = config['reprhtml_audiotag_embed_maxduration_seconds']
        embed = self.duration <= audiotagMaxDur
        html = plotSamplesAsHtml(samples=self.samples, sr=self.sr,
                                 customHeader=header,
                                 withAudiotag=withAudiotag,
                                 profile=profile, path=self.path, figsize=figsize,
                                 embedAudiotag=embed,
                                 audiotagMaxDuration=audiotagMaxDur if embed else 9999999)

        self._reprHtml = html
        return html

    def plotSpetrograph(self, framesize=2048, window='hamming', start=0., dur=0.,
                        axes: Axes | None = None
                        ) -> Axes:
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
            axes: the axes to plot to. A new axes will be created if not given

        Returns:
            the used axes

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
        return plotting.plotPowerSpectrum(samples, self.sr, framesize=framesize,
                                          window=window, axes=axes)

    def plotSpectrogram(self,
                        fftsize=2048,
                        window='hamming',
                        winsize: int = 0,
                        overlap=4,
                        mindb=-120,
                        minfreq: int = 40,
                        maxfreq: int = 12000,
                        yaxis='linear',
                        figsize=(24, 10),
                        axes: Axes | None = None
                        ) -> Axes:
        """
        Plot the spectrogram of this sound using matplotlib

        Args:
            fftsize: the size of the fft.
            window: window type. One of 'hamming', 'hanning', 'blackman', ...
                    (see scipy.signal.get_window)
            winsize: window size in samples, defaults to fftsize
            mindb: the min. amplitude to plot
            overlap: determines the hop size (hop size in samples = fftsize/overlap).
                None to infer a sensible default from the other parameters
            minfreq: the min. freq to plot
            maxfreq: the highes freq. to plot. If None, a default is estimated
                (check maelzel.snd.plotting.config)
            yaxis: one of 'linear' or 'log'
            figsize: the figure size, a tuple (width: int, height: int)
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
                                        maxfreq=maxfreq, axes=axes, yaxis=yaxis,
                                        winsize=winsize, figsize=figsize)

    def plotMelSpectrogram(self,
                           fftsize=2048,
                           overlap=4,
                           winsize: int = 0,
                           nmels=128,
                           axes: Axes | None = None,
                           axislabels=False,
                           cmap='magma',
                           ) -> Axes:
        """
        Plot a mel-scale spectrogram

        Args:
            fftsize: the fftsize in samples
            overlap: the amount of overlap. An overlap of 4 will result in a hop-size of
                winlength samples // overlap
            winsize: the window size in samples. If None, fftsize is used. If given,
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
                                           overlap=overlap, winsize=winsize, axes=axes,
                                           setlabel=axislabels, nmels=nmels, cmap=cmap)

    def openInEditor(self, wait=True, app=None, fmt='wav'
                     ) -> Self | None:
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
        import tempfile
        tmpfile = tempfile.NamedTemporaryFile(suffix="."+fmt, delete=False)
        sndfile = tmpfile.name
        self.write(sndfile)
        # sndfile = tempfile.mktemp(suffix="." + fmt)
        _openInEditor(sndfile, wait=wait, app=app)
        if wait:
            return self.__class__(sndfile)
        return None

    def write(self,
              outfile: str,
              encoding='',
              overflow='fail',
              fmt='',
              bitrate=224,
              **metadata
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
                return
        outfile = _normalizePath(outfile)
        samples = self.samples
        if not fmt:
            fmt = os.path.splitext(outfile)[1][1:].lower()
            assert fmt in {'wav', 'aif', 'aiff', 'flac', 'mp3', 'ogg'}

        import sndfileio
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

    def copy(self) -> Self:
        """
        Return a copy of this Sample

        .. note::

            if self is readonly, the copied Sample will not be readonly.
        """
        return self.__class__(self.samples.copy(), self.sr)

    def _changed(self) -> None:
        # clear cached values, invalidate path
        self._csoundTable = None
        self._reprHtml = ''
        self._asbpf = None
        self._f0 = None
        self.path = ''

    def __add__(self, other: float | Sample) -> Self:
        if isinstance(other, (int, float)):
            return self.__class__(self.samples+other, self.sr)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
            if len(self) == len(other):
                return self.__class__(self.samples+other.samples, self.sr)
            elif len(self) > len(other):
                return self.__class__(self.samples[:len(other)]+other.samples, self.sr)
            else:
                return self.__class__(self.samples + other.samples[:len(self)], self.sr)
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

    def __sub__(self, other: float | Self) -> Self:
        if isinstance(other, (int, float)):
            return self.__class__(self.samples-other, self.sr)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
            if len(self) == len(other):
                return self.__class__(self.samples-other.samples, self.sr)
            elif len(self) > len(other):
                return self.__class__(self.samples[:len(other)]-other.samples, self.sr)
            else:
                return self.__class__(self.samples - other.samples[:len(self)], self.sr)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __isub__(self, other: float | Self) -> None:
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

    def __mul__(self, other: float | Self) -> Self:
        if isinstance(other, (int, float)):
            return self.__class__(self.samples*other, self.sr)
        elif isinstance(other, Sample):
            assert self.numchannels == other.numchannels and self.sr == other.sr
            if len(self) == len(other):
                return self.__class__(self.samples*other.samples, self.sr)
            elif len(self) > len(other):
                return self.__class__(self.samples[:len(other)]*other.samples, self.sr)
            else:
                return self.__class__(self.samples * other.samples[:len(self)], self.sr)
        else:
            raise TypeError(f"Expected a scalar or a sample, got {other}")

    def __imul__(self, other: float | Sample) -> Self:
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

    def __pow__(self, other: float) -> Self:
        return self.__class__(self.samples**other, self.sr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: slice) -> Self:
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
        return self.__class__(self.samples[frame0:frame1], self.sr)
    
    def splice(self, start: int = 0, end: int = 0) -> Self:
        """
        Splice this Sample between the given frames
        
        Args:
            start: start frame (in samples) 
            end: end frame (in samples, 0=end)

        Returns:
            a copy of self spliced between start and end frame
        """
        if start == end == 0:
            return self
        if end == 0:
            end = len(self.samples)
        return self.__class__(self.samples[start:end], sr=self.sr, 
                              readonly=self.readonly, engine=self.engine)

    def fade(self, fadetime: float | tuple[float, float], shape='linear'
             ) -> Self:
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
                _npsnd.applyFade(self.samples, self.sr, fadetime=fadein,
                                 mode='in', shape=shape)
            if fadeout:
                _npsnd.applyFade(self.samples, self.sr, fadetime=fadeout,
                                 mode='out', shape=shape)
        else:
            assert isinstance(fadetime, (int, float))
            _npsnd.applyFade(self.samples, self.sr, fadetime=fadetime,
                             mode='inout', shape=shape)
        self._changed()
        return self

    def prependSilence(self, dur: float) -> Self:
        """
        Return a new Sample with silence of given dur at the beginning

        Args:
            dur: duration of the silence to add at the beginning

        Returns:
            new Sample
        """
        silence = _silentFrames(numframes=int(self.sr*dur), channels=self.numchannels)
        samples = np.concatenate([silence, self.samples])
        return self.__class__(samples, sr=self.sr)

    def appendSilence(self, dur: float) -> Self:
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
        return self.__class__(samples, sr=self.sr)

    def concat(self, *other: Self) -> Self:
        """
        Join (concatenate) this Sample with other(s)

        Args:
            *other: one or more Samples to join together

        Returns:
            the resulting Sample

        .. seealso:: :meth:`Sample.join`
        """
        samples = [self, *other]
        samp = concatenate(samples)
        return self.__class__(samp.samples, sr=samp.sr)

    def _checkWrite(self) -> None:
        if self.readonly:
            raise RuntimeError("This Sample is readonly. Create a copy (which will"
                               " be writable) and operate on that copy")

    def panned(self, pan: float) -> Self:
        """Return a new Sample with panning applied

        Args:
            pan: panning value between 0 (left) and 1 (right)

        Returns:
            the new Sample, always a stereo sample
        """
        if self.numchannels > 2:
            raise ValueError(f"Panning can only be applied to mono or stereo samples, "
                             f"this sample has {self.numchannels} channels")
        samples = _npsnd.panStereo(self.samples, pan)
        return self.__class__(samples, sr=self.sr)

    def applyPanning(self, pan: float) -> Self:
        """Apply panning to the sample in place

        .. note:: This method is only available for stereo samples.

        Args:
            pan: panning value between 0 (left) and 1 (right)

        Returns:
            self
        """
        if self.numchannels != 2:
            raise ValueError(f"Panning can only be applied to stereo samples, "
                             f"this sample has {self.numchannels} channels")

        self._checkWrite()
        _npsnd.applyPanning(self.samples, pan)
        self._changed()
        return self

    def normalize(self, headroom=0.) -> Self:
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
        """Highest sample value in dB"""
        return pt.amp2db(np.abs(self.samples).max())

    def peaksBpf(self, framedur=0.01, overlap=2) -> bpf4.Sampled:
        """
        Create a bpf representing the peaks envelope of the source

        Args:
            framedur: the duration of an analysis frame (in seconds)
            overlap: determines the hop time between analysis frames.
                ``hoptime = framedur / overlap``

        Returns:
            A bpf representing the peaks envelope of the source

        A peak is the absolute maximum value of a sample over a window
        of time (the *framedur* in this case). To use another metric
        for tracking amplitude see :meth:`Sample.rmsBpf` which uses
        rms, or :meth:`Sample.amplitudeBpf` which uses an envelope
        follower

        The resolution of the returned bpf will be ``framedur/overlap``

        .. seealso::

            https://bpf4.readthedocs.io/en/latest/

        """
        return _npsnd.peaksBpf(self.samples, self.sr, res=framedur, overlap=overlap)

    def reverse(self) -> Self:
        """ reverse the sample **in-place**, returns self """
        self._checkWrite()
        self.samples[:] = self.samples[-1::-1]
        self._changed()
        return self

    def rmsBpf(self, dt=0.01, overlap=1) -> bpf4.Sampled:
        """
        Creates a BPF representing the rms of this sample over time

        Args:
            dt (float): The duration of each frame in seconds.
            overlap (int): The number of frames to overlap.

        Returns:
            bpf4.Sampled: A BPF representing the rms of this sample over time.

        Raises:
            ValueError: If dt is not positive.
            ValueError: If overlap is not positive.

        .. seealso:: https://bpf4.readthedocs.io/en/latest/
        """
        return _npsnd.rmsBpf(self.samples, self.sr, dt=dt, overlap=overlap)

    def rms(self) -> float:
        """
        RMS of the samples

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

    def amplitudeBpf(self, attack=0.01, release=0.01, chunktime=0.05, overlap=2) -> bpf4.Sampled:
        """
        Creates a bpf representing the average amplitude over time

        Args:
            attack: attack time in seconds for the envelope follower
            release: decay time in seconds for the envelope follower
            chunktime: chunk time in seconds, averages envelope over this time
            overlap: overlap factor for averaging the envelope

        Returns:
            a bpf representing the average amplitude over time
        """
        return _npsnd.ampBpf(self.samples, self.sr, attack=attack, release=release, chunktime=chunktime, overlap=overlap)

    def mixdown(self, enforceCopy=False) -> Sample:
        """
        Return a new Sample with this sample downmixed to mono

        Args:
            enforceCopy: always return a copy, even if self is already mono

        Returns:
            a mono version of self.
        """
        if self.numchannels == 1:
            return self if not enforceCopy else self.copy()
        return Sample(_npsnd.asmono(self.samples), sr=self.sr)

    def stripLeft(self, threshold=-120.0, margin=0.01, window=0.02) -> Self:
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

    def stripRight(self, threshold=-120.0, margin=0.01, window=0.02) -> Self:
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

    def strip(self, threshold=-120.0, margin=0.01, window=0.02) -> Self:
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
        from maelzel.snd import sndfiletools
        samples, sr = sndfiletools.scrub((self.samples, self.sr), bpf,
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

    def fundamentalAnalysis(self,
                            semitoneQuantization=0,
                            fftsize: int = 0,
                            simplify=0.08,
                            overlap=8,
                            minFrequency=50,
                            minSilence=0.08,
                            onsetThreshold=0.05,
                            onsetOverlap=8,
                            ) -> mono.FundamentalAnalysisMonophonic:
        """
        Analyze the fundamental of this sound, assuming it is a monophonic sound

        This is a wrapper around ``maelzel.transcribe.mono.FundamentalAnalysisMono`` and
        is placed here for visibility and easy of use. To access all parameters,
        use that directly

        Args:
            semitoneQuantization (float): Semitone quantization, 0 to disable quantization
            fftsize (int): FFT size
            simplify (float): Simplification threshold
            overlap (int): Overlap factor
            minFrequency (float): Minimum frequency
            minSilence (float): Minimum silence duration
            onsetThreshold (float): Onset threshold
            onsetOverlap (int): overlap factor for onset analysis

        Returns:
            a :class:`maelzel.transcribe.mono.FundamentalAnalysisMono`

        Example
        ~~~~~~~

            >>> from maelzel.snd import audiosample
            >>> samp = audiosample.Sample("sndfile.wav")
            >>> f0analysis = samp.fundamentalAnalysis()
            >>> notes = [(group.start(), group.duration(), group.meanfreq())
            ...          for group in f0analysis.groups]
        """
        from maelzel.transcribe import mono
        analysis = mono.FundamentalAnalysisMonophonic(samples=self.samples,
                                                      sr=self.sr,
                                                      semitoneQuantization=semitoneQuantization,
                                                      fftSize=fftsize,
                                                      overlap=overlap,
                                                      simplify=simplify,
                                                      minFrequency=minFrequency,
                                                      minSilence=minSilence,
                                                      onsetThreshold=onsetThreshold,
                                                      onsetOverlap=onsetOverlap)
        return analysis

    def onsets(self, fftsize=2048, overlap=4, method='rosita',
               threshold: float | None = None, mingap=0.03) -> np.ndarray:
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
        else:
            raise ValueError(f"method {method} not known. Possible methods: 'rosita'")

    def partialTrackingAnalysis(self,
                                resolution: float = 50.,
                                channel=0,
                                windowsize=0.,
                                freqdrift=0.,
                                hoptime=0.,
                                mindb=-90,
                                ) -> _spectrum.Spectrum:
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
        return Spectrum.analyze(samples=samples,
                                sr=self.sr,
                                resolution=resolution,
                                windowsize=windowsize,
                                hoptime=hoptime,
                                freqdrift=freqdrift,
                                mindb=mindb)

    def spectrumAt(self,
                   time: float,
                   resolution: float = 50.,
                   channel=0,
                   windowsize: float = -1,
                   mindb=-90,
                   minfreq=0,
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

    def fundamentalFreq(self, time: float | None = None, dur=0.2, fftsize=2048, overlap=4,
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

    def fundamental(self, fftsize=2048, overlap=4, unvoiced='negative', minAmpDb=-60, sensitivity=0.7
                    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Track the fundamental frequency of this sample

        Args:
            fftsize: the fft size to use
            overlap: number of overlaps
            unvoiced: one of 'negative' or 'nan'
            minAmpDb: the minimum amplitude in dB. Any sound softer than this
                will be supressed
            sensitivity: onset sensitivity, a value between 0 and 1

        Returns:
            a tuple (times, freqs), both numpy arrays. The frequency array will
            contain a negative frequency whenever the sound is unvoiced (inharmonic,
            no fundamental can be predicted)

        .. seealso:: :func:`maelzel.snd.vamptools.pyinSmoothPitch`,  :func:`maelzel.snd.freqestimate.f0curvePyinVamp`
        """
        from maelzel.snd import vamptools
        samples = _npsnd.getChannel(self.samples, 0, ensureContiguous=True)
        _util.checkChoice("unvoiced", unvoiced, choices=('negative', 'nan'))
        dt, freqs = vamptools.pyinSmoothPitch(samples, self.sr,
                                              fftSize=fftsize,
                                              stepSize=fftsize // overlap,
                                              outputUnvoiced='negative',
                                              lowAmpSuppression=pt.db2amp(minAmpDb),
                                              onsetSensitivity=sensitivity
                                              )
        times = np.arange(0, dt * len(freqs) - dt*0.5, dt)
        assert len(times) == len(freqs), f"{len(times)=}, {len(freqs)=}"
        return times, freqs

    def fundamentalBpf(self,
        fftsize=2048,
        overlap=4,
        unvoiced='negative',
        lowAmpSuppression=pt.db2amp(-60),
        onsetSensitivity=0.7,
        method='pyin-pitchtrack'
        ) -> tuple[bpf4.BpfInterface, bpf4.BpfInterface]:
        """
        Construct a bpf which follows the fundamental of this sample in time

        Args:
            fftsize: the size of the fft, in samples
            overlap: determines the hop size
            unvoiced: method to handle unvoiced sections. One of 'nan', 'negative', 'keep'
            method: one of 'pyin-pitchtrack' or 'pyin-smoothpitch'
            lowAmpSuppression: only analyzes audio louder than this threshold
            onsetSensitivity: onset sensitivity of the pyin algorithm

        Returns:
            a tuple (f0bpf, voicednessbpf), each is a `bpf <https://bpf4.readthedocs.io>`_.
            ``f0bpf`` represents the fundamental freq. over time, ``voicednessbpf``
            represents the voicedness (how "Pitched" the signal is) at a given time
        """
        from maelzel.snd import vamptools
        import bpf4
        samples = self.getChannel(0).samples
        if method == 'pyin-pitchtrack':
            data = vamptools.pyinPitchTrack(
                samples=samples,
                sr=self.sr,
                fftSize=fftsize,
                overlap=overlap,
                lowAmpSuppression=lowAmpSuppression,
                onsetSensitivity=onsetSensitivity,
                outputUnvoiced=unvoiced)
            times = data[:,0]
            freqs = data[:,1]
            voicedness = data[:,2]
            return bpf4.Linear(times, freqs), bpf4.Linear(times, voicedness)
        elif method == 'pyin-smoothpitch':
            dt, freqs = vamptools.pyinSmoothPitch(samples, self.sr,
                                                    fftSize=fftsize,
                                                    stepSize=fftsize // overlap,
                                                    lowAmpSuppression=lowAmpSuppression,
                                                    onsetSensitivity=onsetSensitivity,
                                                    outputUnvoiced=unvoiced)
            return bpf4.Sampled(freqs, dt), bpf4.Const(1.0)
        else:
            raise ValueError(f"Unknown method {method}")


    def chunks(self, chunksize: int, hop: int = 0, pad=False) -> Iterator[np.ndarray]:
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
        import emlib.numpytools as nptools
        return nptools.chunks(self.samples,
                              chunksize=chunksize,
                              hop=hop or None,
                              padwith=(0 if pad else None))

    def firstPitch(self, threshold=-120, minfreq=60, overlap=4, channel=0, chunkdur=0.25
                   ) -> tuple[float, float]:
        """
        Returns the first (monophonic) pitch found

        Args:
            threshold: the silence threhsold
            minfreq: the min. frequency to considere valid
            overlap: pitch analysis overlap
            channel: for multichannel audio, which channel to use
            chunkdur: chunk duration to analyze, in seconds

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
        chunksize = int(chunkdur * self.sr)
        for idx in range(firstidx, lastidx, chunksize):
            fragm = samples[idx:idx+chunksize]
            f0, prob = freqestimate.f0curve(fragm, sr=self.sr, minfreq=minfreq,
                                            overlap=overlap, unvoicedFreqs='nan')
            times, freqs = f0.points()
            mask = ~np.isnan(freqs)
            if not mask.any():
                continue
            selfreqs = freqs[mask]
            seltimes = times[mask]
            idx = min(len(selfreqs)-1, 3)
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

    def addChannels(self, channels: np.ndarray | int) -> Self:
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
        return self.__class__(frames, sr=self.sr)

    @staticmethod
    def mix(samples: list[Sample],
            offsets: list[float] | None = None,
            gains: list[float] | None = None,
            positions: list[float] | None = None
            ) -> Sample:
        """
        Static method: mix the given samples down, optionally with a time offset

        This is a static method. All samples should share the same
        number of channels and sr

        Args:
            samples: the Samples to mix
            offsets: if given, an offset in seconds for each sample
            gains: if given, a gain for each sample
            positions: if given, panning positions for each sample.

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
        return mixSamples(samples, offsets=offsets, gains=gains, positions=positions)

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
        return concatenate(samples)


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


def matchSamplerates(sampleseq: Sequence[Sample], sr: int = 0, forcecopy=False) -> list[Sample]:
    """
    Match the samplerates of the given Samples

    Args:
        sampleseq: a sequence of Sample instances
        sr: the sr to use or None to use the highest samplerate of all samples
        forcecopy: if True, a copy of the Sample is returned even if no resampling
            is needed

    Returns:
        a list of Samples, where all Samples share the same samplerate.
        Only samples which need to be resampled will be resampled. Sample
        instances matching the used samplerate will be returned as is
    """
    numchannels = sampleseq[0].numchannels
    if any(s.numchannels != numchannels for s in sampleseq):
        s = next(s for s in sampleseq if s.numchannels != numchannels)
        raise ValueError(f"All samples should have {numchannels} channels, "
                         f"but one Sample has {s.numchannels} channels")
    if not sr:
        sr = max(s.sr for s in sampleseq)

    if any(s.sr != sr for s in sampleseq):
        sampleseq = [s.resample(sr) if s.sr != sr else s.copy() if forcecopy else s for s in sampleseq]
    else:
        sampleseq = list(sampleseq)
    return sampleseq


def concatenate(sampleseq: Sequence[Sample]) -> Sample:
    """
    Concatenate a sequence of Samples

    Samples should share numchannels. If mismatching samplerates are found,
    all samples are upsampled to the highest sr

    Args:
        sampleseq: a seq. of Samples

    Returns:
        the concatenated samples as one Sample
    """
    s = np.concatenate([s.samples for s in matchSamplerates(sampleseq)])
    return Sample(s, sampleseq[0].sr)


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


def mixSamples(samples: list[Sample],
               offsets: list[float] | None = None,
               gains: list[float] | None = None,
               positions: list[float] | None = None
               ) -> Sample:
    """
    Mix the given samples down, optionally with a time offset

    All samples should share the same number of channels and sr

    Args:
        samples: the Samples to mix
        offsets: if given, an offset in seconds for each sample
        gains: if given, a gain for each sample
        positions: if given, panning positions for each sample (between 0 and 1)
            This will force the output sample to be stereo. Multichannel audio
            does not support panning

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
    nchannels = max(s.numchannels for s in samples)
    sr = samples[0].sr

    if not all(s.sr == sr for s in samples):
        raise ValueError(f"All samples should have the same samplerate, got {[s.sr for s in samples]}")

    if offsets is None:
        offsets = [0.] * len(samples)
    else:
        assert len(offsets) == len(samples)

    if gains is None:
        gains = [1.] * len(samples)
    else:
        assert len(gains) == len(samples)

    if positions:
        assert len(positions) == len(samples)
        if nchannels > 2:
            raise ValueError("Multichannel (> 2) samples are not supported with panning")
        nchannels = 2

    dur = max(s.duration + offset for s, offset in zip(samples, offsets))
    numframes = int(dur * sr)
    if nchannels == 1:
        buf = np.zeros((numframes,), dtype=float)
    else:
        buf = np.zeros((numframes, nchannels), dtype=float)
    for i in range(len(samples)):
        s, gain, offset = samples[i], gains[i], offsets[i]
        startframe = int(offset * sr)
        endframe = startframe + len(s)
        data = s.samples
        if positions and nchannels == 2:
            position = positions[i]
            data = _npsnd.panStereo(data, position)
        buf[startframe:endframe] += data
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
               minfreq=0,
               maxfreq=12000,
               maxcount=0
               ) -> list[tuple[float, float]]:
    """
    Analyze sinusoidal components of these samples at the given time

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
        import loristrck.util
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


def playSamples(samples: np.ndarray,
                sr: int,
                loop=False,
                chan=1,
                gain=1.0,
                speed=1.0,
                skip=0.0,
                dur=0.0,
                block=False
                ) -> PlaybackStream:
    """
    Simple playback for samples

    If more complex playback is needed, use ``Sample(samples, sr).play()``

    Args:
        samples: the samples to play
        sr: sample rate
        loop: should playback be looped?
        chan: first channel to play to. For stereo samples, output
            is routed to consecutive channels starting with this channel
        gain: a gain modifier
        speed: the playback speed. A variation in speed will change
            the pitch accordingly.
        skip: start playback at a given point in time
        dur: duration of playback. 0 indicates to play until the end of the sample
        block: if True, block execution until playback is finished
    Returns:
        a :class:`PlaybackStream`. This can be used to stop playback

    See Also
    ~~~~~~~~

    * :meth:`Sample.getEngine`
    * :meth:`Sample.setEngine`

    """
    numchannels = _npsnd.numChannels(samples)
    mapping = list(range(chan, numchannels + chan))
    if skip:
        samples = samples[int(sr * skip):]

    if dur:
        samples = samples[:int(sr*dur)]

    return _playSamples(samples=samples, mapping=mapping, sr=sr, loop=loop,
                        speed=speed, block=block, gain=gain)
