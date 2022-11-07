"""
Routines for plotting sounds / soundfiles

Uses matplotlib as a backend
"""
from __future__ import annotations

import emlib.misc
import emlib.mathlib
from emlib import numpytools
import numpy as np
from configdict import ConfigDict
import bpf4
import matplotlib.ticker
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union
    import matplotlib.pyplot as plt


# the result of matplotlib.pyplot.colormaps()
_matplotlib_cmaps = [
 'Accent',
 'Accent_r',
 'cividis',
 'cividis_r',
 'cool',
 'cool_r',
 'coolwarm',
 'coolwarm_r',
 'copper',
 'copper_r',
 'cubehelix',
 'cubehelix_r',
 'gnuplot',
 'gnuplot2',
 'gnuplot2_r',
 'gnuplot_r',
 'gray',
 'gray_r',
 'hot',
 'hot_r',
 'hsv',
 'hsv_r',
 'inferno',
 'inferno_r',
 'jet',
 'jet_r',
 'magma',
 'magma_r',
 'nipy_spectral',
 'nipy_spectral_r',
 'ocean',
 'ocean_r',
 'pink',
 'pink_r',
 'plasma',
 'plasma_r',
 'prism',
 'prism_r',
 'rainbow',
 'rainbow_r',
 'seismic',
 'seismic_r',
 'spring',
 'spring_r',
 'summer',
 'summer_r',
 'terrain',
 'terrain_r',
 'turbo',
 'turbo_r',
 'twilight',
 'twilight_r',
 'twilight_shifted',
 'twilight_shifted_r',
 'viridis',
 'viridis_r',
 'winter',
 'winter_r']


config = ConfigDict("maelzel.snd.plotting")
with config as _:
    _('backend', 'matplotlib', choices={'matplotlib'},
      doc="Default backend to use for plotting")
    _('matplotlib.spectrogram.colormap', 'inferno', choices=_matplotlib_cmaps,
      doc="Colormap used when plotting a spectrogram using matplotlib")
    _('matplotlib.samplesplot.figsize', [24, 4],
      doc="Figure size used when plotting audio samples")
    _('matplotlib.spectrogram.figsize', [24, 8],
      doc="Figure size used when plotting a spectrogram using matplotlib")
    _('spectrogram.maxfreq', 12000,
      doc="Highest frequency in a spectrogram")
    _('spectrogram.window', 'hamming', choices={'hamming', 'hanning'})


def plotPowerSpectrum(samples: np.ndarray,
                      samplerate: int,
                      framesize=2048,
                      window: Union[str, tuple[str, float]] = ('kaiser', 9)
                      ) -> None:
    """
    Plot the power spectrum of a sound

    Args:
        samples: the samples to plot
        samplerate: the sr of thesamples
        framesize: the bigger the frame size, the smoother the plot
        window: As passed to scipy.signal.get_window. One of "blackman", "hamming", "hann",
            "bartlett", "flattop", "parzen", "bohman", "blackmanharris", "nuttall", "barthann", "kaiser" (needs beta),
            "gaussian" (needs standard deviation)

    """
    from scipy import signal
    w = signal.get_window(window, framesize)
    return plt.psd(samples, framesize, samplerate, window=lambda s, w=w: s*w)


def _get_channel(samples, channel):
    if len(samples.shape) == 1:
        return samples
    return samples[:, channel]


def _get_num_channels(samples):
    if len(samples.shape) == 1:
        return 1
    return samples.shape[1]


def _iter_channels(samples, start=0, end=0):
    numch = _get_num_channels(samples)
    if end == 0 or end > numch:
        end = numch
    for i in range(start, end):
        yield _get_channel(samples, i)


def _envelope(x, hop):
    return numpytools.overlapping_frames(x, hop_length=hop,
                                         frame_length=hop).max(axis=0)


def _frames_to_time(frames, sr, hop_length, n_fft=None):
    samples = _frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)
    return samples / sr


def _frames_to_samples(frames, hop_length=512, n_fft=None):
    offset = int(n_fft // 2) if n_fft else 0
    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def _plot_matplotlib(samples:np.ndarray, samplerate:int, timelabels:bool) -> plt.Figure:
    numch = _get_num_channels(samples)
    numsamples = samples.shape[0]
    figsize = config['matplotlib.samplesplot.figsize']
    f = plt.figure(figsize=figsize)
    ax1 = None
    if timelabels:
        formatter = matplotlib.ticker.FuncFormatter(
            lambda idx, x:emlib.misc.sec2str(idx/samplerate, msdigits=3))
    else:
        formatter = matplotlib.ticker.FuncFormatter(
            lambda idx, x:f"{idx/samplerate:.3g}")
    locator = TimeLocator(samplerate)
    for i in range(numch):
        if i == 0:
            axes = ax1 = f.add_subplot(numch, 1, i + 1)
        else:
            axes = f.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)
        chan = _get_channel(samples, i)
        # axes.plot(times, chan, linewidth=1)
        axes.plot(chan, linewidth=1)
        axes.xaxis.set_major_formatter(formatter)
        axes.xaxis.set_major_locator(locator)

    ax1.set_xlim(0, numsamples)
    return f


_diff_to_step = bpf4.nointerpol(
        0.01, 1/1000,
        0.025, 1/500,
        0.05, 1/100,
        0.1, 1/50,
        0.25, 1/30,
        0.5, 1/20,
        1, 1/10,
        2, 1/5,
        5, 1/2,
        10, 1,
        20, 2,
        40, 5,
        100, 10,
        200, 20,
        400, 30,
        600, 60,
        2000, 120,
        4000, 240,
        8000, 400,
        16000, 600,
)


class TimeLocator(matplotlib.ticker.LinearLocator):
    """
    A locator for time plotting

    Example
    -------

    .. code-block:: python

        import matplotlib.pyplot as plt
        from maelzel.snd import plotting
        import sndfileio
        samples, sr = sndfileio.sndread("/path/to/soundfile.wav")
        fig, axes = plt.subplots()
        locator = plotting.TimeLocator(sr)
        axes.xaxis.set_major_locator(locator)
        axes.plot(samples)

    """
    def __init__(self, sr: int = 0):
        super().__init__()
        self.sr = sr

    def tick_values(self, valmin: float, valmax: float):
        "secmin and secmax are the axis limits, return the tick locations here"
        if self.sr:
            secmin = valmin/self.sr
            secmax = valmax/self.sr
        else:
            secmin, secmax = valmin, valmax
        diff = secmax-secmin
        step = _diff_to_step(diff)
        firstelem = emlib.mathlib.next_in_grid(secmin, step)
        numticks = int((secmax+step - firstelem) / step)
        if numticks <= 3:
            step = step/2
        ticks = list(emlib.mathlib.frange(firstelem, secmax+step, step))
        if not self.sr:
            return ticks
        return [int(tick*self.sr) for tick in ticks]


def plotWaveform(samples, samplerate, profile:str = None, saveas:str=None,
                 timelabels=True) -> list[plt.Axes]:
    """
    Plot the waveform of a sound using pyplot

    Args:
        samples: a mono or multichannel audio array
        samplerate: the sr
        profile: one of 'low', 'medium', 'high', 'highest'. If None or 'auto' is
            passe, a preset is chosen based on the duration and other parameters given
        saveas: if given, the plot is saved and not displayed
        timelabels: if True, the x axes' labels are shown as MM:SS if needed

    Returns:
        the axes used (one per channel)

    Example
    ~~~~~~~

    .. code-block:: python

        >>> from maelzel.snd import plotting
        >>> import sndfileio
        >>> samples, info = sndfileio.sndget("snd/bach-presto-gmoll.mp3")
        # In this case the preset used will be 'high'
        >>> plotting.plotWaveform(samples, info.sr)

    .. image:: ../assets/snd-plotting-plotWaveform-high.png

    .. code-block:: python

        >>> plotting.plotWaveform(samples, info.sr, preset='low')

    .. image:: ../assets/snd-plotting-plotWaveform-low.png

    .. code-block:: python

        >>> plotting.plotWaveform(samples, info.sr, preset='medium')

    .. image:: ../assets/snd-plotting-plotWaveform-medium.png

    .. code-block:: python

        >>> plotting.plotWaveform(samples, info.sr, preset='high')

    .. image:: ../assets/snd-plotting-plotWaveform-high.png

    .. code-block:: python

        >>> plotting.plotWaveform(samples, info.sr, preset='highest')

    .. image:: ../assets/snd-plotting-plotWaveform-highest.png



    """
    dur = len(samples) / samplerate

    if profile == 'auto' or profile is None:
        if saveas is not None:
            profile = 'highest'
        elif dur > 60*8:
            profile = 'low'
        elif dur > 60*2:
            profile = 'medium'
        elif dur > 60*1:
            profile = 'high'
        else:
            profile = 'highest'

    if profile == 'high' or profile == 'highest':
        if profile == 'high':
            undersample = min(32, len(samples) // (1024 * 8))
            samples = samples[::undersample]
            samplerate = samplerate // undersample
        fig = _plot_matplotlib(samples, samplerate, timelabels=timelabels)
        if saveas:
            plt.close(fig)
            fig.savefig(saveas, transparent=False, facecolor="white", bbox_inches='tight')
        return fig.axes
    elif profile == 'low':
        maxpoints, maxsr = 600, 20
    elif profile == 'medium':
        maxpoints, maxsr = 1200, 40
    else:
        raise ValueError("preset should be one of 'low', 'medium' or 'highest'")

    targetsr = samplerate
    numch = _get_num_channels(samples)
    numsamples = samples.shape[0]
    if maxpoints < numsamples:
        targetsr = min(maxsr, (samplerate * numsamples) // maxpoints)
    hop_length = samplerate // targetsr
    figsize = config['matplotlib.samplesplot.figsize']
    f = plt.figure(figsize=figsize)
    ax1 = None
    timeFormatter = matplotlib.ticker.FuncFormatter(
        lambda s, x:emlib.misc.sec2str(s, msdigits=3))
    locator = TimeLocator(sr=samplerate)
    for i in range(numch):
        if i == 0:
            axes = ax1 = f.add_subplot(numch, 1, i + 1)
        else:
            axes = f.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)

        chan = _get_channel(samples, i)
        env = _envelope(np.ascontiguousarray(chan), hop_length)
        samples_top = env
        samples_bottom = -env
        locs = _frames_to_time(np.arange(len(samples_top)),
                               sr=samplerate,
                               hop_length=hop_length)
        axes.fill_between(locs, samples_bottom, samples_top)
        axes.set_xlim([locs.min(), locs.max()])
        # axes.xaxis.set_major_locator(locator)
        if timelabels:
            axes.xaxis.set_major_formatter(timeFormatter)
    if saveas:
        plt.close(f)
        f.savefig(saveas, transparent=False, facecolor="white", bbox_inches='tight')
    return f.axes


def plotSpectrogram(samples: np.ndarray, samplerate: int, fftsize=2048, window:str=None,
                    overlap:int=None, axes:plt.Axes=None, cmap=None, interpolation='bilinear',
                    minfreq=40, maxfreq=None,
                    mindb=-90
                    ) -> plt.Axes:
    """
    Plot the spectrogram of a sound

    Args:
        samples: a channel of audio data
        samplerate: the sr of the audio data
        fftsize: the size of the fft, in samples
        window: a string passed to scipy.signal.get_window
        overlap: the number of overlaps. If fftsize=2048, an overlap of 4 will result
            in a hoplength of 512 samples. Use None to use a sensible value for the
            number of samples.
        axes: the axes to plot on. If None, new axes will be created
        cmap: colormap, see pyplot.colormaps() (see config['spectrogram.cmap'])
        minfreq: initial min.frequency
        maxfreq: initial max. frequency. If None, a configurable default will be used
            (see config['spectrogram.maxfreq')
        interpolation: one of 'bilinear'
        mindb: the amplitude threshold

    Returns:
        the matplotlib axes object
    """
    from scipy import signal

    if axes is None:
        f: plt.Figure = plt.figure(figsize=config['matplotlib.spectrogram.figsize'])
        axes:plt.Axes = f.add_subplot(1, 1, 1)
    if overlap is None:
        dur = len(samples) / samplerate
        if dur < 10:
            overlap = 4
        elif dur < 60:
            overlap = 2
        else:
            overlap = 1
    hopsize = int(fftsize / overlap)
    noverlap = fftsize - hopsize
    if window is None:
        window = config['spectrogram.window']
    win = signal.get_window(window, fftsize)
    cmap = cmap if cmap is not None else config['matplotlib.spectrogram.colormap']
    axes.specgram(samples,
                  NFFT=fftsize,
                  Fs=samplerate,
                  noverlap=noverlap,
                  window=win,
                  cmap=cmap,
                  interpolation=interpolation,
                  vmin=mindb)
    if maxfreq is None:
        maxfreq = config['spectrogram.maxfreq']
    axes.set_ylim(minfreq, maxfreq)
    return axes
