"""
Routines for plotting sounds / soundfiles

Uses matplotlib as a backend
"""
from __future__ import annotations

import emlib.misc
import emlib.mathlib
from emlib import numpytools
import numpy as np
import bpf4
import matplotlib.ticker
import matplotlib.pyplot as plt
from maelzel.snd.numpysnd import numChannels, getChannel
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union
    import matplotlib.pyplot as plt


logger = logging.getLogger('maelzel.snd')


__all__ = (
    'plotMelSpectrogram',
    'plotPowerSpectrum',
    'plotSpectrogram',
    'plotWaveform'
)


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



def _envelope(x, hop):
    return numpytools.overlapping_frames(x, hop_length=hop,
                                         frame_length=hop).max(axis=0)


def _frames_to_time(frames, sr, hop_length, n_fft=None):
    samples = _frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)
    return samples / sr


def _frames_to_samples(frames, hop_length=512, n_fft=None):
    offset = int(n_fft // 2) if n_fft else 0
    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def _plot_matplotlib(samples: np.ndarray, samplerate: int, timelabels: bool, figsize=(24, 4)
                     ) -> plt.Figure:
    numch = numChannels(samples)
    numsamples = samples.shape[0]
    f = plt.figure(figsize=figsize)
    ax1 = None
    if timelabels:
        formatter = matplotlib.ticker.FuncFormatter(
            lambda idx, x:emlib.misc.sec2str(idx/samplerate, msdigits=3))
    else:
        formatter = matplotlib.ticker.FuncFormatter(
            lambda idx, x:f"{idx/samplerate:.3g}")
    locator = _TimeLocator(samplerate)
    for i in range(numch):
        if i == 0:
            axes = ax1 = f.add_subplot(numch, 1, i + 1)
        else:
            axes = f.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)
        chan = getChannel(samples, i)
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


class _TimeLocator(matplotlib.ticker.LinearLocator):
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


def plotWaveform(samples,
                 samplerate: int,
                 profile='',
                 saveas='',
                 timelabels=True,
                 figsize=(24, 4)
                 ) -> list[plt.Axes]:
    """
    Plot the waveform of a sound using pyplot

    Args:
        samples: a mono or multichannel audio array
        samplerate: the sr
        profile: one of 'low', 'medium', 'high', 'highest'. If not given a preset is chosen
            based on the duration and other parameters given
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

    if profile == 'auto' or not profile:
        if saveas:
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
    numch = numChannels(samples)
    numsamples = samples.shape[0]
    if maxpoints < numsamples:
        targetsr = min(maxsr, (samplerate * numsamples) // maxpoints)
    hop_length = samplerate // targetsr
    f = plt.figure(figsize=figsize)
    ax1 = None
    timeFormatter = matplotlib.ticker.FuncFormatter(
        lambda s, x:emlib.misc.sec2str(s, msdigits=3))
    for i in range(numch):
        if i == 0:
            axes = ax1 = f.add_subplot(numch, 1, i + 1)
        else:
            axes = f.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)

        chan = getChannel(samples, i)
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


def plotSpectrogram(samples: np.ndarray,
                    samplerate: int,
                    fftsize=2048,
                    window: str = 'hamming',
                    overlap: int = None,
                    axes: plt.Axes = None,
                    cmap='inferno',
                    interpolation='bilinear',
                    minfreq=40,
                    maxfreq=16000,
                    mindb=-90,
                    setlabel=False,
                    figsize=(24, 8)
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
        cmap: colormap (see pyplot.colormaps())
        minfreq: initial min. frequency
        maxfreq: initial max. frequency. If None, a configurable default will be used
        interpolation: one of 'bilinear'
        mindb: the amplitude threshold
        figsize: if axes is not given, use this size to determine the figure size.
            The value should be a tuplet (width, height)

    Returns:
        the matplotlib axes object
    """
    if numChannels(samples) > 1:
        logger.info(f"plotSpectrogram only works on mono samples. Will use channel 0")
        samples = getChannel(samples, 0)

    from scipy import signal

    if axes is None:
        f: plt.Figure = plt.figure(figsize=figsize)
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
    win = signal.get_window(window, fftsize)
    axes.specgram(samples,
                  NFFT=fftsize,
                  Fs=samplerate,
                  noverlap=noverlap,
                  window=win,
                  cmap=cmap,
                  interpolation=interpolation,
                  vmin=mindb)
    axes.set_ylim(minfreq, maxfreq)
    if setlabel:
        axes.xaxis.set_label_text('Time')
        axes.yaxis.set_label_text('Hz')
    return axes


def plotMelSpectrogram(samples: np.ndarray,
                       sr: int,
                       fftsize=2048,
                       overlap=4,
                       winlength: int = None,
                       axes=None,
                       setlabel=False,
                       nmels=128,
                       cmap='magma',
                       figsize=(24, 8)
                       ) -> plt.Axes:
    """
    Plot a mel spectrogram

    Args:
        samples: the samples
        sr: the samplerate
        fftsize: fft size
        overlap: number of overlaps per window
        winlength: window length in samples (defaults to fftsize)
        axes: if given, use this axdes to plot into
        setlabel: a label to pass to specshow
        nmels: number of mel bands
        cmap: the color map used
        figsize: the figure size as a tuplet (width, height). Only used if axes is None

    Returns:
        the axes used

    """
    if len(samples.shape) > 1:
        samples = samples[:, 0]
    if winlength is None:
        winlength = fftsize
    from maelzel.snd import rosita
    hoplength = winlength // overlap
    if axes is None:
        fig: plt.Figure = plt.figure(figsize=figsize)
        axes: plt.Axes = fig.add_subplot(1, 1, 1)

    melspec = rosita.melspectrogram(y=samples, sr=sr, n_fft=fftsize,
                                    hop_length=hoplength, win_length=winlength,
                                    power=2.0, n_mels=nmels)
    SdB = rosita.power_to_db(melspec, ref=np.max)
    img = rosita.specshow(SdB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, axes=axes,
                          setlabel=setlabel, cmap=cmap, hop_length=hoplength)
    # axes.set(title='Mel-frequency spectrogram')
    return axes