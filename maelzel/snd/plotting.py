"""
Routines for plotting sounds / soundfiles

Uses matplotlib as a backend
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import functools

import bpf4
import emlib.mathlib
import emlib.misc
import matplotlib.ticker
import numpy as np
from emlib import numpytools

from maelzel.snd.numpysnd import getChannel, numChannels
from maelzel.common import getLogger

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes



__all__ = (
    'plotMelSpectrogram',
    'plotPowerSpectrum',
    'plotSpectrogram',
    'plotWaveform'
)


# the result of matplotlib.pyplot.colormaps()
# _matplotlib_cmaps = [
#     'Accent',
#     'Accent_r',
#     'cividis',
#     'cividis_r',
#     'cool',
#     'cool_r',
#     'coolwarm',
#     'coolwarm_r',
#     'copper',
#     'copper_r',
#     'cubehelix',
#     'cubehelix_r',
#     'gnuplot',
#     'gnuplot2',
#     'gnuplot2_r',
#     'gnuplot_r',
#     'gray',
#     'gray_r',
#     'hot',
#     'hot_r',
#     'hsv',
#     'hsv_r',
#     'inferno',
#     'inferno_r',
#     'jet',
#     'jet_r',
#     'magma',
#     'magma_r',
#     'nipy_spectral',
#     'nipy_spectral_r',
#     'ocean',
#     'ocean_r',
#     'pink',
#     'pink_r',
#     'plasma',
#     'plasma_r',
#     'prism',
#     'prism_r',
#     'rainbow',
#     'rainbow_r',
#     'seismic',
#     'seismic_r',
#     'spring',
#     'spring_r',
#     'summer',
#     'summer_r',
#     'terrain',
#     'terrain_r',
#     'turbo',
#     'turbo_r',
#     'twilight',
#     'twilight_r',
#     'twilight_shifted',
#     'twilight_shifted_r',
#     'viridis',
#     'viridis_r',
#     'winter',
#     'winter_r']


def plotPowerSpectrum(samples: np.ndarray,
                      samplerate: int,
                      framesize=2048,
                      window: str | tuple[str, float] = ('kaiser', 9),
                      axes: Axes | None = None,
                      figsize=(24, 4)
                      ) -> Axes:
    """
    Plot the power spectrum of a sound

    Args:
        samples: the samples to plot
        samplerate: the sr of thesamples
        framesize: the bigger the frame size, the smoother the plot
        window: As passed to scipy.signal.get_window. One of "blackman", "hamming", "hann",
            "bartlett", "flattop", "parzen", "bohman", "blackmanharris", "nuttall", "barthann", "kaiser" (needs beta),
            "gaussian" (needs standard deviation)
        axes: the axes to plot to
        figsize: figure size of the plot

    Returns:
        the axes used

    """
    import matplotlib.pyplot as plt
    if axes is None:
        f: Figure = plt.figure(figsize=figsize)
        axes = f.add_subplot(1, 1, 1)

    from scipy import signal
    w = signal.get_window(window, framesize)
    assert isinstance(w, np.ndarray)
    axes.psd(samples, NFFT=framesize, Fs=samplerate, window=lambda s, w=w: s*w)  # type: ignore
    return axes


def _envelope(x, hop):
    return numpytools.overlapping_frames(x, hop_length=hop,
                                         frame_length=hop).max(axis=0)


def _frames_to_time(frames, sr, hop_length, n_fft=None):
    samples = _frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)
    return samples / sr


def _frames_to_samples(frames, hop_length=512, n_fft=None):
    offset = int(n_fft // 2) if n_fft else 0
    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def _plot_matplotlib(samples: np.ndarray, samplerate: int, timelabels: bool,
                     figsize=(24, 4), tight=True
                     ) -> Figure:
    numch = numChannels(samples)
    numsamples = samples.shape[0]
    import matplotlib.pyplot as plt
    if tight:
        fig = plt.figure(figsize=figsize, layout='constrained')
    else:
        fig = plt.figure(figsize=figsize)
    ax1: Axes | None = None
    if timelabels:
        formatter = matplotlib.ticker.FuncFormatter(
            lambda idx, x:emlib.misc.sec2str(idx/samplerate, msdigits=3))
    else:
        formatter = matplotlib.ticker.FuncFormatter(
            lambda idx, x:f"{idx/samplerate:.3g}")
    locator = _TimeLocator(samplerate)
    for i in range(numch):
        if i == 0:
            axes = ax1 = fig.add_subplot(numch, 1, i + 1)
        else:
            axes = fig.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)
        chan = getChannel(samples, i)
        axes.plot(chan, linewidth=1)
        axes.xaxis.set_major_formatter(formatter)
        axes.xaxis.set_major_locator(locator)

    assert ax1 is not None
    ax1.set_xlim(0, numsamples)
    # use layout="constrained" instead
    # if tight:
    #     fig.tight_layout()
    return fig


@functools.cache
def _difftostep():
    return bpf4.NoInterpol.fromseq(
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
        self.difftostep = _difftostep()

    def tick_values(self, valmin: float, valmax: float):
        "secmin and secmax are the axis limits, return the tick locations here"
        if self.sr:
            secmin = valmin/self.sr
            secmax = valmax/self.sr
        else:
            secmin, secmax = valmin, valmax
        diff = secmax-secmin
        step = self.difftostep(diff)
        firstelem = emlib.mathlib.next_in_grid(secmin, step, offset=0)
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
                 ) -> Figure:
    """
    Plot the waveform of a sound using pyplot

    Args:
        samples: a mono or multichannel audio array
        samplerate: the sr
        profile: one of 'low', 'medium', 'high', 'highest'. If not given a preset is chosen
            based on the duration and other parameters given
        saveas: if given, the plot is saved and not displayed
        timelabels: if True, the x axes' labels are shown as MM:SS if needed
        figsize: the figsize used

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
    import matplotlib.pyplot as plt
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
        fig = _plot_matplotlib(samples, samplerate, timelabels=timelabels, figsize=figsize)
        if saveas:
            plt.close(fig)
            fig.savefig(saveas, transparent=False, facecolor="white", bbox_inches='tight')
        return fig
    elif profile == 'low':
        maxpoints, maxsr = 600, 20
    elif profile == 'medium' or profile == 'middle':
        maxpoints, maxsr = 1200, 40
    else:
        raise ValueError("preset should be one of 'low', 'medium', 'high' or 'highest'")

    targetsr = samplerate
    numch = numChannels(samples)
    numsamples = samples.shape[0]
    if maxpoints < numsamples:
        targetsr = min(maxsr, (samplerate * numsamples) // maxpoints)
    hop_length = samplerate // targetsr
    f = plt.figure(figsize=figsize, layout="constrained")
    # f.set_tight_layout(True)
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
    return f


def plotSpectrogram(samples: np.ndarray,
                    sr: int,
                    fftsize=2048,
                    window: str = 'hamming',
                    winsize: int = 0,
                    overlap=4,
                    axes: Axes | None = None,
                    cmap='inferno',
                    interpolation='bilinear',
                    minfreq=40,
                    maxfreq=16000,
                    mindb=-90,
                    axeslabels=False,
                    figsize=(24, 8),
                    method='specgram',
                    yaxis='linear'
                    ) -> Axes:
    """
    Plot the spectrogram of a sound

    Args:
        samples: a channel of audio data
        sr: sr of the audio data
        fftsize: size of the fft, in samples
        winsize: size of the window, in samples
        window: string passed to scipy.signal.get_window
        overlap: number of overlaps. If fftsize=2048, an overlap of 4 will result
            in a hoplength of 512 samples. Use None to use a sensible value for the
            number of samples.
        axes: axes to plot on. If None, new axes will be created
        cmap: colormap (see pyplot.colormaps())
        minfreq: initial min. frequency
        maxfreq: initial max. frequency. If None, a configurable default will be used
        interpolation: one of 'bilinear'
        mindb: the amplitude threshold
        figsize: if axes is not given, use this size to determine the figure size.
            The value should be a tuplet (width, height)
        yaxis: one of 'linear', 'log'
        method: the actual routine to plot the data, one of 'specgram', 'specshow'
            (based on librosa)
        axeslabels: draw labels for the axes

    Returns:
        the matplotlib axes object
    """
    if numChannels(samples) > 1:
        getLogger("maelzel.snd").info("plotSpectrogram only works on mono samples. Will use channel 0")
        samples = getChannel(samples, 0)

    if yaxis == 'log' or method == 'specshow':
        return _plotSpectrogramRosita(samples=samples, sr=sr, fftsize=fftsize,
                                      window=window, winsize=winsize, overlap=overlap,
                                      axes=axes, cmap=cmap, figsize=figsize,
                                      yaxis=yaxis, maxfreq=maxfreq, minfreq=minfreq)

    import matplotlib.pyplot as plt
    if axes is None:
        f: Figure = plt.figure(figsize=figsize)
        axes = f.add_subplot(1, 1, 1)

    # specgram does not support window sizes different than fftsize
    winsize = fftsize

    hopsize = int(fftsize // overlap)
    noverlap = fftsize - hopsize

    from scipy import signal
    win = signal.get_window(window, winsize)
    axes.specgram(samples,
                  NFFT=fftsize,
                  Fs=sr,
                  noverlap=noverlap,
                  window=win,
                  cmap=cmap,
                  interpolation=interpolation,
                  vmin=mindb)
    axes.set_ylim(minfreq, maxfreq)
    if axeslabels:
        axes.xaxis.set_label_text('Time')
        axes.yaxis.set_label_text('Hz')
    return axes


def plotMelSpectrogram(samples: np.ndarray,
                       sr: int,
                       fftsize=2048,
                       overlap=4,
                       winsize=0,
                       axes: Axes | None = None,
                       setlabel=False,
                       nmels=128,
                       cmap='magma',
                       figsize=(24, 8),
                       minfreq=0,
                       maxfreq=16000
                       ) -> Axes:
    """
    Plot a mel spectrogram

    Args:
        samples: the samples
        sr: the samplerate
        fftsize: fft size
        overlap: number of overlaps per window
        winsize: window size in samples (defaults to fftsize)
        axes: if given, use this axdes to plot into
        setlabel: a label to pass to specshow
        nmels: number of mel bands
        cmap: the color map used
        figsize: the figure size as a tuplet (width, height). Only used if axes is None
        minfreq: the min. freq of the mel spectrum
        maxfreq: the max. freq of the mel spectrum

    Returns:
        the axes used

    """
    import matplotlib.pyplot as plt
    if len(samples.shape) > 1:
        samples = samples[:, 0]
    if not winsize:
        winsize = fftsize
    from maelzel.snd import rosita
    hoplength = winsize // overlap
    if axes is None:
        fig: Figure = plt.figure(figsize=figsize)
        axes = fig.add_subplot(1, 1, 1)

    melspec = rosita.melspectrogram(y=samples, sr=sr, n_fft=fftsize,
                                    hop_length=hoplength, win_length=winsize,
                                    power=2.0, n_mels=nmels,
                                    fmin=minfreq, fmax=maxfreq)
    SdB = rosita.power_to_db(melspec, ref=np.max)
    rosita.specshow(SdB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, axes=axes,
                    setlabel=setlabel, cmap=cmap, hop_length=hoplength)
    # axes.set(title='Mel-frequency spectrogram')
    return axes


def _plotSpectrogramRosita(samples: np.ndarray,
                           sr: int,
                           fftsize=2048,
                           window='hamming',
                           winsize=0,
                           overlap=4,
                           axes: Axes | None = None,
                           cmap='inferno',
                           figsize=(24, 8),
                           yaxis='log',
                           dbamps=True,
                           maxfreq=16000,
                           minfreq=40
                           ) -> Axes:
    from maelzel.snd import rosita
    import matplotlib.pyplot as plt
    if not winsize:
        winsize = fftsize

    if winsize > fftsize:
        raise ValueError(f"Window size ({winsize}) should be <= fft size ({fftsize})")

    hoplength = fftsize // overlap

    # compute stft
    # winarray = signal.get_window(window, winsize)

    stft = rosita.stft(samples, n_fft=fftsize, hop_length=hoplength, win_length=winsize,
                       window=window)
    # out = 2 * np.abs(stft) / np.sum(winarray)
    out = np.abs(stft)
    if dbamps:
        out = rosita.amplitude_to_db(out, ref=np.max)

    # plot result
    if axes is None:
        plt.figure(figsize=figsize)
        axes = plt.axes()

    # axes.set_axis_off()
    rosita.specshow(out, ax=axes, n_fft=fftsize, hop_length=hoplength,
                    cmap=cmap, y_axis=yaxis,
                    x_axis='time',sr=sr, fmax=maxfreq, fmin=minfreq)
    return axes
