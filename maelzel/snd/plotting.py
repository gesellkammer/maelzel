from emlib import numpytools
from scipy import signal
import numpy as np
from configdict import ConfigDict
import matplotlib.pyplot as plt


def _cmaps():
    return plt.colormaps()


config = ConfigDict("maelzel.snd.plotting")
config.addKey('spectrogram.colormap', 'inferno', choices=_cmaps())
config.addKey('samplesplot.figsize', (12, 4))
config.addKey('spectrogram.figsize', (24, 8))
config.addKey('spectrogram.maxfreq', 12000,
              doc="Highest frequency in a spectrogram")
config.addKey('spectrogram.window', 'hamming', choices={'hamming', 'hanning'})
config.load()


def plot_power_spectrum(samples,
                        samplerate,
                        framesize=2048,
                        window=('kaiser', 9)):
    """
    Args:
        samples: the samples to plot
        samplerate: the samplerate of thesamples
        framesize: the bigger the frame size, the smoother the plot
        window: As passed to scipy.signal.get_window
          `blackman`, `hamming`, `hann`, `bartlett`, `flattop`, `parzen`, `bohman`,
          `blackmanharris`, `nuttall`, `barthann`, `kaiser` (needs beta),
          `gaussian` (needs standard deviation)



    """
    w = signal.get_window(window, framesize)

    def func(s):
        return s * w

    import matplotlib.pyplot as plt
    return plt.psd(samples, framesize, samplerate, window=func)


def get_channel(samples, channel):
    if len(samples.shape) == 1:
        return samples
    return samples[:, channel]


def get_num_channels(samples):
    if len(samples.shape) == 1:
        return 1
    return samples.shape[1]


def iter_channels(samples, start=0, end=0):
    numch = get_num_channels(samples)
    if end == 0 or end > numch:
        end = numch
    for i in range(start, end):
        yield get_channel(samples, i)


def _envelope(x, hop):
    return numpytools.overlapping_frames(x, hop_length=hop,
                                         frame_length=hop).max(axis=0)


def _frames_to_time(frames, sr, hop_length, n_fft=None):
    samples = _frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)
    return samples / sr


def _frames_to_samples(frames, hop_length=512, n_fft=None):
    offset = int(n_fft // 2) if n_fft else 0
    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def _plot_matplotlib(samples, samplerate):
    numch = get_num_channels(samples)
    numsamples = samples.shape[0]
    dur = numsamples / samplerate
    times = np.linspace(0, dur, numsamples)
    figsize = config['samplesplot.figsize']
    figsize = figsize[0]*2, figsize[1]
    f = plt.figure(figsize=figsize)
    ax1 = None
    for i in range(numch):
        if i == 0:
            axes = ax1 = f.add_subplot(numch, 1, i + 1)
        else:
            axes = f.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)
        chan = get_channel(samples, i)
        axes.plot(times, chan, linewidth=1)
        plt.xlim([0, dur])
    return True


def _plot_samples_matplotlib2(samples, samplerate, profile):
    import matplotlib.pyplot as plt
    if profile == 'auto':
        dur = len(samples)/samplerate
        if dur > 60*8:
            profile = 'low'
        elif dur > 60*2:
            profile = 'medium'
        elif dur > 60*1:
            profile = 'high'
        else:
            profile = 'highest'
    if profile == 'low':
        maxpoints = 2000
        maxsr = 300
    elif profile == 'medium':
        maxpoints = 4000
        maxsr = 600
    elif profile == 'high':
        undersample = min(32, len(samples) // (1024*8))
        return _plot_matplotlib(samples[::undersample], samplerate//undersample)
    elif profile == 'highest':
        return _plot_matplotlib(samples, samplerate)
    else:
        raise ValueError("profile should be one of 'low', 'medium' or 'high'")
    targetsr = samplerate
    numch = get_num_channels(samples)
    numsamples = samples.shape[0]
    if maxpoints < numsamples:
        targetsr = min(maxsr, (samplerate * numsamples) // maxpoints)
    hop_length = samplerate // targetsr
    figsize = config['samplesplot.figsize']
    if profile == "medium":
        figsize = int(figsize[0]*1.4), figsize[1]
    for i in range(numch):
        f = plt.figure(figsize=figsize)
        if i == 0:
            axes = ax1 = f.add_subplot(numch, 1, i + 1)
        else:
            axes = f.add_subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)

        chan = get_channel(samples, i)
        env = _envelope(np.ascontiguousarray(chan), hop_length)
        samples_top = env
        samples_bottom = -env
        locs = _frames_to_time(np.arange(len(samples_top)),
                               sr=samplerate,
                               hop_length=hop_length)
        axes.fill_between(locs, samples_bottom, samples_top)
        axes.set_xlim([locs.min(), locs.max()])
    return True


def _plot_samples_pyqtgraph(samples, samplerate, profile):
    # TODO
    return False


def plot_samples(samples: np.ndarray, samplerate: int, profile="auto") -> None:
    backends = [
        ('pyqtgraph', _plot_samples_pyqtgraph),
        ('matplotlib', _plot_samples_matplotlib2),
    ]
    for backend, func in backends:
        ok = func(samples, samplerate, profile)
        if ok:
            break


def spectrogram(samples: np.ndarray, samplerate: int, fftsize=2048, window:str=None,
                overlap=4, axes:plt.Axes=None, cmap=None, interpolation='bilinear',
                minfreq=40, maxfreq=None,
                mindb=-90):
    """
    Args:
        samples: a channel of audio data
        samplerate: the samplerate of the audio data
        fftsize: the size of the fft, in samples
        window: a string passed to scipy.signal.get_window
        overlap: the number of overlaps. If fftsize=2048, an overlap of 4 will result
            in a hopsize of 512 samples
        axes: the axes to plot on. If None, new axes will be created
        cmap: colormap, see pyplot.colormaps() (see config['spectrogram.cmap'])
        minfreq: initial min.frequency
        maxfreq: initial max. frequency. If None, a configurable default will be used
            (see config['spectrogram.maxfreq')
        interpolation: one of 'bilinear'
        mindb: the amplitude threshold

    Returns:
        the axes object
    """
    if axes is None:
        f: plt.Figure = plt.figure(figsize=config['spectrogram.figsize'])
        axes:plt.Axes = f.add_subplot(1, 1, 1)
    hopsize = int(fftsize / overlap)
    noverlap = fftsize - hopsize
    if window is None:
        window = config['spectrogram.window']
    win = signal.get_window(window, fftsize)
    cmap = cmap if cmap is not None else config['spectrogram.colormap']
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
