from emlib import numpytools
from scipy import signal
import numpy as np
from configdict import ConfigDict


def _get_cmaps():
    import matplotlib.pyplot as plt
    return plt.colormaps()


config = ConfigDict("emlib:snd_plotting",
                    default={'spectrogram_colormap': 'inferno'},
                    validator={'spectrogram_colormap::choices': _get_cmaps})


def plot_power_spectrum(samples,
                        samplerate,
                        framesize=2048,
                        window=('kaiser', 9)):
    """
    Args:
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
    import matplotlib.pyplot as plt
    numch = get_num_channels(samples)
    numsamples = samples.shape[0]
    dur = numsamples / samplerate
    times = np.linspace(0, dur, numsamples)
    for i in range(numch):
        if i == 0:
            axes = ax1 = plt.subplot(numch, 1, i + 1)
        else:
            axes = plt.subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)
        chan = get_channel(samples, i)
        axes.plot(times, chan, linewidth=1)
        plt.xlim([0, dur])
    return True


def _plot_samples_matplotlib2(samples, samplerate, profile):
    # todo: copiar plot de librosa
    import matplotlib.pyplot as plt
    if profile == 'auto':
        dur = len(samples)/samplerate
        if dur > 60*8:
            # more than 5 minutes
            profile = 'low'
        elif dur > 60*2:
            profile = 'medium'
        else:
            profile = 'high'
    if profile == 'low':
        maxpoints = 10000
        maxsr = 500
    elif profile == 'medium':
        maxpoints = 40000
        maxsr = 2000
    elif profile == 'high':
        return _plot_matplotlib(samples, samplerate)
    else:
        raise ValueError("profile should be one of 'low', 'medium' or 'high'")
    targetsr = samplerate
    hop_length = 1
    numch = get_num_channels(samples)
    numsamples = samples.shape[0]
    if maxpoints is not None:
        if maxpoints < numsamples:
            targetsr = min(maxsr, (samplerate * numsamples) // maxpoints)
        hop_length = samplerate // targetsr
    print(f"targetsr: {targetsr}, hop_length: {hop_length}")
    for i in range(numch):
        if i == 0:
            axes = ax1 = plt.subplot(numch, 1, i + 1)
        else:
            axes = plt.subplot(numch, 1, i + 1, sharex=ax1, sharey=ax1)
        if i < numch - 1:
            plt.setp(axes.get_xticklabels(), visible=False)

        chan = get_channel(samples, i)
        env = _envelope(np.ascontiguousarray(chan), hop_length)
        print(env.max())
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


def plot_samples(samples, samplerate, profile="medium"):
    backends = [
        ('pyqtgraph', _plot_samples_pyqtgraph),
        ('matplotlib', _plot_samples_matplotlib2),
    ]
    for backend, func in backends:
        ok = func(samples, samplerate, profile)
        if ok:
            break


def spectrogram(samples,
                samplerate,
                fftsize=2048,
                window='hamming',
                overlap=4,
                cmap=None,
                mindb=-90):
    mpl_spectrogram(samples=samples,
                    samplerate=samplerate,
                    fftsize=fftsize,
                    window=window,
                    overlap=overlap,
                    mindb=mindb)


def mpl_spectrogram(samples,
                    samplerate,
                    fftsize=2048,
                    window='hamming',
                    overlap=4,
                    axes=None,
                    cmap=None,
                    interpolation='bilinear',
                    mindb=-90):
    """
    samples: a channel of audio data
    samplerate: the samplerate of the audio data
    fftsize: the size of the fft, in samples
    window: a string passed to scipy.signal.get_window
    overlap: the number of overlaps. If fftsize=2048, an overlap of 4 will result
        in a hopsize of 512 samples
    """
    import matplotlib.pyplot as plt
    if axes is None:
        axes = plt.subplot(1, 1, 1)
    hopsize = int(fftsize / overlap)
    noverlap = fftsize - hopsize
    win = signal.get_window(window, fftsize)
    cmap = cmap if cmap is not None else config['spectrogram_colormap']
    axes.specgram(samples,
                  NFFT=fftsize,
                  Fs=samplerate,
                  noverlap=noverlap,
                  window=win,
                  cmap=cmap,
                  interpolation=interpolation,
                  vmin=mindb)
    return axes
