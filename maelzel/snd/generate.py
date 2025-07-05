from __future__ import annotations
import numpy as np
from maelzel.snd import numpysnd as npsnd


def whiteNoise(dur: float, sr=44100, amp=1.) -> np.ndarray:
    """
    Uniformly distributed noise in the range (-1; 1), also called white noise

    Args:
        dur: duration in seconds
        sr: sample rate

    Returns:
        a numpy array holding one channel of samples
    """
    numsamples = int(dur * sr)
    arr = np.random.random(size=numsamples)
    arr *= 2. * amp
    arr -= 1.
    return arr


def _noise_psd(N, psd=lambda f: 1):
    X_white = np.fft.rfft(np.random.randn(N))
    S = psd(np.fft.rfftfreq(N))
    # Normalize S
    S = S / np.sqrt(np.mean(S ** 2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)


def pinkNoiseFFT(dur: float, sr=44100, peaknorm=True) -> np.ndarray:
    """
    Generate pink noise

    Args:
        dur: duration in seconds
        sr: samplerate
        peaknorm: apply max. peak normalization to keep samples between (-1, 1)

    Returns:
        a numpy array holding the audio samples
    """
    numsamps = int(dur * sr)
    white = np.fft.rfft(np.random.randn(numsamps))
    freqs = np.fft.rfftfreq(numsamps)
    S = 1/np.where(freqs == 0, float('inf'), np.sqrt(freqs))
    S = S / np.sqrt(np.mean(S**2))
    shaped = white * S
    out = np.fft.irfft(shaped)
    if peaknorm:
        out *= npsnd.normalizationRatio(out, maxdb=0)
    return out


def pinkNoise(dur: float, sr=44100, state: np.random.RandomState | None = None,
              peaknorm=True
              ) -> np.ndarray:
    """
    Pink noise

    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.

    via: https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py

    We don't use the library directly since it depends on pandas, which itself
    depends on numba, which we try to avoid. This might change in the future

    Args:
        dur: duration
        sr: samplerate
        state: State of PRNG.
        peaknorm: apply max. peak normalization to keep samples between (-1, 1)

    """
    # This method uses the filter with the following coefficients.
    # b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    # a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    # return lfilter(B, A, np.random.randn(N))
    # Another way would be using the FFT
    # x = np.random.randn(N)
    # X = rfft(x) / N
    N = int(dur * sr)
    state = np.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)  # +1 to avoid divide by zero
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    out = _normalize(y)
    if peaknorm:
        out *= npsnd.normalizationRatio(out, maxdb=0)
    return out


def _normalize(y, x=None):
    """
    Normalize power in y to a (standard normal) white noise signal.

    Optionally normalize to power in signal `x`.

    #The mean power of a Gaussian with :math:`\\mu=0` and :math:`\\sigma=1` is 1.
    """
    # return y * np.sqrt( (np.abs(x)**2.0).mean() / (np.abs(y)**2.0).mean() )
    if x is not None:
        x = _ms(x)
    else:
        x = 1.0
    return y * np.sqrt(x / _ms(y))
    # return y * np.sqrt(1.0 / (np.abs(y)**2.0).mean())


def _ms(x):
    """Mean value of signal `x` squared.
    """
    return (np.abs(x)**2.0).mean()


def gaussianNoise(dur: float, sr=44100, siderange=1, mu=0, clip=True
                  ) -> np.ndarray:
    """
    Gaussian distribution noise

    Args:
        dur: duration in seconds
        sr: sample rate
        siderange: 99.9% of samples will fall between the range (mu-siderange, mu+siderange)
        mu: centre of the distribution
        clip: if True, clip any value outside the range

    Returns:
        a numpy array holding one channel of samples

    """
    sigma = siderange / 3.83
    s = np.random.normal(mu, sigma, size=int(dur * sr))
    if clip:
        s.clip(mu-siderange, mu+siderange, out=s)
    return s
