from __future__ import annotations
import numpy as np
import typing as _t
if _t.TYPE_CHECKING:
    import bpf4


def centroidBpf(samples: np.ndarray,
                sr: int,
                fftsize: int = 2048,
                overlap: int = 4,
                winsize: int = 0,
                window='hann'
                ) -> bpf4.Sampled:
    """
    Construct a bpf representing the centroid of the given audio over time

    Args:
        samples: a 1D numpy array representing a mono audio fragment
        sr: the sampling rate
        fftsize: the fft size
        overlap: amount of overlap
        winsize: the size of the window. If not given then winsize is assumed to be
            the same as fftsize. if given it must be <= fftsize
        window: kind of window

    Returns:
        a bpf representing the centroid over time

    """
    from maelzel.snd import rosita
    if len(samples.shape) > 1:
        raise ValueError("Only mono samples are supported")
    winsize = winsize or fftsize
    hopsize = winsize // overlap
    frames = rosita.spectral_centroid(y=samples,
                                      sr=sr,
                                      n_fft=fftsize,
                                      hop_length=hopsize,
                                      win_length=winsize,
                                      window=window)
    import bpf4
    return bpf4.Sampled(frames[0], x0=0, dx=hopsize/sr)
