"""
Multiple backends to perform resampling on numpy arrays
"""
from __future__ import annotations
import numpy as np
import importlib.util
import functools


@functools.cache
def _getBackend() -> str:
    backends = ['soxr', 'resampy', 'samplerate']
    for backend in backends:
        if importlib.util.find_spec(backend) is not None:
            return backend
    return ''


def _resampleResampy(samples: np.ndarray, samplerate: int, newsamplerate: int
                     ) -> np.ndarray:
    try:
        import resampy
    except ImportError:
        raise RuntimeError("resampy is needed, install it via `pip install resampy`"
                           " (see https://github.com/bmcfee/resampy)")
    return resampy.resample(samples, samplerate, newsamplerate)


def _resampleSamplerate(samples: np.ndarray, samplerate: int, newsamplerate: int
                        ) -> np.ndarray:
    try:
        import samplerate as sampleratecffi
        ratio = samplerate / newsamplerate
        return sampleratecffi.resample(samples, ratio, 'sinc_best')
    except ImportError:
        raise ImportError("samplerate package needed, install it via `pip install samplerate`")


def _resampleSoxr(samples: np.ndarray, samplerate: int, newsamplerate: int) -> np.ndarray:
    try:
        import soxr
        out = soxr.resample(samples, samplerate, newsamplerate)
        return out
    except ImportError as e:
        raise ImportError(f"Could not import soxr package, install it via `pip install soxr` "
                          f"original error: {e}")


def resample(samples: np.ndarray, samplerate: int, newsamplerate: int, backend=''
             ) -> np.ndarray:
    """
    Resample samples from sr to newsamplerate

    Args:
        samples: the samples to resample, a float array (normally within the
            range -1 to 1). Can be multichannel
        samplerate: the original sr
        newsamplerate: the new sr
        backend: one of 'soxr', 'resampy' or 'samplerate'. Not all backends are installed
            by default (currently only soxr is installed as a dependency)

    Returns:
        the resampled samples. The shape of the output will be the same as
        the shape of the input
    """
    if not backend:
        backend = _getBackend()
    if not backend:
        raise RuntimeError("no backend found for resampling. Try installing one of the"
                           "following packages:\n"
                           "    * soxr (pip install soxr)\n"
                           "    * resampy (pip install resampy) \n"
                           "    * samplerate (pip install samplerate) \n")
    func = {
        'soxr': _resampleSoxr,
        'resampy': _resampleResampy,
        'samplerate': _resampleSamplerate
    }.get(backend)
    if func is None:
        raise ValueError(f"backend {backend} not known, it should be one of 'soxr', 'resampy', 'samplerate'")        
    return func(samples, samplerate, newsamplerate)
    