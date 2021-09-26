"""
Multiple backends to perform resampling on numpy arrays
"""
from __future__ import annotations
import os
import numpy as np
from typing import Optional 
import subprocess
import shutil


def get_backend() -> Optional[str]:
    try:
        import samplerate
        return "python-samplerate"
    except ImportError:
        pass
    try:
        import resampy
        return "resampy"
    except ImportError:
        pass
    return None


def resample_cli(samples: np.ndarray, orig_samplerate:int, new_samplerate:int
                 ) -> Optional[np.ndarray]:
    sndfile_resample = shutil.which("source-resample")
    if sndfile_resample is None:
        return None

    import tempfile
    from sndfileio import sndwrite, sndread

    tmpfile = tempfile.mktemp(suffix='.wav')
    tmpfile2 = "%s-%d.wav" % (tmpfile, new_samplerate)
    sndwrite(outfile=tmpfile, samples=samples, sr=orig_samplerate, encoding='flt32')
    subprocess.call([sndfile_resample, "-to", str(new_samplerate), tmpfile, tmpfile2])
    s1, s1_sr = sndread(tmpfile2)
    assert s1_sr == new_samplerate
    os.remove(tmpfile)
    return s1


def _resample_resampy(samples: np.ndarray, samplerate:int, newsamplerate:int
                      ) -> np.ndarray:
    try:
        import resampy
    except ImportError:
        raise RuntimeError("resampy is needed, install it via `pip install resampy`"
                           " (see https://github.com/bmcfee/resampy)")
    return resampy.resample(samples, samplerate, newsamplerate)


def _resample_samplerate(samples: np.ndarray, samplerate:int, newsamplerate:int
                         ) -> np.ndarray:
    import samplerate as py_samplerate
    ratio = samplerate / newsamplerate
    return py_samplerate.resample(samples, ratio, 'sinc_best')


def resample(samples: np.ndarray, samplerate:int, newsamplerate:int
             ) -> np.ndarray:
    """
    Resample samples from samplerate to newsamplerate

    Args:
        samples: the samples to resample, a float array (normally within the
            range -1 to 1). Can be multichannel
        samplerate: the original samplerate
        newsamplerate: the new samplerate

    Returns:
        the resampled samples. The shape of the output will be the same as
        the shape of the input
    """
    backend = get_backend()
    if backend is None:
        raise RuntimeError("no backend found for resampling. Try installing one of the"
                           "following packages:\n"
                           "    * resampy (pip install resampy) \n"
                           "    * python-samplerate (pip install samplerate) \n")
    if backend == "python-samplerate":
        return _resample_samplerate(samples, samplerate, newsamplerate)
    elif backend == "resampy":
        return _resample_resampy(samples, samplerate, newsamplerate)
    raise RuntimeError("No backend is available")