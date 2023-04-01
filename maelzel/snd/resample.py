"""
Multiple backends to perform resampling on numpy arrays
"""
from __future__ import annotations
import os
import numpy as np


def _getBackend() -> str:
    try:
        import resampy
        return "resampy"
    except ImportError:
        pass
    try:
        import samplerate
        return "sr"
    except ImportError:
        pass
    return ''


def _resampleViaCli(samples: np.ndarray, samplerate:int, newsamplerate:int
                    ) -> np.ndarray:
    import shutil
    sndfile_resample = shutil.which("sndfile-resample")
    if sndfile_resample is None:
        raise RuntimeError("sndfile-resample is needed")

    import subprocess
    import tempfile
    from sndfileio import sndwrite, sndread

    tmpfile = tempfile.mktemp(suffix='.wav')
    tmpfile2 = "%s-%d.wav" % (tmpfile, newsamplerate)
    sndwrite(outfile=tmpfile, samples=samples, sr=samplerate, encoding='float32')
    subprocess.call([sndfile_resample, "-to", str(newsamplerate), tmpfile, tmpfile2])
    s1, s1_sr = sndread(tmpfile2)
    assert s1_sr == newsamplerate
    os.remove(tmpfile)
    return s1


def _resampleViaResampy(samples: np.ndarray, samplerate:int, newsamplerate:int
                        ) -> np.ndarray:
    try:
        import resampy
    except ImportError:
        raise RuntimeError("resampy is needed, install it via `pip install resampy`"
                           " (see https://github.com/bmcfee/resampy)")
    return resampy.resample(samples, samplerate, newsamplerate)


def _resampleViaSamplerate(samples: np.ndarray, samplerate:int, newsamplerate:int
                           ) -> np.ndarray:
    import samplerate as sampleratecffi
    ratio = samplerate / newsamplerate
    return sampleratecffi.resample(samples, ratio, 'sinc_best')


def resample(samples: np.ndarray, samplerate:int, newsamplerate:int, backend=''
             ) -> np.ndarray:
    """
    Resample samples from sr to newsamplerate

    Args:
        samples: the samples to resample, a float array (normally within the
            range -1 to 1). Can be multichannel
        samplerate: the original sr
        newsamplerate: the new sr
        backend: one of 'resampy', 'sr' or 'cli'

    Returns:
        the resampled samples. The shape of the output will be the same as
        the shape of the input
    """
    if not backend:
        backend = _getBackend()
    if not backend:
        raise RuntimeError("no backend found for resampling. Try installing one of the"
                           "following packages:\n"
                           "    * resampy (pip install resampy) \n"
                           "    * python-sr (pip install sr) \n")
    if backend == "sr":
        return _resampleViaSamplerate(samples, samplerate, newsamplerate)
    elif backend == "resampy":
        return _resampleViaResampy(samples, samplerate, newsamplerate)
    elif backend == 'cli':
        return _resampleViaCli(samples, samplerate=samplerate, newsamplerate=newsamplerate)
    raise RuntimeError("No backend is available")