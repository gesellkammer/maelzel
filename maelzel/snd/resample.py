"""
Multiple backends to perform resampling on numpy arrays
"""

import os
import numpy as np
from typing import Optional as Opt
import subprocess


def _resample_cli(samples: np.ndarray, orig_samplerate:int, new_samplerate:int) -> Opt[np.ndarray]:
    try:
        sndfile_resample = subprocess.check_output("which sndfile-resample", shell=True)
    except subprocess.CalledProcessError:
        return None
    import tempfile
    tmpfile = tempfile.mktemp(suffix='.wav')
    tmpfile2 = "%s-%d.wav" % (tmpfile, new_samplerate)
    from sndfileio import sndwrite, sndread
    sndwrite(samples, orig_samplerate, tmpfile, 'flt32')
    subprocess.call([sndfile_resample, "-to", str(new_samplerate), tmpfile, tmpfile2])
    s1, s1_sr = sndread(tmpfile2)
    assert s1_sr == new_samplerate
    os.remove(tmpfile)
    return s1


def _resample_scikits(samples, orig_samplerate, new_samplerate):
    # type: (np.ndarray, int, int) -> Opt[np.ndarray]
    try:
        import scikits.samplerate
        return scikits.samplerate.resample(
            samples, new_samplerate / orig_samplerate, 'sinc_best')
    except ImportError:
        return None


def _resample_resampy(samples, samplerate, newsamplerate):
    # type: (np.ndarray, int, int) -> Opt[np.ndarray]
    try:
        import resampy
        return resampy.resample(samples, samplerate, newsamplerate)
    except ImportError:
        return None


def resample(samples, samplerate, newsamplerate):
    # type: (np.ndarray, int, int) -> np.ndarray
    """
    Resample samples from samplerate to newsamplerate
    """
    backends = [
        ('resampy', _resample_resampy),
        ('scikits', _resample_scikits),
        ('sndfile-resample', _resample_cli)
    ]
    for backendname, func in backends:
        out = func(samples, samplerate, newsamplerate)
        if out is not None:
            return out
    raise RuntimeError("no backend found for resampling. Backends tested: %s" %
                       [name for name, func in backends])
