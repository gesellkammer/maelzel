"""
Similar to pitchtools, but on numpy arrays
"""
from __future__ import annotations
import numpy as np
import pitchtools as pt


import sys
_EPS = sys.float_info.epsilon


def f2m(freqs: np.ndarray, out: np.ndarray = None, a4: float = None) -> np.ndarray:
    """
    Vectorized version of pitchtools.f2m

    Args:
        freqs: an array of frequencies
        out: if given, put the result in out
        a4: the reference frequency. If not given use the global setting
                (see set_reference_freq)

    Returns:
        the midi pitch as a numpy array


    Formula::

        if freq < 9:
            return 0
        return 12.0 * log(freq/A4, 2) + 69.0

    """
    freqs = np.asarray(freqs, dtype=float)
    a4 = a4 or pt.get_reference_freq()
    if out is None:
        return 12.0 * np.log2(freqs/a4) + 69.0
    x = freqs/a4
    np.log2(x, out=x)
    x *= 12.0
    x += 69.0
    return x


def m2f(midinotes: np.ndarray, out: np.ndarray | None = None, a4: float = None) -> np.ndarray:
    """
    Vectorized version of pitchtools.m2f

    Args:
        midinotes: an array of midinotes
        out: if given, put the result here
        a4: the reference frequency. If not given use the global setting
            (see set_reference_freq)

    Returns:
        the frequencies as a numpy array
    """
    a4 = a4 or pt.get_reference_freq()
    midinotes = np.asarray(midinotes, dtype=float)
    out = np.subtract(midinotes, 69, out=out)
    out /= 12.
    out = np.power(2.0, out, out)
    out *= a4
    return out


def db2amp(db: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """
    Vectorized version of pitchtools.db2amp

    Args:
        db: a np array of db values
        out: if given, put the result here

    Returns:
        the amplitudes as a numpy array
    """
    # amp = 10.0**(0.05*db)
    out = np.multiply(db, 0.05, out=out)
    out = np.power(10, out, out=out)
    return out


def amp2db(amp: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    """
    Vectorized version of pitchtools.amp2db

    Args:
        amp: a np array of db values
        out: if given, put the result here

    Returns:
        the dB values as a numpy array

    """
    X = np.maximum(amp, _EPS, out=out)
    X = np.log10(X, out=X)
    X *= 20
    return X


def logfreqs(notemin=0.0, notemax=139.0, notedelta=1.0) -> np.ndarray:
    """
    Return a list of frequencies corresponding to the pitch range given

    Args:
        notemin: start midi note
        notemax: end midi note
        notedelta: the delta to use between values

    Returns:
        the frequencies between notemin and notemax with the given delta

    Example
    =======

        # generate a list of frequencies of all audible semitones
        >>> logfreqs(0, 139, notedelta=1)

        # generate a list of frequencies of instrumental 1/4 tones
        >>> logfreqs(n2m("A0"), n2m("C8"), 0.5)
    """
    return m2f(np.arange(notemin, notemax + notedelta, notedelta))


def pianofreqs(start='A0', stop='C8') -> np.ndarray:
    """
    Generate an array of the frequencies representing all the piano keys
    """
    n0 = int(pt.n2m(start))
    n1 = int(pt.n2m(stop)) + 1
    return m2f(np.arange(n0, n1, 1))


def ratio2interval(ratios: np.ndarray) -> np.ndarray:
    """
    Vectorized version of pitchtools.r2i
    """
    out = np.log(ratios, 2)
    np.multiply(12, out, out=out)
    return out


def interval2ratio(intervals: np.ndarray) -> np.ndarray:
    """
    Vectorized version of pitchtools.i2r
    """
    out = intervals / 12.
    np.float_power(2, out, out=out)
    return out
