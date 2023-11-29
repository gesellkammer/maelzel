"""
rosita - a minimal version of librosa

https://github.com/librosa/librosa

This is an attempt to provide some of the great functionality
implemented in librosa without the dependency of numba, which
forces other dependency problems regarding numpy's version, etc.

Original license of librosa:

## ISC License

Copyright (c) 2013--2017, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
from __future__ import annotations
import numpy as np
from numpy import fft
from scipy.signal import get_window
import scipy
import warnings
from numpy.lib.stride_tricks import as_strided
import numpyx
import pitchtools as pt
import matplotlib.ticker as mplticker
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.collections import QuadMesh

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10


class ParameterError(ValueError):
    """Exception class for mal-formed inputs"""
    pass


def fix_frames(frames, *, x_min=0, x_max=None, pad=True):
    """Fix a list of frames to lie within [x_min, x_max]

    Examples
    --------
    >>> # Generate a list of frame indices
    >>> frames = np.arange(0, 1000.0, 50)
    >>> frames
    array([   0.,   50.,  100.,  150.,  200.,  250.,  300.,  350.,
            400.,  450.,  500.,  550.,  600.,  650.,  700.,  750.,
            800.,  850.,  900.,  950.])
    >>> # Clip to span at most 250
    >>> librosa.util.fix_frames(frames, x_max=250)
    array([  0,  50, 100, 150, 200, 250])
    >>> # Or pad to span up to 2500
    >>> librosa.util.fix_frames(frames, x_max=2500)
    array([   0,   50,  100,  150,  200,  250,  300,  350,  400,
            450,  500,  550,  600,  650,  700,  750,  800,  850,
            900,  950, 2500])
    >>> librosa.util.fix_frames(frames, x_max=2500, pad=False)
    array([  0,  50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
           550, 600, 650, 700, 750, 800, 850, 900, 950])

    >>> # Or starting away from zero
    >>> frames = np.arange(200, 500, 33)
    >>> frames
    array([200, 233, 266, 299, 332, 365, 398, 431, 464, 497])
    >>> librosa.util.fix_frames(frames)
    array([  0, 200, 233, 266, 299, 332, 365, 398, 431, 464, 497])
    >>> librosa.util.fix_frames(frames, x_max=500)
    array([  0, 200, 233, 266, 299, 332, 365, 398, 431, 464, 497,
           500])

    Parameters
    ----------
    frames : np.ndarray [shape=(n_frames,)]
        List of non-negative frame indices
    x_min : int >= 0 or None
        Minimum allowed frame index
    x_max : int >= 0 or None
        Maximum allowed frame index
    pad : boolean
        If ``True``, then ``frames`` is expanded to span the full range
        ``[x_min, x_max]``

    Returns
    -------
    fixed_frames : np.ndarray [shape=(n_fixed_frames,), dtype=int]
        Fixed frame indices, flattened and sorted

    Raises
    ------
    ParameterError
        If ``frames`` contains negative values
    """

    frames = np.asarray(frames)

    if np.any(frames < 0):
        raise ParameterError("Negative frame index detected")

    if pad and (x_min is not None or x_max is not None):
        frames = np.clip(frames, x_min, x_max)

    if pad:
        pad_data = []
        if x_min is not None:
            pad_data.append(x_min)
        if x_max is not None:
            pad_data.append(x_max)
        frames = np.concatenate((pad_data, frames))

    if x_min is not None:
        frames = frames[frames >= x_min]

    if x_max is not None:
        frames = frames[frames <= x_max]

    return np.unique(frames).astype(int)


def frames_to_time(frames, *, sr=22050, hop_length=512, n_fft=None):
    """Converts frame counts to time (seconds).

    Parameters
    ----------
    frames : np.ndarray [shape=(n,)]
        frame index or vector of frame indices
    sr : number > 0 [scalar]
        audio sampling rate
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        time (in seconds) of each given frame number::

            times[i] = frames[i] * hop_length / sr

    See Also
    --------
    time_to_frames : convert time values to frame indices
    frames_to_samples : convert frame indices to sample indices

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> beat_times = librosa.frames_to_time(beats, sr=sr)
    """

    samples = frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)
    return samples_to_time(samples, sr=sr)


def samples_to_time(samples, sr):
    return np.asanyarray(samples) / float(sr)


def frames_to_samples(frames, *, hop_length=512, n_fft=None):
    """Converts frame indices to audio sample indices.

    Parameters
    ----------
    frames : number or np.ndarray [shape=(n,)]
        frame index or vector of frame indices
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : number or np.ndarray
        time (in samples) of each given frame number::

            times[i] = frames[i] * hop_length

    See Also
    --------
    frames_to_time : convert frame indices to time values
    samples_to_frames : convert sample indices to frame indices

    Examples
    --------
    >>> import sndfileio
    >>> y, sr = sndfileio.sndread("sound.wav")
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> beat_samples = librosa.frames_to_samples(beats)
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def valid_int(x, *, cast=None):
    """Ensure that an input value is integer-typed.
    This is primarily useful for ensuring integrable-valued
    array indices.

    Parameters
    ----------
    x : number
        A scalar value to be cast to int
    cast : function [optional]
        A function to modify ``x`` before casting.
        Default: `np.floor`

    Returns
    -------
    x_int : int
        ``x_int = int(cast(x))``

    Raises
    ------
    ParameterError
        If ``cast`` is provided and is not callable.
    """

    if cast is None:
        cast = np.floor

    if not callable(cast):
        raise ParameterError("cast parameter must be callable")

    return int(cast(x))


def peak_pick(x, *, pre_max, post_max, pre_avg, post_avg, delta, wait):
    """Uses a flexible heuristic to pick peaks in a signal.

    A sample n is selected as an peak if the corresponding ``x[n]``
    fulfills the following three conditions:

    1. ``x[n] == max(x[n - pre_max:n + post_max])``
    2. ``x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta``
    3. ``n - previous_n > wait``

    where ``previous_n`` is the last sample picked as a peak (greedily).

    This implementation is based on [#]_ and [#]_.

    .. [#] Boeck, Sebastian, Florian Krebs, and Markus Schedl.
        "Evaluating the Online Capabilities of Onset Detection Methods." ISMIR.
        2012.

    .. [#] https://github.com/CPJKU/onset_detection/blob/master/onset_program.py

    Parameters
    ----------
    x : np.ndarray [shape=(n,)]
        input signal to peak picks from
    pre_max : int >= 0 [scalar]
        number of samples before ``n`` over which max is computed
    post_max : int >= 1 [scalar]
        number of samples after ``n`` over which max is computed
    pre_avg : int >= 0 [scalar]
        number of samples before ``n`` over which mean is computed
    post_avg : int >= 1 [scalar]
        number of samples after ``n`` over which mean is computed
    delta : float >= 0 [scalar]
        threshold offset for mean
    wait : int >= 0 [scalar]
        number of samples to wait after picking a peak

    Returns
    -------
    peaks : np.ndarray [shape=(n_peaks,), dtype=int]
        indices of peaks in ``x``

    Raises
    ------
    ParameterError
        If any input lies outside its defined range

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          hop_length=512,
    ...                                          aggregate=np.median)
    >>> peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    >>> peaks
    array([  3,  27,  40,  61,  72,  88, 103])

    >>> import matplotlib.pyplot as plt
    >>> times = librosa.times_like(onset_env, sr=sr, hop_length=512)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> D = np.abs(librosa.stft(y))
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[0].plot(times, onset_env, alpha=0.8, label='Onset strength')
    >>> ax[0].vlines(times[peaks], 0,
    ...              onset_env.max(), color='r', alpha=0.8,
    ...              label='Selected peaks')
    >>> ax[0].legend(frameon=True, framealpha=0.8)
    >>> ax[0].label_outer()
    """

    if pre_max < 0:
        raise ParameterError("pre_max must be non-negative")
    if pre_avg < 0:
        raise ParameterError("pre_avg must be non-negative")
    if delta < 0:
        raise ParameterError("delta must be non-negative")
    if wait < 0:
        raise ParameterError("wait must be non-negative")

    if post_max <= 0:
        raise ParameterError("post_max must be positive")

    if post_avg <= 0:
        raise ParameterError("post_avg must be positive")

    if x.ndim != 1:
        raise ParameterError("input array must be one-dimensional")

    # Ensure valid index types
    pre_max = valid_int(pre_max, cast=np.ceil)
    post_max = valid_int(post_max, cast=np.ceil)
    pre_avg = valid_int(pre_avg, cast=np.ceil)
    post_avg = valid_int(post_avg, cast=np.ceil)
    wait = valid_int(wait, cast=np.ceil)

    # Get the maximum of the signal over a sliding window
    max_length = pre_max + post_max
    max_origin = np.ceil(0.5 * (pre_max - post_max))
    # Using mode='constant' and cval=x.min() effectively truncates
    # the sliding window at the boundaries
    mov_max = scipy.ndimage.filters.maximum_filter1d(
        x, int(max_length), mode="constant", origin=int(max_origin), cval=x.min()
    )

    # Get the mean of the signal over a sliding window
    avg_length = pre_avg + post_avg
    avg_origin = np.ceil(0.5 * (pre_avg - post_avg))
    # Here, there is no mode which results in the behavior we want,
    # so we'll correct below.
    mov_avg = scipy.ndimage.filters.uniform_filter1d(
        x, int(avg_length), mode="nearest", origin=int(avg_origin)
    )

    # Correct sliding average at the beginning
    n = 0
    # Only need to correct in the range where the window needs to be truncated
    while n - pre_avg < 0 and n < x.shape[0]:
        # This just explicitly does mean(x[n - pre_avg:n + post_avg])
        # with truncation
        start = n - pre_avg
        start = start if start > 0 else 0
        mov_avg[n] = np.mean(x[start : n + post_avg])
        n += 1
    # Correct sliding average at the end
    n = x.shape[0] - post_avg
    # When post_avg > x.shape[0] (weird case), reset to 0
    n = n if n > 0 else 0
    while n < x.shape[0]:
        start = n - pre_avg
        start = start if start > 0 else 0
        mov_avg[n] = np.mean(x[start : n + post_avg])
        n += 1

    # First mask out all entries not equal to the local max
    detections = x * (x == mov_max)

    # Then mask out all entries less than the thresholded average
    detections = detections * (detections >= (mov_avg + delta))

    # Initialize peaks array, to be filled greedily
    peaks = []

    # Remove onsets which are close together in time
    last_onset = -np.inf

    for i in np.nonzero(detections)[0]:
        # Only report an onset if the "wait" samples was reported
        if i > last_onset + wait:
            peaks.append(i)
            # Save last reported onset
            last_onset = i

    return np.array(peaks)


def tiny(x):
    """Compute the tiny-value corresponding to an input's data type.

    This is the smallest "usable" number representable in ``x.dtype``
    (e.g., float32).

    This is primarily useful for determining a threshold for
    numerical underflow in division or multiplication operations.

    Parameters
    ----------
    x : number or np.ndarray
        The array to compute the tiny-value for.
        All that matters here is ``x.dtype``

    Returns
    -------
    tiny_value : float
        The smallest positive usable number for the type of ``x``.
        If ``x`` is integer-typed, then the tiny value for ``np.float32``
        is returned instead.

    See Also
    --------
    numpy.finfo

    Examples
    --------
    For a standard double-precision floating point number:

    >>> librosa.util.tiny(1.0)
    2.2250738585072014e-308

    Or explicitly as double-precision

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float64))
    2.2250738585072014e-308

    Or complex numbers

    >>> librosa.util.tiny(1j)
    2.2250738585072014e-308

    Single-precision floating point:

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float32))
    1.1754944e-38

    Integer

    >>> librosa.util.tiny(5)
    1.1754944e-38
    """

    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny


def normalize(S, *, norm=np.inf, axis=0, threshold=None, fill=None):
    """Normalize an array along a chosen axis.

    Given a norm (described below) and a target axis, the input
    array is scaled so that::

        norm(S, axis=axis) == 1

    For example, ``axis=0`` normalizes each column of a 2-d array
    by aggregating over the rows (0-axis).
    Similarly, ``axis=1`` normalizes each row of a 2-d array.

    This function also supports thresholding small-norm slices:
    any slice (i.e., row or column) with norm below a specified
    ``threshold`` can be left un-normalized, set to all-zeros, or
    filled with uniform non-zero values that normalize to 1.

    Note: the semantics of this function differ from
    `scipy.linalg.norm` in two ways: multi-dimensional arrays
    are supported, but matrix-norms are not.

    Parameters
    ----------
    S : np.ndarray
        The array to normalize

    norm : {np.inf, -np.inf, 0, float > 0, None}
        - `np.inf`  : maximum absolute value
        - `-np.inf` : minimum absolute value
        - `0`    : number of non-zeros (the support)
        - float  : corresponding l_p norm
            See `scipy.linalg.norm` for details.
        - None : no normalization is performed

    axis : int [scalar]
        Axis along which to compute the norm.

    threshold : number > 0 [optional]
        Only the columns (or rows) with norm at least ``threshold`` are
        normalized.

        By default, the threshold is determined from
        the numerical precision of ``S.dtype``.

    fill : None or bool
        If None, then columns (or rows) with norm below ``threshold``
        are left as is.

        If False, then columns (rows) with norm below ``threshold``
        are set to 0.

        If True, then columns (rows) with norm below ``threshold``
        are filled uniformly such that the corresponding norm is 1.

        .. note:: ``fill=True`` is incompatible with ``norm=0`` because
            no uniform vector exists with l0 "norm" equal to 1.

    Returns
    -------
    S_norm : np.ndarray [shape=S.shape]
        Normalized array

    Raises
    ------
    ValueError
        If ``norm`` is not among the valid types defined above

        If ``S`` is not finite

        If ``fill=True`` and ``norm=0``

    See Also
    --------
    scipy.linalg.norm

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    >>> # Construct an example matrix
    >>> S = np.vander(np.arange(-2.0, 2.0))
    >>> S
    array([[-8.,  4., -2.,  1.],
           [-1.,  1., -1.,  1.],
           [ 0.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  1.]])
    >>> # Max (l-infinity)-normalize the columns
    >>> librosa.util.normalize(S)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # Max (l-infinity)-normalize the rows
    >>> librosa.util.normalize(S, axis=1)
    array([[-1.   ,  0.5  , -0.25 ,  0.125],
           [-1.   ,  1.   , -1.   ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 1.   ,  1.   ,  1.   ,  1.   ]])
    >>> # l1-normalize the columns
    >>> librosa.util.normalize(S, norm=1)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    >>> # l2-normalize the columns
    >>> librosa.util.normalize(S, norm=2)
    array([[-0.985,  0.943, -0.816,  0.5  ],
           [-0.123,  0.236, -0.408,  0.5  ],
           [ 0.   ,  0.   ,  0.   ,  0.5  ],
           [ 0.123,  0.236,  0.408,  0.5  ]])

    >>> # Thresholding and filling
    >>> S[:, -1] = 1e-308
    >>> S
    array([[ -8.000e+000,   4.000e+000,  -2.000e+000,
              1.000e-308],
           [ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.000e+000,   1.000e+000,   1.000e+000,
              1.000e-308]])

    >>> # By default, small-norm columns are left untouched
    >>> librosa.util.normalize(S)
    array([[ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [ -1.250e-001,   2.500e-001,  -5.000e-001,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.250e-001,   2.500e-001,   5.000e-001,
              1.000e-308]])
    >>> # Small-norm columns can be zeroed out
    >>> librosa.util.normalize(S, fill=False)
    array([[-1.   ,  1.   , -1.   ,  0.   ],
           [-0.125,  0.25 , -0.5  ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.125,  0.25 ,  0.5  ,  0.   ]])
    >>> # Or set to constant with unit-norm
    >>> librosa.util.normalize(S, fill=True)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # With an l1 norm instead of max-norm
    >>> librosa.util.normalize(S, norm=1, fill=True)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    """

    # Avoid division-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ValueError(
            "threshold={} must be strictly " "positive".format(threshold)
        )

    if fill not in [None, False, True]:
        raise ValueError("fill={} must be None or boolean".format(fill))

    if not np.all(np.isfinite(S)):
        raise ValueError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ValueError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    elif norm is None:
        return S

    else:
        raise ValueError("Unsupported norm: {}".format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def dtype_r2c(d, *, default=np.complex64):
    """Find the complex numpy dtype corresponding to a real dtype.

    This is used to maintain numerical precision and memory footprint
    when constructing complex arrays from real-valued data
    (e.g. in a Fourier transform).

    A `float32` (single-precision) type maps to `complex64`,
    while a `float64` (double-precision) maps to `complex128`.

    Parameters
    ----------
    d : np.dtype
        The real-valued dtype to convert to complex.
        If ``d`` is a complex type already, it will be returned.
    default : np.dtype, optional
        The default complex target type, if ``d`` does not match a
        known dtype

    Returns
    -------
    d_c : np.dtype
        The complex dtype

    See Also
    --------
    dtype_c2r
    numpy.dtype

    Examples
    --------
    >>> librosa.util.dtype_r2c(np.float32)
    dtype('complex64')

    >>> librosa.util.dtype_r2c(np.int16)
    dtype('complex64')

    >>> librosa.util.dtype_r2c(np.complex128)
    dtype('complex128')
    """
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(float): np.dtype(complex).type,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))



def frame(x, *, frame_length, hop_length, axis=-1, writeable=False, subok=False):
    """Slice a data array into (overlapping) frames.

    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the same input data.

    For example, a one-dimensional input ``x = [0, 1, 2, 3, 4, 5, 6]``
    can be framed with frame length 3 and hop length 2 in two ways.
    The first (``axis=-1``), results in the array ``x_frames``::

        [[0, 2, 4],
         [1, 3, 5],
         [2, 4, 6]]

    where each column ``x_frames[:, i]`` contains a contiguous slice of
    the input ``x[i * hop_length : i * hop_length + frame_length]``.

    The second way (``axis=0``) results in the array ``x_frames``::

        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6]]

    where each row ``x_frames[i]`` contains a contiguous slice of the input.

    This generalizes to higher dimensional inputs, as shown in the examples below.
    In general, the framing operation increments by 1 the number of dimensions,
    adding a new "frame axis" either before the framing axis (if ``axis < 0``)
    or after the framing axis (if ``axis >= 0``).

    Parameters
    ----------
    x : np.ndarray
        Array to frame
    frame_length : int > 0 [scalar]
        Length of the frame
    hop_length : int > 0 [scalar]
        Number of steps to advance between frames
    axis : int
        The axis along which to frame.
    writeable : bool
        If ``True``, then the framed view of ``x`` is read-only.
        If ``False``, then the framed view is read-write.  Note that writing to the framed view
        will also write to the input array ``x`` in this case.
    subok : bool
        If True, sub-classes will be passed-through, otherwise the returned array will be
        forced to be a base-class array (default).

    Returns
    -------
    x_frames : np.ndarray [shape=(..., frame_length, N_FRAMES, ...)]
        A framed view of ``x``, for example with ``axis=-1`` (framing on the last dimension)::

            x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]

        If ``axis=0`` (framing on the first dimension), then::

            x_frames[j] = x[j * hop_length : j * hop_length + frame_length]

    Raises
    ------
    ValueError
        If ``x.shape[axis] < frame_length``, there is not enough data to fill one frame.

        If ``hop_length < 1``, frames cannot advance.

    See Also
    --------
    numpy.lib.stride_tricks.as_strided

    Examples
    --------
    Extract 2048-sample frames from monophonic signal with a hop of 64 samples per frame

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    >>> frames
    array([[-1.407e-03, -2.604e-02, ..., -1.795e-05, -8.108e-06],
           [-4.461e-04, -3.721e-02, ..., -1.573e-05, -1.652e-05],
           ...,
           [ 7.960e-02, -2.335e-01, ..., -6.815e-06,  1.266e-05],
           [ 9.568e-02, -1.252e-01, ...,  7.397e-06, -1.921e-05]],
          dtype=float32)
    >>> y.shape
    (117601,)

    >>> frames.shape
    (2048, 1806)

    Or frame along the first axis instead of the last:

    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64, axis=0)
    >>> frames.shape
    (1806, 2048)

    Frame a stereo signal:

    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    (2, 2048, 1806)

    Carve an STFT into fixed-length patches of 32 frames with 50% overlap

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S.shape
    (1025, 230)
    >>> S_patch = librosa.util.frame(S, frame_length=32, hop_length=16)
    >>> S_patch.shape
    (1025, 32, 13)
    >>> # The first patch contains the first 32 frames of S
    >>> np.allclose(S_patch[:, :, 0], S[:, :32])
    True
    >>> # The second patch contains frames 16 to 16+32=48, and so on
    >>> np.allclose(S_patch[:, :, 1], S[:, 16:48])
    True
    """

    # This implementation is derived from numpy.lib.stride_tricks.sliding_window_view (1.20.0)
    # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html

    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ValueError("Input is too short (n={:d})"
                         " for frame_length={:d}".format(x.shape[axis], frame_length))

    if hop_length < 1:
        raise ValueError(f"Invalid hop_length: {hop_length:d}")

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable)

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


def hz_to_mel(frequencies, *, htk=False):
    """Convert Hz to Mels

    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])

    Parameters
    ----------
    frequencies : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk : bool
        use HTK formula instead of Slaney

    Returns
    -------
    mels : number or np.ndarray [shape=(n,)]
        input frequencies in Mels

    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, *, htk=False):
    """Convert mel bin numbers to frequencies

    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.

    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    Parameters
    ----------
    mels : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk : bool
        use HTK formula instead of Slaney

    Returns
    -------
    frequencies : np.ndarray [shape=(n,)]
        input mels in Hz

    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def fft_frequencies(*, sr: float = 22050, n_fft: int = 2048):
    """
    Alternative implementation of `np.fft.fftfreq`
    """
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def mel_frequencies(n_mels=128, *, fmin=0.0, fmax=11025.0, htk=False):
    """Compute an array of acoustic frequencies tuned to the mel scale.

    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.

    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoaoustical experiments, several implementations coexist
    in the audio signal processing literature [#]_. By default, librosa replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [#]_.
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [#]_ (HTK) according to the following formula::

        mel = 2595.0 * np.log10(1.0 + f / 700.0).

    The choice of implementation is determined by the ``htk`` keyword argument: setting
    ``htk=False`` leads to the Auditory toolbox implementation, whereas setting it ``htk=True``
    leads to the HTK implementation.

    .. [#] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.

    .. [#] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.

    .. [#] Young, S., Evermann, G., Gales, M., Hain, MObjT., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.

    See Also
    --------
    hz_to_mel
    mel_to_hz
    librosa.feature.melspectrogram
    librosa.feature.mfcc

    Parameters
    ----------
    n_mels : int > 0 [scalar]
        Number of mel bins.
    fmin : float >= 0 [scalar]
        Minimum frequency (Hz).
    fmax : float >= 0 [scalar]
        Maximum frequency (Hz).
    htk : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of ``n_mels`` frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def pad_center(data, *, size, axis=-1, **kwargs):
    """Pad an array to a target length along a target axis.

    This differs from `np.pad` by centering the data prior to padding,
    analogous to `str.center`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, size=10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, size=7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, size=7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered
    size : int >= len(data) [scalar]
        Length to pad ``data``
    axis : int
        Axis along which to pad and center the data
    **kwargs : additional keyword arguments
        arguments passed to `np.pad`

    Returns
    -------
    data_padded : np.ndarray
        ``data`` centered and padded to length ``size`` along the
        specified axis

    Raises
    ------
    ValueError
        If ``size < data.shape[axis]``

    See Also
    --------
    numpy.pad
    """
    kwargs.setdefault("mode", "constant")
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))
    if lpad < 0:
        raise ValueError(f"Target size ({size:d}) must be at least input size ({n:d})")

    return np.pad(data, lengths, **kwargs)


def expand_to(x, *, ndim, axes):
    """Expand the dimensions of an input array with

    Parameters
    ----------
    x : np.ndarray
        The input array
    ndim : int
        The number of dimensions to expand to.  Must be at least ``x.ndim``
    axes : int or slice
        The target axis or axes to preserve from x.
        All other axes will have length 1.

    Returns
    -------
    x_exp : np.ndarray
        The expanded version of ``x``, satisfying the following:
            ``x_exp[axes] == x``
            ``x_exp.ndim == ndim``

    See Also
    --------
    np.expand_dims

    Examples
    --------
    Expand a 1d array into an (n, 1) shape

    >>> x = np.arange(3)
    >>> librosa.util.expand_to(x, ndim=2, axes=0)
    array([[0],
       [1],
       [2]])

    Expand a 1d array into a (1, n) shape

    >>> librosa.util.expand_to(x, ndim=2, axes=1)
    array([[0, 1, 2]])

    Expand a 2d array into (1, n, m, 1) shape

    >>> x = np.vander(np.arange(3))
    >>> librosa.util.expand_to(x, ndim=4, axes=[1,2]).shape
    (1, 3, 3, 1)
    """

    # Force axes into a tuple

    try:
        axes = tuple(axes)
    except TypeError:
        axes = tuple([axes])

    if len(axes) != x.ndim:
        raise ValueError(f"Shape mismatch between {axes=}and input {x.shape=}")

    if ndim < x.ndim:
        raise ValueError(
            "Cannot expand x.shape={} to fewer dimensions ndim={}".format(x.shape, ndim)
        )

    shape = [1] * ndim
    for i, axi in enumerate(axes):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def stft(y, *, n_fft=2048, hop_length=None,
         win_length=None, window="hann", center=True,
         dtype=None, pad_mode="constant"):
    """
    Short-time Fourier transform (STFT).

    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.

    This function returns a complex-valued matrix D such that

    - ``np.abs(D[..., f, t])`` is the magnitude of frequency bin ``f``
      at frame ``t``, and

    - ``np.angle(D[..., f, t])`` is the phase of frequency bin ``f``
      at frame ``t``.

    The integers ``t`` and ``f`` can be converted to physical units by means
    of the utility functions `frames_to_sample` and `fft_frequencies`.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)], real-valued
        input signal. Multi-channel is supported.

    n_fft : int > 0 [scalar]
        length of the windowed signal after padding with zeros.
        The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
        The default value, ``n_fft=2048`` samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting ``n_fft`` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm.

    hop_length : int > 0 [scalar]
        number of audio samples between adjacent STFT columns.

        Smaller values increase the number of columns in ``D`` without
        affecting the frequency resolution of the STFT.

        If unspecified, defaults to ``win_length // 4`` (see below).

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window`` of length ``win_length``
        and then padded with zeros to match ``n_fft``.

        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time)
        at the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization trade-off and needs to be adjusted
        according to the properties of the input signal ``y``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        Either:

        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        Defaults to a raised cosine window (`'hann'`), which is adequate for
        most applications in audio signal processing.

        .. see also:: `filters.get_window`

    center : boolean
        If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.

        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.

        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of `librosa.frames_to_samples`.
        Note, however, that ``center`` must be set to `False` when analyzing
        signals with `librosa.stream`.

        .. see also:: `librosa.stream`

    dtype : np.dtype, optional
        Complex numeric type for ``D``.  Default is inferred to match the
        precision of the input signal.

    pad_mode : string or function
        If ``center=True``, this argument is passed to `np.pad` for padding
        the edges of the signal ``y``. By default (``pad_mode="constant"``),
        ``y`` is padded on both sides with zeros.
        If ``center=False``,  this argument is ignored.

        .. see also:: `numpy.pad`

    Returns
    -------
    D : np.ndarray [shape=(..., 1 + n_fft/2, n_frames), dtype=dtype]
        Complex-valued matrix of short-term Fourier transform
        coefficients.

    See Also
    --------
    istft : Inverse STFT
    reassigned_spectrogram : Time-frequency reassigned spectrogram

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S
    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],
           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],
           ...,
           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],
           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],
          dtype=float32)

    Use left-aligned frames, instead of centered frames

    >>> S_left = librosa.stft(y, center=False)

    Use a shorter hop length

    >>> D_short = librosa.stft(y, hop_length=64)

    Display a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S,
    ...                                                        ref=np.max),
    ...                                y_axis='log', x_axis='time', ax=ax)
    >>> ax.set_title('Power spectrogram')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            warnings.warn(f"n_fft={n_fft} is too small for input signal of length={y.shape[-1]}",
                          stacklevel=2)

        padding = [(0, 0) for _ in range(y.ndim)]
        padding[-1] = (int(n_fft // 2), int(n_fft // 2))
        y = np.pad(y, padding, mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise ValueError(f"n_fft={n_fft} is too large for input signal of length={y.shape[-1]}")

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)
    shape[-2] = 1 + n_fft // 2
    stft_matrix = np.empty(shape, dtype=dtype, order="F")

    n_columns = MAX_MEM_BLOCK // (np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[-1])
        stft_matrix[..., bl_s:bl_t] = fft.rfft(fft_window * y_frames[..., bl_s:bl_t], axis=-2)
    return stft_matrix


def _spectrogram(
    *,
    y=None,
    S=None,
    n_fft=2048,
    hop_length=512,
    power=1,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="constant",
):
    """Helper function to retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.

    Parameters
    ----------
    y : None or np.ndarray
        If provided, an audio time series

    S : None or np.ndarray
        Spectrogram input, optional

    n_fft : int > 0
        STFT window size

    hop_length : int > 0
        STFT hop length

    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window``.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    S_out : np.ndarray [dtype=np.float]
        - If ``S`` is provided as input, then ``S_out == S``
        - Else, ``S_out = |stft(y, ...)|**power``
    n_fft : int > 0
        - If ``S`` is provided, then ``n_fft`` is inferred from ``S``
        - Else, copied from input
    """

    if S is not None:
        # Infer n_fft from spectrogram shape, but only if it mismatches
        if n_fft // 2 + 1 != S.shape[-2]:
            n_fft = 2 * (S.shape[-2] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = (
            np.abs(
                stft(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    window=window,
                    pad_mode=pad_mode,
                )
            )
            ** power
        )

    return S, n_fft

def mel(*, sr, n_fft, n_mels=128, fmin=0.0, fmax=None,
        htk=False, norm="slaney", dtype=np.float32):
    """Create a Mel filter-bank.

    This produces a linear transformation matrix to project
    FFT bins onto Mel-frequency bins.

    Parameters
    ----------
    sr : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft : int > 0 [scalar]
        number of FFT components

    n_mels : int > 0 [scalar]
        number of Mel bands to generate

    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``

    htk : bool [scalar]
        use HTK formula instead of Slaney

    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization).

        If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
        See `librosa.util.normalize` for a full description of supported norm values
        (including `+-np.inf`).

        Otherwise, leave all the triangles aiming for a peak value of 1.0

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    M : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    See Also
    --------
    librosa.util.normalize

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    >>> melfb = librosa.filters.mel(sr=22050, n_fft=2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])

    Clip the maximum frequency to 8KHz

    >>> librosa.filters.mel(sr=22050, n_fft=2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(melfb, x_axis='linear', ax=ax)
    >>> ax.set(ylabel='Mel filter', title='Mel filter bank')
    >>> fig.colorbar(img, ax=ax)
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    else:
        weights = normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels.",
            stacklevel=2,
        )

    return weights


def melspectrogram(*, y=None, sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="constant",
    power=2.0,
    n_mels=128,
    **kwargs,
):
    """Compute a mel-scaled spectrogram.

    If a spectrogram input ``S`` is provided, then it is mapped directly onto
    the mel basis by ``mel_f.dot(S)``.

    If a time-series input ``y, sr`` is provided, then its magnitude spectrogram
    ``S`` is first computed, and then mapped onto the mel scale by
    ``mel_f.dot(S**power)``.

    By default, ``power=2`` operates on a power spectrum.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time-series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)]
        spectrogram

    n_fft : int > 0 [scalar]
        length of the FFT window

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.stft`

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        .. see also:: `librosa.filters.get_window`

    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.

        By default, STFT uses zero padding.

    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power, etc.

    **kwargs : additional keyword arguments
        Mel filter bank parameters.

        See `librosa.filters.mel` for details.

    Returns
    -------
    S : np.ndarray [shape=(..., n_mels, t)]
        Mel spectrogram

    See Also
    --------
    librosa.filters.mel : Mel filter bank construction
    librosa.stft : Short-time Fourier Transform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.melspectrogram(y=y, sr=sr)
    array([[3.837e-06, 1.451e-06, ..., 8.352e-14, 1.296e-11],
           [2.213e-05, 7.866e-06, ..., 8.532e-14, 1.329e-11],
           ...,
           [1.115e-05, 5.192e-06, ..., 3.675e-08, 2.470e-08],
           [6.473e-07, 4.402e-07, ..., 1.794e-08, 2.908e-08]],
          dtype=float32)

    Using a pre-computed power spectrogram would give the same result:

    >>> D = np.abs(librosa.stft(y))**2
    >>> S = librosa.feature.melspectrogram(S=D, sr=sr)

    Display of mel-frequency spectrogram coefficients, with custom
    arguments for mel filterbank construction (default is fmax=sr/2):

    >>> # Passing through arguments to the Mel filters
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                     fmax=8000)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> S_dB = librosa.power_to_db(S, ref=np.max)
    >>> img = librosa.display.specshow(S_dB, x_axis='time',
    ...                          y_axis='mel', sr=sr,
    ...                          fmax=8000, ax=ax)
    >>> fig.colorbar(img, ax=ax, format='%+2.0f dB')
    >>> ax.set(title='Mel-frequency spectrogram')
    """

    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Mel filter
    mel_basis = mel(sr=sr, n_fft=n_fft, n_mels=n_mels, **kwargs)

    return np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)


def onset_detect(y=None, sr=22050, onset_envelope=None, hop_length=512,
                 backtrack=False, energy=None, units="frames",
                 normalize=True, delta=0.07, mingap=0.03, n_fft=1024, **kwargs):
    """Locate note onset events by picking peaks in an onset strength envelope.

    The `peak_pick` parameters were chosen by large-scale hyper-parameter
    optimization over the dataset provided by [#]_.

    .. [#] https://github.com/CPJKU/onset_db

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series, must be monophonic

    sr : number > 0 [scalar]
        sampling rate of ``y``

    onset_envelope : np.ndarray [shape=(m,)]
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length (in samples)

    units : {'frames', 'samples', 'time'}
        The units to encode detected onset events in.
        By default, 'frames' are used.

    backtrack : bool
        If ``True``, detected onset events are backtracked to the nearest
        preceding minimum of ``energy``.

        This is primarily useful when using onsets as slice points for segmentation.

    energy : np.ndarray [shape=(m,)] (optional)
        An energy function to use for backtracking detected onset events.
        If none is provided, then ``onset_envelope`` is used.

    normalize : bool
        If ``True`` (default), normalize the onset envelope to have minimum of 0 and
        maximum of 1 prior to detection.  This is helpful for standardizing the
        parameters of `librosa.util.peak_pick`.

        Otherwise, the onset envelope is left unnormalized.

    mingap : float > 0
        Min. time to wait between onsets

    **kwargs : additional keyword arguments
        Additional parameters for peak picking.

        See `librosa.util.peak_pick` for details.

    Returns
    -------
    onsets : np.ndarray [shape=(n_onsets,)]
        estimated positions of detected onsets, in whichever units
        are specified.  By default, frame indices.

        .. note::
            If no onset strength could be detected, onset_detect returns
            an empty list.

    Raises
    ------
    ValueError
        if neither ``y`` nor ``onsets`` are provided

        or if ``units`` is not one of 'frames', 'samples', or 'time'

    See Also
    --------
    onset_strength : compute onset strength per-frame
    onset_backtrack : backtracking onset events
    librosa.util.peak_pick : pick peaks from a time series

    Examples
    --------
    Get onset times from a signal

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.onset.onset_detect(y=y, sr=sr, units='time')
    array([0.07 , 0.232, 0.395, 0.604, 0.743, 0.929, 1.045, 1.115,
           1.416, 1.672, 1.881, 2.043, 2.206, 2.368, 2.554, 3.019])

    Or use a pre-computed onset envelope

    >>> o_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> times = librosa.times_like(o_env, sr=sr)
    >>> onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> D = np.abs(librosa.stft(y))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          x_axis='time', y_axis='log', ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> ax[1].plot(times, o_env, label='Onset strength')
    >>> ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
    ...            linestyle='--', label='Onsets')
    >>> ax[1].legend()
    """

    # First, get the frame->beat strength preset if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ValueError("y or onset_envelope must be provided")

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)

    # Shift onset envelope up to be non-negative
    # (a common normalization step to make the threshold more consistent)
    if normalize:
        # Normalize onset strength function to [0, 1] range
        onset_envelope = onset_envelope - onset_envelope.min()
        # Max-scale with safe division
        onset_envelope /= np.max(onset_envelope) + tiny(onset_envelope)

    # Do we have any onsets to grab?
    if not onset_envelope.any() or not np.all(np.isfinite(onset_envelope)):
        onsets = np.array([], dtype=int)

    else:
        # These parameter settings found by large-scale search
        kwargs.setdefault("pre_max", 0.03 * sr // hop_length)  # 30ms
        kwargs.setdefault("post_max", 0.00 * sr // hop_length + 1)  # 0ms
        kwargs.setdefault("pre_avg", 0.10 * sr // hop_length)  # 100ms
        kwargs.setdefault("post_avg", 0.10 * sr // hop_length + 1)  # 100ms
        # kwargs.setdefault("wait", 0.03 * sr // hop_length)  # 30ms
        # kwargs.setdefault("delta", 0.07)

        # Peak pick the onset envelope
        wait = mingap * sr // hop_length
        onsets = peak_pick(onset_envelope, delta=delta, wait=wait, **kwargs)

        # Optionally backtrack the events
        if backtrack:
            if energy is None:
                energy = onset_envelope

            onsets = onset_backtrack(onsets, energy)

    if units == "frames":
        pass
    elif units == "samples":
        onsets = frames_to_samples(onsets, hop_length=hop_length)
    elif units == "time":
        onsets = frames_to_time(onsets, hop_length=hop_length, sr=sr)
    else:
        raise ValueError("Invalid unit type: {}".format(units))

    return onsets


def onset_strength(
    *,
    y=None,
    sr=22050,
    S=None,
    lag=1,
    max_size=1,
    ref=None,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    **kwargs):
    """Compute a spectral flux onset strength envelope.

    Onset strength at time ``t`` is determined by::

        mean_f max(0, S[f, t] - ref[f, t - lag])

    where ``ref`` is ``S`` after local max filtering along the frequency
    axis [#]_.

    By default, if a time series ``y`` is provided, S will be the
    log-power Mel spectrogram.

    .. [#] Böck, Sebastian, and Gerhard Widmer.
           "Maximum filter vibrato suppression for onset detection."
           16th International Conference on Digital Audio Effects,
           Maynooth, Ireland. 2013.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time-series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, m)]
        pre-computed (log-power) spectrogram

    lag : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.

    ref : None or np.ndarray [shape=(..., d, m)]
        An optional pre-computed reference spectrum, of the same shape as ``S``.
        If not provided, it will be computed from ``S``.
        If provided, it will override any local max filtering governed by ``max_size``.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.
        This corresponds to using a centered frame analysis in the short-time Fourier
        transform.

    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with ``fmax=sr/2``

    aggregate : function
        Aggregation function to use when combining onsets
        at different frequency bins.

        Default: `np.mean`

    **kwargs : additional keyword arguments
        Additional parameters to onset_strength_multi (like n_fft or hop_length)
        Additional parameters to ``feature()``, if ``S`` is not provided.

    Returns
    -------
    onset_envelope : np.ndarray [shape=(..., m,)]
        vector containing the onset strength envelope.
        If the input contains multiple channels, then onset envelope is computed for each channel.

    Raises
    ------
    ValueError
        if neither ``(y, sr)`` nor ``S`` are provided

        or if ``lag`` or ``max_size`` are not positive integers

    See Also
    --------
    onset_detect
    onset_strength_multi

    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> D = np.abs(librosa.stft(y))
    >>> times = librosa.times_like(D)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()

    Construct a standard onset function

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> ax[1].plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,
    ...            label='Mean (mel)')

    Median aggregation, and custom mel options

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          aggregate=np.median,
    ...                                          fmax=8000, n_mels=256)
    >>> ax[1].plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,
    ...            label='Median (custom mel)')

    Constant-Q spectrogram instead of Mel

    >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    >>> onset_env = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))
    >>> ax[1].plot(times, onset_env / onset_env.max(), alpha=0.8,
    ...          label='Mean (CQT)')
    >>> ax[1].legend()
    >>> ax[1].set(ylabel='Normalized strength', yticks=[])
    """

    if aggregate is False:
        raise ValueError(
            "aggregate={} cannot be False when computing full-spectrum onset strength."
        )

    odf_all = onset_strength_multi(
        y=y,
        sr=sr,
        S=S,
        lag=lag,
        max_size=max_size,
        ref=ref,
        detrend=detrend,
        center=center,
        feature=feature,
        aggregate=aggregate,
        channels=None,
        **kwargs,
    )

    return odf_all[..., 0, :]


def onset_backtrack(events, energy):
    """Backtrack detected onset events to the nearest preceding local
    minimum of an energy function.

    This function can be used to roll back the timing of detected onsets
    from a detected peak amplitude to the preceding minimum.

    This is most useful when using onsets to determine slice points for
    segmentation, as described by [#]_.

    .. [#] Jehan, Tristan.
           "Creating music by listening"
           Doctoral dissertation
           Massachusetts Institute of Technology, 2005.

    Parameters
    ----------
    events : np.ndarray, dtype=int
        List of onset event frame indices, as computed by `onset_detect`
    energy : np.ndarray, shape=(m,)
        An energy function

    Returns
    -------
    events_backtracked : np.ndarray, shape=events.shape
        The input events matched to nearest preceding minima of ``energy``.

    Examples
    --------
    Backtrack the events using the onset envelope

    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr)
    >>> times = librosa.times_like(oenv)
    >>> # Detect events without backtracking
    >>> onset_raw = librosa.onset.onset_detect(onset_envelope=oenv,
    ...                                        backtrack=False)
    >>> onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)

    Backtrack the events using the RMS values

    >>> S = np.abs(librosa.stft(y=y))
    >>> rms = librosa.feature.rms(S=S)
    >>> onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])

    Plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].label_outer()
    >>> ax[1].plot(times, oenv, label='Onset strength')
    >>> ax[1].vlines(librosa.frames_to_time(onset_raw), 0, oenv.max(), label='Raw onsets')
    >>> ax[1].vlines(librosa.frames_to_time(onset_bt), 0, oenv.max(), label='Backtracked', color='r')
    >>> ax[1].legend()
    >>> ax[1].label_outer()
    >>> ax[2].plot(times, rms[0], label='RMS')
    >>> ax[2].vlines(librosa.frames_to_time(onset_bt_rms), 0, rms.max(), label='Backtracked (RMS)', color='r')
    >>> ax[2].legend()
    """

    # Find points where energy is non-increasing
    # all points:  energy[i] <= energy[i-1]
    # tail points: energy[i] < energy[i+1]
    minima = np.flatnonzero((energy[1:-1] <= energy[:-2]) & (energy[1:-1] < energy[2:]))

    # Pad on a 0, just in case we have onsets with no preceding minimum
    # Shift by one to account for slicing in minima detection
    minima = fix_frames(1 + minima, x_min=0)

    # Only match going left from the detected events
    return minima[match_events(events, minima, right=False)]


def match_events(events_from, events_to, left=True, right=True):
    from emlib import numpytools
    return numpytools.nearestindex(a=events_from, grid=events_to, left=left, right=right)


def onset_strength_multi(
    *,
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    lag=1,
    max_size=1,
    ref=None,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    channels=None,
    **kwargs,
):
    """Compute a spectral flux onset strength envelope across multiple channels.

    Onset strength for channel ``i`` at time ``t`` is determined by::

        mean_{f in channels[i]} max(0, S[f, t+1] - S[f, t])

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time-series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, m)]
        pre-computed (log-power) spectrogram

    n_fft : int > 0 [scalar]
        FFT window size for use in ``feature()`` if ``S`` is not provided.

    hop_length : int > 0 [scalar]
        hop length for use in ``feature()`` if ``S`` is not provided.

    lag : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.

    ref : None or np.ndarray [shape=(d, m)]
        An optional pre-computed reference spectrum, of the same shape as ``S``.
        If not provided, it will be computed from ``S``.
        If provided, it will override any local max filtering governed by ``max_size``.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.
        This corresponds to using a centered frame analysis in the short-time Fourier
        transform.

    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with ``fmax=sr/2``

        Must support arguments: ``y, sr, n_fft, hop_length``

    aggregate : function or False
        Aggregation function to use when combining onsets
        at different frequency bins.

        If ``False``, then no aggregation is performed.

        Default: `np.mean`

    channels : list or None
        Array of channel boundaries or slice objects.
        If `None`, then a single channel is generated to span all bands.

    **kwargs : additional keyword arguments
        Additional parameters to ``feature()``, if ``S`` is not provided.

    Returns
    -------
    onset_envelope : np.ndarray [shape=(..., n_channels, m)]
        array containing the onset strength envelope for each specified channel

    Raises
    ------
    ValueError
        if neither ``(y, sr)`` nor ``S`` are provided

    See Also
    --------
    onset_strength

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=5)
    >>> D = np.abs(librosa.stft(y))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img1 = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> fig.colorbar(img1, ax=[ax[0]], format="%+2.f dB")

    Construct a standard onset function over four sub-bands

    >>> onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,
    ...                                                     channels=[0, 32, 64, 96, 128])
    >>> img2 = librosa.display.specshow(onset_subbands, x_axis='time', ax=ax[1])
    >>> ax[1].set(ylabel='Sub-bands', title='Sub-band onset strength')
    >>> fig.colorbar(img2, ax=[ax[1]])
    """

    if feature is None:
        feature = melspectrogram
        kwargs.setdefault("fmax", 0.5 * sr)

    if aggregate is None:
        aggregate = np.mean

    if lag < 1 or not isinstance(lag, (int, np.integer)):
        raise ValueError("lag must be a positive integer")

    if max_size < 1 or not isinstance(max_size, (int, np.integer)):
        raise ValueError("max_size must be a positive integer")

    # First, compute mel spectrogram
    if S is None:
        S = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))

        # Convert to dBs
        S = power_to_db(S)

    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    # Compute the reference spectrogram.
    # Efficiency hack: skip filtering step and pass by reference
    # if max_size will produce a no-op.
    if ref is None:
        if max_size == 1:
            ref = S
        else:
            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=-2)
    elif ref.shape != S.shape:
        raise ValueError(
            "Reference spectrum shape {} must match input spectrum {}".format(
                ref.shape, S.shape
            )
        )

    # Compute difference to the reference, spaced by lag
    onset_env = S[..., lag:] - ref[..., :-lag]

    # Discard negatives (decreasing amplitude)
    onset_env = np.maximum(0.0, onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    if aggregate:
        onset_env = sync(onset_env, channels, aggregate=aggregate, pad=pad, axis=-2)

    # compensate for lag
    pad_width = lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    padding = [(0, 0) for _ in onset_env.shape]
    padding[-1] = (int(pad_width), 0)
    onset_env = np.pad(onset_env, padding, mode="constant")

    # remove the DC component
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)

    # Trim to match the input duration
    if center:
        onset_env = onset_env[..., : S.shape[-1]]

    return onset_env


def sync(data, idx, *, aggregate=None, pad=True, axis=-1):
    """Synchronous aggregation of a multi-dimensional array between boundaries

    .. note::
        In order to ensure total coverage, boundary points may be added
        to ``idx``.

        If synchronizing a feature matrix against beat tracker output, ensure
        that frame index numbers are properly aligned and use the same hop length.

    Parameters
    ----------
    data : np.ndarray
        multi-dimensional array of features
    idx : iterable of ints or slices
        Either an ordered array of boundary indices, or
        an iterable collection of slice objects.
    aggregate : function
        aggregation function (default: `np.mean`)
    pad : boolean
        If `True`, ``idx`` is padded to span the full range ``[0, data.shape[axis]]``
    axis : int
        The axis along which to aggregate data

    Returns
    -------
    data_sync : ndarray
        ``data_sync`` will have the same dimension as ``data``, except that the ``axis``
        coordinate will be reduced according to ``idx``.

        For example, a 2-dimensional ``data`` with ``axis=-1`` should satisfy::

            data_sync[:, i] = aggregate(data[:, idx[i-1]:idx[i]], axis=-1)

    Raises
    ------
    ParameterError
        If the index set is not of consistent type (all slices or all integers)

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    Beat-synchronous CQT spectra

    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    >>> beats = librosa.util.fix_frames(beats)



    Plot the results

    >>> import matplotlib.pyplot as plt
    >>> beat_t = librosa.frames_to_time(beats, sr=sr)
    >>> subbeat_t = librosa.frames_to_time(sub_beats, sr=sr)
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(C,
    ...                                                  ref=np.max),
    ...                          x_axis='time', ax=ax[0])
    >>> ax[0].set(title='CQT power, shape={}'.format(C.shape))
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(C_med,
    ...                                                  ref=np.max),
    ...                          x_coords=beat_t, x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Beat synchronous CQT power, '
    ...                 'shape={}'.format(C_med.shape))
    >>> ax[1].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(C_med_sub,
    ...                                                  ref=np.max),
    ...                          x_coords=subbeat_t, x_axis='time', ax=ax[2])
    >>> ax[2].set(title='Sub-beat synchronous CQT power, '
    ...                 'shape={}'.format(C_med_sub.shape))
    """

    if aggregate is None:
        aggregate = np.mean

    shape = list(data.shape)

    if np.all([isinstance(_, slice) for _ in idx]):
        slices = idx
    elif np.all([np.issubdtype(type(_), np.integer) for _ in idx]):
        slices = index_to_slice(np.asarray(idx), idx_min=0, idx_max=shape[axis], pad=pad)
    else:
        raise ParameterError("Invalid index set: {}".format(idx))

    agg_shape = list(shape)
    agg_shape[axis] = len(slices)

    data_agg = np.empty(agg_shape, order="F" if np.isfortran(data) else "C", dtype=data.dtype)

    idx_in = [slice(None)] * data.ndim
    idx_agg = [slice(None)] * data_agg.ndim

    for (i, segment) in enumerate(slices):
        idx_in[axis] = segment
        idx_agg[axis] = i
        data_agg[tuple(idx_agg)] = aggregate(data[tuple(idx_in)], axis=axis)

    return data_agg


def index_to_slice(idx, *, idx_min=None, idx_max=None, step=None, pad=True):
    """Generate a slice array from an index array.

    Parameters
    ----------
    idx : list-like
        Array of index boundaries
    idx_min, idx_max : None or int
        Minimum and maximum allowed indices
    step : None or int
        Step size for each slice.  If `None`, then the default
        step of 1 is used.
    pad : boolean
        If `True`, pad ``idx`` to span the range ``idx_min:idx_max``.

    Returns
    -------
    slices : list of slice
        ``slices[i] = slice(idx[i], idx[i+1], step)``
        Additional slice objects may be added at the beginning or end,
        depending on whether ``pad==True`` and the supplied values for
        ``idx_min`` and ``idx_max``.

    See Also
    --------
    fix_frames

    Examples
    --------
    >>> # Generate slices from spaced indices
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15))
    [slice(20, 35, None), slice(35, 50, None), slice(50, 65, None), slice(65, 80, None),
     slice(80, 95, None)]
    >>> # Pad to span the range (0, 100)
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15),
    ...                             idx_min=0, idx_max=100)
    [slice(0, 20, None), slice(20, 35, None), slice(35, 50, None), slice(50, 65, None),
     slice(65, 80, None), slice(80, 95, None), slice(95, 100, None)]
    >>> # Use a step of 5 for each slice
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15),
    ...                             idx_min=0, idx_max=100, step=5)
    [slice(0, 20, 5), slice(20, 35, 5), slice(35, 50, 5), slice(50, 65, 5), slice(65, 80, 5),
     slice(80, 95, 5), slice(95, 100, 5)]
    """
    # First, normalize the index set
    idx_fixed = fix_frames(idx, x_min=idx_min, x_max=idx_max, pad=pad)

    # Now convert the indices to slices
    return [slice(start, end, step) for (start, end) in zip(idx_fixed, idx_fixed[1:])]


def power_to_db(S, *, ref=1.0, amin=1e-10, top_db=80.0):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``::

            10 * log10(S / ref)

        Zeros in the output correspond to positions where ``S == ref``.

        If callable, the reference value is computed as ``ref(S)``.

    amin : float > 0 [scalar]
        minimum threshold for ``abs(S)`` and ``ref``

    top_db : float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:
        ``max(10 * log10(S)) - top_db``

    Returns
    -------
    S_db : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``

    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Get a power spectrogram from a waveform ``y``

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809],
           ...,
           [-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.],
           ...,
           [-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.]], dtype=float32)

    Or compare to median power

    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578],
           ...,
           [16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578]], dtype=float32)

    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> imgpow = librosa.display.specshow(S**2, sr=sr, y_axis='log', x_axis='time',
    ...                                   ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> imgdb = librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                                  sr=sr, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-Power spectrogram')
    >>> fig.colorbar(imgpow, ax=ax[0])
    >>> fig.colorbar(imgdb, ax=ax[1], format="%+2.0f dB")
    """
    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def times_like(X, *, sr=22050, hop_length=512, n_fft=None, axis=-1):
    """Return an array of time values to match the time axis from a feature matrix.

    Parameters
    ----------
    X : np.ndarray or scalar
        - If ndarray, X is a feature matrix, e.g. STFT, chromagram, or mel spectrogram.
        - If scalar, X represents the number of frames.
    sr : number > 0 [scalar]
        audio sampling rate
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.
    axis : int [scalar]
        The axis representing the time axis of X.
        By default, the last axis (-1) is taken.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        ndarray of times (in seconds) corresponding to each frame of X.

    See Also
    --------
    samples_like :
        Return an array of sample indices to match the time axis from a feature matrix.

    Examples
    --------
    Provide a feature matrix input:

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> D = librosa.stft(y)
    >>> times = librosa.times_like(D)
    >>> times
    array([0.   , 0.023, ..., 5.294, 5.317])

    Provide a scalar input:

    >>> n_frames = 2647
    >>> times = librosa.times_like(n_frames)
    >>> times
    array([  0.00000000e+00,   2.32199546e-02,   4.64399093e-02, ...,
             6.13935601e+01,   6.14167800e+01,   6.14400000e+01])
    """
    samples = samples_like(X, hop_length=hop_length, n_fft=n_fft, axis=axis)
    return samples_to_time(samples, sr=sr)


def samples_like(X, *, hop_length=512, n_fft=None, axis=-1):
    """Return an array of sample indices to match the time axis from a feature matrix.

    Parameters
    ----------
    X : np.ndarray or scalar
        - If ndarray, X is a feature matrix, e.g. STFT, chromagram, or mel spectrogram.
        - If scalar, X represents the number of frames.
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.
    axis : int [scalar]
        The axis representing the time axis of ``X``.
        By default, the last axis (-1) is taken.

    Returns
    -------
    samples : np.ndarray [shape=(n,)]
        ndarray of sample indices corresponding to each frame of ``X``.

    See Also
    --------
    times_like :
        Return an array of time values to match the time axis from a feature matrix.

    Examples
    --------
    Provide a feature matrix input:

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> X = librosa.stft(y)
    >>> samples = librosa.samples_like(X)
    >>> samples
    array([     0,    512, ..., 116736, 117248])

    Provide a scalar input:

    >>> n_frames = 2647
    >>> samples = librosa.samples_like(n_frames)
    >>> samples
    array([      0,     512,    1024, ..., 1353728, 1354240, 1354752])
    """
    if np.isscalar(X):
        frames = np.arange(X)
    else:
        frames = np.arange(X.shape[axis])
    return frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)


def pyin(
    y,
    *,
    fmin,
    fmax,
    sr=22050,
    frame_length=2048,
    win_length=None,
    hop_length=None,
    n_thresholds=100,
    beta_parameters=(2, 18),
    boltzmann_parameter=2,
    resolution=0.1,
    max_transition_rate=35.92,
    switch_prob=0.01,
    no_trough_prob=0.01,
    fill_na=np.nan,
    center=True,
    pad_mode="constant",
):
    """Fundamental frequency (F0) estimation using probabilistic YIN (pYIN).

    pYIN [#]_ is a modificatin of the YIN algorithm [#]_ for fundamental frequency (F0) estimation.
    In the first step of pYIN, F0 candidates and their probabilities are computed using the YIN algorithm.
    In the second step, Viterbi decoding is used to estimate the most likely F0 sequence and voicing flags.

    .. [#] Mauch, Matthias, and Simon Dixon.
        "pYIN: A fundamental frequency estimator using probabilistic threshold distributions."
        2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Args:
        y : np.ndarray [shape=(..., n)]
            audio time series. Multi-channel is supported.
        fmin : number > 0 [scalar]
            minimum frequency in Hertz.
            The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
            though lower values may be feasible.
        fmax : number > 0 [scalar]
            maximum frequency in Hertz.
            The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
            though higher values may be feasible.
        sr : number > 0 [scalar]
            sampling rate of ``y`` in Hertz.
        frame_length : int > 0 [scalar]
            length of the frames in samples.
            By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at
            a sampling rate of 22050 Hz.
        win_length : None or int > 0 [scalar]
            length of the window for calculating autocorrelation in samples.
            If ``None``, defaults to ``frame_length // 2``
        hop_length : None or int > 0 [scalar]
            number of audio samples between adjacent pYIN predictions.
            If ``None``, defaults to ``frame_length // 4``.
        n_thresholds : int > 0 [scalar]
            number of thresholds for peak estimation.
        beta_parameters : tuple
            shape parameters for the beta distribution prior over thresholds.
        boltzmann_parameter : number > 0 [scalar]
            shape parameter for the Boltzmann distribution prior over troughs.
            Larger values will assign more mass to smaller periods.
        resolution : float in `(0, 1)`
            Resolution of the pitch bins.
            0.01 corresponds to cents.
        max_transition_rate : float > 0
            maximum pitch transition rate in octaves per second.
        switch_prob : float in ``(0, 1)``
            probability of switching from voiced to unvoiced or vice versa.
        no_trough_prob : float in ``(0, 1)``
            maximum probability to add to global minimum if no trough is below threshold.
        fill_na : None, float, or ``np.nan``
            default value for unvoiced frames of ``f0``.
            If ``None``, the unvoiced frames will contain a best guess value.
        center : boolean
            If ``True``, the signal ``y`` is padded so that frame
            ``D[:, t]`` is centered at ``y[t * hop_length]``.
            If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
            Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
            time grid by means of ``librosa.core.frames_to_samples``.
        pad_mode : string or function
            If ``center=True``, this argument is passed to ``np.pad`` for padding
            the edges of the signal ``y``. By default (``pad_mode="constant"``),
            ``y`` is padded on both sides with zeros.
            If ``center=False``,  this argument is ignored.

    .. see also:: `np.pad`

    Returns
    -------
    f0: np.ndarray [shape=(..., n_frames)]
        time series of fundamental frequencies in Hertz.
    voiced_flag: np.ndarray [shape=(..., n_frames)]
        time series containing boolean flags indicating whether a frame is voiced or not.
    voiced_prob: np.ndarray [shape=(..., n_frames)]
        time series containing the probability that a frame is voiced.
    .. note:: If multi-channel input is provided, f0 and voicing are estimated separately for each channel.

    See Also
    --------
    librosa.yin :
        Fundamental frequency (F0) estimation using the YIN algorithm.

    Examples
    --------
    Computing a fundamental frequency (F0) curve from an audio input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> f0, voiced_flag, voiced_probs = librosa.pyin(y,
    ...                                              fmin=librosa.note_to_hz('C2'),
    ...                                              fmax=librosa.note_to_hz('C7'))
    >>> times = librosa.times_like(f0)

    Overlay F0 over a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
    >>> ax.set(title='pYIN fundamental frequency estimation')
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    >>> ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    >>> ax.legend(loc='upper right')
    """

    if fmin is None or fmax is None:
        raise ParameterError('both "fmin" and "fmax" must be provided')

    # Set the default window length if it is not already specified.
    if win_length is None:
        win_length = frame_length // 2

    if win_length >= frame_length:
        raise ParameterError(
            "win_length={} cannot exceed given frame_length={}".format(
                win_length, frame_length
            )
        )

    # Set the default hop if it is not already specified.
    if hop_length is None:
        hop_length = frame_length // 4

    # Pad the time series so that frames are centered
    if center:
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (frame_length // 2, frame_length // 2)
        y = np.pad(y, padding, mode=pad_mode)

    # Frame audio.
    y_frames = frame(y, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = max(int(np.floor(sr / fmax)), 1)
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    # Calculate cumulative mean normalized difference function.
    yin_frames = _cumulative_mean_normalized_difference(
        y_frames, frame_length, win_length, min_period, max_period
    )

    # Parabolic interpolation.
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    # Find Yin candidates and probabilities.
    # The implementation here follows the official pYIN software which
    # differs from the method described in the paper.
    # 1. Define the prior over the thresholds.
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    beta_probs = np.diff(beta_cdf)

    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    def _helper(a, b):
        return _pyin_helper(
            a,
            b,
            sr,
            thresholds,
            boltzmann_parameter,
            beta_probs,
            no_trough_prob,
            min_period,
            fmin,
            n_pitch_bins,
            n_bins_per_semitone,
        )

    helper = np.vectorize(_helper, signature="(f,t),(k,t)->(1,d,t),(j,t)")
    observation_probs, voiced_prob = helper(yin_frames, parabolic_shifts)

    # Construct transition matrix.
    max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    # Construct the within voicing transition probabilities
    transition = transition_local(
        n_pitch_bins, transition_width, window="triangle", wrap=False
    )

    # Include across voicing transition probabilities
    t_switch = transition_loop(2, 1 - switch_prob)
    transition = np.kron(t_switch, transition)

    p_init = np.zeros(2 * n_pitch_bins)
    p_init[n_pitch_bins:] = 1 / n_pitch_bins

    states = viterbi(observation_probs, transition, p_init=p_init)

    # Find f0 corresponding to each decoded pitch bin.
    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins

    if fill_na is not None:
        f0[~voiced_flag] = fill_na

    return f0[..., 0, :], voiced_flag[..., 0, :], voiced_prob[..., 0, :]


def _cumulative_mean_normalized_difference(y_frames, frame_length, win_length,
                                           min_period, max_period):
    """Cumulative mean normalized difference function (equation 8 in [#]_)

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.
    frame_length : int > 0 [scalar]
        length of the frames in samples.
    win_length : int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.
    min_period : int > 0 [scalar]
        minimum period.
    max_period : int > 0 [scalar]
        maximum period.

    Returns
    -------
    yin_frames : np.ndarray [shape=(max_period-min_period+1,n_frames)]
        Cumulative mean normalized difference function for each frame.
    """
    # Autocorrelation.
    a = np.fft.rfft(y_frames, frame_length, axis=-2)
    b = np.fft.rfft(y_frames[..., win_length:0:-1, :], frame_length, axis=-2)
    acf_frames = np.fft.irfft(a * b, frame_length, axis=-2)[..., win_length:, :]
    acf_frames[np.abs(acf_frames) < 1e-6] = 0

    # Energy terms.
    energy_frames = np.cumsum(y_frames ** 2, axis=-2)
    energy_frames = (
        energy_frames[..., win_length:, :] - energy_frames[..., :-win_length, :]
    )
    energy_frames[np.abs(energy_frames) < 1e-6] = 0

    # Difference function.
    yin_frames = energy_frames[..., :1, :] + energy_frames - 2 * acf_frames

    # Cumulative mean normalized difference function.
    yin_numerator = yin_frames[..., min_period : max_period + 1, :]
    # broadcast this shape to have leading ones
    tau_range = expand_to(np.arange(1, max_period + 1), ndim=yin_frames.ndim, axes=-2)

    cumulative_mean = (
        np.cumsum(yin_frames[..., 1 : max_period + 1, :], axis=-2) / tau_range
    )
    yin_denominator = cumulative_mean[..., min_period - 1 : max_period, :]
    yin_frames = yin_numerator / (yin_denominator + tiny(yin_denominator))
    return yin_frames


def _parabolic_interpolation(y_frames):
    """Piecewise parabolic interpolation for yin and pyin.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.

    Returns
    -------
    parabolic_shifts : np.ndarray [shape=(frame_length, n_frames)]
        position of the parabola optima
    """

    parabolic_shifts = np.zeros_like(y_frames)
    parabola_a = (
        y_frames[..., :-2, :] + y_frames[..., 2:, :] - 2 * y_frames[..., 1:-1, :]
    ) / 2
    parabola_b = (y_frames[..., 2:, :] - y_frames[..., :-2, :]) / 2
    parabolic_shifts[..., 1:-1, :] = -parabola_b / (
        2 * parabola_a + tiny(parabola_a)
    )
    parabolic_shifts[np.abs(parabolic_shifts) > 1] = 0
    return parabolic_shifts


def _pyin_helper(
    yin_frames,
    parabolic_shifts,
    sr,
    thresholds,
    boltzmann_parameter,
    beta_probs,
    no_trough_prob,
    min_period,
    fmin,
    n_pitch_bins,
    n_bins_per_semitone,
):

    yin_probs = np.zeros_like(yin_frames)

    for i, yin_frame in enumerate(yin_frames.MObjT):
        # 2. For each frame find the troughs.
        is_trough = localmin(yin_frame)

        is_trough[0] = yin_frame[0] < yin_frame[1]
        (trough_index,) = np.nonzero(is_trough)

        if len(trough_index) == 0:
            continue

        # 3. Find the troughs below each threshold.
        # these are the local minima of the frame, could get them directly without the trough index
        trough_heights = yin_frame[trough_index]
        trough_thresholds = np.less.outer(trough_heights, thresholds[1:])

        # 4. Define the prior over the troughs.
        # Smaller periods are weighted more.
        trough_positions = np.cumsum(trough_thresholds, axis=0) - 1
        n_troughs = np.count_nonzero(trough_thresholds, axis=0)

        trough_prior = scipy.stats.boltzmann.pmf(
            trough_positions, boltzmann_parameter, n_troughs
        )

        trough_prior[~trough_thresholds] = 0

        # 5. For each threshold add probability to global minimum if no trough is below threshold,
        # else add probability to each trough below threshold biased by prior.

        probs = trough_prior.dot(beta_probs)

        global_min = np.argmin(trough_heights)
        n_thresholds_below_min = np.count_nonzero(~trough_thresholds[global_min, :])
        probs[global_min] += no_trough_prob * np.sum(
            beta_probs[:n_thresholds_below_min]
        )

        yin_probs[trough_index, i] = probs

    yin_period, frame_index = np.nonzero(yin_probs)

    # Refine peak by parabolic interpolation.
    period_candidates = min_period + yin_period
    period_candidates = period_candidates + parabolic_shifts[yin_period, frame_index]
    f0_candidates = sr / period_candidates

    # Find pitch bin corresponding to each f0 candidate.
    bin_index = 12 * n_bins_per_semitone * np.log2(f0_candidates / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)

    # Observation probabilities.
    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]

    voiced_prob = np.clip(
        np.sum(observation_probs[:n_pitch_bins, :], axis=0, keepdims=True), 0, 1
    )
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob) / n_pitch_bins

    return observation_probs[np.newaxis], voiced_prob


def localmin(x, *, axis=0):
    """Find local minima in an array

    An element ``x[i]`` is considered a local minimum if the following
    conditions are met:

    - ``x[i] < x[i-1]``
    - ``x[i] <= x[i+1]``

    Note that the first condition is strict, and that the first element
    ``x[0]`` will never be considered as a local minimum.

    Examples
    --------
    >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
    >>> localmin(x)
    array([False,  True, False, False,  True, False,  True, False])

    >>> # Two-dimensional example
    >>> x = np.array([[1,0,1], [2, -1, 0], [2, 1, 3]])
    >>> localmin(x, axis=0)
    array([[False, False, False],
           [False,  True,  True],
           [False, False, False]])

    >>> localmin(x, axis=1)
    array([[False,  True, False],
           [False,  True, False],
           [False,  True, False]])

    Parameters
    ----------
    x : np.ndarray [shape=(d1,d2,...)]
        input vector or array
    axis : int
        axis along which to compute local minimality

    Returns
    -------
    m : np.ndarray [shape=x.shape, dtype=bool]
        indicator array of local minimality along ``axis``

    See Also
    --------
    localmax
    """

    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (1, 1)

    x_pad = np.pad(x, paddings, mode="edge")

    inds1 = [slice(None)] * x.ndim
    inds1[axis] = slice(0, -2)

    inds2 = [slice(None)] * x.ndim
    inds2[axis] = slice(2, x_pad.shape[axis])

    return (x < x_pad[tuple(inds1)]) & (x <= x_pad[tuple(inds2)])


def transition_local(n_states, width, *, window="triangle", wrap=False):
    """Construct a localized transition matrix.

    The transition matrix will have the following properties:

        - ``transition[i, j] = 0`` if ``|i - j| > width``
        - ``transition[i, i]`` is maximal
        - ``transition[i, i - width//2 : i + width//2]`` has shape ``window``

    This type of transition matrix is appropriate for state spaces
    that discretely approximate continuous variables, such as in fundamental
    frequency estimation.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    width : int >= 1 or iterable
        The maximum number of states to treat as "local".
        If iterable, it should have length equal to ``n_states``,
        and specify the width independently for each state.

    window : str, callable, or window specification
        The window function to determine the shape of the "local" distribution.

        Any window specification supported by `filters.get_window` will work here.

        .. note:: Certain windows (e.g., 'hann') are identically 0 at the boundaries,
            so and effectively have ``width-2`` non-zero values.  You may have to expand
            ``width`` to get the desired behavior.

    wrap : bool
        If ``True``, then state locality ``|i - j|`` is computed modulo ``n_states``.
        If ``False`` (default), then locality is absolute.

    See Also
    --------
    librosa.filters.get_window

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    Triangular distributions with and without wrapping

    >>> librosa.sequence.transition_local(5, 3, window='triangle', wrap=False)
    array([[0.667, 0.333, 0.   , 0.   , 0.   ],
           [0.25 , 0.5  , 0.25 , 0.   , 0.   ],
           [0.   , 0.25 , 0.5  , 0.25 , 0.   ],
           [0.   , 0.   , 0.25 , 0.5  , 0.25 ],
           [0.   , 0.   , 0.   , 0.333, 0.667]])

    >>> librosa.sequence.transition_local(5, 3, window='triangle', wrap=True)
    array([[0.5 , 0.25, 0.  , 0.  , 0.25],
           [0.25, 0.5 , 0.25, 0.  , 0.  ],
           [0.  , 0.25, 0.5 , 0.25, 0.  ],
           [0.  , 0.  , 0.25, 0.5 , 0.25],
           [0.25, 0.  , 0.  , 0.25, 0.5 ]])

    Uniform local distributions with variable widths and no wrapping

    >>> librosa.sequence.transition_local(5, [1, 2, 3, 3, 1], window='ones', wrap=False)
    array([[1.   , 0.   , 0.   , 0.   , 0.   ],
           [0.5  , 0.5  , 0.   , 0.   , 0.   ],
           [0.   , 0.333, 0.333, 0.333, 0.   ],
           [0.   , 0.   , 0.333, 0.333, 0.333],
           [0.   , 0.   , 0.   , 0.   , 1.   ]])
    """

    if not isinstance(n_states, (int, np.integer)) or n_states <= 1:
        raise ParameterError("n_states={} must be a positive integer > 1")

    width = np.asarray(width, dtype=int)
    if width.ndim == 0:
        width = np.tile(width, n_states)

    if width.shape != (n_states,):
        raise ParameterError(
            "width={} must have length equal to n_states={}".format(width, n_states)
        )

    if np.any(width < 1):
        raise ParameterError("width={} must be at least 1")

    transition = np.zeros((n_states, n_states), dtype=np.float64)

    # Fill in the widths.  This is inefficient, but simple
    for i, width_i in enumerate(width):
        trans_row = pad_center(
            get_window(window, width_i, fftbins=False), size=n_states
        )
        trans_row = np.roll(trans_row, n_states // 2 + i + 1)

        if not wrap:
            # Knock out the off-diagonal-band elements
            trans_row[min(n_states, i + width_i // 2 + 1) :] = 0
            trans_row[: max(0, i - width_i // 2)] = 0

        transition[i] = trans_row

    # Row-normalize
    transition /= transition.sum(axis=1, keepdims=True)

    return transition


def transition_loop(n_states, prob):
    """Construct a self-loop transition matrix over ``n_states``.

    The transition matrix will have the following properties:

        - ``transition[i, i] = p`` for all ``i``
        - ``transition[i, j] = (1 - p) / (n_states - 1)`` for all ``j != i``

    This type of transition matrix is appropriate when states tend to be
    locally stable, and there is no additional structure between different
    states.  This is primarily useful for de-noising frame-wise predictions.

    Parameters
    ----------
    n_states : int > 1
        The number of states

    prob : float in [0, 1] or iterable, length=n_states
        If a scalar, this is the probability of a self-transition.

        If a vector of length ``n_states``, ``p[i]`` is the probability of self-transition in state ``i``

    Returns
    -------
    transition : np.ndarray [shape=(n_states, n_states)]
        The transition matrix

    Examples
    --------
    >>> librosa.sequence.transition_loop(3, 0.5)
    array([[0.5 , 0.25, 0.25],
           [0.25, 0.5 , 0.25],
           [0.25, 0.25, 0.5 ]])

    >>> librosa.sequence.transition_loop(3, [0.8, 0.5, 0.25])
    array([[0.8  , 0.1  , 0.1  ],
           [0.25 , 0.5  , 0.25 ],
           [0.375, 0.375, 0.25 ]])
    """

    if not isinstance(n_states, (int, np.integer)) or n_states <= 1:
        raise ParameterError("n_states={} must be a positive integer > 1")

    transition = np.empty((n_states, n_states), dtype=np.float64)

    # if it's a float, make it a vector
    prob = np.asarray(prob, dtype=np.float64)

    if prob.ndim == 0:
        prob = np.tile(prob, n_states)

    if prob.shape != (n_states,):
        raise ParameterError(
            "prob={} must have length equal to n_states={}".format(prob, n_states)
        )

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError(
            "prob={} must have values in the range [0, 1]".format(prob)
        )

    for i, prob_i in enumerate(prob):
        transition[i] = (1.0 - prob_i) / (n_states - 1)
        transition[i, i] = prob_i

    return transition


def viterbi(prob, transition, *, p_init=None, return_logp=False):
    """Viterbi decoding from observation likelihoods.

    Given a sequence of observation likelihoods ``prob[s, t]``,
    indicating the conditional likelihood of seeing the observation
    at time ``t`` from state ``s``, and a transition matrix
    ``transition[i, j]`` which encodes the conditional probability of
    moving from state ``i`` to state ``j``, the Viterbi algorithm [#]_ computes
    the most likely sequence of states from the observations.

    .. [#] Viterbi, Andrew. "Error bounds for convolutional codes and an
        asymptotically optimum decoding algorithm."
        IEEE transactions on Information Theory 13.2 (1967): 260-269.

    Parameters
    ----------
    prob : np.ndarray [shape=(..., n_states, n_steps), non-negative]
        ``prob[..., s, t]`` is the probability of observation at time ``t``
        being generated by state ``s``.
    transition : np.ndarray [shape=(n_states, n_states), non-negative]
        ``transition[i, j]`` is the probability of a transition from i->j.
        Each row must sum to 1.
    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, a uniform distribution is assumed.
    return_logp : bool
        If ``True``, return the log-likelihood of the state sequence.

    Returns
    -------
    Either ``states`` or ``(states, logp)``:
    states : np.ndarray [shape=(..., n_steps,)]
        The most likely state sequence.
        If ``prob`` contains multiple channels of input, then each channel is
        decoded independently.
    logp : scalar [float] or np.ndarray
        If ``return_logp=True``, the log probability of ``states`` given
        the observations.

    See Also
    --------
    viterbi_discriminative : Viterbi decoding from state likelihoods

    Examples
    --------
    Example from https://en.wikipedia.org/wiki/Viterbi_algorithm#Example

    In this example, we have two states ``healthy`` and ``fever``, with
    initial probabilities 60% and 40%.

    We have three observation possibilities: ``normal``, ``cold``, and
    ``dizzy``, whose probabilities given each state are:

    ``healthy => {normal: 50%, cold: 40%, dizzy: 10%}`` and
    ``fever => {normal: 10%, cold: 30%, dizzy: 60%}``

    Finally, we have transition probabilities:

    ``healthy => healthy (70%)`` and
    ``fever => fever (60%)``.

    Over three days, we observe the sequence ``[normal, cold, dizzy]``,
    and wish to know the maximum likelihood assignment of states for the
    corresponding days, which we compute with the Viterbi algorithm below.

    >>> p_init = np.array([0.6, 0.4])
    >>> p_emit = np.array([[0.5, 0.4, 0.1],
    ...                    [0.1, 0.3, 0.6]])
    >>> p_trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> path, logp = librosa.sequence.viterbi(p_emit, p_trans, p_init=p_init,
    ...                                       return_logp=True)
    >>> print(logp, path)
    -4.19173690823075 [0 0 1]
    """

    n_states, n_steps = prob.shape[-2:]

    if transition.shape != (n_states, n_states):
        raise ParameterError(
            "transition.shape={}, must be "
            "(n_states, n_states)={}".format(transition.shape, (n_states, n_states))
        )

    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ParameterError(
            "Invalid transition matrix: must be non-negative "
            "and sum to 1 on each row."
        )

    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError("Invalid probability values: must be between 0 and 1.")

    # Compute log-likelihoods while avoiding log-underflow
    epsilon = tiny(prob)

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1.0 / n_states)
    elif (
        np.any(p_init < 0)
        or not np.allclose(p_init.sum(), 1)
        or p_init.shape != (n_states,)
    ):
        raise ParameterError(
            "Invalid initial state distribution: " "p_init={}".format(p_init)
        )

    log_trans = np.log(transition + epsilon)
    log_prob = np.log(prob + epsilon)
    log_p_init = np.log(p_init + epsilon)

    def _helper(lp):
        # Transpose input
        try:
            _state, logp = numpyx.viterbi_core(lp.MObjT, log_trans, log_p_init)
        except:
            print("Exception!")
            _state, logp = _viterbi(lp.MObjT, log_trans, log_p_init)
        # Transpose outputs for return
        return _state.MObjT, logp

    if log_prob.ndim == 2:
        states, logp = _helper(log_prob)
    else:
        # Vectorize the helper
        print(f"{log_prob.ndim=}, {log_trans.shape=}, {log_p_init.shape=}")
        viterbi_vect = np.vectorize(
            _helper, otypes=[np.uint64, np.float64], signature="(s,t)->(t),(1)"
        )

        states, logp = viterbi_vect(log_prob)

        # Flatten out the trailing dimension introduced by vectorization
        logp = logp[..., 0]

    if return_logp:
        return states, logp

    return states


def _viterbi(log_prob, log_trans, log_p_init):
    """Core Viterbi algorithm.

    This is intended for internal use only.

    Parameters
    ----------
    log_prob : np.ndarray [shape=(MObjT, m)]
        ``log_prob[t, s]`` is the conditional log-likelihood
        ``log P[X = X(t) | State(t) = s]``
    log_trans : np.ndarray [shape=(m, m)]
        The log transition matrix
        ``log_trans[i, j] = log P[State(t+1) = j | State(t) = i]``
    log_p_init : np.ndarray [shape=(m,)]
        log of the initial state distribution

    Returns
    -------
    None
        All computations are performed in-place on ``state, value, ptr``.
    """
    n_steps, n_states = log_prob.shape

    state = np.zeros(n_steps, dtype=np.uint16)
    value = np.zeros((n_steps, n_states), dtype=np.float64)
    ptr = np.zeros((n_steps, n_states), dtype=np.uint16)

    # factor in initial state distribution
    value[0] = log_prob[0] + log_p_init

    for t in range(1, n_steps):
        # Want V[t, j] <- p[t, j] * max_k V[t-1, k] * A[k, j]
        #    assume at time t-1 we were in state k
        #    transition k -> j

        # Broadcast over rows:
        #    Tout[k, j] = V[t-1, k] * A[k, j]
        #    then take the max over columns
        # We'll do this in log-space for stability

        trans_out = value[t - 1] + log_trans.MObjT

        # Unroll the max/argmax loop to enable numba support
        for j in range(n_states):
            ptr[t, j] = np.argmax(trans_out[j])
            # value[t, j] = log_prob[t, j] + np.max(trans_out[j])
            value[t, j] = log_prob[t, j] + trans_out[j, ptr[t][j]]

    # Now roll backward

    # Get the last state
    state[-1] = np.argmax(value[-1])

    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]

    logp = value[-1:, state[-1]]

    return state, logp


def spectral_centroid(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    S: np.ndarray | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    freq: np.ndarray | None = None,
    win_length: int | None = None,
    window="hann",
    center: bool = True,
    pad_mode="constant",
) -> np.ndarray:
    """Compute the spectral centroid.

    Each frame of a magnitude spectrogram is normalized and treated as a
    distribution over frequency bins, from which the mean (centroid) is
    extracted per frame.

    More precisely, the centroid at frame ``t`` is defined as [#]_::

        centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])

    where ``S`` is a magnitude spectrogram, and ``freq`` is the array of
    frequencies (e.g., FFT frequencies in Hz) of the rows of ``S``.

    .. [#] Klapuri, A., & Davy, M. (Eds.). (2007). Signal processing
        methods for music transcription, chapter 5.
        Springer Science & Business Media.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.reassigned_spectrogram`
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `librosa.filters.get_window`
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          `t` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    centroid : np.ndarray [shape=(..., 1, t)]
        centroid frequencies

    See Also
    --------
    librosa.stft : Short-time Fourier Transform
    librosa.reassigned_spectrogram : Time-frequency reassigned spectrogram

    Examples
    --------
    From time-series input:

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    >>> cent
    array([[1768.888, 1921.774, ..., 5663.477, 5813.683]])

    From spectrogram input:

    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_centroid(S=S)
    array([[1768.888, 1921.774, ..., 5663.477, 5813.683]])

    Using variable bin center frequencies:

    >>> freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
    >>> librosa.feature.spectral_centroid(S=np.abs(D), freq=freqs)
    array([[1768.838, 1921.801, ..., 5663.513, 5813.747]])

    Plot the result

    >>> import matplotlib.pyplot as plt
    >>> times = librosa.times_like(cent)
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax)
    >>> ax.plot(times, cent._CheckedDictT, label='Spectral centroid', color='w')
    >>> ax.legend(loc='upper right')
    >>> ax.set(title='log Power spectrogram')
    """

    # input is time domain:y or spectrogram:s
    #

    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    if not np.isrealobj(S):
        raise ParameterError(
            "Spectral centroid is only defined " "with real-valued input"
        )
    elif np.any(S < 0):
        raise ParameterError(
            "Spectral centroid is only defined " "with non-negative energies"
        )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    if freq.ndim == 1:
        # reshape for broadcasting
        freq = expand_to(freq, ndim=S.ndim, axes=-2)

    # Column-normalize S
    centroid: np.ndarray = np.sum(
        freq * normalize(S, norm=1, axis=-2), axis=-2, keepdims=True
    )
    return centroid


def __coord_chroma(n: int, bins_per_octave: int = 12, **_kwargs) -> np.ndarray:
    """Get chroma bin numbers"""
    return np.linspace(0, (12.0 * n) / bins_per_octave, num=n, endpoint=False)


def __coord_mel_hz(n: int,
                   fmin: float | None = 0.0,
                   fmax: float | None = None,
                   sr: float = 22050,
                   htk: bool = False,
                   **_kwargs
                   ) -> np.ndarray:
    """Get the frequencies for Mel bins"""

    if fmin is None:
        fmin = 0.0
    if fmax is None:
        fmax = 0.5 * sr

    basis = mel_frequencies(n, fmin=fmin, fmax=fmax, htk=htk)
    return basis


def __coord_fft_hz(n: int, sr: float = 22050, n_fft: int | None = None, **_kwargs
                   ) -> np.ndarray:
    """Get the frequencies for FFT bins"""
    if n_fft is None:
        n_fft = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = fft_frequencies(sr=sr, n_fft=n_fft)
    return basis


def __coord_n(n: int, **_kwargs) -> np.ndarray:
    """Get bare positions"""
    return np.arange(n)


def __coord_time(n: int, sr: float = 22050, hop_length: int = 512, **_kwargs
                 ) -> np.ndarray:
    """Get time coordinates from frames"""
    times: np.ndarray = frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    return times


def __mesh_coords(ax_type, coords, n, **kwargs):
    """Compute axis coordinates"""

    if coords is not None:
        if len(coords) not in (n, n + 1):
            raise ParameterError(
                f"Coordinate shape mismatch: {len(coords)}!={n} or {n}+1"
            )
        return coords

    coord_map  = {
        "linear": __coord_fft_hz,
        "fft": __coord_fft_hz,
        "fft_note": __coord_fft_hz,
        "fft_svara": __coord_fft_hz,
        "hz": __coord_fft_hz,
        "log": __coord_fft_hz,
        "mel": __coord_mel_hz,
        # "cqt": __coord_cqt_hz,
        # "cqt_hz": __coord_cqt_hz,
        # "cqt_note": __coord_cqt_hz,
        # "cqt_svara": __coord_cqt_hz,
        # "vqt_fjs": __coord_vqt_hz,
        # "vqt_hz": __coord_vqt_hz,
        # "vqt_note": __coord_vqt_hz,
        "chroma": __coord_chroma,
        "chroma_c": __coord_chroma,
        "chroma_h": __coord_chroma,
        "chroma_fjs": __coord_n,  # We can't use a 12-normalized tick locator here
        "time": __coord_time,
        "h": __coord_time,
        "m": __coord_time,
        "s": __coord_time,
        "ms": __coord_time,
        "lag": __coord_time,
        "lag_h": __coord_time,
        "lag_m": __coord_time,
        "lag_s": __coord_time,
        "lag_ms": __coord_time,
        "tonnetz": __coord_n,
        "off": __coord_n,
        # "tempo": __coord_tempo,
        # "fourier_tempo": __coord_fourier_tempo,
        "frames": __coord_n,
        None: __coord_n,
    }

    if ax_type not in coord_map:
        raise ParameterError(f"Unknown axis type: {ax_type}")
    return coord_map[ax_type](n, **kwargs)


def __check_axes(axes: plt.Axes | None) -> plt.Axes:
    """Check if "axes" is an instance of an axis object. If not, use `gca`."""
    if axes is None:
        axes = plt.gca()
    elif not isinstance(axes, plt.Axes):
        raise ParameterError(f"`axes` must be an instance of matplotlib.axes.Axes. "
                             f"Found type(axes)={type(axes)}")
    return axes


def __scale_axes(axes, ax_type, which):
    """Set the axis scaling"""

    kwargs = dict()
    thresh = "linthresh"
    base = "base"
    scale = "linscale"

    if which == "x":
        scaler = axes.set_xscale
        limit = axes.set_xlim
    else:
        scaler = axes.set_yscale
        limit = axes.set_ylim

    # Map ticker scales
    if ax_type == "mel":
        mode = "symlog"
        kwargs[thresh] = 1000.0
        kwargs[base] = 2
    elif ax_type in ("cqt", "cqt_hz", "cqt_note", "cqt_svara", "vqt_hz", "vqt_note", "vqt_fjs"):
        mode = "log"
        kwargs[base] = 2
    elif ax_type in ("log", "fft_note", "fft_svara"):
        mode = "symlog"
        kwargs[base] = 2
        kwargs[thresh] = float(pt.n2f("C2"))
        kwargs[scale] = 0.5
    elif ax_type in ["tempo", "fourier_tempo"]:
        mode = "log"
        kwargs[base] = 2
        limit(16, 480)
    else:
        return

    scaler(mode, **kwargs)


class TimeFormatter(mplticker.Formatter):
    """A tick formatter for time axes.

    Automatically switches between seconds, minutes:seconds,
    or hours:minutes:seconds.

    Parameters
    ----------
    lag : bool
        If ``True``, then the time axis is interpreted in lag coordinates.
        Anything past the midpoint will be converted to negative time.

    unit : str or None
        Abbreviation of the string representation for axis labels and ticks.
        List of supported units:
        * `"h"`: hour-based format (`H:MM:SS`)
        * `"m"`: minute-based format (`M:SS`)
        * `"s"`: second-based format (`S.sss` in scientific notation)
        * `"ms"`: millisecond-based format (`s.µµµ` in scientific notation)
        * `None`: adaptive to the duration of the underlying time range: similar
        to `"h"` above 3600 seconds; to `"m"` between 60 and 3600 seconds; to
        `"s"` between 1 and 60 seconds; and to `"ms"` below 1 second.


    See also
    --------
    matplotlib.ticker.Formatter


    Examples
    --------

    For normal time

    >>> import matplotlib.pyplot as plt
    >>> times = np.arange(30)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    >>> ax.set(xlabel='Time')

    Manually set the physical time unit of the x-axis to milliseconds

    >>> times = np.arange(100)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(unit='ms'))
    >>> ax.set(xlabel='Time (ms)')

    For lag plots

    >>> times = np.arange(60)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(lag=True))
    >>> ax.set(xlabel='Lag')
    """

    def __init__(self, lag: bool = False, unit: str | None = None):
        if unit not in ["h", "m", "s", "ms", None]:
            raise ParameterError(f"Unknown time unit: {unit}")

        self.unit = unit
        self.lag = lag

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Return the time format as pos"""

        _, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()

        # In lag-time axes, anything greater than dmax / 2 is negative time
        if self.lag and x >= dmax * 0.5:
            # In lag mode, don't tick past the limits of the data
            if x > dmax:
                return ""
            value = np.abs(x - dmax)
            # Do we need to tweak vmin/vmax here?
            sign = "-"
        else:
            value = x
            sign = ""

        if self.unit == "h" or ((self.unit is None) and (vmax - vmin > 3600)):
            s = "{:d}:{:02d}:{:02d}".format(
                int(value / 3600.0),
                int(np.mod(value / 60.0, 60)),
                int(np.mod(value, 60)),
            )
        elif self.unit == "m" or ((self.unit is None) and (vmax - vmin > 60)):
            s = "{:d}:{:02d}".format(int(value / 60.0), int(np.mod(value, 60)))
        elif self.unit == "s":
            s = f"{value:.3g}"
        elif self.unit == None and (vmax - vmin >= 1):
            s = f"{value:.2g}"
        elif self.unit == "ms":
            s = "{:.3g}".format(value * 1000)
        elif self.unit == None and (vmax - vmin < 1):
            s = f"{value:.3f}"

        return f"{sign:s}{s:s}"


class NoteFormatter(mplticker.Formatter):
    """Ticker formatter for Notes

    Parameters
    ----------
    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    See also
    --------
    LogHzFormatter
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax[1].set(ylabel='Note')
    """

    def __init__(self, major: bool = True):
        self.major = major

    def __call__(self, x: float, pos: int | None = None) -> str:
        if x <= 0:
            return ""

        # Only use cent precision if our vspan is less than an octave
        vmin, vmax = self.axis.get_view_interval()
        if not self.major and vmax > 4 * max(1, vmin):
            return ""
        # cents = vmax <= 2 * max(1, vmin)
        return pt.f2n(x)


class LogHzFormatter(mplticker.Formatter):
    """Ticker formatter for logarithmic frequency

    Parameters
    ----------
    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    See also
    --------
    NoteFormatter
    matplotlib.ticker.Formatter

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].yaxis.set_major_formatter(librosa.display.LogHzFormatter())
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax[1].set(ylabel='Note')
    """

    def __init__(self, major: bool = True):
        self.major = major

    def __call__(self, x: float, pos: int | None = None) -> str:
        if x <= 0:
            return ""

        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ""

        return f"{x:g}"


def __decorate_axis(
    axis,
    ax_type,
    key="C:maj",
    Sa=None,
    mela=None,
    thaat=None,
    unicode=True,
    fmin=None,
    unison=None,
    intervals=None,
    bins_per_octave=None,
    n_bins=None,
    setlabel=True
):
    """Configure axis tickers, locators, and labels"""
    time_units = {"h": "hours", "m": "minutes", "s": "seconds", "ms": "milliseconds"}
    import matplotlib.ticker as mplticker

    #if ax_type == "tonnetz":
    #    axis.set_major_formatter(TonnetzFormatter())
    #    axis.set_major_locator(mplticker.FixedLocator(np.arange(6)))
    #    axis.set_label_text("Tonnetz")

    # if ax_type == "chroma":
    #     axis.set_major_formatter(ChromaFormatter(key=key, unicode=unicode))
    #     degrees = core.key_to_degrees(key)
    #     axis.set_major_locator(
    #         mplticker.FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel())
    #     )
    #     axis.set_label_text("Pitch class")
    #
    # elif ax_type == "chroma_h":
    #     if Sa is None:
    #         Sa = 0
    #     axis.set_major_formatter(ChromaSvaraFormatter(Sa=Sa, unicode=unicode))
    #     if thaat is None:
    #         # If no thaat is given, show all svara
    #         degrees = np.arange(12)
    #     else:
    #         degrees = core.thaat_to_degrees(thaat)
    #     # Rotate degrees relative to Sa
    #     degrees = np.mod(degrees + Sa, 12)
    #     axis.set_major_locator(
    #         mplticker.FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel())
    #     )
    #     axis.set_label_text("Svara")

    #elif ax_type == "chroma_c":
    #    if Sa is None:
    #        Sa = 0
    #    axis.set_major_formatter(
    #        ChromaSvaraFormatter(Sa=Sa, mela=mela, unicode=unicode)
    #    )
    #    degrees = core.mela_to_degrees(mela)
    #    # Rotate degrees relative to Sa
    #    degrees = np.mod(degrees + Sa, 12)
    #    axis.set_major_locator(
    #        mplticker.FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel())
    #    )
    #    axis.set_label_text("Svara")

    # elif ax_type == "chroma_fjs":
    #     if fmin is None:
    #         fmin = core.note_to_hz("C1")
    #
    #     if unison is None:
    #         unison = core.hz_to_note(fmin, octave=False, cents=False)
    #
    #     axis.set_major_formatter(
    #         ChromaFJSFormatter(
    #             intervals=intervals,
    #             unison=unison,
    #             unicode=unicode,
    #             bins_per_octave=bins_per_octave,
    #         )
    #     )
    #
    #     if isinstance(intervals, str) and bins_per_octave > 7:
    #         # If intervals are implicit, generate the first 7 and identify
    #         # them in the sorted set
    #         tick_intervals = core.interval_frequencies(
    #             7,
    #             fmin=1,
    #             intervals=intervals,
    #             bins_per_octave=bins_per_octave,
    #             sort=False,
    #         )
    #
    #         all_intervals = core.interval_frequencies(
    #             bins_per_octave,
    #             fmin=1,
    #             intervals=intervals,
    #             bins_per_octave=bins_per_octave,
    #             sort=True,
    #         )
    #
    #         degrees = util.match_events(tick_intervals, all_intervals)
    #     else:
    #         # If intervals are explicit, tick them all
    #         degrees = np.arange(bins_per_octave)
    #
    #     axis.set_major_locator(mplticker.FixedLocator(degrees))
    #     axis.set_label_text("Pitch class")

    # elif ax_type in ["tempo", "fourier_tempo"]:
    #     axis.set_major_formatter(mplticker.ScalarFormatter())
    #     axis.set_major_locator(mplticker.LogLocator(base=2.0))
    #     axis.set_label_text("BPM")

    if ax_type == "time":
        axis.set_major_formatter(TimeFormatter(unit=None, lag=False))
        axis.set_major_locator(
            mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10])
        )
        axis.set_label_text("Time")

    elif ax_type in time_units:
        axis.set_major_formatter(TimeFormatter(unit=ax_type, lag=False))
        axis.set_major_locator(
            mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10])
        )
        axis.set_label_text("Time ({:s})".format(time_units[ax_type]))

    # elif ax_type == "lag":
    #     axis.set_major_formatter(TimeFormatter(unit=None, lag=True))
    #     axis.set_major_locator(
    #         mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10])
    #     )
    #     axis.set_label_text("Lag")

    # elif isinstance(ax_type, str) and ax_type.startswith("lag_"):
    #     unit = ax_type[4:]
    #     axis.set_major_formatter(TimeFormatter(unit=unit, lag=True))
    #     axis.set_major_locator(
    #         mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10])
    #     )
    #     axis.set_label_text("Lag ({:s})".format(time_units[unit]))

    # elif ax_type == "cqt_note":
    #     axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
    #     # Where is C1 relative to 2**k hz?
    #     log_C1 = np.log2(core.note_to_hz("C1"))
    #     C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
    #     axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(C_offset,)))
    #     axis.set_minor_formatter(NoteFormatter(key=key, major=False, unicode=unicode))
    #     axis.set_minor_locator(
    #         mplticker.LogLocator(
    #             base=2.0, subs=C_offset * 2.0 ** (np.arange(1, 12) / 12.0)
    #         )
    #     )
    #     axis.set_label_text("Note")

    # elif ax_type == "cqt_svara":
    #     axis.set_major_formatter(SvaraFormatter(Sa=Sa, mela=mela, unicode=unicode))
    #     # Find the offset of Sa relative to 2**k Hz
    #     sa_offset = 2.0 ** (np.log2(Sa) - np.floor(np.log2(Sa)))
    #
    #     axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(sa_offset,)))
    #     axis.set_minor_formatter(
    #         SvaraFormatter(Sa=Sa, mela=mela, major=False, unicode=unicode)
    #     )
    #     axis.set_minor_locator(
    #         mplticker.LogLocator(
    #             base=2.0, subs=sa_offset * 2.0 ** (np.arange(1, 12) / 12.0)
    #         )
    #     )
    #     axis.set_label_text("Svara")

    # elif ax_type == "vqt_fjs":
    #     if fmin is None:
    #         fmin = core.note_to_hz("C1")
    #     axis.set_major_formatter(
    #         FJSFormatter(
    #             intervals=intervals,
    #             fmin=fmin,
    #             unison=unison,
    #             unicode=unicode,
    #             bins_per_octave=bins_per_octave,
    #             n_bins=n_bins,
    #         )
    #     )
    #     log_fmin = np.log2(fmin)
    #     fmin_offset = 2.0 ** (log_fmin - np.floor(log_fmin))
    #     axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(fmin_offset,)))
    #
    #     axis.set_minor_formatter(
    #         FJSFormatter(
    #             intervals=intervals,
    #             fmin=fmin,
    #             unison=unison,
    #             unicode=unicode,
    #             bins_per_octave=bins_per_octave,
    #             n_bins=n_bins,
    #             major=False,
    #         )
    #     )
    #     axis.set_minor_locator(
    #         mplticker.FixedLocator(
    #             core.interval_frequencies(
    #                 n_bins * 12 // bins_per_octave,
    #                 fmin=fmin,
    #                 intervals=intervals,
    #                 bins_per_octave=12,
    #             )
    #         )
    #     )
    #     axis.set_label_text("Note")

    # elif ax_type == "vqt_hz":
    #     if fmin is None:
    #         fmin = core.note_to_hz("C1")
    #     axis.set_major_formatter(LogHzFormatter())
    #     log_fmin = np.log2(fmin)
    #     fmin_offset = 2.0 ** (log_fmin - np.floor(log_fmin))
    #     axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(fmin_offset,)))
    #     axis.set_minor_formatter(LogHzFormatter(major=False))
    #     axis.set_minor_locator(
    #         mplticker.LogLocator(
    #             base=2.0,
    #             subs=core.interval_frequencies(
    #                 12, fmin=fmin_offset, intervals=intervals, bins_per_octave=12
    #             ),
    #         )
    #     )
    #     axis.set_label_text("Hz")
    #
    # elif ax_type == "vqt_note":
    #     if fmin is None:
    #         fmin = core.note_to_hz("C1")
    #     axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
    #     log_fmin = np.log2(fmin)
    #     fmin_offset = 2.0 ** (log_fmin - np.floor(log_fmin))
    #     axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(fmin_offset,)))
    #     axis.set_minor_formatter(NoteFormatter(key=key, unicode=unicode, major=False))
    #     axis.set_minor_locator(
    #         mplticker.LogLocator(
    #             base=2.0,
    #             subs=core.interval_frequencies(
    #                 12, fmin=fmin_offset, intervals=intervals, bins_per_octave=12
    #             ),
    #         )
    #     )
    #     axis.set_label_text("Note")

    elif ax_type in ["cqt_hz"]:
        axis.set_major_formatter(LogHzFormatter())
        log_C1 = np.log2(pt.n2f("C1"))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(C_offset,)))
        axis.set_major_locator(mplticker.LogLocator(base=2.0))
        axis.set_minor_formatter(LogHzFormatter(major=False))
        axis.set_minor_locator(
            mplticker.LogLocator(
                base=2.0, subs=C_offset * 2.0 ** (np.arange(1, 12) / 12.0)
            )
        )
        if setlabel:
            axis.set_label_text("Hz")

    elif ax_type == "fft_note":
        axis.set_major_formatter(NoteFormatter())
        # Where is C1 relative to 2**k hz?
        log_C1 = np.log2(pt.n2f("C1"))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(mplticker.SymmetricalLogLocator(axis.get_transform()))
        axis.set_minor_formatter(NoteFormatter(major=False))
        axis.set_minor_locator(
            mplticker.LogLocator(base=2.0, subs=2.0 ** (np.arange(1, 12) / 12.0))
        )
        if setlabel:
            axis.set_label_text("Note")

    # elif ax_type == "fft_svara":
    #     axis.set_major_formatter(SvaraFormatter(Sa=Sa, mela=mela, unicode=unicode))
    #     # Find the offset of Sa relative to 2**k Hz
    #     log_Sa = np.log2(Sa)
    #     sa_offset = 2.0 ** (log_Sa - np.floor(log_Sa))
    #
    #     axis.set_major_locator(
    #         mplticker.SymmetricalLogLocator(
    #             axis.get_transform(), base=2.0, subs=[sa_offset]
    #         )
    #     )
    #     axis.set_minor_formatter(
    #         SvaraFormatter(Sa=Sa, mela=mela, major=False, unicode=unicode)
    #     )
    #     axis.set_minor_locator(
    #         mplticker.LogLocator(
    #             base=2.0, subs=sa_offset * 2.0 ** (np.arange(1, 12) / 12.0)
    #         )
    #     )
    #     axis.set_label_text("Svara")

    elif ax_type in ["mel", "log"]:
        axis.set_major_formatter(mplticker.ScalarFormatter())
        axis.set_major_locator(mplticker.SymmetricalLogLocator(axis.get_transform()))
        if setlabel:
            axis.set_label_text("Hz")

    elif ax_type in ["linear", "hz", "fft"]:
        axis.set_major_formatter(mplticker.ScalarFormatter())
        if setlabel:
            axis.set_label_text("Hz")

    elif ax_type in ["frames"]:
        if setlabel:
            axis.set_label_text("Frames")

    elif ax_type in ["off", "none", None]:
        axis.set_label_text("")
        axis.set_ticks([])

    else:
        raise ParameterError(f"Unsupported axis type: {ax_type}")


def specshow(
        data: np.ndarray,
        *,
        x_coords: np.ndarray | None = None,
        y_coords: np.ndarray | None = None,
        x_axis: str | None = None,
        y_axis: str | None = None,
        sr: float = 22050,
        hop_length: int = 512,
        n_fft: int | None = None,
        win_length: int | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        tuning: float = 0.0,
        bins_per_octave: int = 12,
        key: str = "C:maj",
        Sa: float | int | None = None,
        mela: str | int | None = None,
        thaat: str | None = None,
        auto_aspect: bool = True,
        htk: bool = False,
        unicode: bool = True,
        intervals: str | np.ndarray | None = None,
        unison: str | None = None,
        ax: plt.Axes | None = None,
        cmap: str = 'magma',
        setlabel=True,
        **kwargs,
    ) -> QuadMesh:
    """Display a spectrogram/chromagram/cqt/etc.

    For a detailed overview of this function, see :ref:`sphx_glr_auto_examples_plot_display.py`

    Parameters
    ----------
    data : np.ndarray [shape=(d, n)]
        Matrix to display (e.g., spectrogram)

    sr : number > 0 [scalar]
        Sample rate used to determine time scale in x-axis.

    hop_length : int > 0 [scalar]
        Hop length, also used to determine time scale in x-axis

    n_fft : int > 0 or None
        Number of samples per frame in STFT/spectrogram displays.
        By default, this will be inferred from the shape of ``data``
        as ``2 * (d - 1)``.
        If ``data`` was generated using an odd frame length, the correct
        value can be specified here.

    win_length : int > 0 or None
        The number of samples per window.
        By default, this will be inferred to match ``n_fft``.
        This is primarily useful for specifying odd window lengths in
        Fourier tempogram displays.

    x_axis, y_axis : None or str
        Range for the x- and y-axes.

        Valid types are:

        - None, 'none', or 'off' : no axis decoration is displayed.

        Frequency types:

        - 'linear', 'fft', 'hz' : frequency range is determined by
          the FFT window and sampling rate.
        - 'log' : the spectrum is displayed on a log scale.
        - 'fft_note': the spectrum is displayed on a log scale with pitches marked.
        - 'fft_svara': the spectrum is displayed on a log scale with svara marked.
        - 'mel' : frequencies are determined by the mel scale.
        - 'cqt_hz' : frequencies are determined by the CQT scale.
        - 'cqt_note' : pitches are determined by the CQT scale.
        - 'cqt_svara' : like `cqt_note` but using Hindustani or Carnatic svara
        - 'vqt_fjs' : like `cqt_note` but using Functional Just System (FJS)
          notation.  This requires a just intonation-based variable-Q
          transform representation.

        All frequency types are plotted in units of Hz.

        Any spectrogram parameters (hop_length, sr, bins_per_octave, etc.)
        used to generate the input data should also be provided when
        calling `specshow`.

        Categorical types:

        - 'chroma' : pitches are determined by the chroma filters.
          Pitch classes are arranged at integer locations (0-11) according to
          a given key.

        - `chroma_h`, `chroma_c`: pitches are determined by chroma filters,
          and labeled as svara in the Hindustani (`chroma_h`) or Carnatic (`chroma_c`)
          according to a given thaat (Hindustani) or melakarta raga (Carnatic).

        - 'chroma_fjs': pitches are determined by chroma filters using just
          intonation.  All pitch classes are annotated.

        - 'tonnetz' : axes are labeled by Tonnetz dimensions (0-5)
        - 'frames' : markers are shown as frame counts.

        Time types:

        - 'time' : markers are shown as milliseconds, seconds, minutes, or hours.
                Values are plotted in units of seconds.
        - 'h' : markers are shown as hours, minutes, and seconds.
        - 'm' : markers are shown as minutes and seconds.
        - 's' : markers are shown as seconds.
        - 'ms' : markers are shown as milliseconds.
        - 'lag' : like time, but past the halfway point counts as negative values.
        - 'lag_h' : same as lag, but in hours, minutes and seconds.
        - 'lag_m' : same as lag, but in minutes and seconds.
        - 'lag_s' : same as lag, but in seconds.
        - 'lag_ms' : same as lag, but in milliseconds.

        Rhythm:

        - 'tempo' : markers are shown as beats-per-minute (BPM)
            using a logarithmic scale.  This is useful for
            visualizing the outputs of `feature.tempogram`.

        - 'fourier_tempo' : same as `'tempo'`, but used when
            tempograms are calculated in the Frequency domain
            using `feature.fourier_tempogram`.

    x_coords, y_coords : np.ndarray [shape=data.shape[0 or 1]]
        Optional positioning coordinates of the input data.
        These can be use to explicitly set the location of each
        element ``data[i, j]``, e.g., for displaying beat-synchronous
        features in natural time coordinates.

        If not provided, they are inferred from ``x_axis`` and ``y_axis``.

    fmin : float > 0 [scalar] or None
        Frequency of the lowest spectrogram bin.  Used for Mel, CQT, and VQT
        scales.

        If ``y_axis`` is `cqt_hz` or `cqt_note` and ``fmin`` is not given,
        it is set by default to ``note_to_hz('C1')``.

    fmax : float > 0 [scalar] or None
        Used for setting the Mel frequency scales

    tuning : float
        Tuning deviation from A440, in fractions of a bin.

        This is used for CQT frequency scales, so that ``fmin`` is adjusted
        to ``fmin * 2**(tuning / bins_per_octave)``.

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave.  Used for CQT frequency scale.

    key : str
        The reference key to use when using note axes (`cqt_note`, `chroma`).

    Sa : float or int
        If using Hindustani or Carnatic svara axis decorations, specify Sa.

        For `cqt_svara`, ``Sa`` should be specified as a frequency in Hz.

        For `chroma_c` or `chroma_h`, ``Sa`` should correspond to the position
        of Sa within the chromagram.
        If not provided, Sa will default to 0 (equivalent to `C`)

    mela : str or int, optional
        If using `chroma_c` or `cqt_svara` display mode, specify the melakarta raga.

    thaat : str, optional
        If using `chroma_h` display mode, specify the parent thaat.

    intervals : str or array of floats in [1, 2), optional
        If using an FJS notation (`chroma_fjs`, `vqt_fjs`), the interval specification.

        See `core.interval_frequencies` for a description of supported values.

    unison : str, optional
        If using an FJS notation (`chroma_fjs`, `vqt_fjs`), the pitch name of the unison
        interval.  If not provided, it will be inferred from `fmin` (for VQT display) or
        assumed as `'C'` (for chroma display).

    auto_aspect : bool
        Axes will have 'equal' aspect if the horizontal and vertical dimensions
        cover the same extent and their types match.

        To override, set to `False`.

    htk : bool
        If plotting on a mel frequency axis, specify which version of the mel
        scale to use.

            - `False`: use Slaney formula (default)
            - `True`: use HTK formula

        See `core.mel_frequencies` for more information.

    unicode : bool
        If using note or svara decorations, setting `unicode=True`
        will use unicode glyphs for accidentals and octave encoding.

        Setting `unicode=False` will use ASCII glyphs.  This can be helpful
        if your font does not support musical notation symbols.

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    **kwargs : additional keyword arguments
        Arguments passed through to `matplotlib.pyplot.pcolormesh`.

        By default, the following options are set:

            - ``rasterized=True``
            - ``shading='auto'``
            - ``edgecolors='None'``

        The ``cmap`` option if not provided, is inferred from data automatically.
        Set ``cmap=None`` to use matplotlib's default colormap.

    Returns
    -------
    colormesh : `matplotlib.collections.QuadMesh`
        The color mesh object produced by `matplotlib.pyplot.pcolormesh`

    See Also
    --------
    cmap : Automatic colormap detection
    matplotlib.pyplot.pcolormesh

    Examples
    --------
    Visualize an STFT power spectrum using default parameters

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=15)
    >>> fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    >>> img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
    ...                                sr=sr, ax=ax[0])
    >>> ax[0].set(title='Linear-frequency power spectrogram')
    >>> ax[0].label_outer()

    Or on a logarithmic scale, and using a larger hop

    >>> hop_length = 1024
    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),
    ...                             ref=np.max)
    >>> librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
    ...                          x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-frequency power spectrogram')
    >>> ax[1].label_outer()
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    """
    if np.issubdtype(data.dtype, np.complexfloating):
        warnings.warn(
            "Trying to display complex-valued input. " "Showing magnitude instead.",
            stacklevel=2,
        )
        data = np.abs(data)

    kwargs.setdefault("cmap", cmap)
    # kwargs.setdefault("cmap", cmap(data))
    kwargs.setdefault("rasterized", True)
    kwargs.setdefault("edgecolors", "None")
    kwargs.setdefault("shading", "auto")

    all_params = dict(
        kwargs=kwargs,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        tuning=tuning,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=win_length,
        key=key,
        htk=htk,
        unicode=unicode,
        intervals=intervals,
        unison=unison,
    )

    # Get the x and y coordinates
    y_coords = __mesh_coords(y_axis, y_coords, data.shape[0], **all_params)
    x_coords = __mesh_coords(x_axis, x_coords, data.shape[1], **all_params)

    axes = __check_axes(ax)

    out = axes.pcolormesh(x_coords, y_coords, data, **kwargs)

    if ax is None:
        plt.sci(out)

    # Set up axis scaling
    __scale_axes(axes, x_axis, "x")
    __scale_axes(axes, y_axis, "y")

    # Construct tickers and locators
    __decorate_axis(
        axes.xaxis,
        x_axis,
        key=key,
        Sa=Sa,
        mela=mela,
        thaat=thaat,
        unicode=unicode,
        fmin=fmin,
        unison=unison,
        intervals=intervals,
        bins_per_octave=bins_per_octave,
        n_bins=len(x_coords),
        setlabel=setlabel
    )
    __decorate_axis(
        axes.yaxis,
        y_axis,
        key=key,
        Sa=Sa,
        mela=mela,
        thaat=thaat,
        unicode=unicode,
        fmin=fmin,
        unison=unison,
        intervals=intervals,
        bins_per_octave=bins_per_octave,
        n_bins=len(y_coords),
        setlabel=setlabel
    )

    # If the plot is a self-similarity/covariance etc. plot, square it
    #if __same_axes(x_axis, y_axis, axes.get_xlim(), axes.get_ylim()) and auto_aspect:
    #    axes.set_aspect("equal")

    return out
