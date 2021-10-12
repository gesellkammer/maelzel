from __future__ import annotations
import numpy as _np
from scipy import signal as _signal
from scipy.fftpack import hilbert
from emlib.misc import returns_tuple as _returns_tuple
import math
from math import pi
import warnings


@_returns_tuple("b0 b1 b2 a0 a1 a2", "SOSCoeffs")
def biquad_coeffs(filtertype, fc, param, dbgain=0, fs=48000, normalized=True):
    """various iir biquad filters for audio processing
    arguments:
    filtertype = options: "lpf", "hpf", "bpf"
    fc     = filter frequency
    param  = Depending on filter, acts as Q, bandwidth, or shelf slope.
    dbgain = gain  in db for peaking/shelving filters (defaults to 0)
    fs     = sampling frequency (defaults to 48000)
    """
    sel = filtertype
    aval = 10**(dbgain/40.0)
    w0 = 2.0 * pi * fc / float(fs)
    cosw0 = math.cos(w0)
    sinw0 = math.sin(w0)
    alpha = sinw0 / (2.0*param)

    if sel == "lpf":
        b0 = (1 - cosw0)/2.0
        b1 = 1 - cosw0
        b2 = (1 - cosw0)/2.0
        a0 = 1 + alpha
        a1 = -2 * cosw0
        a2 = 1 - alpha
 
    elif sel == "hpf":
        b0 = (1 + cosw0)/2.0
        b1 = -1 - cosw0
        b2 = (1 + cosw0)/2.0
        a0 = 1 + alpha
        a1 = -2 * cosw0
        a2 = 1 - alpha
 
    elif sel == "bpf":
        b0 = sinw0 / 2.0
        b1 = 0.0
        b2 = -1 * sinw0 / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cosw0
        a2 = 1.0 - alpha
         
    else:
        raise NameError("invalid filter type")

    if normalized:
        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0
        return (b0, b1, b2, a0, a1, a2)
    else:
        return (b0, b1, b2, a0, a1, a2)


def sos6(samples, b0, b1, b2, a0, a1, a2):
    xx = samples
    yy = [0]*len(samples)
    for n in range(2, len(xx)):
        yy[n] = b1 * xx[n-1] + b2 * xx[n-2] - a1 * yy[n-1] - a2 * yy[n-2]
    return yy


def biquad(xx, sel, fc, param, dbgain=0, fs=48000):
    """various iir biquad filters for audio processing

    xx     : input signal to be filtered
    sel    : filter type
             options: "lpf", "hpf", "bpf"
    fc     : filter frequency
    param  : Depending on filter, acts as Q, bandwidth, or shelf slope.
    dbgain : gain  in db for peaking/shelving filters (defaults to 0)
    fs     : sampling frequency

    Example
    =======

    # TODO: create an example
    """   
    b0, b1, b2, a0, a1, a2 = biquad_coeffs(sel, fc, param, dbgain, fs, normalized=True)
    return sos6(xx, b0, b1, b2, a0, a1, a2)


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    return the b and a coefficients to design the digital filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = _signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = _signal.lfilter(b, a, data)
    return y


def freq_response(bb, aa, fs, worN=None):
    """
    bb, aa: the filter coefficients
    fs: sampling rate
    worN: see scipy._signal.freqz

    Example
    =======

    >>> bb, aa = butter_bandpass(50, 4000, 44100, order=3)
    >>> xx, yy = freq_response(bb, aa, worN=200)
    >>> pyplot.plot(xx, yy)
    """
    from scipy.signal import freqz
    
    w, h = freqz(bb, aa, worN=worN)
    return (fs * 0.5 / pi) * w, _np.abs(h)


def gen_sine(freq=440, amp=1, iphase=0, dur=1, sr=48000):
    """
    generate a sine tone

    y = A * math.sin(2pi * freq * t + ph) = A * math.sin(w * t + ph)
    """
    ts = _np.linspace(0, dur, sr, endpoint=False)
    samples = _np.sin(freq * (pi*2) * ts + iphase) * amp
    return samples


def gen_square(freq=440, duty=0.5, amp=1, iphase=0, dur=1, sr=48000):
    ts = _np.linspace(0, dur, sr, endpoint=False)
    phase = iphase + ts * (freq * pi * 2)
    return _signal.waveforms.square(phase, duty) * amp


def gen_sine_mod(freq=440, amp=1, iphase=0, dur=1, sr=48000):
    """
    the same as gen_sin but freq and amp can be modulated by a bpf
    """
    from bpf4 import bpf
    freq = bpf.asbpf(freq)
    amp = bpf.asbpf(amp)
    ts = _np.linspace(0, dur, sr, endpoint=False)
    freqs = freq.map(ts)
    amps = amp.map(ts)
    samples = _np.math.sin(freqs*(pi*2) * ts + iphase) * amps
    return samples


def prime_power_delays(N, pathmin, pathmax, sr):
    """
    Calculate the delay lines of a reverberator

    @param N: the number of delay lines (>= 3)
    @param pathmin: minimum acoustic ray length in the reverberator (in meters)
    @param pathmax: max. acoustic ray lngth (meters) - think "room size"
    @param sr: sr
    @return: a list of N elements, each being the delay time in samples
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53]
    # maxdel = 8192
    # ppbs = [13, 8, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2]
    c = 343.0
    dmin = sr * pathmin / c
    dmax = sr * pathmax / c
    delayvals = []
    for i in range(N):
        delay_in_samples = dmin * (dmax/dmin) ** (i/float(N-1))
        ppwr = int(0.5+math.log(delay_in_samples))/math.log(primes[i])
        print(i, primes[i], ppwr)
        delayval = primes[i] ** ppwr
        delayvals.append(delayval)
    return delayvals


def prime_power_delay(i, N, pathmin, pathmax, sr):
    """
    i: which delay (0 to N-1)

    for the rest of the parameters, see prime_power_delays
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 43, 47, 53]
    # maxdel = 8192
    # ppbs = [13, 8, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2]
    c = 343.0
    dmin = sr * pathmin / c
    dmax = sr * pathmax / c
    i = N
    delay_in_samples = dmin * (dmax/dmin) ** (i/float(N-1))
    ppwr = int(0.5+math.log(delay_in_samples))/math.log(primes[i])
    delayval = primes[i] ** ppwr
    return delayval


def decay2feedback(decaydur, delaytime):
    """
    Returns the feedback needed to decrease its volume in 60 dB 
    in the given duration for the indicated delaytime
    """
    return math.exp(-4.605170185988091 * delaytime / decaydur)


def feedback2decay(feedback, delaytime):
    """
    The decay duration needed for a comb filter to decrease its 
    volume in 60 dB given the indicated feedback level (0-1) and
    delaytime
    """
    return -4.605170185988091 * delaytime / math.log(feedback)


def feedback2delaytime(feedback, decaytime):
    return decaytime * math.log(feedback) / -4.605170185988091


def rms(samples):
    """
    samples: a numpy array of samples
    """
    return math.sqrt((samples ** 2).sum() / float(samples.size))


def crest_factor(samples):
    maxvalue = max(samples.max(), -(samples.min()))
    return maxvalue / float(rms(samples))


def lowpass_cheby2(samples, freq, sr, maxorder=12):
    """
    Cheby2-Lowpass Filter

    Filter samples by passing samples only below a certain frequency.
    The main purpose of this cheby2 filter is downsampling.
    
    This method will iteratively design a filter, whose pass
    band frequency is determined dynamically, such that the
    values above the stop band frequency are lower than -96dB.

    samples : Data to filter, type numpy.ndarray.
    freq    : The frequency above which signals are attenuated
              with 95 dB
    sr      : Sampling rate in Hz.
    maxorder: Maximal order of the designed cheby2 filter
    """
    b, a, freq_passband = lowpass_cheby2_coeffs(freq, sr, maxorder)
    return _signal.lfilter(b, a, samples)


@_returns_tuple("a b freq_passband")
def lowpass_cheby2_coeffs(freq, sr, maxorder=12):
    """
    freq    : The frequency above which signals are attenuated
              with 95 dB
    sr      : Sampling rate in Hz.
    maxorder: Maximal order of the designed cheby2 filter
    """
    nyquist = sr * 0.5
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    ws = freq / nyquist    # stop band frequency
    wp = ws                # pass band frequency
    # raise for some bad scenarios
    if ws > 1:
        ws = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    while True:
        if order <= maxorder:
            break
        wp = wp * 0.99
        order, wn = _signal.cheb2ord(wp, ws, rp, rs, analog=False)
    b, a = _signal.cheby2(order, rs, wn, btype='low', analog=False, output='ba')
    return (b, a, wp*nyquist)
    

def envelope(data):
    """
    Envelope of a function.

    Computes the envelope of the given function. The envelope is determined by
    adding the squared amplitudes of the function and it's Hilbert-Transform
    and then taking the square-root. (See [Kanasewich1981]_)
    The envelope at the start/end should not be taken too seriously.

    data: Data to make envelope (numpy.ndarray)
    
    Returns --> Envelope of input data.

    NB: via obspy
    """
    hilb = hilbert(data)
    data = (data ** 2 + hilb ** 2) ** 0.5
    return data


def tau2pole(tau, sr):
    """
    tau: desired smoothing time in seconds
    sr: sampling rate
    """
    return math.exp(-1.0/(tau*sr))


def compressor_makeupgain(thresh, ratio, refdb=0):
    """
    Returns the makeup-gain for the given thresh and ratio
    to achieve 0 dB given a 0 dB output

    thresh: threshold, in dB
    ratio: ratio of compression
    refdb: reference, normally 0 dB
    """
    return refdb - (1.0/ratio) * (refdb - thresh) - thresh


try:
    from fastdsp import sos6
except ImportError:
    pass

