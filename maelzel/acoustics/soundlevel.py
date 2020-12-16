from __future__ import division as _division
from math import log, log10, exp


def soundpressure_to_soundlevel(soundpressure, p0=0.00002):
    # type: (float, float) -> float
    """
    convert soundpressure in Pascal to sound level in dB (dBSPL)

    Lp(dBSPL) = 20 * log10(p/p0)

    p0: threshold of hearing, 0.00002 Pa (20uPa)
    """
    return 20 * log10(soundpressure/p0)


def soundlevel_to_soundpressure(soundlevel, p0=0.00002):
    # type: (float, float) -> float
    """
    convert sound-level in dB to sound-pressure in Pascal

    p = p0 * e^(1/20*Lp*log10(10))

    p0: threshold of hearing, 0.00002 Pa (20uPa) 
    """
    return p0 * exp(1/20*soundlevel*log(10))


def soundintensity_to_soundlevel(soundintensity, I0=10e-12):
    # type: (float, float) -> float
    """
    convert sound intensity (in W/m2) to sound-level (dBSIL)

    Li(dbSIL) = 10 * log10(I/I0)
    """
    return 10 * log10(soundintensity/I0)
    

def soundlevel_to_soundintensity(soundlevel, I0=10e-12):
    # type: (float, float) -> float
    """
    convert soundlevel (dBSPI) to sound-intensity in W/m2

    I = I0 * e^(1/10*Li*log10(10))

    I0: sound-intensity at the threshold of hearing
    """
    return I0 * exp(1/10. * soundlevel * log10(10))


def soundpressure_to_soundintensity(soundpressure, p0=0.00002, I0=10e-12):
    # type: (float, float, float) -> float
    L = soundpressure_to_soundlevel(soundpressure, p0)
    I = soundlevel_to_soundintensity(L, I0)
    return I


def soundintensity_to_soundpressure(soundintensity, p0=0.00002, I0=10e-12):
    # type: (float, float, float) -> float
    L = soundintensity_to_soundlevel(soundintensity, I0)
    p = soundlevel_to_soundpressure(L, p0)
    return p


def amplitude_to_soundlevelfullscale(amp):
    # type: (float) -> float
    """
    convert linear amplitude (as used in DSP, 0 to 1) to dB 
    (~ -120 to 0 depending on the bitrate)

    dBFS = 20 * log10(amp)
    """
    return 20 * log10(amp)


def soundlevelfullscale_to_amplitude(soundlevel):
    # type: (float) -> float
    """
    convert soundlevel fullscale (as used in digital audio) to linear ampitude (0-1)
    """
    return exp(1/20. * soundlevel * log(10))


def dL_coherent_sources(numsources):
    # type: (int) -> float
    """
    calculates the variation in intesity for the sum of numsources equal loud sources

    Example
    -------

    # how much will the sound level increase in an orchestra of 12 first violins,
    # with respect to a solo

    >>> dL_coherent_sources(12)
    10.79181246047625

    # the result will be 10.8 dB higher as the solo violin alone
    """
    return 10*log10(numsources)


I2L = soundintensity_to_soundlevel
L2p = soundlevel_to_soundpressure
p2L = soundpressure_to_soundlevel
L2I = soundlevel_to_soundintensity
I2p = soundintensity_to_soundpressure
p2I = soundpressure_to_soundintensity
amp2db = amplitude_to_soundlevelfullscale
db2amp = soundlevelfullscale_to_amplitude
