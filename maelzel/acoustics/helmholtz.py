"""
Resonators
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
from math import pi, sqrt


C = 343.


def resonantFrequency(V:float, A:float= pi * 0.01 ** 2, L:float=0.05, c:float=C) -> float:
    """
    resonance frequency of a helmholtz-resonator (in Hz)

    Args:
        V: static volume of the cavity          (m^3)
        A: area of the neck                     (m^2)
        L: is the length of the neck            (m)
        c: speed of propagation                 (m/s)

    Returns:
        the resonance freq. of the resonator

    NB: magnitudes given in SI units (m, s, Hz)
    """
    a = sqrt(A / (L*pi))    # radius of the neck if the neck was a tube
    L1 = L + 1.7*a
    return c / pi*2. * sqrt(A/(V*L1))


def resonatorVolume(f:float, A=pi * 0.01 ** 2, L=0.05, c:float=C) -> float:
    """
    Volume of a helmholz resonator of the given frequency (in m^3)
    the volume is given in m^3

    Args:
        f: resonance frequency                  (Hz)
        A: area of the neck                     (m)
           NB: circular area= PI * r^2. r = sqrt(A/PI)
        L: length of the neck                   (m)
        c: speed of propagation                 (m/s)

    Returns:
        the volume
    """
    a = sqrt(A / (L*pi))    # radius of the neck if the neck was a tube
    L1 = L + 1.7*a
    return A/((f*pi*2/c)**2 * L1)


def ductResonantFrequency(L:float, radius:float, n:int, c:float=C) -> float:
    """
    resonance frequency of an unflanged (open) duct

    Args:
        L: length of the duct
        radius: radius of the duct
        n: 1,2,3. vibration mode
        c: sound speed

    Returns:
        the resonance frequency of the open duct of given length and radius

    from 'Acoustic Filters--David Russell.pdf'
    """
    f_n = n * c / (2*(L + 0.61*radius))
    return f_n


def ductLength(freq:float, radius:float, n:int=1, c:float=C) -> float:
    """
    calculate the length of the duct based on the observed resonant frequency

    Args:
        freq: resonant frequency measured
        radius: radius of the duct
        n: harmonic number corresponding to the frequency observed (1 is the fundamental)
        c: propagation speed of the medium

    Returns:
        the length of the unflanged duct

    Example::

        # the most prominent resonance measured is 513 Hz, but in the spectrogram
        # it is observed that this frequency corresponds to the second harmonic
        >>> ductLength(513, 0.07, n=2)
    """
    L = n * c / (2*freq) - (0.61*radius)
    return L


def sphereVolume(r:float) -> float:
    """
    The volume of a sphere with given radius
    """
    return 4/3. * pi * r**3


def sphereRadius(V:float) -> float:
    """
    The radius of a sphere with given volume
    """
    return (V/(4/3.*pi)) ** (1/3.)
