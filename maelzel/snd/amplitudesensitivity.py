"""
Amplitude compensation curves
"""
from __future__ import annotations
from math import sqrt
import numpy as np


_AMPCOMP_MINLEVEL = -0.1575371167435


class AnsiWeightingCurveA:
    """
    ANSI A-Weighting Curve

    Higher frequencies are normally perceived as louder. Following the
    measurements by Fletcher and Munson, the ANSI standard describes a function for loudness vs. frequency.
    Note that this curve is only valid for standardized amplitude.

    The implementation is directly ported from SuperCollider's AmpCompA UGen
    (https://doc.sccode.org/Classes/AmpCompA.html)

    Args:
        root: Root freq relative to which the curve is calculated (usually lowest freq).
        minamp: Amplitude at the minimum point of the curve (around 2512 Hz).
        rootamp: Amplitude at the root frequency.

    """

    def __init__(self, root=0, minamp=0.32, rootamp=1.0, minlevel=_AMPCOMP_MINLEVEL):
        self.root = root
        """Root freq. relative to which the curve is calculated (usually lowest freq)"""

        self.minamp = minamp
        """Amplitude at the min. point of the curve (around 2512 Hz)"""

        self.rootamp = rootamp
        """Amplitude at the root freq."""

        rootlevel = acurveAmplitudeCompensation(root)
        self.scale = (rootamp - minamp) / (rootlevel - minlevel)
        """The scaling factor of the curve"""

        self.offset = minamp - self.scale * minlevel
        """Offset of the curve"""

    def __call__(self, freq: float) -> float:
        """
        Calculate the amplitude compensation at the given frequency

        Args:
            freq: the frequency to evaluate the curve at

        Returns:
            the amplitude compensation factor

        """
        return acurveAmplitudeCompensation(freq) * self.scale + self.offset

    def map(self, freqs: np.ndarray) -> np.ndarray:
        """
        Calculate the amplitude compensation for an array of frequencies

        Args:
            freqs: the frequencies to evaluate the curve at

        Returns:
            the array of amplitude compensations
        """
        return acurveAmplitudeCompensationArray(freqs) * self.scale + self.offset

    def plot(self, minfreq=20, maxfreq=20000, numpoints=500):
        """
        Plot this curve

        Args:
            minfreq: the min. frequency to plot
            maxfreq: the max. frequency to plot
            numpoints: the number of points to plot

        """
        import matplotlib.pyplot as plt
        freqs = np.logspace(minfreq, maxfreq, num=numpoints)
        levels = self.map(freqs)
        plt.plot(freqs, levels)


def acurveAmplitudeCompensationArray(freqs: np.ndarray) -> np.ndarray:
    """
    Calculate the ampltude compensation for freq according to an A-Curve

    Argss:
        freqs: the frequencies to evaluate the curve at

    Returns:
        the resulting amplitude compensations
    """
    k = 3.5041384 * 10e15
    c1 = 20.598997 ** 2
    c2 = 107.65265 ** 2
    c3 = 737.86223 ** 2
    c4 = 12194.217 ** 2
    r = freqs * freqs
    r2 = r*r
    level = r2*r2 * k
    n1 = r + c1
    n2 = r + c4
    levels = level / (n1 * n1 * (c2 + r) * (c3 + r) * n2 * n2)
    return 1 - np.sqrt(levels)


def acurveAmplitudeCompensation(freq: float) -> float:
    """
    Calculate the ampltude compensation for freq according to an A-Curve

    Args:
        freq: the frequency to evaluate the curve at

    Returns:
        the amplitude compensation factor at the given frequency

    """
    k = 3.5041384 * 10e15
    c1 = 20.598997 ** 2
    c2 = 107.65265 ** 2
    c3 = 737.86223 ** 2
    c4 = 12194.217 ** 2
    r = freq * freq
    r2 = r*r
    level = k * r2*r2
    n1 = c1 + r
    n2 = c4 + r
    level = level / (n1*n1*(c2+r) * (c3+r) * n2*n2)
    return 1 - sqrt(level)



defaultCurve = AnsiWeightingCurveA()


def ampcomp(freq: float) -> float:
    return defaultCurve(freq)


def ampcomparray(freqs: np.ndarray | list[float]) -> np.ndarray:
    return defaultCurve.map(np.asarray(freqs))


# Original code from supercollider AmpCompA

"""

const double AMPCOMP_K = 3.5041384 * 10e15;
const double AMPCOMP_C1 = 20.598997 * 20.598997;
const double AMPCOMP_C2 = 107.65265 * 107.65265;
const double AMPCOMP_C3 = 737.86223 * 737.86223;
const double AMPCOMP_C4 = 12194.217 * 12194.217;
const double AMPCOMP_MINLEVEL = -0.1575371167435;

double AmpCompA_calcLevel(double freq)
{
	double r = freq * freq;
	double level = (AMPCOMP_K * r * r * r * r);
	double n1 = AMPCOMP_C1 + r;
	double n2 = AMPCOMP_C4 + r;
	level = level / (
						n1 * n1 *
						(AMPCOMP_C2 + r) *
						(AMPCOMP_C3 + r) *
						n2 * n2
					);
	level = 1. - sqrt(level);
	return level;
}

void AmpCompA_next(AmpCompA *unit, int inNumSamples)
{
	float *out = ZOUT(0);
	float *freq = ZIN(0);

	double scale = unit->m_scale;
	double offset = unit->m_offset;

	LOOP1(inNumSamples,
		ZXP(out) = AmpCompA_calcLevel(ZXP(freq)) * scale + offset;
	);
}

void AmpCompA_Ctor(AmpCompA* unit)
{
	double rootFreq = ZIN0(1);
	double rootLevel = AmpCompA_calcLevel(rootFreq);
	float minLevel = ZIN0(2);
	unit->m_scale = (ZIN0(3) - minLevel) / (rootLevel - AMPCOMP_MINLEVEL);
	unit->m_offset = minLevel - unit->m_scale * AMPCOMP_MINLEVEL;

	SETCALC(AmpCompA_next);
	AmpCompA_next(unit, 1);
}
"""