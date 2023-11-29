from __future__ import annotations
from numbers import Number

import bpf4
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from typing import Sequence, Callable


class Histogram:
    """
    A class representing a Histogram and mapping between percentile and values

    Args:
        values: the values to evaluate
        numbins: the number of bins
    """
    def __init__(self, values: Sequence[Number] | np.ndarray, numbins: int = 20):
        counts, edges = np.histogram(values, bins=numbins)
        self.counts = counts
        """How many values lie within each bin"""

        self.edges = edges
        """The edges of the bins. len(edges) == numbins + 1. edges[0] == min(values), edges[-1] == max(values)"""

        self.numbins = numbins
        """The number of bins"""

        self.values = values
        """The values passed to this Histogram"""

        self._percentiles = np.linspace(0, 1, len(edges))

    def valueToPercentile(self, value: float) -> float:
        """
        Convert a value to its percentile within the distribution

        Args:
            the value to convert

        Returns:
            the percentile, a number between 0 and 1
        """
        return float(np.interp(value, self.edges, self._percentiles))

    def percentileToValue(self, percentile: float) -> float:
        """
        Interpolate a value corresponding to the given percentile

        Args:
            percentile: a percentile in the range [0, 1]

        Returns:
            the corresponding value. The accuracy will depend on the number
            of bins within the histogram

        """
        return np.interp(percentile, self._percentiles, self.edges)

    def __repr__(self):
        return f"Histogram(numbins={self.numbins}, edges={self.edges})"

    def plot(self, axes=None) -> plt.Axes:
        import matplotlib.pyplot as plt
        if axes is None:
            fig, axes = plt.subplots()
        plt.stairs(self.counts, self.edges)
        return axes


def weightedHistogram(values: Sequence[Number],
                      weights: list[Number] | np.ndarray,
                      numbins: int,
                      distribution: float | Callable[[float], float] = 1.0
                      ) -> list[float]:
    """
    Find the bin edges so that the weights in each bin follow the given distribution

    **Use case**: Imaging a list of partials, each with its own energy (energy being the
    integral of amplitude over the duration of the partial). Find the frequency bins so that
    the partials within each bin, together, have the same total energy (if distribution==1.0,
    meaning a linear distribution).

    Args:
        weights: list of weights
        values: codependent dimension, same size as weights
        numbins: number of bins
        distribution: either an exponent or a bpf between (0, 0) and (1, 1). When an exponent is given,
            a value of 1.0 will result in a linear distribution (all bins contain the same total weight),
            a value [0, 1) will result in the lower bins containing more weight than the higher bins,
            and a value higher than 1 will distribute more weight to the higher bins

    Returns:
        the bin edges

    """
    if isinstance(distribution, (int, float)):
        curve = bpf4.expon(0, 0, 1, 1, exp=distribution)
    elif isinstance(distribution, bpf4.BpfInterface):
        curve = distribution
    elif callable(distribution):
        curve = bpf4.asbpf(distribution, bounds=(0, 1))
    else:
        raise TypeError(f"Expected a float or a bpf, got {distribution}")

    relthresholds = curve.map(numbins+1)[1:]
    weightsarr = np.asarray(weights)
    valuesarr = np.asarray(values)
    totalweight = weightsarr.sum()
    sortedindexes = np.argsort(valuesarr)
    sortedvalues = valuesarr[sortedindexes]
    sortedweights = weightsarr[sortedindexes]

    binweight = 0
    edges = [sortedvalues[0]]
    binindex = 0
    threshold = relthresholds[binindex]
    for weight, value in zip(sortedweights, sortedvalues):
        binweight += weight
        if binweight / totalweight > threshold:
            edges.append(value)
            binindex += 1
            if binindex < numbins:
                threshold = relthresholds[binindex]
    if edges[-1] < sortedvalues[-1]:
        edges.append(float(sortedvalues[-1]))
    return edges


