from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy.typing as npt
    from typing import Sequence, Callable
    from matplotlib.axes import Axes


class Quantile1d:
    """
    A class representing a 1-dimensional quantile calculation.

    Args:
        data: The data to calculate the quantile for
    """
    def __init__(self, data: npt.ArrayLike):
        arr = np.asarray(data)
        if not arr.ndim == 1:
            raise ValueError("data must be 1-dimensional")
        if not arr.shape[0] > 0:
            raise ValueError("data must be non-empty")
        self.data = arr.copy()
        self.data.sort()
        self.size = len(arr)

    def quantile(self, value: float) -> float:
        """
        Quantile corresponding to the given value of the data.

        Args:
            value: The value to calculate the quantile for

        Returns:
            The quantile corresponding to the given value (a value between 0 and 1)
        """
        if value < self.data[0]:
            return 0.
        elif value > self.data[-1]:
            return 1.
        idx = int(np.searchsorted(self.data, value))
        if idx == 0:
            return 0.
        val1 = self.data[idx]
        if val1 > value:
            val0 = self.data[idx - 1]
            idx = (value - val0) / (val1 - val0)
        return idx / self.size

    def value(self, q: float, method: str = 'linear') -> float:
        """
        Calculate the value corresponding to the given quantile of the data.

        Args:
            q: The quantile to calculate (between 0 and 1, including both 0 and 1)
            method: The method to use for interpolation, one of 'linear', 'nearest', 'floor', 'ceil'

        Returns:
            The value corresponding to the given quantile
        """
        if not (0 <= q <= 1):
            raise ValueError("invalid q")
        idx = q * self.size
        idx0 = min(int(idx), self.size - 1)
        v0 = float(self.data[idx0])
        if idx0 == idx:
            return v0
        idx1 = idx0 + 1
        if idx1 >= self.size:
            return float(self.data[-1])
        v1 = float(self.data[idx1])
        if method == 'linear':
            return v0 + (v1 - v0) * (idx - idx0)
        elif method == 'nearest':
            return v0 if (idx - idx0) < (idx1 - idx) else v1
        elif method == 'floor':
            return v0
        elif method == 'ceil':
            return v1
        else:
            raise ValueError(f"Invalid method {method}")

    def plot(self, axes=None, show=True) -> Axes:
        import matplotlib.pyplot as plt
        if axes is None:
            fig, axes = plt.subplots()
        quantiles = np.linspace(0, 1, 100)
        values = [self.value(q) for q in quantiles]
        axes.plot(quantiles, values)
        if show:
            plt.show()
        return axes


class Histogram:
    """
    A class representing a Histogram and mapping between percentile and values

    .. note:: Deprecated!

    Args:
        values: the values to evaluate
        numbins: the number of bins
    """
    def __init__(self, values: Sequence[int | float] | np.ndarray, numbins: int = 20):
        import warnings
        warnings.warn("This class is deprecated and will be removed in a future version.")

        valuearray = np.asarray(values)
        counts, edges = np.histogram(valuearray, bins=numbins)
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
        return float(np.interp(percentile, self._percentiles, self.edges))

    def __repr__(self):
        return f"Histogram(numbins={self.numbins}, edges={self.edges})"

    def plot(self, axes=None) -> Axes:
        """
        Plot this histogram

        Args:
            axes: the matplotlib axes to use

        Returns:
            the Axes used. It will be the value passed, if given, or the
            created Axes if None
        """
        import matplotlib.pyplot as plt
        if axes is None:
            fig, axes = plt.subplots()
        plt.stairs(self.counts, self.edges)
        return axes


def weightedHistogram(values: npt.ArrayLike,
                      weights: npt.ArrayLike,
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
        from emlib import mathlib
        relthresholds = mathlib.exponcurve(numbins+1, distribution, 0, 0, 1, 1)
    elif callable(distribution):
        relthresholds = [distribution(x) for x in np.linspace(0, 1, numbins+1)]
    else:
        raise TypeError(f"Expected a float or a bpf, got {distribution}")
    weightsarr = np.asarray(weights)
    valuesarr = np.asarray(values)
    totalweight = weightsarr.sum()
    sortedindexes = np.argsort(valuesarr)
    sortedvalues = valuesarr[sortedindexes]
    sortedweights = weightsarr[sortedindexes]

    binweight = 0
    edges = [float(sortedvalues[0])]
    binindex = 0
    threshold = relthresholds[binindex]
    for weight, value in zip(sortedweights, sortedvalues):
        binweight += weight
        if binweight / totalweight > threshold:
            edges.append(float(value))
            binindex += 1
            if binindex < numbins:
                threshold = relthresholds[binindex]
    if edges[-1] < sortedvalues[-1]:
        edges.append(float(sortedvalues[-1]))
    return edges
