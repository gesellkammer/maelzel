from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
import itertools


if TYPE_CHECKING:
    import numpy.typing as npt
    from typing import Callable
    from matplotlib.axes import Axes


class Quantile1d:
    """
    A class representing a 1-dimensional quantile calculation.

    Args:
        data: The data to calculate the quantile for

    Examples
    ~~~~~~~~

    TODO
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
            The returned value indicates the relative position of the given value
            within the data. A value corresponding to the median returns 0.5
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

    def plot(self, feature="value", n=200, axes=None, show=True) -> Axes:
        import matplotlib.pyplot as plt
        if axes is None:
            fig, axes = plt.subplots()
        if feature == "value":
            quantiles = np.linspace(0, 1, n)
            values = [self.value(q) for q in quantiles]
            axes.plot(quantiles, values)
        elif feature == 'quantile':
            values = np.linspace(self.data[0], self.data[-1], n)
            quantiles = [self.quantile(v) for v in values]
            axes.plot(values, quantiles)
        else:
            raise ValueError(f"One of 'value' or 'quantile', got {feature}")
        if show:
            plt.show()
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
        values: list of values
        weights: list of weights, same size as values
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

    accumweight = 0.
    edges = [float(sortedvalues[0])]
    binindex = 1
    absthresholds = [float(thresh * totalweight) for thresh in relthresholds]
    absthresholds[-1] *= 100
    threshold = absthresholds[binindex]
    for weight, value in zip(sortedweights, sortedvalues):
        accumweight += float(weight)
        if accumweight > threshold:
            binindex += 1
            edges.append(float(value))
            threshold = absthresholds[binindex]
    if edges[-1] < sortedvalues[-1]:
        edges.append(float(sortedvalues[-1]))
    assert len(edges) == numbins + 1, f"{numbins=}, {len(edges)=}, {edges=}, {relthresholds=}"
    assert all(e0 <= e1 for e0, e1 in itertools.pairwise(edges)), f"Edges should be sorted: {edges}, {distribution=}"
    return edges
