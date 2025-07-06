from __future__ import annotations
from . import spectrum as sp
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _segmentsZ(data: np.ndarray, downsample=1, avg=True
               ) -> tuple[np.ndarray, np.ndarray]:
    """

    Args:
        data: a 2D matrix with columns X, Y, Z
        downsample: a downsampling integer factor
        avg: if True, colour each line with the average of the Z value at the
            edges

    Returns:
        a tuple (coordarray, zarray) where coordarray is a 2D array of the points
        (each row holds two values, x, y for each point; there are as many rows
        as there are points) and zarray is an array with the values

    """
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2].copy()
    if downsample > 1:
        X = X[::downsample]
        Y = Y[::downsample]
        Z = Z[::downsample]
    if avg:
        Z = Z[:-1] + Z[1:]
        Z *= 0.5
    points = np.array([X, Y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments, Z


def plotmpl(spectrum: sp.Spectrum,
            axes: Axes | None = None,
            linewidth=1,
            avg=True,
            cmap='inferno',
            exp=1.,
            offset=0.,
            downsample=1,
            autolim=True,
            figsize=(20, 8),
            maxfreq=20000,
            yscale='linear',
            ) -> Axes:
    """
    Plot a Spectrum with matplotlib

    Args:
        spectrum: the Spectrum to plot
        axes: if given, use it to plot into
        linewidth: the linewidth of each partial
        avg: if True, color a segment as the average between two breakpoints
        cmap: the colormap used
        exp: apply an exponential to the amplitude for better contrast
        offset: add an offset to all values to make faint sounds visible
        autolim: auto limit, passed to matplotlib add_collection
        downsample: the amount of downsampling, results in picking one breakpoint every the
            downsample value
        figsize: figure size to set if an axes is not given
        maxfreq: the max. frequency of the plot, sets the y limit

    Returns:
        the axes used (will be the same as `axes` if it was passed)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    downsample = max(downsample, 1)
    if axes is None:
        fig = plt.figure(figsize=figsize, layout="constrained")
        axes = fig.add_subplot(1, 1, 1)
    else:
        fig = None
    bg = mpl.colormaps.get_cmap('inferno')(0.005)[:3]
    axes.set_facecolor(bg)
    axes.autoscale(False)
    axes.use_sticky_edges = False
    allsegments = []
    Zs = []
    for p in spectrum.partials:
        if len(p) <= downsample:
            continue
        segments, Z = _segmentsZ(p.data, downsample=downsample, avg=avg)
        allsegments.extend(segments)
        Zs.append(Z)

    ZZ = np.concatenate(Zs)
    if exp != 1:
        ZZ **= exp
    if offset != 0:
        ZZ += offset
    lc = LineCollection(allsegments, cmap=cmap, array=ZZ)
    lc.set_linewidth(linewidth)

    axes.add_collection(lc, autolim=autolim)
    if fig:
        _ = fig.colorbar(lc)
        # plt.tight_layout()

    plt.sci(lc)
    axes.set_xlim(0, spectrum.end)
    if yscale == 'linear':
        maxfreq = min(maxfreq, int(spectrum.maxFrequency()))
        axes.set_ylim(0, maxfreq)
    else:
        axes.set_yscale(yscale)
        axes.autoscale()
    axes.use_sticky_edges = True
    return axes
