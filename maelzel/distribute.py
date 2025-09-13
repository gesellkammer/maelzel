"""
distribute
==========

This module provides functionality to partition a number in multiple ways,
(partitionFib, partitionExpon) or to subdivide it following a given
distribution or curve.

Since many functions make use of curves, we rely heavily on the package
bpf4_, which allows to define and compute break-point-functions

.. _bpf4: https://bpf4.readthedocs.io/en/latest/
"""
from __future__ import annotations
import numpy as np
import bpf4

import warnings
import logging

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import TypeVar, Callable, Sequence
    import matplotlib.pyplot
    T = TypeVar('T')


PHI = 1.61803398874989484820458683436563811772030917

logger = logging.getLogger(__file__)


# ------------------------------------------------------------
#
#    Utilities
#
# ------------------------------------------------------------

def roundSeqPreservingSum(seq: list[float], maxdelta=1, maxsolutions=1,
                          ensureDirection=True
                          ) -> list[int]:
    """
    Round sequence preserving its integer sum

    Find for each element in seq. an integer within the range x-maxdelta : x+maxdelta
    so that the sum of these integers is the same as the rounded sum of seq.

    .. note::

        This function is implemented using contraint programming. The search
        space grows very quickly as the seq grows so it can take a long time
        for large sequences.

    Args:
        seq: a list of numbers
        maxdelta: the max. deviation the  integer can have from the original item
        maxsolutions: the max. number of solutions to generate before finding the
            best from those solutions. All solutions generated comply with the
            constraints given; the returned solutions as the one which follows the
            original seq. as closest
        ensureDirection: if True, ensure that for any two numbers a and b, if
            a < b then a_rounded <= b_rounded (and similarly if a > b)

    Returns:
        a list of integers representing the original seq. If there are no
        solutions an empty list is returned

    Example
    ~~~~~~~

        >>> from maelzel import distribute
        >>> from emlib.iterlib import pairwise
        >>> import matplotlib.pyplot as plt
        >>> parts = distribute.partitionExpon(40, 5, exp=3)
        >>> parts
        array([ 6.0952381 ,  6.19047619,  6.85714286,  8.66666667, 12.19047619])
        >>> round(sum(parts))
        40
        >>> intparts = distribute.roundSeqPreservingSum(parts, maxsolutions=10)
        >>> intparts, sum(intparts)
        [6, 6, 7, 8, 13], 40
    """
    from math import ceil, floor
    import constraint

    p = constraint.Problem()
    numvars = len(seq)
    seqsum = round(sum(seq))
    variables = list(range(numvars))
    for i, item in enumerate(seq):
        minval = ceil(item) - maxdelta
        maxval = floor(item) + maxdelta
        domain = list(range(minval, maxval + 1))
        p.addVariable(i, domain)
    p.addConstraint(constraint.ExactSumConstraint(seqsum), variables)

    if ensureDirection:
        for idx in range(numvars-1):
            if seq[idx] < seq[idx+1]:
                p.addConstraint(constraint.FunctionConstraint(lambda a, b: a <= b), (idx, idx+1))
            elif seq[idx] > seq[idx+1]:
                p.addConstraint(constraint.FunctionConstraint(lambda a, b: a >= b), (idx, idx + 1))

    if maxsolutions > 0:
        from itertools import islice
        solutions = list(islice(p.getSolutionIter(), maxsolutions))
    else:
        solutions = p.getSolutions()
    if not solutions:
        return []
    solutions.sort(key=lambda sol: sum(abs(v - x) for v, x in zip(list(sol.values()), seq)))
    solution = list(solutions[0].items())
    solution.sort()
    varnames, values = list(zip(*solution))
    return list(values)


# ------------------------------------------------------------
#
#     PARTITIONS
#
# ------------------------------------------------------------

def partitionFib(n: int, numpart: int) -> list[float]:
    """
    Partition *n* into *numpart* partitions with fibonacci proportions

    Args:
        n: the number to partition
        numpart: the number of partitions

    Returns:
        a list of partitions which add up to n

    Example
    ~~~~~~~

        >>> from maelzel import distribute
        >>> from emlib.iterlib import pairwise
        >>> parts = distribute.partitionFib(40, 5)
        >>> parts
        [2.4500439227299493,
         3.964254340907179,
         6.414298263637129,
         10.378552604544307,
         16.792850868181436]
        >>> intparts = distribute.roundSeqPreservingSum(parts)
        >>> intparts
        [3, 4, 7, 10, 16]
        >>> for p1, p2 in pairwise(intparts):
        ...     print(f"{p1 = }\t{p2 = }\t\t{p2/p1 = :.3f}")
        p1 = 3	p2 = 4		p2/p1 = 1.333
        p1 = 4	p2 = 7		p2/p1 = 1.750
        p1 = 7	p2 = 10		p2/p1 = 1.429
        p1 = 10	p2 = 16		p2/p1 = 1.600

    .. note::

        In order to partition into integer values, use :func:`roundSeqPreservingSum`

    """
    from emlib import mathlib
    platonic = [mathlib.fib(i) for i in range(50, 50+numpart)]
    ratio = n / float(sum(platonic))
    seq = [x * ratio for x in platonic]
    return seq


def partitionExpon(n: float, numpart: int, exp=2.0) -> list[float]:
    """
    Partition *n* into numpart following an exponential curve

    Args:
        n: the number to partition
        numpart: the number of items to partition *n* into
        exp: the exponential of the curve

    Returns:
        a seq. of values which sum up to *n* following an exponential curve

    .. note::

        In order to partition into integer values, use :func:`roundSeqPreservingSum`
    """
    c = bpf4.expon(0, 1, 1, 2, exp=exp)
    y0 = c.map(numpart)
    r = n / sum(y0)
    return y0 * r


def chooseBestDistribution(values: Sequence[T], possibleValues: Sequence[T]) -> list[T]:
    """
    Reconstruct the given sequence with items from *possibleValues*

    Try to follow the distribution of values as close as possible
    by drawing elements from *possibleValues*, so that
    ``sum(chosen)`` is as close as possible to ``sum(values)` at any
    moment of the operation.

    Args:
        values: a seq. of values
        possibleValues: a seq. of values to draw from

    Returns:
        a "reconstruction" of the sequenve *values* with items drawn from
        *possibleValues*
    """
    values = sorted(values)
    possibleValues = sorted(possibleValues)
    out = []
    status = 0

    def dist(a, b):
        return abs(a - b)

    for value in values:
        bestfit = sorted((dist(elem, value + status), elem) for elem in possibleValues)[0][1]
        dif = value - bestfit
        status += dif
        out.append(bestfit)
    return out


def partitionCurvedSpace(x: float,
                         numpart: int,
                         curve: bpf4.BpfInterface,
                         minval=1,
                         maxdev=1,
                         accuracy=1):
    """
    Partition a number *x* into *numpart* partitions following a curve

    Args:
        x: the number to partition
        numpart: number of partitions
        curve: a bpf curve where y goes from 0 to some integer (this values is not important)
            It must grow monotonically from 0, meaning that for any x_n f(x_n) >= f(x_(n-1))
        minval: min. value of a partition
        maxdev: max. deviation
        accuracy: a value between 0 and 1

    Reeturns:
        a list of partitions, None if not solution is possible

    .. note::
        curve must be a monotonically growing curve starting at 0.
        See the example using .integrated() to learn how to make a monotonically growing
        curve out of any distribution

    NB: returns None if no solution is possible

    Examples
    ~~~~~~~~

    Divide 23 into 6 partitions following an exponential curve

        >>> import bpf4
        >>> curve = bpf4.expon(0, 0, 1, 1, exp=3)
        >>> partitionCurvedSpace(23, 6, curve)
        [1, 1, 2, 4, 6, 9]

    Partition a distance following, an arbitraty curve, where the y defines the
    relative duration of the partitions. In this case, the curve defined the
    derivative of our space.

        >>> import bpf4
        >>> curve = bpf4.linear(
        ...   0, 1,
        ...   0.5, 0,
        ...   1, 1)
        >>> partitionCurvedSpace(21, 7, curve.integrated())
        [5, 3, 2, 1, 2, 3, 5]

    """
    assert curve(curve.x0) == 0 and curve(curve.x1) > 0, \
        "The curve should be a monotonically growing curve, starting at 0"
    if accuracy > 1 or accuracy < 0:
        raise ValueError("0 < accuracy <= 1")
    elif 0 < accuracy < 1:
        scale = int(1.0/accuracy)
        parts = partitionCurvedSpace(x*scale, numpart, curve, minval=minval*scale,
                                     maxdev=maxdev, accuracy=1)
        if not parts:
            scale += 1
            parts = partitionCurvedSpace(x*scale, numpart, curve, minval=minval*scale,
                                         maxdev=maxdev, accuracy=1)
            if not parts:
                warnings.warn("No parts with this accuracy")
                return None
        fscale = float(scale)

        def roundgrid(x, grid):
            numdig = len(str(grid).split(".")[1])
            return round(round(x*(1.0/grid))*grid, numdig)

        return [roundgrid(part/fscale, accuracy) for part in parts]

    else:  # accuracy == 0
        import constraint

        normcurve = curve.fit_between(0, 1)
        normcurve = (normcurve / normcurve(1)) * x
        optimal_results = np.diff(normcurve.map(numpart+1))
        maxval = int(min(int(max(optimal_results) + 1), x-(numpart-1)*minval))

        # maxval = x - minval * (numpart - 1)
        V = list(range(numpart))
        p = constraint.Problem()
        p.addVariables(V, list(range(minval, maxval)))

        def objective(solution):
            values = list(solution.values())
            return sum(abs(val-res) for val, res in zip(values, optimal_results))

        for var, res in zip(V, optimal_results):
            p.addConstraint((lambda x, res=res: abs(x-res) <= maxdev), [var])
        p.addConstraint(constraint.ExactSumConstraint(x), V)

        solutions = p.getSolutions()
        if not solutions:
            logger.warning("No solutions")
            return None
        solutions.sort(key=objective)
        # each solution is a dict with integers as keys
        best = [value for name, value in sorted(solutions[0].items())]
        return best


def _solution_getvalues(solution):
    return [val for name, val in sorted(solution.items())]


def partitionWithCurve(x: float,
                       numpart: int,
                       curve: bpf4.BpfInterface,
                       method='brentq',
                       ) -> list[float]:
    """
    Partition *x* in *numparts* parts following *curve*

    Args:
        x: the value to partition
        numpart: the number of partitions
        curve: the curve to follow. It is not important over which interval x
            it is defined. The y coord defines the width of the partition (see example)
        excluded: any value

    Returns:
        the list of the partitions

    Example
    ~~~~~~~

    Partition the value 45 into 7 partitions following the given curve

        >>> import bpf4
        >>> curve = bpf4.halfcos(0, 11, 1, 0.5, exp=0.5)
        >>> distr = partitionWithCurve(45, 7, curve)
        >>> distr
        array([ 11.        ,  10.98316635,  10.4796218 ,   7.89530421,
                 3.37336152,   0.76854613,   0.5       ])
        >>> abs(sum(distr) - 45) < 0.001
        True
    """
    x0, x1 = curve.bounds()
    n = x

    def func(r):
        return sum((bpf4.expon(x0, x0, x1, x1, exp=r)|curve).map(numpart)) - n

    try:
        if method == 'brentq':
            from scipy.optimize.zeros import brentq
            r = brentq(func, x0, x1)
            curve = bpf4.expon(x0, x0, x1, x1, exp=r)|curve
            parts = curve.map(numpart)
        elif method == 'fsolve':
            from scipy.optimize import fsolve
            xs = np.linspace(x0, x1, 100)
            rs = [round(float(fsolve(func, x)), 10) for x in xs]
            rs = set(r for r in rs if x0 <= r <= x1)
            parts = []
            for r in rs:
                curve = bpf4.expon(x0, x0, x1, x1, exp=r)|curve
                parts0 = curve.map(numpart)
                parts.extend(parts0)
        else:
            raise ValueError(f"Method {method} unknown. Possible methods: brentq, fsolve")
        if abs(sum(parts) - n) / n > 0.001:
            logger.error(f"Error exceeds threshold: {parts=}, {sum(parts)=}")
        return parts

    except ValueError:
        minvalue = curve(bpf4.util.minimum(curve))
        maxvalue = curve(bpf4.util.maximum(curve))
        if n/numpart < minvalue:
            logger.error("no solution can be found for the given parameters. x is too small "
                         "for the possible values given in the bpf, for this amount of "
                         " partition try either giving a bigger x, lowering the number of "
                         "partitions or allowing smaller possible values in the bpf")
            raise ValueError("No solution found (x is too small)")
        elif n/numpart > maxvalue:
            logger.error("No solution can be found for the given parameters. x is too big "
                         "for the possible values given in the bpf. try either giving a "
                         "smaller x, increasing the number of partitions or allowing bigger "
                         "possible values in the bpf")
            raise ValueError("No solutions found (x is too big)")
        else:
            raise ValueError("???")


def partitionFollowingCurve(n: int, curve: bpf4.BpfInterface, ratio=0.5, margin=0.1,
                            method='brentq') -> list[float]:
    """
    Partition *n* following curve

    The difference with :func:`partitionWithCurve` is that in that function
    you determine the number of partitions manually whereas here the number
    of partitions is determined as a ratio of the possible number of partitions.
    This ensures that you always get a valid result (albeit a trivial one if, for
    example, *n* can only be split in 1 partition of *n*)

    Args:
        n: the number to partition
        curve: a bpf
        ratio: ratio indicates the number of partitions, where 0 means the least
            number of partitions possible and 1 means the most number of partitions possible
        margin: sets the max and min possible partitions as a ratio between the max and min
            value of the given curve.

    Returns:
        the list of the partitions

    .. seealso:: :func:`partitionWithCurve`, :func:`roundSeroundSeqPreservingSum`
    """
    maxval = curve(bpf4.util.maximum(curve))
    minval = curve(bpf4.util.minimum(curve))
    minval2 = minval + (maxval-minval)*margin
    maxval2 = minval + (maxval-minval)*(1-margin)
    minpartitions = n / maxval2
    maxpartitions = n / minval2
    npart = round(minpartitions + (maxpartitions-minpartitions)*ratio)
    return partitionWithCurve(n, int(npart), curve, method)

# ------------------------------------------------------------
#
#     SUBSAMPLED CURVES
#
# ------------------------------------------------------------

def onepulse(x: float, resolution: int, entropy=0.) -> list[int]:
    """
    Represents *x* as a seq. of 0 and 1

    Args:
        x: a float number between 0 and 1
        resolution: int. The number of pulses. NB: these are not binary bits,
            all bits have the same significance
        entropy: (float, 0-1). 0: no entropy (the same output for a given
            input), > 0: output is shuffled, entropy represents the number of
            shuffles

    Returns:
         a list of 0s and 1s representing x

    Example
    ~~~~~~~

        >>> from maelzel import distribute
        >>> distribute.onepulse(0.5, 5)
        [1, 0, 1, 1, 0]
    """
    x = x % 1
    ones = int(x * resolution + 0.5)
    zeros = resolution - ones
    o2 = [1] * ones
    z2 = [0] * zeros
    bins = interleave(o2, z2)
    if entropy > 0:
        from emlib import combinatorics
        bins = combinatorics.unsort(bins, entropy)
    return bins


def ditherCurve(curve: bpf4.BpfInterface, numsamples: int, resolution=2) -> list[int]:
    """
    Sample *curve* applying dithering to smooth transitions

    Args:
        curve: a curve defined between 0-numstates.
        numsamples: the number of samples, determines also the size of the output
        resolution: the number of values to average together to calculate
            the resulting curve. There is a tradeoff between x and y resolution

    Returns:
        a list of ints representing the state at each time

    Example
    ~~~~~~~

        >>> import bpf4
        >>> b = bpf4.linear(0, 0, 1, 2)
        >>> ditherCurve(b, 20)
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2]
    """
    origs = curve.map(numsamples)
    resolution = max(1, resolution-1)
    out = [0]*int(resolution/2)

    def avgnow(win, x):
        return (sum(win) + x) / (len(win)+1)

    for i, orig in enumerate(origs):
        win = out[-resolution:]
        minvalue = int(orig)
        maxvalue = int(orig+0.99999999999)
        states = list(range(minvalue, maxvalue+1))
        diffs = [abs(orig-avgnow(win, x)) for x in states]
        best = sorted(zip(diffs, states))[0][1]
        out.append(best)
    out = out[-numsamples:]
    assert len(out) == numsamples
    return out


def pulseCurve(curve: bpf4.BpfInterface,
               n: int,
               resolution=5,
               entropy=0.,
               x0: float | None = None,
               x1: float | None = None
               ) -> list[int]:
    """
    Generates a list of 0s/1s of length n, following the curve

    For a resolution of 4, this is how some values are represented::

        0    --> 0000
        1    --> 1111
        0.5  --> 0011
        0.25 --> 0001
        0.33 --> 0001

    Args:
        curve: a bpf or a function defined between 0 and 1
        n: the number of values to generate
        resolution: how many rendered values represent a value of the curve.
            In fact, the bit-rate values will be rounded to the nearest lower
            representable value
        entropy: a value between 0 and 1. Nearer to 0, the pulse representation is
            unmodified, nearer to 1, the numbers representing the pulse are shuffled.
            Imagine how to represent 0.5 with a resolution of 4, 0.5 --> 0011
            With increasing entropy, this representation yields other shuffles:
            0.5 --> 0101 or 1001 or 1010 ...
        x0, x1: the range to evaluate the curve. If a bpf is passed, its bounds are used
            as defaults. If a normal function is passed, these values default to 0, 1

    Returns:
        a seq. of 0s and 1s of length n
    """
    if x0 is None:
        try:
            x0 = curve.x0
        except AttributeError:
            x0 = 0
    if x1 is None:
        try:
            x1 = curve.x1
        except AttributeError:
            x1 = 1
    resolutions = _dither_resolutions(n, resolution)
    nums = len(resolutions)
    xs = np.linspace(x0, x1, nums)
    ys = [curve(x) for x in xs]
    out = []
    assert sum(resolutions) == n
    for x, y, resolution in zip(xs, ys, resolutions):
        bins = onepulse(y, resolution, entropy)
        out.extend(bins)
    assert len(out) == n
    return out


def _dither_resolutions(numsamples, resolution):
    nums = int(numsamples / resolution + 0.5)
    intvalue = int(numsamples / nums)
    rest = numsamples - (intvalue * nums)
    resolutions = [intvalue + int(index < rest) for index in range(nums)]
    return resolutions


def interleave(A: list[T], B: list[T], weight=0.5) -> list[T]:
    """
    interleave the elements of A and B

    Args:
        A: a list of elements
        B: another list of elements
        weight: Between 0-1. 0: first the elements of xs, then B, not interleaved;
            0.5: interleave A and B regularly; 1: first the elements of B, then A

    Returns:
        a list with items of A and B interleaved

    Example
    ~~~~~~~

        >>> from maelzel import distribute
        >>> A = ["A", "B", "C"]
        >>> B = ["a", "b", "c", "d", "e"]
        >>> "".join(distribute.interleave(A, B))
        'aAbcBdCe'

    """
    if not B:
        return A
    elif not A:
        return B
    c = bpf4.linear(0, len(A), 0.5, len(A)/len(B), 1, 1/len(A))
    r = xr = c(weight)
    out = []
    L = len(A)+len(B)
    A_index = 0
    B_index = 0

    while True:
        if r >= 1:
            if A_index < len(A):
                out.append(A[A_index])
                A_index += 1
            r -= 1
        else:
            if B_index < len(B):
                out.append(B[B_index])
                B_index += 1
            r += xr
        if len(out) == L:
            break
    return out

# ------------------------------------------------------------
#
#    OTHER
#
# ------------------------------------------------------------


def interleaveWithDynamicWeights(streamSizes: list[int],
                                 weightBpfs: list[Callable[[float], float]]
                                 ) -> list[tuple[int, int]]:
    """
    Interleave items of multiple streams based on dynamic weights

    Args:
        streamSizes: the sizes of each stream
        weightBpfs: for each stream, a bpf indicating the weighting. Each bpf is defined
            between 0-1 (for both x and y coords)

    Returns:
        a list of tuples of the form (streamIndex: int, indexInStream: int)

    Example
    ~~~~~~~

    >>> import bpf4
    >>> A = "AAAAAAAAAAAAAAAAAAA"
    >>> B = "BBBBBBBBBBBBB"
    >>> C = "CCCCC"
    >>> D = "DDD"

    >>> streams = (A, B, C, D)
    >>> streamSizes = (len(A), len(B), len(C), len(D))
    >>> bpfs = (bpf4.linear(0, 1, 1, 1),     # bpfs must be defined within the unity
    ...         bpf4.halfcos(0, 0, 0.5, 1, 1, 0),
    ...         bpf4.linear(0, 1, 1, 0),
    ...         bpf4.expon(0, 0, 1, 1, exp=3)
    ... )
    >>> distributedItems = interleaveWithDynamicWeights(streamSizes, bpfs)
    >>> for stream, idx in distributedItems:
    ...     print(streams[stream][idx], end=' ')
    ACAACABABCABABABCABBABABABDBACABABADAADA
    """
    weights = list(map(bpf4.util.asbpf, weightBpfs))
    weight_total = sum(weights)
    normalized_weights = [w / weight_total for w in weights]
    xss = [np.linspace(0, 1, stream_quant * 2 + 1)[1::2] for stream_quant in streamSizes]
    warped_xss = [bpf4.util.warped(w[::w.ntodx(1000)]).map(np.ascontiguousarray(xs))
                  for w, xs in zip(normalized_weights, xss)]
    # before we flatted all the frames to sort them, we need to attach
    # the stream id to each frame so that we know were it belongs
    annotated_xss = []
    for stream, xs in enumerate(warped_xss):
        annotated_xs = [(x, (stream, index_in_stream)) for index_in_stream, x in enumerate(xs)]
        annotated_xss.extend(annotated_xs)
    annotated_xss.sort()
    frames = [frame for x, frame in annotated_xss]
    return frames


_defaultPalette = [
    '#6acc64',
    '#d65f5f',
    '#ee854a',
    '#956cb4',
    '#82c6e2',
    '#4878d0',
    '#8c613c'
]


def plotFrames(xs, ids: list[str] | None = None, top=1., bottom=0., durs: list[float] | None = None,
               palette: list[str] | None = None, edgecolor='black', ax=None, show=True
               ) -> matplotlib.pyplot.Axes:
    """
    Plot a seq. of stacked frames

    Imagine a section divided in measures. Each of these measures can be thought
    of as a frame. This function permits to visualize their relative duration
    by plotting these frames stacked to the left.

    Args:
        xs: (seq) The start time of each frame
        ids: (seq, optional) The id of each frame. Each id is plotted differently
        top, bottom : (number) The frames are defined between these y values.
            it only makes sense if plotting against something else,
            which shares the x coord
        durs: (seq, optional) The duration of each section.
            If not given, it is assumed that the frames are non-overlapping,
            each frame ending where the next one begins. In this case, there are
            len(xs) - 1 number of frames (the last x value is used to determine the
            width of the last frame)
        palette: (seq) A list of colours, will be applied to the IDs as they are found
        edgecolor: the color of the edges
        ax: if given (a matplotlib.pyplot.Axes) this axes will be used to plot to

    Returns:
        the pyplot.Axes used

    Example
    -------

    >>> from maelzel.distribute import *
    >>> sections = [0, 10, 11, 13, 18]
    >>> ids =      [0, 0,   1,  0,  1]
    >>> plotFrames(sections, ids)

    .. image:: ../assets/distribute-plotFrames2.png

    >>> sections = [0, 10, 11, 13, 18]
    >>> ids =      [0, 2,   3,  5,  4]
    >>> plotFrames(sections, ids)

    .. image:: ../assets/distribute-plotFrames1.png
    """
    import matplotlib.pyplot as plt
    if durs is None:
        durs = list(np.diff(xs))
        durs.append(durs[-1])
    if ids is None:
        ids = np.ones((len(durs),))
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ys = np.ones_like(ids) * top
    bottom = np.ones_like(xs) * bottom
    C = palette if palette is not None else _defaultPalette
    colors = [C[id] for id in ids]
    xs2 = [x + w / 2 for x, w in zip(xs, durs)]
    container = ax.bar(xs2, height=ys, width=durs, bottom=bottom, color=colors,
                       edgecolor=edgecolor)
    ax.bar_label(container, labels=ids, label_type='center')
    if show:
        plt.show()
    return ax


def dohndt(numseats: int, votesPerParty: list[int | float]) -> list[int]:
    """
    Perform a D'Ohndt distribution

    Args:
        numseats: the number of seats to distribute across the parties
        votesPerParty: the votes (can be interpreted as the weight of each party)
            of each party.

    Returns:
        the list of assigned seats per party

    Example
    ~~~~~~~

    Distribute a number of items across streams according to
    a set of weights.

        >>> from maelzel.distribute import dohndt
        >>> levels = 10
        >>> numstreams = 4
        >>> weights = [10, 6, 5, 3]
        >>> assigned = dohndt(levels, weights)
        >>> assigned
        [3, 2, 2, 1]

    """
    numparties = len(votesPerParty)
    assignedSeats = [0] * numparties
    indices = list(range(numparties))
    for seat in range(numseats):
        costs = [(votes/(assigned+1), index)
                 for votes, assigned, index in zip(votesPerParty, assignedSeats, indices)]
        costs.sort(key=lambda cost:cost[0])
        winnerindex = costs[-1][1]
        assignedSeats[winnerindex] += 1
    return assignedSeats


if __name__ == '__main__':
    import doctest
    doctest.testmod()  # optionflags=doctest.NORMALIZE_WHITESPACE)
