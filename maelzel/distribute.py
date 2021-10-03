"""
Module text
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    T = TypeVar('T')
    
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
import numpy as np
from scipy.optimize.zeros import brentq as _brentq
from scipy.optimize import fsolve as _fsolve

import bpf4 as bpf

from emlib.mathlib import frange, fib
from emlib.misc import returns_tuple
from emlib import combinatorics, iterlib, interpol

import warnings
import constraint
import logging


PHI = 1.61803398874989484820458683436563811772030917
ERROR = {}
logger = logging.getLogger(__file__)


# ------------------------------------------------------------
#
#    Utilities
#
# ------------------------------------------------------------


def roundSeqPreservingSum(seq: Sequence[float], minval=1, maxval:int=None
                          ) -> Optional[List[int]]:
    """
    Round the elements of seq preserving the sum
    """
    seqsum = round(sum(seq))
    if maxval is None:
        maxval = int(min(max(seq) + 1, seqsum))
    p = constraint.Problem()
    numvars = len(seq)
    variables = list(range(numvars))
    domain = list(range(minval, maxval+1))
    p.addVariables(variables, domain)

    for var, optimalval in zip(variables, seq):
        func = lambda intval, floatval: abs(intval - floatval) <= 1
        p.addConstraint(partial(func, optimalval), [var])
    p.addConstraint(constraint.ExactSumConstraint(seqsum), variables)

    solutions = p.getSolutions()
    if not solutions:
        return None
    solutions.sort(key=lambda sol: sum(abs(v-x) for v, x in zip(list(sol.values()), seq)))
    solution = list(solutions[0].items())
    solution.sort()
    varnames, values = list(zip(*solution))
    return values


# ------------------------------------------------------------
#
#     PARTITIONS
#
# ------------------------------------------------------------

def _partitionFib(n: int, numpartitions: int, homogeneity=0.) -> List[int]:
    """
    if homogeneity == 0: the partitions build a fib curve, which means that if

    s = partition_fib(n, partitions, homogeneity=0), then

    all(s[i] + s[i+1] == s[i+2] for i in range(len(s-2))) is True

    if homogeneity == 1: the partitions build a linear curve

    first a fib curve is built, then an interpolation between this curve
    and a linear curve is done
    """
    assert 0 <= homogeneity <= 1
    if numpartitions > 60:
        raise ValueError("Two many partitions. Max n == 60")
    n0 = 10
    n1 = n0 + numpartitions
    fib_numbers = list(map(interpol.fib, list(range(n0, n1))))
    sum_fibs = sum(fib_numbers)
    normalized_fibs = [(x - 0) / sum_fibs * n for x in fib_numbers]
    avg_y = n / numpartitions
    partitions = [bpf.linear(0, 1, y, avg_y)(homogeneity) for y in normalized_fibs]
    assert all(partition >= 0 for partition in partitions)
    assert abs(sum(partitions) - n) < 0.0001, partitions
    return sorted(partitions)


def partitionFib(n: int, numpart: int) -> List[float]:
    """
    Partition n into `numpart` partitions with fibonacci proportions

    Args:
        n: the number to partition
        numpart: the number of partitions

    Returns:
        a list of partitions which add up to n
    """
    platonic = [fib(i) for i in range(50, 50+numpart)]
    ratio = n / float(sum(platonic))
    seq = [x * ratio for x in platonic]
    return seq


_Solution = namedtuple('Solution', 'num_partitions homogeneity values')


def partitionExpon(n: float, numpart:int, minval:float, maxval:float, homogeneity=1.
                   ) -> List[float]:
    """
    Partition n into numpart followng an exponential distribution.

    The exponential is determined by the homogeneity value. If homogeneity
    is 1, the distribution is linear.

    Args:
        n: the number to partitio
        numpart: the number of partitions
        minval: the min. value possible
        maxval: the max. value possible
        homogeneity: distribution shape, between 0 (exponential) and 1 (linear)

    Returns:
        a list of values which partitio n
    """
    if numpart == 1:
        return [n]
    assert minval <= n
    assert numpart*minval<=n
    assert int(numpart) == numpart  # numpart must be an integer value
    if maxval > n:
        maxval = n
    exp_now = 1
    c = bpf.core.Linear((0, numpart), (0, n))
    minval_now = c(numpart)-c(numpart-1)
    maxval_now = c(1) - c(0)
    assert minval <= minval_now
    assert maxval >= maxval_now
    linear_distribution = c
    dx = 0.001
    for exp_now in frange(1, dx, -dx):
        c = lambda x: interpol.interpol_expon(x, 0, 0, numpart, n, exp_now)
        minval_now, maxval_now = sorted([c(numpart)-c(numpart-1), c(1)-c(0)])
        if maxval_now > maxval or minval_now < minval:
            break

    def interpol_bpfs(delta):
        def func(x):
            y0 = c(x)
            y1 = linear_distribution(x)
            return y0 + (y1-y0)*delta
        return bpf.asbpf(func, bounds=linear_distribution.bounds())

    curve = interpol_bpfs(homogeneity)
    values = [x1-x0 for x0, x1 in iterlib.pairwise(list(map(curve, list(range(numpart+1)))))]
    values.sort(reverse=True)
    return values


def chooseBestDistribution(values: Sequence[T], possible_elements: Sequence[T]) -> List[T]:
    """
    Try to follow the distribution of values as close as possible
    by drawing elements from possible_elements, so that
    sum(chosen_values) is as close as possible as sum(values)
    and distribution(chosen_values) is as close as poss. as distribution(values)

    values: a seq. of values
    possible_elements: a seq. of values to draw from
    """
    values = sorted(values)
    possible_elements = sorted(possible_elements)
    out = []
    status = 0

    def distance(a, b):
        return abs(a - b)

    for value in values:
        best_fit = sorted((distance(elem, value + status), elem) for elem in possible_elements)[0][1]
        dif = value - best_fit
        status += dif
        out.append(best_fit)
    return out


def partitionFromValues(avg, n, homogeneity, possible_values, func='fib'):
    if func == 'fib':
        part = partitionFib(avg*n, n)
    elif func == 'expon':
        part = partitionExpon(avg*n, n, min(possible_values), max(possible_values), homogeneity)
    else:
        raise NotImplemented('function distribution %s not implemented' % func)
    best_distr = chooseBestDistribution(part, possible_values)
    return best_distr


def partitionRecursively(n, partitions, homogeneity, minval, maxval, kind='fib'):
    """
    partition the value @n in @partitions where partitions is a
    list of partition numbers

    if partition = (2, 7)

    then first partition n in 2, then the two resulting partitions
    are also partitioned so that the final number of partitions = 7

    the last number in partition indicate always the final number of partitions

    for each partition step an homogeneity step is also needed, so
    len(partitions) == len(homogeneity) must be true
    """
    func = {
        'fib': partitionFib,
        'expon': partitionExpon
    }
    p = partitions[0]
    h = homogeneity[0]
    subns = func[kind](n, p, minval, maxval, h)
    if len(partitions) == 1:
        return subns
    collected = []
    for subn in subns:
        ps = list(partitions[1:])
        ps[0] = max(int(ps[0] * (subn / n) + 0.5), 1)
        collected.append(partitionRecursively(subn, ps, homogeneity[1:], minval, subn, type))
    tmp = iterlib.flatten(collected)
    assert abs(sum(tmp) - n) < 0.0001
    return collected


def partitionCurvedSpace(x, numpart, curve, minval=1, maxdev=1, accuracy=1):
    """
    Partition a curved space defined by curve into `numpart` partitions,
    each with a min. value of `minval`

    curve: a bpf curve where y goes from 0 to some integer (this values is not important)
           NB: curve must be a monotonically growing curve starting at 0.
           See the example using .integrated() to learn how to make a monotonically growing
           curve out of any distribution

    NB: returns None if no solution is possible

    Example 1
    =========

    Divide 23 into 6 partitions following an exponential curve

    curve = bpf.expon(3, 0, 0, 1, 1)
    partition_curvedspace_int(23, 6, curve)
    --> [1, 1, 2, 4, 6, 9]

    Example 2
    =========

    Partition a distance following, an arbitraty curve, where the y defines the
    relative duration of the partitions. In this case, the curve defined the
    derivative of our space.

    curve = bpf.linear(
        0, 1,
        0.5, 0,
        1, 1)
    partition_curvedspace_int(21, 7, curve.integrated())
    --> [5, 3, 2, 1, 2, 3, 5]

    Example 3
    =========

    Partition a distance following an arbitrary curve, with fractional values with
    a precission of 0.001

    curve = bpf.linear(0, 1, 0.5, 0, 1, 1)
    dist = 21
    numpart = 7
    upscale = 1000
    parts = partition_curvedspace_int(dist*upscale, numpart, curve.integrated())
    parts = [part/upscale for part in parts]
    --> [5.143, 3.429, 1.714, 0.428, 1.714, 3.429, 5.143]
    """
    assert curve(curve.x0) == 0 and curve(curve.x1) > 0, \
        "The curve should be a monotonically growing curve, starting at 0"
    if accuracy > 1:
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
    normcurve = curve.fit_between(0, 1)
    normcurve = (normcurve / normcurve(1)) * x
    optimal_results = np.diff(normcurve.map(numpart+1))
    maxval = min(int(max(optimal_results) + 1), x-(numpart-1)*minval)

    # maxval = x - minval * (numpart - 1)
    V = list(range(numpart))
    p = constraint.Problem()
    p.addVariables(V, list(range(minval, maxval)))

    def objective(solution):
        values = list(solution.values())
        return sum(abs(val-res) for val, res in zip(values, optimal_results))

    for var, res in zip(V, optimal_results):
        func = lambda x, res: abs(x-res) <= maxdev
        p.addConstraint(partial(func, res), [var])
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


def partitionWithCurve(x, numpart, curve, method='brentq', return_exp=False, excluded=[]):
    """
    Partition `x` in `numparts` parts following bpf

    x : float     --> the value to partition
    numpart : int --> the number of partitions
    curve : bpf   --> the curve to follow.
                      It is not important over which interval x it is defined.
                      The y coord defines the width of the partition (see example)
    return_exp : bool --> | False -> the return value is the list of the partitions
                          | True  -> the return value is a tuple containing the list
                                     of the partitions and the exponent of the
                                     weighting function

    Returns: the list of the partitions

    Example
    =======

    # Partition the value 45 into 7 partitions following the given curve
    >>> import bpf4 as bpf
    >>> curve = bpf.halfcos2(0, 11, 1, 0.5, exp=0.5)
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
        return sum((bpf.expon(x0, x0, x1, x1, exp=r)|curve).map(numpart)) - n
    try:
        if method == 'brentq':
            r = _brentq(func, x0, x1)
            curve = bpf.expon(x0, x0, x1, x1, exp=r)|curve
            parts = curve.map(numpart)
        elif method == 'fsolve':
            xs = np.linspace(x0, x1, 100)
            rs = [round(float(_fsolve(func, x)), 10) for x in xs]
            rs = set(r for r in rs if x0 <= r <= x1 and r not in excluded)
            parts = []
            for r in rs:
                curve = bpf.expon(x0, x0, x1, x1, exp=r)|curve
                parts0 = curve.map(numpart)
                parts.extend(parts0)
    except ValueError:
        minvalue = curve(bpf.minimum(curve))
        maxvalue = curve(bpf.maximum(curve))
        if n/numpart < minvalue:
            s = """
        no solution can be found for the given parameters. x is too small
        for the possible values given in the bpf, for this amount of partitions
        try either giving a bigger x, lowering the number of partitions or
        allowing smaller possible values in the bpf
                """
        elif n/numpart > maxvalue:
            s = """
            no solution can be found for the given parameters. x is too big
        for the possible values given in the bpf. try either giving a
        smaller x, increasing the number of partitions or allowing bigger
        possible values in the bpf
                """
        else:
            s = """???"""
        ERROR['partition_with_curve.func'] = func
        raise ValueError(s)
    if abs(sum(parts) - n)/n > 0.001:
        print("Error exceeds threshold: ", parts, sum(parts))
    if return_exp:
        return parts, r
    return parts


def partitionFollowingCurve(n, curve, ratio=0.5, margin=0.1, method='brentq'):
    """
    partition `n` following curve

    n    : the number to partition
    curve: a bpf
    ratio: ratio indicates the number of partitions
           0 means the least number of partitions possible
           1 means the most number of partitions possible
    """
    maxval = curve(bpf.util.maximum(curve))
    minval = curve(bpf.util.minimum(curve))
    minval2 = interpol.ilin1(margin, minval, maxval)
    maxval2 = interpol.ilin1(1 - margin, minval, maxval)
    minpartitions = n / maxval2
    maxpartitions = n / minval2
    npart = int(interpol.ilin1(ratio, minpartitions, maxpartitions) + 0.5)
    return partitionWithCurve(n, npart, curve, method)

# ------------------------------------------------------------
#
#     SUBSAMPLED CURVES
#
# ------------------------------------------------------------


def binaryCurve(curve: bpf.BpfInterface, n: int, resolution=5) -> List[int]:
    """
    Follow curve in n steps, where each step is a binary value (0, 1) representing
    the distribution

    Args:
        curve: a bpf, defined between 0-1
        n: number of items
        resolution: determines how many binary values are used to represent a given number

    Returns:
        a list of 0s and 1s representing the curve.
        The length of the output is n * resolution
    """
    logger.error("this is deprecated, please use pulse_curve")
    xs = np.linspace(0, 1, n)
    ys = [curve(x) for x in xs]
    out = []
    for x, y in zip(xs, ys):
        ones = int(y * resolution + 0.5)
        zeros = resolution - ones
        o2 = [1] * ones
        z2 = [0] * zeros
        bins = interleave(o2, z2)
        print(x, y, bins)
        out.extend(bins)
    return out


def onepulse(x: float, resolution:int, entropy=0.) -> List[int]:
    """
    Args:
        x: float. The number to represent, between 0-1
        resolution: int. The number of pulses. NB: these are not binary bits,
            all bits have the same significance
        entropy: (float, 0-1). 0 -> no entropy, for a given input expect the same output
            1 -> total shuffle of bits

    Returns:
         a list of 0s and 1s representing x
    """
    x = x % 1
    ones = int(x * resolution + 0.5)
    zeros = resolution - ones
    o2 = [1] * ones
    z2 = [0] * zeros
    bins = interleave(o2, z2)
    if entropy > 0:
        bins = combinatorics.unsort(bins, entropy)
    return bins



def ditherCurve(curve: bpf.BpfInterface, numsamples: int, resolution=2) -> List[int]:
    """
    Sample `curve` applying dithering to smooth transitions

    Example
    ~~~~~~~

        >>> b = bpf.linear(0, 0, 1, 2)
        >>> ditherCurve(b, 20)
        [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2]

    Args:
        curve: a curve defined between 0-numstates.
        numsamples: the number of samples
        resolution: the number of values to average together to calculate
                    the resulting curve. There is a tradeoff between x and y resolution
    """
    origs = curve.map(numsamples)
    #if resolution is None:
    #    resolution = max(2, int(max(origs))-1)
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
        # print(orig, win, best, diffs, [avgnow(win, x) for x in states])
        out.append(best)
    out = out[-numsamples:]
    assert len(out) == numsamples
    return out


def pulseCurve(curve: bpf.BpfInterface, n: int, resolution=5, entropy=0.,
               x0:float=None, x1:float=None
               ) -> List[int]:
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
    # dx = (x1-x0)/n
    #nums = int(n / resolution + 0.5)
    #intvalue = int(n / nums)
    #rest = n - (intvalue * nums)
    #resolutions = [intvalue + int(index < rest) for index in range(nums)]
    resolutions = _dither_resolutions(n, resolution)
    nums = len(resolutions)
    print(resolutions)
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


def interleave(A: List[T], B: List[T], weight=0.5) -> List[T]:
    """
    interleave the elements of A and B

    Args:
        A: a list of elements
        B: another list of elements
        weight: Between 0-1. 0: first the elements of xs, then B, not interleaved;
            0.5: interleave A and B regularly; 1: first the elements of B, then A
    """
    if not B:
        return A
    elif not A:
        return B
    c = bpf.linear(0, len(A), 0.5, len(A)/len(B), 1, 1/len(A))
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


@dataclass
class _FillMatch:
    size: int
    container_index: int
    stream_index: int


@dataclass
class _FillResult:
    matches: List[_FillMatch]
    unfilled_containers: list
    unused_streams: list


def fill(containers, streams) -> _FillResult:
    """
    given a list of caintainers, partition the streams to fill the containers

    Example
    =======

    # try to partition 3 and 4 into three values which fill containers 1, 2, 5
    # in the best way
    >>> fill([1,2,5], [3, 4])
    [[1, 2], 4]
    """
    containers_keep_track = containers[:]
    streams_keep_track = streams[:]
    streams = [(stream, idx) for idx, stream in enumerate(streams)]
    containers = [(container, idx) for idx, container in enumerate(containers)]
    containers = sorted(containers, reverse=True)  # sort the containers from big to small
    streams = sorted(streams, reverse=True)        # also sort them from big to small
    out_streams = [[] for _ in range(len(streams))]
    out = []
    for container, container_id in containers:
        if any(stream[0] >= container for stream in streams):
            best_fit_difference, stream, index_now = sorted((stream[0] - container, stream, i)
                                                            for i, stream in enumerate(streams)
                                                            if stream[0] >= container)[0]
        else:
            best_fit_difference, stream, index_now = sorted((abs(stream[0] - container), stream, i)
                                                            for i, stream in enumerate(streams))[0]
        size = min(container, stream[0])
        print(container, size, stream)
        out_streams[stream[1]].append(size)
        out.append(_FillMatch(size, container_id, stream[1]))
        containers_keep_track[container_id] -= size
        streams_keep_track[stream[1]] -= size
        streams[index_now] = (stream[0] - size, stream[1])
        streams = sorted(streams, reverse=True)
    return _FillResult(out, containers_keep_track, streams_keep_track)


@returns_tuple("stream index_in_stream")
def distributeWeightedStreams(stream_quantities, weight_bpfs):
    """
    Example:

    A = ["AAAAAAAAAAAAAAAAAAAAAAAA"]
    B = ["BBBBBBBBBBBBBBBBBB"]
    C = ["CCCCC"
    D = ["DDD"]

    streams = (A, B, C, D)
    stream_quantities = (len(A), len(B), len(C), len(D))
    weight_bpfs = (bpf.linear(0, 1, 1, 1),     # bpfs must be defined within the unity
                   bpf.halfcos(0, 0, 0.5, 1, 1, 0),
                   bpf.linear(0, 1, 1, 0),
                   bpf.expon(0, 0, 1, 1, exp=3)
    )
    distributed_frames = distribute_weighted_streams(stream_quantities, weight_bpfs)
    for frame in distributed_frames:
        print(streams[frame.stream][frame.index_in_stream])
    """
    weights = list(map(bpf.asbpf, weight_bpfs))
    weight_total = sum(weights)
    # weight_total = reduce(_op.add, weights)
    normalized_weights = [w / weight_total for w in weights]
    xss = [np.linspace(0, 1, stream_quant * 2 + 1)[1::2] for stream_quant in stream_quantities]
    warped_xss = [bpf.warped(w[::w.ntodx(1000)]).map(xs) for w, xs in zip(normalized_weights, xss)]
    # before we flatted all the frames to sort them, we need to attach
    # the stream id to each frame so that we know were it belongs
    annotated_xss = []
    for stream, xs in enumerate(warped_xss):
        annotated_xs = [(x, (stream, index_in_stream)) for index_in_stream, x in enumerate(xs)]
        annotated_xss.extend(annotated_xs)
    annotated_xss.sort()
    frames = [frame for x, frame in annotated_xss]
    return frames


def _test_distribute_weighted_streams():
    A = "AAAAAAAAAAAAAAAAAAAAAAAA"
    C = "CCCCC"
    D = "DDD"

    streams = (A, C, D)
    stream_quantities = (len(A), len(C), len(D))
    weight_bpfs = (bpf.linear(0, 1, 1, 1),     # bpfs must be defined within the unity
                   bpf.halfcos(0, 0, 0.5, 1, 1, 0),
                   bpf.expon(0, 0, 1, 1, exp=3)
                   )
    distributed_frames = distributeWeightedStreams(stream_quantities, weight_bpfs)
    for frame in distributed_frames:
        print(streams[frame.stream][frame.index_in_stream])


def plotFrames(xs, ids=None, top=1, bottom=0, durs=None, palette=None):
    """
    xs  : (seq) The start values of each section
    ids : (seq, optional) The id of each section. Each id is plotted differently
    top, bottom : (number) The frames are defined between these y values.
        it only has sense if you are plotting the frames against something else,
        which shares the x coord
    durs : (seq, optional) The duration of each section.
        If not given, it is assumed that the sections are non-overlapping,
        each section ending where the next one begins. In this case, there are
        len(xs) - 1 number of sections
    palette: (seq) A list of colours, will be applied to the IDs as they are found

    Example
    -------

    >>> sections = [0, 10, 11, 13, 18]
    >>> ids =      [0, 0,   1,  0,  1]
    >>> plotFrames(sections, ids)
    """
    import matplotlib.pyplot as plt
    if durs is None:
        durs = list(np.diff(xs))
        durs.append(durs[-1])
    if ids is None:
        ids = np.ones((len(durs),))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ys = np.ones_like(ids) * top
    bottom = np.ones_like(xs) * bottom
    C = palette if palette is not None else ['red', 'blue', 'green', 'orange', 'grey', '#6666FF', '#33FF66', 'FF6633']
    colors = [C[id] for id in ids]
    ax.bar(xs, ys, durs, bottom, color=colors)
    plt.show()


def fitCurveBetween(n, y0, y1, N=5):
    h = bpf.fib(0, y0, 1, y1)

    def func(r: float) -> float:
        b = bpf.expon(0, 0, 1, 1, exp=r)|h
        samples = b.map(N)
        ratios = samples[1:] / samples[:-1]
        dif = abs(ratios - PHI).sum()
        return dif

    try:
        r = _brentq(func, 0, 20)
        out = bpf.expon(0, 0, 1, 1, exp=r)|h
    except ArithmeticError:
        out = None
    return out


def dohndt(numseats, votes_perparty):
    """
    Perform a D'Ohndt distribution

    numseats: the number of seats to distribute across the parties
    votes_perparty: the votes (can be interpreted as the weight of each party)
                    of each party

    Examples: distribute voices across channels, so that each channel has
              at least a voice and the remaining voices are assigned according
              to a weight

    numvoices = 8
    numchannels = 4
    weightperchannel = bpf.linear(0, 1, 1, 3).map(numchannels)
    assigned = dohndt(numvoices-numchannels, weightperchannel)
    for i in range(len(assigned)):
        assigned[i] += 1
    """
    numparties = len(votes_perparty)
    assigned_perparty = [0] * numparties
    indices = list(range(numparties))
    for seat in range(numseats):
        costs = [(votes/(assigned+1), index)
                 for votes, assigned, index in zip(votes_perparty, assigned_perparty, indices)]
        costs.sort(key=lambda cost:cost[0])
        winnerindex = costs[-1][1]
        assigned_perparty[winnerindex] += 1
    return assigned_perparty


if __name__ == '__main__':
    import doctest
    doctest.testmod()  # optionflags=doctest.NORMALIZE_WHITESPACE)
