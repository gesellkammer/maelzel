from __future__ import annotations
import warnings
from maelzel.common import *

from emlib import iterlib 
from emlib.misc import returns_tuple
import bpf4 as bpf

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from numbers import Real as  number_t



def measureDuration(timesig: Union[str, timesig_t], tempo: number_t) -> F:
    """
    calculate the duration of a given measure with the given tempo

    Args:
        timesig: can be of the form "4/4" or (4, 4)
        tempo:   a tempo value corresponding to the denominator of the time
                 signature

    Examples
    ~~~~~~~~

        >>> measureDuration("3/4", 120)        # 3 quarters, quarter=120
        1.5
        >>> measureDuration((3, 8), 60)        # 3/8 bar, 8th note=60
        3

        >>> assert all(measureDuration((n, 4), 60) == n for n in range(20))
        >>> assert all(measureDuration((n, 8), 120) == n / 2 for n in range(20))
        >>> assert all(measureDuration((n, 16), (8,60)) == n / 2 for n in range(40))
    """
    if isinstance(timesig, str):
        assert "/" in timesig
        num, den = map(int, timesig.split('/'))
    elif isinstance(timesig, (tuple, list)):
        num, den = timesig
    else:
        raise ValueError(
            "timesig must be a string like '4/4' or a tuple (4, 4)")
    if isinstance(tempo, (tuple, list)):
        tempoden, tempoval = tempo
    else:
        tempoval = tempo
        tempoden = den
    quarterdur = F(tempoden, 4)*F(60)/tempoval
    quarters_in_measure = F(4, den)*num
    return quarters_in_measure * quarterdur


@returns_tuple("linear2framed framed2linear")
def framedTime(offsets: List[number_t], durations: List[number_t]
               ) -> Tuple[bpf.BpfInterface, bpf.BpfInterface]:
    """
    Returns two bpfs to convert a value between linear and framed coords, and viceversa

    Args:
        offsets: the start x of each frame
        durations: the duration of each frame

    Returns:
        linear2framed, framed2linear

    Example
    ~~~~~~~

    Imagine you want to apply a linear process to a "track" divided in
    non-contiguous frames. For example, a crescendo in density to all frames
    labeled "A".

    >>> from collections import namedtuple
    >>> Frame = namedtuple("Frame", "id start dur")
    >>> frames = map(Frame, [
    ... # id  start dur
    ... ('A', 0,    0.5),
    ... ('B', 0.5,  1),
    ... ('A', 1.5,  0.5),
    ... ('A', 2.0,  0.5),
    ... ('B', 2.5,  1)
    ... ])
    >>> a_frames = [frame for frame in frames if frame.id == 'A']
    >>> offsets = [frame.offset for frame in a_frames]
    >>> durs = [frame.dur for frame in a_frames]
    >>> density = bpf.linear(0, 0, 1, 1)  # linear crescendo in density
    >>> lin2framed, framed2lin = framedTime(offsets, durs)

    # Now to convert from linear time to framed time, call lin2framed
    >>> lin2framed(0.5)
    1.5
    >>> framed2lin(1.5)
    0.5
    """
    xs = [0] + list(iterlib.partialsum(dur for dur in durations))
    pairs = []
    for (x0, x1), y in zip(iterlib.pairwise(xs), offsets):
        pairs.append((x0, y))
        pairs.append((x1, y + (x1 - x0)))
    xs, ys = zip(*pairs)
    lin2framed = bpf.core.Linear(xs, ys)
    try:
        framed2lin = bpf.core.Linear(ys, xs)
    except ValueError:
        ys = _force_sorted(ys)
        framed2lin = bpf.core.Linear(ys, xs)
    return lin2framed, framed2lin


def _force_sorted(xs):
    EPS = 0
    out = []
    lastx = float('-inf')
    for x in xs:
        if x < lastx:
            x = lastx + EPS
        lastx = x
        out.append(x)
    return out


def findNearestDuration(dur, possibleDurations: List[T], direction="<>") -> T:
    """
    Args:
        dur: a Dur or a float (will be converted to Dur via .fromfloat)
        possibleDurations: a seq of Durs
        direction: "<"  -> find a dur from possible_durations which is lower than dur
                   ">"  -> find a dur from possible_durations which is higher than dur
                   "<>" -> find the nearest dur in possibleDurations

    Example
    ~~~~~~~

    >>> possible_durations = [0.5, 0.75, 1]
    >>> findNearestDuration(0.61, possibleDurations, "<>")
    0.5

    """
    possdurs = sorted(possibleDurations, key=lambda d: float(d))
    inf = float("inf")
    if dur < possibleDurations[0]:
        return possibleDurations[0] if direction != "<" else None
    elif dur > possibleDurations[-1]:
        return possibleDurations[-1] if direction != ">" else None
    if direction == "<":
        nearest = sorted(possdurs, key=lambda d:abs(dur - d) if d < dur else inf)[0]
        return nearest if nearest < inf else None
    elif direction == ">":
        nearest = sorted(possdurs, key=lambda d:abs(dur - d) if d > dur else inf)[0]
        return nearest if nearest < inf else None
    elif direction == "<>":
        nearest = sorted(possdurs, key=lambda d:abs(dur - d))[0]
        return nearest
    else:
        raise ValueError("direction should be one of '>', '<', or '<>'")


DEFAULT_TEMPI = (
    60, 120, 90, 132, 48, 80, 96, 100, 72, 52,
    40, 112, 144, 45, 160, 108, 88, 76, 66, 69)


def tempo2beatdur(tempo):
    return 60 / tempo


@returns_tuple("bestTempi resultingDurations numBeats")
def bestTempo(duration, possible_tempi=DEFAULT_TEMPI,
              num_solutions=5, verbose=True):
    """
    Find best tempi that fit the given duration
    """
    remainings = [(duration % tempo2beatdur(tempo), i)
                  for i, tempo in enumerate(possible_tempi)]
    best_tempi = [possible_tempi[i] for remaining, i in
                  sorted(remainings)[:num_solutions]]
    numbeats = [int(duration / tempo2beatdur(tempo) + 0.4999)
                for tempo in best_tempi]
    resulting_durs = [tempo2beatdur(tempo) * n
                      for tempo, n in zip(best_tempi, numbeats)]
    if verbose:
        for tempo, dur, n in zip(best_tempi, resulting_durs, numbeats):
            print("Tempo: %f \t Resulting duration: %f \t Number of Beats: %d" %
                  (tempo, dur, n))
    else:
        return best_tempi, resulting_durs, numbeats


def translateSubdivision(subdivision, new_tempo, original_tempo=60):
    dur_subdiv = tempo2beatdur(original_tempo) / subdivision
    new_beat = tempo2beatdur(new_tempo)
    dur_in_new_tempo = dur_subdiv / new_beat
    return dur_in_new_tempo


def _ratio_to_dur(num: int, den: int) -> float:
    return int(num) * (4 / int(den))


def parseDur(dur, tempo=60):
    if '//' in dur:
        d = _ratio_to_dur(*dur.split('//'))
    elif '/' in dur:
        d = _ratio_to_dur(*dur.split('/'))
    else:
        d = int(dur)
    return d * (60 / tempo)


def possibleTimesigs(tempo: float) -> List[float]:
    """
    Return possible timesignatures for a given tempo

    Time signatures are given in fractions where 2.5 means 5/8, 3.5 means 7/8
    it is assumed that tempo refers to a quarter note
    """
    fractional_timesigs = [1.5, 2.5, 3.5, 4.5]
    int_timesigs = [2., 3., 4., 5., 6., 7., 8., 9.]
    if tempo > 80:
        return int_timesigs
    return sorted(int_timesigs + fractional_timesigs)


def quartersToTimesig(quarters:float, snap=True, mindiv=64) -> Tuple[int, int]:
    """
    Transform a duration in quarters to a timesig

    =========   ========
    quarters    timesig
    =========   ========
    3           (3, 4)
    1.5         (3, 8)
    1.25        (5, 16)
    4.0         (4, 4)
    =========   =======

    Args:
        quarters: duration in quarter notes
        snap: if True, quantize quarters
        mindiv: min. division

    Returns:
        a timesignature as (num, den)
    """
    if snap:
        if quarters < 1:     # accept a max. of 7/32
            quarters = round(quarters*8)/8
        elif quarters < 2:   # accept a max. of 7/16
            quarters = round(quarters*4)/4
        elif quarters < 8:   # accept a max. of 15/8
            quarters = round(quarters*2)/2
        else:
            quarters = round(quarters)
    mindenom = mindiv >> 2
    f = F.from_float(quarters).limit_denominator(mindenom)
    timesig0 = f.numerator, f.denominator*4
    transforms = {
        (1, 4):(2, 8),
        (2, 4):(4, 8)
    }
    timesig = transforms.get(timesig0, timesig0)
    return timesig


def bestTimesig(duration: float,
                tempo=60,
                timesigs: List[float]=None,
                tolerance=0.25) -> List[float]:
    """
    Best timesignature for the given duration

    Args:
        duration: the duration in quarter notes
        tempo: the tempo
        timesigs: a list of timesigs as fractional quarter notes
            (1.5 = 3/8).
        tolerance: how much can the resulting duration differ from the given
    
    Returns:
        the solutions, sorted from best to worst, where each solution is a float
        representing the time signature (2.5 = 5/8)
    """
    timesigs = timesigs or possibleTimesigs(tempo)
    assert (isinstance(timesigs, (list, tuple)) and
            all(isinstance(t, (int, float, F)) for t in timesigs))
    res = [(abs(timesig * (60 / tempo) - duration), timesig) for timesig in timesigs]
    res.sort()
    solutions = [r[1] for r in res]
    solutions = [sol for sol in solutions
                 if abs(sol * 60. / tempo - duration) <= tolerance]
    if not solutions:
        warnings.warn("No solution for the given tolerance. Try a different tempo")

    return solutions


def bestTimesigWithCombinations(duration: float,
                                tempo: float,
                                timesigs: List[float] = None,
                                maxcombinations=3,
                                tolerance=0.25
                                ) -> List[List[float]]:
    """
    Best timesignature to cover the given duration with multiple measures

    Args:
        duration: the duration
        tempo: the tempo
        timesigs: possible timesignatures, as float (2.5 = 5/8)
        maxcombinations: max number of measures
        tolerance: acceptable difference between the given duration and the resulting
            duration

    Returns:
        the solutions, sorted from best to worst (solutions[0] is the best solution).
        Each solution is a list of floats, where each float represents a time signature
    """
    assert isinstance(duration, (int, float, F))
    assert isinstance(tempo, (int, float, F)) and tempo>0
    assert isinstance(timesigs, (tuple, list))
    assert isinstance(maxcombinations, int) and maxcombinations > 0
    import constraint
    p = constraint.Problem()
    possibledurs = [t * (60.0 / tempo) for t in timesigs]
    possibledurs += [0]
    V = range(maxcombinations)
    p.addVariables(V, possibledurs)

    def objective(solution):
        # this is the function to MINIMIZE
        values = solution.values()
        numcombinations = sum(value > 0 for value in values)
        # TODO: use timesig complexity here to make a more intelligent choice
        return numcombinations

    p.addConstraint(constraint.MinSumConstraint(duration - tolerance))
    p.addConstraint(constraint.MaxSumConstraint(duration + tolerance))
    solutions = p.getSolutions()
    if not solutions:
        warnings.warn("No solutions")
        return []
    solutions.sort(key=objective)

    def getvalues(solution) -> List[float]:
        values = [value for name, value in sorted(solution.items()) if value > 0]
        values.sort()
        return values

    solutions = list(map(getvalues, solutions))
    return solutions


