from __future__ import division as _div
import warnings
from maelzel.common import *

from emlib import iterlib 
from emlib.misc import returns_tuple
from typing import Tuple, List, Union as U

import bpf4 as bpf


def measure_duration(timesig: U[str, timesig_t], tempo: number_t) -> F:
    """
    calculate the duration of a given measure with the given tempo

    timesig: can be of the form "4/4" or (4, 4)
    tempo:   a tempo value corresponding to the denominator of the time
             signature
             
    Examples
    ~~~~~~~~

    >>> measure_duration("3/4", 120)        # 3 quarters, quarter=120
    1.5
    >>> measure_duration((3, 8), 60)        # 3/8 bar, 8th note=60
    3
    
    >>> assert all(measure_duration((n, 4), 60) == n for n in range(20))
    >>> assert all(measure_duration((n, 8), 120) == n / 2 for n in range(20))
    >>> assert all(measure_duration((n, 16), (8,60)) == n / 2 for n in range(40))
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
def framed_time(offsets: List[number_t], durations: List[number_t]
                ) -> Tuple[bpf.BpfInterface, bpf.BpfInterface]:
    """
    Returns two bpfs to convert a value between linear and framed coords, and viceversa

    offsets: the start x of each frame
    durations: the duration of each frame

    Returns: linear2framed, framed2linear

    Example
    ~~~~~~~

    Imagine you want to apply a linear process to a "track" divided in
    non-contiguous frames. For example, a crescendo in density to all frames
    labeled "A".

    >>> from collections import namedtuple
    >>> Frame = namedtuple("Frame", "id start dur")
    >>> frames = map(Frame, [
        # id  start dur
        ('A', 0,    0.5),
        ('B', 0.5,  1),
        ('A', 1.5,  0.5),
        ('A', 2.0,  0.5),
        ('B', 2.5,  1)
    ])
    >>> a_frames = [frame for frame in frames if frame.id == 'A']
    >>> offsets = [frame.start for frame in a_frames]
    >>> durs = [frame.dur for frame in a_frames]
    >>> density = bpf.linear(0, 0, 1, 1)  # linear crescendo in density
    >>> lin2framed, framed2lin = framed_time(offsets, durs)

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


def find_nearest_duration(dur, possible_durations, direction="<>"):
    """
    dur: a Dur or a float (will be converted to Dur via .fromfloat)
    possible_durations: a seq of Durs
    direction: "<"  -> find a dur from possible_durations which is lower than dur
               ">"  -> find a dur from possible_durations which is higher than dur
               "<>" -> find the nearest dur in possible_durations

    Example
    ~~~~~~~

    >>> possible_durations = [0.5, 0.75, 1]
    >>> find_nearest_duration(0.61, possible_durations, "<>")
    0.5

    """
    possdurs = sorted(possible_durations, key=lambda d: float(d))
    inf = float("inf")
    if dur < possible_durations[0]:
        return possible_durations[0] if direction != "<" else None
    elif dur > possible_durations[-1]:
        return possible_durations[-1] if direction != ">" else None
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


@returns_tuple("best_tempi resulting_durs numbeats")
def best_tempo(duration, possible_tempi=DEFAULT_TEMPI,
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


def translate_subdivision(subdivision, new_tempo, original_tempo=60):
    dur_subdiv = tempo2beatdur(original_tempo) / subdivision
    new_beat = tempo2beatdur(new_tempo)
    dur_in_new_tempo = dur_subdiv / new_beat
    return dur_in_new_tempo


def parse_dur(dur, tempo=60):

    def ratio_to_dur(num, den):
        return int(num) * (4 / int(den))
    if '//' in dur:
        d = ratio_to_dur(*dur.split('//'))
    elif '/' in dur:
        d = ratio_to_dur(*dur.split('/'))
    else:
        d = int(dur)
    return d * (60 / tempo)


def possible_timesigs(tempo):
    """
    Return possible timesignatures for a given tempo

    Time signatures are given in fractions where 2.5 means 5/8, 3.5 means 7/8
    it is assumed that tempo refers to a quarter note
    """
    fractional_timesigs = [1.5, 2.5, 3.5, 4.5]
    int_timesigs = [2, 3, 4, 5, 6, 7, 8, 9]
    if tempo > 80:
        return int_timesigs
    return sorted(int_timesigs + fractional_timesigs)


def quarters_to_timesig(quarters:float, snap=True, mindiv=64) -> Tuple[int, int]:
    """
    Transform a duration in quarters to a timesig

    quarters    timesig
    –––––––––––––––––––
    3           (3, 4)
    1.5         (3, 8)
    1.25        (5, 16)
    4.0         (4, 4)

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


@returns_tuple("best allsolutions")
def best_timesig(duration, tempo=60, possibletimesigs=None, maxmeasures=1,
                 tolerance=0.25):
    """
    possibletimesigs: a timesig is defined by a float where
                      1 = 1 * fig defining the tempo.
                      So if tempo=60, 1 is one beat of dur. 60
                      Assuming that tempo is defined for the quarter note,
                      1 = 1/4
                      1.5 = 3/8
                      3.5 = 7/8
                      4 = 4/4
                      etc.

                        If not given, a sensible default is assumed
    if maxmeasures > 1: solutions with multiple measures combined are searched
    """
    timesigs = possibletimesigs or possible_timesigs(tempo)
    assert (isinstance(timesigs, (list, tuple)) and
            all(isinstance(t, (int, float, F)) for t in timesigs))
    if maxmeasures > 1:
        return _besttimesig_with_combinations(duration, tempo, timesigs,
                                              tolerance=tolerance,
                                              maxcombinations=maxmeasures)
    res = [(abs(timesig * (60 / tempo) - duration), timesig) for timesig in timesigs]
    res.sort()
    solutions = [r[1] for r in res]
    solutions = [sol for sol in solutions
                 if abs(sol * 60. / tempo - duration) <= tolerance]
    if not solutions:
        warnings.warn("No solution for the given tolerance. Try a different tempo")
        return None, None
    best = solutions[0]
    return best, solutions


def _besttimesig_with_combinations(duration, tempo, timesigs, maxcombinations=3,
                                   tolerance=0.25):
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
        return None
    solutions.sort(key=objective)

    def getvalues(solution):
        values = [value for name, value in sorted(solution.items()) if value > 0]
        values.sort()
        return tuple(values)

    solutions = list(map(getvalues, solutions))
    best = solutions[0]
    solutions = set(solutions)

    return best, solutions


