from __future__ import division, annotations
import math
from typing import Sequence
import itertools
from .common import division_t, F, asF
from functools import cache
from emlib import iterlib


def subdivisions(numdivs: int,
                 possiblevals: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14),
                 maxval=0,
                 maxdensity=20
                 ) -> list[tuple[int, ...]]:
    if maxval == 0:
        maxval = maxdensity // numdivs
    minval = 1
    used = set()
    out = []
    for i in range(maxval, minval-1, -1):
        if i not in possiblevals or any(x % i == 0 for x in used):
            continue
        used.add(i)
        if numdivs == 1:
            out.append((i,))
        else:
            for subdiv in subdivisions(numdivs - 1, possiblevals=possiblevals, maxval=i, maxdensity=maxdensity):
                out.append((i,) + subdiv)
    return out


def allSubdivisions(maxsubdivs=5,
                    maxdensity=20,
                    possiblevals: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14),
                    permutations=True,
                    blacklist: list[tuple[int, ...]] = None
                    ) -> list[tuple[int, ...]]:
    allsubdivs = []
    for numsubdivs in range(maxsubdivs, 0, -1):
        allsubdivs.extend(subdivisions(numdivs=numsubdivs, possiblevals=possiblevals, maxdensity=maxdensity))

    def issuperfluous(p):
        if len(p) > 1 and all(x == p[0] for x in p) and p[0] in (2, 4, 8) and sum(p) in possiblevals:
            return True
        if len(p) in (2, 4, 8) and all(x in (1, 2, 4, 8) for x in p) and max(p)*len(p) in possiblevals:
            # (2, 4) == (4, 4) == 8, (2, 2, 2, 4) == 16, (4, 4) == 8
            return True
        return False

    allsubdivs = [s for s in allsubdivs if not issuperfluous(s)]
    if permutations:
        out = []
        for p in allsubdivs:
            if len(p) == 1:
                out.append(p)
            else:
                out.extend(set(itertools.permutations(p)))
        allsubdivs = out

    allsubdivs.sort(key=lambda p: sum(p))
    if blacklist:
        blacklistset = set(blacklist)
        allsubdivs = [div for div in allsubdivs
                      if div not in blacklistset]
    return allsubdivs


def partitions(numonsets: int,
               numsubdivs: int,
               possiblevals: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14),
               maxvalue=0,
               maxdensity=0) -> list[tuple[int, ...]]:
    """
    Partition *numonsets* into *numsubdivs* partitions

    Args:
        numonsets: the number to partition
        numsubdivs: the number of partitions
        possiblevals: the possible values for each partition
        maxvalue: if given, a max. value for any partition
        maxdensity: the max. value if a value would be used for all partitions; a way to limit
            concentration in one specific partition

    Returns:
        a list of possible partitions

    Example
    ~~~~~~~

        >>> partitions(6, 3)
        [[4, 1, 1], [3, 2, 1], [2, 2, 2]]
        >>> partitions(7, 3)
        [[5, 1, 1], [4, 2, 1], [3, 3, 1], [3, 2, 2]]

    """

    if numsubdivs == 1 and numonsets in possiblevals:
        return [(numonsets,)]

    if maxdensity == 0:
        maxdensity = numonsets * numsubdivs
    if maxvalue == 0:
        maxvalue = numonsets - numsubdivs + 1
    else:
        maxvalue = min(maxvalue, numonsets - numsubdivs + 1)
    minvalue = int(math.ceil(numonsets / numsubdivs))

    if maxvalue == minvalue == 1:
        # Avoid the unary partition (for ex. for 5, [1, 1, 1, 1, 1], this is the same as [5] itself)
        return []

    out = []
    for i in range(maxvalue, minvalue-1, -1):
        if i * numsubdivs > maxdensity:
            continue
        if i not in possiblevals:
            continue
        subpartitions = partitions(numonsets - i, numsubdivs=numsubdivs - 1, maxvalue=i,
                                   possiblevals=possiblevals, maxdensity=maxdensity)
        for sub in subpartitions:
            out.append((i,) + sub)
    assert all(sum(p) == numonsets for p in out)
    return out


def allPartitions(onsets: Sequence[int],
                  maxsubdivs=8,
                  possiblevals: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14),
                  maxdensity=0,
                  makePermutations=True
                  ) -> list[list[int]]:
    """
    All possible partitions for a given set of onsets

    Args:
        onsets: a list of onsets to partition
        maxsubdivs: the max number of subdivisions for one partition
        possiblevals: possible values for any subdivision
        maxdensity: the max. value for any subdivision if it were to be used for all subdivisions; a way to limit
            concentration in one specific partition
        makePermutations: if True, include all permutations for each partition

    Returns:

    """
    allpartitions = []
    for onset in onsets:
        for subdiv in range(1, maxsubdivs+1):
            divisions = partitions(numonsets=onset, numsubdivs=subdiv, possiblevals=possiblevals, maxdensity=maxdensity)
            allpartitions.extend(divisions)

    # remove partitions like (2, 2, 2, 2) if 8 is already a possible value
    def issuperfluos(p):
        if all(x == p[0] for x in p) and p[0] in (2, 4, 8) and sum(p) in possiblevals:
            return True
        if len(p) in (2, 4, 8) and all(x in (1, 2, 4, 8) for x in p) and max(p)*len(p) in possiblevals:
            # (2, 4) == (4, 4) == 8, (2, 2, 2, 4) == 16, (4, 4) == 8
            return True
        return False

    def issubset(p, q):
        return (y%x==0 for x, y in zip(p, q))

    allpartitions = [p for p in allpartitions if not issuperfluos(p)]

    if makePermutations:
        out = []
        for p in allpartitions:
            if len(p) == 1:
                out.append(p)
            else:
                out.extend(set(itertools.permutations(p)))
        allpartitions = out
    return [list(p) for p in allpartitions]


def simplifyDivision(division: division_t, assignedSlots: list[int]) -> division_t:
    """
    Checks if a division (a partition of the beat) can be substituted by a simpler one

    Args:
        division:
        assignedSlots:

    Returns:

    """
    # a note always lasts to the next one
    if len(division) == 1 and division in (3, 5, 7, 11, 13):
        return division
    currslot = 0
    reduced = []
    for subdiv in division:
        if subdiv == 1:
            reduced.append(1)
        else:
            restSlotsInSubdiv = list(range(currslot+1, currslot+subdiv))
            if not any(slot in assignedSlots for slot in restSlotsInSubdiv):
                reduced.append(1)
            elif subdiv == 4 and not any(slot in assignedSlots for slot in (currslot+1, currslot+3)):
                reduced.append(2)
            elif subdiv == 6 and not any(slot in assignedSlots for slot in (currslot+1, currslot+3, currslot+5)):
                reduced.append(3)
            elif subdiv == 6 and not any(slot in assignedSlots for slot in (currslot + 1, currslot + 2, currslot + 4, currslot+5)):
                reduced.append(2)
            else:
                reduced.append(subdiv)
        currslot += subdiv
    return tuple(reduced)




@cache
def gridDurations(beatDuration: F, division: division_t) -> list[F]:
    """
    Called to recursively generate a grid corresponding to the given division
    of the beat
    """
    if isinstance(division, int):
        dt = beatDuration/division
        grid = [dt] * division
    elif isinstance(division, (list, tuple)):
        if len(division) == 1:
            grid = gridDurations(beatDuration, division[0])
        else:
            numDivisions = len(division)
            subdivDur = beatDuration / numDivisions
            grid = [gridDurations(subdivDur, subdiv) for subdiv in division]
    else:
        raise TypeError(f"Expected an int or a list, got {division} ({type(division)})")
    return grid


@cache
def _divisionGrid0(division: division_t, beatDuration: F) -> list[F]:
    beatDuration = asF(beatDuration)
    durations = gridDurations(beatDuration, division)
    flatDurations = list(iterlib.flatten(durations))
    # flatgrid contains a flat list of the duration of each tick
    # now we need to convert that to offsets
    grid = [F(0)] + list(iterlib.partialsum(flatDurations))
    assert grid[-1] == beatDuration
    return grid


def divisionGrid(division: division_t, beatDuration: F, offset=F(0)) -> list[F]:
    grid = _divisionGrid0(division, beatDuration)
    return [tick + offset for tick in grid]
