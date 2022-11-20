from __future__ import division, annotations
import math
from typing import Sequence
import itertools

from .common import division_t, F, asF
from functools import cache
from emlib import iterlib
from .notation import Notation


def divisionDensity(division: division_t) -> int:
    return max(division) * len(division)


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
        if len(p) > 1 and all(x == p[0] for x in p) and sum(p) in possiblevals:
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
        if blacklist:
            permutations = []
            for div in blacklist:
                if len(div) == 1:
                    permutations.append(div)
                else:
                    permutations.extend(set(itertools.permutations(div)))
            blacklist = permutations

    allsubdivs.sort(key=lambda p: sum(p))
    if blacklist:
        blacklistset = set(blacklist)
        allsubdivs = [div for div in allsubdivs
                      if div not in blacklistset]
    return allsubdivs


def resnap(assignedSlots: list[int],
           grid: list[F],
           newgrid: list[F]
           ) -> list[int]:
    minslot = 0
    maxslot = len(newgrid)
    reassigned = []
    for slot0 in assignedSlots:
        offset = grid[slot0]
        for slotidx in range(minslot, maxslot):
            if offset == newgrid[slotidx]:
                reassigned.append(slotidx)
                minslot = slotidx
                break
    return reassigned


def simplifyDivision(division: division_t, assignedSlots: list[int]) -> division_t:
    """
    Checks if a division (a partition of the beat) can be substituted by a simpler one

    Args:
        division: the division to simplify
        assignedSlots: assigned slots for this division

    Returns:
        the simplified version or the original division if no simplification is possible
    """
    # a note always lasts to the next one
    # assert isinstance(division, tuple)

    if len(assignedSlots) == 1 and assignedSlots[0] == 0:
        return (1,)
    elif len(division) == 1 and division[0] in {3, 5, 7, 11, 13}:
        return division

    assigned = set(assignedSlots)
    cs = 0
    reduced = []
    for subdiv in division:
        if subdiv == 1 or set(range(cs+1, cs+subdiv)).isdisjoint(assigned):
            reduced.append(1)
        elif subdiv == 4 and cs+1 not in assigned and cs+3 not in assigned:
            reduced.append(2)
        elif subdiv == 6:
            if cs+1 not in assigned and cs+3 not in assigned and cs+5 not in assigned:
                reduced.append(3)
            elif {cs+1,cs+2,cs+4, cs+5}.isdisjoint(assigned):
                reduced.append(2)
            else:
                reduced.append(subdiv)
        elif subdiv == 8:
            if {cs+1, cs+2, cs+3,cs+5, cs+6, cs+7}.isdisjoint(assigned):
                reduced.append(2)
            if {cs+1, cs+3, cs+5, cs+7}.isdisjoint(assigned):
                reduced.append(4)
            else:
                reduced.append(subdiv)
        elif subdiv == 9 and {cs+1, cs+2, cs+4, cs+5, cs+7, cs+8}.isdisjoint(assigned):
            reduced.append(3)
        else:
            reduced.append(subdiv)
        cs += subdiv
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


def gridDurationsFlat(beatDuration: F, division: division_t) -> list[F]:
    if isinstance(division, int):
        dt = beatDuration/division
        return [dt] * division

    if len(division) == 1:
        return gridDurationsFlat(beatDuration, division[0])
    else:
        numDivisions = len(division)
        subdivDur = beatDuration / numDivisions
        grid = []
        for subdiv in division:
            grid.extend(gridDurationsFlat(subdivDur, subdiv))
    return grid


@cache
def divisionGrid0(division: division_t, beatDuration: F) -> list[F]:
    # assert isinstance(beatDuration, F)
    # durations = iterlib.flatten(gridDurations(beatDuration, division))
    durations = gridDurationsFlat(beatDuration, division)
    # flatgrid contains a flat list of the duration of each tick
    # now we need to convert that to offsets
    grid = [F(0)]
    grid.extend(iterlib.partialsum(durations))
    # assert grid[-1] == beatDuration
    return grid


def divMinSlotDuration(div: division_t, beatDuration: F) -> F:
    grid = divisionGrid0(div, beatDuration)
    mindur = min(slot1 - slot0 for slot0, slot1 in iterlib.pairwise(grid))
    return mindur


def primeFactors(d: int, excludeBinary=False) -> set:
    assert isinstance(d, int), f"expected int, got {d}"
    factors = set()
    for p in (3, 5, 7, 11, 13, 17, 19):
        if d % p == 0:
            factors.add(p)
    if not excludeBinary:
        if d % 2 == 0:
            factors.add(2)
    return factors


def transferAttributesWithinTies(notations: list[Notation]) -> None:
    """
    When two notes are tied, some attributes need to be copied to the tied note

    This functions works **IN PLACE**.

    Attributes which need to be transferred:

    * gliss: all notes in a tie need to be marked with gliss

    Args:
        notations: the notations to modify

    """
    insideGliss = False
    for n in notations:
        if n.gliss and not insideGliss:
            insideGliss = True
        elif not n.tiedPrev and insideGliss:
            insideGliss = False
        elif n.tiedPrev and insideGliss and not n.gliss:
            n.gliss = True