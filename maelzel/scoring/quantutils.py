"""
Utilities used during quantization
"""
from __future__ import division, annotations
import math
from functools import cache
from emlib import iterlib
from emlib import mathlib
from itertools import pairwise

from maelzel.common import F, F0, F1
from .common import logger, division_t
from . import node as _node
from . import quantdata

import typing as _t
if _t.TYPE_CHECKING:
    from .notation import Notation



@cache
def isNestedTupletDivision(div: division_t) -> bool:
    if isinstance(div, int):
        # A shortcut division, like 3 or 5
        return False
    return not mathlib.ispowerof2(len(div)) and any(not mathlib.ispowerof2(subdiv) for subdiv in div)


def divisionDensity(division: division_t) -> int:
    return max(division) * len(division)


def resnap(assignedSlots: list[int], oldgrid: list[F], newgrid: list[F]) -> list[int]:
    minslot = 0
    maxslot = len(newgrid)
    reassigned = []
    for slot in assignedSlots:
        oldoffset = oldgrid[slot]
        for newslotidx in range(minslot, maxslot):
            newoffset = newgrid[newslotidx]
            if oldoffset == newoffset:
                reassigned.append(newslotidx)
                minslot = newslotidx
                break
        else:
            raise ValueError(f"No corresponding slot {oldoffset=}, {newgrid=}")

    if not len(reassigned) == len(assignedSlots):
        oldoffsets = [oldgrid[i] for i in assignedSlots]
        newoffsets = [newgrid[i] for i in reassigned]
        logger.error(f'{oldoffsets=}, {newoffsets=}, {assignedSlots=}, {reassigned=}, {oldgrid=}, {newgrid=}')
        raise RuntimeError("resnap error")
    return reassigned


def simplifyDivision(division: division_t, assignedSlots: list[int], reduce=True
                     ) -> division_t:
    """
    Checks if a division (a partition of the beat) can be substituted by a simpler one

    Args:
        division: the division to simplify
        assignedSlots: assigned slots for this division
        reduce: if True, try to reduce a division like (1, 2, 1) to (6,)

    Returns:
        the simplified version or the original division if no simplification is possible
    """
    # a note always lasts to the next one
    # assert isinstance(division, tuple)

    if len(assignedSlots) == 1 and assignedSlots[0] == 0:
        return 1,
    elif len(division) == 1 and division[0] in (3, 5, 7, 11, 13):
        return division

    assigned = set(assignedSlots)

    def makeset(start, end, exclude):
        out = set(x for x in range(start, end))
        for item in exclude:
            out.remove(item)
        return out

    cs = 0
    reduced = []
    for subdiv in division:
        if subdiv == 1 or all(slot not in assigned for slot in range(cs+1, cs+subdiv)):
            reduced.append(1)
        elif subdiv == 4 and cs+1 not in assigned and cs+3 not in assigned:
            reduced.append(2)
        elif subdiv == 9 and {cs+1, cs+2, cs+4, cs+5, cs+7, cs+8}.isdisjoint(assigned):
            reduced.append(3)
        elif subdiv % 2 == 1:
            reduced.append(subdiv)
        elif makeset(cs+1, cs+subdiv, (cs+subdiv//2,)).isdisjoint(assigned):
            reduced.append(2)
        elif set(range(cs+1, cs+subdiv, 2)).isdisjoint(assigned):
            reduced.append(subdiv//2)
        else:
            reduced.append(subdiv)
        cs += subdiv

    newdiv: division_t = tuple(reduced)
    assert len(newdiv) == len(division), f'{division=}, {newdiv=}'

    if all(subdiv == 1 for subdiv in newdiv):
        newdiv = (len(newdiv),)

    # last check: unnest and check
    # for example, (1, 2, 1) with slots 0 (0) and 2 (1/2) can be reduced to (2,)
    # first expand (1, 2, 1) to (6,) then reduce again
    if len(newdiv) > 1 and reduce:
        return reduceDivision(division=division, newdiv=newdiv, assignedSlots=assignedSlots)
    return newdiv


def reduceDivision(division: division_t,
                   newdiv: division_t,
                   assignedSlots: list[int],
                   maxslots=20
                   ) -> division_t:
    assert len(newdiv) > 1
    subdiv = math.lcm(*newdiv)
    numslots = subdiv * len(newdiv)
    if numslots > maxslots:
        return newdiv
    expandeddiv = (numslots,)
    oldgrid = divisionGrid0(division=division)
    expandedgrid = divisionGrid0(expandeddiv)
    newslots = resnap(assignedSlots, oldgrid, expandedgrid)
    newdiv2 = simplifyDivision(expandeddiv, newslots, reduce=False)
    return newdiv2 if numslots < sum(newdiv) else newdiv


@cache
def gridDurationsFlat(beatDuration: F, division: int | division_t
                      ) -> list[F]:
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
def divisionGrid0(division: division_t, beatDuration: F = F(1)) -> list[F]:
    durations = gridDurationsFlat(beatDuration, division)
    grid = [F0]
    grid.extend(iterlib.partialsum(durations))
    return grid


@cache
def primeFactors(d: int, excludeBinary=False) -> set:
    """calculate the prime factors of d"""
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


def applyDurationRatio(notations: list[Notation],
                       division: int | division_t,
                       beatOffset: F,
                       beatDur: F
                       ) -> None:
    """
    Applies a duration ratio to each notation.

    A duration ratio converts the actual duration of a notation to its
    notated value and is used to render these as tuplets later

    Args:
        notations: the notations inside the period beatOffset:beatOffset+beatDur
        division: the division of the beat/subbeat.
        beatOffset: the start of the beat
        beatDur: the duration of the beat

    """
    def _apply(durRatio: F, notations: list[Notation]):
        if durRatio == F1:
            for n in notations:
                if not n.durRatios:
                    n.durRatios = (durRatio,)
        else:
            for n in notations:
                n.durRatios += (durRatio,)

        assert all(bool(n.durRatios) for n in notations)

    if isinstance(division, int) or len(division) == 1:
        num: int = division if isinstance(division, int) else division[0]
        durRatio = F(*quantdata.durationRatios[num])
        _apply(durRatio, notations)

    else:
        numSubBeats = len(division)
        now = beatOffset
        dt = beatDur / numSubBeats
        durRatio = F(*quantdata.durationRatios[numSubBeats])
        _apply(durRatio, notations)
        numNotations = 0
        for subdiv in division:
            subdivEnd = now + dt
            subdivNotations = [n for n in notations
                               if now <= n.qoffset and n.end <= subdivEnd]
            applyDurationRatio(notations=subdivNotations, division=subdiv,
                               beatOffset=now, beatDur=dt)
            now += dt
            numNotations += len(subdivNotations)
        assert numNotations == len(notations)

    assert all(n.durRatios is not None for n in notations), f"{notations=}"


def beatToTree(notations: list[Notation], division: int | division_t,
               beatOffset: F, beatDur: F
               ) -> _node.Node:
    if isinstance(division, tuple) and len(division) == 1:
        division = division[0]
    if isinstance(division, int):
        durRatio = quantdata.durationRatios[division]
        return _node.Node(notations, ratio=durRatio)  # type: ignore

    # assert isinstance(division, tuple) and len(division) >= 2
    numSubBeats = len(division)
    now = beatOffset
    dt = beatDur/numSubBeats
    durRatio = quantdata.durationRatios[numSubBeats]
    items = []
    for subdiv in division:
        subdivEnd = now + dt
        subdivNotations = [n for n in notations if now <= n.qoffset < subdivEnd and n.end <= subdivEnd]
        if subdiv == 1:
            items.extend(subdivNotations)
        else:
            items.append(beatToTree(notations=subdivNotations, division=subdiv, beatOffset=now, beatDur=dt))
        now += dt
    return _node.Node(items, ratio=durRatio)


def breakNotationsByBeat(
        notations: list[Notation],
        beatOffsets: _t.Sequence[F]
        ) -> list[tuple[F, F, list[Notation]]]:
    """
    Break the given notations between the given beat offsets, returns the 

    **NB**: Any notations starting after the last offset will not be considered!

    Args:
        notations: the notations to split
        beatOffsets: the boundaries. All notations should be included within the
            boundaries of the bigen offsets

    Returns:
        a list of tuples ((start beat, end beat), notation)

    """
    assert beatOffsets[0] == notations[0].offset
    assert beatOffsets[-1] == notations[-1].end

    timespans = [(beat0, beat1) for beat0, beat1 in pairwise(beatOffsets)]
    splitEvents = []
    for ev in notations:
        if ev.duration > 0:
            splitEvents.extend(ev.splitAtOffsets(beatOffsets))
        else:
            splitEvents.append(ev)

    eventsPerTimespan = []
    for start, end in timespans:
        eventsInTimespan = [ev for ev in splitEvents if start <= ev.offset < end]
        eventsPerTimespan.append(eventsInTimespan)
        assert sum(ev.duration for ev in eventsInTimespan) == end - start
        assert all(start <= ev.offset <= ev.end <= end
                   for ev in eventsInTimespan)
    return [(start, end, events) for (start, end), events in zip(timespans, eventsPerTimespan)]


