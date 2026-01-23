"""
Utilities used during quantization
"""
from __future__ import division, annotations
import numpy as np
from bisect import bisect
import numpyx as npx

from functools import cache
from emlib import mathlib
from itertools import pairwise, accumulate

from maelzel.common import F, F0, F1
from .notation import Notation, Snapped
from . import quantdata
from . import util

import typing as _t
if _t.TYPE_CHECKING:
    _T = _t.TypeVar("_T", int, float, F)
    from .common import division_t


@cache
def divisionNumSlots(div: division_t) -> int:
    flatdiv = div if all(isinstance(item, int) for item in div) else flattenDiv(div)
    return sum(flatdiv)


def outerTuplet(div: division_t) -> tuple[int, int]:
    lendiv = len(div)
    outer = div[0] if lendiv == 1 else lendiv
    if mathlib.ispowerof2(outer):
        return (1, 1)
    den = util.highestPowerLowerOrEqualTo(outer, base=2)
    return outer, den


def flattenDiv(div: tuple[int | tuple, ...]) -> _t.Iterator[int]:
    for item in div:
        if isinstance(item, int):
            yield item
        else:
            yield from flattenDiv(item)


@cache
def isNestedTupletDivision(div: division_t) -> bool:
    if isinstance(div, int):
        # A shortcut division, like 3 or 5
        return False
    return not mathlib.ispowerof2(len(div)) and any(not mathlib.ispowerof2(subdiv) for subdiv in div)


# def divisionDensity(division: division_t) -> int:
#     return max(division) * len(division)


def asymettry(a, b) -> float:
    if a < b:
        a, b = b, a
    return float(a/b)


def _fitToGridList(offsets: list[float], grid: list[float]) -> list[int]:
    out = []
    idx = 0
    gridlen = len(grid)
    for offset in offsets:
        idx = bisect(grid, offset, lo=idx)
        if idx == gridlen:
            idx = gridlen - 1
        elif idx != 0:
            idxleft = idx - 1
            idx = idx if grid[idx] - offset < offset - grid[idxleft] else idxleft
        out.append(idx)
    return out


def _fitToGridNumpy(offsets: np.ndarray[float], grid: np.ndarray[float]) -> list[int]:
    return npx.nearestindexes(grid, offsets).tolist()


def assignSlots(fgrid: list[float] | np.ndarray[float],
                offsets: list[float] | np.ndarray[float]
                ) -> list[int]:
    """
    Snap unquantized events to a given grid

    Args:
        fgrid: the grid as floats.
        offsets: the offsets to fit

    Returns:
        tuple (assigned slots, quantized events)
    """
    # assignedSlots = _fitEventsToGridNearest(events=notations, grid=grid)
    if isinstance(fgrid, np.ndarray):
        assignedSlots = _fitToGridNumpy(offsets, fgrid)
    else:
        assert isinstance(fgrid, list)
        assignedSlots = _fitToGridList(offsets, fgrid)
    return assignedSlots


def makeSnapped(notations: list[Notation], slots: list[int], grid: _t.Sequence[F]
                ) -> list[Snapped]:
    snapped: list[Snapped] = []
    lastidx = len(grid) - 1

    for idx, n in enumerate(notations[:-1]):
        slot0 = slots[idx]
        offset0 = grid[slot0]
        if slot0 < lastidx:
            offset1 = grid[slots[idx+1]]
            dur = offset1 - offset0
            if not n.isRest or dur > 0:
                # do not add gracenote rests
                snapped.append(Snapped(n, offset0, dur))
        elif not n.isRest:
            # is it the last slot (as grace note?)
            snapped.append(Snapped(n, offset0, F0))

    last = notations[-1]
    lastoffset = grid[slots[-1]]
    lastdur = grid[-1] - lastoffset
    if lastdur > 0 or not last.isRest:
        snapped.append(Snapped(last, lastoffset, duration=lastdur))
    # assert sum(n.duration for n in snapped) == grid[-1]
    return snapped


# def resnap(assignedSlots: _t.Sequence[int],
#            oldgrid: _t.Sequence[F],
#            newgrid: _t.Sequence[F]
#            ) -> list[int]:
#     minslot = 0
#     maxslot = len(newgrid)
#     reassigned: list[int] = []
#     for slot in assignedSlots:
#         oldoffset = oldgrid[slot]
#         for newslotidx in range(minslot, maxslot):
#             newoffset = newgrid[newslotidx]
#             if oldoffset == newoffset:
#                 reassigned.append(newslotidx)
#                 minslot = newslotidx
#                 break
#         else:
#             raise ValueError(f"No corresponding slot {oldoffset=}, {newgrid=}")
#
#     if not len(reassigned) == len(assignedSlots):
#         oldoffsets = [oldgrid[i] for i in assignedSlots]
#         newoffsets = [newgrid[i] for i in reassigned]
#         from .common import logger
#         logger.error(f'{oldoffsets=}, {newoffsets=}, {assignedSlots=}, {reassigned=}, {oldgrid=}, {newgrid=}')
#         raise RuntimeError("resnap error")
#     return reassigned


@cache
def _makeset(start: int, end: int, exclude):
    return set(x for x in range(start, end) if x not in exclude)


_primes = {3, 5, 7, 11, 13, 19}


def simplifyDivisionWithSlots(division: division_t, assignedSlots: list[int]
                              ) -> tuple[division_t, list[int]] | tuple[None, None]:

    if len(assignedSlots) == 1 and assignedSlots[0] == 0:
        newdiv = (1,)
        return (newdiv, assignedSlots) if newdiv != division else (None, None)

    lastslot = sum(subdiv for subdiv in division)
    if all(slot == 0 or slot == lastslot for slot in assignedSlots):
        newdiv = (1,)
        newslots = [0 if slot == 0 else 1 for slot in assignedSlots]
        return newdiv, newslots

    if len(division) == 1 and (d0 := division[0]) % 2 == 1 and d0 in _primes:
        return None, None

    if len(division) > 1 and all(subdiv == 1 for subdiv in division):
        newdiv = (len(division),)
        simplified, newslots = simplifyDivisionWithSlots(newdiv, assignedSlots)
        if simplified is not None and simplified != division:
            return simplified, newslots
        return newdiv, assignedSlots

    reduced: list[int] = []
    slots: list[int] = []
    cs = 0
    cs2 = 0
    assigned = set(assignedSlots)
    slotSizes = [s1 - s0 for s0, s1 in pairwise(assignedSlots)]
    numSlots = sum(division)
    slotSizes.append(numSlots - assignedSlots[-1])

    for subdiv in division:
        if cs in assigned:
            slots.append(cs2)

        if subdiv == 1:
            reduced.append(1)
        elif all(s not in assigned for s in range(cs+1, cs+subdiv)):
            # elif not anyBetween(assignedSlots, cs+1, cs+subdiv):
            # no subslots assigned
            reduced.append(1)
        elif subdiv % 2 == 0:
            if subdiv == 4:
                if cs+1 not in assigned and cs+3 not in assigned:
                    reduced.append(2)
                    assert cs+2 in assigned
                    slots.append(cs2+1)
                else:
                    reduced.append(4)
                    slots.extend(cs2+i for i in range(1, 4) if cs+i in assigned)
            elif subdiv == 6:
                #   x       x
                # x 0 0 1 0 0 -> 2
                # x 0 1 0 1 0 -> 3

                vec = [x in assigned for x in range(cs, cs+6)]
                if not vec[1] and not vec[5]:
                    # 0 1 2 3 4 5
                    #   -       -
                    if not vec[2] and not vec[4]:
                        # x - - x - -
                        reduced.append(2)
                        assert vec[3]
                        slots.append(cs2+1)
                    elif not vec[3]:
                        # x - . - . -
                        reduced.append(3)
                        if vec[2]:
                            slots.append(cs2+1)
                        if vec[4]:
                            slots.append(cs2+2)
                    else:
                        reduced.append(6)
                        slots.extend(cs2 + i for i, x in enumerate(vec[1:]) if x)
                else:
                    reduced.append(6)
                    slots.extend(cs2 + i for i, x in enumerate(vec[1:]) if x)
            elif subdiv == 8:
                # 1 0 1 0 1 0 1 0
                if cs+1 not in assigned and cs+3 not in assigned and cs+5 not in assigned and cs+7 not in assigned:
                    # if {cs+1, cs+3, cs+5, cs+7}.isdisjoint(assigned):
                    if cs+2 not in assigned and cs+6 not in assigned:
                        reduced.append(2)
                        assert cs+4 in assigned
                        slots.append(cs2+1)
                    else:
                        reduced.append(4)
                        if cs+2 in assigned:
                            slots.append(cs2+1)
                        if cs+4 in assigned:
                            slots.append(cs2+2)
                        if cs+6 in assigned:
                            slots.append(cs2+3)
                else:
                    reduced.append(8)
                    slots.extend(cs2+i for i in range(1, 8) if cs+i in assigned)
            # from here on: even subdiv (10, 12, ...)
            elif cs+1 not in assigned and cs+subdiv-1 not in assigned:
                # The second and last slot are not assigned
                mid = cs+subdiv//2
                if mid in assigned:
                    if all(i not in assigned for i in range(cs+1, cs+subdiv) if i != mid):
                        # if _makeset(cs+1, cs+subdiv, (cs+subdiv//2,)).isdisjoint(assigned):
                        # 1 0 0 0 0 1 0 0 0 0 -> 2
                        reduced.append(2)
                        slots.append(cs2+1)
                    else:
                        reduced.append(subdiv)
                        slots.extend(cs2+i for i in range(1, subdiv) if cs+i in assigned)
                elif not any(x in assigned for x in range(cs+1, cs+subdiv, 2)):
                    reduced.append(subdiv//2)
                    1/0
                    # TODO
                else:
                    reduced.append(subdiv)
                    slots.extend(cs2+i for i in range(1, subdiv) if cs+i in assigned)
            else:
                reduced.append(subdiv)
                slots.extend(cs2+i for i in range(1, subdiv) if cs+i in assigned)
        elif subdiv == 9:
            # assigned = set(assignedSlots)
            if {cs+1, cs+2, cs+4, cs+5, cs+7, cs+8}.isdisjoint(assignedSlots):
                reduced.append(3)
                if cs+3 in assigned:
                    slots.append(cs2+1)
                if cs+6 in assigned:
                    slots.append(cs2+2)
            else:
                reduced.append(9)
                slots.extend(cs2+i for i in range(1, subdiv) if cs+i in assigned)
        elif subdiv == 15:
            if all(cs+x not in assigned for x in range(1, 15) if x % 5 != 0):
                reduced.append(3)
                if cs+5 in assigned:
                    slots.append(cs2+1)
                if cs+10 in assigned:
                    slots.append(cs2+2)
            elif all(cs+x not in assigned for x in range(1, 15) if x % 3 != 0):
                reduced.append(5)
                if cs+3 in assigned:
                    slots.append(cs2+1)
                if cs+6 in assigned:
                    slots.append(cs2+2)
                if cs+9 in assigned:
                    slots.append(cs2+3)
                if cs+12 in assigned:
                    slots.append(cs2+4)
            else:
                reduced.append(subdiv)
                slots.extend(cs2+i for i in range(1, subdiv) if cs+i in assigned)
        else:
            reduced.append(subdiv)
            slots.extend(cs2+i for i in range(1, subdiv) if cs+i in assigned)
            # slots.extend(assignedSlotsBetween(cs, 1, subdiv, cs2))
        cs += subdiv
        cs2 += reduced[-1]

    newdiv: division_t = tuple(reduced)
    assert len(newdiv) == len(division), f'{division=}, {newdiv=}'

    if all(subdiv == 1 for subdiv in newdiv):
        N = len(newdiv)
        newdiv = (N,)
        if N % 2 == 0 or N % 3 == 0:
            simplified2, slots2 = simplifyDivisionWithSlots(newdiv, slots)
            if simplified2 is not None:
                newdiv, slots = simplified2, slots2

    if newdiv == division:
        return None, None

    if len(slots) != len(assignedSlots):
        # grace notes share slots
        numSlots2 = sum(subdiv for subdiv in newdiv)
        for i, size in enumerate(slotSizes):
            if size == 0:
                if assignedSlots[i] == numSlots:
                    # gracenote at the end of the beat, actually a gracenote to
                    # the next beat
                    slots.append(numSlots2)
                else:
                    slots.insert(i, slots[i])

    assert len(slots) == len(assignedSlots), f"{assignedSlots=}, {slots=}, {division=} -> {newdiv=}"
    return newdiv, slots


# def simplifyDivision(division: division_t, assignedSlots: _t.Sequence[int]
#                      ) -> division_t:
#     """
#     Checks if a division (a partition of the beat) can be substituted by a simpler one
#
#     Args:
#         division: the division to simplify
#         assignedSlots: assigned slots for this division
#
#     Returns:
#         the simplified version or the original division if no simplification is possible
#     """
#     if len(assignedSlots) == 1 and assignedSlots[0] == 0:
#         return (1,)
#
#     if len(division) == 1 and (d0 := division[0]) % 2 == 1 and d0 in (3, 5, 7, 11, 13):
#         return division
#
#     cs = 0
#     reduced = []
#
#     for subdiv in division:
#         if subdiv == 1:
#             reduced.append(1)
#         elif not any(1 <= s - cs < subdiv for s in assignedSlots):
#             # only the first slot is assigned
#             reduced.append(1)
#         elif subdiv % 2 == 0:
#             assigned = set(assignedSlots)
#             if subdiv == 4:
#                 if cs+1 not in assigned and cs+3 not in assigned:
#                     reduced.append(2)
#                 else:
#                     reduced.append(4)
#             elif subdiv == 6:
#                 #   x       x
#                 # x 0 0 1 0 0 -> 2
#                 # x 0 1 0 1 0 -> 3
#                 if cs+1 not in assigned and cs+5 not in assigned:
#                     if cs+2 not in assigned and cs+4 not in assigned:
#                         reduced.append(2)
#                     elif cs+3 not in assigned:
#                         reduced.append(3)
#                     else:
#                         reduced.append(6)
#                 else:
#                     reduced.append(6)
#             elif subdiv == 8:
#                 # 1 0 1 0 1 0 1 0
#                 if cs+1 not in assigned and cs+3 not in assigned and cs+5 not in assigned and cs+7 not in assigned:
#                 # if {cs+1, cs+3, cs+5, cs+7}.isdisjoint(assigned):
#                     if cs+2 not in assigned and cs+6 not in assigned:
#                         reduced.append(2)
#                     else:
#                         reduced.append(4)
#                 else:
#                     reduced.append(8)
#             # from here on: even subdiv (10, 12, ...)
#             elif cs+1 not in assigned and cs+subdiv-1 not in assigned:
#                 # The second and last slot are not assigned
#                 if cs+subdiv//2 in assigned:
#                     if _makeset(cs+1, cs+subdiv, (cs+subdiv//2,)).isdisjoint(assigned):
#                         # 1 0 0 0 0 1 0 0 0 0 -> 2
#                         reduced.append(2)
#                     else:
#                         reduced.append(subdiv)
#                 # elif set(range(cs+1, cs+subdiv, 2)).isdisjoint(assigned):
#                 elif not any(x in assigned for x in range(cs+1, cs+subdiv, 2)):
#                     reduced.append(subdiv//2)
#                 else:
#                     reduced.append(subdiv)
#             else:
#                 reduced.append(subdiv)
#         elif subdiv == 9:
#             # assigned = set(assignedSlots)
#             if {cs+1, cs+2, cs+4, cs+5, cs+7, cs+8}.isdisjoint(assignedSlots):
#                 reduced.append(3)
#             else:
#                 reduced.append(9)
#         else:
#             reduced.append(subdiv)
#         cs += subdiv
#
#     newdiv: division_t = tuple(reduced)
#     assert len(newdiv) == len(division), f'{division=}, {newdiv=}'
#
#     if all(subdiv == 1 for subdiv in newdiv):
#         newdiv = (len(newdiv),)
#
#     return newdiv


# def reduceDivision(division: division_t,
#                    newdiv: division_t,
#                    assignedSlots: _t.Sequence[int],
#                    maxslots=40
#                    ) -> division_t:
#     assert len(newdiv) > 1
#     subdiv = math.lcm(*newdiv)
#     numslots = subdiv * len(newdiv)
#     if numslots > maxslots or numslots >= sum(newdiv):
#         return newdiv
#     expandeddiv = (numslots,)
#     oldgrid = divisionGrid0(division=division)
#     expandedgrid = divisionGrid0(expandeddiv)
#     newslots = resnap(assignedSlots, oldgrid, expandedgrid)
#     newdiv2 = simplifyDivision(expandeddiv, newslots)
#     return newdiv2
#

@cache
def gridDurationsFlat(beatDuration: F, division: division_t
                      ) -> list[F]:
    assert isinstance(division, tuple)
    numDivisions = len(division)
    subdivDur = beatDuration / numDivisions
    grid = []
    for subdiv in division:
        if isinstance(subdiv, int):
            dt = subdivDur / subdiv
            grid.extend([dt] * subdiv)
        else:
            grid.extend(gridDurationsFlat(subdivDur, subdiv))
    return grid


@cache
def divisionGrid0(division: division_t, beatDuration: F) -> list[F]:
    durations = gridDurationsFlat(beatDuration, division)
    grid = [F0]
    grid.extend(accumulate(durations))
    assert grid[-1] == beatDuration
    return grid


@cache
def divisionGrid0Float(division: division_t, beatDuration: F) -> tuple[list[F], list[float]]:
    grid = divisionGrid0(division=division, beatDuration=beatDuration)
    fgrid = [float(slot) for slot in grid]
    return grid, fgrid


@cache
def divisionGrid0Array(division: division_t, beatDuration: F = F(1)) -> tuple[list[F], np.ndarray[float]]:
    grid = divisionGrid0(division=division, beatDuration=beatDuration)
    npgrid = np.array([float(slot) for slot in grid], dtype=float)
    return grid, npgrid


@cache
def primeFactors(d: int, excludeBinary=False) -> set:
    """calculate the prime factors of d"""
    factors = set()
    for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        if d % p == 0:
            factors.add(p)
    if not excludeBinary:
        if d % 2 == 0:
            factors.add(2)
    return factors


def fixGlissWithinTiesInPlace(notations: _t.Sequence[Notation]) -> None:
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
                if not n.durRatios:
                    n.durRatios = (durRatio,)
                else:
                    n.durRatios += (durRatio,)

    def notationsBetween(notations: list[Notation], start: F, end: F) -> list[Notation]:
        # gracenote policy: if a gracenote is at the start, we keep it,
        # at the end we don't
        # No partial notations: a notation needs to fit between start and end
        out = []
        for n in notations:
            noffset = n.offset
            assert noffset is not None
            if noffset >= end:
                break
            if noffset >= start and n.end <= end:
                out.append(n)
        return out

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
        for i, subdiv in enumerate(division):
            subdivEnd = now + dt
            subdivNotations = notationsBetween(notations, now, subdivEnd)
            if i == len(division) - 1 and notations[-1].isGracenote and notations[-1].offset == subdivEnd:
                endgraces = []
                for n in reversed(notations):
                    if not n.isGracenote or (n.offset is not None and n.offset < subdivEnd):
                        break
                    endgraces.append(n)
                if endgraces:
                    subdivNotations.extend(reversed(endgraces))
            applyDurationRatio(notations=subdivNotations, division=subdiv,
                               beatOffset=now, beatDur=dt)
            now += dt
            numNotations += len(subdivNotations)
        if numNotations != len(notations):
            for i, n in enumerate(notations):
                print(i, n)
            raise RuntimeError(f"Failed to apply durations, len mismatch, {numNotations=} != {len(notations)=}, {beatOffset=}, {beatDur=}")


def breakNotationsByBeat(notations: list[Notation],
                         offsets: _t.Sequence[F],
                         forcecopy=False
                         ) -> list[tuple[F, F, list[Notation]]]:
    """
    Break the given unquantized notations between the given beat offsets

    **NB**: Any notations starting after the last offset will not be considered!

    Args:
        notations: the notations to split
        offsets: the boundaries. All notations should be included within the
            boundaries of the given offsets
        forcecopy: ensures that all returned notations are copies of the
            input notations

    Returns:
        a list of tuples (startbeat, endbeat, notation)

    """
    assert not any(n.isQuantized() for n in notations)
    assert offsets[0] == notations[0].offset, f"{offsets=}, {notations=}"
    assert offsets[-1] == notations[-1].end, f"{offsets=}, {notations=}, {offsets[-1]=}, {notations[-1].end=}"

    timespans = [(beat0, beat1) for beat0, beat1 in pairwise(offsets)]
    splitEvents = []

    for ev in notations:
        if ev.duration > 0:
            splitEvents.extend(ev.splitAtOffsets(offsets, forcecopy=True))
        else:
            splitEvents.append(ev.copy() if forcecopy else ev)

    eventsPerTimespan = []
    for start, end in timespans:
        eventsInTimespan = []
        for ev in splitEvents:
            if ev.duration > 0:
                if start <= ev.offset and ev.end <= end:
                    eventsInTimespan.append(ev)
            else:
                if start <= ev.offset and ev.end < end:
                    eventsInTimespan.append(ev)

        # eventsInTimespan = [ev for ev in splitEvents if start <= ev.offset and ev.end <= end]
        eventsPerTimespan.append(eventsInTimespan)
    assert len(splitEvents) == sum(len(evs) for evs in eventsPerTimespan)

    return [(start, end, events) for (start, end), events in zip(timespans, eventsPerTimespan)]


def notationAtOffset(notations: list[Notation], offset: F, exact: bool
                     ) -> int | None:
    """
    Return the index of the notation at the given offset

    Args:
        notations: the notations to search. They all should have an offset set
        offset: the offset to search for
        exact: if True, the notation should start exactly at offset

    Returns:
        the index of the notation present at the given offset, or None
    """
    if exact:
        for i, n in enumerate(notations):
            noffset = n.qoffset
            if noffset > offset:
                return None
            if noffset == offset:
                return i
    else:
        for i, n in enumerate(notations):
            noffset = n.qoffset
            if noffset > offset:
                return None
            if noffset + n.duration >= offset:
                return i
    return None


def insertRestAt(offset: F, seq: list[Notation], fallbackdur=F1) -> Notation:
    """
    Assuming that offset doesn't intersect any notation in seq, create a rest starting at offset

    The duration of the rest will be until the next notation or fallbackdur if the
    rest is at the end of the seq.

    Args:
        offset: the offset to insert a rest at.
        seq: a sequence of Notations
        fallbackdur: the duration of the rest if it is inserted past the last event

    Returns:
        the created rest, which will be part of seq

    """
    assert notationAtOffset(seq, offset, exact=True) is None
    nextidx = next((i for i, n in enumerate(seq) if n.qoffset > offset), None)
    if nextidx is None:
        # offset past last
        n = Notation.makeRest(duration=fallbackdur, offset=offset)
        seq.append(n)
    else:
        nextnot = seq[nextidx]
        n = Notation.makeRest(duration=nextnot.qoffset-offset, offset=offset)
        seq.insert(nextidx, n)
    return n


def insertRestEndingAt(end: F, seq: list[Notation]) -> Notation | None:
    """
    Assuming that end doesn't intersect any notation in seq, create a rest ending at end

    The duration of the rest will be from the previous notation or from 0

    Args:
        end: the time at which the inserted rest should end
        seq: a list of notations

    Returns:
        the inserted rest, None if nothing was inserted
    """
    idx = next((i for i, n in enumerate(seq) if n.qoffset >= end), None)
    if idx is None:
        # end is past last
        last = seq[-1]
        if last.end >= end:
            raise RuntimeError(f"{end} overlaps with {last}, notations: {seq}")
        rest = Notation.makeRest(duration=end - last.end, offset=last.end)
        seq.append(rest)
    else:
        if idx == 0:
            rest = Notation.makeRest(duration=seq[0].qoffset, offset=0)
            seq.insert(0, rest)
        else:
            previdx = idx - 1
            last = seq[previdx]
            if last.end < end:
                rest = Notation.makeRest(duration=end - last.end, offset=last.end)
                seq.insert(idx, rest)
            else:
                assert last.end == end, f"{last=}, {end=}, {seq=}"
                return None
    return rest


# def isRegularDuration(symdur: F) -> bool:
#     """
#     True if symdur is a regular duration
#
#     Args:
#         symdur: the symbolic duration of an event / node
#
#     Returns:
#         True if this duration is regular - can be notated with only one figure,
#         using dots or not
#
#     """
#     return symdur.denominator in (1, 2, 4, 8, 16, 32) and symdur.numerator in (1, 2, 3, 4, 7)


def splitDots(dur: F | tuple[int, int]) -> tuple[F, int]:
    """
    Given a symbolic duration as a fraction, split into figure duration and number of dotrs

    Args:
        dur: the symbolic duration. 1/1: quarter, 3/2: dotted quarter, 3/8: dotten quarter, etc.

    Returns:
        a tuple (maindur: F, numdots: int) where maindur can be 1/1, 2/1, etc. or
        1/2, 1/4, 1/8, etc.

    Example
    -------

        >>> splitDots((7, 8))  # eighth note with two dots
        (F(1, 2), 2)

    """
    if isinstance(dur, tuple):
        num, den = dur
    else:
        num, den = dur.numerator, dur.denominator
    if num == 1:
        assert mathlib.ispowerof2(den), f"Invalid duration: {dur}"
        return dur if isinstance(dur, F) else F(num, den), 0
    elif num == 2 or num == 4:
        assert den == 1
        return dur if isinstance(dur, F) else F(num, den), 0
    if num == 3:
        # 3/2=1., 3/4=1/2., etc
        return F(2, den), 1
    elif num == 7:
        # 7/4=1.., 7/8=1/2..
        return F(4, den), 2
    elif num == 15:
        return F(8, den), 3
    elif num == 31:
        return F(16, den), 4
    elif num == 63:
        return F(32, den), 5
    else:
        raise ValueError(f"Invalid duration: {dur}")


def fillSpan(notations: list[Notation], start: F, end: F
             ) -> list[Notation]:
    """
    Fill a beat/measure with silences / extend unset durations to next notation

    After calling this, the returned list of notations should fill the given
    duration exactly. This function is normally called prior to quantization

    Args:
        notations: a list of unquantized notations inside the beat
        end: the duration to fill
        start: the starting time to fill

    Returns:
        a list of notations which fill the beat exactly

    .. note::

        If any notation has an unset duration, this will extend either to
        the next notation or to fill the given duration

    """
    out = []
    now: F = start
    duration: F = end - start

    if not notations:
        out.append(Notation.makeRest(duration, offset=start))
        return out

    assert all(not n.isQuantized() for n in notations)

    if (n0offset := notations[0].qoffset) > start:
        out.append(Notation.makeRest(n0offset-start, offset=start))
        now = n0offset

    for n0, n1 in pairwise(notations):
        n0offset = n0.qoffset
        if n0offset > now:
            # there is a gap, fill it with a rest
            out.append(Notation.makeRest(offset=now, duration=n0offset - now))
        if n0.duration is None:
            out.append(n0.clone(duration=n1.qoffset - n0offset, spanners=n0.spanners))
        else:
            out.append(n0)
            n0end = n0.end
            if n0end < n1.qoffset:
                out.append(Notation.makeRest(offset=n0end, duration=n1.qoffset - n0end))
        now = n1.qoffset

    # last event
    n = notations[-1]
    assert n.duration is not None
    out.append(n)
    if n.end < end:
        out.append(Notation.makeRest(offset=n.end, duration=end-n.end))
    assert sum(n.duration for n in out) == duration
    assert all(start <= n.offset <= end for n in out)
    return out


@cache
def slotsAtSubdivisions(divs: tuple[int, ...]) -> list[int]:
    return [0] + list(accumulate(divs))
