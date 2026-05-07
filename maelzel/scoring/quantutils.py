"""
Utilities used during quantization
"""
from __future__ import division, annotations
from bisect import bisect

from functools import cache
from itertools import pairwise, accumulate

from maelzel.common import F, F0, F1
from .notation import Notation, Snapped
from maelzel._mathutils import (ispowerof2, highestPowerLowerOrEqualTo)
from . import quantdata


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
    if ispowerof2(outer):
        return (1, 1)
    den = highestPowerLowerOrEqualTo(outer, base=2)
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
    return not ispowerof2(len(div)) and any(not ispowerof2(subdiv) for subdiv in div)


def asymettry(a, b) -> float:
    if a < b:
        a, b = b, a
    return float(a/b)


def assignSlots(events: list[Notation], div: division_t, beatDuration: F) -> list[int]:
    grid = divisionGrid0(div, beatDuration=beatDuration)
    return assignSlotsFrac(events, grid)


def assignSlotsFrac(events: list[Notation], grid: list[F]) -> list[int]:
    result: list[int] = []
    g = 0  # current grid pointer
    for ev in events:
        pos = ev.offset
        # Advance until grid[g+1] is not closer than grid[g]
        while g + 1 < len(grid) - 1 and grid[g + 1] <= pos:
            g += 1

        # At this point grid[g] <= pos < grid[g+1] (or g is the last slot)
        if g + 1 < len(grid) and (pos - grid[g]) > (grid[g + 1] - pos):
            result.append(g + 1)
        else:
            result.append(g)

    return result


def _fitToGridListBisect(offsets: list[float], grid: list[float]) -> list[int]:
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


# def _fitToGridNumpy(offsets: NDArray[np.float64], grid: NDArray[np.float64]) -> list[int]:
#     return npx.nearestindexes(grid, offsets).tolist()


# def assignSlotsForPerfectFit(events: list[Notation],
#                              div: tuple[int, ...],
#                              beatDuration: F,
#                              beatOffset: F = F0
#                              ) -> list[int]:
#     numParts = len(div)
#     partDur = beatDuration / numParts
#     lastSlot = sum(div)
#     beatEnd = beatOffset + beatDuration
#
#     partSlotStart = [0] * numParts
#     for i in range(1, numParts):
#         partSlotStart[i] = partSlotStart[i - 1] + div[i - 1]
#
#     result: list[int] = []
#     for ev in events:
#         if ev.offset == beatEnd:
#             assert ev.duration == 0
#             result.append(lastSlot)
#         else:
#             partIndex, remainder = divmod(ev.offset - beatOffset, partDur)
#             partIndex = int(partIndex)
#             slotDur = partDur / div[partIndex]
#             subIndex, _ = divmod(remainder, slotDur)
#             result.append(partSlotStart[partIndex] + int(subIndex))
#     return result


def snappedToDivision(notations: _t.Sequence[Notation],
                      slots: _t.Sequence[int],
                      div: division_t,
                      beatDuration: F
                      ) -> list[Snapped]:
    """
    Creates Snapped variants for each notation given

    Args:
        notations: the notations to snap
        slots: the assigned slots, one for each notation
        div: the division of the beat
        beatDuration: the duration of the beat

    Returns:
        a list of Snapped objects, one for each notation
    """
    grid = divisionGrid0(div, beatDuration)
    return snappedToGrid(notations, slots=slots, grid=grid)


def snappedToGrid(notations: _t.Sequence[Notation],
                  slots: _t.Sequence[int],
                  grid: _t.Sequence[F]
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
    return snapped


@cache
def _makeset(start: int, end: int, exclude):
    return set(x for x in range(start, end) if x not in exclude)


_primes = {3, 5, 7, 11, 13, 17, 19}


def simplifyDivisionWithSlots(division: division_t, assignedSlots: list[int]
                              ) -> tuple[division_t, list[int]] | None:
    """
    Try to find a simpler division which represents the same time as the given slots

    Args:
        division: the division to simplify
        assignedSlots: the already assigned slots

    Returns:
        a tuple (newdiv, newslots) or None if not possible to simplify

    """

    if len(assignedSlots) == 1 and assignedSlots[0] == 0:
        newdiv = (1,)
        return (newdiv, assignedSlots) if newdiv != division else None

    lastslot = sum(subdiv for subdiv in division)
    if all(slot == 0 or slot == lastslot for slot in assignedSlots):
        newdiv = (1,)
        newslots = [0 if slot == 0 else 1 for slot in assignedSlots]
        return newdiv, newslots

    lendiv = len(division)
    if lendiv == 1 and (d0 := division[0]) % 2 == 1 and d0 in _primes:
        return None

    if lendiv > 1 and all(subdiv == 1 for subdiv in division):
        newdiv = (lendiv,)
        if simplifyRes := simplifyDivisionWithSlots(newdiv, assignedSlots):
            simplified, newSlots = simplifyRes
            assert simplified != division
            return simplified, newSlots
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
                        slots.extend(cs2 + i for i, x in enumerate(vec[1:], start=1) if x)
                else:
                    reduced.append(6)
                    slots.extend(cs2 + i for i, x in enumerate(vec[1:], start=1) if x)

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
                # elif not any(x in assigned for x in range(cs+1, cs+subdiv, 2)):
                #     reduced.append(subdiv//2)
                #     1/0
                #     # TODO
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
                for j in range(1, 5):
                    if cs+j*3 in assigned:
                        slots.append(cs2+j)
            else:
                reduced.append(subdiv)
                slots.extend(cs2+i for i in range(1, subdiv) if cs+i in assigned)
        else:
            reduced.append(subdiv)
            slots.extend(cs2+i for i in range(1, subdiv) if cs+i in assigned)
        cs += subdiv
        cs2 += reduced[-1]

    newdiv: division_t = tuple(reduced)
    assert len(newdiv) == len(division), f'{division=}, {newdiv=}'

    if all(subdiv == 1 for subdiv in newdiv):
        N = len(newdiv)
        newdiv = (N,)
        if N % 2 == 0 or N % 3 == 0:
            if simplifyRes := simplifyDivisionWithSlots(newdiv, slots):
                newdiv, slots = simplifyRes

    if newdiv == division:
        return None

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

    assert isinstance(slots, list) and len(slots) == len(assignedSlots), f"{assignedSlots=}, {slots=}, {division=} -> {newdiv=}"
    return newdiv, slots


def _simplifyDivisionWithSlots(division: division_t, assignedSlots: list[int]
                              ) -> tuple[division_t, list[int]] | None:
    """
    lorem ipsum

    Args:
        division: the division to simplify
        assignedSlots: the already assigned slots

    Returns:
        a tuple (newdiv, newslots) or None if not possible to simplify

    """

    if len(assignedSlots) == 1 and assignedSlots[0] == 0:
        newdiv = (1,)
        return (newdiv, assignedSlots) if newdiv != division else None

    lastslot = sum(subdiv for subdiv in division)
    if all(slot == 0 or slot == lastslot for slot in assignedSlots):
        newdiv = (1,)
        newslots = [0 if slot == 0 else 1 for slot in assignedSlots]
        return newdiv, newslots

    lendiv = len(division)
    if lendiv == 1 and (d0 := division[0]) % 2 == 1 and d0 in _primes:
        return None

    if lendiv > 1 and all(subdiv == 1 for subdiv in division):
        newdiv = (lendiv,)
        if simplifyRes := simplifyDivisionWithSlots(newdiv, assignedSlots):
            simplified, newSlots = simplifyRes
            assert simplified != division
            return simplified, newSlots
        return newdiv, assignedSlots

    reduced: list[int] = []
    slots: list[int] = []
    cs, cs2 = 0, 0
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
                # elif not any(x in assigned for x in range(cs+1, cs+subdiv, 2)):
                #     reduced.append(subdiv//2)
                #     1/0
                #     # TODO
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
                for j in range(1, 5):
                    if cs+j*3 in assigned:
                        slots.append(cs2+j)
            else:
                reduced.append(subdiv)
                slots.extend(cs2+i for i in range(1, subdiv) if cs+i in assigned)
        else:
            reduced.append(subdiv)
            slots.extend(cs2+i for i in range(1, subdiv) if cs+i in assigned)
        cs += subdiv
        cs2 += reduced[-1]

    newdiv: division_t = tuple(reduced)
    assert len(newdiv) == len(division), f'{division=}, {newdiv=}'

    if all(subdiv == 1 for subdiv in newdiv):
        N = len(newdiv)
        newdiv = (N,)
        if N % 2 == 0 or N % 3 == 0:
            if simplifyRes := simplifyDivisionWithSlots(newdiv, slots):
                newdiv, slots = simplifyRes

    if newdiv == division:
        return None

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

    assert isinstance(slots, list) and len(slots) == len(assignedSlots), f"{assignedSlots=}, {slots=}, {division=} -> {newdiv=}"
    return newdiv, slots


@cache
def gridDurationsFlat(division: division_t, beatDuration: F
                      ) -> list[F]:
    numDivisions = len(division)
    subdivDur = beatDuration / numDivisions
    grid = []
    for subdiv in division:
        if isinstance(subdiv, int):
            dt = subdivDur / subdiv
            grid.extend([dt] * subdiv)
        else:
            grid.extend(gridDurationsFlat(subdiv, beatDuration=subdivDur))
    return grid


@cache
def divisionGrid0(division: division_t, beatDuration: F) -> list[F]:
    durations = gridDurationsFlat(division, beatDuration=beatDuration)
    grid = [F0]
    grid.extend(accumulate(durations))
    assert grid[-1] == beatDuration
    return grid


# @cache
# def divisionGrid0Float(division: division_t, beatDuration: F) -> tuple[list[F], list[float]]:
#     grid = divisionGrid0(division=division, beatDuration=beatDuration)
#     fgrid = [float(slot) for slot in grid]
#     return grid, fgrid


# @cache
# def divisionGrid0Array(division: division_t, beatDuration: F = F(1)
#                        ) -> tuple[list[F], NDArray[np.float64]]:
#     grid = divisionGrid0(division=division, beatDuration=beatDuration)
#     npgrid = np.array([float(slot) for slot in grid], dtype=float)
#     return grid, npgrid


@cache
def primeFactors(d: int, excludeBinary=False) -> set:
    """calculate the prime factors of d"""
    factors = set()
    for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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


def _notationsBetween(notations: list[Notation], start: F, end: F) -> list[Notation]:
    # gracenote policy: if a gracenote is at the start, we keep it,
    # at the end we don't
    # No partial notations: a notation needs to fit between start and end
    out = []
    for n in notations:
        noffset = n.offset
        assert noffset is not None
        if noffset >= end:
            break
        if start <= noffset <= end:
        # if noffset >= start and n.end <= end:
            out.append(n)
    return out


def applyDurationRatio(notations: list[Notation],
                       division: int | division_t,
                       beatOffset: F,
                       beatDuration: F
                       ) -> None:
    """
    Applies a duration ratio to each notation, in place

    A duration ratio converts the actual duration of a notation to its
    notated value and is used to render these as tuplets later

    Args:
        notations: the notations inside the period beatOffset:beatOffset+beatDur
        division: the division of the beat/subbeat.
        beatOffset: the start of the beat
        beatDuration: the duration of the beat

    """
    def _apply(durRatio: F, notations: list[Notation]):
        assert durRatio != F1
        for n in notations:
            if not n.durRatios:
                n.durRatios = (durRatio,)
            else:
                n.durRatios += (durRatio,)

    if isinstance(division, int) or len(division) == 1:
        num: int = division if isinstance(division, int) else division[0]
        durRatio = F(*quantdata.durationRatios[num])
        if durRatio != F1:
            _apply(durRatio, notations)
    else:
        numSubBeats = len(division)
        now = beatOffset
        dt = beatDuration / numSubBeats
        durRatio = F(*quantdata.durationRatios[numSubBeats])
        if durRatio != F1:
            _apply(durRatio, notations)
        if all(ispowerof2(div) for div in division):
            return

        numNotations = 0
        subdivs = []
        for i, subdiv in enumerate(division):
            subdivEnd = now + dt
            # A notation can span over multiple subdivisions?
            subdivNotations = _notationsBetween(notations, now, subdivEnd)
            # assert subdivNotations
            subdivs.append(subdivNotations)
            # Add gracenotes at the end to the last subdiv notations. They would
            # be left out, since that start at the end of the subdivision
            if i == len(division) - 1 and notations[-1].isGracenote and notations[-1].offset == subdivEnd:
                endgraces = []
                for n in reversed(notations):
                    if not n.isGracenote or (n.offset is not None and n.offset < subdivEnd):
                        break
                    endgraces.append(n)
                if endgraces:
                    subdivNotations.extend(reversed(endgraces))
            if subdivNotations:
                applyDurationRatio(notations=subdivNotations, division=subdiv,
                                   beatOffset=now, beatDuration=dt)
            now += dt
            numNotations += len(subdivNotations)

        if numNotations != len(notations):
            for i, n in enumerate(notations):
                print(i, n)
            raise RuntimeError(f"Failed to apply dur ratios, {numNotations=} != {len(notations)=}, "
                               f"{division=}, {beatOffset=}, {beatDuration=}, {durRatio=}, {notations=}, {subdivs=}")


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
        n = Notation.Rest(duration=fallbackdur, offset=offset)
        seq.append(n)
    else:
        nextnot = seq[nextidx]
        n = Notation.Rest(duration=nextnot.qoffset - offset, offset=offset)
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
        rest = Notation.Rest(duration=end - last.end, offset=last.end)
        seq.append(rest)
    else:
        if idx == 0:
            rest = Notation.Rest(duration=seq[0].qoffset, offset=0)
            seq.insert(0, rest)
        else:
            previdx = idx - 1
            last = seq[previdx]
            if last.end < end:
                rest = Notation.Rest(duration=end - last.end, offset=last.end)
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
        assert ispowerof2(den), f"Invalid duration: {dur}"
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
        out.append(Notation.Rest(duration, offset=start))
        return out

    if (n0offset := notations[0].qoffset) > start:
        out.append(Notation.Rest(n0offset - start, offset=start))
        now = n0offset

    for n0, n1 in pairwise(notations):
        n0offset = n0.qoffset
        if n0offset > now:
            # there is a gap, fill it with a rest
            out.append(Notation.Rest(offset=now, duration=n0offset - now))
        if n0.duration is None:
            out.append(n0.clone(duration=n1.qoffset - n0offset, spanners=n0.spanners))
        else:
            out.append(n0)
            n0end = n0.end
            if n0end < n1.qoffset:
                out.append(Notation.Rest(offset=n0end, duration=n1.qoffset - n0end))
        now = n1.qoffset

    # last event
    n = notations[-1]
    assert n.duration is not None
    out.append(n)
    if n.end < end:
        out.append(Notation.Rest(offset=n.end, duration=end - n.end))
    assert sum(n.duration for n in out) == duration
    assert all(n.offset is not None and start <= n.offset <= end for n in out)
    return out


@cache
def slotsAtSubdivisions(divs: tuple[int, ...]) -> list[int]:
    return [0] + list(accumulate(divs))


def divisionFitsPerfectly(div: division_t,
                          events: list[Notation],
                          beatDuration: F, offset: F
                          ) -> bool:
    """
    Checks if events fit the given div perfectly. Works for simply nested divs (4,), (3, 4, 5)

    Args:
        div: the division of the beat, as a tuple (subdiv: int, ...)
        events: the events to check
        beatDuration: duration of the beat, in quarternotes
        offset: beat offset

    Returns:
        True if the grid defined by the division fits the events perfectly

    """
    numParts = len(div)
    partDur = beatDuration / numParts  # each top-level subdivision gets equal duration
    for ev in events:
        # which part does this offset fall in?
        partIndex, remainder = divmod(ev.offset - offset, partDur)
        if remainder == 0:
            continue
        partIndex = int(partIndex)
        slotDur = partDur / div[partIndex]
        q, r = divmod(remainder, slotDur)
        if r != 0:
            return False

    return True


def simplifyUnusedSubdivs(div: division_t, events0: list[Notation]) -> division_t:
    # Exclude divisions which are not worth evaluating at full
    numSubdivs = len(div)
    beatdur = events0[-1].end
    scale_num = numSubdivs * beatdur.denominator
    scale_den = beatdur.numerator

    activeSubdivs = [0] * numSubdivs
    last_subdiv = -1
    lastidx = numSubdivs - 1

    for n in events0:
        t = n.offset
        subdiv = (t.numerator * scale_num) // (t.denominator * scale_den)
        if subdiv != last_subdiv:
            activeSubdivs[subdiv] = 1
            last_subdiv = subdiv
            if subdiv == lastidx:
                break

    if all(activeSubdivs):
        return div
    return tuple(1 if not active else s for active, s in zip(activeSubdivs, div))


class SubdivSimplifier:
    """
    Computes which subdivisions are "active" (have an event onset) for a fixed
    set of events, caching the result per subdivision count so that all divisions
    of the same length share one scan of the event list.
    """

    def __init__(self, events: list[Notation]):
        self.events = events
        self.beatdur = events[-1].end
        # numSubdivs -> (activeSubdivs bytearray, all_active bool)
        self._cache: dict[int, tuple[bytearray, bool]] = {}

    def _compute(self, numSubdivs: int) -> tuple[bytearray, bool]:
        beatdur = self.beatdur
        scale_num = numSubdivs * beatdur.denominator
        scale_den = beatdur.numerator
        active = bytearray(numSubdivs)
        last_subdiv = -1
        lastidx = numSubdivs - 1
        numActive = 0

        for n in self.events:
            t = n.offset
            subdiv = (t.numerator * scale_num) // (t.denominator * scale_den)
            if subdiv != last_subdiv:
                last_subdiv = subdiv
                if not active[subdiv]:
                    active[subdiv] = 1
                    numActive += 1
                    if numActive == numSubdivs:
                        return active, True   # all slots filled, no need to continue
                if subdiv == lastidx:
                    break

        return active, False

    def simplify(self, div: division_t) -> division_t:
        numSubdivs = len(div)
        entry = self._cache.get(numSubdivs)
        if entry is None:
            entry = self._compute(numSubdivs)
            self._cache[numSubdivs] = entry

        active, all_active = entry
        if all_active:
            return div
        return tuple(1 if not a else s for a, s in zip(active, div))


class DivisionFitChecker:
    """
    Precomputes per-numParts which events fall off part-boundaries and by how
    much, so that checking a specific division is reduced to a small arithmetic
    test per non-aligned event — with no repeated divmod over all events.
    """

    def __init__(self, events: list[Notation], beatDuration: F, offset: F):
        self.events = events
        self.beatDuration = beatDuration
        self.offset = offset
        # numParts -> list of (partIndex, remainder) for events NOT on a part boundary
        # If the list is None it means ALL events are on boundaries → always fits
        self._cache: dict[int, list[tuple[int, F]] | None] = {}

    def _compute(self, numParts: int) -> list[tuple[int, F]] | None:
        partDur = self.beatDuration / numParts
        misaligned: list[tuple[int, F]] = []
        for ev in self.events:
            partIndex, remainder = divmod(ev.offset - self.offset, partDur)
            if remainder != 0:
                misaligned.append((int(partIndex), remainder))
        return None if not misaligned else misaligned

    def fits(self, div: division_t) -> bool:
        numParts = len(div)
        if numParts not in self._cache:
            self._cache[numParts] = self._compute(numParts)

        misaligned = self._cache[numParts]
        if misaligned is None:
            return True   # every event lands on a part boundary

        partDur = self.beatDuration / numParts
        for partIndex, remainder in misaligned:
            slotDur = partDur / div[partIndex]
            if remainder % slotDur != 0:
                return False
        return True

# --------------------------------

from dataclasses import dataclass


@dataclass
class _PartResult:
    reduced_subdiv: int
    start_active: bool
    inner_reduced_slots: list[int]   # occupied inner slots in reduced-grid local coords
    event_reduced_slots: list[int]   # one per event in this part (with duplicates for grace notes)


class BeatQuantizer:
    """
    For fixed (events, beatDuration, offset), caches per-subdivision work keyed
    by (numParts, partIndex, subdiv) so that all divisions of the same length
    sharing a subdiv at the same position pay only once for both the slot
    snapping and the reduction logic.
    """

    def __init__(self, events: list[Notation], beatDuration: F, offset: F):
        self.events = events
        self.beatDuration = beatDuration
        self.offset = offset
        self._cache: dict[tuple[int, int, int], _PartResult] = {}

    # ------------------------------------------------------------------
    # per-part computation
    # ------------------------------------------------------------------

    def _compute_part(self, numParts: int, partIndex: int, subdiv: int) -> _PartResult:
        partDur = self.beatDuration / numParts
        partStart = self.offset + partIndex * partDur
        partEnd = partStart + partDur
        slotDur = partDur / subdiv
        isLast = (partIndex == numParts - 1)

        occupied: set[int] = set()
        local_slots: list[int] = []

        for ev in self.events:
            pos = ev.offset
            if pos < partStart:
                continue
            if pos > partEnd or (not isLast and pos >= partEnd):
                break
            relPos = pos - partStart
            slot_f = relPos / slotDur
            slot = int(slot_f)
            if slot < subdiv - 1 and slot_f - slot > F(1, 2):
                slot += 1
            slot = min(slot, subdiv - 1)   # clamp beat-end grace notes
            local_slots.append(slot)
            occupied.add(slot)

        start_active = 0 in occupied
        occ = occupied - {0}

        reduced_subdiv, inner_reduced_slots = self._reduce_subdiv(subdiv, occ)

        # Map original local slots -> reduced local slots via uniform factor
        if reduced_subdiv == subdiv:
            event_reduced_slots = local_slots[:]
        else:
            factor = subdiv // reduced_subdiv
            event_reduced_slots = [s // factor for s in local_slots]

        return _PartResult(reduced_subdiv, start_active, inner_reduced_slots, event_reduced_slots)

    @staticmethod
    def _reduce_subdiv(subdiv: int, occ: set[int]) -> tuple[int, list[int]]:
        """
        Given the set of occupied inner slots (1..subdiv-1), return
        (reduced_subdiv, inner_reduced_slots) in the reduced grid's local coords.
        """
        if not occ:
            return 1, []

        if subdiv % 2 == 0:
            if subdiv == 4:
                if 1 not in occ and 3 not in occ:
                    return 2, [1]              # slot 2 -> reduced 1
                return 4, sorted(occ)

            elif subdiv == 6:
                if 1 not in occ and 5 not in occ:
                    if 2 not in occ and 4 not in occ:
                        return 2, [1]          # slot 3 -> reduced 1
                    elif 3 not in occ:
                        # slots 2,4 -> reduced 1,2
                        return 3, [i for i in (1, 2) if i * 2 in occ]
                return 6, sorted(occ)

            elif subdiv == 8:
                if 1 not in occ and 3 not in occ and 5 not in occ and 7 not in occ:
                    if 2 not in occ and 6 not in occ:
                        return 2, [1]          # slot 4 -> reduced 1
                    # slots 2,4,6 -> reduced 1,2,3
                    return 4, [i for i in (1, 2, 3) if i * 2 in occ]
                return 8, sorted(occ)

            else:  # general even (10, 12, ...)
                if 1 not in occ and (subdiv - 1) not in occ:
                    mid = subdiv // 2
                    if mid in occ and not (occ - {mid}):
                        return 2, [1]
                return subdiv, sorted(occ)

        elif subdiv == 9:
            if {1, 2, 4, 5, 7, 8}.isdisjoint(occ):
                return 3, [i for i in (1, 2) if i * 3 in occ]
            return 9, sorted(occ)

        elif subdiv == 15:
            if all(x not in occ for x in range(1, 15) if x % 5 != 0):
                return 3, [i for i in (1, 2) if i * 5 in occ]
            elif all(x not in occ for x in range(1, 15) if x % 3 != 0):
                return 5, [i for i in range(1, 5) if i * 3 in occ]
            return 15, sorted(occ)

        return subdiv, sorted(occ)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def assignAndSimplify(self, division: division_t) -> tuple[division_t, list[int]]:
        """
        Assign events to slots and simplify the division in one pass,
        reusing cached per-part results across all divisions of the same length.
        Always returns (division, slots) — simplified if possible.
        """
        numSubdivs = len(division)

        # Structural early-outs that don't need slot info
        if numSubdivs == 1 and (d0 := division[0]) % 2 == 1 and d0 in _primes:
            slotDur = self.beatDuration / d0
            offset = self.offset + slotDur
            if all(ev.offset < offset for ev in self.events):
                return (1, ), [0] * len(self.events)
            return division, self._assign_only(division)

        if numSubdivs > 1 and all(s == 1 for s in division):
            slots = self._assign_only(division)
            newdiv = (numSubdivs,)
            if res := simplifyDivisionWithSlots(newdiv, slots):
                return res
            return newdiv, slots

        # Main per-part loop
        reduced: list[int] = []
        all_event_slots: list[int] = []   # one per event, in reduced-grid coords
        cs2 = 0

        for partIndex, subdiv in enumerate(division):
            key = (numSubdivs, partIndex, subdiv)
            if key not in self._cache:
                self._cache[key] = self._compute_part(numSubdivs, partIndex, subdiv)
            result = self._cache[key]

            if result.start_active:
                # slot 0 of this part is occupied; its reduced-grid position is cs2
                pass  # accounted for via event_reduced_slots (slot 0 // factor = 0)

            reduced.append(result.reduced_subdiv)
            all_event_slots.extend(cs2 + s for s in result.event_reduced_slots)
            cs2 += result.reduced_subdiv

        newdiv: division_t = tuple(reduced)
        numSlots2 = cs2

        # Boundary special cases (evaluated on reduced slots, which are equivalent)
        if len(all_event_slots) == 1 and all_event_slots[0] == 0:
            return (1,), all_event_slots

        if all(s == 0 or s == numSlots2 for s in all_event_slots):
            return (1,), [0 if s == 0 else 1 for s in all_event_slots]

        # All-ones collapse: (1,1,...,1) -> (N,) with optional further simplification
        if all(s == 1 for s in newdiv):
            N = numSubdivs
            newdiv = (N,)
            if N % 2 == 0 or N % 3 == 0:
                if res := simplifyDivisionWithSlots(newdiv, all_event_slots):
                    return res

        return newdiv, all_event_slots

    def _assign_only(self, division: division_t) -> list[int]:
        grid = divisionGrid0(division, beatDuration=self.beatDuration)
        return assignSlotsFrac(self.events, grid)