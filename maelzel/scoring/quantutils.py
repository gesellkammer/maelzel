"""
Utilities used during quantization
"""
from __future__ import division, annotations
import math
from functools import cache
from emlib import mathlib
from itertools import pairwise, accumulate

from maelzel.common import F, F0, F1
from .notation import Notation
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
    from maelzel.scoring import util
    if len(div) == 1:
        outer = div[0]
    else:
        outer = len(div)
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


def divisionDensity(division: division_t) -> int:
    return max(division) * len(division)


def asymettry(a, b) -> float:
    if a < b:
        a, b = b, a
    return float(a/b)


def resnap(assignedSlots: _t.Sequence[int], oldgrid: _t.Sequence[F], newgrid: _t.Sequence[F]
           ) -> list[int]:
    minslot = 0
    maxslot = len(newgrid)
    reassigned: list[int] = []
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
        from .common import logger
        logger.error(f'{oldoffsets=}, {newoffsets=}, {assignedSlots=}, {reassigned=}, {oldgrid=}, {newgrid=}')
        raise RuntimeError("resnap error")
    return reassigned


@cache
def _makeset(start: int, end: int, exclude):
        return set(x for x in range(start, end) if x not in exclude)


def simplifyDivision(division: division_t, assignedSlots: _t.Sequence[int], reduce=True
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
    d0 = division[0]
    if len(assignedSlots) == 1 and assignedSlots[0] == 0:
        return (1,)
    elif len(division) == 1 and d0 % 2 == 1 and d0 in (3, 5, 7, 11, 13):
        return division

    assigned = set(assignedSlots)

    cs = 0
    reduced = []
    for subdiv in division:
        if subdiv == 1:
            reduced.append(1)
        # elif all(s not in assigned for s in range(cs+1, cs+subdiv)):
        elif all(not 1 <= s-cs < subdiv for s in assigned):
            # only the first slot is assigned
            reduced.append(1)
        elif subdiv == 4:
            if cs+1 not in assigned and cs+3 not in assigned:
                reduced.append(2)
            else:
                reduced.append(4)
        elif subdiv == 6:
            #   x       x
            # x 0 0 1 0 0 -> 2
            # x 0 1 0 1 0 -> 3
            if cs+1 not in assigned and cs+5 not in assigned:
                if cs+2 not in assigned and cs+4 not in assigned:
                    reduced.append(2)
                elif cs+3 not in assigned:
                    reduced.append(3)
                else:
                    reduced.append(6)
            else:
                reduced.append(6)
        elif subdiv == 8:
            if {cs+1, cs+3, cs+5, cs+7}.isdisjoint(assigned):
                if cs+2 not in assigned and cs+6 not in assigned:
                    reduced.append(2)
                else:
                    reduced.append(4)
            else:
                reduced.append(8)
        elif subdiv == 9:
            if {cs+1, cs+2, cs+4, cs+5, cs+7, cs+8}.isdisjoint(assigned):
                reduced.append(3)
            else:
                reduced.append(9)
        elif subdiv % 2 == 1:
            reduced.append(subdiv)
        # from here on: even subdiv
        elif cs+1 not in assigned and cs+subdiv-1 not in assigned:
            # The second and last slot are not assigned
            if cs+subdiv//2 in assigned:
                if _makeset(cs+1, cs+subdiv, (cs+subdiv//2,)).isdisjoint(assigned):
                    # 1 0 0 0 0 1 0 0 0 0 -> 2
                    reduced.append(2)
                else:
                    reduced.append(subdiv)
            # elif set(range(cs+1, cs+subdiv, 2)).isdisjoint(assigned):
            elif not any(x in assigned for x in range(cs+1, cs+subdiv, 2)):
                reduced.append(subdiv//2)
            else:
                reduced.append(subdiv)
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
    assert newdiv
    return newdiv


def reduceDivision(division: division_t,
                   newdiv: division_t,
                   assignedSlots: _t.Sequence[int],
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
    grid.extend(accumulate(durations))
    return grid


@cache
def divisionGrid0Float(division: division_t, beatDuration: F = F(1)) -> tuple[list[F], list[float]]:
    grid = divisionGrid0(division=division, beatDuration=beatDuration)
    fgrid = [float(slot) for slot in grid]
    return grid, fgrid


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
                    if not n.isGracenote or n.offset < subdivEnd:
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


# def beatToTree(notations: list[Notation], division: int | division_t,
#                beatOffset: F, beatDur: F
#                ) -> _node.Node:
#     if isinstance(division, tuple) and len(division) == 1:
#         division = division[0]
#
#     if isinstance(division, int):
#         durRatio = quantdata.durationRatios[division]
#         return _node.Node(notations, ratio=durRatio)  # type: ignore
#
#     # assert isinstance(division, tuple) and len(division) >= 2
#     numSubBeats = len(division)
#     now = beatOffset
#     dt = beatDur/numSubBeats
#     durRatio = quantdata.durationRatios[numSubBeats]
#     items = []
#     for subdiv in division:
#         subdivEnd = now + dt
#         subdivNotations = [n for n in notations if now <= n.qoffset < subdivEnd and n.end <= subdivEnd]
#         if subdiv == 1:
#             items.extend(subdivNotations)
#         else:
#             items.append(beatToTree(notations=subdivNotations, division=subdiv, beatOffset=now, beatDur=dt))
#         now += dt
#     return _node.Node(items, ratio=durRatio)


def breakNotationsByBeat(notations: list[Notation],
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
    assert all (not ev.durRatios for ev in notations), f"{notations=}, {[n.durRatios for n in notations]}"
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
    return [(start, end, events) for (start, end), events in zip(timespans, eventsPerTimespan)]


def notationAtOffset(notations: list[Notation], offset: F, exact: bool) -> int | None:
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
    assert seq
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


def isRegularDuration(symdur: F) -> bool:
    """
    True if symdur is a regular duration

    Args:
        symdur: the symbolic duration of an event / node

    Returns:
        True if this duration is regular - can be notated with only one figure,
        using dots or not

    """
    return symdur.denominator in (1, 2, 4, 8, 16, 32) and symdur.numerator in (1, 2, 3, 4, 7)


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
        notations: a list of notations inside the beat
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
