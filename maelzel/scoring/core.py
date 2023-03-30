from __future__ import annotations
import itertools
import logging

from emlib import iterlib

from .common import *
from .notation import *
from . import definitions
from . import util
from maelzel import scorestruct

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Iterator
    from maelzel.scoring import quant

logger = logging.getLogger("maelzel.scoring")


__all__ = (
    'Notation',
    'Part',
    'Arrangement',
    'stackNotations',
    'stackNotationsInPlace',
    'fillSilences',
    'distributeNotationsByClef',
    'packInParts',
    'mergeNotationsIfPossible',
    'notationsCanMerge',
    'removeOverlap',
    'NotatedDuration',
)


class Part(list):
    """
    A Part is a list of unquantized non-simultaneous :class:`Notation`

    Args:
        events: the events (notes, chords) in this track
        name: a label to identify this track in particular (a name)
        groupid: an identification (given by makeGroupId), used to identify
            tracks which belong to a same tree
        shortname: a name abbreviation used as staf name after the first system

    .. seealso:: :class:`~maelzel.scoring.quant.QuantizedPart`,
    """
    def __init__(self,
                 events: list[Notation] = None,
                 name='',
                 groupid: str = '',
                 shortname='',
                 firstclef: str = '',
                 quantProfile: quant.QuantizationProfile | None = None):

        if events:
            super().__init__(events)
        else:
            super().__init__()
        self.groupid: str = groupid
        self.name: str = name
        self.shortname: str = shortname
        self.firstclef = firstclef
        self.quantProfile = quantProfile

        if events:
            assert all(isinstance(n, Notation) for n in events)
            _repairGracenoteAsTargetGliss(self)

    def __getitem__(self, item) -> Notation:
        return super().__getitem__(item)

    def __iter__(self) -> Iterator[Notation]:
        return super().__iter__()

    def __repr__(self) -> str:
        s0 = super().__repr__()
        return "Part"+s0

    def dump(self) -> None:
        """Dump this Part to stdout"""
        for n in self:
            print(n)

    def distributeByClef(self) -> list[Part]:
        """
        Distribute the notations in this Part into multiple parts, based on pitch
        """
        return distributeNotationsByClef(self, groupid=self.groupid)

    def needsMultipleClefs(self) -> bool:
        """
        True if the notations in this Part extend over the range of one clef
        """
        midinotes: list[float] = sum((n.pitches for n in self), [])
        return util.midinotesNeedMultipleClefs(midinotes)

    def stack(self) -> None:
        """
        Stack the notations of this part **in place**.

        Stacking means filling in any unresolved offset/totalDuration of the notations
        in this part. After this operation, all Notations in this Part have an
        explicit totalDuration and start. See :meth:`stacked` for a version which
        returns a new Part instead of operating in place
        """
        stackNotationsInPlace(self)

    def meanPitch(self) -> float:
        """
        The mean pitch of this part, weighted by the totalDuration of each pitch

        Returns:
            a float representing the mean pitch as midinote
        """
        pitch, dur = 0., 0.
        for n in self:
            if n.isRest:
                continue
            dur = n.duration or 1.
            pitch += n.meanPitch() * dur
            dur += dur
        return pitch / dur

    def fillGaps(self, mingap=1/64) -> None:
        """
        Fill gaps between notations in this Part, in place
        """
        if not self.hasGaps():
            return
        newevents = fillSilences(self, mingap=mingap, offset=0)
        self.clear()
        self.extend(newevents)
        assert not self.hasGaps()

    def hasGaps(self) -> bool:
        """Does this Part have gaps?"""
        assert all(n.offset is not None and n.duration is not None for n in self)
        return any(n0.end < n1.offset for n0, n1 in iterlib.pairwise(self))

    def stacked(self) -> Part:
        """
        Stack the Notations to make them adjacent if they have unset offset/totalDuration

        Similar to :meth:`stack`, **this method returns a new Part** instead of
        operating in place.
        """
        notations = stackNotations(self)
        return Part(events=notations, name=self.name, groupid=self.groupid,
                    shortname=self.shortname, firstclef=self.firstclef)

    # def groupNotationsByMeasure(self, struct: scorestruct.ScoreStruct
    #                             ) -> list[list[Notation]]:
    #     currMeasure = -1
    #     groups = []
    #     for n in self:
    #         assert n.offset is not None and n.duration is not None
    #         loc = struct.beatToLocation(n.offset)
    #         if loc is None:
    #             logger.error(f"Offset {n.offset} outside of scorestruct, for {n}")
    #             logger.error(f"Scorestruct: totalDuration = {struct.totalDurationBeats()} quarters\n{struct.dump()}")
    #             raise ValueError(f"Offset {float(n.offset):.3f} outside of score structure "
    #                              f"(max. offset: {float(struct.totalDurationBeats()):.3f})")
    #         elif loc[0] == currMeasure:
    #             groups[-1].append(n)
    #         else:
    #             # new measure
    #             currMeasure = loc[0]
    #             groups.append([n])
    #     return groups
    #

class Arrangement(list):
    """
    An Arrangement is a list of Parts
    """
    def __init__(self, parts: list[Part] = None, title: str = ''):
        if parts:
            super().__init__(parts)
        else:
            super().__init__()
        self.title = title


def _repairGracenoteAsTargetGliss(notations: list[Notation]) -> None:
    """
    Removes superfluous end glissandi notes **in place**

    To be called after notations are "stacked". Removes superfluous
    end glissandi notes when the endgliss is the same as the next note
    """
    toBeRemoved = []
    skip = False
    for n0, n1, n2 in iterlib.window(notations, 3):
        if skip:
            skip = False
            continue
        if n0.gliss and n1.isGracenote and n1.pitches == n2.pitches:
            # check if the gracenote is empty
            if not n1.hasAttributes():
                toBeRemoved.append(n1)
                skip = True
    for item in toBeRemoved:
        notations.remove(item)
    stackNotationsInPlace(notations)


def stackNotationsInPlace(events: list[Notation], start=F(0), overrideOffset=False
                          ) -> None:
    """
    Stacks notations to the left, in place

    Stacks events together by placing an event at the end of the
    previous event whenever an event does not define its own offset

    Args:
        events: a list of Notations (or a Part)
        start: the start time, will override the offset of the first event
        overrideOffset: if True, offsets are overriden even if they are defined
    """
    if all(ev.offset is not None and ev.duration is not None
           for ev in events):
        return
    now = _ if (_:=events[0].offset) is not None else start if start is not None else F(0)
    assert now is not None and now >= 0
    lasti = len(events)-1
    for i, ev in enumerate(events):
        if ev.offset is None or overrideOffset:
            assert ev.duration is not None
            ev.offset = now
        elif ev.duration is None:
            if i == lasti:
                raise ValueError("The last event should have a totalDuration")
            ev.duration = events[i+1].offset - ev.offset
        now += ev.duration
    assert all(ev1.offset <= ev2.offset for ev1, ev2 in iterlib.pairwise(events))
    removeOverlap(events)


def stackNotations(events: list[Notation], start=F(0), overrideOffset=False
                   ) -> list[Notation]:
    """
    Stacks Notations to the left, returns the new notations

    This function stacks events together by placing an event at the end of the
    previous event whenever an event does not define its own offset, or sets
    the totalDuration of an event if events are specified via offset alone

    Args:
        events: a list of notations
        start: the start time, will override the offset of the first event
        overrideOffset: if True, offsets are overriden even if they are defined

    Returns:
        a list of stacked events
    """
    events = [ev.copy() for ev in events]
    stackNotationsInPlace(events, start=start, overrideOffset=overrideOffset)
    return events


def removeOverlap(notations: list[Notation], maxgap=F(1, 10000)) -> None:
    """
    Fix overlap between notations, in place.

    If two notations overlap, the first notation is cut, preserving the
    offset of the second notation. If there is a gap smaller than a given
    threshold the end of the note left of the gap is extended to match
    the offset of the note right of the gap

    Args:
        notations: the notations to fix
        maxgap: max. gap to allow between notations. Any gap smaller than
            this will be removed, growing the previous notation to fill
            the gap (the start times are left unmodified)

    Returns:
        the fixed notations
    """
    if len(notations) < 2:
        return
    for n0, n1 in iterlib.pairwise(notations):
        assert n0.duration is not None and n0.offset is not None
        assert n1.offset is not None
        assert n0.offset <= n1.offset, "Notes are not sorted!"
        if n0.end > n1.offset or n1.offset - n0.end < maxgap:
            n0.duration = n1.offset - n0.offset
            assert n0.end == n1.offset


def removeOverlapInplace(notations: list[Notation], threshold=F(1,1000)) -> None:
    """
    Remove overlap between notations.

    This should be only used to remove small overlaps product of rounding errors.
    """
    removed = []
    for n0, n1 in iterlib.pairwise(notations):
        assert n0.offset <= n1.offset, "Notes are not sorted!"
        diff = n0.end - n1.offset
        if diff > 0:
            if diff > threshold:
                raise ValueError(f"Notes overlap by too much: {diff}, {n0}, {n1}")
            duration = n1.offset - n0.offset
            if duration <= 0:
                removed.append(n0)
            else:
                n0.duration = duration
    for n in removed:
        notations.remove(n)


def fillSilences(notations: list[Notation], mingap=1/64, offset: time_t = None
                 ) -> list[Notation]:
    """
    Return a list of Notations filled with rests

    Args:
        notations: the notes to fill
        mingap: min. gap between two notes. If any notes differ by less
                   than this, the first note absorvs the gap
        offset: if given, marks the start time to fill. If notations start after
            this offset a rest will be crated from this offset to the start
            of the first notation
    Returns:
        a list of new Notations
    """
    assert notations
    assert all(isinstance(n, Notation) and n.offset is not None and n.duration is not None
               for n in notations)
    if offset is not None:
        assert all(n.offset >= offset for n in notations
                   if n.offset is not None)

    out: list[Notation] = []
    n0 = notations[0]
    if offset is not None and n0.offset is not None and n0.offset > offset:
        out.append(makeRest(duration=n0.offset, offset=offset))
    for ev0, ev1 in iterlib.pairwise(notations):
        assert isinstance(ev0.offset, F) and isinstance(ev0.duration, F)
        gap = ev1.offset - (ev0.offset + ev0.duration)
        if gap < 0:
            if abs(gap) < 1e-14 and ev0.duration > 1e-13:
                out.append(ev0.clone(duration=ev1.offset - ev0.offset))
            else:
                raise ValueError(f"negative gap! = {gap}")
        elif gap > mingap:
            out.append(ev0)
            rest = makeRest(duration=gap, offset=ev0.offset+ev0.duration)
            assert rest.offset is not None and rest.duration is not None
            out.append(rest)
        else:
            # adjust the dur of n0 to match start of n1
            out.append(ev0.clone(duration=ev1.offset - ev0.offset))
    out.append(notations[-1])
    for n0, n1 in iterlib.pairwise(out):
        assert n0.end == n1.offset, f'{n0=}, {n1=}'
    return out


def _groupById(notations: list[Notation]) -> list[Union[Notation, list[Notation]]]:
    """
    Given a seq. of events, elements which are grouped together are wrapped
    in a list, whereas elements which don't belong to any tree are
    appended as is

    """
    out: list[Union[Notation, list[Notation]]] = []
    for groupid, elementsiter in itertools.groupby(notations, key=lambda n: n.groupid):
        if not groupid:
            out.extend(elementsiter)
        else:
            elements = list(elementsiter)
            elements.sort(key=lambda elem: elem.offset or 0)
            out.append(elements)
    return out


def _pitchToClef(pitch: float, splitPoint=60) -> str:
    if splitPoint <= pitch <= 93:
        return 'g'
    elif 93 < pitch:
        return '15a'
    else:
        return 'f'


def distributeNotationsByClef(notations: list[Notation],
                              filterRests=False,
                              groupid: str = ''
                              ) -> list[Part]:
    """
    Split the notations into parts

    Assuming that events are not simultanous, split the events into
    different Parts if the range makes it necessary, where each
    Part can be represented without clef changes. We don't enforce that the
    notations are not simultaneous within a part

    Args:
        notations: the events to split
        filterRests: if True, rests are skipped
        groupid: if True, mark the created Parts with the same groupid. When rendering
            it is possible to show parts with the same id as belonging to one staff
            tree. To create a groupid see

    Returns:
         list of Parts (between 1 and 3, one for each clef)
    """
    # The order is important
    parts = {'15a': [], 'g': [], 'f': []}
    pitchedNotations = (n for n in notations if not n.isRest)
    lowPitch = sum(60 - p for n in pitchedNotations for p in n.pitches if p <= 60)
    splitPoint = 60 if lowPitch > 8 else 56
    lastj = len(notations) - 1
    for j, notation in enumerate(notations):
        assert notation.offset is not None
        if notation.isRest:
            if not filterRests:
                parts['g'].append(notation)
        elif len(notation.pitches) == 1:
            clef = _pitchToClef(notation.pitches[0], splitPoint)
            parts[clef].append(notation)
        else:
            # a chord
            indexesPerClef = {'g': [], 'f': [], '15a': []}
            pitchindexToClef = [notation.getClefHint(i) or _pitchToClef(pitch, splitPoint)
                                for i, pitch in enumerate(notation.pitches)]
            if set(pitchindexToClef) == 1:
                parts[pitchindexToClef[0]].append(notation)
            else:
                for i, clef in enumerate(pitchindexToClef):
                    indexesPerClef[clef].append(i)
                    if j < lastj and notation.gliss:
                        notations[j+1].setClefHint(clef, i)
                for clef, indexes in indexesPerClef.items():
                    if not indexes:
                        continue
                    partialChord = notation.extractPartialNotation(indexes)
                    parts[clef].append(partialChord)

    parts = [Part(part) for part in parts.values() if part]
    if groupid:
        for p in parts:
            p.groupid = groupid
    return parts


def packInParts(notations: list[Notation], maxrange=36,
                keepGroupsTogether=True) -> list[Part]:
    """
    Pack a list of possibly simultaneous notations into tracks

    The notations within one track are NOT simulatenous. Notations belonging
    to the same tree are kept in the same track.

    Args:
        notations: the Notations to pack
        maxrange: the max. distance between the highest and lowest Notation
        keepGroupsTogether: if True, items belonging to a same tree are
            kept in a same track

    Returns:
        a list of Parts

    """
    from maelzel import packing
    items = []
    groups = _groupById(notations)
    for group in groups:
        if isinstance(group, Notation):
            n = group
            if n.isRest and not n.attachments and not n.dynamic:
                continue
            assert n.offset is not None and n.duration is not None
            items.append(packing.Item(obj=n, offset=n.offset,
                                      dur=n.duration, step=n.meanPitch()))
        else:
            assert isinstance(group, list)
            if keepGroupsTogether:
                dur = (max(n.end for n in group if n.end is not None) -
                       min(n.offset for n in group if n.offset is not None))
                step = sum(n.meanPitch() for n in group)/len(group)
                item = packing.Item(obj=group, offset=group[0].offset or 0, dur=dur, step=step)
                items.append(item)
            else:
                items.extend(packing.Item(obj=n, offset=n.offset or 0, dur=n.duration or 1,
                                          step=n.meanPitch())
                             for n in group)

    packedTracks = packing.packInTracks(items, maxAmbitus=maxrange)
    return [Part(track.unwrap()) for track in packedTracks]


def removeRedundantDynamics(notations: list[Notation],
                            resetAfterRest=True,
                            minRestDuration: time_t = F(1, 16)) -> None:
    """
    Removes redundant dynamics, in place

    A dynamic is redundant if it is the same as the last dynamic and
    it is a dynamic level (ff, mf, ppp, but not sf, sfz, etc). It is
    possible to force a dynamic by adding a ``!`` sign to the dynamic
    (pp!)

    Args:
        notations: the notations to remove redundant dynamics from
        resetAfterRest: if True, any dynamic after a rest is not considered
            redundant
        minRestDuration: the min. totalDuration of a rest to reset dynamic, in quarternotes
    """
    lastDynamic = ''
    for n in notations:
        if n.tiedPrev:
            continue
        if n.isRest and not n.dynamic:
            if resetAfterRest and n.duration > minRestDuration:
                lastDynamic = ''
        elif n.dynamic and n.dynamic in definitions.dynamicLevels:
            if n.dynamic[-1] == '!':
                lastDynamic = n.dynamic[:-1]
            elif n.dynamic == lastDynamic:
                n.dynamic = ''
            else:
                lastDynamic = n.dynamic