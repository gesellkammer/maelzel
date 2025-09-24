from __future__ import annotations
import uuid
import itertools
from typing import TYPE_CHECKING
from itertools import pairwise

from emlib import iterlib

from maelzel._util import reprObj
from maelzel.common import F, F0, F1

from . import util
from .common import NotatedDuration, logger
from .notation import Notation

if TYPE_CHECKING:
    from typing import Iterator
    from maelzel.scoring import quant
    from maelzel.common import time_t
    from . import attachment


__all__ = (
    'Notation',
    'UnquantizedPart',
    'UnquantizedScore',
    'NotatedDuration',
    'fillSilences',
    'resolveOffsets',
    'packInParts',
    'removeSmallOverlaps',
    'distributeNotationsByClef',
)


def _parseGroupname(name: str, separator="::") -> tuple[str, str]:
    parts = name.split(separator, maxsplit=1)
    return (parts[0], '') if len(parts) == 1 else (parts[0], parts[1])


class UnquantizedPart:
    """
    An UnquantizedPart is a list of unquantized non-simultaneous :class:`Notation`

    .. seealso:: :class:`~maelzel.scoring.quant.QuantizedPart`,
    """
    def __init__(self,
                 notations: list[Notation],
                 name='',
                 shortname='',
                 groupid: str = '',
                 groupname='',
                 showName=True,
                 quantProfile: quant.QuantizationProfile | None = None,
                 firstClef='',
                 possibleClefs: tuple[str, ...] = (),
                 resolve=True
                 ):
        """

        Args:
            notations: the notations in this part
            name: the name of the part
            shortname: an abbreviated name for this part
            groupid: parts with the same groupid will be grouped
            groupname: the name of the group, if necessary
            showName: show/hide the name of this part
            quantProfile: a profile can be attached for later quantization
            resolve: resolve all missing offsets explicitely
        """
        self.notations: list[Notation] = notations

        self.groupid: str = groupid
        """A UUID identifying this Part (can be left unset)"""

        self.groupName: tuple[str, str] | None = _parseGroupname(groupname) if groupname else None
        """Used as staff group name for parts grouped together. It can include a shortname as
        <name>::<shortname>"""

        self.name: str = name
        """The name of the part"""

        self.shortName: str = shortname
        """A shortname to use as abbreviation"""

        self.quantProfile = quantProfile
        """A quantization profile can be attached for later quantization"""

        self.showName = showName
        """If True, show the part name when rendered"""

        self.hooks: list[attachment.Hook] = []
        """Callbacks to be triggered at different stages"""

        self.attachments: list[attachment.Attachment] = []
        """A list of Attachments for the part itself"""

        self.firstClef: str = firstClef
        """Initial clef for this part"""

        self.possibleClefs = possibleClefs
        """Clefs to choose from for automatic clef changes during quantization"""

        self.check()

        if resolve:
            resolveOffsets(self.notations)

        self._repairNotations()

    def _repairNotations(self):
        wasmodified = _repairGracenoteAsTargetGliss(self.notations)
        if wasmodified:
            resolvedOffsets(self.notations)

    def check(self) -> None:
        """
        Check that this part is valid

        Raises ValueError if an error is found
        """
        if not all(isinstance(n, Notation) for n in self.notations):
            raise TypeError(f"Expected a list of Notations, got {self.notations}")

        overlap = self.findOverlap()
        if overlap is not None:
            overlaptime, idx = overlap
            ev1 = self.notations[idx]
            ev2 = self.notations[idx+1]
            raise ValueError(f"Found overlap of {overlap} between {ev1} and {ev2}")
        return

    def findOverlap(self) -> tuple[F, int] | None:
        """
        Find the next overlap within the items of this part

        Returns:
            a tuple (overlapamount: F, eventindex: int), or None if no overlap
        """
        idx = 0
        for (n1, offset1), (n2, offset2) in pairwise(self.iterWithOffset()):
            overlap = offset2 - (offset1 + n1.duration)
            if overlap < 0:
                return (overlap, idx)
            idx += 1
        return None

    def setGroup(self, groupid: str, name='', shortname='', showPartName=False) -> None:
        """
        Set group attributes for this part

        Args:
            groupid: the groupid this part belongs to. All parts with the same groupid
                are grouped together
            name: name of the group
            shortname: abbreviation for the group name
            showPartName: if True, show the name of each part, even if they are
                within a group
        """
        self.groupid = groupid
        self.groupName = (name, shortname)
        self.showName = showPartName

    def __getitem__(self, item) -> Notation:
        return self.notations.__getitem__(item)

    def __iter__(self) -> Iterator[Notation]:
        return iter(self.notations)

    def __repr__(self) -> str:
        return reprObj(self, priorityargs=('notations',), hideFalsy=True)

    def dump(self, indents=0, file=None) -> None:
        """Dump this to stdout"""
        indentstr = "  " * indents
        for n in self.notations:
            print(indentstr, n, file=file)

    @staticmethod
    def groupParts(parts: list[UnquantizedPart],
                   name='',
                   shortname='',
                   groupid='',
                   showPartNames=False
                   ) -> str:
        """
        Mark the given parts as belonging to one group

        Args:
            parts: the parts to group
            name: a name for the group
            shortname: short name used for all systems after the first
            groupid: an explicit group id to use. If not given, a group id is created
            showPartNames: if True, the names for each part are shown (if present).
                Otherwise part names are hidden and only the group name is shown

        Returns:
            the group id assigned to the parts

        """
        if not groupid:
            groupid = makeGroupId()
        for part in parts:
            part.setGroup(groupid=groupid, name=name, shortname=shortname, showPartName=showPartNames)
        return groupid

    def distributeByClef(self) -> list[UnquantizedPart]:
        """
        Distribute the notations in this Part into multiple parts, based on pitch
        """
        return distributeNotationsByClef(self.notations, groupid=self.groupid)

    def needsMultipleClefs(self) -> bool:
        """
        True if the notations in this Part extend over the range of one clef
        """
        midinotes: list[float] = sum((n.pitches for n in self), [])  # type: ignore
        return util.midinotesNeedMultipleClefs(midinotes)

    def stack(self) -> None:
        """
        Stack the notations of this part **inplace**.

        Stacking means filling in any unresolved offset in this part.
        After this operation, all Notations in this UnquantizedPart have an
        explicit offset.

        .. seealso:: :meth:`UnquantizedPart.iterWithOffset`

        """
        resolveOffsets(self.notations)

    def iterWithOffset(self) -> Iterator[tuple[Notation, F]]:
        """
        Iterate over the notations in this part with their resolved offsets

        The notations are not modified

        Returns:
            an iterator of tuple(notation: Notation, offset: F)

        .. seealso:: :meth:`UnquantizedPart.stack`
        """
        return resolvedOffsets(self.notations)

    def meanPitch(self) -> float:
        """
        The mean pitch of this part, weighted by the duration of each pitch

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

    def fillGaps(self, mingap=F(1, 64)) -> None:
        """
        Fill gaps between notations in this Part, inplace
        """
        if not self.hasGaps():
            return
        self.notations = fillSilences(self.notations, mingap=mingap, offset=F0)
        assert not self.hasGaps()

    def hasGaps(self) -> bool:
        """Does this Part have gaps?"""
        now = F(0)
        for n, offset in self.iterWithOffset():
            if offset > now:
                return True
            now = offset + n.duration
        return False


class UnquantizedScore:
    """
    An UnquantizedScore is a list of UnquantizedParts
    """
    def __init__(self, parts: list[UnquantizedPart], title: str = ''):
        self.parts = parts
        self.title = title

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, item) -> UnquantizedPart:
        return self.parts.__getitem__(item)

    def __iter__(self) -> Iterator[UnquantizedPart]:
        return iter(self.parts)

    def append(self, part: UnquantizedPart):
        self.parts.append(part)

    def dump(self, indents=0, file=None):
        for i, part in enumerate(self.parts):
            print(f"Part #{i}")
            part.dump(indents=1, file=file)


def _repairGracenoteAsTargetGliss(notations: list[Notation]) -> bool:
    """
    Removes superfluous end glissandi notes **inplace**

    To be called after notations are "stacked". Removes superfluous
    end glissandi notes when the endgliss is the same as the next note

    Returns:
        True if notations was modified
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
    return len(toBeRemoved) > 0


def resolvedOffsets(notations: list[Notation]
                    ) -> Iterator[tuple[Notation, F]]:
    """
    Iterate over notations rendering each notation together with its resolved offset

    Notations are not modified

    Args:
        notations: the notations to iterate over

    Returns:
        an iterator of tuple(notation, offset)

    """
    now = notations[0].offset
    if now is None:
        now = F(0)
    for i, n in enumerate(notations):
        if n.offset is not None:
            if n.offset < now:
                raise ValueError(f"Notations not sorted, {n} starts before "
                                 f"the end of the previous event, {notations[i-1]}. "
                                 f"Notations: {notations}")
            now = n.offset
        yield n, now
        now += n.duration


def resolveOffsets(notations: list[Notation], start=F(0), overrideOffset=False
                   ) -> None:
    """
    Fills all offsets, in place

    Notations with an unset offset are stacked to the end of the previous notation

    Args:
        notations: a list of Notations (or a Part)
        start: the start time, will override the offset of the first event
        overrideOffset: if True, offsets are overriden even if they are defined
    """
    if all(ev.offset is not None and ev.duration is not None
           for ev in notations):
        return
    now = _ if (_ := notations[0].offset) is not None else start
    for i, n in enumerate(notations):
        assert n.duration is not None
        if n.offset is None or overrideOffset:
            n.offset = now
        now += n.duration
    for n1, n2 in pairwise(notations):
        if n1.end > n2.qoffset:
            raise ValueError(f"Notations are not sorted: {n1}, {n2}")
    removeSmallOverlaps(notations)


def removeSmallOverlaps(notations: list[Notation], threshold=F(1, 1000)) -> None:
    """
    Remove overlap between notations.

    This should be only used to remove small overlaps product of rounding errors.
    """
    if len(notations) < 2:
        return
    for n0, n1 in pairwise(notations):
        n1offset = n1.qoffset
        n0offset = n0.qoffset
        diff = n1offset - n0.end
        if diff > 0 and diff < threshold:
            # small gap between notations
            n0.duration = n1offset - n0offset
        elif diff < 0:
            # overlap
            if abs(diff) > threshold:
                raise ValueError(f"Notes overlap by too much: {diff=}, {n0=}, {n1=}")
            duration = n1offset - n0offset
            if duration < 0:
                raise ValueError(f"Notations are not sorted: {n0=}, {n1=}")
            n0.duration = duration


def fillSilences(notations: list[Notation],
                 mingap=F(1, 64),
                 offset=F0
                 ) -> list[Notation]:
    """
    Return a list of Notations filled with rests

    Args:
        notations: the notes to fill, should have offset set
        mingap: min. gap between two notes. If any notes differ by less
                   than this, the first note absorvs the gap (becomes longer or shorter)
        offset: if given, marks the start time to fill. If notations start after
            this offset a rest will be crated from this offset to the start
            of the first notation

    Returns:
        a list of new Notations
    """
    assert notations and all(n.offset is not None and n.offset >= offset for n in notations)

    out: list[Notation] = []
    n0 = notations[0]
    if n0.offset is not None and n0.offset > offset:
        out.append(Notation.makeRest(duration=n0.offset, offset=offset))
    for ev0, ev1 in pairwise(notations):
        gap = ev1.qoffset - (ev0.qoffset + ev0.duration)
        if gap == 0:
            out.append(ev0)
        elif gap > mingap:
            out.append(ev0)
            rest = Notation.makeRest(duration=gap, offset=ev0.qoffset+ev0.duration)
            out.append(rest)
        elif gap < 0:
            if abs(gap) < 1e-14 and ev0.duration > 1e-13:
                n = ev0.clone(duration=ev1.qoffset - ev0.qoffset)
                logger.debug("Small negative gap", n)
                out.append(n)
            else:
                raise ValueError(f"Items overlap, {gap=}, {ev0=}, {ev1=}")
        else:
            # gap <= mingap: adjust the dur of n0 to match start of n1
            logger.debug(f"Small gap ({gap}), absorve the gap in the first note")
            out.append(ev0.clone(duration=ev1.qoffset - ev0.qoffset))

    out.append(notations[-1])
    assert all(n0.end == n1.offset for n0, n1 in pairwise(out)), f"failed to fill gaps: {out}"
    return out


def _groupById(notations: list[Notation]) -> list[Notation | list[Notation]]:
    """
    Given a seq. of events, elements which are grouped together are wrapped
    in a list, whereas elements which don't belong to any group are
    appended as is

    """
    out: list[Notation | list[Notation]] = []
    for groupid, elementsiter in itertools.groupby(notations, key=lambda n: n.groupid):
        if not groupid:
            out.extend(elementsiter)
        else:
            elements = list(elementsiter)
            elements.sort(key=lambda elem: elem.offset or 0)
            out.append(elements)
    return out


def distributeNotationsByClef(notations: list[Notation],
                              maxstaves=3,
                              groupid: str = '',
                              name='',
                              shortname='',
                              ) -> list[UnquantizedPart]:
    """
    Distribute the given notations amongst parts with different clefs

    Args:
        notations: the notations to distribute
        maxstaves: max. number of staves
        groupid: a groupid to use for all created parts
        name: a name to use for the resulting group
        shortname: an abbreviation for the name of the group

    Returns:
        a list of UnquantizedParts, sorted from low to high
    """
    from . import clefutils
    partpairs = clefutils.explodeNotations(notations, maxStaves=maxstaves)
    # parts are sorted from low to high
    parts = [UnquantizedPart(notations, firstClef=clef) for clef, notations in partpairs]
    if len(parts) > 1:
        # lowest part
        parts[0].possibleClefs = clefutils.clefsBetween(maxclef=parts[1].firstClef)
        # highest part
        parts[-1].possibleClefs = clefutils.clefsBetween(minclef=parts[-2].firstClef)

    if len(parts) > 2:
        for i in range(len(parts)-2):
            parts[i+1].possibleClefs = clefutils.clefsBetween(minclef=parts[i].firstClef, maxclef=parts[i+2].firstClef)

    if groupid:
        for p in parts:
            p.groupid = groupid

    if name:
        if len(parts) == 1:
            parts[0].name = name
            parts[0].shortName = shortname
        else:
            for i, part in enumerate(parts):
                part.name = f'{name}-{i + 1}'
                if shortname:
                    part.shortName = f'{shortname}{i + 1}'

    return parts


def packInParts(notations: list[Notation],
                maxrange=36,
                keepGroupsTogether=True,
                ) -> list[UnquantizedPart]:
    """
    Pack a list of possibly simultaneous notations into tracks

    The notations within one track are NOT simultaneous. Notations belonging
    to the same group are kept in the same track.

    Args:
        notations: the Notations to pack
        maxrange: the max. distance between the highest and lowest Notation
        keepGroupsTogether: if True, items belonging to a same group are
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
            items.append(packing.Item(obj=n, offset=float(n.offset),
                                      dur=float(n.duration), step=n.meanPitch()))
        else:
            assert isinstance(group, list)
            if keepGroupsTogether:
                dur = (max(n.end for n in group if n.end is not None) -
                       min(n.offset for n in group if n.offset is not None))
                step = sum(n.meanPitch() for n in group)/len(group)
                offset = group[0].offset
                item = packing.Item(obj=group, offset=float(offset) if offset else 0., dur=float(dur), step=step)
                items.append(item)
            else:
                items.extend(packing.Item(obj=n, offset=float(n.offset) if n.offset else 0., dur=float(n.duration),
                                          step=n.meanPitch())
                             for n in group)

    packedTracks = packing.packInTracks(items, maxrange=maxrange)
    assert packedTracks is not None
    return [UnquantizedPart(track.unwrap()) for track in packedTracks]


def removeRedundantDynamics(notations: list[Notation],
                            resetAfterRest=True,
                            minRestDuration: time_t = F(1, 16),
                            resetAfterQuarters=0) -> None:
    """
    Removes redundant dynamics, inplace

    A dynamic is redundant if it is the same as the last dynamic and
    it is a dynamic level (ff, mf, ppp, but not sf, sfz, etc). It is
    possible to force a dynamic by adding a ``!`` sign to the dynamic
    (pp!)

    Args:
        notations: the notations to remove redundant dynamics from
        resetAfterRest: if True, any dynamic after a rest is not considered
            redundant
        minRestDuration: the min. duration of a rest to reset dynamic, in quarternotes
        resetAfterQuarters: if given, an explicit dynamic after this number of quarters
            will not be removed
    """
    lastDynamic = ''
    now = F(0)
    lastDynamicBeat = F(0)
    for n in notations:
        if resetAfterQuarters and now - lastDynamicBeat > resetAfterQuarters:
            lastDynamic = ''
        now += n.duration
        if n.tiedPrev:
            continue
        if n.isRest and not n.dynamic:
            if resetAfterRest and n.duration > minRestDuration:
                lastDynamic = ''
        elif n.dynamic:
            if n.dynamic[-1] == "!":
                lastDynamic = n.dynamic[:-1]
                lastDynamicBeat = now
            elif n.dynamic == lastDynamic:
                n.dynamic = ''
            else:
                lastDynamic = n.dynamic
                lastDynamicBeat = now


def makeGroupId(parent: str = '') -> str:
    """
    Create an id to group notations together

    Args:
        parent: if given it will be prepended as {parent}/{groupid}

    Returns:
        the group id as string
    """
    groupid = str(uuid.uuid1())
    return groupid if parent is None else f'{parent}/{groupid}'
