from __future__ import annotations
import uuid
import itertools
from typing import TYPE_CHECKING
from itertools import pairwise

from emlib import iterlib

from maelzel._util import reprObj
from maelzel.common import F, F0

from . import util
from .common import logger
from .notation import Notation
from dataclasses import dataclass

if TYPE_CHECKING:
    from typing import Iterator, Sequence
    from maelzel.scoring import quant
    from . import attachment
    from maelzel.scorestruct import ScoreStruct



__all__ = (
    'Notation',
    'UnquantizedPart',
    'UnquantizedScore',
    'fillSilences',
    'resolveOffsets',
    'distributeByClef',
    'removeSmallOverlaps',
)


@dataclass
class GroupDef:
    id: str
    name: str = ''
    abbrev: str = ''
    kind: str = 'group'

    def rank(self) -> int:
        return 0 if self.kind == '' else 1 if self.kind == 'group' else 2


class UnquantizedPart:
    """
    An UnquantizedPart is a list of unquantized non-simultaneous :class:`Notation`

    .. seealso:: :class:`~maelzel.scoring.quant.QuantizedPart`,
    """
    _groupRegistry: dict[str, GroupDef] = {}

    def __init__(self,
                 notations: list[Notation],
                 name='',
                 abbrev='',
                 showName=True,
                 quantProfile: quant.QuantizationProfile | None = None,
                 firstClef='',
                 possibleClefs: tuple[str, ...] = (),
                 resolve=True,
                 scorestruct: ScoreStruct | None = None,
                 ):
        """

        Args:
            notations: the notations in this part
            name: the name of the part
            abbrev: an abbreviated name for this part
            groupid: parts with the same groupid will be grouped
            groupName: the name of the group, if necessary
            showName: show/hide the name of this part
            quantProfile: a profile can be attached for later quantization
            resolve: resolve all missing offsets explicitely
        """
        self.notations: list[Notation] = notations

        self.groups: list[str] = []
        """Ids of the groups/parts to which this part belongs"""
        # Nesting of groups is determined by the fact that a group must fully contain
        # another group, meaning that for groups to share a part one of the groups
        # must be a subgroup of the other

        self.name: str = name
        """The name of the part"""

        self.abbrev: str = abbrev
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

        self.scorestruct = scorestruct
        """The scorestruct for this part, or None if not specified"""

        self.check()

        if resolve:
            resolveOffsets(self.notations)

        self._repairNotations()

    @property
    def partid(self) -> str:
        for id in self.groups:
            group = self._groupRegistry.get(id)
            if group is not None and group.kind == 'part':
                return id
        return ''

    def _repairNotations(self):
        wasmodified = _repairGracenoteAsTargetGliss(self.notations)
        if wasmodified:
            resolveOffsets(self.notations)
            
    def notationAfter(self, offset: F, end: F = F0) -> Notation | None:
        """
        The first notation at or after offset
        
        Args:
            offset: the time offset to start searching 
            end: if given, notation should start before end

        Returns:

        """
        if offset == F0:
            return self.notations[0]
        
        for n, noffset in self.iterWithOffset():
            if noffset >= offset and (end == 0 or noffset < end):
                return n
        return None
    
    def notationsBetween(self, start: F, end: F = F0, strict=True) -> list[Notation]:
        out: list[Notation] = []
        if strict:
            for n, offset in self.iterWithOffset():
                if end and offset >= end:
                    break
                if start <= offset and (end == F0 or n.duration + offset <= end):
                    out.append(n)
        else:
            for n, offset in self.iterWithOffset():
                if end and offset >= end:
                    break
                if start < n.duration + offset and (end == F0 or offset < end):
                    out.append(n)
        return out
                    
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

    def setGroup(self, groupid='', name='', abbrev='') -> str:
        """
        Set group attributes for this part.

        This adds this part to a new/existing group. A part can be part of
        multiple groups as long as a group is a subgroup of the other. If this
        part is already marked as belonging to the given group, nothing happends

        Args:
            groupid: the groupid this part belongs to. All parts with the same groupid
                are grouped together
            name: name of the group
            abbrev: abbreviation for the group name
        """
        if groupid and groupid in self.groups:
            return groupid
        groupid = self._registerGroup(id=groupid, name=name, abbrev=abbrev)
        assert groupid
        self.groups.append(groupid)
        return groupid

    @property
    def groupName(self) -> tuple[str, str]:
        if not self.groups:
            raise ValueError("This part does not belong to a group")
        elif len(self.groups) > 1:
            raise ValueError("This part belongs to multiple groups")
        group = self._groupRegistry.get(self.groups[0])
        if not group:
            raise ValueError(f"Group {self.groupid} not found in groupRegistry")
        return group.name, group.abbrev

    @classmethod
    def _registerGroup(cls, name='', abbrev='', id='', kind='group') -> str:
        if id and id in cls._groupRegistry:
            return id
        elif not id:
            id = makeGroupId()
        group = GroupDef(id=id, name=name, abbrev=abbrev, kind=kind)
        cls._groupRegistry[id] = group
        return id

    @classmethod
    def makeMultivoicePart(cls,
                           voices: Sequence[UnquantizedPart],
                           name='',
                           abbrev='',
                           id=''
                           ) -> str:
        """
        Mark the given parts as being voices of a multivoice part

        Args:
            voices: the voices to mark as belonging to a part (2 to 4)
            name: the name of the part
            abbrev: the abbreviation of the name
            id: an id, if known. If not given a new id is created.

        Returns:
            the part id

        """
        assert 2 <= len(voices) <= 4, f"Invalid number of voices: {voices}"
        if not id or id not in cls._groupRegistry:
            id = cls._registerGroup(name=name, abbrev=abbrev, kind='part', id=id)
        assert id
        for voice in voices:
            voice.groups.append(id)
        return id

    def asVoice(self, id='', name='', abbrev='') -> str:
        """
        Mark this part as belonging to a multivoice part

        Args:
            id: the id of the part. This should be shared between all voices
            name: the name of the part.
            abbrev: an optional abbreviation for the name

        Returns:
            the part id (the same as given or a new one if not set)
        """
        if not id or id not in self._groupRegistry:
            id = self._registerGroup(name=name, abbrev=abbrev, kind='part', id=id)
        self.groups.append(id)
        return id

    def __getitem__(self, item) -> Notation:
        return self.notations.__getitem__(item)

    def __iter__(self) -> Iterator[Notation]:
        return iter(self.notations)

    def __repr__(self) -> str:
        return reprObj(self, first=('notations',), hideFalsy=True)

    def dump(self, indents=0, file=None) -> None:
        """Dump this to stdout"""
        indentstr = "  " * indents
        info = []
        if self.name:
            info.append(f"name: {self.name}")
        if self.firstClef:
            info.append(f"firstClef: {self.firstClef}")
        if self.groups:
            groups = []
            for id in self.groups:
                group = self._groupRegistry[id]
                groups.append(f"{id}: kind={group.kind}, name={group.name}")
            info.append(f"groups: {groups})")
        print(indentstr[:-1], ", ".join(info))
        for n in self.notations:
            print(indentstr, n, file=file)

    @classmethod
    def makeGroup(cls,
                  parts: list[UnquantizedPart],
                  name='',
                  abbrev='',
                  groupid='',
                  showPartNames=False
                  ) -> str:
        """
        Mark the given parts as belonging to one group

        Args:
            parts: the parts to group
            name: a name for the group
            abbrev: short name used for all systems after the first
            groupid: an explicit group id to use. If not given, a group id is created
            showPartNames: if True, the names for each part are shown (if present).
                Otherwise part names are hidden and only the group name is shown

        Returns:
            the group id assigned to the parts

        """
        if not groupid:
            groupid = cls._registerGroup(name=name, abbrev=abbrev)

        for part in parts:
            part.setGroup(groupid=groupid)
            part.showName = showPartNames
        return groupid

    def distributeByClef(self, maxStaves: int) -> list[UnquantizedPart]:
        """
        Distribute the notations in this Part into multiple parts, based on pitch
        """
        return distributeByClef(self.notations, maxStaves=maxStaves)

    def needsMultipleClefs(self) -> bool:
        """
        True if the notations in this Part extend over the range of one clef
        """
        midinotes: list[float] = sum((n.pitches for n in self), [])  # type: ignore
        return util.midinotesNeedMultipleClefs(midinotes)

    def iterWithOffset(self) -> Iterator[tuple[Notation, F]]:
        """
        Iterate over the notations in this part with their resolved offsets

        The notations are not modified

        Returns:
            an iterator of tuple(notation: Notation, offset: F)

        .. seealso:: :meth:`UnquantizedPart.stack`
        """
        notations = self.notations
        now = notations[0].offset
        if now is None:
            now = F(0)
        for i, n in enumerate(notations):
            if n.offset is not None:
                if n.offset < now:
                    raise ValueError(f"Notations not sorted, {i=}, {n.offset=}, {now=}, "
                                     f"{n} starts before the end of the previous event, {notations[i-1]}. "
                                     f"Notations: {notations}")
                now = n.offset
            yield n, now
            now += n.duration

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
        return pitch / float(dur)

    def fillGaps(self, mingap=F(1, 64)) -> None:
        """
        Fill gaps between notations in this Part, inplace
        """
        if not self.hasGaps():
            return
        self.notations = fillSilences(self.notations, mingap=mingap, start=F0)
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
    def __init__(self, parts: list[UnquantizedPart],
                 title: str = '',
                 scorestruct: ScoreStruct | None = None):
        self.parts = parts
        self.title = title
        self.scorestruct = scorestruct
        self._groupidToName: dict[str, str] = {}

    # def addPart(self, voices: Sequence[UnquantizedPart], id: str, name='', abbrev=''):
    #     assert all(v in self.parts for v in voices)

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

    def __repr__(self):
        parts = [f"parts={self.parts}"]
        if self.title:
            parts.append(f"title={self.title}")
        return f"UnquantizedScore({', '.join(parts)}"


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
        if n0.gliss and n1.isGracenote and all (p in n2.pitches for p in n1.pitches):
            # check if the gracenote is empty
            if not n1.hasAttributes():
                toBeRemoved.append(n1)
                skip = True
    for item in toBeRemoved:
        notations.remove(item)
    return len(toBeRemoved) > 0


def resolveOffsets(notations: list[Notation], start=F0
                   ) -> None:
    """
    Fills all offsets, in place

    Notations with an unset offset are stacked to the end of the previous notation

    Args:
        notations: a list of Notations (or a Part)
        start: the start time, will override the offset of the first event
    """
    n0offset = notations[0].offset
    now = n0offset if n0offset is not None else start
    for i, n in enumerate(notations):
        if n.offset is not None:
            now = n.offset
        now += n.duration
    assert all(n1.end <= n2.offset for n1, n2 in pairwise(notations))


def removeSmallOverlaps(notations: list[Notation], threshold=F(1, 1000)
                        ) -> None:
    """
    Remove overlap between notations, in place

    This should only be used to remove small overlaps product of rounding errors.
    Attack times are never modified, only durations

    Args:
        notations: the notations to remove overlap from
        threshold: how much overlap should be removed. Any overlap higher than
            this results in a ValueError

    """
    if len(notations) < 2:
        return

    for n0, n1 in pairwise(notations):
        assert n1.offset is not None
        diff = n1.offset - n0.end
        if diff != 0:
            if abs(diff) > threshold:
                raise ValueError(f"Too much overlap")
            elif diff > 0:
                n0.duration = n1.offset - n0.offset
            else:
                if (duration := n1.offset - n0.offset) >= 0:
                    n0.duration = duration
                raise ValueError(f"Notations are not sorted: {n0=}, {n1=}")


def fillSilences(notations: list[Notation],
                 mingap=F(1, 64),
                 start=F0,
                 ) -> list[Notation]:
    """
    Return a list of Notations filled with rests

    Args:
        notations: the notes to fill, should have offset set
        mingap: min. gap between two notes. If any notes differ by less
                   than this, the first note absorvs the gap (becomes longer or shorter)
        start: if given, marks the start time to fill. If notations start after
            this offset a rest will be crated from this offset to the start
            of the first notation

    Returns:
        a list of new Notations
    """
    assert notations and all(n.offset is not None and n.offset >= start for n in notations)
    out: list[Notation] = []
    n0 = notations[0]
    if n0.offset is not None and n0.offset > start:
        out.append(Notation.makeRest(duration=n0.offset - start, offset=start))
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
                logger.debug("Small negative gap in notation %s", n)
                out.append(n)
            else:
                raise ValueError(f"Items overlap, {gap=}, {ev0=}, {ev1=}")
        else:
            # gap <= mingap: adjust the dur of n0 to match start of n1
            logger.debug("Small gap (%s), absorve the gap in the first note", gap)
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


def distributeByClef(notations: list[Notation],
                     maxStaves: int,
                     minStaves=1,
                     singleStaffRange=12,
                     staffPenalty=1.2,
                     groupNotesInSpanners=False
                     ) -> list[UnquantizedPart]:
    """
    Distribute the given notations amongst parts with different clefs

    Args:
        notations: the notations to distribute
        maxStaves: max. number of staves
        groupid: a groupid to use for all created parts
        name: a name to use for the resulting group
        abbrev: an abbreviation for the name of the group
        singleStaffRange: if notations fit within this range only one staff
            is used.
        groupNotesInSpanners: keep notations sharing a spanner together

    Returns:
        a list of UnquantizedParts, sorted from low to high
    """
    from . import clefutils
    partpairs = clefutils.explodeNotations(notations,
                                           maxStaves=maxStaves,
                                           minStaves=minStaves,
                                           singleStaffRange=singleStaffRange,
                                           staffPenalty=staffPenalty,
                                           groupNotesInSpanners=groupNotesInSpanners)
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

    return parts


def makeGroupId() -> str:
    """
    Create an id to group notations together
    """
    return str(uuid.uuid1())


def collectGroups(parts: Sequence[UnquantizedPart],
                  kind=''
                  ) -> list[tuple[GroupDef, list[UnquantizedPart]]]:
    """
    Collect all groups within a sequence of parts

    Args:
        parts: the parts, generally all the parts of a score
        kind: one of 'group', 'part' or '' to match any

    Returns:
        a list of tuples (groupdef, parts) where groupdef is the
        GroupDef with information about the group (kind, name, ...)
        and parts are the parts belonging to that group

    """
    assert kind in ('group', 'part', '')
    id2parts = {}
    for part in parts:
        for groupid in part.groups:
            group = UnquantizedPart._groupRegistry[groupid]
            if not kind or group.kind == kind:
                id2parts.setdefault(groupid, []).append(part)
    return [(UnquantizedPart._groupRegistry[id], parts) for id, parts in id2parts.items()]