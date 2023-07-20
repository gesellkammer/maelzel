from __future__ import annotations
import itertools
from emlib import iterlib
from maelzel._util import reprObj
from .common import *
from .notation import *
from . import definitions
from . import util
from . import attachment


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Iterator, Callable
    from maelzel.scoring import quant


__all__ = (
    'Notation',
    'UnquantizedPart',
    'UnquantizedScore',
    'NotatedDuration',
    'fillSilences',
    'resolveOffsets',
    'packInParts',
    'notationsCanMerge',
    'mergeNotationsIfPossible',
    'removeSmallOverlaps',
    'distributeNotationsByClef',
)


def _parseGroupname(name: str, separator="::") -> tuple[str, str]:
    parts = name.split(separator, maxsplit=1)
    return (parts[0], '') if len(parts) == 1 else (parts[0], parts[1])



class UnquantizedPart:
    """
    An UnquantizedPart is a list of unquantized non-simultaneous :class:`Notation`

    Args:
        notations: the events (notes, chords) in this part
        name: a label to identify this track in particular (a name)
        groupid: an identification (given by makeGroupId), used to identify
            tracks which belong to a same tree
        shortname: a name abbreviation used as staf name after the first system

    .. seealso:: :class:`~maelzel.scoring.quant.QuantizedPart`,
    """
    def __init__(self,
                 notations: list[Notation] = None,
                 name='',
                 shortname='',
                 groupid: str = '',
                 groupname='',
                 showName=True,
                 quantProfile: quant.QuantizationProfile | None = None,
                 ):

        self.notations: list[Notation] = notations

        self.groupid: str = groupid
        """A UUID identifying this Part (can be left unset)"""

        self.groupname: tuple[str, str] | None = _parseGroupname(groupname) if groupname else None
        """Used as staff group name for parts grouped together. It can include a shortname as
        <name>::<shortname>"""

        self.name: str = name
        """The name of the part"""

        self.shortname: str = shortname
        """A shortname to use as abbreviation"""

        self.quantProfile = quantProfile
        """A quantization profile can be attached to an UnquantizedPart for later quantization"""

        self.showName = showName
        """If True, show the part name when rendered"""

        self.hooks: list[attachment.Hook] = []
        """Callbacks to be triggered at different stages"""

        self.attachments: list[attachment.Attachment] = []
        """A list of Attachments for the part itself"""

        if notations:
            assert all(isinstance(n, Notation) for n in notations)
            _repairGracenoteAsTargetGliss(self.notations)

    def setGroup(self, groupid: str, name='', shortname='', showPartName=False):
        self.groupid = groupid
        self.groupname = (name, shortname)
        self.showName = showPartName

    def __getitem__(self, item) -> Notation:
        return self.notations.__getitem__(item)

    def __iter__(self) -> Iterator[Notation]:
        return iter(self.notations)

    def __repr__(self) -> str:
        return reprObj(self, priorityargs=('notations',))

    def dump(self) -> None:
        """Dump this to stdout"""
        for n in self.notations:
            print(n)

    @staticmethod
    def groupParts(parts: list[UnquantizedPart], name='', shortname='', groupid='', showPartNames=False
                   ) -> str:
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
        midinotes: list[float] = sum((n.pitches for n in self), [])
        return util.midinotesNeedMultipleClefs(midinotes)

    def stack(self) -> None:
        """
        Stack the notations of this part **inplace**.

        Stacking means filling in any unresolved offset in this part.
        After this operation, all Notations in this UnquantizedPart have an
        explicit offset.
        """
        resolveOffsets(self.notations)

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

    def fillGaps(self, mingap=1/64) -> None:
        """
        Fill gaps between notations in this Part, inplace
        """
        if not self.hasGaps():
            return
        self.notations = fillSilences(self.notations, mingap=mingap, offset=0)
        assert not self.hasGaps()

    def hasGaps(self) -> bool:
        """Does this Part have gaps?"""
        assert all(n.offset is not None and n.duration is not None for n in self.notations)
        return any(n0.end < n1.offset for n0, n1 in iterlib.pairwise(self.notations))


class UnquantizedScore:
    """
    An UnquantizedScore is a list of UnquantizedParts
    """
    def __init__(self, parts: list[UnquantizedPart] = None, title: str = ''):
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


def _repairGracenoteAsTargetGliss(notations: list[Notation]) -> None:
    """
    Removes superfluous end glissandi notes **inplace**

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
    resolveOffsets(notations)


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
    now = _ if (_ := notations[0].offset) is not None else start if start is not None else F(0)
    assert now is not None and now >= 0
    for i, n in enumerate(notations):
        assert n.duration is not None
        if n.offset is None or overrideOffset:
            assert n.duration is not None
            n.offset = now
        now += n.duration
    for n1, n2 in iterlib.pairwise(notations):
        if n1.end > n2.offset:
            raise ValueError(f"Notations are not sorted: {n1}, {n2}")
    removeSmallOverlaps(notations)


def removeSmallOverlaps(notations: list[Notation], threshold=F(1, 1000)) -> None:
    """
    Remove overlap between notations.

    This should be only used to remove small overlaps product of rounding errors.
    """
    if len(notations) < 2:
        return
    mindur = threshold * 4
    for n0, n1 in iterlib.pairwise(notations):
        diff = n0.end - n1.offset
        if diff < 0:
            if abs(diff) > threshold or n0.duration < mindur:
                raise ValueError(f"Notes overlap by too much: {n0=}, {n1=}")
            n0.duration = abs(diff)
        elif diff > 0:
            if diff > threshold:
                raise ValueError(f"Notes overlap by too much: {diff=}, {n0=}, {n1=}")
            duration = n1.offset - n0.offset
            if duration < 0:
                raise ValueError(f"Note with negative duration: {n0=}, {n1=}")
            n0.duration = duration


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


def _findMostSuitableClefs(notations: list[Notation], maxstaves: int) -> list[str]:
    clefs2 = [('g', 'f'), ('g8', 'f'), ('g15', 'f'), ('g15', 'g'), ('g', 'f8')]
    pass


def distributeNotationsByClef(notations: list[Notation],
                              maxstaves=3,
                              groupid: str = '',
                              name='',
                              shortname='',
                              ) -> list[UnquantizedPart]:
    from . import clefutils
    partpairs = clefutils.explodeNotations(notations, maxstaves=maxstaves)
    parts = [UnquantizedPart(notations) for clef, notations in partpairs]

    if groupid:
        for p in parts:
            p.groupid = groupid

    if name:
        if len(parts) == 1:
            parts[0].name = name
            parts[0].shortname = shortname
        else:
            for i, part in enumerate(parts):
                part.name = f'{name}-{i + 1}'
                if shortname:
                    part.shortname = f'{shortname}{i + 1}'

    return parts

def _distributeNotationsByClef(notations: list[Notation],
                              maxstaves=3,
                              filterRests=False,
                              groupid: str = '',
                              name='',
                              shortname='',
                              ) -> list[UnquantizedPart]:
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
    if maxstaves == 1:
        raise ValueError("maxstaves should be between 2 and 3")

    if maxstaves == 2:
        clefs = _findMostSuitableClefs(notations, maxstaves)
    elif maxstaves == 3:
        clefs = ('15a', 'g', 'f')
    else:
        raise ValueError("maxstaves should be between 2 and 3")

    # The order is important
    parts = {clef: [] for clef in clefs}
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


    parts = [UnquantizedPart(part) for part in parts.values() if part]
    if groupid:
        for p in parts:
            p.groupid = groupid
    if name:
        if len(parts) == 1:
            parts[0].name = name
            parts[0].shortname = shortname
        else:
            for i, part in enumerate(parts):
                part.name = f'{name}-{i+1}'
                if shortname:
                    part.shortname = f'{shortname}{i+1}'
    return parts


def packInParts(notations: list[Notation],
                maxrange=36,
                keepGroupsTogether=True
                ) -> list[UnquantizedPart]:
    """
    Pack a list of possibly simultaneous notations into tracks

    The notations within one track are NOT simulatenous. Notations belonging
    to the same tree are kept in the same track.

    Args:
        notations: the Notations to _packold
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

    packedTracks = packing.packInTracks(items, maxrange=maxrange)
    return [UnquantizedPart(track.unwrap()) for track in packedTracks]


def removeRedundantDynamics(notations: list[Notation],
                            resetAfterRest=True,
                            minRestDuration: time_t = F(1, 16)) -> None:
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
