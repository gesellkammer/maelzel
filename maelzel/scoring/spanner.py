from __future__ import annotations

import copy

from maelzel._util import reprObj

from . import definitions
from . import util
from .common import logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Sequence 
    from typing_extensions import Self
    from maelzel.scoring import Notation


class Spanner:
    """A spanner is a line that connects two or more notes.

    Attributes:
        kind: One of start or end
        uuid: A uuid for this spanner. uuid is shared with the partner spanner
        linetype: One of solid, dashed, dotted
        placement: above or below
        color: A valid css color
    """
    endingAtTie = 'last'
    basePriority = 0
    lilyPlacementPost = True
    defaultLinetype = 'solid'

    """Should the spanner end at the first or at the last note of a tie"""

    def __init__(self, kind: str = 'start', uuid: str = '', linetype='', placement='',
                 color=''):
        assert kind in {'start', 'end'}
        if kind != 'start':
            assert uuid, "A uuid is needed when continuing or closing a spanner"
        self.kind = kind
        """One of start or end"""

        self.uuid: str = uuid or util.makeuuid(8)
        """A uuid for this spanner. uuid is shared with the partner spanner"""

        self.linetype = linetype
        """One of solid, dashed, dotted or empty string"""

        self.placement = placement
        """above or below"""

        self.color = color
        """A valid css color"""

        self.parent: Notation | None = None
        """If given, the Notation to which this spanner is attached. Not always present"""

        self.nestingLevel = 0
        """The nesting level of this spanner. """

        assert self.uuid

    @staticmethod
    def make(cls: str, kind='start', linetype='', placement='', color='') -> Spanner:
        spannercls = Spanner._strToClass(cls)
        spanner = spannercls(kind=kind, linetype=linetype, placement=placement, color=color)
        return spanner

    @staticmethod
    def _strToClass(s: str) -> type[Spanner]:
        cls = {
            'slur': Slur,
            'beam': Beam
        }.get(s.lower())
        if cls is None:
            raise ValueError(f"Unknown spanner class: {s}")
        return cls

    @staticmethod
    def fromStr(s: str) -> Spanner:
        cls = Spanner._strToClass(s)
        return cls()

    def __hash__(self):
        return hash((self.uuid, self.kind))

    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        def parent(p: Notation | None):
            if p is None:
                return ''
            return p._namerepr()

        return reprObj(self, hideFalsy=True, convert={'parent': parent})

    def priority(self) -> int:
        if self.kind == 'end':
            return 0 + self.basePriority
        else:
            return 1 + self.basePriority

    def copy(self) -> Self:
        """
        Create a copy of this Spanner

        Returns:
            the copy of this spanner
        """
        return copy.copy(self)

    def makeEndSpanner(self) -> Self:
        """
        Create an end spanner corresponding to this start/continue spanner

        Returns:
            a clone of this spanner of kind 'end'
        """
        if self.kind == 'end':
            raise ValueError("This is already an end spanner")
        out = self.copy()
        out.kind = 'end'
        assert out.uuid == self.uuid
        return out

    def resolveLinetype(self) -> str:
        return self.linetype or self.defaultLinetype


def markSpannerNestingLevel(notations: Iterable[Notation]) -> list[Spanner]:
    """
    Marks the nesting levels on these notations, returns a list of open spanners
    Args:
        notations: 

    Returns:

    """
    openSpanners: dict[str, Spanner] = {}
    openSpannersByClass: dict[type, list[str]] = {}
    for n in notations:
        if not n.spanners:
            continue
        for spanner in n.spanners:
            if spanner.kind == 'start':
                openSpanners[spanner.uuid] = spanner
                cls = type(spanner)
                if (uuids := openSpannersByClass.get(cls)) is not None:
                    uuids.append(spanner.uuid)
                    level = len(uuids)
                else:
                    openSpannersByClass[cls] = [spanner.uuid]
                    level= 1
                spanner.nestingLevel = level
            else:
                assert spanner.kind == 'end', f"Invalid spanner: {spanner}"
                startspanner = openSpanners.pop(spanner.uuid, None)
                if startspanner is None:
                    logger.error(f"No start spanner found for {spanner}")
                else:
                    spanner.nestingLevel = startspanner.nestingLevel
                    uuids = openSpannersByClass[type(spanner)]
                    uuids.remove(startspanner.uuid)
    if openSpanners:
        for uuid, spanner in openSpanners.items():
            logger.error(f"No end spanner found for {spanner}, {openSpannersByClass=}")
        return list(openSpanners.values())
    return []


def matchOrfanSpanners(notations: Iterable[Notation], removeUnmatched=False) -> None:
    """Match orphan spanners with their partners.

    Args:
        notations: An iterable of notations to match.
        removeUnmatched: Whether to remove unmatched spanners.

    Returns:
        None
    """
    unmatched = collectUnmatchedSpanners(notations)
    byclass: dict[type, list[Spanner]] = {}
    for spanner, n in unmatched:
        spanner.parent = n
        byclass.setdefault(type(spanner), []).append(spanner)
    for spannercls, spanners in byclass.items():
        stack: list[Spanner] = []
        for spanner in spanners:
            if spanner.kind == 'start':
                stack.append(spanner)
            elif spanner.kind == 'end':
                if stack:
                    startspanner = stack.pop()
                    spanner.uuid = startspanner.uuid
                elif removeUnmatched:
                    assert spanner.parent is not None
                    n = spanner.parent
                    n.removeSpanner(spanner)


def collectUnmatchedSpanners(notations: Iterable[Notation],
                             ) -> list[tuple[Spanner, Notation]]:
    """
    Collect all spanner in notations which do not have a partner spanner

    As a side-effect, removes any duplicate spanners. Duplicates sind spanners
    which have the same uuid and kind. They might appear as an error in
    note splitting/merging

    Args:
        notations: the notations to analyze

    Returns:
        a list of (spanner, notation), where spanner is an unmatched spanner
        and notation is the notation this spanner is attached to

    """
    seen: set[tuple[str, str]] = set()
    duplicates: set[tuple[str, str]] = set()

    for n in notations:
        if n.spanners:
            for spanner in n.spanners:
                if (spanner.uuid, spanner.kind) in seen:
                    duplicates.add((spanner.uuid, spanner.kind))
                else:
                    seen.add((spanner.uuid, spanner.kind))

    if duplicates:
        for uuid, kind in duplicates:
            parts = [n for n in notations if n.spanners and any(sp.uuid == uuid and sp.kind == kind for sp in n.spanners)]
            if kind == 'start':
                for part in parts[1:]:
                    part.spanners = [_ for _ in part.spanners if _.uuid != uuid]
            else:
                for part in parts[:-1]:
                    part.spanners = [_ for _ in part.spanners if _.uuid != uuid]
    
    if not seen:
        return []

    out: list[tuple[Spanner, Notation]] = []
    for n in notations:
        if n.spanners:
            for sp in n.spanners:
                other = 'end' if sp.kind == 'start' else 'start'
                if (sp.uuid, other) not in seen:
                    out.append((sp, n))
    return out
    

def spannersIndex(notations: Iterable[Notation]) -> dict[tuple[str, str], Spanner]:
    return {(s.uuid, s.kind): s for n in notations
            if n.spanners
            for s in n.spanners}


def removeUnmatchedSpanners(notations: Iterable[Notation]) -> int:
    unmatched = collectUnmatchedSpanners(notations)
    for spanner, notation in unmatched:
        notation.removeSpanner(spanner)
    return len(unmatched)


class Slur(Spanner):
    endingAtTie = 'last'
    basePriority = 1
    

class Beam(Spanner):
    """A forced beam"""
    endingAtTie = 'last'
    basePriority = 1
    defaultLinetype = 'solid'


class Bracket(Spanner):
    endingAtTie = 'first'

    def __init__(self, kind='start', uuid: str = '', linetype='solid',
                 placement='above', text='', lineend=''):
        super().__init__(kind=kind, uuid=uuid, placement=placement, linetype=linetype)
        self.text = text
        self.lineend = lineend


class Slide(Spanner):
    """
    A line between two noteheads

    The notes to which the noteheads belong do not need to be adjacent
    """
    endingAtTie = 'first'

    def __init__(self, kind='start', uuid: str = '', linetype='solid',
                 placement='', text='', color=''):
        super().__init__(kind=kind, uuid=uuid, placement=placement,
                         linetype=linetype, color=color)
        self.text = text


class TrillLine(Spanner):
    """
    A trill line

    Args:
        startmark: the start marking. One of 'trill', 'bisb' (for bisbigliando)
        alteration: an alteration can be added to the right of the startmark
        trillpitch: a notename can be given to be placed in parenthesis after the
            main note
    """
    def __init__(self, kind='start', uuid='', startmark='trill', placement='',
                 alteration='', trillpitch=''):
        if alteration:
            assert alteration in definitions.alterations
        super().__init__(kind=kind, uuid=uuid, placement=placement)
        self.startmark = startmark
        self.alteration = alteration
        self.trillpitch = trillpitch


class OctaveShift(Spanner):
    lilyPlacementPost = False

    def __init__(self, kind='start', octaves=1, uuid=''):
        placement = 'above' if octaves >= 1 else 'below'
        super().__init__(kind=kind, uuid=uuid, placement=placement)
        self.octaves = octaves

    def lilyStart(self) -> str:
        return rf"\ottava #{self.octaves}"

    def lilyEnd(self) -> str:
        return r"\ottava #0"


class Hairpin(Spanner):
    """Hairpin crescendo / diminuendo spanner.

    Args:
        kind: The kind of hairpin ("start" / "end")
        uuid: The unique identifier of the hairpin.
        direction: The direction of the hairpin ("<", ">").
        niente: Whether the hairpin is from/to niente.
        placement: The placement of the hairpin ("above" / "below").

    Returns:
        None
    """
    endingAtTie = 'first'
    basePriority = 2

    def __init__(self, kind: str, uuid: str = '', direction='<', niente=False, placement=''):
        super().__init__(kind=kind, uuid=uuid, placement=placement)
        self.direction = direction
        self.niente = niente


class LineSpan(Spanner):
    def __init__(self, kind='start', uuid='', linetype='solid', placement='',
                 starttext='', endtext='', middletext='', verticalAlign='',
                 starthook=False, endhook=False):
        super().__init__(kind=kind, uuid=uuid, linetype=linetype, placement=placement)
        self.starttext = starttext
        self.endtext = endtext
        self.middletext = middletext
        self.verticalAlign = verticalAlign
        self.starthook = starthook
        self.endhook = endhook


def solveHairpins(notations: Sequence[Notation], startDynamic='mf') -> None:
    """
    Resolve end spanners for hairpins

    A hairpin can be created without an end spanner. In that case the end
    of the hairpin is the next dynamic

    Args:
        notations: The notations to solve hairpins for.
        startDynamic: The starting dynamic for hairpins.

    """
    hairpins = {(s.uuid, s.kind) for n in notations if n.spanners
                for s in n.spanners if isinstance(s, Hairpin)}

    currentDynamic = startDynamic
    lastHairpin: Hairpin | None = None
    for n in notations:
        if n.dynamic and n.dynamic != currentDynamic:
            currentDynamic = n.dynamic
            if lastHairpin:
                n.addSpanner(lastHairpin.makeEndSpanner())
                lastHairpin = None
        if n.spanners:
            for spanner in n.spanners:
                if isinstance(spanner, Hairpin) and spanner.kind == 'start' and (spanner.uuid, 'end') not in hairpins:
                    lastHairpin = spanner


def moveEndSpannersToEndOfLogicalTie(notations: Iterable[Notation]) -> None:
    """
    Move end spanners to end of logical tie

    On modern notation, some spanners, like slurs, end at the end of the
    last note's logical tie. That means, if the note to which the
    end spanner is attached to is tied, then the spanner should  actually
    end on the last note of the logical tie.

    Args:
        notations: the notations to analyze

    """
    spannerClasses = (Slur, )
    stack: list[Spanner] = []
    for n in notations:
        if n.tiedPrev and not n.tiedNext and stack:
            for spanner in stack:
                n.addSpanner(spanner)
            stack.clear()
        elif n.spanners and n.tiedNext and any(isinstance(s, spannerClasses) for s in n.spanners):
            for spanner in n.spanners.copy():
                if spanner.kind == 'end' and isinstance(spanner, spannerClasses):
                    stack.append(spanner)
                    n.spanners.remove(spanner)
