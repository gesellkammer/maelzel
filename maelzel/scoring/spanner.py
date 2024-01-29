from __future__ import annotations
from . import util
from . import definitions
import copy
from .common import logger
from maelzel._util import reprObj

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.scoring import Notation
    from typing import TypeVar, Iterable
    SpannerT = TypeVar('SpannerT', bound='Spanner')


class Spanner:
    endingAtTie = 'last'
    basePriority = 0
    lilyPlacementPost = True

    """Should the spanner end at the first or at the last note of a tie"""

    def __init__(self, kind: str = 'start', uuid: str = '', linetype='solid', placement='',
                 color=''):
        assert kind in {'start', 'end'}
        if kind != 'start':
            assert uuid, f"A uuid is needed when continuing or closing a spanner"
        self.kind = kind
        """One of start or end"""
        self.uuid = uuid or util.makeuuid(8)
        """A uuid for this spanner. uuid is shared with the partner spanner"""
        self.linetype = linetype
        """One of solid, dashed, dotted"""
        self.placement = placement
        """above or below"""
        self.color = color
        """A valid css color"""
        self.parent: Notation | None = None
        """If given, the Notation to which this spanner is attached. Not always present"""
        self.nestingLevel = 1
        """The nesting level of this spanner. """

        assert self.uuid

    def __hash__(self):
        return hash((self.uuid, self.kind))

    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return reprObj(self, hideFalsy=True)

    def priority(self) -> int:
        if self.kind == 'end':
            return 0 + self.basePriority
        else:
            return 1 + self.basePriority

    def copy(self: SpannerT) -> SpannerT:
        """
        Create a copy of this Spanner

        Returns:
            the copy of this spanner
        """
        return copy.copy(self)

    def makeEndSpanner(self: SpannerT) -> SpannerT:
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


def markNestingLevels(notations: Iterable[Notation]) -> None:
    openSpannersByClass: dict[type, list[Spanner]] = {}
    uuidToLevel: dict[str, int] = {}
    for n in notations:
        if not n.spanners:
            continue
        for spanner in n.spanners:
            if spanner.kind == 'start':
                openspanners = openSpannersByClass.setdefault(type(spanner), [])
                openspanners.append(spanner)
                for i, spanner2 in enumerate(openspanners):
                    spanner2.nestingLevel = len(openspanners) - i
                    uuidToLevel[spanner2.uuid] = spanner2.nestingLevel
            elif spanner.kind == 'end':
                openspanners = openSpannersByClass.get(type(spanner))
                if openspanners:
                    startspanner = next((s for s in openspanners if s.uuid == spanner.uuid), None)
                    if startspanner is not None:
                        openspanners.remove(startspanner)
                        level = uuidToLevel.get(startspanner.uuid)
                        if level is not None:
                            spanner.nestingLevel = level


def matchOrfanSpanners(notations: Iterable[Notation], removeUnmatched=False) -> None:
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
    registry: dict[tuple[str, str], tuple[Spanner, Notation]] = {}
    unmatched: list[tuple[Spanner, Notation]] = []
    toberemoved = []
    for n in notations:
        if n.spanners:
            for spanner in n.spanners:
                key = (spanner.uuid, spanner.kind)
                if key in registry:
                    logger.warning(f"Duplicate spanner in {n}: {spanner}\n"
                                   f"   already seen in {registry[key][1]}  -- removing it")
                    toberemoved.append((n, spanner))
                else:
                    registry[key] = (spanner, n)

    if toberemoved:
        for n, spanner in toberemoved:
            n.spanners.remove(spanner)

    for (uuid, kind), spannerpair in registry.items():
        other = 'start' if kind == 'end' else 'end'
        if not registry.get((uuid, other)):
            unmatched.append(spannerpair)
    return unmatched


def spannersIndex(notations: Iterable[Notation]) -> dict[tuple[str, str], Spanner]:
    return {(s.uuid, s.kind): s
            for n in notations
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


def solveHairpins(notations: Iterable[Notation], startDynamic='mf') -> None:
    """
    Resolve end spanners for hairpins

    A hairpin can be created without an end spanner. In that case the end
    of the hairpin is the next dynamic

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
                if isinstance(spanner, Hairpin) and spanner.kind == 'start' and not (spanner.uuid, 'end') in hairpins:
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


