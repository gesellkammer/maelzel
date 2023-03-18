from __future__ import annotations
from . import util
from . import definitions
import weakref
import copy
from .common import logger

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
        assert kind in {'start', 'end', 'continue'}
        if kind != 'start':
            assert uuid, f"A uuid is needed when continuing or closing a spanner"
        self.kind = kind
        self.uuid = uuid or util.makeuuid(8)
        self.linetype = linetype
        self.placement = placement
        self.color = color
        assert self.uuid

    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f'{cls}(kind={self.kind}, uuid={self.uuid})'

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

    def endSpanner(self: SpannerT) -> SpannerT:
        """
        Create an end spanner corresponding to this start/continue spanner

        Returns:
            a clone of this spanner of kind 'end'
        """
        if self.kind == 'end':
            return self
        out = self.copy()
        out.kind = 'end'
        assert out.uuid == self.uuid
        return out

    def lilyStart(self) -> str:
        raise NotImplementedError

    def lilyEnd(self) -> str:
        raise NotImplementedError


def removeUnmatchedSpanners(notations: Iterable[Notation]) -> int:
    registry: dict[tuple[str, str], tuple[Spanner, Notation]] = {}
    count = 0
    for n in notations:
        if n.spanners:
            for spanner in n.spanners.copy():
                key = (spanner.uuid, spanner.kind)
                if key in registry:
                    logger.debug(f"Duplicate spanner in {n}: {spanner}\n"
                                 f"   already seen in {registry[key][1]}  -- removing it")
                    n.removeSpanner(spanner)
                else:
                    registry[key] = (spanner, n)

    for (uuid, kind), (spanner, n) in registry.items():
        assert kind in ('start', 'end')
        other = 'start' if kind == 'end' else 'end'
        partner = registry.get((uuid, other))
        if not partner:
            logger.debug(f"Found unmatched spanner: {spanner} ({kind=}) in notation {n}, removing")
            logger.debug(f"Registry: {registry.keys()}")
            n.removeSpanner(spanner)
            count += 1
    return count


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
                 placement='', text=''):
        super().__init__(kind=kind, uuid=uuid, placement=placement, linetype=linetype)
        self.text = text


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

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f'{cls}(direction={self.direction}, kind={self.kind}, uuid={self.uuid})'


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

