from __future__ import annotations
from . import util
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Type, TypeVar
    T = TypeVar('T', bound='Spanner')


class Spanner:
    endingAtTie = 'last'
    basePriority = 0

    """Should the spanner end at the first or at the last note of a tie"""

    def __init__(self, kind: str, uuid: str = ''):
        assert kind in {'start', 'end', 'continue'}
        if kind != 'start':
            assert uuid, f"A uuid is needed when continuing or closing a spanner"
        self.kind = kind
        self.uuid = uuid or util.makeuuid(8)

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



class Slur(Spanner):
    endingAtTie = 'last'
    basePriority = 1


class Hairpin(Spanner):
    endingAtTie = 'first'
    basePriority = 2

    def __init__(self, kind: str, uuid: str = '', direction='<'):
        super().__init__(kind=kind, uuid=uuid)
        self.direction = direction

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f'{cls}(direction={self.direction}, kind={self.kind}, uuid={self.uuid})'


