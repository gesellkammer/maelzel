from __future__ import annotations

from dataclasses import dataclass
from maelzel.common import F
from maelzel.scoring import core


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.scoring.common import division_t
    from maelzel.scoring.quant import QuantizedPart
    from typing import Sequence, Any


@dataclass
class QuantizedBeatDef:
    """
    Represent a quantized beat without the actual data
    """
    offset: F
    duration: F = F(1)
    division: division_t = (1,)
    weight: int = 0

    @property
    def end(self) -> F:
        return self.offset + self.duration


class PartNode:
    def __init__(self, kind: str, items: Sequence[QuantizedPart | PartNode], name='', abbrev='', id=''):
        assert kind in ('', 'group', 'part')
        self.kind = kind
        self.items = items if isinstance(items, list) else list(items)
        self.name = name
        self.abbrev = abbrev
        self.id = id or core.makeGroupId()

    # def copy(self) -> PartNode:
    #     return PartNode(self.kind, items=self.items, name=self.name, abbrev=self.abbrev, id=self.id)
    #
    # def clone(self, **kws) -> PartNode:
    #     node = self.copy()
    #     for k, v in kws.items():
    #         setattr(node, k, v)
    #     return node

    def __contains__(self, item: QuantizedPart):
        return item in self.items

    def rank(self) -> int:
        return 0 if self.kind == '' else 1 if self.kind == 'group' else 2

    def _show(self, indent=0):
        indentstr = '  ' * indent
        itemsrepr = []
        for item in self.items:
            itemsrepr.append(item._show(indent+1) if isinstance(item, PartNode) else indentstr+repr(item))

        info = [self.kind]
        if self.name:
            info.append(f"name={self.name}")
        if self.abbrev:
            info.append(f"abbrev={self.abbrev}")
        lines = [f"{indentstr}{self.__class__.__name__}({', '.join(info)})"]
        lines.extend(itemsrepr)
        return "\n".join(lines)

    def __repr__(self):
        return self._show()

    def serialize(self) -> list[dict[str, Any] | QuantizedPart]:
        """
        Serialize this PartNode

        Called on the rute serializes the entire tree.

        Returns a flat list where each element is a dict,
        representing an "operation" (opening or closing a group
        or a multivoice part), or a part itself.

        For the case of a score with the following structure::

            Score
              part1
              part2
              group1:
                part3
                part1:
                  part4
                  part5
              group2:
                part6
                part7

        Returns a list::

            part1
            part2
            {'kind': 'group', 'open': True, name: 'group1', ...}
            part3
            {'kind': 'part', 'open': True, name: 'part1', ...}
            part4
            part5
            {'kind': 'part', 'open': False, ...}
            {'kind': 'group', 'open': False, ...}
            {'kind': 'group', 'open': True, ...}
            part6
            part7
            {'kind': 'group', 'open': False, ...}


        """
        d = {'kind': self.kind,
             'name': self.name,
             'id': self.id,
             'open': True,
             'abbrev': self.abbrev}
        if self.kind == 'part':
            part = self.items[0]
            # assert isinstance(part, QuantizedPart)
            d['struct'] = part.struct  # type: ignore
            d['clef'] = part.firstClef or part.bestClef()
            if not self.name:
                d['name'] = next((voice.name for voice in self.items), '')
        out: list[dict | QuantizedPart] = [d]
        for item in self.items:
            if isinstance(item, PartNode):
                out.extend(item.serialize())
            else:
                out.append(item)
        out.append({'kind': self.kind, 'open': False, 'id': self.id, 'name': self.name})
        return out

