from __future__ import annotations

from dataclasses import dataclass
from maelzel.common import F
from maelzel.scoring import core


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.scoring.common import division_t
    from maelzel.scoring.quant import QuantizedPart


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
    def __init__(self, kind: str, items: list[QuantizedPart | PartNode], name='', abbrev='', id=''):
        assert kind in ('', 'group', 'part')
        if not id:
            id = core.makeGroupId()
        self.kind = kind
        self.items = items
        self.name = name
        self.abbrev = abbrev
        self.id = id

    def copy(self) -> PartNode:
        return PartNode(self.kind, items=self.items, name=self.name, abbrev=self.abbrev, id=self.id)

    def clone(self, **kws) -> PartNode:
        node = self.copy()
        for k, v in kws.items():
            setattr(node, k, v)
        return node

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

    def serialize(self) -> list[dict | QuantizedPart]:
        d = {'kind': self.kind,
             'name': self.name,
             'id': self.id,
             'open': True,
             'abbrev': self.abbrev}
        if self.kind == 'part':
            part = self.items[0]
            d['struct'] = part.struct  # type: ignore
            d['clef'] = part.firstClef or part.bestClef()
        if self.kind == 'part':
            if not self.name:
                d['name'] = next((voice.name for voice in self.items), '')
        out: list[dict | QuantizedPart] = [d]
        for item in self.items:
            if isinstance(item, PartNode):
                out.extend(item.serialize())
            else:
                out.append(item)
        out.append({'kind': self.kind, 'name': self.name, 'open': False, 'id': self.id})
        return out

    def append(self, item: QuantizedPart | PartNode):
        self.items.append(item)