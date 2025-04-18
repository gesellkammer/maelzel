
from __future__ import annotations
from maelzel.core.mobj import MObj
from .config import CoreConfig
from . import environment
from .workspace import Workspace, getConfig
from maelzel import scoring
from maelzel.common import F, F0

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.common import time_t
    from typing_extensions import Self
    from typing import Any, Callable
    from .synthevent import PlayArgs, SynthEvent


_EMPTYSLICE = slice(None, None, None)


class MObjList(MObj):
    """
    A sequence of music objects

    This class serves as a base for container classes like Score
    **It should not be instantiated by itself**

    """
    def __init__(self,
                 label='',
                 properties: dict[str, Any] = None,
                 offset: F = None,
                 dur: F = F0):
        """a list of MusicObj inside this container"""

        super().__init__(dur=dur, offset=offset, label=label, properties=properties)

    def append(self, obj: MObj) -> None:
        """
        Append an item

        Args:
            obj: the object to append
        """
        self.getItems().append(obj)
        self._changed()

    def getItems(self) -> list[MObj]:
        raise NotImplementedError()

    def __iter__(self):
        return iter(self.getItems())

    def __getitem__(self, idx):
        if idx == _EMPTYSLICE:
            return self.getItems()
        return self.getItems().__getitem__(idx)

    def __len__(self):
        return len(self.getItems())

    def __hash__(self):
        items = [type(self).__name__, self.label, self.offset, len(self)]
        if self.symbols:
            items.extend(self.symbols)
        ownitems = self.getItems()
        if ownitems:
            items.extend(ownitems)
        out = hash(tuple(items))
        return out

    def pitchRange(self) -> tuple[float, float] | None:
        ranges = [item.pitchRange() for item in self]
        ranges = [r for r in ranges if r is not None]
        return min(r[0] for r in ranges), max(r[1] for r in ranges)

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig = None,
                      parentOffset: F | None = None
                      ) -> list[scoring.Notation]:

        """
        Returns the scoring events corresponding to this object

        Args:
            groupid: if given, all events are given this groupid
            config: the configuration used

        Returns:
            the scoring notations representing this object
        """
        raise NotImplementedError()

    def _synthEvents(self,
                     playargs: PlayArgs,
                     parentOffset: F,
                     workspace: Workspace
                     ) -> list[SynthEvent]:
        if self.playargs:
            playargs = playargs.updated(self.playargs)
        out = []
        parentOffset = self.parent.absOffset() if self.parent else F(0)
        for item in self.getItems():
            events = item._synthEvents(playargs=playargs, workspace=workspace,
                                       parentOffset=parentOffset)
            out.extend(events)
        return out

    def quantizePitch(self, step=0.) -> Self:
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        items = [i.quantizePitch(step) for i in self]
        return self.clone(items=items)

    def timeShift(self, timeoffset: time_t):
        resolved = self.withExplicitOffset()
        items = [item.timeShift(timeoffset) for item in resolved]
        return self.clone(items=items)

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Self:
        newitems = [item.pitchTransform(pitchmap) for item in self]
        return self.clone(items=newitems)

    def dump(self, indents=0, forcetext=False):
        if environment.insideJupyter and not forcetext:
            from IPython.display import HTML, display
            header = f'{"  " * indents}<strong>{type(self).__name__}</strong>'
            display(HTML(header))
        else:
            print(f'{"  "*indents}{type(self).__name__}')
            if self.playargs:
                print("  "*(indents+1), self.playargs)
        for item in self:
            item.dump(indents+1)
