
from __future__ import annotations
from .event import MObj
from .config import CoreConfig
from . import environment
from .workspace import Workspace, getConfig
from maelzel import scoring
from maelzel.common import F

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .chain import Voice
    from maelzel.scorestruct import ScoreStruct
    from ._typedefs import *
    from typing import Any, Callable, TypeVar
    from .synthevent import PlayArgs, SynthEvent
    MObjT = TypeVar("MObjT", bound="MObj")


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
                 offset: time_t = None,
                 dur: time_t = None):
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
        pitchRanges = [item.pitchRange() for item in self]
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

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

    def quantizePitch(self: MObjT, step=0.) -> MObjT:
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        items = [i.quantizePitch(step) for i in self]
        return self.clone(items=items)

    def timeShift(self, timeoffset: time_t):
        resolved = self.withExplicitTimes()
        items = [item.timeShift(timeoffset) for item in resolved]
        return self.clone(items=items)

    def pitchTransform(self: MObjT, pitchmap: Callable[[float], float]) -> MObjT:
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

