
from __future__ import annotations
from .mobj import MObj
from .tools import packInVoices
from .config import CoreConfig
from . import environment
from .workspace import Workspace, getConfig
from maelzel import scoring
from emlib import misc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .chain import Voice
    from maelzel.scorestruct import ScoreStruct
    from ._typedefs import *
    from typing import Any, Callable, TypeVar
    from .synthevent import PlayArgs, SynthEvent
    MObjT = TypeVar("MObjT", bound="MObj")


class MObjList(MObj):
    """
    A sequence of music objects

    This class serves as a base for all container classes (Chain, Group, Voice)
    **It should not be instantiated by itself**

    """
    def __init__(self,
                 label='',
                 properties: dict[str, Any] = None,
                 start: time_t = None,
                 dur: time_t = None):
        """a list of MusicObj inside this container"""

        super().__init__(dur=dur, start=start, label=label, properties=properties)

    def append(self, obj: MObj) -> None:
        """
        Append an item

        Args:
            obj: the object to append
        """
        self._getItems().append(obj)
        self._changed()

    def _changed(self) -> None:
        items = self._getItems()
        self.start = min(it.start for it in items)
        end = max(it.end for it in items)
        self.dur = end - self.start
        super()._changed()

    def _getItems(self) -> list[MObj]:
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._getItems())

    def __getitem__(self, idx):
        return self._getItems().__getitem__(idx)

    def __len__(self):
        return len(self._getItems())

    def __hash__(self):
        items = [type(self).__name__, self.label, self.start, len(self)]
        if self.symbols:
            items.extend(self.symbols)
        ownitems = self._getItems()
        if ownitems:
            items.extend(ownitems)
        out = hash(tuple(items))
        return out

    def pitchRange(self) -> tuple[float, float] | None:
        pitchRanges = [item.pitchRange() for item in self]
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

    def scoringEvents(self, groupid: str = None, config: CoreConfig = None
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

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        playargs.fillWith(self.playargs)
        return misc.sumlist(item._synthEvents(playargs.copy(), workspace)
                            for item in self._getItems())

    def quantizePitch(self: MObjT, step=0.) -> MObjT:
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        items = [i.quantizePitch(step) for i in self]
        return self.clone(items=items)

    def timeShift(self, timeoffset: time_t):
        resolved = self.resolved()
        items = [item.timeShift(timeoffset) for item in resolved]
        return self.clone(items=items)

    def pitchTransform(self: MObjT, pitchmap: Callable[[float], float]) -> MObjT:
        newitems = [item.pitchTransform(pitchmap) for item in self]
        return self.clone(items=newitems)

    def dump(self, indents=0):
        if environment.insideJupyter:
            from IPython.display import HTML, display
            header = f'{"  " * indents}<strong>{type(self).__name__}</strong>'
            display(HTML(header))
        else:
            print(f'{"  "*indents}{type(self).__name__}')
            if self._playargs:
                print("  "*(indents+1), self.playargs)
        for item in self:
            item.dump(indents+1)

    def makeVoices(self) -> list[Voice]:
        """
        Construct a list of Voices from this object
        """
        return packInVoices(self._getItems())

    def adaptToScoreStruct(self, newstruct: ScoreStruct, oldstruct: ScoreStruct = None):
        newitems = [item.adaptToScoreStruct(newstruct, oldstruct)
                    for item in self]
        return self.clone(items=newitems)
