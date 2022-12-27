from __future__ import annotations

from maelzel.common import F
from .mobj import MObj, MContainer
from .config import CoreConfig
from .chain import Voice, Chain
from .workspace import getConfig
from .mobjlist import MObjList
from maelzel.scorestruct import ScoreStruct
from maelzel import scoring

from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    from ._typedefs import *


__all__ = (
    'Score',
)


class Score(MContainer, MObjList):
    """
    A Score is a list of Voices

    Args:
        voices: the voices of this score.
        scorestruct: it is possible to attach a ScoreStruct to a score instead of depending
            on the active scorestruct
        title: a title for this score

    """
    _acceptsNoteAttachedSymbols = False

    __slots__ = ('voices', '_modified')

    def __init__(self,
                 voices: Sequence[Voice | Chain] = (),
                 scorestruct: ScoreStruct | None = None,
                 title=''):
        asvoices: list[Voice] = [item if isinstance(item, Voice) else item.asVoice()
                                 for item in voices]
        for voice in asvoices:
            voice.parent = self

        self.voices: list[Voice] = asvoices
        """the voices of this score"""

        MObjList.__init__(self, label=title, offset=F(0))
        self._scorestruct = scorestruct
        self._modified = True

    def scorestruct(self) -> ScoreStruct | None:
        return self._scorestruct

    def setScoreStruct(self, scorestruct: ScoreStruct) -> None:
        """
        Set the ScoreStruct for this Score

        Scores are the only objects in maelzel.core which can have a
        ScoreStruct attached to them. This ScoreStruct will be
        used for any object embedded downstream

        Args:
            scorestruct: the ScoreStruct

        """
        self._scorestruct = scorestruct
        self._changed()


    def __hash__(self):
        items = [type(self).__name__, self.label, self.offset, len(self.voices)]
        if self.symbols:
            items.extend(self.symbols)
        if self.voices:
            items.extend(self.voices)
        out = hash(tuple(items))
        return out

    def __repr__(self):
        if not self.voices:
            info = ''
        else:
            info = f'{len(self.voices)} voices'
            # info = f'voices={self.voices}'
        return f'Score({info})'

    def _changed(self) -> None:
        self._modified = True
        self.dur = None

    def getItems(self) -> list[Voice]:
        return self.voices

    def append(self, voice: Voice | Chain) -> None:
        if isinstance(voice, Chain):
            voice = voice.asVoice()
        voice.parent = self
        self.voices.append(voice)
        self._changed()

    def resolveDur(self) -> F:
        if self.dur is not None:
            return self.dur

        if not self.voices:
            return F(0)

        dur = max(v.resolveDur() for v in self.voices)
        self.dur = dur
        return dur

    def scoringParts(self, config: CoreConfig | None = None
                     ) -> list[scoring.Part]:
        parts = []
        for voice in self.voices:
            voiceparts = voice.scoringParts(config or getConfig())
            parts.extend(voiceparts)
        return parts

    def scoringEvents(self, groupid='', config: CoreConfig | None = None
                      ) -> list[scoring.Notation]:
        parts = self.scoringParts(config or getConfig())
        flatevents = []
        for part in parts:
            flatevents.extend(part)
        # TODO: deal with groupid
        return flatevents

    def childOffset(self, child: MObj) -> F:
        offset = child._detachedOffset()
        return offset if offset is not None else F(0)

    def childDuration(self, child: MObj) -> F:
        return child.resolveDur()

    def absoluteOffset(self) -> F:
        return F(0)