from __future__ import annotations

from maelzel.common import F
from .config import CoreConfig
from .chain import Voice, Chain
from .workspace import getConfig
from .mobj import MObj
from .mobjlist import MObjList
from maelzel.scorestruct import ScoreStruct
from maelzel import scoring

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._typedefs import *


__all__ = (
    'Score',
)


class Score(MObjList):
    """
    A Score is a list of Voices

    Args:
        voices: the voices of this score.
        scorestruct: it is possible to attach a ScoreStruct to a score instead of depending
            on the active scorestruct
        title: a title for this score

    """
    _acceptsNoteAttachedSymbols = False

    __slots__ = ('voices',)

    def __init__(self,
                 voices: list = None,
                 scorestruct: ScoreStruct = None,
                 title: str = ''):
        asvoices = []
        if voices:
            for obj in voices:
                if isinstance(obj, Voice):
                    assert obj.offset == 0
                    asvoices.append(obj)
                elif isinstance(obj, Chain):
                    asvoices.append(obj.asVoice())
                else:
                    raise TypeError(f"Cannot convert {obj} to a voice")
        else:
            voices = []
        self.voices: list[Voice] = voices
        """the voices of this score"""

        super().__init__(label=title, offset=F(0))
        self._scorestruct = scorestruct
        if scorestruct:
            for v in self.voices:
                v.setScoreStruct(scorestruct)
        self._changed()

    def __repr__(self):
        if not self.voices:
            info = ''
        else:
            info = f'{len(self.voices)} voices'
            # info = f'voices={self.voices}'
        return f'Score({info})'

    def _changed(self) -> None:
        self.dur = self.resolvedDur()

    def getItems(self) -> list[Voice]:
        return self.voices

    def append(self, voice: Voice) -> None:
        struct = self.scorestruct
        voicestruct = voice.scorestruct
        if struct:
            if voicestruct:
                if struct != voicestruct:
                    raise ValueError("The voice has a scorestruct attached different from the score's")
            else:
                voice.setScoreStruct(struct)
        elif voicestruct:
            # the score has no scorestruct but the voice has, adopt that
            assert all(voice.scorestruct == voicestruct for voice in self.voices)
            self.setScoreStruct(voicestruct)
        self.voices.append(voice)
        self._changed()

    def resolvedDur(self) -> F:
        return max(v.resolvedDur() for v in self.voices)

    def scoringParts(self, config: CoreConfig = None
                     ) -> list[scoring.Part]:
        parts = []
        for voice in self.voices:
            voiceparts = voice.scoringParts(config or getConfig())
            parts.extend(voiceparts)
        return parts

    def scoringEvents(self, groupid: str = None, config: CoreConfig = None
                      ) -> list[scoring.Notation]:
        parts = self.scoringParts(config or getConfig())
        flatevents = []
        for part in parts:
            flatevents.extend(part)
        # TODO: deal with groupid
        return flatevents

    def setScoreStruct(self, scorestruct: ScoreStruct):
        self._scorestruct = scorestruct
        for v in self.voices:
            v.setScoreStruct(scorestruct)
