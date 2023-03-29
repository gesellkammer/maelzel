from __future__ import annotations

from maelzel.common import F
from .mobj import MObj, MContainer
from .event import MEvent
from .config import CoreConfig
from .chain import Voice, Chain
from .workspace import getConfig
from .synthevent import PlayArgs, SynthEvent
from .workspace import Workspace
from maelzel.scorestruct import ScoreStruct
from maelzel import scoring
from ._common import UNSET

from typing import TYPE_CHECKING, Sequence, Callable
if TYPE_CHECKING:
    from ._typedefs import *


__all__ = (
    'Score',
)


def _asvoice(o: MObj):
    if isinstance(o, MEvent):
        return Voice([o])
    elif isinstance(o, Chain):
        return o.asVoice()
    else:
        raise TypeError(f"Cannot create a Voice from {o} (type: {type(o)})")


class Score(MObj, MContainer):
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
                 voices: Sequence[Voice | Chain | MEvent] = (),
                 scorestruct: ScoreStruct | None = None,
                 title=''):
        asvoices: list[Voice] = [item if isinstance(item, Voice) else _asvoice(item)
                                 for item in voices]
        for voice in asvoices:
            voice.parent = self

        self.voices: list[Voice] = asvoices
        """the voices of this score"""

        super().__init__(label=title, offset=F(0))

        self._scorestruct = scorestruct
        self._modified = True

    @staticmethod
    def fromMusicxml(musicxml: str, enforceParsedSpelling=True) -> Score:
        """
        Create a Score from musicxml text

        Args:
            musicxml: the musicxml text to parse (read from a .musicxml file)
            enforceParsedSpelling: if True, the enharmonic spelling defined in the
                musicxml text will be enforced

        Returns:
            a Score
        """
        from maelzel.core import musicxml as mxml
        return mxml.parseMusicxml(musicxml, enforceParsedSpelling=enforceParsedSpelling)

    def scorestruct(self) -> ScoreStruct | None:
        """The attached ScoreStruct, if present"""
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
        self._dur = None

    def _update(self):
        if not self._modified:
            return
        self._dur = max(v.dur for v in self.voices) if self.voices else F(0)
        self._modified = False

    def append(self, voice: Voice | Chain) -> None:
        """Append a Voice to this Score"""
        if isinstance(voice, Chain):
            voice = voice.asVoice()
        voice.parent = self
        self.voices.append(voice)
        self._changed()

    @property
    def dur(self) -> F:
        "The duration of this object"
        if self._modified:
            self._update()
        assert self._dur is not None
        return self._dur

    def scoringParts(self, config: CoreConfig | None = None
                     ) -> list[scoring.Part]:
        parts = []
        for voice in self.voices:
            voiceparts = voice.scoringParts(config or getConfig())
            parts.extend(voiceparts)
        return parts

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig = None,
                      parentOffset: F | None = None
                      ) -> list[scoring.Notation]:
        parts = self.scoringParts(config or getConfig())
        flatevents = []
        for part in parts:
            flatevents.extend(part)
        # TODO: deal with groupid
        return flatevents

    def _synthEvents(self,
                     playargs: PlayArgs,
                     parentOffset: F,
                     workspace: Workspace
                     ) -> list[SynthEvent]:
        if self.playargs:
            playargs = playargs.overwrittenWith(self.playargs)
        parentOffset = self.parent.absoluteOffset() if self.parent else F(0)
        out = []
        for voice in self.voices:
            events = voice._synthEvents(playargs=playargs, workspace=workspace,
                                        parentOffset=parentOffset)
            out.extend(events)
        return out

    def __copy__(self):
        voices = [voice for voice in self.voices]
        return Score(voices=voices, scorestruct=self._scorestruct, title=self.label)

    def copy(self):
        return self.__deepcopy__()

    def __deepcopy__(self, memodict={}):
        voices = [voice for voice in self.voices]
        return Score(voices=voices.copy(), scorestruct=self._scorestruct, title=self.label)

    def clone(self,
              voices: list[Voice] = UNSET,
              scorestruct: ScoreStruct = UNSET,
              label: str = UNSET,
              ):
        return Score(voices=self.voices.copy() if voices is UNSET else voices,
                     scorestruct=self.scorestruct() if scorestruct is UNSET else scorestruct,
                     title=self.label if label is UNSET else label)

    def childOffset(self, child: MObj) -> F:
        offset = child._detachedOffset()
        return offset if offset is not None else F(0)

    def absoluteOffset(self) -> F:
        return F(0)

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Score:
        voices = [voice.pitchTransform(pitchmap) for voice in self.voices]
        return self.clone(voices=voices)