from __future__ import annotations

from maelzel.common import F, F0
from .mobj import MObj, MContainer
from .event import MEvent
from .config import CoreConfig
from .chain import Voice, Chain, PartGroup
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
    'show'
)


def _asvoice(o: MObj):
    if isinstance(o, MEvent):
        return Voice([o])
    elif isinstance(o, Chain):
        return o.asVoice()
    else:
        raise TypeError(f"Cannot create a Voice from {o} (type: {type(o)})")


class Score(MContainer):
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

        super().__init__(label=title, offset=F0)

        self._scorestruct = scorestruct
        self._modified = True

    def dump(self, indents=0, forcetext=False) -> None:
        for i, part in enumerate(self.voices):
            print("  "*indents + f"Voice #{i}, name='{part.name}'")
            part.dump(indents=indents+1, forcetext=forcetext)

    def _resolveGlissandi(self, force=False) -> None:
        for voice in self.voices:
            voice._resolveGlissandi(force=force)

    @staticmethod
    def read(path: str) -> Score:
        """
        Read a Score from musicxml, MIDI, ...

        Args:
            path: the path to the file

        Returns:
            a Score

        .. seealso:: :meth:`Score.fromMusicxml`, :meth:`Score.fromMIDI`, :meth:`Score.write`

        """
        import os
        ext = os.path.splitext(path)[1].lower()
        if ext == '.xml' or ext == '.musicxml':
            xmltext = open(path).read()
            return Score.fromMusicxml(xmltext)
        elif ext == '.mid' or ext == '.midi':
            return Score.fromMIDI(path)
        else:
            raise ValueError(f"Format '{ext}' is not supported. At the moment only"
                             f" musicxml and MIDI are supported ")

    @staticmethod
    def fromMIDI(midifile: str) -> Score:
        """
        Parse a MIDI file, returns a :class:`Score`

        Args:
            midifile: the midi file to parse

        Returns:
            the correponding Score.

        .. seealso:: :meth:`Score.fromMusicxml`, :meth:`Score.read`, :meth:`Score.write`
        """
        return Score()

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

        .. seealso:: :meth:`Score.fromMIDI`, :meth:`Score.read`, :meth:`Score.write`
        """
        from maelzel.core import musicxmlparser as mxml
        return mxml.parseMusicxml(musicxml, enforceParsedSpelling=enforceParsedSpelling)

    @staticmethod
    def pack(objects: list[MEvent | Chain | Voice], maxrange=36, mingap=0) -> Score:
        """
        Pack the given objects into a Score

        Args:
            objects: a list of notes, chords, chains, etc. Voices are packed as is and not joined
                with other voices
            maxrange: the max. pitch range for a voice
            mingap: a min. gap between items in a voice

        Returns:
            the packed Score
        """
        from maelzel import packing
        voices = []
        items = []
        for obj in objects:
            if isinstance(obj, Voice):
                voices.append(obj)
            elif isinstance(obj, (MEvent, Chain)):
                items.append(packing.Item(obj, offset=obj.absOffset(), dur=obj.dur,
                                          step=obj.meanPitch()))
            else:
                raise TypeError(f"Cannot pack {obj}")
        tracks = packing.packInTracks(items, maxrange=maxrange, mingap=mingap)
        if not tracks:
            raise ValueError("Cannot pack the given objects")
        for track in tracks:
            voice = Voice(track.unwrap())
            voices.append(voice)
        # Sort from high to low
        voices.sort(key=lambda voice: voice.meanPitch(), reverse=True)
        sco = Score(voices)
        return sco

    def __getitem__(self, item):
        return self.voices.__getitem__(item)

    def scorestruct(self, resolve=False) -> ScoreStruct | None:
        """The attached ScoreStruct, if present"""
        if self._scorestruct:
            return self._scorestruct
        elif resolve:
            return Workspace.active.scorestruct
        else:
            return None

    def setScoreStruct(self, scorestruct: ScoreStruct) -> None:
        """
        Set the ScoreStruct for this Score

        Scores are the only objects in `maelzel.core` which can have a
        ScoreStruct attached to them. This ScoreStruct will be
        used for any object embedded downstream

        Args:
            scorestruct: the ScoreStruct

        """
        self._scorestruct = scorestruct
        self._changed()

    def makeGroup(self,
                  parts: list[Voice],
                  name: str = '',
                  shortname: str = '',
                  showPartNames=False):
        for part in parts:
            if part.parent and part.parent is not self:
                raise RuntimeError(f"Cannot make a group with a part which belongs to another"
                                   f" score (part={part}, parent={part.parent})")
        PartGroup(parts=parts, name=name, shortname=shortname, showPartNames=showPartNames)
        for part in parts:
            if not any(v is part for v in self.voices):
                raise RuntimeError(f"Parts can only be bundled into a group if they are already "
                                   f"part of this Score, but {part} is not")

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
        self._dur = max(v.dur for v in self.voices) if self.voices else F0
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
        """The duration of this object"""
        if self._modified:
            self._update()
        assert self._dur is not None
        return self._dur

    def scoringParts(self, config: CoreConfig | None = None
                     ) -> list[scoring.UnquantizedPart]:
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
            playargs = playargs.updated(self.playargs)
        parentOffset = self.parent.absOffset() if self.parent else F0
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
              voices: list[Voice] = None,
              scorestruct: ScoreStruct = None,
              label: str = None,
              ):
        return Score(voices=self.voices.copy() if voices is None else voices,
                     scorestruct=self.scorestruct() if scorestruct is None else scorestruct,
                     title=self.label if label is None else label)

    def childOffset(self, child: MObj) -> F:
        offset = child._detachedOffset()
        return offset if offset is not None else F0

    def absOffset(self) -> F:
        return F0

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Score:
        voices = [voice.pitchTransform(pitchmap) for voice in self.voices]
        return self.clone(voices=voices)


def show(*objs: MObj | list[MObj], **kws) -> None:
    flatobjs = []
    for obj in objs:
        if isinstance(obj, MObj):
            flatobjs.append(obj)
        elif isinstance(obj, (tuple, list)):
            flatobjs.extend(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} not supported ({obj})")
    from maelzel.core import Score
    sco = Score.pack(flatobjs)
    sco.show(**kws)