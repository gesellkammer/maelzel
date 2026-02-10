from __future__ import annotations

from maelzel.common import F, F0, asF
from .mobj import MObj, MContainer
from .event import MEvent
from .config import CoreConfig
from .chain import Voice, Chain, VoiceGroup
from .workspace import Workspace
from maelzel.scorestruct import ScoreStruct

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Iterator, Sequence, Callable
    from typing_extensions import Self
    from .synthevent import PlayArgs, SynthEvent
    from maelzel import scoring
    from maelzel.common import time_t


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

    __slots__ = ('voices', 'groups', '_modified', '_scorestruct', 'fusedParts')

    def __init__(self,
                 voices: Sequence[Voice | Chain | MEvent] = (),
                 scorestruct: ScoreStruct | str | None = None,
                 title=''):

        super().__init__(label=title, offset=F0)

        asvoices: list[Voice] = [item if isinstance(item, Voice) else _asvoice(item)
                                 for item in voices]
        for voice in asvoices:
            voice.parent = self

        self.voices: list[Voice] = asvoices
        """the voices of this score"""

        self.fusedParts: list[VoiceGroup] = []
        """Contains voices which have been fused to one part"""

        self.groups: set[VoiceGroup] = set()
        """Groups added via makeGroup are added here for reference"""

        self._scorestruct: ScoreStruct | None = None
        self._modified = True
        self._config: dict[str, Any] = {}
        self._dur = self._calculateDuration()
        if scorestruct:
            self.setScoreStruct(scorestruct)

    def setScoreStruct(self, scorestruct: ScoreStruct | str | None) -> None:
        """
        Set the ScoreStruct for this score and its children

        This ScoreStruct will be used for any object embedded
        downstream.

        Args:
            scorestruct: the ScoreStruct, or None to remove any scorestruct
                previously set

        """
        if isinstance(scorestruct, str):
            scorestruct = ScoreStruct(scorestruct)
        self._scorestruct = scorestruct
        self._changed()

    def scorestruct(self) -> ScoreStruct | None:
        """
        Returns the ScoreStruct for this score, if set

        .. seealso:: :meth:`activeScorestruct() <maelzel.core.mobj.MObj.activeScorestruct>`
        """
        return self._scorestruct


    def getConfig(self, prototype: CoreConfig | None = None) -> CoreConfig | None:
        if not self._config:
            return None
        return (prototype or Workspace.active.config).clone(self._config)

    def dump(self, indents=0, forcetext=False) -> None:
        self._update()
        for i, part in enumerate(self.voices):
            print("  "*indents + f"Voice #{i}, name='{part.name}'")
            part.dump(indents=indents+1, forcetext=forcetext)

    def _resolveGlissandi(self, force=False) -> None:
        for voice in self.voices:
            voice._resolveGlissandi(force=force)

    @staticmethod
    def read(path: str) -> Score:
        """
        Read a Score from musicxml, ...

        Args:
            path: the path to the file

        Returns:
            a Score

        .. seealso:: :meth:`Score.fromMusicxml`, :meth:`Score.write`

        """
        import os
        ext = os.path.splitext(path)[1].lower()
        if ext == '.xml' or ext == '.musicxml':
            xmltext = open(path).read()
            return Score.fromMusicxml(xmltext)
        else:
            raise ValueError(f"Format '{ext}' is not supported. At the moment only"
                             f" musicxml and MIDI are supported ")

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

    def remap(self, deststruct: ScoreStruct, sourcestruct: ScoreStruct | None = None,
              setStruct=True
              ) -> Score:
        """
        Creates a clone, remapping times from source scorestruct to destination scorestruct

        The absolute time remains the same

        Args:
            deststruct: the destination scorestruct
            sourcestruct: the source scorestructure, or None to use the resolved scoresturct
            setStruct: if True, explicitely sets deststruct as the score structure for
                this chain/voice

        Returns:
            a clone of self remapped to the destination scorestruct

        """
        voices = [v.remap(deststruct, sourcestruct) for v in self.voices]
        out = self.clone(voices=voices, scorestruct=deststruct)
        if setStruct:
            out.setScoreStruct(deststruct)
        return out

    def __len__(self):
        return len(self.voices)

    def index(self, child: Voice) -> int:
        return self.voices.index(child)

    @staticmethod
    def pack(*objects: MEvent | Chain | Voice | list[MEvent],
             group=True,
             maxrange=36,
             mingap: time_t = F0
             ) -> Score:
        """
        Pack the given objects into a Score

        Args:
            objects: a list of notes, chords, chains, etc. Voices are packed as is and
                not joined with other voices
            group: if True, any sequence of objects are kept within one staff, as long
                as they do not overlap
            maxrange: the max. pitch range for a voice
            mingap: a min. gap between items in a voice

        Returns:
            the packed Score
        """
        from maelzel import packing
        flatobjs = _flattenObjects(objects, group=group)
        voices = []
        items: list[packing.Item[MEvent|Chain]] = []
        for obj in flatobjs:
            if isinstance(obj, Voice):
                voices.append(obj)
            elif isinstance(obj, (MEvent, Chain)):
                pitch = obj.meanPitch()
                if pitch is None:
                    pitch = items[-1].step if items else 60
                item = packing.Item(obj, offset=float(obj.absOffset()), dur=float(obj.dur), step=pitch)
                items.append(item)
            else:
                raise TypeError(f"Cannot pack {obj}")
        tracks = packing.packInTracks(items, maxrange=maxrange, mingap=asF(mingap))
        if not tracks:
            raise ValueError("Cannot pack the given objects")
        for track in tracks:
            voice = Voice(track.unwrap())
            voices.append(voice)
        # Sort from high to low
        voices.sort(key=lambda voice: voice.meanPitch(), reverse=True)
        sco = Score(voices)
        return sco

    def __iter__(self) -> Iterator[Voice]:
        return iter(self.voices)

    def __getitem__(self, item):
        return self.voices.__getitem__(item)

    def __contains__(self, item) -> bool:
        return item in self.voices

    def addPart(self,
                voices: list[Voice],
                name='',
                abbrev='') -> None:
        """
        Group multiple voices within one part

        The voices will be confined to one staff when notated

        Args:
            voices: the voices to place together. A list of 2, 3 or 4 voices
            name: a name to use for the staf. If not given, a name constructed
                from the names of the given voices will be used
            abbrev: an optional abbreviation for the staf name
        """
        part = VoiceGroup(voices=voices, name=name, abbrev=abbrev)
        if part in self.fusedParts:
            oldpart = self.fusedParts[self.fusedParts.index(part)]
            raise ValueError(f"Part containing voices {voices} already exists: {oldpart}")
        self.fusedParts.append(part)
        newVoices = sum(1 for v in voices if v not in self.voices)
        if newVoices:
            if newVoices != len(voices):
                raise ValueError("Some voices within this part are already part of this score")
            self.voices.extend(voices)
            self._changed()

    def addGroup(self,
                 voices: list[Voice],
                 name: str = '',
                 abbrev: str = '',
                 showPartNames=False) -> None:
        """
        Create a group from a list of voices

        A group of voices can be created for notational purposes, to group those
        voices under one name, add a shortname to the group, etc.

        Args:
            name: the name of the group. It will be used when rendering as notation
            abbrev: a short name to use for all systems after the first one
            showPartNames: do not hide the names of the parts which form this group
        """
        voices = [voice if voice.parent is self else voice.copy()
                 for voice in voices]
        group = VoiceGroup(voices=voices, name=name, abbrev=abbrev, showPartNames=showPartNames)
        self.groups.add(group)
        newVoices = [v for v in voices if v not in self.voices]
        voiceIds = [id(v) for v in voices]
        if newVoices:
            self.voices.extend(newVoices)
            if len(newVoices) != len(voices):
                # Voices within a group need to be adjacent
                voices = []
                for v in self.voices:
                    if v is voices[0]:
                        voices.extend(voices)
                    elif id(v) not in voiceIds:
                        voices.append(v)
                self.voices = voices
                self._changed()

    def __hash__(self):
        items = [type(self).__name__, self.label, self.offset, len(self.voices)]
        if self.symbols:
            items.append(hash(tuple(self.symbols)))
        if self.voices:
            items.append(hash(tuple(self.voices)))
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

    def _calculateDuration(self) -> F:
        return max(v.dur for v in self.voices) if self.voices else F0

    def _update(self):
        if not self._modified:
            return
        self._dur = self._calculateDuration()
        self._modified = False

    def append(self, voice: Voice | Chain) -> None:
        """Append a Voice to this Score"""
        if isinstance(voice, Chain):
            voice = voice.asVoice()
        voice.parent = self
        self.voices.append(voice)
        if not self._modified:
            self._changed()

    @property
    def dur(self) -> F:
        """The duration of this object"""
        if self._modified:
            self._update()
        return self._dur

    def scoringParts(self, config: CoreConfig | None = None
                     ) -> list[scoring.core.UnquantizedPart]:
        self._update()
        parts = []
        config, iscustom = self._resolveConfig(config, forceCopy=True)
        config['show.voiceMaxStaves'] = 1
        for voice in self.voices:
            voiceparts = voice.scoringParts(config=config)
            parts.extend(voiceparts)
        return parts

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig | None = None,
                      parentOffset: F | None = None
                      ) -> list[scoring.Notation]:
        parts = self.scoringParts(config or Workspace.active.config)
        flatevents = []
        for part in parts:
            flatevents.extend(part)
        # TODO: deal with groupid
        return flatevents

    def _asVoices(self) -> list[Voice]:
        return self.voices

    def _synthEvents(self,
                     playargs: PlayArgs,
                     parentOffset: F,
                     workspace: Workspace
                     ) -> list[SynthEvent]:
        if self.playargs:
            playargs = playargs.updated(self.playargs)
        out = []
        for voice in self.voices:
            events = voice._synthEvents(playargs=playargs, workspace=workspace,
                                        parentOffset=F0)
            out.extend(events)
        return out

    def __copy__(self) -> Self:
        voices = [voice for voice in self.voices]
        return self.__class__(voices=voices, scorestruct=self._scorestruct, title=self.label)

    def copy(self):
        return self.__deepcopy__()

    def __deepcopy__(self, memodict={}) -> Self:
        voices = [voice for voice in self.voices]
        return self.__class__(voices=voices.copy(), scorestruct=self._scorestruct, title=self.label)

    def clone(self,
              voices: list[Voice] | None = None,
              scorestruct: ScoreStruct | None = None,
              label='',
              ) -> Self:
        return self.__class__(voices=self.voices.copy() if voices is None else voices,
                              scorestruct=self.scorestruct() if scorestruct is None else scorestruct,
                              title=label or self.label)

    def _childOffset(self, child: MObj) -> F:
        offset = child._detachedOffset()
        return offset if offset is not None else F0

    def absOffset(self) -> F:
        return F0

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Self:
        voices = [voice.pitchTransform(pitchmap) for voice in self.voices]
        return self.clone(voices=voices)

    def isPolymetric(self) -> bool:
        """
        True if this score has multiple meters (simultaneously) at any moment

        Multiple simultaneous tempi are also considered to make a score polymetric
        """
        if len(self.voices) < 2:
            return False

        structs = [part.activeScorestruct() for part in self.voices]
        uniquestructs = set(structs)
        if len(uniquestructs) < 2:
            return False

        struct0 = uniquestructs.pop()
        for i, mdef in enumerate(struct0.measures):
            for struct in uniquestructs:
                mdef2 = struct.measure(i)
                if mdef.timesig != mdef2.timesig or mdef.quarterTempo != mdef.quarterTempo:
                    return True
        return False

    def numMeasures(self) -> int:
        """
        Number of measures needed to encompass this score

        Returns:
            the number of measures in this voice
        """
        struct = self.activeScorestruct()
        end = self.dur
        endidx, endbeat = struct.beatToLocation(end)
        return endidx + 1


def show(*objs: MObj | Sequence[MObj], group=True, **kws) -> Score:
    """
    Packs all objects into a score and displays them as notation

    Args:
        objs: objects to pack. Either single Notes, Chains, etc. List
            of events can be given and will be kept together if group=True
        group: if True, objects grouped within a list are kept within a
            staff if they do not overlap
        kws: any keyword is passed to the :meth:`~MObj.show` method

    Returns:
        the resulting Score
    """
    sco = Score.pack(*objs, group=group)
    sco.show(**kws)
    return sco


def _flattenObjects(objs: Sequence[MObj | Sequence[MEvent]], group=True) -> list[MObj]:
    flatobjs = []
    for obj in objs:
        if isinstance(obj, MObj):
            flatobjs.append(obj)
        elif isinstance(obj, (tuple, list)):
            if group and not _objectsOverlap(obj):
                assert all(isinstance(item, MEvent) for item in obj)
                chain = Chain(obj)
                flatobjs.append(chain)
            else:
                flatobjs.extend(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} not supported ({obj})")
    return flatobjs


def _objectsOverlap(objs: Sequence[MObj]) -> bool:
    now = F0
    for obj in objs:
        if obj.offset is not None:
            if obj.offset < now:
                return True
            now = obj.offset
        now += obj.dur
    return False