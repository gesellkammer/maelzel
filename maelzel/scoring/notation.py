"""
A Notation represents a note/chord/rest
"""
from __future__ import annotations
from dataclasses import dataclass
import uuid

from .common import *
from .util import *
from .attachment import *
from emlib.iterlib import first
from . import definitions
from . import spanner as _spanner
import pitchtools as pt
import copy


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    import maelzel.core

__all__ = (
    'Notation',
    'makeNote',
    'makeChord',
    'makeRest',
    'makeGroupId',
    'mergeNotations',
    'notationsToCoreEvents',
    'notationsCanMerge',
    'notationsCannotMerge',
    'durationsCanMerge',
    'mergeNotationsIfPossible'
)


_UNSET = object()


class Notation:
    """
    This represents a notation (a rest, a note or a chord)

    Args:
        duration: the duration of this Notation, in quarter-notes. A value of
            None indicates an unset duration. During quantization an unset
            duration is interpreted as lasting to the next notation.
            0 indicates a grace note
        pitches: if given, a list of pitches as midinote or notename. If a notename
            is given, the spelling is fixed. Otherwise a suitable spelling is calculated
            based on the context of this notation.
        offset: the offset of this Notation, in quarter-notes.
        isRest: is this a rest?
        tiedPrev: is this Notation tied to the previous one?
        tiedNext: is it tied to the next
        dynamic: the dynamic of this notation, one of "p", "pp", "f", etc.
        group: a str identification, can be used to group Notations together
        durRatios: a list of tuples (x, y) indicating a tuple relationship.
            For example, a Notation used to represent one 8th note in a triplet
            would have the duration 1/3 and durRatios=[(3, 2)]. Multiplying the
            duration by the durRatios would result in the notated value, in this
            case 1/2 (1 being a quarter note). A value of None is the same as
            a value of [(1, 1)] (no modification)
        gliss: if True, a glissando will be rendered between this note and the next
        color: the color of this notations
        stem: if given, one of

    """
    __slots__ = ("duration",
                 "pitches",
                 "offset",
                 "isRest",
                 "tiedPrev",
                 "tiedNext",
                 "dynamic",
                 "durRatios",
                 "groupid",
                 "gliss",
                 "noteheads",
                 "color",
                 "stem",
                 "properties",
                 "fixedNotenames",
                 "sizeFactor",
                 "spanners",
                 "attachments",
                 "accidentalTraits"
                 )

    def __init__(self,
                 duration: time_t,
                 pitches: list[pitch_t] = None,
                 offset: time_t = None,
                 isRest=False,
                 tiedPrev=False,
                 tiedNext=False,
                 dynamic: str = '',
                 durRatios: list[F] = None,
                 group='',
                 gliss: bool = None,
                 color='',
                 stem='',
                 sizeFactor=1,  # size is relative: 0 is normal, +1 is bigger, -1 is smaller
                 properties: dict[str, Any] = None,
                 _init=True
                 ):

        assert duration is not None

        if _init:
            if dynamic:
                dynamic = definitions.normalizeDynamic(dynamic, '')
            duration = asF(duration)
            if pitches:
                pitches = [asmidi(p) for p in pitches]
            if offset is not None:
                offset = asF(offset)
            if isRest:
                tiedNext = False
                tiedPrev = False

            assert not stem or stem in definitions.stemTypes, \
                f"Stem types: {definitions.stemTypes}"

        self.duration: F = duration
        self.pitches: list[float] = pitches
        self.offset: F | None = offset
        self.isRest = isRest
        self.tiedNext = tiedNext
        self.tiedPrev = tiedPrev
        self.dynamic = dynamic
        self.durRatios = durRatios
        self.groupid = group
        self.gliss = gliss
        self.noteheads: dict[int, definitions.Notehead] | None = None
        self.color = color
        self.stem = stem
        self.sizeFactor = sizeFactor
        self.properties: dict[str, Any] | None = properties
        self.fixedNotenames: dict[int, str] | None = None
        self.attachments: list[Attachment] = []
        self.spanners: list[_spanner.Spanner] | None = None

        if self.isRest:
            assert self.duration > 0
            assert not self.pitches or (len(self.pitches) == 1 and self.pitches[0] == 0)
        else:
            assert self.pitches and all(p > 0 for p in self.pitches)
            for i, n in enumerate(pitches):
                if isinstance(n, str):
                    self.fixNotename(n, i)

    def getAttachments(self,
                       cls: str | type = '',
                       predicate: Callable = None,
                       anchor: int = _UNSET
                       ) -> list[Attachment]:
        """
        Get a list of Attachments matching the given criteria

        Args:
            cls: the class to match (the class itself or its name, case is not relevant)
            predicate: a function (attachment) -> bool
            anchor: if given, the anchor index to match. Some attachments are anchored to
                a specific component (pitch) in the notation (for example a notehead or
                an accidental trait are attached to a specific pitch of the chord)

        Returns:
            the list of attachments matching the given criteria
        """
        attachments = self.attachments
        if cls:
            if isinstance(cls, str):
                cls = cls.lower()
                attachments = [a for a in attachments
                               if type(a).__name__.lower() == cls]
            else:
                attachments = [a for a in attachments if isinstance(a, cls)]
        if predicate:
            attachments = [a for a in attachments if predicate(a)]

        if anchor is not _UNSET:
            attachments = [a for a in attachments if a.anchor == anchor]

        return attachments

    def findSpanner(self, cls: str  | type, kind='') -> Optional[_spanner.Spanner]:
        if not self.spanners:
            return
        if isinstance(cls, str):
            clsname = cls.lower()
            return first(s for s in self.spanners
                         if type(s).__name__.lower() == clsname and (not kind or s.kind==kind))
        else:
            return first(s for s in self.spanners
                         if isinstance(s, cls) and (not kind or s.kind==kind))

    def findAttachment(self,
                       cls: str | type,
                       anchor: int | None = _UNSET,
                       predicate: Callable = None
                       ) -> Attachment | None:
        """
        Find an attachment by class or classname

        Similar to getAttachments, returns only one attachment or None
        Args:
            cls: the class to match (the class itself or its name, case is not relevant)
            predicate: a function (attachment) -> bool
            anchor: if given, the anchor index to match. Some attachments are anchored to
                a specific component (pitch) in the notation (for example a notehead or
                an accidental trait are attached to a specific pitch of the chord)

        Returns:
            an Attachment matching the given criteria, or None
        """
        attachments = self.getAttachments(cls=cls, anchor=anchor, predicate=predicate)
        if attachments:
            return attachments[0]

    def addAttachment(self, attachment: Attachment) -> Notation:
        """
        Add an attachment to this Notation

        An attachment is any kind of note attached symbol or text expression
        (a Fermata, a Text, an Articulation, an Ornament, etc.). To add a spanner
        (a slur, a bracket, etc.) see addSpanner.

        .. note::
            Some kinds of attachments are **exclusive**. Adding an exclusive
            attachment (like a fermata) will remove any previous such attachment.

        Args:
            attachment: an instance of scoring.attachment.Attachment

        Returns:
            self

        """
        if self.attachments is None:
            self.attachments = []
        if attachment.exclusive:
            cls = type(attachment)
            if any(isinstance(a, cls) for a in self.attachments):
                self.attachments = [a for a in self.attachments
                                    if not isinstance(a, cls)]
        if attachment.anchor is not None:
            assert 0 <= attachment.anchor < len(self.pitches)
        self.attachments.append(attachment)
        return self

    def setNotehead(self,
                    notehead: definitions.Notehead | str,
                    idx: int | None = None,
                    merge=False) -> None:
        """
        Set a notehead in this notation

        Args:
            notehead: a Notehead
            idx: the index, corresponding to the pitch at the same index,
                or None to set all noteheads

        """
        if self.noteheads is None:
            self.noteheads = {}

        if isinstance(notehead, str):
            notehead = definitions.Notehead(shape=notehead)

        if idx is not None:
            if not(0 <= idx < len(self.pitches)):
                raise IndexError(f'Index {idx} out of range. This notation has {len(self.pitches)} '
                                 f'pitches: {self.pitches}')
            indexes = [idx]
        else:
            indexes = range(len(self.pitches))

        for i in indexes:
            if merge and (oldnotehead := self.noteheads.get(i)) is not None:
                oldnotehead.update(notehead)
            else:
                self.noteheads[i] = notehead.copy()

    def addArticulation(self, articulation: str | Articulation) -> Notation:
        """
        Add an articulation to this Notation.

        See ``definitions.articulations`` for possible values. We understand
        articulation in a broad sense as any symbol attached to a note/chord

        Args:
            articulation: an Articulation object, or one of accent, staccato, tenuto,
                marcato, staccatissimo, espressivo, portato, arpeggio, upbow,
                downbow, flageolet, open, closed, stopped, snappizz

        Returns:
            self
        """
        if isinstance(articulation, str):
            articulation = definitions.normalizeArticulation(articulation)
            if not articulation:
                raise ValueError(f"Articulation {articulation} unknown. "
                                 f"Possible values: {definitions.articulations}")
            articulation = Articulation(articulation)
        assert isinstance(articulation, Articulation)
        return self.addAttachment(articulation)

    def removeAttachments(self, predicate: Callable[Attachment, bool]) -> None:
        """
        Remove attachments where predicate is  True

        Args:
            predicate: a function taking an Attachment, returns True if it should
                be removed

        """
        self.attachments[:] = [a for a in self.attachments
                               if not(predicate(a))]

    def removeAttachmentsByClass(self, cls: str | type) -> None:
        if isinstance(cls, str):
            cls = cls.lower()
            predicate = lambda a, cls=cls: type(a).__name__.lower() == cls
        else:
            predicate = lambda a, cls: isinstance(a, cls)
        self.removeAttachments(predicate=predicate)

    def addSpanner(self, spanner: _spanner.Spanner, end: Notation = None) -> Notation:
        """
        Add a Spanner to this Notation

        Spanners always are bound in pairs. A 'start' spanner is attached to
        one Notation, an 'end' spanner with the same uuid is attached to
        the notation where the spanner should end (see :meth:`Spanner.bind`)

        Args:
            spanner: the spanner to add.

        Returns:
            self

        """
        if self.spanners is None:
            self.spanners = []
        self.spanners.append(spanner)
        self.spanners.sort(key=lambda spanner: spanner.priority())
        if end:
            end.addSpanner(spanner.endSpanner())
        return self

    @classmethod
    def makeArtificialHarmonic(cls,
                               basepitch: pitch_t,
                               interval: int,
                               **kws
                               ) -> Notation:
        if not isinstance(basepitch, str):
            basepitch = pt.m2n(basepitch)
        touchpitch = pt.transpose(basepitch, interval)
        n = cls(pitches=[basepitch, touchpitch], **kws)
        n.fixNotename(basepitch, 0)
        n.fixNotename(touchpitch, 1)
        n.setNotehead(definitions.Notehead('harmonic'), 1)
        return n

    def fixNotename(self, notename: str, idx: int | None = None, fail=True) -> None:
        """
        Fix the spelling for the pitch at index **in place**

        Args:
            notename: if given, it will be fixed to the given notename.
                If nothing is given, it will be fixed to n2m(self.pitches[idx])
                Alternatively 'enharmonic' can be given as notename, in which
                case the enharmonic variant of the current notename will be used
            idx: the index of the note to modify. If None, a matching pitch in this notation
                is searched
            fail: if idx was set to None (to search for a matching fit) and there
                is not match, an Exception will be raised if fail is set to True.
                Otherwise we fail silently.

        .. seealso:: :meth:`Notation.notenames`
        """
        if self.fixedNotenames is None:
            self.fixedNotenames = {}

        if notename == 'enharmonic':
            notename = pt.enharmonic(self.notename(idx))

        if idx is None:
            spellingPitch = pt.n2m(notename)
            idx = next((idx for idx in range(len(self.pitches))
                        if abs(spellingPitch - self.pitches[idx]) < 0.04), None)
            if idx is None:
                if fail:
                    raise ValueError(f"No pitch in this notation matches the given notename {notename}"
                                     f" (pitches: {self.notenames})")
                return

        self.fixedNotenames[idx] = notename

    def getFixedNotename(self, idx: int = 0) -> Optional[str]:
        """
        Returns the fixed notename of this notation, if any

        Args:
            idx: 0 in the case of a note, the index of the note if representing a chord

        Returns:
            the fixed spelling of the note, if exists (None otherwise)

        """
        return self.fixedNotenames.get(idx) if self.fixedNotenames else None

    def fixedSlots(self, semitoneDivs=2) -> dict[int, int] | None:
        """
        Calculate the fixed slots within this chord

        Args:
            semitoneDivs: the number of divisions of the semitone.

        Returns:
            a dict mapping slot to alteration direction
        """
        print("here!", self)
        if not self.fixedNotenames:
            return None
        fixedSlots = {}
        for notename in self.fixedNotenames.values():
            notated = pt.notated_pitch(notename)
            slot = notated.microtone_index(divs_per_semitone=semitoneDivs)
            fixedSlots[slot] = notated.alteration_direction(min_alteration=0.5)
        return fixedSlots

    @property
    def isGraceNote(self) -> bool:
        return not self.isRest and self.duration == 0

    def meanPitch(self) -> float:
        """
        The mean pitch of this note/chord

        This is provided to have a generalized way of quering the pitch
        of a note/chord for packing

        Returns:
            the pitchof this note or the avg. pitch if it is a chord. Rests
            do not have a mean pitch and calling this on a rest will raise
            ValueError
        """
        L = len(self.pitches)
        if self.isRest or L == 0:
            raise ValueError("No pitches to calculate mean")
        return self.pitches[0] if L == 1 else sum(self.pitches) / L

    @property
    def end(self) -> Optional[F]:
        """
        The end time of this notation (if set)
        """
        if self.duration is not None and self.offset is not None:
            return self.offset + self.duration
        return None

    def _setPitches(self, pitches: list[pitch_t], resetFixedNotenames=True) -> None:
        if len(pitches) != self.pitches:
            if self.noteheads:
                logger.info("Notation: setting new pitches in place. Noteheads will be reset")
                self.noteheads = {}
        self.pitches = [asmidi(p) for p in pitches] if pitches else []
        if resetFixedNotenames:
            self.fixedNotenames = None

    def transferFixedSpellingTo(self, other: Notation):
        if not self.fixedNotenames:
            return
        for notename in self.fixedNotenames.values():
            other.fixNotename(notename, None, fail=False)

    def clone(self, **kws) -> Notation:
        """
        Clone this Notation, overriding any value.

        Args:
            kws: keyword arguments, as passed to the Notation constructor.
                Any parameter given will override the corresponding value in
                this Notation
        """
        noteheads = kws.get('noteheads')
        if noteheads:
            assert isinstance(noteheads, dict), f'{self=}, {noteheads=}'

        out = self.copy()
        pitches = kws.pop('pitches', None)
        if pitches:
            out._setPitches(pitches)
            self.transferFixedSpellingTo(out)
        for key, value in kws.items():
            setattr(out, key, value)
        return out

    def copy(self) -> Notation:
        """
        Copy this Notation as is
        """

        return self.__copy__()

    def __deepcopy__(self, memo=None):
        return self.__copy__()

    def __copy__(self) -> Notation:
        # return copy.deepcopy(self)
        out = Notation(duration=self.duration,
                       pitches=self.pitches.copy() if self.pitches else None,
                       offset=self.offset,
                       isRest=self.isRest,
                       tiedPrev=self.tiedPrev,
                       tiedNext=self.tiedNext,
                       dynamic=self.dynamic,
                       durRatios=self.durRatios.copy() if self.durRatios else None,
                       group=self.groupid,
                       gliss=self.gliss,
                       color=self.color,
                       stem=self.stem,
                       sizeFactor=self.sizeFactor,
                       properties=self.properties.copy() if self.properties else None,
                       _init=False)
        if self.attachments:
            out.attachments = self.attachments.copy()
        if self.fixedNotenames:
            out.fixedNotenames = self.fixedNotenames.copy()
        if self.spanners:
            out.spanners = self.spanners.copy()
        if self.noteheads:
            out.noteheads = self.noteheads.copy()
        return out

    def symbolicDuration(self) -> F:
        """
        The symbolic duration of this Notation.

        This represents the notated figure (1=quarter, 1/2=eighth note,
        1/4=16th note, etc)
        """
        dur = self.duration
        if self.durRatios:
            for durRatio in self.durRatios:
                dur *= durRatio
        return dur

    def notename(self, index=0, addExplicitMark=False) -> str:
        """
        Returns the notename corresponding to the given pitch index

        If there is a fixed notename for the pitch, that will be returned; otherwise
        the notename corresponding to the pitch

        Args:
            index: the index of the pitch (in self.pitches)
            addExplicitMark: if True, the notename is suffixed with a '!' sign if
                the spelling has been fixed
        Returns:
            the notename corresponing to the given pitch

        """
        if index < 0:
            index = len(self.pitches) + index
        assert 0 <= index < len(self.pitches), f"Invalid index {index}, num. pitches={len(self.pitches)}"
        if fixed := self.getFixedNotename(index):
            return fixed if not addExplicitMark else fixed+'!'
        return pt.m2n(self.pitches[index])

    def pitchIndex(self, semitoneDivs=2, index=0) -> int:
        """
        The index of the nearest pitch/microtone

        For example, if divs_per_semitone is 2, then

        ====   ================
        note   microtone index
        ====   ================
        4C     0
        5C     0
        4C+    1
        4C#    2
        4Db    2
        …      …
        ====   ================
        """
        notename = self.notename(index=index)
        if semitoneDivs == 1:
            return pt.notated_pitch(notename).chromatic_index
        return pt.notated_pitch(notename).microtone_index(divs_per_semitone=semitoneDivs)

    @property
    def notenames(self) -> list[str]:
        return [self.getFixedNotename(i) or pt.m2n(p) for i, p in enumerate(self.pitches)]

    def verticalPosition(self, index=0) -> int:
        """
        The vertical position of the notated note at given index

        The vertical position is the position within the staff in terms of
        lines/spaces. It is calculated as octave*7 + diatonic_index

        =====   ===================
        Note     Vertical Position
        =====   ===================
        4C       28
        4C#      28
        4D       29
        4Eb      30
        ...      ...
        =====   ===================

        Args:
            index: the index of the pitch within this notation

        Returns:
            the vertical position

        """
        return pt.vertical_position(self.notename(index))

    def addText(self,
                text: Union[str, Text],
                placement='above',
                fontsize: int|float = None,
                fontstyle: str = None,
                box: str|bool = False
                ) -> None:
        """
        Add a text annotation to this Notation.

        Args:
            text: the text of the annotation, or a Text object itself
                If passed a Text objecj, all other parameters will not be
                considered
            placement: where to place the text annotation, one of 'above' or 'below'
            fontsize: the size of the font
            box: if True, the text is enclosed in a box. A string indicates the shape
                of the box
        """
        if isinstance(text, Text):
            assert text.text.strip()
            annotation = text
        else:
            assert not text.isspace()
            annotation = Text(text=text, placement=placement, fontsize=fontsize,
                              fontstyle=fontstyle, box=box)
        self.addAttachment(annotation)

    def notatedDuration(self) -> NotatedDuration:
        """
        The duration of the notated figure, in quarter-notes, independent of any tuples.

        A quarter-note inside a triplet would have a notatedDuration of 1
        """
        return notatedDuration(self.duration, self.durRatios)

    def canMergeWith(self, other: Notation) -> bool:
        return notationsCanMerge(self, other)
    
    def mergeWith(self, other: Notation) -> Notation:
        """Merge this Notation with ``other``"""
        return mergeNotations(self, other)

    def setProperty(self, key: str, value = _UNSET) -> None:
        """
        Set any property of this Notation.

        Properties can be used, for example, for any rendering backend to
        pass directives which are specific to that rendering backend.
        """
        if self.properties is None:
            self.properties = {}
        if value is not _UNSET:
            self.properties[key] = value
        elif self.properties:
            del self.properties[key]

    def getProperty(self, key: str, default=None) -> Any:
        """
        Get the value of a property. If the key is not found, return ``default``
        """
        if not self.properties:
            return default
        return self.properties.get(key, default)

    def setClefHint(self, clef: str, idx: int = None) -> None:
        """
        Sets a hint regarding which clef to use for this notation

        .. warning::

            This is mostly used internally and is an experimental feature.
            It is conceived for the case where two notations
            are bound by a glissando and they should be placed together,
            even if the pitch of some of them might indicate otherwise

        Args:
            clef: the clef to set, one of 'g', 'f' or '15a'
            idx: the index of the pitch within a chord, or None to apply to
                the whole notation

        """
        if idx is None:
            self.setProperty('.clefHint', clef)
        else:
            hint = self.getProperty('.clefHint', {})
            hint[idx] = clef
            self.setProperty('.clefHint', hint)

    def clearClefHints(self) -> None:
        self.setProperty('.clefHint')

    def getClefHint(self, idx: int = 0) -> Optional[str]:
        """
        Get any clef hint for this notation or a particular pitch thereof

        .. warning::

            This is mostly used internally and is an experimental feature which
            might be implemented using other means in the future

        Args:
            idx: which pitch index to query

        Returns:
            the clef hint, if any

        """
        hints = self.getProperty('.clefHint')
        if not hints:
            return None
        elif isinstance(hints, str):
            return hints
        else:
            return hints.get(idx)

    def __repr__(self):
        info = []
        if self.offset is None:
            info.append(f"None, dur={showT(self.duration)}")
        elif self.duration == 0:
            info.append(f"{showT(self.offset)}:grace")
        else:
            info.append(f"{showT(self.offset)}:{showT(self.end)}")

        if self.durRatios and self.durRatios != [F(1)]:
            info.append(",".join(showF(r) for r in self.durRatios))

        if self.isRest:
            info.append("rest")
        elif self.pitches:
            if len(self.pitches) > 1:
                info.append("[" + " ".join(self.notenames) + "]")
            else:
                info.append(self.notenames[0])
            if self.gliss:
                info.append("gliss")

        if self.tiedPrev:
            info.append("tiedPrev")
        if self.tiedNext:
            info.append("tiedNext")
        if self.dynamic:
            info.append(self.dynamic)
        if self.noteheads:
            descrs = [f'{i}:{n.description()}' for i, n in self.noteheads.items()]
            info.append(f'noteheads={descrs}')

        for attr in ('attachments', 'properties', 'spanners', 'color'):
            val = getattr(self, attr)
            if val:
                info.append(f"{attr}={val}")

        infostr = " ".join(info)
        return f"«{infostr}»"

    def transferAttributesTo(self: Notation, dest: Notation) -> None:
        """
        Copy attributes of self to dest
        """
        exclude = {'duration', 'pitches', 'offset', 'durRatios', 'group',
                   'properties'}

        for attr in self.__slots__:
            if attr not in exclude:
                setattr(dest, attr, getattr(self, attr))

        if self.properties:
            for prop, value in self.properties.items():
                dest.setProperty(prop, value)

        if self.attachments:
            for a in self.attachments:
                dest.addAttachment(a)

    def __len__(self) -> int:
        return len(self.pitches)

    def accidentalDirection(self, index=0, minAlteration=0.5) -> int:
        """
        Returns the direction of the alteration in this notation

        Args:
            index: index of the pitch within this Notation
            minAlteration: threshold (with minAlteration 0.5
                C+ gets a direction of +1, whereas C+25 still gets a direction
                of 0

        Returns:
            one of -1, 0 or +1, corresponding to the direction of the alteration
            (flat, natural or sharp)
        """
        n = self.notename(index=index)
        notated = pt.notated_pitch(n)
        return notated.alteration_direction(min_alteration=minAlteration)

    def extractPartialNotation(self, indexes: list[int]) -> Notation:
        """
        Extract part of a chord with any attachments corresponding to the given pitches

        Args:
            indexes: the indexes of the pitches to extract

        Returns:
            a new Notation with the given pitches
        """
        indexes.sort()
        pitches = [self.pitches[index] for index in indexes]
        mappedIndexes = {index: indexes.index(index) for index in indexes}
        if self.noteheads:
            noteheads = {}
            for index in indexes:
                if (notehead := self.noteheads.get(index)) is not None:
                    noteheads[mappedIndexes[index]] = notehead
            if not noteheads:
                noteheads = None

        else:
            noteheads = None
        attachments = []
        for a in self.attachments:
            if a.anchor in indexes:
                a = copy.copy(a)
                a.anchor = mappedIndexes[a.anchor]
                attachments.append(a)
            elif a.anchor is None:
                attachments.append(a)

        out = self.clone(pitches=pitches,
                         noteheads=noteheads)
        out.attachments = attachments
        self.transferFixedSpellingTo(out)
        out.clearClefHints()
        for idx in indexes:
            if hint := self.getClefHint(idx):
                out.setClefHint(hint, mappedIndexes[idx])
        return out


def mergeNotations(a: Notation, b: Notation) -> Notation:
    """
    Merge two compatible notations to one.

    For two notations to be mergeable they need to:

    - be adjacent or have unset offset
    - have a duration
    - have the same pitch/pitches.

    All other attributes are taken from the first notation and the
    duration of this first notation is extended to cover both notations
    """
    if a.pitches != b.pitches:
        raise ValueError("Attempting to merge two Notations with "
                         "different pitches")
    assert a.duration is not None and b.duration is not None
    assert b.offset is None or (a.end == b.offset)
    out = a.clone(duration=a.duration + b.duration,
                  tiedNext=b.tiedNext)
    if b.fixedNotenames:
        b.transferFixedSpellingTo(out)
    return out


def makeGroupId(parent: str | None = None) -> str:
    """
    Create an id to group notations together

    Returns:
        the group id as string
    """
    subgroup = str(uuid.uuid1())
    if parent is None:
        return subgroup
    assert isinstance(parent, str), f"Expected a str, got {parent}"
    return parent + "/" + subgroup


def makeNote(pitch: pitch_t,
             duration: time_t = None,
             offset: time_t = None,
             annotation: str = None,
             gliss=False,
             withId=False,
             gracenote=False,
             enharmonicSpelling: str = None,
             dynamic: str = '',
             **kws
             ) -> Notation:
    """
    Utility function to create a note Notation

    Args:
        pitch: the pitch as midinote or notename. If given a pitch as str,
            the note in question is fixed at the given enharmonic representation.
        duration: the duration of this Notation. Use None to leave this unset,
            0 creates a grace note
        offset: the offset of this Notation (None to leave unset)
        annotation: an optional text annotation for this note
        gliss: does this Notation start a glissando?
        withId: if True, this Notation has a group id and this id
            can be used to mark multiple notes as belonging to a same group
        gracenote: make this a grace note.
        enharmonicSpelling: if given, this spelling of pitch will be used
        dynamic: a dynamic such as 'p', 'mf', 'ff', etc.
        **kws: any keyword accepted by Notation

    Returns:
        the created Notation
    """
    if gracenote:
        duration = 0
    else:
        duration = asF(duration) if duration is not None else None
    offset = asF(offset) if offset is not None else None
    out = Notation(pitches=[pitch], duration=duration, offset=offset, gliss=gliss,
                   dynamic=dynamic, **kws)
    if annotation:
        out.addText(annotation)
    if withId:
        out.groupid = str(id(out))
    if enharmonicSpelling:
        out.fixNotename(enharmonicSpelling)
    return out


def makeChord(pitches: list[pitch_t],
              duration: time_t = None,
              offset: time_t = None,
              annotation: str = None,
              dynamic='',
              fixed=False,
              **kws
              ) -> Notation:
    """
    Utility function to create a chord Notation

    Args:
        pitches: the pitches as midinotes or notenames. If given a note as str,
            the note in question is fixed at the given enharmonic representation.
        duration: the duration of this Notation. Use None to leave this unset,
            use 0 to create a grace note
        offset: the offset of this Notation (None to leave unset)
        annotation: a text annotation
        dynamic: a dynamic for this chord
        fixed: if True, fix the spelling of any pitch given as notename
        **kws: any keyword accepted by Notation

    Returns:
        the created Notation
    """
    out = Notation(pitches=pitches, duration=duration, offset=offset,
                   dynamic=dynamic, **kws)
    if fixed:
        for i, pitch in enumerate(pitches):
            if isinstance(pitch, str):
                out.fixNotename(pitch, i)

    if annotation:
        if isinstance(annotation, str) and annotation.isspace():
            logger.warning("Trying to add an empty annotation")
        else:
            out.addText(annotation)
    return out


def makeRest(duration: time_t,
             offset: time_t = None,
             dynamic: str = '') -> Notation:
    """
    Shortcut function to create a rest notation.

    A rest is only needed when stacking notations within a container like
    Chain or Track, to signal a spacing between notations.
    Just explicitely setting the offset of a notation has the
    same effect

    Args:
        duration: the duration of the rest
        offset: the start time of the rest. Normally a rest's offset
            is left unspecified (None)

    Returns:
        the created rest (a Notation)
    """
    assert duration > 0
    return Notation(duration=asF(duration), offset=None if offset is None else asF(offset),
                    isRest=True, _init=False, dynamic=dynamic)


def notationsToCoreEvents(notations: list[Notation]
                          ) -> list[maelzel.core.Note | maelzel.core.Chord]:
    """
    Convert notations to their corresponding maelzel.core Note or Chord

    Args:
        notations: a list of Notations to convert

    Returns:
        a list of Note/Chord, corresponding to the input notations

    """
    from maelzel.core import Note, Chord, Rest
    out = []
    for n in notations:
        assert isinstance(n, Notation), f"Expected a Notation, got {n}\n{notations=}"
        if n.isRest:
            out.append(Rest(n.duration, offset=n.offset))
        elif len(n.pitches) == 1:
            # note
            pitch = n.getFixedNotename(0) or n.pitches[0]
            note = Note(pitch=pitch,
                        dur=n.duration,
                        offset=n.offset,
                        dynamic=n.dynamic,
                        tied=n.tiedNext,
                        fixed=isinstance(pitch, str),
                        gliss=n.gliss,
                        properties=n.properties)
            # TODO: add attachments, etc.
            out.append(note)
        else:
            # chord
            notenames = [n.getFixedNotename(i) or n.pitches[i]
                         for i in range(len(n))]
            chord = Chord(notes=notenames,
                          dur=n.duration,
                          offset=n.offset,
                          dynamic=n.dynamic,
                          tied=n.tiedNext,
                          gliss=n.gliss,
                          properties=n.properties)
            out.append(chord)
    return out


def durationsCanMerge(n0: Notation, n1: Notation) -> bool:
    """
    True if these Notations can be merged based on duration and start/end

    Two durations can be merged if their sum is regular, meaning
    the sum has a numerator of 1, 2, 3, 4, or 7 (3 means a dotted
    note, 7 a double dotted note) and the denominator is <= 64
    (1/1 being a quarter note)

    Args:
        n0: one Notation
        n1: the other Notation

    Returns:
        True if they can be merged
    """
    dur0 = n0.symbolicDuration()
    dur1 = n1.symbolicDuration()
    sumdur = dur0 + dur1
    num, den = sumdur.numerator, sumdur.denominator
    if den > 64 or num not in {1, 2, 3, 4, 7}:
        return False

    # Allow: r8 8 + 4 = r8 4.
    # Don't allow: r16 8. + 8. r16 = r16 4. r16
    #grid = F(1, den)
    #if (num == 3 or num == 7) and ((n0.offset % grid) > 0 or (n1.end % grid) > 0):

    if num not in {1, 2, 3, 4, 6, 7, 8, 12, 16, 32}:
        return False
    return True


def notationsCannotMerge(n0: Notation, n1: Notation) -> str:
    if n0.isRest and n1.isRest:
        if n0.durRatios != n1.durRatios:
            return 'Duration ratios not compatible'
        if not durationsCanMerge(n0, n1):
            return 'Durations cannot merge'
    elif n0.isRest or n1.isRest:
        return 'A rest and a pitches notation cannot merge'
    else:
        if not (n0.tiedNext and n1.tiedPrev):
            return 'Notations not tied'
        if n0.durRatios != n1.durRatios:
            return 'Duration ratios not equal'
        if n0.pitches != n1.pitches:
            return 'Pitches not equal'
        if n1.dynamic or n0.dynamic != n1.dynamic:
            return 'Dynamics differ'
        if n1.attachments or not set(n1.attachments).issubset(set(n0.attachments)):
            return 'Attachments differ'
        if n0.noteheads != n1.noteheads:
            return 'Noteheads differ'
        if n1.gliss:
            return 'Last notation has a glissando'
        if not durationsCanMerge(n0, n1):
            return 'Durations cannot merge'
    return ''


def notationsCanMerge(n0: Notation, n1: Notation) -> bool:
    """
    Returns True if n0 and n1 can me merged

    Two Notations can merge if the resulting duration is regular. A regular
    duration is one which can be represented via **one** notation (a quarter,
    a half, a dotted 8th, a double dotted 16th are all regular durations,
    5/8 of a quarter is not)

    """
    if n0.isRest and n1.isRest:
        return (n0.durRatios == n1.durRatios and
                durationsCanMerge(n0, n1))

    # TODO: decide what to do about spanners
    if not (
        n0.tiedNext and
        n1.tiedPrev and
        n0.durRatios == n1.durRatios and
        n0.pitches == n1.pitches
    ):
        return False

    if n1.dynamic and n1.dynamic != n0.dynamic:
        return False

    if n1.attachments and not set(n1.attachments).issubset(set(n0.attachments)):
        return False

    if n0.noteheads != n1.noteheads:
        visible = {idx: notehead for idx, notehead in n1.noteheads.items()
                       if not notehead.hidden}
        if ((n0.noteheads or visible) and n0.noteheads != visible):
            return False

    if not n0.gliss and n1.gliss:
        return False

    if not durationsCanMerge(n0, n1):
        return False

    return True


def mergeNotationsIfPossible(notations: list[Notation]) -> list[Notation]:
    """
    Merge the given notations into one, if possible

    If two consecutive notations have same .durRatio and merging them
    would result in a regular note, merge them::

        8 + 8 = q
        q + 8 = q·
        q + q = h
        16 + 16 = 8

    In general::

        1/x + 1/x     2/x
        2/x + 1/x     3/x  (and viceversa)
        3/x + 1/x     4/x  (and viceversa)
        6/x + 1/x     7/x  (and viceversa)
    """
    assert len(notations) > 1
    out = [notations[0]]
    for n1 in notations[1:]:
        if notationsCanMerge(out[-1], n1):
            out[-1] = out[-1].mergeWith(n1)
        else:
            out.append(n1)
    assert len(out) <= len(notations)
    assert sum(n.duration for n in out) == sum(n.duration for n in notations)
    return out


def transferAttributesWithinTies(notations: list[Notation]) -> None:
    """
    When two notes are tied some attributes need to be copied to the tied note

    This functions works **IN PLACE**. Attributes which need to be transferred:

    * gliss: all notes in a tie need to be marked with gliss

    Args:
        notations: the notations to modify

    """
    insideGliss = False
    for n in notations:
        if n.gliss and not insideGliss:
            insideGliss = True
        elif not n.tiedPrev and insideGliss:
            insideGliss = False
        elif n.tiedPrev and insideGliss and not n.gliss:
            n.gliss = True

@dataclass
class SnappedNotation:
    notation: Notation
    offset: F
    duration: F

    def snapped(self) -> Notation:
        return self.notation.clone(offset=self.offset, duration=self.duration)

    def __repr__(self):
        return repr(self.snapped())
