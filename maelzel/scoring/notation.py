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
from emlib import mathlib
from . import definitions
from . import spanner as _spanner
import pitchtools as pt
import copy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable
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
    'durationsCanMerge',
    'mergeNotationsIfPossible',
    'tieNotationParts'
)


_UNSET = object()


class Notation:
    """
    This represents a notation (a rest, a note or a chord)

    Args:
        duration: the totalDuration of this Notation, in quarter-notes. A value of
            None indicates an unset totalDuration. During quantization an unset
            totalDuration is interpreted as lasting to the next notation.
            0 indicates a grace note
        pitches: if given, a list of pitches as midinote or notename. If a notename
            is given, the spelling is fixed. Otherwise a suitable spelling is calculated
            based on the context of this notation.
        offset: the offset of this Notation, in quarter-notes.
        isRest: is this a rest?
        tiedPrev: is this Notation tied to the previous one?
        tiedNext: is it tied to the next
        dynamic: the dynamic of this notation, one of "p", "pp", "f", etc.
        group: a str identification, can be used to tree Notations together
        durRatios: a list of tuples (x, y) indicating a tuple relationship.
            For example, a Notation used to represent one 8th note in a triplet
            would have the totalDuration 1/3 and durRatios=[(3, 2)]. Multiplying the
            totalDuration by the durRatios would result in the notated value, in this
            case 1/2 (1 being a quarter note). A value of None is the same as
            a value of [(1, 1)] (no modification)
        gliss: if True, a glissando will be rendered between this note and the next
        color: the color of this notations
        stem: if given, one of

    """
    _privateKeys = {
        '.snappedGracenote',
        '.clefHint',
        '.breakBeam',
        '.graceGroup',
        '.forceTupletBracket'
    }

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
                 "__weakref__"
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
        "The duration of this Notation, in quarternotes"

        self.pitches: list[float] = pitches
        "The pitches of this Notation (without spelling, just midinotes)"

        self.offset: F | None = offset
        "The start time in quarternotes"

        self.isRest = isRest
        "Is this a Rest?"

        self.tiedNext = tiedNext
        "Is this Notation tied to the next one?"

        self.tiedPrev = tiedPrev
        "Is this Notation tied to the previous one?"

        self.dynamic = dynamic
        "A dynamic mark"

        self.durRatios = durRatios
        """A set of ratios to apply to .duration to convert it to its notated duration
        
        see :meth:`Notation.notatedDuration`
        """

        self.groupid = group
        "The group id this Notation belongs to, if applicable"

        self.gliss = gliss
        "Is this Notation part of a glissando?"

        self.noteheads: dict[int, definitions.Notehead] | None = None
        "A dict mapping pitch index to notehead definition"

        self.color = color
        "The color of this entire Notation"

        self.stem = stem
        "A stem modifier (one of 'normal', 'hidden'"

        self.sizeFactor: int = sizeFactor
        "A size factor applied to this Notation (0: normal, 1: bigger, 2: even bigger, -1: smaller, etc.)"

        self.properties: dict[str, Any] | None = properties
        "A dict of user properties. To be set via setProperty"

        self.fixedNotenames: dict[int, str] | None = None
        "A dict mapping pitch index to spelling"

        self.attachments: list[Attachment] = []
        "Attachments are gathered here"

        self.spanners: list[_spanner.Spanner] | None = None
        "A list of spanners this Notations is part of"

        if self.isRest:
            assert self.duration > 0
            assert not self.pitches or (len(self.pitches) == 1 and self.pitches[0] == 0)
        else:
            assert self.pitches and all(p > 0 for p in self.pitches)
            for i, n in enumerate(pitches):
                if isinstance(n, str):
                    self.fixNotename(n, i)

    def __hash__(self):
        return id(self)

    def quantizedPitches(self, divs=4) -> list[float]:
        """Quantize the pitches of this Notation

        Args:
            divs: the number of divisions per semitone

        Returns:
            the quantized pitches as midinotes
        """
        return [round(p*4)/4 for p in self.pitches]

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
        assert self.attachments is not None
        # if self.attachments is None:
        #     self.attachments = []
        if False and attachment.exclusive:
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
                    merge=False
                    ) -> None:
        """
        Set a notehead in this notation

        Args:
            notehead: a Notehead or the notehead shape, as string (one of 'normal',
                'hidden', 'cross', 'harmonic', 'rhombus', 'square', etc.). See
                maelzel.scoring.definitions.noteheadShapes for a complete list
            idx: the index, corresponding to the pitch at the same index,
                or None to set all noteheads
            merge: if True and there is already a Notehead set for the given index,
                the new properties are merged with the properties of the already
                existing notehead

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

    def removeAttachments(self, predicate: Callable[[Attachment], bool]) -> None:
        """
        Remove attachments where predicate is  True

        Args:
            predicate: a function taking an Attachment, returns True if it should
                be removed

        """
        self.attachments[:] = [a for a in self.attachments
                               if not(predicate(a))]

    def removeAttachmentsByClass(self, cls: str | type) -> None:
        """Remove attachments which match the given class"""
        if isinstance(cls, str):
            cls = cls.lower()
            predicate = lambda a, cls=cls: type(a).__name__.lower() == cls
        else:
            predicate = lambda a, cls: isinstance(a, cls)
        self.removeAttachments(predicate=predicate)

    def hasSpanner(self, uuid: str) -> bool:
        """Returns true if a spanner with the given uuid is found"""
        return any(s.uuid == uuid for s in self.spanners) if self.spanners else False

    def addSpanner(self, spanner: _spanner.Spanner, end: Notation = None) -> Notation:
        """
        Add a Spanner to this Notation

        Spanners always are bound in pairs. A 'start' spanner is attached to
        one Notation, an 'end' spanner with the same uuid is attached to
        the notation where the spanner should end (see :meth:`Spanner.bind`)

        Args:
            spanner: the spanner to add.
            end: the end anchor of the spanner

        Returns:
            self

        """
        if self.spanners is None:
            self.spanners = []
        if spanner in self.spanners:
            raise ValueError(f"Spanner {spanner} was already added to this Notation ({self})")
        elif any(s.uuid == spanner.uuid for s in self.spanners):
            raise ValueError(f"A spanner with the uuid {spanner.uuid} is already part of this Notation")
        self.spanners.append(spanner)
        # self.spanners.sort(key=lambda spanner: spanner.priority())
        if end:
            end.addSpanner(spanner.endSpanner())
        self.spanners.sort(key=lambda spanner: spanner.priority())
        return self

    def transferSpanner(self, spanner: _spanner.Spanner, other: Notation):
        """Move the given spanner to another Notation

        This is done when replacing a Notation within a Node but there is a need
        to keep the spanner
        """
        self.spanners.remove(spanner)
        if not other.spanners or spanner not in other.spanners:
            if not other.hasSpanner(spanner.uuid):
                other.addSpanner(spanner)

    def removeSpanner(self, spanner: _spanner.Spanner) -> None:
        """
        Removes the given spanner from this Notation and from its partner

        Args:
            spanner: the spanner to remove

        """
        if not self.spanners:
            raise ValueError(f"spanner {spanner} not found in notation {self}")
        try:
            self.spanners.remove(spanner)
        except ValueError as e:
            raise ValueError(f"Cannot remove {spanner} from {self}: spanner not found. Spanners: {self.spanners}")

    def removeSpanners(self) -> None:
        """Remove all spanners from this Notation"""
        if self.spanners:
            for spanner in self.spanners:
                self.removeSpanner(spanner)


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
                is no match, an Exception will be raised if fail is set to True.
                Otherwise, we fail silently.

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
                                     f" (pitches: {self.resolveNotenames()})")
                return

        self.fixedNotenames[idx] = notename

    def getFixedNotename(self, idx: int = 0) -> str | None:
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
        if not self.fixedNotenames:
            return None
        fixedSlots = {}
        for notename in self.fixedNotenames.values():
            notated = pt.notated_pitch(notename)
            slot = notated.microtone_index(semitone_divisions=semitoneDivs)
            fixedSlots[slot] = notated.alteration_direction(min_alteration=0.5)
        return fixedSlots

    @property
    def isGracenote(self) -> bool:
        """Is this a gracenote?"""
        return not self.isRest and self.duration == 0

    @property
    def isRealnote(self) -> bool:
        """A real note is not a rest and not a gracenote"""
        return not self.isRest and self.duration > 0

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
    def end(self) -> F | None:
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

    def copyFixedSpellingTo(self, other: Notation):
        """Copy fixed spelling to *other*"""
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
        if noteheads := kws.get('noteheads'):
            assert isinstance(noteheads, dict), f'{self=}, {noteheads=}'

        out = self.copy()
        if (pitches := kws.pop('pitches', None)) is not None:
            out._setPitches(pitches)
            self.copyFixedSpellingTo(out)
        if kws:
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
        if self.noteheads:
            out.noteheads = self.noteheads.copy()
        if self.spanners:
            out.spanners = self.spanners.copy()
        return out

    def splitNotationAtOffsets(self: Notation, offsets: Sequence[F]
                               ) -> list[Notation]:
        """
        Splits a Notation at the given offsets

        Args:
            self: the Notation to split
            offsets: the offsets at which to split n

        Returns:
            the parts after splitting

        Example::

            >>> splitNotationAtOffsets(Notation(F(0.5), totalDuration=F(1)))
            [Notation(0.5, totalDuration=0.5), Notation(1, totalDuration=0.5)]

        """
        if not offsets:
            raise ValueError("offsets is empty")

        assert self.duration >= 0

        intervals = mathlib.split_interval_at_values(self.offset, self.end, offsets)
        assert all(isinstance(x0, F) and isinstance(x1, F)
                   for x0, x1 in intervals)

        if len(intervals) == 1:
            return [self]

        parts: list[Notation] = [self.clone(offset=start, duration=end - start)
                                 for start, end in intervals]

        # Remove superfluous dynamic/articulation
        for part in parts[1:]:
            part.dynamic = ''
            # part.removeAttachments(lambda item: isinstance(item, (attachment.Articulation, attachment.Text)))
            if part.spanners:
                part.spanners.clear()

        if not self.isRest:
            tieNotationParts(parts)
            parts[0].tiedPrev = self.tiedPrev
            parts[-1].tiedNext = self.tiedNext

        assert sum(part.duration for part in parts) == self.duration
        assert parts[0].offset == self.offset
        assert parts[-1].end == self.end
        if not self.isRest:
            assert parts[0].tiedPrev == self.tiedPrev
            assert parts[-1].tiedNext == self.tiedNext, f"{self=}, {parts=}"
        return parts

    def symbolicDuration(self) -> F:
        """
        The symbolic totalDuration of this Notation.

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

    def pitchclassIndex(self, semitoneDivs=2, index=0) -> int:
        """
        The index of the nearest pitch/microtone

        Args:
            semitoneDivs: the number of divisions per semitone (1=chromatic, 2=quartertones, ...)
            index: the index of the pitch within this Notation

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
        return pt.pitchclass(notename, semitone_divisions=semitoneDivs)

    def resolveNotenames(self, addFixedAnnotation=False) -> list[str]:
        """Resolve the enharmonic spellings for this Notation

        Args:
           addFixedAnnotation: if True, enforce the returned spelling

        Returns:
            the notenames of each pitch in this Notation
        """
        out = []
        for i, p in enumerate(self.pitches):
            notename = self.getFixedNotename(i)
            if not notename:
                notename = pt.m2n(p)
            elif addFixedAnnotation and not notename.endswith('!'):
                notename += '!'
            out.append(notename)
        return out

    @property
    def _notenames(self) -> list[str]:
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
                text: str | Text,
                placement='above',
                fontsize: int | float = None,
                fontstyle: str = None,
                box: str | bool = False,
                exclusive=False
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
            fontstyle: the style ('bold', 'italic')
            exclusive: if True, only one text annotation with the given text and attributes
                is allowed. This enables to set a given text for a Notation without needing
                to check at the callsite that this text is already present
        """
        if isinstance(text, Text):
            assert not text.text.isspace()
            annotation = text
        else:
            assert not text.isspace()
            annotation = Text(text=text, placement=placement, fontsize=fontsize,
                              fontstyle=fontstyle, box=box)
        if exclusive and self.attachments:
            for attach in self.attachments:
                if attach == annotation:
                    return
        self.addAttachment(annotation)

    def notatedDuration(self) -> NotatedDuration:
        """
        The totalDuration of the notated figure, in quarter-notes, independent of any tuples.

        A quarter-note inside a triplet would have a notatedDuration of 1
        """
        return notatedDuration(self.duration, self.durRatios)

    def canMergeWith(self, other: Notation) -> bool:
        """Can this Notation merge with *other*?"""
        return notationsCanMerge(self, other)
    
    def mergeWith(self, other: Notation) -> Notation:
        """Merge this Notation with ``other``"""
        return mergeNotations(self, other)

    def setProperty(self, key: str, value=_UNSET) -> None:
        """
        Set any property of this Notation.

        Properties can be used, for example, for any rendering backend to
        pass directives which are specific to that rendering backend.

        Args:
            key: the key to set
            value: if not given then the key is deleted if found, similar to
                ``dict.pop(key, None)``
        """
        if value is _UNSET:
            if not self.properties:
                return
            self.properties.pop(key, None)
        else:
            if self.properties is None:
                self.properties = {}
            if key.startswith('.'):
                assert key in self._privateKeys, f"Key {key} unknown. Possible private keys: {self._privateKeys}"
            self.properties[key] = value

    def getProperty(self, key: str, default=None) -> Any:
        """
        Get the value of a property. If the key is not found, return ``default``
        """
        if not self.properties:
            return default
        if key.startswith('.'):
            assert key in self._privateKeys, f"Key {key} unknown. Possible private keys: {self._privateKeys}"
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
        """Remove any clef hints from this Notation

        .. seealso:: :meth:`Notation.getClefHint`, :meth:`Notation.setClefHint`"""
        self.setProperty('.clefHint')

    def getClefHint(self, idx: int = 0) -> str | None:
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
            if self.duration.denominator < 100:
                info.append(f"{self.duration.numerator}/{self.duration.denominator}")
            else:
                info.append(showT(self.duration))

        if self.durRatios and self.durRatios != [F(1)]:
            info.append(",".join(showF(r) for r in self.durRatios))

        if self.isRest:
            info.append("rest")
        elif self.pitches:
            if len(self.pitches) > 1:
                info.append("[" + " ".join(self.resolveNotenames()) + "]")
            else:
                info.append(self.resolveNotenames()[0])
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

    def copyAttributesTo(self: Notation, dest: Notation, spelling=True) -> None:
        """
        Copy attributes of self to dest
        """
        assert dest is not self

        exclude = {'totalDuration', 'pitches', 'offset', 'durRatios', 'tree',
                   'properties', 'attachments', 'spanners', '__weakref__'}

        for attr in self.__slots__:
            if attr not in exclude:
                value = getattr(self, attr)
                if isinstance(value, (list, dict)):
                    value = value.copy()
                setattr(dest, attr, value)

        if self.properties:
            if dest.properties:
                for prop, value in self.properties.items():
                    dest.setProperty(prop, value)
            else:
                dest.properties = self.properties.copy()

        if self.attachments:
            for i, a in enumerate(self.attachments):
                dest.addAttachment(a)

        if spelling:
            self.copyFixedSpellingTo(dest)

    def copyAttachmentsTo(self, dest: Notation) -> None:
        """Copy any attachments in self to *dest* Notation"""
        if self.attachments:
            for a in self.attachments:
                dest.addAttachment(a)

    def __len__(self) -> int:
        return len(self.pitches)

    def hasAttributes(self) -> bool:
        """
        True if this notation has information attached

        Information is any dynanic, attachments, spanners, etc.
        """
        return bool(self.dynamic or
                    self.attachments or
                    self.spanners or
                    self.noteheads)

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
        self.copyFixedSpellingTo(out)
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
    - have a totalDuration
    - have the same pitch/pitches.

    All other attributes are taken from the first notation and the
    totalDuration of this first notation is extended to cover both notations
    """
    if a.pitches != b.pitches:
        raise ValueError("Attempting to merge two Notations with "
                         "different pitches")
    assert a.duration is not None and b.duration is not None
    assert b.offset is None or (a.end == b.offset)
    out = a.clone(duration=a.duration + b.duration,
                  tiedNext=b.tiedNext)

    if b.fixedNotenames:
        b.copyFixedSpellingTo(out)

    spanners = mergeSpanners(a, b)
    out.spanners = spanners
    return out


def mergeSpanners(a: Notation, b: Notation
                  ) -> list[_spanner.Spanner] | None:
    """
    Merge the spanner of two Notations

    Shared spanners (for example, a crescendo from a to b) are removed

    Args:
        a: the first notation
        b: the second notation

    Returns:
        a list of merged spanners, or None if both a and b have no spanners
    """
    if not a.spanners and not b.spanners:
        spanners = None
    elif not a.spanners:
        spanners = b.spanners
    elif not b.spanners:
        spanners = a.spanners
    else:
        spanners = a.spanners + b.spanners

    return spanners


def makeGroupId(parent: str = '') -> str:
    """
    Create an id to tree notations together

    Args:
        parent: if given it will be prepended as {parent}/{groupid}

    Returns:
        the tree id as string
    """
    groupid = str(uuid.uuid1())
    return groupid if parent is None else f'{parent}/{groupid}'


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
        duration: the totalDuration of this Notation. Use None to leave this unset,
            0 creates a grace note
        offset: the offset of this Notation (None to leave unset)
        annotation: an optional text annotation for this note
        gliss: does this Notation start a glissando?
        withId: if True, this Notation has a tree id and this id
            can be used to mark multiple notes as belonging to a same tree
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
        duration: the totalDuration of this Notation. Use None to leave this unset,
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
             dynamic: str = '',
             annotation: str = ''
             ) -> Notation:
    """
    Shortcut function to create a rest notation.

    A rest is only needed when stacking notations within a container like
    Chain or Track, to signal a spacing between notations.
    Just explicitely setting the offset of a notation has the
    same effect

    Args:
        duration: the totalDuration of the rest
        offset: the start time of the rest. Normally a rest's offset
            is left unspecified (None)
        dynamic: if given, attach this dynamic to the rest
        annotation: if given, attach this text annotation to the rest

    Returns:
        the created rest (a Notation)
    """
    assert duration > 0
    out = Notation(duration=asF(duration),
                   offset=None if offset is None else asF(offset),
                   dynamic=dynamic,
                   isRest=True,
                   _init=False)
    if annotation:
        out.addText(annotation)
    return out


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
    True if these Notations can be merged based on totalDuration and start/end

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
    # grid = F(1, den)
    # if (num == 3 or num == 7) and ((n0.offset % grid) > 0 or (n1.end % grid) > 0):

    if num not in {1, 2, 3, 4, 6, 7, 8, 12, 16, 32}:
        return False
    return True


def notationsCanMerge(n0: Notation, n1: Notation) -> bool:
    """
    Returns True if n0 and n1 can me merged

    Two Notations can merge if the resulting totalDuration is regular. A regular
    totalDuration is one which can be represented via **one** notation (a quarter,
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

    if not n0.gliss and (n0.noteheads or n1.noteheads) and n0.noteheads != n1.noteheads:
        if not n1.noteheads:
            return False

        n1visiblenoteheads = {idx: notehead for idx, notehead in n1.noteheads.items()
                              if not notehead.hidden}
        if n0.noteheads != n1visiblenoteheads:
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


def tieNotationParts(parts: list[Notation]) -> None:
    """ Tie these notations in place """

    for part in parts[:-1]:
        part.tiedNext = True

    hasGliss = parts[0].gliss
    for part in parts[1:]:
        part.tiedPrev = True
        part.dynamic = ''
        part.removeAttachments(lambda a: isinstance(a, (Text, Articulation)))
        if hasGliss:
            part.gliss = True



@dataclass
class SnappedNotation:
    notation: Notation
    offset: F
    duration: F

    def makeSnappedNotation(self, extraOffset: F | None = None) -> Notation:
        """
        Clone the original notation to be snapped to offset and totalDuration

        Args:
            extraOffset: if given, an extra offset to apply to the snapped notation

        Returns:
            the snapped notation

        """
        offset = self.offset if not extraOffset else self.offset + extraOffset
        notation = self.notation.clone(offset=offset, duration=self.duration)
        if self.duration == 0 and self.notation.duration > 0:
            notation.setProperty('.snappedGracenote', True)
            notation.setProperty('originalDuration', self.notation.duration)
        return notation

    def __repr__(self):
        return repr(self.makeSnappedNotation())
