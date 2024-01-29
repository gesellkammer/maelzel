"""
A Notation represents a note/chord/rest
"""
from __future__ import annotations
from dataclasses import dataclass
import uuid
import copy
import pitchtools as pt
from emlib.iterlib import pairwise, first

from maelzel.common import UNSET, Unset, F, F1
from maelzel._util import showF, showT
from .common import *
from .attachment import *
from . import util
from . import definitions
from . import spanner as _spanner

from typing import TYPE_CHECKING, TypeVar, cast as _cast
if TYPE_CHECKING:
    from typing import Callable, Sequence, Any
    import maelzel.core
    import maelzel.core.symbols


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
    'tieNotations',
    'splitNotationsAtOffsets'
)


_EMPTYLIST = []


class Notation:
    """
    This represents a notation (a rest, a note or a chord)

    Args:
        duration: the duration of this Notation, in quarter-notes. 0 indicates
            a grace note
        pitches: if given, a list of pitches as midinote or notename.
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
        fixNotenames: if True, pitches given as strings are fixed to the given spelling

    """
    _privateKeys = {
        '.clefHint',
        '.graceGroup',
        '.mergeable',
        '.forceTupletBracket',
        '.snappedGracenote',   # Is this a note which has been snapped to 0 duration?
        '.originalDuration'    # For snapped notes, it is useful to keep track of the original duration
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
                 durRatios: tuple[F, ...] = (),
                 group='',
                 gliss=False,
                 color='',
                 sizeFactor=1,  # size is relative: 0 is normal, +1 is bigger, -1 is smaller
                 properties: dict[str, Any] = None,
                 fixNotenames=False,
                 _init=True
                 ):

        assert duration is not None

        if _init:
            if dynamic:
                dynamic = definitions.normalizeDynamic(dynamic, '')
            duration = asF(duration)
            if pitches is not None:
                midinotes = [asmidi(p) for p in pitches]
            else:
                midinotes = []

            if offset is not None:
                offset = asF(offset)
            if isRest:
                tiedNext = False
                tiedPrev = False

        else:
            midinotes = [] if pitches is None else _cast(list[float], pitches)

        if durRatios:
            assert isinstance(durRatios, tuple) and all(isinstance(r, F) for r in durRatios)
            assert F1 not in durRatios

        self.duration: F = duration
        "The duration of this Notation, in quarternotes"

        self.pitches: list[float] = midinotes
        "The pitches of this Notation (without spelling, just midinotes)"

        self.offset: F | None = offset
        "The start time in quarternotes"

        self.isRest: bool = isRest
        "Is this a Rest?"

        self.tiedNext: bool = tiedNext
        "Is this Notation tied to the next one?"

        self.tiedPrev: bool = tiedPrev
        "Is this Notation tied to the previous one?"

        self.dynamic: str = dynamic
        "A dynamic mark"

        self.durRatios: tuple[F, ...] = durRatios
        """A set of ratios to apply to .duration to convert it to its notated duration
        
        see :meth:`Notation.notatedDuration`
        """

        self.groupid: str = group
        "The group id this Notation belongs to, if applicable"

        self.gliss: bool = gliss
        "Is this Notation part of a glissando?"

        self.noteheads: dict[int, definitions.Notehead] | None = None
        "A dict mapping pitch index to notehead definition"

        self.color: str = color
        "The color of this entire Notation"

        self.sizeFactor: int = sizeFactor
        "A size factor applied to this Notation (0: normal, 1: bigger, 2: even bigger, -1: smaller, etc.)"

        self.properties: dict[str, Any] | None = properties
        "A dict of user properties. To be set via setProperty"

        self.fixedNotenames: dict[int, str] | None = None
        "A dict mapping pitch index to spelling"

        self.attachments: list[Attachment] | None = None
        "Attachments are gathered here"

        self.spanners: list[_spanner.Spanner] | None = None
        "A list of spanners this Notations is part of"

        if self.isRest:
            assert self.duration > 0
            assert not self.pitches or (len(self.pitches) == 1 and self.pitches[0] == 0)
        else:
            if not pitches or any(p <= 0 for p in self.pitches):
                raise ValueError(f"Invalid pitches: {self.pitches}")
            if fixNotenames:
                for i, n in enumerate(pitches):
                    if isinstance(n, str):
                        self.fixNotename(n, i)

    @property
    def quantized(self) -> bool:
        """Is this Notation quantized?"""
        return self.offset is not None

    @property
    def qoffset(self) -> F:
        """Quantized offset, ensures that it is never None"""
        if self.offset is None:
            raise ValueError(f"This Notation does not have a fixed offset: {self}")
        return self.offset

    def __hash__(self):
        attachhash = 0 if not self.attachments else hash(tuple(str(a) for a in self.attachments))
        pitcheshash = tuple(self.pitches) if self.pitches else 0
        return hash((self.duration, pitcheshash, self.tiedNext, self.tiedPrev,
                     self.dynamic, self.gliss, attachhash))
        # return id(self)

    def fusedDurRatio(self) -> F:
        num, den = 1, 1
        for ratio in self.durRatios:
            num *= ratio.numerator
            den *= ratio.denominator
        return F(num, den)

    @staticmethod
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
            duration: the duration of the rest
            offset: the start time of the rest. Normally a rest's offset
                is left unspecified (None)
            dynamic: if given, attach this dynamic to the rest
            annotation: if given, attach this text annotation to the rest

        Returns:
            the created rest (a Notation)
        """
        return makeRest(duration=duration, offset=offset, dynamic=dynamic, annotation=annotation)

    def quantizedPitches(self, divs=4) -> list[float]:
        """Quantize the pitches of this Notation

        Args:
            divs: the number of divisions per semitone

        Returns:
            the quantized pitches as midinotes
        """
        return [round(p*divs)/divs for p in self.pitches]

    def getAttachments(self,
                       cls: str | type = '',
                       predicate: Callable = None,
                       anchor: int | None | Unset = UNSET
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
        if not self.attachments:
            return []

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

        if anchor is not UNSET:
            attachments = [a for a in attachments if a.anchor == anchor or a.anchor is None]

        return attachments

    def findAttachment(self,
                       cls: str | type,
                       anchor: int | None | Unset = UNSET,
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
        if not self.attachments:
            return None
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
            attachment (like a certain text) will remove any previous such attachment.

        Args:
            attachment: an instance of scoring.attachment.Attachment

        Returns:
            self

        """
        if self.attachments is None:
            self.attachments = []
        if attachment.exclusive:
            # An exclusive attachment is exclusive at the class level (like a Harmonic or an ornament)
            cls = type(attachment)
            if any(isinstance(a, cls) for a in self.attachments):
                logger.debug(f"An attachment of class {cls} already present in this notation, "
                             f"replacing the old one by the new one ({attachment})")
                self.attachments = [a for a in self.attachments
                                    if not isinstance(a, cls)]
        else:
            if attachment in self.attachments:
                logger.warning(f"Attachment {attachment} already present in this notation ({self})")
                return self

        if attachment.anchor is not None:
            assert 0 <= attachment.anchor < len(self.pitches)
        self.attachments.append(attachment)
        return self

    @property
    def isStemless(self) -> bool:
        """Is this Notation stemless?

        This property can be set by adding a StemTraits attachment
        """
        if self.attachments:
            attach = first(a for a in self.attachments if isinstance(a, StemTraits))
            if attach is not None and attach.hidden:
                return True
        return False

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

    def getNotehead(self, index=0) -> definitions.Notehead | None:
        """
        Get a Notehead by index

        Args:
            index: the notehead index. This corresponds to the pitch index in self.pitches

        Returns:
            the Notehead or None if not defined

        """
        if not self.noteheads:
            return None
        return self.noteheads.get(index)

    def addArticulation(self, articulation: str | Articulation, color='', placement='') -> Notation:
        """
        Add an articulation to this Notation.

        See ``definitions.articulations`` for possible values. We understand
        articulation in a broad sense as any symbol attached to a note/chord

        Args:
            articulation: an Articulation object, or one of accent, staccato, tenuto,
            marcato, staccatissimo, espressivo, portato, arpeggio, upbow,
            downbow, flageolet, open, closed, stopped, snappizz
            color: if given, color of the articulation
            placement: one of 'above', 'below'. If not given, the default placement
                for the given articulation is used

        Returns:
            self
        """
        if isinstance(articulation, str):
            articulation = definitions.normalizeArticulation(articulation)
            if not articulation:
                raise ValueError(f"Articulation {articulation} unknown. "
                                 f"Possible values: {definitions.articulations}")
            articulation = Articulation(articulation, color=color, placement=placement)
        else:
            if color or placement:
                articulation = articulation.copy()
                if color:
                    articulation.color = color
                if placement:
                    articulation.placement = placement
        assert isinstance(articulation, Articulation)
        return self.addAttachment(articulation)

    def removeAttachments(self, predicate: Callable[[Attachment], bool]) -> None:
        """
        Remove attachments where predicate is  True

        Args:
            predicate: a function taking an Attachment, returns True if it should
                be removed

        """
        if self.attachments:
            self.attachments[:] = [a for a in self.attachments
                                   if not(predicate(a))]

    def removeAttachmentsByClass(self, cls: str | type) -> None:
        """Remove attachments which match the given class"""
        if isinstance(cls, str):
            cls = cls.lower()
            predicate = lambda a, cls=cls: type(a).__name__.lower() == cls
        else:
            predicate = lambda a, cls=cls: isinstance(a, cls)
        self.removeAttachments(predicate=predicate)

    def hasSpanner(self, uuid: str, kind='') -> bool:
        """Returns true if a spanner with the given uuid is found"""
        return bool(self.findSpanner(uuid, kind=kind)) if self.spanners else False

    def findSpanner(self, uuid: str, kind='') -> _spanner.Spanner | None:
        """
        Find a spanner with the given attributes

        Args:
            uuid: the uuid of the spanner
            kind: the kind of the spanner, one of 'start' / 'end'
        """
        if not self.spanners:
            return None
        if kind:
            assert kind == 'start' or kind == 'end'
            return next((s for s in self.spanners if s.uuid == uuid and s.kind == kind), None)
        return next((s for s in self.spanners if s.uuid == uuid), None)

    def addSpanner(self, spanner: _spanner.Spanner, end: Notation = None
                   ) -> Notation:
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

        if self.findSpanner(uuid=spanner.uuid, kind=spanner.kind):
            raise ValueError(f"Spanner {spanner} was already added to this Notation ({self})")
        elif partner := self.findSpanner(uuid=spanner.uuid, kind='start' if spanner.kind == 'end' else 'end'):
            logger.warning(f"A Notation cannot be assigned both start and end of a spanner. Removing "
                           f"the partner spanner"
                           f"{self=}, {spanner=}, {partner=}, {end=}")
            self.removeSpanner(partner)
        else:
            self.spanners.append(spanner)
            if end:
                end.addSpanner(spanner.makeEndSpanner())
            self.spanners.sort(key=lambda spanner: spanner.priority())
        return self

    def resolveHarmonic(self, removeAttachment=False) -> Notation:
        """
        Realize an artificial harmonic as a chord with the corresponding noteheads

        Returns:
            the modified Notation or self if this Notation is not a harmonic
        """
        if not self.attachments:
            logger.warning(f"Notation has no attachments: {self}")
            return self
        elif len(self.pitches) > 1:
            logger.error(f"Cannot set a chord as artificial harmonic for notation {self}")
            return self

        harmonic = next((a for a in self.attachments if isinstance(a, Harmonic)), None)
        if not harmonic:
            logger.warning(f"Notation has no harmonic attachment: {self}")
            return self

        if harmonic.interval == 0:
            n = self.copy()
            n.setNotehead('harmonic')
            if n.attachments:
                n.attachments.remove(harmonic)
        else:
            fund = self.notename(0)
            touched = pt.transpose(fund, harmonic.interval)
            n = self.clone(pitches=(fund, touched))
            n.fixNotename(touched, idx=1)
            n.setNotehead('harmonic', idx=1)

        if removeAttachment and n.attachments:
            n.attachments = [a for a in n.attachments if not isinstance(a, Harmonic)]
        return n

    def transferSpanner(self, spanner: _spanner.Spanner, other: Notation) -> bool:
        """Move the given spanner to another Notation

        Args:
            spanner: the spanner to transfer
            other: the destination notation

        Returns:
            True if the spanner was actually transferred

        This is done when replacing a Notation within a Node but there is a need
        to keep the spanner
        """
        assert self.spanners and spanner in self.spanners

        if other is self:
            raise ValueError(f"Cannot transfer a spanner to self ({self=}, {spanner=}")

        else:
            if other.addSpanner(spanner):
                self.spanners.remove(spanner)
            return True

    def removeSpanner(self, spanner: _spanner.Spanner | str) -> None:
        """
        Removes the given spanner from this Notation

        Args:
            spanner: the spanner to remove or the uuid of the spanner to remove

        """
        if not self.spanners:
            raise ValueError(f"spanner {spanner} not found in notation {self}")
        if spanner.parent and spanner.parent is not self:
            logger.error(f"This spanner {spanner} has a different parent! parent={spanner.parent}, self={self}")

        if isinstance(spanner, _spanner.Spanner):
            spanner.parent = None
            self.spanners.remove(spanner)
        else:
            for spannerobj in (s for s in self.spanners if s.uuid == spanner):
                self.removeSpanner(spannerobj)

    def removeSpanners(self) -> None:
        """Remove all spanners from this Notation"""
        if self.spanners:
            for spanner in self.spanners:
                self.removeSpanner(spanner)

    def checkIntegrity(self, fix=False) -> list[str]:
        """
        Checks the integrity of self

        Args:
            fix: if True, attempts to fix the probelms found, if possible

        Returns:
            a list of error messages

        """
        out = []
        if self.spanners:
            for spanner in self.spanners.copy():
                if partner := self.findSpanner(uuid=spanner.uuid, kind='start' if spanner.kind == 'end' else 'end'):
                    msg = (f"Found notation with both start and end spanner of same uuid, "
                           f"{spanner=}, {partner=}")
                    logger.warning(msg)
                    out.append(msg)
                    if fix:
                        out.append(f"Removed spanner pair ({spanner}, {partner}) from {self}")
                        self.removeSpanner(spanner)
                        self.removeSpanner(partner)
        return out

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
        Fix the spelling for the pitch at index **inplace**

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
            if idx is None:
                if len(self.pitches) > 1:
                    raise ValueError(f"Note index not given, but it is needed since this"
                                     f" is a chord with pitches {self.pitches}")
                notename = self.notename(0)
            else:
                notename = self.notename(idx)
            notename = pt.enharmonic(notename)

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

    @property
    def mergeable(self) -> bool:
        return self.getProperty('.mergeable', True)

    @mergeable.setter
    def mergeable(self, value: bool):
        self.setProperty('.mergeable', value)

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
    def end(self) -> F:
        """
        The end time of this notation.

        Raises an exception if this Notation has no offset
        """
        if self.offset is None:
            raise ValueError(f"This notations has no offset: {self}")
        return self.offset + self.duration

    def _setPitches(self, pitches: list[pitch_t], resetFixedNotenames=True) -> None:
        if len(pitches) != self.pitches:
            if self.noteheads:
                logger.info("Notation: setting new pitches inplace. Noteheads will be reset")
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

    def __copy__(self) -> Notation:
        """
        Copy this Notation as is
        """
        return self.copy()

    def asRest(self) -> Notation:
        return self.__class__(isRest=True,
                              duration=self.duration,
                              offset=self.offset,
                              dynamic=self.dynamic)

    def cloneAsTie(self,
                   duration: F,
                   offset: F | None,
                   tiedPrev=True,
                   tiedNext: bool | None = None,
                   gliss: bool = None,
                   ) -> Notation:
        """
        Clone self so that the cloned Notation can be used within a logical tie

        This is used when a notation is split across a measure or a beam
        or within a tuplet

        Returns:
            The cloned Notation
        """
        if self.isRest:
            return Notation(isRest=True,
                            duration=duration,
                            offset=offset,
                            pitches=None,
                            )

        out = Notation(duration=duration,
                       offset=offset,
                       pitches=self.pitches,
                       tiedPrev=tiedPrev,
                       tiedNext=tiedNext if tiedNext is not None else self.tiedNext,
                       dynamic='',
                       gliss=gliss if gliss is not None else self.gliss,
                       color=self.color,
                       durRatios=self.durRatios)

        if self.attachments:
            for attach in self.attachments:
                if attach.copyToSplitNotation:
                    out.addAttachment(attach)

        if self.noteheads is not None:
            out.noteheads = self.noteheads.copy()

        if self.fixedNotenames is not None:
            out.fixedNotenames = self.fixedNotenames

        return out

    def __deepcopy__(self, memo=None):
        return self.copy()

    def copy(self) -> Notation:
        """Copy this Notation"""
        out = Notation(duration=self.duration,
                       pitches=self.pitches.copy() if self.pitches else None,
                       offset=self.offset,
                       isRest=self.isRest,
                       tiedPrev=self.tiedPrev,
                       tiedNext=self.tiedNext,
                       dynamic=self.dynamic,
                       durRatios=self.durRatios,
                       group=self.groupid,
                       gliss=self.gliss,
                       color=self.color,
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

    def splitAtOffsets(self: Notation, offsets: Sequence[F]
                       ) -> list[Notation]:
        """
        Splits a Notation at the given offsets

        Args:
            self: the Notation to split
            offsets: the offsets at which to split n

        Returns:
            the parts after splitting

        Example::

            >>> splitAtOffsets(Notation(F(0.5), duration=F(1)))
            [Notation(0.5, duration=0.5), Notation(1, duration=0.5)]

        """
        if not offsets:
            raise ValueError("offsets is empty")

        assert self.duration >= 0 and self.offset is not None
        intervals = util.splitInterval(self.offset, self.end, offsets)
        assert all(isinstance(x0, F) and isinstance(x1, F)
                   for x0, x1 in intervals)

        if len(intervals) == 1:
            return [self]

        start0, end0 = intervals[0]
        parts: list[Notation] = [self.clone(offset=start0, duration=end0-start0)]
        parts.extend((self.cloneAsTie(offset=start, duration=end - start)
                      for start, end in intervals[1:]))

        tieNotations(parts)
        parts[0].tiedPrev = self.tiedPrev
        parts[-1].tiedNext = self.tiedNext

        assert sum(part.duration for part in parts) == self.duration
        assert parts[0].offset == self.offset
        assert parts[-1].end == self.end
        if not self.isRest:
            assert parts[0].tiedPrev == self.tiedPrev
            assert parts[-1].tiedNext == self.tiedNext, f"{self=}, {parts=}"
        return parts

    def hasRegularDuration(self) -> bool:
        symdur = self.symbolicDuration()
        return symdur.denominator in (1, 2, 4, 8, 16) and symdur.numerator in (1, 2, 3, 4, 7)

    def symbolicDuration(self) -> F:
        """
        The symbolic duration of this Notation.

        This represents the notated figure (1=quarter, 1/2=eighth note,
        1/4=16th note, etc)
        """
        return self.duration * self.fusedDurRatio()

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
            addFixedAnnotation: if True, enforce the returned spelling by adding
            a '!' suffix.

        Returns:
            the notenames of each pitch in this Notation
        """
        if self.isRest:
            return _EMPTYLIST

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
                italic=False,
                weight='normal',
                fontfamily='',
                box: str | bool = False,
                exclusive=False,
                role=''
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
            italic: if True, use italic style
            weight: one of 'normal', 'bold'
            fontfamily: the font used
            role: either unset or one of 'measure', ...
            exclusive: if True, only one text annotation with the given text and attributes
                is allowed. This enables to set a given text for a Notation without needing
                to check at the callsite that this text is already present
        """
        if isinstance(text, Text):
            assert not text.text.isspace()
            annotation = text
        else:
            assert not text.isspace()
            annotation = Text(text=text,
                              placement=placement,
                              fontsize=fontsize,
                              fontfamily=fontfamily,
                              italic=italic,
                              weight=weight,
                              box=box,
                              role=role)
        if exclusive and self.attachments:
            for attach in self.attachments:
                if attach == annotation:
                    return
        self.addAttachment(annotation)

    def notatedDuration(self) -> NotatedDuration:
        """
        The duration of the notated figure as a NotatedDuration

        A quarter-note inside a triplet would have a notatedDuration of 1
        """
        return util.notatedDuration(self.duration, self.durRatios)

    def canMergeWith(self, other: Notation) -> bool:
        """Can this Notation merge with *other*?"""
        return notationsCanMerge(self, other)
    
    def mergeWith(self, other: Notation) -> Notation:
        """Merge this Notation with ``other``"""
        return mergeNotations(self, other)

    def setProperty(self, key: str, value) -> None:
        """
        Set any property of this Notation.

        Properties can be used, for example, for any rendering backend to
        pass directives which are specific to that rendering backend.

        Args:
            key: the key to set
            value: the value of the property.
        """
        if key.startswith('.'):
            assert key in self._privateKeys, f"Key {key} unknown. Possible private keys: {self._privateKeys}"
        if self.properties is None:
            self.properties = {}
        self.properties[key] = value

    def delProperty(self, key: str) -> None:
        """
        Remove the given property

        Args:
            key: the key to remove

        """
        if self.properties:
            self.properties.pop(key, None)

    def getProperty(self, key: str, default=None, setdefault=None) -> Any:
        """
        Get the value of a property.

        Args:
            key: the key to query
            setdefault: if given, sets properties[key] = value if not already set
            default: like setdefault but never modifies the actual properties

        Returns:
            the value of the given property, or a fallback value
        """
        if key.startswith('.'):
            assert key in self._privateKeys, f"Key {key} unknown. Possible private keys: {self._privateKeys}"
        if not self.properties:
            if setdefault is not None:
                self.setProperty(key, setdefault)
                return setdefault
            return default
        if setdefault is not None:
            return self.properties.setdefault(key, setdefault)
        return self.properties.get(key, default)

    def setClefHint(self, clef: str, idx: int = None) -> None:
        """
        Sets a hint regarding which clef to use for this notation

        .. warning::

            This is mostly used internally and is an experimental feature.
            It is conceived for the case where two notations
            are bound by a glissando, and they should be placed together,
            even if the pitch of some of them might indicate otherwise

        Args:
            clef: the clef to set, one of 'treble', 'bass' or 'treble8', 'treble15' or 'bass8'
            idx: the index of the pitch within a chord, or None to apply to
                the whole notation

        """
        normalizedclef = definitions.clefs.get(clef)
        if normalizedclef is None:
            raise ValueError(f"Clef {clef} not known. Possible clefs: {definitions.clefs.keys()}")
        if idx is None:
            self.setProperty('.clefHint', normalizedclef)
        else:
            hint = self.getProperty('.clefHint', {})
            hint[idx] = normalizedclef
            self.setProperty('.clefHint', hint)

    def clearClefHints(self) -> None:
        """Remove any clef hints from this Notation

        .. seealso:: :meth:`Notation.getClefHint`, :meth:`Notation.setClefHint`"""
        self.delProperty('.clefHint')

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
        if self.isRest:
            info.append("rest")
        elif self.pitches:
            if len(self.pitches) > 1:
                s = "[" + " ".join(self.resolveNotenames()) + "]"
            else:
                s = self.resolveNotenames()[0]
            if self.tiedPrev:
                s = f"~{s}"
            if self.tiedNext:
                s += "~"
            if self.gliss:
                s += "gliss"
            info.append(s)
        if self.offset is None:
            info.append(f"None, dur={showT(self.duration)}")
        elif self.duration == 0:
            info.append(f"{showT(self.offset)}:grace")
        else:
            info.append(f"{showT(self.offset)}:{showT(self.end)}")
            if int(self.duration) == self.duration or self.duration.denominator >= 100:
                info.append(showT(self.duration) + '♩')
            else:
                info.append(f"{self.duration.numerator}/{self.duration.denominator}♩")
            
        if self.durRatios and self.durRatios != [F(1)]:
            info.append(",".join(showF(r) for r in self.durRatios))

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

        exclude = {'duration', 'pitches', 'offset', 'durRatios', 'group',
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
                of 0)

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
        if self.attachments:
            for a in self.attachments:
                if a.anchor is not None and a.anchor in indexes:
                    anchor = mappedIndexes[a.anchor]
                    a = copy.copy(a)
                    a.anchor = anchor
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
    - have a duration
    - have the same pitch/pitches.

    All other attributes are taken from the first notation and the
    duration of this first notation is extended to cover both notations
    """
    assert type(a) == type(b)
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
    Create an id to group notations together

    Args:
        parent: if given it will be prepended as {parent}/{groupid}

    Returns:
        the group id as string
    """
    groupid = str(uuid.uuid1())
    return groupid if parent is None else f'{parent}/{groupid}'


def makeNote(pitch: pitch_t,
             duration: time_t,
             offset: time_t = None,
             annotation='',
             gliss=False,
             withId=False,
             enharmonicSpelling='',
             dynamic='',
             **kws
             ) -> Notation:
    """
    Utility function to create a note Notation

    Args:
        pitch: the pitch as midinote or notename. If given a pitch as str,
            the note in question is fixed at the given enharmonic representation.
        duration: the duration of this Notation. Use 0 tp create a grace note
        offset: the offset of this Notation (None to leave unset)
        annotation: an optional text annotation for this note
        gliss: does this Notation start a glissando?
        withId: if True, this Notation has a group id and this id
            can be used to mark multiple notes as belonging to a same group
        enharmonicSpelling: if given, this spelling of pitch will be used
        dynamic: a dynamic such as 'p', 'mf', 'ff', etc.
        **kws: any keyword accepted by Notation

    Returns:
        the created Notation
    """
    duration = asF(duration)
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


def makeChord(pitches: Sequence[pitch_t],
              duration: time_t,
              offset: time_t = None,
              annotation: str | Text = '',
              dynamic='',
              fixed=False,
              **kws
              ) -> Notation:
    """
    Utility function to create a chord Notation

    Args:
        pitches: the pitches as midinotes or notenames. If given a note as str,
            the note in question is fixed at the given enharmonic representation.
        duration: the duration of this Notation. Use 0 to create a chord grace note
        offset: the offset of this Notation (None to leave unset)
        annotation: a text annotation
        dynamic: a dynamic for this chord
        fixed: if True, fix the spelling of any pitch given as notename
        **kws: any keyword accepted by Notation

    Returns:
        the created Notation
    """
    pitchlist = pitches if isinstance(pitches, list) else list(pitches)
    out = Notation(pitches=pitchlist, duration=duration, offset=offset,
                   dynamic=dynamic, **kws)
    if fixed:
        for i, pitch in enumerate(pitches):
            if isinstance(pitch, str):
                out.fixNotename(pitch, i)

    if annotation:
        if isinstance(annotation, str):
            if annotation.isspace():
                raise ValueError("Trying to add an empty annotation")
            out.addText(annotation)
        elif isinstance(annotation, Text):
            out.addAttachment(annotation)
        else:
            raise TypeError(f"Expected a str or Text, got {annotation}")
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
        duration: the duration of the rest
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
    Convert notations to their corresponding `maelzel.core` Note or Chord

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
            out.append(Rest(n.duration))
        else:
            if len(n.pitches) == 1:
                # note
                pitch = n.getFixedNotename(0) or n.pitches[0]
                event = Note(pitch=pitch,
                             dur=n.duration,
                             dynamic=n.dynamic,
                             tied=n.tiedNext,
                             fixed=isinstance(pitch, str),
                             gliss=n.gliss,
                             properties=n.properties,
                             )
                # TODO: add attachments, etc.
            else:
                # chord
                notenames = [n.getFixedNotename(i) or n.pitches[i]
                             for i in range(len(n))]
                event = Chord(notes=notenames,
                              dur=n.duration,
                              dynamic=n.dynamic,
                              tied=n.tiedNext,
                              gliss=n.gliss,
                              properties=n.properties)
            _transferAttachments(n, event)
            out.append(event)
    return out


def _transferAttachments(n: Notation, event: maelzel.core.MEvent) -> None:
    from maelzel.core import symbols
    if n.attachments:
        for attach in n.attachments:
            if isinstance(attach, Articulation):
                symbol = symbols.Articulation(attach.kind, placement=attach.placement,
                                              color=attach.color)
                event.addSymbol(symbol)
            elif isinstance(attach, Fermata):
                event.addSymbol(symbols.Fermata(kind=attach.kind))
            elif isinstance(attach, Harmonic):
                event.addSymbol(symbols.Harmonic(kind=attach.kind, interval=attach.interval))
            else:
                print(f"TODO: implemenet transfer for {attach}")

    if n.spanners:
        for spanner in n.spanners:
            logger.debug(f"Processing spanner {spanner}")
            if isinstance(spanner, _spanner.Slur):
                event.addSpanner(symbols.Slur(kind=spanner.kind, uuid=spanner.uuid))
            elif isinstance(spanner, _spanner.Hairpin):
                event.addSpanner(symbols.Hairpin(direction=spanner.direction, uuid=spanner.uuid,
                                                 kind=spanner.kind))
            else:
                print(f"TODO: implement transfer for {spanner}")


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
    # grid = F(1, den)
    # if (num == 3 or num == 7) and ((n0.offset % grid) > 0 or (n1.end % grid) > 0):

    if num not in {1, 2, 3, 4, 6, 7, 8, 12, 16, 32}:
        return False
    return True


def notationsCanMerge(n0: Notation, n1: Notation) -> bool:
    """
    Returns True if n0 and n1 can be merged

    Two Notations can merge if the resulting duration is regular. A regular
    duration is one which can be represented via **one** notation (a quarter,
    a half, a dotted 8th, a double dotted 16th are all regular durations,
    5/8 of a quarter is not)

    """
    if not n1.mergeable:
        return False

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

    if n1.attachments:
        if not n0.attachments:
            return False
        if not set(n1.attachments).issubset(set(n0.attachments)):
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


def tieNotations(notations: list[Notation]) -> None:
    """ Tie these notations inplace """

    for n in notations[:-1]:
        n.tiedNext = True

    hasGliss = notations[0].gliss
    for n in notations[1:]:
        n.tiedPrev = True
        n.dynamic = ''
        n.removeAttachments(lambda a: isinstance(a, (Text, Articulation)))
        if hasGliss:
            n.gliss = True


def splitNotationsAtOffsets(notations: list[Notation],
                            offsets: Sequence[F]
                            ) -> list[tuple[TimeSpan, list[Notation]]]:
    """
    Split the given notations between the given offsets

    **NB**: Any notations starting after the last offset will not be considered!

    Args:
        notations: the notations to split
        offsets: the boundaries.

    Returns:
        a list of tuples (timespan, notation)

    """
    timeSpans = [TimeSpan(beat0, beat1) for beat0, beat1 in pairwise(offsets)]
    splitEvents = []
    for ev in notations:
        if ev.duration > 0:
            splitEvents.extend(ev.splitAtOffsets(offsets))
        else:
            splitEvents.append(ev)

    eventsPerBeat = []
    for timeSpan in timeSpans:
        eventsInBeat = []
        for ev in splitEvents:
            if timeSpan.start <= ev.offset < timeSpan.end:
                assert ev.end <= timeSpan.end
                eventsInBeat.append(ev)
        eventsPerBeat.append(eventsInBeat)
        assert sum(ev.duration for ev in eventsInBeat) == timeSpan.end - timeSpan.start
        assert all(timeSpan.start <= ev.offset <= ev.end <= timeSpan.end
                   for ev in eventsInBeat)
    return list(zip(timeSpans, eventsPerBeat))


@dataclass
class SnappedNotation:
    notation: Notation
    offset: F
    duration: F

    def makeSnappedNotation(self, extraOffset: F | None = None) -> Notation:
        """
        Clone the original notation to be snapped to offset and duration

        Args:
            extraOffset: if given, an extra offset to apply to the snapped notation

        Returns:
            the snapped notation

        """
        offset = self.offset if not extraOffset else self.offset + extraOffset
        notation = self.notation.clone(offset=offset, duration=self.duration)
        if self.duration == 0 and self.notation.duration > 0:
            notation.setProperty('.snappedGracenote', True)
            notation.setProperty('.originalDuration', self.notation.duration)
        return notation

    def __repr__(self):
        return repr(self.makeSnappedNotation())



