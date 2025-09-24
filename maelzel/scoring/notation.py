"""
A Notation represents a note/chord/rest
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import pitchtools as pt
from itertools import pairwise
from emlib import mathlib

from maelzel.common import UNSET, UnsetType, F, F0, asF, asmidi
from maelzel._util import showF, showT, hasoverlap
from .common import (logger, NotatedDuration)
from . import attachment as att
from . import util
from . import definitions
from . import quantdata

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Sequence, Any, TypeVar
    import maelzel.core
    import maelzel.core.symbols
    import maelzel.core.mevent
    from maelzel.common import time_t, pitch_t
    from .common import division_t
    from . import spanner as _spanner
    AttachmentT = TypeVar('AttachmentT', bound=att.Attachment)
    from .quantdefs import QuantizedBeatDef


__all__ = (
    'Notation',
    'notationsToCoreEvents',
    'durationsCanMerge',
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
        fixNotenames: if True, pitches given as strings are fixed to the given spelling

    """
    _privateKeys = {
        '.clefHint',
        '.graceGroup',
        '.forceTupletBracket',
        '.snappedGracenote',   # Is this a note which has been snapped to 0 duration?
        '.originalDuration',    # For snapped notes, it is useful to keep track of the original duration
        '.forwardTies',
        '.backwardTies'
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
                 "mergeableNext",
                 "mergeablePrev",
                 "__weakref__",
                 "_symbolicDuration"
                 )

    def __init__(self,
                 duration: time_t,
                 pitches: Sequence[pitch_t],
                 offset: time_t | None = None,
                 isRest=False,
                 tiedPrev=False,
                 tiedNext=False,
                 dynamic: str = '',
                 durRatios: tuple[F, ...] = (),
                 group='',
                 gliss=False,
                 properties: dict[str, Any] | None = None,
                 fixNotenames=False,
                 _init=True
                 ):

        assert duration is not None
        assert pitches or isRest

        if _init:
            if dynamic:
                dynamic = definitions.normalizeDynamic(dynamic, '')

            if offset is not None:
                offset = asF(offset)

            duration = asF(duration)

            if isRest:
                tiedNext = False
                tiedPrev = False
                midinotes = ()
            else:
                pitches = sorted(pitches, key=asmidi)
                midinotes = tuple(asmidi(p) for p in pitches)

        else:
            assert isinstance(pitches, tuple)
            midinotes = pitches
            assert offset is None or isinstance(offset, F)

        if durRatios:
            assert isinstance(durRatios, tuple) and all(isinstance(r, F) for r in durRatios)

        self.duration: F = asF(duration)
        "The duration of this Notation, in quarternotes"

        self.pitches: tuple[float | int, ...] = midinotes  # type: ignore
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

        # self.color: str = color
        # "The color of this entire Notation"

        # self.sizeFactor: int = sizeFactor
        # "A size factor applied to this Notation (0: normal, 1: bigger, 2: even bigger, -1: smaller, etc.)"

        self.properties: dict[str, Any] | None = properties
        "A dict of user properties. To be set via setProperty"

        self.fixedNotenames: dict[int, str] | None = None
        "A dict mapping pitch index to spelling"

        self.attachments: list[att.Attachment] | None = None
        "Attachments are gathered here"

        self.spanners: list[_spanner.Spanner] | None = None
        "A list of spanners this Notations is part of"

        self.mergeablePrev = True
        "Can this Notation be merged with a previous Notation"

        self.mergeableNext = True
        "Can this Notation be merged with a next Notation"

        self._symbolicDuration: F = F0

        if self.isRest:
            assert self.duration > 0
            assert not self.pitches
        else:
            if not pitches or any(p <= 0 for p in self.pitches):
                raise ValueError(f"Invalid pitches: {self.pitches}")
            if fixNotenames:
                for i, n in enumerate(pitches):
                    if isinstance(n, str):
                        self.fixNotename(n, i)


    @classmethod
    def makeNote(cls,
                 pitch: pitch_t,
                 duration: time_t,
                 offset: time_t = None,
                 annotation='',
                 gliss=False,
                 withId=False,
                 enharmonicSpelling='',
                 dynamic='',
                 **kws
                 ) -> Notation:
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

    @classmethod
    def makeChord(cls,
                  pitches: Sequence[pitch_t],
                  duration: time_t,
                  offset: time_t = None,
                  annotation: str | att.Text = '',
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
            elif isinstance(annotation, att.Text):
                out.addAttachment(annotation)
            else:
                raise TypeError(f"Expected a str or Text, got {annotation}")
        return out

    @classmethod
    def makeRest(cls,
                 duration: time_t,
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
                       pitches=(),
                       _init=False)
        if annotation:
            out.addText(annotation)
        return out

    def pitchRange(self) -> tuple[float, float]:
        return self.pitches[0], self.pitches[-1]

    def isQuantized(self) -> bool:
        """Is this Notation quantized?"""
        return self.offset is not None and len(self.durRatios) > 0

    @property
    def qoffset(self) -> F:
        """Quantized offset, raises ValueError if this notation is not quantized"""
        if (offset := self.offset) is not None:
            return offset
        raise ValueError(f"This Notation does not have a fixed offset: {self}")

    def __hash__(self):
        attachhash = 0 if not self.attachments else hash(tuple(str(a) for a in self.attachments))
        pitcheshash = tuple(self.pitches) if self.pitches else 0
        return hash((self.duration, pitcheshash, self.tiedNext, self.tiedPrev,
                     self.dynamic, self.gliss, attachhash))

    def fusedDurRatio(self) -> F:
        """
        The fused duration ratio of this notation

        This is the result of applying all duration ratios this
        notation may have. This operation is only valid for quantized
        notations and will raise ValueError otherwise.

        Returns:
            the fused duration ratio

        Raises:
            ValueError: if this notation is not quantized

        .. seealso:: :meth:`Notation.isQuantized`
        """
        if not self.durRatios:
            raise ValueError(f"This Notation is not quantized, {self=}")
        num, den = 1, 1
        for ratio in self.durRatios:
            num *= ratio.numerator
            den *= ratio.denominator
        return F(num, den)

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
                       predicate: Callable | None = None,
                       anchor: int | None | UnsetType = UNSET
                       ) -> list[att.Attachment]:
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
                       cls: type[AttachmentT],
                       pitchanchor: int | None | UnsetType = UNSET,
                       ) -> AttachmentT | None:
        """
        Find an attachment by class or classname

        Similar to getAttachments, returns only one attachment or None

        Args:
            cls: the class to match (the class itself or its name, case is not relevant)
            pitchanchor: if given, the anchor index to match. Some attachments are anchored to
                None, meaning they are anchored to the entire Notation and not a specific
                pitch. For example, an AccidentalTrait which applies to an entire chord
                (for example, to force accidentals or set colors) can be anchored to None
                Setting this argument to None will filter out an AccidentalTrait
                anchored to a specific pitch.

        Returns:
            an Attachment matching the given criteria, or None
        """
        if not self.attachments:
            return None

        if pitchanchor is UNSET:
            return next((a for a in self.attachments if isinstance(a, cls)), None)
        else:
            return next((a for a in self.attachments if isinstance(a, cls) and a.anchor == pitchanchor), None)

    def addAttachment(self, attachment: att.Attachment, pitchanchor: int | None = None
                      ) -> Notation:
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
            pitchanchor: for pitch anchored symbols, the index of the pitch to add
                this attachment to. Alternatively the anchor can be set in the
                attachment itself

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
        if pitchanchor is not None:
            attachment.anchor = pitchanchor
        self.attachments.append(attachment)
        return self

    @property
    def isStemless(self) -> bool:
        """Is this Notation stemless?

        This property can be set by adding a StemTraits attachment
        """
        if self.attachments:
            attach = next((a for a in self.attachments if isinstance(a, att.StemTraits)), None)
            if attach is not None and attach.hidden:
                return True
        return False

    def setNotehead(self,
                    notehead: definitions.Notehead | str,
                    index: int | None = None,
                    merge=False
                    ) -> None:
        """
        Set a notehead in this notation

        Args:
            notehead: a Notehead or the notehead shape, as string (one of 'normal',
                'hidden', 'cross', 'harmonic', 'rhombus', 'square', etc.). See
                maelzel.scoring.definitions.noteheadShapes for a complete list
            index: the index, corresponding to the pitch at the same index,
                or None to set all noteheads
            merge: if True and there is already a Notehead set for the given index,
                the new properties are merged with the properties of the already
                existing notehead

        """
        if self.noteheads is None:
            self.noteheads = {}

        if isinstance(notehead, str):
            notehead = definitions.Notehead(shape=notehead)

        if index is not None:
            if not(0 <= index < len(self.pitches)):
                raise IndexError(f'Index {index} out of range. This notation has {len(self.pitches)} '
                                 f'pitches: {self.pitches}')
            indexes = [index]
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

    def addArticulation(self, articulation: str | att.Articulation, color='', placement='') -> Notation:
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
            return self.addAttachment(att.Articulation(articulation, color=color, placement=placement))
        else:
            assert isinstance(articulation, att.Articulation)
            if color or placement:
                articulation = articulation.copy()
                if color:
                    articulation.color = color
                if placement:
                    articulation.placement = placement
            return self.addAttachment(articulation)

    def removeAttachments(self, predicate: Callable[[att.Attachment], bool]) -> None:
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
        if not self.attachments:
            return
        if isinstance(cls, str):
            cls = cls.lower()
            self.attachments[:] = [a for a in self.attachments
                                   if not(type(a).__name__.lower() == cls)]
        else:
            self.attachments[:] = [a for a in self.attachments
                                   if not(isinstance(a, cls))]

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

    def addSpanner(self, spanner: _spanner.Spanner | str, end: Notation | None = None
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
            
        if isinstance(spanner, str):
            from . import spanner as _spanner
            spanner = _spanner.Spanner.fromStr(spanner)

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
                assert spanner.kind == 'start'
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

        harmonic = next((a for a in self.attachments if isinstance(a, att.Harmonic)), None)
        if not harmonic:
            logger.warning(f"Notation has no harmonic attachment: {self}")
            return self

        assert isinstance(harmonic, att.Harmonic)
        if harmonic.interval == 0:
            n = self.copy()
            n.setNotehead('harmonic')
            if n.attachments:
                n.attachments.remove(harmonic)
        else:
            fund = self.notename(0)
            touched = pt.transpose(fund, harmonic.interval)
            n = self.clone(pitches=(fund, touched))
            n.fixNotename(touched, index=1)
            n.setNotehead('harmonic', index=1)

        if removeAttachment and n.attachments:
            n.attachments = [a for a in n.attachments if not isinstance(a, att.Harmonic)]
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
        assert other is not self, f"Cannot transfer a spanner to self ({self=}, {spanner=}"

        print(f"transferring spanner {spanner} form {self} to {other}")
        other.addSpanner(spanner)
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

        if isinstance(spanner, str):
            for spannerobj in (s for s in self.spanners if s.uuid == spanner):
                self.removeSpanner(spannerobj)
        else:
            if spanner.parent and spanner.parent is not self:
                logger.error(f"This spanner {spanner} has a different parent! parent={spanner.parent}, self={self}")
            spanner.parent = None
            self.spanners.remove(spanner)

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
        """
        Create a Notation representing an artificial harmonic

        This is mainly circumscribed to string instruments

        Args:
            basepitch: the pitch to press down
            interval: the interval over the base pitch
            **kws: any kws passed to the Notation constructor

        Returns:
            a chord representing the given artificial harmonic, consisting
            of two pitches, the base pitch and the half pressed pitch,
            where the higher pitch has a harmonic notehead. In both cases
            the enharmonic spelling is fixed in order to keep the interval
        """
        if not isinstance(basepitch, str):
            basepitch = pt.m2n(basepitch)
        touchpitch = pt.transpose(basepitch, interval)
        n = cls(pitches=[basepitch, touchpitch], **kws)
        n.fixNotename(basepitch, 0)
        n.fixNotename(touchpitch, 1)
        n.setNotehead(definitions.Notehead('harmonic'), index=1)
        return n

    def clearFixedNotenames(self) -> None:
        """
        Remove any fixed enharmonic spelling set for this notation

        """
        if self.fixedNotenames:
            self.fixedNotenames.clear()

    def fixNotename(self, notename: str, index: int | None = None) -> None:
        """
        Fix the spelling for the pitch at index **inplace**

        Args:
            notename: if given, it will be fixed to the given notename.
                If nothing is given, it will be fixed to n2m(self.pitches[idx])
            index: the index of the note to modify. If None, a matching pitch in this notation
                is searched. ValueError is raised if no pitch is found

        .. seealso:: :meth:`Notation.notenames`
        """
        if self.fixedNotenames is None:
            self.fixedNotenames = {}

        tolerance = 0.04

        if index is None:
            if len(self.pitches) == 1:
                index = 0
            else:
                spellingPitch = pt.n2m(notename)
                index = next((idx for idx in range(len(self.pitches))
                              if abs(spellingPitch - self.pitches[idx]) < tolerance), None)
                if index is None:
                    raise ValueError(f"No pitch in this notation matches the given notename {notename}={pt.n2m(notename)}"
                                     f" (pitches: {self.pitches=}, {index=}, {[pt.m2n(p) for p in self.pitches]})")

        self.fixedNotenames[index] = notename

    def getFixedNotename(self, idx: int = 0) -> str | None:
        """
        Returns the fixed notename of this notation, if any

        Args:
            idx: 0 in the case of a note, the index of the note if representing a chord

        Returns:
            the fixed spelling of the note, if exists (None otherwise)

        """
        return self.fixedNotenames.get(idx) if self.fixedNotenames else None

    def tieHints(self, direction='forward', clear=False) -> set[int]:
        """
        Get any tie hints set

        Tie hints indicate which pitches within a chord are actually tied
        to the next/previous notation. This can be set manually and is
        set after quantization

        Args:
            direction: one of "forward" or "backward"
            clear: if True, clear the set if applicable
        """
        if direction == 'forward':
            key = '.forwardTies'
            assert self.tiedNext
        elif direction == 'backward':
            key = '.backwardTies'
            assert self.tiedPrev
        else:
            raise KeyError(f"direction must be one of 'forward', 'backward', got {direction}")
        hints = self.getProperty(key)
        if hints is None:
            hints = set()
            self.setProperty(key, hints)
        else:
            assert isinstance(hints, set)
            if clear:
                hints.clear()
        return hints

    def tiedPitches(self, direction='forward') -> tuple[float, ...] | None:
        """
        The list of tied pitches or None if not tied or ties not set

        Args:
            direction:

        Returns:

        """
        if direction == 'forward':
            if not self.tiedNext:
                return None
        else:
            if not self.tiedPrev:
                return None
        if len(self.pitches) == 1:
            return self.pitches
        else:
            hints = self.tieHints(direction=direction)
            if not hints:
                return None
            return tuple(self.pitches[idx] for idx in hints)

    def setTieHint(self, idx: int, direction="forward") -> None:
        """
        Set a tie hint for a specific pitch in this notation

        Args:
            idx: the index of the pitch
            direction: one of "forward" or "backward"
        """
        if direction == "forward" and not self.tiedNext:
            raise ValueError(f"This Notation is not tied forward: {self}")
        elif direction == "backward" and not self.tiedPrev:
            raise ValueError(f"This Notation is not tied backward: {self}")
        self.tieHints(direction).add(idx)

    def getTieHint(self, idx: int, direction="forward") -> bool:
        """
        True if the pitch with the given idx has a tie hint set

        Args:
            idx: the index of the pitch
            direction: one of "forward or "backward"
        """
        if idx < 0 or idx >= len(self.pitches):
            raise ValueError(f"idx {idx} out of range, {self.pitches=}")
        return idx in self.tieHints(direction)

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
        self.pitches = tuple(asmidi(p) for p in pitches) if pitches else ()
        if resetFixedNotenames:
            self.fixedNotenames = None

    def copyFixedSpellingTo(self, other: Notation):
        """Copy fixed spelling to *other*"""
        if not self.fixedNotenames:
            return
        for notename in self.fixedNotenames.values():
            if pt.n2m(notename) in other.pitches:
                other.fixNotename(notename, index=None)

    def clone(self, copyFixedNotenames=True, spanners=True, **kws) -> Notation:
        """
        Clone this Notation, overriding any value.

        Args:
            copyFixedNotenames: transfer any fixed notenames to the cloned
                notation
            kws: keyword arguments, as passed to the Notation constructor.
                Any parameter given will override the corresponding value in
                this Notation
        """
        if noteheads := kws.get('noteheads'):
            assert isinstance(noteheads, dict), f'{self=}, {noteheads=}'

        out = self.copy(spanners=spanners)
        if (pitches := kws.pop('pitches', None)) is not None:
            out._setPitches(pitches)  # type: ignore
            if self.fixedNotenames and copyFixedNotenames:
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

    def asRest(self, dynamic=False) -> Notation:
        """
        Clone this Notations as a rest

        Args:
            dynamic: if True, add a dynamic to the rest if self has a dynamic

        Returns:
            a notation representing a rest with the same offset, duration and
            any other attribute set for this notation which can be applied to
            a rest
        """
        return self.__class__(isRest=True,
                              duration=self.duration,
                              offset=self.offset,
                              dynamic=self.dynamic if dynamic else '',
                              pitches=())

    def cloneAsTie(self,
                   duration: F,
                   offset: F | None,
                   tiedPrev=True,
                   tiedNext: bool | None = None,
                   gliss: bool | None = None,
                   ) -> Notation:
        """
        Clone self so that the cloned Notation can be used within a logical tie

        The returned notation is thought to be tied to self, as a continuation.
        This is used when a notation is split across a measure or a beam
        or within a tuplet

        Returns:
            The cloned Notation
        """

        if self.isRest:
            return Notation(isRest=True,
                            duration=duration,
                            offset=offset,
                            durRatios=self.durRatios,
                            pitches=())

        out = Notation(duration=duration,
                       offset=offset,
                       pitches=self.pitches,
                       tiedPrev=tiedPrev,
                       tiedNext=tiedNext if tiedNext is not None else self.tiedNext,
                       dynamic='',
                       gliss=gliss if gliss is not None else self.gliss,
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

    def copy(self, spanners=True) -> Notation:
        """Copy this Notation"""
        properties = None if self.properties is None else copy.deepcopy(self.properties)
        out = Notation(duration=self.duration,
                       pitches=self.pitches,
                       offset=self.offset,
                       isRest=self.isRest,
                       tiedPrev=self.tiedPrev,
                       tiedNext=self.tiedNext,
                       dynamic=self.dynamic,
                       durRatios=self.durRatios,
                       group=self.groupid,
                       gliss=self.gliss,
                       properties=properties,
                       _init=False)
        if self.attachments:
            out.attachments = self.attachments.copy()

        if self.fixedNotenames:
            out.fixedNotenames = self.fixedNotenames.copy()

        if self.noteheads:
            out.noteheads = self.noteheads.copy()

        out.mergeableNext = self.mergeableNext
        out.mergeablePrev = self.mergeablePrev

        if spanners and self.spanners:
            out.spanners = self.spanners.copy()
        return out

    def _breakIrregularDurationInBeat(self: Notation,
                                      beatDur: F,
                                      beatDivision: int | division_t,
                                      beatOffset: F = F0
                                      ) -> list[Notation] | None:
        """
        Breaks a notation with irregular duration into its parts during quantization

        - a Notations should not extend over a subdivision of the beat if the
          subdivisions in question are coprimes
        - within a subdivision, a Notation should not result in an irregular multiple of the
          subdivision. Irregular multiples are all numbers which have prime factors other than
          2 or can be expressed with a dot
          Regular durations: 2, 3, 4, 6, 7 (double dotted), 8, 12, 16, 24, 32
          Irregular durations: 5, 9, 10, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27,
          28, 29, 30, 31

        Args:
            beatDur: the duration of the beat
            beatDivision: the division of the beat, either a division tuple or an int
            beatOffset: the offset of the beat

        Returns:
            a list of tied Notations representing the original notation, or None
            if the notation does not need to be split into parts

        Raises:
            ValueError if the notation cannot be split

        """
        if not beatOffset <= self.qoffset and self.end <= beatOffset + beatDur:
            raise ValueError(f"The notation should be defined within the given beat boundaries, "
                             f"got {self=}, {beatOffset=}, {beatDur=}")

        if self.duration == 0:
            return None
        elif self.isQuantized() and self.hasRegularDuration():
            return None

        if isinstance(beatDivision, (tuple, list)) and len(beatDivision) == 1:
            beatDivision = beatDivision[0]

        if isinstance(beatDivision, int):
            return _breakIrregularDurationInSimpleDivision(self, beatDur=beatDur, div=beatDivision, beatOffset=beatOffset)

        # beat is not subdivided regularly. check if n extends over subdivision
        numDivisions = len(beatDivision)
        divDuration = beatDur / numDivisions

        ticks = list(mathlib.fraction_range(beatOffset, beatOffset+beatDur+divDuration, divDuration))
        assert len(ticks) == numDivisions + 1

        subdivisionTimespans = list(pairwise(ticks))
        subdivisions = list(zip(subdivisionTimespans, beatDivision))
        subns = self.splitAtOffsets(ticks)
        allparts: list[Notation] = []
        for subn in subns:
            # find the subdivision
            for timespan, numslots in subdivisions:
                if hasoverlap(timespan[0], timespan[1], subn.qoffset, subn.end):
                    if self.duration == 0 or (self.isQuantized() and self.hasRegularDuration()):
                        allparts.append(self)
                    else:
                        parts = _breakIrregularDurationInBeat(n=subn, beatDur=divDuration, beatDivision=numslots, beatOffset=timespan[0])
                        if parts:
                            allparts.extend(parts)
                        else:
                            allparts.append(subn)
        assert sum(part.duration for part in allparts) == self.duration
        Notation.tieNotations(allparts)
        return allparts

    def breakIrregularDurationInNode(self: Notation, beatstruct: Sequence[QuantizedBeatDef]) -> list[Notation]:
        # this is called on each part of a notation when split at a beat boundary
        assert self.duration > 0
        assert self.isQuantized() and not self.hasRegularDuration()
        from maelzel.scoring import util
        beatoffsets = [b.offset for b in beatstruct]
        fragments = util.splitInterval(self.qoffset, self.end, beatoffsets)
        N = len(fragments)
        assert N > 0,  f"??? {self=}, {beatoffsets=}"
        if N == 1:
            # does not cross any beats
            beat = next((b for b in beatstruct if b.offset <= self.qoffset and self.end <= b.end), None)
            assert beat is not None, f"Could not find beat for {self}, beats={beatstruct}"
            parts = self._breakIrregularDurationInBeat(beatDur=beat.duration, beatDivision=beat.division, beatOffset=beat.offset)
            assert parts is not None
            return parts
        elif N == 2:
            n0, n1 = self.splitAtOffset(fragments[1][0])
            parts = []
            for part in (n0, n1):
                if part.hasRegularDuration():
                    parts.append(part)
                else:
                    parts.extend(Node.breakIrregularDurationInNode(part, beatstruct=beatstruct))
            Notation.tieNotations(parts)
            return parts
        else:
            parts = []
            offset0, end0 = fragments[0]
            offset1, end1 = fragments[1][0], fragments[-2][1]
            offset2, end2 = fragments[-1]
            n0 = self.clone(offset=offset0, duration=end0 - offset0, spanners=False)
            n1 = self.clone(offset=offset1, duration=end1 - offset1, spanners=False)
            n2 = self.clone(offset=offset2, duration=end2 - offset2, spanners=False)
            for part in (n0, n1, n2):
                if part.hasRegularDuration():
                    parts.append(part)
                else:
                    parts.extend(Node.breakIrregularDurationInNode(part, beatstruct=beatstruct))
            Notation.tieNotations(parts)
            self._copySpannersToSplitNotation(parts)
            return parts
        
    @staticmethod
    def splitNotationsAtOffsets(notations: list[Notation],
                                offsets: Sequence[F],
                                forcecopy=False,
                                nomerge=False
                                ) -> list[Notation]:
        """
        Split notations at the given offsets.

        The returned notations do not extend over the offsets

        Args:
            notations: the notations to split. Their offset must be set
            offsets: the offsets at which to split the notations. The offsets must be sorted in ascending order.
            forcecopy: if True, all notations are copied even if they are not split
            nomerge: if True, mark the split notations as not mergeable

        Returns:
            A list of notations that do not extend over the offsets.
        """
        out = []
        for n in notations:
            if n.duration == 0:
                out.append(n if not forcecopy else n.copy())
            else:
                assert n.offset is not None, f"Notation.offset must be set for {n}"
                if any(n.offset < offset < n.end for offset in offsets):
                    out.extend(n.splitAtOffsets(offsets, nomerge=nomerge))
                else:
                    out.append(n if not forcecopy else n.copy())
        return out

    @staticmethod
    def tieNotations(notations: list[Notation]) -> None:
        _tieNotations(notations)

    def splitAtOffset(self, offset: F, tie=True, nomerge=False) -> tuple[Notation, Notation]:
        """
        Split this notations at the given offset

        Here we do not check if the resulting parts have a correct quantization
        or a regular duration

        Args:
            offset: the offset to split this notation at
            tie: if True, tie the returned notations
            nomerge: if True, mark the split notes as not mergeable between them

        Returns:
            a tuple of two notations, the part left to the offset and
            the part right to the offset.

        Raises:
            ValueError: if offset does not split this notation
        """
        assert self.offset is not None
        if not (self.offset < offset < self.end):
            raise ValueError(f"Offset {offset} is not contained within this notations: {self}")

        left = self.clone(offset=self.offset, duration=offset - self.offset)
        right = self.clone(offset=offset, duration=self.end - offset)

        left.mergeablePrev = self.mergeablePrev
        left.mergeableNext = not nomerge
        right.mergeableNext = self.mergeableNext
        right.mergeablePrev = not nomerge

        if tie:
            left.tiedNext = True
            right.tiedPrev = True

        if self.spanners is not None:
            right.spanners = [sp for sp in self.spanners if sp.kind != 'start']
            left.spanners = [sp for sp in self.spanners if sp.kind != 'end']
            assert {_.uuid for _ in left.spanners}.isdisjoint({_.uuid for _ in right.spanners})

        return left, right

    def splitAtOffsets(self: Notation, offsets: Sequence[F], nomerge=False
                       ) -> list[Notation]:
        """
        Splits a Notation at the given offsets

        Args:
            offsets: the offsets at which to split n. The offsets must be sorted in ascending order.
            nomerge: if True, mark the parts as not-mergeable

        Returns:
            the parts after splitting

        Example::

            >>> splitAtOffsets(Notation(F(0.5), duration=F(1)))
            [Notation(0.5, duration=0.5), Notation(1, duration=0.5)]

        """
        if not offsets:
            raise ValueError("offsets is empty")

        assert self.offset is not None
        intervals = util.splitInterval(self.offset, self.end, offsets)

        if len(intervals) == 1:
            return [self]

        start0, end0 = intervals[0]
        parts: list[Notation] = [self.clone(offset=start0, duration=end0-start0)]
        parts.extend((self.cloneAsTie(offset=start, duration=end - start)
                      for start, end in intervals[1:]))

        _tieNotations(parts)
        first, last = parts[0], parts[-1]

        if self.spanners:
            self._copySpannersToSplitNotation(parts)

        first.tiedPrev = self.tiedPrev
        first.mergeablePrev = self.mergeablePrev
        last.tiedNext = self.tiedNext
        last.mergeableNext = self.mergeableNext

        if nomerge:
            for part in parts[:-1]:
                part.mergeableNext = False
            for part in parts[1:]:
                part.mergeablePrev = False

        assert sum(part.duration for part in parts) == self.duration
        return parts

    def hasRegularDuration(self) -> bool:
        """
        Does this notations have a regular duration?

        This is only valid if the notation has been quantized.

        Returns:
            True is the duration of this Notation is regular (the symbolic
            duration can be represented by ONE rhythmic figure, ie a ¼ note,
            ⅛ note, etc., with or without dots. This is independent of the
            notation beeing part of a tuplet.

        Raises:
            ValueError: if the notation has not been quantized
        """
        symdur = self.symbolicDuration()
        return symdur.denominator in (1, 2, 4, 8, 16, 32) and symdur.numerator in (1, 2, 3, 4, 7)

    def symbolicDuration(self) -> F:
        """
        The symbolic duration of this Notation.

        This method can only be called for quantized notations.
        This represents the notated figure (1=quarter, 1/2=eighth note,
        1/4=16th note, etc)

        Raises:
            ValueError: if the notation has not been quantized
        """
        if self._symbolicDuration > 0:
            return self._symbolicDuration
        self._symbolicDuration = dur = self.duration * self.fusedDurRatio()
        return dur

    def setPitches(self, pitches: Sequence[float | str], fixNotenames=False) -> None:
        """
        Set the pitches of this notation, in place

        Args:
            pitches: a list of midinotes or notenames, or any combination
            fixNotenames: if True, fix the notenames for those pitches given
                as strings. Notenames can also be fixed with a '!' suffix
        """
        self.clearFixedNotenames()
        pitches = sorted(pitches, key=lambda p: pt.n2m(p) if isinstance(p, str) else p)
        midinotes = tuple(p if isinstance(p, (int, float)) else pt.n2m(p)
                          for p in pitches)
        self.pitches = midinotes
        if fixNotenames:
            for i, pitch in enumerate(pitches):
                if isinstance(pitch, str):
                    if pitch[-1] == '!':
                        pitch = pitch[:-1]
                    self.fixNotename(pitch, index=i)
        else:
            for i, pitch in enumerate(pitches):
                if isinstance(pitch, str) and pitch[-1] == '!':
                    self.fixNotename(pitch[:-1], index=i)

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
            return fixed if not addExplicitMark else fixed + '!'
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

    def resolveNotenames(self, addFixedAnnotation=False, removeFixedAnnotation=False
                         ) -> list[str]:
        """Resolve the enharmonic spellings for this Notation

        Args:
            addFixedAnnotation: if True, enforce the returned spelling by adding
                a '!' suffix.
            removeFixedAnnotation: if True, remove any fixed annotation marks ('!')
                from the notenames

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
            if notename.endswith('!'):
                if removeFixedAnnotation:
                    notename = notename[:-1]
            elif addFixedAnnotation:
                notename += '!'
            out.append(notename)
        return out

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
                text: str | att.Text,
                placement='above',
                fontsize: int | float | None = None,
                italic=False,
                weight='normal',
                fontfamily='',
                box: str | bool = False,
                exclusive=False,
                role='',
                relativeSize=False
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
            relativeSize: if True, the font size is relative to the staff size
        """
        if isinstance(text, att.Text):
            assert not text.text.isspace()
            annotation = text
        else:
            assert not text.isspace()
            annotation = att.Text(text=text,
                                  placement=placement,
                                  fontsize=fontsize,
                                  fontfamily=fontfamily,
                                  italic=italic,
                                  weight=weight,
                                  box=box,
                                  role=role,
                                  relativeSize=relativeSize)
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

    def mergeWith(self, other: Notation, check=True) -> Notation:
        """Merge this Notation with ``other``"""
        return _mergeNotations(self, other, check=check)

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

    def _setClefHint(self, clef: str, index: int | None = None) -> None:
        """
        Sets a hint regarding which clef to use for this notation

        .. warning::

            This is mostly used internally for the case where two notations
            are bound by a glissando, and they should be placed together,
            even if the pitch of some of them might indicate otherwise

        Args:
            clef: the clef to set, one of 'treble', 'bass' or 'treble8', 'treble15' or 'bass8'
            index: the index of the pitch within a chord, or None to apply to
                the whole notation

        """
        normalizedclef = definitions.clefs.get(clef)
        if normalizedclef is None:
            raise ValueError(f"Clef {clef} not known. Possible clefs: {definitions.clefs.keys()}")
        if index is None:
            self.setProperty('.clefHint', normalizedclef)
        else:
            hint = self.getProperty('.clefHint', {})
            hint[index] = normalizedclef
            self.setProperty('.clefHint', hint)

    def _clearClefHints(self) -> None:
        """Remove any clef hints from this Notation

        .. seealso:: :meth:`Notation.getClefHint`, :meth:`Notation.setClefHint`"""
        self.delProperty('.clefHint')

    def _getClefHint(self, index: int = 0) -> str | None:
        """
        Get any clef hint for this notation or a particular pitch thereof

        .. warning::

            This is mostly used internally and is an experimental feature which
            might be implemented using other means in the future

        Args:
            index: which pitch index to query

        Returns:
            the clef hint, if any

        """
        hints = self.getProperty('.clefHint')
        if not hints:
            return None
        elif isinstance(hints, str):
            return hints
        else:
            return hints.get(index)

    def _namerepr(self) -> str:
        if self.isRest:
            return 'r'
        if len(self.pitches) > 1:
            s = "[" + " ".join(self.resolveNotenames()) + "]"
        else:
            s = self.resolveNotenames()[0]
        if self.tiedPrev:
            s = f"~{s}"
        if not self.mergeablePrev:
            s = "|" + s
        if self.tiedNext:
            s += "~"
        if not self.mergeableNext:
            s += "|"
        if self.gliss:
            s += ":gliss"
        return s

    def __repr__(self):
        info = []
        info.append(self._namerepr())
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

        if self.durRatios and self.durRatios != (F(1),):
            info.append(",".join(showF(r) for r in self.durRatios))

        if self.dynamic:
            info.append(self.dynamic)
        if self.noteheads:
            descrs = [f'{i}:{n.description()}' for i, n in self.noteheads.items()]
            info.append(f'noteheads={descrs}')

        for attr in ('attachments', 'properties', 'spanners'):
            val = getattr(self, attr)
            if val:
                info.append(f"{attr}={val}")

        infostr = " ".join(info)
        if self.isQuantized():
            return f"«{infostr}»"
        return f"‹{infostr}›"

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
                dest.addAttachment(a.copy())

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
    
    @staticmethod
    def mergeNotationsIfPossible(notations: list[Notation]) -> list[Notation]:
        """
        Merge the given notations into one, if possible.
    
        Notations which cannot be merged are added to the returned list
    
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
        return _mergeNotationsIfPossible(notations)

    def canMergeWith(self, n1: Notation) -> bool:
        """
        Returns True if self and n1 can be merged

        Two Notations can merge if they are quantized and the resulting
        duration is regular. A regular duration is one which can be
        represented via **one** notation (a quarter, a half, a dotted 8th,
        a double dotted 16th are all regular durations, 5/8 of a quarter is not)

        """
        if not self.mergeableNext or not n1.mergeablePrev:
            return False

        if (self.isRest != n1.isRest):
            return False

        quantized = self.isQuantized()

        if quantized != n1.isQuantized():
            raise ValueError(f"A quantized notation cannot be merged with an "
                             f"unquantized notation: {self=}, {n1=}")

        if quantized:
            if self.durRatios != n1.durRatios or not durationsCanMerge(self.symbolicDuration(), n1.symbolicDuration()):
                return False

        if self.isRest and n1.isRest:
            canmerge = (not n1.dynamic and not n1.attachments and not n1.noteheads)
            n0spanners = self.spanners and any(sp.kind == 'end' for sp in self.spanners)
            n1spanners = n1.spanners and any(sp.kind == 'start' for sp in n1.spanners)
            canmerge = canmerge and not (n1spanners or n0spanners)
            return canmerge

        # Two notes/chords

        # TODO: decide what to do about spanners
        if (not self.tiedNext or
            not n1.tiedPrev or
            self.durRatios != n1.durRatios or
            self.pitches != n1.pitches or
            self.noteheads != n1.noteheads or
                (n1.dynamic and n1.dynamic != self.dynamic)

        ):
            return False

        if n1.attachments:
            if not self.attachments:
                return False
            if not set(n1.attachments).issubset(set(self.attachments)):
                return False

        if not self.gliss and (self.noteheads or n1.noteheads) and self.noteheads != n1.noteheads:
            if not n1.noteheads:
                return False

            n1visiblenoteheads = {idx: notehead for idx, notehead in n1.noteheads.items()
                                  if not notehead.hidden}
            if self.noteheads != n1visiblenoteheads:
                return False

        if not self.gliss and n1.gliss:
            return False

        return True

    def _copySpannersToSplitNotation(self, parts: list[Notation]) -> None:
        if not self.spanners:
            return
        assert self.offset == parts[0].offset and self.duration == sum(p.duration for p in parts)
        parts[0].spanners = [_ for _ in self.spanners if _.kind == 'start']
        parts[-1].spanners = [_ for _ in self.spanners if _.kind == 'end']
        for p in parts[1:-1]:
            p.spanners = None

    def extractPartialNotation(self, indexes: list[int], spanners=True) -> Notation:
        """
        Extract part of a chord with any attachments corresponding to the given pitches

        Args:
            indexes: the indexes of the pitches to extract
            spanners: add any spanners in self to the extracted notation

        Returns:
            a new Notation with the given pitches
        """
        indexes.sort()
        pitches = [self.pitches[index] for index in indexes]
        mappedIndexes = {idx: indexes.index(idx) for idx in indexes}
        if self.noteheads:
            noteheads = {}
            for index in indexes:
                if (notehead := self.noteheads.get(index)) is not None:
                    noteheads[mappedIndexes[index]] = notehead
        else:
            noteheads = None
        fixedNotenames = {}
        if self.fixedNotenames:
            for index in indexes:
                if (notename := self.fixedNotenames.get(index)) is not None:
                    fixedNotenames[mappedIndexes[index]] = notename

        attachments = []
        if self.attachments:
            for att in self.attachments:
                if att.anchor is not None and att.anchor in indexes:
                    anchor = mappedIndexes[att.anchor]
                    att = copy.copy(att)
                    att.anchor = anchor
                    attachments.append(att)
                elif att.anchor is None:
                    attachments.append(att)

        out = self.clone(pitches=pitches,
                         noteheads=noteheads,
                         spanners=spanners)
        out.fixedNotenames = fixedNotenames
        out.attachments = attachments
        # self.copyFixedSpellingTo(out)
        out._clearClefHints()
        for idx in indexes:
            if hint := self._getClefHint(idx):
                out._setClefHint(hint, mappedIndexes[idx])
        return out


def _mergeNotations(a: Notation, b: Notation, check=True) -> Notation:
    """
    Merge two compatible notations to one.

    For two notations to be mergeable they need to:

    - be adjacent or have unset offset
    - have a duration
    - have the same pitch/pitches.
    - both need to be quantized or both need to be not quantized

    All other attributes are taken from the first notation and the
    duration of this first notation is extended to cover both notations
    """
    if check:
        assert type(a) is type(b), f"{a=}, {b=}"
        if a.pitches != b.pitches:
            raise ValueError("Attempting to merge two Notations with "
                             "different pitches")
        assert a.duration is not None and b.duration is not None
        if a.isRest != b.isRest:
            raise ValueError(f"Cannot merge a notation with a rest, {a=}, {b=}")
    quantized = a.isQuantized()
    if quantized != b.isQuantized():
        raise ValueError(f"Cannot merge a quantized and an unquantized notation, {a=}, {b=}")

    out = a.clone(duration=a.duration + b.duration,
                  tiedNext=b.tiedNext,
                  mergeableNext=b.mergeableNext)

    assert out.duration == a.duration + b.duration

    if quantized:
        assert a.end == b.offset
        if not out.hasRegularDuration():
            raise ValueError(f"Cannot merge {a=} with {b=}, the resulting notation does not"
                             f" hava a regular duration: {out}")

    if b.fixedNotenames:
        b.copyFixedSpellingTo(out)

    spanners = mergeSpanners(a, b)
    out.spanners = spanners
    out.mergeableNext = b.mergeableNext
    out.mergeablePrev = a.mergeablePrev
    return out


def mergeSpanners(a: Notation, b: Notation
                  ) -> list[_spanner.Spanner] | None:
    """
    Merge the spanner of two Notations

    We assume that a and b are to be merged. At this stage we just merge everything
    since we cannot decide here if there are spanners which need to be removed
    or not transferred...

    Shared spanners (for example, a crescendo from a to b) are removed

    Args:
        a: the first notation
        b: the second notation

    Returns:
        a list of merged spanners, or None if both a and b have no spanners
    """
    if a.spanners and b.spanners:
        return a.spanners + b.spanners
    else:
        return a.spanners or b.spanners


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
            event = Rest(dur=n.duration, dynamic=n.dynamic)
        elif len(n.pitches) == 1:
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


def _transferAttachments(source: Notation, dest: maelzel.core.mevent.MEvent) -> None:
    """
    Transfer attachments from a Notation object to a MEvent object.

    Args:
        source (Notation): The Notation object to transfer attachments from.
        dest (MEvent): The MEvent object to transfer attachments to.

    Returns:
        None
    """
    from maelzel.core import symbols
    from . import spanner as _spanner
    
    if source.attachments:
        for attach in source.attachments:
            if isinstance(attach, att.Articulation):
                symbol = symbols.Articulation(attach.kind, placement=attach.placement,
                                              color=attach.color)
                dest.addSymbol(symbol)
            elif isinstance(attach, att.Fermata):
                dest.addSymbol(symbols.Fermata(kind=attach.kind))
            elif isinstance(attach, att.Harmonic):
                dest.addSymbol(symbols.Harmonic(kind=attach.kind, interval=attach.interval))
            else:
                print(f"TODO: implemenet transfer for {attach}")

    if source.spanners:
        for spanner in source.spanners:
            if isinstance(spanner, _spanner.Slur):
                dest.addSpanner(symbols.Slur(kind=spanner.kind, uuid=spanner.uuid, linetype=spanner.linetype, color=spanner.color))
            elif isinstance(spanner, _spanner.Hairpin):
                dest.addSpanner(symbols.Hairpin(direction=spanner.direction, uuid=spanner.uuid,
                                                kind=spanner.kind))
            else:
                raise ValueError(f"Spanner {spanner} not implemented yet")


def durationsCanMerge(symbolicdur1: F, symbolicdur2: F) -> bool:
    """
    True if these symbolic durations can merge

    Two durations can be merged if their sum is regular, meaning
    the sum has a numerator of 1, 2, 3, 4, or 7 (3 means a dotted
    note, 7 a double dotted note) and the denominator is <= 64
    (1/1 being a quarter note)

    Args:
        symbolicdur1: first symbolic duration
        symbolicdur2: seconds symbolic duration

    Returns:
        True if they can be merged
    """
    assert mathlib.ispowerof2(symbolicdur1.denominator) and mathlib.ispowerof2(symbolicdur2.denominator), f"{symbolicdur1=}, {symbolicdur2}="
    sumdur = symbolicdur1 + symbolicdur2
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


def _mergeNotationsIfPossible(notations: list[Notation]) -> list[Notation]:
    """
    Merge the given notations into one, if possible.

    Notations which cannot be merged are added to the returned list

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
        last = out[-1]
        if last.canMergeWith(n1):
            out[-1] = last.mergeWith(n1)
        else:
            out.append(n1)
    assert len(out) <= len(notations)
    assert sum(n.duration for n in out) == sum(n.duration for n in notations)
    return out


def _tieNotations(notations: list[Notation]) -> None:
    """ Tie these notations inplace """
    for n in notations[:-1]:
        n.tiedNext = True

    n0 = notations[0]
    hasGliss = n0.gliss
    for n in notations[1:]:
        n.tiedPrev = True
        n.dynamic = ''
        n.removeAttachments(lambda a: isinstance(a, (att.Text, att.Articulation)))
        if hasGliss:
            n.gliss = True


@dataclass
class SnappedNotation:
    """
    Represents a notation that has been snapped to a specific offset and duration.

    Attributes:
        notation: the original notation
        offset: the offset of the snapped notation
        duration: the duration of the snapped notation

    """
    notation: Notation
    offset: F
    duration: F

    def applySnap(self, extraOffset: F | None = None) -> Notation:
        """
        Clone the original notation to be snapped to offset and duration

        Args:
            extraOffset: if given, an extra offset to apply to the snapped notation

        Returns:
            the snapped notation

        """
        offset = self.offset if not extraOffset else self.offset + extraOffset
        notation = self.notation.clone(offset=offset, duration=self.duration)
        notation.spanners = self.notation.spanners
        if self.duration == 0 and self.notation.duration > 0:
            if notation.isRest and not notation.hasAttributes():
                raise ValueError(f"A rest should not be snapped to a gracenote: {self=}, {self.notation=}")
            notation.setProperty('.snappedGracenote', True)
            notation.setProperty('.originalDuration', self.notation.duration)
        return notation

    def __repr__(self):
        return f"SnappedNotation(notation={self.notation}, offset={self.offset}, duration={self.duration})"


def _breakIrregularDurationInBeat(n: Notation,
                                  beatDur: F,
                                  beatDivision: int | division_t,
                                  beatOffset: F = F0
                                  ) -> list[Notation] | None:
    """
    Breaks a notation with irregular duration into its parts

    - a Notations should not extend over a subdivision of the beat if the
      subdivisions in question are coprimes
    - within a subdivision, a Notation should not result in an irregular multiple of the
      subdivision. Irregular multiples are all numbers which have prime factors other than
      2 or can be expressed with a dot
      Regular durations: 2, 3, 4, 6, 7 (double dotted), 8, 12, 16, 24, 32
      Irregular durations: 5, 9, 10, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27,
      28, 29, 30, 31

    Args:
        n: the Notation to break
        beatDur: the duration of the beat
        beatDivision: the division of the beat, either a division tuple or an int
        beatOffset: the offset of the beat

    Returns:
        a list of tied Notations representing the original notation, or None
        if the notation does not need to be split into parts

    Raises:
        ValueError if the notation cannot be split

    """

    assert beatOffset <= n.qoffset and n.end <= beatOffset + beatDur, f"{n=}, {beatOffset=}, {beatDur=}"

    if n.duration == 0:
        return None
    elif n.isQuantized() and n.hasRegularDuration():
        return None
    
    if isinstance(beatDivision, (tuple, list)) and len(beatDivision) == 1:
        beatDivision = beatDivision[0]

    if isinstance(beatDivision, int):
        parts = _breakIrregularDurationInSimpleDivision(n, beatDur=beatDur,
                                                        div=beatDivision, beatOffset=beatOffset)
        return parts

    # beat is not subdivided regularly. check if n extends over subdivision
    numDivisions = len(beatDivision)
    divDuration = beatDur/numDivisions

    ticks = list(mathlib.fraction_range(beatOffset, beatOffset+beatDur+divDuration, divDuration))
    assert len(ticks) == numDivisions + 1

    subdivisionTimespans = list(pairwise(ticks))
    subdivisions = list(zip(subdivisionTimespans, beatDivision))
    subns = n.splitAtOffsets(ticks)
    allparts: list[Notation] = []
    for subn in subns:
        # find the subdivision
        for timespan, numslots in subdivisions:
            if hasoverlap(timespan[0], timespan[1], subn.qoffset, subn.end):
                if n.duration == 0 or (n.isQuantized() and n.hasRegularDuration()):
                    allparts.append(n)
                else:
                    parts = _breakIrregularDurationInBeat(n=subn,
                                                          beatDur=divDuration,
                                                          beatDivision=numslots,
                                                          beatOffset=timespan[0])
                    if parts:
                        allparts.extend(parts)
                    else:
                        allparts.append(subn)
    assert sum(part.duration for part in allparts) == n.duration
    Notation.tieNotations(allparts)
    n._copySpannersToSplitNotation(allparts)
    return allparts


def _breakIrregularDurationInSimpleDivision(n: Notation,
                                            beatDur: F,
                                            div: int,
                                            beatOffset: F = F0,
                                            minPartDuration=F(1,64)
                                            ) -> list[Notation] | None:
    """
    Split irregular durations within a beat during quantization

    An irregular duration is a duration which cannot be expressed as a quarter/eights/16th/etc.
    For example a beat filled with a sextuplet with durations (1, 5), the second
    note is irregular and must be split. Since it begins in an uneven slot, it is
    split as 1+4

    Args:
        n: the Notation to split. Its duration should be a multiple of
            the slot duration, where slotdur=beatDur/div
        beatDur: the duration of the beat in which n is circumscribed
        beatOffset: the offset of the beat to the beginning of the measure
        minPartDuration: the min. duration of a part when splitting n

    Returns:
        a list of notations which together reconstruct the original notation,
        or None if the notation given is already regular

    ::

        5  -> 4+1 if n starts in an even slot, 1+4 if it starts in an odd slot
        9  -> 8+1 / 1+8
        10 -> 8+2 / 2+8
        11 -> 8+3 / 3+8
        13 -> 12+1 / 1+12
        15 -> 12+3 / 3+12
        17 -> 16+1 / 1+16
        18 -> 16+2 == 8+1
        19 -> 16+3 / 3+16
        20 -> 16+4 == 4+1
        21 -> 16+4+1 (quarter~16th~64th)
        22 -> 16+6 (quarter~16th·)
        23 -> 16+7 (quarter~16th··)
        25 -> 24+1 (16+9 == q~8th~64th)
        higher -> error

    """
    assert n.duration <= beatDur
    slotdur = beatDur/div
    nslots = n.duration/slotdur

    if nslots.denominator != 1:
        raise ValueError(f"Duration is not quantized with given division.\n  {n=}, {div=}, {slotdur=}, {nslots=}")

    if nslots.numerator in quantdata.regularDurations:
        return None

    slotindex = (n.qoffset-beatOffset)/slotdur
    assert int(slotindex) == slotindex
    slotindex = int(slotindex)

    if not slotindex.denominator == 1:
        raise ValueError(f"Offset is not quantized with given division. n={n}, division={div}")

    numslots = int(n.duration / slotdur)
    if numslots == 1:
        return [n]
    elif numslots > 25:
        raise ValueError("Division not supported")

    slotDivisions = quantdata.splitIrregularSlots(numslots=numslots, slotindex=slotindex)

    offset = F(n.qoffset)
    parts: list[Notation] = []
    # durRatio = _intDivisionToRatio(div)
    for slots in slotDivisions:
        partDur = slotdur * slots
        assert partDur > minPartDuration
        parts.append(n.clone(offset=offset, duration=partDur))
        offset += partDur
    Notation.tieNotations(parts)
    if n.spanners:
        n._copySpannersToSplitNotation(parts)

    assert sum(part.duration for part in parts) == n.duration
    assert (p0 := parts[0]).offset == n.offset and p0.tiedPrev == n.tiedPrev, f"{n=}, {p0=}"
    assert (p1 := parts[-1]).end == n.end and p1.tiedNext == n.tiedNext
    return parts


def _intDivisionToRatio(div: int) -> F:
    if mathlib.ispowerof2(div):
        return F(1)
    pow2 = util.highestPowerLowerOrEqualTo(div, 2)
    return F(div, pow2)
