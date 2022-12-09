"""

All objects within the realm of **maelzel.core** inherit from :class:`MObj`.
A :class:`MObj` **exists in time** (in has a duration and a time offset),
it **can display itself as notation** and, if appropriate, **play itself as audio**.

Time: Absolute Time / Quarternote Time
--------------------------------------

A :class:`MObj` has always an *offset* and *dur* attribute. These can be unset (``None``),
meaning that they are not explicitely determined. For example, a sequence of notes might
be defined by simply setting their duration (the ``dur`` attribute); their ``start`` time
would be determined by stacking them one after the other.

These time attributes (*offset*, *dur*, *end*) refer to a symbolic *quarternote* time.
To map a *quarternote* time to *absolute* time a score structure
(:class:`maelzel.scorestruct.ScoreStruct`) is needed, which provides information about
tempo.

Score Structure
---------------

A Score Structure (:class:`~maelzel.scorestruct.ScoreStruct`) is a set of tempos and measure
definitions. **It does not contain any material itself**: it is only the "skeleton" of a score.

At any moment there is always an **active score structure**, the default being an endless
score with a *4/4* time-signature and a tempo of *60 bpm*.

Playback
--------

For playback we rely on `csound <https://csound.com/>`_. When the method :meth:`MObj.play` is
called, a :class:`MObj` generates a list of :class:`~maelzel.core.synthevent.SynthEvent`,
which tell *csound* how to play a :class:`Note`, :class:`Chord`, or an entire :class:`Score`.
Using csound it is possible to define instrumental presets using any kind of synthesis or
by simply loading a set of samples or a soundfont. See :meth:`MObj.play`
and :py:mod:`maelzel.core.playback`

"""

from __future__ import annotations

import math
import functools

from emlib import misc
from emlib import iterlib

import pitchtools as pt

from maelzel import scoring
from maelzel.scoring import enharmonics
from maelzel.common import F, asF
from ._common import UNSET, MAXDUR, logger
from . import _util
from .mobj import *
from .workspace import getConfig, Workspace
from . import notation
from .synthevent import PlayArgs, SynthEvent
from . import symbols

from typing import TYPE_CHECKING, overload as _overload

if TYPE_CHECKING:
    from .config import CoreConfig
    from typing import TypeVar, Callable, Any, Sequence, Iterator
    from ._typedefs import *
    MEventT = TypeVar("MEventT", bound="MEvent")
    import music21 as m21

__all__ = (
    'MEvent',
    'Note',
    'Chord',
    'Rest',

    'asNote',
    'asEvent',
    'Gracenote'
)


class MEvent(MObj):
    """
    A discrete event in time (a Note, Chord, etc)
    """
    _acceptsNoteAttachedSymbols = True

    def isRest(self) -> bool:
        return False

    def isGracenote(self) -> bool:
        """
        Is this a grace note?

        A grace note has a pitch but no duration

        Returns:
            True if this can be considered a grace note
        """
        return not self.isRest() and self.dur == 0

    def _markAsGracenote(self, slash=False):
        self.setProperty('grace', True)
        self.setProperty('grace-slash', slash)

    def canBeLinkedTo(self, other: MEvent) -> bool:
        """
        Can self be linked to *other* within a line, assuming other follows self?

        A line is a sequence of events (notes, chords) where
        one is linked to the next by either a tied note or a gliss
        leading to the next pitch, etc

        This method should not take offset time into account: it should
        simply return if self can be linked to other assuming that
        other follows self
        """
        return False

    def _copyAttributesTo(self, other: MEvent) -> None:
        if self.symbols:
            other.symbols = self.symbols.copy()
        if self.playargs:
            other.playargs = self.playargs.copy()
        if self.properties:
            other.properties = self.properties.copy()

    def mergeWith(self: MEventT, other: MEventT) -> MEventT | None:
        """
        Merge this with other, return None if not possible

        Args:
            other: the event to concatenato to this. Only events of the same type
                can be merged (a Note with a Note, a Chord with a Chord)

        Returns:
            the merged event, or None

        """
        raise NotImplemented

    @property
    def name(self) -> str:
        raise NotImplementedError('Subclass should implement this')


@functools.total_ordering
class Note(MEvent):
    """
    A Note represents a one-pitch event

    Internally the pitch of a Note is represented as a fractional
    midinote: there is no concept of enharmonicity. Notes created
    as ``Note(61)``, ``Note("4C#")`` or ``Note("4Db")`` will all result
    in the same Note.

    A Note makes a clear division between the value itself and the
    representation as notation or as sound. **Playback specific options**
    (instrument, pan position, etc) can be set via the
    :meth:`~Note.setPlay` method.

    **Any aspects regarding **notation** (articulation, enharmonic variant, etc)
    can be set via :meth:`~Note.setSymbol`

    .. note::

        The *pitch* parameter can be used to set the pitch and other attributes in
        multiple ways.

        * To set the pitch from a frequency, use `pitchtools.f2m` or use a string as '400hz'.
        * The spelling of the notename can be fixed by suffixing the notename with
          a '!' sign. For example, ``Note('4Db!')`` will fix the spelling of this note
 -         to Db instead of C#. This is the same as ``Note('4Db', fixed=True)``
        * The duration can be set as ``Note('4C#:0.5')`` to set a duration of 0.5, or
          also Note('4C#/8') to set a duration of an 8th note
        * Dynamics can also be set: ``Note('4C#:mf')``. Multiple settings can be combined:
          ``Note('4C#:0.5:mf')``.
        * Offset time can be set like: ``Note('4C#:offset=0.5')``

    Args:
        pitch: a midinote or a note as a string. A pitch can be
            a midinote (a float) or a notename (a string).
        dur: the duration of this note (optional)
        amp: amplitude 0-1 (optional)
        offset: offset time fot the note, in quarternotes (optional). If None,
            the offset time will depend on the context (previous notes) where
            this Note is evaluated.
        gliss: if given, defines a glissando. It can be either the endpitch of
            the glissando, or True, in which case the endpitch remains undefined
        label: a label str to identify this note
        dynamic: allows to attach a dynamic expression to this Note. This dynamic
-            is only for notation purposes, it does not modify playback
        tied: is this Note tied to the next?
        fixed: if True, fix the spelling of the note to the notename given (this is
            only taken into account if the pitch was given as a notename)
        _init: if True, fast initialization is performed, skipping any checks. This is
            used internally for fast copying/cloning of objects.

    Attributes:
        amp: the amplitude (0-1), or None
        pitch: the sounding pitch, as midinote
        gliss: the end pitch (as midinote), or None
        tied: True if this Note is tied to another
        dynamic: the dynamic of this note, or None. See :ref:`config_play_usedynamics`
        pitchSpelling: the notated pitch, can differ from the pitch attribute
    """

    __slots__ = ('pitch', 'amp', '_gliss', 'tied', 'dynamic', 'pitchSpelling')

    def __init__(self,
                 pitch: pitch_t,
                 dur: time_t | None = None,
                 amp: float | None = None,
                 offset: time_t | None = None,
                 gliss: pitch_t | bool = False,
                 label: str = '',
                 dynamic: str = '',
                 tied = False,
                 properties: dict[str, Any] | None = None,
                 fixed=False,
                 _init=True
                 ):
        pitchSpelling = ''
        if _init:
            if isinstance(pitch, str):
                if ":" in pitch:
                    props = _util.parseNote(pitch)
                    dur = dur if dur is not None else props.dur
                    if isinstance(props.notename, list):
                        raise ValueError(f"Can only accept a single pitch, got {props.notename}")
                    pitch = props.notename
                    if p := props.properties:
                        offset = offset or p.pop('offset', None)
                        dynamic = dynamic or p.pop('dynamic', None)
                        tied = tied or p.pop('tied', False)
                        gliss = gliss or p.pop('gliss', False)
                        fixed = p.pop('fixPitch', False) or fixed
                        label = label or p.pop('label', '')
                        properties = p if not properties else misc.dictmerge(p, properties)
                elif "/" in pitch:
                    parsednote = _util.parseNote(pitch)
                    pitch = parsednote.notename
                    dur = parsednote.dur
                pitch = pitch.lower()
                if pitch == 'rest' or pitch == 'r':
                    pitch, amp = 0, 0
                else:
                    pitch, tied, fixed = _util.parsePitch(pitch)
                    pitchSpelling = pt.notename_upper(pitch)
                    pitch = pt.str2midi(pitch)

                if not fixed and Workspace.active.config['fixStringNotenames']:
                    fixed = True

            else:
                assert 0 <= pitch <= 200, f"Expected a midinote (0-127) but got {pitch}"

            if dur is not None:
                dur = asF(dur)

            if offset is not None:
                offset = asF(offset)

            if not isinstance(gliss, bool):
                gliss = _util.asmidi(gliss)

            if amp and amp > 0:
                assert pitch > 0

            if dynamic:
                assert dynamic in scoring.definitions.dynamicLevels

            assert properties is None or isinstance(properties, dict)

        super().__init__(dur=dur, offset=offset, label=label, properties=properties)
        self.pitch: float = pitch
        "The pitch of this note"

        self.amp: float | None = amp
        "The playback amplitude 0-1 of this note"

        self._gliss: float | bool = gliss
        self.tied = tied
        "Is this note tied?"

        self.dynamic: str | None = dynamic
        "A dynamic. If given and no amp was set, its value will inform the playback"

        self.pitchSpelling = '' if not fixed else pitchSpelling
        "The pitch representation of this Note. Can be different from the sounding pitch"

    @staticmethod
    def makeRest(dur: time_t, offset: time_t = None, label: str = '') -> Note:
        assert dur and dur > 0
        if offset:
            offset = asF(offset)
        return Note(pitch=0, dur=asF(dur), offset=offset, amp=0, label=label, _init=False)

    def transpose(self, interval: float, inplace=False) -> Note:
        """
        Transpose this note by the given interval

        If inplace is True the operation is done inplace and the
        returned note can be ignored.

        .. note::
            If this Note has a set spelling, its spelling will also be transposed
            To remove a fixed spelling do ``note.pitchSpelling = ''``

        Args:
            interval: the transposition interval (1=one semitone)
            inplace: if True, the note itself is modified

        Returns:
            The transposed note

        """
        out = self if inplace else self.copy()
        out._setpitch(self.pitch+interval)
        return out

    def asGracenote(self, slash=True) -> Note:
        return Gracenote(self.pitch, slash=slash)

    def mergeWith(self, other: Note) -> Note | None:
        """
        Merge this Note with another Note, if possible

        Args:
            other: the other note to merge this with

        Returns:
            the merged note or None if merging is not possible

        """
        if self.isRest() and other.isRest():
            assert self.dur and other.dur
            out = self.clone(dur=self.dur + other.dur)
            return out

        if (not self.tied or
                self.gliss or
                self.isRest() or
                self.isRest() or
                self.pitch != other.pitch or
                self.amp != other.amp or
                self.dynamic != other.dynamic
        ):
            return None

        return self.clone(dur=self.dur + other.dur, tied=other.tied)

    def _setNotatedPitch(self, notename: str) -> None:
        """
        Set the pitch representation when this object is rendered as notation

        This will not change the actual pitch or its playback. The pitch may differ
        from the actual pitch of the object.

        Args:
            notename: the pitch to use to represent this object. Can be set
                to an empty string to remove any pitch previously set

        """
        if not notename:
            self._removeSymbolsOfClass(symbols.NotatedPitch)
            self.pitchSpelling = ''
        else:
            self.addSymbol(symbols.NotatedPitch(notename))
            self.pitchSpelling = notename

    def canBeLinkedTo(self, other: MEvent) -> bool:
        if self.isRest() and other.isRest():
            return False
        if self._gliss is True:
            return True
        if isinstance(other, Note):
            if self.tied and self.pitch == other.pitch:
                return True
        elif isinstance(other, Chord):
            if self.pitch in other.pitches:
                return True
        return False

    def isGracenote(self) -> bool:
        return not self.isRest() and self.dur == 0

    @property
    def gliss(self):
        """the end pitch (as midinote), or None"""
        return self._gliss

    @gliss.setter
    def gliss(self, gliss: pitch_t | bool):
        """
        Set the gliss attribute of this Note, in place
        """
        self._gliss = gliss if isinstance(gliss, bool) else _util.asmidi(gliss)

    def __eq__(self, other) -> bool:
        if isinstance(other, Note):
            return hash(self) == hash(other)
            # return self.pitch == other.pitch
        elif isinstance(other, str):
            return self.pitch == pt.str2midi(other)
        else:
            return self.pitch == other

    def copy(self) -> Note:
        out = Note(self.pitch, dur=self.dur, amp=self.amp, gliss=self._gliss, tied=self.tied,
                   dynamic=self.dynamic, offset=self.offset, label=self.label,
                   _init=False)
        out.pitchSpelling = self.pitchSpelling
        out._scorestruct = self._scorestruct
        self._copyAttributesTo(out)
        return out

    def clone(self,
              pitch: pitch_t = UNSET,
              dur: time_t | None = UNSET,
              amp: time_t | None = UNSET,
              offset: time_t | None = UNSET,
              gliss: pitch_t | bool = UNSET,
              label: str = UNSET,
              tied: bool = UNSET,
              dynamic: str = UNSET) -> Note:
        """
        Clone this note with overridden attributes

        Returns a new note
        """
        out = Note(pitch=pitch if pitch is not UNSET else self.pitch,
                   dur=dur if dur is not UNSET else self.dur,
                   amp=amp if amp is not UNSET else self.amp,
                   offset=offset if offset is not UNSET else self.offset,
                   gliss=gliss if gliss is not UNSET else self.gliss,
                   label=label if label is not UNSET else self.label,
                   tied=tied if tied is not UNSET else self.tied,
                   dynamic=dynamic if dynamic is not UNSET else self.dynamic,
                   _init=False)
        if pitch is not UNSET:
            if self.pitchSpelling:
                out.pitchSpelling = pt.transpose(self.pitchSpelling, pitch - self.pitch)
        else:
            out.pitchSpelling = self.pitchSpelling
        self._copyAttributesTo(out)
        return out

    def __hash__(self) -> int:
        hashsymbols = hash(tuple(self.symbols)) if self.symbols else 0
        return hash((self.pitch, self.dur, self.offset, self._gliss, self.label,
                     self.dynamic, self.tied, self.pitchSpelling, hashsymbols))

    def asChord(self) -> Chord:
        """ Convert this Note to a Chord of one note """
        gliss = self.gliss
        if gliss and isinstance(self.gliss, (int, float)):
            gliss = [gliss]
        properties = self.properties.copy() if self.properties else None
        chord = Chord([self], amp=self.amp, dur=self.dur, offset=self.offset,
                      gliss=gliss, label=self.label, properties=properties, _init=False)

        if self.symbols:
            for s in self.symbols:
                chord.addSymbol(s)
        if self.playargs:
            chord.playargs = self.playargs.copy()
        return chord

    def isRest(self) -> bool:
        """ Is this a Rest? """
        return self.amp == 0 and self.pitch == 0

    def convertToRest(self) -> None:
        """Convert this Note to a rest, inplace"""
        self.amp = 0
        self.pitch = 0
        self.pitchSpelling = ''

    def pitchRange(self) -> tuple[float, float] | None:
        return (self.pitch, self.pitch)

    def freqShift(self, freq:float) -> Note:
        """
        Return a copy of self, shifted in freq.

        Example::

            # Shifting a note by its own freq. sounds one octave higher
            >>> n = Note("C3")
            >>> n.freqShift(n.freq)
            C4
        """
        out = self.copy()
        out._setpitch(pt.f2m(self.freq + freq))
        return out

    def __lt__(self, other:pitch_t) -> bool:
        if isinstance(other, Note):
            return self.pitch < other.pitch
        elif isinstance(other, (float, Rational)):
            return self.pitch < other
        elif isinstance(other, str):
            return self.pitch < pt.str2midi(other)
        else:
            raise NotImplementedError()

    def __abs__(self) -> Note:
        if self.pitch >= 0:
            return self
        return self.clone(pitch=-self.pitch)

    def _setpitch(self, pitch: float) -> None:
        interval = pitch - self.pitch
        self.pitch = pitch
        if self.pitchSpelling:
            self.pitchSpelling = pt.transpose(self.pitchSpelling, interval)

    @property
    def freq(self) -> float:
        """The frequency of this Note (according to the current A4 value)"""
        return pt.m2f(self.pitch)

    @freq.setter
    def freq(self, value:float) -> None:
        self.pitch = pt.f2m(value)

    @property
    def name(self) -> str:
        """The notename of this Note"""
        if self.isRest():
            return 'Rest'
        return self.pitchSpelling or pt.m2n(self.pitch)

    @name.setter
    def name(self, notename: str):
        self.pitchSpelling = notename
        self.pitch = pt.n2m(notename)

    @property
    def pitchclass(self) -> int:
        """The pitch-class of this Note (an int between 0-11)"""
        return round(self.pitch) % 12


    @property
    def cents(self) -> int:
        """The fractional part of this pitch, rounded to the cent"""
        return _util.midicents(self.pitch)

    @property
    def centsrepr(self) -> str:
        """A string representing the .cents of this Note"""
        return _util.centsshown(self.cents, divsPerSemitone=4)

    def overtone(self, n:float) -> Note:
        """
        Return a new Note representing the `nth` overtone of this Note

        Args:
            n: the overtone number (1 = fundamental)

        Returns:
            a new Note
        """
        return Note(pt.f2m(self.freq * n), _init=False)

    def scoringEvents(self,
                      groupid: str = None,
                      config: CoreConfig = None,
                      ) -> list[scoring.Notation]:
        if not config:
            config = getConfig()
        dur = self.resolvedDur()
        if self.isRest():
            rest = scoring.makeRest(self.dur, offset=self.offset, dynamic=self.dynamic)
            if annot := self._scoringAnnotation():
                rest.addText(annot)
            if self.symbols:
                for symbol in self.symbols:
                    if symbol.appliesToRests:
                        symbol.applyTo(rest)
            return [rest]

        if self.parent:
            if config['show.asoluteOffsetForDetachedObjects']:
                offset = self.absoluteOffset()
            else:
                offset = self.resolvedOffset()
        else:
            offset = self.offset

        notation = scoring.makeNote(pitch=self.pitch,
                                    duration=asF(dur),
                                    offset=offset,
                                    gliss=bool(self.gliss),
                                    dynamic=self.dynamic,
                                    group=groupid,
                                    gracenote=self.isGracenote())
        if self.pitchSpelling:
            notation.fixNotename(self.pitchSpelling, idx=0)

        if self.tied:
            notation.tiedNext = True
            assert not self.gliss

        notes = [notation]
        if self.gliss and not isinstance(self.gliss, bool):
            offset = self.end if self.end is not None else None
            groupid = groupid or str(hash(self))
            notes[0].groupid = groupid
            assert self.gliss >= 12, f"self.gliss = {self.gliss}"
            notes.append(scoring.makeNote(pitch=self.gliss,
                                          gracenote=True,
                                          offset=offset,
                                          group=groupid))
            if config['show.glissEndStemless']:
                notes[-1].stem = 'hidden'

        if self.label:
            annot = self._scoringAnnotation()
            if annot is not None:
                notes[0].addText(annot)
        if self.symbols:
            for symbol in self.symbols:
                symbol.applyToTiedGroup(notes)
        return notes

    def _asTableRow(self, config: CoreConfig = None) -> list[str]:
        if self.isRest():
            elements = ["REST"]
        else:
            notename = self.name
            if self.tied:
                notename += "~"

            if self.symbols:
                notatedPitch = next((s for s in self.symbols
                                     if isinstance(s, symbols.NotatedPitch)), None)
                if notatedPitch and notatedPitch.notename != notename:
                    notename = f'{notename} ({notatedPitch.notename})'

            elements = [notename]
            config = config or getConfig()
            if config['reprShowFreq']:
                elements.append("%dHz" % int(self.freq))
            if self.amp is not None and self.amp < 1:
                elements.append("%ddB" % round(pt.amp2db(self.amp)))

        if self.dur:
            if self.dur >= MAXDUR:
                elements.append("dur=inf")
            else:
                elements.append(f"{_util.showT(self.dur)}♩")

        if self.offset is not None:
            elements.append(f"offset={_util.showT(self.offset)}")

        if self.gliss:
            if isinstance(self.gliss, bool):
                elements.append(f"gliss={self.gliss}")
            else:
                elements.append(f"gliss={pt.m2n(self.gliss)}")

        if self.symbols:
            elements.append(f"symbols={self.symbols}")

        return elements

    def __repr__(self) -> str:
        if self.isRest():
            if self.offset is not None:
                return f"Rest:{_util.showT(self.dur)}♩:offset={_util.showT(self.offset)}"
            return f"Rest:{_util.showT(self.dur)}♩"
        elements = self._asTableRow()
        if len(elements) == 1:
            return elements[0]
        else:
            s = ":".join(elements)
            return s

    def __float__(self) -> float: return float(self.pitch)

    def __int__(self) -> int: return int(self.pitch)

    def __add__(self, other: num_t) -> Note:
        if isinstance(other, (int, float)):
            out = self.copy()
            out._setpitch(self.pitch + other)
            out._gliss = self.gliss if isinstance(self.gliss, bool) else self.gliss + other
            return out
        raise TypeError(f"can't add {other} ({other.__class__}) to a Note")

    def __xor__(self, freq) -> Note: return self.freqShift(freq)

    def __sub__(self, other: num_t) -> Note:
        return self + (-other)

    def quantizePitch(self, step=0.) -> Note:
        """
        Returns a new Note, rounded to step.

        If step is 0, the default quantization value is used (this can be
        configured via ``getConfig()['semitoneDivisions']``

        .. note::
            - If this note has a pitch gliss, the target pitch is also quantized
            - Any set pitch spelling is deleted
        """
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        out = self.clone(pitch=round(self.pitch / step) * step)
        if self._gliss > 1:  # a pitched glissando
            self._gliss = round(self._gliss / step) * step
        out.pitchSpelling = ''
        return out

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace,
                     ) -> list[SynthEvent]:
        if self.isRest():
            return []
        conf = workspace.config
        scorestruct = workspace.scorestruct
        if self.playargs:
            playargs = playargs.overwrittenWith(self.playargs)
            # playargs.overwriteWith(self.playargs)
        # playargs.fillDefaults(conf)
        if self.amp is not None:
            amp = self.amp
        else:
            if conf['play.useDynamics']:
                dyn = self.dynamic or conf['play.defaultDynamic']
                amp = workspace.dynamicCurve.dyn2amp(dyn)
            else:
                amp = conf['play.defaultAmplitude']

        endmidi = self.gliss if not isinstance(self.gliss, bool) else self.pitch
        offset = self.offset or 0.
        dur = self.dur or F(1)
        starttime = float(scorestruct.beatToTime(offset))
        endtime   = float(scorestruct.beatToTime(offset + dur))
        transp = playargs.get('transpose', 0)
        if starttime >= endtime:
            raise ValueError(f"Trying to play an event with 0 or negative duration: {endtime-starttime}. "
                             f"Object: {self}")
        bps = [[starttime, self.pitch+transp, amp],
               [endtime,   endmidi+transp,    amp]]
        if sustain:=playargs.get('sustain'):
            bps.append([endtime + sustain, endmidi+transp, amp])

        return [SynthEvent.fromPlayArgs(bps=bps, playargs=playargs, tiednext=self.tied)]

    def resolvedDynamic(self, conf: CoreConfig = None) -> str:
        if conf is None:
            conf = getConfig()
        return self.dynamic or conf['play.defaultDynamic']

    def resolvedAmp(self, workspace: Workspace = None) -> float:
        """
        Get the amplitude of this object, or a default amplitude

        Returns a default amplitude if no amplitude was define (self.amp is None).
        The default amplitude can be customized via
        ``getConfig()['play.defaultAmplitude']``

        Returns:
            the amplitude (a value between 0-1, where 0 corresponds to 0dB)
        """
        if self.amp:
            return self.amp
        if workspace is None:
            workspace = Workspace.active
        conf = workspace.config
        if conf['play.useDynamics']:
            return workspace.dynamicCurve.dyn2amp(self.resolvedDynamic(conf))
        else:
            return conf['play.defaultAmplitude']

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Note:
        if self.isRest():
            return self
        pitch = pitchmap(self.pitch)
        gliss = self.gliss if isinstance(self.gliss, bool) else pitchmap(self.gliss)
        out = self.copy()
        out._setpitch(pitch)
        out._gliss = gliss

    # def trill(self,
    #           other: Note,
    #           notedur: time_t = F(1, 8)
    #           ) -> Chain:
    #     """
    #     Create a Chain of Notes representing a trill
    #
    #     Args:
    #         other: the second note of the trill (can also  be a chord)
    #         notedur: duration of each note. This value will only be used
    #             if the trill notes have an unset duration
    #
    #     Returns:
    #         A realisation of the trill as a :class:`Chain` of at least the
    #         given *dur* (can be longer if *dur* is not a multiple
    #         of *notedur*)
    #     """
    #     totaldur = self.resolvedDur()
    #     note1 = self.clone(dur=notedur)
    #     note2 = asNote(other, dur=notedur)
    #     seq = Chain([note1, note2])
    #     return seq.cycle(totaldur)


def Rest(dur: time_t, offset: time_t = None, label='', dynamic: str = None) -> Note:
    """
    Creates a Rest. A Rest is a Note with pitch 0 and amp 0.

    To test if an item is a rest, call :meth:`~MObj.isRest`

    Args:
        dur: duration of the Rest
        offset: time offset of the Rest
        label: a label for this rest
        dynamic: a rest may have a dynamic

    Returns:
        the created rest
    """
    assert dur and dur > 0
    return Note(pitch=0, dur=dur, offset=offset, amp=0, label=label,
                dynamic=dynamic, _init=False)


def asNote(n: Note|float|str, **kws) -> Note:
    """
    Convert *n* to a Note, optionally setting its amp, offset or dur

    Args:
        n: the pitch
        kws: any keyword passed to Note

    Returns:
        the corresponding Note

    """
    return n if isinstance(n, Note) else Note(n, **kws)


class Chord(MEvent):
    """
    A Chord is a stack of Notes

    a Chord can be instantiated as::

        Chord(note1, note2, ...)
        Chord([note1, note2, ...])
        Chord("C4 E4 G4", ...)

    Where each note is either a Note, a notename ("C4", "E4+", etc) or a midinote

    Args:
        notes: the notes of this chord. Can be a list of pitches, where each
            pitch is either a fractional midinote or a notename (str); notes
            can be already created ``Note`` instances; or a string with multiple
            notes separated by a space
        amp: the amplitude (volume) of this chord
        dur: the duration of this chord (in quarternotes)
        offset: the offset time (in quarternotes)
        gliss: either a list of end pitches (with the same size as the chord), or
            True to leave the end pitches unspecified (a gliss to the next chord)
        label: if given, it will be used for printing purposes
        tied: if True, this chord should be tied to a following chord, if possible
        dynamic: a dynamic for this chord ('pp', 'mf', 'ff', etc). Dynamics range from
            'pppp' to 'ffff'
        fixed: if True, any note in this chord which was given as a notename (str)
            will be fixed in the given spelling (**NB**: the spelling can also
            be fixed by suffixing any notename with a '!' sign)
        properties: properties are a space given to the user to attach any information to
            this object

    Attributes:
        amp: the amplitude of the chord itself (each note can have an individual amp)
        notes: the notes which build this chord
        gliss: if True, this Chord makes a gliss to another chord. Also a list
            of pitches can be given as gliss, these indicate the end pitch of the gliss
            as midinote
        tied: is this Chord tied to another Chord?
    """

    __slots__ = ('amp', 'gliss', 'notes', 'tied', 'dynamic', '_notatedPitches')

    def __init__(self,
                 notes: str | list[Note|int|float|str],
                 dur: time_t = None,
                 amp: float = None,
                 offset: time_t = None,
                 gliss: str|bool|Sequence[pitch_t] = False,
                 label: str = '',
                 tied = False,
                 dynamic: str = '',
                 properties: dict[str, Any] = None,
                 fixed=False,
                 _init=True
                 ) -> None:
        """
        a Chord can be instantiated as:

            Chord(note1, note2, ...) or
            Chord([note1, note2, ...])
            Chord("C4 E4 G4")

        where each note is either a Note, a notename ("C4", "E4+", etc), a midinote
        or a tuple (midinote, amp)

        Args:
            amp: the amplitude (volume) of this chord
            dur: the duration of this chord (in quarternotes)
            offset: the offset time (in quarternotes)
            gliss: either a list of end pitches (with the same size as the chord), or
                True to leave the end pitches unspecified (a gliss to the next chord)
            label: if given, it will be used for printing purposes
            fixed: if True, any note in this chord which was given as a notename (str)
                will be fixed in the given spelling (**NB**: the spelling can also
                be fixed by suffixing any notename with a '!' sign)

        """
        self.amp = amp

        if dur is not None:
            dur = asF(dur)

        if not notes:
            super().__init__(dur=dur, offset=None, label=label)
            return

        super().__init__(dur=dur, offset=offset, label=label, properties=properties)

        if dur == 0:
            self._markAsGracenote()

        # notes might be: Chord([n1, n2, ...]) or Chord("4c 4e 4g", ...)
        if isinstance(notes, str):
            if ',' in notes:
                notes = [Note(n.strip(), amp=amp, fixed=fixed) for n in notes.split(',')]
            else:
                notes = [Note(n.strip(), amp=amp, fixed=fixed) for n in notes.split()]

        if _init:
            notes2 = []
            for n in notes:
                if isinstance(n, Note):
                    if n.dur is None and n.offset is None:
                        notes2.append(n)
                    else:
                        notes2.append(n.clone(dur=None, offset=None))
                elif isinstance(n, (int, float, str)):
                    notes2.append(Note(n, amp=amp))
                else:
                    raise TypeError(f"Expected a Note or a pitch, got {n}")
            notes2.sort(key=lambda n: n.pitch)
            notes = notes2
            if any(n.tied for n in notes):
                tied = True

            if not isinstance(gliss, bool):
                gliss = pt.as_midinotes(gliss)
                assert len(gliss) == len(notes), (f"The destination chord of the gliss should have "
                                                  f"the same length as the chord itself, "
                                                  f"{notes=}, {gliss=}")
        self.notes: list[Note] = notes
        self.gliss: bool | list[float] = gliss
        self.tied: bool = tied
        self.dynamic: str = dynamic

    def copy(self):
        notes = [n.copy() for n in self.notes]
        out = Chord(notes=notes, dur=self.dur, amp=self.amp, offset=self.offset,
                    gliss=self.gliss, label=self.label, tied=self.tied,
                    dynamic=self.dynamic,
                    _init=False)
        self._copyAttributesTo(out)
        return out

    def __len__(self) -> int:
        return len(self.notes)

    @_overload
    def __getitem__(self, idx: int) -> Note: ...

    @_overload
    def __getitem__(self, idx: slice) -> Chord: ...

    def __getitem__(self, idx):
        out = self.notes.__getitem__(idx)
        if isinstance(out, list):
            out = self.__class__(out)
        return out

    def __iter__(self) -> Iterator[Note]:
        return iter(self.notes)

    @property
    def name(self) -> str:
        return ",".join(self._bestSpelling())

    def asGracenote(self, slash=True) -> Note:
        return Gracenote(self.pitches, slash=slash)

    def setNotatedPitch(self, notenames: str | list[str]) -> None:
        if isinstance(notenames, str):
            return self.setNotatedPitch(notenames.split())
        if not len(notenames) == len(self.notes):
            raise ValueError(f"The number of given fixed spellings ({notenames}) does not correspond"
                             f"to the number of notes in this chord ({self._bestSpelling()})")
        for notename, n in zip(notenames, self.notes):
            if notename:
                n.pitchSpelling = notename
                # n.setNotatedPitch(notename)


    def canBeLinkedTo(self, other: MObj) -> bool:
        if self.gliss is True:
            return True
        if isinstance(other, Note):
            if self.tied and any(p == other.pitch for p in self.pitches):
                return True
        elif isinstance(other, Chord):
            if self.tied and any(p in other.pitches for p in self.pitches):
                return True
        return False

    def mergeWith(self, other: Chord) -> Chord | None:
        if not isinstance(other, Chord):
            return None

        if (not self.tied or other.gliss or self.pitches != other.pitches):
            return None

        if any(n1 != n2 for n1, n2 in zip(self.notes, other.notes)):
            return None

        return self.clone(dur=self.dur + other.dur)

    def pitchRange(self) -> tuple[float, float] | None:
        return min(n.pitch for n in self.notes), max(n.pitch for n in self.notes)

    def scoringEvents(self, groupid: str = None, config: CoreConfig = None
                      ) -> list[scoring.Notation]:
        if not config:
            config = getConfig()
        notenames = [note.name for note in self.notes]
        annot = self._scoringAnnotation()
        dur = self.resolvedDur()

        notation = scoring.makeChord(pitches=notenames, duration=dur, offset=self.offset,
                                     annotation=annot, group=groupid, dynamic=self.dynamic,
                                     tiedNext=self.tied)

        # Transfer any pitch spelling
        for i, note in enumerate(self.notes):
            if note.pitchSpelling:
                notation.fixNotename(note.pitchSpelling, i)

        if self.symbols:
            for s in self.symbols:
                s.applyTo(notation)

        # Transfer note symbols
        for i, n in enumerate(self.notes):
            if n.symbols:
                for symbol in n.symbols:
                    if isinstance(symbol, symbols.PitchAttachedSymbol):
                        symbol.applyToPitch(notation, idx=i)
                    else:
                        logger.debug(f"Cannot apply symbol {symbol} to a pitch inside chord {self}")

        # Add gliss.
        notations = [notation]
        if self.gliss:
            notation.gliss = True
            if not isinstance(self.gliss, bool):
                groupid = scoring.makeGroupId(groupid)
                notation.groupid = groupid
                endEvent = scoring.makeChord(pitches=self.gliss, duration=0,
                                             offset=self.end, group=groupid)
                if config['show.glissEndStemless']:
                    endEvent.stem = 'hidden'
                notations.append(endEvent)
        return notations

    def asmusic21(self, **kws) -> m21.stream.Stream:
        cfg = getConfig()
        arpeggio = _normalizeChordArpeggio(kws.get('arpeggio', None), self, cfg)
        if arpeggio:
            from . import chain
            dur = cfg['show.arpeggioDuration']
            notes = [n.clone(dur=dur) for n in self.notes]
            return chain.Chain(notes).asmusic21()
        events = self.scoringEvents()
        scoring.stackNotationsInPlace(events, start=self.offset)
        parts = scoring.distributeNotationsByClef(events)
        return notation.renderWithActiveWorkspace(parts, backend='music21').asMusic21()

    def __hash__(self):
        if isinstance(self.gliss, bool):
            glisshash = int(self.gliss)
        elif isinstance(self.gliss, list):
            glisshash = hash(tuple(self.gliss))
        else:
            glisshash = hash(self.gliss)
        symbolshash = hash(tuple(self.symbols)) if self.symbols else 0

        data = (self.dur, self.offset, self.label, glisshash, self.dynamic,
                symbolshash,
                *(hash(n) for n in self.notes))
        return hash(data)

    def append(self, note: float|str|Note) -> None:
        """ append a note to this Chord """
        note = asNote(note)
        if note.freq < 17:
            logger.debug(f"appending a note with very low freq: {note.freq}")
        self.notes.append(note)
        self._changed()

    def extend(self, notes) -> None:
        """ extend this Chord with the given notes """
        for note in notes:
            self.notes.append(asNote(note))
        self._changed()

    def insert(self, index:int, note:pitch_t) -> None:
        self.notes.insert(index, asNote(note))
        self._changed()

    def filter(self, predicate) -> Chord:
        """
        Return a new Chord with only the notes which satisfy the given predicate

        Example::

            # filter out notes which do not belong to the C-major triad
            >>> ch = Chord("C3 D3 E4 G4")
            >>> ch2 = ch.filter(lambda note: (note.notename % 12) in {0, 4, 7})

        """
        return self._withNewNotes([n for n in self if predicate(n)])

    def transposeTo(self, fundamental:pitch_t) -> Chord:
        """
        Return a copy of self, transposed to the new fundamental

        .. note::
            the fundamental is the lowest note in the chord

        Args:
            fundamental: the new lowest note in the chord

        Returns:
            A Chord transposed to the new fundamental
        """
        step = _util.asmidi(fundamental) - self[0].pitch
        return self.transpose(step)

    def freqShift(self, freq:float) -> Chord:
        """
        Return a copy of this chord shifted in frequency
        """
        return Chord([note.freqShift(freq) for note in self])

    def _withNewNotes(self, notes) -> Chord:
        return self.clone(notes=notes)

    def quantizePitch(self, step=0.) -> Chord:
        """
        Returns a copy of this chord, with the pitches quantized.

        Two notes with the same pitch are considered equal if they quantize to the same
        pitch, independently of their amplitude. In the case of two equal notes,
        only the first one is kept.
        """
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        seenmidi = set()
        notes = []
        for note in self:
            note2 = note.quantizePitch(step)
            if note2.pitch not in seenmidi:
                seenmidi.add(note2.pitch)
                notes.append(note2)
        return self._withNewNotes(notes)

    def __setitem__(self, i:int, obj:pitch_t) -> None:
        self.notes.__setitem__(i, asNote(obj))
        self._changed()

    def __add__(self, other:pitch_t) -> Chord:
        if isinstance(other, Note):
            # append the note
            s = self.copy()
            s.append(other)
            return s
        elif isinstance(other, (int, float)):
            # transpose
            s = [n + other for n in self]
            return Chord(s)
        elif isinstance(other, (Chord, str)):
            # join chords together
            return Chord(self.notes + _asChord(other).notes)
        raise TypeError(f"Can't add a Chord to a {other.__class__.__name__}")

    def loudest(self, n:int) -> Chord:
        """
        Return a new Chord with the loudest `n` notes from this chord
        """
        out = self.copy()
        out.sort(key='amp', reverse=True)
        return out[:n]

    def sort(self, key='pitch', reverse=False) -> None:
        """
        Sort **INPLACE**.

        If inplace sorting is undesired, use ``x = chord.copy(); x.sort()``

        Args:
            key: either 'pitch' or 'amp'
            reverse: similar as sort
        """
        if key == 'pitch':
            self.notes.sort(key=lambda n: n.pitch, reverse=reverse)
        elif key == 'amp':
            self.notes.sort(key=lambda n:n.amp, reverse=reverse)
        else:
            raise KeyError(f"Unknown sort key {key}. Options: 'pitch', 'amp'")

    def _resolvePlayargs(self, playargs: PlayArgs, config: dict | None = None) -> PlayArgs:
        playargs = playargs.filledWith(self.playargs)
        playargs.fillDefaults(config or getConfig())
        return playargs

    @property
    def pitches(self) -> list[float]:
        return [n.pitch for n in self.notes]

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        conf = workspace.config
        scorestruct = workspace.scorestruct
        if self.playargs:
            playargs = playargs.overwrittenWith(self.playargs)
            # playargs.overwriteWith(self.playargs)
        # playargs.fillDefaults(conf)
        if conf['chordAdjustGain']:
            gain = playargs.get('gain') or 1.0
            playargs['gain'] = gain / math.sqrt(len(self))
        endpitches = self.pitches
        if self.gliss:
            if isinstance(self.gliss, list):
                endpitches = self.gliss
            elif self.properties and (glisstarget:=self.properties.get('glisstarget')):
                # This property is filled when resolving a sequence
                endpitches = glisstarget

        events = []
        start, dur = self.resolvedOffset(), self.resolvedDur()
        starttime = float(scorestruct.beatToTime(start))
        endtime = float(scorestruct.beatToTime(start + dur))
        useDynamics = conf['play.useDynamics']
        if self.amp is not None:
            chordamp = self.amp
        else:
            if not useDynamics:
                chordamp = conf['play.defaultAmplitude']
            else:
                dyn = self.dynamic or conf['play.defaultDynamic']
                chordamp = workspace.dynamicCurve.dyn2amp(dyn)
        transpose = playargs.get('transpose', 0.)
        for note, endpitch in zip(self.notes, endpitches):
            if note.amp:
                noteamp = note.amp
            elif note.dynamic and useDynamics:
                noteamp = workspace.dynamicCurve.dyn2amp(note.dynamic)
            else:
                noteamp = chordamp
            bps = [[starttime, note.pitch+transpose, noteamp],
                   [endtime,   endpitch+transpose,   noteamp]]
            if sustain:=playargs.get('sustain'):
                bps.append([endtime + sustain, endpitch+transpose, noteamp])
            events.append(SynthEvent.fromPlayArgs(bps=bps, playargs=playargs))
        return events

    def _bestSpelling(self) -> tuple[str]:
        notenames = [n.pitchSpelling + '!' if n.pitchSpelling else n.name
                     for n in self.notes]
        return enharmonics.bestChordSpelling(notenames)

    def __repr__(self):
        # «4C+14,4A 0.1q -50dB»
        elements = [" ".join(self._bestSpelling())]
        if self.dur:
            if self.dur >= MAXDUR:
                elements.append("dur=inf")
            else:
                elements.append(f"{float(self.dur):.3g}♩")
        if self.offset is not None:
            elements.append(f'offset={float(self.offset):.3g}')
        if self.gliss:
            if isinstance(self.gliss, bool):
                elements.append("gliss")
            else:
                endpitches = ','.join([pt.m2n(_) for _ in self.gliss])
                elements.append(f"gliss={endpitches}")

        if len(elements) == 1:
            return f'‹{elements[0]}›'
        else:
            return f'‹{elements[0].ljust(3)} {" ".join(elements[1:])}›'

    def dump(self, indents=0):
        elements = f'offset={self.offset}, dur={self.dur}, gliss={self.gliss}'
        print(f"{'  '*indents}Chord({elements})")
        if self.playargs:
            print("  "*(indents+1), self.playargs)
        for n in reversed(self.notes):
            print("  "*(indents+2), repr(n))

    def mappedAmplitudes(self, curve, db=False) -> Chord:
        """
        Return new Chord with the amps of the notes modified according to curve

        Example::

            # compress all amplitudes to 30 dB
            >>> import bpf4 as bpf
            >>> chord = Chord([Note(60, amp=0.5), Note(65, amp=0.1)])
            >>> curve = bpf.linear(-90, -30, -30, -12, 0, 0)
            >>> chord2 = chord.mappedAmplitudes(curve, db=True)

        Args:
            curve: a func mapping ``amp -> amp``
            db: if True, the value returned by func is interpreted as dB
                if False, it is interpreted as amplitude (0-1)

        Returns:
            the resulting chord
        """
        notes = []
        if db:
            for note in self:
                db = curve(pt.amp2db(note.amp))
                notes.append(note.clone(amp=pt.db2amp(db)))
        else:
            for note in self:
                amp2 = curve(note.amp)
                notes.append(note.clone(amp=amp2))
        return Chord(notes)

    def setAmplitudes(self, amp: float) -> None:
        """
        Set the amplitudes of the notes in this chord to `amp` (in place)

        This modifies the Note objects within this chord, without modifying
        the amplitude of the chord itself

        .. note::

            Each note (instance of Note) within a Chord can have its own ampltidue.
            The Chord itself can also have an amplitude, which is used as fallback
            for any note with unset amplitude. These two amplitudes do not multiply

        Args:
            amp: the new amplitude for the notes of this chord

        Example
        ~~~~~~~

            >>> chord = Chord("3f 3b 4d# 4g#", dur=4)
            >>> chord.amp = 0.5                 # Fallback amplitude
            >>> chord[0:-1].setAmplitude(0.01)  # Sets the amp of all notes but the last
            >>> chord.events()
            [SynthEvent(delay=0, dur=4, gain=0.5, chan=1, fade=(0.02, 0.02), instr=_piano)
             bps 0.000s: 53       0.01
                 4.000s: 53       0.01    ,
             SynthEvent(delay=0, dur=4, gain=0.5, chan=1, fade=(0.02, 0.02), instr=_piano)
             bps 0.000s: 59       0.01
                 4.000s: 59       0.01    ,
             SynthEvent(delay=0, dur=4, gain=0.5, chan=1, fade=(0.02, 0.02), instr=_piano)
             bps 0.000s: 63       0.01
                 4.000s: 63       0.01    ,
             SynthEvent(delay=0, dur=4, gain=0.5, chan=1, fade=(0.02, 0.02), instr=_piano)
             bps 0.000s: 68       0.5
                 4.000s: 68       0.5     ]

        Notice that the last event, corresponding to the highest note, has taken the ampltidue
        of the chord (0.5) as default, since it was unset. The other notes have an amplitude
        of 0.01, as set via :meth:`~Chord.setAmplitudes`
        """
        return self.scaleAmplitudes(factor=0., offset=amp)

    def scaleAmplitudes(self, factor: float, offset=0.0) -> None:
        """
        Scale the amplitudes of the notes within this chord **in place**

        .. note::

            Each note (instance of Note) within a Chord can have its own ampltidue.
            The Chord itself can also have an amplitude, which is used as fallback
            for any note with unset amplitude. These two amplitudes do not multiply

        Args:
            factor: a factor to multiply the amp of each note
            offset: a value to add to the amp of each note
        """
        for n in self.notes:
            amp = n.amp if n.amp is not None else self.amp if self.amp is not None else 1.0
            n.amp = amp * factor + offset

    def equalize(self, curve:Callable[[float], float]) -> None:
        """
        Scale the amplitude of the notes according to their frequency, **in place**

        Args:
            curve: a func mapping freq to gain
        """
        for note in self:
            gain = curve(note.freq)
            note.amp *= gain

    def _isTooCrowded(self) -> bool:
        """
        Is this chord two dense that it needs to be arpeggiated when shown?
        """
        return any(abs(n0.pitch - n1.pitch) <= 1 and abs(n1.pitch - n2.pitch) <= 1
                   for n0, n1, n2 in iterlib.window(self, 3))

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Chord:
        transpositions = [pitchmap(note.pitch) - note.pitch for note in self.notes]
        newnotes = [n.transpose(interval) for n, interval in zip(self.notes, transpositions)]
        return self.clone(notes=newnotes,
                          gliss=self.gliss if isinstance(self.gliss, bool) else list(map(pitchmap, self.gliss)))


def _asChord(obj, amp: float = None, dur: float = None) -> Chord:
    """
    Create a Chord from `obj`

    Args:
        obj: a string with spaces in it, a list of notes, a single Note, a Chord
        amp: the amp of the chord
        dur: the duration of the chord

    Returns:
        a Chord
    """
    if isinstance(obj, Chord):
        out = obj
    elif isinstance(obj, (list, tuple, str)):
        out = Chord(obj)
    elif hasattr(obj, "asChord"):
        out = obj.asChord()
    elif isinstance(obj, (int, float)):
        out = Chord([Note(obj)])
    else:
        raise ValueError(f"cannot express this as a Chord: {obj}")
    return out if amp is None and dur is None else out.clone(amp=amp, dur=dur)


def Gracenote(pitch: pitch_t | list[pitch_t], slash=True) -> Note | Chord:
    """
    Create a gracenote (a note or chord)

    The resulting gracenote can be a note or a chord, depending on pitch

    Args:
        pitch: a single pitch (as midinote, notename, etc), a list of pitches or string
            representing one or more pitches
        slash: if True, the gracenote will be marked as slashed

    Returns:
        the Note/Chord representing the gracenote. A gracenote is basically a
        note/chord with 0 duration

    Example
    ~~~~~~~

        >>> grace = Gracenote('4F')
        >>> grace2 = Note('4F', dur=0)
        >>> grace == grace2
        True
        >>> gracechord = Gracenote('4F 4A')
        >>> gracechord2 = Chord('4F 4A', dur=0)
        >>> gracechord == gracechord2
        True

    """
    out = asEvent(pitch, dur=0)
    assert isinstance(out, (Note, Chord))
    out._markAsGracenote(slash=slash)
    return out


def asEvent(obj, **kws) -> MEvent:
    """
    Convert obj to a Note or Chord, depending on the input itself

    =============================  ==========
    Input                          Output
    =============================  ==========
    int, float (midinote)          Note
    list (of notes)                Chord
    notename as string ("C4")      Note
    str with notenames ("C4 E4")   Chord
    =============================  ==========

    Args:
        obj: the object to convert
        kws: any keyword passed to the constructor (Note, Chord)

    """
    if isinstance(obj, MEvent):
        return obj
    elif isinstance(obj, str):
        if " " in obj:
            return Chord(obj.split(), **kws)
        elif "," in obj:
            notedef = _util.parseNote(obj)
            dur = kws.pop('dur', None) or notedef.dur
            if notedef.properties:
                kws = misc.dictmerge(notedef.properties, kws)
            return Chord(notedef.notename, dur=dur, **kws)
        return Note(obj, **kws)
    elif isinstance(obj, (list, tuple)):
        return Chord(obj, **kws)
    elif isinstance(obj, (int, float)):
        return Note(obj, **kws)
    else:
        raise TypeError(f"Cannot convert {obj} to a Note or Chord")


def _normalizeChordArpeggio(arpeggio: str | bool | None, chord: Chord, config: CoreConfig
                            ) -> bool:
    if arpeggio is None:
        arpeggioconfig = config['show.arpeggiateChord']
        assert isinstance(arpeggioconfig, (str, bool))

    if isinstance(arpeggio, bool):
        return arpeggio
    elif arpeggio == 'auto':
        return chord._isTooCrowded()
    else:
        raise ValueError(f"arpeggio should be True, False, 'auto' (got {arpeggio})")
