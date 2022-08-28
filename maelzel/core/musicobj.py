"""

All objects within the realm of **maelzel.core** inherit from :class:`MusicObj`.
A :class:`MusicObj` **exists in time** (in has a start and duration attribute), it **can display itself
as notation** and, if appropriate, **play itself as audio**.

Time: Absolute Time / Quarternote Time
--------------------------------------

A :class:`MusicObj` has always a *start* and *dur* attribute. These can be unset (``None``),
meaning that they are not explicitely determined. For example, a sequence of notes might
be defined by simply setting their duration (the ``dur`` attribute); their ``start`` time
would be determined by stacking them one after the other.

These time attributes (*start*, *dur*, *end*) refer to an abstract, `quarternote` time.
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

For playback we rely on `csound <https://csound.com/>`_. When the method :meth:`MusicObj.play` is
called, a :class:`MusicObj` generates a list of :class:`~maelzel.core.synthevent.SynthEvent`,
which tell *csound* how to play a :class:`Note`, :class:`Chord`, or an entire :class:`Score`.
Using csound it is possible to define instrumental presets using any kind of synthesis or
by simply loading a set of samples or a soundfont. See :meth:`MusicObj.play`
and :py:mod:`maelzel.core.playback`

"""

from __future__ import annotations

import math
import functools

from emlib import misc
from emlib import iterlib

import pitchtools as pt

from maelzel import scoring
from maelzel.scorestruct import ScoreStruct

from ._common import Rat, asRat, UNSET, MAXDUR, logger
from . import _util
from .musicobjbase import *
from .workspace import getConfig, Workspace
from . import environment
from . import notation
from .synthevent import PlayArgs, SynthEvent
from . import _musicobjtools
from . import symbols
from maelzel.colortheory import safeColors as _safeColors

from typing import TYPE_CHECKING, overload as _overload

if TYPE_CHECKING:
    from .config import CoreConfig
    from typing import Optional, TypeVar, Callable, Any, Sequence, Iterator
    from ._typedefs import *
    T = TypeVar("T", bound="MusicObj")
    import music21 as m21


__all__ = (
    'MusicObj',
    'MusicEvent',
    'Note',
    'Chord',
    'Rest',

    'asNote',
    'asEvent',
    'stackEvents',

    'Chain',
    'Voice',

    'makeGracenote'
)


class MusicEvent(MusicObj):


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

    def canBeLinkedTo(self, other: MusicEvent) -> bool:
        """
        Can self be linked to *other* within a line, assuming other follows self?

        A line is a sequence of events (notes, chords) where
        one is linked to the next by either a tied note or a gliss
        leading to the next pitch, etc

        This method should not take start time into account: it should
        simply return if self can be linked to other assuming that
        other follows self
        """
        return False

    def _copyAttributesTo(self, other: MusicEvent) -> None:
        if self.symbols:
            other.symbols = self.symbols.copy()
        if self._playargs:
            other._playargs = self._playargs.copy()
        if self._properties:
            other._properties = self._properties.copy()


@functools.total_ordering
class Note(MusicEvent):
    """
    A Note represents a one-pitch event

    It is possible to specify a duration, a start time (.start),
    an amplitude and an endpitch, resulting in a glissando.

    Internally the pitch of a Note is represented as a fractional
    midinote: there is no concept of enharmonicity. Notes created
    as ``Note(61)``, ``Note("4C#")`` or ``Note("4Db")`` will all result
    in the same Note.

    A Note makes a clear division between the value itself and the
    representation as notation or as sound. Playback specific options
    (instrument, pan position, etc) can be set via the
    :meth:`~Note.setPlay` method.

    Any aspects regarding notation (articulation, enharmonic variant, etc)
    can be set via :meth:`~Note.setSymbol`

    Args:
        pitch: a midinote or a note as a string. A pitch can be
            a midinote or a notename as a string. To set the pitch from a frequency,
            use `pitchtools.f2m`
        dur: the duration of this note (optional)
        amp: amplitude 0-1 (optional)
        start: start fot the note (optional). If None, the start time will depend
            on the context (previous notes) where this Note is evaluated.
        gliss: if given, defines a glissando. It can be either the endpitch of
            the glissando, or True, in which case the endpitch remains undefined
        label: a label str to identify this note
        dynamic: allows to attach a dynamic expression to this Note. This dynamic
            is only for notation purposes, it does not modify playback
        tied: is this Note tied to the next?
        _init: if True, fast initialization is performed, skipping any checks. This is
            used internally for fast copying/cloning of objects.

    Attributes:
        amp: the amplitude (0-1), or None
        pitch: the sounding pitch, as midinote
        gliss: the end pitch (as midinote), or None
        tied: True if this Note is tied to another
        dynamic: the dynamic of this note, or None. See :ref:`config_play_usedynamics`
    """

    __slots__ = ('pitch', 'amp', '_gliss', 'tied', 'dynamic')

    def __init__(self,
                 pitch: pitch_t,
                 dur: time_t = None,
                 amp: float = None,
                 start: time_t = None,
                 gliss: Union[pitch_t, bool] = False,
                 label: str = '',
                 dynamic: str = None,
                 tied = False,
                 properties: dict[str, Any] = None,
                 _init=True
                 ):
        if _init:
            if isinstance(pitch, str) and (":" in pitch or "/" in pitch):
                props = _util.parseNote(pitch)
                dur = dur or props.dur
                pitchl = props.pitch.lower()
                if pitchl == 'rest' or pitchl == 'r':
                    pitch = 0
                    amp = 0
                else:
                    pitch = _util.asmidi(props.pitch)

                if props.properties:
                    start = start or props.properties.pop('start', None)
                    dynamic = dynamic or props.properties.pop('dynamic', None)
                    tied = tied or props.properties.pop('tied', False)
                    gliss = gliss or props.properties.pop('gliss', False)
                    properties = props.properties if not properties else misc.dictmerge(props.properties, properties)
            else:
                pitch = _util.asmidi(pitch)

            if dur is not None:
                dur = asRat(dur)
            if start is not None:
                start = asRat(start)

            if not isinstance(gliss, bool):
                gliss = _util.asmidi(gliss)

            if amp and amp > 0:
                assert pitch > 0

            if dynamic:
                assert dynamic in scoring.definitions.dynamicLevels

        super().__init__(dur=dur, start=start, label=label, properties=properties)
        self.pitch: float = pitch
        self.amp: Optional[float] = amp
        self._gliss: Union[float, bool] = gliss
        self.tied = tied
        self.dynamic: Optional[str] = dynamic

    @staticmethod
    def makeRest(dur: time_t, start: time_t = None, label: str = '') -> Note:
        assert dur and dur > 0
        if start:
            start = asRat(start)
        return Note(pitch=0, dur=asRat(dur), start=start, amp=0, label=label, _init=False)

    def asGracenote(self, slash=True) -> Note:
        return makeGracenote(self.pitch, slash=slash)

    def canBeLinkedTo(self, other: MusicEvent) -> bool:
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
    def gliss(self, gliss: Union[pitch_t, bool]):
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
                   dynamic=self.dynamic, start=self.start, label=self.label,
                   _init=False)
        self._copyAttributesTo(out)
        return out

    def clone(self,
              pitch: pitch_t = UNSET,
              dur: Optional[time_t] = UNSET,
              amp: Optional[time_t] = UNSET,
              start: Optional[time_t] = UNSET,
              gliss: Union[pitch_t, bool] = UNSET,
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
                   start=start if start is not UNSET else self.start,
                   gliss=gliss if gliss is not UNSET else self.gliss,
                   label=label if label is not UNSET else self.label,
                   tied=tied if tied is not UNSET else self.tied,
                   dynamic=dynamic if dynamic is not UNSET else self.dynamic,
                   _init=False)
        self._copyAttributesTo(out)
        return out

    def __hash__(self) -> int:
        hashsymbols = hash(tuple(self.symbols)) if self.symbols else 0
        return hash((self.pitch, self.dur, self.start, self._gliss, self.label,
                     self.dynamic, self.tied, hashsymbols))

    def asChord(self) -> Chord:
        """ Convert this Note to a Chord of one note """
        gliss = self.gliss
        if gliss and isinstance(self.gliss, (int, float)):
            gliss = [gliss]
        return Chord([self], amp=self.amp, dur=self.dur, start=self.start,
                     gliss=gliss, label=self.label)

    def isRest(self) -> bool:
        """ Is this a Rest? """
        return self.amp == 0 and self.pitch == 0

    def convertToRest(self) -> None:
        """Convert this Note to a rest, inplace"""
        self.amp = 0
        self.pitch = 0

    def pitchRange(self) -> Optional[tuple[float, float]]:
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
        return self.clone(pitch=pt.f2m(self.freq + freq))

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
        return pt.m2n(self.pitch)

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

    def scoringEvents(self, groupid:str=None, config: CoreConfig = None
                      ) -> list[scoring.Notation]:
        if not config:
            config = getConfig()
        dur = self.dur if self.dur is not None else Rat(1)
        if self.isRest():
            rest = scoring.makeRest(self.dur, offset=self.start)
            if annot := self._scoringAnnotation():
                rest.addAnnotation(annot)
            if self.symbols:
                for symbol in self.symbols:
                    if symbol.appliesToRests:
                        symbol.applyTo(rest)
            return [rest]
        note = scoring.makeNote(pitch=self.pitch,
                                duration=asRat(dur),
                                offset=self.start,
                                gliss=bool(self.gliss),
                                playbackGain=self.amp,
                                dynamic=self.dynamic,
                                group=groupid)
        if self.tied:
            note.tiedNext = True
            assert not self.gliss

        notes = [note]
        if self.gliss and not isinstance(self.gliss, bool):
            start = self.end if self.end is not None else None
            groupid = groupid or str(hash(self))
            notes[0].groupid = groupid
            assert self.gliss >= 12, f"self.gliss = {self.gliss}"
            notes.append(scoring.makeNote(pitch=self.gliss, gracenote=True,
                                          offset=start, group=groupid))
            if config['show.glissEndStemless']:
                notes[-1].stem = 'hidden'

        if self.label:
            annot = self._scoringAnnotation()
            if annot is not None:
                notes[0].addAnnotation(annot)
        if self.symbols:
            for symbol in self.symbols:
                symbol.applyToTiedGroup(notes)
        return notes

    def _asTableRow(self) -> list[str]:
        if self.isRest():
            elements = ["REST"]
        else:
            notename = pt.m2n(self.pitch)
            elements = [notename]
            config = getConfig()
            if config['repr.showFreq']:
                elements.append("%dHz" % int(self.freq))
            if self.amp is not None and self.amp < 1:
                elements.append("%ddB" % round(pt.amp2db(self.amp)))
        if self.dur:
            if self.dur >= MAXDUR:
                elements.append("dur=inf")
            else:
                elements.append(f"{_util.showTime(self.dur)}♩")
        if self.tied:
            elements[-1] += "~"
        if self.start is not None:
            elements.append(f"start={_util.showTime(self.start)}")
        if self.gliss:
            if isinstance(self.gliss, bool):
                elements.append(f"gliss={self.gliss}")
            else:
                elements.append(f"gliss={pt.m2n(self.gliss)}")
        return elements

    def __repr__(self) -> str:
        if self.isRest():
            if self.start is not None:
                return f"Rest:{_util.showTime(self.dur)}♩:start={_util.showTime(self.start)}"
            return f"Rest:{_util.showTime(self.dur)}♩"
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
            pitch = self.pitch + other
            gliss = self.gliss if isinstance(self.gliss, bool) else self.gliss + other
            return self.clone(pitch=pitch, gliss=gliss)
        raise TypeError(f"can't add {other} ({other.__class__}) to a Note")

    def __xor__(self, freq) -> Note: return self.freqShift(freq)

    def __sub__(self, other: num_t) -> Note:
        return self + (-other)

    def quantizePitch(self, step=0.) -> Note:
        """
        Returns a new Note, rounded to step.

        If step is 0, the default quantization value is used (this can be
        configured via ``getConfig()['semitoneDivisions']``
        """
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        return self.clone(pitch=round(self.pitch / step) * step)

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace,
                     ) -> list[SynthEvent]:
        if self.isRest():
            return []
        conf = workspace.config
        scorestruct = workspace.scorestruct
        if self._playargs:
            playargs.overwriteWith(self._playargs)
        playargs.fillWithConfig(conf)
        if self.amp is not None:
            amp = self.amp
        else:
            if conf['play.useDynamics']:
                dyn = self.dynamic or conf['play.defaultDynamic']
                amp = workspace.dynamicCurve.dyn2amp(dyn)
            else:
                amp = conf['play.defaultAmplitude']

        endmidi = self.gliss if not isinstance(self.gliss, bool) else self.pitch
        start = self.start or 0.
        dur = self.dur or Rat(1)
        starttime = float(scorestruct.beatToTime(start))
        endtime   = float(scorestruct.beatToTime(start + dur))
        if starttime >= endtime:
            raise ValueError(f"Trying to play an event with 0 or negative duration: {endtime-starttime}. "
                             f"Object: {self}")
        bps = [[starttime, self.pitch, amp],
               [endtime,   endmidi,    amp]]
        if sustain:=playargs.get('sustain'):
            bps.append([endtime + sustain, endmidi, amp])

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
        return self.clone(pitch=pitch, gliss=gliss)

    def trill(self,
              other: Note,
              notedur: time_t = Rat(1, 8)
              ) -> Chain:
        """
        Create a Chain of Notes representing a trill

        Args:
            other: the second note of the trill (can also  be a chord)
            notedur: duration of each note. This value will only be used
                if the trill notes have an unset duration

        Returns:
            A realisation of the trill as a :class:`Chain` of at least the
            given *dur* (can be longer if *dur* is not a multiple
            of *notedur*)
        """
        totaldur = self.resolvedDur()
        note1 = self.clone(dur=notedur)
        note2 = asNote(other, dur=notedur)
        seq = Chain([note1, note2])
        return seq.cycle(totaldur)


def Rest(dur: time_t, start: time_t = None, label='') -> Note:
    """
    Creates a Rest. A Rest is a Note with pitch 0 and amp 0.

    To test if an item is a rest, call :meth:`~MusicObj.isRest`

    Args:
        dur: duration of the Rest
        start: start of the Rest

    Returns:
        the created rest
    """
    assert dur and dur > 0
    return Note(pitch=0, dur=dur, start=start, amp=0, label=label, _init=False)


def asNote(n: Note|float|str, **kws) -> Note:
    """
    Convert *n* to a Note, optionally setting its amp, start or dur

    Args:
        n: the pitch
        kws: any keyword passed to Note

    Returns:
        the corresponding Note

    """
    return n if isinstance(n, Note) else Note(n, **kws)


class Chord(MusicEvent):
    """
    A Chord is a stack of Notes

    a Chord can be instantiated as::

        Chord(note1, note2, ...)
        Chord([note1, note2, ...])
        Chord("C4 E4 G4", ...)

    Where each note is either a Note, a notename ("C4", "E4+", etc) or a midinote

    Args:
        amp: the amplitude (volume) of this chord
        dur: the duration of this chord (in quarternotes)
        start: the start time (in quarternotes)
        gliss: either a list of end pitches (with the same size as the chord), or
            True to leave the end pitches unspecified (a gliss to the next chord)
        label: if given, it will be used for printing purposes

    Attributes:
        amp: the amplitude of the chord itself (each note can have an individual amp)
        notes: the notes which build this chord
        gliss: if True, this Chord makes a gliss to another chord. Also a list
            of pitches can be given as gliss, these indicate the end pitch of the gliss
            as midinote
        tied: is this Chord tied to another Chord?
    """

    __slots__ = ('amp', 'gliss', 'notes', 'tied', 'dynamic')

    def __init__(self,
                 notes: Union[list[Union[Note, int, float, str]], str],
                 dur: time_t = None,
                 amp: float = None,
                 start: time_t = None,
                 gliss: Union[str, Sequence[pitch_t], bool] = False,
                 label: str = '',
                 tied = False,
                 dynamic: str = '',
                 properties: dict[str, Any] = None,
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
            start: the start time (in quarternotes)
            gliss: either a list of end pitches (with the same size as the chord), or
                True to leave the end pitches unspecified (a gliss to the next chord)
            label: if given, it will be used for printing purposes
        """
        self.amp = amp
        if dur is not None:
            assert dur > 0
            dur = asRat(dur)

        if not notes:
            super().__init__(dur=None, start=None, label=label)
            return

        # notes might be: Chord([n1, n2, ...]) or Chord("4c 4e 4g", ...)
        if isinstance(notes, str):
            notes = [Note(n, amp=amp) for n in notes.split()]

        super().__init__(dur=dur, start=start, label=label, properties=properties)
        if _init:
            notes2 = []
            for n in notes:
                if isinstance(n, Note):
                    if n.dur is None and n.start is None:
                        notes2.append(n)
                    else:
                        notes2.append(n.clone(dur=None, start=None))
                elif isinstance(n, (int, float, str)):
                    notes2.append(Note(n, amp=amp))
                else:
                    raise TypeError(f"Expected a Note or a pitch, got {n}")
            notes2.sort(key=lambda n: n.pitch)
            notes = notes2

            if not isinstance(gliss, bool):
                gliss = pt.as_midinotes(gliss)
                assert len(gliss) == len(self.notes), (f"The destination chord of the gliss should have "
                                                       f"the same length as the chord itself, "
                                                       f"{self.notes=}, {gliss=}")
        self.notes: list[Note] = notes
        self.gliss: Union[bool, list[float]] = gliss
        self.tied: bool = tied
        self.dynamic: str = dynamic

    def copy(self):
        notes = [n.copy() for n in self.notes]
        out = Chord(notes=notes, dur=self.dur, amp=self.amp, start=self.start,
                    gliss=self.gliss, label=self.label, tied=self.tied,
                    dynamic=self.dynamic,
                    _init=False)
        self._copyAttributesTo(out)
        return out

    def __len__(self) -> int:
        return len(self.notes)

    @_overload
    def __getitem__(self, idx) -> Note: ...

    @_overload
    def __getitem__(self, slice) -> Chord: ...

    def __getitem__(self, idx):
        out = self.notes.__getitem__(idx)
        if isinstance(out, list):
            out = self.__class__(out)
        return out

    def __iter__(self) -> Iterator[Note]:
        return iter(self.notes)

    def canBeLinkedTo(self, other: MusicObj) -> bool:
        if self.gliss is True:
            return True
        if isinstance(other, Note):
            if self.tied and any(p == other.pitch for p in self.pitches):
                return True
        elif isinstance(other, Chord):
            if self.tied and any(p in other.pitches for p in self.pitches):
                return True
        return False

    def pitchRange(self) -> Optional[tuple[float, float]]:
        return min(n.pitch for n in self.notes), max(n.pitch for n in self.notes)

    def scoringEvents(self, groupid:str = None, config: CoreConfig = None
                      ) -> list[scoring.Notation]:
        if not config:
            config = getConfig()
        pitches = [note.pitch for note in self.notes]
        annot = self._scoringAnnotation()
        dur = self.dur if self.dur is not None else Rat(1)

        chord = scoring.makeChord(pitches=pitches, duration=dur, offset=self.start,
                                  annotation=annot, group=groupid, dynamic=self.dynamic,
                                  tiedNext=self.tied)
        noteheads = [n.getSymbol('notehead')
                     for n in self.notes]
        if any(noteheads):
            for i, notehead in enumerate(noteheads):
                if notehead:
                    assert isinstance(notehead, symbols.Notehead)
                    notehead.applyToNotehead(chord, i)
        if self.symbols:
            for s in self.symbols:
                s.applyTo(chord)
        notations = [chord]
        if self.gliss:
            chord.gliss = True
            if not isinstance(self.gliss, bool):
                groupid = scoring.makeGroupId(groupid)
                chord.groupid = groupid
                endEvent = scoring.makeChord(pitches=self.gliss, duration=0,
                                             offset=self.end, group=groupid)
                if config['show.glissEndStemless']:
                    endEvent.stem = 'hidden'
                notations.append(endEvent)
        return notations

    def asmusic21(self, **kws) -> m21.stream.Stream:
        cfg = getConfig()
        arpeggio = _musicobjtools.normalizeChordArpeggio(kws.get('arpeggio', None), self, cfg)
        if arpeggio:
            dur = cfg['show.arpeggioDuration']
            notes = [n.clone(dur=dur) for n in self.notes]
            return Chain(notes).asmusic21()
        events = self.scoringEvents()
        scoring.stackNotationsInPlace(events, start=self.start)
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

        data = (self.dur, self.start, self.label, glisshash, self.dynamic,
                symbolshash,
                *(hash(n) for n in self.notes))
        return hash(data)

    def append(self, note:Union[float, str, Note]) -> None:
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
            >>> ch2 = ch.filter(lambda note: (note.pitch % 12) in {0, 4, 7})

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
        raise TypeError("Can't add a Chord to a %s" % other.__class__.__name__)

    def loudest(self, n:int) -> Chord:
        """
        Return a new Chord with the loudest `n` notes from this chord
        """
        return self.copy().sort(key='amp', reverse=True)[:n]

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

    def _resolvePlayargs(self, playargs: PlayArgs, config: dict=None) -> PlayArgs:
        playargs = playargs.filledWith(self.playargs)
        playargs.fillWithConfig(config or getConfig())
        return playargs

    @property
    def pitches(self) -> list[float]:
        return [n.pitch for n in self.notes]

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        conf = workspace.config
        scorestruct = workspace.scorestruct
        if self._playargs:
            playargs.overwriteWith(self._playargs)
        playargs.fillWithConfig(conf)
        if conf['chord.adjustGain']:
            gain = playargs.get('gain', 1.0) / math.sqrt(len(self))
            playargs['gain'] = gain
        endpitches = self.pitches
        if self.gliss:
            if isinstance(self.gliss, list):
                endpitches = self.gliss
            elif self._properties and (glisstarget:=self._properties.get('glisstarget')):
                # This property is filled when resolving a sequence
                endpitches = glisstarget

        events = []
        start, dur = self.resolvedStart(), self.resolvedDur()
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

        for note, endpitch in zip(self.notes, endpitches):
            if note.amp:
                noteamp = note.amp
            elif note.dynamic and useDynamics:
                noteamp = workspace.dynamicCurve.dyn2amp(note.dynamic)
            else:
                noteamp = chordamp
            bps = [[starttime, note.pitch, noteamp],
                   [endtime,   endpitch,   noteamp]]
            if sustain:=playargs.get('sustain'):
                bps.append([endtime + sustain, endpitch, noteamp])
            events.append(SynthEvent.fromPlayArgs(bps=bps, playargs=playargs))
        return events

    def asChain(self) -> Chain:
        """ Convert this Chord to a Chain """
        return Chain(self.notes)

    def __repr__(self):
        # «4C+14,4A 0.1q -50dB»
        elements = [" ".join(pt.m2n(p.pitch) for p in self.notes)]
        if self.dur:
            if self.dur >= MAXDUR:
                elements.append("dur=inf")
            else:
                elements.append(f"{float(self.dur):.3g}♩")
        if self.start is not None:
            elements.append(f'start={float(self.start):.3g}')
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
        elements = f'start={self.start}, dur={self.dur}, gliss={self.gliss}'
        print(f"{'  '*indents}Chord({elements})")
        if self._playargs:
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
        newpitches = [pitchmap(p) for p in self.pitches]
        newnotes = [n.clone(pitch=pitch) for n, pitch in zip(self.notes, newpitches)]
        return self.clone(notes=newnotes,
                          gliss=self.gliss if isinstance(self.gliss, bool) else list(map(pitchmap, self.gliss)))


def _asChord(obj, amp:float=None, dur:float=None) -> Chord:
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


def _itemsAreStacked(items: list[MusicEvent | Chain]) -> bool:
    for item in items:
        if isinstance(item, MusicEvent):
            if item.start is None or item.dur is None:
                return False
        elif isinstance(item, Chain):
            if item.start is None or not _itemsAreStacked(item.items):
                return False
    return True


def stackEvents(events: list[T],
                defaultDur: time_t = Rat(1),
                offset: time_t = Rat(0),
                inplace = False,
                recurse = False,
                check = False
                ) -> list[T]:
    """
    Stack events to the left, making any unset start and duration explicit

    After setting all start times and durations an offset is added, if given

    Args:
        events: the events to modify, either in place or as a copy
        defaultDur: the default duration used when an event has no duration and
            the next event does not have an explicit start
        inplace: if True, events are modified in place
        offset: an offset to add to all start times after stacking them
        recurse: if True, stack also events inside subchains

    Returns:
        the modified events. If inplace is True, the returned events are the
        same as the events passed as input

    """

    if not events:
        raise ValueError("no events given")

    if check and _itemsAreStacked(events):
        return events

    if not inplace:
        events = [ev.copy() for ev in events]
        stackEvents(events=events, defaultDur=defaultDur, offset=offset,
                    inplace=True, recurse=recurse)
        return events
    # All start times given in the events are interpreted as being relative
    # *start* is not taken into account
    now = Rat(0)
    lasti = len(events) - 1
    for i, ev in enumerate(events):
        if ev.start is None:
            ev.start = now
        if isinstance(ev, MusicEvent):
            if ev.dur is None:
                if i == lasti:
                    ev.dur = defaultDur
                else:
                    nextev = events[i+1]
                    if nextev.start is None:
                        ev.dur = defaultDur
                    else:
                        ev.dur = nextev.start - ev.start
            now = ev.end
        elif isinstance(ev, Chain):
            if recurse:
                # stackEvents(ev.items, offset=ev.start, inplace=True, recurse=True)
                stackEvents(ev.items, inplace=True, recurse=True)
            dur = ev.resolvedDur()
            now = ev.start + dur

    if offset:
        for ev in events:
            ev.start += offset
    assert all(ev.start is not None for ev in events)
    assert all(ev.dur is not None for ev in events
               if isinstance(ev, MusicEvent))
    return events


class Chain(MusicObj):
    """
    A Chain is a sequence of Notes, Chords or other Chains

    Attributes:
        items: the items of this Chain. Each item is either a MusicEvent (a Note or Chord)
            or a subchain.
        start: the offset of this chain or None if the start time depends on the
            position of the chain within another chain

    Args:
        items: the items of this Chain. The start time of any object, if given, is
            interpreted as relative to the start of the chain.
        start: start time of the chain itself
        label: a label for this chain
        properties: any properties for this chain. Properties can be anything,
            they are a way for the user to attach data to an object
    """
    _acceptsNoteAttachedSymbols = False

    def __init__(self,
                 items: list[MusicEvent | Chain | str] = None,
                 start: time_t = None,
                 label: str = '',
                 properties: dict[str, Any] = None):
        if start is not None:
            start = asRat(start)
        if items is not None:
            items = [item if isinstance(item, (MusicEvent, Chain)) else asEvent(item)
                     for item in items]
            for i0, i1 in iterlib.pairwise(items):
                assert i0.start is None or i1.start is None or i0.start <= i1.start, f'{i0=}, {i1=}'
        else:
            items = []

        super().__init__(start=start, dur=None, label=label, properties=properties)
        self.items: list[MusicEvent | 'Chain'] = items

    def __hash__(self):
        items = [type(self).__name__, self.label, self.start, len(self.items)]
        if self.symbols:
            items.extend(self.symbols)
        items.extend(self.items)
        out = hash(tuple(items))
        return out

    def clone(self, items=UNSET, start=UNSET, label='', properties=UNSET) -> Chain:
        return Chain(items=self.items if items is UNSET else items,
                     start=self.start if start is UNSET else start,
                     label=self.label if label is UNSET else label,
                     properties=self.properties if properties is UNSET else properties)

    def copy(self) -> Chain:
        items = [item.copy() for item in self.items]
        return Chain(items=items, start=self.start, label=self.label, properties=self._properties)

    def isStacked(self) -> bool:
        """
        True if items in this chain have a defined offset and duration
        """
        return self.start is not None and _itemsAreStacked(self.items)

    def fillGapsWithRests(self) -> None:
        """
        Fill any gaps with rests

        A gap is produced when an event within a chain has an explicit start time
        later than the offset calculated by stacking the previous objects in terms
        of their duration
        """
        # TODO
        pass


    def flat(self, removeRedundantOffsets=True) -> Chain:
        """
        A flat version of this Chain

        A Chain can contain other Chains. This method serializes all objects inside
        this Chain and any sub-chains to a flat chain of notes/chords.

        If this Chain is already flat, meaning that it does not contain any
        Chains, self is returned unmodified.

        As a side-effect all offsets (start times) are made explicit

        Args:
            removeRedundantOffsets: remove any redundant start times. A start time is
                redundant if it merely confirms the time offset of an object as
                determined by the durations of the previous objects.

        Returns:
            a chain with exclusively Notes and/or Chords
        """
        if all(isinstance(item, MusicEvent) for item in self.items):
            return self
        chain = self.resolved()
        offset = chain.start if chain.start is not None else Rat(0)
        items = _musicobjtools.flattenObjs(chain.items, offset)
        if chain.start is not None:
            for item in items:
                item.start -= chain.start
        out = self.clone(items=items)
        if removeRedundantOffsets:
            out.removeRedundantOffsets()
        return out

    def pitchRange(self) -> Optional[tuple[float, float]]:
        pitchRanges = [item.pitchRange() for item in self.items]
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

    def resolved(self, start: time_t = None) -> Chain:
        """
        Copy of self with explicit times

        The items in the returned object have an explicit start and
        duration.

        .. note:: use a start time of 0 to have an absolute start
            time set for each item.

        Args:
            start: a start time to fill or override self.start.

        Returns:
            a clone of self with dur and start set to explicit
            values

        """
        if start is not None:
            offset = self.resolvedStart() - start
            if offset < 0:
                raise ValueError(f"This would result in a negative offset: {offset}")
            clonedStart = start
        else:
            offset = 0
            clonedStart = self.start
        if self.isStacked():
            return self
        items = stackEvents(self.items, offset=offset, recurse=True)
        return self.clone(items=items, start=clonedStart)

    def resolvedStart(self) -> Rat:
        ownstart = self.start or Rat(0)
        if not self.items:
            return ownstart
        item = self.items[0]
        return ownstart if item.start is None else ownstart + item.start

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        chain = self.flat(removeRedundantOffsets=False)
        conf = workspace.config
        if self._playargs:
            playargs.overwriteWith(self._playargs)
        items = stackEvents(chain.items, inplace=True, offset=self.start)
        if any(n.isGracenote() for n in self.items
               if isinstance(n, (Note, Chord))):
            _musicobjtools.addDurationToGracenotes(items, Rat(1, 14))
        if conf['play.useDynamics']:
            _musicobjtools.fillTempDynamics(items, initialDynamic=conf['play.defaultDynamic'])
        return _musicobjtools.chainSynthEvents(items, playargs=playargs, workspace=workspace)

    def timeShiftInPlace(self, timeoffset):
        if any(item.start is None for item in self.items):
            stackEvents(self.items, inplace=True)
        for item in self.items:
            item.start += timeoffset
        self._changed()

    def movedTo(self, start: time_t):
        offset = start - self.items[0].start
        return self.timeShift(offset)

    def moveTo(self, start: time_t):
        offset = start - self.items[0].start
        self.timeShiftInPlace(offset)

    def resolvedDur(self, start: time_t = None) -> Rat:
        if not self.items:
            return Rat(0)

        defaultDur = Rat(1)
        accum = Rat(0)
        items = self.items
        lasti = len(items) - 1
        if start is None:
            start = self.resolvedStart()

        for i, ev in enumerate(items):
            if ev.start is not None:
                accum = ev.start
            if isinstance(ev, MusicEvent):
                if ev.dur:
                    accum += ev.dur
                elif i == lasti:
                    accum += defaultDur
                else:
                    nextev = items[i + 1]
                    accum += defaultDur if nextev.start is None else nextev.start - accum
            else:
                # a Chain
                accum += ev.resolvedDur()

        return accum

    def append(self, item: Union[Note, Chord]) -> None:
        """
        Append an item to this chain

        Args:
            item: the item to add
        """
        self.items.append(item)
        if len(self.items) > 1:
            butlast = self.items[-2]
            last = self.items[-1]
            if isinstance(butlast, Note) and butlast.gliss is True and isinstance(last, Note):
                butlast.gliss = last.pitch
        self._changed()

    def _changed(self):
        if self.items:
            self.dur = self.resolvedDur()
        else:
            self.start = None
            self.dur = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[MusicEvent]:
        return iter(self.items)

    @_overload
    def __getitem__(self, idx: int) -> MusicEvent: ...

    @_overload
    def __getitem__(self, slice_: slice) -> Chain: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.items[idx]
        else:
            return self.__class__(self.items.__getitem__(idx))

    def _dumpRows(self, indents=0) -> list[str]:
        selfstart = round(float(self.start.limit_denominator(1000)), 3) if self.start is not None else None
        if environment.insideJupyter:
            namew = max((sum(len(n.name) for n in event.notes)+len(event.notes)
                         for event in self.recurse()
                         if isinstance(event, Chord)),
                        default=10)
            header = f"<code>{'  '*indents}</code><strong>{type(self).__name__}</strong> &nbsp;" \
                     f"start: <code>{_util.htmlSpan(selfstart, ':blue1')}</code>"
            rows = [header]
            columnnames = f"{'  ' * indents}{'start'.ljust(6)}{'dur'.ljust(6)}{'name'.ljust(namew)}{'gliss'.ljust(6)}{'dyn'.ljust(5)}playargs"
            row = f"<code>  {_util.htmlSpan(columnnames, ':grey1')}</code>"
            rows.append(row)
            for item in self.items:
                if isinstance(item, MusicEvent):
                    if item.isRest():
                        name = "Rest"
                    elif isinstance(item, Note):
                        name = item.name
                    elif isinstance(item, Chord):
                        name = ",".join(n.name for n in item.notes)
                    else:
                        raise TypeError(f"Expected Note or Chord, got {item}")

                    if item.tied:
                        name += "~"
                    start = f"{float(item.start):.3g}" if item.start is not None else "None"
                    dur = f"{float(item.dur):.3g}" if item.dur is not None else "None"
                    rowtxt = f"{'  '*indents}{start.ljust(6)}{dur.ljust(6)}{name.ljust(namew)}{str(item.gliss).ljust(6)}{str(item.dynamic).ljust(5)}{self._playargs}</code>"
                    row = f"<code>  {_util.htmlSpan(rowtxt, ':blue1')}</code>"
                    rows.append(row)
                elif isinstance(item, Chain):
                    rows.extend(item._dumpRows(indents=indents+1))
            return rows
        else:
            rows = [f"{' ' * indents}Chain"]
            for item in self.items:
                if isinstance(item, MusicEvent):
                    sublines = repr(item).splitlines()
                    for subline in sublines:
                        rows.append(f"{'  ' * (indents + 1)}{subline}")
                else:
                    rows.extend(item._dumpRows(indents=indents+1))
            return rows

    def dump(self, indents=0):
        rows = self._dumpRows(indents=indents)
        if environment.insideJupyter:
            html = '<br>'.join(rows)
            from IPython.display import HTML, display
            display(HTML(html))
        else:
            for row in rows:
                print(row)

    def __repr__(self):
        if len(self.items) < 10:
            itemstr = ", ".join(repr(_) for _ in self.items)
        else:
            itemstr = ", ".join(repr(_) for _ in self.items[:10]) + ", …"
        cls = self.__class__.__name__
        namedargs = []
        if self.start is not None:
            namedargs.append(f'start={self.start}')
        if namedargs:
            info = ', ' + ', '.join(namedargs)
        else:
            info = ''
        return f'{cls}([{itemstr}]{info})'

    def _repr_html_header(self):
        itemcolor = _safeColors['blue2']
        items = self.items if len(self.items) < 10 else self.items[:10]
        itemstr = ", ".join(f'<span style="color:{itemcolor}">{repr(_)}</span>'
                            for _ in items)
        if len(self.items) >= 10:
            itemstr += ", …"
        cls = self.__class__.__name__
        namedargs = []
        if self.start is not None:
            namedargs.append(f'start={self.start}')
        if namedargs:
            info = ', ' + ', '.join(namedargs)
        else:
            info = ''
        return f'{cls}([{itemstr}]{info})'

    def cycle(self, dur: time_t, crop=True):
        """
        Cycle the items in this chain until the given duration is reached

        Args:
            dur: the total duration
            crop: if True, the last event will be cropped to fit
                the given total duration. Otherwise, it will last
                its given duration, even if that would result in
                a total duration longer than the given one

        Returns:
            the resulting Chain
        """
        defaultDur = Rat(1)
        accumDur = Rat(0)
        maxDur = asRat(dur)
        items: list[MusicEvent] = []
        ownitems = stackEvents(self.items)
        for item in iterlib.cycle(ownitems):
            dur = item.dur if item.dur else defaultDur
            if dur > maxDur - accumDur:
                if crop:
                    dur = maxDur - accumDur
                else:
                    break
            if item.dur is None or item.start is not None:
                item = item.clone(dur=dur, start=None)
            assert isinstance(item, MusicEvent)
            items.append(item)
            accumDur += item.dur
            if accumDur == maxDur:
                break
        return self.__class__(items, start=self.start)

    def removeRedundantOffsets(self):
        """
        Remove over-secified start times in this Chain **inplace**
        """
        # This is the relative position (independent of the chain's start)
        now = Rat(0)
        for item in self.items:
            if isinstance(item, MusicEvent):
                if item.dur is None:
                    raise ValueError(f"This Chain contains events with unspecified duration: {item}")
                if item.start is None:
                    now += item.dur
                else:
                    if item.start < now:
                        raise ValueError(f"Items overlap: {item}, {now=}")
                    elif item.start > now:
                        now = item.end
                    else:
                        # item.start == now
                        item.start = None
                        now += item.dur
            elif isinstance(item, Chain):
                item.removeRedundantOffsets()
        if self.start == 0:
            self.start = None

    def asVoice(self) -> Voice:
        """Convert this Chain to a Voice"""
        resolved = self.resolved(start=0)
        resolved.removeRedundantOffsets()
        return Voice(resolved.items, label=self.label, start=0)

    def makeVoices(self) -> list[Voice]:
        return [self.asVoice()]

    def scoringEvents(self, groupid: str = None, config: CoreConfig = None
                      ) -> list[scoring.Notation]:
        """
        Returns the scoring events corresponding to this object

        Args:
            groupid: if given, all events are given this groupid

        Returns:
            the scoring notations representing this object
        """
        if config is None:
            config = getConfig()
        items = self.flat(removeRedundantOffsets=False)
        notations: list[scoring.Notation] = []
        for item in items:
            notations.extend(item.scoringEvents(groupid=groupid, config=config))
        scoring.stackNotationsInPlace(notations)
        if self.start is not None and self.start > 0:
            for notation in notations:
                notation.offset += self.start

        for n0, n1 in iterlib.pairwise(notations):
            if n0.tiedNext and not n1.isRest:
                n1.tiedPrev = True

        if self.symbols:
            for s in self.symbols:
                for n in notations:
                    s.applyTo(n)
        return notations

    def scoringParts(self, options: scoring.render.RenderOptions = None
                     ) -> list[scoring.Part]:
        notations = self.scoringEvents(config=getConfig())
        if not notations:
            return []
        scoring.stackNotationsInPlace(notations)
        part = scoring.Part(notations, label=self.label)
        return [part]

    def quantizePitch(self:T, step=0.) -> Chain:
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        items = [i.quantizePitch(step) for i in self.items]
        return self.clone(items=items)

    def timeShift(self, timeoffset: time_t) -> Chain:
        if self.start is not None:
            return self.clone(start=self.start+timeoffset)
        items = stackEvents(self.items, offset=timeoffset)
        return self.clone(items=items)

    def timeTransform(self, timemap: Callable[[Rat], Rat], inplace=False) -> Chain:
        start = self.resolvedStart()
        start2 = timemap(start)
        if inplace:
            stackEvents(self.items, inplace=True)
            for item in self.items:
                item.start = timemap(item.start + start) - start2
                item.dur = timemap(item.end + start) - start2 - item.start
            self.start = start2
            return self
        else:
            items = stackEvents(self.items, inplace=False)
            for item in items:
                item.start = timemap(item.start + start) - start2
                item.dur = timemap(item.end + start) - start2 - item.start
            return self.clone(items=items, start=start2)

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Chain:
        newitems = [item.pitchTransform(pitchmap) for item in self.items]
        return self.clone(items=newitems)

    def recurse(self, reverse=False) -> Iterator[MusicEvent]:
        """
        Yields all Notes/Chords in this chain, recursing through sub-chains if needed

        This method guarantees that the yielded events are the actual objects included
        in this chain or its sub-chains. This is usefull when used in combination with
        methods like addSpanner, which modify the objects themselves.

        Args:
            reverse: if True, recurse the chain in reverse

        Returns:
            an iterator over all notes/chords within this chain and its sub-chains

        """
        if not reverse:
            for item in self.items:
                if isinstance(item, MusicEvent):
                    yield item
                elif isinstance(item, Chain):
                    yield from item.recurse(reverse=False)
        else:
            for item in reversed(self.items):
                if isinstance(item, MusicEvent):
                    yield item
                else:
                    yield from item.recurse(reverse=True)

    def addSpanner(self, spannercls: str, endobj: MusicObj = None) -> None:
        first = next(self.recurse())
        last = next(self.recurse(reverse=True))
        first.addSpanner(spannercls, endobj=last)


def makeGracenote(pitch, slash=True) -> Note | Chord:
    """
    Create a gracenote

    The resulting gracenote can be a note or a chord, depending on pitch

    Args:
        pitch: a single pitch (as midinote, notename, etc) or a list of pitches
        slash: if True, the gracenote will be marked as slashed

    Returns:
        the Note/Chord representing the gracenote. A gracenote is basically a
        note/chord with 0 duration

    """
    out = asEvent(pitch, dur=0)
    assert isinstance(out, (Note, Chord))
    out.properties['grace'] = True
    out.properties['grace-slash'] = slash
    return out


class Voice(Chain):
    """
    A Voice is a sequence of non-overlapping objects

    It is **very** similar to a Chain, the only difference being that its start
    is always 0.

    Voice vs Chain
    ~~~~~~~~~~~~~~

    * A Voice can contain a Chain, but not vice versa.
    * A Voice does not have a start offset, its start is always 0.
    """

    _acceptsNoteAttachedSymbols = False

    def __init__(self,
                 items: list[Union[MusicEvent, str]] = None,
                 label=''):
        super().__init__(items=items, label=label, start=Rat(0))


def asEvent(obj, **kws) -> MusicEvent:
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
    if isinstance(obj, MusicEvent):
        return obj
    elif isinstance(obj, str):
        if " " in obj:
            return Chord(obj.split(), **kws)
        elif "," in obj:
            notedef = _util.parseNote(obj)
            dur = kws.pop('dur', None) or notedef.dur
            if notedef.properties:
                kws = misc.dictmerge(notedef.properties, kws)
            return Chord(notedef.pitch, dur=dur, **kws)
        return Note(obj, **kws)
    elif isinstance(obj, (list, tuple)):
        return Chord(obj, **kws)
    elif isinstance(obj, (int, float)):
        return Note(obj, **kws)
    else:
        raise TypeError(f"Cannot convert {obj} to a Note or Chord")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if environment.insideJupyter:
#    from . import jupytertools
#    jupytertools.m21JupyterHook()


