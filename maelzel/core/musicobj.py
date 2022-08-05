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
from numbers import Rational

from emlib.misc import firstval as _firstval

from emlib import misc
from emlib import mathlib
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
    'asNote',
    'Rest',

    'Chord',
    'asChord',

    'Line',

    'asEvent',
    'stackEvents',

    'MusicObjList',
    'Chain',
    'Voice',
    'Score',

    'makeTrill',
    'packInVoices',
    'resetImageCache',
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


@functools.total_ordering
class Note(MusicEvent):
    """
    In its simple form, a Note is used to represent a pitch.

    A Note must have a pitch. It is possible to specify
    a duration, a start time (.start), an amplitude
    and an endpitch, resulting in a glissando.

    Internally the pitch of a Note is represented as a fractional
    midinote, so there is no concept of enharmonicity. Notes created
    as ``Note(61)``, ``Note("4C#")`` or ``Note("4Db")`` will all result
    in the same Note.

    Like all MusicObjs, a Note makes a clear division between
    the value itself and the representation as notation or as sound.
    Playback specific options (instrument, pan position, etc) can be
    set via the :meth:`~Note.setPlay` method. Any aspects regarding notation
    (articulation, enharmonic variant, etc) can be set via :meth:`~Note.setSymbol`

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
        _init: if True, fast initialization is performed, skipping any checks

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
                assert dynamic in scoring.definitions.availableDynamics

        super().__init__(dur=dur, start=start, label=label, properties=properties)
        self.pitch: float = pitch
        self.amp: Optional[float] = amp
        self._gliss: Union[float, bool] = gliss
        self.tied = tied
        self.dynamic: Optional[str] = dynamic

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
                   dynamic=self.dynamic, start=self.start, _init=False,
                   label=self.label)
        if self.symbols:
            out.symbols = self.symbols.copy()
        if self._playargs:
            out._playargs = self._playargs.copy()
        if self._properties:
            out._properties = self._properties.copy()
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
        if self.symbols:
            out.symbols = self.symbols.copy()
        if self._playargs:
            out._playargs = self._playargs.copy()
        if self._properties:
            out._properties = self._properties.copy()
        return out

    def __hash__(self) -> int:
        hashsymbols = hash(tuple(self.symbols)) if self.symbols else 0
        return hash((self.pitch, self.dur, self.start, self.gliss, self.label, self.dynamic,
                     hashsymbols))

    def asChord(self) -> Chord:
        """ Convert this Note to a Chord of one note """
        gliss = self.gliss
        if gliss and isinstance(self.gliss, (int, float)):
            gliss = [gliss]
        return Chord([self], amp=self.amp, dur=self.dur, start=self.start,
                     gliss=gliss, label=self.label)

    def isRest(self) -> bool:
        """ Is this a Rest? """
        return self.amp == 0

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
        return Note(pt.f2m(self.freq * n))

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
            elements = [pt.m2n(self.pitch)]
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
            return f"Rest({_util.showTime(self.dur)}♩)"
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
            if conf['play.useDynamic']:
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
        if playargs.sustain:
            bps.append([endtime + playargs.sustain, endmidi, amp])

        return [SynthEvent.fromPlayArgs(bps=bps, playargs=playargs, tiednext=self.tied)]

    def makeGliss(self, endpitch:pitch_t=None, dur:time_t=None, endamp:float=None
                  ) -> Line:
        """
        Create a Line between this Note and ``endpitch``
        
        If this note does not have a defined gliss note (note.gliss is set) then
        endpitch must be given

        If this Note has no predefined duration, ``dur`` must be specified

        Args:
            endpitch: the destination pitch, will override this note's own
                end pitch (given as the value of note.gliss)
            dur: the duration of the gliss., in case this Note has not
                defined duration
            endamp: the destination amplitude
        """
        if endpitch is None:
            assert self.gliss > 1
            endpitch = self.gliss
        else:
            endpitch = _util.asmidi(endpitch)
        dur = dur or self.resolvedDur()
        startamp = self.resolvedAmp()
        endamp = _firstval(endamp, self.amp, startamp)
        start = self.start or 0
        breakpoints = [(start, self.pitch, startamp),
                       (start+dur, endpitch, endamp)]
        return Line(breakpoints)

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


def Rest(dur:time_t=1, start:time_t=None, label='') -> Note:
    """
    Creates a Rest. A Rest is a Note with pitch 0 and amp 0.

    To test if an item is a rest, call :meth:`~MusicObj.isRest`

    Args:
        dur: duration of the Rest
        start: start of the Rest

    Returns:
        the creaed rest
    """
    assert dur is not None and dur > 0
    return Note(pitch=0, dur=dur, start=start, amp=0, label=label)


def asNote(n: Union[float, str, Note],
           amp:float=None, dur:time_t=None, start:time_t=None) -> Note:
    """
    Convert *n* to a Note, optionally setting its amp, start or dur

    Args:
        n: the pitch
        amp: 0-1
        dur: the duration of the Note
        start: the offset of the note

    Returns:
        ``n`` as a Note

    """
    if isinstance(n, Note):
        if any(x is not None for x in (amp, dur, start)):
            return n.clone(amp=amp, dur=dur, start=start)
        return n
    elif isinstance(n, (int, float)):
        return Note(n, amp=amp, dur=dur, start=start)
    elif isinstance(n, str):
        midi = pt.str2midi(n)
        return Note(midi, amp=amp, dur=dur, start=start)
    #elif isinstance(n, Pitch):
    #    return Note(pitch=n.midi)
    raise ValueError(f"cannot express this as a Note: {n} ({type(n)})")


class Line(MusicEvent):
    """ 
    A Line is a sequence of breakpoints
    
    Args:
        bps: breakpoints, a tuple/list of the form (delay, pitch, [amp=1, ...]), where
            delay is the time offset to the beginning of the line; pitch is the pitch
            as notename or midinote and amp is an amplitude between 0-1. If values are
            missing from one row they are carried from the previous
        start: time offset of the line itself
        label: a label to add to the line
        relative: if True, the first value of each breakpoint is a time offset
            from previous breakpoint
    
    A bp has the form ``(delay, pitch, [amp=1, ...])``, where:

    - delay is the time offset to the first breakpoint.
    - pitch is the pitch as midinote or notename
    - amp is the amplitude (0-1), optional

    pitch, amp and any other following data can be 'carried'::

        Line((0, "D4"), (1, "D5", 0.5), ..., fade=0.5)

    Also possible::

    >>> bps = [(0, "D4"), (1, "D5"), ...]
    >>> Line(bps)   # without *

    a Line stores its breakpoints as: ``[delayFromFirstBreakpoint, pitch, amp, ...]``

    Attributes:
        bps: the breakpoints of this line, a list of tuples of the form
            ``(delay, pitch, [amp, ...])``, where delay is always relative
            to the start of the line (the delay of the first breakpoint is always 0)
    """

    __slots__ = ('bps', 'dynamic', 'tied', 'gliss')
    _acceptsNoteAttachedSymbols = True

    def __init__(self,
                 *bps,
                 start: num_t = None,
                 label="",
                 relative=False,
                 dynamic='',
                 tied = False,
                 gliss = False):
        # [[[0, 60, 1], [1, 60, 2]]]
        if len(bps) == 1 and isinstance(bps[0], list) and isinstance(bps[0][0], (tuple, list)):
            bps = bps[0]

        if any(len(bp) < 2 for bp in bps):
            raise ValueError("A breakpoint should be at least (delay, pitch)", bps)

        if len(bps[0]) == 2:
            bps[0].append(1.)

        l = len(bps)
        if any(len(bp) != l for bp in bps):
            bps = _util.carryColumns(bps)

        bps = _util.as2dlist(bps)

        for bp in bps:
            bp[0] = asRat(bp[0])
            bp[1] = _util.asmidi(bp[1])

        if relative:
            now = 0
            for bp in bps:
                now += bp[0]
                bp[0] = now

        if bps[0][0] > 0 and start is not None:
            dt = bps[0][0]
            start += dt
            for row in bps:
                row[0] -= dt

        assert all(bp1[0] > bp0[0] for bp0, bp1 in iterlib.pairwise(bps))

        super().__init__(dur=bps[-1][0], start=start, label=label)
        self.bps: list[list] = bps
        """The breakpoints of this line, a list of tuples (delay, pitch, [amp, ...])"""

        self.dynamic = dynamic
        self.gliss = gliss
        self.tied = tied

    def resolvedStart(self) -> Rat:
        if self.start is not None:
            return self.start
        return self.bps[0][0]

    def resolvedDur(self, start: time_t = None) -> Rat:
        return self.bps[-1][0] - self.bps[0][0]

    def offsets(self) -> list[Rat]:
        """ Return absolute offsets of each breakpoint """
        start = self.start or Rat(0)
        return [bp[0] + start for bp in self.bps]

    def translateBreakpointsToAbsTime(self,
                                      score: ScoreStruct,
                                      asfloat=True
                                      ) -> list[list[num_t, ...]]:
        """
        Translate beat to absolute time within the breakpoints of this Line

        Args:
            score: the scorestructure to use to translate quarter notes to
                abs time
            asfloat: if True, convert all times to float

        Returns:
            a copy of this Lines breakpoints where all timing is given in
            absolute time
        """
        start = self.start or Rat(0)
        bps = []
        for bp in self.bps:
            bp2 = bp.copy()
            t = score.beatToTime(bp[0] + start)
            if asfloat:
                t = float(t)
            bp2[0] = t
            bps.append(bp2)
        return bps

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        conf = workspace.config
        if self._playargs:
            playargs.overwriteWith(self._playargs)
        playargs.fillWithConfig(conf)
        bps = self.translateBreakpointsToAbsTime(workspace.scorestruct, asfloat=True)
        return [SynthEvent.fromPlayArgs(bps, playargs=playargs)]

    def __hash__(self):
        rowhashes = [hash(tuple(bp)) for bp in self.bps]
        attrs = (self.start, self.dynamic, self.gliss, self.tied)
        rowhashes.extend(attrs)
        return hash(tuple(rowhashes))

    def __repr__(self):
        return f"Line(start={self.start}, bps={self.bps})"

    def quantizePitch(self, step=0) -> Line:
        """
        Returns a new line, rounded to step

        Some breakpoints might not be relevant after quantization. These
        will be simplified, meaning that the returned line might have less
        breakpoints than the original line.

        Args:
            step: the semitone division used (0.5 = quantize to nearest 1/4 tone)

        Returns:
            a line with its pitches quantized by the given step
        """
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        bps = [ (bp[0], _util.quantizeMidi(bp[1], step)) + bp[2:]
                for bp in self.bps ]
        if len(bps) >= 3:
            bps = misc.simplify_breakpoints(bps, coordsfunc=lambda bp:(bp[0], bp[1]),
                                            tolerance=0.01)
        return Line(bps)

    def scoringEvents(self,
                      groupid: str = None,
                      config: CoreConfig = None
                      ) -> list[scoring.Notation]:
        if config is None:
            config = getConfig()
        start = self.start or Rat(0)
        # groupid = scoring.makeGroupId(groupid)
        notations: list[scoring.Notation] = []

        for bp0, bp1 in iterlib.pairwise(self.bps):
            pitch = bp0[1]
            dur = bp1[0] - bp0[0]
            offset = bp0[0] + start
            ev = scoring.makeNote(pitch=pitch, offset=offset, duration=dur,
                                  gliss=bp0[1] != bp1[1], group=groupid)
            if bp0[1] == bp1[1]:
                ev.tiedNext = True
            notations.append(ev)

        if self.bps[-1][1] != self.bps[-2][1]:
            # add a last note if last pair needed a gliss (to have a destination note)
            n = notations[-1]
            n.gliss = True
            lastbp = self.bps[-1]
            notations.append(scoring.makeNote(pitch=lastbp[1],
                                              offset=lastbp[0] + start,
                                              gracenote=True,
                                              group=groupid))
        if notations:
            scoring.fixOverlap(notations)
            annot = self._scoringAnnotation()
            if annot:
                notations[0].addAnnotation(annot)
        if self.symbols:
            for symbol in self.symbols:
                symbol.applyToTiedGroup(notations)
        for n0, n1 in iterlib.window(notations, 2):
            if n0.tiedNext:
                n1.tiedNext = True
        if self.tied:
            logger.warning("Tied lines are not yet supported")
        if self.gliss:
            logger.warning("Gliss at the end of lines is not yet supported")

        if self.dynamic:
            notations[0].dynamic = self.dynamic
        return notations

    def timeTransform(self, timemap: Callable[[num_t], num_t], inplace=False) -> Line:
        if not inplace:
            bps = []
            for bp in self.bps:
                t1 = timemap(bp[0] + self.start)
                bp2 = bp.copy()
                bp2[0] = t1
                bps.append(bp2)
            return Line(bps, label=self.label)
        else:
            for bp in self.bps:
                t1 = timemap(bp[0] + self.start)
                bp[0] = timemap(bp[0] + self.start)

    def dump(self, indents=0):
        attrs = []
        if self.start:
            attrs.append(f"start={self.start}")
        if self.label:
            attrs.append(f"label={self.label}")
        infostr = ", ".join(attrs)
        print("Line:", infostr)
        rows = []
        for bp in self.bps:
            row = ["%.6g" % _ for _ in bp]
            rows.append(row)
        headers = ("start", "pitch", "amp", "p4", "p5", "p6", "p7", "p8")
        misc.print_table(rows, headers=headers, showindex=False, )

    def timeShift(self, timeoffset: time_t) -> Line:
        return Line(self.bps, start=(self.start or Rat(0)) + timeoffset)

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Line:
        newpitches = [pitchmap(bp[1]) for bp in self.bps]
        newbps = self.bps.copy()
        for bp, pitch in zip(newbps, newpitches):
            bp[1] = pitch
        return self.clone(bps=newbps)


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
                 properties: dict[str, Any] = None
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
        self._hash = 0
        if dur is not None:
            assert dur > 0
            dur = asRat(dur)

        if not notes:
            super().__init__(dur=None, start=None, label=label)
            return

        # notes might be: Chord([n1, n2, ...]) or Chord("4c 4e 4g", ...)
        if isinstance(notes, str):
            notes = [Note(n, amp=amp) for n in notes.split()]

        # determine dur & start
        if dur is None:
            dur = max((n.dur for n in notes if isinstance(n, Note) and n.dur is not None),
                      default=None)
        if start is None:
            start = min((n.start for n in notes if isinstance(n, Note) and n.start is not None),
                        default=None)
        super().__init__(dur=dur, start=start, label=label, properties=properties)

        notes2 = []
        for n in notes:
            if isinstance(n, Note):
                if n.dur is None and n.start is None:
                    notes2.append(n)
                else:
                    notes2.append(n.clone(dur=None, start=None))
            else:
                notes2.append(asNote(n, amp=amp, dur=None, start=None))
        self.notes: list[Note] = list(set(notes2))
        self.sort()

        if not isinstance(gliss, bool):
            gliss = pt.as_midinotes(gliss)
            assert len(gliss) == len(self.notes), (f"The destination chord of the gliss should have "
                                                   f"the same length as the chord itself, "
                                                   f"{self.notes=}, {gliss=}")
        self.gliss: Union[bool, list[float]] = gliss

        self.tied: bool = tied
        self.dynamic: str = dynamic

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
                                  annotation=annot, group=groupid, dynamic=self.dynamic)

        noteheads = [n.getSymbol('notehead')
                     for n in self.notes]
        if any(noteheads):
            for i, notehead in enumerate(noteheads):
                if notehead:
                    assert isinstance(notehead, symbols.Notehead)
                    notehead.applyToNotehead(chord, i)

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
        if self.symbols:
            for s in self.symbols:
                s.applyToTiedGroup(notations)
        return notations

    def asmusic21(self, **kws) -> m21.stream.Stream:
        config = getConfig()
        arpeggio = _normalizeChordArpeggio(kws.get('arpeggio', None), self)
        if arpeggio:
            dur = config['show.arpeggioDuration']
            notes = [n.clone(dur=dur) for n in self.notes]
            return Chain(notes).asmusic21()
        events = self.scoringEvents()
        scoring.stackNotationsInPlace(events, start=self.start)
        parts = scoring.distributeNotationsByClef(events)
        return notation.renderWithActiveWorkspace(parts, backend='music21').asMusic21()

    def __hash__(self):
        #if self._hash:
        #    return self._hash
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
        self._hash = hash(data)
        return self._hash

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
            return Chord(self.notes + asChord(other).notes)
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
        gain = _ if (_:=playargs.gain) is not None else 1.0
        if conf['chord.adjustGain']:
            gain *= 1 / math.sqrt(len(self))
        playargs.gain = gain
        endpitches = self.pitches
        if self.gliss:
            if isinstance(self.gliss, list):
                endpitches = self.gliss
            elif self._properties and (glisstarget:=self._properties.get('glisstarget')):
                # This property is filled when resolving a sequence
                endpitches = glisstarget

        events = []
        dur = self.resolvedDur()
        start = self.start or 0
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
            if playargs.sustain:
                bps.append([endtime + playargs.sustain, endpitch, noteamp])
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


def asChord(obj, amp:float=None, dur:float=None) -> Chord:
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
        assert isinstance(out, Chord)
    elif isinstance(obj, (int, float)):
        out = Chord([Note(obj)])
    else:
        raise ValueError(f"cannot express this as a Chord: {obj}")
    if amp is not None or dur is not None:
        out = out.clone(amp=amp, dur=dur)
    return out


def _normalizeChordArpeggio(arpeggio: Union[str, bool], chord: Chord) -> bool:
    config = getConfig()
    if arpeggio is None: arpeggio = config['show.arpeggiateChord']

    if isinstance(arpeggio, bool):
        return arpeggio
    elif arpeggio == 'auto':
        return chord._isTooCrowded()
    else:
        raise ValueError(f"arpeggio should be True, False, 'auto' (got {arpeggio})")


def stackEventsRelative(events: list[T],
                        defaultDur: time_t = Rat(1),
                        inplace=False,
                        offset: time_t = Rat(0)
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

    Returns:
        the modified events. If inplace is True, the returned events are the
        same as the events passed as input

    """

    if not events:
        raise ValueError("no events given")

    if not inplace:
        events = [ev.copy() for ev in events]
        return stackEventsRelative(events=events, defaultDur=defaultDur, offset=offset, inplace=True)

    # All start times given in the events are interpreted as being relative
    # *start* is not taken into account
    now = Rat(0)
    lasti = len(events) - 1
    for i, ev in enumerate(events):
        if ev.start is None:
            ev.start = now
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

    if offset:
        for ev in events:
            ev.start += offset
    assert all(ev.start is not None and ev.dur is not None for ev in events)
    return events


def stackEvents(events: list[MusicObj],
                defaultDur: time_t = Rat(1),
                start: time_t = Rat(0),
                inplace=False,
                force=False
                ) -> list[MusicObj]:
    """
    Place `events` in succession

    Args:
        events: the events (Notes / Chords) to stack against each other
        defaultDur: the duration given to events which don't have an explicit duration
        start: the start time for the event stack. This will be used only if the first
            event doesn't have an explicit start. To force the first element to be shifted
            to this value, the *force* parameter needs to be ``True``
        inplace: if True, the events are modified in place. In this case the returned
            events list is the same as the list given.

    Returns:
        the resulting events. It is ensured that in the returned events there is no
        intersection between the events and all have start and dur set. The returned
        events are always copies/clones of the input events

    .. note::

        When a note has no duration, the following note should have an explitic
        start time

        >>> notes = [
        ...     Note(60, dur=1),
        ...     Note(61, dur=0.5),
        ...     Note(62),
        ...     Note(63, start=4),
        ...     Note(64, start=5, dur=0.5)
        ... ]
        >>> stackEvents(notes)
        [4C:262Hz:1♩:start=0,
         4C#:278Hz:0.5♩:start=1,
         4D:294Hz:2.5♩:start=1.5,
         4D#:312Hz:1♩:start=4,
         4E:331Hz:0.5♩:start=5]
         >>> stackEvents(notes, start=1)
         [4C:262Hz:1♩:start=1,
          4C#:278Hz:0.5♩:start=2,
          4D:294Hz:1.5♩:start=2.5,
          4D#:312Hz:1♩:start=4,
          4E:331Hz:0.5♩:start=5]
    """
    if inplace:
        _stackEventsInPlace(events=events, defaultDur=defaultDur, start=start, force=force)
        return events

    if not events:
        return events

    if all(ev.start is not None and ev.dur is not None for ev in events):
        return [ev.copy() for ev in events]

    if start is None:
        start = events[0].start
        if start is None:
            start = Rat(0)
    elif not force and events[0].start is not None:
        start = events[0].start

    now = start
    assert now is not None and now >= 0

    if len(events) == 1:
        ev = events[0]
        if ev.start is not None and ev.dur is not None:
            return events
        return [ev.clone(start=start if ev.start is None else ev.start,
                         dur=defaultDur if ev.dur is None else ev.dur)]
    out = []
    lasti = len(events) - 1
    for i, ev in enumerate(events):
        if ev.dur is None or ev.start is None:
            start = ev.start if ev.start is not None else now
            if ev.dur:
                dur = ev.dur
            elif i < lasti:
                nextev = events[i+1]
                if nextev.start is not None:
                    dur = nextev.start - start
                    if dur < 0:
                        raise ValueError(f"This event ({ev}) has no explicit duration, but the next "
                                         f"event ({nextev}) has an explicit start before the start of "
                                         "this event")
                else:
                    dur = defaultDur
            else:
                dur = defaultDur
            ev = ev.clone(start=start, dur=dur)
        assert ev.dur > 0
        now = ev.end
        out.append(ev)

    assert all(isinstance(ev.start, Rat)  and ev.start >= 0 and ev.dur >= 0 for ev in out)

    for ev1, ev2 in iterlib.pairwise(out):
        assert ev1.end <= ev2.start, \
            f"{ev1=}, end={round(float(ev1.end), 3)}; {ev2=}, start={round(float(ev2.start), 3)}"
    return out


def _stackEventsInPlace(events: Sequence[MusicObj],
                        defaultDur: time_t = None,
                        start: time_t = Rat(0),
                        force=False
                        ) -> None:
    """
    Similar to stackEvents, but modifies the events themselves

    Args:
        events: the events to stack against each other
        defaultDur: the duration given to events which don't have an explicit duration
        start: the start time for the event stack. This will be used only if the first
            event doesn't have an explicit start. To force the first element to be shifted
            to this value, the *force* parameter needs to be ``True``
        force: if True and the first event has a start time, it is forced
            to use the given start time
    """
    # events = list(iterlib.flatten(events, exclude=(Chord,)))
    if not events:
        return

    if force:
        assert start is not None, f"If force is True, start needs to be given"
        events[0].start = start

    if all(ev.start is not None and ev.dur is not None for ev in events):
        # Nothing to do
        return

    if start is None:
        start = events[0].start
        if start is None:
            start = Rat(0)

    if defaultDur is None:
        defaultDur = Rat(1)

    if len(events) == 1:
        ev = events[0]
        if ev.start is None:
            ev.start = start
        if ev.dur is None:
            ev.dur = defaultDur
        return

    now = events[0].start if events[0].start is not None else start
    assert now is not None and now >= 0
    lasti = len(events) - 1
    for i, ev in enumerate(events):
        if ev.dur is None:
            if i < lasti:
                if (startNext:=events[i + 1].start) is not None:
                    dur = startNext - now
                else:
                    dur = defaultDur
            else:
                dur = defaultDur
            ev.start = now
            ev.dur = dur
        elif ev.start is None:
            ev.start = now
        elif ev.start < now:
            ev.moveTo(now)
        assert ev.dur is not None and ev.start is not None
        now = ev.end
    for ev1, ev2 in iterlib.pairwise(events):
        assert ev1.start <= ev2.start, f"{ev1=}, {ev2=}"
        assert ev1.end <= ev2.start, f"{ev1=}, {ev2=}"


def _fillDurationsIfPossible(items: list[MusicObj]) -> None:
    now = items[0].start
    for i, item in items[:-1]:
        if item.start is not None:
            now = item.start
        if item.dur is None:
            nextitem = items[i+1]
            if nextitem.start is not None and now is not None:
                item.dur = nextitem.start - now
        if now is not None and item.dur is not None:
            now += item.dur
        else:
            now = None


def _sequentialDuration(items: list[MusicObj]) -> Optional[Rat]:
    if all(item.start is None for item in items):
        if all(item.dur is not None for item in items):
            return sum(item.dur for item in items)
        return None


class MusicObjList(MusicObj):
    """
    A sequence of music objects

    This class serves as a base for all container classes (Chain, Group, Voice)
    **It should not be instantiated by itself**

    """
    def __init__(self,
                 label='',
                 properties: dict[str, Any] = None,
                 start: time_t = None,
                 dur: time_t = None):
        """a list of MusicObj inside this container"""

        super().__init__(dur=dur, start=start, label=label, properties=properties)

    def append(self, obj: MusicObj) -> None:
        self._getItems().append(obj)
        self._changed()

    def _changed(self) -> None:
        items = self._getItems()
        self.start = min(it.start for it in items)
        end = max(it.end for it in items)
        self.dur = end - self.start
        super()._changed()

    def _getItems(self) -> list[MusicObj]:
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._getItems())

    def __getitem__(self, idx):
        return self._getItems().__getitem__(idx)

    def __len__(self):
        return len(self._getItems())

    def __hash__(self):
        items = [type(self).__name__, self.label, self.start, len(self)]
        if self.symbols:
            items.extend(self.symbols)
            # items.extend(hash(s) for s in self.symbols)
        items.extend(self._getItems())
        out = hash(tuple(items))
        return out

    def pitchRange(self) -> Optional[tuple[float, float]]:
        pitchRanges = [item.pitchRange() for item in self]
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

    def scoringEvents(self, groupid:str=None, config: CoreConfig = None
                      ) -> list[scoring.Notation]:
        """
        Returns the scoring events corresponding to this object

        Args:
            groupid: if given, all events are given this groupid

        Returns:
            the scoring notations representing this object
        """
        raise NotImplementedError()

    def _mergedItems(self) -> list[MusicObj]:
        return self._getItems()

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        playargs.fillWith(self.playargs)
        return misc.sumlist(item._synthEvents(playargs.copy(), workspace)
                            for item in self._mergedItems())

    def quantizePitch(self:T, step=0.) -> T:
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        items = [i.quantizePitch(step) for i in self]
        return self.clone(items=items)

    def timeShift(self, timeoffset: time_t):
        resolved = self.resolved()
        items = [item.timeShift(timeoffset) for item in resolved]
        return self.clone(items=items)

    def timeTransform(self:T, timemap: Callable[[Rat], Rat], inplace=False) -> T:
        if inplace:
            stackEvents(self.items, start=self.start, inplace=True)
            for item in self.items:
                item.timeTransform(timemap, inplace=True)
            return self
        else:
            items = stackEvents(self.items, start=self.start)
            for item in items:
                item.timeTransform(timemap, inplace=True)
            return self.clone(items=items)

    def pitchTransform(self: T, pitchmap: Callable[[float], float]) -> T:
        newitems = [item.pitchTransform(pitchmap) for item in self]
        return self.clone(items=newitems)

    def dump(self, indents=0):
        print(f'{"  "*indents}{repr(self)}')
        if self._playargs:
            print("  "*(indents+1), self.playargs)
        for item in self:
            item.dump(indents+1)

    def makeVoices(self) -> list[Voice]:
        """
        Construct a list of Voices from this object
        """
        return _musicobjtools.packInVoices(self._getItems())

    def adaptToScoreStruct(self, newstruct: ScoreStruct, oldstruct: ScoreStruct = None):
        newitems = [item.adaptToScoreStruct(newstruct, oldstruct)
                    for item in self]
        return self.clone(items=newitems)

    def addSpanner(self, spannercls: str, endobj: MusicObj = None) -> None:
        """
        Adds a spanner symbol to this object

        A spanner is a slur, line or any other symbol attached to two or more
        objects. A spanner always has a start and an end.

        Args:
            spannercls: one of 'slur', 'cresc', 'dim'
            endobj: not needed

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> a = Note("4C")
            >>> b = Note("4E")
            >>> c = Note("4G")
            >>> chain = Chain([a, b, c])
            >>> chain.addSpanner('slur')

        .. seealso:: :meth:`Spanner.bind() <maelzel.core.symbols.Spanner.bind>`

        """
        assert endobj is None, 'This class does not accept an end object'
        items = self._getItems()
        startobj = items[0]
        if isinstance(startobj, (Note, Chord)):
            startobj.addSpanner(spannercls, endobj=items[-1])
        elif isinstance(startobj, MusicObjList):
            startobj.addSpanner(spannercls)


def _fixGliss(items: list[MusicObj]) -> None:

    for i0, i1 in iterlib.window(items, 2):
        if isinstance(i0, MusicObjList):
            _fixGliss(list(i0))
        elif (isinstance(i0, Note) and isinstance(i1, Note) and
                not i1.isRest() and i0.gliss is True):
            i0.gliss = i1.pitch

def _makeLine(notes: list[Note], workspace: Workspace = None) -> Line:
    assert all(n0.end == n1.start for n0, n1 in iterlib.pairwise(notes))
    bps = []
    for note in notes:
        bp = [note.start, note.pitch, note.resolvedAmp(workspace=workspace)]
        bps.append(bp)
    lastnote = notes[-1]
    if lastnote.dur > 0:
        pitch = lastnote.gliss if lastnote.gliss else lastnote.pitch
        bps.append([lastnote.end, pitch, lastnote.resolvedAmp()])
    return Line(bps, label=notes[0].label)


def _mergeItems(items: list[MusicEvent], workspace: Workspace = None
                ) -> list[MusicEvent]:
    """
    Merge notes/chords with ties/gliss into Lines, which are better capable of
    rendering notation and playback for those cases.

    Notes and Chord without glissandi re appended as-is.
    Notes which can be merged as Lines are fused together
    and the resulting Line is added
    """
    groups = _musicobjtools.groupEvents(items)
    return [_makeLine(item, workspace=workspace)
            if isinstance(item, list) else item
            for item in groups]



class Chain(MusicObj):
    _acceptsNoteAttachedSymbols = False

    def __init__(self,
                 items: list[Union[MusicEvent, str]] = None,
                 start: time_t = None,
                 label: str = '',
                 properties: dict[str, Any] = None):
        if start is not None:
            start = asRat(start)
        if items:
            items = [item if isinstance(item, (Note, Chord, Line)) else asEvent(item)
                     for item in items]
            if all(item.start is not None for item in items):
                items.sort(key=lambda item: item.start)

            # stackEvents(items, start=start, inplace=True)
        super().__init__(start=start, dur=None, label=label, properties=properties)
        self.items: list[MusicEvent] = items
        self._merged = None

    def __hash__(self):
        items = [type(self).__name__, self.label, self.start, len(self.items)]
        if self.symbols:
            items.extend(self.symbols)
        items.extend(self.items)
        out = hash(tuple(items))
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
        items = stackEventsRelative(self.items, offset=offset)
        return self.clone(items=items, start=clonedStart)

    def resolvedStart(self) -> Rat:
        ownstart = self.start or Rat(0)
        if not self.items:
            return ownstart
        item = self.items[0]
        return ownstart if item.start is None else ownstart + item.start

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        conf = workspace.config
        if self._playargs:
            playargs.overwriteWith(self._playargs)
        items = stackEventsRelative(self.items, inplace=False, offset=self.start)
        if any(n.isGracenote() for n in self.items
               if isinstance(n, (Note, Chord))):
            _musicobjtools.addDurationToGracenotes(items, Rat(1, 14))
        if conf['play.useDynamics']:
            _musicobjtools.fillTempDynamics(items, initialDynamic=conf['play.defaultDynamic'])
        return _musicobjtools.chainSynthEvents(items, playargs=playargs, workspace=workspace)

        items = _mergeItems(items, workspace=workspace)
        out = misc.sumlist(item._synthEvents(playargs.copy(), workspace)
                           for item in items)
        return out

    def timeShiftInPlace(self, timeoffset):
        if any(item.start is None for item in self.items):
            stackEventsRelative(self.items, inplace=True)
        for item in self.items:
            item.start += timeoffset
        self._changed()

    def movedTo(self, start: time_t):
        offset = start - self.items[0].start
        return self.timeShift(offset)

    def moveTo(self, start: time_t):
        offset = start - self.items[0].start
        self.timeShiftInPlace(offset)

    def _mergedItems(self) -> list[Union[Note, Chord, Line]]:
        if not self._merged:
            self._merged = _mergeItems(self.items)
        return self._merged

    def resolvedDur(self, start: time_t = None) -> Rat:
        if not self.items:
            return Rat(0)

        defaultDur = Rat(1)
        accum = Rat(0)
        items = self.items
        lasti = len(items) - 1

        for i, ev in enumerate(items):
            if ev.start is not None:
                accum = ev.start
            if ev.dur:
                accum += ev.dur
            elif i == lasti:
                accum += defaultDur
            else:
                nextev = items[i + 1]
                accum += defaultDur if nextev.start is None else nextev.start - accum
        return accum - (start if start is not None else self.resolvedStart())

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
        self._hash = None
        self._merged = None

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

    def dump(self, indents=0):
        if environment.insideJupyter:
            rows = []
            for item in self.items:
                start = round(float(item.start), 3) if item.start is not None else None
                if item.isRest():
                    name = "Rest"
                else:
                    name = item.name if isinstance(item, Note) else ", ".join(n.name for n in item)
                    if item.tied:
                        name += "~"
                dur = "None" if item.dur is None else round(float(item.dur), 3)
                row = (start, name, dur, item.gliss, item.dynamic, str(self._playargs))
                rows.append(row)
            tableHtml = misc.html_table(rows=rows, headers='time pitch dur gliss dynamic playargs'.split())
            start = round(float(self.start.limit_denominator(1000)), 3) if self.start is not None else None
            html = f"<p><strong>{type(self).__name__}</strong>(<small>start={start}</small>)</p>"
            html += tableHtml
            from IPython.display import HTML, display
            display(HTML(html))
        else:
            print(f"{' ' * indents}Chain")
            for item in self.items:
                sublines = repr(item).splitlines()
                for subline in sublines:
                    print(f"{'  ' * (indents + 1)}{subline}")

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
        from maelzel import colortheory
        itemcolor = colortheory.safeColors['blue2']
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
        ownitems = stackEventsRelative(self.items)
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
        # _stackEventsInPlace(items, defaultDur=defaultDur)
        return self.__class__(items, start=self.start)

    def removeRedundantTimings(self):
        """
        Modify self to remove over-secified start times in the items
        """
        start = self.resolvedStart()
        now = start
        for i, item in enumerate(self.items[1:], start=1):
            if item.dur is None:
                break
            if item.start and item.start == now:
                item.start = None
            now += item.dur

    def asVoice(self) -> Voice:
        """Convert this Chain to a Voice"""
        resolved = self.resolved(start=0)
        resolved.removeRedundantTimings()
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
        items = stackEventsRelative(self.items, offset=self.start)
        # groupid = scoring.makeGroupId(groupid)
        notations = sum((item.scoringEvents(groupid=groupid, config=config)
                         for item in items), [])

        if self.symbols:
            for s in self.symbols:
                for n in notations:
                    s.applyTo(n)
        return notations

    def scoringParts(self, options: scoring.render.RenderOptions = None
                     ) -> list[scoring.Part]:
        notations = self.scoringEvents(config=getConfig())
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
        items = stackEventsRelative(self.items, offset=timeoffset)
        return self.clone(items=items)

    def timeTransform(self, timemap: Callable[[Rat], Rat], inplace=False) -> Chain:
        start = self.resolvedStart()
        start2 = timemap(start)
        if inplace:
            stackEventsRelative(self.items, inplace=True)
            for item in self.items:
                item.start = timemap(item.start + start) - start2
                item.dur = timemap(item.end + start) - start2 - item.start
            self.start = start2
            return self
        else:
            items = stackEventsRelative(self.items, inplace=False)
            for item in items:
                item.start = timemap(item.start + start) - start2
                item.dur = timemap(item.end + start) - start2 - item.start
            return self.clone(items=items, start=start2)

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Chain:
        newitems = [item.pitchTransform(pitchmap) for item in self.items]
        return self.clone(items=newitems)


class Voice(Chain):
    def __init__(self,
                 items: list[Union[MusicEvent, str]] = None,
                 start: time_t = Rat(0),
                 label=''):
        super().__init__(items=items, label=label, start=start)




class _Voice(MusicObjList):
    """
    A Voice is a sequence of non-overlapping objects

    Voice vs Chain
    ~~~~~~~~~~~~~~

    * A Voice can contain a Chain, but not vice versa.
    * A Voice does not have a start offset, its start is always 0. Notes/Chords
      inside a Voice have an absolute start
    """
    _acceptsNoteAttachedSymbols = False

    def __init__(self,
                 items: list[Union[MusicEvent, str]] = None,
                 start: time_t = Rat(0),
                 label=''):
        if items:
            items = [item if isinstance(item, MusicObj) else asEvent(item)
                     for item in items]

        self.items: list[MusicEvent] = items
        super().__init__(label=label, dur=None, start=start)
        self._merged: Optional[list[MusicEvent]] = None

    def _getItems(self):
        return self.items

    def __repr__(self) -> str:
        parts = []
        if self.label:
            parts.append(self.label)
        maxitems = 12
        items = self.items[:maxitems]
        parts.append(str(items))
        s = ', '.join(parts)
        if len(self.items)>maxitems:
            s += ", …"
        return f"Voice({s})"

    def _changed(self) -> None:
        if self.items:
            self.dur = self.items[-1].end-self.items[0].start
            self.start = self.items[0].start
        super()._changed()

    def _mergedItems(self) -> list[MusicEvent]:
        if not self._merged:
            self._merged = _mergeItems(self.items)
        return self._merged

    def isEmptyBetween(self, start: time_t, end: time_t) -> bool:
        """
        Is this Voice empty between start and end?

        Args:
            start: start time to query
            end: end time to query

        Returns:
            True if this Voice is empty between start and end

        """
        if not self.items or start >= self.end or end < self.start:
            return True
        return all(mathlib.intersection(i.start, i.end, asRat(start), asRat(end)) is None
                   for i in self.items)

    def needsSplit(self) -> bool:
        """
        Does this Voice need to be split?

        Returns:
            False
        """
        return False

    def append(self, item: MusicEvent) -> None:
        self.items.append(item)
        self._changed()

    def pitchRange(self) -> Optional[tuple[float, float]]:
        pitchRanges = [item.pitchRange() for item in self.items]
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

    def extend(self, objs:list[MusicEvent]) -> None:
        self.items.extend(objs)
        self._changed()

    def scoringEvents(self, groupid:str=None, config: CoreConfig = None
                      ) -> list[scoring.Notation]:
        if config is None:
            config = getConfig()
        subgroup = scoring.makeGroupId(groupid)
        return misc.sumlist(item.scoringEvents(subgroup, config=config)
                            for item in _mergeItems(self.items))

    def scoringParts(self, options: scoring.render.RenderOptions = None
                     ) -> list[scoring.Part]:
        notations = self.scoringEvents(config=getConfig())
        scoring.stackNotationsInPlace(notations)
        part = scoring.Part(notations, label=self.label)
        return [part]

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        playargs.fillWith(self.playargs)
        return misc.sumlist(item._synthEvents(playargs.copy(), workspace)
                            for item in _mergeItems(self.items))


def _asVoice(obj: Union[MusicObj, list[MusicObj]]) -> Voice:
    if isinstance(obj, Voice):
        return obj
    elif isinstance(obj, Chain):
        return obj.asVoice()
    elif isinstance(obj, (Chord, Note)):
        return _asVoice([obj])
    elif isinstance(obj, (list, tuple)):
        return Chain(obj).asVoice()
    else:
        raise TypeError(f"Cannot convert {obj} of type {type(obj)} to a Voice")


class Score(MusicObjList):
    """
    A Score is a list of Voices
    """
    _acceptsNoteAttachedSymbols = False
    __slots__ = ('voices', '_scorestruct')

    def __init__(self,
                 voices: list,
                 label:str='',
                 scorestruct: ScoreStruct = None):
        asvoices = []
        for obj in voices:
            if isinstance(obj, Voice):
                asvoices.append(obj)
            elif isinstance(obj, Chain):
                asvoices.append(obj.asVoice())
            elif isinstance(obj, Score):
                asvoices.extend(obj.voices)
            else:
                raise TypeError(f"Cannot convert {obj} to a voice")
        starts = [v.resolvedStart() for v in asvoices]
        durs = [v.resolvedDur() for v in asvoices]
        ends = [start + dur for start, dur in zip(starts, durs)]
        self.voices: list[Voice] = asvoices
        super().__init__(label=label, start=0)
        self._scorestruct = scorestruct

    def __repr__(self):
        if not self.voices:
            info = ''
        else:
            # info = f'{len(self.voices)} voices'
            info = f'voices={self.voices}'
        return f'Score({info})'


    def _getItems(self) -> list[Voice]:
        return self.voices

    def _changed(self) -> None:
        starts = [v.resolvedStart() for v in self.voices]
        durs = [v.resolvedDur() for v in self.voices]
        ends = [start + dur for start, dur in zip(starts, durs)]
        end = max(ends)
        start = min(starts)
        self.start = start
        self.dur = end - start

    def scoringParts(self, options: scoring.render.RenderOptions = None
                     ) -> list[scoring.Part]:
        parts = []
        for voice in self.voices:
            voiceparts = voice.scoringParts(options=options)
            parts.extend(voiceparts)
        return parts

    def attachedScoreStruct(self) -> Optional[ScoreStruct]:
        if self._scorestruct is not None:
            return self._scorestruct
        structs = set(struct for v in self.voices
                      if (struct:=v.attachedScoreStruct()) is not None)
        if not structs:
            return None
        if len(structs) > 1:
            logger.warning("Multiple scorestructs are attached to voices within"
                         "this score. Returning the first one found")
        struct = next(iter(structs))
        assert isinstance(struct, ScoreStruct)
        return struct


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# notenames
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generateNotes(start=12, end=127) -> dict[str, Note]:
    """
    Generates all notes for interactive use.

    From an interactive session:

    .. code-block:: python

        >>> from maelzel.core import *
        >>> locals().update(generateNotes())
    """
    notes = {}
    for i in range(start, end):
        notename = pt.m2n(i)
        octave = notename[0]
        rest = notename[1:]
        rest = rest.replace('#', 'x')
        original_note = rest + str(octave)
        notes[original_note] = Note(i)
        if "x" in rest or "b" in rest:
            enharmonic_note = _util.enharmonic(rest)
            enharmonic_note += str(octave)
            notes[enharmonic_note] = Note(i)
    return notes


def splitNotes(notes: list[Note], splitpoints:list[float], deviation=None
               ) -> list[list[Note]]:
    """
    Split notes at given splitpoints.

    This can be used to split a group of notes into multiple staves. This assumes
    that notes are not synchronous (they do not overlap)

    Args:
        notes: the notes to split
        splitpoints: a list of splitpoints
        deviation: an acceptable deviation to fit all notes
            in one group (config: 'splitAcceptableDeviation')

    Returns:
        A list of list of notes, where each list contains notes either above,
        below or between splitpoints
    """
    splitpoints = sorted(splitpoints)
    tracks = []
    above = notes
    for splitpoint in splitpoints:
        above, below = _musicobjtools.splitNotesOnce(above, splitpoint=splitpoint,
                                                     deviation=deviation)
        if below:
            tracks.append(below)
        if not above:
            break
    return tracks

def asEvent(obj, **kws) -> Union[Note, Chord]:
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

    """
    if isinstance(obj, (Note, Chord, Line)):
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


def makeTrill(note1: Union[Note, Chord],
              note2: Union[Note, Chord],
              dur: time_t,
              notedur: time_t = Rat(1, 8)
              ) -> Chain:
    """
    Create a Chain of Notes representing a trill

    Args:
        note1: the first note of the trill (can also be a chord)
        note2: the second note of the trill (can also  be a chord)
        dur: total duration of the trill
        notedur: duration of each note. This value will only be used
            if the trill notes have an unset duration

    Returns:
        A realisation of the trill as a :class:`Chain` of at least the
        given *dur* (can be longer if *dur* is not a multiple
        of *notedur*)
    """
    note1 = asNote(note1)
    note2 = asNote(note2)
    note1 = note1.clone(dur=note1.dur or notedur or Rat(1, 8))
    note2 = note2.clone(dur=note2.dur or notedur or Rat(1, 8))
    seq = Chain([note1, note2])
    return seq.cycle(dur)


def packInVoices(objs: list[MusicObj]) -> list[Voice]:
    """
    Distribute these objects across multiple voices

    Ensures that objects within a voice do not overlap

    Args:
        objs: the objects (Notes, Chords, etc) to distribute

    Returns:
        a list of Voices
    """
    return _musicobjtools.packInVoices(objs)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if environment.insideJupyter:
    from . import jupytertools
    jupytertools.m21JupyterHook()


