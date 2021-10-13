"""
Module documentation

# Musical Objects

## Time

A MusicObj has always a start and dur attribute. They refer to an abstract time.
When visualizing a MusicObj as musical notation these times are interpreted/converted
to beats and score locations based on a score structure.

### Score Structure

A minimal score structure is a default time-signature (4/4) and a default tempo (60). If
the user does not set a different score structure, an endless score with these default
values will always be used.

"""

from __future__ import annotations

from math import sqrt

import music21 as m21

from emlib import misc
from emlib.misc import firstval
from emlib.mathlib import intersection
from emlib import iterlib

import pitchtools as pt
from maelzel import scoring

from ._common import Rat, asRat, UNSET, MAXDUR, logger
from .musicobjbase import *
from .workspace import activeConfig
from . import play
from . import tools
from . import environment
from . import notation
from .pitch import Pitch
from .csoundevent import PlayArgs, CsoundEvent
from . import _musicobjtools
from maelzel.scorestruct import ScoreStruct
import functools
import csoundengine

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from ._typedefs import *


@functools.total_ordering
class Note(MusicObj):
    """
    In its simple form, a Note is used to represent a Pitch.

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
    set via the `.setplay` method. Any aspects regarding notation
    (articulation, enharmonic variant, etc) can be set via `.setSymbol`

    Args:
        pitch: a midinote or a note as a string. A pitch can be
            a midinote or a notename as a string.
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

    Attributes:
        pitch: the pitch of this Note, as midinote
        dur: the duration of the Note (a Rat), or None
        amp: the amplitude (0-1), or None
        start: the start time (as Rat), or None
        gliss: the end pitch (as midinote), or None
        label: a string label
    """

    __slots__ = ('pitch', 'amp', '_gliss', 'tied')

    def __init__(self,
                 pitch: pitch_t,
                 dur: time_t = None,
                 amp: float = None,
                 start: time_t = None,
                 gliss: Union[pitch_t, bool] = False,
                 label: str = '',
                 dynamic: str = None,
                 tied = False
                 ):

        MusicObj.__init__(self, dur=dur, start=start, label=label)
        self.pitch: float = tools.asmidi(pitch)
        self.amp: Optional[float] = amp
        self._gliss: Union[float, bool] = gliss if isinstance(gliss, bool) else tools.asmidi(gliss)
        self.tied = tied
        if dynamic:
            self.setSymbol('Dynamic', dynamic)

    @property
    def gliss(self):
        return self._gliss

    @gliss.setter
    def gliss(self, gliss: Union[pitch_t, bool]):
        """
        Set the gliss attribute of this Note, in place
        """
        self._gliss = gliss if isinstance(gliss, bool) else tools.asmidi(gliss)

    def clone(self,
              pitch: pitch_t = UNSET,
              dur: Optional[time_t] = UNSET,
              amp: Optional[time_t] = UNSET,
              start: Optional[time_t] = UNSET,
              gliss: Union[pitch_t, bool] = UNSET,
              label: str = UNSET,
              tied: bool = UNSET) -> Note:
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
                   tied=tied if tied is not UNSET else self.tied)
        if self._symbols:
            out._symbols = self._symbols.copy()
        if self._playargs:
            out._playargs =self._playargs.copy()
        return out

    def __hash__(self) -> int:
        hashsymbols = hash(tuple(self._symbols)) if self._symbols else 0
        return hash((self.pitch, self.dur, self.start, self.gliss, self.label,
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
        if self._symbols:
            self._symbols.clear()

    def pitchRange(self) -> Optional[Tuple[float, float]]:
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
        else:
            raise NotImplementedError()

    @property
    def freq(self) -> float:
        return pt.m2f(self.pitch)

    @freq.setter
    def freq(self, value:float) -> None:
        self.pitch = pt.f2m(value)

    @property
    def name(self) -> str:
        return pt.m2n(self.pitch)

    @property
    def pitchclass(self) -> int:
        return round(self.pitch) % 12


    @property
    def cents(self) -> int:
        return tools.midicents(self.pitch)

    @property
    def centsrepr(self) -> str:
        return tools.centsshown(self.cents,
                                divsPerSemitone=activeConfig()['semitoneDivisions'])

    def overtone(self, n:float) -> Note:
        """
        Return a new Note representing the `nth` overtone of this Note

        Args:
            n: the overtone number (1 = fundamental)

        Returns:
            a new Note
        """
        return Note(pt.f2m(self.freq * n))

    def scoringEvents(self, groupid:str=None) -> List[scoring.Notation]:
        config = activeConfig()
        dur = self.dur or config['defaultDuration']
        assert dur is not None
        if self.isRest():
            rest = scoring.makeRest(self.dur, offset=self.start)
            annot = self._scoringAnnotation()
            if annot:
                rest.addAnnotation(annot)
            return [rest]

        note = scoring.makeNote(pitch=self.pitch,
                                duration=asRat(dur),
                                offset=self.start,
                                gliss=bool(self.gliss),
                                playbackGain=self.amp,
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
        if self._symbols:
            tools.applySymbols(self._symbols, notes)
        return notes

    def _asTableRow(self) -> List[str]:
        if self.isRest():
            elements = ["REST"]
        else:
            elements = [pt.m2n(self.pitch)]
            config = activeConfig()
            if config['repr.showFreq']:
                elements.append("%dHz" % int(self.freq))
            if self.amp is not None and self.amp < 1:
                elements.append("%ddB" % round(pt.amp2db(self.amp)))
        if self.dur:
            if self.dur >= MAXDUR:
                elements.append("dur=inf")
            else:
                elements.append(f"{tools.showTime(self.dur)}♩")
        if self.start is not None:
            elements.append(f"start={tools.showTime(self.start)}")
        if self.gliss:
            if isinstance(self.gliss, bool):
                elements.append(f"gliss={self.gliss}")
            else:
                elements.append(f"gliss={pt.m2n(self.gliss)}")
        return elements

    def __repr__(self) -> str:
        elements = self._asTableRow()
        if len(elements) == 1:
            return elements[0]
        else:
            s = ":".join(elements)
            return s

    def __str__(self) -> str: return self.name

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
            step = 1/activeConfig()['semitoneDivisions']
        return self.clone(pitch=round(self.pitch / step) * step)

    def csoundEvents(self, playargs: PlayArgs, scorestruct:ScoreStruct, conf:dict
                     ) -> List[CsoundEvent]:
        playargs.fillWith(self.playargs)
        playargs.fillWithConfig(conf)
        amp = 1.0 if self.amp is None else self.amp
        endmidi = self.gliss if self.gliss > 1 else self.pitch
        start = self.start or 0.
        dur = self.dur or 1.0
        starttime = float(scorestruct.beatToTime(start))
        endtime   = float(scorestruct.beatToTime(start + dur))
        bps = [[starttime, self.pitch, amp],
               [endtime,   endmidi,    amp]]
        return [CsoundEvent.fromPlayArgs(bps=bps, playargs=playargs, tiednext=self.tied)]

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
            endpitch = tools.asmidi(endpitch)
        dur = dur or self.resolvedDuration()
        startamp = self.resolvedAmp()
        endamp = firstval(endamp, self.amp, startamp)
        start = self.start or 0
        breakpoints = [(start, self.pitch, startamp),
                       (start+dur, endpitch, endamp)]
        return Line(breakpoints)

    def resolvedAmp(self) -> float:
        """
        Get the amplitude of this object, or a default amplitude

        Returns a default amplitude if no amplitude was define (self.amp is None).
        The default amplitude can be customized via
        ``getConfig()['play.defaultAmplitude']``

        Returns:
            the amplitude (a value between 0-1, where 0 corresponds to 0dB)
        """
        return self.amp if self.amp is not None else \
            activeConfig()['play.defaultAmplitude']

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Note:
        pitch = pitchmap(self.pitch)
        gliss = self.gliss if isinstance(self.gliss, bool) else pitchmap(self.gliss)
        return self.clone(pitch=pitch, gliss=gliss)


def Rest(dur:time_t=1, start:time_t=None) -> Note:
    """
    Creates a Rest. A Rest is a Note with pitch 0 and amp 0.

    To test if an item is a rest, call isRest

    Args:
        dur: duration of the Rest
        start: start of the Rest

    Returns:
        the creaed rest
    """
    assert dur is not None and dur > 0
    return Note(pitch=0, dur=dur, start=start, amp=0)


def asNote(n: Union[float, str, Note, Pitch],
           amp:float=None, dur:time_t=None, start:time_t=None) -> Note:
    """
    Convert ``n`` to a Note

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
    elif isinstance(n, Pitch):
        return Note(pitch=n.midi)
    raise ValueError(f"cannot express this as a Note: {n} ({type(n)})")


class Line(MusicObj):
    """ 
    A Line is a seq. of breakpoints, where each bp is of the form
    (delay, pitch, [amp=1, ...]), where:

    - delay is the time offset to the first breakpoint.
    - pitch is the pitch as midinote or notename
    - amp is the amplitude (0-1), optional

    pitch, amp and any other following data can be 'carried'::

        Line((0, "D4"), (1, "D5", 0.5), ..., fade=0.5)

    Also possible::

    >>> bps = [(0, "D4"), (1, "D5"), ...]
    >>> Line(bps)   # without *

    a Line stores its breakpoints as: ``[delayFromFirstBreakpoint, pitch, amp, ...]``
    """

    __slots__ = ('bps',)
    _acceptsNoteAttachedSymbols = True

    def __init__(self, *bps, label="", delay:num_t=0, relative=False):
        """
        Args:
            bps: breakpoints, a tuple/list of the form (delay, pitch, [amp=1, ...]), where
                delay is the time offset to the beginning of the line
                pitch is the pitch as notename or midinote
                amp is an amplitude between 0-1
            delay: time offset of the line itself
            label: a label to add to the line
            relative: if True, the first value of each breakpoint is a time offset
                from previous breakpoint
        """
        # [[[0, 60, 1], [1, 60, 2]]]
        if len(bps) == 1 and isinstance(bps[0], list) and isinstance(bps[0][0], (tuple, list)):
            bps = bps[0]
        bps = tools.carryColumns(bps)
        
        if len(bps[0]) < 2:
            raise ValueError("A breakpoint should be at least (delay, pitch)", bps)

        bps = tools.as2dlist(bps)
        if len(bps[0]) < 3:
            for bp in bps:
                if len(bp) == 2:
                    bp.append(1.)

        for bp in bps:
            bp[1] = tools.asmidi(bp[1])

        if relative:
            now = 0
            for bp in bps:
                now += bp[0]
                bp[0] = now

        if bps[0][0] > 0:
            dt = bps[0][0]
            delay += dt
            for row in bps:
                row[0] -= dt

        for bp in bps:
            assert all(isinstance(x, (float, int, Rat)) for x in bp), f"bp={bp}"
        assert all(bp1[0]>bp0[0] for bp0, bp1 in iterlib.pairwise(bps))

        super().__init__(dur=bps[-1][0], start=delay, label=label)
        self.bps: List[List[float]] = bps
        
    def getOffsets(self) -> List[num_t]:
        """ Return absolute offsets of each breakpoint """
        start = self.start
        return [bp[0] + start for bp in self.bps]

    def translateBps(self, score:ScoreStruct) -> List[Tuple[num_t, ...]]:
        """
        Translate beat to absolute time within the breakpoints of this Line

        Args:
            score: the scorestructure to use to translate quarter notes to
                abs time

        Returns:
            a copy of this Lines breakpoints where all timing is given in
            absolute time
        """
        bps = []
        for bp in self.bps:
            t = bp[0] + self.start
            t = float(score.beatToTime(t))
            bp2 = [t]
            bp2 += bp[1:]
            bps.append(bp2)
        return bps

    def csoundEvents(self, playargs: PlayArgs, score:ScoreStruct, conf: dict
                     ) -> List[CsoundEvent]:
        playargs.fillWith(self.playargs)
        playargs.fillWithConfig(conf)
        bps = self.translateBps(score)
        return [CsoundEvent.fromPlayArgs(bps, playargs=playargs)]

    def __hash__(self):
        rowhashes = [hash(tuple(bp)) for bp in self.bps]
        rowhashes.append(self.start)
        return hash(tuple(rowhashes))

    def __repr__(self):
        return f"Line(start={self.start}, bps={self.bps})"

    def quantizePitch(self, step=0) -> Line:
        """ Returns a new object, rounded to step """
        if step == 0:
            step = 1/activeConfig()['semitoneDivisions']
        bps = [ (bp[0], tools.quantizeMidi(bp[1], step)) + bp[2:]
                for bp in self.bps ]
        if len(bps) >= 3:
            bps = misc.simplify_breakpoints(bps, coordsfunc=lambda bp:(bp[0], bp[1]),
                                            tolerance=0.01)
        return Line(bps)

    def scoringEvents(self, groupid:str=None) -> List[scoring.Notation]:
        offsets = self.getOffsets()
        groupid = scoring.makeGroupId(groupid)
        notations: List[scoring.Notation] = []
        for bp0, bp1 in iterlib.pairwise(self.bps):
            pitch = bp0[1]
            assert pitch > 0
            dur = bp1[0] - bp0[0]
            offset = bp0[0] + self.start
            assert dur > 0
            ev = scoring.makeNote(pitch=pitch, offset=offset, duration=dur,
                                  gliss=bp0[1] != bp1[1], group=groupid)
            if bp0[1] == bp1[1]:
                ev.tiedNext = True
            notations.append(ev)
        if(self.bps[-1][1] != self.bps[-2][1]):
            # add a last note if last pair needed a gliss (to have a destination note)
            n = notations[-1]
            n.gliss = True
            notations.append(scoring.makeNote(pitch=self.bps[-1][1],
                                              offset=offsets[-1],
                                              group=groupid,
                                              duration=0))
        if notations:
            scoring.fixOverlap(notations)
            annot = self._scoringAnnotation()
            if annot:
                notations[0].addAnnotation(annot)
        if self._symbols:
            tools.applySymbols(self._symbols, notations)
        for n0, n1 in iterlib.window(notations, 2):
            if n0.tiedNext:
                n1.tiedNext = True
        return notations

    def timeTransform(self, timemap: Callable[[float], float]) -> Line:
        bps = []
        for bp in self.bps:
            t1 = timemap(bp[0] + self.start)
            bp2 = bp.copy()
            bp2[0] = t1
            bps.append(bp2)
        return Line(bps, label=self.label)

    def dump(self, indents=0):
        elems = []
        if self.start:
            elems.append(f"start={self.start}")
        if self.label:
            elems.append(f"label={self.label}")
        infostr = ", ".join(elems)
        print("Line:", infostr)
        rows = []
        for bp in self.bps:
            row = ["%.6g" % _ for _ in bp]
            rows.append(row)
        headers = ("start", "pitch", "amp", "p4", "p5", "p6", "p7", "p8")
        misc.print_table(rows, headers=headers, showindex=False, )

    def timeShift(self, timeoffset: time_t) -> Line:
        out = self.copy()
        out.start = self.start + timeoffset
        return out

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Line:
        newpitches = [pitchmap(bp[1]) for bp in self.bps]
        newbps = self.bps.copy()
        for bp, pitch in zip(newbps, newpitches):
            bp[1] = pitch
        return self.clone(bps=newbps)


class Chord(MusicObj):

    __slots__ = ('amp', 'gliss', 'notes', 'tied')

    def __init__(self, *notes,
                 dur: time_t = None,
                 amp: float = None,
                 start: time_t = None,
                 gliss: Union[str, List[pitch_t], Tuple[pitch_t], bool] = False,
                 label: str='',
                 tied = False,
                 dynamic: str = None
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
        self.notes: List[Note] = []
        if not isinstance(gliss, bool):
            gliss = pt.as_midinotes(gliss)

        self.gliss: Union[bool, List[float]] = gliss

        if not notes:
            super().__init__(dur=None, start=None, label=label)
            return

        # notes might be: Chord([n1, n2, ...]) or Chord(n1, n2, ...)
        if misc.isgeneratorlike(notes):
            notes = list(notes)
        n0 = notes[0]
        if len(notes) == 1:
            if isinstance(n0, (Chord, Chain)):
                notes = list(n0)
            elif isinstance(n0, (list, tuple)):
                notes = notes[0]
            elif isinstance(n0, str):
                notes = n0.split()
                notes = [Note(n) for n in notes]
        # determine dur & start
        if dur is None:
            dur = max((n.dur for n in notes if isinstance(n, Note) and n.dur is not None),
                      default=None)
        if start is None:
            start = min((n.start for n in notes if isinstance(n, Note) and n.start is not None),
                        default=None)
        super().__init__(dur=dur, start=start, label=label)

        for note in notes:
            if isinstance(note, Note):
                # we erase any duration or offset of the individual notes
                note = note.clone(dur=None, start=None)
                if self.gliss:
                    note.gliss = False
            else:
                assert isinstance(note, (int, float, str))
                note = asNote(note, amp=amp, dur=None, start=None)
            self.notes.append(note)
        self.notes = list(set(self.notes))
        if self.gliss:
            assert all(not n.gliss for n in self.notes)

        self.sort()
        self.tied = tied
        if dynamic:
            self.setSymbol('Dynamic', dynamic)

    def __len__(self) -> int:
        return len(self.notes)

    def __getitem__(self, idx) -> Union[Note, Chord]:
        out = self.notes.__getitem__(idx)
        if isinstance(out, list):
            out = self.__class__(out)
        return out

    def __iter__(self) -> Iterable[Note]:
        return iter(self.notes)

    def pitchRange(self) -> Optional[Tuple[float, float]]:
        return min(n.pitch for n in self.notes), max(n.pitch for n in self.notes)

    def scoringEvents(self, groupid:str = None) -> List[scoring.Notation]:
        config = activeConfig()
        pitches = [note.pitch for note in self.notes]
        annot = self._scoringAnnotation()
        dur = self.dur if self.dur is not None else config['defaultDuration']
        chord = scoring.makeChord(pitches=pitches, duration=dur, offset=self.start,
                                  annotation=annot, group=groupid)
        #if config['show.fillDynamicFromAmplitude'] and self.amp is not None and self.getSymbol('dynamic') is None:
        #    chord.dynamic = tools.amplitudeToDynamic()
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
        if self._symbols:
            tools.applySymbols(self._symbols, notations)
        return notations

    def asmusic21(self, **kws) -> m21.stream.Stream:
        config = activeConfig()
        arpeggio = _normalizeChordArpeggio(kws.get('arpeggio', None), self)
        if arpeggio:
            dur = config['show.arpeggioDuration']
            notes = [n.clone(dur=dur) for n in self.notes]
            return Chain(notes).asmusic21()
        events = self.scoringEvents()
        scoring.stackNotationsInPlace(events, start=self.start)
        parts = scoring.distributeNotationsByClef(events)
        return notation.renderWithCurrentWorkspace(parts).asMusic21()

    def __hash__(self):
        if self._hash:
            return self._hash
        if isinstance(self.gliss, bool):
            glisshash = int(self.gliss)
        elif isinstance(self.gliss, list):
            glisshash = hash(tuple(self.gliss))
        else:
            glisshash = hash(self.gliss)
        data = (self.dur, self.start, self.label, glisshash, *(n.pitch for n in self.notes))
        self._hash = hash(data)
        return self._hash

    def append(self, note:Union[float, str, Note, Pitch]) -> None:
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
        step = tools.asmidi(fundamental) - self[0].pitch
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
            step = 1/activeConfig()['semitoneDivisions']
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

    def splitByAmp(self, numChords=8, maxNotesPerChord=16) -> List[Chord]:
        """
        Split this chord into several chords, according to note amplitude

        Args:
            numChords: the number of chords to split this chord into
            maxNotesPerChord: max. number of notes per chord

        Returns:
            a list of Chords
        """
        midis = [note.pitch for note in self.notes]
        amps = [note.amp for note in self.notes]
        chords = tools.splitByAmp(midis, amps, numGroups=numChords,
                                  maxNotesPerGroup=maxNotesPerChord)
        return [Chord(chord) for chord in chords]

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
        playargs.fillWithConfig(config or activeConfig())
        return playargs

    @property
    def pitches(self) -> List[float]:
        return [n.pitch for n in self.notes]

    def csoundEvents(self, playargs: PlayArgs, scorestruct:ScoreStruct, config:dict
                     ) -> List[CsoundEvent]:
        playargs.fillWith(self.playargs)
        playargs.fillWithConfig(config)

        gain = firstval(playargs.gain, 1.0)
        if config['chord.adjustGain']:
            gain *= 1 / sqrt(len(self))
        playargs.gain = gain
        endpitches = self.gliss if isinstance(self.gliss, list) else self.pitches
        events = []
        dur = self.resolvedDuration()
        start = self.start or 0
        starttime = float(scorestruct.beatToTime(start))
        endtime = float(scorestruct.beatToTime(start + dur))
        for note, endpitch in zip(self.notes, endpitches):
            amp = firstval(note.amp, self.amp, 1.)
            bps = [[starttime, note.pitch, amp],
                   [endtime, endpitch, amp]]
            events.append(CsoundEvent.fromPlayArgs(bps=bps, playargs=playargs))
        return events

    def asChain(self) -> Chain:
        """ Convert this Chord to an Chain """
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
                elements.append("gliss=True")
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

    def setAmplitude(self, amp: float) -> None:
        """
        Set the amplitudes of the notes in this chord to `amp` (in place)
        """
        return self.scaleAmplitudes(factor=0, offset=amp)

    def scaleAmplitudes(self, factor:float, offset=0.0) -> None:
        """
        Scale the amplitudes of the notes within this chord **in place**
        """
        for n in self.notes:
            n.amp = n.amp * factor + offset

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
        out = Chord(asNote(obj))
    else:
        raise ValueError(f"cannot express this as a Chord: {obj}")
    if amp is not None or dur is not None:
        out = out.clone(amp=amp, dur=dur)
    return out


def asEvent(obj, **kws) -> Union[Note, Chord]:
    """Convert `obj` to an event (a Note or a Chord, depending on obj)"""
    if isinstance(obj, (Note, Chord)):
        return obj
    out = asMusic(obj, **kws)
    assert isinstance(out, (Note, Chord))
    return out


def _normalizeChordArpeggio(arpeggio: Union[str, bool], chord: Chord) -> bool:
    config = activeConfig()
    if arpeggio is None: arpeggio = config['chord.arpeggio']

    if isinstance(arpeggio, bool):
        return arpeggio
    elif arpeggio == 'auto':
        return chord._isTooCrowded()
    else:
        raise ValueError(f"arpeggio should be True, False, 'auto' (got {arpeggio})")


def stackEvents(events: List[MusicObj],
                defaultDur:time_t=None,
                start:time_t=Rat(0)
                ) -> List[MusicObj]:
    """
    Place `events` one after the other`

    Args:
        events: the events to stack against each other
        defaultDur: the duration given to events which don't have an explicit duration
        start: the start time for the event stack (will be used if the first event
            doesn't have an explicit start)

    Returns:
        the resulting events. It is ensured that in the returned events there is no
        intersection between the events and all have start and dur set

    """
    if not events:
        return events
    if all(ev.start is not None and ev.dur is not None for ev in events):
        return events
    if start is None:
        start = events[0].start
        if start is None:
            start = Rat(0)
    assert start is not None
    if defaultDur is None:
        defaultDur = asRat(activeConfig()['defaultDuration'])
    assert defaultDur is not None
    now = events[0].start if events[0].start is not None else start
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
                startNext = events[i + 1].start
                if startNext is not None:
                    dur = startNext - start
                else:
                    dur = defaultDur
            else:
                dur = defaultDur
            ev = ev.clone(start=start, dur=dur)
        assert ev.dur > 0
        now = ev.end
        out.append(ev)
    for ev1, ev2 in iterlib.pairwise(out):
        assert all(isinstance(x, Rat) for x in (ev1.start, ev1.end, ev2.start, ev2.end))
        assert ev1.end <= ev2.start, \
            f"ev1: {ev1} (end={ev1.end}, {type(ev1.end)}), ev2: {ev2} (start={ev2.start})"
    return out


def stackEventsInPlace(events: Sequence[MusicObj],
                       defaultDur:time_t=None,
                       start:time_t=Rat(0)
                       ) -> None:
    """
    Similar to stackEvents, but modifies the events themselves

    Args:
        events: the events to stack against each other
        defaultDur: the duration given to events which don't have an explicit duration
        start: the start time for the event stack (will be used if the first event
            doesn't have an explicit start)
    """
    if all(ev.start is not None and ev.dur is not None for ev in events):
        return
    for ev in events:
        if isinstance(ev, MusicObjList):
            stackEventsInPlace(ev)
    assert start is not None
    if defaultDur is None:
        defaultDur = activeConfig()['defaultDuration']
        assert defaultDur is not None
    now = events[0].start if events[0].start is not None else start
    assert now is not None and now >= 0
    if len(events) == 1:
        ev = events[0]
        if ev.start is None:
            ev.start = start
        if ev.dur is None:
            ev.dur = defaultDur
        return
    lasti = len(events) - 1
    for i, ev in enumerate(events):
        if ev.dur is None:
            if i < lasti:
                startNext = events[i + 1].start
                assert startNext is not None
                dur = startNext - now
            else:
                dur = defaultDur
            ev.start = now
            ev.dur = dur
        elif ev.start is None:
            ev.start = now
        assert ev.dur is not None and ev.start is not None
        now = ev.end
    for ev1, ev2 in iterlib.pairwise(events):
        assert ev1.start <= ev2.start, f"{ev1=}, {ev2=}"
        assert ev1.end <= ev2.start, f"{ev1=}, {ev2=}"


class MusicObjList(MusicObj):
    """
    A sequence of music objects (Chain, Group).

    They do not need to be sequencial (they can overlap, like Group)
    """
    __slots__ = ('items')

    def __init__(self, items: List[MusicObj], label:str=''):
        self.items: List[MusicObj] = []
        if items:
            assert all(item.dur is not None and item.start is not None
                       for item in items)
            self.items.extend(items)
            end = max(it.end for it in items)
            start = min(it.start for it in items)
            dur = end - start
        else:
            dur = None
            start = None
        super().__init__(dur=dur, start=start, label=label)

    def append(self, obj: MusicObj) -> None:
        self.items.append(obj)
        self._changed()

    def _changed(self) -> None:
        self.start = min(it.start for it in self.items)
        end = max(it.end for it in self.items)
        self.dur = end - self.start
        super()._changed()

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items.__getitem__(idx)

    def __len__(self):
        return len(self.items)

    def __hash__(self):
        hashes = [hash(type(self).__name__), hash(self.label)]
        hashes.extend(hash(i) for i in self.items)
        return hash(tuple(hashes))

    def pitchRange(self) -> Optional[Tuple[float, float]]:
        pitchRanges = [item.pitchRange() for item in self.items]
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

    def scoringEvents(self, groupid:str=None) -> List[scoring.Notation]:
        """
        Returns the scoring events corresponding to this object

        Args:
            groupid: if given, all events are given this groupid

        Returns:
            the scoring notations
        """
        groupid = scoring.makeGroupId(groupid)
        return sum((i.scoringEvents(groupid=groupid) for i in self._mergedItems()), [])

    def _mergedItems(self) -> List[MusicObj]:
        return self.items

    def csoundEvents(self, playargs: PlayArgs, scorestruct: ScoreStruct, conf:dict
                     ) -> List[CsoundEvent]:
        playargs.fillWith(self.playargs)
        return misc.sumlist(item.csoundEvents(playargs.copy(), scorestruct, conf)
                            for item in self._mergedItems())

    def quantizePitch(self:T, step=0.) -> T:
        if step == 0:
            step = 1/activeConfig()['semitoneDivisions']
        items = [i.quantizePitch(step) for i in self.items]
        return self.clone(items=items)

    def timeShift(self:T, timeoffset: time_t) -> T:
        items = [item.timeShift(timeoffset) for item in self.items]
        return self.clone(items=items)

    def timeTransform(self:T, timemap: Callable[[float], float]) -> T:
        items = [item.timeTransform(timemap) for item in self.items]
        return self.clone(items=items)

    def pitchTransform(self: T, pitchmap: Callable[[float], float]) -> T:
        newitems = [item.pitchTransform(pitchmap) for item in self.items]
        return self.clone(items=newitems)

    def dump(self, indents=0):
        print(f'{"  "*indents}{repr(self)}')
        if self._playargs:
            print("  "*(indents+1), self.playargs)
        for item in self.items:
            item.dump(indents+1)

    def makeVoices(self) -> List[Voice]:
        return _musicobjtools.packInVoices(self.items)


def _fixGliss(items: List[MusicObj]) -> None:

    for i0, i1 in iterlib.window(items, 2):
        if isinstance(i0, MusicObjList):
            _fixGliss(i0)
        elif (isinstance(i0, Note) and isinstance(i1, Note) and
                not i1.isRest() and i0.gliss is True):
            i0.gliss = i1.pitch

def _makeLine(notes: List[Note]) -> Line:
    assert all(n0.end == n1.start for n0, n1 in iterlib.pairwise(notes))
    bps = []
    for note in notes:
        bp = [note.start, note.pitch, note.resolvedAmp()]
        bps.append(bp)
    lastnote = notes[-1]
    if lastnote.dur > 0:
        pitch = lastnote.gliss if lastnote.gliss else lastnote.pitch
        bps.append([lastnote.end, pitch, lastnote.resolvedAmp()])
    return Line(bps, label=notes[0].label)


def _mergeLines(items: List[Union[Note, Chord]]) -> List[Union[Note, Chord, Line]]:
    """
    Merge notes/chords with ties/gliss into Lines, which are better capable of
    rendering notation and playback for those cases.

    Notes and Chords are appended as-is, notes which can be merged as Lines are
    fused together and the resulting Line is added
    """
    groups = []
    lineStarted = False
    gapAfter = set()
    for i0, i1 in iterlib.window(items, 2):
        if i0.end < i1.start:
            gapAfter.add(i0)
    for item in items:
        if isinstance(item, Note) and not item.isRest():
            if lineStarted:
                groups[-1].append(item)
                if not item.tied and not (item.gliss is True):
                    lineStarted = False
            else:
                if item.tied or item.gliss is True:
                    lineStarted = True
                    groups.append([item])
                else:
                    groups.append(item)
        else:
            lineStarted = False
            groups.append(item)
    return [_makeLine(item) if isinstance(item, list) else item
            for item in groups]


class Chain(MusicObjList):
    """
    A seq. of non-simultaneous Notes / Chords

    A Chain is used to express a series of notes or chords which come
    one after the other. All notes and chors within a Chain have an
    explicit start and duration, but there can be gaps between
    items. Such gaps are either possible by setting the explicit start
    time of an event later than the end time of the previous event, or
    by inserting a Rest.

    Within a Chain, any note/chord with a glissando set to True will
    result in a glissando to the next note/chord in the chain.
    """
    _acceptsNoteAttachedSymbols = False

    def __init__(self, items: List[Union[Note, Chord]] = None, start:time_t=None,
                 label:str=''):
        if start is not None:
            start = asRat(start)
        if items:
            items = [asEvent(item) for item in items]
            items = stackEvents(items, start=start)
        super().__init__(items=items, label=label)
        self._merged = None

    def _mergedItems(self) -> List[Union[Note, Chord, Line]]:
        if not self._merged:
            self._merged = _mergeLines(self.items)
        return self._merged

    def resolvedDuration(self) -> Rat:
        if not self.items:
            return Rat(0)
        end = self.items[-1].end
        assert end is not None
        start = self.items[0].start
        assert start is not None
        return end - start

    def append(self, item:Union[Note, Chord]) -> None:
        """
        Append an item to this chain

        Args:
            item: the item to add
        """
        if item.start is None or item.dur is None:
            start = item.start if item.start is not None else (self.end or Rat(0))
            item = item.withExplicitTime(start=start)
        assert item.start is not None and item.dur is not None and item.end is not None
        assert self.end is not None
        assert item.start >= self.end
        self.items.append(item)
        if len(self.items) > 1:
            butlast = self.items[-2]
            last = self.items[-1]
            if isinstance(butlast, Note) and butlast.gliss is True and isinstance(last, Note):
                butlast.gliss = last.pitch
        self._changed()

    def _changed(self):
        if self.items:
            self.start = self.items[0].start
            end = self.items[-1].end
            self.dur = end - self.start
        else:
            self.start = None
            self.dur = None
        self._hash = None
        self._merged = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterable[Union[Note, Chord]]:
        return iter(self.items)

    def __getitem__(self, idx):
        out = self.items.__getitem__(idx)
        if isinstance(out, MusicObj):
            return out
        elif isinstance(out, list):
            return self.__class__(out)
        else:
            raise ValueError(f"__getitem__ returned {out}, expected Chord or list of Chords")

    def dump(self, indents=0):
        if environment.insideJupyter:
            rows = []
            for item in self.items:
                row = [round(float(item.start), 3)]
                if isinstance(item, Note):
                    row.append(item.name)
                else:
                    row.append(", ".join(pt.m2n(n) for n in item))
                row.append(item.dur)
                row.append(item.gliss)
                row.append(str(self._playargs))
                rows.append(row)
            misc.print_table(rows, headers='time pitch dur gliss playargs'.split())
        else:
            print(f"{' '*indents}Chain")
            for item in self.items:
                sublines = repr(item).splitlines()
                for subline in sublines:
                    print(f"{'  '*(indents+1)}{subline}")

    def __repr__(self):
        if len(self.items) < 10:
            itemstr = ", ".join(repr(_) for _ in self.items)
        else:
            itemstr = ", ".join(repr(_) for _ in self.items[:10]) + ", …"
        return f'Chain([{itemstr}])'

    def __hash__(self):
        if not self._hash:
            self._hash = hash(tuple(hash(i) ^ 0x1234 for i in self.items))
        return self._hash

    def cycle(self, dur:time_t, crop=True) -> Chain:
        """
        Cycle the items in this seq. until the given duration is reached

        Args:
            dur: the total duration
            crop: if True, the last event will be cropped to fit
                the given total duration. Otherwise, it will last
                its given duration, even if that would result in
                a total duration longer than the given one

        Returns:
            the resulting Chain
        """
        cfg = activeConfig()
        defaultDur = Rat(cfg['defaultDuration'])
        accumDur = Rat(0)
        maxDur = asRat(dur)
        items = []
        for item in iterlib.cycle(self):
            if item.dur is None:
                item = item.clone(dur=defaultDur)
            if item.dur > maxDur - accumDur:
                if not crop:
                    break
                item = item.clone(dur=maxDur - accumDur)
            if item.start is not None:
                item = item.clone(start=None)
            items.append(item)
            accumDur += item.dur
            if accumDur == maxDur:
                break
        stackEventsInPlace(items, defaultDur=defaultDur)
        return Chain(items, start=self.start)

    def asVoice(self) -> Voice:
        """Convert this Chain to a Voice"""
        return Voice(self.items, label=self.label)

    def makeVoices(self) -> List[Voice]:
        return [self.asVoice()]


class Voice(MusicObjList):
    """
    A Voice is a seq. of non-overlapping objects

    A Voice can contain an Chain, but not vice versa
    """
    _acceptsNoteAttachedSymbols = False

    def __init__(self, items:List[MusicObj]=None, label:str= ''):
        self.instrs: Dict[MusicObj, str] = {}
        if items:
            items = items.copy()
            stackEventsInPlace(items, start=asRat(0))
        self._merged: Optional[List[Note, Chord, Line]] = None
        super().__init__(items=items, label=label)

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

    def _mergedItems(self) -> List[Union[Note, Chord, Line]]:
        if not self._merged:
            self._merged = _mergeLines(self.items)
        return self._merged

    def isEmptyBetween(self, start:time_t, end:num_t) -> bool:
        if not self.items or start >= self.end or end < self.start:
            return True
        return all(intersection(i.start, i.end, start, end) is None
                    for i in self.items)

    def needsSplit(self) -> bool:
        return False

    def add(self, obj:MusicObj) -> None:
        """
        Add this object to this Voice.

        If obj has already a given start, it will be inserted at that offset, otherwise
        it will be appended to the end of this Voice.

        1) To insert an untimed object (for example, a Note with start=None) to the Voice
           at a given offset, set its .start attribute or do voice.add(chord.clone(start=...))

        2) To append a timed object at the end of this voice (overriding the start
           time of the object), do voice.add(obj.clone(start=voice.endTime()))

        Args:
            obj: the object to add (a Note, Chord, Event, etc.)
        """
        if obj.start is None or obj.dur is None:
            obj = obj.withExplicitTime(start=self.end)
        assert obj.start is not None and obj.end is not None
        if not self.isEmptyBetween(obj.start, obj.end):
            msg = f"obj {obj} ({obj.start}:{obj.end}) does not fit in voice"
            raise ValueError(msg)
        assert obj.dur is not None and obj.dur>0
        if obj.start is None:
            obj = obj.clone(start=self.end)
        self.items.append(obj)
        if obj.start < self.end:
            self.items.sort(key=lambda obj:obj.start)
        self._changed()

    def extend(self, objs:List[MusicObj]) -> None:
        objs.sort(key=lambda obj:obj.start or 0)
        start = objs[0].start
        assert start is not None and start >= self.end
        for obj in objs:
            self.items.append(obj)
        self._changed()

    def scoringEvents(self, groupid:str=None) -> List[scoring.Notation]:
        subgroup = scoring.makeGroupId(groupid)
        return misc.sumlist(item.scoringEvents(subgroup)
                            for item in _mergeLines(self.items))

    def scoringParts(self, options: scoring.render.RenderOptions = None
                     ) -> List[scoring.Part]:
        notations = self.scoringEvents()
        scoring.stackNotationsInPlace(notations)
        part = scoring.Part(notations, label=self.label)
        return [part]

    def csoundEvents(self, playargs: PlayArgs, scorestruct: ScoreStruct, conf: dict
                     ) -> List[CsoundEvent]:
        playargs.fillWith(self.playargs)
        return misc.sumlist(item.csoundEvents(playargs.copy(), scorestruct, conf)
                            for item in _mergeLines(self.items))


def _asVoice(obj: Union[MusicObj, List[MusicObj]]) -> Voice:
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

    _acceptsNoteAttachedSymbols = False

    def __init__(self, voices: List[MusicObj] = None, label:str=''):
        if voices:
            voices = [_asVoice(v) for v in voices]

        super().__init__(items=voices, label=label)
        self.start = min(v.start for v in self.voices)
        end = max(v.end for v in self.voices)
        self.dur = end - self.start

    @property
    def voices(self):
        return self.items

    def scoringParts(self, options=None) -> List[scoring.Part]:
        parts = []
        for voice in self.voices:
            voiceparts = voice.scoringParts(options=options)
            parts.extend(voiceparts)
        return parts


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# notenames
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generateNotes(start=12, end=127) -> Dict[str, Note]:
    """
    Generates all notes for interactive use.

    From an interactive session,

    locals().update(generate_notes())
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
            enharmonic_note = tools.enharmonic(rest)
            enharmonic_note += str(octave)
            notes[enharmonic_note] = Note(i)
    return notes


def splitNotes(notes: Sequence[Note], splitpoints:List[float], deviation=None
               ) -> List[List[Note]]:
    """
    Split notes at given splitpoints.

    This can be used to split a group of notes into multiple staves. This assumes
    that notes are not synchonous (they do not overlap)

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


def resetImageCache() -> None:
    """
    Reset the image cache. Useful when changing display format
    """
    renderObject.cache_clear()


def asMusic(obj, **kws) -> Union[Note, Chord]:
    """
    Convert obj to a Note or Chord, depending on the input itself

    ::
        int, float      -> Note
        list (of notes) -> Chord
        "C4"            -> Note
        "C4 E4"         -> Chord
    """
    if isinstance(obj, (Note, Chord)):
        return obj
    elif isinstance(obj, str):
        if " " in obj:
            return Chord(obj.split(), **kws)
        return Note(obj, **kws)
    elif isinstance(obj, (list, tuple)):
        return Chord(obj, **kws)
    elif isinstance(obj, (int, float)):
        return Note(obj, **kws)
    else:
        raise TypeError(f"Cannot convert {obj} to a Note or Chord")


class Group(MusicObjList):
    """
    A Group represents a group of objects. These can be simultaneous

    There are no group of groups: if a group is placed inside another group,
    the items of the inner group are placed "ungrouped" inside the outer group

    Example::

        >>> a, b = Note(60, dur=2), Note(61, start=2, dur=1)
        >>> h = Group((a, b))
    """

    def __init__(self, items:Sequence[MusicObj], label:str=''):
        assert isinstance(items, (list, tuple))
        flatitems = []
        for item in items:
            if isinstance(item, Group):
                flatitems.extend(item)
            else:
                flatitems.append(item)
        items = [i.withExplicitTime() for i in flatitems]
        items.sort(key=lambda item:item.start)
        start = min(i.start for i in items)
        super().__init__(items=items, label=label)
        self.dur = self.end - start

    @property
    def end(self) -> Rat:
        return max(i.end or Rat(0) for i in self.items)

    def add(self, obj:MusicObj) -> None:
        self.items.append(obj)

    def __getitem__(self, idx) -> Union[MusicObj, List[MusicObj]]:
        return self.items[idx]

    def __repr__(self):
        objstr = self.items.__repr__()
        return f"Group({objstr})"

    def rec(self, outfile:str=None, sr:int=None, **kws) -> str:
        return recMany(self.items, outfile=outfile, sr=sr, **kws)

    def scoringParts(self, options=None) -> List[scoring.Part]:
        events = self.scoringEvents()
        return scoring.packInParts(events)


def playMany(objs: Sequence[MusicObj], **kws) -> csoundengine.synth.SynthGroup:
    """
    Play multiple objects with the same parameters

    Args:
        objs: the objects to play
        kws: any keywords passed to play

    Returns:
        a SynthGroup holding all generated Synths

    """
    events = sum((obj.events(**kws) for obj in objs), [])
    return play.playEvents(events)


def recMany(objs: Sequence[MusicObj], outfile:str=None, sr:int=None, **kws
            ) -> str:
    """
    Record many objects with the same parameters

    Args:
        objs: the objects to record
        outfile: the path of the generated sound file
        sr: the sample rate
        kws: any keywords passed to rec

    Returns:
        the path of the generated soundfile. This is only needed if
        outfile was None, in which case the path of the generated
        recording is returned
    """
    events = sum((obj.events(**kws) for obj in objs), [])
    return play.recEvents(outfile=outfile, events=events, sr=sr)


def trill(note1: Union[Note, Chord], note2: Union[Note, Chord],
          totaldur: time_t, notedur:time_t=None) -> Chain:
    """
    Create a trill

    Args:
        note1: the first note of the trill (can also be a chord)
        note2: the second note of the trill (can also  be a chord)
        totaldur: total duration of the trill
        notedur: duration of each note. This value will only be used
            if the trill notes have an unset duration

    Returns:
        A realisation of the trill as an Chain of at least the
        given totaldur (can be longer if totaldur is not a multiple
        of notedur)
    """
    note1 = asChord(note1)
    note2 = asChord(note2)
    note1 = note1.clone(dur=note1.dur or notedur or Rat(1, 8))
    note2 = note2.clone(dur=note2.dur or notedur or Rat(1, 8))
    seq = Chain([note1, note2])
    return seq.cycle(totaldur)


def packInVoices(objs: List[MusicObj]) -> List[Voice]:
    return _musicobjtools.packInVoices(objs)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if environment.insideJupyter:
    tools.m21JupyterHook()


