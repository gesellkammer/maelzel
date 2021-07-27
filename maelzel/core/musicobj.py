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

from ._common import *
from .musicobjbase import *
from .workspace import getConfig
from . import play
from . import tools
from . import environment
from . import notation
from .pitch import Pitch
from .csoundevent import PlayArgs, CsoundEvent
from maelzel.scorestruct import ScoreStruct
import functools
import csoundengine

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Optional as Opt, TYPE_CHECKING, List, Iterable as Iter, \
        Sequence as Seq, Dict


@functools.total_ordering
class Note(MusicObj):
    """
    In its simple form, a Note is used to represent a Pitch.

    A Note must have a pitch. It is possible to specify
    an a duration, a time offset (.start), an amplitude
    and an endpitch, resulting in a glissando.
    Internally the pitch of a Note is represented as a fractional
    midinote, so there is no concept of enharmonicity. Notes created
    as ``Note(61)``, ``Note("4C#")`` or ``Note("4Db")`` will all result
    in the same Note.

    Args:
        pitch: a midinote or a note as a string. A pitch can be
            a midinote or a notename as a string.
        dur: the duration of this note (optional)
        amp: amplitude 0-1 (optional)
        start: start fot the note (optional)
        gliss: if given, defines a glissando. It can be either the endpitch of
            the glissando, or True, in which case the endpitch remains undefined
        label: a label str to identify this note

    Attributes:
        pitch: the pitch of this Note, as midinote
        dur: the duration of the Note (a Rat), or None
        amp: the amplitude (0-1), or None
        start: the start time (as Rat), or None
        endpitch: the end pitch (as midinote), or None
        label: a string label
    """

    __slots__ = ('pitch', 'amp', 'gliss')

    def __init__(self,
                 pitch: pitch_t,
                 dur: time_t = None,
                 amp:float=None,
                 start:time_t=None,
                 gliss:U[pitch_t, bool]=False,
                 label:str='',
                 dynamic:str=None,
                 ):

        MusicObj.__init__(self, dur=dur, start=start, label=label)
        self.pitch: float = tools.asmidi(pitch)
        self.amp: Opt[float] = amp
        self.gliss: U[float, bool] = gliss if isinstance(gliss, bool) else tools.asmidi(gliss)
        if dynamic:
            self.setSymbol('Dynamic',self.dynamic)

    def clone(self,
              pitch: pitch_t = UNSET,
              dur: Opt[time_t] = UNSET,
              amp: Opt[time_t] = UNSET,
              start: Opt[time_t] = UNSET,
              gliss: U[pitch_t, bool] = UNSET,
              label: str = UNSET) -> Note:
        """

        Args:
            pitch: a pitch to override the current pitch
            dur: a duration to override the current duration (or None to remove it)
            amp: an amplitude to override current value (None to remove it)
            start: a start time to override current value (None to remove it)
            gliss: the end pitch or True for an undefined gliss. (None to remove it)
            label: the label of this note

        Returns:
            a new note
        """
        out = Note(pitch if pitch is not UNSET else self.pitch,
                   dur if dur is not UNSET else self.dur,
                   amp if amp is not UNSET else self.amp,
                   start if start is not UNSET else self.start,
                   gliss if gliss is not UNSET else self.gliss,
                   label if label is not UNSET else self.label)
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
    def midi(self) -> float:
        return self.pitch

    @midi.setter
    def midi(self, value: float) -> None:
        self.pitch = value

    @property
    def name(self) -> str:
        return pt.m2n(self.pitch)

    @property
    def pitchclass(self) -> int:
        return round(self.midi) % 12


    @property
    def cents(self) -> int:
        return tools.midicents(self.pitch)

    @property
    def centsrepr(self) -> str:
        return tools.centsshown(self.cents,
                                divsPerSemitone=getConfig()['semitoneDivisions'])

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
        config = getConfig()
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
                                playbackGain=self.amp, group=groupid)
        notes = [note]
        if not isinstance(self.gliss, bool):
            start = self.end if self.end is not None else None
            groupid = groupid or str(hash(self))
            notes[0].groupid = groupid
            assert self.gliss >= 12
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
            config = getConfig()
            if config['repr.showFreq']:
                elements.append("%dHz" % int(self.freq))
            if self.amp is not None and self.amp < 1:
                elements.append("%ddB" % round(amp2db(self.amp)))
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
            return f'‹{elements[0].ljust(3)} {" ".join(elements[1:])}›'

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
        configured via ``currentConfig()['semitoneDivisions']``
        """
        if step == 0:
            step = 1/getConfig()['semitoneDivisions']
        return self.clone(pitch=round(self.pitch / step) * step)

    def csoundEvents(self, playargs: PlayArgs, scorestruct:ScoreStruct, conf:dict
                     ) -> List[CsoundEvent]:
        playargs.fillWith(self.playargs)
        playargs.fillWithConfig(conf)
        amp = 1.0 if self.amp is None else self.amp
        endmidi = self.gliss if self.gliss > 1 else self.pitch
        start = self.start or 0
        dur = self.dur or 1.0
        starttime = float(scorestruct.beatToTime(start))
        endtime   = float(scorestruct.beatToTime(start + dur))
        bps = [[starttime, self.pitch, amp],
               [endtime, endmidi,   amp]]
        return [CsoundEvent.fromPlayArgs(bps=bps, playargs=playargs)]

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
            endnote = tools.asmidi(endpitch)
        dur = dur or self.resolvedDuration()
        startamp = self.resolvedAmp()
        endamp = firstval(endamp, self.amp, startamp)
        start = self.start or 0
        breakpoints = [(start, self.pitch, startamp),
                       (start+dur, endnote, endamp)]
        return Line(breakpoints)

    def resolvedAmp(self) -> float:
        """
        Get the amplitude of this object, or a default amplitude

        Returns a default amplitude if no amplitude was define (self.amp is None).
        The default amplitude can be customized via
        ``currentConfig()['play.defaultAmplitude']``

        Returns:
            the amplitude (a value between 0-1, where 0 corresponds to 0dB)
        """
        return self.amp if self.amp is not None else \
            getConfig()['play.defaultAmplitude']

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Note:
        return self.clone(pitch=pitchmap(self.pitch))

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


def asNote(n: U[float, str, Note, Pitch],
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

    A Note can also be created via `asNote((pitch, amp))`
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
        rowhashes = []
        rowhashes = [hash(tuple(bp)) for bp in self.bps]
        rowhashes.append(self.start)
        return hash(tuple(rowhashes))

    def __repr__(self):
        return f"Line(start={self.start}, bps={self.bps})"

    def quantizePitch(self, step=0) -> Line:
        """ Returns a new object, rounded to step """
        if step == 0:
            step = 1/getConfig()['semitoneDivisions']
        bps = [ (bp[0], tools.quantizeMidi(bp[1], step)) + bp[2:]
                for bp in self.bps ]
        if len(bps) >= 3:
            bps = misc.simplify_breakpoints(bps, coordsfunc=lambda bp:(bp[0], bp[1]),
                                            tolerance=0.01)
        return Line(bps)

    def scoringEvents(self, groupid:str=None) -> List[scoring.Notation]:
        offsets = self.getOffsets()
        groupid = groupid or scoring.makeGroupId()
        notations = []
        for bp0, bp1 in iterlib.pairwise(self.bps):
            pitch = bp0[1]
            assert pitch > 0
            dur = bp1[0] - bp0[0]
            offset = bp0[0] + self.start
            assert dur > 0
            ev = scoring.makeNote(pitch=pitch, offset=offset, duration=dur,
                                  gliss=bp0[1] != bp1[1], group=groupid)
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
        return notations

    def timeTransform(self, timemap: Callable[[float], float]) -> Line:
        bps = []
        start2 = timemap(self.start)
        for bp in self.bps:
            t0 = bp[0] + self.start
            t1 = timemap(t0)
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

    __slots__ = ('amp', 'gliss', 'notes')

    def __init__(self, *notes,
                 dur:time_t=None,
                 amp:float=None,
                 start=None,
                 gliss=False,
                 label:str=''
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
                True to leave the end pitches unspecified
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
        self.gliss: U[bool, List[float]] = gliss

        if not notes:
            return

        # notes might be: Chord([n1, n2, ...]) or Chord(n1, n2, ...)
        if misc.isgeneratorlike(notes):
            notes = list(notes)
        n0 = notes[0]
        if len(notes) == 1:
            if isinstance(n0, (Chord, EventSeq)):
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
        MusicObj.__init__(self, dur=dur, start=start, label=label)

        for note in notes:
            if isinstance(note, Note):
                # we erase any duration or offset of the individual notes
                note = note.clone(dur=None, start=None)
            else:
                assert isinstance(note, (int, float, str))
                note = asNote(note, amp=amp, dur=None, start=None)
            self.notes.append(note)
        self.notes = list(set(self.notes))
        self.sort()

    def __len__(self) -> int:
        return len(self.notes)

    def __getitem__(self, idx) -> U[Note, Chord]:
        out = self.notes.__getitem__(idx)
        if isinstance(out, list):
            out = self.__class__(out)
        return out

    def __iter__(self) -> Iter[Note]:
        return iter(self.notes)

    def scoringEvents(self, groupid:str = None) -> List[scoring.Notation]:
        config = getConfig()
        pitches = [note.pitch for note in self.notes]
        annot = self._scoringAnnotation()
        dur = self.dur if self.dur is not None else config['defaultDuration']
        chord = scoring.makeChord(pitches=pitches, duration=dur, offset=self.start,
                                  annotation=annot, group=groupid)
        notations = [chord]
        if self.gliss:
            if self.gliss is True:
                chord.gliss = True
            else:
                groupid = groupid or scoring.makeGroupId()
                chord.groupid = groupid
                endEvent = scoring.makeChord(pitches=self.gliss,
                                             duration=0,
                                             offset=self.end,
                                             group=groupid)
                notations.append(endEvent)
        if self._symbols:
            tools.applySymbols(self._symbols, notations)
        return notations

    def asmusic21(self, **kws) -> m21.stream.Stream:
        config = getConfig()
        arpeggio = _normalizeChordArpeggio(kws.get('arpeggio', None), self)
        if arpeggio:
            dur = config['show.arpeggioDuration']
            return EventSeq(self.notes, itemDefaultDur=dur).asmusic21()
        events = self.scoringEvents()
        scoring.stackNotationsInPlace(events, start=self.start)
        parts = scoring.distributeNotationsByClef(events)
        return notation.renderWithCurrentWorkspace(parts).asMusic21()

    def __hash__(self):
        if self._hash:
            return self._hash
        glisshash = hash(float(self.gliss))
        data = (self.dur, self.start, self.label, glisshash, *(n.pitch for n in self.notes))
        self._hash = h = hash(data)
        return h

    def append(self, note:U[float, str, Note, Pitch]) -> None:
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
            step = 1/getConfig()['semitoneDivisions']
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

        Returns:
            self
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
    def pitches(self) -> List[float]:
        return [n.midi for n in self.notes]

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
            amp = note.amp if note.amp is not None else 1.
            bps = [[starttime, note.pitch, amp],
                   [endtime, endpitch, amp]]
            events.append(CsoundEvent.fromPlayArgs(bps=bps, playargs=playargs))
        return events

    def asSeq(self, dur=None) -> EventSeq:
        """ Convert this Chord to an EventSeq """
        return EventSeq(self.notes, itemDefaultDur=dur or self.dur)

    def __repr__(self):
        # «4C+14,4A 0.1q -50dB»
        elements = [" ".join(pt.m2n(p.midi) for p in self.notes)]
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
            return f'«{elements[0]}»'
        else:
            return f'«{elements[0].ljust(3)} {" ".join(elements[1:])}»'

    def dump(self, indents=0):
        elements = f'start={self.start}, dur={self.dur}, gliss={self.gliss}'
        print(f"{'  '*indents}Chord({elements})")
        if self._playargs:
            print("  "*(indenst+1), self.playargs)
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
                db = curve(amp2db(note.amp))
                notes.append(note.clone(amp=db2amp(db)))
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
        mewpitches = [pitchmap(p) for p in self.pitches]
        newnotes = [n.clone(pitch=pitch) for n, pitch in zip(self.notes, newpitches)]
        return self.clone(notes=newnotes)


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


def asEvent(obj, **kws) -> U[Note, Chord]:
    """Convert `obj` to an event (a Note or a Chord, depending on obj)"""
    if isinstance(obj, (Note, Chord)):
        return obj
    return N(obj, **kws)


def _normalizeChordArpeggio(arpeggio: U[str, bool], chord: Chord) -> bool:
    config = getConfig()
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
    assert start is not None
    if defaultDur is None:
        defaultDur = asRat(getConfig()['defaultDuration'])
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


def stackEventsInPlace(events: List[MusicObj],
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
    assert start is not None
    if defaultDur is None:
        defaultDur = getConfig()['defaultDuration']
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
        now += ev.dur
    for ev1, ev2 in iterlib.pairwise(events):
        assert ev1.start <= ev2.start
        assert ev1.end == ev2.start


class SeqMusicObj(MusicObj):
    __slots__ = ('items')

    def __init__(self, items: List[MusicObj], label:str=''):
        if items:
            assert all(item.dur is not None and item.start is not None
                       for item in items)
            self.items = items
            end = max(it.end for it in items)
            start = min(it.start for it in items)
            dur = end - start
        else:
            self.items = []
            dur = None
            start = None
        super().__init__(dur=dur, start=start, label=label)

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

    def scoringEvents(self, groupid:str=None) -> List[scoring.Notation]:
        """
        Returns the scoring events corresponding to this object

        Args:
            groupid: if given, all events are given this groupid

        Returns:
            the scoring notations
        """
        return sum((i.scoringEvents(groupid=groupid) for i in self.items), [])

    def csoundEvents(self, playargs: PlayArgs, scorestruct: ScoreStruct, conf:dict
                     ) -> List[CsoundEvent]:
        playargs.fillWith(self.playargs)
        events = []
        for item in self.items:
            itemevents = item.csoundEvents(playargs.copy(), scorestruct, conf)
            events.extend(itemevents)
        return events

    def _play(self, **kws) -> csoundengine.synth.SynthGroup:
        """
        Args:
            kws: any kws is passed directly to each individual event

        Returns:
            a SynthGroup collecting all the synths of each item
        """
        synths: List[csoundengine.synth.Synth] = []
        for i in self.items:
            itemsynths = i.play(**kws)
            synths.extend(itemsynths)
        return csoundengine.synth.SynthGroup(synths)

    def quantizePitch(self:T, step=0.) -> T:
        if step == 0:
            step = getConfig()['semitoneDivisions']
        return type(self)([i.quantizePitch(step) for i in self.items])

    def timeShift(self:T, timeoffset: time_t) -> T:
        return type(self)([item.timeShift(timeoffset) for item in self.items],
                          label=self.label)

    def timeTransform(self:T, timemap: Callable[[float], float]) -> T:
        items = [item.timeTransform(timemap) for item in self.items]
        return type(self)(items, label=self.label)

    def pitchTransform(self: T, pitchmap: Callable[[float], float]) -> T:
        newitems = [item.pitchTransform(pitchmap) for item in self.items]
        return self.clone(items=newitems)

    def dump(self, indents=0):
        print(f'{"  "*indents}{repr(self)}')
        if self._playargs:
            print("  "*(indents+1), self.playargs)
        for item in self.items:
            item.dump(indents+1)


class EventSeq(SeqMusicObj):
    """
    A seq. of Notes or Chords

    Within an EventSeq, start times are relative to the sequence itself

    """
    _acceptsNoteAttachedSymbols = False

    def __init__(self, items: List[MusicObj] = None, start:time_t=None, label:str=''):
        start = asRat(start)
        if start is not None:
            start = asRat(start)
        if items:
            items = [asEvent(item) for item in items]
            items = stackEvents(items, defaultDur=self.itemDefaultDur, start=start)
        super().__init__(items=items, label=label)

    def resolvedDuration(self) -> Rat:
        if not self.items:
            return Rat(0)
        end = self.items[-1].end
        assert end is not None
        start = self.items[0].start
        assert start is not None
        return end - start

    def append(self, item:U[Note, Chord]) -> None:
        """
        Append an item to this seq.

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
        assert self.start is not None
        self.dur = item.end - self.start

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iter[Chord]:
        return iter(self.items)

    def __getitem__(self, idx):
        out = self.items.__getitem__(idx)
        if isinstance(out, MusicObj):
            return out
        elif isinstance(out, list):
            return self.__class__(out)
        else:
            raise ValueError(f"__getitem__ returned {out}, expected Chord or list of Chords")

    def scoringEvents(self, groupid:str=None) -> List[scoring.Notation]:
        events = []
        start = self.start if self.start is not None else Rat(0)
        groupid = groupid or scoring.makeGroupId()
        for item in self.items:
            scoringEvents = item.scoringEvents(groupid)
            for ev in scoringEvents:
                if ev.duration is None and ev.offset is None:
                    ev.duration = self.dur
            events.extend(scoringEvents)
        if events and events[0].offset is None:
            events[0].offset = start
        return events

    def dump(self, indents=0):
        lines = [f"{' '*indents}EventSeq"]
        for item in self.items:
            sublines = repr(item).splitlines()
            for subline in sublines:
                lines.append(f"{'  '*(indents+1)}{subline}")
        return "\n".join(lines)

    def __repr__(self):
        if len(self.items) < 10:
            itemstr = ", ".join(repr(_) for _ in self.items)
        else:
            itemstr = ", ".join(repr(_) for _ in self.items[:10]) + ", …"
        return f'EventSeq([{itemstr}])'

    def __hash__(self):
        if not self._hash:
            self._hash = hash(tuple(hash(i) ^ 0x1234 for i in self.items))
        return self._hash

    def csoundEvents(self, playargs: PlayArgs, scorestruct:ScoreStruct, conf:dict
                     ) -> List[CsoundEvent]:
        allevents = []
        playargs.fillWith(self.playargs)
        for item in self.items:
            allevents.extend(item.csoundEvents(playargs.copy(), scorestruct, conf))
        return allevents

    def cycle(self, dur:time_t, crop=True) -> EventSeq:
        """
        Cycle the items in this seq. until the given duration is reached

        Args:
            dur: the total duration
            crop: if True, the last event will be cropped to fit
                the given total duration. Otherwise, it will last
                its given duration, even if that would result in
                a total duration longer than the given one

        Returns:
            the resulting EventSeq
        """
        items = []
        defaultDur = self.dur
        it = iterlib.cycle(self)
        totaldur = Rat(0)
        dur = asRat(dur)
        while totaldur < dur:
            item = next(it)
            maxdur = dur - totaldur
            if crop:
                if item.dur is None or item.dur > maxdur:
                    item = item.clone(dur=maxdur)
            elif item.dur is None:
                if crop:
                    item = item.clone(dur=min(defaultDur, maxdur))
                else:
                    item = item.clone(dur=defaultDur)
            assert item.dur is not None
            totaldur += item.dur
            items.append(item)
        return EventSeq(items, start=self.start)

    def clone(self, items:List[U[Note, Chord]]=None, dur:time_t=None, start:time_t=None,
              **kws) -> EventSeq:
        """Returns a copy of self with the given attributes modified"""
        items = items if items is not None else self.items
        dur = dur if dur is not None else self.dur
        start = start if start is not None else self.start
        return EventSeq(items, itemDefaultDur=dur, start=start, label=self.label)

    def quantizePitch(self, step=0.) -> EventSeq:
        step = step or 1/getConfig()['semitoneDivisions']
        items = [i.quantizePitch(step) for i in self.items]
        return self.clone(items)

    def timeShift(self, timeoffset:time_t) -> EventSeq:
        items = [i.timeShift(timeoffset) for i in self.items]
        return self.clone(items=items)

    def asVoice(self) -> Voice:
        """Convert this EventSeq to a Voice"""
        return Voice(self.items)




class Voice(SeqMusicObj):
    """
    A Voice is a seq. of non-overlapping objects
    """
    _acceptsNoteAttachedSymbols = False

    def __init__(self, items:List[MusicObj]=None, label:str= ''):
        self.instrs: Dict[MusicObj, str] = {}
        if items:
            items = stackEvents(items, start=asRat(0))
        super().__init__(items=items, label=label)

    def __repr__(self) -> str:
        parts = []
        if self.label:
            parts.append(self.label)
        parts.append(str(self.items))
        s = ', '.join(parts)
        return f"Voice({s})"

    def _changed(self):
        if self.items:
            self.dur = self.items[-1].end-self.items[0].start
            self.start = self.items[0].start
        super()._changed()

    def endTime(self) -> Rat:
        if not self.items:
            return Rat(0)
        end = self.items[-1].end
        assert end is not None
        return end

    def startTime(self) -> Rat:
        if not self.items:
            return Rat(0)
        start = self.items[0].start
        assert start is not None
        return start

    def isEmptyBetween(self, start:time_t, end:num_t) -> bool:
        if not self.items:
            return True
        if start >= self.endTime():
            return True
        if end < self.startTime():
            return True
        return all(intersection(i.start, i.end, start, end) is None
                    for i in self.items)

    def needsSplit(self) -> bool:
        pass

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
            obj = obj.withExplicitTime(start=self.endTime())
        assert obj.start is not None and obj.end is not None
        if not self.isEmptyBetween(obj.start, obj.end):
            msg = f"obj {obj} ({obj.start}:{obj.end}) does not fit in voice"
            raise ValueError(msg)
        assert obj.dur is not None and obj.dur>0
        if obj.start is None:
            obj = obj.clone(start=self.endTime())
        self.items.append(obj)
        if obj.start < self.endTime():
            self.items.sort(key=lambda obj:obj.start)
        self._changed()

    def extend(self, objs:List[MusicObj]) -> None:
        objs.sort(key=lambda obj:obj.start or 0)
        start = objs[0].start
        assert start is not None and start >= self.endTime()
        for obj in objs:
            self.items.append(obj)
        self._changed()

    def scoringParts(self) -> List[scoring.Part]:
        notations = self.scoringEvents()
        scoring.stackNotationsInPlace(notations)
        part = scoring.Part(notations, label=self.label)
        return [part]


def _asVoice(obj: U[MusicObj, List[MusicObj]]) -> Voice:
    if isinstance(obj, Voice):
        return obj
    elif isinstance(obj, EventSeq):
        return obj.asVoice()
    elif isinstance(obj, (Chord, Note)):
        return _asVoice([obj])
    elif isinstance(obj, (list, tuple)):
        return EventSeq(obj).asVoice()
    else:
        raise TypeError(f"Cannot convert {obj} of type {type(obj)} to a Voice")


class Score(SeqMusicObj):

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

    def scoringParts(self) -> List[scoring.Part]:
        parts = []
        for voice in self.voices:
            voiceparts = voice.scoringParts()
            parts.extend(voiceparts)
        return parts




def _asTimedObj(obj: MusicObj, start:num_t=Rat(0),
                dur:num_t=None) -> MusicObj:
    """
    A TimedObj has a start time and a duration
    """
    if obj.start is not None and obj.dur is not None:
        return obj
    if dur is None:
        cfg = getConfig()
        dur = cfg['defaultDuration']
    dur = obj.dur if obj.dur is not None else dur
    assert dur > 0 and start >= 0
    start = obj.start if obj.start is not None else start
    return obj.clone(dur=asRat(dur), start=asRat(start))

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


def _setJupyterHooks() -> None:
    """
    Sets jupyter display hooks for multiple classes
    """
    # MusicObj.setJupyterHook()
    Pitch.setJupyterHook()
    tools.m21JupyterHook()


def _splitNotesOnce(notes: U[Chord, Seq[Note]], splitpoint:float, deviation=None,
                    ) -> Tuple[List[Note], List[Note]]:
    """
    Split a list of notes into two lists, one above and one below the splitpoint

    Args:
        notes: a seq. of Notes
        splitpoint: the pitch to split the notes
        deviation: an acceptable deviation to fit all notes
            in one group (config: 'splitAcceptableDeviation')

    Returns:
        notes above and below

    """
    deviation = deviation or getConfig()['splitAcceptableDeviation']
    if all(note.pitch > splitpoint - deviation for note in notes):
        above = [n for n in notes]
        below = []
    elif all(note.pitch < splitpoint + deviation for note in notes):
        above = []
        below = [n for n in notes]
    else:
        above, below = [], []
        for note in notes:
            (above if note.pitch > splitpoint else below).append(note)
    return above, below


def splitNotes(notes: Seq[Note], splitpoints:List[float], deviation=None
               ) -> List[List[Note]]:
    """
    Split notes at given splitpoints.

    This can be used to split a group of notes into multiple staves

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
        above, below = _splitNotesOnce(above, splitpoint=splitpoint, deviation=deviation)
        if below:
            tracks.append(below)
        if not above:
            break
    return tracks


def splitNotesIfNecessary(notes:List[Note], splitpoint:float, deviation=None
                          ) -> List[List[Note]]:
    """
    Like _splitNotesOnce, but returns only groups which have notes in them

    This can be used to split in more than one staves.

    Args:
        notes: the notes to split
        splitpoint: the split point
        deviation: an acceptable deviation, if all notes could fit in one part

    Returns:
        a list of parts (a part is a list of notes)

    """
    return [p for p in _splitNotesOnce(notes, splitpoint, deviation) if p]


def resetImageCache() -> None:
    """
    Reset the image cache. Useful when changing display format
    """
    renderObject.cache_clear()


def asMusic(obj) -> U[Note, Chord]:
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
            return Chord(obj.split())
        return Note(obj)
    elif isinstance(obj, (list, tuple)):
        return Chord(obj)
    elif isinstance(obj, (int, float)):
        return Note(obj)
    else:
        raise TypeError(f"Cannot convert {obj} to a Note or Chord")


def gliss(a, b, dur:time_t=None) -> U[Note, Chord]:
    """
    Create a gliss. between a and b. a should implement
    the method .gliss (either a Note or a Chord)

    Args:
        a: the start object
        b: the end object (should have the same type as obj1)
        dur: the duration of the glissando

    Returns:
        the resulting Note or Chord
    """
    m1 = asMusic(a)
    m2 = asMusic(b)
    assert isinstance(m2, type(m1))
    return m1.makeGliss(m2, dur=dur)


class Group(SeqMusicObj):
    """
    A Group represents a group of objects. They can be simultaneous

    Example::

        >>> a, b = Note(60, dur=2), Note(61, start=2, dur=1)
        >>> h = Group((a, b))

    """

    def __init__(self, items:Seq[MusicObj], label:str=''):
        assert isinstance(items, (list, tuple))
        items = [i.withExplicitTime() for i in items]
        items.sort(key=lambda item:item.start)
        start = min(i.start for i in items)
        SeqMusicObj.__init__(self, items=items, label=label)
        self.dur = self.end - start

    @property
    def end(self) -> Rat:
        return max(i.end or Rat(0) for i in self.items)

    def add(self, obj:MusicObj) -> None:
        self.items.append(obj)

    def __getitem__(self, idx) -> U[MusicObj, List[MusicObj]]:
        return self.items[idx]

    def __repr__(self):
        objstr = self.items.__repr__()
        return f"Group({objstr})"

    def rec(self, outfile:str=None, sr:int=None, **kws) -> str:
        return recMany(self.items, outfile=outfile, sr=sr, **kws)

    def scoringParts(self) -> List[scoring.Part]:
        events = self.scoringEvents()
        return scoring.packInParts(events)


def _collectPlayEvents(objs: List[MusicObj], **kws) -> List[CsoundEvent]:
    """
    Collect events of multiple objects using the same parameters

    Args:
        objs: a seq. of objects
        **kws: keywords passed to play

    Returns:
        a list of the events
    """
    return sum((obj.events(**kws) for obj in objs), [])


def playMany(objs, **kws) -> csoundengine.synth.SynthGroup:
    """
    Play multiple objects with the same parameters

    Args:
        objs: the objects to play
        kws: any keywords passed to play

    Returns:
        a SynthGroup holding all generated Synths

    """
    return play.playEvents(_collectPlayEvents(objs, **kws))


def recMany(objs: List[MusicObj], outfile:str=None, sr:int=None, **kws
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
        outfile was None, in which
    """
    allevents = _collectPlayEvents(objs, **kws)
    return play.recEvents(outfile=outfile, events=allevents, sr=sr)


def trill(note1: U[Note, Chord], note2: U[Note, Chord],
          totaldur: time_t, notedur:time_t=None) -> EventSeq:
    """
    Create a trill

    Args:
        note1: the first note of the trill (can also be a chord)
        note2: the second note of the trill (can also  be a chord)
        totaldur: total duration of the trill
        notedur: duration of each note. This value will only be used
            if the trill notes have an unset duration

    Returns:
        A realisation of the trill as an EventSeq of at least the
        given totaldur (can be longer if totaldur is not a multiple
        of notedur)
    """
    note1 = asChord(note1)
    note2 = asChord(note2)
    note1 = note1.clone(dur=note1.dur or notedur or Rat(1, 8))
    note2 = note2.clone(dur=note2.dur or notedur or Rat(1, 8))
    seq = EventSeq([note1, note2])
    return seq.cycle(totaldur)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if environment.insideJupyter:
    _setJupyterHooks()


