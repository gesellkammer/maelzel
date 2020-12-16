from __future__ import annotations

from . import util
from .common import *
from .core import Notation

from pathlib import Path
import dataclasses
from typing import Tuple, Optional as Opt, List, Union as U
from bisect import bisect
from collections import defaultdict

import music21 as m21
from emlib import misc

from maelzel.music import m21tools



@dataclasses.dataclass
class MeasureDef:
    """
    A measure definition. This does not hold any other data (notes) but the information
    of the measure itself, to be used inside a ScoreStructure
    """
    timesig: timesig_t
    quarterTempo: F
    annotation: str = ""
    timesigInherited: bool = False
    tempoInherited: bool = False
    barline: str = ""

    def __post_init__(self):
        misc.assert_type(self.timesig, (int, int))
        self.quarterTempo = asF(self.quarterTempo)

    def numberOfBeats(self):
        return 4 * F(*self.timesig)

    def duration(self) -> F:
        if self.quarterTempo is None or self.timesig is None:
            raise ValueError("MeasureDef not fully defined")
        return self.numberOfBeats() * (F(60)/self.quarterTempo)


@dataclasses.dataclass
class ScoreLocation:
    measureNum: int
    beat: F = 0

    def __repr__(self):
        beat = float(self.beat.limit_denominator(10000))
        return f"ScoreLocation(measureNum={self.measureNum}, {beat=})"

    def __post_init__(self):
        assert isinstance(self.beat, F)


class ScoreStructure:
    def __init__(self, endless=False, autoextend=False):
        """
        A ScoreStructure hold all the information about a score but no data
        (actual measures with notes inside, etc). A ScoreStructure is used
        to quantize and render scores.

        A ScoreStructure consists of some metadata and a list of MeasureDefs,
        where each MeasureDef defines the properties of the measure at the given
        index. If a ScoreStructure is marked as endless, it is possible to query
        it (ask for a MeasureDef, convert beats to time, etc.) outside of the defined
        measures.

        Args:
            endless: mark this ScoreStructure as endless
            autoextend: this is only valid if the score is marked as endless. If True,
                querying this ScoreStructure outside its defined boundaries will create
                the necessary MeasureDefs to cover the point in question. In concrete, if
                a ScoreStructure has only one MeasureDef, .getMeasureDef(3) will create
                three MeasureDefs, cloning the first MeasureDef, and return the last one.
                Also .beatToTime and .timeToBeat will create new MeasureDefs.

        Example::
            # create an endless score with a given time signature
            >>> scorestruct = ScoreStructure(endless=True)
            >>> scorestruct.addMeasure((4, 4), quarterTempo=72)
            # this is the same as:
            >>> scorestruct = ScoreStructure.fromTimesig((4, 4), 72)

            # Create the beginning of Sacre
            >>> s = ScoreStructure()
            >>> s.addMeasure((4, 4), 50)
            >>> s.addMeasure((3, 4))
            >>> s.addMeasure((4, 4))
            >>> s.addMeasure((2, 4))
            >>> s.addMeasure((3, 4), numMeasures=2)

        A ScoreStructure can also be created from a string (see .fromString to learn
        about the format), which can be read from a file
        """
        self.measuredefs: List[MeasureDef] = []

        self.title = None
        self.endless = endless
        self.autoextend = autoextend
        self._changed = True
        self._offsetsIndex: List[F] = []

    @staticmethod
    def fromString(s: str, initialTempo=60, initialTimeSignature=(4, 4)
                   ) -> ScoreStructure:
        """
        Format::

            measureNum, timeSig, tempo

        * Tempo refers always to a quarter note
        * Any value can be left out: , 5/8,
        * The measure number and/or the tempo can both be left out::

            These are all the same (assuming the last defined measure is 1):
            2, 5/8, 63
            5/8, 63
            5/8
        * measure numbers start at 0
        * comments start with `#` and last until the end of the line

        Example::

            0, 4/4, 60
            ,3/4,80     # Assumes measureNum=1
            10, 5/8, 120
            30,,        # last measure (inclusive, the score will have 31 measures)
        """
        tempo = initialTempo
        timesig = initialTimeSignature
        measure = -1
        struct = ScoreStructure()

        def lineStrip(l:str) -> str:
            if "#" in l:
                l = l.split("#")[0]
            return l.strip()

        for i, line0 in enumerate(s.splitlines()):
            line = lineStrip(line0)
            if not line:
                continue
            newMeasure, newTimesig, newTempo = util.parseScoreStructLine(line)
            if newMeasure is None:
                newMeasure = measure + 1
            if newTempo is not None:
                tempo = newTempo
            if newTimesig is not None:
                timesig = newTimesig
            for m in range(measure, newMeasure):
                struct.addMeasure(timesig=timesig, quarterTempo=tempo)
            measure = newMeasure

        return struct

    @staticmethod
    def fromTimesig(timesig: timesig_t, quarterTempo=60, numMeasures:int=None
                    ) -> ScoreStructure:
        """
        Creates a ScoreStructure from a time signature and tempo.
        If numMeasures is given, the resulting ScoreStructure will
        have as many measures defined and will be finite. Otherwise
        the ScoreStructure will be flagged as endless

        Args:
            timesig: the time signature, a tuple (num, den)
            quarterTempo: the tempo of a quarter note
            numMeasures: the number of measures of this score. If None
                is given, the ScoreStructure will be endless

        Returns:
            a ScoreStructure

        """
        if numMeasures is None:
            out = ScoreStructure(endless=True)
            out.addMeasure(timesig=timesig, quarterTempo=quarterTempo)
        else:
            out = ScoreStructure(endless=False)
            out.addMeasure(timesig, quarterTempo, numMeasures=numMeasures)
        return out

    def numMeasures(self) -> int:
        """
        Returns the number of defined measures, independently of
        this ScoreStructure being endless or not
        """
        return len(self.measuredefs)

    def getMeasureDef(self, idx:int) -> MeasureDef:
        """
        Returns the MeasureDef at the given index. If the scorestruct
        is endless and the index is outside of the defined range, the
        returned MeasureDef will be the last defined MeasureDef
        """
        if idx < len(self.measuredefs):
            return self.measuredefs[idx]
        # outside of defined measures
        if not self.endless:
            raise IndexError(f"index out of range. The score has "
                             f"{len(self.measuredefs)} measures defined")
        if not self.autoextend:
            return self.measuredefs[-1]
        for n in range(len(self.measuredefs)-1, idx):
            self.addMeasure()

    def addMeasure(self, timesig: timesig_t=None, quarterTempo: number_t=None,
                   annotation:str=None, numMeasures:int=1) -> None:
        """
        Add a measure definition to this score structure

        Args:
            timesig: the time signature of the new measure. If not given, the last
                time signature will be used
            quarterTempo: the tempo of a quarter note. If not given, the last tempo
                will be used
            annotation: each measure can have a text annotation
            numMeasures: if this is > 1, multiple measures of the same kind can be
                added

        Example::

            # Create a 4/4 score, 32 measures long
            >>> s = ScoreStructure()
            >>> s.addMeasure((4, 4), 52, numMeasures=32)
        """
        if timesig is None:
            timesigInherited = True
            timesig = self.measuredefs[-1].timesig if self.measuredefs else (4, 4)
        else:
            timesigInherited = False
        if quarterTempo is None:
            tempoInherited = True
            quarterTempo = self.measuredefs[-1].quarterTempo if self.measuredefs else F(60)
        else:
            tempoInherited = False
        measuredef = MeasureDef(timesig=timesig, quarterTempo=quarterTempo,
                                annotation=annotation, timesigInherited=timesigInherited,
                                tempoInherited=tempoInherited)
        self.measuredefs.append(measuredef)
        self._changed = True
        if numMeasures > 1:
            self.addMeasure(numMeasures=numMeasures-1)

    def __str__(self) -> str:
        lines = ["ScoreStructure(["]
        for m in self.measuredefs:
            lines.append("    " + str(m))
        lines.append("])")
        return "\n".join(lines)

    def _update(self):
        if not self._changed:
            return
        now = F(0)
        starts = []
        for mdef in self.measuredefs:
            starts.append(now)
            now += mdef.duration()
        self._changed = False
        self._offsetsIndex = starts

    def beatToTime(self, measure: int, beat:F=F(0)) -> F:
        """
        Return the elapsed time at the given score location

        Args:
            measure: the measure number (starting with 0)
            beat: the beat within the measure

        Returns:
            a time in seconds (as a Fraction to avoid rounding problems)
        """
        self._update()
        if measure > len(self.measuredefs) - 1:
            if not self.endless:
                raise ValueError("Measure outside of score")
            if self.autoextend:
                for _ in range(len(self.measuredefs)-1, measure):
                    self.addMeasure()
            else:
                last = len(self.measuredefs)-1
                lastTime = self.beatToTime(last)
                mdef = self.getMeasureDef(last)
                mdur = mdef.duration()
                fractionalDur = beat * F(60)/mdef.quarterTempo
                return lastTime + (measure - last)*mdur + fractionalDur

        now = self._offsetsIndex[measure]
        mdef = self.measuredefs[measure]

        measureBeats = mdef.numberOfBeats()
        if beat > measureBeats:
            raise ValueError(f"Beat outside of measure, measure={mdef}")

        return now+F(60)/mdef.quarterTempo*beat

    def timeToBeat(self, time: time_t) -> Opt[ScoreLocation]:
        """
        Find the location in score corresponding to the given time in seconds

        Args:
            time: the time in seconds

        Returns:
            a ScoreLocation (.measureNum, .beat). If the score is not
             endless and the time is outside the score, None is returned
        """
        if not self.measuredefs:
            raise IndexError("This ScoreStructure is empty")

        self._update()
        time = asF(time)
        idx = bisect(self._offsetsIndex, time)
        if idx < len(self.measuredefs):
            m = self.measuredefs[idx-1]
            assert self._offsetsIndex[idx-1]<=time<self._offsetsIndex[idx]
            dt = time-self._offsetsIndex[idx-1]
            beat = dt*m.quarterTempo/F(60)
            return ScoreLocation(idx-1, beat)

        # is it within the last measure?
        m = self.measuredefs[idx-1]
        dt = time - self._offsetsIndex[idx-1]
        if dt <= m.duration():
            beat = dt*m.quarterTempo/F(60)
            return ScoreLocation(idx-1, beat)
        # outside of score
        if not self.endless:
            return None
        raise NotImplementedError("endless score not implemented here")

    def elapsedTime(self, a:Tuple[int, F], b:Tuple[int, F]=None) -> F:
        """
        Returns the elapsed time between two score locations

        Args:
            a: a tuple (measureIndex, beatOffset)
            b: a tuple (measureIndex, beatOffset)

        Returns:
            the elapsed time, as a Fraction

        """
        if b is None:
            b = a
            a = (0, F(0))
        t0 = self.beatToTime(a[0], a[1])
        t1 = self.beatToTime(b[0], b[1])
        return t1 - t0

    def show(self) -> None:
        """
        Render and show this ScoreStructure
        """
        score = self.asMusic21(fillMeasures=False)
        score.show('musicxml.pdf')

    def dump(self):
        tempo = -1
        N = len(str(len(self.measuredefs)))
        fmt = "%0" + str(N) + "d" + ", %d/%d"
        for i, m in enumerate(self.measuredefs):
            num, den = m.timesig
            parts = [fmt % (i, num, den)]
            if m.quarterTempo != tempo:
                parts.append(f", {m.quarterTempo}")
                tempo = m.quarterTempo
            print("".join(parts))

    def asMusic21(self, fillMeasures=False) -> m21.stream.Score:
        """
        Return the score structure as a music21 Score

        NB 1: to set score title/composer,
              use emlib.music.m21tools.scoreSetMetadata

        TODO: render barlines according to measureDef
        """
        s = m21.stream.Part()
        lasttempo = self.measuredefs[0].quarterTempo or F(60)
        lastTimesig = self.measuredefs[0].timesig or (4, 4)
        s.append(m21tools.makeMetronomeMark(number=lasttempo))

        for measuredef in self.measuredefs:
            tempo = measuredef.quarterTempo or lasttempo
            if tempo != lasttempo:
                lasttempo = tempo
                s.append(m21tools.makeMetronomeMark(number=tempo))
            timesig = measuredef.timesig or lastTimesig
            lastTimesig = timesig
            num, den = timesig
            s.append(m21.meter.TimeSignature(f"{num}/{den}"))
            if measuredef.annotation:
                textExpression = m21tools.makeTextExpression(measuredef.annotation)
                s.append(textExpression)
            if fillMeasures:
                s.append(m21.note.Note(pitch=60, duration=m21.duration.Duration(measuredef.numberOfBeats())))
            else:
                s.append(m21.note.Rest(duration=m21.duration.Duration(measuredef.numberOfBeats())))
        score = m21.stream.Score()
        score.append(s)
        if self.title:
            m21tools.scoreSetMetadata(score, title=self.title)
        return score

    def write(self, path: U[str, Path]) -> None:
        """
        Write this as .xml, .ly or render as .pdf or .ly

        Args:
            path: the path of the written file. The extension determines the format
        """
        m21score = self.asMusic21(fillMeasures=True)
        path = Path(path)
        if path.suffix == ".xml":
            m21score.write("xml", path)
        elif path.suffix == ".pdf":
            m21tools.writePdf(m21score, str(path), 'musicxml.pdf')
        elif path.suffix == ".ly":
            m21tools.saveLily(m21score, path)
        else:
            raise ValueError(f"Extension {path.suffix} not supported, should be one of .xml, .pdf, .ly")


def _canBeMerged(n0: Notation, n1: Notation) -> bool:
    """
    Returns True if n0 and n1 can me merged into one Notation
    with a regular duration

    NB: a regular duration is one which can be represented via
    one notation (a quarter, a half, a dotted 8th, a double dotted 16th are
    all regular durations, 5/8 of a quarter is not --which is a shame)
    """
    if (not n0.tiedNext or
            not n1.tiedPrev or
            n0.durRatios != n1.durRatios or
            n0.pitches != n1.pitches
            ):
        return False
    # durRatios are the same so check if durations would sum to a regular duration
    dur0 = n0.symbolicDuration()
    dur1 = n1.symbolicDuration()
    sumdur = dur0 + dur1
    num, den = sumdur.numerator, sumdur.denominator
    return den < 64 and num in {1, 2, 3, 7}


def mergeNotationsIfPossible(notations: List[Notation]) -> List[Notation]:
    """
    If two consecutive notations have same .durRatio and merging them
    would result in a regular note, merge them.

    8 + 8 = q
    q + 8 = q·
    q + q = h
    16 + 16 = 8

    In general:

    1/x + 1/x     2/x
    2/x + 1/x     3/x  (and viceversa)
    3/x + 1/x     4/x  (and viceversa)
    6/x + 1/x     7/x  (and viceversa)
    """
    assert len(notations) > 1
    out = [notations[0]]
    for n1 in notations[1:]:
        if _canBeMerged(out[-1], n1):
            out[-1] = out[-1].mergeWith(n1)
        else:
            out.append(n1)
    assert len(out) <= len(notations)
    assert sum(n.duration for n in out) == sum(n.duration for n in notations)
    return out


@dataclasses.dataclass
class DurationGroup:
    """
    A DurationGroup groups together Notations under time modifier (a tuple)
    A DurationGroup consists of a sequence of Notations or DurationGroups,
    allowing to define nested tuples or beats
    """
    durRatio: Tuple[int, int]
    items: List[U[Notation, 'DurationGroup']]

    def symbolicDuration(self) -> F:
        """
        The symbolic duration of this Notation. This represents
        the notated figure (1=quarter, 1/2=eighth note, 1/4=16th note, etc)
        """
        return sum(item.symbolicDuration() for item in self.items)

    def __repr__(self):
        parts = [f"DurationGroup({self.durRatio[0]}/{self.durRatio[1]}, "]
        for item in self.items:
            if isinstance(item, Notation):
                parts.append("  " + str(item))
            else:
                s = str(item)
                for line in s.splitlines():
                    parts.append("  " + line)
        parts.append(")")
        return "\n".join(parts)

    def mergeNotations(self) -> DurationGroup:
        i0 = self.items[0]
        out = [i0 if isinstance(i0, Notation) else i0.mergeNotations()]
        for i1 in self.items[1:]:
            if isinstance(out[-1], Notation) and isinstance(i1, Notation):
                if _canBeMerged(out[-1], i1):
                    out[-1] = out[-1].mergeWith(i1)
                else:
                    out.append(i1)
            elif isinstance(i1, Notation):
                assert isinstance(out[-1], DurationGroup)
                out.append(i1)
            else:
                assert isinstance(out[-1], Notation) and isinstance(i1, DurationGroup)
                out.append(i1.mergeNotations())
        return DurationGroup(durRatio=self.durRatio, items=out)


def _splitGroupAt(group: List[Notation], splitPoints: List[F]
                  ) -> List[List[Notation]]:
    subgroups = defaultdict(lambda:list())
    for n in group:
        for i, splitPoint in enumerate(splitPoints):
            if n.end <= splitPoint:
                subgroups[i-1].append(n)
                break
        else:
            subgroups[len(splitPoints)].append(n)
    return [subgroup for subgroup in subgroups.values() if subgroup]

