from __future__ import annotations

import os
import shutil
from pathlib import Path
from dataclasses import dataclass, replace as _dataclassReplace
from bisect import bisect

import emlib.img
import emlib.misc
import emlib.textlib
import music21 as m21
from maelzel.music import m21tools
from maelzel.rational import Rat as F


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple, Optional as Opt, List, Union as U, Iterator as Iter
    timesig_t = Tuple[int, int]
    number_t = U[int, float, F]


def asF(x: number_t) -> F:
    if isinstance(x, F):
        return x
    elif hasattr(x, 'numerator'):
        return F(x.numerator, x.denominator)
    return F(x)


def _parseTimesig(s: str) -> Tuple[int, int]:
    try:
        num, den = s.split("/")
    except ValueError:
        raise ValueError(f"Could not parse timesig: {s}")
    return int(num), int(den)

def _asTimesig(t: U[str, timesig_t]) -> timesig_t:
    if isinstance(t, tuple):
        assert len(t) == 2
        return t
    elif isinstance(t, str):
        return _parseTimesig(t)
    else:
        raise TypeError(f"Expected a tuple (5, 8) or a string '5/8', got {t}, {type(t)}")


@dataclass
class _ScoreLine:
    measureNum: Opt[int]
    timesig: Opt[timesig_t]
    tempo: Opt[float]
    label: str = ''


def _parseScoreStructLine(line: str) -> _ScoreLine:
    """
    parse a line of a ScoreStruct definition

    Args:
        line: a line of the format [measureNum, ] timesig [, tempo]

    Returns:
        a tuple (measureNum, timesig, tempo), where only timesig
        is required
    """

    parts = [_.strip() for _ in line.split(",")]
    lenparts = len(parts)
    label = ''
    if lenparts == 1:
        timesigS = parts[0]
        measure = None
        tempo = None
    elif lenparts == 2:
        if "/" in parts[0]:
            timesigS, tempoS = parts
            measure = None
            tempo = float(tempoS)
        else:
            measureNumS, timesigS = parts
            measure = int(measureNumS)
            tempo = None
    elif lenparts == 3:
        if "/" not in parts[0]:
            measureNumS, timesigS, tempoS = [_.strip() for _ in parts]
            measure = int(measureNumS) if measureNumS else None
        else:
            measure = None
            timesigS, tempoS, label = [_.strip() for _ in parts]
        tempo = float(tempoS) if tempoS else None
    elif lenparts == 4:
        measureNumS, timesigS, tempoS, label = [_.strip() for _ in parts]
        measure = int(measureNumS) if measureNumS else None
        tempo = float(tempoS) if tempoS else None
    else:
        raise ValueError(f"Parsing error at line {line}")
    timesig = _parseTimesig(timesigS) if timesigS else None
    if label:
        label = label.replace('"', '')
    return _ScoreLine(measureNum=measure, timesig=timesig, tempo=tempo, label=label)


@dataclass
class MeasureDef:
    """
    A measure definition.

    It does not hold any other data (notes) but the information
    of the measure itself, to be used inside a ScoreStruct
    """
    timesig: timesig_t
    quarterTempo: F
    annotation: str = ""
    timesigInherited: bool = False
    tempoInherited: bool = False
    barline: str = ""

    def __post_init__(self):
        assert isinstance(self.timesig, tuple) and len(self.timesig) == 2
        assert all(isinstance(i, int) for i in self.timesig)
        self.quarterTempo = asF(self.quarterTempo)

    def numberOfBeats(self) -> F:
        """
        The duration of this measure in beats

        Timesig  numberOfBeats
        ----------------------
        4/4      4
        5/8      2.5
        3/8      1.5
        """
        return 4 * F(*self.timesig)

    def duration(self) -> F:
        """
        The duration of this measure, in seconds
        """
        if self.quarterTempo is None or self.timesig is None:
            raise ValueError("MeasureDef not fully defined")
        return self.numberOfBeats() * (F(60)/self.quarterTempo)

    def clone(self, **kws):
        return _dataclassReplace(self, **kws)


@dataclass
class ScoreLocation:
    measureNum: int
    beat: F = 0

    def __repr__(self):
        return f"ScoreLocation(measureNum={self.measureNum}, beat={float(self.beat):.4g})"

    def __post_init__(self):
        assert isinstance(self.measureNum, int)
        self.beat = asF(self.beat)

    def __iter__(self):
        return iter((self.measureNum, self.beat))


class ScoreStruct:
    def __init__(self, timesig:U[timesig_t, str]=None, quarterTempo:int=None,
                 endless=True, autoextend=False):
        """
        A ScoreStruct holds the structure of a score but no content

        A ScoreStruct consists of some metadata and a list of MeasureDefs,
        where each MeasureDef defines the properties of the measure at the given
        index. If a ScoreStruct is marked as endless, it is possible to query
        it (ask for a MeasureDef, convert beats to time, etc.) outside of the defined
        measures.

        Args:
            endless: mark this ScoreStruct as endless

            autoextend: this is only valid if the score is marked as endless. If True,
              querying this ScoreStruct outside its defined boundaries will create the
              necessary MeasureDefs to cover the point in question. In concrete,
              if a ScoreStruct has only one MeasureDef, .getMeasureDef(3) will create
              three MeasureDefs, cloning the first MeasureDef, and return the last one.

        Example::
            # create an endless score with a given time signature
            >>> s = ScoreStruct(endless=True)
            >>> s.addMeasure((4, 4), quarterTempo=72)
            # this is the same as:
            >>> s = ScoreStruct.fromTimesig((4, 4), 72)

            # Create the beginning of Sacre
            >>> s = ScoreStruct()
            >>> s.addMeasure((4, 4), 50)
            >>> s.addMeasure((3, 4))
            >>> s.addMeasure((4, 4))
            >>> s.addMeasure((2, 4))
            >>> s.addMeasure((3, 4), numMeasures=2)

        A ScoreStruct can also be created from a string (see .fromString to learn
        about the format), which can be read from a file
        """
        self.measuredefs: List[MeasureDef] = []

        self.title = None
        self.endless = endless
        self.autoextend = autoextend
        self._modified = True
        self._offsetsIndex: List[F] = []
        if timesig or quarterTempo:
            if not timesig:
                timesig = (4, 4)
            elif not quarterTempo:
                quarterTempo = 60
            self.addMeasure(timesig, quarterTempo=quarterTempo)

    @staticmethod
    def fromString(s: str, initialTempo=60, initialTimeSignature=(4, 4), endless=False
                   ) -> ScoreStruct:
        """
        Args:
            s: the score as string
            initialTempo: the initial tempo, for the case where the initial measure/s
                do not include a tempo
            initialTimeSignature: the initial time signature
            endless: if True, make this ScoreStruct endless. The same can be achieved
                by ending the score with the line '...'

        Format::

            measureNum, timeSig, tempo

        * Tempo refers always to a quarter note
        * Any value can be left out: , 5/8,
        * The measure number and/or the tempo can both be left out::

            These are all the same:
            1, 5/8, 63
            5/8, 63
            5/8

        * measure numbers start at 0
        * comments start with `#` and last until the end of the line
        * A line with a single "." repeats the last defined measure
        * A score ending with the line ... is an endless score

        Example::

            0, 4/4, 60, "mark A"
            ,3/4,80     # Assumes measureNum=1
            10, 5/8, 120
            30,,
            .
            .      # last measure (inclusive, the score will have 33 measures)
        """
        tempo = initialTempo
        timesig = initialTimeSignature
        measureNum = -1
        lines = emlib.textlib.splitAndStripLines(s)
        if lines[-1].strip() == '...':
            endless = True
            lines = lines[:-1]
        struct = ScoreStruct(endless=endless)

        def lineStrip(l:str) -> str:
            if "#" in l:
                l = l.split("#")[0]
            return l.strip()

        for i, line0 in enumerate(lines):
            line = lineStrip(line0)
            if not line:
                continue

            if line == ".":
                assert len(struct.measuredefs) > 0
                lastmeas = struct.measuredefs[-1]
                struct.addMeasure()
                measureNum += 1
                continue

            mdef = _parseScoreStructLine(line)
            if mdef.measureNum is None:
                mdef.measureNum = measureNum + 1
            else:
                assert mdef.measureNum > measureNum
                struct.addMeasure(numMeasures=mdef.measureNum - measureNum - 1)

            struct.addMeasure(timesig=mdef.timesig, quarterTempo=mdef.tempo,
                              annotation=mdef.label)
            measureNum = mdef.measureNum

        return struct

    @staticmethod
    def fromTimesig(timesig: U[timesig_t, str], quarterTempo=60, numMeasures:int=None
                    ) -> ScoreStruct:
        """
        Creates a ScoreStruct from a time signature and tempo.

        If numMeasures is given, the resulting ScoreStruct will
        have as many measures defined and will be finite. Otherwise
        the ScoreStruct will be flagged as endless

        Args:
            timesig: the time signature, a tuple (num, den)
            quarterTempo: the tempo of a quarter note
            numMeasures: the number of measures of this score. If None
                is given, the ScoreStruct will be endless

        Returns:
            a ScoreStruct

        """
        timesig = _asTimesig(timesig)
        if numMeasures is None:
            out = ScoreStruct(endless=True)
            out.addMeasure(timesig=timesig, quarterTempo=quarterTempo)
        else:
            out = ScoreStruct(endless=False)
            out.addMeasure(timesig, quarterTempo, numMeasures=numMeasures)
        return out

    def copy(self) -> ScoreStruct:
        """
        Create a copy of this ScoreSturct
        """
        s = ScoreStruct(endless=self.endless, autoextend=self.autoextend)
        s.title = self.title
        s.measuredefs = self.measuredefs.copy()
        return s

    def numDefinedMeasures(self) -> int:
        """
        Returns the number of defined measures

        (independently of this ScoreStruct being endless or not)
        """
        return len(self.measuredefs)

    def __len__(self):
        """Returns the number of defined measures"""
        return len(self.measuredefs)

    def getMeasureDef(self, idx:int) -> MeasureDef:
        """
        Returns the MeasureDef at the given index.

        If the scorestruct is endless and the index is outside of the defined
        range, the returned MeasureDef will be the last defined MeasureDef.
        If this ScoreStruct was created with autoextend=True, any query
        outside of the defined range of measures will extend the score
        to that point
        """
        if idx < len(self.measuredefs):
            return self.measuredefs[idx]
        # outside of defined measures
        if not self.endless:
            raise IndexError(f"index {idx} out of range. The score has "
                             f"{len(self.measuredefs)} measures defined")
        if not self.autoextend:
            # we are "outside" the defined score
            m = self.measuredefs[-1]
            if m.annotation:
                m = m.clone(annotation='', tempoInherited=True, timesigInherited=True)
            return m

        for n in range(len(self.measuredefs)-1, idx):
            self.addMeasure()

    def __getitem__(self, item:int) -> MeasureDef:
        if isinstance(item, int):
            return self.getMeasureDef(item)
        print(item, dir(item))

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
            >>> s = ScoreStruct()
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
        self._modified = True
        if numMeasures > 1:
            self.addMeasure(numMeasures=numMeasures-1)

    def __str__(self) -> str:
        lines = ["ScoreStruct(["]
        for m in self.measuredefs:
            lines.append("    " + str(m))
        lines.append("])")
        return "\n".join(lines)

    def markAsModified(self, value=True) -> None:
        """
        Call this when a MeasureDef inside this ScoreStruct is modified

        By marking it as modified any internal cache is invalidated
        """
        self._modified = value

    def _update(self) -> None:
        if not self._modified:
            return
        now = F(0)
        starts = []
        for mdef in self.measuredefs:
            starts.append(now)
            now += mdef.duration()
        self._modified = False
        self._offsetsIndex = starts

    def locationToTime(self, measure: int, beat:number_t=F(0)) -> F:
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
                lastTime = self.locationToTime(last)
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

    def tempoAtTime(self, time: number_t) -> F:
        loc = self.timeToLocation(time)
        measuredef = self.getMeasureDef(loc.measureNum)
        return measuredef.quarterTempo

    def timeToLocation(self, time: number_t) -> Opt[ScoreLocation]:
        """
        Find the location in score corresponding to the given time in seconds

        Args:
            time: the time in seconds

        Returns:
            a ScoreLocation (.measureNum, .beat). If the score is not
             endless and the time is outside the score, None is returned
        """
        if not self.measuredefs:
            raise IndexError("This ScoreStruct is empty")

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

    def beatToLocation(self, beat:number_t) -> Opt[ScoreLocation]:
        """
        Return the location in score corresponding to the given beat

        The beat is the time-offset in quarter-notes. Given a beat
        (in quarter-notes), return the score location
        (measure, beat offset within the measure). Tempo does not
        play any role within this calculation.

        See also .locationToBeat, which performs the opposite operation

        Example:
            Given the following score: 4/4, 3/4, 4/4

            input       output
            ---------------------------------
            4           ScoreLocation(1, 0)
            5.5         ScoreLocation(1, 1.5)
            8           ScoreLocation(2, 1.0)
        """
        assert len(self.measuredefs) >= 1
        numMeasures = 0
        rest = asF(beat)
        for i, mdef in enumerate(self.measuredefs):
            numBeats = mdef.numberOfBeats()
            if rest < numBeats:
                return ScoreLocation(i, rest)
            rest -= numBeats
            numMeasures += 1
        # we are at the end of the defined measure, but we did not find beat yet.
        if not self.endless:
            return None
        beatsPerMeasures = self.measuredefs[-1].numberOfBeats()
        numMeasures += int(rest / beatsPerMeasures)
        restBeats = rest % beatsPerMeasures
        return ScoreLocation(numMeasures, restBeats)

    def beatToTime(self, beat: U[number_t, Tuple[int, number_t]]) -> F:
        """
        Convert beat-time to real-time

        Args:
            beat: the beat location, as beat-time (in quarter-notes) or as
                a tuple (measure index, beat inside measure)
        """
        if isinstance(beat, tuple):
            loc = beat
        else:
            loc = self.beatToLocation(beat)
        return self.locationToTime(*loc)

    def timeToBeat(self, t: number_t) -> F:
        loc = self.timeToLocation(t)
        beat = self.locationToBeat(loc.measureNum, loc.beat)
        return beat

    def iterMeasureDefs(self) -> Iter[MeasureDef]:
        """
        Iterate over all measure definitions in this ScoreStruct.
        If it is marked as endless then the last defined measure
        will be returned indefinitely.
        """
        for mdef in self.measuredefs:
            yield mdef
        if not self.endless:
            raise StopIteration
        lastmdef = self.measuredefs[-1]
        while True:
            yield lastmdef

    def __iter__(self) -> Iter[MeasureDef]:
        return self.iterMeasureDefs()

    def locationToBeat(self, measure:int, beat:number_t=F(0)) -> F:
        """
        Returns the number of quarter notes up to the given location

        This value is independent of any tempo given.
        """
        if not self.endless and measure > self.numDefinedMeasures():
            raise ValueError(f"This scorestruct has {self.numDefinedMeasures()} and is not"
                             f"marked as endless. Measure {measure} is out of scope")
        accum = F(0)
        for i, mdef in enumerate(self.iterMeasureDefs()):
            if i < measure:
                accum += mdef.numberOfBeats()
            else:
                if beat > mdef.numberOfBeats():
                    raise ValueError(f"beat {beat} outside of measure {i}: {mdef}")
                accum += asF(beat)
                break
        return accum

    def timeDifference(self,
                       start:U[number_t, Tuple[int, number_t]],
                       end:U[number_t, Tuple[int, number_t]]
                       ) -> F:
        """
        Returns the elapsed time between two score locations.

        Args:
            start: the start location, as a beat or as a tuple (measureIndex, beatOffset)
            end: the end location, as a beat or as a tuple (measureIndex, beatOffset)

        Returns:
            the elapsed time, as a Fraction

        """
        return self.beatToTime(end) - self.beatToTime(start)

    def timeDifferenceBetweenBeats(self, startBeat: number_t, endBeat: number_t) -> F:
        return self.beatToTime(endBeat) - self.beatToTime(startBeat)

    def beatDifference(self, start:Tuple[int,F], end:Tuple[int, F]) -> F:
        """
        difference in beats between the two score locations

        Args:
            start: the start location, a tuple (measureIndex, beatOffset)
            end: the end location, a tuple (measureIndex, beatOffset)

        Returns:
            the distance between the two locations, in beats
        """
        return self.locationToBeat(*end) - self.locationToBeat(*start)

    def show(self) -> None:
        """
        Render and show this ScoreStruct
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

    def hasUniqueTempo(self) -> bool:
        t = self.measuredefs[0].quarterTempo
        return all(m.quarterTempo == t for m in self.measuredefs)

    def __repr__(self) -> str:
        if self.hasUniqueTempo() and self.hasUniqueTimesig():
            m0 = self.measuredefs[0]
            return f'ScoreStruct(tempo={m0.quarterTempo}, timesig={m0.timesig})'
        else:
            tempo = -1
            parts = []
            maxdefs = 10
            for m in self.measuredefs[:maxdefs]:
                num, den = m.timesig
                if m.quarterTempo != tempo:
                    tempo = m.quarterTempo
                    parts.append(f"{num}/{den}@{tempo}")
                else:
                    parts.append(f"{num}/{den}")
            s = ", ".join(parts)
            if len(self.measuredefs) > maxdefs:
                s += " …"
            return f"ScoreStruct([{s}])"

    def _repr_html_(self) -> str:
        colnames = ['Meas. Index', 'Timesig', 'Tempo (quarter note)', 'Label']

        parts = [f'<h5><strong>ScoreStruct<strong></strong></h5>']
        tempo = -1
        rows = []
        for i, m in enumerate(self.measuredefs):
            num, den = m.timesig
            if m.quarterTempo != tempo:
                tempo = m.quarterTempo
                tempostr = ("%.3f" % tempo).rstrip("0").rstrip(".")
            else:
                tempostr = ""
            row = (str(i), f"{num}/{den}", tempostr, m.annotation or "")
            rows.append(row)
        if self.endless:
            rows.append(("...", "", "", ""))
        rowstyle = 'font-size: small;'
        htmltable = emlib.misc.html_table(rows, colnames, rowstyles=[rowstyle]*4)
        parts.append(htmltable)
        return "".join(parts)

    def render(self, outfile: str) -> None:
        fmt = os.path.splitext(outfile)[1]
        m21format = 'xml' + fmt
        outfile2 = self.asMusic21().write(m21format, outfile)
        if fmt == ".png":
            emlib.img.pngRemoveTransparency(outfile2)
        assert os.path.exists(outfile2)
        if os.path.exists(outfile):
            os.remove(outfile)
        shutil.move(outfile2, outfile)

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
        s.append(m21tools.makeMetronomeMark(number=float(lasttempo)))

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
        score.insert(0, s)
        if self.title:
            m21tools.scoreSetMetadata(score, title=self.title)
        return score

    def setTempo(self, quarterTempo: float, measureNum:int=0) -> None:
        """
        Set the tempo of the given measure, until the next tempo change

        Args:
            quarterTempo: the new tempo
            measureNum: the first measure to modify

        """
        if measureNum > len(self):
            raise IndexError(f"Index {measureNum} out of rage; this ScoreStruct has only "
                             f"{len(self)} measures defined")
        mdef = self.measuredefs[measureNum]
        mdef.quarterTempo = quarterTempo
        mdef.tempoInherited = False
        for m in self.measuredefs[measureNum+1:]:
            if m.tempoInherited:
                m.quarterTempo = quarterTempo
            else:
                break

    def hasUniqueTimesig(self) -> bool:
        lastTimesig = self.measuredefs[0].timesig
        for m in self.measuredefs:
            if m.timesig != lastTimesig:
                return False
        return True

    def write(self, path: U[str, Path]) -> None:
        """
        Write this as .xml, .ly or render as .pdf or .png

        Args:
            path: the path of the written file. The extension determines the format
        """
        m21score = self.asMusic21(fillMeasures=True)
        path = Path(path)
        if path.suffix == ".xml":
            m21score.write("xml", path)
        elif path.suffix == ".pdf":
            m21tools.writePdf(m21score, str(path), 'musicxml.pdf')
        elif path.suffix == ".png":
            m21score.write("musicxml.png", path)
        elif path.suffix == ".ly":
            m21tools.saveLily(m21score, path.as_posix())
        else:
            raise ValueError(f"Extension {path.suffix} not supported, "
                             f"should be one of .xml, .pdf, .png or .ly")



