from __future__ import annotations

import copy
from pathlib import Path
from dataclasses import dataclass, replace as _dataclass_replace
from bisect import bisect
import sys

import emlib.img
import emlib.misc
from emlib import iterlib
import emlib.textlib
import music21 as m21
from numbers import Rational
from maelzel.common import F, asF

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterator, Sequence, Union
    timesig_t = tuple[int, int]
    number_t = Union[float, Rational]
    import maelzel.core
    from maelzel import scoring


__all__ = (
    'asF',
    'ScoreStruct',
    'MeasureDef'
)

_unicodeFractions = {
    (3, 16): '³⁄₁₆',
    (5, 16): '⁵⁄₁₆',
    (2, 8): '²⁄₈',
    (3, 8): '⅜',
    (4, 8): '⁴⁄₈',
    (5, 8): '⅝',
    (6, 8): '⁶⁄₈',
    (7, 8): '⅞',
    (2, 4): '²⁄₄',
    (3, 4): '¾',
    (4, 4): '⁴⁄₄',
    (5, 4): '⁵⁄₄',
    (6, 4): '⁶⁄₄'
}


class TimeSignature:
    def __init__(self, *parts: tuple[int, int]):
        self.parts = parts
        minden = max(den for num, den in parts)
        numerators = [num * minden // den for num, den in parts]
        self.raw = (int(sum(numerators)), minden)
        self.normalizedParts = [(num, minden) for num in numerators]

    @property
    def numerator(self) -> int:
        return self.raw[0]

    @property
    def denominator(self) -> int:
        return self.raw[1]

    def __repr__(self):
        if len(self.parts) == 1:
            num, den = self.parts[0]
            return f"TimeSignature({num}/{den})"
        elif all(den == self.parts[0][1] for num, den in self.parts):
            nums = "+".join(str(p[0]) for p in self.parts)
            return f"TimeSignature({nums}/{self.parts[0][1]})"
        else:
            parts = '+'.join(f"{n}/{d}" for n, d in self.parts)
            return f"TimeSignature({parts})"

    @classmethod
    def parse(cls, timesig) -> TimeSignature:
        if isinstance(timesig, tuple):
            if all(isinstance(_, tuple) for _ in timesig):
                # ((3, 8), (3, 8), (2, 8))
                return TimeSignature(*timesig)
            elif len(timesig) == 2 and isinstance(timesig[1], int):
                num, den = timesig
                if isinstance(num, tuple):
                    # ((3, 3, 2), 8)
                    parts = [(_, den) for _ in num]
                    return TimeSignature(*parts)
                else:
                    assert isinstance(num, int)
                    return TimeSignature((num, den))
            else:
                raise ValueError(f"Cannot parse timesignature: {timesig}")
        elif isinstance(timesig, str):
            # 3/4, 3+3+2/8 or 3/8+3/8+2/8
            parts = timesig.count("+") + 1
            if parts == 1:
                assert "/" in timesig
                numstr, denstr = timesig.split("/")
                return TimeSignature((int(numstr), int(denstr)))
            else:
                if timesig.count("/") == 1:
                    numstr, denstr = timesig.split("/")
                    den = int(denstr)
                    parts = [(int(num), den) for num in numstr.split("+")]
                    return TimeSignature(*parts)
                else:
                    parts = [tuple(map(int, p.split("/"))) for p in timesig.split("+")]
                    return TimeSignature(*parts)
        else:
            raise TypeError(f"Expected a str or a tuple, got {timesig}")


def _parseTimesig(s: str) -> tuple[int, int]:
    try:
        num, den = s.split("/")
    except ValueError:
        raise ValueError(f"Could not parse timesig: {s}")
    if "+" in num:
        parts = num.split("+")
        # Compound timesigs are not supported, we just add
        num = sum(int(p) for p in parts)
        return num, int(den)
    return int(num), int(den)


def _asTimesig(t: str | timesig_t) -> timesig_t:
    if isinstance(t, tuple):
        assert len(t) == 2
        return t
    elif isinstance(t, str):
        return _parseTimesig(t)
    else:
        raise TypeError(f"Expected a tuple (5, 8) or a string '5/8', got {t}, {type(t)}")


@dataclass
class _ScoreLine:
    measureIndex: int | None
    timesig: timesig_t | None
    tempo: float | None
    label: str = ''


@dataclass
class RehearsalMark:
    text: str
    box: str = 'square'


class KeySignature:
    def __init__(self, fifths: int, mode='major'):
        self.fifths = fifths
        self.mode = mode


def _parseScoreStructLine(line: str) -> _ScoreLine:
    """
    parse a line of a ScoreStruct definition

    Args:
        line: a line of the format [measureIndex, ] timesig [, tempo]

    Returns:
        a tuple (measureIndex, timesig, tempo), where only timesig
        is required
    """
    line = line.strip()
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
            try:
                tempo = float(tempoS)
            except ValueError:
                raise ValueError(f"Could not parse the tempo ({tempoS}) as a number (line: {line})")
        else:
            measureIndexS, timesigS = parts
            try:
                measure = int(measureIndexS)
            except ValueError:
                raise ValueError(f"Could not parse the measure index '{measureIndexS}' while parsing line: '{line}'")
            tempo = None
    elif lenparts == 3:
        if "/" not in parts[0]:
            measureIndexS, timesigS, tempoS = [_.strip() for _ in parts]
            measure = int(measureIndexS) if measureIndexS else None
        else:
            measure = None
            timesigS, tempoS, label = [_.strip() for _ in parts]
        tempo = float(tempoS) if tempoS else None
    elif lenparts == 4:
        measureIndexS, timesigS, tempoS, label = [_.strip() for _ in parts]
        measure = int(measureIndexS) if measureIndexS else None
        tempo = float(tempoS) if tempoS else None
    else:
        raise ValueError(f"Parsing error at line {line}")
    timesig = _parseTimesig(timesigS) if timesigS else None
    if label:
        label = label.replace('"', '')
    return _ScoreLine(measureIndex=measure, timesig=timesig, tempo=tempo, label=label)


@dataclass
class MeasureDef:
    """
    A measure definition.

    It does not hold any other data (notes) but the information
    of the measure itself, to be used inside a ScoreStruct

    """
    timesig: timesig_t
    """The time signature, a tuple (int, int)"""

    quarterTempo: F | int
    """The beats per minute for the quarter note"""

    annotation: str = ""
    """A text annotation for the measure itself (a measure annotation)"""

    timesigInherited: bool = False
    """Is the timesignature inherited from a previous measure?"""

    tempoInherited: bool = False
    """Is the tempo inherited from a previous measure?"""

    barline: str = ""
    """The kind of barline (one of 'normal', 'double', 'solid', ''='default')"""

    subdivisionStructure: list[int] | None = None
    """for irregular measures indicates the grouping of beats (for 7/8 could be [2, 3, 2])"""

    rehearsalMark: RehearsalMark | None = None
    """A rehearsal mark for this measure"""

    keySignature: KeySignature | None = None

    properties: dict | None = None
    """A place to put any attribute for this measure"""

    maxEighthTempo: F = F(48)

    durationBeats: F = F(0)
    """The duration of this measure, in beats (quarternotes)"""

    durationSecs: F = F(0)
    """The duration of this measure in seconds"""

    def __post_init__(self):
        assert isinstance(self.timesig, tuple) and len(self.timesig) == 2
        assert all(isinstance(i, int) for i in self.timesig), f"Expected a tuple of ints, got {self.timesig}"
        if self.subdivisionStructure:
            assert isinstance(self.subdivisionStructure, (tuple, list))
            assert all(isinstance(part, int) for part in self.subdivisionStructure)
        self.quarterTempo = asF(self.quarterTempo)
        n, d = self.timesig
        self.durationBeats = F(4*n, d)
        self.durationSecs = self.durationBeats * (F(60) / self.quarterTempo)

    def __hash__(self) -> int:
        return hash((self.timesig, self.quarterTempo, self.annotation))

    def subdivisions(self) -> list[F]:
        """
        Returns a list of the subdivisions of this measure.

        A subdivision is a duration, in quarters.

        Returns:
            a list of durations which sum up to the duration of this measure

        Example
        -------

            >>> MeasureDef(timesig=(3, 4), quarterTempo=60).subdivisions()
            [1, 1, 1]
            >>> MeasureDef(timesig=(3, 8), quarterTempo=60).subdivisions()
            [0.5, 0.5, 0.5]
            >>> MeasureDef(timesig=(7, 8), quarterTempo=40).subdivisions()
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            >>> MeasureDef(timesig=(7, 8), quarterTempo=150).subdivisions()
            [1.0, 1.0, 1.5]
            >>> MeasureDef((7, 8), quarterTempo=150, subdivisionStructure=[2, 3, 2]).subdivisions()
            [1, 1.5, 1]
        """
        return measureBeatDurations(timesig=self.timesig, quarterTempo=self.quarterTempo,
                                    subdivisionStructure=self.subdivisionStructure,
                                    maxEighthTempo=self.maxEighthTempo)

    def clone(self, **kws):
        """Clone this MeasureDef with modified attributes"""
        return _dataclass_replace(self, **kws)

    def timesigRepr(self) -> str:
        """
        Returns a string representation of this measure's time signature
        Returns:

        """
        num, den = self.timesig
        if self.subdivisionStructure:
            # 3+3+2/8
            return f'{num}/{den} ({"+".join(str(sub) for sub in self.subdivisionStructure)})'
        return f'{num}/{den}'

    def setBarline(self, barstyle: str) -> None:
        """

        Args:
            barstyle: a valid linestyle for the barline. One of single, double, solid,
                dotted or dashed.

        Returns:

        """
        assert barstyle in {'single', 'final', 'double', 'solid', 'dotted', 'dashed', 'tick', 'short',
                            'double-thin', 'none'}, \
            f'Unknown barstyle: {barstyle}'
        self.barline = barstyle


def _inferSubdivisions(num: int, den: int, quarterTempo
                       ) -> list[int]:
    subdivs = []
    while num > 3:
        subdivs.append(2)
        num -= 2
    if num:
        subdivs.append(num)
    return subdivs


def measureQuarterDuration(timesig: timesig_t) -> F:
    """
    The duration in quarter notes of a measure according to its time signature

    Args:
        timesig: a tuple (num, den)

    Returns:
        the duration in quarter notes

    Examples::

        >>> measureQuarterDuration((3,4))
        3

        >>> measureQuarterDuration((5, 8))
        2.5

    """
    num, den = timesig
    quarterDuration = F(num)/den * 4
    return quarterDuration


def measureBeatDurations(timesig: timesig_t,
                         quarterTempo: F,
                         maxEighthTempo: Rational | float = 48,
                         subdivisionStructure: list[int] = None
                         ) -> list[F]:
    """

    Args:
        timesig: the timesignature of the measure
        quarterTempo: the tempo for a quarter note
        maxEighthTempo: max quarter tempo to divide a measure like 5/8 in all
            eighth notes instead of, for example, 2+2+1
        subdivisionStructure: if given, a list of subdivision lengths. For example,
            a 5/8 measure could have a subdivision structure of [2, 3] or [3, 2]

    Returns:
        a list of durations, as Fraction

    ::

        4/8 -> [1, 1]
        2/4 -> [1, 1]
        3/4 -> [1, 1, 1]
        5/8 -> [1, 1, 0.5]
        5/16 -> [0.5, 0.5, 0.25]

    """
    quarterTempo = asF(quarterTempo)
    quarters = measureQuarterDuration(timesig)
    num, den = timesig
    if subdivisionStructure:
        return [F(num, den // 4) for num in subdivisionStructure]
    elif den == 4 or den == 2:
        return [F(1)] * quarters.numerator
    elif den == 8:
        if quarterTempo <= maxEighthTempo:
            # render all beats as 1/8 notes
            return [F(1, 2)]*num
        subdivstruct = _inferSubdivisions(num=num, den=den, quarterTempo=quarterTempo)
        return [F(num, den // 4) for num in subdivstruct]
    elif den == 16:
        if num % 2 == 0:
            timesig = (num//2, 8)
            return measureBeatDurations(timesig, quarterTempo=quarterTempo)
        beats = [F(1, 2)] * (num//2)
        beats.append(F(1, 4))
        return beats
    else:
        raise ValueError(f"Invalid time signature: {timesig}")


def measureBeatOffsets(timesig: timesig_t,
                       quarterTempo: F | int,
                       subdivisionStructure: list[int] = None
                       ) -> list[F]:
    """
    Returns a list with the offsets of all beats in measure.

    The last value refers to the offset of the end of the measure

    Args:
        timesig: the timesignature as a tuple (num, den)
        quarterTempo: the tempo correponding to a quarter note
        subdivisionStructure: if given, a list of subdivision lengths. For example,
            a 5/8 measure could have a subdivision structure of [2, 3] or [3, 2]

    Returns:
        a list of fractions representing the start time of each beat, plus the
        end time of the measure (== the start time of the next measure)

    Example::
        >>> measureBeatOffsets((5, 8), 60)
        [Fraction(0, 1), Fraction(1, 1), Fraction(2, 1), Fraction(5, 2)]
        # 0, 1, 2, 2.5
    """
    quarterTempo = asF(quarterTempo)
    beatDurations = measureBeatDurations(timesig, quarterTempo=quarterTempo,
                                         subdivisionStructure=subdivisionStructure)
    beatOffsets = [F(0)] + list(iterlib.partialsum(beatDurations))
    return beatOffsets


def _bisect(seq, value) -> int:
    l = len(seq)
    if l > 10:
        return bisect(seq, value)
    for i, x in enumerate(seq):
        if value < x:
            return i
    return l


class ScoreStruct:
    """
    A ScoreStruct holds the structure of a score but no content

    A ScoreStruct consists of some metadata and a list of :class:`MeasureDefs`,
    where each :class:`MeasureDef` defines the properties of the measure at the given
    index. If a ScoreStruct is marked as *endless*, it is possible to query
    it (convert beats to time, etc.) outside of the defined measures.

    The ScoreStruct class is used extensively within :py:mod:`maelzel.core` (see
    `scorestruct-and-maelzel-core`)

    Args:
        score: if given, a score definition as a string (see below for the format)
        timesig: time-signature. If no score is given, a timesig can be passed to
            define a basic scorestruct with a time signature and a default or
            given tempo
        quarterTempo: the tempo of a quarter note, if given
        endless: mark this ScoreStruct as endless. Defaults to True
        title: title metadata for the score, used when rendering
        composer: composer metadata for this score, used when rendering

    Example
    -------

    .. code-block:: python

        # create an endless score with a given time signature
        s = ScoreStruct(endless=True)
        s.addMeasure((4, 4), quarterTempo=72)

        # this is the same as:
        s = ScoreStruct.fromTimesig((4, 4), 72)

        # Create the beginning of Sacre
        s = ScoreStruct()
        s.addMeasure((4, 4), 50)
        s.addMeasure((3, 4))
        s.addMeasure((4, 4))
        s.addMeasure((2, 4))
        s.addMeasure((3, 4), numMeasures=2)

        # The same can be achieved via a score string:
        s = ScoreStruct(r'''
        4/4, 50
        3/4
        4/4
        2/4
        3/4
        .
        ''')

        # Or everything in one line:
        s = ScoreStruct('4/4, 50; 3/4; 4/4; 2/4; 3/4; 3/4 ')

    **Format**

    A definitions are divided by new line or by ;. Each line has the form::

        measureIndex, timeSig, tempo


    * Tempo refers always to a quarter note
    * Any value can be left out: , 5/8,
    * measure numbers start at 0
    * comments start with `#` and last until the end of the line
    * A line with a single "." repeats the last defined measure
    * A score ending with the line ... is an endless score

    The measure number and/or the tempo can both be left out. The following definitions are
    all the same::

        1, 5/8, 63
        5/8, 63
        5/8

    **Example**::

        0, 4/4, 60, "mark A"
        ,3/4,80     # Assumes measureIndex=1
        10, 5/8, 120
        30,,
        .
        .      # last measure (inclusive, the score will have 33 measures)

    """
    def __init__(self,
                 score: str = None,
                 timesig: timesig_t | str = None,
                 quarterTempo: int = None,
                 endless: bool = True,
                 title='',
                 composer=''):

        # holds the time offset (in seconds) of each measure
        self._timeOffsets: list[F] = []

        self._beatOffsets: list[F] = []

        # the quarternote duration of each measure
        self._quarternoteDurations: list[F] = []

        self._modified = True
        self._prevScoreStruct: ScoreStruct | None = None

        self._lastIndex = 0

        if score:
            if timesig or quarterTempo:
                raise ValueError("Either a score as string or a timesig / quarterTempo can be given"
                                 "but not both")
            s = ScoreStruct._parseScore(score)
            self.measuredefs = s.measuredefs
            self.endless = endless
        else:
            self.measuredefs: list[MeasureDef] = []
            self.endless = endless
            if timesig or quarterTempo:
                if not timesig:
                    timesig = (4, 4)
                elif not quarterTempo:
                    quarterTempo = 60
                self.addMeasure(timesig, quarterTempo=quarterTempo)

        self.title = title
        self.composer = composer

    def __hash__(self) -> int:
        hashes = [hash(x) for x in (self.title, self.endless)]
        hashes.extend(hash(mdef) for mdef in self.measuredefs)
        return hash(tuple(hashes))

    def __eq__(self, other: ScoreStruct) -> int:
        return hash(self) == hash(other)

    @staticmethod
    def _parseScore(s: str, initialTempo=60, initialTimeSignature=(4, 4), endless=False
                    ) -> ScoreStruct:
        """
        Create a ScoreStruct from a string definition

        Args:
            s: the score as string. See below for format
            initialTempo: the initial tempo, for the case where the initial measure/s
                do not include a tempo
            initialTimeSignature: the initial time signature
            endless: if True, make this ScoreStruct endless. The same can be achieved
                by ending the score with the line '...'

        **Format**

        A definitions are divided by new line or by ;. Each line has the form::

            measureIndex, timeSig, tempo


        * Tempo refers always to a quarter note
        * Any value can be left out: , 5/8,
        * measure numbers start at 0
        * comments start with `#` and last until the end of the line
        * A line with a single "." repeats the last defined measure
        * A score ending with the line ... is an endless score

        The measure number and/or the tempo can both be left out. The following definitions are
        all the same::

            1, 5/8, 63
            5/8, 63
            5/8

        **Example**::

            0, 4/4, 60, "mark A"
            ,3/4,80     # Assumes measureIndex=1
            4/4         # Assumes measureIndex=2, inherits tempo 80
            10, 5/8, 120
            12,,96      # At measureIndex 12, change tempo to 96
            30,,
            .
            .      # last measure (inclusive, the score will have 33 measures)
        10, 4/4 q=60 label='Mylabel'
        3/4 q=42
        20, q=60 label='foo'

        """
        tempo = initialTempo
        timesig = initialTimeSignature
        measureIndex = -1
        lines = emlib.textlib.splitAndStripLines(s, r'[\n;]')
        if lines[-1].strip() == '...':
            endless = True
            lines = lines[:-1]
        struct = ScoreStruct(endless=endless)

        def lineStrip(l: str) -> str:
            if "#" in l:
                l = l.split("#")[0]
            return l.strip()

        for i, line0 in enumerate(lines):
            line = lineStrip(line0)
            if not line:
                continue

            if line == ".":
                assert len(struct.measuredefs) > 0
                struct.addMeasure()
                measureIndex += 1
                continue

            mdef = _parseScoreStructLine(line)
            if mdef.measureIndex is None:
                mdef.measureIndex = measureIndex + 1
            else:
                assert mdef.measureIndex > measureIndex
                if mdef.measureIndex - measureIndex > 1:
                    struct.addMeasure(numMeasures=mdef.measureIndex - measureIndex - 1)

            struct.addMeasure(timesig=mdef.timesig, quarterTempo=mdef.tempo,
                              annotation=mdef.label)
            measureIndex = mdef.measureIndex
        
        return struct

    @staticmethod
    def fromTimesig(timesig: timesig_t | str, quarterTempo=60, numMeasures: int = None
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
        s = ScoreStruct(endless=self.endless, title=self.title, composer=self.composer)
        s.measuredefs = copy.deepcopy(self.measuredefs)
        s.markAsModified()
        return s

    def numDefinedMeasures(self) -> int:
        """
        Returns the number of defined measures

        (independently of this ScoreStruct being endless or not)
        """
        return len(self.measuredefs)

    def __len__(self):
        """
        Returns the number of defined measures (even if the score is defined as endless)
        """
        return len(self.measuredefs)

    def getMeasureDef(self, idx: int, extend=False) -> MeasureDef:
        """
        Returns the MeasureDef at the given index.

        Args:
            idx: the measure index (measures start at 0)

        If the scorestruct is endless and the index is outside of the defined
        range, the returned MeasureDef will be the last defined MeasureDef.

        The same result can be achieved via ``__getitem__``

        Example
        -------

            >>> from maelzel.scorestruct import ScoreStruct
            >>> s = ScoreStruct(r'''
            ... 4/4, 50
            ... 3/4
            ... 5/4, 72
            ... 6/8
            ... ''')
            >>> s.getMeasureDef(2)
            MeasureDef(timesig=(5, 4), quarterTempo=72, annotation='', timesigInherited=False,
                       tempoInherited=True, barline='', subdivisionStructure=None)
            >>> s[2]
            MeasureDef(timesig=(5, 4), quarterTempo=72, annotation='', timesigInherited=False,
                       tempoInherited=True, barline='', subdivisionStructure=None)

        """
        if idx < len(self.measuredefs):
            return self.measuredefs[idx]

        # outside of defined measures
        if not self.endless:
            raise IndexError(f"index {idx} out of range. The score has "
                             f"{len(self.measuredefs)} measures defined")
        if not extend:
            # we are "outside" the defined score
            m = self.measuredefs[-1]
            if m.annotation:
                m = m.clone(annotation='', tempoInherited=True, timesigInherited=True)
            return m

        for n in range(len(self.measuredefs)-1, idx):
            self.addMeasure()

        return self.measuredefs[-1]

    def __getitem__(self, item: int) -> MeasureDef:
        if isinstance(item, int):
            return self.getMeasureDef(item)
        print(item, dir(item))

    def addMeasure(self,
                   timesig: timesig_t | str = None,
                   quarterTempo: number_t = None,
                   annotation: str = None,
                   numMeasures=1,
                   rehearsalMark: str | RehearsalMark = None,
                   subdivisions: Sequence[int] = None,
                   keySignature: tuple[int, str] | KeySignature = None,
                   **kws
                   ) -> None:
        """
        Add a measure definition to this score structure

        Args:
            timesig: the time signature of the new measure. If not given, the last
                time signature will be used. The timesig can be given as str in the
                form "num/den". For a compound time signature use, for example
                "(3+2)/8". Compound time signatures of the form "3/4+3/8" are not
                supported yet.
            quarterTempo: the tempo of a quarter note. If not given, the last tempo
                will be used
            annotation: each measure can have a text annotation
            numMeasures: if this is > 1, multiple measures of the same kind can be
                added
            rehearsalMark: if given, add a rehearsal mark to the new measure definition.
                A rehearsal mark can be a text or a RehearsalMark, which enables you
                to customize the rehearsal mark further
            subdivisions: the subdivisions of the measure. For example, it is possible to
                specify how a 7/8 measure is divided by passing (3, 2, 2) (instead of,
                for example, ``2+2+3``)
            keySignature: either a KeySignature object or a tuple (fifths, mode); for example
                for A-Major, ``(3, 'major')``
            **kws: any extra keyword argument will be saved as a property of the MeasureDef

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

        if isinstance(rehearsalMark, str):
            rehearsalMark = RehearsalMark(rehearsalMark)

        if isinstance(keySignature, tuple):
            fifths, mode = keySignature
            keySignature = KeySignature(fifths=fifths, mode=mode)

        measuredef = MeasureDef(timesig=timesig if isinstance(timesig, tuple) else _parseTimesig(timesig),
                                quarterTempo=quarterTempo,
                                annotation=annotation, timesigInherited=timesigInherited,
                                tempoInherited=tempoInherited,
                                rehearsalMark=rehearsalMark,
                                subdivisionStructure=list(subdivisions) if subdivisions else None,
                                properties=kws,
                                keySignature=keySignature)

        self.measuredefs.append(measuredef)
        if numMeasures > 1:
            self.addMeasure(numMeasures=numMeasures-1)

        self._modified = True

    def addRehearsalMark(self, idx: int, mark: RehearsalMark | str, box: str = 'square') -> None:
        """
        Add a rehearsal mark to this scorestruct

        The measure definition for the given index must already exist or the score must
        be set to autoextend

        Args:
            idx: the measure index
            mark: the rehearsal mark, as text or as a RehearsalMark
            box: one of 'square', 'circle' or '' to avoid drawing a box around the rehearsal mark
        """
        if idx >= len(self.measuredefs) and not self.endless:
            raise IndexError(f"Measure index {idx} out of range. "
                             f"This score has {len(self.measuredefs)} measures")
        mdef = self.getMeasureDef(idx, extend=True)
        if isinstance(mark, str):
            mark = RehearsalMark(mark, box=box)
        mdef.rehearsalMark = mark

    def ensureDurationInMeasures(self, numMeasures: int) -> None:
        """
        Extends this score to have at least the given number of measures

        If the scorestruct already has reached the given length this operation
        does nothing

        Args:
            numMeasures: the minimum number of measures this score should have
        """
        measureDiff = numMeasures - self.numDefinedMeasures()
        if measureDiff > 0:
            self.addMeasure(numMeasures=measureDiff)

    def ensureDurationInSeconds(self, duration: F) -> None:
        """
        Ensure that this scorestruct is long enough to include the given time

        This is of relevance in certain edge cases including endless scorestructs:

        * When creating a clicktrack from an endless score.
        * When exporting a scorestruct to midi

        Args:
            duration: the duration in seconds to ensure

        """
        mindex, mbeat = self.timeToLocation(duration)
        if mindex is None:
            raise ValueError(f"duration {duration} outside of score")
        self.ensureDurationInMeasures(mindex + 1)

    def totalDurationBeats(self) -> F:
        """
        The duration of this score, in beats (quarters)

        Raises ValueError if this score is endless
        """
        if self.endless:
            raise ValueError("An endless score does not have a duration in beats")
        return sum(m.durationBeats for m in self.measuredefs)

    def totalDuratioSecs(self) -> F:
        """
        The duration of this score, in seconds

        Raises ValueError if this score is endless
        """
        if self.endless:
            raise ValueError("An endless score does not have a duration in seconds")
        return sum(m.durationSecs for m in self.measuredefs)

    def markAsModified(self, value=True) -> None:
        """
        Call this when a MeasureDef inside this ScoreStruct is modified

        By marking it as modified any internal cache is invalidated
        """
        self._modified = value

    def _update(self) -> None:
        accumTime = F(0)
        accumBeats = F(0)
        starts = []
        quarterDurs = []
        beatOffsets = []

        for mdef in self.measuredefs:
            starts.append(accumTime)
            beatOffsets.append(accumBeats)
            durBeats = mdef.durationBeats
            quarterDurs.append(durBeats)
            accumTime += F(60) / mdef.quarterTempo * durBeats
            accumBeats += durBeats
        self._modified = False
        self._timeOffsets = starts
        self._beatOffsets = beatOffsets
        self._quarternoteDurations = quarterDurs

    def locationToTime(self, measure: int, beat: number_t = F(0)) -> F:
        """
        Return the elapsed time at the given score location

        Args:
            measure: the measure number (starting with 0)
            beat: the beat within the measure

        Returns:
            a time in seconds (as a Fraction to avoid rounding problems)
        """
        if self._modified:
            self._update()

        numdefs = len(self.measuredefs)
        if measure > numdefs - 1:
            if measure == numdefs and beat == 0:
                mdef = self.measuredefs[-1]
                return self._timeOffsets[-1] + mdef.durationSecs

            if not self.endless:
                raise ValueError("Measure outside of score")

            last = numdefs - 1
            lastTime = self._timeOffsets[last]
            mdef = self.measuredefs[last]
            mdur = mdef.durationSecs
            fractionalDur = beat * 60 / mdef.quarterTempo
            return lastTime + (measure - last) * mdur + fractionalDur
        else:
            now = self._timeOffsets[measure]
            mdef = self.measuredefs[measure]
            measureBeats = self._quarternoteDurations[measure]
            assert beat <= measureBeats, f"Beat outside of measure, measure={mdef}"
            qtempo = mdef.quarterTempo
            return now + F(60 * qtempo.denominator, qtempo.numerator) * beat

    def tempoAtTime(self, time: number_t) -> F:
        """
        Returns the tempo active at the given time (in seconds)

        Args:
            time: point in the timeline (in seconds)

        Returns:
            the quarternote-tempo at the given time

        """
        measureindex, measurebeat = self.timeToLocation(time)
        measuredef = self.getMeasureDef(measureindex)
        return measuredef.quarterTempo

    def timeToLocation(self, time: number_t) -> tuple[int|None, F]:
        """
        Find the location in score corresponding to the given time in seconds

        Args:
            time: the time in seconds

        Returns:
            a tuple (measureindex, measurebeat) where measureindex can be None if the score
            is not endless and time is outside the score

        .. seealso:: :meth:`beatToLocation`
        """
        if not self.measuredefs:
            raise IndexError("This ScoreStruct is empty")

        if self._modified:
            self._update()
        time = asF(time)
        idx = bisect(self._timeOffsets, time)
        if idx < len(self.measuredefs):
            m = self.measuredefs[idx-1]
            assert self._timeOffsets[idx-1]<=time<self._timeOffsets[idx]
            dt = time-self._timeOffsets[idx-1]
            beat = dt*m.quarterTempo/F(60)
            return idx-1, beat

        # is it within the last measure?
        m = self.measuredefs[idx-1]
        dt = time - self._timeOffsets[idx-1]
        if dt < m.durationSecs:
            beat = dt*m.quarterTempo/F(60)
            return idx-1, beat
        # outside of score
        if not self.endless:
            return (None, F(0))
        lastMeas = self.measuredefs[-1]
        measDur = lastMeas.durationSecs
        numMeasures = dt / measDur
        beat = (numMeasures - int(numMeasures)) * lastMeas.durationBeats
        return len(self.measuredefs)-1 + int(numMeasures), beat

    def beatToLocation(self, beat: number_t) -> tuple[int|None, F]:
        """
        Return the location in score corresponding to the given beat

        The beat is the time-offset in quarter-notes. Given a beat
        (in quarter-notes), return the score location
        (measure, beat offset within the measure). Tempo does not
        play any role within this calculation.

        Returns:
            a tuple (measure index, beat), where measureindex will be
            None if the beat is outside of the score

        .. note::

            In the special case where a ScoreStruct is not endless and the
            beat is exactly at the end of the last measure, we return
            ``(numMeasures, 0)``

        .. seealso:: :meth:`locationToBeat`, which performs the opposite operation

        Example
        ~~~~~~~

        Given the following score: 4/4, 3/4, 4/4

        ========   =======================
         input       output
        ========   =======================
         4          (1, 0)
         5.5        (1, 1.5)
         8          (2, 1.0)
        ========   =======================
        """
        numdefs = len(self.measuredefs)
        assert numdefs >= 1, "This scorestruct is empty"
        if self._modified:
            self._update()

        if not isinstance(beat, F):
            beat = asF(beat)

        if beat > self._beatOffsets[-1]:
            # past the end
            rest = beat - self._beatOffsets[-1]
            if not self.endless:
                return (None, 0) if rest > 0 else (numdefs, F(0))
            beatsPerMeasure = self.measuredefs[-1].durationBeats
            idx = numdefs - 1
            idx += int(rest / beatsPerMeasure)
            restBeats = rest % beatsPerMeasure
            return (idx, restBeats)
        else:
            lastIndex = self._lastIndex
            lastOffset = self._beatOffsets[lastIndex]
            if lastOffset <= beat <= lastOffset + self._quarternoteDurations[lastIndex]:
                idx = lastIndex
            else:
                ridx = bisect(self._beatOffsets, beat)
                idx = ridx - 1
                self._lastIndex = idx
            rest = beat - self._beatOffsets[idx]
            return (idx, rest)

    def beatToTime(self, beat: number_t) -> F:
        """
        Convert beat-time to real-time

        Args:
            beat: the quarter-note beat

        Returns:
            the corresponding time

        Example
        ~~~~~~~

            >>> from maelzel.scorestruct import ScoreStruct
            >>> sco = ScoreStruct.fromTimesig('4/4', quarterTempo=120)
            >>> sco.beatToTime(2)
            1.0
            >>> sco.timeToBeat(2)
            4.0

        .. seealso:: :meth:`~ScoreStruct.timeToBeat`
        """
        return self.locationToTime(*self.beatToLocation(beat))

    def timeToBeat(self, t: number_t) -> F:
        """
        Convert a time to a quarternote offset according to this ScoreStruct

        Args:
            t: the time (in absolute seconds)

        Returns:
            A quarternote offset

        will raise ValueError if the given time is outside of this score structure

        Example
        ~~~~~~~

            >>> from maelzel.scorestruct import ScoreStruct
            >>> sco = ScoreStruct.fromTimesig('4/4', quarterTempo=120)
            >>> sco.beatToTime(2)
            1.0
            >>> sco.timeToBeat(2)
            4.0

        .. seealso:: :meth:`~ScoreStruct.beatToTime`
        """
        measureindex, measurebeat = self.timeToLocation(t)
        if measureindex is None:
            raise ValueError(f"time {t} outside of score")
        beat = self.locationToBeat(measureindex, measurebeat)
        return beat

    def iterMeasureDefs(self) -> Iterator[MeasureDef]:
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

    def __iter__(self) -> Iterator[MeasureDef]:
        return self.iterMeasureDefs()

    def toBeat(self, x: number_t | tuple[int, number_t]) -> F:
        """
        Convert a time in secs or a location (measure, beat) to a quarter-note beat

        Args:
            x: the time/location to convert

        Returns:
            the corresponding quarter note beat according to this ScoreStruct

        Example
        ~~~~~~~

            >>> sco = ScoreStruct.fromTimesig('3/4', 120)
            # Convert time to beat
            >>> sco.toBeat(0.5)
            1.0
            # Convert score location (measure 1, beat 2) to beats
            >>> sco.toBeat((1, 2))
            5.0

        .. seealso:: :meth:`~ScoreSctruct.toTime`
        """
        if isinstance(x, tuple):
            return self.locationToBeat(*x)
        else:
            return self.timeToBeat(x)

    def toTime(self, x: number_t | tuple[int, number_t]) -> F:
        """
        Convert a quarter-note beat or a location (measure, beat) to an absolute time in secs

        Args:
            x: the beat/location to convert

        Returns:
            the corresponding time according to this ScoreStruct

        Example
        ~~~~~~~

            >>> sco = ScoreStruct.fromTimesig('3/4', 120)
            # Convert time to beat
            >>> sco.toTime(1)
            0.5
            # Convert score location (measure 1, beat 2) to beats
            >>> sco.toTime((1, 2))
            2.5

        .. seealso:: :meth:`~ScoreSctruct.toBeat`

        """
        return self.locationToTime(*x) if isinstance(x, tuple) else self.beatToTime(x)

    def locationToBeat(self, measure: int, beat: number_t = F(0)) -> F:
        """
        Returns the number of quarter notes up to the given location

        This value is independent of any tempo given.

        Args:
            measure: the measure number (measures start at 0)
            beat: the beat within the given measure (beat 0 = start of the measure), in
                quarter notes.

        Returns:
            the location translated to quarter notes.

        Example
        -------

        >>> s = ScoreStruct._parseScore(r'''
        ... 3/4, 120
        ... 3/8
        ... 4/4
        ... ''')
        >>> s.locationToBeat(1, 0.5)
        3.5
        >>> s.locationToTime(1, 0.5)
        1.75

        """
        if self._modified:
            self._update()
        beat = asF(beat)
        if measure < self.numDefinedMeasures():
            # Use the index
            measureOffset = self._beatOffsets[measure]
            quartersInMeasure = self._quarternoteDurations[measure]
            if beat > quartersInMeasure:
                raise ValueError(f"Measure {measure} has {quartersInMeasure} quarters, but given "
                                 f"offset {beat} is too large")
            return measureOffset + beat
        elif not self.endless:
            raise ValueError(f"This scorestruct has {self.numDefinedMeasures()} and is not"
                             f"marked as endless. Measure {measure} is out of scope")
        # It is endless and out of the defined measures
        # TODO

        accum = F(0)
        for i, mdef in enumerate(self.iterMeasureDefs()):
            if i < measure:
                accum += mdef.durationBeats
            else:
                if beat > mdef.durationBeats:
                    raise ValueError(f"beat {beat} outside of measure {i}: {mdef}")
                accum += asF(beat)
                break
        return accum

    def timeDelta(self,
                  start: number_t | tuple[int, number_t],
                  end: number_t | tuple[int, number_t]
                  ) -> F:
        """
        Returns the elapsed time between two beats or score locations.

        Args:
            start: the start location, as a beat or as a tuple (measureIndex, beatOffset)
            end: the end location, as a beat or as a tuple (measureIndex, beatOffset)

        Returns:
            the elapsed time, as a Fraction

        Example
        -------

            >>> from maelzel.scorestruct import ScoreStruct
            >>> s = ScoreStruct('4/4,60; 3/4; 3/8')
            >>> s.timeDelta((0, 0.5), (2, 0.5))
            7
            >>> s.timeDelta(3, (1, 2))
            3

        .. seealso:: :meth:`~ScoreStruct.beatDelta`

        """
        startTime = self.locationToTime(*start) if isinstance(start, tuple) else self.beatToTime(start)
        endTime = self.locationToTime(*end) if isinstance(end, tuple) else self.beatToTime(end)
        return endTime - startTime

    def beatDelta(self, 
                  start: number_t | tuple[int, number_t],
                  end: number_t | tuple[int, number_t]) -> F:
        """
        Difference in beats between the two score locations or two times

        Args:
            start: the start moment as a location (a tuple (measureIndex, beatOffset) or as
                a time
            end: the end location, a tuple (measureIndex, beatOffset)

        Returns:
            the distance between the two locations, in beats

        Example
        -------

            >>> from maelzel.scorestruct import ScoreStruct
            >>> s = ScoreStruct('4/4, 120; 3/4; 3/8; 5/8')
            # delta, in quarternotes, between time=2secs and location (2, 0)
            >>> s.beatDelta(2., (2, 0))
            5

        .. seealso:: :meth:`~ScoreStruct.timeDelta`
        """
        startBeat = self.locationToBeat(*start) if isinstance(start, tuple) else self.timeToBeat(start)
        endBeat = self.locationToBeat(*end) if isinstance(start, tuple) else self.timeToBeat(end)
        return endBeat - startBeat

    def show(self, fmt='png', app: str = '', scalefactor: float = None, backend: str = None
             ) -> None:
        """
        Render and show this ScoreStruct
        """
        import tempfile
        from maelzel.core import environment
        outfile = tempfile.mktemp(suffix='.' + fmt)
        self.write(outfile, backend=backend)
        if fmt == 'png':
            from maelzel.core import jupytertools
            if environment.insideJupyter and not app:
                jupytertools.jupyterShowImage(outfile, scalefactor=scalefactor, maxwidth=1200)
            else:
                emlib.misc.open_with_app(outfile, app=app)
        else:
            emlib.misc.open_with_app(outfile, app=app)

    def dump(self) -> None:
        """
        Dump this ScoreStruct to stdout
        """
        tempo = -1
        N = len(str(len(self.measuredefs)))
        fmt = "%0" + str(N) + "d" + ", %d/%d"
        for i, m in enumerate(self.measuredefs):
            num, den = m.timesig
            parts = [fmt % (i, num, den)]
            if m.quarterTempo != tempo:
                parts.append(f", {m.quarterTempo}")
                tempo = m.quarterTempo
            if m.annotation:
                parts.append(f", annotation={m.annotation}")
            if m.rehearsalMark:
                parts.append(f", rehearsal={m.rehearsalMark.text}")
            if m.barline:
                parts.append(f", barline={m.barline}")
            if m.keySignature:
                parts.append(f", keySignature={m.keySignature.fifths}")
            print("".join(parts))

    def hasUniqueTempo(self) -> bool:
        """
        Returns True if this ScoreStruct has no tempo changes
        """
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

    def __enter__(self):
        if 'maelzel.core.workspace' in sys.modules:
            from maelzel.core import workspace
            w = workspace.getWorkspace()
            self._prevScoreStruct = w.scorestruct
            w.scorestruct = self
        else:
            raise RuntimeError("No active maelzel.core Workspace. A ScoreStruct can only be "
                               "called when maelzel.core has been importated")

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._prevScoreStruct is not None
        from maelzel.core import workspace
        workspace.getWorkspace().scorestruct = self._prevScoreStruct

    def _repr_html_(self) -> str:
        colnames = ['Meas. Index', 'Timesig', 'Tempo (quarter note)', 'Label', 'Rehearsal']

        if any(m.keySignature is not None for m in self.measuredefs):
            colnames.append('Key')
            haskey = True
        else:
            haskey = False

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
            rehearsal = m.rehearsalMark.text if m.rehearsalMark is not None else ''
            timesig = m.timesigRepr()
            row = [str(i), timesig, tempostr, m.annotation or "", rehearsal]
            if haskey:
                row.append(str(m.keySignature.fifths) if m.keySignature else '-')
            rows.append(row)
        if self.endless:
            rows.append(("...", "", "", "", ""))
        rowstyle = 'font-size: small;'
        htmltable = emlib.misc.html_table(rows, colnames, rowstyles=[rowstyle]*5)
        parts.append(htmltable)
        return "".join(parts)

    def _render(self, backend: str = None) -> scoring.render.Renderer:
        from maelzel import scoring
        measures = [scoring.quant.QuantizedMeasure(timesig=m.timesig, quarterTempo=m.quarterTempo)
                    for m in self.measuredefs]
        part = scoring.quant.QuantizedPart(struct=self, measures=measures)
        qscore = scoring.quant.QuantizedScore([part], title=self.title, composer=self.composer)
        options = scoring.render.RenderOptions()
        return scoring.render.renderQuantizedScore(qscore, options=options, backend=backend)

    def asMusic21(self, fillMeasures=False) -> m21.stream.Score:
        """
        Return the score structure as a music21 Score

        Args:
            fillMeasures: if True, measures are filled with a note. This can be useful
                if you need to export the musicxml as midi

        Returns:
            a music21 Score representing this score structure

        .. image:: ../assets/scorestruct-asmusic21.png

        TODO: render barlines according to measureDef
        """
        from maelzel.music import m21tools
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
                s.append(m21.note.Note(pitch=60, duration=m21.duration.Duration(float(measuredef.durationBeats))))
            else:
                s.append(m21.note.Rest(duration=m21.duration.Duration(float(measuredef.durationBeats))))
        score = m21.stream.Score()
        score.insert(0, s)
        m21tools.scoreSetMetadata(score, title=self.title)
        return score

    def setTempo(self, quarterTempo: float, measureIndex: int = 0) -> None:
        """
        Set the tempo of the given measure, until the next tempo change

        Args:
            quarterTempo: the new tempo
            measureIndex: the first measure to modify

        """
        if measureIndex > len(self):
            raise IndexError(f"Index {measureIndex} out of rage; this ScoreStruct has only "
                             f"{len(self)} measures defined")
        mdef = self.measuredefs[measureIndex]
        mdef.quarterTempo = quarterTempo
        mdef.tempoInherited = False
        for m in self.measuredefs[measureIndex+1:]:
            if m.tempoInherited:
                m.quarterTempo = quarterTempo
            else:
                break

    def hasUniqueTimesig(self) -> bool:
        """
        Returns True if this ScoreStruct does not have any time-signature change
        """
        lastTimesig = self.measuredefs[0].timesig
        for m in self.measuredefs:
            if m.timesig != lastTimesig:
                return False
        return True

    def write(self, path: str | Path, backend: str = None) -> None:
        """
        Export this score structure

        Write this as musicxml (.xml), lilypond (.ly), MIDI (.mid) or render as
        pdf or png. The format is determined by the extension of the file

        .. note:: when saving as MIDI, notes are used to fill each beat as an empty
            MIDI score is not supported by the MIDI standard

        Args:
            path: the path of the written file
            backend: for pdf or png only - the backend to use for rendering, one
                of 'lilypond' or 'music21'
        """
        path = Path(path)
        if path.suffix == ".xml":
            m21score = self.asMusic21(fillMeasures=False)
            m21score.write("xml", path)
        elif path.suffix == ".pdf":
            r = self._render(backend=backend)
            r.write(str(path))
        elif path.suffix == ".png":
            r = self._render(backend=backend)
            r.write(str(path))
        elif path.suffix == ".ly":
            m21score = self.asMusic21(fillMeasures=True)
            from maelzel.music import m21tools
            m21tools.saveLily(m21score, path.as_posix())
        elif path.suffix == '.mid' or path.suffix == '.midi':
            sco = _filledScoreFromStruct(self)
            sco.write(str(path))
        else:
            raise ValueError(f"Extension {path.suffix} not supported, "
                             f"should be one of .xml, .pdf, .png or .ly")

    def exportMidi(self, midifile: str) -> None:
        """
        Export this ScoreStruct as MIDI

        Args:
            midifile: the path of the MIDI file to generate

        """
        m21score = self.asMusic21(fillMeasures=False)
        m21score.write("midi", midifile)

    def exportMidiClickTrack(self, midifile: str) -> None:
        """
        Generate a MIDI click track from this ScoreStruct

        Args:
            midifile: the path of the MIDI file to generate

        .. seealso:: :func:`maelzel.core.tools.makeClickTrack`
        """
        from maelzel.core import tools
        click = tools.makeClickTrack(self)
        m21click = click.asmusic21()
        m21click.write('midi', midifile)

    def setBarline(self, measureIndex: int, linetype: str) -> None:
        """
        Set the barline type

        Args:
            measureIndex: the measure index to modify
            linetype: one of 'single', 'double', 'final'

        """
        assert linetype in {'single', 'double', 'final'}
        self.getMeasureDef(measureIndex, extend=True).barline = linetype

    def makeClickTrack(self,
                       minMeasures: int = 0,
                       clickdur: F = None,
                       strongBeatPitch='5C',
                       weakBeatPitch='5G',
                       playTransposition=24,
                       ) -> maelzel.core.Score:
        """
        Create a click track from this ScoreStruct

        The returned score can be displayed as notation via :meth:`maelzel.core.Score.show`
        or exported as pdf or midi.

        This is a shortcut to :func:`maelzel.core.tools.makeClickTrack`. Use that for more
        customization options

        .. note::

            The duration of the playback can be set individually from the duration
            of the displayed pitch

        Args:
            clickdur: the length of each tick. Use None to use the duration of the beat.
            strongBeatPitch: the pitch used as a strong beat (at the beginning of each
                measure
            weakBeatPitch: the pitch used as a weak beat
            playTransposition: the transposition interval between notated pitch and
                playback pitch

        Returns:
            a maelzel.core.Score

        Example
        -------

            >>> from maelzel.core import *
            >>> scorestruct = ScoreStruct(r"4/4,72; .; 5/8; 3/8; 2/4,96; .; 5/4; 3/4")
            >>> clicktrack = scorestruct.makeClickTrack()
            >>> clicktrack.write('click.pdf')
            >>> clicktrack.play()

        .. image:: ../assets/clicktrack2.png
        """
        from maelzel.core import tools
        if minMeasures < self.numDefinedMeasures():
            struct = self
        else:
            struct = self.copy()
            struct.ensureDurationInMeasures(minMeasures)
        return tools.makeClickTrack(struct, clickdur=clickdur,
                                    strongBeatPitch=strongBeatPitch,
                                    weakBeatPitch=weakBeatPitch,
                                    playpreset='_click',
                                    playparams={'ktransp': playTransposition})


def _filledScoreFromStruct(struct: ScoreStruct, pitch='4C') -> maelzel.core.Score:
    """
    Creates a maelzel.core Score representing the given ScoreStruct

    Args:
        struct: the scorestruct to construct
        pitch: the pitch to use to fill the measures

    Returns:
        the resulting maelzel.core Score

    """
    now = 0
    events = []
    from maelzel.core import Note, Voice, Score
    import pitchtools
    midinote = pitch if isinstance(pitch, (int, float)) else pitchtools.n2m(pitch)
    for i, m in enumerate(struct.measuredefs):
        num, den = m.timesig
        dur = 4/den * num
        if i == len(struct.measuredefs) - 1:
            events.append(Note(midinote if i%2==0 else midinote+2, offset=now, dur=dur))
        now += dur
    voice = Voice(events)
    return Score([voice], scorestruct=struct)