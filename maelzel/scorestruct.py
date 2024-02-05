from __future__ import annotations

import copy
from pathlib import Path
from dataclasses import dataclass
from bisect import bisect
import sys
import functools

import emlib.textlib
from emlib import iterlib
from maelzel.common import F, asF, F0, num_t, timesig_t
from maelzel._util import aslist

from typing import TYPE_CHECKING, overload as _overload
if TYPE_CHECKING:
    from typing import Iterator, Sequence, Union
    import maelzel.core
    from maelzel.scoring.renderoptions import RenderOptions
    from maelzel.scoring.renderer import Renderer


__all__ = (
    'asF',
    'ScoreStruct',
    'MeasureDef',
    'measureBeatStructure',
    'TimeSignature'
)


@dataclass
class TimeInterval:
    start: F
    end: F

    @property
    def duration(self) -> F:
        return self.end - self.start

    def __iter__(self):
        return iter((self.start, self.end))

    def __getitem__(self, idx: int):
        return (self.start, self.end)[idx]

    def __len__(self):
        return 2


@functools.cache
def beatWeightsByTimeSignature(num: int, den: int) -> tuple[int, ...]:
    """
    Given a time signature, returns a sequence of beat weights

    A beat weight is a number from 0 to 2, where 0 means unweighted,
    1 means weak weight and 2 means strong weight. For example, for
    a 4/4 measure the returned weights are (2, 0, 1, 0), where the
    first beat is strong, the second unweighted, the 3rd has a weak
    weight and the 4th unweighted again.

    Args:
        num: numerator of the time signature
        den: denominator ot the time signature

    Returns:
        a sequence of weights (a value 0, 1, or 2) as a tuple

    """
    if num % 2 == 0:
        # 2, 0, 1, 0, 1, 0, ...
        weights = [1, 0] * (num//2)
        weights[0] = 2
        return tuple(weights)
    elif num % 3 == 0:
        weights = [1, 0, 0] * (num//3)
        weights[0] = 2
        return tuple(weights)
    elif num % 5 == 0:
        return tuple([2, 0, 1, 0, 0] * (num//5))
    else:
        # All uneven signatures: binary pairs and the last ternary
        weights = [1, 0] * (num//2)
        weights.append(0)
        weights[0] = 2
        return tuple(weights)


class TimeSignature:
    """
    A time signature

    In its simplest form a type signature consists of one part, a tuple
    (numerator, denominator). For example, a 4/4 time signature can be
    represented as TimeSignature((4, 4))

    Args:
        parts: the parts of this time signature, a seq. of tuples (numerator, denominator)
        subdivisions: subdivisions as multiples of the denominator. Only valid for
            non-compound signatures. It is used to structure subdivisions for a single
            part. For example, 7/8 subdivided as 2+3+2 can be expressed as
            TimeSignature((7, 8), subdivisionStruct=(2, 3, 2)).
    """
    def __init__(self,
                 *parts: tuple[int, int],
                 subdivisions: Sequence[int] = ()):
        self.parts: tuple[tuple[int, int], ...] = parts
        """
        The parts of this timesig, as originally passed at creation
        """

        minden = max(den for num, den in parts)
        numerators = [num * minden // den for num, den in parts]

        self.normalizedParts: tuple[tuple[int, int], ...] = tuple((num, minden) for num in numerators)
        """
        The normalized parts with a shared common denominator
        
        parts: (3/4 3/8), common den: 8, normalizedParts: (6/8, 3/8), fusedSignature: 9/8
        """

        self.fusedSignature: tuple[int, int] = (int(sum(numerators)), minden)
        """
        One signature epresenting all compound parts
         
        The fused signature is based on the min. common multiple of the compound parts. 
        For example, a signature 3/4+3/16 will have a fused signature of 15/16 (3*4+3).
        For non-compound signatures, the fused signature is the same as the time signature
        itself."""

        self.subdivisionStruct: tuple[int, ...] = tuple(subdivisions)
        """
        Subdivisions as multiples of the fused denominator. 
        """

    def copy(self) -> TimeSignature:
        return TimeSignature(*self.parts, subdivisions=self.subdivisionStruct)

    def __hash__(self) -> int:
        return hash((self.parts, self.subdivisionStruct))

    def __eq__(self, other: TimeSignature):
        return isinstance(other, TimeSignature) and self.parts == other.parts and self.subdivisionStruct == other.subdivisionStruct

    @property
    def numerator(self) -> int:
        """The numerator of this time signature

        For non-compound signatures, this is the same as the numerator
        of its only part. For compound signatures, this corresponds to the nominator
        of the fused signature

        >>> TimeSignature((7, 8)).numerator
        7
        >>> TimeSignature((2, 4), (5, 8)).numerator
        9
        """
        return self.fusedSignature[0]

    @property
    def denominator(self) -> int:
        """The denominator of this time signature

        For non-compound signatures, this is the same as the denominator
        of its only part. For compound signatures, this corresponds to the denominator
        of the fused signature

        >>> TimeSignature((7, 8)).denominator
        8
        >>> TimeSignature((2, 4), (5, 8)).denominator
        8

        """
        return self.fusedSignature[1]

    @property
    def quarternoteDuration(self) -> F:
        """The duration of this time signature, in quarternotes"""
        num, den = self.fusedSignature
        return F(num, den) * 4

    def __str__(self):
        parts = [f"{num}/{den}" for num, den in self.parts]
        return "+".join(parts)

    def __repr__(self):
        if len(self.parts) == 1:
            num, den = self.parts[0]
            if self.subdivisionStruct:
                subdiv = '-'.join(map(str, self.subdivisionStruct))
                return f"TimeSignature({num}/{den}({subdiv}))"
            return f"TimeSignature({num}/{den})"
        elif all(den == self.parts[0][1] for num, den in self.parts):
            nums = "+".join(str(p[0]) for p in self.parts)
            return f"TimeSignature({nums}/{self.parts[0][1]})"
        else:
            parts = '+'.join(f"{n}/{d}" for n, d in self.parts)
            return f"TimeSignature({parts})"

    @classmethod
    def parse(cls, timesig: str | tuple, subdivisionStruct: Sequence[int] = ()
              ) -> TimeSignature:
        """
        Parse a time signature definition

        Args:
            timesig: a time signature as a string. For compound signatures, use
                a + sign between parts.
            subdivisionStruct: the subdivision structure as multiples of the
                fused signature denominator.

        Returns:
            the time signature

        """
        if isinstance(timesig, tuple):
            if all(isinstance(_, tuple) for _ in timesig):
                # ((3, 8), (3, 8), (2, 8))
                return TimeSignature(*timesig)
            elif len(timesig) == 2 and isinstance(timesig[1], int):
                num, den = timesig
                if isinstance(num, tuple):
                    # ((3, 3, 2), 8)
                    parts = [(_, den) for _ in num]
                    return TimeSignature(*parts, subdivisions=subdivisionStruct)
                else:
                    assert isinstance(num, int)
                    return TimeSignature((num, den), subdivisions=subdivisionStruct)
            else:
                raise ValueError(f"Cannot parse timesignature: {timesig}")
        elif isinstance(timesig, str):
            # Possible signatures: 3/4, 3/8+3/8+2/8, 5/8(3-2), 5/8(3-2)+3/16
            parts = timesig.split("+")
            parsedParts = [_parseTimesigPart(part) for part in parts]
            if len(parsedParts) == 1:
                signature, subdivs = parsedParts[0]
                if subdivs and subdivisionStruct:
                    raise ValueError("Duplicate subdivision structure")
                return TimeSignature(signature, subdivisions=subdivs or subdivisionStruct)
            signatures, subdivs = zip(*parsedParts)
            # We ignore subdivisions for compound signatures
            return TimeSignature(*signatures, subdivisions=subdivisionStruct)

        else:
            raise TypeError(f"Expected a str or a tuple, got {timesig}")

    def isHeterogeneous(self) -> bool:
        """
        Is this a compound meter with heterogeneous denominators?

        Heterogeneous meters are 3/4+3/8, 4/4+1/8, 3/8+3/16. Homogeneous meters are
        3/8+2/8, 3/4+4/4

        Returns:
            true if this is a compound meter with heterogenous denominators
        """
        if len(self.parts) == 1:
            return False
        denoms = set(denom for num, denom in self.parts)
        return len(denoms) >= 2

    def qualifiedSubdivisionStruct(self) -> tuple[int, tuple[int, ...]]:
        """
        The qualified subdivision structure, a tuple (denominator, subdivisions)
        where the denominator is the max. common denom. of the parts of this signature
        and the subdivisions are the subdivisions as given in the subdivisionStruct

        This method will raise ValueError if this time signature does not have a
        subdivision structure

        Example
        ~~~~~~~

            >>> t = TimeSignature((7, 8), subdivisionStruct=(3, 2, 2))
            >>> t.qualifiedSubdivisionStruct()
            (8, (3, 2, 2))
            >>> t2 = TimeSignature((2, 4), (3, 16), subdivisionStruct=(4, 4, 3))
            >>> t2.fusedSignature
            (11, 16)
            >>> t2.qualifiedSubdivisionStruct()
            (16, (4, 4, 3))
        """
        if not self.subdivisionStruct:
            raise ValueError(f"This time signature does not have a subdivision structure")
        return self.fusedSignature[1], self.subdivisionStruct


def _parseTimesigPart(s: str) -> tuple[tuple[int, int], tuple[int, ...]]:
    """
    Given a string in the form 5/8(3-2), returns ((5, 8), (3, 2))

    For 5/8, returns ((5, 8), ())
    """
    if "(" in s:
        assert s.count("(") == 1 and s[-1] == ")"
        p1, p2 = s[:-1].split("(")
        nums, dens = p1.split("/")
        num, den = int(nums), int(dens)
        subdivs = tuple(int(subdiv) for subdiv in p2.split("-"))
        return ((num, den), subdivs)
    else:
        nums, dens = s.split("/")
        return ((int(nums), int(dens)), ())


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


def _asTimeSignature(timesig: str | timesig_t | TimeSignature
                     ) -> TimeSignature:
    if isinstance(timesig, TimeSignature):
        return timesig
    elif isinstance(timesig, str):
        return TimeSignature.parse(timesig)
    elif isinstance(timesig, tuple):
        return TimeSignature(timesig)
    else:
        raise TypeError(f"Cannot convert {timesig} to a TimeSignature")


# def _asTimesig(t: str | timesig_t) -> timesig_t:
#     if isinstance(t, tuple):
#         assert len(t) == 2
#         return t
#     elif isinstance(t, str):
#         return _parseTimesig(t)
#     else:
#         raise TypeError(f"Expected a tuple (5, 8) or a string '5/8', got {t}, {type(t)}")


@dataclass
class _ScoreLine:
    measureIndex: int | None
    timesig: TimeSignature | None
    tempo: float | None
    label: str = ''
    barline: str = ''
    rehearsalMark: str = ''


@dataclass
class RehearsalMark:
    text: str
    box: str = ''


class KeySignature:
    def __init__(self, fifths: int, mode='major'):
        self.fifths = fifths
        self.mode = mode if mode else 'major'


def _parseScoreStructLine(line: str) -> _ScoreLine:
    """
    parse a line of a ScoreStruct definition

    The line has the format ``[measureIndex, ] timesig [, tempo] [keywords]

    Where timesig has the format ``num/den`` and keywords have the format
    ``keyword=value``, where keyword can be one of ``rehearsalmark``, ``label``
    and ``barline``. *rehearsalmark* will add a rehearsal mark to the measure,
    *label* will add a text label to the measure and *barline* will customize
    the right barline. Possible values for *barline* are single, double, solid,
    dotted or dashed


    Args:
        line: a line of the format [measureIndex, ] timesig [, tempo]

    Returns:
        a tuple (measureIndex, timesig, tempo), where only timesig
        is required
    """
    line = line.strip()
    args = []
    keywords = {}
    for part in [_.strip() for _ in line.split(",")]:
        if '=' in part:
            key, value = part.split('=', maxsplit=1)
            key = key.strip()
            value = value.strip()
            if value[0] == value[-1] in "'\"":
                value = value[1:-1]
            keywords[key] = value
        else:
            args.append(part)
    numargs = len(args)
    label = ''
    barline = ''
    rehearsalmark = ''
    if numargs == 1:
        timesigS = args[0]
        measure = None
        tempo = None
    elif numargs == 2:
        if "/" in args[0]:
            timesigS, tempoS = args
            measure = None
            try:
                tempo = float(tempoS)
            except ValueError:
                raise ValueError(f"Could not parse the tempo ({tempoS}) as a number (line: {line})")
        else:
            measureIndexS, timesigS = args
            try:
                measure = int(measureIndexS)
            except ValueError:
                raise ValueError(f"Could not parse the measure index '{measureIndexS}' while parsing line: '{line}'")
            tempo = None
    elif numargs == 3:
        if "/" not in args[0]:
            measureIndexS, timesigS, tempoS = [_.strip() for _ in args]
            measure = int(measureIndexS) if measureIndexS else None
        else:
            measure = None
            timesigS, tempoS, label = [_.strip() for _ in args]
        tempo = float(tempoS) if tempoS else None
    elif numargs == 4:
        measureIndexS, timesigS, tempoS, label = [_.strip() for _ in args]
        measure = int(measureIndexS) if measureIndexS else None
        tempo = float(tempoS) if tempoS else None
    else:
        raise ValueError(f"Parsing error at line {line}")
    timesig = TimeSignature.parse(timesigS) if timesigS else None

    for k, v in keywords.items():
        k = k.lower()
        if k == 'label':
            label = v
        elif k == 'barline':
            assert v in _barstyles, f"Expected a barline style ({_barstyles}), got {v}"
            barline = v
        elif k == 'rehearsalmark':
            rehearsalmark = v
        else:
            raise ValueError(f"Key {k} unknown (value: {v}) while reading score line {line}")
    if label:
        label = label.replace('"', '')
    return _ScoreLine(measureIndex=measure, timesig=timesig, tempo=tempo,
                      label=label, barline=barline, rehearsalMark=rehearsalmark)


_barstyles = {'single', 'final', 'double', 'solid', 'dotted', 'dashed', 'tick', 'short',
              'double-thin', 'none'}


class MeasureDef:
    """
    A MeasureDef defines one Measure within a ScoreStruct (time signature, tempo, etc.)
    """
    __slots__ = (
        '_timesig',
        '_quarterTempo',
        '_barline',
        'annotation',
        'timesigInherited',
        'tempoInherited',
        'rehearsalMark',
        'keySignature',
        'properties',
        'maxEighthTempo',
        'parent',
        'readonly'
    )

    def __init__(self,
                 timesig: TimeSignature,
                 quarterTempo: F | int,
                 annotation='',
                 timesigInherited=False,
                 tempoInherited=False,
                 barline='',
                 rehearsalMark: RehearsalMark = None,
                 keySignature: KeySignature = None,
                 properties: dict = None,
                 maxEighthTempo=48,
                 parent: ScoreStruct = None,
                 readonly=True
                 ):
        assert not barline or barline in _barstyles, \
            f"Unknown barline style: '{barline}', possible values: {_barstyles}"

        assert isinstance(timesig, TimeSignature), f"Expected a TimeSignature, got {timesig}"
        self._timesig: TimeSignature = timesig
        self._quarterTempo = asF(quarterTempo)
        self.annotation = annotation
        """Any text annotation for this measure"""

        self.timesigInherited = timesigInherited
        """Is the time-signature of this measure inherited?"""

        self.tempoInherited = tempoInherited
        """Is the tempo of this measure inherited?"""

        self._barline = barline
        """The barline style, or '' to use default"""

        self.rehearsalMark = rehearsalMark
        """If given, a RehearsalMark for this measure"""

        self.keySignature = keySignature
        """If given, a key signature"""

        self.properties = properties
        """User defined properties can be placed here. None by default"""

        self.maxEighthTempo: int = maxEighthTempo
        """The max. tempo at which an eighth note can be a beat of its own"""

        self.parent = parent
        """The parent ScoreStruct of this measure, if any"""

        self.readonly = readonly
        """Is this measure definition read only?"""

    @property
    def durationQuarters(self) -> F:
        """The duration of this measure in quarter-notes"""
        return self.timesig.quarternoteDuration

    @property
    def durationSecs(self) -> F:
        """The duration of this measure in seconds"""
        return self.durationQuarters * (F(60) / self._quarterTempo)

    @property
    def timesig(self) -> TimeSignature:
        """The time signature of this measure. Can be explicit or inherited"""
        return self._timesig

    @timesig.setter
    def timesig(self, timesig):
        if self.readonly:
            raise ValueError("This MeasureDef is readonly")
        self._timesig = TimeSignature.parse(timesig)
        self.timesigInherited = False
        if self.parent:
            self.parent.modified(timing=True, attributes=True)

    @property
    def quarterTempo(self) -> F:
        """The tempo relative to a quarternote"""
        return self._quarterTempo

    @quarterTempo.setter
    def quarterTempo(self, tempo: F | int):
        if self.readonly:
            raise ValueError("This MeasureDef is readonly")
        self._quarterTempo = asF(tempo)
        self.tempoInherited = False
        if self.parent:
            self.parent.modified(timing=True, attributes=True)

    @property
    def barline(self) -> str:
        """The barline style, or '' to use default"""
        return self._barline

    @barline.setter
    def barline(self, linestyle: str):
        if self.readonly:
            raise ValueError("This MeasureDef is readonly")
        if linestyle not in _barstyles:
            raise ValueError(f'Unknown barstyle: {linestyle}, possible values: {_barstyles}')
        self._barline = linestyle

    def beatStructure(self) -> list[BeatStructure]:
        """
        Beat structure of this measure

        Returns:
            a list of tuple with the form (beatOffset: F, beatDur: F, beatWeight: int)
            for each beat of this measure
        """

        return measureBeatStructure(self.timesig, quarterTempo=self.quarterTempo,
                                    subdivisionStructure=self.subdivisionStructure())

    def asScoreLine(self) -> str:
        """
        The representation of this MeasureDef as a score line
        """
        num = self.timesig.numerator
        den = self.timesig.denominator
        parts = [f'{num}/{den}, {self.quarterTempo}']
        if self.annotation:
            parts.append(self.annotation)
        return ', '.join(parts)

    def __copy__(self):
        return MeasureDef(timesig=self._timesig,
                          quarterTempo=self._quarterTempo,
                          annotation=self.annotation,
                          timesigInherited=self.timesigInherited,
                          tempoInherited=self.tempoInherited,
                          keySignature=self.keySignature,
                          rehearsalMark=self.rehearsalMark,
                          barline=self.barline)

    def copy(self) -> MeasureDef:
        return self.__copy__()

    def __repr__(self):
        parts = [f'timesig={self._timesig}, quarterTempo={self._quarterTempo}']
        if self.annotation:
            parts.append(f'annotation="{self.annotation}"')
        if self.timesigInherited:
            parts.append('timesigInherited=True')
        if self.tempoInherited:
            parts.append('tempoInherited=True')
        if self.barline:
            parts.append(f'barline={self.barline}')
        if self.keySignature:
            parts.append(f'keySignature={self.keySignature}')
        if self.rehearsalMark:
            parts.append(f'rehearsalMark={self.rehearsalMark}')
        return f'MeasureDef({", ".join(parts)})'

    def __hash__(self) -> int:
        return hash((self.timesig, self.quarterTempo, self.annotation))

    def subdivisionStructure(self) -> tuple[int, tuple[int, ...]]:
        """
        Max. common denominator for subdivisions and the subdivisions as multiples of it

        For example, for 3/4+3/8, returns (8, (2, 2, 2, 3))

        Returns:
            a tuple (max. common denominator, subdivisions as multiples of common denominator)

        """
        subdivs = self.subdivisions()
        denom = self.timesig.fusedSignature[1]
        multiples = tuple((subdiv * denom // 4) for subdiv in subdivs)
        return denom, multiples

    def subdivisions(self) -> list[F]:
        """
        Returns a list of the durations representing the subdivisions of this measure.

        A subdivision is a duration, in quarters.

        Returns:
            a list of durations which sum up to the duration of this measure

        Example
        -------

            >>> MeasureDef(timesig=TimeSignature((3, 4)), quarterTempo=60).subdivisions()
            [1, 1, 1]
            >>> MeasureDef(timesig=TimeSignature((3, 8)), quarterTempo=60).subdivisions()
            [0.5, 0.5, 0.5]
            >>> MeasureDef(timesig=TimeSignature((7, 8)), quarterTempo=40).subdivisions()
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            >>> MeasureDef(timesig=TimeSignature((7, 8)), quarterTempo=150).subdivisions()
            [1.0, 1.0, 1.5]
            >>> MeasureDef(TimeSignature((7, 8), subdivisionStruct=(2, 3, 2)), quarterTempo=150).subdivisions()
            [1, 1.5, 1]
        """
        return measureSubdivisions(timesig=self.timesig, quarterTempo=self.quarterTempo,
                                   maxEighthTempo=self.maxEighthTempo)

    def timesigRepr(self) -> str:
        """
        Returns a string representation of this measure's time signature

        Returns:
            a string representation of this measure's time-signature

        """
        parts = [f"{num}/{den}" for num, den in self.timesig.parts]
        partstr = "+".join(parts)
        if self.timesig.subdivisionStruct:
            subdivs = self.subdivisions()
            partstr += f"({subdivs})"
        return partstr


def inferSubdivisions(num: int, den: int, quarterTempo
                      ) -> tuple[int, ...]:
    if (den == 8 or den == 16) and num % 3 == 0:
        return tuple([3] * (num // 3))
    subdivs = []
    while num > 3:
        subdivs.append(2)
        num -= 2
    if num:
        subdivs.append(num)
    return tuple(subdivs)


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


def _checkSubdivisionStructure(s: tuple[int, tuple[int, ...]]) -> None:
    assert isinstance(s, tuple) and len(s) == 2, s
    assert isinstance(s[0], int) and isinstance(s[1], tuple), s
    assert all(isinstance(div, int) for div in s[1]), s


def measureSubdivisions(timesig: TimeSignature,
                        quarterTempo: F,
                        subdivisionStructure: tuple[int, tuple[int, ...]] = None,
                        maxEighthTempo: int = 48
                        ) -> list[F]:
    if len(timesig.parts) == 1:
        if subdivisionStructure is None and timesig.subdivisionStruct:
            subdivisionStructure = timesig.qualifiedSubdivisionStruct()
        return beatDurations(timesig=timesig.parts[0], quarterTempo=quarterTempo,
                             subdivisionStructure=subdivisionStructure,
                             maxEighthTempo=maxEighthTempo)
    subdivs = []
    for part in timesig.parts:
        # TODO: use the subdivision structure in the timesig, if present
        beatdurs = beatDurations(timesig=part, quarterTempo=quarterTempo,
                                 maxEighthTempo=maxEighthTempo)
        subdivs.extend(beatdurs)
    return subdivs


def beatDurations(timesig: timesig_t,
                  quarterTempo: F,
                  maxEighthTempo: num_t = 48,
                  subdivisionStructure: tuple[int, tuple[int, ...]] = None
                  ) -> list[F]:
    """
    Returns the beat durations for the given time signature

    Args:
        timesig: the timesignature of the measure or of the part of the measure
        quarterTempo: the tempo for a quarter note
        maxEighthTempo: max quarter tempo to divide a measure like 5/8 in all
            eighth notes instead of, for example, 2+2+1
        subdivisionStructure: if given, a tuple (denominator, list of subdivision lengths)
            For example, a 5/8 measure could have a subdivision structure of (8, (2, 3)) or
            (8, (3, 2)).

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
        _checkSubdivisionStructure(subdivisionStructure)
        subdivden, subdivnums = subdivisionStructure
        subdivisions = [F(num, subdivden // 4) for num in subdivnums]
        if sum(subdivisions) != quarters:
            raise ValueError(f"The sum of the subdivisions ({sum(subdivisions)}) does not"
                             f"match the number of quarters ({quarters}) in this time "
                             f"signature ({timesig[0]}/{timesig[1]}). Subdivision structure: "
                             f"{subdivisionStructure}")
        return subdivisions
    elif den == 4 or den == 2:
        return [F(1)] * quarters.numerator
    elif den == 8:
        if quarterTempo <= maxEighthTempo:
            # render all beats as 1/8 notes
            return [F(1, 2)]*num
        subdivstruct = inferSubdivisions(num=num, den=den, quarterTempo=quarterTempo)
        return [F(num, den // 4) for num in subdivstruct]
    elif den == 16:
        beatdurs = beatDurations((num, 8), quarterTempo=quarterTempo*2)
        return [dur/2 for dur in beatdurs]
    else:
        raise ValueError(f"Invalid time signature: {timesig}")


@dataclass
class BeatStructure:
    offset: F
    duration: F
    weight: int = 0

    def isBinary(self) -> bool:
        return self.duration.numerator != 3

    @property
    def end(self) -> F:
        return self.offset + self.duration


# def possibleSubdivisions(timesig: tuple[int, int],
#                          quarterTempo: F
#                          ) -> list[tuple[int, ...]]:
#     """
#     Given a timesig in the form (num, den), returns a list of possible subdivisions
#
#     =================  =======================================
#      time sig          subdivisions
#     =================  =======================================
#      (5, 8)             (2, 3), (3, 2)
#      (6, 8)             (2, 2, 2), (3, 3)
#      (7, 8)             (2, 2, 3), (2, 3, 2), (3, 2, 2)
#      (9, 8)             (2, 2, 2, 3), (2, 2, 3, 2), (2, 3, 2, 2), (3, 2, 2, 2), (3, 3, 3)
#      (11, 8)            (2, 2, 2, 2, 3), ...
#     =================  =======================================
#     """
#     num, den = timesig
#     divisions = {
#         5: [(2, 3), (3, 2)],
#         6: [(2, 2, 2), (3, 3)],
#         7: [(2, 2, 3), (2, 3, 2), (3, 2, 2)],
#         9: [(2, 2, 2, 3), (2, 2, 3, 2), (2, 3, 2, 2), (3, 2, 2, 2), (3, 3, 3)],
#         11: [(2, 2, 2, 2, 3), (2, 2, 2, 3, 2), (2, 2, 3, 2, 2), (2, 3, 2, 2, 2), (3, 2, 2, 2, 2),
#              (2, 3, 3, 3), (3, 2, 3, 3), (3, 3, 2, 3), (3, 3, 3, 2)],
#     }
#     return divisions.get(num) or [_trivialSubdivisions(num)]


def _trivialSubdivisions(num: int) -> tuple[int, ...]:
    if num % 2 == 0:
        out = tuple([2] * (num//2))
    else:
        out = tuple([2] * ((num//2) - 1) + [3])
    assert sum(out) == num
    return out


@functools.cache
def measureBeatStructure(timesig: TimeSignature,
                         quarterTempo: Union[F, int],
                         subdivisionStructure: tuple[int, tuple[int, ...]] = None
                         ) -> list[BeatStructure]:
    """
    Returns the beat structure for this measure

    Args:
        timesig: the time signature
        quarterTempo: the tempo of the quarter note
        subdivisionStructure: the subdivision structure in the
            form (denominator: int, subdivisions). For example a 7/8 bar divided
            in 3+2+2 would have a subdivision strucutre of (8, (3, 2, 2)). A
            4/4 measure divided in 3/8+3/8+2/8+2/8 would be (8, (3, 3, 2, 2))

    Returns:
        a list of (beat offset: F, beat duration: F, beat weight: int)
    """
    durations = measureSubdivisions(timesig=timesig, quarterTempo=quarterTempo,
                                    subdivisionStructure=subdivisionStructure)

    N = len(durations)
    if N == 1:
        weights = [1]
    elif N % 2 == 0:
        weights = [1, 0] * (N//2)
    elif N % 3 == 0:
        weights = [1, 0, 0] * (N//3)
    else:
        weights = [1, 0] * (N//2)
        weights.append(0)

    weights[0] = 2

    now = F(0)
    beatOffsets = []
    for i, dur in enumerate(durations):
        beatOffsets.append(now)
        now += dur
        if dur.numerator == 3:
            weights[i] = 1

    assert len(beatOffsets) == len(durations) == len(weights)
    return [BeatStructure(offset, duration, weight)
            for offset, duration, weight in zip(beatOffsets, durations, weights)]


def measureBeatOffsets(timesig: timesig_t,
                       quarterTempo: F | int,
                       subdivisionStructure: tuple[int, ...] = None
                       ) -> list[F]:
    """
    Returns a list with the offsets of all beats in measure.

    The last value refers to the offset of the end of the measure

    Args:
        timesig: the timesignature as a tuple (num, den)
        quarterTempo: the tempo correponding to a quarter note
        subdivisionStructure: if given, a list of subdivision lengths. For example,
            a 5/8 measure could have a subdivision structure of (2, 3) or (3, 2)

    Returns:
        a list of fractions representing the start time of each beat, plus the
        end time of the measure (== the start time of the next measure)

    Example::
        >>> measureBeatOffsets((5, 8), 60)
        [Fraction(0, 1), Fraction(1, 1), Fraction(2, 1), Fraction(5, 2)]
        # 0, 1, 2, 2.5
    """
    quarterTempo = asF(quarterTempo)
    subdivstruct = None if subdivisionStructure is None else (timesig[1], subdivisionStructure)
    beatdurs = beatDurations(timesig,
                             quarterTempo=quarterTempo,
                             subdivisionStructure=subdivstruct)
    beatOffsets = [F(0)] + list(iterlib.partialsum(beatdurs))
    return beatOffsets


class ScoreStruct:
    """
    A ScoreStruct holds the structure of a score but no content

    A ScoreStruct consists of some metadata and a list of :class:`MeasureDefs`,
    where each :class:`MeasureDef` defines the properties of the measure at the given
    index. If a ScoreStruct is marked as *endless*, it is possible to query
    it (convert beats to time, etc.) outside the defined measures.

    The ScoreStruct class is used extensively within :py:mod:`maelzel.core` (see
    `scorestruct-and-maelzel-core`)

    Args:
        score: if given, a score definition as a string (see below for the format)
        timesig: time-signature. If no score is given, a timesig can be passed to
            define a basic scorestruct with a time signature and a default or
            given tempo
        tempo: the tempo of a quarter note, if given. Even if using a time-signature with
            a smaller denominator (like 3/8), the tempo is always given in reference to
            a quarter note.
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
                 tempo: int = None,
                 endless: bool = True,
                 title='',
                 composer='',
                 readonly=False):

        # holds the time offset (in seconds) of each measure
        self._timeOffsets: list[F] = []

        self._beatOffsets: list[F] = []

        # the quarternote duration of each measure
        self._quarternoteDurations: list[F] = []

        self._attributesModified = True
        self._timingModified = True
        self._prevScoreStruct: ScoreStruct | None = None

        self._lastIndex = 0

        self.readonly = False
        """Is this ScoreStruct read-only?"""

        self.title = title
        """Title metadata"""

        self.composer = composer
        """The composer metadata"""

        if score:
            if timesig or tempo:
                raise ValueError("Either a score as string or a timesig / quarterTempo can be given"
                                 "but not both")
            s = ScoreStruct._parseScore(score)
            self.measuredefs = s.measuredefs
            self.endless = endless
        else:
            self.measuredefs: list[MeasureDef] = []
            self.endless = endless
            if timesig or tempo:
                if not timesig:
                    timesig = (4, 4)
                elif not tempo:
                    tempo = 60
                self.addMeasure(timesig, quarterTempo=tempo)

        self.readonly = readonly
        self._hash: int | None = None

    def __hash__(self) -> int:
        if self._hash is None:
            hashes = [hash(x) for x in (self.title, self.endless)]
            hashes.extend(hash(mdef) for mdef in self.measuredefs)
            self._hash = hash(tuple(hashes))
        return self._hash

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

        def lineStrip(line: str) -> str:
            if "#" in line:
                line = line.split("#")[0]
            return line.strip()

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

            struct.addMeasure(
                timesig=mdef.timesig,
                quarterTempo=mdef.tempo,
                annotation=mdef.label,
                rehearsalMark=RehearsalMark(mdef.rehearsalMark) if mdef.rehearsalMark else None,
                barline=mdef.barline
            )
            measureIndex = mdef.measureIndex
        
        return struct

    def copy(self) -> ScoreStruct:
        """
        Create a copy of this ScoreSturct
        """
        s = ScoreStruct(endless=self.endless, title=self.title, composer=self.composer)
        s.measuredefs = copy.deepcopy(self.measuredefs)
        return s

    def numMeasures(self) -> int:
        """
        Returns the number of measures in this score structure

        If self is endless, it returns the number of defined measures

        Example
        ~~~~~~~

        We create an endless structure (which is the default) where the last
        defined measure is at index 9. The number of measures is 10

            >>> from maelzel.scorestruct import *
            >>> struct = ScoreStruct(r'''
            ... 4/4
            ... .
            ... 3/4
            ... 6, 4/4
            ... 9, 3/4
            ''')
            >>> struct.numMeasures()
            10

        """
        return len(self.measuredefs)

    def __len__(self):
        """
        Returns the number of defined measures (even if the score is defined as endless)

        This is the same as :meth:`ScoreStruct.numMeasures`
        """
        return self.numMeasures()

    def getMeasureDef(self, idx: int, extend=False) -> MeasureDef:
        """
        Returns the MeasureDef at the given index.

        Args:
            idx: the measure index (measures start at 0)
            extend: if True and the index given is outside the defined
                measures, the score will be extended, repeating the last
                defined measure

        If the scorestruct is endless and the index is outside the defined
        range, the returned MeasureDef will be a copy of the last defined MeasureDef.

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

        # outside defined measures
        if not self.endless:
            raise IndexError(f"index {idx} out of range. The score has "
                             f"{len(self.measuredefs)} measures defined")

        if not extend:
            # "outside" of the defined score: return a copy of the last
            # measure so that any modification will not have any effect
            # Make the parent None so that it does not get notified if tempo or timesig
            # change
            return self.measuredefs[-1].copy()

        for n in range(len(self.measuredefs)-1, idx):
            self.addMeasure()

        return self.measuredefs[-1]

    @_overload
    def __getitem__(self, item: int) -> MeasureDef: ...

    @_overload
    def __getitem__(self, item: slice) -> list[MeasureDef]: ...

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.getMeasureDef(item)
        assert isinstance(item, slice)
        return [self.getMeasureDef(idx) for idx in range(item.start, item.stop, item.step)]

    def addMeasure(self,
                   timesig: tuple[int, int] | str | TimeSignature = None,
                   quarterTempo: num_t = None,
                   annotation='',
                   numMeasures=1,
                   rehearsalMark: str | RehearsalMark = None,
                   keySignature: tuple[int, str] | KeySignature = None,
                   barline='',
                   **kws
                   ) -> None:
        """
        Add a measure definition to this score structure

        Args:
            timesig: the time signature of the new measure. If not given, the last
                time signature will be used. The timesig can be given as str in the
                form "num/den". For a compound time signature use "3/8+2/8". To
                specify the internal subdivision use a TimeSignature object or a
                string in the form "5/8(3-2)"
            quarterTempo: the tempo of a quarter note. If not given, the last tempo
                will be used
            annotation: each measure can have a text annotation
            numMeasures: if this is > 1, multiple measures of the same kind can be
                added
            rehearsalMark: if given, add a rehearsal mark to the new measure definition.
                A rehearsal mark can be a text or a RehearsalMark, which enables you
                to customize the rehearsal mark further
            keySignature: either a KeySignature object or a tuple (fifths, mode); for example
                for A-Major, ``(3, 'major')``. Mode can also be left as an ampty string
            barline: if needed, the right barline of the measure can be set to one of
                'single', 'final', 'double', 'solid', 'dotted', 'dashed', 'tick', 'short',
                'double-thin' or 'none'
            **kws: any extra keyword argument will be saved as a property of the MeasureDef

        Example::

            # Create a 4/4 score, 32 measures long
            >>> s = ScoreStruct()
            >>> s.addMeasure((4, 4), 52, numMeasures=32)
        """
        if self.readonly:
            raise RuntimeError("This ScoreStruct is read-only")

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

        if not isinstance(timesig, TimeSignature):
            timesig = TimeSignature.parse(timesig)

        measuredef = MeasureDef(timesig=timesig,
                                quarterTempo=quarterTempo,
                                annotation=annotation,
                                timesigInherited=timesigInherited,
                                tempoInherited=tempoInherited,
                                rehearsalMark=rehearsalMark,
                                properties=kws,
                                keySignature=keySignature,
                                barline=barline,
                                parent=self,
                                readonly=self.readonly)

        self.measuredefs.append(measuredef)
        if numMeasures > 1:
            self.addMeasure(numMeasures=numMeasures-1)

        self._timingModified = True

    def addRehearsalMark(self, idx: int, mark: RehearsalMark | str, box: str = 'square'
                         ) -> None:
        """
        Add a rehearsal mark to this scorestruct

        The measure definition for the given index must already exist or the score must
        be set to autoextend

        Args:
            idx: the measure index
            mark: the rehearsal mark, as text or as a RehearsalMark
            box: one of 'square', 'circle' or '' to avoid drawing a box around the rehearsal mark
        """
        if self.readonly:
            raise RuntimeError("This ScoreStruct is read-only")

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
        measureDiff = numMeasures - self.numMeasures()
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
            raise ValueError(f"duration {duration} outside score")
        self.ensureDurationInMeasures(mindex + 1)

    def durationQuarters(self) -> F:
        """
        The duration of this score, in quarters

        Raises ValueError if this score is endless
        """
        if self.endless:
            raise ValueError("An endless score does not have a duration in beats")
        return asF(sum(m.durationQuarters for m in self.measuredefs))

    def durationSecs(self) -> F:
        """
        The duration of this score, in seconds

        Raises ValueError if this score is endless
        """
        if self.endless:
            raise ValueError("An endless score does not have a duration in seconds")
        return asF(sum(m.durationSecs for m in self.measuredefs))

    def _update(self) -> None:
        if not self._attributesModified and not self._timingModified:
            return

        if self._attributesModified:
            self._fixInheritedAttributes()
            self._attributesModified = False

        if self._timingModified:
            accumTime = F(0)
            accumBeats = F(0)
            starts = []
            quarterDurs = []
            beatOffsets = []

            for mdef in self.measuredefs:
                starts.append(accumTime)
                beatOffsets.append(accumBeats)
                durBeats = mdef.durationQuarters
                quarterDurs.append(durBeats)
                accumTime += F(60) / mdef.quarterTempo * durBeats
                accumBeats += durBeats

            self._timeOffsets = starts
            self._beatOffsets = beatOffsets
            self._quarternoteDurations = quarterDurs
            self._timingModified = False

    def locationToTime(self, measure: int, beat: num_t = F(0)) -> F:
        """
        Return the elapsed time at the given score location

        Args:
            measure: the measure number (starting with 0)
            beat: the beat within the measure

        Returns:
            a time in seconds (as a Fraction to avoid rounding problems)
        """
        self._update()

        numdefs = len(self.measuredefs)
        if measure > numdefs - 1:
            if measure == numdefs and beat == 0:
                mdef = self.measuredefs[-1]
                return self._timeOffsets[-1] + mdef.durationSecs

            if not self.endless:
                raise ValueError("Measure outside score")

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
            assert beat <= measureBeats, f"Beat outside measure, measure={mdef}"
            qtempo = mdef.quarterTempo
            return now + F(60 * qtempo.denominator, qtempo.numerator) * beat

    def tempoAtTime(self, time: num_t) -> F:
        """
        Returns the tempo active at the given time (in seconds)

        Args:
            time: point in the timeline (in seconds)

        Returns:
            the quarternote-tempo at the given time

        """
        measureindex, measurebeat = self.timeToLocation(time)
        if measureindex is None:
            raise ValueError(f"time {time} outside of score")
        measuredef = self.getMeasureDef(measureindex)
        return measuredef.quarterTempo

    def timeToLocation(self, time: num_t) -> tuple[int | None, F]:
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

        self._update()

        time = asF(time)
        idx = bisect(self._timeOffsets, time)
        if idx < len(self.measuredefs):
            m = self.measuredefs[idx-1]
            assert self._timeOffsets[idx - 1] <= time < self._timeOffsets[idx]
            dt = time-self._timeOffsets[idx-1]
            beat = dt*m.quarterTempo/F(60)
            return idx-1, beat

        # is it within the last measure?
        m = self.measuredefs[idx-1]
        dt = time - self._timeOffsets[idx-1]
        if dt < m.durationSecs:
            beat = dt*m.quarterTempo/F(60)
            return idx-1, beat
        # outside score
        if not self.endless:
            return None, F0
        lastMeas = self.measuredefs[-1]
        measDur = lastMeas.durationSecs
        numMeasures = dt / measDur
        beat = (numMeasures - int(numMeasures)) * lastMeas.durationQuarters
        return len(self.measuredefs)-1 + int(numMeasures), beat

    def beatToLocation(self, beat: num_t) -> tuple[int, F]:
        """
        Return the location in score corresponding to the given beat

        The beat is the time-offset in quarter-notes. Given a beat
        (in quarter-notes), return the score location
        (measure, beat offset within the measure). Tempo does not
        play any role within this calculation.

        Returns:
            a tuple (measure index, beat). Raises ValueError if beat
            is not defined within this score

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

        self._update()

        if not isinstance(beat, F):
            beat = asF(beat)

        if beat > self._beatOffsets[-1]:
            # past the end
            rest = beat - self._beatOffsets[-1]
            if not self.endless:
                if rest > 0:
                    raise ValueError(f"The given beat ({beat}) is outside the score")
                return (numdefs, F0)
            beatsPerMeasure = self.measuredefs[-1].durationQuarters
            idx = numdefs - 1
            idx += int(rest / beatsPerMeasure)
            restBeats = rest % beatsPerMeasure
            return idx, restBeats
        else:
            lastIndex = self._lastIndex
            lastOffset = self._beatOffsets[lastIndex]
            if lastOffset <= beat < lastOffset + self._quarternoteDurations[lastIndex]:
                idx = lastIndex
            else:
                ridx = bisect(self._beatOffsets, beat)
                idx = ridx - 1
                self._lastIndex = idx
            rest = beat - self._beatOffsets[idx]
            return idx, rest

    def b2t(self, beat: num_t) -> F:
        """Beat to time"""
        meas, beat = self.beatToLocation(beat)
        return self.locationToTime(meas, beat)

    def t2b(self, t: num_t) -> F:
        """Time to beat"""
        meas, beat = self.timeToLocation(t)
        if meas is None:
            raise ValueError(f"time {t} outside score")
        return self.locationToBeat(meas, beat)

    def beatToTime(self, beat: num_t) -> F:
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
        meas, offset = self.beatToLocation(beat)
        return self.locationToTime(meas, offset)

    def remapTo(self, deststruct: ScoreStruct, location: num_t | tuple[int, num_t]) -> F:
        """
        Remap a beat from this struct to another struct

        Args:
            location: the beat offset in quarternotes or a location (measureindex, offset)
            deststruct: the destination scores structure

        Returns:
            the beat within deststruct which keeps the same absolute time
        """
        abstime = self.time(location)
        return deststruct.timeToBeat(abstime)

    def remapSpan(self, sourcestruct: ScoreStruct, offset: num_t, duration: num_t
                  ) -> tuple[F, F]:
        """
        Remap a time span from a source score structure to this score structure

        Args:
            sourcestruct: the source score strcuture
            offset: the offset
            duration: the duration

        Returns:
            a tuple(offset, dur) where these represent the start and duration within this
            scorestruct which coincide in absolute time with the offset and duration given

        """
        starttime = sourcestruct.beatToTime(offset)
        endtime = sourcestruct.beatToTime(offset + duration)
        startbeat = self.timeToBeat(starttime)
        endbeat = self.timeToBeat(endtime)
        return startbeat, endbeat - startbeat

    def remapFrom(self, sourcestruct: ScoreStruct, location: num_t | tuple[int, num_t]) -> F:
        """
        Remap a beat from sourcestruct to this this struct

        Args:
            location: the beat offset in quarternotes or a location (measureindex, offset)
            sourcestruct: the source score structure

        Returns:
            the beat within this struct which keeps the same absolute time as
            the given beat within sourcestruct
        """
        abstime = sourcestruct.time(location)
        return self.timeToBeat(abstime)

    def timeToBeat(self, t: num_t) -> F:
        """
        Convert a time to a quarternote offset according to this ScoreStruct

        Args:
            t: the time (in absolute seconds)

        Returns:
            A quarternote offset

        will raise ValueError if the given time is outside this score structure

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
            raise ValueError(f"time {t} outside score")
        beat = self.locationToBeat(measureindex, measurebeat)
        return beat

    def iterMeasureDefs(self) -> Iterator[MeasureDef]:
        """
        Iterate over all measure definitions in this ScoreStruct.

        If it is marked as `endless`, then the last defined measure
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

    def beat(self, a: num_t | tuple[int, num_t], b: num_t | None = None
             ) -> F:

        """
        Convert a time in secs or a location (measure, beat) to a quarter-note beat

        Args:
            a: the time/location to convert. Either a time
            b: when passign a location, the beat within the measure (`a` contains
                the measure index)

        Returns:
            the corresponding quarter note beat according to this ScoreStruct

        Example
        ~~~~~~~

            >>> sco = ScoreStruct.fromTimesig('3/4', 120)
            # Convert time to beat
            >>> sco.beat(0.5)
            1.0
            # Convert score location (measure 1, beat 2) to beats
            >>> sco.beat((1, 2))
            5.0
            # Also supported, similar to the previous operation:
            >>> sco.beat(1, 2)
            5.0

        .. seealso:: :meth:`~ScoreSctruct.time`
        """
        if isinstance(a, tuple):
            assert b is None
            return self.locationToBeat(*a)
        elif b is not None:
            assert isinstance(a, int)
            return self.locationToBeat(a, b)
        else:
            return self.timeToBeat(a)

    def time(self, a: num_t | tuple[int, num_t], b: num_t | None = None
             ) -> F:
        """
        Convert a quarter-note beat or a location (measure, beat) to an absolute time in secs

        Args:
            a: the beat/location to convert. Either a beat, a tuple (measureindex, beat) or
                the measureindex itself, in which case `b` is also needed
            b: if given, then `a` is the measureindex and `b` is the beat

        Returns:
            the corresponding time according to this ScoreStruct

        Example
        ~~~~~~~

            >>> sco = ScoreStruct.fromTimesig('3/4', 120)
            # Convert time to beat
            >>> sco.time(1)
            0.5
            # Convert score location (measure 1, beat 2) to beats
            >>> sco.time((1, 2))
            2.5

        .. seealso:: :meth:`~ScoreSctruct.beat`

        """
        if isinstance(a, tuple):
            assert b is None
            return self.locationToTime(*a)
        elif b is not None:
            assert isinstance(a, int)
            return self.locationToTime(a, b)
        else:
            return self.beatToTime(a)

    def ltob(self, measure: int, beat: num_t) -> F:
        """
        A shortcut to locationToBeat

        Args:
            measure: the measure index (measures start at 0
            beat:  the beat within the given measure

        Returns:
            the corresponding beat in quarter notes
        """
        return self.locationToBeat(measure, beat)

    def asBeat(self, location: num_t | tuple[int, F]) -> F:
        """
        Given a beat or a location (measureidx, relativeoffset), returns an absolute beat

        Args:
            location: the location

        Returns:
            the absolute beat in quarter notes
        """
        return self.locationToBeat(*location) if isinstance(location, tuple) else location

    def locationToBeat(self, measure: int, beat: num_t = F(0)) -> F:
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
        self._update()
        beat = asF(beat)
        if measure < self.numMeasures():
            # Use the index
            measureOffset = self._beatOffsets[measure]
            quartersInMeasure = self._quarternoteDurations[measure]
            if beat > quartersInMeasure:
                raise ValueError(f"Measure {measure} has {quartersInMeasure} quarters, but given "
                                 f"offset {beat} is too large")
            return measureOffset + beat
        elif not self.endless:
            raise ValueError(f"This scorestruct has {self.numMeasures()} and is not"
                             f"marked as endless. Measure {measure} is out of scope")
        # It is endless and out of the defined measures
        # TODO

        accum = F(0)
        for i, mdef in enumerate(self.iterMeasureDefs()):
            if i < measure:
                accum += mdef.durationQuarters
            else:
                if beat > mdef.durationQuarters:
                    raise ValueError(f"beat {beat} outside measure {i}: {mdef}")
                accum += asF(beat)
                break
        return accum

    def measureOffsets(self, startIndex=0, stopIndex=0) -> list[F]:
        """
        Returns a list with the time offsets of each measure

        Args:
            startIndex: the measure index to start with. 0=last measure definition
            stopIndex: the measure index to end with (not included)

        Returns:
            a list of time offsets (start times), one for each measure in the
            interval selected
        """
        if not stopIndex:
            stopIndex = self.numMeasures()
        return [self.locationToBeat(idx) for idx in range(startIndex, stopIndex)]

    def measuresBetween(self, start: F, end: F) -> list[MeasureDef]:
        """
        List of measures defined between the given times as beats

        Args:
            start: start beat in quarter-tones
            end: end beat in quarter-tones

        Returns:

        """
        startloc = self.beatToLocation(start)
        idx0 = startloc[0]
        endloc = self.beatToLocation(end)
        idx1 = endloc[0] + 1
        return [self.getMeasureDef(idx=i) for i in range(idx0, idx1)]

    def timeDelta(self,
                  start: num_t | tuple[int, num_t],
                  end: num_t | tuple[int, num_t]
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
                  start: num_t | tuple[int, num_t],
                  end: num_t | tuple[int, num_t]) -> F:
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
        startBeat = self.beat(start)
        endBeat = self.beat(end)
        return endBeat - startBeat

    def show(self,
             fmt='png',
             app: str = '',
             scalefactor: float = 1.0,
             backend: str = None,
             renderoptions: RenderOptions = None
             ) -> None:
        """
        Render and show this ScoreStruct

        Args:
            fmt: the format to render to, one of 'png' or 'pdf'
            app: if given, the app used to open the produced document
            scalefactor: if given, a scale factor to enlarge or reduce the prduce image
            backend: the backend used (None to use a sensible default). If given, one of
                'lilypond' or 'musicxml'
            renderoptions: if given, these options will be used for rendering this
                score structure as image.

        Example
        ~~~~~~~

            >>> from maelzel.scorestruct import ScoreStruct
            >>> sco = ScoreStruct(r'''
            ... ...
            ... ''')
            >>> from maelzel.scoring.render import RenderOptions

        """
        import tempfile
        from maelzel.core import environment
        import emlib.misc

        outfile = tempfile.mktemp(suffix='.' + fmt)
        self.write(outfile, backend=backend, renderoptions=renderoptions)

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
        from maelzel.core import environment
        if environment.insideJupyter:
            from IPython.display import display, HTML
            display(HTML(self._repr_html_()))
        else:
            tempo = -1
            for i, m in enumerate(self.measuredefs):
                parts = []
                parts.append(str(m.timesig))
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
                if m.quarterTempo != tempo:
                    tempo = m.quarterTempo
                    parts.append(f"{m.timesig}@{tempo}")
                else:
                    parts.append(f"{m.timesig}")
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
        import emlib.misc
        colnames = ['Meas. Index', 'Timesig', 'Tempo (quarter note)', 'Label', 'Rehearsal', 'Barline']

        if any(m.keySignature is not None for m in self.measuredefs):
            colnames.append('Key')
            haskey = True
        else:
            haskey = False

        parts = [f'<p><strong>ScoreStruct</strong></p>']
        tempo = -1
        rows = []
        for i, m in enumerate(self.measuredefs):
            # num, den = m.timesig
            if m.quarterTempo != tempo:
                tempo = m.quarterTempo
                tempostr = ("%.3f" % tempo).rstrip("0").rstrip(".")
            else:
                tempostr = ""
            rehearsal = m.rehearsalMark.text if m.rehearsalMark is not None else ''
            timesig = m.timesigRepr()
            if m.timesigInherited:
                timesig = f'({timesig})'
            row = [str(i), timesig, tempostr, m.annotation or "", rehearsal, m.barline]
            if haskey:
                row.append(str(m.keySignature.fifths) if m.keySignature else '-')
            rows.append(row)
        if self.endless:
            rows.append(("...", "", "", "", "", ""))
        rowstyle = 'font-size: small;'
        htmltable = emlib.misc.html_table(rows, colnames, rowstyles=[rowstyle]*len(colnames))
        parts.append(htmltable)
        return "".join(parts)

    def _render(self, backend: str = None, renderoptions: RenderOptions = None
                ) -> Renderer:
        from maelzel import scoring
        quantprofile = scoring.quant.QuantizationProfile()
        measures = [scoring.quant.QuantizedMeasure(timesig=m.timesig, quarterTempo=m.quarterTempo,
                                                   quantprofile=quantprofile, beats=[])
                    for m in self.measuredefs]
        part = scoring.quant.QuantizedPart(struct=self, measures=measures, quantprofile=quantprofile)
        qscore = scoring.quant.QuantizedScore([part], title=self.title, composer=self.composer)
        if not renderoptions:
            renderoptions = scoring.render.RenderOptions()
        if backend:
            renderoptions.backend = backend
        return scoring.render.renderQuantizedScore(qscore, options=renderoptions)

    def setTempo(self, tempo: float, reference=1, measureIndex: int = 0) -> None:
        """
        Set the tempo of the given measure, until the next tempo change

        Args:
            tempo: the new tempo
            reference: the reference duration (1=quarternote, 2=halfnote, 0.5: 8th note, etc)
            measureIndex: the first measure to modify

        """
        if self.readonly:
            raise RuntimeError("This ScoreStruct is read-only")

        if measureIndex > len(self) and not self.endless:
            raise IndexError(f"Index {measureIndex} out of rage; this ScoreStruct has only "
                             f"{len(self)} measures defined")
        quarterTempo = asF(tempo) / asF(reference)
        mdef = self.getMeasureDef(measureIndex, extend=True)
        mdef.quarterTempo = quarterTempo
        mdef.tempoInherited = False
        for m in self.measuredefs[measureIndex+1:]:
            if m.tempoInherited:
                m.quarterTempo = quarterTempo
            else:
                break

    def setTimeSignature(self, measureIndex, timesig: tuple[int, int] | str | TimeSignature
                         ) -> None:
        if self.readonly:
            raise RuntimeError("This ScoreStruct is read-only")

        if measureIndex > len(self) and not self.endless:
            raise IndexError(f"Index {measureIndex} out of rage; this ScoreStruct has only "
                             f"{len(self)} measures defined")
        timesig = _asTimeSignature(timesig)
        mdef = self.getMeasureDef(measureIndex, extend=True)
        mdef.timesig = timesig
        mdef.timesigInherited = False
        for m in self.measuredefs[measureIndex + 1:]:
            if m.timesigInherited:
                m.timesig = timesig
            else:
                break

    def modified(self, attributes=False, timing=False) -> None:
        """
        Notify this ScoreStruct that some of its measure definitions have been modified

        Args:
            attributes: if True, some attributes of some measuredefs have been
                modified (barline style, subdivision, etc)
            timing: if True, any aspect modifying timing has been altered (tempo, time
                signature, new measures, etc). This forces the update of the internal index

        Example
        ~~~~~~~

            >>> from maelzel.scorestruct import ScoreStruct
            >>> s = ScoreStruct(r'''
            ... 4/4, 60
            ... 3/4
            ... 4/4
            ... .
            ... .
            ... 5/8
            ... ''')
            >>> measure = s.getMeasureDef(3)
            >>> measure.timesig = 6/8
            >>> s.modified()
        """
        self._attributesModified = attributes
        self._timingModified = timing
        self._hash = None

    def _fixInheritedAttributes(self):
        m0 = self.measuredefs[0]
        timesig = m0.timesig
        tempo = m0.quarterTempo
        for m in self.measuredefs[1:]:
            if m.tempoInherited:
                m._quarterTempo = tempo
            else:
                tempo = m._quarterTempo
            if m.timesigInherited:
                m._timesig = timesig
            else:
                timesig = m._timesig

    def hasUniqueTimesig(self) -> bool:
        """
        Returns True if this ScoreStruct does not have any time-signature change
        """
        lastTimesig = self.measuredefs[0].timesig
        for m in self.measuredefs:
            if m.timesig != lastTimesig:
                return False
        return True

    def write(self,
              path: str | Path,
              backend: str = None,
              renderoptions: RenderOptions = None
              ) -> None:
        """
        Export this score structure

        Write this as musicxml (.xml), lilypond (.ly), MIDI (.mid) or render as
        pdf or png. The format is determined by the extension of the file. It is
        also possible to write the score as text (in its own format) in order to
        load it later (.txt)

        .. note:: when saving as MIDI, notes are used to fill each beat because an empty
            MIDI score is not supported by the MIDI standard

        Args:
            path: the path of the written file
            backend: for pdf or png only - the backend to use for rendering, one
                of 'lilypond' or 'musicxml'
            renderoptions: if given, they will be used to customize the rendering
                process.
        """
        self._update()
        path = Path(path)
        if path.suffix == ".xml":
            raise ValueError("musicxml output is not supported yet")
        elif path.suffix in (".pdf", '.png', '.ly'):
            r = self._render(backend=backend, renderoptions=renderoptions)
            r.write(str(path))
        elif path.suffix == '.mid' or path.suffix == '.midi':
            sco = _filledScoreFromStruct(self)
            sco.write(str(path))
        elif path.suffix == '.txt':
            text = self.asText()
            with open(path, 'w') as f:
                f.write(text)
        else:
            raise ValueError(f"Extension {path.suffix} not supported, "
                             f"should be one of .xml, .pdf, .png or .ly")

    def exportMidiClickTrack(self, midifile: str) -> None:
        """
        Generate a MIDI click track from this ScoreStruct

        Args:
            midifile: the path of the MIDI file to generate

        .. seealso:: :func:`maelzel.core.clicktrack.makeClickTrack`
        """
        from maelzel.core import clicktrack
        click = clicktrack.makeClickTrack(self)
        ext = Path(midifile).suffix.lower()
        if ext != '.mid' and ext != '.midi':
            raise ValueError(f"Expected a .mid or .midi extension, got {ext} ({midifile})")
        click.write(midifile)

    def setEnd(self, numMeasures: int) -> None:
        """
        Set an end measure to this ScoreStruct

        If the scorestruct has less defined measures as requested, then it is extended
        by duplicating the last defined measure as needed. Otherwise, the scorestruct is
        cropped. The scorestruct ceases to be endless if that was the case previously

        Args:
            numMeasures: the requested number of measures after the operation

        """
        self.endless = False
        if numMeasures < len(self.measuredefs):
            self.measuredefs = self.measuredefs[:numMeasures]
            self._timingModified = True
        else:
            last = self.measuredefs[-1]

            self.addMeasure(timesig=last.timesig,
                            quarterTempo=last.quarterTempo,
                            subdivisions=last.subdivisionStructure,
                            keySignature=last.keySignature,
                            numMeasures=numMeasures - len(self.measuredefs))

    def setBarline(self, measureIndex: int, linetype: str) -> None:
        """
        Set the right barline type

        Args:
            measureIndex: the measure index to modify
            linetype: one of 'single', 'double', 'final', 'solid', 'dashed'

        """
        if self.readonly:
            raise RuntimeError("This ScoreStruct is read-only")
        assert linetype in _barstyles, f"Unknown style '{linetype}', possible styles: {_barstyles}"
        self.getMeasureDef(measureIndex, extend=True).barline = linetype

    def asText(self) -> str:
        """
        This ScoreStruct as parsable text format

        Returns:
            this score as text
        """
        lines = []
        for i, measuredef in enumerate(self.measuredefs):
            line = measuredef.asScoreLine()
            lines.append(f'{i}, {line}')
        if self.endless:
            lines.append('...')
        return '\n'.join(lines)

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
                measure)
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
        from maelzel.core.clicktrack import makeClickTrack
        if minMeasures < self.numMeasures():
            struct = self
        else:
            struct = self.copy()
            struct.ensureDurationInMeasures(minMeasures)
        score = makeClickTrack(struct,
                               clickdur=clickdur,
                               strongBeatPitch=strongBeatPitch,
                               weakBeatPitch=weakBeatPitch)
        score.setPlay(itransp=playTransposition)
        return score


def _filledScoreFromStruct(struct: ScoreStruct, pitch='4C') -> maelzel.core.Score:
    """
    Creates a :class:`maelzel.core.score.Score` representing the given ScoreStruct

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
    if isinstance(pitch, (int, float)):
        midinote = float(pitch)
    else:
        midinote = pitchtools.n2m(pitch)
    for i, measuredef in enumerate(struct.measuredefs):
        dur = measuredef.durationQuarters
        if i == len(struct.measuredefs) - 1:
            events.append(Note(pitch=midinote, offset=now, dur=dur))
        now += dur
    voice = Voice(events)
    return Score([voice], scorestruct=struct)
