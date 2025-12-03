from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from bisect import bisect
import functools
import re
import copy as _copy

import emlib.textlib
from maelzel.common import F, asF, F0

from typing import TYPE_CHECKING, overload as _overload
if TYPE_CHECKING:
    from typing import Iterator, Iterable, Any
    from typing_extensions import Self
    import maelzel.core
    from maelzel.common import num_t, timesig_t, beat_t
    from maelzel.scoring.renderoptions import RenderOptions
    from maelzel.scoring.renderer import Renderer


__all__ = (
    'ScoreStruct',
    'MeasureDef',
    'measureBeatStructure',
    'TimeSignature',
    'convertTempo',
    'figureDuration'
)


_powersof2 = (1, 2, 4, 8, 16, 32, 64, 128)


def _partialsum(seq: Iterable[F], start=F0) -> list[F]:
    """
    for each elem in seq return the partial sum

    .. code::

        n0 -> n0
        n1 -> n0 + n1
        n2 -> n0 + n1 + n2
        n3 -> n0 + n1 + n2 + n3
    """
    accum = start
    out = []
    for i in seq:
        accum += i
        out.append(accum)
    return out


class TimeSignature:
    """
    A time signature

    In its simplest form a type signature consists of one part, a tuple
    (numerator, denominator). For example, a 4/4 time signature can be
    represented as TimeSignature((4, 4))

    Args:
        parts: the parts of this time signature, a seq. of tuples (numerator, denominator).
            Use TimeSignature.parse to parse a signature given as string
        subdivisions: subdivisions as multiples of the denominator. Only valid for
            non-compound signatures. It is used to structure subdivisions for a single
            part. For example, 7/8 subdivided as 2+3+2 can be expressed as
            TimeSignature((7, 8), subdivisionStruct=(2, 3, 2)).
    """
    def __init__(self,
                 *parts: tuple[int, int],
                 subdivisions: tuple[int, ...] = ()):

        self.parts: tuple[tuple[int, int], ...] = parts
        """
        The parts of this timesig, a seq. of tuples (num, den)
        """
        assert isinstance(subdivisions, tuple)

        minden = max(den for num, den in parts)
        numerators = [num * minden // den for num, den in parts]

        self.normalizedParts: tuple[tuple[int, int], ...] = tuple((num, minden) for num in numerators)
        """
        The normalized parts with a shared common denominator

        parts: (3/4 3/8), common den: 8, normalizedParts: (6/8, 3/8), fusedSignature: 9/8
        """

        self.fusedSignature: tuple[int, int] = (int(sum(numerators)), minden)
        """
        One signature representing all compound parts

        The fused signature is based on the min. common multiple of the compound parts.
        For example, a signature 3/4+3/16 will have a fused signature of 15/16 (3*4+3).
        For non-compound signatures, the fused signature is the same as the time signature
        itself."""

        self.subdivisionStruct: tuple[int, ...] = subdivisions
        """
        Subdivisions as multiples of the fused denominator.
        """

        if subdivisions and sum(subdivisions) != self.fusedSignature[0]:
            raise ValueError(f"Invalid subdivision structure: {subdivisions}, parts={parts}, "
                             f"{self.fusedSignature=}")

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
    def duration(self) -> F:
        """The duration of this time signature, in quarternotes"""
        num, den = self.fusedSignature
        return F(num, den) * 4

    def __str__(self):
        parts = [f"{num}/{den}" for num, den in self.parts]
        return "+".join(parts)

    def _reprInfo(self) -> str:
        if len(self.parts) == 1:
            num, den = self.parts[0]
            if self.subdivisionStruct:
                subdiv = '-'.join(map(str, self.subdivisionStruct))
                return f"{num}/{den}({subdiv})"
            return f"{num}/{den}"
        elif all(den == self.parts[0][1] for num, den in self.parts):
            nums = "+".join(str(p[0]) for p in self.parts)
            return f"{nums}/{self.parts[0][1]}"
        else:
            return '+'.join(f"{n}/{d}" for n, d in self.parts)

    def __repr__(self):
        return f"TimeSignature({self._reprInfo()})"

    @classmethod
    def parse(cls, timesig: str | tuple, subdivisions: tuple[int, ...] = ()
              ) -> TimeSignature:
        """
        Parse a time signature definition

        Args:
            timesig: a time signature as a string. For compound signatures, use
                a + sign between parts.
            subdivisions: the subdivision structure as multiples of the
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
                    return TimeSignature(*parts, subdivisions=subdivisions)
                else:
                    assert isinstance(num, int)
                    return TimeSignature((num, den), subdivisions=subdivisions)
            else:
                raise ValueError(f"Cannot parse timesignature: {timesig}")
        elif isinstance(timesig, str):
            # Possible signatures: 3/4, 3/8+3/8+2/8, 5/8(3-2), 5/8(3+2)+3/16
            parts = re.split(r"\+(?![^(]*\))", timesig)
            parsedParts = [_parseTimesigPart(part) for part in parts]
            if len(parsedParts) == 1:
                signature, subdivs = parsedParts[0]
                if subdivs and subdivisions:
                    raise ValueError("Duplicate subdivision structure")
                return TimeSignature(signature, subdivisions=subdivs or subdivisions)
            signatures, subdivs = zip(*parsedParts)
            # We ignore subdivisions for compound signatures
            return TimeSignature(*signatures, subdivisions=subdivisions)

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
            raise ValueError("This time signature does not have a subdivision structure")
        return self.fusedSignature[1], self.subdivisionStruct


def _parseTimesigPart(s: str) -> tuple[tuple[int, int], tuple[int, ...]]:
    """
    Given a string in the form 5/8(3-2), returns ((5, 8), (3, 2))

    Possible parts: 5/8, 5/8(3-2), 5/8(3+2),

    For 5/8, returns ((5, 8), ())
    """
    if "(" in s:
        assert s.count("(") == 1 and s[-1] == ")", f"Invalid time signature part: {s}"
        p1, p2 = s[:-1].split("(")
        nums, dens = p1.split("/")
        num, den = int(nums), int(dens)
        subdivparts = re.split(r"[+\-]", p2)
        subdivs = tuple(int(subdiv) for subdiv in subdivparts)
        return ((num, den), subdivs)
    else:
        fracparts = s.split("/")
        if len(fracparts) != 2:
            raise ValueError(f"Invalid time signature: {s}")
        nums, dens = fracparts
        return ((int(nums), int(dens)), ())


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


@dataclass
class _ScoreLine:
    measureidx: int | None
    timesig: TimeSignature | None
    tempodef: TempoDef | None
    label: str = ''
    barline: str = ''
    mark: str = ''


@dataclass
class Mark:
    text: str
    box: str = ''

    def __post_init__(self):
        assert self.text

    def __repr__(self):
        parts = ['"' + self.text + '"']
        if self.box:
            parts.append(f"box={self.box}")
        return f"Mark({", ".join(parts)})"


@dataclass
class TempoDef:
    tempo: F
    base: int = 4
    dots: int = 0

    @property
    def quarterTempo(self) -> F:
        return asQuarterTempo(self.tempo, base=self.base, dots=self.dots)

    def __iter__(self):
        return iter((self.tempo, self.base, self.dots))


@dataclass
class KeySignature:
    fifths: int
    mode: str = 'major'

    def __post_init__(self):
        assert isinstance(self.fifths, int) and -7 <= self.fifths <= 7
        assert self.mode in ('', 'major', 'minor'), f"Invalid mode: {self.mode}"


@functools.cache
def asQuarterTempo(tempo: F, base: int, dots: int = 0) -> F:
    """
    Convert a generic tempo to a quarternote tempo

    Args:
        tempo: tempo value
        base: base duration, where 4=quarternote, 8=8th note, etc.
        dots: number of dots

    Returns:
        The tempo corresponding to a quarter note
    """
    return convertTempo(tempo, source=(base, dots), dest=(4, 0))


def convertTempo(tempo: F | int | float,
                 source: tuple[int, int] | F | int,
                 dest: tuple[int, int] | F | int) -> F:
    """
    Convert a tempo from a source reference (base, dots) to a destination reference

    Args:
        tempo: the source tempo
        source: source reference, a tuple (base, dots)
        dest: destination reference

    Returns:
        the tempo corresponding to the new reference

    Example
    -------

        >>> convertTempo(60, (4, 0), (8, 0))
        120
    """
    sourcefig = _asFigure(source)
    destfig = _asFigure(dest)
    sourcedur = figureDuration(*sourcefig)
    destdur = figureDuration(*destfig)
    return tempo * sourcedur / destdur


def _asFigure(fig: tuple[int, int] | F | int) -> tuple[int, int]:
    if isinstance(fig, tuple):
        return fig
    elif isinstance(fig, F):
        return durationToFigure(fig)
    elif isinstance(fig, int):
        return (fig, 0)
    elif isinstance(fig, float):
        return durationToFigure(F(fig))
    else:
        raise ValueError(f"Cannot convert to a figure: {fig}")



def _parseTempoRefvalue(ref: str) -> tuple[int, int]:
    if "." not in ref:
        refvaluestr = ref
        numdots = 0
    else:
        try:
            refvaluestr, dots = ref.split(".", maxsplit=1)
            numdots = len(dots) + 1
        except ValueError as e:
            raise ValueError(f"Could not parse tempo: '{ref}'") from e

    try:
        refvalue = int(refvaluestr)
    except ValueError as e:
        raise ValueError(f"Could not parse tempo '{ref}', invalid reference value '{refvaluestr}'") from e

    if refvalue not in _powersof2:
        raise ValueError(f"Could not parse tempo: '{ref}', reference value {refvalue} should be a power of 2")
    return refvalue, numdots


def _parseTempo(s: str) -> TempoDef:
    """
    Parse a tempo

    =======    ===== ========== ======= ==============
    Value      tempo  refvalue  numdots quartertempo
    =======    ===== ========== ======= ==============
    60         60      4         0       60
    4=60       60      4         0       60
    8=60       60      8         0       120
    4.=60      60      4         1       90
    4..=60     60      4         2       112.5
    =======    ===== ========== ======= ==============


    Args:
        s: the string to parse

    Returns:
        A TempoDef

    .. code-block:: python

        >>> _parseTempo('8=60').quartertempo
        120
        >>> _parseTempo('4.=60').quartertempo
        90
    """
    if "=" not in s:
        try:
            tempo = F(s)
            return TempoDef(tempo, 4, 0)
        except ValueError as e:
            raise ValueError(f"Could not parse tempo: {s}") from e

    parts = [_.strip() for _ in s.split("=")]
    if len(parts) != 2:
        raise ValueError(f"Could not parse tempo: {s}")
    ref, tempostr = parts
    tempo = float(tempostr)
    refvalue, numdots = _parseTempoRefvalue(ref)
    return TempoDef(F(tempo), refvalue, numdots)


def _parseLine(line: str) -> _ScoreLine:
    """
    parse a line of a ScoreStruct definition

    The line has the format ``[measureidx, ] timesig [, tempo] [keywords]

    * timesig has the format ``num/den``
    * keywords have the format ``keyword=value``, where keyword can be one of
        ``rehearsalmark`` or ``mark``, ``label`` and ``barline`` (case is not important).

    * *rehearsalmark* / *mark* adds a rehearsal mark
    * *label* adds a text label and
    * *barline* customizes the right barline
      (possible values: 'single', 'double', 'solid', 'dotted' or 'dashed')

    Args:
        line: a line of the format [measureidx, ] timesig [, tempo]

    Returns:
        a tuple (measureidx, timesig, tempo), where only timesig
        is required
    """
    line = line.strip()
    tempodef: TempoDef | None = None
    parts = [_.strip() for _ in line.split(",")]
    # The first argument is special, because it is the only one
    # which can be an index
    measureidx: int | None = None
    timesig = ''
    kwargs = {}
    keywords = ('rehearsalmark', 'mark', 'label', 'barline')

    for i, part in enumerate(parts):
        if i == 0 and part.isdecimal():
            measureidx = int(part)
        elif '/' in part:
            # A time signature
            timesig = part
        elif i > 0 and part.isdecimal():
            # A tempo without reference figure
            # TODO: allow tempos with a fractional part
            tempodef = TempoDef(F(int(part), 1))
        elif "=" in part:
            # Either a tempo or a keyword=value pair
            key, val = part.split('=', maxsplit=1)
            key = key.strip().lower()
            if re.match(r"(1|2|4|8|16|32|64)\.*", key):
                refvalue, numdots = _parseTempoRefvalue(key)
                tempodef = TempoDef(tempo=F(val), base=refvalue, dots=numdots)
            else:
                val = val.strip()
                if val[0] == val[-1] in "'\"":
                    val = val[1:-1]

                if key not in keywords:
                    raise ValueError(f"Key '{key}' not recognized, possible keys: {keywords}")
                val = val.replace('"', '')
                if key == 'rehearsalmark' or key == 'mark':
                    kwargs['mark'] = val
                else:
                    kwargs[key] = val
        else:
            raise ValueError(f"Could not parse {part!r} in line '{line}'")

    return _ScoreLine(measureidx=measureidx,
                      timesig=TimeSignature.parse(timesig) if timesig else None,
                      tempodef=tempodef,
                      **kwargs)


_barstyles = {'single', 'final', 'double', 'solid', 'dotted', 'dashed', 'tick', 'short',
              'double-thin', 'none'}


class MeasureDef:
    """
    A MeasureDef defines one Measure within a ScoreStruct (time signature, tempo, etc.)

    Args:
        timesig, the TimeSignature
        tempo: tempo value for this measure.
        parent: the parent ScoreStruct, if applicable
        annotation: a text annotation for this measure
        timesigInherited: is the time signature inherited from previous measures?
        tempoInherited: is the tempo inherited?
        barline: a barline for this measure, one of 'single', 'final', 'double', 'solid', 'dotted' or 'dashed'
        mark: rehearsal mark
        key: the key signature for this measure
        properties: user defined properties, any key: value pair is allowed
        const: make this measure read-only, any modification raises an error
        tempoRef: the figure corresponding to the given tempo. A tuplet (base, dots),
            where base is 4=quarter, 8=8th note, etc., and dots indicates the number of dots
        subdivTempo: max. tempo at which a subdivision can be a beat of its own
        breakTempo: tempo threshold at which syncopations over a beat are broken.
            This is a quarter note tempo
    """
    __slots__ = (
        '_timesig',
        '_quarterTempo',
        '_barline',
        '_tempo',
        'annotation',
        'timesigInherited',
        'tempoInherited',
        'mark',
        'key',
        'properties',
        'parent',
        'const',
        'duration',
        'tempoRef',
        '_subdivTempo',
        '_breakTempo',
        '_durationSecs'
    )

    def __init__(self,
                 timesig: TimeSignature,
                 tempo: F | int | float,
                 annotation='',
                 barline='',
                 mark: Mark | str = '',
                 tempoRef: tuple[int, int] = (4, 0),
                 key: KeySignature | None = None,
                 parent: ScoreStruct | None = None,
                 timesigInherited=False,
                 tempoInherited=False,
                 properties: dict | None = None,
                 subdivTempo: int | None = None,
                 breakTempo: int | None = None,
                 const=True,
                 ):
        if barline and barline not in _barstyles:
            raise ValueError(f"Unknown barline style: '{barline}', possible values: {_barstyles}")

        if mark and isinstance(mark, str):
            mark = Mark(mark)

        assert isinstance(timesig, TimeSignature), f"Expected a TimeSignature, got {timesig}"
        self._timesig: TimeSignature = timesig
        self._tempo = asF(tempo)
        self._quarterTempo = asQuarterTempo(self._tempo, tempoRef[0], tempoRef[1])
        self.annotation = annotation
        """Any text annotation for this measure"""

        self.timesigInherited = timesigInherited
        """Is the time-signature of this measure inherited?"""

        self.tempoInherited = tempoInherited
        """Is the tempo of this measure inherited?"""

        self._barline = barline
        """The barline style, or '' to use default"""

        self.mark: Mark | None = mark or None
        """If given, a RehearsalMark for this measure"""

        self.key = key
        """If given, a key signature"""

        self.properties = properties
        """User defined properties can be placed here. None by default"""

        self.parent = parent
        """The parent ScoreStruct of this measure, if any"""

        self.const = const
        """Is this measure read only?"""

        self._subdivTempo: int | None = subdivTempo
        """The max. tempo at which an eighth note can be a beat of its own"""

        self._breakTempo: int | None = breakTempo

        self.duration: F = self.timesig.duration
        """Measure duration in quarters"""

        self._durationSecs: F = F0

        self.tempoRef: tuple[int, int] = tempoRef
        """Reference figure for the tempo, as a tuple (base, numdots), where base 4=quarternote"""

    @property
    def durationSecs(self) -> F:
        """The duration of this measure in seconds"""
        if not (dur := self._durationSecs):
            self._durationSecs = dur = self.duration * (F(60) / self._quarterTempo)
        return dur

    @property
    def timesig(self) -> TimeSignature:
        """The time signature of this measure. Can be explicit or inherited"""
        return self._timesig

    @timesig.setter
    def timesig(self, timesig):
        if self.const:
            raise ValueError("This MeasureDef is readonly")
        self._timesig = TimeSignature.parse(timesig)
        self.timesigInherited = False
        if self.parent:
            self.parent.modified()

    @property
    def quarterTempo(self) -> F:
        """The tempo relative to a quarternote"""
        return self._quarterTempo

    @quarterTempo.setter
    def quarterTempo(self, tempo: F | int):
        """Set the tempo in relation to a quarter note"""
        ratio = asF(tempo) / self._quarterTempo
        self.setTempo(tempo=self._tempo * ratio)

    @property
    def tempo(self) -> F:
        """The tempo of this measure, corresponding to the rhythmic figure in .temporef"""
        return self._tempo

    @tempo.setter
    def tempo(self, tempo: F | int | tuple[F | int, tuple[int, int]]):
        """Modify the tempo of this measure

        If a tempo is given, the tempo reference stays unmodified.
        Alternatively a tuple (tempo: int, temporef: tuple[int, int]) can
        can be given, in which case both tempo value and reference are
        modified
        """
        if isinstance(tempo, tuple):
            tempoval, temporef = tempo
            self.setTempo(tempo=tempoval, reference=temporef)
        else:
            self.setTempo(tempo=tempo)

    def tempoDef(self) -> TempoDef:
        """Tempo definition for this measure (.tempo, base., .dots)"""
        refvalue, numdots = self.tempoRef
        return TempoDef(tempo=self._tempo, base=refvalue, dots=numdots)

    def _setTempo(self, tempo: F, reference: tuple[int, int], inherited: bool) -> None:
        self.tempoRef = reference
        base, dots = self.tempoRef
        self._tempo = tempo
        self._quarterTempo = asQuarterTempo(tempo, base, dots)
        self.tempoInherited = inherited

    def setTempo(self, tempo: F | int | float, reference: tuple[int, int] | F | None = None
                 ) -> None:
        """Set the tempo of this measure

        Args:
            tempo: tempo value corresponding to the rhythmic figure in reference
            reference: rhythmic figure as base for the tempo value. A quarterntote is
                given as (4, 0), a dotted 8th note as (8, 1). A Fraction can be
                given instead with the duration of the rhythmic value as a fraction
                of the quarter note (quarter=F(1), dotted 8th note=F(3, 4))
        """
        if self.const:
            raise ValueError("This MeasureDef is readonly")
        if not reference:
            reference = self.tempoRef
        elif isinstance(reference, F):
            reference = durationToFigure(reference)
        self._setTempo(asF(tempo), reference, inherited=False)
        if self.parent:
            self.parent.modified()

    @property
    def barline(self) -> str:
        """The barline style, or '' to use default"""
        return self._barline

    @barline.setter
    def barline(self, linestyle: str):
        if self.const:
            raise ValueError("This MeasureDef is readonly")
        if linestyle not in _barstyles:
            raise ValueError(f'Unknown barstyle: {linestyle}, possible values: {_barstyles}')
        self._barline = linestyle
        
    def subdivTempoThresh(self, fallback=96) -> int:
        """
        Resolves the subdivision tempo threshold

        This is the tempo at which a subdivision of the beat (as given
        in the denominator of the time signature) can become a beat
        itself. First the value specified for this measure is used,
        if given; otherwise the value specified for the score or
        a fallback

        Returns:
            the tempo threshold

        """
        if self._subdivTempo:
            return self._subdivTempo
        elif self.parent and self.parent.subdivTempo:
            return self.parent.subdivTempo
        return fallback

    def beatWeightTempoThresh(self, fallback=52) -> int:
        """
        Resolves the beat-weight tempo threshold

        This is the tempo at which the weight acquires extra weight.
        This is used in the context of notation

        Args:
            fallback: tempo used if no specification was given, either for this
                measure or for the score itself

        Returns:
            the tempo threshold at which a beat has a higher weight

        """
        assert isinstance(fallback, (int, float, F))
        if self._breakTempo:
            return self._breakTempo
        if self.parent and self.parent.breakTempo:
            return self.parent.breakTempo
        return fallback

    def beatStructure(self) -> list[BeatDef]:
        """
        Beat structure of this measure

        Returns:
            a list of tuple with the form (beatOffset: F, beatDur: F, beatWeight: int)
            for each beat of this measure
        """
        return measureBeatStructure(self.timesig,
                                    tempo=self.tempo,
                                    tempoRef=self.tempoRef,
                                    subdivisions=self.subdivisionStructure(),
                                    subdivTempo=self.subdivTempoThresh(),
                                    breakTempo=self.beatWeightTempoThresh())

    def asScoreLine(self) -> str:
        """
        The representation of this MeasureDef as a score line
        """
        num = self.timesig.numerator
        den = self.timesig.denominator
        base, dots = self.tempoRef
        if base == 4 and dots == 0:
            parts = [f'{num}/{den}, {self.tempo}']
        else:
            fig = str(base)
            if dots:
                fig += "." * dots
            parts = [f'{num}/{den}, {fig}={self.tempo}']
        if self.annotation:
            parts.append(self.annotation)
        return ', '.join(parts)

    def __copy__(self):
        return _copy.deepcopy(self)

    def copy(self) -> MeasureDef:
        return self.__copy__()

    def __repr__(self):
        parts = []
        timesig = str(self._timesig)
        parts.append(timesig)
        fig = unicodeDuration(self.tempoRef)
        parts.append(f"{fig}={self._tempo}")
        if self.tempoRef != (4, 0):
            parts.append(f'quarterTempo={self._quarterTempo}')
        if self.annotation:
            parts.append(f'annotation="{self.annotation}"')
        if self.timesigInherited:
            parts.append('timesigInherited✓')
        if self.tempoInherited:
            parts.append('tempoInherited✓')
        if self.barline:
            parts.append(f'barline={self.barline}')
        if self.key:
            parts.append(f'keySignature={self.key}')
        if self.mark:
            parts.append(f'mark={self.mark}')
        if self.const:
            parts.append('const✓')
        return f'MeasureDef({", ".join(parts)})'

    def __hash__(self) -> int:
        return hash((self.timesig, self._tempo, self.tempoRef, self.annotation))

    def subdivisionStructure(self) -> tuple[int, tuple[int, ...]]:
        """
        Max. common denominator for subdivisions and the subdivisions as multiples of it

        For example, for 3/4+3/8, returns (8, (2, 2, 2, 3))

        Returns:
            a tuple (max. common denominator, subdivisions as multiples of common denominator)

        """
        subdivdurs = self.subdivisions()
        maxdenom = max(dur.denominator for dur in subdivdurs)
        commondur = F(1, maxdenom)
        nums = [dur / commondur for dur in subdivdurs]
        assert all(num.denominator == 1 for num in nums)
        multiples = tuple(num.numerator for num in nums)
        struct = (commondur/4).denominator, multiples
        _checkSubdivisionStructure(struct, self.timesig)
        return struct

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
        return measureSubdivisions(timesig=self.timesig,
                                   tempo=self.tempo,
                                   tempoRef=self.tempoRef,
                                   subdivTempo=self.subdivTempoThresh())

    def timesigRepr(self) -> str:
        """
        Returns a string representation of this measure's time signature

        Returns:
            a string representation of this measure's time-signature

        """
        return self.timesig._reprInfo()


@functools.cache
def figureDuration(base: int, dots: int) -> F:
    """
    Convert a notated figure to a duration in quarter notes

    Args:w
        base: the base duration, where 4=quarter note, 8=eighth note, etc
        dots: number of dots

    Returns:
        the duration in quarter notes


    =====  =====  =========
    base   dots   duration
    =====  =====  =========
    4      0      1
    4      1      3/2
    4      2      7/4
    4      3      15/8
    8      0      1/2
    8      1      1/2 * 3/2
    =====  =====  =========


    """
    refdur = F(4, base)
    if dots > 0:
        den = 2 ** dots
        num = (2 ** (dots + 1)) - 1
        refdur = refdur * num / den
    return refdur


def _subdivRepr(f: F, timesigDen: int) -> str:
    f = f * (timesigDen // 4)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def inferSubdivisions(num: int, den: int,
                      tempo: F,
                      tempoRef=(4, 0),
                      minTempoBinary=48,
                      minTempoTernary: int = 0,
                      maxTempo=96
                      ) -> tuple[int, ...]:
    """
    Infer the subdivisions of a measure

    Args:
        num: numerator of the time signature. Compound time signatures are not
            supported but the parts of the time signature can be passed to
            this function
        den: denominator of the time signautre
        tempo: tempo value
        tempoRef: reference ryhthmic figure for the given tempo
        minTempoBinary: min. quarter tempo for a binary beat. A beat resulting
            in a tempo slower than this will be subdivided in two
        minTempoTernary: min. quarter tempo for a ternary beat. A beat resulting
            in a tempo slower than this will be subdivided in three. If not given, the same
            value passed to minBinaryTempo is used (adapted to a ternary beat)
        maxTempo: if the pulse quarter tempo is slower than this value, the measure
            is split into the pulses as indicated in the time signature. For example,
            for a time signature of 7/8, if the tempo of an 8th note is slower
            than `maxTempo`, the measure is subdivided in 7 8th note beats. Any
            tempo faster than this value will group beats together, either as 2+2+1+1+1
            or 2+2+3, depending on the tempo.

    Returns:
        the subdivisions, as multiples of the denominator

    """
    # TODO: make this more clever...
    if not minTempoTernary:
        minTempoTernary = minTempoBinary * 2 / 3

    qtempo = convertTempo(tempo, tempoRef, (4, 0))
    pulsetempo = convertTempo(qtempo, (4, 0), (den, 0))

    if minTempoBinary <= pulsetempo <= maxTempo:
        out = tuple([1] * num)
    elif num % 2 == 0 and pulsetempo / 2 >= minTempoBinary:
        out = tuple([2] * (num // 2))
    elif num % 3 == 0 and pulsetempo / 3 >= minTempoTernary:
        out = tuple([3] * (num // 3))
    else:
        subdivs = []
        while num > 3:
            subdivs.append(2)
            num -= 2
        if num:
            assert num == 1 or num == 3
            if num == 3 and pulsetempo / 3 >= minTempoTernary:
                subdivs.append(3)
            else:
                subdivs.extend([1] * num)
        out = tuple(subdivs)
    return out


@functools.cache
def measureDuration(timesig: timesig_t | TimeSignature) -> F:
    """
    The duration of a measure in quarter notes, based on its time signature

    Args:
        timesig: a tuple (num, den)

    Returns:
        the duration in quarter notes

    Examples::

        >>> measureDuration((3,4))
        3
        >>> measureDuration((5, 8))
        2.5

    """
    if isinstance(timesig, TimeSignature):
        return timesig.duration
    num, den = timesig
    return F(num)/den * 4


def _checkSubdivisionStructure(s: tuple[int, tuple[int, ...]],
                               timesig: TimeSignature | timesig_t | None = None,
                               ) -> None:
    assert isinstance(s, tuple) and len(s) == 2, s
    assert isinstance(s[0], int) and isinstance(s[1], tuple), s
    assert all(isinstance(div, int) for div in s[1]), s
    assert s[0] in _powersof2
    if timesig is not None:
        measuredur = measureDuration(timesig)
        dev, nums = s
        assert sum(F(num*4, dev) for num in nums) == measuredur


@functools.cache
def measureSubdivisions(timesig: TimeSignature,
                        tempo: F,
                        tempoRef=(4, 0),
                        subdivisionStructure: tuple[int, tuple[int, ...]] = (),
                        subdivTempo: int = 96
                        ) -> list[F]:
    """
    Returns a list of beat durations

    Args:
        timesig: the time signature
        tempo: tempo value
        tempoRef: tempo reference figure
        subdivisionStructure: subdivision structure, as a tuple (denom, (*subdivs))
        subdivTempo: ??

    Returns:
        a list of durations, one for each beat subdivision

    """
    if subdivisionStructure:
        _checkSubdivisionStructure(subdivisionStructure, timesig)

    if len(timesig.parts) == 1:
        if subdivisionStructure and timesig.subdivisionStruct:
            subdivisionStructure = timesig.qualifiedSubdivisionStruct()
        subdivs = beatDurations(timesig=timesig.parts[0],
                                tempo=tempo,
                                tempoRef=tempoRef,
                                subdivisions=subdivisionStructure,
                                subdivMaxTempo=subdivTempo)
    else:
        subdivs = []
        for part in timesig.parts:
            # TODO: use the subdivision structure in the timesig, if present
            beatdurs = beatDurations(timesig=part,
                                     tempo=tempo,
                                     tempoRef=tempoRef,
                                     subdivMaxTempo=subdivTempo)
            subdivs.extend(beatdurs)
    assert all(isinstance(div, F) for div in subdivs), f"{subdivs=}"
    assert sum(div for div in subdivs) == measureDuration(timesig)
    return subdivs


def beatDurations(timesig: timesig_t,
                  tempo: F,
                  tempoRef=(4, 0),
                  subdivMaxTempo=96,
                  subdivisions: tuple[int, tuple[int, ...]] = ()
                  ) -> list[F]:
    """
    Returns the beat durations for the given time signature

    Args:
        timesig: the timesignature of the measure or of the part of the measure
            (compound signatures are not supported here)
        tempo: the tempo value
        tempoRef: reference duration for the tempo given. (4, 0) indicates a
            quarter note without dots
        subdivMaxTempo: max tempo for a subdivision. A slower tempo
            will divide beats into subdivisions. For example, a 5/8 measure
            with a quarterTempo of 40 and a subdivMaxTempo of 96 will
            subdivide the measure into 5 8th notes. A faster
            tempo (in this case faster than 48) will result in 2+2+1
        subdivisions: if given, a tuple (denominator, list of subdivision lengths)
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
    quarterTempo = convertTempo(tempo, tempoRef, (4, 0))
    measuredur = measureDuration(timesig)
    timesignum, timesigden = timesig
    if subdivisions:
        _checkSubdivisionStructure(subdivisions)
        subdivden, subdivnums = subdivisions
        out = [F(4*subdivnum, subdivden) for subdivnum in subdivnums]
    elif timesigden == 4:
        if quarterTempo < subdivMaxTempo/2 or (tempoRef == (8, 0) and tempo <= subdivMaxTempo):
            out = [F(1, 2)] * (2 * measuredur.numerator)
        else:
            out = [F(1)] * measuredur.numerator
    elif timesigden == 8:
        if tempoRef == (8, 0) or quarterTempo <= subdivMaxTempo/2:
            # render all beats as 1/8 notes
            out = [F(1, 2)]*timesignum
        else:
            subdivstruct = inferSubdivisions(num=timesignum, den=timesigden, tempo=tempo, tempoRef=tempoRef)
            out = [F(num, timesigden // 4) for num in subdivstruct]
    else:
        # 9/16, 8.=60: convert to 9/8, 4.=60, factor = 2, divide by factor
        factor = F(timesigden, 8)
        # beatdurs = beatDurations((timesignum, 8), tempo=tempo*factor, tempoRef=tempoRef)
        beatdurs = beatDurations((timesignum, 8), tempo=tempo, tempoRef=(int(tempoRef[0]/factor), tempoRef[1]))
        out = [F(dur)/factor for dur in beatdurs]
    if (subdivs := sum(out)) != measuredur:
        raise ValueError(f"The sum of the subdivisions ({subdivs}) does not"
                         f"match the number of quarters ({measuredur}) in this time "
                         f"signature ({timesig[0]}/{timesig[1]}). Subdivision structure: "
                         f"{subdivisions}, durations: {out}")
    return out


@dataclass
class BeatDef:
    """A Beat definition"""
    offset: F
    "The offset within the measure"
    duration: F
    "Duration of the beat in quarter notes"
    weight: int = 0
    "Weight of the beat, used for breaking syncopations"

    def isBinary(self) -> bool:
        return self.duration.numerator % 2 == 0

    @property
    def end(self) -> F:
        "End time of the beat"
        return self.offset + self.duration


@functools.cache
def measureBeatStructure(timesig: TimeSignature | tuple[int, int],
                         tempo: F | int | float,
                         tempoRef: tuple[int, int] = (4, 0),
                         subdivisions: tuple[int, tuple[int, ...]] = (),
                         breakTempo = 52,
                         subdivTempo = 96
                         ) -> list[BeatDef]:
    """
    Beat structure for this measure (a list of beat definitions)

    Args:
        timesig: the time signature
        tempo: tempo in reference to the figure defined in temporef
        subdivisions: the subdivision structure in the
            form (denominator: int, subdivisions). For example a 7/8 bar divided
            in 3+2+2 would have a subdivision strucutre of (8, (3, 2, 2)). A
            4/4 measure divided in 3/8+3/8+2/8+2/8 would be (8, (3, 3, 2, 2))
        breakTempo: a beat resulting in a tempo higher than this
            is by default assigned a weak weight. This means that beats with
            a tempo slower than this are always considered strong beats, indicating
            that beams and syncopations across these beats should be broken
        subdivTempo: a regular subdivision of a beat resulting in a
            tempo lower than this can be promoted to a beat of its own. For example,
            with a quarterTempo of 44, a 5/8 measure would be seen as 5 beats, each
            of 1/8 note length. For a faster tempo, this would result in a beat
            of 2/8 and a ternary beat of 5/8

    Returns:
        a list of BeatDefs
    """
    assert isinstance(breakTempo, (int, float, F))
    assert isinstance(subdivTempo, (int, float, F))
    if subdivisions:
        _checkSubdivisionStructure(subdivisions, timesig)

    qtempo = asQuarterTempo(tempo, tempoRef[0], tempoRef[1])

    if isinstance(timesig, tuple):
        timesig = TimeSignature(timesig)
    subdivDurs = measureSubdivisions(timesig=timesig,
                                     tempo=tempo,
                                     tempoRef=tempoRef,
                                     subdivisionStructure=subdivisions,
                                     subdivTempo=subdivTempo)

    N = len(subdivDurs)
    if N == 1:
        weights = [1]
    elif N % 2 == 0:
        weights = [1, 0] * (N//2)
    elif N % 3 == 0:
        weights = [1, 0, 0] * (N//3)
    else:
        weights = [1, 0] * (N//2)
        weights.append(0)

    now = F(0)
    beatOffsets = []
    weakBeatDurThreshold = F(60) / breakTempo
    for i, dur in enumerate(subdivDurs):
        beatOffsets.append(now)
        now += dur
        if dur.numerator == 3:
            weights[i] = 1
        else:
            beatRealDur = dur * (F(60)/qtempo)
            if beatRealDur > weakBeatDurThreshold:
                weights[i] += 1

    # weights[0] = max(weights) + 1
    assert len(beatOffsets) == len(subdivDurs) == len(weights)
    return [BeatDef(offset, duration, weight)
            for offset, duration, weight in zip(beatOffsets, subdivDurs, weights)]


def beatOffsets(timesig: timesig_t,
                tempo: F | int,
                tempoRef=(4, 0),
                subdivisions: tuple[int, ...] = ()
                ) -> list[F]:
    """
    Returns the offsets of the beats within a measure (first offset is always 0)

    The last value refers to the offset of the end of the measure

    Args:
        timesig: the timesignature as a tuple (num, den)
        tempo: the tempo of the measure
        tempoRef: reference duration for the tempo value, (4, 0) indicates
            a quarter note without dots
        subdivisions: if given, a list of subdivision lengths. For example,
            a 5/8 measure could have a subdivision structure of (2, 3) or (3, 2)

    Returns:
        a list of fractions representing the start time of each beat, plus the
        end time of the measure (== the start time of the next measure)

    Example::
        >>> beatOffsets((5, 8), 60)
        [Fraction(0, 1), Fraction(1, 1), Fraction(2, 1), Fraction(5, 2)]
        # 0, 1, 2, 2.5
    """
    subdivstruct = () if not subdivisions else (timesig[1], subdivisions)
    beatdurs = beatDurations(timesig,
                             tempo=tempo,
                             tempoRef=tempoRef,
                             subdivisions=subdivstruct)
    beatOffsets = [F(0)]
    beatOffsets += _partialsum(beatdurs)
    return beatOffsets


class ScoreStruct:
    """
    Holds the structure of a score (but no content, only measure definitions)

    A ScoreStruct consists of some metadata and a list of :class:`MeasureDef`,
    where each :class:`MeasureDef` defines the properties of the measure at the given
    index (time signature, tempo, etc.). If a ScoreStruct is marked as *endless*, its
    last measure definition extends to infinity.

    The ScoreStruct class is used extensively within :py:mod:`maelzel.core` (see
    `scorestruct-and-maelzel-core`)
    
    .. note::

        Within the context of maelzel.core, a ScoreStruct needs to be
        activated (see :meth:`~ScoreStruct.activate`) to be used
        implicitely in the current Workspace.

    Args:
        score: if given, a score definition as a string (see below for the format), or a
            time signature
        tempo: initial quarter-note tempo. Even if using a time-signature with
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
        s.addMeasure((4, 4), tempo=72)

        # this is the same as:
        s = ScoreStruct((4, 4), 72)

        # Create the beginning of Sacre
        s = ScoreStruct()
        s.addMeasure((4, 4), 50)
        s.addMeasure((3, 4))
        s.addMeasure((4, 4))
        s.addMeasure((2, 4))
        s.addMeasure((3, 4), numMeasures=2)

        # The same can be achieved by passing the entire score at once:
        s = ScoreStruct(r'''
        4/4, 50
        3/4
        4/4
        2/4
        3/4
        .
        ''')

        # Or everything in one line:
        s = ScoreStruct('4/4, 4=50; 3/4; 4/4; 2/4; 3/4; 3/4 ')

    **Format**

    Measure definitions are divided by new line or by ;. A measure definition
    can have many forms::

        measureidx, timesig, tempo, [properties]
        timesig, tempo
        timesig
        tempo

    * Tempo refers by default to a quarter note but any other reference value can
      be given (4=60, 8=76, 8.=60, etc)
    * measure numbers start at 0
    * comments start with `#` and last until the end of the line
    * A line with a single "." repeats the last defined measure
    * A score ending with the line ... is an endless score, which extends the last timesignature
      and tempo
    * A measure can have the following properties: "mark" (a rehearsal mark, ``mark=B``),
      "barline" (``barline=dashed``), "label" (a text annotation, ``label=leggiero``)

    The measure number and/or the tempo can both be left out. The following definitions are
    all the same::

        1, 5/8, 63
        5/8, 4=63
        5/8

    **Example**::

        0, 4/4, 60, mark="A"
        3/4, 80     # Assumes measureidx=1
        10, 5/8, 120
        ...

    """
    def __init__(self,
                 score: str | timesig_t = '',
                 tempo: int | float | F = 0,
                 endless: bool = True,
                 title='',
                 composer='',
                 const=False,
                 breakTempo: int = 52,
                 subdivTempo: int = 92):

        # holds the time offset (in seconds) of each measure
        self._timeOffsets: list[F] = []

        self._beatOffsets: list[F] = []

        # the quarternote duration of each measure
        self._quarternoteDurations: list[F] = []

        self._prevScoreStruct: ScoreStruct | None = None
        self._needsUpdate = True
        self._lastIndex = 0

        self.title = title
        """Title metadata"""

        self.composer = composer
        """The composer metadata"""

        self.measures: list[MeasureDef] = []
        """The measure definitions"""

        self.endless = endless
        """Is this an endless scorestruct?"""

        self.const = const
        """Is this ScoreStruct read-only?"""
        
        self.breakTempo: int | None = breakTempo
        """Tempo at which a syncopation across the beat is broken (during quantization)"""

        self.subdivTempo: int | None = subdivTempo
        """A subdivision slower than this becomes an independent beat (during quantization)"""

        self._hash: int | None = None
        """Cached hash"""

        if score:
            if isinstance(score, tuple):
                self.addMeasure(score, tempo=tempo or 60)
            else:
                self._parseScore(score, initialTempo=tempo or 60)
        elif tempo:
            self.addMeasure((4, 4), tempo=tempo)


    def __hash__(self) -> int:
        if self._hash is None:
            hashes = [hash(x) for x in (self.title, self.endless)]
            hashes.extend(hash(mdef) for mdef in self.measures)
            self._hash = hash(tuple(hashes))
        return self._hash

    def __eq__(self, other: ScoreStruct) -> int:
        return hash(self) == hash(other)

    def _parseScore(self, s: str, initialTempo=60, initialTimeSignature=(4, 4)
                    ) -> None:
        """
        Create a ScoreStruct from a string definition

        Args:
            s: the score as string. See below for format
            initialTempo: the initial tempo, for the case where the initial measure/s
                do not include a tempo
            initialTimeSignature: the initial time signature

        **Format**

        A definitions are divided by new line or by ;. Each line has the form::

            measureidx, timeSig, tempo


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
            ,3/4,80     # Assumes measureidx=1
            4/4         # Assumes measureidx=2, inherits tempo 80
            10, 5/8, 120
            12,,96      # At measureidx 12, change tempo to 96
            30,,
            .
            .      # last measure (inclusive, the score will have 33 measures)
        10, 4/4 q=60 label='Mylabel'
        3/4 q=42
        20, q=60 label='foo'

        """
        assert not self.measures
        measureidx = -1
        lines = re.split(r'[\n;]', s)
        lines = emlib.textlib.linesStrip(lines)
        if lines[-1].strip() == '...':
            self.endless = True
            lines = lines[:-1]

        def lineStrip(line: str) -> str:
            if "#" in line:
                line = line.split("#")[0]
            return line.strip()

        for i, line0 in enumerate(lines):
            line = lineStrip(line0)
            if not line:
                continue

            if line == ".":
                if not self.measures:
                    raise ValueError("Cannot repeat last measure definition since there are "
                                     "no measures defined yet")
                self.addMeasure()
                measureidx += 1
                continue

            mdef = _parseLine(line)
            if i == 0:
                if mdef.timesig is None:
                    mdef.timesig = TimeSignature(initialTimeSignature)
                if mdef.tempodef is None:
                    mdef.tempodef = TempoDef(tempo=F(initialTempo))

            if mdef.measureidx is None:
                mdef.measureidx = measureidx + 1
            else:
                assert mdef.measureidx > measureidx
                if mdef.measureidx - measureidx > 1:
                    self.addMeasure(count=mdef.measureidx - measureidx - 1)

            self.addMeasure(
                timesig=mdef.timesig or '',
                tempo=mdef.tempodef.tempo if mdef.tempodef else None,
                temporef=(mdef.tempodef.base, mdef.tempodef.dots) if mdef.tempodef else (4, 0),
                annotation=mdef.label,
                mark=Mark(mdef.mark) if mdef.mark else None,
                barline=mdef.barline
            )
            measureidx = mdef.measureidx

    def copy(self) -> ScoreStruct:
        """
        Create a deep copy of this ScoreStruct
        """
        s = ScoreStruct(endless=self.endless, title=self.title, composer=self.composer)
        s.measures = [m.copy() for m in self.measures]
        for m in s.measures:
            m.parent = s
        s.modified()
        return s

    def activate(self) -> Self:
        """
        Set this scorestruct as active for the current workspace within maelzel.core

        .. seealso:: :func:`~maelzel.core.workspace.setScoreStruct`
        """
        from maelzel.core import Workspace
        Workspace.active.scorestruct = self
        return self

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
        return len(self.measures)

    def __len__(self):
        """
        Returns the number of defined measures (even if the score is defined as endless)

        This is the same as :meth:`ScoreStruct.numMeasures`
        """
        return self.numMeasures()

    def measure(self, idx: int, extend: bool | None = None) -> MeasureDef:
        """
        Returns the MeasureDef at the given index.

        Args:
            idx: the measure index (measures start at 0)
            extend: if True and the index given is outside the defined
                measures, the score will be extended, repeating the last
                defined measure. For endless scores, the default is to
                extend the measure definitions.

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
        self._update()

        if idx < len(self.measures):
            measuredef = self.measures[idx]
            assert measuredef.parent is self
            return measuredef

        if extend is None:
            extend = self.endless

        # outside defined measures
        if not extend:
            if not self.endless:
                raise IndexError(f"index {idx} out of range. The score has "
                                 f"{len(self.measures)} measures defined")

            # "outside" of the defined score: return a copy of the last
            # measure so that any modification will not have any effect
            # Make the parent None so that it does not get notified if tempo or timesig
            # change
            out = self.measures[-1].copy()
            out.const = True
            return out

        for n in range(len(self.measures)-1, idx):
            self.addMeasure()

        mdef = self.measures[-1]
        assert mdef.parent is self
        return mdef

    @_overload
    def __getitem__(self, item: int) -> MeasureDef: ...

    @_overload
    def __getitem__(self, item: slice) -> list[MeasureDef]: ...

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.measure(item)
        assert isinstance(item, slice)
        return [self.measure(idx) for idx in range(item.start, item.stop, item.step)]

    def _extend(self, num: int):
        mdef = self.measures[-1].copy()
        mdef.tempoInherited = True
        mdef.timesigInherited = True
        for _ in range(num):
            self.measures.append(mdef.copy())

    def addMeasure(self,
                   timesig: tuple[int, int] | str | TimeSignature = '',
                   tempo: num_t | None = None,
                   index: int | None = None,
                   annotation='',
                   mark: str | Mark | None = None,
                   keySignature: tuple[int, str] | KeySignature | None = None,
                   barline='',
                   properties: dict[str, Any] | None = None,
                   temporef: tuple[int, int] | str = (4, 0),
                   count=1,
                   ) -> None:
        """
        Add a measure definition to this score structure

        Args:
            timesig: the time signature of the new measure. If not given, the last
                time signature will be used. The timesig can be given as str in the
                form "num/den". For a compound time signature use "3/8+2/8". To
                specify the internal subdivision use a TimeSignature object or a
                string in the form "5/8(3-2)"
            tempo: the tempo of a quarter note. If not given, the last tempo
                will be used
            annotation: each measure can have a text annotation
            index: if given, add measure at the given index
            count: number of measures of the same kind to add
            mark: add a rehearsal mark to the new measure definition.
                A rehearsal mark can be a text or a RehearsalMark, which enables you
                to customize the rehearsal mark further
            keySignature: either a KeySignature object or a tuple (fifths, mode); for example
                for A-Major, ``(3, 'major')``. Mode can also be left as an ampty string
            barline: if needed, the right barline of the measure can be set to one of
                'single', 'final', 'double', 'solid', 'dotted', 'dashed', 'tick', 'short',
                'double-thin' or 'none'
            temporef: reference rhythmic figure for the given tempo. Only used if
                a tempo is given. If given as a tuplet, a tuplet (figurevale: int, numdots: int),
                where 4=quarter note, 8=1/8th note, etc., and numdots represents the number of dots.
                If given as a string, a string of the form "4", "8.", etc., as above described.
            properties: user defined properties for the measure, can be anything.

        Example::

            # Create a 4/4 score, 32 measures long
            >>> s = ScoreStruct()
            >>> s.addMeasure((4, 4), 52, numMeasures=32)
        """
        if self.const:
            raise RuntimeError("This ScoreStruct is read-only")

        if not timesig:
            timesigInherited = True
            timesig = self.measures[-1].timesig if self.measures else (4, 4)
        else:
            timesigInherited = False
        if not tempo:
            tempoInherited = True
            if self.measures:
                last = self.measures[-1]
                tempo = last.tempo
                temporef = last.tempoRef
            else:
                tempo, temporef = F(60), (4, 0)
        else:
            tempoInherited = False

        if isinstance(mark, str):
            mark = Mark(mark)

        if isinstance(keySignature, tuple):
            fifths, mode = keySignature
            assert isinstance(fifths, int) and isinstance(mode, str)
            keySignature = KeySignature(fifths=fifths, mode=mode)

        if not isinstance(timesig, TimeSignature):
            timesig = TimeSignature.parse(timesig)

        if isinstance(temporef, str):
            parts = temporef.split(".")
            refvalue = int(parts[0])
            numdots = len(parts) - 1
            temporef = (refvalue, numdots)

        assert keySignature is None or isinstance(keySignature, KeySignature)

        measure = MeasureDef(
            timesig=timesig,
            tempo=tempo,
            annotation=annotation,
            timesigInherited=timesigInherited,
            tempoInherited=tempoInherited,
            mark=mark,
            properties=properties,
            key=keySignature,
            barline=barline,
            parent=self,
            tempoRef=temporef,
            const=self.const)

        if index:
            if index < len(self.measures):
                if count > 1:
                    raise ValueError(f"Setting an already existing measure ({index}), multiple measures are not supported")
                self.measures[index] = measure
            else:
                self._extend(index - len(self.measures))
        else:
            self.measures.append(measure)

        if count > 1:
            self._extend(count - 1)

        self.modified()

    def addMark(self, idx: int, mark: Mark | str, box: str = 'square'
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
        if self.const:
            raise RuntimeError("This ScoreStruct is read-only")

        if idx >= len(self.measures) and not self.endless:
            raise IndexError(f"Measure index {idx} out of range. "
                             f"This score has {len(self.measures)} measures")
        mdef = self.measure(idx, extend=True)
        if isinstance(mark, str):
            mark = Mark(mark, box=box)
        mdef.mark = mark

    def ensureNumMeasures(self, numMeasures: int) -> None:
        """
        Ensures that this score has at least the given number of measures

        If the scorestruct already has reached the given length this operation
        does nothing. Otherwise the score is extended.

        Args:
            numMeasures: the minimum number of measures this score should have
        """
        if (count := numMeasures - self.numMeasures()) > 0:
            self._extend(count)

    def ensureDurationInSeconds(self, secs: F) -> None:
        """
        Ensure that this scorestruct is long enough to include the given time

        This is of relevance in the following use cases:

        * When creating a clicktrack from an endless score.
        * When exporting a scorestruct to midi

        Args:
            secs: the duration to ensure, in seconds

        """
        mindex, mbeat = self.timeToLocation(secs)
        if mindex is None:
            # Outside of this score's time range.
            assert not self.endless

            raise ValueError(f"duration {secs} outside score")
        self.ensureNumMeasures(mindex + 1)

    def durationQuarters(self) -> F:
        """
        The duration of this score, in quarternotes

        Raises ValueError if this score is endless
        """
        if self.endless:
            raise ValueError("An endless score does not have a duration in beats")
        return asF(sum(m.duration for m in self.measures))

    def durationSecs(self) -> F:
        """
        The duration of this score, in seconds

        Raises ValueError if this score is endless
        """
        if self.endless:
            raise ValueError("An endless score does not have a duration in seconds")
        return asF(sum(m.durationSecs for m in self.measures))

    def _update(self) -> None:
        if not self._needsUpdate:
            return

        # if mdef := next((mdef for mdef in self.measuredefs if mdef.parent is not self), None):
        #     raise ValueError(f"Wrong parent: {mdef}, parentid={id(mdef.parent)}, self={id(self)}")

        self._fixInheritedAttributes()

        accumTime = F(0)
        accumBeats = F(0)
        starts = []
        quarterDurs = []
        beatOffsets = []

        for mdef in self.measures:
            starts.append(accumTime)
            beatOffsets.append(accumBeats)
            durBeats = mdef.duration
            quarterDurs.append(durBeats)
            accumTime += F(60) / mdef.quarterTempo * durBeats
            accumBeats += durBeats

        self._timeOffsets = starts
        self._beatOffsets = beatOffsets
        self._quarternoteDurations = quarterDurs
        self._needsUpdate = False

    def locationToTime(self, measure: int, beat: num_t = F(0)) -> F:
        """
        Return the elapsed time at the given score location

        Args:
            measure: the measure number (starting with 0)
            beat: the beat within the measure

        Returns:
            a time in seconds (as a Fraction to avoid rounding problems)
        """
        if self._needsUpdate:
            self._update()

        numdefs = len(self.measures)
        if measure > numdefs - 1:
            if measure == numdefs and beat == 0:
                mdef = self.measures[-1]
                return self._timeOffsets[-1] + mdef.durationSecs

            if not self.endless:
                raise ValueError("Measure outside score")

            last = numdefs - 1
            lastTime = self._timeOffsets[last]
            mdef = self.measures[last]
            mdur = mdef.durationSecs
            fractionalDur = beat * 60 / mdef.quarterTempo
            return lastTime + (measure - last) * mdur + fractionalDur
        else:
            now = self._timeOffsets[measure]
            mdef = self.measures[measure]
            measureBeats = self._quarternoteDurations[measure]
            assert beat <= measureBeats, "Beat outside measure"
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
        measuredef = self.measure(measureindex)
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
        if self._needsUpdate:
            self._update()

        if not self.measures:
            raise IndexError("This ScoreStruct is empty")

        time = asF(time)
        idx = bisect(self._timeOffsets, time)
        if idx < len(self.measures):
            m = self.measures[idx-1]
            assert self._timeOffsets[idx - 1] <= time < self._timeOffsets[idx]
            dt = time-self._timeOffsets[idx-1]
            beat = dt*m.quarterTempo/F(60)
            return idx-1, beat

        # is it within the last measure?
        m = self.measures[idx-1]
        dt = time - self._timeOffsets[idx-1]
        if dt < m.durationSecs:
            beat = dt*m.quarterTempo/F(60)
            return idx-1, beat
        # outside score
        if not self.endless:
            return None, F0
        lastMeas = self.measures[-1]
        measDur = lastMeas.durationSecs
        numMeasures = dt / measDur
        beat = (numMeasures - int(numMeasures)) * lastMeas.duration
        return len(self.measures)-1 + int(numMeasures), beat

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
        if self._needsUpdate:
            self._update()

        numdefs = len(self.measures)
        assert numdefs >= 1, "This scorestruct is empty"

        if not isinstance(beat, F):
            beat = asF(beat)

        lastBeatOffset = self._beatOffsets[-1]
        if beat > lastBeatOffset:
            # past the end
            rest = beat - lastBeatOffset
            if not self.endless:
                if rest > 0:
                    raise ValueError(f"The given beat ({beat}) is outside the score")
                return (numdefs, F0)
            beatsPerMeasure = self.measures[-1].duration
            numExtraMeasures = int(rest / beatsPerMeasure)
            idx = numdefs - 1 + numExtraMeasures
            restBeats = rest - numExtraMeasures*beatsPerMeasure
            # restBeats = rest % beatsPerMeasure
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
        for mdef in self.measures:
            yield mdef
        if not self.endless:
            raise StopIteration
        lastmdef = self.measures[-1]
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
            measureidx, beat = a
            return self.locationToBeat(measureidx, beat)
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
        if b is None:
            # a is a beat or a location
            if isinstance(a, tuple):
                measure, beat = a
                return self.locationToTime(measure, beat)
            else:
                return self.beatToTime(a)
        else:
            assert isinstance(a, int)
            return self.locationToTime(a, b)

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

    def asBeat(self, location: num_t | tuple[int, float | F]) -> F:
        """
        Given a beat or a location (measureidx, relativeoffset), returns an absolute beat

        Args:
            location: the location

        Returns:
            the absolute beat in quarter notes
        """
        if isinstance(location, tuple):
            measure, beat = location
            return self.locationToBeat(measure, beat)
        else:
            return location if isinstance(location, F) else F(location)  # type: ignore


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

        >>> s = ScoreStruct(r'''
        ... 3/4, 120
        ... 3/8
        ... 4/4
        ... ''')
        >>> s.locationToBeat(1, 0.5)
        3.5
        >>> s.locationToTime(1, 0.5)
        1.75

        """
        if self._needsUpdate:
            self._update()

        if not isinstance(measure, int):
            raise TypeError(f"Expected a measure index, got {measure=}")

        if not isinstance(beat, (int, float, F)):
            raise TypeError(f"Expected a number as beat, got {beat=}")

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
        for i, mdef in enumerate(self):
            if i < measure:
                accum += mdef.duration
            else:
                if beat > mdef.duration:
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
        return [self.measure(idx=i) for i in range(idx0, idx1)]

    def timeDelta(self,
                  start: beat_t,
                  end: beat_t
                  ) -> F:
        """
        Returns the elapsed time between two beats or score locations.

        Args:
            start: the start location, as a beat or as a tuple (measureidx, beatOffset)
            end: the end location, as a beat or as a tuple (measureidx, beatOffset)

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
        startTime = self.beatToTime(self.asBeat(start))
        endTime = self.beatToTime(self.asBeat(end))
        return endTime - startTime

    def beatDelta(self,
                  start: num_t | tuple[int, num_t],
                  end: num_t | tuple[int, num_t]) -> F:
        """
        Difference in beats between the two score locations or two times

        Args:
            start: the start moment as a location (a tuple (measureidx, beatOffset) or as
                a time
            end: the end location, a tuple (measureidx, beatOffset)

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
             app='',
             scalefactor: float = 1.0,
             backend='',
             renderoptions: RenderOptions | None = None
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
                emlib.misc.open_with_app(outfile, app=app or None)
        else:
            emlib.misc.open_with_app(outfile, app=app)

    def dump(self, index=True, beatstruct=True) -> None:
        """
        Dump this ScoreStruct to stdout
        """
        self._update()

        from maelzel import _util
        if _util.pythonSessionType() == 'jupyter':
            from IPython.display import display, HTML
            display(HTML(self._repr_html_()))
        else:
            tempo = -1
            for i, m in enumerate(self.measures):
                parts = []
                if index:
                    parts.append(f"{i}: {m.timesig}")
                else:
                    parts.append(str(m.timesig))
                if m.quarterTempo != tempo:
                    parts.append(f", {m.quarterTempo}")
                    tempo = m.quarterTempo
                if m.annotation:
                    parts.append(f", annotation={m.annotation}")
                if m.mark:
                    parts.append(f", rehearsal={m.mark.text}")
                if m.barline:
                    parts.append(f", barline={m.barline}")
                if m.key:
                    parts.append(f", keySignature={m.key.fifths}")
                if beatstruct:
                    beatstruct = m.beatStructure()
                    s = "+".join(f"{beat.duration.numerator}/{beat.duration.denominator}" for beat in beatstruct)
                    parts.append(f", beatstruct={s}")
                print("".join(parts))

    def hasUniqueTempo(self) -> bool:
        """
        Returns True if this ScoreStruct has no tempo changes
        """
        self._update()

        t = self.measures[0].quarterTempo
        return all(m.quarterTempo == t for m in self.measures)

    def __repr__(self) -> str:
        self._update()

        if self.hasUniqueTempo() and self.hasUniqueTimesig():
            m0 = self.measures[0]
            info = [str(m0.timesig), f"tempo={m0.quarterTempo}"]
            if m0.key:
                info.append(f"keySignature={m0.key}")
            infostr = ", ".join(info)
            return f'ScoreStruct({infostr})'
        else:
            tempo = -1
            parts = []
            maxdefs = 10
            for m in self.measures[:maxdefs]:
                if m.quarterTempo != tempo:
                    tempo = m.quarterTempo
                    parts.append(f"{m.timesig}@{tempo}")
                else:
                    parts.append(f"{m.timesig}")
            s = ", ".join(parts)
            if len(self.measures) > maxdefs:
                s += " …"
            return f"ScoreStruct([{s}])"

    def __enter__(self):
        from maelzel.core import getWorkspace
        w = getWorkspace()
        self._prevScoreStruct = w.scorestruct
        w.scorestruct = self
        return self

    def __exit__(self, *args, **kws):
        assert self._prevScoreStruct is not None
        from maelzel.core import getWorkspace
        getWorkspace().scorestruct = self._prevScoreStruct

    def _repr_html_(self) -> str:
        self._update()
        import emlib.misc
        colnames = ['Meas. Index', 'Timesig', 'Tempo', 'Label', 'Rehearsal', 'Barline', 'Beats']

        if any(m.key is not None for m in self.measures):
            colnames.append('Key')
            haskey = True
        else:
            haskey = False

        allparts = ['<p><strong>ScoreStruct</strong></p>']
        tempodef = TempoDef(F0, 0, 0)
        rows = []
        for i, m in enumerate(self.measures):
            # num, den = m.timesig
            if (newtempodef := m.tempoDef()) != tempodef:
                reffigure, numdots = m.tempoRef
                refstr = unicodeDuration((reffigure, numdots))
                tempovaluestr = ("%.3f" % m.tempo).rstrip("0").rstrip(".")
                tempostr = f"{refstr}={tempovaluestr}"
                tempodef = newtempodef
            else:
                tempostr = ""
            rehearsal = m.mark.text if m.mark is not None else ''
            timesig = m.timesigRepr()
            if m.timesigInherited:
                timesig = f'({timesig})'
            beatstruct = m.beatStructure()
            parts = []
            for beat in beatstruct:
                fig = unicodeDuration(beat.duration)
                parts.append(fig)
                # if beat.duration.denominator == 1:
                #     parts.append(str(beat.duration.numerator))
                # else:
                #     parts.append(f"{beat.duration.numerator}/{beat.duration.denominator}")
            if all(part==parts[0] for part in parts):
                beatstruct = f"{len(parts)}×{parts[0]}"
            else:
                beatstruct = "+".join(parts)
            row = [str(i), timesig, tempostr, m.annotation or "", rehearsal, m.barline, beatstruct]
            if haskey:
                row.append(str(m.key.fifths) if m.key else '-')
            rows.append(row)
        if self.endless:
            rows.append(("...",) + ("",) * (len(rows[-1]) - 1))
        rowstyle = 'font-size: small;'
        htmltable = emlib.misc.html_table(rows, colnames, rowstyles=[rowstyle]*len(colnames))
        allparts.append(htmltable)
        return "".join(allparts)

    def _render(self, backend='', renderoptions: RenderOptions | None = None
                ) -> Renderer:
        self._update()
        from maelzel.scoring import quant
        from maelzel.scoring import render
        quantprofile = quant.QuantizationProfile()
        measures = [quant.QuantizedMeasure(timesig=m.timesig, tempo=m.tempo, tempoRef=m.tempoRef,
                                           quantprofile=quantprofile, beats=[])
                    for m in self.measures]
        part = quant.QuantizedPart(struct=self, measures=measures, quantProfile=quantprofile)
        qscore = quant.QuantizedScore([part], title=self.title, composer=self.composer)
        if not renderoptions:
            renderoptions = render.RenderOptions()
        if backend:
            renderoptions.backend = backend
        return render.renderQuantizedScore(qscore, options=renderoptions)

    def setTempo(self, measureidx: int, tempo: F | int | float, reference: F | tuple[int, int] = (4, 0)) -> None:
        """
        Set the tempo of the given measure, until the next tempo change

        Args:
            measureidx: first measure to modify
            tempo: the new tempo
            reference: the reference duration (1=quarternote, 2=halfnote, 0.5: 8th note, etc)

        """
        if self.const:
            raise RuntimeError("This ScoreStruct is read-only")

        if measureidx > len(self) and not self.endless:
            raise IndexError(f"Index {measureidx} out of rage; this ScoreStruct has only "
                             f"{len(self)} measures defined")
        meas = self.measure(measureidx)
        tempo = asF(tempo)
        if isinstance(reference, F):
            reference = durationToFigure(reference)
        meas._setTempo(tempo=tempo, reference=reference, inherited=False)
        for m in self.measures[measureidx+1:]:
            if m.tempoInherited:
                m._setTempo(tempo=tempo, reference=reference, inherited=True)
            else:
                break
        self.modified()

    def setTimeSignature(self, measureidx, timesig: tuple[int, int] | str | TimeSignature
                         ) -> None:
        if self.const:
            raise RuntimeError("This ScoreStruct is read-only")

        if measureidx > len(self) and not self.endless:
            raise IndexError(f"Index {measureidx} out of rage; this ScoreStruct has only "
                             f"{len(self)} measures defined")
        timesig = _asTimeSignature(timesig)
        mdef = self.measure(measureidx, extend=True)
        mdef.timesig = timesig
        mdef.timesigInherited = False
        for m in self.measures[measureidx + 1:]:
            if m.timesigInherited:
                m.timesig = timesig
            else:
                break

    def modified(self) -> None:
        """
        mark this ScoreStruct as modified

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
        self._needsUpdate = True
        self._hash = None

    def _fixInheritedAttributes(self):
        m = self.measures[0]
        timesig = m.timesig
        tempo = m.tempo
        temporef = m.tempoRef
        for m in self.measures[1:]:
            if m.tempoInherited:
                m._tempo = tempo
                m.tempoRef = temporef
            else:
                tempo = m._tempo
                temporef = m.tempoRef
            if m.timesigInherited:
                m._timesig = timesig
            else:
                timesig = m._timesig

    def hasUniqueTimesig(self) -> bool:
        """
        Returns True if this ScoreStruct does not have any time-signature change
        """
        self._update()
        lastTimesig = self.measures[0].timesig
        for m in self.measures:
            if m.timesig != lastTimesig:
                return False
        return True

    def write(self,
              path: str | Path,
              backend='',
              renderoptions: RenderOptions | None = None
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
        Set an end measure to this ScoreStruct, in place

        If the scorestruct has less defined measures as requested, then it is extended
        by duplicating the last defined measure as needed. Otherwise, the scorestruct is
        cropped. The scorestruct ceases to be endless if that was the case previously

        Args:
            numMeasures: the requested number of measures after the operation

        """
        self.endless = False
        if numMeasures < len(self.measures):
            self.measures = self.measures[:numMeasures]
        else:
            last = self.measures[-1]

            self.addMeasure(timesig=last.timesig,
                            tempo=last.quarterTempo,
                            keySignature=last.key,
                            count=numMeasures - len(self.measures))

    def setBarline(self, measureidx: int, linetype: str) -> None:
        """
        Set the right barline type

        Args:
            measureidx: the measure index to modify
            linetype: one of 'single', 'double', 'final', 'solid', 'dashed'

        """
        if self.const:
            raise RuntimeError("This ScoreStruct is read-only")
        assert linetype in _barstyles, f"Unknown style '{linetype}', possible styles: {_barstyles}"
        self.measure(measureidx, extend=True).barline = linetype

    def asText(self) -> str:
        """
        This ScoreStruct as parsable text format

        Returns:
            this score as text
        """
        lines = []
        for i, measuredef in enumerate(self.measures):
            line = measuredef.asScoreLine()
            lines.append(f'{i}, {line}')
        if self.endless:
            lines.append('...')
        return '\n'.join(lines)

    def makeClickTrack(self,
                       minMeasures: int = 0,
                       clickdur: F | None = None,
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
            struct.ensureNumMeasures(minMeasures)
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
    for i, measuredef in enumerate(struct.measures):
        dur = measuredef.duration
        if i == len(struct.measures) - 1:
            events.append(Note(pitch=midinote, offset=now, dur=dur))
        now += dur
    voice = Voice(events)
    return Score([voice], scorestruct=struct)


def unicodeDuration(dur: tuple[int, int] | F) -> str:
    """
    Returns the given notated duration as a unicode str

    Args:
        dur: either a duration as a Fraction or a figure as tuple (base: int, dots: int),
            where base represents the notated figure (4=quarter note, 8=eigth note, etc)
            and dots is the number of dots

    Returns:
        the unicode representation
    """
    if isinstance(dur, F):
        base, dots = durationToFigure(dur)
    else:
        base, dots = dur

    figures = {
        1: "𝅝",
        2: "𝅗𝅥",
        4: "𝅘𝅥",
        8: "𝅘𝅥𝅮",
        16: "𝅘𝅥𝅯",
        32: "𝅘𝅥𝅰",
        64: "𝅘𝅥𝅱",
    }
    base = figures.get(base)
    if not base:
        raise ValueError(f"Invalid figure, expected a power of 2 between 1 and 64, got {base}")
    return base + '𝅭' * dots


def durationToFigure(dur: F, maxdots=5) -> tuple[int, int]:
    """
    Convert a duration in quarter notes (as a fraction) to a notated duration

    Args:
        dur: duration as a fraction
        maxdots: max. number of dots allowed

    Returns:
        a tuple (basedur: int, numdots: int) where basedur represents the
        base duration (4=quarter note, 8=8th note, etc.) and numdots represents
        the number of dots


    .. note::

        A dotted note has duration = base * (2^(dots+1) - 1) / 2^dots
        Relative to quarter note: (4/base) * (2^(dots+1) - 1) / 2^dots = num/denom
    """
    # Simplify: 4 * denom * (2^(dots+1) - 1) = base * num * 2^dots

    # Count trailing zeros in numerator (these become dots)
    # numerator must be odd times a power of 2 minus that power

    # Try to express as: numerator/denominator = (4/base) * (2^(n+1) - 1) / 2^n
    # This means: numerator * 2^n = denominator * 4 * (2^(n+1) - 1) / base
    num, den = dur.numerator, dur.denominator

    for numdots in range(maxdots+1):
        # (2^(dots+1) - 1) is the pattern for dots
        dotsFactor = (2 ** (numdots + 1)) - 1

        # Check if numerator * 2^dots / dots_factor gives us an integer
        if (num * (2 ** numdots)) % dotsFactor == 0:
            k = (num * (2 ** numdots)) // dotsFactor
            base = (4 * den) // k

            # Verify and check base is valid
            if k * base == 4 * den and base in _powersof2:
                return (base, numdots)

    raise ValueError(f"Cannot represent {num}/{den} as a standard "
                     f"duration with dots")