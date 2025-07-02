from __future__ import annotations

import random

import pitchtools as pt
import math

from maelzel.common import F, asF
from maelzel.scoring.common import NotatedDuration

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterator, Sequence
    from maelzel.common import timesig_t

# This module can only import .common from .


def asSimplestNumberType(f: F) -> int | float:
    """
    convert a fraction to the simplest number type it represents
    """
    fl = float(f)
    i = int(fl)
    return i if i == fl else fl


def roundMidinote(a: float, divsPerSemitone=4) -> float:
    """
    Round a midi note to the nearest division

    Args:
        a: the midinote
        divsPerSemitone: the number of subdivisions per semitone

    Returns:
        the rounded midinote

    Example
    ~~~~~~~

        >>> from maelzel.scoring import util
        >>> util.roundMidinote(60.1)
        60.0
        >>> util.roundMidinote(60.7, divsPerSemitone=2)
        60.5
        >>> util.roundMidinote(60.7, dicsPerSemitone=4)
        60.75

    """
    rounding_factor = 1/divsPerSemitone
    return round(a/rounding_factor)*rounding_factor


def measureQuarterDuration(timesig: timesig_t) -> F:
    """
    The duration in quarter notes of a measure according to its time signature

    Args:
        timesig: a tuple (num, den)

    Returns:
        the duration in quarter notes

    .. rubric:: Example

    >>> measureQuarterDuration((3,4))
    Fraction(3, 1)

    >>> measureQuarterDuration((5, 8))
    Fraction(5, 2)

    """
    assert isinstance(timesig, tuple) and isinstance(timesig[0], int) and isinstance(timesig[1], int)
    num, den = timesig
    quarterDuration = F(num)/den * 4
    return quarterDuration


def measureTimeDuration(timesig: timesig_t, quarterTempo: F) -> F:
    """
    The duration of a measure in seconds

    Args:
        timesig: the time signature, a tuple (num, den)
        quarterTempo: the tempo of the quarter note

    Returns:
        The duration **in seconds**

    """
    assert isinstance(quarterTempo, (int, F))
    quarters = measureQuarterDuration(timesig=timesig)
    return (F(60) / quarterTempo) * quarters


def midinotesNeedMultipleClefs(midinotes: list[float], threshold=1) -> bool:
    """
    True if multiple clefs are needed to represent these midinotes

    This can be used to determine of they need to be split
    among multiple staves.

    Args:
        midinotes: the pitches to evaluate
        threshold: how many events should be outside a clef's range
            to declare the need for multiple clefs

    Returns:
        True if midinotes can't be represened through one clef alone
    """
    G, F, G15a = 0, 0, 0
    for midinote in midinotes:
        if 55 < midinote <= 93:
            G += 1
        elif 93 < midinote:
            G15a += 1
        else:
            F += 1
        if int(G>threshold)+int(G15a>threshold)+int(F>threshold)>1:
            return True
    return False


def centsShown(centsdev: int, divsPerSemitone=4, snap=2, addplus=False) -> str:
    """
    Given a cents deviation from a chromatic pitch, return
    a string to be shown along the notation to indicate the
    distance to its corresponding microtone.

    Args:
        centsdev: the deviation from the chromatic pitch
        divsPerSemitone: divisions per semitone
        snap: if the difference to the microtone is within this error,
            we "snap" the pitch to the microtone and do not
            show any annotation
        addplus: if True, adds a plus sign for any possitive deviation

    Returns:
        the string to be shown alongside the notated pitch

    Example
    ~~~~~~~

        >>> centsShown(55, divsPerSemitone=4)
        "5"
    """
    # cents can be also negative (see self.cents)
    divsPerSemitone = divsPerSemitone
    pivot = int(round(100 / divsPerSemitone))
    dist = min(centsdev%pivot, -centsdev%pivot)
    if dist <= snap:
        return ""
    if centsdev < 0:
        # NB: this is not a normal - (minus) sign! We do this to avoid it being confused
        # with a syllable separator during rendering (this is currently the case
        # in musescore
        return f"–{-centsdev}"
    return f'+{centsdev:d}' if addplus else str(int(centsdev))


def snapToGrids(x: F, ticks: list[F], offsets: list[F] | F = F(0), mode='nearest') -> F:
    """
    Snap x to the a grid created by the given grids

    Args:
        x: an unquantized value
        ticks: deltas which constitute a grid
        offsets: offsets for each grid. Either a list of offsets or a single offset
        mode: 'nearest', 'floor', 'ceil'

    Returns:
        the nearest value to x within the grid, according to the specified mode

    Example
    ~~~~~~~

       >>> snapToGrids(0.4, [1/4, 1/3])
       1/2

       >>> snapToGrids(0.29, [1/4, 1/3])
       1/3

    """
    def snapRound(x: F, tick: F, offset: F) -> F:
        return tick * round((x - offset) * (1 / tick)) + offset

    def snapFloor(x: F, tick: F, offset: F) -> F:
        return tick * int((x - offset) * (1 / tick)) + offset

    def snapCeil(x: F, tick: F, offset: F) -> F:
        xtick = (x - offset) * (1 / tick)
        xtickfloor = int(xtick)
        if xtick > xtickfloor:
            xtick = xtickfloor + 1
        return tick * xtick + offset

    func = {
        'floor': snapFloor,
        'ceil': snapCeil,
        'nearest': snapRound,
    }.get(mode)
    if func is None:
        raise ValueError(f"mode should be one of 'floor', 'ceil', 'round', but got {mode}")
    if not isinstance(offsets, list):
        offsets = [offsets] * len(ticks)
    if len(offsets) != len(ticks):
        raise ValueError(f"offsets and ticks should have the same length, {ticks=}, {offsets=}")
    quants = [func(x, t, o) for t, o in zip(ticks, offsets)]
    quants.sort(key=lambda quant: abs(quant - x))
    return quants[0]


def nextInGrid(x: float|F, ticks: list[F]) -> F:
    """
    Snap x to the right within a grid created by the given ticks

    Args:
        x: an unquantized value
        ticks: deltas which constitute a grid

    Returns:
        the nearest value to x within the grid

    Example
    ~~~~~~~

       >>> nextInGrid(0.4, [1/4, 1/3])
       0.5

       >>> nextInGrid(0.29, [1/4, 1/3])
       0.33333333

    """
    return snapToGrids(asF(x) + F(1, 999999999), ticks, mode='ceil')


# def snapTime(start: F,
#              duration: F,
#              divisors: list[int],
#              mindur=F(1, 16),
#              durdivisors: list[int] | None = None
#              ) -> tuple[F, F]:
#     """
#     Quantize an event to snap to a grid defined by divisors and durdivisors
# 
#     Args:
#         start: the start of the event
#         duration: the duration of the event
#         divisors: a list of divisions of the pulse
#         mindur: the min. duration of the note
#         durdivisors: if left unspecified, then the same list of divisors
#             is used for start and end of the note. Otherwise, it is possible
#             to define a specific grid for the end also
# 
#     Returns:
#         a tuple (start, duration) quantized to the given grids
#     """
#     if durdivisors is None:
#         durdivisors = divisors
#     ticks = [F(1, div) for div in divisors]
#     durticks = [F(1, div) for div in durdivisors]
#     start = snapToGrids(start, ticks)
#     end = snapToGrids(start + duration, durticks)
#     if end - start <= mindur:
#         end = nextInGrid(start + mindur, ticks)
#     return (start, end-start)


def showB(b: bool) -> str:
    """Show *b* as boolean"""
    return "T" if b else "F"


def quarterDurationToBaseDuration(d: F) -> int:
    """
    Convert duration in quarters to its base duration


    Args:
        d: the duration

    Returns:
        the base duration


    ================  ===============
    Quarter Duration  Base Duration
    ================  ===============
    4/1               1
    2/1               2
    1/1               4
    1/2               8
    1/4               16
    1/8               32
    1/16              64
    ================  ===============

    """
    assert ((d.denominator == 1 and d.numerator in {4, 2, 1}) or
            (d.numerator == 1 and d.denominator in {2, 4, 8, 16, 32})
            )
    return (d/4).denominator


_baseDurationToName = {
    1: 'whole',
    2: 'half',
    4: 'quarter',
    8: 'eighth',
    16: '16h',
    32: '32nd',
    64: '64th'
}


def baseDurationToName(baseDur: int) -> str:
    """
    Convert base duration to its name

    Args:
        baseDur: the base duration (1, 2, 4, 8, …)

    Returns:
        the corresponding name

    ====  ========
    Base  Name
    ====  ========
     1    while
     2    half
     4    quarter
     8    eighth
     16   16th
     32   32nd
     64   64th
    ====  ========
    """
    return _baseDurationToName[baseDur]


def baseDurationToQuarterDuration(b: int) -> F:
    """Convert base duration to quarter duration

    Example
    ~~~~~~~

        >>> baseDurationToQuarterDuration(4)
        1
        >>> baseDurationToQuarterDuration(8)
        0.5
    """
    return F(1, b)*4


def durationRatiosToTuplets(durRatios: Sequence[F]) -> list[tuple[int, int]]:
    tuplets = [(dr.numerator, dr.denominator) for dr in durRatios if dr != F(1)]
    return tuplets


def highestPowerLowerOrEqualTo(limit: int, base: int) -> int:
    """
    Returns the highest power of base which does not exceed limit

    Args:
        limit: the limit to the power of base
        base: the base

    Returns:
        the highest power of base not exceeding limit

    Example
    ~~~~~~~

        >>> highestPowerLowerOrEqualTo(100, 2)
        64

    .. seealso:: :func:`lowestPowerHigherOrEqualTo`
    """
    return base ** int(math.log(limit, base))


def lowestPowerHigherOrEqualTo(limit: int, base: int) -> int:
    return base ** int(math.ceil(math.log(limit, base)))


def parseScoreStructLine(line: str
                         ) -> tuple[int|None, timesig_t|None, float|None]:
    """
    Parse a line of a ScoreStructure definition

    Args:
        line: a line of the format [measureNum, ] timesig [, tempo]

    Returns:
        a tuple (measureNum, timesig, tempo), where only timesig
        is required
    """
    def parseTimesig(s: str) -> tuple[int, int]:
        try:
            num, den = s.split("/")
        except ValueError:
            raise ValueError(f"Could not parse timesig: {s}")
        return int(num), int(den)

    parts = [_.strip() for _ in line.split(",")]
    lenparts = len(parts)
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
        measureNumS, timesigS, tempoS = [_.strip() for _ in parts]
        measure = int(measureNumS) if measureNumS else None
        tempo = int(tempoS) if tempoS else None
    else:
        raise ValueError(f"Parsing error at line {line}")
    timesig = parseTimesig(timesigS) if timesigS else None
    return measure, timesig, tempo


def centsDeviation(pitch: float, divsPerSemitone=4) -> int:
    """
    The cents deviation to the nearest pitch in grid

    Args:
        pitch: the pitch to query
        divsPerSemitone: the number of divisions per semitone

    Returns:
        the deviationin cents from the nearest pitch in grid

    ======== ===============  ===============
    pitch    4 divs/semitone  2 divs/semitone
    ======== ===============  ===============
    60.0       0                0
    60.05      5                5
    60.1       10              10
    60.15      15              15
    60.2       20              20
    60.25      25              25
    60.3       30              30
    60.35      35              35
    60.4       40              40
    60.45      45              45
    60.5       50              50
    60.55      55              55
    60.6       60              60
    60.65     -35              65
    60.7      -30              70
    60.75     -25             -25
    60.8      -20             -20
    60.85     -15             -15
    60.9      -10             -10
    60.95     -5              -5
    ======== ===============  ===============
    """
    return pt.pitch_round(pitch, divsPerSemitone)[1]


def centsAnnotation(pitch: float | Sequence[float],
                    divsPerSemitone=4,
                    order='ascending',
                    addplus=False,
                    snap=2,
                    separator=',') -> str:
    """
    Generates the string used to annotate a note/chord when showCents is true

    Args:
        pitch: midinote/s as float
        divsPerSemitone: subdivisions of the semitone
        order: 'ascending' or 'descending'
        addplus: if True, add a plus sign for possitive deviations
        snap: if the difference to the nearest microtone is within this error,
            no annotation is shown
        separator: separator used for chords

    Returns:
        a string which can be attached to a note/chord to show
        the cents deviation from the notaten pitches

    """
    if isinstance(pitch, (int, float)):
        centsdev = centsDeviation(pitch, divsPerSemitone=divsPerSemitone)
        return str(centsdev)

    if order == 'ascending':
        pitches = sorted(pitch)
    else:
        pitches = sorted(pitch, reverse=True)

    centsdevs = [centsDeviation(p, divsPerSemitone) for p in pitches]

    annotations = [centsShown(centsdev, addplus=addplus, snap=snap) for centsdev in centsdevs]
    return separator.join(annotations) if any(annotations) else ""


def notatedDuration(duration: F, durRatios: Sequence[F] = ()
                    ) -> NotatedDuration:
    assert duration >= 0
    if duration == 0:
        # a grace note
        return NotatedDuration(0)

    dur = duration
    if durRatios and any(dr != 1 for dr in durRatios):
        tuplets = durationRatiosToTuplets(durRatios)
        for durRatio in durRatios:
            dur *= durRatio
    else:
        tuplets = None
    num, den = dur.numerator, dur.denominator
    assert den in {1, 2, 4, 8, 16, 32, 64}, f"Irregular denominator: {den}, {dur=}"
    assert num in {1, 3, 7} or den == 1
    if num == 1:
        # 1/1, 1/2, 1/4, 1/8
        # 1/2 -> base is 8 (8th note)
        return NotatedDuration(base=den*4, dots=0, tuplets=tuplets)
    elif num == 3:
        # 3/4, 3/8, ...,
        # 3/4 -> base 8 (8th note, dotted)
        return NotatedDuration(base=den*2, dots=1, tuplets=tuplets)
    elif num == 7:
        # 7/8 -> base 8 (8th note, double dotted)
        return NotatedDuration(base=den, dots=2, tuplets=tuplets)
    elif den == 1 and num in {2, 4}:
        return NotatedDuration(base=4//num, dots=0, tuplets=tuplets)
    else:
        raise ValueError(f"Invalid duration: {dur}")


_uuid_alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'


def makeuuid(size=8) -> str:
    return ''.join(random.choices(_uuid_alphabet, k=size))


def splitInterval(start: F, end: F, offsets: Sequence[F]
                  ) -> list[tuple[F, F]]:
    """
    Split interval (start, end) at the given offsets

    Args:
        start: start of the interval
        end: end of the interval
        offsets: offsets to split the interval at. Must be sorted

    Returns:
        a list of (start, end) segments where no segment extends over any
        of the given offsets

    Example
    ~~~~~~~

        >>> splitInterval(F(1), F(3), [F(1.5), F(2)])
        [(F(1), F(1.5)), (F(1.5),  F(2)), (F(2), F(3))]

    """
    assert end > start
    assert offsets

    if offsets[0] > end or offsets[-1] < start:
        # no intersection, return the original time range
        return [(start, end)]

    out = []
    for offset in offsets:
        if offset >= end:
            break
        if start < offset:
            out.append((start, offset))
            start = offset
    if start != end:
        out.append((start, end))

    assert len(out) >= 1
    return out


def fractionRange(start: F, stop: F, step: F = F(1)
                   ) -> Iterator[F]:
    """ Like range, but yielding Fractions """
    while start < stop:
        yield start
        start += step
