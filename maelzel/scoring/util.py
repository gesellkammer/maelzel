from __future__ import annotations
import random
from emlib import iterlib
from emlib import misc
import pitchtools as pt
from .common import F, asF, NotatedDuration
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from .common import timesig_t
    from numbers import Real

# This module can only import .common from .


def asSimplestNumberType(f: F) -> Union[int, float]:
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
    The totalDuration in quarter notes of a measure according to its time signature

    Args:
        timesig: a tuple (num, den)

    Returns:
        the totalDuration in quarter notes

    Examples::

        >>> measureQuarterDuration((3,4))
        Fraction(3, 1)

        >>> measureQuarterDuration((5, 8))
        Fraction(5, 2)

    """
    misc.assert_type(timesig, (int, int))
    num, den = timesig
    quarterDuration = F(num)/den * 4
    assert isinstance(quarterDuration, F), f"Expected type F, got {type(quarterDuration).__name__}={quarterDuration}"
    return quarterDuration


def measureTimeDuration(timesig: timesig_t, quarterTempo: F) -> F:
    """
    The totalDuration of a measure in seconds

    Args:
        timesig: the time signature, a tuple (num, den)
        quarterTempo: the tempo of the quarter note

    Returns:
        The totalDuration **in seconds**

    """
    misc.assert_type(timesig, (int, int))
    quarterTempo = asF(quarterTempo)
    quarters = measureQuarterDuration(timesig=timesig)
    dur = quarters * (F(60)/quarterTempo)
    assert isinstance(dur, F), f"Expected type F, got {type(dur).__name__}={dur}"
    return dur




def _measureOffsets(timesig: timesig_t, quarterTempo: Real) -> List[F]:
    """
    Returns a list with the offsets of all beats in measure.

    The last value refers to the offset of the end of the measure

    Args:
        timesig: the timesignature as a tuple (num, den)
        quarterTempo: the tempo correponding to a quarter note

    Returns:
        a list of fractions representing the start time of each beat, plus the
        end time of the measure (== the start time of the next measure)

    Example::
        >>> measureOffsets((5, 8), 60)
        [Fraction(0, 1), Fraction(1, 1), Fraction(2, 1), Fraction(5, 2)]
        # 0, 1, 2, 2.5
    """
    misc.assert_type(timesig, (int, int))
    quarterTempo = asF(quarterTempo)

    beatDurations = measureBeats(timesig, quarterTempo=quarterTempo)
    beatOffsets = [F(0)] + list(iterlib.partialsum(beatDurations))
    return beatOffsets


def clefNameFromMidinotes(midis: List[float]) -> str:
    """
    Find the best clef to fit these notes

    Args:
        midis: a list of midinotes

    Returns:
        a string describing the clef type, one of 'treble', 'bass', 'treble8va', 'bass8vb'
    """
    if not midis:
        return "treble"
    avg = sum(midis)/len(midis)
    if avg>80:
        return "treble8"
    elif avg>58:
        return "treble"
    elif avg>36:
        return "bass"
    else:
        return "bass8"


def midinotesNeedSplit(midinotes: List[float], splitpoint=60, margin=4) -> bool:
    """
    Do these midinotes need to be split across multiple stafs?

    Args:
        midinotes: the midnotes to analyze
        splitpoint: the splitpoint
        margin: ??

    Returns:
        True if the midinotes need to be split

    """
    if len(midinotes) == 0:
        return False
    numabove = sum(int(m > splitpoint - margin) for m in midinotes)
    numbelow = sum(int(m < splitpoint + margin) for m in midinotes)
    return bool(numabove and numbelow)


def midinotesNeedMultipleClefs(midinotes: List[float], threshold=1) -> bool:
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


def centsShown(centsdev: int, divsPerSemitone=4, snap=2) -> str:
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
    return str(int(centsdev))


def nextInGrid(x: Union[float, F], ticks: List[F]) -> F:
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
    out = misc.snap_to_grids(x + F(1, 9999999), ticks, mode='ceil')
    return asF(out)


def snapTime(start: F,
             duration: F,
             divisors: List[int],
             mindur=F(1, 16),
             durdivisors: List[int]=None) -> Tuple[F, F]:
    """
    Quantize an event to snap to a grid defined by divisors and durdivisors

    Args:
        start: the start of the event
        duration: the totalDuration of the event
        divisors: a list of divisions of the pulse
        mindur: the min. totalDuration of the note
        durdivisors: if left unspecified, then the same list of divisors
            is used for start and end of the note. Otherwise, it is possible
            to define a specific grid for the end also

    Returns:
        a tuple (start, totalDuration) quantized to the given grids
    """
    if durdivisors is None:
        durdivisors = divisors
    ticks = [F(1, div) for div in divisors]
    durticks = [F(1, div) for div in durdivisors]
    start = asF(misc.snap_to_grids(start, ticks))
    end = asF(misc.snap_to_grids(start + duration, durticks))
    if end - start <= mindur:
        end = nextInGrid(start + mindur, ticks)
    return (start, end-start)


def showF(f:F, maxdenom=1000) -> str:
    """
    Show a fraction, limit den to *maxdenom*

    Args:
        f:

    Returns:

    """
    if f.denominator > maxdenom:
        f2 = f.limit_denominator(maxdenom)
        return "*%d/%d" % (f2.numerator, f2.denominator)
    return "%d/%d" % (f.numerator, f.denominator)


def showT(f:Optional[F]) -> str:
    """Show *f* as time"""
    if f is None:
        return "None"
    return f"{float(f):.3f}"


def showB(b:bool) -> str:
    """Show *b* as boolean"""
    return "MObjT" if b else "F"


def parseNotehead(s: str) -> Tuple[str, Optional[bool]]:
    """
    Parse notehead description

    Args:
        s: the notehead description (a str)

    Returns:
        a tuple (notehead:str, filled: bool)


    **Format**:

    - "diamond"
    - "diamond.unfilled"
    - "harmonic.filled"

    """
    defaultFilled = {
        'harmonic': False,
    }

    parts = s.split(".")
    l = len(parts)
    if l == 1:
        shape = parts[0]
        filled = defaultFilled.get(shape)
        return (parts[0], filled)
    if l == 2:
        notehead = parts[0]
        filled = parts[1] == "filled"
        return (notehead, filled)
    raise ValueError("Notehead format not understood")


def quarterDurationToBaseDuration(d: F) -> int:
    """
    Convert totalDuration in quarters to its base totalDuration


    Args:
        d: the totalDuration

    Returns:
        the base totalDuration


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
    Convert base totalDuration to its name

    Args:
        baseDur: the base totalDuration (1, 2, 4, 8, …)

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
    """Convert base totalDuration to quarter totalDuration

    Example
    ~~~~~~~

        >>> baseDurationToQuarterDuration(4)
        1
        >>> baseDurationToQuarterDuration(8)
        0.5
    """
    return F(1, b)*4


def durationRatiosToTuplets(durRatios: List[F]) -> List[Tuple[int, int]]:
    tuplets = [(dr.numerator, dr.denominator) for dr in durRatios if dr != F(1)]
    return tuplets


def parseScoreStructLine(line: str
                         ) -> Tuple[Optional[int], Optional[timesig_t], Optional[float]]:
    """
    Parse a line of a ScoreStructure definition

    Args:
        line: a line of the format [measureNum, ] timesig [, tempo]

    Returns:
        a tuple (measureNum, timesig, tempo), where only timesig
        is required
    """
    def parseTimesig(s: str) -> Tuple[int, int]:
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

def fixedPitchCentsAnnotation(note: str) -> str:
    """

    4C+60    4C+     60
    4C+      4C+     -
    4C+30    4C>     30
    4C+80    4Db-20 -20

    Args:
        note:

    Returns:

    """

def centsAnnotation(pitch: Union[float, List[float]], divsPerSemitone=4,
                    order='ascending') -> str:
    """
    Generates the string used to annotate a note/chord when showCents is true
    
    Args:
        pitch: midinote/s as float
        divsPerSemitone: subdivisions of the semitone
        order: 'ascending' or 'descending'

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
    annotations = [str(centsShown(centsdev)) for centsdev in centsdevs]
    return ",".join(annotations) if any(annotations) else ""


def notatedDuration(duration: F, durRatios: Optional[List[F]]
                    ) -> NotatedDuration:
    assert duration >= 0
    if duration == 0:  #  a grace note
        return NotatedDuration(0)

    dur = duration
    if durRatios and any(dr != F(1) for dr in durRatios):
        tuplets = durationRatiosToTuplets(durRatios)
        for durRatio in durRatios:
            dur *= durRatio
    else:
        tuplets = None
    num, den = dur.numerator, dur.denominator
    assert den in {1, 2, 4, 8, 16, 32, 64}
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
        raise ValueError(f"Invalid totalDuration: {dur}")


_uuid_alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'


def makeuuid(size=8) -> str:
    return ''.join(random.choices(_uuid_alphabet, k=size))
