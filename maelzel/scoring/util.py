from __future__ import annotations

import random

import pitchtools as pt

from maelzel.common import F
from maelzel.scoring.common import NotatedDuration
from maelzel._mathutils import ispowerof2
import functools

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence

# This module can only import .common from .

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
     1    whole
     2    half
     4    quarter
     8    eighth
     16   16th
     32   32nd
     64   64th
    ====  ========
    """
    return _baseDurationToName[baseDur]


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


def _notatedDuration(duration: F, durRatios: Sequence[F] = ()
                    ) -> NotatedDuration:
    assert duration >= 0
    if duration == 0:
        # a grace note
        return NotatedDuration(0)

    dur = duration
    if durRatios and any(dr != 1 for dr in durRatios):
        tuplets = [(d.numerator, d.denominator) for d in durRatios if d != F(1)]
        for dr in durRatios:
            dur *= dr
    else:
        tuplets = None
    num, den = dur.numerator, dur.denominator
    assert den in {1, 2, 4, 8, 16, 32, 64, 128}, f"Irregular denominator: {den}, {dur=}, orig. duration: {duration}, {durRatios=}"
    assert den == 1 or num in {1, 3, 7}
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


def durRepr(dur: F, maxdenom=100) -> str:
    if dur == 0:
        return "𝆔"
    elif int(dur) == dur or dur.denominator >= maxdenom:
        return f"{dur:.3f}".rstrip('0').rstrip('.') + '♩'
    else:
        return f"{dur.numerator}/{dur.denominator}♩"


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
    if num == 0:
        return (0, 0)

    if num == 1:
        assert ispowerof2(den), f"Invalid duration: {num}/{den}"
        # 1/1 -> 4
        # 1/2 -> 8
        # 1/4 -> 16
        return (den * 4, 0)

    for numdots in range(maxdots+1):
        # (2^(dots+1) - 1) is the pattern for dots
        dotsFactor = (2 ** (numdots + 1)) - 1

        # Check if numerator * 2^dots / dots_factor gives us an integer
        if (num * (2 ** numdots)) % dotsFactor == 0:
            k = (num * (2 ** numdots)) // dotsFactor
            assert k > 0, f"Invalid duration: {dur}, {num=}, {den=}, {dotsFactor=}, {numdots=}"
            base = (4 * den) // k

            # Verify and check base is valid
            if k * base == 4 * den and ispowerof2(base):
                return (base, numdots)

    raise ValueError(f"Cannot represent {num}/{den} as a standard "
                     f"duration with dots")



@functools.cache
def figureToDuration(base: int, dots: int) -> F:
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


@functools.cache
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
        if dur.numerator == 0:
            return "𝆔"
        if dur.numerator == 5:
            part1 = unicodeDuration(dur - F(1, dur.denominator))
            part2 = unicodeDuration(F(1, dur.denominator))
            return part1 + "͜" + part2
        elif dur.numerator in (1, 2, 3, 4, 6, 7, 8, 15):
            base, dots = durationToFigure(dur)
        else:
            raise ValueError(f"Invalid symbolic duration: {dur}")
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
        128: "1/128",
        256: "1/256"
    }

    figbase = figures.get(base)
    if not figbase:
        raise ValueError(f"Invalid figure, expected a power of 2 between 1 and 256, got {base}")
    return figbase + '·' * dots

