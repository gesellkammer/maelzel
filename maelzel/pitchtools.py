"""
Set of routinges to work with musical pitches, convert
to and from frequencies, notenames, etc.

There is a set of functions which all follow global settings
regarding, for example, the frequency for A4. In order to use
a custom value without interfering with any other clients of
the library, create a custom Converter

Example:

    >>> cnv = Converter(a4=435)
    >>> print(cnv.n2f("4C"))
    258.7
"""

from __future__ import annotations
import math
import re as _re
from typing import Tuple, List, NamedTuple, Union as U

import sys

_EPS = sys.float_info.epsilon


number_t = U[int, float]

_flats  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B", "C"]
_sharps = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C"]

_notes2 = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

_r1 = _re.compile(r"(?P<pch>[A-Ha-h][b|#]?)(?P<oct>[-]?[\d]+)(?P<micro>[-+><↓↑][\d]*)?")
_r2 = _re.compile(r"(?P<oct>[-]?\d+)(?P<pch>[A-Ha-h][b|#]?)(?P<micro>[-+><↓↑]\d*)?")


class NoteParts(NamedTuple):
    octave: int
    noteName: str
    alteration: str
    centsDeviation: int


class ParsedMidinote(NamedTuple):
    pitchindex: int
    alteration: float
    octave: int
    chromatic_pitch: str


class NotatedPitch(NamedTuple):
    octave: int
    diatonic_index: int
    diatonic_step: str
    chromatic_index: int
    chromatic_step: str
    diatonic_alteration: float
    chromatic_alteration: float
    accidental_name: str


class Converter:
    def __init__(self, a4=442.0, eightnote_symbol=True):
        """
        Convert between midinote, frequency and notename.

        Args:
            a4: the reference frequency
            eightnote_symbol: if True, a special symbol is used
                (">", "<") when a note is exactly 25 cents higher
                or lower (for example, "4C>"). Otherwise, a notename
                would be, for example, "4C+25"
        """
        self.a4 = a4
        self.eighthnote_symbol = eightnote_symbol

    def set_reference_freq(self, a4:float) -> None:
        self.a4 = a4

    def get_reference_freq(self) -> float:
        return self.a4

    def f2m(self, freq: float) -> float:
        """
        Convert a frequency in Hz to a midi-note

        See also: set_reference_freq, temporaryA4
        """
        if freq<9:
            return 0
        return 12.0*math.log(freq/self.a4, 2)+69.0

    def freqround(self, freq:float) -> float:
        return self.m2f(round(self.f2m(freq)))

    def m2f(self, midinote: float) -> float:
        """
        Convert a midi-note to a frequency

        See also: set_reference_freq, temporaryA4
        """
        return 2**((midinote-69)/12.0)*self.a4

    def m2n(self, midinote: float) -> str:
        """
        Convert midinote to notename

        Args:
            midinote: a midinote (60=C4)

        Returns:
            the notename corresponding to midinote.

        """
        octave, note, microtonal_alteration, cents = self.midi_to_note_parts(midinote)
        if cents == 0:
            return str(octave)+note+microtonal_alteration
        if cents>0:
            if cents<10:
                return f"{octave}{note}{microtonal_alteration}+0{cents}"
            return f"{octave}{note}{microtonal_alteration}+{cents}"
        else:
            if -10<cents:
                return f"{octave}{note}{microtonal_alteration}-0{abs(cents)}"
            return f"{octave}{note}{microtonal_alteration}{cents}"

    def n2m(self, note: str) -> float:
        return n2m(note)

    def n2f(self, note: str) -> float:
        return self.m2f(n2m(note))

    def f2n(self, freq: float) -> str:
        return self.m2n(self.f2m(freq))

    def pianofreqs(self, start="A0", stop="C8") -> List[float]:
        """
        Generate an array of the frequencies representing all the piano keys

        Args:
            start: the starting note
            stop: the ending note

        Returns:
            a list of frequencies
        """
        m0 = int(n2m(start))
        m1 = int(n2m(stop))
        midinotes = range(m0, m1+1)
        freqs = [self.m2f(m) for m in midinotes]
        return freqs

    def str2midi(self, s: str) -> float:
        """
        Accepts all that n2m accepts but with the addition of
        frequencies

        Possible values:

        "100hz", "200Hz", "4F+20hz", "8C-4hz"

        The hz part must be at the end
        """
        ending = s[-2:]
        if ending != "hz" and ending != "Hz":
            return self.n2m(s)
        srev = s[::-1]
        minusidx = srev.find("-")
        plusidx = srev.find("+")
        if minusidx<0 and plusidx<0:
            return self.f2m(float(s[:-2]))
        if minusidx>0 and plusidx>0:
            if minusidx<plusidx:
                freq = -float(s[-minusidx:-2])
                notename = s[:-minusidx-1]
            else:
                freq = float(s[-plusidx:-2])
                notename = s[:-plusidx-1]
        elif minusidx>0:
            freq = -float(s[-minusidx:-2])
            notename = s[:-minusidx-1]
        else:
            freq = float(s[-plusidx:-2])
            notename = s[:-plusidx-1]
        return self.f2m(self.n2f(notename)+freq)

    def midi_to_note_parts(self, midinote: float) -> NoteParts:
        """
        Convert a midinote into its parts as a note: 
            octave, notename, alteration, cents deviation

        Args:
            midinote: the midinote to analyze

        Returns:
            a NoteParts instance, a named tuple with the fields: `octave`, `noteName`,
            `alteracion` and `centsDeviation`

        """
        i = int(midinote)
        micro = midinote-i
        octave = int(midinote/12.0)-1
        ps = int(midinote%12)
        cents = int(micro*100+0.5)
        if cents == 0:
            return NoteParts(octave, _sharps[ps], "", 0)
        elif cents == 50:
            if ps in (1, 3, 6, 8, 10):
                return NoteParts(octave, _sharps[ps+1], "-", 0)
            return NoteParts(octave, _sharps[ps], "+", 0)
        elif cents == 25 and self.eighthnote_symbol:
            return NoteParts(octave, _sharps[ps], ">", 0)
        elif cents == 75 and self.eighthnote_symbol:
            ps += 1
            if ps>11:
                octave += 1
            if ps in (1, 3, 6, 8, 10):
                return NoteParts(octave, _flats[ps], "<", 0)
            else:
                return NoteParts(octave, _sharps[ps], "<", 0)
        elif cents>50:
            cents = 100-cents
            ps += 1
            if ps>11:
                octave += 1
            return NoteParts(octave, _flats[ps], "", -cents)
        else:
            return NoteParts(octave, _sharps[ps], "", cents)

    def normalize_notename(self, notename: str) -> str:
        return self.m2n(self.n2m(notename))


def n2m(note: str) -> float:
    """
    Converta notename to a midinote

    Two formats are supported (the 2nd format is preferable:

    # first format: C#2, D4, Db4+20, C4*, Eb5~
    # snd format  : 2C#, 4D+, 7Eb-14

    + = 1/4 note sharp
    - = 1/4 note flat
    * = 1/8 note sharp
    ~ = 1/8 note flat
    """
    if not isinstance(note, str):
        raise TypeError(f"expected a str, got {note} of type {type(note)}")

    if note[0].isalpha():
        m = _r1.search(note)
    else:
        m = _r2.search(note)
    if not m:
        raise ValueError("Could not parse note " + note)
    groups = m.groupdict()
    pitchstr = groups["pch"]
    octavestr = groups["oct"]
    microstr = groups["micro"]

    pc = _notes2[pitchstr[0].lower()]

    if len(pitchstr) == 2:
        alt = pitchstr[1]
        if alt == "#":
            pc += 1
        elif alt == "b":
            pc -= 1
        else:
            raise ValueError("Could not parse alteration in " + note)
    octave = int(octavestr)
    if not microstr:
        micro = 0.0
    elif microstr == "+":
        micro = 0.5
    elif microstr == "-":
        micro = -0.5
    elif microstr == ">" or microstr == "↑":
        micro = 0.25
    elif microstr == "<" or microstr == "↓":
        micro = -0.25
    else:
        micro = int(microstr) / 100.0

    if pc > 11:
        pc = 0
        octave += 1
    elif pc < 0:
        pc = 12 + pc
        octave -= 1
    return (octave + 1) * 12 + pc + micro


def _pitchname(pitchidx: int, micro: float) -> str:
    """
    Given a pitchindex (0-11) and a microtonal alteracion (between -0.5 and +0.5),
    return the pitchname which better represents pitchindex

    0, 0.4      -> C
    1, -0.2     -> Db
    3, 0.4      -> D#
    3, -0.2     -> Eb
    """
    blacknotes = {1, 3, 6, 8, 10}
    if micro < 0:
        if pitchidx in blacknotes:
            return _flats[pitchidx]
        else:
            return _sharps[pitchidx]
    elif micro == 0:
        return _sharps[pitchidx]
    else:
        if pitchidx in blacknotes:
            return _sharps[pitchidx]
        return _flats[pitchidx]


def parse_midinote(midinote: float) -> ParsedMidinote:
    """
    Convert a midinote into its pitch components:
        pitchindex, alteration, octave, chromaticPitch

    63.2   -> (3, 0.2, 4, "D#")
    62.8   -> (3, -0.2, 4, "Eb")
    """
    i = int(midinote)
    micro = midinote - i
    octave = int(midinote / 12.0) - 1
    ps = int(midinote % 12)
    cents = int(micro * 100 + 0.5)
    if cents == 50:
        if ps in (1, 3, 6, 8, 10):
            ps += 1
            micro = -0.5
        else:
            micro = 0.5
    elif cents > 50:
        micro = micro - 1.0
        ps += 1
        if ps == 12:
            octave += 1
            ps = 0
    pitchname = _pitchname(ps, micro)
    return ParsedMidinote(ps, round(micro, 2), octave, pitchname)


def ratio2interval(ratio: float) -> float:
    """
    Given two frequencies f1 and f2, calculate the interval between them

    f1 = n2f("C4")
    f2 = n2f("D4")
    interval = ratio2interval(f2/f1)   # --> 2 (semitones)
    """
    return 12 * math.log(ratio, 2)


def interval2ratio(interval: float) -> float:
    """
    Calculate the ratio r so that f1*r gives f2 so that
    the interval between f2 and f1 is the given one

    f1 = n2f("C4")
    r = interval2ratio(7)  # a 5th higher
    f2 = f2n(f1*r)  # --> G4
    """
    return 2 ** (interval / 12.0)


r2i = ratio2interval
i2r = interval2ratio


def pitchbend2cents(pitchbend: int, maxcents=200) -> int:
    """
    Convert a MIDI pitchband to the amount to set bent by the pitchwheel

    Args:
        pitchbend:
        maxcents:

    Returns:

    """
    return int(((pitchbend / 16383.0) * (maxcents * 2.0)) - maxcents + 0.5)


def cents2pitchbend(cents: int, maxcents=200) -> int:
    return int((cents + maxcents) / (maxcents * 2.0) * 16383.0 + 0.5)


_centsrepr = {
    '+': 50,
    '-': -50,
    '*': 25,
    '>': 25,
    '<': -25,
}

def split_notename(notename: str) -> Tuple[int, str, int, int]:
    """
    Return (octave, letter, alteration (1=#, -1=b), cents)

    4C#+10  -> (4, "C", 1, 10)
    Eb4-15  -> (4, "E", -1, -15)
    """

    def parse_centstr(centstr: str) -> int:
        if not centstr:
            return 0
        cents = _centsrepr.get(centstr)
        if cents is None:
            cents = int(centstr)
        return cents

    if not notename[0].isdecimal():
        # C#4-10
        cursor = 1
        letter = notename[0]
        l1 = notename[1]
        if l1 == "#":
            alter = 1
            octave = int(notename[2])
            cursor = 3
        elif l1 == "b":
            alter = -1
            octave = int(notename[2])
            cursor = 3
        else:
            alter = 0
            octave = int(notename[1])
            cursor = 2
        centstr = notename[cursor:]
        cents = parse_centstr(centstr)
    else:
        # 4C#-10
        octave = int(notename[0])
        letter = notename[1]
        rest = notename[2:]
        cents = 0
        alter = 0
        if rest:
            r0 = rest[0]
            if r0 == "b":
                alter = -1
                centstr = rest[1:]
            elif r0 == "#":
                alter = 1
                centstr = rest[1:]
            else:
                centstr = rest
            cents = parse_centstr(centstr)
    return octave, letter.upper(), alter, cents


def split_cents(notename: str) -> Tuple[str, int]:
    """
      input        output
    -------------------------
      "4E-"      ("4E", -50)
      "5C#+10"   ("5C#", 10)

    """
    octave, letter, alter, cents = split_notename(notename)
    alterchar = "b" if alter == -1 else "#" if alter == 1 else ""
    return str(octave) + letter + alterchar, cents


def enharmonic(notename: str) -> str:
    """
    original       enharmonic
    --------------------------
    4C+50          4Db-50
    4A+25          unchanged
    4G#+25         4Ab+25
    4Eb-25         4D#-25
    4G+30          unchanged

    """
    midinote = n2m(notename)
    diatonicsteps = "CDEFGAB"
    if int(midinote) == midinote:
        return notename
    octave, letter, alteration, cents = split_notename(notename)
    sign = "+" if cents > 0 else "-" if cents < 0 else ""
    pitchidx = diatonicsteps.index(letter)
    if alteration != 0:
        # a black key
        if alteration == 1:
            # turn sharp to flat
            basenote = diatonicsteps[pitchidx+1] + "b"
            return f"{octave}{basenote}{sign}{abs(cents)}"
        else:
            # turn flat into sharp
            basenote = diatonicsteps[pitchidx-1] + "#"
            return f"{octave}{basenote}{sign}{abs(cents)}"
    else:
        if cents == 50:
            # 4D+50 -> 4Eb-50
            # 4B+50 -> 5C-50
            if letter == "B":
                return f"{octave+1}C-"
            basenote = diatonicsteps[pitchidx+1] + "b"
            return f"{octave}{basenote}-"
        elif cents == -50:
            # 4D-50 -> 4C#+50
            # 4C-50 -> 3B+50
            if letter == "C":
                return f"{octave-1}B+"
            basenote = diatonicsteps[pitchidx+1]+"b"
            return f"{octave}{basenote}-"
        else:
            return notename


def pitch_round(midinote: float, semitoneDivisions=4) -> Tuple[str,int]:
    """
    Round midinote to the next pitch according to semitoneDivisions,
    returns the rounded notename and the cents deviation from the
    original pitch to the next semitone

    Args:
        midinote: the midinote to round, as float
        semitoneDivisions: the number of division per semitone

    Returns:
        a tuple (rounded note, cents deviation)

    Example::

        >>> pitch_round(60.2)
        ("4C", 20)

        >>> pitch_round(60.75)
        ("4D<", -25)
    """
    rounding_factor = 1 / semitoneDivisions
    rounded_midinote = round(midinote/rounding_factor)*rounding_factor
    notename = m2n(rounded_midinote)
    basename, cents = split_cents(notename)
    mididev = midinote-n2m(basename)
    centsdev = int(round(mididev*100))
    return notename, centsdev


def freq2mel(freq: float) -> float:
    return 1127.01048 * math.log(1. + freq/700)


def mel2freq(mel:float) -> float:
    return 700. * (math.exp(mel / 1127.01048) - 1.0)

_centsToAccidentalName = {
# cents   name
    0:   'natural',
    25:  'natural-up',
    50:  'quarter-sharp',
    75:  'sharp-down',
    100: 'sharp',
    125: 'sharp-up',
    150: 'three-quarters-sharp',

    -25: 'natural-down',
    -50: 'quarter-flat',
    -75: 'flat-up',
    -100:'flat',
    -125:'flat-down',
    -150:'three-quarters-flat'
}


def accidental_name(alterationCents: int, semitoneDivisions=4) -> str:
    assert semitoneDivisions in {1, 2, 4}, "semitoneDivisions should be 1, 2, or 4"
    if alterationCents < -150 or alterationCents > 150:
        raise ValueError(f"alterationCents should be between -150 and 150, got {alterationCents}")
    centsResolution = 100 // semitoneDivisions
    alterationCents = round(alterationCents / centsResolution) * centsResolution
    return _centsToAccidentalName[alterationCents]


def _roundres(x:float, resolution:float) -> float:
    return round(x/resolution)*resolution


def notated_pitch(midinote: float, divsPerSemitone=4) -> NotatedPitch:
    """

    Args:
        midinote:
        divsPerSemitone:

    Returns:

    """
    rounded_midinote = _roundres(midinote, 1/divsPerSemitone)
    parsed_midinote = parse_midinote(rounded_midinote)
    notename = m2n(rounded_midinote)
    octave, letter, alter, cents = split_notename(notename)
    basename, cents = split_cents(notename)
    chromaticStep = letter + ("#" if alter == 1 else "b" if alter == -1 else "")
    diatonicAlteration = alter+cents/100
    try:
        diatonic_index = "CDEFGAB".index(letter)
    except ValueError:
        raise ValueError(f"note step is not diatonic: {letter}")

    return NotatedPitch(octave=octave,
                        diatonic_index=diatonic_index,
                        diatonic_step=letter,
                        chromatic_index=parsed_midinote.pitchindex,
                        chromatic_step=chromaticStep,
                        diatonic_alteration=diatonicAlteration,
                        chromatic_alteration=cents/100,
                        accidental_name=accidental_name(int(diatonicAlteration*100)))


# --- Global functions ---

_converter = Converter()
midi_to_note_parts = _converter.midi_to_note_parts
set_reference_freq = _converter.set_reference_freq
get_reference_freq = _converter.get_reference_freq
f2m = _converter.f2m
m2f = _converter.m2f
m2n = _converter.m2n
n2f = _converter.n2f
f2n = _converter.f2n
freqround = _converter.freqround
normalize_notename = _converter.normalize_notename
str2midi = _converter.str2midi


# --- Amplitude converters ---

def db2amp(db: float) -> float:
    """
    convert dB to amplitude (0, 1)

    db: a value in dB
    """
    return 10.0 ** (0.05 * db)


def amp2db(amp: float) -> float:
    """
    convert amp (0, 1) to dB

    20.0 * log10(amplitude)

    :type amp: float|np.ndarray
    :rtype: float|np.ndarray

    """
    amp = max(amp, _EPS)
    return math.log10(amp) * 20

