"""
String flageolets
"""
from __future__ import annotations
import math
import pitchtools as pt
from dataclasses import dataclass

from maelzel.common import F

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence


def _m2f(midinote: float, a4=442.) -> float:
    return 2 ** ((midinote - 69) / 12.0) * a4


def _f2m(freq: float, a4=442.) -> float:
    if freq < 9:
        return 0
    return 12.0 * math.log(freq / a4, 2) + 69.0


@dataclass
class Fret:
    fret: float
    midinote: float
    a4: float = 442

    @property
    def note(self):
        return pt.m2n(self.midinote)

    @property
    def freq(self):
        return _m2f(self.midinote, self.a4)

    def __repr__(self):
        return f"Fret(fret={self.fret}, note={self.note})"


@dataclass
class Node:
    midinote: float
    freq: float
    frets: list[Fret]

    @property
    def note(self):
        return pt.m2n(self.midinote)

    def __repr__(self):
        return "Node %s (%.2f Hz, %.2f) Frets: %s" % (
            self.note, self.freq, self.midinote, ", ".join(map(str, self.frets)))


class InstrumentString:
    """
    Defines the string of an instrument

    Args:
        pitch: The pitch of the string.
        fretsPerOctave: The number of frets per octave.

    Attributes:
        midi: The MIDI note corresponding to the pitch.
        fretsPerOctave: The number of frets per octave.
        freq: The frequency of the string.

    """
    def __init__(self, pitch: float | str, fretsPerOctave=12, name='', a4=442):
        self.midi = pitch if isinstance(pitch, (int, float)) else pt.n2m(pitch)
        self.fretsPerOctave = fretsPerOctave
        self.freq = _m2f(self.midi, a4=a4)
        self.name = name
        self.a4=a4

    @property
    def note(self) -> str:
        return pt.m2n(self.midi)

    def ratio2fret(self, ratio: float) -> float:
        """
        Convert a ratio to a fret number.

        Args:
            ratio: the position on the string (0-1)

        Returns:
            The fret number corresponding to the ratio.
        """
        if ratio != 1:
            return math.log(-1 / (ratio - 1)) / math.log(2) * self.fretsPerOctave
        else:
            return 0

    def fret2midi(self, fret: float) -> float:
        """
        Convert a fret number to a MIDI note.

        Args:
            fret: The fret number to convert.

        Returns:
            The MIDI note corresponding to the fret number.
        """
        return self.midi + 12 * (fret / self.fretsPerOctave)

    def fret2note(self, fret: float) -> str:
        """
        Convert a fret number to a note.

        Args:
            fret: The fret number to convert.

        Returns:
            The note corresponding to the fret number.
        """
        return pt.m2n(self.fret2midi(fret))

    def midi2fret(self, midinote: float) -> float:
        """
        Convert a MIDI note to a fret number.

        Args:
            midinote: The MIDI note to convert.

        Returns:
            The fret number corresponding to the MIDI note.
        """
        fret = (midinote - self.midi) / 12 * self.fretsPerOctave
        return fret

    def note2fret(self, note: str) -> float:
        """
        Convert a note to a fret number.

        Args:
            note: The note to convert.

        Returns:
            The fret number corresponding to the note.
        """
        return self.midi2fret(pt.n2m(note))

    def findNode(self, harmonic=2, minfret=0, maxfret=24) -> Node:
        """
        Find a node for the given harmonic between the specified frets

        Args:
            harmonic: The harmonic number to find
            minfret: min. fret number to search
            maxfret: max. fret number to search

        Returns:
            The fret number at which the natural harmonic is found

        NB: the fundamental (f0) is the 1st harmonic

        """
        frets: list[Fret] = []
        for i in range(1, harmonic):
            if math.gcd(i, harmonic) > 1:
                continue
            fret = self.ratio2fret(i / harmonic)
            if minfret <= fret <= maxfret:
                positionAsMidi = self.midi + 12 * (fret / self.fretsPerOctave)
                frets.append(Fret(fret, positionAsMidi, a4=self.a4))
        freq = self.freq * harmonic
        return Node(midinote=_f2m(freq, a4=self.a4), freq=freq, frets=frets)

    def flageolets(self, note, minfret=0, maxfret=24, kind=None):
        """
        Find the flageolets in this string which produce the given note

        Args:
            kind: None, or 'all': all type of flageolets
                  'natural': only natural flageolets
                'artificial: only artificial flageolets (3m, 3M, 4th, 5th)
        """
        raise NotImplementedError("not yet...")

    def nearestNode(self, note: str | float, maxHarmonic=16):
        """
        Find node closest to the given position

        Args:
            note: The position in the string as a note (str or midinumber)
            maxHarmonic: Consider only harmonics lower or equal to this harmonic

        Returns:
            a tuple (harmonic: int, note: float, fret: ??), where harmonic is ...,
            note is the node as midi pitch, and fret is ...
        """
        midinote = note if isinstance(note, (int, float)) else pt.n2m(note)
        fq = _m2f(midinote, self.a4)
        if fq < self.freq:
            raise ValueError("The given note is lower than the fundamental")
        ratio = F(self.freq/fq).limit_denominator(maxHarmonic)
        harmonic = ratio.denominator
        frets = self.findNode(harmonic).frets
        diff, fret_pos = min((abs(fret.freq - fq), fret) for fret in frets)
        resulting_freq = self.freq * harmonic
        return harmonic, _f2m(resulting_freq, self.a4), fret_pos

    def __mul__(self, other):
        return self.findNode(other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return self.freq + other

    def __mod__(self, other):
        return self.nearestNode(other)

    def __repr__(self):
        return f"InstrumentString({self.note}={self.freq}hz)"


class StringedInstrument:
    def __init__(self, pitches: Sequence[str], referenceFreq=442):
        self.strings = [InstrumentString(pitch, a4=referenceFreq) for pitch in pitches]

    def __getitem__(self, idx: int) -> InstrumentString:
        return self.strings[idx]

    @property
    def i(self):
        return self.strings[0]

    @property
    def ii(self):
        return self.strings[1]

    @property
    def iii(self):
        return self.strings[2]

    @property
    def iv(self):
        return self.strings[3]

    @property
    def v(self):
        return self.strings[4]

    @classmethod
    def violin(cls) -> StringedInstrument:
        return cls("5E 4A 4D 3G".split())

    @classmethod
    def viola(cls) -> StringedInstrument:
        return cls("4A 4D 3G 3C".split())

    @classmethod
    def cello(cls) -> StringedInstrument:
        return StringedInstrument("3A 3D 2G 2C".split())

    @classmethod
    def bass(cls) -> StringedInstrument:
        return StringedInstrument("2G 2D 1A 1E 0B".split())
