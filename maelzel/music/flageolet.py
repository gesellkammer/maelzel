"""
String flageolets
"""
from __future__ import annotations
from maelzel.common import F
import pitchtools as pt
import math
from emlib.misc import returns_tuple as _returns_tuple
from maelzel.core import Note
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence


@dataclass
class Fret:
    fret: float
    midinote: float

    @property
    def note(self):
        return pt.m2n(self.midinote)

    @property
    def freq(self):
        return pt.m2f(self.midinote)

    def __repr__(self):
        return f"Fret(fret={self.fret}, note={self.note})"


@dataclass
class Node:
    midinote: float
    frets: list[Fret]

    @property
    def freq(self):
        return pt.m2f(self.midinote)

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
    def __init__(self, pitch: float | str, fretsPerOctave=12):
        self.midi = pitch if isinstance(pitch, (int, float)) else pt.n2m(pitch)
        self.fretsPerOctave = fretsPerOctave
        self.freq = pt.m2f(self.midi)

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
                frets.append(Fret(fret, positionAsMidi))
        freq = self.freq * harmonic
        return Node(pt.f2m(freq), frets)

    def flageolets(self, note, minfret=0, maxfret=24, kind=None):
        """
        Find the flageolets in this string which produce the given note

        Args:
            kind: None, or 'all': all type of flageolets
                  'natural': only natural flageolets
                'artificial: only artificial flageolets (3m, 3M, 4th, 5th)
        """
        raise NotImplementedError("not yet...")

    @_returns_tuple("harmonic note fret")
    def nearestNode(self, note: str | float, maxHarmonic=16):
        """
        Find node closest to the given position

        Args:
            note: The position in the string as a note (str or midinumber)
            maxHarmonic: Consider only harmonics lower or equal to this harmonic
        """
        fq = pt.n2f(note) if isinstance(note, str) else pt.m2f(note)
        if fq < self.freq:
            raise ValueError("The given note is lower than the fundamental")
        ratio = F(self.freq/fq).limit_denominator(maxHarmonic)
        harmonic = ratio.denominator
        frets = self.findNode(harmonic).frets
        diff, fret_pos = min((abs(fret.freq - fq), fret) for fret in frets)
        resulting_freq = self.freq * harmonic
        return harmonic, Note(pt.f2m(resulting_freq)), fret_pos

    def __mul__(self, other):
        return self.findNode(other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return self.freq + other

    def __mod__(self, other):
        return self.nearestNode(other)

    def __repr__(self):
        return f"{self.freq} Hz | {pt.f2n(self.freq)} | {pt.f2m(self.freq)} midi"


class StringedInstrument:
    def __init__(self, pitches: Sequence[str]):
        self.strings = [InstrumentString(pitch) for pitch in pitches]

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


violin = StringedInstrument("5E 4A 4D 3G".split())
viola = StringedInstrument("4A 4D 3G 3C".split())
cello = StringedInstrument("3A 3D 2G 2C".split())
bass = StringedInstrument("2G 2D 1A 1E 0B".split())
