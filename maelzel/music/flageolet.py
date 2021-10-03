"""
String flageolets
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
if TYPE_CHECKING:
    from typing import *
from collections import namedtuple as _namedtuple
from maelzel.rational import Rat
from pitchtools import *
from emlib.misc import returns_tuple as _returns_tuple
from maelzel.core import Note


class Fret(NamedTuple):
    fret: float
    midinote: float
    
    @property
    def note(self):
        return m2n(self.midinote)

    @property
    def freq(self):
        return m2f(self.midinote)
    
    def __repr__(self):
        return f"Fret({self.freq}, {self.note})" 
    

class Node(NamedTuple):
    midinote: float
    frets: List[int]
    
    @property
    def freq(self):
        return m2f(self.midinote)

    @property 
    def note(self):
        return m2n(self.midinote)

    def __repr__(self):
        return "Node %s (%.2f Hz, %.2f) Frets: %s" % (
            self.note, self.freq, self.midinote, ", ".join(map(str, self.frets)))


class InstrumentString:
    """
    Defines the string of an instrument
    """
    def __init__(self, pitch: Union[float, str], frets_per_octave=12):
        """
        """
        self.midi = pitch if isinstance(pitch, (int, float)) else n2m(pitch)
        self.frets_per_octave = frets_per_octave
        self.freq = m2f(self.midi)
        
    def ratio2fret(self, ratio:float) -> float:
        """
        Args:
            ratio: the position on the string (0-1)
        """
        if ratio != 1:
            return math.log(-1 / (ratio - 1)) / math.log(2) * self.frets_per_octave
        else:
            return 0

    def fret2midi(self, fret: float) -> float:
        return self.midi + 12 * (fret / self.frets_per_octave)

    def fret2note(self, fret: float) -> str:
        return m2n(self.fret2midi(fret))

    def midi2fret(self, midinote: float) -> float:
        fret = (midinote - self.midi) / 12 * self.frets_per_octave
        return fret

    def note2fret(self, note: str) -> float:
        return self.midi2fret(n2m(note))

    def find_node(self, harmonic=2, minfret=0, maxfret=24):
        """
        Find a node for the given harmonic between the specified frets

        Args:
            harmonic (int)         : The harmonic number to find
            minfret, maxfret (int) : Search within these fret-numbers

        Returns:
            The fret number at which the natural harmonic is found

        NB: the fundamental (f0) is the 1st harmonic

        """
        frets = []
        for i in range(1, harmonic):
            if math.gcd(i, harmonic) > 1:
                continue
            fret = self.ratio2fret(i / harmonic)
            if fret >= minfret and fret <= maxfret:
                position_as_midi = self.midi + 12 * (fret / self.frets_per_octave)
                frets.append(Fret(fret, position_as_midi))
        freq = self.freq * harmonic
        return Node(f2m(freq), frets)

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
    def nearest_node(self, note, max_harmonic=16):
        """
        Find node closest to the given position

        note (str|num)     : The position in the string as a note (str or midinumber)
        max_harmonic (int) : Consider only harmonics lower or equal to this harmonic
        """
        fq = n2f(note) if isinstance(note, str) else m2f(note)
        if fq < self.freq:
            raise ValueError("The given note is lower than the fundamental")
        ratio = Rat(self.freq/fq).limit_denominator(max_harmonic)
        harmonic = ratio.denominator
        frets = self.find_node(harmonic).frets
        diff, fret_pos = min((abs(fret.freq - fq), fret) for fret in frets)
        resulting_freq = self.freq * harmonic
        return harmonic, Note(f2m(resulting_freq)), fret_pos

    def __mul__(self, other):
        return self.find_node(other)
        
    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return self.freq + other

    def __mod__(self, other):
        return self.nearest_node(other)

    def __repr__(self):
        return "%f Hz | %s | %f midi" % (self.freq, f2n(self.freq), f2m(self.freq))


class StringedInstrument:
    def __init__(self, pitches: List[Union[float, str]]):
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
