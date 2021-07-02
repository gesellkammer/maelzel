from collections import namedtuple as _namedtuple
from fractions import Fraction as _Fraction
from pitchtools import *
from emlib.misc import returns_tuple as _returns_tuple
from maelzel.core import Note

class Fret(_namedtuple("Fret", "fret midinote")):
    @property
    def note(self):
        return m2n(self.midinote)

    @property
    def freq(self):
        return m2f(self.midinote)
    
    def __repr__(self):
        return "Fret %.2f %s (%.1f)" % (self.fret, self.note, self.midinote)


class Node(_namedtuple("Node", "midinote frets")):
    @property
    def freq(self):
        return m2f(self.midinote)

    @property 
    def note(self):
        return m2n(self.midinote)

    def __repr__(self):
        return "Node %s (%.2f Hz, %.2f) Frets: %s" % (
            self.note, self.freq, self.midinote, ", ".join(map(str, self.frets)))


class InstrumentString(object):
    """
    Defines the string of an instrument
    """
    def __init__(self, freq, frets_per_octave=12):
        """
        """
        if isinstance(freq, str):
            freq = n2f(freq)
        self._frets_per_octave = frets_per_octave
        self.freq = freq
        self.midi = f2m(freq)

    def ratio2fret(self, ratio):
        """
        ratio (float, 0-1) : the position on the string
        """
        if ratio != 1:
            return math.log(-1 / (ratio - 1)) / math.log(2) * self._frets_per_octave
        else:
            return 0

    def fret2midi(self, fret):
        return self.midi + 12 * (fret / self._frets_per_octave)

    def fret2note(self, fret):
        return m2n(self.fret2midi(fret))

    def midi2fret(self, midinote):
        fret = (midinote - self.midi) / 12 * self._frets_per_octave
        return fret

    def note2fret(self, note):
        return self.midi2fret(n2m(note))

    def find_node(self, harmonic=2, minfret=0, maxfret=24):
        """
        Find a node for the given harmonic between the specified frets

        harmonic (int)         : The harmonic number to find
        minfret, maxfret (int) : Search within these fret-numbers

        NB: the fundamental (f0) is the 1st harmonic

        Returns => The fret number at which the natural harmonic is found
        """
        frets = []
        for i in range(1, harmonic):
            if math.gcd(i, harmonic) > 1:
                continue
            fret = self.ratio2fret(i / harmonic)
            if fret >= minfret and fret <= maxfret:
                position_as_midi = self.midi + 12 * (fret / self._frets_per_octave)
                frets.append(Fret(fret, position_as_midi))
        freq = self.freq * harmonic
        return Node(f2m(freq), frets)

    def flageolets(self, note, minfret=0, maxfret=24, kind=None):
        """
        Find the flageolets in this string which produce the given note

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
        ratio = _Fraction(self.freq/fq).limit_denominator(max_harmonic)
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

_Violin = _namedtuple("Violin", "i ii iii iv")
_Viola = _namedtuple("Viola","i ii iii iv")
_Cello = _namedtuple("Cello","i ii iii iv")
_Bass = _namedtuple("Bass", "i ii iii iv v")

violin = _Violin(*map(InstrumentString, "5E 4A 4D 3G".split()))
viola = _Viola(*map(InstrumentString, "4A 4D 3G 3C".split()))
cello = _Cello(*map(InstrumentString, "3A 3D 2G 2C".split()))
bass = _Bass(*map(InstrumentString, "2G 2D 1A 1E 0B".split()))


def nearest_node(fundamental, note, max_harmonic=16):
    """
    Given an arbitrary fundamental, return the nearest node
    to the given note.

    fundamental: a midinote or a notename
    note: a midinote or a notename
    max_harmonic: an integer

    NB: for an artificial harmonic, note should be within reach of 
        the fundamental

    Example: TODO
    """
    freq = _note2freq(fundamental)
    s = InstrumentString(freq)
    return s.nearest_node(note, max_harmonic=max_harmonic)


def find_node(fundamental, harmonic=2, minfret=0, maxfret=24):
    freq = _note2freq(fundamental)
    s = InstrumentString(freq)
    return s.find_node(harmonic, minfret=minfret, maxfret=maxfret)


def _note2freq(note):
    return n2f(note) if isinstance(note, str) else m2f(note)
    
