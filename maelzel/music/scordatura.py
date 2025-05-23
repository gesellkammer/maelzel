from __future__ import annotations
from pitchtools import m2n, n2m, f2m, m2f
from emlib.iterlib import window as _window
from maelzel.music.flageolet import InstrumentString


class DetunedString:
    def __init__(self, name='IV', written='3G', sounding='3C'):
        self._name = name
        self._written = written
        self._sounding = sounding
        self._written_midi = n2m(written)
        self._sounding_midi = n2m(sounding)
        self._flageolet_string = InstrumentString(self._sounding)

    def note2sound(self, note: str) -> str:
        """
        Return the sounding note of the given played note

        Args:
            note: the played note (as a notename)

        Returns:
            the sounding note (as note name)

        """
        midinote = n2m(note)
        dx = midinote - self._written_midi
        if dx < 0:
            raise ValueError("This note is impossible in this string")
        sounding = self._sounding_midi + dx
        return m2n(sounding)

    def sound2note(self, note: str) -> str:
        midinote = n2m(note)
        dx = midinote - self._sounding_midi
        if dx < 0:
            raise ValueError("note not present in this string")
        written = self._written_midi + dx
        return m2n(written)

    def note2harmonic(self, written_note, kind='n'):
        """
        'n' = natural harmonic
        '4' = 4th harmonic
        '3M' = major third
        '3m' = minor third

        return the sounding pitch of the harmonic at the given written position

        if an artificial harmonic, the position refers to the position of the 'lower' finger
        """
        raise NotImplementedError

    def sound2harmonic(self, note: str, kind='all', tolerance=0.5):
        """
        find the harmonics in this string which can produce the given sound as result.

        Args:
            note: the note to produce (a string note)
            kind: kind of harmonic. One of [4, 3M, 3m, natural, all]
            tolerance: the acceptable difference between the desired note and the result
                       (in semitones)
        """
        midinote = n2m(note)
        if kind == '4' or kind == 4:
            f0 = midinote - 24
            out = self.sound2note(m2n(f0))
        elif kind == '3M' or kind == 3:
            f0 = midinote - 28
            out = self.sound2note(m2n(f0))
        elif kind == '3m':
            f0 = midinote - 31
            out = self.sound2note(m2n(f0))
        elif kind in ('n', 'natural'):
            fundamental = m2f(self._sounding_midi)
            harmonics = [f2m(fundamental * harmonic) for harmonic in range(12)]
            acceptable_harmonics = []
            for harmonic in harmonics:
                if abs(harmonic - midinote) <= tolerance:
                    acceptable_harmonics.append(harmonic)
            if len(acceptable_harmonics) > 0:
                # now find the position of the node in the string
                results = []
                nodes = []
                for harmonic in acceptable_harmonics:
                    fret = self._flageolet_string.ratio2fret(m2f(harmonic) / fundamental)
                    nodes.append(fret)
                for node in nodes:
                    for fret_pos in node.frets_pos:
                        results.append(fret_pos[1])  # we only append the pitch
                out = [self.sound2note(result) for result in results]
            else:
                out = None
        elif kind == 'all':
            out = []
            for kind in ('4 3M 3m n'.split()):
                out.append(self.sound2harmonic(note, kind=kind, tolerance=tolerance))
            return out
        return kind, out


def aslist(x):
    if isinstance(x, list):
        return x
    return aslist(x)


class DetunedInstrument:
    def __init__(self, i, ii, iii, iv):
        self.i, self.ii, self.iii, self.iv = i, ii, iii, iv

    def __getitem__(self, idx):
        if idx == 1:
            return self.i
        elif idx == 2:
            return self.ii
        elif idx == 3:
            return self.iii
        elif idx == 4:
            return self.iv
        else:
            raise ValueError("The index indicates the string, 1-4")

    def sound2note(self, note, strings=None):
        if strings is None:
            strings = (self.i, self.ii, self.iii, self.iv)
        notes = note.split()
        if len(notes) > 1:
            midinotes = [n2m(n) for n in notes]
            midinotes.sort(reverse=True)
            notes = [m2n(m) for m in midinotes]
            lines = []
            for strings in _window((self.i, self.ii, self.iii, self.iv), len(notes)):
                for string, note in zip(strings, notes):
                    lines.append(self.sound2note(note, [string]))
                lines.append('\n')
            return ''.join(lines)
        else:
            lines = []
            for string in strings:
                out = string.sound2note(note)
                if out:
                    lines.append("%s --> %s\n" % (string._name, out))
            return ''.join(lines)

    def note2sound(self, note):
        for string in (self.i, self.ii, self.iii, self.iv):
            out = string.note2sound(note)
            if out:
                print(string._name, '-->', out)

    def __repr__(self):
        return ' '.join(string._sounding for string in (self.iv, self.iii, self.ii, self.i))


def normalize_note(note):
    return m2n(n2m(note))


class Violin(DetunedInstrument):
    def __init__(self, g='3G', d='4D', a='4A', e='5E'):
        tunings = g.split()
        if len(tunings) == 4:
            g, d, a, e = tunings
        g, d, a, e = map(normalize_note, (g, d, a, e))
        iv, iii, ii, i = (
            DetunedString(name='IV', written='3G', sounding=g),
            DetunedString(name='III', written='4D', sounding=d),
            DetunedString(name='II', written='4A', sounding=a),
            DetunedString(name='I', written='5E', sounding=e)
        )
        super(Violin, self).__init__(i, ii, iii, iv)
        self.g, self.d, self.a, self.e = iv, iii, ii, i


class Viola(DetunedInstrument):
    def __init__(self, c='3C', g='3G', d='4D', a='4A'):
        tunings = g.split()
        if len(tunings) == 4:
            c, g, d, a = tunings
        c, g, d, a = map(normalize_note, (c, g, d, a))
        iv, iii, ii, i = (
            DetunedString(name='IV', written='3C', sounding=c),
            DetunedString(name='III', written='3G', sounding=g),
            DetunedString(name='II', written='4D', sounding=d),
            DetunedString(name='I', written='4A', sounding=a)
        )
        super().__init__(i, ii, iii, iv)
        self.c, self.g, self.d, self.a = iv, iii, ii, i
