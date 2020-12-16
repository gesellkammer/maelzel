import music21 as m21
from emlib.music import m21tools
from . import core
# from . import quantization


def m21Note(note: core.Note) -> m21.note.Note:
    """
    Create a m21.Note from a scoring Note
    Offset will not be taken into account

    Args:
        note: the scoring Note to convert

    Returns:
        the corresponding m21 Note
    """
    m21note, centsdev = m21tools.makeNote(note.pitch, quarterLength=note.dur)
    return m21note


def m21Chord(chord: core.Chord) -> m21.chord.Chord:
    """
    Create a m21.Chord from a scoring Chord
    Offset will not be taken into account

    Args:
        chord: the scoring Chord to convert

    Returns:
        the corresponding m21 Chord
    """
    m21chord, dentsdevs = m21tools.makeChord(chord.pitches,
                                             quarterLength=chord.dur)
    return m21chord

