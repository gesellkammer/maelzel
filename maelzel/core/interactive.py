from __future__ import annotations
from maelzel.core import Note, Chord
from maelzel.core import _tools
import pitchtools as pt


def generateNotes(start=12, end=127) -> dict[str, Note]:
    """
    Generates all notes for interactive use.

    From an interactive session:

    .. code-block:: python

        >>> from maelzel.core import *
        >>> locals().update(generateNotes())
    """
    notes = {}
    for i in range(start, end):
        notename = pt.m2n(i)
        octave = notename[0]
        rest = notename[1:]
        rest = rest.replace('#', 'x')
        original_note = rest + str(octave)
        notes[original_note] = Note(i)
        if "x" in rest or "b" in rest:
            enharmonic_note = _util.enharmonic(rest)
            enharmonic_note += str(octave)
            notes[enharmonic_note] = Note(i)
    return notes