from __future__ import annotations
from maelzel.music import packing
from maelzel.rational import Rat
from . import musicobj
from .workspace import activeConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from .musicobj import Note, Chord, Voice, MusicObj


def packInVoices(objs: List[MusicObj]) -> List[Voice]:
    """
    Distribute the items across voices
    """
    items = []
    unpitched = []
    for obj in objs:
        assert obj.start is not None and obj.dur is not None, \
            "Only objects with an explict start / duration can be packed"
        r = obj.pitchRange()
        if r is None:
            unpitched.append(r)
        else:
            pitch = (r[0] + r[1]) / 2
            item = packing.Item(obj,
                                offset=Rat(obj.start.numerator, obj.start.denominator),
                                dur=Rat(obj.dur.numerator, obj.dur.denominator),
                                step=pitch)
            items.append(item)
    tracks = packing.packInTracks(items)
    voices = [musicobj.Voice(track.unwrap()) for track in tracks]
    return voices


def splitNotesOnce(notes: Union[Chord, Sequence[Note]], splitpoint:float, deviation=None,
                    ) -> Tuple[List[Note], List[Note]]:
    """
    Split a list of notes into two lists, one above and one below the splitpoint

    Args:
        notes: a seq. of Notes
        splitpoint: the pitch to split the notes
        deviation: an acceptable deviation to fit all notes
            in one group (config: 'splitAcceptableDeviation')

    Returns:
        notes above and below

    """
    deviation = deviation or activeConfig()['splitAcceptableDeviation']
    if all(note.pitch > splitpoint - deviation for note in notes):
        above = [n for n in notes]
        below = []
    elif all(note.pitch < splitpoint + deviation for note in notes):
        above = []
        below = [n for n in notes]
    else:
        above, below = [], []
        for note in notes:
            (above if note.pitch > splitpoint else below).append(note)
    return above, below


def splitNotesIfNecessary(notes:List[Note], splitpoint:float, deviation=None
                          ) -> List[List[Note]]:
    """
    Like _splitNotesOnce, but returns only groups which have notes in them

    This can be used to split in more than one staves, which should not overlap

    Args:
        notes: the notes to split
        splitpoint: the split point
        deviation: an acceptable deviation, if all notes could fit in one part

    Returns:
        a list of parts (a part is a list of notes)

    """
    return [p for p in splitNotesOnce(notes, splitpoint, deviation) if p]