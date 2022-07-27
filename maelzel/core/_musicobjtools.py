from __future__ import annotations
from maelzel import packing
from maelzel.rational import Rat
from . import musicobj
from .workspace import getConfig
from emlib.iterlib import pairwise

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
    deviation = deviation or getConfig()['splitAcceptableDeviation']
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


def fillTempDynamics(items: list[Union[Note, Chord]], initialDynamic='mf',
                     resetMinGap=1
                     ) -> None:
    """
    Fill notes/chords with context dynamic as temporary (in place)

    To mark that the dynamic is temporary the property 'tempdynamic' is set to
    True. This routine is used for playback when config['play.useDynamics'] is True,
    to set the dynamic from the last dynamic if a note/chord of the same voice does
    not have a dynamic set

    Args:
        items: the notes/chords to modify. We assume that these form a voice, a consecutive
            list of notes without overlap
        initialDynamic: the dynamic to use at the beginning, if no dynamic present
        resetMinGap: if a distance between two events is longer than this the dynamic
            is reset to the initial dynamic. Set it to 0 to never reset

    Returns:

    """
    if not items:
        return
    elif len(items) == 1:
        item = items[0]
        if not item.dynamic:
            item.dynamic = initialDynamic
            item.properties['tempdynamic'] = True
    else:
        lastDynamic = initialDynamic
        lastEnd  = 0
        for item in items:
            if resetMinGap > 0 and item.start - lastEnd > resetMinGap:
                lastDynamic = initialDynamic
            if not item.dynamic:
                item.dynamic = lastDynamic
                item.properties['tempdynamic'] = True
            else:
                lastDynamic = item.dynamic
            lastEnd = item.end


def addDurationToGracenotes(chain: list[Note], dur: Rat) -> None:
    """
    Adds real duration to gracenotes within chain

    Previous to playback, gracenotes have a duration of 0. When playing back
    they are assigned a duration, which is substracted from the previous "real"
    note or silence.

    Args:
        chain: the sequence of notes to modify (in place)
        dur: the duration of a single gracenote

    Returns:

    """
    lastRealNote = None
    d = {}
    # first we build a registry mapping real notes to their grace notes
    for i, n in enumerate(chain):
        if not n.isGracenote():
            lastRealNote = i
        else:
            if lastRealNote is None:
                # first in the sequence!
                print("What to do???")
            else:
                gracenotes = d.get(lastRealNote)
                if gracenotes:
                    gracenotes.append(i)
                else:
                    d[lastRealNote] = [i]
    for realnoteIndex, gracenotesIndexes in d.items():
        realnote = chain[realnoteIndex]
        assert realnote.dur is not None
        realnote.dur -= dur * len(gracenotesIndexes)
        assert realnote.dur > 0
        for i, gracenoteIndex in enumerate(gracenotesIndexes):
            gracenote = chain[gracenoteIndex]
            gracenote.dur = dur
            deltapos = (len(gracenotesIndexes) - i) * dur
            gracenote.start -= deltapos



