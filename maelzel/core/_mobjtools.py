from __future__ import annotations
from maelzel.common import F, F0
from emlib.iterlib import first

from .mevent import MEvent
from .event import Note, Chord
from .workspace import getConfig


from typing import Sequence


def splitNotesOnce(notes: Chord | Sequence[Note], splitpoint: float, deviation=None,
                   ) -> tuple[list[Note], list[Note]]:
    """
    Split a list of notes into two lists, one above and one below the splitpoint

    Args:
        notes: a seq. of Notes
        splitpoint: the pitch to split the notes
        deviation: an acceptable deviation to fit all notes
            in one part (config: 'splitAcceptableDeviation')

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


def fillTempDynamics(items: list[MEvent], initialDynamic='mf',
                     resetMinGap=1
                     ) -> None:
    """
    Fill notes/chords with context dynamic as temporary (inplace)

    To mark that the dynamic is temporary the property '.tempdynamic' is set to
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
            item.setProperty('.tempdynamic', True)
    else:
        lastDynamic = initialDynamic
        lastEnd = F0
        for item in items:
            itemOffset = item.relOffset()
            if resetMinGap > 0 and itemOffset - lastEnd > resetMinGap:
                lastDynamic = initialDynamic
            if not item.dynamic:
                item.dynamic = lastDynamic
                item.setProperty('.tempdynamic', True)
            else:
                lastDynamic = item.dynamic
            lastEnd = itemOffset + item.dur


def addDurationToGracenotes(events: list[MEvent], dur: F) -> None:
    """
    Adds real duration to gracenotes within chain

    Previous to playback, gracenotes have a duration of 0. Before playing
    they are assigned a duration, which is substracted from the previous "real"
    note or silence.

    Args:
        events: the sequence of notes to modify (inplace)
        dur: the duration of a single gracenote

    """
    lastRealNote = None
    d = {}
    # first we build a registry mapping real notes to their grace notes
    now = events[0].offset
    assert now is not None
    for i, n in enumerate(events):
        if not n.isGrace():
            lastRealNote = i
        else:
            if lastRealNote is None:
                # First in the sequence is a gracenote. Diminish the dur of the next real note,
                # make the gracenote "on the beat"
                nextrealidx = first(j for j, n in enumerate(events[i+1:])
                                    if not n.isGrace() and n.dur > 0)
                if nextrealidx is None:
                    raise ValueError(f"No real notes in {events=}")
                nextreal = events[nextrealidx+i+1]
                dur = min(dur, nextreal.dur / (nextrealidx + 1))
                assert dur > 0
                assert nextreal.dur > dur, f"{nextreal=}, {dur=}, {i=}, {nextrealidx=}"
                nextreal.dur -= dur
                assert nextreal.offset is not None
                nextreal.offset += dur
                n.dur = dur
                n.offset = now
                now += dur
            else:
                gracenotes = d.get(lastRealNote)
                if gracenotes:
                    gracenotes.append(i)
                else:
                    d[lastRealNote] = [i]

    for realnoteIndex, gracenotesIndexes in d.items():
        realnote = events[realnoteIndex]
        assert realnote.dur is not None and realnote.dur > 0
        maxGraceDur = realnote.dur / (len(gracenotesIndexes) + 1)
        graceDur = min(dur, maxGraceDur)
        realnote.dur -= graceDur * len(gracenotesIndexes)
        assert realnote.dur > 0, f"{realnote=}"
        for i, gracenoteIndex in enumerate(gracenotesIndexes):
            gracenote = events[gracenoteIndex]
            gracenote.dur = graceDur
            deltapos = (len(gracenotesIndexes) - i) * graceDur
            gracenote.offset -= deltapos


def groupLinkedEvents(items: list[MEvent],
                      mingap=F(1, 1000)) -> list[MEvent | list[MEvent]]:
    """
    Group linked events together

    Two events are linked if they are adjacent and the first event is either tied
    or has a glissando to the second event. This is used, for example, to merge
    the synth events of such linked groups into one line

    Args:
        items: a list of Note|Chord
        mingap: the min. gap between events to start a new group

    Returns:
        a list where each item is a note, a chord or itself a list of such
        notes or chords. A list indicates a linked event, where each item
        in such list is linked by either a tie or a gliss

    """
    lastitem = items[0]
    groups = [[lastitem]]
    for item in items[1:]:
        assert item.offset is not None and lastitem.end is not None
        gap = item.offset - lastitem.end
        if gap < 0:
            raise ValueError(f"Events supperpose: {lastitem=}, {item=}")
        elif gap > mingap:
            groups.append([item])
        elif not lastitem._canBeLinkedTo(item):
            groups.append([item])
        else:
            groups[-1].append(item)
        lastitem = item
    return [group[0] if len(group) == 1 else group for group in groups]


def splitLinkedGroupIntoLines(objs: list[MEvent]
                              ) -> list[list[Note]]:
    """
    Given a group as a list of Notes/Chords, split it in subgroups matching
    each note with its continuation.

    When one chords is followed by another chord and the first chord
    should do a glissando to the second, each note in the first chord is matched with
    a second note of the second chord (possibly duplicating the notes).

    This is purely intended for playback, so the duplication is not important.

    """
    if all(isinstance(obj, Note) for obj in objs):
        return [objs]  # type: ignore

    finished: list[list[Note]] = []
    started: list[list[Note]] = []
    continuations: dict[Note, Note] = {}
    for obj in objs:
        if isinstance(obj, Chord):
            for i, note in enumerate(obj.notes):
                note.offset = obj.offset
                note.dur = obj.dur
                note.gliss = obj.gliss if isinstance(obj.gliss, bool) else obj.gliss[i]
                note.tied = obj.tied
                if obj.playargs:
                    if not note.playargs:
                        note.playargs = obj.playargs.copy()
                    else:
                        note.playargs.fillWith(obj.playargs)

    # gliss pass
    if len(objs) > 1:
        ev0 = objs[0]
        for ev1 in objs[1:]:
            if isinstance(ev0, Chord) and ev0.gliss is True:
                if isinstance(ev1, Chord):
                    # Notes are matched in sort order (which is normally by pitch)
                    for n0, n1 in zip(ev0.notes, ev1.notes):
                        continuations[n0] = n1
                elif isinstance(ev1, Note):
                    for n0 in ev0.notes:
                        continuations[n0] = ev1
            ev0 = ev1

    for objidx, obj in enumerate(objs):
        if isinstance(obj, Chord):
            notes = obj.notes
        elif isinstance(obj, Note):
            notes = [obj]
        else:
            raise TypeError(f"Expected notes or chords, got {obj}")
        usednotes = set()
        assert all(n.offset is not None for n in notes)
        if not started:
            # No started group, so all notes here will start group
            for note in notes:
                started.append([note])
        else:
            # there are started groups, so iterate through started groups and
            # find if there are matches.
            for groupidx, group in enumerate(started):
                last = group[-1]
                if last.tied:
                    matchidx = next((i for i, n in enumerate(notes) if n.pitch == last.pitch), None)
                    if matchidx is not None:
                        group.append(notes[matchidx])
                        # notes.pop(matchidx)
                        usednotes.add(notes[matchidx])
                elif last.gliss is True:
                    if continuation := continuations.get(last):
                        group.append(continuation)
                        if continuation in notes:
                            usednotes.add(continuation)
                            # notes.remove(continuation)
                    else:
                        matchidx = min(range(len(notes)),
                                       key=lambda idx: abs(notes[idx].pitch - last.pitch))
                        group.append(notes[matchidx])
                        usednotes.add(notes[matchidx])
                else:
                    # This group's last note is not tied and has no gliss: this is the
                    # end of this group, so add it to finished
                    finished.append(group)
                    started.pop(groupidx)
            # Are there notes left? If yes, this notes did not match any started group,
            # so they must start a group themselves
            for note in notes:
                if note not in usednotes:
                    started.append([note])

    # We finished iterating, are there any started groups? Finish them
    finished.extend(started)
    return finished
