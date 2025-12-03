from __future__ import annotations

from maelzel.common import F, F0
from .mevent import MEvent


def fillTempDynamics(items: list[MEvent],
                     dynamic='mf',
                     key='.tempdynamic',
                     reset=1,
                     apply=False
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
        dynamic: the dynamic to use at the beginning, if no dynamic present
        key: the key to set within the event's properties
        reset: if a distance between two events is longer than this the dynamic
            is reset to the initial dynamic. Set it to 0 to never reset

    """
    if not items:
        return
    elif len(items) == 1:
        item = items[0]
        if not item.dynamic:
            if apply:
                item.dynamic = dynamic
            item.setProperty(key, dynamic)
    else:
        lastDynamic = dynamic
        lastEnd = F0
        for item in items:
            itemOffset = item.relOffset()
            if reset > 0 and itemOffset - lastEnd > reset:
                lastDynamic = dynamic
            if not item.dynamic:
                if apply:
                    item.dynamic = lastDynamic
                item.setProperty(key, lastDynamic)
            else:
                lastDynamic = item.dynamic
            lastEnd = itemOffset + item.dur

def copyEventsModifiedByGracenotes(events: list[MEvent]) -> list[MEvent]:
    out = []
    insideGraceGroup = False
    for idx, ev in enumerate(events):
        if ev.isGrace():
            ev = ev.copy()
            insideGraceGroup = True
        else:
            if insideGraceGroup:
                insideGraceGroup = False
                ev = ev.copy()
        out.append(ev)
    return out


def addDurationToGracenotes(events: list[MEvent], dur: F
                            ) -> None:
    """
    Adds real duration to gracenotes within chain (in place)

    Previous to playback, gracenotes have a duration of 0. Before playing
    they are assigned a duration, which is substracted from the previous "real"
    note or silence.

    Args:
        events: the sequence of notes to modify (inplace)
        dur: the duration of a single gracenote

    """
    lastRealNote = -1
    d: dict[int, list[int]] = {}
    # first we build a registry mapping real notes to their grace notes
    now = events[0].relOffset()
    assert now is not None
    for i, n in enumerate(events):
        if not n.isGrace():
            lastRealNote = i
        else:
            if lastRealNote < 0:
                # First in the sequence is a gracenote. Diminish the dur of the next real note,
                # make the gracenote "on the beat"
                nextrealidx = next((j for j, n in enumerate(events[i+1:])
                                    if not n.isGrace() and n.dur > 0), None)
                if nextrealidx is None:
                    raise ValueError(f"No real notes in {events=}")
                nextreal = events[nextrealidx+i+1]
                dur = min(dur, nextreal.dur / (nextrealidx + 1))
                assert dur > 0 and nextreal.dur > dur, f"{nextreal=}, {dur=}, {i=}, {nextrealidx=}"
                nextreal.dur -= dur
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
        maxGraceDur = realnote.dur / (len(gracenotesIndexes) + 1)
        graceDur = min(dur, maxGraceDur)
        realnote.dur -= graceDur * len(gracenotesIndexes)
        for i, gracenoteIndex in enumerate(gracenotesIndexes):
            gracenote = events[gracenoteIndex]
            gracenote.dur = graceDur
            deltapos = (len(gracenotesIndexes) - i) * graceDur
            gracenote.offset -= deltapos


def groupLinkedEvents(items: list[MEvent],
                      mingap=F(1, 1000)) -> list[MEvent | list[MEvent]]:
    """
    Group linked events together. Events are time-sorted and non-contiguous

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
    # assert all(ev1.relOffset() < ev2.relOffset() for ev1, ev2 in itertools.pairwise(items))
    lastitem = items[0]
    groups: list[MEvent | list[MEvent]] = [lastitem]
    for item in items[1:]:
        assert item.offset is not None and lastitem.end is not None
        gap = item.offset - lastitem.end
        if gap < 0:
            raise ValueError(f"Events supperpose: {lastitem=}, {item=}")
        if gap <= mingap and lastitem._canBeLinkedTo(item):
            if isinstance(groups[-1], list):
                groups[-1].append(item)
            else:
                groups[-1] = [groups[-1], item]
        else:
            groups.append(item)
        lastitem = item
    return groups


# def splitLinkedGroupIntoLines(objs: list[MEvent]
#                               ) -> list[list[event.Note]]:
#     """
#     Given a group as a list of Notes/Chords, split it in subgroups matching
#     each note with its continuation.
#
#     When one chords is followed by another chord and the first chord
#     should do a glissando to the second, each note in the first chord is matched with
#     a second note of the second chord (possibly duplicating the notes).
#
#     This is purely intended for playback, so the duplication is not important.
#
#     """
#     if all(isinstance(obj, event.Note) for obj in objs):
#         return [objs]  # type: ignore
#
#     finished: list[list[event.Note]] = []
#     started: list[list[event.Note]] = []
#     continuations: dict[event.Note, event.Note] = {}
#     for obj in objs:
#         if isinstance(obj, event.Chord):
#             for i, note in enumerate(obj.notes):
#                 note.offset = obj.offset
#                 note.dur = obj.dur
#                 note.gliss = obj.gliss if isinstance(obj.gliss, bool) else obj.gliss[i]
#                 note.tied = obj.tied
#                 if obj.playargs:
#                     if not note.playargs:
#                         note.playargs = obj.playargs.copy()
#                     else:
#                         note.playargs.fillWith(obj.playargs)
#
#     # gliss pass
#     if len(objs) > 1:
#         ev0 = objs[0]
#         for ev1 in objs[1:]:
#             if isinstance(ev0, event.Chord) and ev0.gliss is True:
#                 if isinstance(ev1, event.Chord):
#                     # Notes are matched in sort order (which is normally by pitch)
#                     for n0, n1 in zip(ev0.notes, ev1.notes):
#                         continuations[n0] = n1
#                 elif isinstance(ev1, event.Note):
#                     for n0 in ev0.notes:
#                         continuations[n0] = ev1
#             ev0 = ev1
#
#     for objidx, obj in enumerate(objs):
#         if isinstance(obj, event.Chord):
#             notes = obj.notes
#         elif isinstance(obj, event.Note):
#             notes = [obj]
#         else:
#             raise TypeError(f"Expected notes or chords, got {obj}")
#         usednotes = set()
#         assert all(n.offset is not None for n in notes)
#         if not started:
#             # No started group, so all notes here will start group
#             for note in notes:
#                 started.append([note])
#         else:
#             # there are started groups, so iterate through started groups and
#             # find if there are matches.
#             for groupidx, group in enumerate(started):
#                 last = group[-1]
#                 if last.tied:
#                     matchidx = next((i for i, n in enumerate(notes) if n.pitch == last.pitch), None)
#                     if matchidx is not None:
#                         group.append(notes[matchidx])
#                         # notes.pop(matchidx)
#                         usednotes.add(notes[matchidx])
#                 elif last.gliss is True:
#                     if continuation := continuations.get(last):
#                         group.append(continuation)
#                         if continuation in notes:
#                             usednotes.add(continuation)
#                             # notes.remove(continuation)
#                     else:
#                         matchidx = min(range(len(notes)),
#                                        key=lambda idx: abs(notes[idx].pitch - last.pitch))
#                         group.append(notes[matchidx])
#                         usednotes.add(notes[matchidx])
#                 else:
#                     # This group's last note is not tied and has no gliss: this is the
#                     # end of this group, so add it to finished
#                     finished.append(group)
#                     started.pop(groupidx)
#             # Are there notes left? If yes, this notes did not match any started group,
#             # so they must start a group themselves
#             for note in notes:
#                 if note not in usednotes:
#                     started.append([note])
#
#     # We finished iterating, are there any started groups? Finish them
#     finished.extend(started)
#     return finished
