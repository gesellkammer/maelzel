from __future__ import annotations

import itertools

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
    # all gracenotes are copied, and all real notes before a gracenote
    out = []

    def copy(ev: MEvent) -> MEvent:
        out = ev.copy()
        out._relOffset = ev._relOffset
        return out

    for ev0, ev1 in itertools.pairwise(events):
        if ev0.isGrace() or ev1.isGrace():
            out.append(copy(ev0))
        else:
            out.append(ev0)

    last = events[-1]
    out.append(last if not last.isGrace() else copy(last))
    # assert all(ev0.relEnd() <= ev1.relOffset() for ev0, ev1 in itertools.pairwise(events))
    return out


def addDurationToGracenotes(events: list[MEvent], dur: F, inplace=False
                            ) -> list[MEvent]:
    """
    Adds real duration to gracenotes within chain (in place)

    Previous to playback, gracenotes have a duration of 0. Before playing
    they are assigned a duration, which is substracted from the previous "real"
    note or silence.

    Args:
        events: the sequence of notes to modify (inplace)
        dur: the duration of a single gracenote

    """
    if not inplace:
        events = copyEventsModifiedByGracenotes(events)

    lastRealNote = -1
    d: dict[int, list[int]] = {}
    # first we build a registry mapping real notes to their grace notes. Gracenotes
    # come BEFORE the
    now = events[0].relOffset()
    for i, n in enumerate(events):
        if not n.isGrace():
            lastRealNote = i
            if n.offset is None:
                # n._relOffset = now
                now += n.dur
            else:
                now = n.offset
        elif lastRealNote >= 0:
            d.setdefault(lastRealNote, []).append(i)
        else:
            # First in the sequence is a gracenote. Diminish the dur of the next real note,
            # make the gracenote "on the beat"
            nextrealidx = next((j for j, n in enumerate(events[i+1:])
                                if not n.isGrace() and n.dur > 0), None)
            if nextrealidx is None:
                raise ValueError(f"No real notes in {events=}")
            nextreal = events[nextrealidx+i+1]
            dur = min(dur, nextreal.dur / (nextrealidx + 1))
            assert dur > 0 and nextreal.dur > dur, f"{nextreal=}, {dur=}, {i=}, {nextrealidx=}"
            n.dur = dur
            n._relOffset = now
            nextoffset = now + dur
            if nextreal.offset is None:
                nextreal._relOffset = nextoffset
                nextreal.dur -= dur
            elif nextreal.offset < nextoffset:
                nextreal.dur -= nextoffset - nextreal.offset
                nextreal.offset = nextoffset
            now += dur

    for realidx, graceidxs in d.items():
        realev = events[realidx]
        maxGraceDur = realev.dur / (len(graceidxs) + 1)
        graceDur = min(dur, maxGraceDur)
        realend = realev.relEnd()
        realev.dur -= graceDur * len(graceidxs)
        for i, graceidx in enumerate(graceidxs):
            grace = events[graceidx]
            grace.dur = graceDur
            deltapos = (len(graceidxs) - i) * graceDur
            grace.offset = realend - deltapos

    for e0, e1 in itertools.pairwise(events):
        if e0.relEnd() > e1.relOffset():
            raise ValueError(f"Items supperpose: {e0}, end={e0.relOffset()}, {e1}, start={e1.relOffset()}")
    return events


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
    lastitem = items[0]
    groups: list[MEvent | list[MEvent]] = [lastitem]
    for item in items[1:]:
        # assert item.offset is not None and lastitem.end is not None
        gap = item.relOffset() - lastitem.relEnd()
        # gap = item.offset - lastitem.end
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