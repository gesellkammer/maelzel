from __future__ import annotations
from maelzel import packing
from maelzel.rational import Rat
from emlib.iterlib import pairwise
from . import musicobj as mobj
from .workspace import getConfig, Workspace
from maelzel.core.synthevent import SynthEvent, PlayArgs
from numbers import Rational

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from .musicobj import Note, Chord, Chain, MusicObj, MusicEvent
    from .config import CoreConfig




def splitNotesOnce(notes: Union[Chord, Sequence[Note]], splitpoint: float, deviation=None,
                    ) -> Tuple[list[Note], list[Note]]:
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


def splitNotesIfNecessary(notes: list[Note], splitpoint: float, deviation=None
                          ) -> list[list[Note]]:
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
        lastEnd = 0
        for item in items:
            if resetMinGap > 0 and item.start - lastEnd > resetMinGap:
                lastDynamic = initialDynamic
            if not item.dynamic:
                item.dynamic = lastDynamic
                item.properties['tempdynamic'] = True
            else:
                lastDynamic = item.dynamic
            lastEnd = item.end


def addDurationToGracenotes(chain: list[MusicEvent], dur: Rat) -> None:
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
                # First in the sequence is a gracenote. Diminish the dur of the next real note,
                # make the gracenote "on the beat"
                assert i == 0 and len(chain) > 1 and not chain[1].isGracenote()
                nextreal = chain[1]
                assert nextreal.dur > dur
                nextreal.dur -= dur
                nextreal.start += dur
                n.dur = dur
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


def groupLinkedEvents(items: list[MusicEvent]) -> list[MusicEvent | list[MusicEvent]]:
    """
    Group linked events together

    Two events are linked if they are adjacent and the first event is either tied
    or has a glissando to the second event

    Args:
        items: a list of Note|Chord

    Returns:
        a list of individual notes, chords or groups, where a group is itself a
        list of notes/chords
    """
    lastitem = items[0]
    groups = [[lastitem]]
    for item in items[1:]:
        if lastitem.canBeLinkedTo(item):
            groups[-1].append(item)
        else:
            groups.append([item])
        lastitem = item
    return [group[0] if len(group) == 1 else group for group in groups]


def splitLinkedGroupIntoLines(objs: list[MusicEvent]
                              ) -> list[list[Note]]:
    """
    Given a group as a list of Notes/Chords, split it in subgroups matching
    each note with its continuation.

    For example, when one chords is followed by another chord and the first chord
    should do a glissando to the second, each note in the first chord is matched with
    a second note of the second chord (possibly duplicating the notes).

    This is purely intended for playback, so the duplication is not important.

    """
    if all(isinstance(obj, mobj.Note) for obj in objs):
        return [objs]

    finished: list[list[Note]] = []
    started: list[list[Note]] = []
    continuations: dict[Note, Note] = {}
    for obj in objs:
        if isinstance(obj, mobj.Chord):
            for note in obj.notes:
                note.start = obj.start
                note.dur = obj.dur
                note.gliss = obj.gliss
                note.tied = obj.tied
                if obj._playargs:
                    note.playargs.fillWith(obj._playargs)

    # gliss pass
    for ev0, ev1 in pairwise(objs):
        if isinstance(ev0, mobj.Chord) and ev0.gliss is True:
            if isinstance(ev1, mobj.Chord):
                for n0, n1 in zip(ev0.notes, ev1.notes):
                    continuations[n0] = n1
            elif isinstance(ev1, mobj.Note):
                for n0 in ev0.notes:
                    continuations[n0] = ev1

    for objidx, obj in enumerate(objs):
        notes = obj.notes if isinstance(obj, mobj.Chord) else [obj]
        usednotes = set()
        assert all(n.start is not None for n in notes)
        if not started:
            # No started groups, so all notes here will start groups
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
                    if continuation:=continuations.get(last):
                        group.append(continuation)
                        if continuation in notes:
                            usednotes.add(continuation)
                            # notes.remove(continuation)
                    else:
                        matchidx = min(range(len(notes)),
                                       key=lambda idx: abs(notes[idx].pitch - last.pitch))
                        group.append(notes[matchidx])
                        usednotes.add(notes[matchidx])
                        # notes.pop(matchidx)
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


def chainSynthEvents(objs: list[MusicEvent], playargs: PlayArgs, workspace: Workspace
                      ) -> list[SynthEvent]:
    """
    Calculate synthevents for a chain of events

    Args:
        objs: a sequence of events
        playargs: playargs for the sequence
        workspace: the activeworkspace

    Returns:
        the corresponding list of synthevents
    """
    synthevents = []
    groups = groupLinkedEvents(objs)
    struct = workspace.scorestruct
    conf = workspace.config
    for group in groups:
        if isinstance(group, mobj.MusicEvent):
            events = group._synthEvents(playargs.copy(), workspace=workspace)
            synthevents.extend(events)
        elif isinstance(group, list):
            lines = splitLinkedGroupIntoLines(group)
            # A line of notes
            for line in lines:
                bps = [[float(struct.toTime(item.start)), item.pitch, item.resolvedAmp(workspace=workspace)]
                       for item in line]
                lastev = line[-1]
                pitch = lastev.gliss or lastev.pitch
                assert lastev.end is not None
                bps.append([float(struct.toTime(lastev.end)), pitch, lastev.resolvedAmp(workspace=workspace)])
                for bp in bps:
                    assert all(isinstance(x, (int, float)) for x in bp), f"bp: {bp}\n{bps=}"
                firstev = line[0]
                # TODO: optimize / revise the playargs handling
                evplayargs = playargs.copy()
                if firstev._playargs:
                    evplayargs.overwriteWith(firstev._playargs)
                evplayargs.fillWithConfig(conf)
                synthevents.append(SynthEvent.fromPlayArgs(bps=bps, playargs=evplayargs))
        else:
            raise TypeError(f"Did not expect {group}")
    return synthevents


def normalizeChordArpeggio(arpeggio: Union[str, bool], chord: Chord, config: CoreConfig
                           ) -> bool:
    if arpeggio is None:
        arpeggio = config['show.arpeggiateChord']
    if isinstance(arpeggio, bool):
        return arpeggio
    elif arpeggio == 'auto':
        return chord._isTooCrowded()
    else:
        raise ValueError(f"arpeggio should be True, False, 'auto' (got {arpeggio})")


def flattenObjs(objs: list[MusicEvent | Chain], offset=Rat(0)) -> list[MusicEvent]:
    collected = []
    for obj in objs:
        assert obj.start is not None, \
            f"This function should be called with objects with resolved start, got {obj}"
        if isinstance(obj, mobj.MusicEvent):
            assert obj.dur is not None
            collected.append(obj.clone(start=obj.start+offset))
        elif isinstance(obj, mobj.Chain) and obj.items:
            collected.extend(flattenObjs(obj.items, offset=offset+obj.start))
        else:
            raise TypeError(f"Expected a Note/Chord or a Chain, got {obj} ({type(obj)})")
    return collected


def resolvedTimes(events: list[MusicEvent | Chain],
                  defaultDur = Rat(1),
                  offset = Rat(0),
                  ) -> list:
    """
    Stack events to the left, making any unset start and duration explicit

    After setting all start times and durations an offset is added, if given

    Args:
        events: the events to modify, either in place or as a copy
        defaultDur: the default duration used when an event has no duration and
            the next event does not have an explicit start
        inplace: if True, events are modified in place
        offset: an offset to add to all start times after stacking them
        recurse: if True, stack also events inside subchains

    Returns:
        the modified events. If inplace is True, the returned events are the
        same as the events passed as input

    """

    if not events:
        raise ValueError("no events given")

    now = Rat(0)
    lasti = len(events) - 1
    out = []
    for i, ev in enumerate(events):
        if ev.start is None:
            start = now
        else:
            start = ev.start
        dur = defaultDur
        if isinstance(ev, MusicEvent):
            if ev.dur is None:
                if i < lasti:
                    nextev = events[i+1]
                    if nextev.start is not None:
                        dur = nextev.start - start
                        # ev.dur = nextev.start - ev.start
            now = start + dur
            out.append((ev, start, dur))
        elif isinstance(ev, Chain):
            subitems = resolvedTimes(ev.items)
            out.append(subitems)
            dur = _resolvedTimesDur(ev.items)
            now = start + dur

    return out


def _resolvedTimesDur(items: list[tuple[MusicEvent, Rational, Rational] | list]) -> Rational:
    return sum(item[2] if isinstance(item, tuple) else _resolvedTimesDur(item)
               for item in items)
