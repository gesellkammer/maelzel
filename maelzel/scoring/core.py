from __future__ import annotations
from .util import *
import itertools
import dataclasses
from typing import Optional as Opt, Union as U, List, Any, Dict

import logging


logger = logging.getLogger("maelzel.scoring")


@dataclasses.dataclass
class Notation:
    """
    This represents a notation (either a rest or a note/chord)

    offset: the offset of this Notation. This offset can represent either an absolute
        offset in seconds or a quarterLength. It is left unspecified in purpose.
        A value of None indicates an unset offset
    duration: the duration of this Notation. A value of None indicates an
        unset duration. During quantization an unset duration is interpreted
        as lasting to the next notation. 0 indicates a grace note
    rest: is this a rest?
    tiedBack: is this Notation tied to the previous one?
    tiedForward: is it tied to the next
    dynamic: the dynamic of this notation, one of "p", "pp", "f", etc.
    group: a str identification, can be used to group Notations together
    durRatios: a list of tuples (x, y) indicating a tuple relationship.
        For example, a Notation used to represent one 8th note in a triplet
        would have the duration 1/3 and durRatios=[(3, 2)]. Multiplying the
        duration by the durRatios would result in the notated value, in this
        case 1/2 (1 being a quarter note). A value of None is the same as
        a value of [(1, 1)] (no modification)
    gliss: if True, a glissando will be rendered between this note and the next
    notehead: the type of notehead, with the following format:
        <notehead>[.<filled/unfilled>]
        Examples: "rectangle", "diamond.unfilled", "regular.filled"
    noteheadHidden: should the notehead be hidden when rendered?
    noteheadParenthsis: parenthesize notehead
    instr: the name of the instrument to play this notation, used for playback
    priority: this is used to sort gracenotes if many gracenotes appear one
        after the other.
    playbackGain: a float between 0-1, or None to leave unset. This is only
        useful for playback, has no effect in the Notation

    """
    duration: Opt[F] = None
    offset: Opt[F] = None
    pitches: Opt[List[float]] = None
    rest: bool = False
    tiedPrev: bool = False
    tiedNext: bool = False
    dynamic: str = ""
    annotations: Opt[List[Annotation]] = None
    articulation: str = ""
    durRatios: Opt[List[F]] = None
    group: str = ""
    gliss: bool = None
    notehead: str = None
    noteheadHidden: bool = False
    noteheadParenthesis: bool = False
    accidentalHidden: bool = False
    accidentalNeeded: bool = True
    instr: str = ""
    priority: int = 0    # this is used to sort gracenotes
    playbackGain: Opt[float] = None   # either None or a float 0-1
    properties: Opt[Dict[str:Any]] = None

    def __post_init__(self):
        if self.rest:
            self.tiedNext = False
            self.tiedPrev = False
        assert self.offset is None or isinstance(self.offset, F)
        assert self.duration is None or isinstance(self.duration, F)
        # assert not self.notehead or self.notehead in availableNoteheads
        assert not self.articulation or self.articulation in availableArticulations

    def isGraceNote(self) -> bool:
        return self.duration == 0

    def meanPitch(self) -> float:
        L = len(self.pitches)
        if self.rest or L == 0:
            raise ValueError("No pitches to calculate mean")
        return self.pitches[0] if L == 1 else sum(self.pitches) / L

    @property
    def end(self) -> Opt[F]:
        if self.duration is not None and self.offset is not None:
            return self.offset + self.duration
        return None

    def clone(self, **kws) -> Notation:
        return dataclasses.replace(self, **kws)

    def symbolicDuration(self) -> F:
        """
        The symbolic duration of this Notation. This represents
        the notated figure (1=quarter, 1/2=eighth note, 1/4=16th note, etc)
        """
        dur = self.duration
        if self.durRatios:
            for durRatio in self.durRatios:
                dur *= durRatio
        return dur

    def addAnnotation(self, text:U[str, Annotation], placement:str='above', fontSize:int=None) -> None:
        if isinstance(text, Annotation):
            annotation = text
        else:
            annotation = Annotation(text=text, placement=placement, fontSize=fontSize)
        if self.annotations is None:
            self.annotations = []
        self.annotations.append(annotation)

    def addArticulation(self, articulation:str):
        assert articulation in availableArticulations
        self.articulation = articulation

    def notatedDuration(self) -> NotatedDuration:
        return notatedDuration(self.duration, self.durRatios)

    def mergeWith(self, other:Notation) -> Notation:
        return mergeNotations(self, other)

    def setProperty(self, key, value) -> None:
        if self.properties is None:
            self.properties = {}
        self.properties[key] = value

    def getProperty(self, key:str, default=None) -> Any:
        if not self.properties:
            return default
        return self.properties.get(key, default)

    def __repr__(self):
        info = []
        if self.offset is None:
            info.append(f"None, dur={showT(self.duration)}")
        else:
            info.append(f"{showT(self.offset)}:{showT(self.end)}")

        if self.durRatios and self.durRatios != [F(1)]:
            info.append(",".join(showF(r) for r in self.durRatios))
        if self.tiedPrev:
            info.append("tiedPrev")
        if self.tiedNext:
            info.append("tiedNext")
        if self.rest:
            info.append("rest")
        elif self.pitches:
            if len(self.pitches) > 1:
                info.append("[" + " ".join(m2n(p) for p in self.pitches) + "]")
            else:
                info.append(m2n(self.pitches[0]))
            if self.gliss:
                info.append("gliss")
        if self.group:
            info.append(f"group={self.group}")

        infostr = " ".join(info)
        return "«" + infostr + "»"


def makeNote(pitch:pitch_t, duration:time_t = None, offset:time_t = None,
             annotation:str=None, gliss=False,
             **kws) -> Notation:
    """
    Utility function to create a note Notation

    Args:
        pitch: the pitch as midinote or notename
        duration: the duration of this Notation. Use None to leave this unset,
            0 creates a grace note
        offset: the offset of this Notation (None to leave unset)
        annotation: an optional text annotation for this note
        gliss: does this Notation start a glissando?
        **kws: any keyword accepted by Notation

    Returns:
        the created Notation
    """
    duration = asFractionOrNone(duration)
    offset = asFractionOrNone(offset)
    midinote = asmidi(pitch)
    out = Notation(pitches=[midinote], duration=duration, offset=offset, gliss=gliss,
                   **kws)
    if annotation:
        out.addArticulation(annotation)
    return out

def makeChord(pitches: List[pitch_t], duration:time_t=None, offset:time_t=None,
              annotation:str=None, **kws) -> Notation:
    """
    Utility function to create a chord Notation

    Args:
        pitches: the pitches as midinotes or notenames
        duration: the duration of this Notation. Use None to leave this unset,
            use 0 to create a grace note
        offset: the offset of this Notation (None to leave unset)
        annotation: a text annotation
        **kws: any keyword accepted by Notation

    Returns:
        the created Notation
    """
    duration = asFractionOrNone(duration)
    offset = asFractionOrNone(offset)
    midinotes = [asmidi(pitch) for pitch in pitches]
    out = Notation(pitches=midinotes, duration=duration, offset=offset, **kws)
    if annotation:
        out.addAnnotation(annotation)
    return out


def makeRest(duration: time_t, offset:time_t=None) -> Notation:
    offset = asFractionOrNone(offset)
    return Notation(duration=asF(duration), offset=offset, rest=True)


def mergeNotations(n0: Notation, n1: Notation) -> Notation:
    if n0.pitches != n1.pitches:
        raise ValueError("Attempting to merge two Notations with "
                         "different pitches")
    return dataclasses.replace(n0,
                               duration=n0.duration+n1.duration,
                               tiedNext=n1.tiedNext)


class Part(list):
    def __init__(self, events: Iter[Notation]=None, label:str=None, groupid:str=None):
        """
        A Track is a list of non-simultaneous events (a Part)

        Args:
            events: the events (notes, chords) in this track
            label: a label to identify this track in particular (a name)
            groupid: an identification (given by makeId), used to identify
                tracks which belong to a same group
        """
        if events:
            #assert all(ev.duration is not None and
            #           ev.duration>=0 and
            #           ev.offset is not None
            #           for ev in events)
            #assert all(ev0.end <= ev1.offset for ev0, ev1 in iterlib.pairwise(events))
            super().__init__(events)
        else:
            super().__init__()
        self.groupid:str = groupid
        self.label:str = label

    def __getitem__(self, item) -> Notation:
        return super().__getitem__(item)

    def __iter__(self) -> Iter[Notation]:
        return super().__iter__()

    def split(self) -> List[Part]:
        return splitNotationsByClef(self, groupid=self.groupid)

    def needsSplit(self) -> bool:
        midinotes = sum((n.pitches for n in self), [])
        return midinotesNeedMultipleClefs(midinotes)

    def stack(self) -> None:
        stackNotationsInPlace(self)


def stackNotationsInPlace(events: List[Notation], start=F(0), overrideOffset=False) -> None:
    """
    This function stacks events together by placing an event at the end of the
    previous event whenever an event does not define its own offset

    Args:
        events: a list of events or a Track
        start: the start time, will override the offset of the first event
        overrideOffset: if True, offsets are overriden even if they are defined

    Returns:
        a list of stacked events
    """
    if all(ev.offset is not None and ev.duration is not None for ev in events):
        return
    now = misc.firstval(events[0].offset, start, F(0))
    assert now is not None and now>=0
    lasti = len(events)-1
    for i, ev in enumerate(events):
        if ev.offset is None or overrideOffset:
            assert ev.duration is not None
            ev.offset = now
        elif ev.duration is None:
            if i == lasti:
                raise ValueError("The last event should have a duration")
            ev.duration = events[i+1].offset - ev.offset
        now += ev.duration
    for ev1, ev2 in iterlib.pairwise(events):
        assert ev1.offset <= ev2.offset


def stackNotations(events: List[Notation], start=F(0), overrideOffset=False
                   ) -> List[Notation]:
    """
    This function stacks events together by placing an event at the end of the
    previous event whenever an event does not define its own offset, or sets
    the duration of an event if events are specified via offset alone

    Args:
        events: a list of notations
        start: the start time, will override the offset of the first event
        overrideOffset: if True, offsets are overriden even if they are defined

    Returns:
        a list of stacked events
    """
    if all(ev.offset is not None and ev.duration is not None for ev in events):
        return events
    assert all(ev.offset is not None or ev.duration is not None for ev in events)
    now = events[0].offset if events[0].offset is not None else start
    assert now is not None and now >= 0
    out = []
    lasti = len(events) - 1
    for i, ev in enumerate(events):
        if ev.offset is None or overrideOffset:
            assert ev.duration is not None
            ev = ev.clone(offset=now, duration=ev.duration)
        elif ev.duration is None:
            if i == lasti:
                raise ValueError("The last event should have a duration")
            ev = ev.clone(duration=events[i+1].offset - ev.offset)
        now += ev.duration
        out.append(ev)
    for ev1, ev2 in iterlib.pairwise(out):
        assert ev1.offset <= ev2.offset
    return out


def fixOverlap(events: List[Notation]) -> List[Notation]:
    """
    Fix overlap between events. If two events overlap,
    the first event is cut, preserving the offset of the
    second event

    Args:
        events: the events to fix

    Returns:
        the fixed events
    """
    if len(events) < 2:
        return events
    out = []
    for n0, n1 in iterlib.pairwise(events):
        assert n0.duration is not None and n0.offset is not None
        assert n1.offset is not None
        assert n0.offset <= n1.offset, "Notes are not sorted!"
        if n0.end > n1.offset:
            n0 = dataclasses.replace(n0, duration=n1.offset - n0.offset)
        out.append(n0)
    out.append(events[-1])
    return out


def fillSilences(events: List[Notation], mingap=1/64) -> List[Notation]:
    """
    Return a list of Notations filled with rests

    Args:
        events: the notes to fill
        mingap: min. gap between two notes. If any notes differ by less
                   than this, the first note absorvs the gap
    Returns:
        a list of new Notations
    """
    assert events
    assert all(isinstance(ev, Notation) for ev in events)
    out: List[Notation] = []
    if events[0].offset > 0:
        out.append(Notation(offset=F(0), duration=events[0].offset, rest=True))
    for ev0, ev1 in iterlib.pairwise(events):
        gap = ev1.offset - (ev0.offset + ev0.duration)
        assert gap >= 0, f"negative gap! = {gap}"
        if gap > mingap:
            out.append(ev0)
            rest = Notation(offset=ev0.offset+ev0.duration, duration=gap, rest=True)
            out.append(rest)
        else:
            # adjust the dur of n0 to match start of n1
            out.append(ev0.clone(duration=ev1.offset - ev0.offset))
    out.append(events[-1])
    assert not any(n0.end != n1.offset for n0, n1 in iterlib.pairwise(out)), out
    return out


def _groupById(notations: List[Notation]) -> List[U[Notation, List[Notation]]]:
    """
    Given a seq. of events, elements which are grouped together are wrapped
    in a list, whereas elements which don't belong to any group are
    appended as is

    """
    out = []
    for groupid, elementsiter in itertools.groupby(notations, key=lambda n:n.group):
        if not groupid:
            out.extend(elementsiter)
        else:
            elements = list(elementsiter)
            elements.sort(key=lambda elem:elem.offset)
            out.append(elements)
    return out


def splitNotationsByClef(events: List[Notation], groupid=None) -> List[Part]:
    """
    Assuming that events are not simultanous, split the events into
    different Parts if the range makes it necessary, where each
    Part can be represented without clef changes

    Args:
        events: the events to split
        groupid: if given, this id will be used to identify the
            generated tracks (see makeId)

    Returns:
         list of Parts (between 1 and 3, one for each clef)
    """
    G = []
    F = []
    G15a = []

    for event in events:
        assert event.offset is not None
        if event.rest:
            continue
        elif len(event.pitches) == 1:
            pitch = event.pitches[0]
            if 55 < pitch <= 93:
                G.append(event)
            elif 93 < pitch:
                G15a.append(event)
            else:
                F.append(event)
        else:
            # a chord
            chordG = []
            chordF = []
            chord15a = []
            for pitch in event.pitches:
                if 55 < pitch <= 93:
                    chordG.append(pitch)
                elif 93 < pitch:
                    chord15a.append(pitch)
                else:
                    chordF.append(pitch)
            if chordG:
                G.append(event.clone(pitches=chordG))
            if chordF:
                F.append(event.clone(pitches=chordF))
            if chord15a:
                G15a.append(event.clone(pitches=chord15a))
    groupid = groupid or makeId()
    parts = [Part(part, groupid=groupid, label=name)
              for part, name in ((G15a, "G15a"), (G, "G"), (F, "F")) if part]
    return parts


def packInParts(events: List[Notation], maxrange=36) -> List[Part]:
    """
    Pack a list of possibly simultaneous events into tracks, where the events
    within one track are NOT simulatenous. Events belonging to the same group
    are kept in the same track.

    Returns a list of Parts
    """
    from maelzel.music import packing
    items = []
    groups = _groupById(events)
    for group in groups:
        if isinstance(group, Notation):
            event = group
            if event.rest:
                continue
            item = packing.Item(obj=event, offset=event.offset, dur=event.duration,
                                step=event.meanPitch())
        elif isinstance(group, list):
            dur = group[-1].end - group[0].offset
            step = sum(event.avgPitch() for event in group)/len(group)
            item = packing.Item(obj=group, offset=group[0].offset, dur=dur, step=step)
        else:
            raise TypeError(f"Expected an Event or a list thereof, got {type(group)}")
        items.append(item)

    packedTracks = packing.pack_in_tracks(items, maxrange=maxrange)

    def unwrapPackingTrack(track: packing.Track) -> List[Notation]:
        out = []
        for item in track:
            obj = item.obj
            if isinstance(obj, list):
                out.extend(obj)
            else:
                out.append(obj)
        return out

    return [Part(unwrapPackingTrack(track)) for track in packedTracks]



