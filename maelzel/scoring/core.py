from __future__ import annotations

from emlib import iterlib
import pitchtools as pt
from .util import *
from . import definitions
import itertools
import copy
from typing import Optional as Opt, Union as U, List, Any, Dict, \
    Iterator as Iter
import uuid
import logging



logger = logging.getLogger("maelzel.scoring")


class Notation:
    """
    This represents a notation (a rest, a note or a chord)

    Args:
        offset: the offset of this Notation, in quarter-notes.
        duration: the duration of this Notation, in quarter-notes. A value of
            None indicates an unset duration. During quantization an unset
            duration is interpreted as lasting to the next notation.
            0 indicates a grace note
        isRest: is this a rest?
        tiedPrev: is this Notation tied to the previous one?
        tiedNext: is it tied to the next
        dynamic: the dynamic of this notation, one of "p", "pp", "f", etc.
        group: a str identification, can be used to group Notations together
        durRatios: a list of tuples (x, y) indicating a tuple relationship.
            For example, a Notation used to represent one 8th note in a triplet
            would have the duration 1/3 and durRatios=[(3, 2)]. Multiplying the
            duration by the durRatios would result in the notated value, in this
            case 1/2 (1 being a quarter note). A value of None is the same as
            a value of [(1, 1)] (no modification)
        gliss: if True, a glissando will be rendered between this note and the next
        notehead: the type of notehead, with the format <notehead> or
            <notehead>.<filled/unfilled>. Examples: "cross", "square.unfilled"
        noteheadHidden: should the notehead be hidden when rendered?
        noteheadParenthesis: parenthesize notehead
        instr: the name of the instrument to play this notation, used for playback
        priority: this is used to sort gracenotes if many gracenotes appear one
            after the other.
        playbackGain: a float between 0-1, or None to leave unset. This is only
            useful for playback, has no effect in the Notation
        fixedNotenames: an optional dict mapping pitch index to notename

    """
    __slots__ = ("duration",
                 "pitches",
                 "offset",
                 "isRest",
                 "tiedPrev",
                 "tiedNext",
                 "dynamic",
                 "annotations",
                 "articulation",
                 "durRatios",
                 "group",
                 "gliss",
                 "notehead",
                 "noteheadHidden",
                 "noteheadParenthesis",
                 "accidentalHidden",
                 "accidentalNeeded",
                 "color",
                 "stem",
                 "instr",
                 "priority",
                 "playbackGain",
                 "properties",
                 "fixedNotenames"
                 )

    def __init__(self,
                 duration:time_t = None,
                 pitches: List[float] = None,
                 offset: time_t = None,
                 isRest=False,
                 tiedPrev=False,
                 tiedNext=False,
                 dynamic:str='',
                 annotations:List[Annotation]=None,
                 articulation:str='',
                 durRatios: List[F]=None,
                 group='',
                 gliss:bool=None,
                 notehead:str='',
                 noteheadHidden=False,
                 noteheadParenthesis=False,
                 accidentalHidden=False,
                 accidentalNeeded=True,
                 color='',
                 stem='',
                 instr='',
                 priority=0,
                 playbackGain:float=None,
                 properties:Dict[str, Any]=None,
                 ):

        assert not notehead or notehead in definitions.noteheadShapes, \
            f"Possible noteheads: {definitions.noteheadShapes}"
        assert not articulation or articulation in definitions.availableArticulations, \
            f"Available articulations: {definitions.availableArticulations}"
        assert not stem or stem in definitions.stemTypes, \
            f"Stem types: {definitions.stemTypes}"
        assert not dynamic or dynamic in definitions.availableDynamics, \
            f"Available dynamics: {definitions.availableDynamics}"

        self.duration:Opt[F] = None if duration is None else asF(duration)
        self.pitches:List[float] = pitches
        self.offset:Opt[F] = None if offset is None else asF(offset)
        self.isRest = isRest
        if isRest:
            self.tiedNext = False
            self.tiedPrev = False
        else:
            self.tiedPrev = tiedPrev
            self.tiedNext = tiedNext
        self.dynamic = dynamic
        self.articulation = articulation
        self.annotations = annotations
        self.durRatios = durRatios
        self.group = group
        self.gliss = gliss
        self.notehead = notehead
        self.noteheadHidden = noteheadHidden
        self.noteheadParenthesis = noteheadParenthesis
        self.accidentalHidden = accidentalHidden
        self.accidentalNeeded = accidentalNeeded
        self.color = color
        self.stem = stem
        self.instr = instr
        self.priority = priority
        self.playbackGain = playbackGain
        self.properties: Opt[Dict[str,Any]] = properties
        self.fixedNotenames: Opt[Dict[int, str]] = None

    def setFixedNotenameByIndex(self, idx:int, notename:str):
        if self.fixedNotenames is None:
            self.fixedNotenames = {}
        self.fixedNotenames[idx] = notename

    def getFixedNotenameByIndex(self, idx:int, default=None) -> Opt[str]:
        if not self.fixedNotenames:
            return default
        return self.fixedNotenames.get(idx, default)

    def isGraceNote(self) -> bool:
        return self.duration == 0

    def meanPitch(self) -> float:
        L = len(self.pitches)
        if self.isRest or L == 0:
            raise ValueError("No pitches to calculate mean")
        return self.pitches[0] if L == 1 else sum(self.pitches) / L

    @property
    def end(self) -> Opt[F]:
        if self.duration is not None and self.offset is not None:
            return self.offset + self.duration
        return None

    def clone(self, **kws) -> Notation:
        """
        Clone this Notation, overriding any value.

        Args:
            kws: keyword arguments, as passed to the Notation constructor.
                Any parameter given will override the corresponding value in
                this Notation
        """
        out = self.copy()
        for key, value in kws.items():
            setattr(out, key, value)
        return out

    def copy(self) -> Notation:
        """
        Copy this Notation as is
        """
        return copy.deepcopy(self)

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

    def addAnnotation(self, text:U[str, Annotation], placement:str='above',
                      fontSize:int=None) -> None:
        """
        Add a text annotation to this Notation.

        Args:
            text: the text of the annotation, or an Annotation object itself
                If passed an Annotation, all other parameters will not be
                considered
            placement: where to place the annotation, one of 'above' or 'below'
            fontSize: the size of the font
        """
        if isinstance(text, Annotation):
            assert text.text.strip()
            annotation = text
        else:
            assert not text.isspace()
            annotation = Annotation(text=text, placement=placement, fontSize=fontSize)
        if self.annotations is None:
            self.annotations = []
        self.annotations.append(annotation)

    def addArticulation(self, articulation:str):
        """
        Add an articulation to this Notation. See definitions.availableArticulations
        for possible values.
        """
        assert articulation in definitions.availableArticulations
        self.articulation = articulation

    def notatedDuration(self) -> NotatedDuration:
        """The duration of the notated figure, in quarter-notes,
        independent of any tuples. A quarter-note inside a triplet
        would have a notatedDuration of 1"""
        return notatedDuration(self.duration, self.durRatios)

    def mergeWith(self, other:Notation) -> Notation:
        """Merge this Notation with ``other``"""
        return mergeNotations(self, other)

    def setProperty(self, key:str, value) -> None:
        """
        Set any property of this Notation. Properties can be
        used, for example, for any rendering backend to pass directives which
        are specific to that rendering backend.
        """
        if self.properties is None:
            self.properties = {}
        self.properties[key] = value

    def getProperty(self, key:str, default=None) -> Any:
        """
        Get the value of a property. If the key is not found, return ``default``
        """
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
        if self.isRest:
            info.append("rest")
        elif self.pitches:
            if len(self.pitches) > 1:
                info.append("[" + " ".join(m2n(p) for p in self.pitches) + "]")
            else:
                info.append(m2n(self.pitches[0]))
            if self.gliss:
                info.append("gliss")
        if self.group:
            if len(self.group) < 6:
                info.append(f"group={self.group}")
            else:
                info.append(f"group={self.group[:8]}…")

        infostr = " ".join(info)
        return "«" + infostr + "»"


def makeGroupId() -> str:
    """
    Create an id to group notations together

    Returns:
        the group id as string
    """
    return str(uuid.uuid1())


def makeNote(pitch:pitch_t, duration:time_t = None, offset:time_t = None,
             annotation:str=None, gliss=False, withId=False,
             gracenote=False,
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
        withId: if True, this Notation has a group id and this id
            can be used to mark multiple notes as belonging to a same group
        gracenote: make this a grace note.
        **kws: any keyword accepted by Notation

    Returns:
        the created Notation
    """
    if gracenote:
        duration = 0
    else:
        duration = asFractionOrNone(duration)
    offset = asFractionOrNone(offset)
    midinote = asmidi(pitch)
    out = Notation(pitches=[midinote], duration=duration, offset=offset, gliss=gliss,
                   **kws)
    if annotation:
        out.addArticulation(annotation)
    if withId:
        out.group = str(id(out))
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
        if isinstance(annotation, str) and annotation.isspace():
            logger.warning("Trying to add an empty annotation")
        else:
            out.addAnnotation(annotation)
    return out


def makeRest(duration: time_t, offset:time_t=None) -> Notation:
    """
    Shortcut function to create a rest notation. A rest is only
    needed when stacking notations within a container like
    EventSeq or Track, to signal a spacing between notations.
    Just explicitely setting the offset of a notation has the
    same effect

    Args:
        duration: the duration of the rest
        offset: the start time of the rest. Normally a rest's offset
            is left unspecified (None)
    """
    return Notation(duration=asF(duration), offset=offset, isRest=True)


def mergeNotations(a: Notation, b: Notation) -> Notation:
    """
    Merge two compatible notations to one. For two notations to be
    mergeable they need to:

    - be adjacent or have unset offset
    - have a duration
    - have the same pitch/pitches.

    All other attributes are taken from the first notation and the
    duration of this first notation is extended to cover both notations
    """
    if a.pitches != b.pitches:
        raise ValueError("Attempting to merge two Notations with "
                         "different pitches")
    assert a.duration is not None and b.duration is not None
    assert b.offset is None or (a.end == b.offset)
    out = a.clone(duration=a.duration + b.duration,
                  tiedNext=b.tiedNext)
    return out


def durationsCanMerge(n0: Notation, n1: Notation) -> bool:
    """
    Returns True if these two Notations can be merged solely based
    on their duration and their start / end position
    """
    dur0 = n0.symbolicDuration()
    dur1 = n1.symbolicDuration()
    sumdur = dur0 + dur1
    num, den = sumdur.numerator, sumdur.denominator
    if den > 64 or num not in {1, 2, 3, 4, 7}:
        return False

    # Allow: r8 8 + 4 = r8 4.
    # Don't allow: r16 8. + 8. r16 = r16 4. r16
    grid = F(1, den)
    if (num == 3 or num == 7) and ((n0.offset % grid) > 0 or (n1.end % grid) > 0):
        return False
    return True

def notationsCanMerge(n0: Notation, n1: Notation) -> bool:
    """
    Returns True if n0 and n1 can me merged

    Two Notations can merge if the resulting duration is regular. A regular
    duration is one which can be represented via **one** notation (a quarter,
    a half, a dotted 8th, a double dotted 16th are all regular durations,
    5/8 of a quarter is not)

    """
    if n0.isRest and n1.isRest:
        return (n0.durRatios == n1.durRatios and
                durationsCanMerge(n0, n1))
    if (not n0.tiedNext or
            not n1.tiedPrev or
            n0.durRatios != n1.durRatios or
            n0.pitches != n1.pitches
            ):
        return False
    # durRatios are the same so check if durations would sum to a regular duration
    return durationsCanMerge(n0, n1)


def mergeNotationsIfPossible(notations: List[Notation]) -> List[Notation]:
    """
    If two consecutive notations have same .durRatio and merging them
    would result in a regular note, merge them.

    8 + 8 = q
    q + 8 = q·
    q + q = h
    16 + 16 = 8

    In general:

    1/x + 1/x     2/x
    2/x + 1/x     3/x  (and viceversa)
    3/x + 1/x     4/x  (and viceversa)
    6/x + 1/x     7/x  (and viceversa)
    """
    assert len(notations) > 1
    out = [notations[0]]
    for n1 in notations[1:]:
        if notationsCanMerge(out[-1], n1):
            out[-1] = out[-1].mergeWith(n1)
        else:
            out.append(n1)
    assert len(out) <= len(notations)
    assert sum(n.duration for n in out) == sum(n.duration for n in notations)
    return out


class Part(list):
    """
    A Part is a list of non-simultaneous events

    Args:
        events: the events (notes, chords) in this track
        label: a label to identify this track in particular (a name)
        groupid: an identification (given by makeGroupId), used to identify
            tracks which belong to a same group
    """
    def __init__(self, events: Iter[Notation]=None, label:str=None, groupid:str=None):

        if events:
            super().__init__(events)
        else:
            super().__init__()
        self.groupid:str = groupid
        self.label:str = label
        # fixEnharmonicsInPlace(self)

    def __getitem__(self, item) -> Notation:
        return super().__getitem__(item)

    def __iter__(self) -> Iter[Notation]:
        return super().__iter__()

    def __repr__(self) -> str:
        s0 = super().__repr__()
        return "Part"+s0

    def distributeByClef(self) -> List[Part]:
        """
        Distribute the notations in this Part into multiple parts,
        depending on their pitch
        """
        return distributeNotationsByClef(self, groupid=self.groupid)

    def needsMultipleClefs(self) -> bool:
        """
        Returns True if the notations in this Part extend over the range
        of one particular clef
        """
        midinotes = sum((n.pitches for n in self), [])
        return midinotesNeedMultipleClefs(midinotes)

    def stack(self) -> None:
        """
        Stack the notations of this part **in place**. Stacking means filling in any
        unresolved offset/duration of the notations in this part. After this operation,
        all Notations in this Part have an explicit duration and start. See
        :meth:`stacked` for a version which returns a new Part instead of operating in
        place
        """
        stackNotationsInPlace(self)

    def stacked(self) -> Part:
        """
        Similar to :meth:`stack`, stacks the Notations in this Part to make them
        adjacent whenever they have unset offset/duration. **This method returns a
        new Part** instead of operating in place.
        """
        notations = stackNotations(self)
        return Part(notations, label=self.label, groupid=self.groupid)


def fixEnharmonicsInPlace(events: List[Notation]):
    # TODO
    pass

def stackNotationsInPlace(events: List[Notation], start=F(0), overrideOffset=False
                          ) -> None:
    """
    This function stacks events together by placing an event at the end of the
    previous event whenever an event does not define its own offset

    Args:
        events: a list of Notations (or a Part)
        start: the start time, will override the offset of the first event
        overrideOffset: if True, offsets are overriden even if they are defined
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


def fixOverlap(notations: List[Notation]) -> List[Notation]:
    """
    Fix overlap between notations. If two notations overlap,
    the first notation is cut, preserving the offset of the
    second notation

    Args:
        notations: the notations to fix

    Returns:
        the fixed notations
    """
    if len(notations) < 2:
        return notations
    out = []
    for n0, n1 in iterlib.pairwise(notations):
        assert n0.duration is not None and n0.offset is not None
        assert n1.offset is not None
        assert n0.offset <= n1.offset, "Notes are not sorted!"
        if n0.end > n1.offset:
            n0 = n0.clone(duration=n1.offset - n0.offset)
        out.append(n0)
    out.append(notations[-1])
    return out


def fillSilences(notations: List[Notation], mingap=1/64) -> List[Notation]:
    """
    Return a list of Notations filled with rests

    Args:
        notations: the notes to fill
        mingap: min. gap between two notes. If any notes differ by less
                   than this, the first note absorvs the gap
    Returns:
        a list of new Notations
    """
    assert notations
    assert all(isinstance(ev, Notation) for ev in notations)
    out: List[Notation] = []
    if notations[0].offset > 0:
        out.append(makeRest(duration=notations[0].offset))
    for ev0, ev1 in iterlib.pairwise(notations):
        gap = ev1.offset - (ev0.offset + ev0.duration)
        assert gap >= 0, f"negative gap! = {gap}"
        if gap > mingap:
            out.append(ev0)
            rest = makeRest(offset=ev0.offset+ev0.duration, duration=gap)
            out.append(rest)
        else:
            # adjust the dur of n0 to match start of n1
            out.append(ev0.clone(duration=ev1.offset - ev0.offset))
    out.append(notations[-1])
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


def distributeNotationsByClef(notations: List[Notation], groupid=None) -> List[Part]:
    """
    Assuming that events are not simultanous, split the events into
    different Parts if the range makes it necessary, where each
    Part can be represented without clef changes. We don't enforce that the
    notations are not simultaneous within a part

    Args:
        notations: the events to split
        groupid: if given, this id will be used to identify the
            generated tracks (see makeGroupId)

    Returns:
         list of Parts (between 1 and 3, one for each clef)
    """
    G = []
    F = []
    G15a = []

    for notation in notations:
        assert notation.offset is not None
        if notation.isRest:
            continue
        elif len(notation.pitches) == 1:
            pitch = notation.pitches[0]
            if 55 < pitch <= 93:
                G.append(notation)
            elif 93 < pitch:
                G15a.append(notation)
            else:
                F.append(notation)
        else:
            # a chord
            chordG = []
            chordF = []
            chord15a = []
            for pitch in notation.pitches:
                if 55 < pitch <= 93:
                    chordG.append(pitch)
                elif 93 < pitch:
                    chord15a.append(pitch)
                else:
                    chordF.append(pitch)
            if chordG:
                G.append(notation.clone(pitches=chordG))
            if chordF:
                F.append(notation.clone(pitches=chordF))
            if chord15a:
                G15a.append(notation.clone(pitches=chord15a))
    # groupid = groupid or makeGroupId()
    # parts = [Part(part, groupid=groupid, label=name)
    #           for part, name in ((G15a, "G15a"), (G, "G"), (F, "F")) if part]
    parts = [Part(part) for part in (G15a, G, F) if part]
    return parts


def packInParts(notations: List[Notation], maxrange=36,
                keepGroupsTogether=True) -> List[Part]:
    """
    Pack a list of possibly simultaneous notations into tracks

    The notations within one track are NOT simulatenous. Notations belonging
    to the same group are kept in the same track.

    Args:
        notations: the Notations to pack
        maxrange: the max. distance between the highest and lowest Notation
        keepGroupsTogether: if True, items belonging to a same group are
            kept in a same track

    Returns:
        a list of Parts

    """
    from maelzel.music import packing
    items = []
    groups = _groupById(notations)
    for group in groups:
        if isinstance(group, Notation):
            n = group
            if not n.isRest:
                items.append(packing.Item(obj=n, offset=n.offset,
                                          dur=n.duration, step=n.meanPitch()))
        else:
            assert isinstance(group, list)
            if keepGroupsTogether:
                dur = max(n.end for n in group) - min(n.offset for n in group)
                step = sum(n.meanPitch() for n in group)/len(group)
                item = packing.Item(obj=group, offset=group[0].offset, dur=dur, step=step)
                items.append(item)
            else:
                items.extend(packing.Item(obj=n, offset=n.offset, dur=n.duration,
                                          step=n.meanPitch())
                             for n in group)

    packedTracks = packing.packInTracks(items, maxAmbitus=maxrange)
    return [Part(track.unwrap()) for track in packedTracks]



