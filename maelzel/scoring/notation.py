from __future__ import annotations
import copy
import uuid

from .util import *
from . import definitions
import pitchtools as pt


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *

__all__ = (
    'Notation',
    'makeNote',
    'makeChord',
    'makeRest',
    'makeGroupId',
    'mergeNotations'
)


_UNSET = object()


class Notation:
    """
    This represents a notation (a rest, a note or a chord)

    Args:
        duration: the duration of this Notation, in quarter-notes. A value of
            None indicates an unset duration. During quantization an unset
            duration is interpreted as lasting to the next notation.
            0 indicates a grace note
        pitches: if given, a list of pitches as midinote or notename. If a notename
            is given, the spelling is fixed. Otherwise a suitable spelling is calculated
            based on the context of this notation.
        offset: the offset of this Notation, in quarter-notes.
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
        color: the color of this notations
        stem: if given, one of
        priority: this is used to sort gracenotes if many gracenotes appear one
            after the other.
        playbackGain: a float between 0-1, or None to leave unset. This is only
            useful for playback, has no effect in the Notation

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
                 "groupid",
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
                 pitches: List[pitch_t] = None,
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

        self.duration:Optional[F] = None if duration is None else asF(duration)
        self.pitches: List[float] = [asmidi(p) for p in pitches] if pitches else []
        self.offset:Optional[F] = None if offset is None else asF(offset)
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
        self.groupid = group
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
        self.properties: Optional[Dict[str,Any]] = properties
        self.fixedNotenames: Optional[Dict[int, str]] = None
        if self.isRest:
            assert self.duration > 0
            assert not self.pitches or (len(self.pitches) == 1 and self.pitches[0] == 0)
        else:
            assert self.pitches and all(p > 0 for p in self.pitches)
            for i, n in enumerate(pitches):
                if isinstance(n, str):
                    self.fixNotename(n, i)

    def fixNotename(self, notename:str= '', idx:int=0) -> None:
        """
        Fix the notename for the pitch at index **in place**

        Args:
            notename: if given, it will be fixed to the given notename.
                If nothing is given, it will be fixed to n2m(self.pitches[idx])
                Alternatively 'enharmonic' can be given as notename, in which
                case the enharmonic variant of the current notename will be used
            idx: the index of the note to modify

        See Also
        ~~~~~~~~

        notenames
        """
        if self.fixedNotenames is None:
            self.fixedNotenames = {}
        if not notename:
            notename = pt.m2n(self.pitches[idx])
        elif notename == 'enharmonic':
            notename = pt.enharmonic(self.notename(idx))
        self.fixedNotenames[idx] = notename

    def getFixedNotename(self, idx:int = 0) -> Optional[str]:
        if self.fixedNotenames:
            return self.fixedNotenames.get(idx)

    def isGraceNote(self) -> bool:
        return self.duration == 0

    def meanPitch(self) -> float:
        L = len(self.pitches)
        if self.isRest or L == 0:
            raise ValueError("No pitches to calculate mean")
        return self.pitches[0] if L == 1 else sum(self.pitches) / L

    @property
    def end(self) -> Optional[F]:
        if self.duration is not None and self.offset is not None:
            return self.offset + self.duration
        return None

    def _setPitches(self, pitches: List[pitch_t]) -> None:
        self.pitches = [asmidi(p) for p in pitches] if pitches else []
        for i, n in enumerate(pitches):
            if isinstance(n, str):
                self.fixNotename(n, i)

    def clone(self, **kws) -> Notation:
        """
        Clone this Notation, overriding any value.

        Args:
            kws: keyword arguments, as passed to the Notation constructor.
                Any parameter given will override the corresponding value in
                this Notation
        """
        out = self.copy()
        pitches = kws.pop('pitches', None)
        for key, value in kws.items():
            setattr(out, key, value)
        if pitches:
            out._setPitches(pitches)
        else:
            out.pitches = self.pitches
            if self.fixedNotenames:
                out.fixedNotenames = self.fixedNotenames.copy()
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

    def notename(self, index=0) -> str:
        """
        Returns the notename corresponding to the given pitch index

        If there is a fixed notename for the pitch, that will returned; otherwise
        the notename corresponding to the pitch

        Args:
            index: the index of the pitch (in self.pitches)

        Returns:
            the notename corresponing to the given pitch

        """
        return self.getFixedNotename(index) or pt.m2n(self.pitches[index])

    def pitchIndex(self, semitoneDivs=2, index=0) -> int:
        """
        The index of the nearest pitch/microtone

        For example, if divs_per_semitone is 2, then

        ====   ================
        note   microtone index
        ====   ================
        4C     0
        5C     0
        4C+    1
        4C#    2
        4Db    2
        …      …
        ====   ================
        """
        notename = self.notename(index=index)
        if semitoneDivs == 1:
            return pt.notated_pitch(notename).chromatic_index
        return pt.notated_pitch(notename).microtone_index(divs_per_semitone=semitoneDivs)

    @property
    def notenames(self) -> List[str]:
        return [self.getFixedNotename(i) or pt.m2n(p) for i, p in enumerate(self.pitches)]

    def verticalPosition(self, index=0) -> int:
        """
        Return the vertical position of the notated note at given index

        The vertical position is the position within the staff in terms of
        lines/spaces. It is calculated as octave*7 + diatonic_index

        =====   ===================
        Note     Vertical Position
        =====   ===================
        4C       28
        4C#      28
        4D       29
        4Eb      30
        ...      ...
        =====   ===================

        Args:
            index: the index of the pitch within this notation

        Returns:
            the vertical position

        """
        return pt.vertical_position(self.notename(index))

    def addAnnotation(self, text:Union[str, Annotation], placement:str='above',
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
        """
        The duration of the notated figure, in quarter-notes, independent of any tuples.

        A quarter-note inside a triplet would have a notatedDuration of 1
        """
        return notatedDuration(self.duration, self.durRatios)

    def mergeWith(self, other:Notation) -> Notation:
        """Merge this Notation with ``other``"""
        return mergeNotations(self, other)

    def setProperty(self, key:str, value) -> None:
        """
        Set any property of this Notation.
        Properties can be used, for example, for any rendering backend to
        pass directives which are specific to that rendering backend.
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
        elif self.duration == 0:
            info.append(f"{showT(self.offset)}:grace")
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
                info.append("[" + " ".join(self.notenames) + "]")
            else:
                info.append(self.notenames[0])
            if self.gliss:
                info.append("gliss")
        if self.groupid:
            if len(self.groupid) < 6:
                info.append(f"group={self.groupid}")
            else:
                info.append(f"group={self.groupid[:8]}…")

        if self.properties:
            info.append(f"properties={self.properties}")

        infostr = " ".join(info)
        return f"«{infostr}»"

    def transferAttributesTo(self: Notation, dest: Notation) -> None:
        """
        Copy attributes of self to dest
        """
        dest.tiedPrev = self.tiedPrev
        dest.gliss = self.gliss
        dest.articulation = self.articulation
        dest.noteheadHidden = self.noteheadHidden
        if self.annotations:
            if dest.annotations is None:
                dest.annotations = self.annotations
            else:
                dest.annotations.extend(self.annotations)

    def __len__(self) -> int:
        return len(self.pitches)

    def accidentalDirection(self, index=0, min_alteration=0.5) -> int:
        """
        Returns the direction of the alteration in this notation

        Args:
            index: index of the pitch within this Notation
            min_alteration: threshold (with min_alteration 0.5
                C+ gets a direction of +1, whereas C+25 still gets a direction
                of 0

        Returns:
            one of -1, 0 or +1, corresponding to the direction of the alteration
            (flat, natural or sharp)
        """
        n = self.notename(index=index)
        notated = pt.notated_pitch(n)
        return notated.alteration_direction(min_alteration=min_alteration)


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


def makeGroupId(parent=None) -> str:
    """
    Create an id to group notations together

    Returns:
        the group id as string
    """
    subgroup = str(uuid.uuid1())
    if parent is None:
        return subgroup
    return parent + "/" + subgroup


def makeNote(pitch:pitch_t, duration:time_t = None, offset:time_t = None,
             annotation:str=None, gliss=False, withId=False,
             gracenote=False, enharmonicSpelling: str = None,
             **kws) -> Notation:
    """
    Utility function to create a note Notation

    Args:
        pitch: the pitch as midinote or notename. If given a pitch as str,
            the note in question is fixed at the given enharmonic representation.
        duration: the duration of this Notation. Use None to leave this unset,
            0 creates a grace note
        offset: the offset of this Notation (None to leave unset)
        annotation: an optional text annotation for this note
        gliss: does this Notation start a glissando?
        withId: if True, this Notation has a group id and this id
            can be used to mark multiple notes as belonging to a same group
        gracenote: make this a grace note.
        enharmonicSpelling: if given, this spelling of pitch will be used
        **kws: any keyword accepted by Notation

    Returns:
        the created Notation
    """
    if gracenote:
        duration = 0
    else:
        duration = asF(duration) if duration is not None else None
    offset = asF(offset) if offset is not None else None
    assert 'isRest' not in kws
    out = Notation(pitches=[pitch], duration=duration, offset=offset, gliss=gliss, **kws)
    assert not out.isRest
    if annotation:
        out.addAnnotation(annotation)
    if withId:
        out.groupid = str(id(out))
    if enharmonicSpelling:
        out.fixNotename(enharmonicSpelling)
    return out


def makeChord(pitches: List[pitch_t], duration:time_t=None, offset:time_t=None,
              annotation:str=None, **kws) -> Notation:
    """
    Utility function to create a chord Notation

    Args:
        pitches: the pitches as midinotes or notenames. If given a note as str,
            the note in question is fixed at the given enharmonic representation.
        duration: the duration of this Notation. Use None to leave this unset,
            use 0 to create a grace note
        offset: the offset of this Notation (None to leave unset)
        annotation: a text annotation
        **kws: any keyword accepted by Notation

    Returns:
        the created Notation
    """
    duration = asF(duration) if duration is not None else None
    offset = asF(offset) if offset is not None else None
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
    Chain or Track, to signal a spacing between notations.
    Just explicitely setting the offset of a notation has the
    same effect

    Args:
        duration: the duration of the rest
        offset: the start time of the rest. Normally a rest's offset
            is left unspecified (None)
    """
    assert duration > 0
    return Notation(duration=asF(duration), offset=offset, isRest=True)
