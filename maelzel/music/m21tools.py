"""
Tools to work with music21
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    number_t = Union[int, float]
    pitch_t = Union[int, float, str]

import os
import tempfile
import warnings
import shutil
import glob

from maelzel.rational import Rat as F

import subprocess
from dataclasses import dataclass
import music21 as m21

from emlib import iterlib
from emlib import misc

from pitchtools import n2m, m2n, split_notename, split_cents
from maelzel.music import m21fix
from maelzel.music.timing import quartersToTimesig


def _splitchord(chord: m21.chord.Chord, 
                partabove: m21.stream.Part, 
                partbelow: m21.stream.Part, 
                split=60) -> Tuple[List[m21.note.Note], List[m21.note.Note]]:
    above, below = [], []
    for i in range(len(chord)):
        note = chord[i]
        if note.pitch.pitch >= split:
            above.append(note)
        else:
            below.append(note)

    def addnotes(part, notes, lyric=None):
        if not notes:
            rest = m21.note.Rest()
            if lyric:
                rest.lyric = lyric
            part.append(rest)
        else:
            ch = m21.chord.Chord(notes)
            lyric = "\n".join(note.lyric for note in reversed(notes) 
                              if note.lyric is not None)
            if lyric:
                ch.lyric = lyric
            part.append(ch)

    addnotes(partabove, above)
    addnotes(partbelow, below, lyric=chord.lyric)
    return above, below


def _asmidi(x) -> float:
    if isinstance(x, str):
        return n2m(x)
    elif isinstance(x, (int, float)):
        assert 0 <= x < 128, f"Expected a midinote (between 0-127), but got {x}"
        return x
    raise TypeError(f"Expected a midinote as number of notename, but got {x} ({type(x)})")


def logicalTies(stream:m21.stream.Stream) -> List[List[m21.note.NotRest]]:
    out = []
    current = []
    events = list(stream.getElementsByClass(m21.note.NotRest))
    if len(events) == 1:
        return [events]
    for ev0, ev1 in iterlib.pairwise(events):
        # if n0.pitch.ps != n1.pitch.ps or n0.tie is None:
        if ev0.tie is None:
            current.append(ev0)
            out.append(current)
            current = []
        elif ev0.tie is not None and ev0.tie.type in ('start', 'continue'):
            current.append(ev0)
    if events:
        current.append(events[-1])
    out.append(current)
    return out


def noteLogicalTie(note: m21.note.Note, stream: m21.stream.Stream
                   ) -> List[m21.note.Note]:
    """
    Return the group of notes tied to `note`

    Args:
        note: the note starting the tied group
        stream: the stream the note belongs to

    Returns:
        the list of notes tied to `note`, with `note`
        as the first element.

    """
    assert isinstance(note, m21.note.Note)
    if note.tie is None or note.tie.type != "start":
        return [note]
    out = [note]
    n0 = note
    while True:
        n1 = n0.next(m21.note.GeneralNote)
        if isinstance(n1, m21.note.Rest):
            break
        if not isinstance(n1, m21.note.Note):
            break
        out.append(n1)
        if n1.tie is None or n1.tie.type == "stop":
            break
        n0 = n1
    return out


def splitChords(chords: Sequence[m21.chord.Chord], split=60, force=False) -> m21.stream.Score:
    """
    split a seq. of music21 Chords in two staffs
    """
    assert isinstance(split, (int, float))
    for chord in chords:
        assert isinstance(chord, m21.chord.Chord)
    partabove = m21.stream.Part()
    partabove.append(m21.clef.TrebleClef())
    partbelow = m21.stream.Part()
    partbelow.append(m21.clef.BassClef())
    allabove = []
    allbelow = []
    for chord in chords:
        above, below = _splitchord(chord, partabove=partabove, partbelow=partbelow, split=split)
        allabove.extend(above)
        allbelow.extend(below)
    parts = []
    if allabove or force:
        parts.append(partabove)
    if allbelow or force:
        parts.append(partbelow)
    return m21.stream.Score(parts)


def isTiedToPrevious(note:m21.note.Note) -> bool:
    """
    Is this note tied to the previous note?
    """
    prev = note.previous('Note')
    if not prev:
        return False
    tie: m21.tie.Tie = prev.tie
    return tie is not None and tie.type in ('start', 'continued')


def getAttacks(stream: m21.stream.Stream) -> List[m21.note.NotRest]:
    return [tie[0] for tie in logicalTies(stream) if tie]


def endTime(obj: m21.note.GeneralNote) -> F:
    return obj.offset + obj.quarterLength


def attackPairs(stream: m21.stream.Stream) -> Iterable[Tuple[m21.note.NotRest, m21.note.NotRest]]:
    for g0, g1 in iterlib.pairwise(logicalTies(stream)):
        if g0 is None or g1 is None:
            continue
        if endTime(g0[-1]) == g1[0].offset:
            yield g0[0], g1[0]


def splitVoice(voice: m21.stream.Stream, split: int=60) -> m21.stream.Score:
    """
    split a music21 Voice in two staffs
    """
    above = []
    below = []
    for obj in voice:
        if obj.isClassOrSubclass((m21.note.GeneralNote,)):
            above.append(obj)
            continue
        rest = m21.note.Rest(duration=obj.duration)
        if isinstance(obj, m21.note.Rest):
            above.append(obj)
            below.append(obj)
        else:
            if obj.pitch.pitch >= split:
                above.append(obj)
                below.append(rest)
            else:
                below.append(obj)
                above.append(rest)
    partabove = m21.stream.Part()
    partabove.append(bestClef(above))
    for obj in above:
        partabove.append(obj)
    
    partbelow = m21.stream.Part()
    partbelow.append(bestClef(below))
    for obj in below:
        partbelow.append(obj)

    return m21.stream.Score([partabove, partbelow])


def bestClef(objs):
    avg = meanMidi(objs)
    if avg > 80:
        return m21.clef.Treble8vaClef()
    elif avg > 58:
        return m21.clef.TrebleClef()
    elif avg > 36:
        return m21.clef.BassClef()
    else:
        return m21.clef.Bass8vbClef()


def meanMidi(objs: Sequence[m21.Music21Object]):
    n, s = 0, 0
    stream = m21.stream.Stream()
    for obj in objs:
        stream.append(obj)
    for obj in stream.flat:
        try: 
            for pitch in obj.pitches:
                s += pitch.ps 
                n += 1
        except AttributeError:
            pass
    if n:
        return s/n
    return 0


def makeNoteSeq(midinotes: Sequence[float], dur=1, split=False) -> m21.stream.Stream:
    """
    Take a sequence of midi midinotes and create a Part (or a Score if
    split is True and midinotes need to be split between two staffs)

    midinotes: a seq. of midi values (fractional values are allowed)
    """
    s = m21.stream.Part()
    centroid = sum(midinotes)/len(midinotes)
    if centroid < 60:
        s.append(m21.clef.BassClef())
    for n in midinotes:
        s.append(m21.note.Note(n, quarterLength=dur))
    if split == 'auto' or split:
        if needsSplit(midinotes):
            return splitVoice(s)
        return s if not needsSplit(midinotes) else splitVoice(s)
    else:
        return s


def needsSplit(notes: Union[Sequence[float], m21.stream.Stream], splitpoint=60) -> bool:
    """

    notes:
        list of midinotes, or a m21 stream
    splitpoint:
        the note to use as splitpoint

    returns True if splitting is necessary
    """

    def _needsSplitMidinotes(midinotes):
        midi0 = min(midinotes)
        midi1 = max(midinotes)
        if midi0 < splitpoint - 7 and midi1 > splitpoint + 7:
            return True
        return False

    if isinstance(notes, list) and isinstance(notes[0], (int, float)):
        return _needsSplitMidinotes(notes)
    elif isinstance(notes, m21.stream.Stream):
        midinotes = [note.pitch.pitch for note in notes.getElementsByClass(m21.note.Note)]
        return _needsSplitMidinotes(midinotes)
    else:
        raise TypeError(f"expected a list of midinotes or a m21.Stream, got {notes}")


def makeTimesig(num_or_dur: Union[int, float], den:int=0) -> m21.meter.TimeSignature:
    """
    Create a m21 TimeSignature from either a numerator, denominator or from
    a duration in quarter notes.

    makeTimesig(2.5) -> 5/8
    makeTimesig(4)   -> 4/4
    """
    if den == 0:
        num, den = quartersToTimesig(num_or_dur)
    else:
        num = num_or_dur
    return m21.meter.TimeSignature(f"{num}/{den}")


_durationTypeFromFraction = {
    F(4): 'whole',
    F(2): 'half',
    F(1): 'quarter',
    F(1, 2): 'eighth',
    F(1, 4): '16th',
    F(1, 8): '32nd',
    F(1, 16): '64th'
}

def asF(f: number_t) -> F:
    if isinstance(f, F):
        return f
    return F(f)


def durationTypeFromQuarterDur(dur: number_t) -> str:
    return _durationTypeFromFraction[asF(dur)]


def makeDuration(durType: Union[str, number_t], dots=0, durRatios: List[F]=None,
                 tupleType:str='') -> m21.duration.Duration:
    if isinstance(durType, str):
        assert durType in {'half', 'quarter', 'eighth', '16th', '32nd', '64th'}
        dur = m21.duration.Duration(durType, dots=dots)
    else:
        durTypeStr = _durationTypeFromFraction[asF(durType)]
        dur = m21.duration.Duration(durTypeStr, dots=dots)

    if durRatios:
        ownRatio = durRatios[-1]
        for r in durRatios[:-1]:
            if r != 1:
                tup = m21.duration.Tuplet(r.numerator, r.denominator)
                dur.appendTuplet(tup)
        if ownRatio != 1:
            tup = m21.duration.Tuplet(ownRatio.numerator, ownRatio.denominator)
            if tupleType:
                tup.type = tupleType
            dur.appendTuplet(tup)
    return dur

def _makeDuration(durType: Union[str, number_t], dots=0, durRatios: List[F]=None,
                 ) -> m21.duration.Duration:
    """
    Args:
        durType: the notated duration, BEFORE applying any time modification
            1 = quarter note, 0.5 = eighth note, etc. Or a string, like 'half',
            'quarter', 'eighth', '16th', etc.
        dots: how many dots
        durRatios: the duration ratios to apply, if any. None indicates no time
            modification

    Returns:
        a music21 Duration to be used in a Note, Chord, etc.
    """
    if isinstance(durType, str):
        assert durType in {'half', 'quarter', 'eighth', '16th', '32nd', '64th'}
        dur = m21.duration.Duration(durType, dots=dots)
    else:
        durTypeStr = _durationTypeFromFraction[asF(durType)]
        dur = m21.duration.Duration(durTypeStr, dots=dots)
    if durRatios:
        for ratio in durRatios:
            if ratio != 1:
                tup = m21.duration.Tuplet(ratio.numerator, ratio.denominator)
                dur.appendTuplet(tup)

    return dur


def makeTie(obj: Union[m21.note.Note, m21.chord.Chord], tiedPrev:bool, tiedNext:bool) -> None:
    if tiedPrev and tiedNext:
        obj.tie = m21.tie.Tie('continue')
    elif tiedPrev:
        obj.tie = m21.tie.Tie('stop')
    elif tiedNext:
        obj.tie = m21.tie.Tie('start')
    else:
        obj.tie = None

_centsToAccidentalName = {
# cents   name
    0:   'natural',
    25:  'natural-up',
    50:  'quarter-sharp',
    75:  'sharp-down',
    100: 'sharp',
    125: 'sharp-up',
    150: 'three-quarters-sharp',

    -25: 'natural-down',
    -50: 'quarter-flat',
    -75: 'flat-up',
    -100:'flat',
    -125:'flat-down',
    -150:'three-quarters-flat'
}

_standardAccidentals = {'natural', 'sharp', 'flat',
                        'quarter-sharp', 'quarter-flat',
                        'three-quarters-sharp', 'three-quarters-down'}


def accidentalName(cents:int) -> str:
    """
    Given a number of cents, return the name of the accidental
    """
    if not (-150 <= cents <= 150):
        raise ValueError("cents should be between -150 and +150")
    rndcents = round(cents/25)*25
    name = _centsToAccidentalName.get(rndcents)
    assert name is not None
    return name


def makeAccidental(cents:int) -> m21.pitch.Accidental:
    """
    Make an accidental with possibly 1/8 tone alterration

    Example: create a C# 1/8 tone higher (C#+25)

    note = m21.note.Note(61)
    note.pitch.accidental = makeAccidental(125)
    """
    assert -150 <= cents <= 150
    name = accidentalName(round(cents/25)*25)
    semitone = cents/100.
    alter = round(semitone/0.5)*0.5
    accidental = m21.pitch.Accidental()
    accidental.alter = alter
    # the non-standard-name should be done in the end, because otherwise
    # other settings (like .alter) will wipe it
    nonStandard = name not in _standardAccidentals
    accidental.set(name, allowNonStandardValue=nonStandard)
    return accidental


def m21Notename(pitch: Union[str, float]) -> str:
    """
    Convert a midinote or notename (like "4C+10") to a notename accepted
    by music21. The decimal part (the non-chromatic part, in this case "+10")
    will be discarded

    Args:
        pitch: a midinote or a notename as returned by m2n
    
    Returns:
        the notename accepted by music21
    """
    notename = pitch if isinstance(pitch, str) else m2n(pitch)
    if "-" in notename or "+" in notename:
        sep = "-" if "-" in notename else "+"
        notename = notename.split(sep)[0]
    if notename[0].isdecimal():
        # 4C# -> C#4
        return notename[1:] + notename[0]
    # C#4 -> C#4
    return notename


def hideAccidental(note: m21.note.Note) -> None:
    accidental = note.pitch.accidental
    accidental.style.hideObjectOnPrint = True
    accidental.displayStatus = False
    accidental.displayType = "never"


def hideStem(obj: m21.note.NotRest) -> None:
    obj.stemDirection = "noStem"


def measureFixAccidentals(s: m21.stream.Measure) -> None:
    seen = {}
    for i, event in enumerate(s.getElementsByClass(m21.note.NotRest)):
        if isinstance(event, m21.note.Note):
            if i == 0 and isTiedToPrevious(event):
                print("Hiding accidental: ", event)
                hideAccidental(event)
            notes = [event]
        elif isinstance(event, m21.chord.Chord):
            notes = event.notes
        else:
            raise TypeError(f"Expected a Note or a Chord, got {type(event)}")
        for note in notes:
            lastSeen = seen.get(note.pitch.step)
            if lastSeen is None and note.pitch.accidental.name == "natural":
                hideAccidental(note)
            elif lastSeen == note.pitch.accidental.name:
                hideAccidental(note)
            seen[note.pitch.step] = note.pitch.accidental.name


def makePitch(pitch: Union[str, float], divsPerSemitone=4, hideAccidental=False
              ) -> Tuple[m21.pitch.Pitch, int]:
    """
    This is used to make a Pitch for a m21.Note (see makeNote)
    
    Args:
        pitch: a notename or a midinote as float
        divsPerSemitone: the number of divisions of the semitone. 4=1/8th tones
        hideAccidental: if True, hide the accidental

    Returns:
        a tuple (m21.pitch.Pitch, centsdev)
        where centsdev is the deviation in cents between the original pitch and
        the returned pitch, which is quantized to the divsPerSemitone specified

    """
    assert(isinstance(pitch, (str, int, float)))
    midinote = n2m(pitch) if isinstance(pitch, str) else pitch
    rounding_factor = 1/divsPerSemitone
    rounded_midinote = round(midinote/rounding_factor)*rounding_factor
    notename = m2n(rounded_midinote)
    octave, letter, alter, cents = split_notename(notename)
    basename, cents = split_cents(notename)
    m21notename = m21Notename(basename)
    cents += 100 if alter == "#" else -100 if alter == "b" else 0
    accidental = makeAccidental(cents)
    out = m21.pitch.Pitch(m21notename)
    if hideAccidental:
        accidental.style.hideObjectOnPrint = True
        accidental.displayStatus = False
        accidental.displayType = "never"
    out.accidental = accidental

    mididev = midinote-n2m(basename)
    centsdev = int(round(mididev*100))
    return out, centsdev


def _centsshown(centsdev, divsPerSemitone=4) -> str:
    """
    Given a cents deviation from a chromatic pitch, return
    a string to be shown along the notation, to indicate the
    true tuning of the note. If we are very close to a notated
    pitch (depending on divsPerSemitone), then we don't show
    anything. Otherwise, the deviation is always the deviation
    from the chromatic pitch

    Args:
        centsdev: the deviation from the chromatic pitch
        divsPerSemitone: if given, overrides the value in the config

    Returns:
        the string to be shown alongside the notated pitch
    """
    # cents can be also negative (see self.cents)
    pivot = int(round(100 / divsPerSemitone))
    dist = min(centsdev%pivot, -centsdev%pivot)
    if dist <= 2:
        return ""
    if centsdev < 0:
        # NB: this is not a normal - sign! We do this to avoid it being confused
        # with a syllable separator during rendering (this is currently the case
        # in musescore
        return f"–{-centsdev}"
    return str(int(centsdev))


availableNoteheads = {"slash", "triangle", "diamond", "square", "cross", "x",
                      "circle-x", "inverted-triangle", "arrow-down", "arrow-up",
                      "circled", "slashed", "back-slashed", "normal", "cluster", "circle-dot",
                      "rectangle", "do", "re", "mi", "fa", "sol", "la", "none"}


def setNotehead(note: m21.note.NotRest, notehead: str, fill:bool=None,
                parenthesis=False) -> None:
    """
    Modify the notehead of the given note/chord

    Args:
        note: the note/chord to modify
        notehead: the notehead type (one of availableNoteheads).
            "none" to hide the notehead
        fill: filled or unfilled?
        parenthesis: should the notehead be parenthesized?
    """
    if notehead == "none":
        hideNotehead(note)
        return
    assert notehead in availableNoteheads, \
        f"Unknown notehead {notehead}, possible values: {availableNoteheads}"
    note.notehead = notehead
    note.noteheadFill = fill
    note.noteheadParenthesis = parenthesis


def makeNotation(pitch: Union[pitch_t, Sequence[pitch_t]], divsPerSemitone=4, centsAsLyrics=False,
                 notehead: str=None, noteheadFill=None, hideAccidental=False,
                 stem:str=None, color:str='', **options
                 ) -> Union[m21.note.Note, m21.chord.Chord]:
    """
    Construct a Note or a Chord, depending on the number of pitches given

    Args:
        pitch: the pitch/pitches (as midinote or notename)
        divsPerSemitone: divisions per semitone (4=1/8 tones, possible values: 1, 2, 4)
        centsAsLyrics: display the cents deviation as text
        notehead: the notehead to use, or None to leave the default
        noteheadFill: in connection with notehead, determines the shape of the notehead
        hideAccidental: if True, hide the accidental
        stem: None to leave untouched, or one of 'hidden', ...
        color: the color of the note/chord
        options: any option will be passed to m21.note.Note or m21.chord.Chord

    Returns:
        the Note/Chord
    """

    if isinstance(pitch, (list, tuple)):
        pitches = [_asmidi(p) for p in pitch]
        out, centdevs = makeChord(pitches=pitches, divsPerSemitone=divsPerSemitone,
                                  centsAsLyrics=centsAsLyrics, notehead=notehead,
                                  noteheadFill=noteheadFill, hideAccidental=hideAccidental,
                                  **options)
    else:
        out, centdev = makeNote(pitch=_asmidi(pitch), divsPerSemitone=divsPerSemitone,
                                 centsAsLyrics=centsAsLyrics, notehead=notehead,
                                 noteheadFill=noteheadFill, hideAccidental=hideAccidental,
                                 **options)
    if stem == 'hidden':
        hideStem(out)

    if color:
        out.style.color = color
    return out


def _makeNote(pitch: Union[str, float], divsPerSemitone=4, centsAsLyrics=False,
             notehead: str=None, noteheadFill=None, hideAccidental=False,
             **options
             ) -> Tuple[m21.note.Note, int]:
    """
    Given a pitch as a (fractional) midinote or a notename, create a
    m21 Note with a max. 1/8 tone resolution.

    Any keyword option will be passed to m21.note.Note (for example,
    `duration` or `quarterLength`)

    Args:
        pitch: the pitch of the resulting note (for example, 60.20, or "4C+20")
        divsPerSemitone: divisions per semitone (4=1/8 tones, possible values: 1, 2, 4)
        centsAsLyrics: display the cents deviation as text
        notehead: the notehead to use, or None to leave the default
        noteheadFill: in connection with notehead, determines the shape of the notehead
        hideAccidental: if True, hide the accidental
        options: any option will be passed to m21.note.Note

    Returns:
        tuple (m21.Note, cents deviation from the returned note)
    """
    assert isinstance(pitch, (str, int, float))
    pitch, centsdev = makePitch(pitch=pitch, divsPerSemitone=divsPerSemitone,
                                hideAccidental=hideAccidental)
    note = m21.note.Note(60, **options)
    note.pitch = pitch
    if centsAsLyrics:
        lyric = centsAnnotation(centsdev)
        if lyric:
            note.lyric = lyric
    if notehead is not None:
        setNotehead(note, notehead, fill=noteheadFill)
    return note, centsdev


def makeNote(pitch: Union[str, float], divsPerSemitone=4,
             notehead: str=None, noteheadFill:bool=None, hideAccidental=False,
             tiedToPrevious=False,
             **options
             ) -> Tuple[m21.note.Note, int]:
    """
    Create a music21 Note

    Args:
        pitch: either a note name or a midi number
        divsPerSemitone: divisions per semitone
        notehead: the kind of notehead. Possible noteheads: XXX
        noteheadFill: should the notehead be filled (black)?
        hideAccidental: force a hidden accidental
        tiedToPrevious: is this note tied to the previous?
        **options: any options here are passed to m21.note.Note directly

    Returns:
        a tuple (m21.note.Note, cents deviation from the given pitch)

    """
    midinote = _asmidi(pitch)
    pitch, centsdev = makePitch(pitch=midinote, divsPerSemitone=divsPerSemitone,
                                hideAccidental=hideAccidental)
    note = m21.note.Note(midinote, **options)
    note.pitch = pitch
    if tiedToPrevious:
        note.pitch.accidental.setAttributeIndependently('alter', float(int(note.pitch.alter)))
    if notehead is not None:
        setNotehead(note, notehead, fill=noteheadFill)
    return note, centsdev


def makeChord(pitches: Sequence[float], divsPerSemitone:int=4, centsAsLyric=False,
              notehead: Optional[str] = None, noteheadFill=None, hideAccidental=False,
              tiedToPrevious=False,
              **options
              ) -> Tuple[m21.chord.Chord, List[int]]:
    """
    Create a m21 Chord with the given pitches, adjusting the accidentals to divsPerSemitone
    (up to 1/8 tone). If showcents is True, the cents deviations to the written pitch
    are placed as a lyric attached to the chord.
    The cents deviations are returned as a second argument

    Args:
        pitches: the midi notes
        divsPerSemitone: divisions per semitone (1, 2, 4)
        centsAsLyric: if True, cents deviation is added as lyric
        notehead: the notehead as str, or None to use the default
        noteheadFill: should the notehead be filled or hollow (None=default)
        hideAccidental: if True, hide the accidentals
        tiedToPrevious: is this chord tied to the previous?
        options: options passed to the Chord constructor (duration, quarterLength, etc)

    Returns:
        a tuple (Chord, list of cents deviations)
    """
    notes, centsdevs = [], []
    pitches = sorted(pitches)
    for pitch in pitches:
        note, centsdev = makeNote(pitch, divsPerSemitone=divsPerSemitone, showcents=False,
                                  hideAccidental=hideAccidental, tiedToPrevious=tiedToPrevious)
        notes.append(note)
        centsdevs.append(centsdev)
    chord = m21.chord.Chord(notes, **options)
    if centsAsLyric:
        centsdevs.reverse()
        annotation = centsAnnotation(centsdevs)
        if annotation:
            chord.lyric = annotation
    if notehead:
        setNotehead(chord, notehead, fill=noteheadFill)

    return chord, centsdevs


def centsAnnotation(centsdev:Union[int, Sequence[int]], divsPerSemi=4) -> str:
    """
    Given a cents deviation or a list thereof, construct an annotation
    as it would be placed as a lyric for a chord or a note
    
    Args:
        centsdev: the deviation from the written pitch in cents,
            or a list of deviations in the case of a chord
        divsPerSemi: divisions of the semitone
    
    Returns:
        an annotation string to be attached to a chord or a note
    """
    if isinstance(centsdev, int):
        return _centsshown(centsdev, divsPerSemitone=divsPerSemi)
    else:
        annotations = [str(_centsshown(dev, divsPerSemi)) for dev in centsdev]
        return ",".join(annotations) if any(annotations) else ""


def addGraceNote(pitch:Union[float, str, Sequence[float]], anchorNote:m21.note.GeneralNote, dur=1/2,
                 nachschlag=False, context='Measure') -> m21.note.Note:
    """
    Add a grace note (or a nachschlag) to anchor note.A
    Anchor note should be part of a stream, to which the grace note will be added

    Args:
        pitch: the pitch of the grace note (as midinote, or notename)
        anchorNote: the note the grace note will be added to
        dur: the written duration of the grace note
        nachschlag: if True, the grace note is added as nachschlag, after the anchor
        context: the context where anchor note is defined, as a str, or the
                 stream itself
    Returns:
        the added grace note
    """
    stream = context if isinstance(context, m21.stream.Stream) else anchorNote.getContextByClass(context)
    if isinstance(pitch, (list, tuple)):
        grace = makeChord(pitch, quarterLength=dur)[0].getGrace()
    else:
        grace = makeNote(pitch, quarterLength=dur)[0].getGrace()
    grace.duration.slash = False
    if nachschlag:
        grace.priority = 2
    stream.insert(anchorNote.getOffsetBySite(stream), grace)
    return grace


def findNextAttack(obj:Union[m21.note.NotRest, m21.chord.Chord]
                   ) -> Tuple[Optional[m21.note.NotRest], bool]:
    n = obj
    originalPitches = obj.pitches
    isContiguous = True
    while True:
        n = n.next(m21.note.GeneralNote)
        if n is None:
            # end of stream, no attack found
            return (None, False)
        if isinstance(n, m21.note.Rest):
            isContiguous = False
            continue
        if n.pitches != originalPitches:
            return n, isContiguous
        if n.tie:
            if n.tie.type == 'start':
                return n, isContiguous


def addGliss(start:m21.note.GeneralNote, end:m21.note.GeneralNote, linetype='solid',
             stream:Union[str, m21.stream.Stream]='Measure', hideTiedNotes=True,
             continuous=True, text:str=None
             ) -> m21.spanner.Spanner:
    """
    Add a glissando between start and end. Both notes should already be part
    of a stream

    Args:
        start: start note
        end: end note
        linetype: line type of the glissando
        stream: a concrete stream or a context class
        hideTiedNotes: if True, tied notes after start will be hidden (the notehead
            only, actually)
        continuous: if True, the slyde type is set to continuous. This will export
            as <slide> in musicxml (if False, the spanner is exported as <glissando>)
        text: text to use as a label along the line (normally "gliss.")

    Returns:
        the created Glissando
    """
    if not isinstance(start, m21.note.NotRest):
        raise TypeError(f"Expected a Note or a Chord, got {type(start)}")
    assert linetype in m21.spanner.Line.validLineTypes
    gliss = m21.spanner.Glissando([start, end])
    gliss.lineType = linetype
    # this creates a <slide> tag when converted to xml
    if continuous:
        gliss.slideType = 'continuous'
    gliss.label = text
    context = stream if isinstance(stream, m21.stream.Stream) else start.getContextByClass(stream)
    context.insert(start.getOffsetBySite(context), gliss)
    if hideTiedNotes:
        if isinstance(start, m21.note.Note):
            tiedNotes = noteLogicalTie(start, context)
            for tiedNote in tiedNotes[1:]:
                if not tiedNote.duration.isGrace:
                    hideNotehead(tiedNote)
        else:
            raise ValueError("hiding noteheads of chords is not supported yet")
    return gliss


_articulationNameToClass = {
    'accent': m21.articulations.Accent,
    'tenuto': m21.articulations.Tenuto,
    'staccato': m21.articulations.Staccato,
    'strongAccent': m21.articulations.StrongAccent
}


def makeArticulation(articulation: str) -> m21.articulations.Articulation:
    """
    Create an articulation based on the name
    """
    cls = _articulationNameToClass.get(articulation)
    if cls is None:
        raise KeyError(f"Articulation {articulation} unknown. "
                       f"Possible values: {_articulationNameToClass.keys()}")
    return cls()


def addArticulation(n: m21.note.GeneralNote, articulation: str) -> None:
    """
    Add an articulation to a note / chord
    """
    if n.articulations is None:
        n.articulations = []
    n.articulations.append(makeArticulation(articulation))


def _noteScalePitch(note: m21.note.Note, factor: number_t) -> None:
    """
    Scale the pitch of note INPLACE

    Args:
        note: a m21 note
        factor: the factor to multiply pitch by
    """
    midinote = float(note.pitch.ps * factor)
    pitch, centsdev = makePitch(midinote)
    note.pitch = pitch   


def stackPartsInplace(parts:Sequence[m21.stream.Stream], outstream:m21.stream.Stream) -> None:
    """
    This solves the problem that a Score will stack Parts vertically,
    but append any other stream horizontally
    """
    for part in parts:
        part = misc.astype(m21.stream.Part, part)
        outstream.insert(0, part)


def stackParts(parts:Sequence[m21.stream.Stream], outstream:m21.stream.Stream=None
               ) -> m21.stream.Score:
    """
    Create a score from the different parts given

    This solves the problem that a Score will stack Parts vertically,
    but append any other stream horizontally
    """
    if outstream is None:
        outstream = m21.stream.Score()
    stackPartsInplace(parts, outstream=outstream)
    return outstream


def attachToObject(obj:m21.Music21Object, thingToAttach:m21.Music21Object, contextclass):
    context = obj.getContextByClass(contextclass)
    offset = obj.getOffsetBySite(context)
    context.insert(offset, thingToAttach)


def addTextExpression(note:m21.note.GeneralNote, text:str, placement="above",
                      contextclass='Measure', fontSize:float=None,
                      letterSpacing:float=None, fontWeight:str=None) -> None:
    """
    Add a text expression to note. The note needs to be already inside a stream,
    since the text expression is added to the stream, next to the note

    Args:
        note: the note (or chord) to add text expr. to
        text: the text
        placement: above or below
        contextclass: the context in which note is defined (passed to getContextByClass)
        fontSize: the size of the font
        letterSpacing: the spacing between letters
        fontWeight: the weight of the text
    """
    textexpr = makeTextExpression(text=text, placement=placement, fontSize=fontSize,
                                  letterSpacing=letterSpacing, fontWeight=fontWeight)
    attachToObject(note, textexpr, contextclass)


def makePart(clef:str=None, partName=None, abbreviatedName=None,
             timesig:Tuple[int, int]=None) -> m21.stream.Part:
    """
    Returns an empty Part with the attributes given

    Args:
        clef: one of 'treble', 'bass', 'treble8', 'bass8', 'ato', 'tenor'
        partName: the name of the part
        abbreviatedName: abbreviated name of the part
        timesig: initial time signature
    """
    part = m21.stream.Part()
    if partName:
        part.insert(0, m21.instrument.Instrument(partName))
        # part.partName = partName
    if abbreviatedName:
        part.partAbbreviation = abbreviatedName
    if clef:
        part.insert(0, makeClef(clef))
    if timesig:
        part.insert(0, makeTimesig(timesig[0], timesig[1]))
    return part


def makeTextExpression(text:str,
                       placement="above",
                       fontSize:float=None,
                       letterSpacing:float=None,
                       fontWeight:str=None
                       ) -> m21.expressions.TextExpression:
    """
    Create a TextExpression. You still need to attach it to something
    (see addTextExpression)

    Args:
        text: the text of the expression
        placement: one of "above", "below"
        fontSize: the size of the font
        letterSpacing: spacing between letters
        fontWeight: weight of the font

    Returns:
        the TextExpression
    """
    textexpr = m21.expressions.TextExpression(text)
    textexpr.positionPlacement = placement
    if fontSize:
        textexpr.style.fontSize = fontSize
    if letterSpacing:
        textexpr.style.letterSpacing = letterSpacing
    if fontWeight:
        textexpr.style.fontWeight = fontWeight
    return textexpr


_clefNameToClass = {
    'bass': m21.clef.BassClef,
    'bass8': m21.clef.Bass8vbClef,
    'bass8vb':m21.clef.Bass8vbClef,
    'viola': m21.clef.AltoClef,
    'alto': m21.clef.AltoClef,
    'violin':m21.clef.TrebleClef,
    'g': m21.clef.TrebleClef,
    'f': m21.clef.BassClef,
    'c': m21.clef.AltoClef,
    'tenor': m21.clef.TenorClef,
    'treble': m21.clef.TrebleClef,
    'treble8': m21.clef.Treble8vaClef,
    'treble8va': m21.clef.Treble8vaClef
}


def makeClef(clef:str) -> m21.clef.Clef:
    """
    Create a music21 clef

    Args:
        clef: the clef to create. Possible values: treble, alto, tenor, bass, treble8,
            bass8. There are aliases also (g=violin=treble, f=bass, c=viola=alto)

    Returns:
        a m21.clef.Clef of the specified type

    """
    cls = _clefNameToClass.get(clef)
    if cls is None:
        raise KeyError(f"Clef {clef} unknown. Possible names: {_clefNameToClass.keys()}")
    return cls()


def makeMetronomeMark(number:Union[int, float], text:str=None, referent:str=None
                      ) -> m21.tempo.MetronomeMark:
    referentNote = m21.note.Note(type=referent) if referent else None
    mark = m21.tempo.MetronomeMark(number=number, text=text, referent=referentNote)
    mark.positionPlacement = "above"
    return mark


def makeExpressionsFromLyrics(part: m21.stream.Part, **kws):
    """
    Iterate over notes and chords in part and if a lyric is present
    move it to a text expression

    Args:
        part: the part to modify
        kws: any attribute passed to makeTextExpression (placement, fontSize, etc)
    """
    part.makeMeasures(inPlace=True)
    for event in part.getElementsByClass(m21.note.NotRest):
        if event.lyric:
            text = event.lyric
            event.lyric = None 
            addTextExpression(event, text=text, **kws)
    

def hideNotehead(event:m21.note.NotRest, hideAccidental=True) -> None:
    if isinstance(event, m21.note.Note):
        event.style.hideObjectOnPrint = True
        # event.notehead = "none"
        event.notehead = None
        if hideAccidental:
            event.pitch.accidental.displayStatus = False
    elif isinstance(event, m21.chord.Chord):
        for note in event:
            note.style.hideObjectOnPrint = True
            if hideAccidental:
                note.pitch.accidental.displayStatus = False
    else:
        raise TypeError(f"expected a Note or a Chord, got {type(event)}")


def addDynamic(note:m21.note.GeneralNote, dynamic:str, contextclass='Measure') -> None:
    attachToObject(note, m21.dynamics.Dynamic(dynamic), contextclass)


@dataclass
class TextAnnotation:
    text: str
    placement: str = "above"
    fontSize: Optional[int] = None


barlineTypes = {
    'regular'
    'double',
    'heavy',
    'dashed',
    'final',
    'tick',
    'light-light',
    'short',
    'none',
    'heavy-heavy'
}


def makeMeasure(timesig: Tuple[int, int],
                timesigIsNew = True,
                barline: str = '',
                metronome: int = None,
                metronomeReferent: str = "quarter",
                annotation: Union[str, TextAnnotation] = "") -> m21.stream.Measure:
    """
    Create a Measure

    Args:
        timesig: the time signature as a tuple (num, den)
        timesigIsNew: is this a new timesignature?
        barline: the type of the barline. See barlineTypes.
        metronome: if given, add a metronome mark to this measure
        metronomeReferent: if metronome is given, set the referent for the
            value (one of "quarter", "eighth", etc)
        annotation: an optional annotation for the measure

    Returns:
        the created Measure
    """
    m = m21.stream.Measure()
    if timesigIsNew:
        m.timeSignature = makeTimesig(*timesig)
        m.timeSignatureIsNew = timesigIsNew
    if barline:
        assert barline in barlineTypes, \
            f"Unknown barline {barline}, known values: {barlineTypes}"
        m.rightBarline = barline
    if metronome:
        tempoMark = makeMetronomeMark(metronome, referent=metronomeReferent)
        m.insert(0, tempoMark)
    if annotation:
        if isinstance(annotation, TextAnnotation):
            text = annotation.text
            placement = annotation.placement
            fontSize = annotation.fontSize
        else:
            text, placement, fontSize = annotation, "above", None
        m.insert(0, makeTextExpression(text=text, placement=placement, fontSize=fontSize))
    return m


def scoreSchema(durs: Sequence[float],
                default='rest',
                barlines: Dict[int, str] = None,
                measureLabels: Dict[int, str]   = None,
                notes: Dict[int, float]  = None,
                separators: Dict[int, dict] = None,
                tempo: int = None,
                ) -> m21.stream.Part:
    """
    Make an empty score where each measure is indicated by the duration
    in quarters.

    Args:

        durs: a seq. of durations, where each duration indicates the length
              of each measure.
              e.g: 1.5 -> 3/8, 1.25 -> 5/16, 4 -> 4/4
        barlines:
            if given, a dictionary of measure_index: barline_style
            Possible styles are: 'regular', 'double', 'dashed' or 'final'
        measureLabels:
            if given, a dictionary of measure_index: label
            The label will be attached as a text expression to the measure
        notes:
            if given, a dictionary of measure_idx: midinote
            This note will be used instead of the default
        separators:
            if given, a dict of measure_idx: sep_dict where sep_dict
            can have the keys {'dur': duration, 'fill': 'rest' / 'fill': midinote}
            A separator adds a measure before the given idx. Separators don't affect
            measure indices used in other indicators (barlines, labels, notes)
        default: either 'rest' or a midinote, will be used to fill measures
        tempo: a tempo to be added to the score.

    Returns:
        a `music21.Part`, which can be wrapped in a Score or used as is
    """
    part = m21.stream.Part()
    measnum = 0
    for i, dur in enumerate(durs):
        measnum += 1
        sep = separators.get(i) if separators else None
        if sep:
            sepdur = sep.get(dur, 1)
            sepfill = sep.get('fill', 'rest')
            sepmeas = m21.stream.Measure(number=measnum)
            sepmeas.timeSignature = makeTimesig(sepdur)
            if sepfill == 'rest' or sepfill == 0:
                sepmeas.append(m21.note.Rest(quarterLength=sepdur))
            else:
                sepmeas.append(m21.note.Note(sepfill, quarterLength=sepdur))
            part.append(sepmeas)
            measnum += 1
        meas = m21.stream.Measure(number=i+1)
        meas.timeSignature = makeTimesig(dur)
        barline = barlines.get(i) if barlines else None
        if barline:
            meas.rightBarline = barline
        label = measureLabels.get(i) if measureLabels else None
        if label:
            meas.append(m21.expressions.TextExpression(label))
        midinote = notes.get(i) if notes is not None else None
        if midinote:
            meas.append(m21.note.Note(midinote, quarterLength=dur))
        elif default == 'rest':
            meas.append(m21.note.Rest(quarterLength=dur))
        else:
            meas.append(m21.note.Note(default, quarterLength=dur))
        part.append(meas)

    part[-1].rightBarline = "final"
    if tempo is not None:
        part[0].insert(0, m21.tempo.MetronomeMark(number=tempo))
    return part


def getXml(m21stream: m21.stream.Stream) -> str:
    """
    Generate musicxml from the given m21 stream, return it as a str

    Args:
        m21stream:  a m21 stream

    Returns:
        the xml generated, as string
    """
    exporter = m21.musicxml.m21ToXml.GeneralObjectExporter(m21stream)
    return exporter.parse().decode('utf-8')


def saveLily(m21stream, outfile: str) -> str:
    """
    Save to lilypond via musicxml2ly. Returns the saved path

    Args:
        m21stream: (m21.Stream) the stream to save
        outfile: (str) the name of the outfile

    Returns:
        (str) the saved path
    """
    from maelzel.music import lilytools
    xmlpath = str(m21stream.write('xml'))
    if not os.path.exists(xmlpath):
        raise RuntimeError("Could not write stream to xml")
    lypath = lilytools.musicxml2ly(str(xmlpath), outfile=outfile)
    return lypath


def renderViaLily(m21obj:m21.Music21Object, fmt:str=None, outfile:str=None, show=False) -> str:
    """
    Create a pdf or png via lilypond, bypassing the builtin converter
    (using musicxml2ly instead)

    To use the builtin method, use stream.write('lily.pdf') or
    stream.write('lily.png')

    Args:
        m21obj: the stream to convert to lilypond
        fmt: one of 'png' or 'pdf'
        outfile: if given, the name of the lilypond file generated. Otherwise
            a temporary file is created
        show: if True, show the resulting file

    Returns:
        the path of the saved file (pdf or png)
    """
    if outfile is None and fmt is None:
        fmt = 'png'
    elif fmt is None:
        assert outfile is not None
        fmt = os.path.splitext(outfile)[1][1:]
    elif outfile is None:
        assert fmt in ('png', 'pdf')
        outfile = tempfile.mktemp(suffix="."+fmt)
    else:
        ext = os.path.splitext(outfile)[1][1:]
        if fmt != ext:
            raise ValueError(f"outfile has an extension ({ext}) which does not match the format given ({fmt})")
    assert fmt in ('png', 'pdf')
    from maelzel.music import lilytools
    xmlpath = str(m21obj.write('xml'))
    if not os.path.exists(xmlpath):
        raise RuntimeError("Could not write stream to xml")
    lypath = lilytools.musicxml2ly(str(xmlpath))
    if not os.path.exists(lypath):
        raise RuntimeError(f"Error converting {xmlpath} to lilypond {lypath}")    
    if fmt == 'png':
        outfile2 = lilytools.renderLily(lypath, outfile)
        # outfile2 = lilytools.lily2png(lypath, outfile)
    elif fmt == 'pdf':
        outfile2 = lilytools.renderLily(lypath, outfile)
    else:
        raise ValueError(f"fmt should be png or pdf, got {fmt}")
    assert outfile2 is not None
    if not os.path.exists(outfile2):
        raise RuntimeError(f"Error converting lilypond file {lypath} to {fmt} {outfile}")
    if show:
        misc.open_with_standard_app(outfile)
    return outfile


def makeImage(m21obj, outfile:str=None, fmt='xml.png', musicxml2ly=True
              ) -> str:
    """
    Generate an image from m21obj

    Args:
        m21obj: the object to make an image from
        outfile: the file to write to, or None to create a temporary
        fmt: the format, one of "xml.png" or "lily.png"
        musicxml2ly: if fmt is lily.png and musicxml2ly is True, then conversion
                     is performed via the external tool `musicxml2ly`, otherwise
                     the conversion routine provided by music21 is used

    Returns:
        the path to the generated image file
    """
    if isinstance(m21obj, m21.stream.Stream):
        m21obj = m21fix.fixStream(m21obj, inPlace=True)
    method, fmt3 = fmt.split(".")
    if method == 'lily' and musicxml2ly:
        if fmt3 not in ('png', 'pdf'):
            raise ValueError(f"fmt should be one of 'lily.png', 'lily.pdf' (got {fmt})")
        if outfile is None:
            outfile = tempfile.mktemp(suffix="."+fmt3)
        path = renderViaLily(m21obj, fmt=fmt3, outfile=outfile)
    else:
        tmpfile = m21obj.write(fmt)
        if outfile is not None:
            os.rename(tmpfile, outfile)
            path = outfile
        else:
            path = tmpfile
    return str(path)


def showImage(m21obj, fmt='xml.png'):
    imgpath = makeImage(m21obj, fmt=fmt)
    from maelzel.core import tools
    tools.pngShow(imgpath)


def writePdf(m21stream, outfile, fmt='musicxml.pdf') -> None:
    """
    Write the stream to a pdf

    Args:
        m21stream: a stream (preferably a Score)
        outfile: the .pdf to write to
        fmt: the format used ('musicxml.pdf' to use MuseScore, 'lily.pdf' to use lilypond)
    """
    base, ext = os.path.splitext(outfile)
    assert ext == '.pdf'
    xmlfile = base + '.xml'
    m21stream.write(fmt, xmlfile)
    assert os.path.exists(outfile)


def scoreSetMetadata(score: m21.stream.Score, title="", composer="") -> None:
    """
    Add title/composer to the given score. This metadata is used when converting
    to pdf and rendered as text

    Args:
        score: the score to modify in place
        title: a title for this score
        composer: composer text is added at the right corner

    """
    score.insert(0, m21.metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = composer


def addBestClefs(part: m21.stream.Stream, threshold=4) -> None:
    """
    Add clefs to part whenever needed. For a clef change it is necessary
    that at least `threshold` notes are outside the ambitus of the current
    clef.

    Args:
        part: a stream holding the part to be modified
        threshold: the min. number of notes outside the ambitus to justify a clef change
    """
    notes = list(part.getElementsByClass(m21.note.Note))
    if not notes:
        warnings.warn("The part is empty")
        return
    currentClef = bestClef(notes[:threshold])
    attachToObject(notes[0], currentClef)
    for group in notes[threshold::threshold]:
        bestnow = bestClef(group)
        if bestnow != currentClef:
            currentClef = bestnow
            attachToObject(group[0], currentClef, contextclass='Measure')


def iterPart(part: m21.stream.Part, cls=m21.note.GeneralNote
            ) -> Iterable[Tuple[m21.stream.Measure, m21.note.GeneralNote]]:
    """
    Iterates over all items in a Part matching a specific class. For each
    item a tuple (measure, item) is yielded

    Args:
        part: the Part to iterate
        cls: the class to use as selector within each measure

    Returns:
        an iterator where each yielded element is a tuple (measure, item), where
        item matches the class of `cls`

    Example::

        # iterate over the notes of a part, in order to modify them
        # (otherwise one should call .flat)
        for location in

    """
    for measure in part.getElementsByClass(m21.stream.Measure):
        for item in measure.getElementsByClass(cls):
            yield (measure, item)


def fixNachschlag(n: m21.note.Note, durtype:str=None, priority=0) -> None:
    """
    Fix a grace note in place, so that it is displayed correctly in the case
    it is a Nachschlag (a grace note after another note)

    Args:
        n: the note to fix. It must be a grace note
        durtype: the displayed duration. It must be an eighth note or shorter
        priority: higher priorities are sorted later for the same offset
    """
    assert n.duration.isGrace
    assert durtype in {'eighth', '16th', '32nd', None}
    n.duration.slash = False
    if durtype is not None:
        n.duration.type = durtype
    n.priority = priority


def _fixNachschlaege(part: m21.stream.Part) -> None:
    """
    Fix Nachschläge within a part, in place

    A Nachschlag is a grace note placed after the main note.
    In musicxml, for a nachschlag to be rendered properly it must be
    of type "eighth" and cannot be slashed.
    """
    for loc0, loc1 in iterlib.pairwise(iterPart(part, m21.note.GeneralNote)):
        n0 = loc0[1]
        if n0.duration.isGrace and loc1[1].isRest:
            assert isinstance(n0, m21.note.Note)
            fixNachschlag(n0)


def fixNachschlaege(part: m21.stream.Part, convertToRealNote=False, duration=1/8) -> None:
    """
    Fix Nachschläge within a part, in place

    A Nachschlag is a grace note placed after the main note.
    In musicxml, for a nachschlag to be rendered properly it must be
    of type "eighth" and cannot be slashed.
    """
    for loc0, loc1 in iterlib.pairwise(iterPart(part, m21.note.GeneralNote)):
        n0 = loc0[1]
        if n0.duration.isGrace and loc1[1].isRest:
            assert isinstance(n0, m21.note.Note)
            fixNachschlag(n0)
            if convertToRealNote and duration > 0:
                m1 = loc1[0]
                n1 = loc1[1]
                realizedDuration = min(duration, n1.quarterLength/2)
                loc0[0].remove(n0)
                m1.remove(n1)

                n1.quarterLength = n1.quarterLength - realizedDuration
                replacement, centsdev = makeNote(n0.pitch.pitch, quarterLength=realizedDuration)
                m1.insert(n1.offset, replacement)
                m1.insert(n1.offset + realizedDuration, n1)


def _musescorePath() -> Optional[str]:
    us = m21.environment.UserSettings()
    musicxmlpath = us['musescoreDirectPNGPath']
    if os.path.exists(musicxmlpath):
        return str(musicxmlpath)
    path = shutil.which('musescore')
    if path is not None:
        return path
    return None


def _musescoreRenderMusicxmlToPng(xmlfile:str, outfile: str, page=1, trim=True):
    musescore = _musescorePath()
    if musescore is None:
        raise RuntimeError("MuseScore not found")
    args = [musescore, '--no-webview']
    if trim:
        args.extend(['--trim-image', '10'])
    args.extend(['--export-to', outfile, xmlfile])
    subprocess.call(args, stderr=subprocess.PIPE)
    generatedFiles = glob.glob(os.path.splitext(outfile)[0] + "-*.png")
    if not generatedFiles:
        raise RuntimeError("No output files generated")
    for generatedFile in generatedFiles:
        generatedPage = int(os.path.splitext(generatedFile)[0].split("-")[-1])
        if generatedPage == page:
            os.rename(generatedFile, outfile)
            return
    raise RuntimeError(f"Page not found, generated files: {generatedFiles}")


def renderMusicxml(xmlfile: str, outfile: str, method:str=None) -> None:
    """
    Convert a saved musicxml file to pdf or png

    Args:
        xmlfile: the musicxml file to convert
        outfile: the output file. The extension determines the output
            format. Possible formats pdf and png
        method: if given, will determine the method used to render. Use
            None to indicate a default method.
            Possible values: 'musescore'


    Supported methods::

        | format  |  methods   |
        |---------|------------|
        | pdf     |  musescore |
        | png     |  musescore |

    """
    fmt = os.path.splitext(outfile)[1]
    if fmt == ".pdf":
        method = method or 'musescore'
        if method == 'musescore':
            musescore = _musescorePath()
            if musescore is None:
                raise RuntimeError("MuseScore not found")
            subprocess.call([musescore,
                             '--no-webview',
                             '--export-to', outfile,
                             xmlfile],
                            stderr=subprocess.PIPE)
            if not os.path.exists(outfile):
                raise RuntimeError(f"Could not generate pdf file {outfile} from {xmlfile}")
        else:
            raise ValueError(f"method {method} unknown, possible values: 'musescore'")
    elif fmt == '.png':
        method = method or 'musescore'
        if method == 'musescore':
            _musescoreRenderMusicxmlToPng(xmlfile, outfile)
        else:
            raise ValueError(f"method {method} unknown, possible values: 'musescore'")