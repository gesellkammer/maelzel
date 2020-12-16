"""
Functions and code snippets to make working with abjad less frustrating

"""
from dataclasses import dataclass
from fractions import Fraction as F
import copy
import enum

import abjad as abj
from emlib.music import lilytools
from emlib import typehints as t
from emlib import iterlib
import textwrap
import music21 as m21



def voiceMeanPitch(voice: abj.Voice) -> float:
    notes = [obj for obj in abj.iterate(voice).leaves() if isinstance(obj, abj.Note)]
    pitches = [noteGetMidinote(n) for n in notes]
    if not pitches:
        return 60
    avg = sum(pitches) / len(pitches)
    return avg


def noteGetMidinote(note: abj.Note) -> float:
    return note.written_pitch.number + 60


def voiceSetClef(voice: abj.Voice) -> abj.Voice:
    pitch = voiceMeanPitch(voice)
    if pitch < 48:
        clefid = "bass_8"
    elif pitch < 62:
        clefid = "bass"
    elif pitch < 80:
        clefid = "treble"
    else:
        clefid = "treble^15"
    clef = abj.Clef(clefid)
    # abj.attach(clef, next(abj.iterate(voice).by_leaf()))
    abj.attach(clef, next(abj.iterate(voice).leaves()))
    return voice


class TieTypes(enum.Enum):
    NOTTIED = 0
    TIEDFORWARD = 1
    TIEDBACKWARD = 2
    TIEDBOTH = 3


def noteTied(note: abj.core.Note) -> TieTypes:
    """
    Returns an int describind the state:
        0: not tied
        1: tied forward
        2: tied backwards
        3: tied in both directions

    tied = noteTied(note)
    forward = tied & 0b01
    backwards = tied & 0b10
    both = tied & 0b11
    """
    logical_tie = abj.inspect(note).logical_tie()
    if len(logical_tie) == 1:
        return TieTypes.NOTTIED
    if note is logical_tie[0]:
        return TieTypes.TIEDFORWARD
    elif note is logical_tie[-1]:
        return TieTypes.TIEDBACKWARD
    else:
        return TieTypes.TIEDBOTH


def addLiteral(obj, text:str, position:str= "after") -> None:
    """
    add a lilypond literal to obj

    obj: an abjad object (a note, a chord)
    text: the text to add
    position: one of 'before', 'after'
    """
    assert position in ('before', 'after')
    abj.attach(abj.LilyPondLiteral(text, format_slot=position), obj)


def isAttack(note:abj.Note) -> bool:
    """
    an attack is the first of a group of tied notes. The group
    can be of length 1 for a single note.
    """
    insp = abj.inspect(note)
    logical_tie = insp.logical_tie()
    tiehead = logical_tie[0] is note
    if insp.grace() and len(logical_tie) > 1 and tiehead:
        print("grace note with tie?")
        for t in logical_tie[1:]:
            abj.detach(t, note, by_id=True)
    print(note, tiehead)
    return tiehead


def getAttacks(voice:abj.Voice) -> t.List[abj.Note]:
    """
    Returns a list of notes or gracenotes which represent an attack
    (they are not tied to a previous note)
    """
    
    attacks = []
    for leaf in abj.iterate(voice).leaves():
        if isinstance(leaf, abj.Note) and isAttack(leaf):
            attacks.append(leaf)
    return attacks


def scoreToLily(score: abj.Score, pageSize: str=None, orientation: str=None,
                staffSize: int=None) -> abj.LilyPondFile:
    """
    Create a LilyPondFile from a score by adding a header and setting score layout
    
    Args:
        score: the abjad score
        pageSize: a3 or a4
        orientation: portrait, landscape
        staffSize: the size of a staff, in points

    Returns:
        the LilyPondFile
    """
    if pageSize is None and orientation is None:
        paperDef = None
    else:
        paperSize = pageSize.lower() if pageSize else 'a4'
        paperOrientation = orientation if orientation else 'portrait'
        assert orientation in ('landscape', 'portrait')
        assert paperSize in ('a4', 'a3')
        paperDef = (paperSize, paperOrientation)
    lilyfile = abj.LilyPondFile.new(score,
                                    global_staff_size=staffSize,
                                    default_paper_size=paperDef)
    lilyfileAddHeader(lilyfile)
    return lilyfile


def saveLily(score: abj.Score, outfile: str=None,
             pageSize: str=None, orientation: str=None, staffSize: int=None) -> str:
    """
    Save the score as a .ly file, returns the path of the saved file

    Args:
        score: the abj.Score to save
        outfile: the path of the lilypond file (or None to save to a temp file)
        pageSize: the size as str, one of "a4", "a3"
        orientation: one of 'landscape', 'portrait'
        staffSize: the size of the staff, in points. Default=12

    Returns:
        the path to the saved file
    """
    import tempfile
    if outfile is None:
        outfile = tempfile.mktemp(suffix=".ly")
    lilyfile = scoreToLily(score, pageSize=pageSize, orientation=orientation, staffSize=staffSize)
    with open(outfile, "w") as f:
        f.write(format(lilyfile))
    return outfile


def savePdf(score: abj.Score, outfile: str,
            pageSize: str=None, orientation: str=None, staffSize: int=None) -> None:
    """
    Save this score as pdf

    Args:
        score: the abj.Score to save
        outfile: the path to save to
        pageSize: the size as str, one of "a4", "a3"
        orientation: one of 'landscape', 'portrait'
        staffSize: the size of the staff, in points. Default=12
    """
    # we generate a lilyfile, then a pdf from there
    import tempfile
    lilyfile = tempfile.mktemp(suffix=".ly")
    saveLily(score, lilyfile, pageSize=pageSize, orientation=orientation, staffSize=staffSize)
    lilytools.lily2pdf(lilyfile, outfile)


def voicesToScore(voices: t.List[abj.Voice]) -> abj.Score:
    """
    voices: a list of voices as returned by [makevoice(notes) for notes in ...]

    Args:
        voices: a list of voices
    
    Return: 
        an abjad Score
    """
    voices.sort(key=voiceMeanPitch, reverse=True)
    staffs = [abj.Staff([voice]) for voice in voices]
    score = abj.Score(staffs)
    return score


def lilyfileFindBlock(lilyfile: abj.LilyPondFile, blockname:str) -> t.Opt[int]:
    """
    Find the index of a Block. This is used to find an insertion
    point to put macro definitions
    
    Args:
        lilyfile: an abjad LilypondFile
        blockname: the name of the block to find

    Returns:
        the index of the block, or None if not found
    """
    for i, item in enumerate(lilyfile.items):
        if isinstance(item, abj.Block) and item.name == blockname:
            return i
    return None


def lilyfileAddHeader(lilyfile: abj.LilyPondFile, enablePointAndClick=False) -> None:
    """
    Adds a header to the given LyliPondFile
    """
    gliss_header = textwrap.dedent(r"""
        glissandoSkipOn = {
            \override NoteColumn.glissando-skip = ##t
            \hide NoteHead
            \override NoteHead.no-ledgers = ##t
        }

        glissandoSkipOff = {
            \revert NoteColumn.glissando-skip
            \undo \hide NoteHead
            \revert NoteHead.no-ledgers
        }

    """)
    blocks = [gliss_header]
    if not enablePointAndClick:
        blocks.append(r"\pointAndClickOff")
    blocktext = "\n".join(blocks)
    idx = lilyfileFindBlock(lilyfile, "score")
    lilyfile.items.insert(idx, blocktext)


def voiceAddAnnotation(voice: abj.Voice, annotations: t.List[t.Opt[str]], 
                       fontSize:int=10, attacks=None) -> None:
    """
    Add the annotations to each note in this voice

    Args:
        voice: the voice to add annotations to
        annotations: a list of annotations. There should be one annotation
            per attack (if no annotation is needed for a specific attack,
            that slot should be None).
            An annotation can have a prefix: 
                _ is a bottom annotation (default) 
                ^ is an top annotation 
        attacks: the result of calling getAttacks. Used for the case where attacks
            have already been calculated before.
        fontSize: the size to use for the annotations
    """
    # prefix = "_" if orientation == "down" else "^"
    attacks = attacks or getAttacks(voice)
    if len(attacks) != len(annotations):
        for p in attacks: print(p)
        for p in annotations: print(p)
        # raise ValueError("Annotation mismatch")
    for attack, annotstr in zip(attacks, annotations):
        if not annotstr:
            continue
        annots = annotstr.split(";")
        for annot in annots:
            if annot[0] not in  "_^":
                prefix = "_"
            else:
                prefix = annot[0]
                annot = annot[1:]
            if fontSize <= 0:
                literal = f'{prefix}"{annot}"'
            else:
                literal = fr"{prefix}\markup {{\abs-fontSize #{fontSize} {{ {annot} }} }}"
            addLiteral(attack, literal, "after")
            # abjAddLiteral(attack, literal, "after")


def voiceAddGliss(voice: abj.Voice, glisses: t.List[bool], 
                  usemacros=True, skipsame=True, attacks=None) -> None:
    """
    Add glissando to the notes in the given voice

    Args:
        voice: an abjad Voice
        glisses: a list of bools, where each value indicates if the corresponding
                        note should produce a sgliss.
    """
    attacks = attacks or getAttacks(voice)
    assert len(attacks) == len(glisses)
    # We use macros defined in the header. These are added when the file is saved
    # later on
    glissandoSkipOn = textwrap.dedent(r"""
        \override NoteColumn.glissando-skip = ##t
        \hide NoteHead
        \override NoteHead.no-ledgers =  ##t    
    """)
    glissandoSkipOff = textwrap.dedent(r"""
        \revert NoteColumn.glissando-skip
        \undo \hide NoteHead
        \revert NoteHead.no-ledgers
    """)

    def samenote(n0: abj.Note, n1: abj.Note) -> bool:
        return n0.written_pitch == n1.written_pitch

    for (note0, note1), gliss in zip(iterlib.pairwise(attacks), glisses):
        if gliss:
            if samenote(note0, note1) and skipsame:
                continue
            if usemacros:
                addLiteral(note0, r"\glissando \glissandoSkipOn ", "after")
                addLiteral(note1, r"\glissandoSkipOff", "before")
            else:
                addLiteral(note0, r"\glissando " + glissandoSkipOn, "after")
                addLiteral(note1, glissandoSkipOff, "before")


def objDuration(obj) -> F:
    """
    Calculate the duration of obj. Raises TypeError if obj has no duration

    1/4 = 1 quarter note
    """
    if isinstance(obj, (abj.core.Tuplet, abj.core.Note, abj.core.Rest)):
        dur = abj.inspect(obj).duration()
        return dur
    raise TypeError(f"dur. not implemented for {type(obj)}")


def _abjTupleGetDurationType(tup: abj.core.Tuplet) -> str:
    tupdur = abj.inspect(tup).duration()
    mult = tup.multiplier
    if mult.denominator <= 3:
        durtype = {
            2: 'quarter',
            4: 'eighth',
            8: '16th',
            12: '16th',
            16: '32nd'
        }.get(tupdur.denominator)
    elif mult.denominator <= 7:
        durtype = {
            1: 'quarter',
            2: 'eighth',
            4: '16th',
            8: '32nd',
            12: '32nd',
            16: '64th'
        }.get(tupdur.denominator)
    elif mult.denominator <= 15:
        durtype = {
            4: '16th',
            8: '32nd',
            12: '32nd',
            16: '64th'
        }.get(tupdur.denominator)
    else:
        raise ValueError(f"??? {tup} dur: {tupdur}")
    if durtype is None:
        raise ValueError(f"tup {tup} ({tup.multiplier}), dur: {tupdur}")
    return durtype


def _abjDurClassify(num, den) -> t.Tup[str, int]:
    durname = {
        1: 'whole',
        2: 'half',
        4: 'quarter',
        8: 'eighth',
        16: '16th',
        32: '32nd',
        64: '64th'
    }[den]
    dots = {
        1: 0,
        3: 1,
        7: 2
    }[num]
    return durname, dots


def noteGetMusic21Duration(abjnote: abj.Leaf, tuplet: abj.Tuplet=None
                          ) -> m21.duration.Duration:
    dur = abjnote.written_duration * 4
    dur = F(dur.numerator, dur.denominator)
    dur = m21.duration.Duration(dur)
    if tuplet:
        dur.appendTuplet(tuplet)
    return dur


def noteToMusic21(abjnote: abj.Note, tuplet: abj.Tuplet=None) -> m21.note.Note:
    """
    Convert an abjad to a music21 note

    Args:
        abjnote: the abjad note to convert to
        tuplet: a lilipond tuplet, if applies
    
    Returns:
        the m21 note
    """
    dur = noteGetMusic21Duration(abjnote, tuplet)
    pitch = noteGetMidinote(abjnote)
    m21note = m21.note.Note(pitch, duration=dur)
    return m21note


def extractMatching(abjobj, matchfunc):
    if not hasattr(abjobj, '__iter__'):
        if matchfunc(abjobj):
            yield abjobj
    else:
        for elem in abjobj:
            yield from extractMatching(elem, matchfunc)


def _abjtom21(abjobj, m21stream, level=0, durfactor=4, 
              tup=None, state=None) -> m21.stream.Stream:
    """
    Convert an abjadobject to a m21 stream
    
    Args:
        abjobj: the abjad object to convert
        m21stream: the stream being converted to
        level: the level of recursion
        durfactor: current duration factor
        tup: current m21 tuplet
        state: a dictionary used to pass global state

    Returns: 
        the music21 stream
    """
    indent = "\t"*level
    if state is None:
        state = {}
    debug = state.get('debug', False)

    def append(stream, obj, msg=""):
        assert stream is not obj
        if debug:
            print(indent, f"{stream}  <- {obj}    {msg}")
        stream.append(obj)

    if hasattr(abjobj, '__iter__'):
        if debug:
            print(indent, "iter", type(abjobj), abjobj)
        if isinstance(abjobj, abj.core.Voice):       # Voice
            # voice = m21.stream.Voice()
            m21voice = m21.stream.Part()
            for meas in abjobj:
                _abjtom21(meas, m21voice, level+1, durfactor, tup, state=state)
            append(m21stream, m21voice, "stream append voice")
        elif isinstance(abjobj, abj.core.Staff):
            abjstaff = abjobj
            m21staff = m21.stream.Part()
            leaves = abj.select(abjstaff).leaves()
            measures = leaves.group_by_measure()
            # TODO: deal with timesignatures
            for meas in measures:
                m21meas = m21.stream.Measure()
                for elem in meas:
                    _abjtom21(elem, m21meas, level+1, durfactor, tup, state=state)
                append(m21staff, m21meas, "staff append meas")
            append(m21stream, m21staff, "stream append staff")
        elif isinstance(abjobj, abj.core.Tuplet):      # Tuplet
            mult = abjobj.multiplier
            newtup = m21.duration.Tuplet(mult.denominator, mult.numerator, bracket=True)
            m21durtype = _abjTupleGetDurationType(abjobj)
            newtup.setDurationType(m21durtype)
            if debug:
                print(indent, "Tuple!", mult, mult.numerator, mult.denominator, newtup)
            for elem in abjobj:
                _abjtom21(elem, m21stream, level+1, durfactor*abjobj.multiplier, tup=newtup, state=state)
            if debug:
                print(indent, "closing tuple")
        elif isinstance(abjobj, abj.core.Container):
            # a measure in a Voice
            # TODO: time signature
            meas = m21.stream.Measure()
            for elem in abjobj:
                _abjtom21(elem, meas, level+1, durfactor, tup, state=state)
            append(m21stream, meas, "container: stream append meas")
        else:
            if debug:
                print("????", type(abjobj), abjobj)
    else:
        if debug:
            print("\t"*level, "tup: ", tup, "no iter", type(abjobj), abjobj)
        if isinstance(abjobj, (abj.core.Rest, abj.core.Note)):
            # check if it has gracenotes
            graces = abj.inspect(abjobj).before_grace_container()
            # graces = abj.inspect(abjobj).grace_container()
            if graces:
                for grace in graces:
                    if isinstance(grace, abj.Note):
                        # add grace note to m21 stream
                        m21grace = noteToMusic21(grace).getGrace()
                        append(m21stream, m21grace)
            # abjdur = abjobj.written_duration
            # durtype, dots = _abjDurClassify(abjdur.numerator, abjdur.denominator)
            # dur = m21.duration.Duration(durtype, dots=dots)

            abjdur = abjobj.written_duration*4
            dur = m21.duration.Duration(F(abjdur.numerator, abjdur.denominator))
            if tup:
                dur.appendTuplet(tup)
            if isinstance(abjobj, abj.core.Rest):
                append(m21stream, m21.note.Rest(duration=dur))
            else:
                note = abjobj
                pitch = noteGetMidinote(note)
                m21note = m21.note.Note(pitch, duration=copy.deepcopy(dur))
                tie = noteTied(note)
                if debug:
                    print("\t"*level, "tie: ", tie)
                if tie == TieTypes.TIEDFORWARD or tie == TieTypes.TIEDBOTH:
                    m21note.tie = m21.tie.Tie()
                append(m21stream, m21note)
        else:
            if debug:
                print("\t"*level, "**** ???", type(abjobj), abjobj)
    return m21stream


def abjadToMusic21(abjadStream, debug=False) -> m21.stream.Stream:
    """
    Convert an abjad stream to a music21 stream

    Args:
        abjadStream: an abjad stream
        debug: If True, print debugging information

    Returns: 
        the corresponding music21 stream
    """
    m21stream = m21.stream.Stream()
    out = _abjtom21(abjadStream, m21stream, state={'debug': debug})
    return out
