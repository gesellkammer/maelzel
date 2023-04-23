"""
Functions and code snippets to make working with abjad less frustrating

"""
from __future__ import annotations

try:
    import abjad as abj
except ImportError:
    raise ImportError("Abjad was not found. Install it in order to use this module")


from maelzel.common import F
import enum
from maelzel.music import lilytools
from emlib import iterlib
import textwrap

from typing import Optional


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
    an attack is the first of a tree of tied notes. The tree
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


def getAttacks(voice:abj.Voice) -> list[abj.Note]:
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
    lilytools.renderLily(lilyfile=lilyfile, outfile=outfile)


def voicesToScore(voices: list[abj.Voice]) -> abj.Score:
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


def lilyfileFindBlock(lilyfile: abj.LilyPondFile, blockname:str) -> Optional[int]:
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


def voiceAddAnnotation(voice: abj.Voice, annotations: list[Optional[str]],
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


def voiceAddGliss(voice: abj.Voice, glisses: list[bool],
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
    Calculate the totalDuration of obj. Raises TypeError if obj has no totalDuration

    1/4 = 1 quarter note
    """
    if isinstance(obj, (abj.core.Tuplet, abj.core.Note, abj.core.Rest)):
        dur = abj.inspect(obj).durationSecs()
        return dur
    raise TypeError(f"dur. not implemented for {type(obj)}")


def _abjTupleGetDurationType(tup: abj.core.Tuplet) -> str:
    tupdur = abj.inspect(tup).durationSecs()
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


def _abjDurClassify(num, den) -> tuple[str, int]:
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


def extractMatching(abjobj, matchfunc):
    if not hasattr(abjobj, '__iter__'):
        if matchfunc(abjobj):
            yield abjobj
    else:
        for elem in abjobj:
            yield from extractMatching(elem, matchfunc)



