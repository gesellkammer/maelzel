from . import musicobj
import music21 as m21

def importMusic21Part(m21part: m21.stream.Part) -> musicobj.Voice:
    # for now, do not take care of score structure
    for x in m21part:
        if isinstance(x, m21.stream.Measure):
            for n in x.getElementsByClass(m21.note.GeneralNote):
                print(n, n.duration)

def m21NoteToNote(note: m21.note.Note) -> musicobj.Note:
    pass

