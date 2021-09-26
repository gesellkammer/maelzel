from maelzel.core import *
from maelzel import core
from pitchtools import *


def generateNotes(minoctave=0, maxoctave=9):
    newnotes = {}
    dosharp = "CDFGA"
    doflat = "DEGAB"
    for octave in range(minoctave, maxoctave+1):
        for noteclass in "CDEFGAB":
            name = f"{noteclass}{octave}"
            newnotes[name] = n = Note(name)
            if noteclass in dosharp:
                sharp = f"{noteclass}x{octave}"
                newnotes[sharp] = n + 1
            if noteclass in doflat:
                flat = f"{noteclass}b{octave}"
                newnotes[flat] = n - 1
    return newnotes


N = Note
Ch = Chord
R = Rest

globals().update(generateNotes())

