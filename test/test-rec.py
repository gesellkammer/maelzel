from maelzel.core import *
n = Note(69, dur=4)
n.rec('a4-sine.wav', instr='sin', fade=0.1, nchnls=1)
