import argparse
from maelzel.core import *

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='test-rec.wav')
args = parser.parse_args()

n = Note(69, dur=4)
n.rec(args.output, instr='sin', fade=0.1, nchnls=1)
