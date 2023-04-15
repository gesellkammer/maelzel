import argparse
from maelzel.core import *

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='test-notation.pdf')
args = parser.parse_args()


pitches = [60+i*0.5 for i in range(24)]
notes = [Note(p, F(1, 7)) for p in pitches]
voice = Voice(notes)
voice.write(args.output)
