from maelzel.core import *
from pitchtools import *
from maelzel.snd.audiosample import Sample
import numpy as np
import os
from pathlib import Path
import platform

outfolder = Path('output')
os.makedirs(outfolder, exist_ok=True)

def systemid():
    v = sys.version_info
    return f'{platform.system()}-{platform.machine()}-{v.major}.{v.minor}'


cfg = CoreConfig()
cfg['show.respellPitches'] = False
cfg['show.centsDeviationAsTextAnnotation'] = False
cfg['chordAdjustGain'] = False
cfg['show.voiceMaxStaves'] = 3
cfg.activate()

cl = Clip(os.path.abspath("../notebooks/snd/colours-german-male.flac"), pitch="4E")

dt = 1/16
times = np.arange(0, cl.durSecs(), dt)
chords = [cl.chordAt(t, mindb=-55, dur=dt, maxcount=12, ampfactor=10, maxfreq=m2f(100),
                     minfreq=40) or Rest(dt) for t in times]
chain1 = Chain(chords)

chain1.write(outfolder / f"clip-chords-1-{systemid()}.pdf")
chain1.rec(outfile=outfolder/f"clip-chords-1-{systemid()}.flac",
           gain=0.5, instr='sin', fade=(0.05, 0.05), sustain=0.05)


dt = 1/8
times = np.arange(0, cl.durSecs(), dt)
chords = [cl.chordAt(t, mindb=-55, dur=dt, maxcount=8, ampfactor=10, maxfreq=m2f(126),
                     minfreq=40) or Rest(dt) for t in times]
chain2 = Chain(chords)
chain2 = chain2.quantizePitch(step=0.5)

chain2.write(outfolder / f"clip-chords-2-{systemid()}.pdf")
chain2.rec(outfile=outfolder/f"clip-chords-2-{systemid()}.flac",
           gain=1.0, instr='.piano', fade=(0., 0.1), sustain=0.1)


