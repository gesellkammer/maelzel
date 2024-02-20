from maelzel.snd.audiosample import Sample
from maelzel.snd import rosita
import matplotlib.pyplot as plt
import numpy as np
import sys, os, platform, argparse

outfolder = 'output'

os.makedirs(outfolder, exist_ok=True)


def systemid():
    v = sys.version_info
    return f'{platform.system()}-{platform.machine()}-{v.major}.{v.minor}'


def centroid(sndfile):
    s = Sample(sndfile)

    centroid = rosita.spectral_centroid(y=s.getChannel(0).samples, sr=s.sr, n_fft=2048, hop_length=512)
    print(centroid.shape)
    centroid0 = centroid.T[:,0]
    times = np.arange(0, s.duration, 512/s.sr)

    ax = s.plotSpectrogram()
    ax.plot(times, centroid0, color='#ffffff', linewidth=2)
    suffix = systemid()
    outfile = os.path.join(outfolder, f'spectrogram-{suffix}.png')
    print("Writing centroid plot to", outfile)
    ax.get_figure().savefig(outfile)

centroid("../notebooks/snd/istambul2.flac")
