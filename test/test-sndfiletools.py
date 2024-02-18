from maelzel.snd import sndfiletools
import pitchtools as pt
import matplotlib.pyplot as plt
import time


def peakbpf():
    peaks = sndfiletools.peakbpf("snd/finneganswake.wav", channel=1, resolution=0.05)
    t, maxpeak = sndfiletools.maxPeak("snd/finneganswake.wav")
    print(f"Max. peak: {maxpeak} at {t} seconds")
    for t, peak in zip(*peaks.points()):
        db = pt.amp2db(peak)
        print(f"{t=:.3f}, {peak=:.4f} ({db:.1f} dB)")
    peaks.plot()
    

if __name__ == "__main__":
    peakbpf()
    plt.show()

