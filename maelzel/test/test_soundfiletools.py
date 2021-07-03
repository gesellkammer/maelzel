from maelzel.snd import _sndfiletools_old
import matplotlib.pyplot as plt
import time


def peakbpf():
    bpf = _sndfiletools_old.peakbpf("snd/finneganswake.wav", channel=1, resolution=0.01)
    t, maxpeak = _sndfiletools_old.maxPeak("snd/finneganswake.wav")
    print(f"Max. peak: {maxpeak} at {t} seconds")
    bpf.plot()
    


    
if __name__ == "__main__":
    peakbpf()
    plt.show()

