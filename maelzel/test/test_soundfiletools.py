from maelzel.snd import sndfiletools
import matplotlib.pyplot as plt
import time


def peakbpf():
    bpf = sndfiletools.peakbpf("snd/finneganswake.wav", channel=1, resolution=0.01)
    t, maxpeak = sndfiletools.maxPeak("snd/finneganswake.wav")
    print(f"Max. peak: {maxpeak} at {t} seconds")
    bpf.plot()
    


    
if __name__ == "__main__":
    peakbpf()
    plt.show()

