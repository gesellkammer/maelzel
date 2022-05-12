from __future__ import annotations
import numpy as np
import bpf4 as bpf
from emlib.numpytools import chunks
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    import csoundengine


def onsetsAubio(samples: np.ndarray, sr:int, method='mkl', winsize=1024,
                hopsize=512, threshold=0.03, mingap=0.050, silencedb=-70
                ) -> List[float]:
    """
    Detect onsets in samples

    Args:
        samples: the samples, as numpy array (1D), between -1 and 1
        sr: the sample rate of samples
        winsize: the size of the fft window size, in samples
        hopsize: the hop size, in samples
        threshold: depends on the method. The lower this value, the more probable
            is it that an onset is detected
        method: the method to detect onsets. One of:
            - `energy`: local energy,
            - `hfc`: high frequency content,
            - `complex`: complex domain,
            - `phase`: phase-based method,
            - `wphase`: weighted phase deviation,
            - `specdiff`: spectral difference,
            - `kl`: Kullback-Liebler,
            - `mkl`: modified Kullback-Liebler,
            - `specflux`: spectral flux.
        mingap: the min. amount of time (in seconds) between two onsets
        silencedb: onsets will only be detected if the amplitude exceeds this value (in dB)

    Returns:
        a list of floats, representing the times of the offsets
    """
    assert isinstance(samples, np.ndarray) and len(samples.shape) == 1
    try:
        import aubio
    except ImportError:
        raise ImportError("aubio (https://github.com/aubio/aubio) is needed for this "
                          "functionality")
    ao = aubio.onset(method, buf_size=winsize, hop_size=hopsize)
    ao.set_threshold(threshold)
    ao.set_silence(silencedb)
    ao.set_minioi_s(mingap)
    samples = samples.astype('float32')
    onsets = [ao.get_last()/sr for chunk in chunks(samples, hopsize, padwith=0.0)
              if ao(chunk)]
    return onsets


def playTicks(times: List[float], engine: csoundengine.Engine = None, chan=1,
              midinote:Union[int, float, List[float]] = 69,
              amp=0.5, attack=0.01, decay=0.05, sustain=0.5, release=0.100, latency=0.2
              ) -> csoundengine.synth.SynthGroup:
    """
    Given a list of times offsets, play these as ticks

    Args:
        times: a list of time offsets
        chan: which channel to play the ticks to
        midinote: the pitch of the ticks or a list of pitches, one for each tick
        amp: the amplitude of the ticks
        attack: attack duration
        decay: decay duration
        sustain: sustain amplitude
        release: release dur.
        latency: extra delay added to each tick, to account for the
            initial latency product of scheduling a large amount of events

    Returns:
        a csoundengine.SynthGroup, which constrols playback.

    Examples
    ~~~~~~~~

        >>> import sndfileio
        >>> samples, info = sndfileio.sndget("/path/to/sound.wav")
        >>> onsets = onsetsAubio(samples, info.sr)
        >>> synthgroup = playTicks(onsets)
        # if needed to stop the playback at any moment:
        >>> synthgroup.stop()
    """
    import csoundengine
    if engine is None:
        engine = csoundengine.getEngine("maelzel.snd")
    session = engine.session()
    instr = session.defInstr("features.tick", body=r"""
        iPitch, iAmp, iAtt, iDec, iSust, iRel, iChan passign 5
        iFreq mtof iPitch
        a0 vco2 iAmp, iFreq, 12  ; triangular shape
        aenv adsr iAtt, iDec, iSust, iRel
        outch iChan, a0*aenv
        """)
    dur = attack + decay + release
    args = [midinote, amp, attack, decay, sustain, release, chan]
    if isinstance(midinote, (int, float)):
        midinotes = [midinote] * len(times)
    else:
        midinotes = midinote

    synths = []
    with engine.lockedClock():
        for t, m in zip(times, midinotes):
            args[0] = m
            synths.append(session.sched(instr.name, delay=t+latency, dur=dur, pargs=args))
    return csoundengine.synth.SynthGroup(synths)


def onsets(samples: np.ndarray, sr: int, winsize=2048,
           hopsize=512, threshold=0.07, mingap=0.050
           ) -> tuple[np.ndarray, bpf.BpfInterface]:
    """
    Detect onsets

    This is based on `rosita`, a minimal version of librosa with some fixes
    and simplifications (it avoids having to add numba as a dependency)

    The onset detection algorithm uses the variation in mel spectrum to calculate
    an onset strength in time. Peaks above the given threshold are detected
    as onsets.

    Args:
        samples: mono samples
        sr: samplerate
        winsize: the size of the fft window
        hopsize: samples to skip between windows
        threshold: the delta needed to trigger an onset
        mingap: min. time gap between onsets

    Returns:
        a tuple (onset array, onset strength bpf)
    """
    from maelzel.snd import rosita
    env = rosita.onset_strength(y=samples, sr=sr, hop_length=hopsize, n_fft=winsize)
    envtimes = rosita.times_like(env, sr=sr, hop_length=hopsize, n_fft=winsize)
    onsets = rosita.onset_detect(samples, sr, onset_envelope=env, hop_length=hopsize,
                                 units='time', delta=threshold, mingap=mingap, n_fft=winsize,
                                 backtrack=False)
    onsetbpf = bpf.core.Linear(envtimes, env)
    return onsets, onsetbpf


def plotOnsets(samples: np.ndarray, sr: int, onsets: np.ndarray,
               onsetbpf: bpf.BpfInterface = None, samplesgain=20,
               envalpha=0.8, samplesalpha=0.4, onsetsalpha=0.3):
    """
    Plot the results of onsets detection

    Args:
        samples: the samples from which onsets were detected
        sr: the samplerate of samples
        onsets: the onsets as returned via onsetsRosita
        onsetbpf: the onsetbpf as returned via onsetsRosita
        samplesgain: a gain to apply to the samples for plotting
        envalpha: alpha channel for onsets strength
        samplesalpha: alpha channel for samples plot
        onsetsalpha: alpha channel for onsets

    Example
    -------

        >>> from maelzel.snd.audiosample import Sample
        >>> from maelzel.snd import features
        >>> s = Sample("/path/to/sndfile.wav").getChannel(0)
        >>> onsets, onsetstrength = features.onsets(s.samples, s.sr)
        >>> import csoundengine
        >>> e = csoundengine.Engine()
        >>> with e.lockedClock():
        ...     s.play(engine=e)
        ...     playTicks(times=onsets, engine=e)
        >>> features.plotOnsetsRosita(samples=s.samples, sr=s.sr, onsets=onsets,
        ...                           onsetbpf=onsetstrength)


    """
    import matplotlib.pyplot as plt
    if onsetbpf:
        xs, ys = onsetbpf.points()
        plt.plot(xs, ys, alpha=envalpha)
    duration = len(samples) / sr
    plt.plot(np.arange(0, duration, 1 / sr), samples ** 2 * samplesgain, alpha=samplesalpha, linewidth=1)
    for onset in onsets:
        plt.axvline(x=onset, alpha=onsetsalpha, linewidth=1)