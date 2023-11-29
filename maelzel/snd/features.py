from __future__ import annotations
import numpy as np
import bpf4
import bpf4.core
from emlib.numpytools import chunks
from pitchtools import db2amp, amp2db

from maelzel.snd import _common
from maelzel.snd.numpysnd import numChannels, rmsbpf

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import csoundengine


def onsetsAubio(samples: np.ndarray,
                sr: int,
                method='mkl',
                winsize=1024,
                hopsize=512,
                threshold=0.03,
                mingap=0.050,
                silencedb=-70
                ) -> list[float]:
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
        a list of floats, representing the times of the onsets
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


def playTicks(times: list[float] | np.ndarray,
              engine: csoundengine.Engine | None = None,
              chan=1,
              midinote: int|float|list[float] = 69,
              amp=0.5,
              attack=0.01,
              decay=0.05,
              sustain=0.5,
              release=0.100,
              extraLatency=0.
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
        engine = _common.getEngine()

    session = engine.session()
    instr = session.defInstr("features.tick", body=r"""
        |iPitch, iAmp, iAtt, iDec, iSust, iRel, iChan|
        iFreq mtof iPitch
        a0 vco2 iAmp, iFreq, 12  ; triangular shape
        aenv adsr iAtt, iDec, iSust, iRel
        outch iChan, a0*aenv
        """, priority=1)
    dur = attack + decay + release
    if isinstance(midinote, (int, float)):
        midinotes = [midinote] * len(times)
    else:
        midinotes = midinote
    with engine.lockedClock():
        synths = []
        for time, pitch in zip(times, midinotes):
            args = dict(iPitch=pitch, iAmp=amp, iAtt=attack, iDec=decay,
                        iSust=sustain, iRel=release, iChan=chan)
            synths.append(session.sched(instr.name, delay=time+extraLatency, dur=dur, args=args))
    return csoundengine.SynthGroup(synths)


def onsets(samples: np.ndarray,
           sr: int,
           winsize=2048,
           hopsize=512,
           threshold=0.07,
           mingap=0.050,
           backtrack=False,
           ) -> tuple[np.ndarray, bpf4.core.BpfBase]:
    """
    Detect onsets

    This is based on `rosita`, a minimal version of librosa with some fixes
    and simplifications (it avoids having to add numba as a dependency)

    The onset detection algorithm uses the variation in mel spectrum to calculate
    an onset strength in time. Peaks above the given threshold are detected
    as onsets.

    Args:
        samples: mono samples
        sr: sr
        winsize: the size of the fft window
        hopsize: samples to skip between windows
        threshold: the delta needed to trigger an onset
        mingap: min. time gap between onsets

    Returns:
        a tuple (onset array, onset strength bpf)
    """
    if (n := numChannels(samples)) != 1:
        raise ValueError(f"Only mono samples are accepted, but got {n} channels of audio")

    from maelzel.snd import rosita
    env = rosita.onset_strength(y=samples, sr=sr, hop_length=hopsize, n_fft=winsize)
    envtimes = rosita.times_like(env, sr=sr, hop_length=hopsize, n_fft=winsize)
    onsets = rosita.onset_detect(samples, sr, onset_envelope=env, hop_length=hopsize,
                                 units='time', delta=threshold, mingap=mingap, n_fft=winsize,
                                 backtrack=backtrack)
    onsetbpf = bpf4.core.Linear(envtimes, env)
    return onsets, onsetbpf


def plotOnsets(samples: np.ndarray,
               sr: int,
               onsets: np.ndarray,
               onsetbpf: bpf4.core.BpfBase = None,
               samplesgain=20,
               envalpha=0.8,
               samplesalpha=0.4,
               onsetsalpha=0.3,
               figsize: tuple[int, int] | None = None,
               offsets: np.ndarray | list[float] = None
               ) -> None:
    """
    Plot the results of onsets detection

    Args:
        samples: the samples from which onsets were detected
        sr: the sr of samples
        onsets: the onsets as returned via onsetsRosita
        onsetbpf: the onsetbpf as returned via onsetsRosita
        samplesgain: a gain to apply to the samples for plotting
        envalpha: alpha channel for onsets strength
        samplesalpha: alpha channel for samples plot
        onsetsalpha: alpha channel for onsets
        offsets: if given, a region is plotted instead of a line. An offset of
            0 indicates that the given onset has no offset and in this case
            also a line will be plotted

    Example
    -------

        >>> from maelzel.snd.audiosample import Sample
        >>> from maelzel.snd import features
        >>> s = Sample("/path/to/sndfile.wav").getChannel(0)
        >>> onsets, onsetstrength = features.onsets(s.samples, s.sr)
        >>> features.plotOnsets(samples=s.samples, sr=s.sr, onsets=onsets,
        ...                     onsetbpf=onsetstrength)


    """
    import matplotlib.pyplot as plt
    if figsize:
        plt.figure(figsize=figsize)
    if onsetbpf:
        xs, ys = onsetbpf.points()
        plt.plot(xs, ys, alpha=envalpha)
    duration = len(samples) / sr
    plt.plot(np.arange(0, duration, 1 / sr), samples ** 2 * samplesgain, alpha=samplesalpha, linewidth=1)
    if offsets:
        for onset, offset in zip(onsets, offsets):
            if offset > 0:
                plt.axvspan(xmin=onset, xmax=offset, alpha=onsetsalpha)
            else:
                plt.axvline(x=onset, ymin=0, alpha=onsetsalpha, linewidth=1)

    else:
        for onset in onsets:
            plt.axvline(x=onset, alpha=onsetsalpha, linewidth=1, ymin=0)


def filterOnsets(onsets: np.ndarray,
                 samples: np.ndarray,
                 sr: int,
                 minampdb = -60,
                 rmscurve: bpf4.core.BpfInterface = None,
                 rmsperiod = 0.05,
                 onsetStrengthBpf: bpf4.core.BpfInterface = None,
                 ) -> np.ndarray:
    """
    Returns a selection array where a value of 1 marks an onset as relevant

    The returned array can be used to remove superfluous onsets, based on
    secondary features (rms, ...)

    Args:
        onsets: the list of onsets
        samples: the samples for which these onsets where calculated
        sr: the sample rate of the samples
        minampdb: the min. amptliude of audio in order for an onset to be valid (in dB)
        rmsperiod: the period in seconds to use for calculating the RMS
        rmscurve: an rms curve can be given if it has been already calculated.
        onsetStrengthBpf: the onset strength as returned by the :func:`onsets` function

    Returns:
        a tuple (onsets selection array, rmscurve), where the array is of the same size
        of *onsets*. For each onset a value of 1 marks the onset as relevant, a value
        of 0 indicates that the onset might be superfluous. During the filtering a rms
        curve is calculated. This curve is returned to the user, who might use it
        later (for example, to calculate the offsets via :func:`findOffsets`

    Example
    ~~~~~~~

    TODO!!!

    """
    sel = np.ones_like(onsets, dtype=bool)
    if onsetStrengthBpf:
        before = onsets - 0.05
        after = onsets + 0.15
        sel *= onsetStrengthBpf.map(after) - onsetStrengthBpf.map(before) > -0.1

    if rmsperiod > 0:
        if rmscurve is None:
            rmscurve = rmsbpf(samples, sr=sr, dt=rmsperiod, overlap=2)
        rmsdelay = rmsperiod * 2
        sel *= rmscurve.map(onsets + rmsdelay) > db2amp(minampdb)
    return sel


def findOffsets(onsets: list[float] | np.ndarray,
                samples: np.ndarray,
                sr: int,
                rmscurve: bpf4.core.BpfInterface = None,
                silenceThreshold=-60,
                relativeThreshold: int = 90,
                rmsperiod=0.05,
                notfoundValue=-1
                ) -> list[float]:
    """For each onset find its corresponding offset

    If no offset is found before the next onset, the corresponding offset
    time is set to be *notfoundValue*

    Args:
        onsets: the onset times
        samples: the samples for which the onsets where calculated
        sr: the samplerate of the samples
        silenceThreshold: silence threshold in dB
        relativeThreshold: if the sound falls this amount of dB relative to the onset
            then it is also considered an offset (possitive dB)
        notfoundValue: the value used to indicate that no offset was found for a given
            onset, indicating that a new onset was found before the previous onset
            was allowed to decay into silence

    Returns:
        a list of offsets, one for each onset given. An offset is -1 if a new onset
        is found before any silence is found
    """
    if rmscurve is None:
        rmscurve = rmsbpf(samples, sr, dt=rmsperiod, overlap=2)
    assert rmscurve is not None
    end = len(samples) / sr
    offsets = []
    lasti = len(onsets) - 1
    for i, onset in enumerate(onsets):
        nextonset = onsets[i + 1] if i < lasti else end
        rmsfragm = rmscurve[onset:nextonset]
        threshdb = max(amp2db(rmscurve(onset)) - relativeThreshold, silenceThreshold)
        thresh = db2amp(threshdb)
        intersect = rmsfragm - thresh
        zeros = intersect.zeros(maxzeros=1)
        if zeros:
            zero = zeros[0]
            assert onset < zero < nextonset, f'{onset=}, {zero=}, {nextonset=}'
        else:
            zero = notfoundValue
        offsets.append(zero)
    for i in range(len(onsets)-1):
        assert offsets[i] < 0 or (onsets[i] < offsets[i] < onsets[i+1]), f"{i=}, {onsets[i]=}, {offsets[i]=}, {offsets[i+1]=}"
    return offsets


def centroidbpf(samples: np.ndarray,
                sr: int,
                fftsize: int = 2048,
                overlap: int = 4,
                winsize: int | None = None,
                window='hann'
                ) -> bpf4.core.Sampled:
    """
    Construct a bpf representing the centroid of the given audio over time

    Args:
        samples: a 1D numpy array representing a mono audio fragment
        sr: the sampling rate
        fftsize: the fft size
        overlap: amount of overlap
        winsize: the size of the window. If not given then winsize is assumed to be
            the same as fftsize. if given it must be <= fftsize
        window: kind of window

    Returns:
        a bpf representing the centroid over time

    """
    from maelzel.snd import rosita
    if len(samples.shape) > 1:
        raise ValueError("Only mono samples are supported")
    winsize = winsize or fftsize
    hopsize = winsize // overlap
    frames = rosita.spectral_centroid(y=samples,
                                      sr=sr,
                                      n_fft=fftsize,
                                      hop_length=hopsize,
                                      win_length=winsize,
                                      window=window)
    return bpf4.core.Sampled(frames[0], x0=0, dx=hopsize/sr)
