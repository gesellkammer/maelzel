from __future__ import annotations
import numpy as np
from emlib.numpytools import chunks
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    import csoundengine


def onsets_aubio(samples: np.ndarray, sr:int, method='mkl', winsize=1024,
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
        raise ImportError("aubio is needed for this functionality. Install it"
                          "via ")
    ao = aubio.onset(method, buf_size=winsize, hop_size=hopsize)
    ao.set_threshold(threshold)
    ao.set_silence(silencedb)
    ao.set_minioi_s(mingap)
    samples = samples.astype('float32')
    onsets = [ao.get_last()/sr for chunk in chunks(samples, hopsize, padwith=0.0)
              if ao(chunk)]
    return onsets


def play_ticks(times, chan=1, midinote=69, amp=0.5, attack=0.01, decay=0.05,
               sustain=0.5, release=0.100, latency=0.2
               ) -> csoundengine.synth.SynthGroup:
    """
    Given a list of times offsets, play these as ticks

    Args:
        times: a list of time offsets
        chan: which channel to play the ticks to
        midinote: the pitch of the ticks
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
        >>> onsets = onsets_aubio(samples, info.sr)
        >>> synthgroup = play_ticks(onsets)
        # if needed to stop the playback at any moment:
        >>> synthgroup.stop()
    """
    import csoundengine
    session = csoundengine.Engine().session()
    instr = session.defInstr("features.tick", body=r"""
        iPitch, iAmp, iAtt, iDec, iSust, iRel, iChan passign 5
        iFreq mtof iPitch
        a0  oscil iAmp, iFreq
        aenv adsr iAtt, iDec, iSust, iRel
        outch iChan, a0*aenv
        """)
    dur = attack + decay + release
    args = [midinote, amp, attack, decay, sustain, release, chan]
    synths = [session.schedEvent(instr.name, delay=t+latency, dur=dur, args=args)
              for t in times]
    return csoundengine.synth.SynthGroup(synths)