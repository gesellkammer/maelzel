from __future__ import annotations
import os
import itertools
import numpy as np
from maelzel.snd import numpysnd


_spectralFilterTable = r'''
| iaudiotab, isr, ipairstab, ichan=1, iwet=1, ifftsize=2048, 
  iwinsize=2048, iwintype=0, ioverlap=8 |  
inumsamps = ftlen(iaudiotab)
p3 = min(p3, inumsamps / isr)
ihopsize = ifftsize / ioverlap
ifilterfreqs0[] tab2array ipairstab, 0, 0, 2
ifiltergains0[] tab2array ipairstab, 1, 0, 2

asig poscil3 1, isr/inumsamps, iaudiotab

fsig1 pvsanal asig, ifftsize, ihopsize, iwinsize, iwintype
i__overlap, inumbins, i__wsize, i__fmt pvsinfo fsig1

kmags0[]  init inumbins
kfreqs0[] init inumbins
kfilteridxs[]  init inumbins
kfiltergains[] init inumbins

kframe pvs2array kmags0, kfreqs0, fsig1
if changed(kframe) == 1 then
  ; only do work when a new fft frame is available
  kfilteridxs bisect kfreqs0, ifilterfreqs0
  kfiltergains interp1d kfilteridxs, ifiltergains0, "linear"
  kmags0 *= kfiltergains
  kmags0 limit kmags0, 0, 1
endif
fsig2 pvsfromarray kmags0, kfreqs0, ihopsize, iwinsize, iwintype
aout pvsynth fsig2
aout = aout * iwet + (1 - iwet) * delay:a(asig, iwinsize/sr)
outch ichan, aout
'''


def spectralFilter(samples: np.ndarray,
                   sr: int,
                   pairs: list[float] | tuple[list[float], list[float]],
                   fftsize=2048,
                   overlap=8,
                   winsize: int = 0,
                   wintype='hamming',
                   realign=True,
                   wet=1.,
                   tail=0.,
                   outfile='',
                   verbose=False
                   ) -> np.ndarray:
    """
    Filter the given audios by applying a spectral envelope

    Args:
        samples: audio samples
        sr: samplerate
        pairs: either a flat list of pairs of the form [freq_0, gain_0, ..., freq_n, gain_n]
            or a tuple (freqs, gains)
        fftsize: fft size
        overlap: overlap factor, hopsize = winsize // overlap
        winsize: window size. If given, must be >= fftsize. If not given,
            fftsize is used
        wintype: window type, one of 'hamming', 'hann', kaiser'
        wet: determines the mix of wet and dry signals. A value of 1. outputs only
            the wet signal. In general the output signal will be delayed by winsize/sr
        realign: shift the audio to compensate for the delay in the spectral
            processing
        tail: extra render time at the end.
        outfile: if given, the audio samples are written to this outfile
        verbose: output debugging information

    Returns:
        the resulting samples
    """
    sampledur = len(samples) / sr
    if not winsize:
        winsize = fftsize
    elif winsize < fftsize:
        raise ValueError(f"The window size should be at least as big as the fft size,"
                         f"got {winsize=}, {fftsize=}")

    if isinstance(pairs, list) and isinstance(pairs[0], (int, float)):
        # flat pairs
        flatpairs = pairs
    else:
        assert isinstance(pairs, tuple) and len(pairs) == 2
        freqs, gains = pairs
        flatpairs = list(itertools.chain(*zip(freqs, gains)))

    wintypeNumber = {
        'hamming': 0,
        'hann': 1,
        'kaiser': 2
    }.get(wintype)

    if wintypeNumber is None:
        raise ValueError(f"Expected one of 'hamming', 'hann', 'kaiser', got '{wintype}'")

    if not isinstance(overlap, int) and overlap >= 1:
        raise ValueError(f"Expected an integer, got {overlap}")

    nchnls = numpysnd.numChannels(samples)

    from csoundengine.offline import OfflineSession
    renderer = OfflineSession(sr=sr, nchnls=nchnls, ksmps=64)

    channelTables = []
    for n in range(nchnls):
        chan = numpysnd.getChannel(samples, n)
        table = renderer.makeTable(chan, sr=sr)
        channelTables.append(table)

    pairstable = renderer.makeTable(data=flatpairs)

    renderer.defInstr('spectralFilterTable', _spectralFilterTable)
    for i, channeltab in enumerate(channelTables):
        renderer.sched('spectralFilterTable', 0, dur=sampledur,
                       args=dict(iaudiotab=channeltab.tabnum,
                                 isr=sr,
                                 ipairstab=pairstable.tabnum,
                                 ichan=i+1,
                                 iwet=wet,
                                 ifftsize=fftsize,
                                 iwinsize=winsize,
                                 ioverlap=overlap,
                                 iwintype=wintypeNumber)
                       )
    job = renderer.render(outfile=outfile, verbose=verbose, tail=tail, wait=True)
    if not os.path.exists(job.outfile):
        raise RuntimeError(f"Could not generate output file '{job.outfile}', file not found. "
                           f"Args used: {job.process.args}")
    import sndfileio
    outsamples, _ = sndfileio.sndread(job.outfile)
    if realign:
        outsamples = outsamples[fftsize:]
        if outfile:
            # We were asked to save to a file, so save the realigned audio
            sndfileio.sndwrite(outfile, outsamples, sr=sr)
    return outsamples
