from __future__ import annotations
import os
import numpy as np
from maelzel.snd import numpysnd

import typing as _t

_removeSustainInstr = r'''
|ionsettab, iaudiotab, isr, ichan=1, iwet=1, ifftsize=2048, iwinsize=2048, iwintype=0, ioverlap=8, ifreezemargin=0.1, imorphtime=0.2, ireduction=1, irelthreshold=0.01|
inumsamps = ftlen(iaudiotab)
idur = inumsamps / isr
p3 = min(p3, idur)
asig poscil3 1, isr/inumsamps, iaudiotab
ihopsize = ifftsize / ioverlap
fsig1 pvsanal asig, ifftsize, ihopsize, iwinsize, iwintype
i_overlap, i_numbins, i_winsize, i_format pvsinfo fsig1

kfreqs0[] init i_numbins
kmags0[]  init i_numbins
kmags1[]  init i_numbins
kmags2[]  init i_numbins
kmags3[]  init i_numbins
kmags4[]  init i_numbins
kmagstest[] init i_numbins

ionsets[] tab2array ionsettab
kindex = floor:k(bisect(eventtime(), ionsets))
konset = changed(kindex)
kfreezetrig delayk konset, ifreezemargin
kframe pvs2array kmags0, kfreqs0, fsig1

if kfreezetrig > 0 then
  kmags1 = kmags0
endif

if konset > 0 then
  kmags3 = kmags2
  kmags2 = kmags1
endif

kenv = linenv(konset, 0, 0, imorphtime, 1)
kmags4 = linlin(kenv, kmags3, kmags2)
kmagstest = kmags0 - kmags4 * ireduction
kmagstest limit kmagstest, 0, 1
if sumarray(kmagstest) / sumarray(kmags0) > irelthreshold then
    kmags0 = kmagstest
endif

fsig3 pvsfromarray kmags0, kfreqs0, ihopsize, iwinsize, 0
aout pvsynth fsig3
if iwet < 1 then
    aout = aout * iwet + (1-iwet) * delay:a(asig, iwinsize/sr)
endif
outch ichan, aout
'''


def removeSustain(samples: np.ndarray,
                  sr: int,
                  fftsize=2048,
                  overlap=8,
                  wintype='hamming',
                  winsize: int = 0,
                  onsets: _t.Sequence[float] | np.ndarray | None = None,
                  transientMargin=0.1,
                  morphTime=0.15,
                  reductionFactor=1.0,
                  backend='csound',
                  wet=1.,
                  verbose=False,
                  outfile='',
                  onsetThreshold=0.07,
                  onsetFFTsize=2048,
                  onsetHopsize=256,
                  csoundKsmps=64,
                  realign=True,
                  tail=0.
                  ) -> np.ndarray:
    """
    Remove the sustain of notes

    After each onset a snapshot of the audio is taken. Within the next
    onset the previous snapshot is removed from the current audio,
    reducing any resonance/sustain of the previous note into the new
    note. This is helpful for piano sounds or in highly reverberated
    recordings to help with fundamental tracking

    Args:
        samples: audio samples
        sr: sample rate
        fftsize: fft size
        overlap: overlap factor
        wintype: window type, one of 'hamming', 'hann', kaiser'
        winsize: window size, defaults to fftsize. If given, must be bigger than fftsize
        onsets: if given, these times are used instead of calculating onsets here
        transientMargin: a time margin after the onset when it is assumed that the signal
            is stable, without the inhamonicity of attack transients
        morphTime: morphing time between previous and new spectral template.
        reductionFactor: how much to reduce. Normally a value between 0-1, but
            negative values and values above 1 can also be used
        backend: at the moment, only 'csound'
        onsetThreshold: used when calculating onsets
        wet: determines the mix of wet and dry signals. A value of 1. outputs only
            the wet signal. In general the output signal will be delayed by winsize/sr
        verbose: show debugging information
        outfile: if given, the audio samples are written to this outfile
        onsetFFTsize: size of the fft used for detecting offsets. Only valid if no onsets
            are passed
        onsetHopsize: hopsize used for detecting offsets (overlap = onsetFFTsize // onsetHopsize)
        csoundKsmps: ksmps used for rendering when using the csound backend
        realign: if True, remove the time delay introduced by the fft analysis
        tail: extra render time at the end.

    Returns:
        the modified samples
    """
    if backend != 'csound':
        raise ValueError(f"At the moment only 'csound' is supported, got '{backend}'")

    wintypeNumber = {
        'hamming': 0,
        'hann': 1,
        'kaiser': 2
    }.get(wintype)
    if wintypeNumber is None:
        raise ValueError(f"Expected one of 'hamming', 'hann', 'kaiser', got '{wintype}'")

    if not isinstance(overlap, int) and overlap >= 1:
        raise ValueError(f"Expected an integer, got {overlap}")

    if onsets is None:
        from maelzel.snd import features
        onsets, onsetStrength = features.onsets(samples=samples,
                                                sr=sr,
                                                winsize=onsetFFTsize,
                                                hopsize=onsetHopsize,
                                                threshold=onsetThreshold,
                                                backtrack=True)
    else:
        onsets = np.asarray(onsets)

    if onsets[0] > 0.001:
        onsets = np.insert(onsets, 0, 0.)

    from csoundengine.offline import OfflineSession
    nchnls = numpysnd.numChannels(samples)
    renderer = OfflineSession(sr=sr, nchnls=nchnls, ksmps=csoundKsmps, withBusSupport=False)
    channelTables = []
    for n in range(nchnls):
        chan = numpysnd.getChannel(samples, n)
        table = renderer.makeTable(chan, sr=sr)
        channelTables.append(table)

    onsetTable = renderer.makeTable(data=onsets)

    renderer.defInstr('deverb', _removeSustainInstr)
    sampledur = len(samples) / sr
    if not winsize:
        winsize = fftsize
    elif winsize < fftsize:
        raise ValueError(f"The window size should be at least as big as the fft size,"
                         f"got {winsize=}, {fftsize=}")
    for i, channeltab in enumerate(channelTables):
        renderer.sched('deverb', 0, dur=sampledur,
                       args=dict(ionsettab=onsetTable.tabnum,
                                 iaudiotab=channeltab.tabnum,
                                 isr=sr,
                                 ichan=i+1,
                                 iwet=wet,
                                 ifftsize=fftsize,
                                 iwinsize=winsize,
                                 iwintype=wintypeNumber,
                                 ioverlap=overlap,
                                 ifreezemargin=transientMargin,
                                 imorphtime=morphTime,
                                 ireduction=reductionFactor)
                       )
    job = renderer.render(outfile=outfile, verbose=verbose, tail=tail, wait=True)
    if not os.path.exists(job.outfile):
        args = job.process.args if job.process else None
        raise RuntimeError(f"Could not generate output file '{job.outfile}', file not found. "
                           f"Args used: {args}")
    import sndfileio
    outsamples, _ = sndfileio.sndread(job.outfile)
    if realign:
        outsamples = outsamples[fftsize:]
        if outfile:
            # We were asked to save to a file, so save the realigned audio
            sndfileio.sndwrite(outfile, outsamples, sr=sr)
    return outsamples
