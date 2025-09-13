"""
Monophonic fundamental analyisis and transcription
"""
from __future__ import annotations
import numpy as np
from dataclasses import astuple as _astuple

from maelzel.snd import features
from maelzel.snd import freqestimate
from maelzel.snd import audiosample
from maelzel.snd.numpysnd import rmsBpf
from pitchtools import PitchConverter, db2amp
from math import isnan
from emlib import iterlib

from maelzel.scorestruct import ScoreStruct
from maelzel import stats

from .breakpoint import Breakpoint, BreakpointGroup, simplifyBreakpoints, simplifyBreakpointsByDensity
from .options import TranscriptionOptions

from typing import TYPE_CHECKING, Iterator
if TYPE_CHECKING:
    from maelzel.core import Voice
    from matplotlib.axes import Axes


__all__ = (
    'FundamentalAnalysisMonophonic',
)


def _fixNanFreqs(breakpoints: list[tuple[float, float, float]], fallbackFreq: float
                 ) -> list[tuple[float, float, float]]:
    # find the nearest freq and set it as negative to indicate unvoiced
    freq = next((bp[1] for bp in breakpoints if not isnan(bp[1])), None)
    if freq is None:
        # no freqs found in the breakpoints
        return [(t, fallbackFreq, a) for t, f, a in breakpoints]
    out = []
    for bp in breakpoints:
        if isnan(bp[1]):
            bp = (bp[0], -freq, bp[2])
        else:
            freq = bp[1]
        out.append(bp)
    return out


def _quantizeFreq(pitchconv: PitchConverter, freq: float, divs: int) -> float:
    if freq == 0:
        return 0
    elif freq > 0:
        return pitchconv.m2f(round(pitchconv.f2m(freq) * divs) / divs)
    else:
        return -pitchconv.m2f(round(pitchconv.f2m(-freq) * divs) / divs)


class FundamentalAnalysisMonophonic:
    """
    Perform monophonic f0 analysis (one source, one sound at a a time)

    Algorithm: the audio is split in fragments via onset/offset detection. Each of
    these fragments is split in groups of breakpoints, representing the pitch/amplitude
    variation of the fundamental within the given time range. Each breakpoint also holds
    other features of the audio, like its voicedness (a low value is an indication of noisy
    sound without enough harmonic content to determine the pitch). The analysis includes
    other parameters as global curves (rms, spectral centroid) which can be used for
    transcription

    Args:
        samples: the samples to analyze
        sr: the sampling rate
        onsetMinDb: the min. amplitude in dB for an onset to be considered
        rmsPeriod: the period over which to calculate an rms curve of the samples.
        fftSize: the window size used for analysis
        overlap: the window overlap used (hopsize = windowsize / overlap)

        minFrequency: the min. frequency used when detecting the fundamental pitch over time.
            If 0, perform a detection pass to estimate the lowest fundamental frequency of the
            sample
        minSilence: shortest silence allowed. Any silence shorter than this will be absorved
            by the previous fragment
        simplify: the amount of simplification. The default simplification used is Visvalingam-Wyatt,
            in which case this value represents the area threshold measured in semitones.
            The higher this value, the simpler the resulting curve. NB: breakpoints are
            only simplified within an onset-offset pair. Onsets are not simplified. To reduce
            the number of onsets you can increase the onsetThreshold
        semitoneQuantization: if given, breakpoints are quantized to the next fraction
            of the semitone given. A value of 2 will quantize all breakpoints to the nearest
            quarter tone. It might be necessary to also customize the A4 reference freq
        onsetThreshold: the threshold used for detecting onsets. The higher the value, the more
            prominent an onset must be in order to be detected.
        onsetOverlap: overlap for onset analysis.
        onsetBacktrack: if True, backtrack the onset to the nearest previous minimum
        accentPercentile: any onset with a strength higher than this percentile will be
            marked as accent. This is an indication that a certain onset is above a certain
            average strength.
        simplificationMethod: A note takes place within an onset-offset time range. The
            fundamental is sampled regularly within this time range and the resulting breakpoints
            are simplified in terms of their pitch. This parameter indicates the simplification
            method used (one of 'visvalingam' or 'douglaspeucker'). For reference see
            https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm and
            https://en.wikipedia.org/wiki/Visvalingam%E2%80%93Whyatt_algorithm
        markLowFrequenciesAsUnvoiced: if True mark f0 measurements which lie below the given
            min. frequency as unvoiced. Under certain circumstances the f0 tracking algorithm
            might predict frequencies lower than the given min. frequency. An alternative
            way to mitigate such false predictions is to increase the overlap
        removeSustain: remove sustain via :func:`maelzel.snd.deverb.removeSustain` prior
            to analysis. This is helpful for audio signals like piano with pedal or
            long reverberation, where the sustain of one note is still audible on
            the next one, confusing the algorithm
        lowAmpSuppression: supress low amplitude pitch estimates. 0.01=-40dB, 0.001=-60dB

    Attributes:
        groups: each group is a list of Breakpoints representing a note
        rms: the RMS curve of the sound. A mpf mapping to to rms
        f0: the fundamental curve, mapping time to frequency. Parts with low confidence
            are marked as negative
        voicedness: a curve mapping time to voicedness. The highe this value the higher the
            confidence of the f0 prediction
        onsetStrength: a curve mapping time to onset strength. Onset strength is based on
            spectral flow and is not directly correlated with variations in amplitude

    """
    def __init__(self,
                 samples: np.ndarray,
                 sr: int,
                 onsetMinDb=-60,
                 rmsPeriod=0.05,
                 fftSize: int = 0,
                 overlap=8,
                 minFrequency=50,
                 minSilence=0.08,
                 simplify=0.1,
                 semitoneQuantization=0,
                 onsetThreshold=0.07,
                 onsetOverlap=8,
                 onsetBacktrack=True,
                 accentPercentile=0.1,
                 simplificationMethod='visvalingam',
                 referenceFrequency=442,
                 markLowFrequenciesAsUnvoiced=True,
                 unvoicedMinAmplitudePercentile=0,
                 lowAmpSuppression=0.01,
                 removeSustain=False,
                 ):

        if sr > 48000:
            import warnings
            warnings.warn("Analysis from audio with a samplerate higher than 48k is"
                          " not supported at the moment")

        if not minFrequency and not fftSize:
            minFrequency, _ = freqestimate.detectMinFrequency(samples, sr=sr, refine=False,
                                                              lowAmpSuppression=lowAmpSuppression)
            if minFrequency == 0:
                raise ValueError("Could not detect any pitched sound")
        if not fftSize:
            fftSize = max(2048, freqestimate.frequencyToWindowSize(int(minFrequency), sr=sr, powerof2=True))

        onsetsFftSize = min(2048 if sr <= 48000 else 4098, fftSize)
        onsets, onsetStrengthBpf = features.onsets(samples=samples,
                                                   sr=sr,
                                                   winsize=onsetsFftSize,
                                                   hopsize=onsetsFftSize // onsetOverlap,
                                                   threshold=onsetThreshold,
                                                   backtrack=onsetBacktrack)
        hopsize = fftSize // overlap
        hoptime = hopsize / sr
        # Secondary features:
        rmscurve = rmsBpf(samples, sr, dt=rmsPeriod, overlap=2)

        centroidFftsize = min(2048 if sr <= 48000 else 4098, fftSize)
        centroidcurve = features.centroidBpf(samples=samples,
                                             sr=sr,
                                             fftsize=centroidFftsize,
                                             overlap=min(8, overlap))

        pitchconv = PitchConverter(a4=referenceFrequency)
        dbs = rmscurve.amp2db().map(200)
        # dbHistogram = histogram.Histogram(dbs, numbins=numHistogramBins)
        # onsetStrengthHistogram = histogram.Histogram(onsetStrengthBpf.points()[1], numbins=numHistogramBins)
        onsetStrengthQuantile = stats.Quantile1d(onsetStrengthBpf.points()[1])
        accentStrength = onsetStrengthQuantile.value(accentPercentile)

        onsetsMask = features.filterOnsets(onsets, samples=samples, sr=sr, rmscurve=rmscurve,
                                           minampdb=onsetMinDb, rmsperiod=rmsPeriod)
        onsets = onsets[onsetsMask]
        offsets = features.findOffsets(onsets, samples=samples, sr=sr, rmscurve=rmscurve)

        if removeSustain:
            from maelzel.snd.deverb import removeSustain
            samples = removeSustain(samples=samples, sr=sr,
                                    fftsize=fftSize,
                                    overlap=overlap,
                                    onsets=onsets)

        if minFrequency > 100:
            from maelzel.snd import filters
            samples = filters.spectralFilter(samples, sr=sr, pairs=[0, 0, minFrequency, 0, minFrequency+1, 1, sr, 1])

        f0, voicedcurve = freqestimate.f0curvePyinVamp(sig=samples,
                                                       sr=sr,
                                                       fftsize=fftSize,
                                                       overlap=overlap,
                                                       unvoicedFreqs='nan',
                                                       lowAmpSuppression=lowAmpSuppression)
        groupSamplingPeriod = hoptime * 0.68
        # The min. duration of a note. Groups with a shorter duration will not be
        # split into breakpoints
        minDuration = max(groupSamplingPeriod, minSilence)

        timeCorrection = fftSize / sr

        def makeBreakpoint(t: float, freq: float, amp: float, linked: bool, kind=''
                           ) -> Breakpoint:
            strength = onsetStrengthBpf(t + timeCorrection)
            fpos = abs(freq) if not isnan(freq) else 0
            assert fpos == 0 or fpos > 20, f"{freq=}"
            return Breakpoint(t, fpos, amp,
                              voiced=bool(freq>0),
                              linked=linked,
                              onsetStrength=strength,
                              kind=kind,
                              freqConfidence=voicedcurve(t))

        # Algorithm:
        # * for each onset calculate an offset
        # * if there is an offset, sample f0 between onset and offset and simplify. Both onset and
        #   * offset should be included in the simplification
        # * if there is no offset, sample f0 between onset and next onset and simplify.
        #   * The next onset should NOT be part of the group
        #   * TODO: decide what to do for static notes
        # * In both cases, do not simplify if the duration between onset-offset (or onset and next
        #   onset) is less than minDuration. In this case make a group with just the onset breakpoint

        def makeGroup(onset: float, offset: float) -> list[Breakpoint]:
            if offset - onset < minDuration:
                return [makeBreakpoint(onset, freq=f0(onset), amp=rmscurve(onset+timeCorrection),
                                       linked=False, kind='onset')]
            numBreakpoints = round((offset - onset) / groupSamplingPeriod)
            times = np.linspace(onset, offset, numBreakpoints)
            freqs = f0.map(times)  # type: ignore
            amps = rmscurve.map(times + timeCorrection)
            breakpoints0 = list(zip(times, freqs, amps))
            breakpoints0 = _fixNanFreqs(breakpoints0, fallbackFreq=0)
            if semitoneQuantization:
                breakpoints0 = [(bp[0], _quantizeFreq(pitchconv, bp[1], semitoneQuantization), bp[2])
                                for bp in breakpoints0]
            group = [makeBreakpoint(t, f, a, linked=True)
                     for t, f, a in breakpoints0]
            first = group[0]
            first.kind = 'onset'
            first.isaccent = bool(first.onsetStrength > accentStrength)
            if simplify > 0:
                group = simplifyBreakpoints(group,
                                            method=simplificationMethod,
                                            param=simplify,
                                            pitchconv=pitchconv)
            group[-1].linked = False
            return group

        groups: list[BreakpointGroup] = []
        for i, onset, offset in zip(range(len(onsets)), onsets, offsets):
            if offset > 0:
                # There is an offset, sample f0 between onset and offset and simplify.
                # Both onset and offset should be included
                group = makeGroup(onset, offset)
                if len(group) >= 2:
                    group[-1].kind = 'offset'
            else:
                # There is no offset: sample f0 between onset and next onset (or until end of
                # the samples). Simplify between those onsets, then REMOVE the last breakpoint
                if i < len(onsets) - 1:
                    offset = float(onsets[i+1])
                else:
                    offset = len(samples)/sr
                group = makeGroup(onset, offset)

            if markLowFrequenciesAsUnvoiced:
                for bp in group:
                    if bp.freq < minFrequency:
                        bp.voiced = False

            if unvoicedMinAmplitudePercentile > 0 and all(not bp.voiced for bp in group) and group[0].getProperty('ampPercentile') is not None:
                if all(bp.getProperty('ampPercentile', 1.) < unvoicedMinAmplitudePercentile for bp in group):  # type: ignore
                    continue

            groups.append(BreakpointGroup(group))

        # TODO: minSilence - remove short silences between breakpoints within a group

        self.sample = audiosample.Sample(samples, sr=sr)
        """Sample holding the samples used for this analysis"""

        self.groups: list[BreakpointGroup] = groups
        """Each group is a list of Breakpoints representing a note"""

        self.rms = rmscurve
        """The RMS curve of the sound. A bpf mapping time to rms"""

        self.dbQuantile = stats.Quantile1d(dbs)

        self.f0 = f0
        """The f0 curve, mapping time to frequency. Parts with low confidence are marked as negative values"""

        self.voicedness = voicedcurve
        """A curve mapping time to voicedness of the sound"""

        self.onsetStrength = onsetStrengthBpf
        """A curve mapping time to onset strength"""

        self.onsetStrengthQuantile = onsetStrengthQuantile
        """Onset strenth quantile object"""

        self.centroid = centroidcurve
        """Spectral centroid over time (bpf)"""

        self._pitchconv = pitchconv

        for bp1, bp2 in iterlib.pairwise(self.flatBreakpoints()):
            bp1.duration = bp2.time - bp1.time

    def flatBreakpoints(self) -> Iterator[Breakpoint]:
        """
        Returns an iterator over all breakpoints in this analysis

        """
        for group in self.groups:
            yield from group

    def _repr_html_(self):
        import tabulate
        columnnames = Breakpoint.fields()
        rows = [_astuple(bp) for bp in self.flatBreakpoints()]
        html = tabulate.tabulate(rows, tablefmt='html', headers=columnnames, floatfmt=".4f")
        return html

    def play(self, voicedInstr='saw', unvoicedInstr='.bandnoise', unvoicedbw=0.9,
             unvoicedGain=0.):
        """
        Play the analysis as a sequence of notes.

        Args:
            voicedInstr: Instrument to use for voiced notes.
            unvoicedInstr: Instrument to use for unvoiced notes.
            unvoicedbw: Bandwidth for unvoiced notes.
            unvoicedGain: Gain for unvoiced notes.

        Returns:
            A list of notes representing the analysis.
        """
        from maelzel.core import Note, Voice, defPreset
        defPreset('.bandnoise', r'''
            |kbw=0.9|
            aout1 = beosc(kfreq, kbw) * a(kamp)
        ''')
        notes = []
        struct = ScoreStruct(tempo=60)
        for b0, b1 in iterlib.pairwise(self.flatBreakpoints()):
            assert b0.time + b0.duration <= b1.time, f"{b0=}, {b1=}"
            if b0.voiced:
                n = Note(self._pitchconv.f2m(b0.freq), offset=struct.t2b(b0.time), dur=b0.duration,
                         tied=b0.linked, gliss=b0.linked)
                n.setPlay(instr=voicedInstr)
                notes.append(n)
            elif unvoicedGain > 0:
                if b0.linked:
                    dur = b0.duration
                else:
                    dur = min(b0.duration, 0.1)
                n = Note(self._pitchconv.f2m(b0.freq), offset=struct.t2b(b0.time),
                         dur=dur, tied=b0.linked, gliss=b0.linked)
                n.setPlay(instr=unvoicedInstr, kbw=unvoicedbw, gain=unvoicedGain)
                notes.append(n)
        v = Voice(notes)
        v.play()

    def plot(self,
             linewidth=2,
             axes: Axes | None = None,
             spanAlpha=0.2,
             onsetAlpha=0.4,
             spanColor='red'
             ) -> Axes:
        """
        Plot the breakpoints of this analysis

        Args:
            linewidth: the line width used in the plot
            axes: if given, a pyplot Axes instance to plot on
            spanAlpha: the alpha value for the axvspan used to mark the
                onset-offset regions
            onsetAlpha: alpha value for onset lines
            spanColor: color for onset-offset span

        Returns:
            the axes used
        """
        if not axes:
            import matplotlib.pyplot as plt
            axes = plt.axes()

        for group in self.groups:
            group.plot(ax=axes, linewidth=linewidth, spanAlpha=spanAlpha, onsetAlpha=onsetAlpha, spanColor=spanColor)

        return axes

    def simplify(self, algorithm='visvalingam', threshold=0.05) -> None:
        """Simplify the breakpoints inside each group, inplace

        Groups themselves are never simplified

        Args:
            algorithm: at the moment only 'visvalingam' (Visvalingam-Wyatt) is supported
            threshold: the simplification parameter passed to the algorithm. For visvalingam
                this is the surface of the triangle being evaluated.

        """
        groups = [BreakpointGroup(simplifyBreakpoints(group, method=algorithm, param=threshold, pitchconv=self._pitchconv))
                  for group in self.groups]
        self.groups = groups

    def transcribe(self,
                   scorestruct: ScoreStruct | None = None,
                   options: TranscriptionOptions | None = None
                   ) -> Voice:
        """
        Convert the analyized data to a maelzel.core.Voice

        Args:
            scorestruct: the score structure to use for conversion. If not given a default
                scorestruct is used
            options: transcription options (see :class:`maelzel.transcribe.core.TranscribeOptions`)

        Returns:
            the resulting :class:`maelzel.core.Voice`

        """
        if options is None:
            options = TranscriptionOptions()
        else:
            options = options.copy()

        options.unvoicedMinAmpDb = min(self.dbQuantile.value(0.05),
                                       options.unvoicedMinAmpDb)

        voice = transcribeVoice(groups=self.groups,
                                scorestruct=scorestruct,
                                options=options)
        voice.stack()

        if scorestruct is None:
            scorestruct = ScoreStruct(tempo=60)

        if self.centroid:
            for note in voice:
                if not note.isRest():
                    centroidfreq = self.centroid(float(scorestruct.time(note.absOffset())))
                    note.setProperty('centroid', int(centroidfreq))

        return voice


def transcribeVoice(groups: list[list[Breakpoint]] | list[BreakpointGroup],
                    scorestruct: ScoreStruct | None = None,
                    options: TranscriptionOptions | None = None,
                    ) -> Voice:
    """
    Convert a list of breakpoint groups to a maelzel.core.Voice

    Args:
        groups: a list of groups, where each group is a list of breakpoints
            representing a note
        scorestruct: the score structure to use for conversion
        options: the transcription options (see :class:`maelzel.transcribe.core.TranscribeOptions`)

    Returns:
        the  resulting maelzel.core Voice

    """
    from maelzel.core import Note, Voice, Workspace
    from maelzel.core import symbols

    if scorestruct is None:
        scorestruct = Workspace.activeScoreStruct()

    if options is None:
        options = TranscriptionOptions()

    notes = []
    # lastgroupidx = len(groups) - 1
    pitchconv = PitchConverter(a4=options.a4)
    unvoicedPitch = options.unvoicedPitch
    numslurs = 0

    if options.simplify > 0:
        groups = [simplifyBreakpoints(group, param=options.simplify, pitchconv=pitchconv)
                  for group in groups]

    if options.maxDensity > 0:
        groups = [simplifyBreakpointsByDensity(group, maxdensity=options.maxDensity, pitchconv=pitchconv)
                  for group in groups]

    for groupidx, group in enumerate(groups):
        fragment = []
        assert group
        if len(group) == 1:
            bp = group[0]
            assert not bp.voiced
            notes.append(Note(unvoicedPitch, amp=bp.amp,
                              offset=scorestruct.timeToBeat(bp.time), dur=bp.duration))
            continue

        for bp1, bp2 in iterlib.pairwise(group):
            pitch = pitchconv.f2m(bp1.freq) or unvoicedPitch
            offset = scorestruct.timeToBeat(bp1.time)
            end = scorestruct.timeToBeat(bp2.time)
            note = Note(pitch, amp=bp1.amp, offset=offset, dur=end - offset, gliss=options.addGliss)
            note.setProperty('voiced', bp1.voiced)
            if not bp1.voiced and options.unvoicedNotehead:
                note.addSymbol(symbols.Notehead(options.unvoicedNotehead))
            if options.addAccents and bp1.isaccent:
                note.addSymbol('accent')
            if pitch == unvoicedPitch:
                note.setProperty('unvoicedGroup', True)
                note.gliss = False
            else:
                note.setPlay(linkednext=True)
            fragment.append(note)

        last = group[-1]
        # TODO
        # nextgroup = groups[groupidx + 1] if groupidx < lastgroupidx else None

        if last.kind == 'offset' and last.freq:
            fragment.append(Note(pitchconv.f2m(last.freq) or unvoicedPitch, dur=0, amp=last.amp))
        else:
            fragment[-1].gliss = False

        if options.addSlurs and len(fragment) > 1:
            n0, n1 = fragment[0], fragment[-1]
            n0.addSpanner('slur', n1)
            if options.debug:
                n0.addText(f'{numslurs}(')
                n1.addText(f'){numslurs}')
            numslurs += 1

        notes.extend(fragment)

    for n1, n2 in iterlib.pairwise(notes):
        if n2.properties and n2.properties.get('unvoicedGroup'):
            n1.gliss = False
            n2.gliss = False

    # Filter unvoiced groups which are too faint
    unvoicedMinAmp = db2amp(options.unvoicedMinAmpDb)
    notes = [n for n in notes
             if not n.getProperty('unvoicedGroup') or n.amp > unvoicedMinAmp]

    return Voice(notes)


# ------------------------------
