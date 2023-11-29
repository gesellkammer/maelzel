from __future__ import annotations
import numpy as np
from dataclasses import dataclass

import bpf4
import pitchtools as pt
from emlib import mathlib
from emlib import iterlib

from maelzel import histogram
from maelzel.snd import amplitudesensitivity

from . import spectrum as sp
from .partial import Partial
from .track import Track

from typing import Callable, Sequence


def _estimateMinFreq(spectrum: sp.Spectrum) -> float:
    f0, voicedness = spectrum.fundamental()
    return float(f0.map(1000).min())


def _ratePartial(track: Track, partial: Partial, maxrange: int | None = None, mingap=0.1) -> float:
    """
    The higher, the best. -1 indicates that the partial does not fit the track
    """
    maxrange = maxrange or track.maxrange
    partialPitch = partial.meanpitch()
    if not track.partials:
        margin = partial.start
        mingap = 0
        prevPitch = partialPitch

    elif track.end < partial.start:
        margin = partial.start - track.end
        if margin < mingap:
            return -1
        prevPitch = pt.f2m(track.partials[-1].meanfreq())
    elif track.start >= partial.start:
        margin = track.start - partial.end
        if margin < mingap:
            return -1
        prevPitch = pt.f2m(track.partials[0].meanfreq())
    else:
        prevPartialIdx = track.partialBefore(partial.start)
        assert prevPartialIdx is not None, f"{partial=}, partials={track.partials}, {track.start=}"
        prevPartial = track.partials[prevPartialIdx]
        if prevPartial.end > partial.start - mingap:
            return -1
        nextPartial = track.partials[prevPartialIdx  + 1]
        if nextPartial.start < partial.end + mingap:
            return -1
        margin = min(partial.start - prevPartial.end, nextPartial.start - partial.end)
        assert margin >= mingap
        prevPitch = pt.f2m(prevPartial.meanfreq())
    marginRating = bpf4.smooth(mingap, 1, 1, 0.01, 5, 0.0001)(margin)
    marginWeight, rangeWeight, wrangeWeight = 3, 1, 1
    if not track.partials:
        return mathlib.weighted_euclidian_distance([(marginRating, marginWeight),
                                                    (1, rangeWeight),
                                                    (1, wrangeWeight)])
    trackminnote, trackmaxnote = track.minnote, track.maxnote
    rangeWithPartial = max(trackmaxnote, partialPitch) - min(trackminnote, partialPitch)
    if rangeWithPartial > maxrange:
        return -1
    rangeRating = bpf4.expon(0, 1, maxrange, 0.0001, exp=1)(rangeWithPartial)
    trackPitch = track.meanpitch()
    pitchdiff = abs(trackPitch - pt.f2m(partial.meanfreq()))
    wrangeRating = bpf4.halfcos(0, 1, maxrange, 0.0001, exp=0.5)(pitchdiff)
    total = mathlib.weighted_euclidian_distance([(marginRating, marginWeight),
                                                 (rangeRating, rangeWeight),
                                                 (wrangeRating, wrangeWeight)])
    return total


def _bestTrack(tracks: list[Track], partial: Partial, mingap=0.1) -> Track | None:
    bestrating, besttrack = 0., None
    for track in tracks:
        rating = _ratePartial(track, partial, mingap=mingap)
        if rating >= bestrating:
            besttrack = track
        else:
            assert len(track.partials) > 0
    return besttrack


def _pack(spectrum: sp.Spectrum,
          numtracks: int,
          maxrange: int,
          mingap: float,
          chanexp: float,
          method: str,
          numchannels: int | None = None,
          minfreq=50.,
          maxfreq=12000.):

    from maelzel import distribute
    from .packchannel import Channel

    if numchannels is None:
        numchannels = int(numtracks / 2 + 0.5)
    numchannels = min(numtracks, numchannels)
    chanFreqCurve = bpf4.expon(0, pt.f2m(minfreq*0.9), 1, pt.f2m(maxfreq), exp=chanexp)
    splitpoints = list(chanFreqCurve.map(numchannels+1))
    channels: list[Channel] = [Channel(f0, f1) for f0, f1 in iterlib.pairwise(splitpoints)]
    for partial in spectrum.partials:
        for ch in channels:
            if ch.minfreq <= partial.meanfreq() < ch.maxfreq:
                ch.partials.append(partial)
                break
    # TODO: partial assignment can be optimized via searchsorted

    chanWeights = [ch.weight() for ch in channels]
    numtracksPerChan = [numtracks+1 for numtracks in distribute.dohndt(numtracks-numchannels, chanWeights)]
    tracks: list[Track] = []
    rejected0: list[Partial] = []
    for ch, tracksPerChan in zip(channels, numtracksPerChan):
        ch.pack(tracksPerChan, maxrange=maxrange, mingap=mingap, method=method)
        tracks.extend(ch.tracks)
        rejected0.extend(ch.rejected)

    # Try to fit rejected partials
    rejected = rejected0
    rejected1 = []
    for partial in rejected:
        track = _bestTrack(tracks, partial)
        if track is not None:
            track.append(partial)
        else:
            rejected1.append(partial)
    rejected.extend(rejected1)
    tracks = [track for track in tracks if len(track.partials) > 0]

    tracks.sort(key=lambda track: sum(p.audibility() for p in track.partials))
    return tracks, rejected


def _packold(spectrum: sp.Spectrum,
             maxrange: int,
             numtracks: int,
             mingap=0.1,
             packmethod='weight'
             ) -> tuple[list[Track], list[Partial]]:
    minchannels = min(3, numtracks)
    maxchannels = max(numtracks, minchannels)
    numPossibleChannels = min(maxchannels - minchannels, 5)
    possibleChannels = [int(x) for x in np.linspace(minchannels, maxchannels, numPossibleChannels)]
    minfreq = _estimateMinFreq(spectrum)
    results = []
    chanexps = [0.1, 0.4, 0.7, 1.3]
    for numchannels in possibleChannels:
        for chanexp in chanexps:
            tracks, rejected = _pack(spectrum=spectrum,
                                     numtracks=numtracks,
                                     mingap=mingap,
                                     numchannels=numchannels,
                                     maxrange=maxrange,
                                     minfreq=minfreq,
                                     chanexp=chanexp,
                                     method=packmethod)
            assignedPartials = (partial for track in tracks for partial in track.partials)
            assignedWeight  = sum(p.audibility() for p in assignedPartials)
            rejectedWeight = sum(p.audibility() for p in rejected)
            totalWeight = assignedWeight + rejectedWeight
            if totalWeight:
                rating = assignedWeight / totalWeight
            else:
                rating = 0
            # n = sum(len(track.partials) for track in tracks)
            results.append((rating, numchannels, chanexp, tracks, rejected))
    results.sort(key=lambda result: result[0])
    rating, numchannels, exp, tracks, rejected = results[-1]
    return tracks, rejected


@dataclass
class SplitResult:
    tracks: list[Track]
    """The fitted tracks"""

    noisetracks: list[Track]
    """Noise tracks, if any"""

    residual: list[Partial]
    """Partials not fitted within tracks and noisetracks"""

    distribution: float | bpf4.core.BpfInterface
    """The frequency distribution used to split the spectrum into bands"""

    def voicedPartials(self) -> list[Partial]:
        return sum((tr.partials for tr in self.tracks), [])

    def noisePartials(self) -> list[Partial]:
        return sum((tr.partials for tr in self.noisetracks), [])

    def partials(self) -> list[Partial]:
        partials = self.voicedPartials()
        partials.extend(self.noisePartials())
        return partials

    def __repr__(self):
        return f"SplitResult(tracks: {len(self.tracks)}, noisetracks: {len(self.noisetracks)}, " \
               f"residual: {len(self.residual)} partials)"


def optimizeSplit(partials: list[Partial],
                  maxtracks: int,
                  maxrange: int = 36,
                  relerror=0.1,
                  distributions: Sequence[float] | None = (0.2, 0.3, 0.6, 1., 2., 3.5, 6.),
                  numbands: int = None,
                  mingap=0.1,
                  noisetracks: int = 0,
                  noisefreq=4000,
                  noisebw=0.05,
                  debug=False
                  ) -> SplitResult:
    if distributions is not None:
        results = []
        for distr in distributions:
            if debug:
                print("Evaluating distribution", distr)
            tracks, residualtracks, unfitted = splitInTracks(
                partials,
                maxtracks=maxtracks,
                maxrange=maxrange,
                relerror=relerror,
                distribution=distr,
                numbands=numbands,
                mingap=mingap,
                audibilityCurveWeight=1.,
                maxnoisetracks=noisetracks,
                noisefreq=noisefreq,
                noisebw=noisebw,
                debug=debug)
            totalEnergy = 0.
            for track in tracks:
                totalEnergy += sum(p.audibility() for p in track.partials)
            results.append((totalEnergy, distr, tracks, residualtracks, unfitted))
        best = max(results, key=lambda result: result[0])
        if debug:
            print(f"Best distribution: {best[1]} (energy: {best[0]})")
        return SplitResult(distribution=best[1], tracks=best[2], noisetracks=best[3], residual=best[4])
    else:
        from scipy import optimize
        results = {}
        totalEnergy = sum(p.audibility() for p in partials)
        curve = bpf4.linear(0, 0.2, 0.5, 1, 1, 6)

        def func(distr0):
            distr = curve(distr0)
            if debug:
                print("Evaluating distribution", distr)
            tracks, residualtracks, unfitted = splitInTracks(
                partials,
                maxtracks=maxtracks,
                maxrange=maxrange,
                relerror=relerror,
                distribution=distr,
                numbands=numbands,
                mingap=mingap,
                audibilityCurveWeight=1.,
                maxnoisetracks=noisetracks,
                noisefreq=noisefreq,
                noisebw=noisebw,
                debug=debug)
            results[distr0] = (distr, tracks, residualtracks, unfitted)
            packedEnergy = sum(sum(p.audibility() for p in track) for track in tracks)
            return 1 - (packedEnergy - totalEnergy)

        r = optimize.minimize_scalar(func, bounds=(0, 1), tol=0.01)
        distr, tracks, residualtracks, unfitted = results[r['x']]
        return SplitResult(distribution=distr, tracks=tracks, noisetracks=residualtracks, residual=unfitted)


def splitInTracks(partials: list[Partial],
                  maxtracks: int,
                  maxrange: int = 36,
                  relerror=0.1,
                  distribution: float | Callable[[float], float] = 0.8,
                  numbands: int = None,
                  mingap=0.1,
                  audibilityCurveWeight=1.,
                  maxnoisetracks: int = 0,
                  noisefreq=4000,
                  noisebw=0.05,
                  debug=False
                  ) -> tuple[list[Track], list[Track], list[Partial]]:
    """
    Split the partials into tracks

    Args:
        partials: the partials
        maxtracks: max. number of tracks
        maxrange: max. range pro track
        relerror: ??
        distribution: frequency distribution. A callable mapping
        numbands:
        mingap:
        audibilityCurveWeight:
        maxnoisetracks:
        noisefreq:
        noisebw:
        debug:

    Returns:

    """
    from maelzel import packing
    from scipy import optimize

    # First, try to fit self without reduction
    items = [packing.Item(obj=partial, offset=partial.start, dur=partial.duration, step=pt.f2m(partial.meanfreq()))
             for partial in partials]

    packingtracks = packing.packInTracks(items, maxrange=maxrange, maxtracks=maxtracks, method='append', mingap=mingap)
    if packingtracks is not None:
        assert len(packingtracks) <= maxtracks
        return [Track(partials=track.unwrap()) for track in packingtracks], [], []

    # Enumerate the partials to be able to access the corresponding Item
    for i, p in enumerate(partials):
        p.label = i

    # We need to reduce the spectrum
    if numbands is None:
        numbands = max(4, maxtracks // 3)
    else:
        numbands = max(1, numbands)

    bands = splitInBands(partials, numbands=numbands, distribution=distribution)
    if debug:
        for band in bands:
            print(f"len={len(band.partials)}, {band.minfreq=:.0f}, {band.maxfreq=:.0f}, {min(p.meanfreq() for p in band.partials):.0f}, "
                  f"{max(p.meanfreq() for p in band.partials):.0f}")

    histograms = [histogram.Histogram([p.audibility(curvefactor=audibilityCurveWeight) for p in band.partials])
                  for band in bands]

    results = {}
    totalEnergy = sum(p.audibility() for p in partials)

    def _pack(percentile: float) -> tuple[float, list[packing.Track]]:
        """
        Pack partials by the given percentile

        Args:
            percentile: the audibility percentile

        Returns:
            a tuple (relative packed energy, packed trackes)

        """
        percentile = float(percentile)
        if debug:
            print("Testing percentile", percentile)
        if percentile <= 0:
            return 0., []
        elif percentile > 1:
            percentile = 1

        selected = []
        for band, hist in zip(bands, histograms):
            threshold = hist.percentileToValue(percentile)
            bandselection = [p for p in band.partials
                             if p.audibility(curvefactor=audibilityCurveWeight) >= threshold]
            if debug:
                print(f"... Partials from band {band.minfreq}:{band.maxfreq}: {len(bandselection)}")
            selected.extend(bandselection)

        # selected.sort(key=lambda p: p.label)
        selecteditems = [items[p.label] for p in selected]
        tracks = packing.packInTracks(selecteditems, maxrange=maxrange, method='append', mingap=mingap)
        if tracks is None:
            return 0., []
        elif len(tracks) > maxtracks:
            # return 0., []
            tracks.sort(key=lambda track: sum(item.obj.audibility() for item in track.items), reverse=True)
            tracks = tracks[:maxtracks]

        packedEnergy = 0.
        for track in tracks:
            packedEnergy += sum(p.audibility() for p in track.unwrap())
        relenergy = packedEnergy / totalEnergy
        if debug:
            print("........ packed relative energy", relenergy)
        return relenergy, tracks

    def _packeval(percentile: float) -> float:
        """
        Pack partials by the given percentile

        Args:
            percentile: the audibility percentile

        Returns:
            the relative packed energy. This must be maximized

        """
        percentile = float(percentile)
        relenergy, tracks = _pack(percentile)
        results[percentile] = tracks
        return relenergy

    if debug:
        percentiles = np.arange(0.001, 1.05, 0.05)
        relenergies = [_packeval(perc) for perc in percentiles]
        import matplotlib.pyplot as plt
        plt.plot(percentiles, [1 - rele for rele in relenergies])
        print(":::::::::::::::::::::::::::::::::::::::::::::")

    result = optimize.minimize_scalar(lambda percentile: 1 - _packeval(percentile), bracket=(0.001, 0.99), tol=relerror)
    percentile = float(result['x'])
    if debug:
        print("Solution percentile: ", percentile)
    if percentile in results:
        packingtracks = results[percentile]
    else:
        _, packingtracks = _pack(percentile)
    out = []
    selectedindexes = []
    for track in packingtracks:
        trackpartials = track.unwrap()
        out.append(trackpartials)
        selectedindexes.extend(p.label for p in trackpartials)

    selectedset = set(selectedindexes)
    unfitted = [p for p in partials if p.label not in selectedset]
    tracks = [Track(partials=track) for track in out]
    unfitted = fitPartialsInTracks(tracks, unfitted)
    if debug:
        print("Unfitted partials before noise:", len(unfitted), ", Selected:", len(selectedset))
    if maxnoisetracks == 0 or len(unfitted) == 0:
        noisetracks = []
    else:
        noisepartials = [p for p in unfitted
                         if p.meanfreq() > noisefreq and p.meanbw() > noisebw]
        if debug:
            print("Unfitted partials:", len(unfitted), "Noise partials:", len(noisepartials))
        noisetracks, _, unfittednoise = splitInTracks(noisepartials, maxtracks=maxnoisetracks, maxnoisetracks=0,
                                                      numbands=2, mingap=0.05, maxrange=60,
                                                      relerror=0.2)
        for track in noisetracks:
            for partial in track.partials:
                selectedset.add(partial.label)
        unfitted = [p for p in unfitted if p.label not in selectedset]

    tracks.sort(key=lambda track: track.minnote)
    return tracks, noisetracks, unfitted


@dataclass
class SpectralBand:
    """
    Represents partials within a spectral range
    """
    minfreq: float
    maxfreq: float
    partials: list[Partial]

    def audibility(self):
        """How audible are the partials in the band"""
        return sum(p.audibility() for p in self.partials)

    def energy(self):
        """The total energy of the partials within this band"""
        return sum(p.energy() for p in self.partials)


def splitInBands(partials: list[Partial],
                 numbands: int,
                 distribution: float | Callable[[float], float] = 1.0
                 ) -> list[SpectralBand]:
    """

    Args:
        partials: the partials to split
        numbands: number of bands
        distribution: either an exponent or a bpf between (0, 0) and (1, 1). When an exponent is given,
            a value of 1.0 will result in a linear distribution (all bins contain the same total weight),
            a value [0, 1) will result in the lower bins containing more weight than the higher bins,
            and a value higher than 1 will distribute more weight to the higher bins

    Returns:
        a list of SpectralBands
    """
    if numbands <= 1:
        return [SpectralBand(minfreq=0, maxfreq=24000, partials=partials)]

    freqs = [p.meanfreq() for p in partials]
    energies = [p.energy() for p in partials]
    ampfactors = amplitudesensitivity.ampcomparray(freqs)
    energies *= ampfactors
    freqedges = histogram.weightedHistogram(freqs, energies, numbins=numbands, distribution=distribution)
    bands = [SpectralBand(minfreq=minfreq, maxfreq=maxfreq, partials=[])
             for minfreq, maxfreq in iterlib.pairwise(freqedges)]
    assert len(bands) > 0, f"#freqs: {len(freqs)}, #energies: {energies}, numbads: {numbands}, distribution: {distribution}"

    bandindexes = np.searchsorted(freqedges, freqs) - 1
    bandindexes.clip(0, len(bands)-1, out=bandindexes)
    for idx, partial in zip(bandindexes, partials):
        assert 0 <= idx < len(bands), f"Band index out of range, got {idx}, but there are {len(bands)} bands"
        bands[idx].partials.append(partial)

    return bands


def fitPartialsInTracks(tracks: list[Track], partials: list[Partial], mingap=0.1, debug=False
                        ) -> list[Partial]:
    """
    Fit the given partials within the given tracks

    A track is a list of partials with non-simultaneous partials

    Args:
        tracks: a list of tracks. They will be modified in place
        partials: a list of partials
        mingap: a min. gap between partials within a track
        debug: if True print debugging information

    Returns:
        a list of residual partials. The tracks given are modified in place
    """
    residual2 = []
    partials.sort(key=lambda p: p.audibility(), reverse=True)
    for partial in partials:
        track = _bestTrack(tracks, partial, mingap=mingap)
        if track is not None:
            track.append(partial)
        else:
            if debug:
                ratings = [_ratePartial(track, partial, mingap=mingap) for track in tracks]
                print("Could not fit partial", [(rating, len(track.partials)) for rating, track in zip(ratings, tracks)])
            residual2.append(partial)

    for i, track in enumerate(tracks):
        if not track.check():
            track.dump()
            raise RuntimeError("Track has overlapping partials")

    return residual2
