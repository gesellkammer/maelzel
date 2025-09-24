from __future__ import annotations
import numpy as np
from dataclasses import dataclass

import bpf4
import pitchtools as pt
from emlib import iterlib

# from maelzel import histogram
from maelzel.common import asF
from maelzel import stats
from maelzel.mathutils import linexp
from .partialtrack import PartialTrack

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Sequence
    from .partial import Partial
    from . import spectrum as sp

def _estimateMinFreq(spectrum: sp.Spectrum) -> float:
    f0, voicedness = spectrum.fundamental()
    return float(f0.map(1000).min())


def _ratePartial(track: PartialTrack, partial: Partial, maxrange: int | None = None, mingap=0.1) -> float:
    """
    The higher, the best. -1 indicates that the partial does not fit the track
    """
    maxrange = maxrange or track.maxrange
    partialPitch = partial.meanpitch()
    if not track.partials:
        margin = partial.start
        mingap = 0

    elif track.end < partial.start:
        margin = partial.start - track.end
        if margin < mingap:
            return -1
    elif track.start >= partial.start:
        margin = track.start - partial.end
        if margin < mingap:
            return -1
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
    marginRating = bpf4.Smooth.fromseq(mingap, 1, 1, 0.01, 5, 0.0001)(margin)
    marginWeight, rangeWeight, wrangeWeight = 3, 1, 1
    import emlib.mathlib
    if not track.partials:
        return emlib.mathlib.weighted_euclidian_distance([(marginRating, marginWeight),
                                                          (1, rangeWeight),
                                                          (1, wrangeWeight)])
    trackminnote, trackmaxnote = track.minnote, track.maxnote
    rangeWithPartial = max(trackmaxnote, partialPitch) - min(trackminnote, partialPitch)
    if rangeWithPartial > maxrange:
        return -1
    rangeRating = linexp(rangeWithPartial, 1., 0, 1, maxrange, 0.001)
    trackPitch = track.meanpitch()
    pitchdiff = abs(trackPitch - pt.f2m(partial.meanfreq()))
    wrangeRating = bpf4.Halfcos.fromseq(0, 1, maxrange, 0.0001, exp=0.5)(pitchdiff)
    total = emlib.mathlib.weighted_euclidian_distance([(marginRating, marginWeight),
                                                       (rangeRating, rangeWeight),
                                                       (wrangeRating, wrangeWeight)])
    return total


def bestTrack(tracks: list[PartialTrack], partial: Partial, mingap=0.1) -> PartialTrack | None:
    bestrating, besttrack = 0., None
    for track in tracks:
        rating = _ratePartial(track, partial, mingap=mingap)
        if rating >= bestrating:
            besttrack = track
        else:
            assert len(track.partials) > 0
    return besttrack


@dataclass
class SplitResult:
    tracks: list[PartialTrack]
    """The fitted tracks"""

    noisetracks: list[PartialTrack]
    """Noise tracks, if any"""

    residual: list[Partial]
    """Partials not fitted within tracks and noisetracks"""

    distribution: float | bpf4.BpfInterface | Callable[[float], float]
    """The frequency distribution used to split the spectrum into bands"""

    def voicedPartials(self) -> list[Partial]:
        return sum((tr.partials for tr in self.tracks), [])

    def voicedSpectrum(self) -> sp.Spectrum:
        from maelzel.partialtracking import spectrum
        return spectrum.Spectrum(self.voicedPartials())

    def noisePartials(self) -> list[Partial]:
        return sum((tr.partials for tr in self.noisetracks), [])

    def noiseSpectrum(self) -> sp.Spectrum:
        from maelzel.partialtracking import spectrum
        return spectrum.Spectrum(self.noisePartials())

    def partials(self) -> list[Partial]:
        partials = self.voicedPartials()
        partials.extend(self.noisePartials())
        return partials

    def __repr__(self):
        return f"SplitResult(tracks: {len(self.tracks)}, noisetracks: {len(self.noisetracks)}, " \
               f"residual: {len(self.residual)} partials)"

    def __iter__(self):
        return iter((self.tracks, self.noisetracks, self.residual))


def optimizeSplit(partials: list[Partial],
                  maxtracks: int,
                  maxrange: int = 36,
                  relerror=0.1,
                  distributions: Sequence[float] | None = (0.2, 0.3, 0.6, 1., 2., 3.5, 6.),
                  numbands: int = 3,
                  mingap=0.1,
                  noisetracks: int = 0,
                  noisefreq=4000,
                  noisebw=0.05,
                  method='insert',
                  debug=False,
                  mindistr=0.1,
                  maxdistr=6
                  ) -> SplitResult:
    if distributions is not None:
        # Pick best distribution from the given distributions
        results: list[tuple[float, float, list[PartialTrack], list[PartialTrack], list[Partial]]] = []
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
        # Find best distribution which maximizes audibility
        from scipy import optimize
        distrToResult: dict[float, tuple[float, list[PartialTrack], list[PartialTrack], list[Partial]]] = {}
        totalEnergy = sum(p.audibility() for p in partials)
        curve = bpf4.Linear.fromseq(0, mindistr, 0.5, mindistr*0.68+maxdistr*0.32, 1, maxdistr)

        def func(distr0: float) -> float:
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
                method=method,
                debug=debug)
            distrToResult[distr0] = (distr, tracks, residualtracks, unfitted)
            packedEnergy = sum(sum(p.audibility() for p in track) for track in tracks)
            return 1 - (packedEnergy - totalEnergy)

        # r = optimize.minimize_scalar(func, bounds=(0, 1), tol=0.01)
        r = optimize.minimize_scalar(func, bounds=(0, 1), tol=0.01)
        assert isinstance(r, optimize.OptimizeResult)
        bestdistr = r['x']
        distr, tracks, residualtracks, unfitted = distrToResult[bestdistr]
        return SplitResult(distribution=distr, tracks=tracks, noisetracks=residualtracks, residual=unfitted)


def splitInTracks(partials: list[Partial],
                  maxtracks: int,
                  maxrange: int = 36,
                  relerror=0.1,
                  distribution: float | Callable[[float], float] = 0.8,
                  numbands: int = 3,
                  mingap=0.1,
                  audibilityCurveWeight=1.,
                  maxnoisetracks: int = 0,
                  noisefreq=4000,
                  noisebw=0.05,
                  method='insert',
                  indexPeriod=0.,
                  debug=False
                  ) -> tuple[list[PartialTrack], list[PartialTrack], list[Partial]]:
    """
    Split the partials into tracks

    Args:
        partials: the partials
        maxtracks: max. number of tracks
        maxrange: max. range pro track
        relerror: relative error passed to the optimization routine
        distribution: frequency distribution. A callable mapping
        numbands: number of bands to divide the spectrum
        mingap: min. time gap between two partials within a track
        audibilityCurveWeight: the weight of the frequency dependent audibility curve
        maxnoisetracks: max. number of tracks to allocate for noise
        noisefreq: only partials with an avg. freq higher than this will be considered as
            noise
        noisebw: only partials with an avg. bandwidth higher than this will be
            considered as noise
        debug: print debug information
        method: one of 'append' or 'insert'

    Returns:
        a tuple (partialTracks, noiseTracks, unfittedPartials)

    """
    from maelzel import packing
    from scipy import optimize

    # First, try to fit self without reduction
    items = [packing.Item(obj=partial, offset=partial.start, dur=partial.duration, step=pt.f2m(partial.meanfreq()), weightfunc=lambda p: p.audibility())
             for partial in partials]

    packingtracks = packing.packInTracks(items, maxrange=maxrange, maxtracks=maxtracks,
                                         method=method, mingap=asF(mingap),
                                         indexperiod=indexPeriod)

    if packingtracks is not None:
        assert len(packingtracks) <= maxtracks
        return [PartialTrack(partials=track.unwrap()) for track in packingtracks], [], []

    # Enumerate the partials to be able to access the corresponding Item
    for i, p in enumerate(partials):
        p.label = i

    # We need to reduce the spectrum

    bands = splitInBands(partials, numbands=numbands, distribution=distribution)
    if debug:
        print("Split in bands. Number of bands: ", len(bands))
        for band in bands:
            print(f"len={len(band.partials)}, {band.minfreq=:.0f}, {band.maxfreq=:.0f}")

    quantiles = [stats.Quantile1d([p.audibility(curvefactor=audibilityCurveWeight) for p in band.partials])
                 for band in bands]

    if debug and numbands > 1:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(quantiles), 1, figsize=(8, 6 * len(quantiles)))
        for i, quantile in enumerate(quantiles):
            quantile.plot(feature='value', axes=axes[i], show=False)
        plt.show()

    percentileToTracks: dict[float, list[PartialTrack]] = {}
    totalEnergy = sum(p.audibility() for p in partials)

    def _pack(percentile: float) -> tuple[float, list[PartialTrack]]:
        """
        Pack partials by the given percentile

        Args:
            percentile: the audibility percentile

        Returns:
            a tuple (relative packed energy, packed trackes)

        """
        percentile = float(percentile)
        if percentile <= 0:
            return 0., []
        elif percentile > 1:
            percentile = 1

        if debug:
            print("Testing percentile", percentile)

        selected = []
        for band, bandquantile in zip(bands, quantiles):
            threshold = bandquantile.value(percentile)
            bandselection = [p for p in band.partials
                             if p.audibility(curvefactor=audibilityCurveWeight) >= threshold]
            if debug:
                print(f"... Partials from band {band.minfreq:.1f}:{band.maxfreq:.1f}: {len(bandselection)}, audibility threshold={threshold:.5g}")
            selected.extend(bandselection)

        selecteditems = [items[p.label] for p in selected]
        if debug:
            print(f"... Selected partials: {len(selecteditems)}")
        tracks = packing.packInTracks(selecteditems, maxrange=maxrange, method=method, mingap=asF(mingap))
        if tracks is None:
            return 0., []
        elif len(tracks) > maxtracks:
            # return 0., []
            tracks.sort(key=lambda track: sum(item.obj.audibility() for item in track.items), reverse=True)
            tracks = tracks[:maxtracks]

        partialtracks = [PartialTrack(track.unwrap()) for track in tracks]
        packedEnergy = 0.
        for track in partialtracks:
            packedEnergy += sum(partial.audibility() for partial in track)
        relenergy = packedEnergy / totalEnergy
        if debug:
            print("........ packed relative energy", relenergy)
        return relenergy, partialtracks

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
        percentileToTracks[percentile] = tracks
        return relenergy

    if method == 'append':
        minimizedPercentile = optimize.minimize_scalar(lambda percentile: 1 - _packeval(percentile), bracket=(0.05, 0.95), options=dict(xtol=relerror))
        assert isinstance(minimizedPercentile, optimize.OptimizeResult)
        percentile = float(minimizedPercentile['x'])
        if percentile in percentileToTracks:
            partialtracks = percentileToTracks[percentile]
        else:
            _, partialtracks = _pack(percentile)
    else:
        # percentile = 0.4
        relenergy, partialtracks = _pack(percentile=0.4)

    selectedindexes = []
    for track in partialtracks:
        selectedindexes.extend(p.label for p in track.partials)

    selectedset = set(selectedindexes)
    unfitted = [p for p in partials if p.label not in selectedset]
    unfitted = fitPartialsInTracks(partialtracks, unfitted)
    noisetracks: list[PartialTrack]
    if maxnoisetracks == 0 or len(unfitted) == 0:
        noisetracks = []
    else:
        noisepartials = [p for p in unfitted
                         if p.meanfreq() > noisefreq and p.meanbw() > noisebw]
        noisetracks, _, unfittednoise = splitInTracks(noisepartials, maxtracks=maxnoisetracks, maxnoisetracks=0,
                                                      numbands=3, mingap=0.05, maxrange=60, method=method,
                                                      relerror=0.2)

        for track in noisetracks:
            for partial in track.partials:
                selectedset.add(partial.label)
        unfitted = [p for p in unfitted if p.label not in selectedset]

    partialtracks.sort(key=lambda track: track.minnote)
    if debug:
        totalfitted = sum(len(t.partials) for t in partialtracks)
        noisepartialsfitted = sum(len(t.partials) for t in noisetracks)
        print(f"Fitted {totalfitted} partials in {len(partialtracks)} tracks, fitted {noisepartialsfitted} noise partials in {len(noisetracks)} noise tracks. Unfitted partials: {len(unfitted)}")
    return partialtracks, noisetracks, unfitted


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
    from maelzel.snd import amplitudesensitivity
    ampfactors = amplitudesensitivity.ampcomparray(freqs)
    energies *= ampfactors
    freqedges = stats.weightedHistogram(freqs, energies, numbins=numbands, distribution=distribution)
    bands = [SpectralBand(minfreq=minfreq, maxfreq=maxfreq, partials=[])
             for minfreq, maxfreq in iterlib.pairwise(freqedges)]
    assert 0 < len(bands) <= numbands, f"#freqs: {len(freqs)}, #energies: {energies}, numbads: {numbands}, distribution: {distribution}"

    bandindexes = np.searchsorted(freqedges, freqs) - 1
    bandindexes.clip(0, len(bands)-1, out=bandindexes)
    for idx, partial in zip(bandindexes, partials):
        assert 0 <= idx < len(bands), f"Band index out of range, got {idx}, but there are {len(bands)} bands"
        bands[idx].partials.append(partial)

    return bands


def fitPartialsInTracks(tracks: list[PartialTrack], partials: list[Partial], mingap=0.1, debug=False
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
        track = bestTrack(tracks, partial, mingap=mingap)
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
