"""
Transcribe a partial-tracking spectrum
"""
from __future__ import annotations
from emlib import iterlib
from maelzel.partialtracking.partial import Partial
from maelzel.partialtracking import spectrum as sp

from .core import Breakpoint, simplifyBreakpoints, TranscribeOptions

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from maelzel.partialtracking.track import Track
    import maelzel.core as mc
    from maelzel.scorestruct import ScoreStruct


def partialToBreakpoints(partial: Partial,
                         bandwidthThreshold=0.001,
                         simplify=0.
                         ) -> list[Breakpoint]:
    breakpoints = []
    data = partial.data
    lasti = partial.numbreakpoints - 1
    for i in range(partial.numbreakpoints):
        bw = data[i, 4]
        bp = Breakpoint(time=data[i, 0],
                        freq=data[i, 1],
                        amp=data[i, 2],
                        voiced=bw < bandwidthThreshold,
                        linked=i < lasti,
                        )
        breakpoints.append(bp)
    if simplify:
        breakpoints = simplifyBreakpoints(breakpoints, param=simplify)

    for bp1, bp2 in iterlib.pairwise(breakpoints):
        bp1.duration = bp2.time - bp1.time

    return breakpoints


def trackToVoice(partials: list[Partial],
                 scorestruct: ScoreStruct | None = None,
                 options: TranscribeOptions | None = None
                 ) -> mc.Voice:

    from maelzel.transcribe import mono

    breakpointGroups = [partialToBreakpoints(partial)
                        for partial in partials]

    voice = mono.transcribeVoice(groups=breakpointGroups,
                                 scorestruct=scorestruct,
                                 options=options)
    return voice


def transcribeTracks(tracks: list[Track],
                     noisetracks: list[Track] | None = None,
                     scorestruct: ScoreStruct | None = None,
                     options: TranscribeOptions | None = None
                     ) -> mc.Score:
    import maelzel.core

    pitchedVoices = [trackToVoice(track.partials, scorestruct=scorestruct, options=options)
                     for track in tracks]

    noiseVoices = [trackToVoice(track.partials, scorestruct=scorestruct, options=options)
                   for track in noisetracks]

    for i, voice in enumerate(pitchedVoices):
        voice.name = f'V{i}'

    for i, voice in enumerate(noiseVoices):
        voice.name = f'N{i}'

    noiseVoices.sort(key=lambda voice: voice.meanPitch(), reverse=True)
    pitchedVoices.sort(key=lambda voice: voice.meanPitch(), reverse=True)

    allvoices = noiseVoices + pitchedVoices
    score = maelzel.core.Score(voices=allvoices)
    return score


def transcribe(spectrum: sp.Spectrum,
               maxtracks: int,
               noisetracks=0,
               maxrange=36,
               mingap=0.1,
               noisebw=0.001,
               noisefreq=3500,
               scorestruct: ScoreStruct | None = None,
               options: TranscribeOptions | None = None
               ) -> tuple[mc.Score, list[Partial]]:

    result = spectrum.splitInTracks(maxtracks=maxtracks,
                                    noisetracks=noisetracks,
                                    maxrange=maxrange,
                                    mingap=mingap,
                                    noisebw=noisebw,
                                    noisefreq=noisefreq)
    score = transcribeTracks(result.tracks,
                             noisetracks=result.noisetracks,
                             scorestruct=scorestruct,
                             options=options)
    return score, result.residual

