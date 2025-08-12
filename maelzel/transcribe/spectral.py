"""
Transcribe a partial-tracking spectrum
"""
from __future__ import annotations
from emlib import iterlib
from maelzel.partialtracking.partial import Partial
from maelzel.partialtracking import spectrum as sp

from .breakpoint import Breakpoint, simplifyBreakpoints
from .options import TranscriptionOptions

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.partialtracking.partialtrack import PartialTrack
    import maelzel.core as mc
    from maelzel.scorestruct import ScoreStruct
    

__all__ = (
    'transcribeTracks',
    'transcribe',
    'TranscriptionOptions'
)


def partialToBreakpoints(partial: Partial,
                         bandwidthThreshold=0.001,
                         simplify=0.
                         ) -> list[Breakpoint]:
    """
    Convert a partial to a list of breakpoints.

    Args:
        partial (Partial): The partial to convert.
        bandwidthThreshold (float): The threshold for determining if a breakpoint is voiced.
        simplify (float): The parameter for simplifying the breakpoints.

    Returns:
        list[Breakpoint]: The list of breakpoints.
    """
    breakpoints = []
    data = partial.data
    lasti = partial.numbreakpoints - 1
    for i in range(partial.numbreakpoints):
        bw = float(data[i, 4])
        bp = Breakpoint(time=float(data[i, 0]),
                        freq=float(data[i, 1]),
                        amp=float(data[i, 2]),
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
                 options: TranscriptionOptions | None = None
                 ) -> mc.Voice:
    """
    Convert a track to a voice.

    Args:
        partials: List of partials to convert.
        scorestruct: Score structure to use for transcription.
        options: Transcription options to use.

    Returns:
        A voice object representing the track.
    """
    from maelzel.transcribe import mono

    breakpointGroups = [partialToBreakpoints(partial)
                        for partial in partials]

    voice = mono.transcribeVoice(groups=breakpointGroups,
                                 scorestruct=scorestruct,
                                 options=options)
    return voice


def transcribeTracks(tracks: list[PartialTrack],
                     noisetracks: list[PartialTrack] | None = None,
                     scorestruct: ScoreStruct | None = None,
                     options: TranscriptionOptions | None = None
                     ) -> mc.Score:
    """
    Transcribe a list of tracks into a score.

    Args:
        tracks: List of tracks to transcribe.
        noisetracks: List of noise tracks to transcribe.
        scorestruct: Score structure to use for transcription.
        options: Transcription options to use.

    Returns:
        A score object representing the transcribed tracks.
    """
    import maelzel.core
    if options is None:
        options = TranscriptionOptions()

    voices = [trackToVoice(track.partials, scorestruct=scorestruct, options=options)
              for track in tracks]

    for i, voice in enumerate(voices):
        voice.name = f'V{i}'

    voices.sort(key=lambda voice: voice.meanPitch(), reverse=True)

    if noisetracks is not None:
        noiseVoices = [trackToVoice(track.partials, scorestruct=scorestruct, options=options)
                       for track in noisetracks]
        for i, voice in enumerate(noiseVoices):
            voice.name = f'N{i}'
        noiseVoices.sort(key=lambda voice: voice.meanPitch(), reverse=True)
        voices.extend(noiseVoices)

    score = maelzel.core.Score(voices=voices)
    return score


def transcribe(spectrum: sp.Spectrum,
               maxtracks: int,
               noisetracks=0,
               maxrange=36,
               mingap=0.1,
               noisebw=0.001,
               noisefreq=3500,
               scorestruct: ScoreStruct | None = None,
               options: TranscriptionOptions | None = None
               ) -> tuple[mc.Score, list[Partial]]:
    """
    Transcribe the spectrum as a Score

    Args:
        spectrum: the Spectrum to transcribe
        maxtracks: the max. number of tracks
        noisetracks: number of tracks used to pack noisy partials / residual
        maxrange: the max. range of a track, in semitones
        mingap: the min. gap between partials within a track
        noisebw: the bandwidth of a partial to be considered noise
        noisefreq: partials above this frequency can be qualified as noise
        scorestruct: a ScoreStruct used for transcription
        options: the TranscribeOptions used, or None to use default options

    Returns:
        a tuple (score, residualpartials)

    """

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
