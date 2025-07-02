from __future__ import annotations
from . import pack
from .partialtrack import PartialTrack
from .partial import Partial


class Channel:
    """
    A Channel represents a frequency range within a sound spectrum.

    Args:
        minfreq: The minimum frequency of the channel.
        maxfreq: The maximum frequency of the channel.
        partials: A list of partials within the channel.
    """
    def __init__(self, minfreq: float, maxfreq: float, partials: list[Partial] | None = None):
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.partials: list[Partial] = partials or []
        self.tracks: list[PartialTrack] = []
        self.rejected: list[Partial] = []

    def pack(self, numtracks: int, maxrange: int, mingap: float, method='weight') -> None:
        """
        Packs the partials into tracks within the channel.

        Args:
            numtracks: The number of tracks to pack the partials into.
            maxrange: The maximum range of frequencies for each track.
            mingap: The minimum gap between partials on the same track.
            method: The packing method to use ('weight' or 'time').
        """
        if method == 'weight':
            self.packByWeight(numtracks, maxrange=maxrange, mingap=mingap)
        else:
            self.packByTime(numtracks, maxrange=maxrange, mingap=mingap)

    def weight(self) -> float:
        if not self.partials:
            return 0
        return sum(p.audibility() for p in self.partials)


    def packByTime(self, numtracks: int, maxrange: float, mingap: float):
        """
        Packs the partials into tracks within the channel based on their start time.

        Args:
            numtracks: The number of tracks to pack the partials into.
            maxrange: The maximum range of frequencies for each track.
            mingap: The minimum gap between partials on the same track.
        """
        partials = sorted(self.partials, key=lambda partial: partial.start)
        tracks = [PartialTrack() for _ in range(numtracks)]
        rejected: list[Partial] = []
        for partial in partials:
            possibleTracks = [track for track in tracks
                              if not track.partials or track.partials[-1].end + mingap <= partial.start]
            if possibleTracks:
                track = min(possibleTracks, key=lambda track: partial.start - track.end)
                track.append(partial)
            else:
                rejected.append(partial)
        self.tracks = tracks
        self.rejected = rejected

    def packByWeight(self, numtracks: int, maxrange: int, mingap: float):
        """
        Packs the partials into tracks within the channel based on their weight.

        Args:
            numtracks: The number of tracks to pack the partials into.
            maxrange: The maximum range of frequencies for each track.
            mingap: The minimum gap between partials on the same track.
        """
        partials: list[Partial] = sorted(self.partials, key=lambda p: p.audibility())
        tracks: list[PartialTrack] = [PartialTrack() for _ in range(numtracks)]
        rejected = []
        for partial in partials:
            track = pack.bestTrack(tracks, partial)
            if track is not None:
                track.append(partial)
            else:
                rejected.append(partial)
        self.tracks = tracks
        self.rejected = rejected
