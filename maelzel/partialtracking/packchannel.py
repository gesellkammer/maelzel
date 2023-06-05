from __future__ import annotations
from .partial import Partial
from .track import Track


class Channel:
    def __init__(self, minfreq: float, maxfreq: float, partials: list[Partial] | None = None):
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.partials: list[Partial] = partials or []
        self.tracks: list[Track] = []
        self.rejected: list[Partial] = []

    def pack(self, numtracks: int, maxrange: int, mingap: float, method='weight') -> None:
        if method == 'weight':
            self.packByWeight(numtracks, maxrange=maxrange, mingap=mingap)
        else:
            self.packByTime(numtracks, maxrange=maxrange, mingap=mingap)

    def weight(self) -> float:
        if not self.partials:
            return 0
        return sum(p.audibility() for p in self.partials)


    def packByTime(self, numtracks: int, maxrange: float, mingap: float):
        partials = sorted(self.partials, key=lambda partial: partial.start)
        tracks = [Track(mingap=mingap) for _ in range(numtracks)]
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
        partials: list[Partial] = sorted(self.partials, key=lambda p: p.audibility())
        tracks: list[Track] = [Track(mingap=mingap) for _ in range(numtracks)]
        rejected = []
        for partial in partials:
            track = _bestTrack(tracks, partial)
            if track is not None:
                track.append(partial)
            else:
                rejected.append(partial)
        self.tracks = tracks
        self.rejected = rejected