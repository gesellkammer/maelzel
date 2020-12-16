from fractions import Fraction
from typing import NamedTuple, Iterator as Iter, List, Any
from emlib.iterlib import pairwise

"""
Implements a general routine for packing a seq. of
items (notes, partials, etc) into a series of tracks.

A Track is a seq. of non-overlapping items

In order to be attached to a Track, each object (note, partial, etc.)
must be wrapped inside an Item, defining an offset, duration and step
"""


def _overlap(x0, x1, y0, y1):
    """ do (x0, x1) and (y0, y1) overlap? """
    if x0 < y0:
        return x1 > y0
    return y1 > x0


class Item(NamedTuple):
    """ an Item is an object which can be packed in a Track """
    obj: Any
    offset: Fraction
    dur: Fraction
    step: float

    @property
    def end(self):
        """ end time of item """
        return self.offset + self.dur


class Track(list):
    """ A Track is a list of Items """
    def append(self, item: Item) -> None:
        assert isinstance(item, Item)
        super().append(item)

    def extend(self, items: List[Item]) -> None:
        assert all(isinstance(item, Item) for item in items)
        super().extend(items)

    def __getitem__(self, idx: int) -> Item:
        out = super().__getitem__(idx)
        assert isinstance(out, Item)
        return out


def pack_in_tracks(items: Iter[Item], maxrange=36) -> List[Track]:
    """
    items: a seq. of Items
    maxrange: the maximum pitch range of a track, in semitones
    """
    tracks: List[Track] = []
    items2: List[Item] = sorted(items, key=lambda itm: itm.offset)
    for item in items2:
        track = _best_track(tracks, item, maxrange=maxrange)
        if track is None:
            track = Track()
            tracks.append(track)
        track.append(item)
    assert all(_checktrack(track) for track in tracks)
    return tracks


def dumptrack(track: Track) -> None:
    """ print track """
    print("--------")
    for item in track:
        print(f"{float(item.offset):.4f} - {float(item.end):.4f} {item.step}")


def dumptracks(tracks: List[Track]) -> None:
    """ print tracks """
    for track in tracks:
        dumptrack(track)

# ------------------------------------------------------------------------


def _track_getrange(track):
    if not track:
        return None, None
    note0 = 99999999999
    note1 = 0
    for item in track:
        step = item.step
        if step < note0:
            note0 = step
        elif step > note1:
            note1 = step
    return note0, note1


def _best_track(tracks: List[Track], item: Item, maxrange: int):
    """
    tracks: list of tracks
    node: node to fit
    trackrange: the maximum range a track can have
    """
    possibletracks = [track for track in tracks
                      if _fits_in_track(track, item, maxrange=maxrange)]
    if not possibletracks:
        return None
    results = [(_rate_fit(track, item), track) for track in possibletracks]
    results.sort()
    _, track = results[0]
    return track


def _fits_in_track(track: Track, item: Item, maxrange: int):
    if not track:
        return True
    for packednode in track:
        if _overlap(packednode.offset, packednode.end, item.offset, item.end):
            return False
    tracknote0, tracknote1 = _track_getrange(track)
    step = item.step
    note0 = min(tracknote0, step)
    note1 = max(tracknote1, step)
    if note1 - note0 < maxrange:
        return True
    return False


def _rate_fit(track: Track, item: Item):
    """
    Return a value representing how goog this item fits in track
    Assumes that it fits both horizontally and vertically.
    The lower the value, the best the fit
    """
    assert isinstance(track, list)
    assert isinstance(item, Item)
    if not track:
        time1 = 0
    else:
        time1 = track[-1].end
    assert time1 <= item.offset
    rating = item.offset - time1
    return rating


def _checktrack(track: Track) -> bool:
    if not track:
        return True
    for item0, item1 in pairwise(track):
        if item0.end > item1.offset:
            return False
    return True


