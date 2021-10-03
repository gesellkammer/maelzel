"""
Implements a general routine for packing a seq. of
items (notes, partials, etc) into a series of tracks.

A Track is a seq. of non-overlapping items

In order to be attached to a Track, each object (note, partial, etc.)
must be wrapped inside an Item, defining an offset, duration and step
"""

from __future__ import annotations
from maelzel.rational import Rat
from typing import List, Tuple, Optional as Opt, Union as U
from emlib.iterlib import pairwise
import operator
import bisect


number_t = U[int, float, Rat]


def _overlap(x0: number_t, x1: number_t, y0: number_t, y1: number_t) -> bool:
    """ do (x0, x1) and (y0, y1) overlap? """
    if x0 < y0:
        return x1 > y0
    return y1 > x0


def asF(x: number_t) -> Rat:
    if isinstance(x, Rat):
        return x
    return Rat(x)


class Item:
    """
    an Item is used to wrap an object to be packed in a Track

    """
    __slots__ = ("obj", "offset", "dur", "step", "weight")

    def __init__(self, obj, offset: number_t, dur: number_t, step: float,
                 weight: float = 1.0):
        """
        Args:
            obj (Any): the object to pack
            offset (Rat): the start time of the object
            dur (Rat): the duration of the object
            step (float): the pitch step. This is used to distribute
                the item into a track
            weight: an item can be assigned a weight and this weight can be
                used for packing to give priority to certain items.
                Currently unused

        """
        self.obj = obj
        self.offset = asF(offset)
        self.dur = asF(dur)
        self.step = step
        self.weight = weight

    @property
    def end(self) -> Rat:
        """ end time of item """
        return self.offset + self.dur

    def __lt__(self, other:Item) -> bool:
        return self.offset < other.offset

    def __le__(self, other: Item) -> bool:
        return self.offset <= other.offset

    def __eq__(self, other: Item) -> bool:
        return self.offset == other.offset

    def __ge__(self, other: Item) -> bool:
        return self.offset >= other.offset

    def __gt__(self, other: Item) -> bool:
        return self.offset > other.offset


class Track(list):
    """ A Track is a list of non-simultaneous Items """
    def __init__(self, items=None, keepSorted=True):
        if items:
            super().__init__(items)
        else:
            super().__init__()
        self.keepSorted = keepSorted
        self._sortKey = operator.attrgetter('offset')

    def append(self, item: Item) -> None:
        """
        Append an item to this Track and keep it sorted
        Does not explicitely check if item fits in track
        """
        if not self or not self.keepSorted or self[-1].offset < item.offset:
            super().append(item)
        else:
            bisect.insort(self, item)

    def sort(self, *, key=None, reverse=False) -> None:
        super().sort(key=key or self._sortKey, reverse=reverse)

    def extend(self, items: List[Item]) -> None:
        assert all(isinstance(item, Item) for item in items)
        if not self.keepSorted:
            super().extend(items)
        else:
            items.sort(key=self._sortKey)
            needsSort = len(self) > 0 and self.end() >= items[0].offset
            super().extend(items)
            if needsSort:
                self.sort()

    def __getitem__(self, idx: int) -> Item:
        out = super().__getitem__(idx)
        assert isinstance(out, Item)
        return out

    def dump(self) -> None:
        """ print track """
        print("--------")
        for item in self:
            print(f"{float(item.offset):.4f} - {float(item.end):.4f} {item.step}")

    def ambitus(self) -> Tuple[float, float]:
        """
        Returns a tuple (min. step, max. step) for this track.
        """
        if not self:
            raise ValueError("This Track is empty")
        minstep = float("inf")
        maxstep = float("-inf")
        for item in self:
            step = item.step
            if step < minstep:
                minstep = step
            elif step > maxstep:
                maxstep = step
        return minstep, maxstep

    def start(self) -> Rat:
        if self.keepSorted:
            return self[0].offset
        else:
            return min(item.offset for item in self)

    def end(self) -> Rat:
        if self.keepSorted:
            return self[-1].end
        return max(item.end for item in self)

    def unwrap(self) -> list:
        """
        In order to implement generic packing the strategy is to:

        1. pack each object as an Item, explicitely copying into the Item
            the relvant information from the packed object: offset, duration, step
        2. call packInTracks. Items are distributed into a list of Tracks
        3. call unwrap for each Track to retrieve the packed objects
        """
        out = []
        for item in self:
            obj = item.obj
            if isinstance(obj, list):
                out.extend(obj)
            else:
                out.append(obj)
        return out

    def isWellFormed(self) -> bool:
        if not self:
            return True
        for item0, item1 in pairwise(self):
            if item0.end > item1.offset:
                return False
        return True


def packInTracks(items: List[Item],
                 maxAmbitus=36,
                 maxJump:int=None) -> List[Track]:
    """
    Distribute the items into tracks, minimizing the amount of
    tracks needed to pack the given items.

    To pack an arbitrary list of objects:

    1. Wrap these objects into Items
    2. call ``packInTracks`` to distribute these Items into Tracks
    3. call ``track.unwrap()`` for each track to retrieve the objects
        packed in that Track

    Args:
        items: a seq. of Items
        maxAmbitus: the maximum step range of a track. An item
            can be added to a track if the resulting range of the track
            would be smaller than ``maxTrackRange``. This is to minimize
            packing very disimilate items into one Track, in this is desired
        maxJump: if given, limit the step difference between any adjacent items

    Returns:
        a list of the packed Tracks
    """
    tracks: List[Track] = []
    items2: List[Item] = sorted(items, key=lambda item: item.offset)
    for item in items2:
        track = _bestTrack(tracks, item, maxAmbitus=maxAmbitus,
                           maxJump=maxJump)
        if track is None:
            track = Track()
            tracks.append(track)
        track.append(item)
    assert all(track.isWellFormed() for track in tracks)
    return tracks


def dumpTracks(tracks: List[Track]) -> None:
    """ print tracks """
    for track in tracks:
        track.dump()


# ------------------------------------------------------------------------


def _bestTrack(tracks: List[Track], item: Item, maxAmbitus: int,
               maxJump:int=None) -> Opt[Track]:
    """
    Returns the best track in tracks to pack item into

    Args:
        tracks: list of tracks
        item: node to fit
        maxAmbitus: the maximum range a track can have
        maxJump: if given, sets a maximum step difference
            between two adjacent items in a track

    Returns:
        the best trackto place item, or None if item does not fit
    """
    possibletracks = [track for track in tracks
                      if _fitsInTrack(track, item, maxAmbitus=maxAmbitus, maxJump=maxJump)]
    if not possibletracks:
        return None
    results = [(_rateFit(track, item), track) for track in possibletracks]
    results.sort()
    _, track = results[0]
    return track


def _fitsInTrack(track: Track, item: Item, maxAmbitus: int,
                 maxJump: int=None) -> bool:
    """
    Returns True if item can be added to track, False otherwise
    """
    if not track:
        return True

    for packeditem in track:
        if _overlap(packeditem.offset, packeditem.end, item.offset, item.end):
            return False

    step = item.step
    step0, step1 = track.ambitus()
    if max(step, step1) - min(step, step0) > maxAmbitus:
        return False

    if maxJump is not None:
        idx = bisect.bisect(track, item.offset)
        if idx < len(track):
            if idx == 0:
                # no item to the left
                if abs(step - track[0].step) > maxJump:
                    return False
            else:
                left = track[idx - 1]
                right = track[idx]
                if abs(left.step-step) > maxJump or abs(right.step-step) > maxJump:
                    return False

    return True


def _rateFit(track: Track, item: Item) -> float:
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


def _checkTrack(track: Track) -> bool:
    if not track:
        return True
    for item0, item1 in pairwise(track):
        if item0.end > item1.offset:
            return False
    return True


