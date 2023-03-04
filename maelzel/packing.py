"""
Implements a general routine for packing a seq. of
items (notes, partials, etc) into a series of non-simultaneous containers (tracks)

A Track is a seq. of non-overlapping items

In order to be attached to a Track, each object (note, partial, etc.)
must be wrapped inside an Item, defining an offset, totalDuration and step


"""

from __future__ import annotations
from maelzel.common import F
from emlib.iterlib import pairwise
import operator
import bisect


def _overlap(x0: F | float, x1: F | float, y0: F | float, y1: F | float) -> bool:
    """ do (x0, x1) and (y0, y1) overlap? """
    if x0 < y0:
        return x1 > y0
    return y1 > x0


def asF(x) -> F:
    return x if isinstance(x, F) else F(x)


class Item:
    """
    an Item is used to wrap an object to be packed in a Track

    Attributes:
        obj: the object itself
        offset: the offset of the item (a time or x coordinate)
        dur: the "totalDuration" or "width" of the object
        step: an arbitrary y-value for this object. This is used to tree
            together items which are similar, or to discard adding an item to
            a certain track if this step does not fit the track
        weight: an arbitrary weight for the object. It might be used to give
            this item a higher priority over other items when packing
    """
    __slots__ = ("obj", "offset", "dur", "step", "weight")

    def __init__(self, obj, offset: F | float, dur: F | float, step: float,
                 weight: float = 1.0):
        """
        Args:
            obj (Any): the object to pack
            offset (F): the start time of the object
            dur (F): the totalDuration of the object
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
    def end(self) -> F:
        """ end time of item """
        return self.offset + self.dur

    def __lt__(self, other: Item) -> bool:
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

    def __init__(self, items: list[Item] = None, keepSorted=True):
        """
        Args:
            items: the items  to add to this track
            keepSorted: keep the track sorted when appending elements?
        """
        if items:
            super().__init__(items)
        else:
            super().__init__()
        self.keepSorted = keepSorted
        self._sortKey = operator.attrgetter('offset')

    def append(self, item: Item) -> None:
        """
        Append an item to this Track

        Keeps the Track sorted if the Track was created with keepSorted==True

        .. note::

            Does not explicitely check if item fits in track. This check should
            be done before appending
        """
        if not self or not self.keepSorted or self[-1].offset < item.offset:
            super().append(item)
        else:
            bisect.insort(self, item)

    def sort(self, *, key=None, reverse=False) -> None:
        """Sort this Track"""
        super().sort(key=key or self._sortKey, reverse=reverse)

    def extend(self, items: list[Item]) -> None:
        """Extend this Track"""
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

    def ambitus(self) -> tuple[float, float]:
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

    def start(self) -> F:
        """
        The offset of the first item
        """
        if self.keepSorted:
            return self[0].offset
        else:
            return min(item.offset for item in self)

    def end(self) -> F:
        """
        The end value of the last item
        """
        if self.keepSorted:
            return self[-1].end
        return max(item.end for item in self)

    def unwrap(self) -> list:
        """
        Unwraps the values inside this track

        Returns:
            a list with the original items wrapped in each Item in this Track

        In order to implement generic packing the strategy is to:

        1. pack each object as an Item, explicitely copying into the Item
            the relvant information from the packed object: offset, totalDuration, step
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

    def hasNoOverlap(self) -> bool:
        """
        Returns True if the items in this track do not overlap
        """
        if not self:
            return True
        for item0, item1 in pairwise(self):
            if item0.end > item1.offset:
                return False
        return True


def packInTracks(items: list[Item],
                 maxAmbitus: float = float('inf'),
                 maxJump: int = None
                 ) -> list[Track]:
    """
    Pack the items into tracks, minimizing the amount of tracks needed

    To pack an arbitrary list of objects:

    1. Wrap these objects into Items
    2. call ``packInTracks`` to distribute these Items into Tracks
    3. call ``track.unwrap()`` for each track to retrieve the objects
        packed in that Track

    Args:
        items: a seq. of Items
        maxAmbitus: the maximum step range of a track. An item
            can be added to a track if the resulting range of the track
            would be smaller than this value. This is to minimize
            packing very disimilate items into one Track, if this is desired
        maxJump: if given, limit the step difference between any adjacent items

    Returns:
        a list of the packed Tracks
    """
    tracks: list[Track] = []
    items2: list[Item] = sorted(items, key=operator.attrgetter('offset'))
    for item in items2:
        track = _bestTrack(tracks, item, maxAmbitus=maxAmbitus,
                           maxJump=maxJump)
        if track is None:
            track = Track()
            tracks.append(track)
        track.append(item)
    assert all(track.hasNoOverlap() for track in tracks)
    return tracks


def dumpTracks(tracks: list[Track]) -> None:
    """ print tracks """
    for track in tracks:
        track.dump()


# ------------------------------------------------------------------------


def _bestTrack(tracks: list[Track], item: Item, maxAmbitus: float,
               maxJump: int = None
               ) -> Track | None:
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


def _fitsInTrack(track: Track, item: Item, maxAmbitus: float,
                 maxJump: int = None
                 ) -> bool:
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
    Return a value representing how good this item fits in track

    Assumes that it fits both horizontally and vertically.
    The lower the value, the best the fit

    Args:
        track: the track to rate
        item: the item which should be placed in track

    Returns:
        a penalte. The lower the penalty, the better the fit
    """
    assert isinstance(track, list)
    assert isinstance(item, Item)
    if not track:
        time1 = 0
    else:
        time1 = track[-1].end
    assert time1 <= item.offset
    penalty = item.offset - time1
    return float(penalty)


def _checkTrack(track: Track) -> bool:
    if not track:
        return True
    for item0, item1 in pairwise(track):
        if item0.end > item1.offset:
            return False
    return True

