"""
Implements a general routine for packing a seq. of
items (notes, partials, etc.) into a series of non-simultaneous containers (tracks)

A Track is a seq. of non-overlapping items

In order to be attached to a Track, each object (note, partial, etc.)
must be wrapped inside an Item, defining an offset, duration and step


"""

from __future__ import annotations
from maelzel.common import F, F0
from emlib.iterlib import pairwise
import operator
import bisect
from math import sqrt, inf


def asF(x) -> F:
    return x if isinstance(x, F) else F(x)


class Item:
    """
    an Item is used to wrap an object to be packed in a Track

    Attributes:
        obj: the object itself
        offset: the offset of the item (a time or x coordinate)
        dur: the "duration" or "width" of the object
        step: an arbitrary y-value for this object. This is used to pack
            together items which are similar, or to discard adding an item to
            a certain track if this step does not fit the track
        weight: an arbitrary weight for the object. It might be used to give
            this item a higher priority over other items when packing
    """
    __slots__ = ("obj", "offset", "dur", "step", "weight", "end")

    def __init__(self,
                 obj,
                 offset: F | float,
                 dur: F | float,
                 step: float,
                 weight: float = 1.0):
        """
        Args:
            obj (Any): the object to _packold
            offset (F): the start time of the object
            dur (F): the duration of the object
            step (float): the pitch step. This is used to distribute
                the item into a track
            weight: an item can be assigned a weight and this weight can be
                used for packing to give priority to certain items.
                Currently unused

        """
        self.obj = obj
        self.offset = asF(offset)
        self.dur = asF(dur)
        self.end = self.offset + self.dur
        self.step = step
        self.weight = weight

    def __repr__(self):
        objrepr = repr(self.obj)
        if len(objrepr) > 12:
            objrepr = f"{objrepr[:12]}…"
        return f"Item(offset={float(self.offset):.3f}, dur={float(self.dur):.3f}, step={self.step:.2f}, obj={objrepr})"

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


class Track:
    """ A Track is a list of non-simultaneous Items """

    def __init__(self, items: list[Item] = None):
        """
        Args:
            items: the items  to add to this track
        """
        self.items: list[Item] = items if items else []
        self._sortkey = operator.attrgetter('offset')
        self._modified = True
        self._minstep: float = 0.
        self._maxstep: float = 0.

    def _update(self) -> None:
        if not self.items:
            return
        self._minstep = min(item.step for item in self.items)
        self._maxstep = max(item.step for item in self.items)
        self._modified = False

    def minstep(self) -> float:
        if not self.items:
            raise ValueError("No items")
        if self._modified:
            self._update()
        return self._minstep

    def maxstep(self) -> float:
        if not self.items:
            raise ValueError("No items")
        if self._modified:
            self._update()
        return self._maxstep

    def append(self, item: Item) -> None:
        """
        Append an item to this Track

        Keeps the Track sorted if the Track was created with keepSorted==True

        .. note::

            Does not explicitely check if item fits in track. This check should
            be done before appending
        """
        if not self.items or self.items[-1].end <= item.offset:
            self.items.append(item)
        else:
            bisect.insort(self.items, item)
        self._modified = True

    def sort(self, *, key=None, reverse=False) -> None:
        """Sort this Track"""
        self.items.sort(key=key or self._sortkey, reverse=reverse)

    def extend(self, items: list[Item]) -> None:
        """Extend this Track"""
        assert all(isinstance(item, Item) for item in items)
        self.items.extend(items)
        self.sort()

    def __len__(self) -> int:
        return len(self.items)

    def dump(self) -> None:
        """ print track """
        print("--------")
        for item in self.items:
            print(f"{float(item.offset):.4f} - {float(item.end):.4f} {item.step}")

    def ambitus(self) -> tuple[float, float]:
        """
        Returns a tuple (min. step, max. step) for this track.
        """
        return self.minstep(), self.maxstep()

    def start(self) -> F:
        """
        The offset of the first item
        """
        return self.items[0].offset if self.items else F0

    def end(self) -> F:
        """
        The end value of the last item
        """
        return self.items[-1].end if self.items else F0

    def unwrap(self) -> list:
        return [item.obj for item in self.items]

    def _unwrap(self) -> list:
        """
        Unwraps the values inside this track

        Returns:
            a list with the original items wrapped in each Item in this Track

        In order to implement generic packing the strategy is to:

        1. _packold each object as an Item, explicitely copying into the Item
            the relvant information from the packed object: offset, duration, step
        2. call packInTracks. Items are distributed into a list of Tracks
        3. call unwrap for each Track to retrieve the packed objects
        """
        out = []
        for item in self.items:
            obj = item.obj
            if isinstance(obj, list):
                out.extend(obj)
            else:
                out.append(obj)
        return out

    def hasoverlap(self) -> bool:
        if not self.items:
            return False
        for item0, item1 in pairwise(self.items):
            if item0.end > item1.offset:
                return True
        return False


def packInTracks(items: list[Item],
                 maxrange: float = inf,
                 maxjump: float = inf,
                 method='append',
                 maxtracks: int = None,
                 mingap=0.,
                 ) -> list[Track] | None:
    """
    Pack the items into tracks, minimizing the amount of tracks needed

    To _packold an arbitrary list of objects:

    1. Wrap these objects into Items
    2. call ``packInTracks`` to distribute these Items into Tracks
    3. call ``track.unwrap()`` for each track to retrieve the objects
        packed in that Track

    Args:
        items: a seq. of Items

        maxrange: the maximum step range of a track. An item
            can be added to a track if the resulting range of the track
            would be smaller than this value. This is to minimize
            packing very disimilate items into one Track, if this is desired
        maxjump: if given, limit the step difference between any adjacent items
        method: one of 'append' or 'insert'. When 'append' is selected, a new
            Item will always be appended at the end of a Track. With 'insert'
            an Item can be inserted between two Items within a Track.
        mingap: a min. gap between packed items.
        maxtracks: if given, packing will fail early if the number of tracks is exceeded, returning None

    Returns:
        a list of the packed Tracks, or None if failed to _packold the items within the given max. number
        of tracks
    """
    tracks: list[Track] = []
    items2: list[Item] = sorted(items, key=operator.attrgetter('offset'))
    for item in items2:
        if method == 'insert':
            track = _bestTrackInsert(tracks, item, maxrange=maxrange, maxjump=maxjump)
        elif method == 'append':
            track = _bestTrackAppend(tracks, item, maxrange=maxrange, maxjump=maxjump,
                                     mingap=mingap)
        else:
            raise ValueError(f"Expected 'insert' or 'append', got {method=}")
        if track is None:
            if maxtracks and tracks and len(tracks) >= maxtracks:
                return None
            track = Track()
            tracks.append(track)
        track.append(item)
    assert all(not track.hasoverlap() for track in tracks)
    return tracks


def dumpTracks(tracks: list[Track]) -> None:
    """ print tracks """
    for track in tracks:
        track.dump()


# ------------------------------------------------------------------------

def _fits(track: Track, itemoffset: F, maxjump: float, step: float, maxrange: float) -> bool:
    assert track.items
    lastitem = track.items[-1]
    if lastitem.end > itemoffset:
        return False
    minstep = min(track.minstep(), step)
    maxstep = max(track.maxstep(), step)
    if maxstep - minstep > maxrange or abs(lastitem.step - step) > maxjump:
        return False
    return True


def _rateFitAppend(track: Track, item: Item) -> float:
    if not track.items:
        return float(item.offset)
    lastitem = track.items[-1]
    time1 = lastitem.end
    offsetPenalty = float(item.offset - time1)
    jumpPenalty = abs(item.step - lastitem.step)
    # penalty: the lower, the better
    penalty = sqrt(offsetPenalty ** 2 * 1 +
                   jumpPenalty ** 2 * 0.05)
    return float(penalty)


def _bestTrackAppend(tracks: list[Track], item: Item, maxrange=inf,
                     maxjump=inf, mingap=0.
                     ) -> Track | None:
    """
    Returns the best track in tracks to append the item to

    Args:
        tracks: list of existing tracks
        item: the item to append
        maxrange: the max. ambitus of the track
        maxjump: the max. jump between the previous item and this item. 0 allows any jump
        mingap: a min. gap between the end of the last event and the start of
            the next event

    Returns:
        the best track, or None if the item cannot be appended

    """
    itemoffset = item.offset - mingap
    step = item.step

    possibletracks = []
    for track in tracks:
        if track.items:
            if track.end() > itemoffset:
                break
            possibletracks.append(track)
        elif _fits(track, itemoffset=itemoffset, maxjump=maxjump, step=step, maxrange=maxrange):
            possibletracks.append(track)

    if not possibletracks:
        return None
    results = []
    for track in possibletracks:
        penalty = _rateFitAppend(track, item) if track.items else float(item.offset)
        if penalty == 0:
            return track
        results.append((penalty, track))
    return min(results)[1]


def _bestTrackInsert(tracks: list[Track], item: Item, maxrange: float,
                     maxjump=inf, mingap=0
                     ) -> Track | None:
    """
    Returns the best track in tracks to _packold item into

    Args:
        tracks: list of tracks
        item: node to fit
        maxrange: the maximum range a track can have
        maxjump: maximum step difference between two adjacent items in a track
        mingap: min. time gap between two items

    Returns:
        the best trackto place item, or None if item does not fit
    """
    possibletracks = [track for track in tracks
                      if _fitsInTrack(track, item, maxrange=maxrange, maxjump=maxjump, mingap=mingap)]
    if not possibletracks:
        return None
    results = [(_rateFit(track, item), track) for track in possibletracks]
    results.sort()
    _, track = results[0]
    return track


def _fitsInTrack(track: Track,
                 item: Item,
                 maxrange: float,
                 maxjump: float = 0.,
                 mingap=0.
                 ) -> bool:
    """
    Returns True if item can be added to track, False otherwise

    Args:
        track: the track to evaluate
        item: the item to potentially add to the track
        maxrange: the max. ambitus of the track. The ambitus is the range between the lowest
            and the highest item
        maxjump: how big can be the jump from one item to this new item

    Returns:
        True if the item can be added to the track
    """
    if not track.items:
        return True

    itemoffset = item.offset - mingap
    itemend = item.end + mingap
    if track.end() > itemoffset:
        for packeditem in track.items:
            if packeditem.offset > itemend:
                break
            if packeditem.end > itemoffset:
                return False

    step = item.step
    step0, step1 = track.ambitus()
    if max(step, step1) - min(step, step0) > maxrange:
        return False

    if maxjump is not None:
        idx = bisect.bisect(track.items, item.offset)
        if idx < len(track.items):
            if idx == 0:
                # no item to the left
                if abs(step - track.items[0].step) > maxjump:
                    return False
            else:
                left = track.items[idx - 1]
                right = track.items[idx]
                if abs(left.step-step) > maxjump or abs(right.step - step) > maxjump:
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
    if not track:
        time1 = 0
    else:
        time1 = track.items[-1].end
    assert time1 <= item.offset
    penalty = item.offset - time1
    return float(penalty)


def _checkTrack(track: Track) -> bool:
    if not track:
        return True
    for item0, item1 in pairwise(track.items):
        if item0.end > item1.offset:
            return False
    return True

