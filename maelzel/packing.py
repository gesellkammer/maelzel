"""
Implements a general routine for packing a seq. of
items (notes, partials, etc.) into a series of non-simultaneous containers (tracks)

A Track is a seq. of non-overlapping items

In order to be attached to a Track, each object (note, partial, etc.)
must be wrapped inside an Item, defining an offset, duration and step


"""

from __future__ import annotations
import bisect
import operator
from math import sqrt, inf, ceil
from maelzel.common import F, F0


import typing as _t
if _t.TYPE_CHECKING:
    import numpy as np


def asF(x) -> F:
    return x if isinstance(x, F) else F(x)


class Item[T]:
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
    __slots__ = ("obj", "offset", "dur", "step", "_weight", "end", "weightfunc")

    def __init__(self,
                 obj: T,
                 offset: float,
                 dur: float,
                 step: float,
                 weight: float | None = None,
                 weightfunc: _t.Callable[[T], float] | None = None):
        """
        Args:
            obj (Any): the object to _packold
            offset (F): the start time of the object
            dur (F): the duration of the object
            step (float): the pitch step. This is used to distribute
                the item into a track
            weight: an item can be assigned a weight and this weight can be
                used for packing to give priority to certain items.
            weightfunc: as an alternative to precalculating a weight, a
                weight function of the form `(obj: T) -> weight: float`
                can be passed, to calculate a weight as needed

        """
        self.obj: T = obj
        # self.offset = asF(offset)
        self.offset = float(offset)
        # self.dur = asF(dur)
        self.dur = float(dur)
        self.end = self.offset + self.dur
        self.step = step
        self._weight = weight
        self.weightfunc = weightfunc

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

    def hasWeight(self) -> bool:
        return self._weight is not None or self.weightfunc is not None

    def weight(self, default=1.0) -> float:
        """
        Weight of this Item

        Args:
            default: value used if this item has no weight
                and no weight function was given

        Returns:
            the weight of this item, normally a value between 0 and 1
        """
        if self._weight is not None:
            return self._weight
        elif self.weightfunc:
            self._weight = w = self.weightfunc(self.obj)
            return w
        return default


class Track[T]:
    """ A Track is a list of non-simultaneous Items """

    def __init__(self, items: list[Item[T]] | None = None):
        """
        Args:
            items: the items  to add to this track
        """
        self.items: list[Item] = items if items else []
        self._sortkey = operator.attrgetter('offset')
        self._modified = True
        self._minstep: float | None = None
        self._maxstep: float | None = None
        self._offsets: list[float] | None = None
        self._index: np.ndarray | None = None
        self._indexstart = 0.
        self._indexperiod = 0.

    def __iter__(self) -> _t.Iterator[Item[T]]:
        return iter(self.items)

    def buildIndex(self, end: float, period: float, start=0.):
        import numpy as np
        n = int((end - start) / period)
        self._index = np.ones((n,), dtype=int)
        self._index *= -1  # -1 indicates unused, otherwise indicates the index of the item
        self._indexperiod = period
        self._indexstart = start

    def _update(self) -> None:
        if not self._modified or not self.items:
            return
        self._minstep = None
        self._maxstep = None
        self._modified = False

    def minstep(self) -> float:
        if not self.items:
            raise ValueError("No items")
        if self._modified:
            self._update()
        if self._minstep is None:
            self._minstep = min(item.step for item in self.items)
        return self._minstep

    def maxstep(self) -> float:
        if not self.items:
            raise ValueError("No items")
        if self._modified:
            self._update()
        if self._maxstep is None:
            self._maxstep = max(item.step for item in self.items)
        return self._maxstep

    def offsets(self) -> list[float]:
        if self._modified:
            self._update()
        if self._offsets is None:
            self._offsets = [item.offset for item in self.items]
        return self._offsets

    def insert(self, item: Item[T], idx: int | None = None) -> None:
        if idx is None:
            idx = bisect.bisect(self.offsets(), item.offset)
        self.items.insert(idx, item)
        if self._minstep is not None:
            self._minstep = min(self._minstep, item.step)
        if self._maxstep is not None:
            self._maxstep = max(self._maxstep, item.step)
        if self._offsets is not None:
            self._offsets.insert(idx, item.offset)
        if self._index is not None:
            self._addItemToIndex(item, idx)

    def append(self, item: Item[T]) -> None:
        """
        Append an item to this Track

        Keeps the Track sorted if the Track was created with keepSorted==True

        .. note::

            Does not explicitely check if item fits in track. This check should
            be done before appending
        """
        if self.items and self.items[-1].end > item.offset:
            raise ValueError(f"item {item} has an offset lower than the end "
                             f"of the last item (last={self.items[-1]})")
        self.items.append(item)
        if self._minstep is not None:
            self._minstep = min(self._minstep, item.step)
        if self._maxstep is not None:
            self._maxstep = max(self._maxstep, item.step)
        if self._offsets is not None:
            self._offsets.append(item.offset)
        if self._index is not None:
            self._addItemToIndex(item, len(self.items))

    def _addItemToIndex(self, item: Item[T], idx: int):
        assert self._index is not None
        slot0 = ceil((item.offset - self._indexstart) / self._indexperiod)
        slot1 = int((item.end - self._indexstart) / self._indexperiod)
        if slot1 >= len(self._index):
            self._index.resize(slot1*2)
        self._index[slot0:slot1] = idx

    def sort(self, *, key=None, reverse=False) -> None:
        """Sort this Track"""
        self.items.sort(key=key or self._sortkey, reverse=reverse)

    def extend(self, items: list[Item]) -> None:
        """Extend this Track"""
        assert all(isinstance(item, Item) for item in items)
        self.items.extend(items)
        self.sort()
        self._modified = True

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
        if self._modified:
            self._update()
        if self._minstep is None:
            self._minstep = min(item.step for item in self.items)
        if self._maxstep is None:
            self._maxstep = max(item.step for item in self.items)
        return self._minstep, self._maxstep

    def start(self) -> float:
        """
        The offset of the first item
        """
        return self.items[0].offset if self.items else 0

    def end(self) -> float:
        """
        The end value of the last item
        """
        return self.items[-1].end if self.items else 0

    def unwrap(self) -> list[T]:
        return [item.obj for item in self.items]

    def hasoverlap(self) -> bool:
        if not self.items:
            return False
        item0 = self.items[0]
        for item1 in self.items[1:]:
            if item0.end > item1.offset:
                return True
            item0 = item1
        return False

    def emptyBetween(self, start: float, end: float) -> tuple[int, float] | None:
        """
        Checks if the time range (start, end) is empty

        Args:
            start: start time
            end: duration

        Returns:
            a tuple (index, gap) or None

        Example
        ~~~~~~~

        if fit := track.fits(start, dur):
            idx, gap = fit

        """
        endgap = start - self.end()
        if endgap >= 0:
            return len(self.items), endgap

        if self._index is not None:
            slot0 = int((start - self._indexstart) / self._indexperiod)
            slot1 = ceil((end - self._indexstart) / self._indexperiod)
            if slot1 < len(self._index) and (self._index[slot0:slot1+1] >= 0).any():
                # not empty
                return None

        items = self.items
        idx = bisect.bisect(self.offsets(), start)
        if idx == 0:
            gap = items[0].offset - end
        elif idx == len(items):
            gap = start - items[-1].end
        else:
            itemR = items[idx]
            itemL = items[idx - 1]
            gapR = itemR.offset - end
            gapL = start - itemL.end
            gap = -1 if gapR < 0 or gapL < 0 else min(gapR, gapL)
        return None if gap < 0 else (idx, gap)


def packInTracks[T](items: list[Item[T]],
                    maxrange: float = inf,
                    maxjump: float = inf,
                    method='insert',
                    maxtracks: int | None = None,
                    mingap=F0,
                    indexperiod: float = 0.
                    ) -> list[Track[T]] | None:
    """
    Pack the items into tracks, minimizing the amount of tracks needed

    To pack an arbitrary list of objects:

    1. Wrap these objects into Items
    2. call :func:`packInTracks` to distribute these Items into Tracks
    3. call :meth:`Track.unwrap` for each track to retrieve the objects
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
        a list of the packed Tracks, or None if the number of tracks exceeded the maximum given
    """
    tracks: list[Track] = []
    # presort by offset
    items2: list[Item] = sorted(items, key=operator.attrgetter('offset'))
    if method == 'insert' and all(item.hasWeight() for item in items2):
        # We sort again by weight. If all weights are the same, the previous
        # sort order is kept
        items2.sort(key=lambda item: item.weight(), reverse=True)

    start = items2[0].offset
    end = items2[-1].end
    assert method in ('insert', 'append')
    for item in items2:
        if method == 'insert':
            result = _bestTrackInsert(tracks, item, maxrange=maxrange, maxjump=maxjump)
            if result:
                track, idx = result
                track.insert(item, idx)
            else:
                newtrack = Track([item])
                if indexperiod:
                    newtrack.buildIndex(end=end, period=indexperiod, start=start)
                tracks.append(newtrack)
        else:
            track = _bestTrackAppend(tracks, item, maxrange=maxrange, maxjump=maxjump,
                                     mingap=mingap)
            if track:
                track.append(item)
            else:
                tracks.append(Track([item]))

        if maxtracks and tracks and len(tracks) >= maxtracks:
            return None

    assert all(not track.hasoverlap() for track in tracks)
    return tracks


def dumpTracks(tracks: list[Track]) -> None:
    """ print tracks """
    for track in tracks:
        track.dump()


# ------------------------------------------------------------------------

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
                     maxjump=inf, mingap=F0
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
    possibletracks = []
    for track in tracks:
        if track.items:
            if track.end() > itemoffset:
                break
            possibletracks.append(track)
        elif _fitsInTrack(track, item=item, maxjump=maxjump, maxrange=maxrange):
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
                     ) -> tuple[Track, int] | None:
    """
    Returns the best track in tracks to pack item into

    Args:
        tracks: list of tracks
        item: node to fit
        maxrange: the maximum range a track can have
        maxjump: maximum step difference between two adjacent items in a track
        mingap: min. time gap between two items

    Returns:
        the best trackto place item, or None if item does not fit
    """
    results: list[tuple[float, float, Track, int]] = []
    step = item.step
    for track in tracks:
        step0, step1 = track.ambitus()
        if step - step0 > maxrange or step1 - step > maxrange:
            continue
        fits = track.emptyBetween(item.offset, item.end)
        if fits:
            idx, gap = fits
            results.append((gap, abs((step0 + step1) * 0.5 - step), track, idx))
    if not results:
        return None
    best = min(results, key=lambda row: (row[0], row[1]))
    gap, stepdiff, track, insertidx = best
    return track, insertidx


def _fitsInTrack(track: Track,
                 item: Item,
                 maxrange: float,
                 maxjump: float = 0.,
                 mingap=0.
                 ) -> float:
    """
    Returns the distance to the next item (negative if item does not fit)

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
        return 0.

    step = item.step
    step0, step1 = track.ambitus()
    if max(step, step1) - min(step, step0) > maxrange:
        return -1

    maxjump = 0

    itemoffset = item.offset - mingap
    endgap = itemoffset - track.end()
    if endgap >= 0:
        gap = endgap
    else:
        idxright = bisect.bisect(track.items, item)
        if idxright == 0:
            gap = track.items[0].offset - item.end
        elif idxright == len(track.items):
            gap = item.offset - track.items[-1].end
        else:

            itemR = track.items[idxright]
            itemL = track.items[idxright - 1]
            gapR = itemR.offset - item.end
            gapL = item.offset - itemL.end
            gap = -1 if gapR < 0 or gapL < 0 else min(gapR, gapL)

    if maxjump:
        raise ValueError("maxjump not supported")

    #if gap >= 0:
    #    assert all(it.end <= item.offset or it.offset >= item.end for it in track.items)
    return float(gap)


def _checkTrack(track: Track) -> bool:
    if not track:
        return True
    item0 = track.items[0]
    for item1 in track.items[1:]:
        if item0.end > item1.offset:
            return False
        item0 = item1
    return True
