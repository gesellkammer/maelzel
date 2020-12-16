from collections import deque, namedtuple as _namedtuple
import random as _random
from numbers import Number as _Number
from operator import attrgetter
import logging

from bpf4 import bpf as _bpf

from emlib import iterlib
from emlib import lib as _lib
from emlib.containers import RecordList as _RecordList

logger = logging.getLogger("emlib")


"""
talea: always a real seq. of numbers, representing durations
color: a seq. (can be an iterator) of any objects
"""


# ------------------------------------------------------------
#
# Helpers    
#
# ------------------------------------------------------------


def _normalize_color(c):
    if isinstance(c, str):
        return c.split()
    return c


def _normalize_mask(mask):
    if isinstance(mask, str):
        if not all(x in 'xo' or x in 'x-' for x in mask):
            raise ValueError("could not understand mask")
        mask = [1 if x == 'x' else 0 for x in mask]
    assert all(m in (0, 1) for m in mask)
    return mask


def _isnan(x):
    return not (x == x)


def _parse_mask(m, defaultmask=(1,)):
    """
    A talea/color can be any kind of iterator
    A mask can be any kind of iterator
    m can be either one or two iterators: m=talea or m=(talea, mask)
    m can't be a tuple
    """
    if not isinstance(m, tuple):
        return m, defaultmask
    if len(m) != 2:
        raise ValueError("A talea/color should be either an iterator or a list\n"
                         "A tuple is reserved for a masked iterator")
    seq, mask = m
    mask = _normalize_mask(mask)
    return seq, mask


def _almosteq(a, b):
    """ are a and b equal floats? """
    return abs(a - b) < 1e-12


class Masked(object):
    def __init__(self, stream, mask):
        """
        mask: an iterator of 0s or 1s
        """
        self.stream = stream
        self.mask = _normalize_mask(mask)

    def __iter__(self):
        maskstream = iterlib.cicle(self.mask)
        for x, m in zip(self.stream, maskstream):
            if m:
                yield x

# ------------------------------------------------------------
#
# Item: an item of a talea color   
#
# ------------------------------------------------------------


class Item(_namedtuple("BaseItem", "color dur start")):
    def __repr__(self):
        return "start=%.3f dur=%.3f | %s" % (self.start, self.dur, self.color)

    @property
    def end(self):
        return self.start + self.dur


class Items(_RecordList): 
    pass


# ------------------------------------------------------------------------------
#
# Talea Protocol 
#
# _get_iterator: virtual. The only thing that needs to be defined externally
# __iter__     : one can always iterate over a talea
# take_between : basen on __iter__, it provides modes of selection
# and iteration over the items rendered by __iter__
#    
# ------------------------------------------------------------------------------


class _TaleaProtocol(object):

    def __init__(self, transform=None, histlen=10):
        """
        transform: a function of the form

        def transform(item):
            return item._replace(...) if item.dur > 0.1 else None

        It will be called for each item and should return an Item or None,
        in which case the item is skipped. The transform can also raise a StopIteration
        exception, in which case the iteration is stopped

        The transform can be also set later:

        >>> t = Talea([3, 4, 2, 5], "A B C".split())                      
        # compress it in time with high-compression at t=10
        >>> t2 = t.warp(bpf.linear(0, 1, 10, 0.1, 20, 1))
        # filter all items which are smaller than 0.3
        >>> t.transform = lambda item: item if item.dur >= 0.3 else None  

        """
        self.transform = transform
        self.history = deque([], histlen)
        self.timeoffset = 0

    def __iter__(self):
        """
        should return an iterator over Items, where Item is

        Item = namedtuple('Item', "color start dur")

        color can be anything, even a sequence of colors
        """
        now = self.timeoffset
        maxnone = 1000
        accumnone = 0

        def _item_is_sane(item):
            return not _isnan(item.dur)
        for dur, color in self._get_iterator():
            item = Item(color=color, start=now, dur=dur)
            if self.transform is not None:
                item = self.transform(item)
            if item is not None:
                assert _item_is_sane(item)
                yield item
                now += item.dur
                accumnone = 0
            else:
                accumnone += 1
                if accumnone > maxnone:
                    raise ValueError("the ammount of cancelled items (set to None)"
                                     "reached the limit. Check your transform function!")

    def take_between(self, t0, t1, left=None, right=None):
        """
        returns a **list** of items that lie between the interval (t0, t1)

        left:
            None       -> do nothing. Only items which are fully included
                          within the time frame indicated.
            'align'    -> align all frames to the left so that the
                          first full item after t0 begins now at t0
            'overflow' -> if an item begins before t0 and exceeds t0
                          it is included
            'clip'     -> if an item begins before t0, it is cut and
                          only the portion after t0 is given
        right:
            The same as left ('align' is not implemented)

        transform, if given, will override the original transform of this Talea

        NB: 'align' can only be applied to left or to right, not to both

        >>> t = Talea([1, 1, 1], "A B C D".split())
        >>> t.take_between(0.5, 3, left=None, right=None)
        [1.000 -- dur 1.000 : B
         2.000 -- dur 1.000 : C
        ]

        >>> t.take_between(0.5, 3, left='align', right=None)
        [0.500 -- dur 1.000 : A
         1.500 -- dur 1.000 : B
        ]

        >>> t.take_between(0.5, 2.5, left='align', right='clip')
        [0.500 -- dur 1.000 : A
         1.500 -- dur 1.000 : B
         2.500 -- dur 0.500 : C
        ]

        >>> t.take_between(0.5, 2.5, left='align', right='overflow')
        [0.500 -- dur 1.000 : A
         1.500 -- dur 1.000 : B
         2.500 -- dur 1.000 : C
        ]

        etc.
        """
        i = iter(self)
        items = []
        extra_dur = 0
        for item in i:
            assert not _isnan(item.dur)
            if item.start >= (t1 + extra_dur):
                break
            if _lib.intersection(item.start, item.end, t0, t1):
                if item.start <= t0:
                    # extra_dur = item.start - t0
                    extra_dur = t0 - item.start
                items.append(item)
        t1 += extra_dur
        if not items:
            return items
        #######################
        # some sanity checks    
        #######################
        if not all(item.dur >= 0 for item in items):
            print("WARNING! item.dur <= 0")
            print([item for item in items if item.dur <= 0])
        assert all(abs(i0.end - i1.start) < 1e-13 for i0, i1 in iterlib.pairwise(items))
        assert all(_lib.intersection(item.start, item.end, t0, t1) for item in items)

        #################
        # LEFT EDGE
        #################
        if left is None:
            if items[0].start < t0:
                items = items[1:]
        elif left == 'align':
            if items[0].start < t0:
                items.pop(0)
                if not items:  # this could be the only item
                    return []
                else:
                    shift = items[0].start - t0
                    items = [item._replace(start=item.start - shift) for item in items]
        elif left == 'overflow':
            pass  # do nothing
        elif left == 'clip':
            it0 = items[0]
            assert it0.end > t0
            if it0.start < t0:
                dur = it0.dur - (t0 - it0.start)
                items[0] = it0._replace(start=t0, dur=dur)
        else:
            raise ValueError("mode not understood for 'left'")
        ##################
        # RIGHT EDGE
        ##################
        if right is None:
            items = [item for item in items if (item.end) <= t1]
        elif right == 'align':
            raise NotImplementedError("right align is not implemented")
        elif right == 'clip':
            lastitem = items[-1]
            if lastitem.end > t1:
                assert lastitem.start < t1
                dur = lastitem.dur - (lastitem.end - t1)
                assert dur > 0
                items[-1] = lastitem._replace(dur=dur)
        return Items(items)

    def _get_iterator(self):
        """
        New classes should define _get_iterator

        should return an iterator yielding (dur, color)
        """
        pass

    def warp(self, timeshape):
        """
        Time-warp this talea by a given time varying factor

        Example
        =======

        >>> t = Talea([2, 4, 3], "A B C D".split())
        >>> tw = t.warp(0.5)
        >>> items = tw.take_between(0, 10)
        >>> [item.dur for item in items]
        [1.0, 2.0, 1.5, 1.0, 2.0, 1.5, 1.0]

        # Now create a shape, so that at time 0 the talea is in its
        # original form, and linearly decreases to a 0.5 compression
        # by time 10.
        >>> from bpf4 import bpf
        >>> shape = _bpf.linear(0, 1, 10, 0.5)
        >>> tw =    rp(shape)
        >>> [item.dur for item in tw.take_between(0, 10)]
        ## TODO: put solution here
        """
        timeshape = _bpf.asbpf(timeshape)
        return _WarpedTalea(self, timeshape)

    def overlap(self, threshold, factor, overlap_direction=1):
        """
        A new talea based on this one with a certain (time-varying) overlap
        of items.

        Args
        ----

        threshold: a (time-varying) threshold of duration, after which an overlap is
                  triggered
        factor: a (time-varying) factor of the duration threshold to go back in time
                1   -> will go back the same value set as threshold
                0.5 -> will go back the half of the threshold

        overlap_direction: if positive, the overlapped section is repeated in the
                           same direction as in the source
                           if nevative, the overlapped section is repeated backwards
                           (like a rewind tape)

        Example
        -------

        >>> t0 = Talea([2, 3, 1], "A B C D".split())
        >>> t0.take_between(0, 10)
        # TODO: put result here

        >>> toverlap = t0.overlap(threshold=4, factor=0.5)
        >>> toverlap.take_between(0, 10)
        """
        return _OverlapTalea(self, threshold, factor)


class _TaleaBase(_TaleaProtocol):

    def __init__(self, talea, color, transform=None):
        super(_TaleaBase, self).__init__(transform)
        assert all(isinstance(dur, _Number) for dur in talea)
        self._talea = talea
        self._color = color

    @property
    def cycle_duration(self):
        return sum(self._talea)

# ------------------------------------------------------------
##
# Talea    
##
# ------------------------------------------------------------


class Talea(_TaleaBase):

    def __init__(self, talea, color, transform=None):
        """
        talea     -> a seq of durations. It can be masked (see below)
        color     -> a seq of any kind of item. It can be masked (see below)
        transform -> a function that takes an Item and returns either an Item or None

        Masking: a mask is a seq of 0 and 1s or a string in the form "xxooxo", 
                 where x means 1 and o means 0.
                 A mask is normally not the same length as the seq and acts as a
                 perforation in the talea

        Examples
        ========

        # color can be a single string, in which case it will be splitted
        >>> t = Talea((2, 4, 3), "A B C D")   
        >>> t.take_between(0, 15)
        [Item(color='A', start=0, dur=2),
         Item(color='B', start=2, dur=4),
         Item(color='C', start=6, dur=3),
         Item(color='D', start=9, dur=2),
         Item(color='A', start=11, dur=4)]

        >>> t = Talea((2, 4, 3), ("A B C D", "xxxxo"))
        >>> t.take_between(0, 15)
        [Item(color='A', start=0, dur=2),
         Item(color='B', start=2, dur=4),
         Item(color='C', start=6, dur=3),
         Item(color='D', start=9, dur=2),
         Item(color='B', start=11, dur=4)]

        See how the A with duration 3 was skipped

        The same mask applied to the talea renders a different result:
        >>> t = Talea(((2, 4, 3), "xxxxo"), "A B C D")
        >>> t.take_between(0, 15)
        [Item(color='A', start=0, dur=2),
         Item(color='B', start=2, dur=4),
         Item(color='C', start=6, dur=3),
         Item(color='D', start=9, dur=2),
         Item(color='A', start=11, dur=3),
         Item(color='B', start=14, dur=2)]

        Here the duration 4 was skipped, but without any influence on the color
        """
        talea, taleamask = _parse_mask(talea, defaultmask=(1,))
        color, colormask = _parse_mask(color, defaultmask=(1,))
        color = _normalize_color(color)
        super(Talea, self).__init__(talea, color, transform)
        self._taleamask = taleamask 
        self._maskcolor = colormask

    def _get_iterator(self):
        colorseq = zip(iterlib.cycle(self._color), iterlib.cycle(self._maskcolor))
        taleaseq = zip(iterlib.cycle(self._talea), iterlib.cycle(self._taleamask))
        maskedcolor = (x[0] for x in filter(lambda color_mask:color_mask[1], colorseq))
        maskedtalea = (x[0] for x in filter(lambda talea_mask:talea_mask[1], taleaseq))
        return zip(maskedtalea, maskedcolor)

######################################################
# WarpedTalea
######################################################


class _WarpedTalea(_TaleaProtocol):

    """
    Normally you dont create a _WarpedTalea directly.
    You create a Talea and call its 'warp' method
    """

    def __init__(self, source, shape):
        super(_WarpedTalea, self).__init__()
        self._source = source
        self._shape = shape
        r2v = (1 / shape).render(10000).integrated()
        self._real2virt = r2v
        self._virt2real = r2v.inverted()
        self._bounds = self._virt2real.bounds()
        if not _almosteq(self._virt2real.x1, r2v(r2v.x1)):
            raise ValueError("bounds are wrong")

    def __iter__(self):
        v2r = self._virt2real

        def item_is_sane(item):
            return not _isnan(item.dur) and not _isnan(item.start)
        x0, x1 = self._bounds
        for item in self._source:
            if item.start >= x1 or item.end <= x0:
                break        # <--- only yield inside the warped area
            start = v2r(item.start)
            if item.end > x1:
                dur = v2r(x1) - start
            else:    
                dur = v2r(item.end) - start
            item2 = item._replace(start=start, dur=dur)
            if not item_is_sane(item2):
                s = " : ".join(map(str, (item.start, item.dur, item2.start, item2.dur)))
                raise ValueError("item is not sane: %s" % s)
            yield item2

    def taleatime_to_warpedtime(self, t):
        return self._virt2real(t)

    def warpedtime_to_taleatime(self, t):
        return self._real2virt(t)

######################################################
# OverlapTalea
######################################################


class _OverlapTalea(_TaleaProtocol):

    def __init__(self, source, threshold, overlap, overlap_direction=1):
        """
        source            -> the source talea
        threshold         -> a (time-varying) threshold of duration after which an overlap is 
                             triggered. It can be viewed as the window length of a sliding window
                             over the items of the talea
        overlap           -> the overlap-factor of the threshold to go back. Like the overlap in windowed
                             FFT analysis
        overlap_direction -> if positive, the overlapped section is first backed up and then the
                             items are rendered in their correct time order. Its a repetition
                             if negative, the overlapped section is rendered LIFO, resulting
                             in an inversion of the overlapped section
                             NB: it can be dynamic, changing over time. It will be sampled
                             when triggered by the treshold

        Example
        =======

        For simplification, assuming a talea of 4 values, all with the same duration,
        a threshold of the whole cycle and a factor of 0.75 would result in:

        ABCD
         BCDE
          CDEF
           DEFG
        """
        super(_OverlapTalea, self).__init__()
        self._source = source
        self._threshold = _bpf.asbpf(threshold)
        self._factor = _bpf.asbpf(overlap)
        self._overlap_direction = _bpf.asbpf(overlap_direction)      
        self._constantoverlap = (isinstance(threshold, _Number) and
                                 isinstance(self._factor, _Number))

    def __iter__(self):
        it, it0 = iterlib.tee(self._source, 2)
        now = it0.next().start
        pastitems = []
        yielded_items = []
        accum = 0
        accumpast = 0
        take_from_future = True
        thresh = self._threshold(now)
        quota = self._threshold(now) * self._factor(now)
        while True:
            # from FUTURE
            if take_from_future:      
                item = it.next()
                item = item._replace(start=now)
                accum += item.dur
                yielded_items.append(item)
                yield item
                now += item.dur
                if accum >= thresh:
                    accumpast = 0
                    threshback = quota + (accum - thresh)
                    pastitems = []
                    take_from_future = False
                    accum = thresh - accum
            # from PAST
            else:  
                item = yielded_items.pop()
                accumpast += item.dur
                pastitems.append(item)
                if accumpast >= threshback:
                    overlapdir = self._overlap_direction(now)
                    overlapped = pastitems if overlapdir < 0 else reversed(pastitems)
                    for item in overlapped:
                        item = item._replace(start=now)
                        yielded_items.append(item)
                        yield item
                        now += item.dur
                    take_from_future = True
                    thresh = self._threshold(now) - threshback + (accumpast - threshback)
                    factor = self._factor(now)
                    if factor >= 1:
                        raise ValueError("Overlap factor must be lower than 1")
                    quota = self._threshold(now) * factor
                    accum = 0 

    def warp(self, timeshape):
        """
        Time-warp this talea by a given time varying factor (see the Parent class)
        """
        if not self._constantoverlap:
            logger.warning(
                "Attempting to warp a Talea with a time varying overlap"
                "Are you sure? Normally a talea is first warped and then overlapped."
                "Attempting to overlap and then warp will result in misinterpreted"
                "timings for the threshold and factor envelopes defining the overlap")
        return super(_OverlapTalea, self).warp(timeshape)

# ------------------------------------------------------------
##
# TaleaX    
##
# ------------------------------------------------------------


class TaleaX(_TaleaProtocol):
    def __init__(self, talea, durfunc=None, transform=None, **colors):
        """
        One Talea, Many Colors. All colors share the same talea
        Both talea and color(s) can have a mask. The color(s) have to 
        be given by name (see example)

        Arguments
        ---------

        durfunc --> a function which is passed an item an returns a duration
                    This is a hook to implement filtering based on start-time, 
                    or on context, etc.

        Example
        -------

        A masked talea with two colors, one masked and one unmasked

        MaskedMulti(talea=([2, 3, 4, 5], "xxxoxxo"), 
                    color1=("A B C D", "xxxxo"), 
                    color2="NN MM NN")
        """
        super(TaleaX, self).__init__(transform=transform)
        talea, taleamask = _parse_mask(talea)
        self._durfunc = durfunc if durfunc is not None else lambda item:item.dur
        field_names = colors.keys()
        colormasks = [colors[name] for name in field_names]
        colors, masks = [], []
        for colormask in colormasks:
            color, mask = _parse_mask(colormask)
            colors.append(color)
            masks.append(mask)
        self._colors = list(map(_normalize_color, colors))
        self._masks = list(map(_normalize_mask, masks))
        self._talea, self._taleamask = talea, taleamask
        self._item_ctor = _namedtuple("I", field_names)
        self._field_names = field_names

    def _get_iterator(self):
        seqs = [zip(iterlib.cycle(color), iterlib.cycle(mask))
                for color, mask in zip(self._colors, self._masks)]
        maskedseqs = [(x[0] for x in filter(lambda mask:mask[1], seq)) for seq in seqs]
        taleaseq = zip(iterlib.cycle(self._talea), iterlib.cycle(self._taleamask))
        maskedtalea = (x[0] for x in filter(lambda talea_mask:talea_mask[1], taleaseq))
        return zip(*([maskedtalea] + maskedseqs))

    def __iter__(self):
        now = 0
        zipped = self._get_iterator()

        def resolve_color(c, now):
            if callable(c):
                return c(now)
            elif hasattr(c, 'next'):
                return c.next()
            return c
        while True:
            item = zipped.next()
            dur = item[0]
            colors = [resolve_color(color, now) for color in item[1:]]
            colors = self._item_ctor(*colors)
            item = Item(color=colors, start=now, dur=dur)
            newdur = self._durfunc(item)
            if newdur != dur:
                item = item._replace(dur=newdur)
            if self.transform:
                item = self.transform(item, self.history)
            if item is None:
                continue
            now += item.dur
            self.history.append(item)
            yield item

    def flatiter(self):
        durfunc = self._durfunc
        self._durfunc = lambda item:item.dur
        g0 = iter(self).next()
        attrs = [f for f in g0._fields if f != 'color'] + \
                ['color.%s' % f for f in g0.color._fields]
        fields = [attr.split('.')[-1] for attr in attrs]
        FlatItem = _namedtuple('Item', fields)
        for item in iter(self):
            d = dict([(field, attrgetter(attr)(item)) 
                      for attr, field in zip(attrs, fields)])
            flatitem = FlatItem(**d)
            newdur = durfunc(flatitem)
            item = flatitem._replace(dur=newdur)
            yield item
        self._durfunc = durfunc
        
    def take(self, num, flat=True):
        if flat:
            return Items(iterlib.take(self.flatiter(), num))
        else:
            return Items(iterlib.take(self, num))
        

# ------------------------------------------------------------
##
# The simplest Talea possible    
##
# ------------------------------------------------------------


class TaleaSimple(_TaleaProtocol):

    def __init__(self, talea, color):
        self.talea = talea
        self.color = color
        N = len(talea) * len(color) * 10
        N = min(N, 10000)
        xs = [0] + list(iterlib.take(N, iterlib.parsum(iterlib.cycle(talea))))
        indices = range(len(xs))
        self.bpf = _bpf.nointerpol(xs, indices)
        self.t0, self.t1 = self._bpf.bounds()

    def get_color_index(self, time):
        index = int(self.bpf(time))
        numcycle, rest = divmod(index, len(self.color))
        return rest

    def __call__(self, time):
        return self.color_at(time)

    def color_at(self, time):
        return self.color[self.get_color_index(time)]

    def take_between(self, t0, t1):
        """
        return a list of pairs (color, start time, dur)
        """
        xs = self._bpf.points()[0]
        xs_in_range = xs[(xs[1:] >= t0) * (xs[:-1] <= t1)]
        out = []
        for i in range(len(xs_in_range)):
            x0 = xs_in_range[i]
            try:
                x1 = xs_in_range[i + 1]
            except IndexError:
                x1 = t1
            x0, x1 = _lib.intersection(x0, x1, t0, t1)
            if x0 < x1:
                color = self.color_at((x0 + x1) * 0.5)
                out.append(Item(color=color, start=x0, dur=x1 - x0))
        return out

    @property
    def cycle_duration(self):
        return sum(self.talea)

    def set_cycle_duration(self, cycledur):
        """ 
        return a new TaleaColor with the same proportions as this one
        but with the given cycle duration
        """
        ratio = cycledur / self.cycle_duration
        new_talea = [d * ratio for d in self.talea]
        return Talea(new_talea, self.color)

    def __iter__(self):
        t = 0
        maxt = max(self.talea)
        while True:
            color, t0, dur = self.take_between(t, t + maxt)[0]
            yield Item(color=color, start=t, dur=dur)
            t += dur

# ------------------------------------------------------------
#
# Probability Talea    
#
# ------------------------------------------------------------


def probs(dictionary, n=None, shuffle=True, exclude=None, exclude_s=None, maxiter=1000):
    """
    given a dictionary with keys and probabilities, return 
    a sequence of n length with elements chosen from the 
    keys of the dictionary, according to the given probabilities

    exclude is a func receiving the possible output. should return True or False
    exclude_s is the same as exclude but receives the output as a joint string

    Example:

    >>> probs({'A':0.8, 'B':0.2}, 10)
    ['A', 'A', 'A', 'B', A', 'A', 'A', 'A', 'B', 'A']

    # Exclude some unwanted distributions
    >>> probs({'A':0.8, 'B':0.2}, 10, exclude_s=lambda s:("ABABA" not in s))
    """
    import bisect
    keys, probs = zip(*dictionary.iteritems())
    sumprobs = sum(probs)
    if n is None:
        n = sumprobs
    n = max(n, 1)
    relprobs = [prob / sumprobs for prob in probs]
    for _ in range(maxiter):
        stackedprobs = []
        x = 0
        for relprob in relprobs:
            stackedprobs.append(x)
            x += relprob

        def rand2key(x):
            index = bisect.bisect_right(stackedprobs, x) - 1
            return keys[index]
        out = []
        knums = [int(relprob * n + 0.5) for relprob in relprobs]
        if sum(knums) > n:
            while sum(knums) > n:
                i = _random.randint(0, len(knums) - 1)
                knums[i] -= 1
        elif sum(knums) < n:
            while sum(knums) < n:
                i = _random.randint(0, len(knums) - 1)
                knums[i] += 1
        for i, knum in enumerate(knums):
            ks = [keys[i]] * knum
            out.extend(ks)
        if shuffle:
            for i in range(n**2):
                _random.shuffle(out)
        isvalid = True
        if exclude is not None:
            isvalid = not exclude(out)
        if exclude_s is not None:
            isvalid = not exclude_s("".join(map(str, out)))
        if isvalid:
            break
    return out
