from collections import deque
from emlib import iterlib
from emlib import distribute
from emlib.music.misc import split_mask, normalize_mask


def morph(streams, curve, numsteps, hook=None):
    """
    streams: a seq of iterators yielding one value per iteration
    curve: a curve x->any interval, y->(0, len(streams)-1)
    numsteps: the number of steps to apply the curve to
    hook: if given, a function (stepnum, state) -> index, will
          override the curve

    curve is used to index the streams
    """
    indexes = distribute.dither_curve(curve, numsteps)
    out = []
    seqs = [iterlib.take(stream, numsteps) for stream in streams]
    for i, index in enumerate(indexes):
        if hook:
            index = hook(i, out)
        seq = seqs[index]
        value = seq[i]
        out.append(value)
    return out


def masked(stream, mask):
    """
    A masked stream

    stream: a seq. of elements
    mask: a seq. of bools. or a string like ("--x-x-")

    Example:

    masked(range(20), (0, 1)) -> yields the even numbers between 0 and 20
    """
    mask = normalize_mask(mask)
    maskstream = iterlib.cycle(mask)
    for x, maskvalue in zip(stream, maskstream):
        if maskvalue:
            yield x


def traverse(seq, chunks, advance=1):
    """
    seq:
      a seq of anything (colors, durations)

    chunks:
      an iterator (possibly cyclic) yielding the size of each chunk or 
      a function like (cursor) -> chunk

      A chunk is a size (how many elements from seq at cursor to take)
      or a tuple (size, mask) where mask indicates which elements to take    

    advance:
      how much to advance after each chunk. A value or an iterator
    
    seq = [7, 6, 5, 4, 3, 2, 1]

    def genchunks():
        masks = {
            3: ['xx-', 'xxx'],
            4: ['xxx-', 'x-xx']
        }
        while True:
            chunksize = random.choice([3, 4])
            mask = random.choice(masks[chunksize])
            yield chunksize, mask

    traverse(seq, chunks=genchunks())
    --> [[7, 6, 4], [6, 5, 4], [5, 4], [4, 2, 1], [3, 1], [2, 1]]
    """
    cursor = 0
    out = []
    if not callable(chunks):
        chunks = iter(chunks)
        chunksfunc = lambda cursor: next(chunks)
    else:
        chunksfunc = chunks
    it_advance = iter(advance) if hasattr(advance, '__iter__') else iter(iterlib.repeat(advance))

    def parsemask(obj):
        if isinstance(obj, tuple):
            value, mask = obj
            mask = [1 if x in (1, "x") else 0 for x in mask.lower()]
        else:
            value = obj
            mask = [1]*value
        return value, mask

    while True:
        chunksize, chunkmask = split_mask(chunksfunc(cursor))
        if chunkmask is None:
            chunkmask = [1] * chunksize
        chunksize = min(chunksize, len(seq) - cursor)
        if chunksize <= 0:
            break
        chunkmask = chunkmask[:chunksize] 
        assert len(chunkmask) == chunksize
        chunk = [seq[cursor+i] for i, maskvalue in enumerate(chunkmask) if maskvalue == 1]
        out.append(chunk)
        cursor += next(it_advance)
    return out


class Zip(object):
    def __init__(self, streams, constraintfunc=None, histsize=20):
        self.streams = streams
        self.constraintfunc = constraintfunc
        self.history = deque([], histsize)

    def __iter__(self):
        constraintfunc = self.constraintfunc
        for items in zip(*self.streams):
            if constraintfunc is None:
                yield items
                continue
            items = constraintfunc(items, self.history)
            if items is not None:
                yield items
                self.history.append(items)
