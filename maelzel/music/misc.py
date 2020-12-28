from __future__ import division as _div
from emlib.misc import returns_tuple as _returns_tuple
from maelzel.common import *

# #------------------------------------------------------------
# #
# #    GENERAL ROUTINES    
# #
# #------------------------------------------------------------


def _is_pandas_dataframe(obj):
    attrs = ["ix", "index", "start", "dur", "pitch"]
    return all(hasattr(obj, attr) for attr in attrs)


@_returns_tuple("starts durs pitches")
def normalize_frames(frames):
    """
    Converts a seq of Frames or a pandas.DataFrame to a list of starts, durations and pitches

    frames: a seq. of Frames or a pandas.DataFrame with columns (start, dur, pitch)
            where Frame is a namedtuple(..., "start dur pitch)

    SEE ALSO: lib.dataframe2namedtuple
    """
    # is it a seq. of namedtuples?
    if isinstance(frames, (tuple, list)) and isframe(frames[0]):
        starts, durs, pitches = [], [], []
        for frame in frames:
            starts.append(frame.start)
            durs.append(frame.dur)
            pitches.append(frame.pitch)
    elif _is_pandas_dataframe(frames):
        starts = list(frames.start)
        durs = list(frames.dur)
        pitches = list(frames.pitch)
    else:
        raise ValueError("frames not of a suitable format")
    return starts, durs, pitches


def isframe(obj):
    """
    Does obj respond to the Frame protocol ("start", "dur", "pitch")
    """
    return (isinstance(obj, tuple) and 
            all(hasattr(obj, a) for a in ("start", "dur", "pitch")))


def split_mask(obj, defaultmask=None):
    """
    A masked sequence is a tuple of the sort (value, mask)
    where:
        value can be anything
        mask is a seq. or a string of the sort
            [1, 1, 0, 1], "xx-x", "1101", "XX_X", etc

    This is used in many places to represent selecting objects from
    a sequence, like a talea-color or a markov chain
    
    NB: if obj is not a tuple, it is assumed that there is no mask, and 
        None is returned as the mask
    """
    if isinstance(obj, tuple):
        value, mask = obj
        mask = normalize_mask(mask)
    else:
        value = obj
        mask = defaultmask
    return value, mask


def normalize_mask(mask, truevalues=(1, "x")):
    """
    convert a string mask into a binary mask

    Possible string representations
    XX--X, xx--x, xxoxo
    """
    return [1 if x in truevalues else 0 for x in mask.lower()]


del U
