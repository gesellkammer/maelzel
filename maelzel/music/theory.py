from emlib.music import core
from emlib.iterlib import combinations
from math import sqrt


def quantizedInterval(pitch1: core.pitch_t, pitch2: core.pitch_t) -> int:
    midi1 = core.asMidi(pitch1)
    midi2 = core.asMidi(pitch2)
    interval = round(abs(midi1 - midi2))
    return interval


def chordConsonance(*pitches):
    midinotes = [core.asMidi(p) for p in pitches]
    intervals = [quantizedInterval(m1, m2) for m1, m2 in combinations(midinotes, 2)]
    modintervals = [i%12 for i in intervals]
    modIntervalConsonance = {
        0: 10,
        7: 9,
        5: 7,
        4: 5,
        3: 5,
        9: 5,
        8: 5,
        2: 2,
        10: 2,
    }

    intervalHarmonicConsonance = {
        0: 10,
        12: 10,
        15: 2,
        16: 4,
        19: 8,
        24: 7,
        26: 1,
        28: 6,
        31: 5,
        34: 4,
        36: 4,
        38: 2
    }
    consonances = [modIntervalConsonance.get(modinterval, 0) for modinterval in modintervals]
    consonance = sum(consonances)
    harmonicConsonance = sum(intervalHarmonicConsonance.get(i, 0) for i in intervals)
    result = sqrt((consonance*4.0)**2 + (harmonicConsonance*1.0)**2)
    return result


