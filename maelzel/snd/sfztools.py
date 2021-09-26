"""
Utilities to edit and generate sfz files
"""
from __future__ import annotations
import os
import math
from maelzel.snd import audiosample as smpl
from pitchtools import n2m, f2m
from emlib.iterlib import pairwise
import collections
from typing import Tuple, List


def region_from_file(sndfile: str, key='auto', offset=0, relpath=True, **kws) -> str:
    """
    Returns a <region> definition as a string

    Args:
        sndfile: the path of the soundfile
        key: the note to assign to this sample, or 'auto' if the pitch of the sample
             is to be used. In this case, the frequency is estimated and the
             closest midi-note is used. key can also be a function, in which case
             it is called with the given path. It should return a midinote.
        offset: an offset in samples or 'auto' to calculate it
        relpath: use a relative path

    Returns:
        the <region> definition string
    """
    if key == 'auto':
        key = int(f2m(estimate_freq(sndfile)) + 0.5)
    key_str = str(key)
    attrs = ["%s=%s" % (k, v) for k, v in kws.items()]
    attr_str = " ".join(attrs)
    if relpath:
        samplename = os.path.split(sndfile)[1]
    else:
        samplename = sndfile
    if offset == 0:
        offset_str = ""
    elif offset == 'auto':
        offset_str = str(estimate_offset(sndfile))
    else:
        offset_str = str(offset)
    out = "<region> sample=%s key=%s %s %s" % (samplename, key_str, offset_str,
                                               attr_str)
    return out.strip()


def regions_from_files(files: List[str],
                       key='auto',
                       offset='auto',
                       fillkeys=True,
                       relpath=True,
                       minkey=36,
                       skipbadestimations=True,
                       printregions=True,
                       offset_thresh=-55):
    """
    files: a list of soundfiles
    key: 'auto': read the file and determine the main pitch
         a function: it is called with the path, it returns a midinote
         an integer or string: the first note of the region, 
         all other samples are assigned chormatically
         a list of integers or strings: each file is assigned a specific key
    """
    if key == 'auto' and offset == 'auto':
        freqs_and_offsets = [estimate_freq_and_offset(f, minamp=offset_thresh)
                             for f in files]
        keys = [int(f2m(freq) + 0.5) for freq, _ in freqs_and_offsets]
        offsets = [o for f, o in freqs_and_offsets]
    elif key == 'auto':
        keys = [int(f2m(estimate_freq(f)) + 0.5) for f in files]
        offsets = [offset] * len(keys)
    elif offset == 'auto':
        keys = list(map(key, files))
        offsets = [estimate_offset(f) for f in files]
    else:
        if isinstance(key, int):
            keys = list(range(key, key + len(files)))
        elif isinstance(key, str):
            key = n2m(key)
            keys = list(range(key, key + len(files)))
        elif isinstance(key, collections.Callable):
            keys = list(map(key, files))
        else:
            keys = key
            assert len(keys) == len(files)
        offsets = [offset] * len(keys)
    if relpath:
        samples = [os.path.split(f)[1] for f in files]
    else:
        samples = files
    i = 0
    for f, key in zip(files, keys):
        if key < minkey:
            print("error in the frequency estimation: key of %s out of range"
                  "Trying another strategy"
                  % f)
            key2 = int(estimate_freq(f, strategy='fft') + 0.5)
            if key2 < minkey:
                print("    no luck...")
                if not skipbadestimations:
                    raise RuntimeError(
                        "could not estimate the frequency of %s, aborting"
                        % f)
                else:
                    print("estimation failed, skipping...")
                    key2 = -1
            if key2 >= minkey:
                keys[i] = key2
        i += 1
    rows = sorted(zip(samples, keys, offsets),
                  key=lambda sample_key_offset: sample_key_offset[1])
    regions = []
    if not fillkeys:
        raise ValueError("XXX fix this")
        # for sample, key, offset in samples_and_keys:
        #     regions.append("<region> sample=%s key=%d offset=%d" %
        #                    (sample, key, offset))
    else:
        samples, keys, offsets = list(zip(*rows))
        avgs = [int((key1 + key0) / 2. + 0.5) for key0, key1 in pairwise(keys)]
        lokeys = [keys[0] - (avgs[0] - keys[0])] + avgs
        centers = keys
        hikeys = avgs + [keys[-1] + (keys[-1] - avgs[-1])]
        hikeys = [max(center, k - 1) for k, center in zip(hikeys, centers)]
        for sample, offset, center, lokey, hikey in zip(
                samples, offsets, centers, lokeys, hikeys):
            assert lokey <= center <= hikey
            if offset is None:
                offset = 0
            region = ("<region> sample=%s offset=%d pitch_keycenter=%d "
                      "lokey=%d hikey=%d"
                      % (sample, offset, center, lokey, hikey))
            regions.append(region)
    if printregions:
        print("\n".join(regions))
    return regions


def normalize_filename(path: str) -> str:
    return path.replace(" ", "_").replace("__", "_")


def estimate_freq(sndfile: str, strategy='autocorr') -> float:
    """
    Estimate the freq. of source

    Args:
        sndfile (str): the path of the soundfile
        strategy (str): one of 'autocorr', 'fft'

    Returns:
        the estimated freq.
    """
    s = smpl.Sample(sndfile)
    return s.estimateFreq(start=0.1, strategy=strategy)


def estimate_offset(sndfile: str, minamp=-80) -> float:
    """
    Args:
        sndfile: the path to a soundfile
        minamp: the min. amplitude to considere as sound

    Returns:
        the starting time of the first sound, in seconds
    """
    s = smpl.Sample(sndfile)
    offset = s.firstSound(threshold=minamp)
    return offset


def estimate_freq_and_offset(sndfile: str, minamp=-80) -> Tuple[float, float]:
    s = smpl.Sample(sndfile)
    freq = s.estimateFreq()
    offset = s.firstSound(minamp)
    return freq, offset


def log2(x:float) -> float:
    return math.log(x, 2)


def group_template(name=None, release=0.1, loopmode='one_shot', **attrs):
    if attrs:
        attr_strs = ["%s=%s" % (k, str(v)) for k, v in attrs.items()]
        attr_str = '\n'.join(attr_strs)
    else:
        attr_str = ""
    s = """<group>
name=%s
ampeg_release=%s
loop_mode=%s
%s
""" % (name, release, loopmode, attr_str)
    return s
