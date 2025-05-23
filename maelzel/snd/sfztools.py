"""
Utilities to edit and generate sfz files
"""
from __future__ import annotations
import os
import math
from maelzel.snd import audiosample
import pitchtools as pt
from emlib.iterlib import pairwise
import collections


def regionFromFile(sndfile: str, key='auto', offset=0, relpath=True, a4=442, **kws) -> str:
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
    conv = pt.PitchConverter(a4=a4)
    if key == 'auto':
        key = int(conv.f2m(estimateFreq(sndfile)) + 0.5)
    attrstr = " ".join([f"{k}={v}" % (k, v) for k, v in kws.items()])
    samplename = os.path.split(sndfile)[1] if relpath else sndfile
    if offset == 0:
        offsetstr = ""
    elif offset == 'auto':
        offsetstr = str(estimateOffset(sndfile))
    else:
        offsetstr = str(offset)
    out = f"<region> sample={samplename} key={key} {offsetstr} {attrstr}"
    return out.strip()


def regionsFromFiles(files: list[str],
                       key='auto',
                       offset='auto',
                       fillkeys=True,
                       relpath=True,
                       minkey=36,
                       skipbadestimations=True,
                       printregions=True,
                       a4=442,
                       offset_thresh=-55):
    """
    files: a list of soundfiles
    key: 'auto': read the file and determine the main pitch
         a function: it is called with the path, it returns a midinote
         an integer or string: the first note of the region,
         all other samples are assigned chormatically
         a list of integers or strings: each file is assigned a specific key
    """
    conv = pt.PitchConverter(a4=a4)
    if key == 'auto' and offset == 'auto':
        freqs_and_offsets = [estimateFreqAndOffset(f, minamp=offset_thresh)
                             for f in files]
        keys = [int(conv.f2m(freq) + 0.5) for freq, _ in freqs_and_offsets]
        offsets = [o for f, o in freqs_and_offsets]
    elif key == 'auto':
        keys = [int(conv.f2m(estimateFreq(f)) + 0.5) for f in files]
        offsets = [offset] * len(keys)
    elif offset == 'auto':
        keys = list(map(key, files))
        offsets = [estimateOffset(f) for f in files]
    else:
        if isinstance(key, int):
            keys = list(range(key, key + len(files)))
        elif isinstance(key, str):
            key = pt.n2m(key)
            keys = list(range(key, key + len(files)))
        elif callable(key):
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
            key2 = int(estimateFreq(f) + 0.5)
            if key2 < minkey:
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


def estimateFreq(sndfile: str, start=0., minfreq=40) -> float:
    """
    Estimate the freq. of source

    Args:
        sndfile (str): the path of the soundfile
        strategy (str): one of 'autocorr', 'fft'

    Returns:
        the estimated freq.
    """
    if start:
        s = audiosample.Sample(sndfile, end=start+0.5)
    else:
        s = audiosample.Sample(sndfile, end=4)
    f0time, f0freq = s.firstPitch(minfreq=minfreq)
    freq = s.fundamentalFreq(max(f0time+0.05, start))
    return 0. if freq is None else freq


def estimateOffset(sndfile: str, minamp=-80) -> float | None:
    """
    Args:
        sndfile: the path to a soundfile
        minamp: the min. amplitude to considere as sound

    Returns:
        the starting time of the first sound, in seconds
    """
    s = audiosample.Sample(sndfile)
    offset = s.firstSound(threshold=minamp)
    return offset


def estimateFreqAndOffset(sndfile: str, minamp=-80) -> tuple[float, float | None]:
    s = audiosample.Sample(sndfile)
    offset = s.firstSound(minamp)
    if offset is None:
        return 0., None
    freq = s.fundamentalFreq(offset+0.05)
    return freq if freq is not None else 0., offset


def log2(x: float) -> float:
    return math.log(x, 2)


def groupTemplate(name: str, release=0.1, loopmode='one_shot', **attrs):
    if attrs:
        attrstr = '\n'.join([f"{k}={v}" for k, v in attrs.items()])
    else:
        attrstr = ""
    return f"""\
<group>
name={name}
ampeg_release={release}
loop_mode={loopmode}
{attrstr}
"""
