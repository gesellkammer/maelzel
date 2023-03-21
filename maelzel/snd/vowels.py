"""
Utilities to generate vocal and manipulate vocal sounds

"""
from __future__ import annotations
from math import pi, sqrt

import bpf4 as bpf
from pitchtools import *
from dataclasses import dataclass
import csoundengine

from emlib.iterlib import flatten
from emlib.misc import runonce

from maelzel.core import Chord, Note
from maelzel.snd import playback

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


@dataclass
class Vowel:
    dbs: List[float]
    bws: List[float]
    freqs: List[float]

    def shift(self, freq):
        freqs = [formantfreq + freq for formantfreq in self.freqs]
        return Vowel(dbs=self.dbs, bws=self.bws, freqs=freqs)

    def transpose(self, interval):
        ratio = interval2ratio(interval)
        freqs = [f * ratio for f in self.freqs]
        return Vowel(dbs=self.dbs, bws=self.bws, freqs=freqs)

    def scalebw(self, factor):
        bws = [bw * factor for bw in self.bws]
        return Vowel(bws=bws, dbs=self.dbs, freqs=self.freqs)


def getVowel(descr: Union[str, Tuple[str, str]]) -> Vowel:
    """
    Args:
        descr: a string of the form "vocal:register", like "i:bass", or
               a tuple ('i', 'bass')
    Returns:
        the corresponding Vowel obj
    """
    if isinstance(descr, str):
        vowel, register = descr.split(":")
    elif isinstance(descr, tuple):
        vowel, register = descr
    else:
        raise TypeError(
            "descr should be a str like 'i:bass' or a tuple like ('i', 'bass')"
        )
    return formants[vowel][register]


# The praat data was taken by synthesizing a vowel and analyzing it later
# The bandwidth for the third formant gives in some very big values, these
# have been reduced. The analysis does not yield intesity values for the
# bandwidths.

formants: Dict[str, Dict[str, Vowel]] = {
    'e': {
        'bass':
        Vowel(dbs=[0, -12, -9, -12, -18],
              bws=[40, 80, 100, 120, 120],
              freqs=[400, 1620, 2400, 2800, 3100]),
        'counterten':
        Vowel(dbs=[0, -14, -18, -20, -20],
              bws=[70, 80, 100, 120, 120],
              freqs=[440, 1800, 2700, 3000, 3300]),
        'soprano':
        Vowel(dbs=[0, -20, -15, -40, -56],
              bws=[60, 100, 120, 150, 200],
              freqs=[350, 2000, 2800, 3600, 4950]),
        'alto':
        Vowel(dbs=[0, -24, -30, -35, -60],
              bws=[60, 80, 120, 150, 200],
              freqs=[400, 1600, 2700, 3300, 4950]),
        'tenor':
        Vowel(dbs=[0, -14, -12, -14, -20],
              bws=[70, 80, 100, 120, 120],
              freqs=[400, 1700, 2600, 3200, 3580]),
        'praat-male':
        Vowel(dbs=[0, -12, -9, -12, -60],
              bws=[90, 190, 220, 260, 120],
              freqs=[432, 1895, 2170, 2600, 3580]),
        'male':
        Vowel(dbs=[0, -12, -9, -12, -60],
              bws=[90, 190, 220, 260, 120],
              freqs=[432, 1895, 2170, 2600, 3580])
    },
    'o': {
        'bass':
        Vowel([0, -11, -21, -20, -40], [40, 80, 100, 120, 120],
              [400, 750, 2400, 2600, 2900]),
        'counterten':
        Vowel([0, -10, -26, -22, -34], [40, 80, 100, 120, 120],
              [430, 820, 2700, 3000, 3300]),
        'soprano':
        Vowel([0, -11, -22, -22, -50], [70, 80, 100, 130, 135],
              [450, 800, 2830, 3800, 4950]),
        'alto':
        Vowel([0, -9, -16, -28, -55], [70, 80, 100, 130, 135],
              [450, 800, 2830, 3500, 4950]),
        'tenor':
        Vowel([0, -10, -12, -12, -26], [40, 80, 100, 120, 120],
              [400, 800, 2600, 2800, 3000]),
        'praat-male':
        Vowel([0, -10, -12, -12, -60], [107, 94, 483, 224, 120],
              [540, 940, 2050, 2570, 3000]),
        # 'male':       Vowel([0, -10, -12, -12, -26], [40, 80, 100, 120, 120], [400, 800, 2600, 2800, 3000])
        'male':
        Vowel([0, -10, -12, -12, -43], [73, 87, 291, 172, 120],
              [470, 870, 2325, 2685, 3000])
    },
    'a': {
        'bass':
        Vowel([0, -7, -9, -9, -20], [60, 70, 110, 120, 130],
              [600, 1040, 2250, 2450, 2750]),
        'counterten':
        Vowel([0, -6, -23, -24, -38], [80, 90, 120, 130, 140],
              [660, 1120, 2750, 3000, 3350]),
        'soprano':
        Vowel([0, -6, -32, -20, -50], [80, 90, 120, 130, 140],
              [800, 1150, 2900, 3900, 4950]),
        'alto':
        Vowel([0, -4, -20, -36, -60], [80, 90, 120, 130, 140],
              [800, 1150, 2800, 3500, 4950]),
        'tenor':
        Vowel([0, -6, -7, -8, -22], [80, 90, 120, 130, 140],
              [650, 1080, 2650, 2900, 3250]),
        'praat-male':
        Vowel([0, -6, -7, -8, -60], [82, 130, 350, 260, 120],
              [825, 1300, 2100, 2580, 3000]),
        'male':
        Vowel([0, -6, -7, -8, -41], [81, 110, 235, 195, 130],
              [737, 1190, 2375, 2740, 3125])
    },
    'u': {
        'bass':
        Vowel([0, -20, -32, -28, -36], [40, 80, 100, 120, 120],
              [350, 600, 2400, 2675, 2950]),
        'counterten':
        Vowel([0, -20, -23, -30, -34], [40, 60, 100, 120, 120],
              [370, 630, 2750, 3000, 3400]),
        'soprano':
        Vowel([0, -16, -35, -40, -60], [50, 60, 170, 180, 200],
              [325, 700, 2700, 3800, 4950]),
        'alto':
        Vowel([0, -12, -30, -40, -64], [50, 60, 170, 180, 200],
              [325, 700, 2530, 3500, 4950]),
        'tenor':
        Vowel([0, -20, -17, -14, -26], [40, 60, 100, 120, 120],
              [350, 600, 2700, 2900, 3300]),
        'praat-male':
        Vowel([0, -12, -26, -20, -60], [80, 63, 250, 180, 220],
              [366, 733, 1855, 2540, 3540]),
        'male':
        Vowel([0, -12, -26, -20, -60], [80, 63, 250, 180, 220],
              [366, 733, 1855, 2540, 3540])
    },
    'i': {
        'bass':
        Vowel([0, -30, -16, -22, -28], [60, 90, 100, 120, 120],
              [250, 1750, 2600, 3050, 3340]),
        'counterten':
        Vowel([0, -24, -24, -36, -36], [40, 90, 100, 120, 120],
              [270, 1850, 2900, 3350, 3590]),
        'soprano':
        Vowel([0, -12, -26, -26, -44], [60, 90, 100, 120, 120],
              [270, 2140, 2950, 3900, 4950]),
        'alto':
        Vowel([0, -20, -30, -36, -60], [50, 100, 120, 150, 200],
              [350, 1700, 2700, 3700, 4950]),
        'tenor':
        Vowel([0, -15, -18, -20, -30], [40, 90, 100, 120, 120],
              [290, 1870, 2800, 3250, 3540]),
        'praat-male':
        Vowel([0, -12, -18, -20, -60], [100, 230, 450, 260, 220],
              [290, 1926, 2297, 2610, 3540]),
        'male':
        Vowel([0, -17, -20, -25, -42], [60, 136, 216, 166, 153],
              [283, 1882, 2665, 3070, 3556])
    }
}


def _listmul(seq, scalar):
    return [item * scalar for item in seq]


def _sumcolumns(rows):
    return [sum(row[i] for row in rows) for i in range(len(rows[0]))]


def interpolateVowel(vowels: List[str], weights:List[float]=None) -> Vowel:
    """
    Create a mix between different vowels according to the weights given
    If no weights are given, all vowels are mixed equally

    Args:
        vowels: a str of the form vowel:register, like "i:bass"
        weights: if given, a numerical weight for each vowel.
    """
    if weights is None:
        weights = [1] * len(vowels)
    if isinstance(vowels[0], str):
        vowels = [getVowel(vowel) for vowel in vowels]
    sumweights = sum(weights)
    normalizedWeights = [w / sumweights for w in weights]
    rows = list(zip(vowels, normalizedWeights))
    avgfreqs = _sumcolumns(
        [_listmul(vowel.freqs, weight) for vowel, weight in rows])
    avgdbs = _sumcolumns(
        [_listmul(vowel.dbs, weight) for vowel, weight in rows])
    avgbws = _sumcolumns(
        [_listmul(vowel.bws, weight) for vowel, weight in rows])
    return Vowel(dbs=avgdbs, bws=avgbws, freqs=avgfreqs)


def vowelInstr(kind='fof2') -> csoundengine.instr.Instr:
    """
    Args:
        kind: one of 'fof2', TODO

    Returns:
        the csoundengine Instr
    """
    if kind == 'fof2':
        return _vowelInstrFof2()
    else:
        raise KeyError(f"no instr with kind {kind}")


@runonce
def _vowelInstrFof2():
    body = """
        itotdur = p3
        iamp = p4
        imidi0 = p5
        imidi1 = p6
        ivibrate = p7
        ivibinterval = p8
        iform1 = p9
        iband1 = p10
        iamp1  = p11
        iform2 = p12
        iband2 = p13
        iamp2  = p14
        iform3 = p15
        iband3 = p16
        iamp3  = p17
        iform4 = p18
        iband4 = p19
        iamp4  = p20
        iform5 = p21
        iband5 = p22
        iamp5  = p23

        ivibamount = 2 ^ (-ivibinterval / 12)   ; vibrato should be always downwards

        iBurstRise init 0.003
        iBurstDur init 0.020
        iBurstDec init 0.007

        kmidi linseg imidi0, p3, imidi1
        kfund mtof kmidi
        ; avib = oscili:a(ivibamount*0.5, ivibrate) - (ivibamount*0.5)
        avib0 = oscili:a(1, ivibrate)
        avib = (avib0 + 1) / 2 * (1 - ivibamount) + ivibamount
        iolaps = 200
        ifna = -1
        ; ifnb = gi_fof2rise
        ifnb ftgenonce 0, 0, 4096, 7, 0, 4096, 1
        koct init 0
        kgliss init 1
        afund = avib * kfund
        a1  fof2 iamp1, afund, iform1, koct, iband1, iBurstRise, iBurstDur, iBurstDec, iolaps, ifna, ifnb, itotdur, rnd(1), kgliss
        a2  fof2 iamp2, afund, iform2, koct, iband2, iBurstRise, iBurstDur, iBurstDec, iolaps, ifna, ifnb, itotdur, rnd(1), kgliss
        a3  fof2 iamp3, afund, iform3, koct, iband3, iBurstRise, iBurstDur, iBurstDec, iolaps, ifna, ifnb, itotdur, rnd(1), kgliss
        a4  fof2 iamp4, afund, iform4, koct, iband4, iBurstRise, iBurstDur, iBurstDec, iolaps, ifna, ifnb, itotdur, rnd(1), kgliss
        a5  fof2 iamp5, afund, iform5, koct, iband5, iBurstRise, iBurstDur, iBurstDec, iolaps, ifna, ifnb, itotdur, rnd(1), kgliss
        aout = a1 + a2 + a3 + a4 + a5
        aenv linsegr 0, 0.05, iamp, 0.1, 0
        aout *= aenv
        outs aout, aout
    """
    instr = csoundengine.instr.Instr('vowelsfof2', body)
    return instr


def instrData(vowel: Union[str, Vowel]) -> List[float]:
    vowel = asVowel(vowel)
    amps = [db2amp(db) for db in vowel.dbs]
    data = list(flatten(zip(vowel.freqs, vowel.bws, amps)))
    return data


def _synthVowelFof2(engine: csoundengine.Engine,
                    midinote: Union[float, Tuple[float, float]],
                    vowel: Union[str, Vowel],
                    dur:float,
                    vibrate=0.,
                    vibamount=0.25,
                    gain=1.0
                    ) -> csoundengine.synth.AbstrSynth:
    """
    Synthesize the vowel via csoundengine

    Args:
        midinote: either a midinote as float, or a tuple (start, end)
        vowel: the vowel to synthesize
        dur: the totalDuration
        vibrate: the vibrato rate
        vibamount: how much vibrato
        gain: the gain of the note

    Returns:
        the synth
    """
    midi0, midi1 = midinote if isinstance(midinote, tuple) else (midinote, midinote)
    vowel = asVowel(vowel)
    data = instrData(vowel)
    args = [gain, midi0, midi1, vibrate, vibamount] + data
    session = engine.session()
    instr = _vowelInstrFof2()
    session.registerInstr(instr)
    synth = session.sched(instr.name, 0, dur=dur, args=args)
    return synth


def synthVowel(midinote: Union[float, Tuple[float, float]],
               vowel: Union[str, Vowel],
               dur=4.,
               gain=1.0,
               method='fof2',
               vibrate=0.,
               vibamount=0.25,
               enginename: str = ''
               ) -> csoundengine.synth.AbstrSynth:
    """
    Synthesize a vowel

    Args:
        midinote: the pitch to synthesize, either a single pitch or a glissando if a tuple
            (start pitch, end pitch) is given
        vowel: the vowel to use
        dur: the totalDuration
        gain: gain
        method: 'fof2' at the moment
        vibrate: vibrato rate
        vibamount: amount of vibrato

    Returns:
        the synth

    """
    if enginename:
        engine = csoundengine.getEngine(enginename)
    else:
        engine = playback.getEngine()
    if method == 'fof2':
        return _synthVowelFof2(engine,
                               midinote,
                               vowel,
                               dur,
                               gain=gain,
                               vibamount=vibamount,
                               vibrate=vibrate)
    else:
        raise ValueError(
            f"method {method} not supported. It should be one of ['noise'] ")


# ------------------------------------------------------------


def _overtones_tri(f0, maxfreq):
    """
    It is possible to approximate a triangle wave with additive synthesis by summing
    odd harmonics of the fundamental while multiplying every other odd harmonic by −1
    (or, equivalently, changing its phase by π) and multiplying the amplitude of the
    harmonics by one over the square of their mode number, n, (which is equivalent
    to one over the square of their relative frequency to the fundamental)
               1
    An = ---------------
           (fn / f0)**2
    """
    overtones = []
    for n in range(1, 10000000, 2):
        fn = f0 * n
        if fn > maxfreq:
            break
        an = 1 / ((fn / f0)**2)
        overtones.append((fn, an))
    return overtones


def _overtones_saw(f0: float, maxfreq: float) -> List[Tuple[float, float]]:
    """
    Calculate overtones for a saw signal

    A saw waveform includes all harmonics, Amp_n = 1/n

    Args:
        f0: fundamental
        maxfreq: max. freq to calculate

    Returns:
        a list of tuples (freq, amp)

    """
    overtones = []
    for n in range(1, 1000000, 1):
        fn = f0 * n
        if fn > maxfreq:
            break
        an = 1 / n
        overtones.append((fn, an))
    return overtones


def _overtones_square(f0, maxfreq):
    overtones = []
    for n in range(1, 1000000, 2):
        fn = f0 * n
        if fn > maxfreq:
            break
        # an = 2/(pi*n)
        an = 4 / pi * 1 / n
        overtones.append((fn, an))
    return overtones


def _overtones_const(f0, maxfreq):
    overtones = []
    for n in range(1, 1000000):
        fn = f0 * n
        if fn > maxfreq:
            break
        overtones.append((fn, 1))
    return overtones


def makeOvertones(f0: float, model='saw', maxfreq=8000
                  ) -> List[Tuple[float, float]]:
    """
    Generate overtones for the given fundamental

    Args:
        f0: the fundamental
        model: one of 'saw', 'tri', 'square', 'const' (all overtones have the
            same amplitude)
        maxfreq: max. frequency

    Returns:
        a list of tuples (freq, amplitude) for each overtone generated
    """
    if model == 'saw':
        pairs = _overtones_saw(f0, maxfreq=maxfreq)
    elif model == 'tri':
        pairs = _overtones_tri(f0, maxfreq=maxfreq)
    elif model == 'square':
        pairs = _overtones_square(f0, maxfreq=maxfreq)
    elif model == 'const':
        pairs = _overtones_const(f0, maxfreq=maxfreq)
    else:
        raise ValueError(f"model '{model}' not known")
    return pairs


def findVowel(freqs: List[float]) -> str:
    """
    Find the vowel whose formant freqs are nearest to the freqs given
    
    Args:
        freqs: formant frequencies
        
    Returns:
        a string describing the vowel with the format {vowel}:{register} 
        ("a:tenor")
    """

    def vowelDistance(voweldef: Vowel, freqs: List[float]) -> float:
        vowelfreqs = voweldef.freqs[:len(freqs)]
        dist = sqrt(
            sum((f2m(vowelfreq) - f2m(freq))**2
                for vowelfreq, freq in zip(vowelfreqs, freqs)))
        return dist

    results = []
    for vowel, registers in formants.items():
        for register, voweldef in registers.items():
            dist = vowelDistance(voweldef, freqs)
            results.append((dist, vowel, register))
    results.sort()
    _, vowel, register = results[0]
    return f"{vowel}:{register}"


def asVowel(vowel: Union[str, Vowel]) -> Vowel:
    """
    Converts a string to a Vowel.

    If a Vowel is passed, it is returned as is

    Args:
        vowel: a Vowel or a vowel description in the format
            '<vowel>:<voice>', like 'i:bass'

    Returns:
        the Vowel
    """
    if isinstance(vowel, Vowel):
        return vowel
    elif isinstance(vowel, str):
        return getVowel(vowel)
    else:
        raise TypeError(
            f"expected a Vowel or a vowel desc. (like 'i:bass'), but got {vowel}"
        )


@runonce
def listVowels():
    allvowels = []
    for voweltype, voices in formants.items():
        vowels = [f'{voweltype}:{voicetype}' for voicetype in voices.keys()]
        allvowels.extend(vowels)
    allvowels.sort()
    return allvowels


class VowelFilter:
    def __init__(self, vowel:Union[str, Vowel], bwmul=1.0, gain=1.0, bwknee=0.1) -> None:
        """
        a VowelFilter uses a vowel definition to generate a spectral surface,
        mapping frequency to gain

        Args:
            vowel: a Vowel or a vowel description
            bwmul: a multiplier to the bandwidths defined in a Vowel
            bwknee: as a ratio of the bandwidth
        """
        self.vowel = asVowel(vowel)
        self.bwmul = bwmul
        self.gain = gain
        self.bwknee = bwknee
        self.curve = self.makeCurve()

    def ampAt(self, freq: float) -> float:
        return self.curve(freq)

    def makeCurve(self) -> bpf.BpfInterface:
        numFormants = len(self.vowel.freqs)
        curves = []
        transition = 1 - self.bwknee
        for i in range(numFormants):
            freq = self.vowel.freqs[i]
            amp = db2amp(self.vowel.dbs[i]) * self.gain
            bw = self.vowel.bws[i] * self.bwmul
            bw2 = bw * 0.5
            transitionbw = bw2 * transition
            b = bpf.halfcos(0, 0, freq - bw2 - transitionbw, 0, freq - bw2,
                            amp, freq + bw2, amp, freq + bw2 + transitionbw, 0)
            curves.append(b)
        return bpf.core.Max(*curves)

    def filter(self, overtones: List[Tuple[float, float]], mindb=-90, wet=1.0):
        """
        Returns a Chord with the filtered overtones

        Args:
            overtones: as generated via makeOvertones (a list of tuples (freq, amp))
            mindb: min. amplitude to be present in the result
            wet: the result is a combination of the original overtones and the
                filtered result. When wet is 1, the result is composed only of
                the filtered result.
        
        Returns:
            a Chord with the filtered overtones
        """
        return vocalChord(overtones, self, mindb=mindb, wet=wet)


def vocalChordFromF0(f0, vowel, model='saw', mindb=-90, wet=1.0):
    overtones = makeOvertones(f0=f0, model=model)
    vowelfilter = VowelFilter(asVowel(vowel))
    return vocalChord(overtones=overtones,
                      vowelfilter=vowelfilter,
                      mindb=mindb,
                      wet=wet)


def vocalChord(overtones: List[Tuple[float, float]],
               vowelfilter: Union[VowelFilter, str],
               mindb=-90,
               wet=1.0) -> Chord:
    """
    Args:
        overtones: a list of (freq, amplitude), as returned, for instance, by makeOvertones
        vowelfilter: a VowelFilter or a string description of a vowel (eg: "a:male")
        mindb: after filtering only overtones with a min. db of mindb will be kept

    See Also
    ~~~~~~~~

    vocalChordFromF0

    Example
    ~~~~~~~

        >>> exciter = makeOvertones(50, model='saw')
        >>> filter = VowelFilter(asVowel("a:male")
        >>> chord = vocalChord(exciter, filter, mindb=-60)
        >>> chord.show()
        >>> chord.clone(dur=10).play(instr='sin')
    """
    if isinstance(vowelfilter, str):
        vowelfilter = VowelFilter(asVowel(vowelfilter))
    pairs = []
    for freq, amp in overtones:
        ampwet = vowelfilter.ampAt(freq) * amp
        amp2 = amp * (1 - wet) + ampwet * wet
        if amp2db(amp2) > mindb:
            pairs.append((freq, amp2))
    return Chord([Note(f2m(freq), amp=amp) for freq, amp in pairs])
