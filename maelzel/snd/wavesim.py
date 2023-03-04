import os
import itertools
import logging
from math import sqrt

from maelzel.snd.audiosample import Sample
from pitchtools import db2amp
import csoundengine
from csoundengine import csoundlib
from maelzel.snd import vowels


from typing import List

logger = logging.getLogger("maelzel.wavesim")


ENGINE = 'maelzel.wavesim'


def _getEngine() -> csoundengine.Engine:
    return csoundengine.getEngine(ENGINE)


class Instr:
    def __init__(self, name, instrBody, instrInit, nchnls, args=None, minDur=0.1, simTime=0.5):
        self.name = name
        self.instrBody = instrBody
        self.instrInit = instrInit
        self.nchnls = nchnls
        self.args = args if args is not None else []
        self.minDur = minDur
        self.simTime = simTime
        self._csoundInstr = None

    def getArgs(self, gain=1) -> List[float]:
        """
        Returns the pargs passed to the instr when scheduled
        These are p4, p5, ...  (if any)

        p2 and p3 are not needed
        """
        args = self.args
        if gain != 1:
            args = [args[0] * gain] + args[1:]
        return args 
    
    def _chordTime(self) -> float:
        return self.simTime * 0.5

    def makeInstr(self):
        """
        Creates a CsoundInstr out of this Instr
        """
        if self._csoundInstr is None:
            logger.debug(f"creating CsoundInstr {self.name}")
            session = _getEngine().session()
            self._csoundInstr = session.defInstr(name=self.name, body=self.instrBody,
                                                 init=self.instrInit)
        return self._csoundInstr

    def _getEvents(self, dur, gain=1, delay=0):
        args = self.getArgs(gain=gain)
        event = [delay, dur] + args
        return [event]

    def rec(self, dur, outfile:str=None, sr=44100, ksmps=64, block=True) -> str:
        """
        Record this Instr, return the soundfile generated

        Args:
            dur: the totalDuration of the recording
            outfile: if given, the path to the generated soundfile.
                Otherwise a temp. file is generated
            sr: the sr of the sound
            ksmps: the ksmps used
            block: if True, wait until recording is finished

        Returns:
            the path of the recording
        """
        dur = max(dur, self.minDur)
        events = self._getEvents(dur=dur, gain=gain)
        outfile, popen = csoundlib.recInstr(body=self.instrBody, init=self.instrInit, outfile=outfile,
                                            events=events, sr=sr, ksmps=ksmps, nchnls=self.nchnls)
        if block:
            popen.wait()
        return outfile

    def makeSample(self, dur:float) -> Sample:
        """
        Run this Instr offline, return the samples as a Sample
        """
        outfile = self.rec(dur)
        sample = Sample(outfile)
        os.remove(outfile)
        return sample

    def play(self, dur, gain=1):
        instr = self.makeInstr()
        args = self.getArgs(gain=gain)
        return instr.play(dur=dur, args=args)


class Vowel(Instr):
    def __init__(self, midinote, vowel, gain=1, vibrate=5.3, vibamount=0, method="fof2"):
        vowel = vowels.asVowel(vowel)
        instr = vowels.vowelInstr(method)
        data = vowels.instrData(vowel)
        midi0, midi1 = midinote if isinstance(midinote, tuple) else (midinote, midinote)
        args = [gain, midi0, midi1, vibrate, vibamount] + data
        super().__init__(name='wavesim.vowel',
                         instrBody=instr.body,
                         instrInit=instr.initcode,
                         nchnls=1,
                         args=args)


class FreqMod(Instr):

    def __init__(self, freq, modmul, modindex=1, carmul=1.0, amp=1.0):
        """
        Frequency Modulation

        We define the carrier and modulating signals as two sine waves which share a common
        frequency.

        carrier_freq = freq * carmul
        mod_freq = freq * modmul

        freq     : base frequency
        carmul   : carrier_freq = freq * carmul. Normally this is 1
        modmul   : modulating signal multiplier, modfreq = freq * modmul
        modindex : the index of modulation. The bigger, the more sidebands are generated
        amp      : amplitude of the end signal
        """
        body = """
        ifreq = p4
        iamp  = p5
        icar  = p6
        imod  = p7
        indx  = p8
        ifade = 0.03
        asig foscil iamp, ifreq, icar, imod, indx, -1
        asig *= linseg(0, ifade, 1, p3-ifade*2, 1, ifade, 0)
        out asig
        """
        self.freq = freq
        self.modmul = modmul
        self.modindex = modindex
        self.carmul = carmul
        self.amp = amp
        super().__init__(name='wavesim.freqmod',
                         instrBody=body,
                         instrInit="",
                         nchnls=1,
                         minDur=0.1,
                         args = [self.freq, self.amp, self.carmul, self.modmul, self.modindex])


class FreqMod2(Instr):
    def __init__(self, freq, carmul, modmul1, modmul2, idx1, idx2, amp=1.0):
        """
        FM Synthesis: 2 Modulators -> 1 Carrier

              MOD1   MOD2
               |       |
               +–––+–––+
                   |
                  CAR
                   |
                  OUT

        freq    : base frequency
        carmul  : carrier freq multiplier.     freq_carrier = freq * carmul
        modmul1 : modulator 1 freq multiplier. freq_mod1    = freq * modmul1
        modmul2 : modulator 2 freq multiplier. freq_mod2    = freq * modmul2
        idx1    : modulation index 1
        idx2    : modulation index 2
        amp     : output gain
        """
        self.freq = freq
        self.carmul = carmul
        self.modmul1 = modmul1
        self.modmul2 = modmul2
        self.idx1 = idx1
        self.idx2 = idx2
        self.amp = amp
        body = """
        iamp      = p4
        iBaseFreq = p5
        iCarRat   = p6
        iModRat1  = p7
        iModRat2  = p8
        iModIdx1  = p9
        iModIdx2  = p10

        ifade = 0.02
        kpeakdeviation1 = iBaseFreq * iModIdx1
        kpeakdeviation2 = iBaseFreq * iModIdx2
        aMod1 oscili kpeakdeviation1, iBaseFreq * iModRat1
        aMod2 oscili kpeakdeviation2, iBaseFreq * iModRat2
        aCar  oscili 1, (iBaseFreq*iCarRat) + aMod1 + aMod2
        aCar *= linsegr:a(0, ifade, iamp, ifade, 0)
        out aCar
        """
        super().__init__(name='wavesim.freqmod2', instrBody=body, instrInit="", nchnls=1, minDur=0.1,
                         args=[amp, freq, carmul, modmul1, modmul2, idx1, idx2])


class FreqMod2Stacked(Instr):
    def __init__(self, freq, carmul, modmul1, modmul2, idx1, idx2, amp=1.0):
        """
        2 Modulators, stacked
        1 Carrier
        """
        self.freq = freq
        self.carmul = carmul
        self.modmul1 = modmul1
        self.modmul2 = modmul2
        self.idx1 = idx1
        self.idx2 = idx2
        self.amp = amp
        body = """
        iamp      = p4
        iBaseFreq = p5
        iCarRat   = p6
        iModRat1  = p7
        iModRat2  = p8
        iModIdx1  = p9
        iModIdx2  = p10

        ifade = 0.02
        kpeakdeviation1 = iBaseFreq * iModIdx1
        kpeakdeviation2 = iBaseFreq * iModIdx2
        aMod1 oscili kpeakdeviation1, iBaseFreq * iModRat1
        aMod2 oscili kpeakdeviation2, iBaseFreq * iModRat2 + aMod1
        aCar oscili 1, (iBaseFreq*iCarRat) + aMod2
        aCar *= linsegr:a(0, ifade, iamp, ifade, 0)
        out aCar
        """
        super().__init__(name='wavesim.freqmod2stacked', instrBody=body, instrInit="", nchnls=1, minDur=0.1,
                         args=[amp, freq, carmul, modmul1, modmul2, idx1, idx2])


class RingModAll(Instr):
    def __init__(self, freqs, offsets=None, gain=1):
        """
        Ring-modulation of all possible 2-combinations of signals

        numSidebands = !len(freqs) / !(len(freqs) - 2)

        len(freqs)   num sidebands
        2            2*1 = 2
        3            3*2 = 6
        4            4*3 = 12
        5            5*4 = 20

        offsets: if given, it must be a sequence of equal length of freqs and turns ring-mod
                 into amp-mod
        """
        self.freqs = freqs
        self.offsets = offsets if offsets is not None else [0] * len(freqs)
        self.gain = gain
        """
        iamp = p4
        ifreq_0 = p5
        ioffset_0 = p6
        ifreq_1 = p7
        ioffset_1 = p8
        ...
        
        a_1 oscili iamp_1, ifreq_1
        ...
        aout = 0
        aout += a_1 * a_2
        aout += a_1 * a_3
        aout += a_2 * a_3
        ...
        aout *= lingsegr:a(0, 0.05, iamp, 0.05, 0)
        
        """
        numosc = len(freqs)
        body = self.genBody(numosc)
        super().__init__(name='wavesim.ringmodx%d'%numosc, instrBody=body, instrInit="", nchnls=1, minDur=0.2)

    def getArgs(self, gain=1):
        args = [self.gain * gain]
        for freq, offset in zip(self.freqs, self.offsets):
            args.append(freq)
            args.append(offset)
        return args

    @staticmethod
    def genBody(n):
        lines = []
        oscgain = 0.5
        lines.append( "iamp = p4" )
        for i in range(n):
            lines.append( f"ifreq_{i} = p{5+i*2}" )
            lines.append( f"ioffset_{i} = p{6+i*2}" )
        for i in range(n):
            lines.append( f"a_{i} oscili {oscgain}, ifreq_{i}" )
        lines.append( "aout = 0" )
        for i, j in itertools.combinations(list(range(n)), 2):
            lines.append( f"aout += (a_{i} + ioffset_{i}) * (a_{j} + ioffset_{j})" )
        lines.append( "aout *= linsegr:a(0, 0.05, iamp, 0.05, 0)" )
        lines.append( "out aout" )
        return "\n".join(lines)


def scaleAmps(n, maxdb=-3):
    return 1 / sqrt(n) * db2amp(maxdb)


class StaticSinesMono(Instr):
    @staticmethod
    def genBody(n):
        lines = [
            "igain = p4",
            "aout = 0"
        ]
        _ = lines.append
        for i in range(n):
            _(f"ifreq_{i} = p{i*2+5}")
            _(f"iamp_{i}  = p{i*2+6}")
        for i in range(n):
            _(f"aout += oscili:a(iamp_{i}, ifreq_{i})")
        _("aout *= linsegr:a(0, 0.05, igain, 0.1, 0)")
        _("out aout")
        return "\n".join(lines)

    def __init__(self, freqs, amps=None, gain=1):
        self.freqs = freqs
        self.amps = amps if amps is not None else [scaleAmps(len(freqs))] * len(freqs)
        self.gain = gain
        numsines = len(freqs)
        body = self.genBody(numsines)
        super().__init__(name='wavemod.staticsinesmono%d'%numsines, instrBody=body, instrInit="", nchnls=1, minDur=0.2)

    def getArgs(self, gain=1.0):
        args = [self.gain * gain]
        for freq, offset in zip(self.freqs, self.amps):
            args.append(freq)
            args.append(offset)
        return args



