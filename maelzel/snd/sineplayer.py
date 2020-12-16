from configdict import ConfigDict
import sys
import ctcsound

config = ConfigDict("emlib:synthplayer",
                    default={
                        'sr': 44100,
                        'linux.backend': 'jack',
                        'multisine.maxosc': 200,
                    },
                    validator={'linux.backend::choices': ['jack', 'portaudi']})

_csd_multisine = '''

sr     = {sr}
ksmps  = {ksmps}
nchnls = 2
0dbfs  = 1

#define MAXPARTIALS #{numosc}#

gkFreqs[] init $MAXPARTIALS
gkAmps[]  init $MAXPARTIALS
gkBws[]   init $MAXPARTIALS

gkgain init 1

alwayson 2, {numosc}
schedule 1, 0, 1

instr 1
    kidx = 0
    while kidx < lenarray(gkFreqs) do
        gkFreqs[kidx] = 1000
        gkAmps[kidx] = 0
        gkBws[kidx] = 0
        kidx += 1
    od
    turnoff
endin

instr 2
    inumosc = p4
    aout beadsynt 1, 1, gkFreqs, gkAmps, gkBws, inumosc, -1, -1, 0
    aout *= interp(gkgain)
    outch 1, aout, 2, aout
endin

instr 100
    idx = int(p4)
    ifreq = p5
    iamp = p6
    ibw = p7
    iramptime = p3
    kfreq = sc_lag(k(ifreq), iramptime, i(gkFreqs, idx))
    kamp = sc_lag(k(iamp), iramptime, i(gkAmps, idx))
    ; kbw = sc_lag(k(ibw), iramptime, i(gkBws, idx))
    kbw = ibw
    gkFreqs[idx] = kfreq
    gkAmps[idx] = kamp
    gkBws[idx] = kbw
endin

instr 200
    puts "instr 200!", 1
    igain = i(gkgain)
    gkgain linseg igain, p3-(ksmps/sr), 0
endin
'''


class MultiSineSynth:
    def __init__(self, sr=None, backend=None, porttime=0.05, maxosc=None):
        """
        notes:
            a seq. of notes, where each note is a seq. of [freq, amp, bw], or
            just [freq, amp]
            Example data = [
                [440, 0.4, 0.1],
                [220, 0.1],
                [800, 0.5, 0.5]
            ]

        sr:
            samplerate. Can be None, in which case we use the default sr
        backend:
            audio backend for your platform
        porttime:
            the time used to smooth out values of freq and amp for each osc.
        maxosc:
            the maximum number of oscillators.
        """
        cfg = config
        sr = sr if sr is not None else cfg['sr']
        backend = backend if backend is not None else cfg[
            f'{sys.platform}.backend']
        self.sr = sr
        self.backend = backend
        self.porttime = porttime
        self.fadetime = 0.2

        self._numosc = maxosc if maxosc is not None else cfg['multisine.maxosc']
        self._freqs = [0] * self._numosc
        self._amps = [0] * self._numosc
        self._bws = [0] * self._numosc
        self._cs = None
        self._pt = None
        self._exited = False
        self._csdstr = _csd_multisine
        self._start_csound()

    def _start_csound(self):
        cs = ctcsound.Csound()
        orc = self._csdstr.format(sr=self.sr,
                                  ksmps=128,
                                  backend=self.backend,
                                  numosc=self._numosc)
        options = ["-d", "-odac", "-+rtaudio=%s" % self.backend, "-m 0"]
        for opt in options:
            cs.setOption(opt)
        cs.compileOrc(orc)
        cs.start()
        pt = ctcsound.CsoundPerformanceThread(cs.csound())
        pt.play()
        self._cs = cs
        self._pt = pt

    def setOsc(self, idx, freq, amp, bw=0, delay=0):
        if idx > self._numosc - 1:
            raise IndexError("osc out of range")
        dur = self.porttime
        # the first argument, 0, indicates that we indicate time as
        # relative
        self._pt.scoreEvent(0, 'i', [100, delay, dur, idx, freq, amp, bw])

    def playNote(self, dur, freq, amp, bw=0, idx=None, delay=0):
        if idx is None:
            idx = self._getidx()
        self.setOsc(idx, freq, amp, bw, delay=delay)
        self.setOsc(idx, freq, 0, bw, delay=delay + dur)

    def fadeout(self, delay=0):
        self._pt.scoreEvent(0, 'i', [200, delay, self.fadetime])

    def stop(self):
        self._pt.stop()
        self._cs.stop()
        self._cs.cleanup()
        self._exited = True
