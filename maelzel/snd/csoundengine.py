from __future__ import annotations

import sys
import os
import ctcsound
import uuid as _uuid
from sndfileio import sndinfo
from typing import Optional as Opt, Union as U, Sequence as Seq, KeysView, List, Dict
import ctypes as _ctypes
import atexit as _atexit
import textwrap as _textwrap
import logging

from configdict import ConfigDict as _ConfigDict

from emlib.pitchtools import m2f
from emlib.iterlib import pairwise, chunks

from string import Template as _Template
from collections import namedtuple as _namedtuple, deque as _deque
from contextlib import contextmanager as _contextmanager, closing as _closing
from dataclasses import dataclass
from queue import Queue

from maelzel.snd import csound

import time
import signal
import weakref
import numpy as np
import socket
import subprocess


__all__ = [
    'unschedAll',
    'getManager',
    'stopEngine',
    'getEngine',
    'activeEngines',
    'config',
    'logger',
    'testout'
]


modulename = 'maelzel.csoundengine'

logger = logging.getLogger(modulename)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                  CONFIG                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

config = _ConfigDict(modulename.replace(".", ":"))
_ = config.addKey

_('sr', 0,
  choices=(0, 22050, 44100, 48000, 88200, 96000),
  doc='0 indicates the default sr for the backend')
_('numchannels', 0,
  range=(0, 128),
  doc='Number of channels to use. 0 = default for used device')
_('ksmps', 64,
  choices=(16, 32, 64, 128, 256),
  doc="corresponds to csound's ksmps")
_('linux.backend', 'jack',
  choices=('jack', 'pa_cb', 'pa_bl', 'pulse', 'alsa'))
_('fallback_backend', 'pa_cb',
  choices=('pa_cb', 'pa_bl'))
_('A4', 442,
  range=(410, 460))
_('multisine.maxosc', 200)
_('check_pargs', False,
  doc='Check number of pargs passed to instr')
_('fail_if_unmatched_pargs', False,
  doc='Fail if the # of passed pargs doesnt match the # of pargs'),
_('wait_poll_interval', 0.020,
  doc='seconds to wait when polling for a synth to finish')
_('set_sigint_handler', True,
  doc='Set a sigint handler to prevent csound crash with CTRL-C')
_('generalmidi_soundfont', '')
_('suppress_output', False,
  doc='Supress csound´s debugging information')
_('unknown_parameter_fail_silently', True,
  doc='Do not raise an Error if a synth is asked to set an unknown parameter')
_('define_builtin_instrs', True,
  doc="Whenever an InstrManager is created, it will have all builtin "
      "instruments defined")

config.load()


# Constants
_CSOUND_INSTRS = {
    'turnoff':     2,
    'chnset':      3,
    'filltable':   4,
    'freetable':   5,
    'maketable':   6,
    'playsndfile': 7,
    'uniqinstance': 8,
    'scheduniq': 9,
    'strset': 10,
    'playgen1': 11,
    'ftsetparams': 12,
    'pwrite': 13,
    'automate': 14,
    'testaudio': 15,
    'tabwrite': 16
}

_CSOUND_TABLES = {
    'responses': 1,
    'subgains': 2
}

_CSOUND_CONSTS = {
    'numtokens': 1000,
    'event_max_size': 1999,
    'reserved_instrs': 20,
    'reserved_tables': 200,
    'highest_instrnum': 12000
}

_NUMTOKENS                = 1000
_CSOUND_EVENT_MAX_SIZE    = 1999
_CSOUND_RESERVED_INSTRS   = 20
_CSOUND_RESERVED_TABLES   = 200
_CSOUND_HIGHEST_INSTRNUM  = 12000

_MYFLTPTR = _ctypes.POINTER(ctcsound.MYFLT)

_ORC_TEMPLATE = _Template("""
sr     = {sr}
ksmps  = {ksmps}
nchnls = {nchnls}
0dbfs  = 1
A4     = {a4}

gi__responses ftgen  ${responses}, 0, ${numtokens}, -2, 0
gi__subgains  ftgen  ${subgains},  0, 100, -2, 1


instr ${lastnum}
    ; this is used to prevent a crash when an opcode is defined as part 
    ; of globalcode, and later on an instr is defined with a high instrnum
    turnoff
endin

opcode _panstereo, aa, aak
    ; kpos: 0-1
    a0, a1, kpos xin
    imax = 1.4142
    kamp0 = bpf:k(kpos, 0, imax, 0.5, 1, 1, 0)
    kamp1 = bpf:k(kpos, 0, 0,    0.5, 1, 1, imax)
    if timeinstk() == 1 then
        println "kamp0: %f, kamp1: %f", kamp0, kamp1
    endif
    a0 *= kamp0
    a1 *= kamp1
    xout a0, a1
endop

{globalcode}

instr _notifyDealloc
    iwhich = p4
    outvalue "__dealloc__", iwhich
    turnoff
endin

instr ${turnoff}
    iwhich = p4
    turnoff2 iwhich, 4, 1
    turnoff
endin

instr ${tabwrite}
    itab = p4
    iidx = p5
    ival = p6
    tabw_i ival, iidx, itab
    turnoff
endin

instr ${chnset}
    Schn = p4
    ival = p5
    chnset ival, Schn
    turnoff
endin

; this is used to fill a table with the given pargs
instr ${filltable}
    itoken, ifn, itabidx, ilen passign 4
    iArgs[] passign 8, 8+ilen
    copya2ftab iArgs, ifn, itabidx
    if itoken > 0 then
        outvalue "__sync__", itoken
    endif
    turnoff
endin

; can make a table with data or just empty of a given size
; data must be smaller than 2000 which is the current limit
; for pfields
instr ${maketable}  
    ; args: 
    ;   itoken: the token used to sync with the engine
    ;   itabnum: the table number (can be 0 to let csound assign a number)
    ;   ilen: the size of the table
    ;   iempty: should the table be empty or should it be filled with data?
    ;           in this case, the rest p-args should containt ilen datapoints
    itoken = p4
    itabnum = p5
    ilen = p6
    iempty = p7
    if (iempty == 1) then
        ifn ftgen itabnum, 0, -ilen, -2, 0
    else
        iValues[] passign 8, 8+ilen
        ifn ftgen itabnum, 0, -ilen, -2, iValues
    endif
    ; notify host that token is ready
    if itoken > 0 then
        tabw_i ifn, itoken, gi__responses
        outvalue "__sync__", itoken
    endif
    turnoff
endin

instr _automate_pwrite
    ip1 = p4
    ipindex = p5
    inumpairs = p6
    imode = p7
    iparam = p8
    idataidx = 9
    
    ilen = inumpairs*2
    iValues[] passign idataidx, idataidx+ilen
    iXs[] slicearray iValues, 0, ilen-1, 2
    iYs[] slicearray iValues, 1, ilen-1, 2
    kt timeinsts
    if imode == 0 then
        ky bpf kt, iXs, iYs
    elseif imode == 1 then
        ky bpfcos kt, iXs, iYs
    elseif imode == 2 then
        ky bpf kt, iXs, iYs
        ilagtime = max(iparam, 0.05)
        ky = lag(ky, ilagtime)
    else
        initerror sprintf("imode %d not supported", imode)
    endif
    pwrite ip1, ipindex, ky
endin


instr ${automate}
    iargtab = p4
    iargidx = p5
    inumpairs = p6
    imode = p7   ; 0 = linear, 1 = cos, 2 = lag
    iparam = p8
    idataidx = 9

    if ftexists:i(iargtab) == 0 then
        initerror sprintf("Table %d doesn't exist", iargtab)
    endif
    if ftlen(iargtab) <= iargidx then
        initerror sprintf("Table too small (%d <= %d)", ftlen(iargtab), iargidx)
    endif

    ilen = inumpairs*2
    iValues[] passign idataidx, idataidx+ilen
    ; WTF: slicearray's end index is inclusive
    iXs[] slicearray iValues, 0, ilen-1, 2
    iYs[] slicearray iValues, 1, ilen-1, 2
    kt timeinsts
    if imode == 0 then
        ky bpf kt, iXs, iYs
    elseif imode == 1 then
        ky bpfcos kt, iXs, iYs
    elseif imode == 2 then
        ky bpf kt, iXs, iYs
        ilagtime = max(iparam, 0.05)
        ky = lag(ky, ilagtime)
    else
        throwerror "init", sprintf("imode %d not supported", imode)
    endif
    if ftexists:k(iargtab) == 0 then
        throwerror "warning", \
            sprintf("automate: dest table (%d) was freed, stopping", iargtab)
        turnoff
    endif
    tablew ky, iargidx, iargtab
endin

instr ${uniqinstance}
    itoken = p4
    ip1    = p5
    iuniq uniqinstance ip1
    tabw_i iuniq, itoken, gi__responses
    outvalue "__sync__", itoken
endin

instr ${freetable}
    ifn = p4
    idelay = p5
    ftfree ifn, 0
    turnoff
endin

instr ${playsndfile}
    Spath  = p4
    ichan  = p5
    ipitch = p6
    idur = filelen(Spath) / ipitch
    iwsize = ipitch == 1 ? 1 : 4
    aouts[] diskin2 Spath, ipitch, 0, 0, 0, iwsize
    inumouts = lenarray(aouts)
    ichans[] genarray ichan, ichan+inumouts-1
    aenv linsegr 0, 0.01, 1, 0.01, 0
    aouts = aouts * aenv
    poly0 inumouts, "outch", ichans, aouts
    if timeinsts() > idur then
        turnoff
    endif
endin

instr ${playgen1}
    ;             4     5      6       7     8     9          10
    ;             gain, speed, tabnum, chan, fade, starttime, gaingroup
    pset 0, 0, 0, 0,    1,     1,      1,    0.05, 0,         0
    kgain = p4
    kspeed = p5
    ksampsplayed = 0
    itabnum, ichan, ifade, ioffset, igaingroup passign 6
    inumsamples = nsamp(itabnum)
    itabsr = ftsr(itabnum)
    istartframe = ioffset * itabsr
    ksampsplayed += ksmps * kspeed
    ; ar[] loscilx xamp, kcps, ifn, iwsize, ibas, istrt
    aouts[] loscilx kgain, kspeed, itabnum, 4, 1, istartframe
    aenv = linsegr:a(0, ifade, 1, ifade, 0)
    ksubgain = table:k(igaingroup, gi__subgains)
    aenv *= a(ksubgain)
    aouts = aouts * aenv
    inumouts = lenarray(aouts)
    ichans[] genarray ichan, ichan+inumouts-1
    poly0 inumouts, "outch", ichans, aouts
    if ksampsplayed >= inumsamples then
        turnoff
    endif
endin

instr ${scheduniq}
    itoken = p4
    inumpargs = p5
    ip6 = uniqinstance(p6)
    if inumpargs == 3 then
        schedule ip6, p7, p8
    elseif inumpargs == 4 then
        schedule ip6, p7, p8, p9
    elseif inumpargs == 5 then
        schedule ip6, p7, p8, p9, p10
    elseif inumpargs == 6 then
        schedule ip6, p7, p8, p9, p10
    elseif inumpargs == 7 then
        schedule ip6, p7, p8, p9, p10, p11
    endif
    tabw_i ip6, itoken, gi__responses
    turnoff
endin

instr ${strset}
    Sstr = p4
    idx  = p5
    strset idx, Sstr
    turnoff
endin

instr ${ftsetparams}
    itabnum, isr, inumchannels passign 4
    ftsetparams itabnum, isr, inumchannels
    turnoff
endin

instr ${pwrite}
    ip1, ipnum, ivalue passign 4
    pwrite ip1, ipnum, ivalue
    turnoff
endin

instr ${testaudio}
    pset 0, 0, 0, 0
    imode passign 4
    if imode == 0 then
        a0 pinker
    elseif imode == 1 then
        a0 oscili 0.1, 1000
    endif
    outch 0, a0
endin
    
""").safe_substitute(lastnum=_CSOUND_HIGHEST_INSTRNUM, **_CSOUND_INSTRS, **_CSOUND_TABLES, **_CSOUND_CONSTS)

# Types

_SynthDef = _namedtuple("_SynthDef", "qname instrnum")


@dataclass
class TableProxy:
    tabnum: int
    sr: int
    nchnls: int
    numframes: int
    manager: Opt[InstrManager]
    path: str = ""
    freeself: bool = False
    _array: Opt[np.ndarray] = None

    def __post_init__(self):
        self.path = os.path.abspath(os.path.expanduser(self.path))

    def getData(self) -> np.ndarray:
        if self.manager is None:
            raise NoEngineError
        if self._array is None:
            self._array = self.manager.getEngine().getCsound().table(self.tabnum)
        return self._array

    def play(self, *args, **kws) -> AbstrSynth:
        return self.manager.playSample(self.tabnum, *args, **kws)

    def getDuration(self):
        return self.numframes / self.sr

    def __del__(self):
        if not self.freeself:
            return
        engine = self.manager.getEngine()
        if engine and engine.started:
            engine.freeTable(self.tabnum)


# ~~~~~~~~~ Exceptions ~~~~~~~~~

class NoEngineError(Exception): pass

class InstrumentNotRegistered(Exception): pass

class CsoundError(Exception): pass

class CsoundNotStartedError(Exception): pass

class RenderError(Exception): pass


# ~~~~~~~~~~ Globals ~~~~~~~~~

# this is used to cache searches and set global state
# (sigint handlers, etc)
_registry = {}


# ~~~~~~~~~~ Utilities ~~~~~~~~~~

def fluidsf2Path() -> Opt[str]:
    """
    Returns the path of the fluid sf2 file
    """
    key = 'fluidsf2_path'
    path = _registry.get(key)
    if path:
        return path
    userpath = config['generalmidi_soundfont']
    if userpath and os.path.exists(userpath):
        _registry[key] = userpath
        return userpath
    if sys.platform == 'linux':
        paths = ["/usr/share/sounds/sf2/FluidR3_GM.sf2"]
        path = next((path for path in paths if os.path.exists(path)), None)
        if path:
            _registry[key] = path
            return path
    else:
        raise RuntimeError("only works for linux right now")
    return None
   

def testout(dur=8, nchnls=2, sr:int = None, backend:str = None, outdev="dac") -> None:
    """
    Test audio out. Creates an ad-hoc engine and outputs a test pattern

    Args:
        dur: the duration of the test
        nchnls: the number of channels to output to
        sr: the sample rate (use None to use the system sr in the cases where
            this is possible)
        backend: the audio backend used (jack, portaudio, etc)
        outdev: the output device to use (use "dac" for default)

    This function will block for the duration of the test, after which the engine
    is stopped and freed
    """
    body = """
    instr 1
        iperiod = 1
        kchn init -1
        ktrig metro 1/iperiod
        kchn = (kchn + ktrig) % nchnls 
        anoise pinker
        outch kchn+1, anoise*0.1
        printk2 kchn
    endin
    """
    engine = CsoundEngine(name='testout', nchnls=nchnls, sr=sr,
                          backend=backend, outdev=outdev)
    engine.defInstr(instr=body)
    engine.sched(1, dur=dur)
    try:
        time.sleep(dur+0.5)
    except KeyboardInterrupt:
        pass
    engine.stop()


def _sigintHandler(sig, frame):
    print(frame)
    raise KeyboardInterrupt("SIGINT (CTRL-C) while waiting")


def setSigintHandler():
    """
    Set own sigint handler to prevent CTRL-C from crashing csound

    It will do nothing if this was already set
    """
    if _registry.get('sigint_handler_set'):
        return 
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigintHandler)
    _registry['original_sigint_handler'] = original_handler
    _registry['sigint_handler_set'] = True


def removeSigintHandler():
    """
    Reset the sigint handler to its original state
    This will do nothing if our own handler was not set
    in the first place
    """
    if not _registry.get('sigint_handler_set'):
        return 
    signal.signal(signal.SIGINT, _registry['original_sigint_handler'])
    _registry['sigint_handler_set'] = False


@_contextmanager
def safeSigint():
    setSigintHandler()
    try:
        yield
    except:
        raise  # Exception is dropped if we don't reraise it.
    finally:
        removeSigintHandler()


def _findFreePort():
    """
    Find a free port (for UDP communication)

    Returns:
        the port number
    """
    with _closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _is_jack_running() -> bool:
    proc = subprocess.Popen(["jack_control", "status"])
    running = proc.wait() == 0
    return running


def _get_jack_sr() -> int:
    cs = ctcsound.Csound()
    cs.setOption("-+rtaudio=jack")
    cs.setOption("-odac")
    cs.setOption("--use-system-sr")
    cs.start()
    sr = cs.sr()
    cs.stop()
    return sr

        
class CsoundEngine:

    def __init__(self, name:str = "Unnamed", sr:int = None, ksmps:int = None,
                 backend:str = None, outdev="dac", a4:int = None, nchnls:int = None,
                 globalcode:str = "", oscport:int = 0, quiet=None, udpport:int = None,
                 extraOptions=None):
        """
        Args:
            name:       the name of the engine
            sr:         sample rate
            ksmps:      samples per k-cycle
            backend:    passed to -+rtaudio
            outdev:     passed to -o
            a4:         freq of a4
            nchnls:     number of channels (passed to nchnls)
            globalcode: code to evaluate as instr0
            oscport:    port to use to communicate with this engine
            quiet:      if True, suppress output of csound (-m 0)
            udpport:    the udpport to use for real-time messages. Use 0 to autoassign
                        a port, None to disable
            extraOptions: extra command line options
        """
        if name in _engines:
            raise KeyError(f"engine {name} already exists")
        cfg = config
        backend = backend if backend is not None else cfg[f'{sys.platform}.backend']
        backends = csound.get_audiobackends_names()
        if backend not in backends:
            # should we fallback?
            fallback_backend = cfg['fallback_backend']
            if not fallback_backend:
                raise CsoundError(f"The backend {backend} is not available, no fallback backend defined")
            logger.error(f"The backend {backend} is not available. Fallback backend: {fallback_backend}")
            backend = fallback_backend
        # if outdev is None:
        #     outdev = csound.default_dac_for_backend(backend)
        extraOptions = extraOptions if extraOptions is not None else []
        sr = sr if sr is not None else cfg['sr']
        if sr == 0:
            sr = csound.get_sr(backend)
            if not sr:
                # failed to get sr for backend
                sr = 44100
                logger.error("Failed to get sr for backend {backend}, falling back to default sr: {sr}")
        if a4 is None: a4 = cfg['A4']
        if ksmps is None: ksmps = cfg['ksmps']
        extraOptions = extraOptions if extraOptions is not None else []
        if quiet is None: quiet = cfg['suppress_output']
        if quiet:
            extraOptions.append('-m0')
            extraOptions.append('-d')
        self.name = name
        self.sr = sr
        self.backend = backend
        self.a4 = a4
        self.ksmps = ksmps
        self.outdev = outdev
        self.uuid = _getUUID()
        self.nchnls = nchnls if nchnls is not None else cfg['numchannels']
        self.oscport = oscport
        self.qualifiedName = f"{self.name}:{self.uuid}"
        self.globalcode = globalcode
        self.started = False
        self.extraOptions = extraOptions
        if udpport is None:
            self.udpport = None
        elif udpport == 0:
            self.udpport = _findFreePort()
        else:
            assert udpport > 1024
            self.udpport = udpport
        if self.udpport is not None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udpsocket = sock
            self._sendAddr = ("127.0.0.1", self.udpport)
        else:
            self._udpsocket = None
            self._sendAddr = None

        self._performanceThread: Opt[ctcsound.CsoundPerformanceThread] = None
        self._csound: Opt[ctcsound.Csound] = None            # the csound object
        self._fracnumdigits = 5        # number of fractional digits used for unique instances
        self._exited = False           # are we still running?
        self._csdstr = _ORC_TEMPLATE   # the template to create new engines
        self._instanceCounter = {}     # counters to create unique instances for each instrument
        self._instrRegistry = {}       # instrument definitions are registered here
        self._outvalueCallbacks = {}   # a dict of callbacks, reacting to outvalue opcodes
        self._initDone = False         # have we set our callback, etc, yet?
        # Maps used for strSet / strGet
        self._indexToStr: Dict[int:str] = {}
        self._strToIndex: Dict[str:int] = {}
        self._strLastIndex = 20

        # global code added to this engine
        self._globalcode = {}

        # a history of events
        self._history = _deque([], 1000)

        # this will be a numpy array pointing to a csound table of
        # NUMTOKENS size. When an instrument wants to return a value to the
        # host, the host sends a token, the instr sets table[token] = value
        # and calls 'outvale "__sync__", token' to signal that an answer is
        # ready
        self._responsesTable = None

        self._subgainsTable = None

        # reserved channels used by the engine
        self._reservedChannels = set()

        # tokens start at 1, leave token 0 to signal that no sync is needed
        # tokens are used as indices to _responsesTable, which is an alias of
        # gi__responses
        self._tokens = list(range(1, _NUMTOKENS))

        # a pool of table numbers
        # self._tablePool = set(list(range(_CSOUND_RESERVED_TABLES, _CSOUND_RESERVED_TABLES+2000)))
        self._tablePool = _deque(list(range(_CSOUND_RESERVED_TABLES, _CSOUND_RESERVED_TABLES+2000)))

        # a dict of token:callback, used to register callbacks when asking for
        # feedback from csound
        self._responseCallbacks = {}

        # a dict mapping tableindex to fractional instr number
        self._assignedTables: Dict[int, float] = {}
        _engines[name] = self

        self._tableCache: Dict[int, np.ndarray] = {}

    def __repr__(self):
        return f"CsoundEngine(name={self.name}, backend={self.backend}, " \
               f"out={self.outdev}, nchnls={self.nchnls})"

    def __del__(self):
        self.stop()

    def _getToken(self) -> int:
        """
        Get a unique token, to pass to csound for a sync response
        """
        return self._tokens.pop()

    def _releaseToken(self, token:int) -> None:
        """ Release token back to pool when done """
        self._tokens.append(token)

    def assignTable(self, instrnum=-1, tabnum:int = None) -> int:
        assigned = self._assignedTables
        if tabnum is None:
            tabnum = self._tablePool.pop()
            assert tabnum not in assigned and tabnum is not None
            assigned[tabnum] = instrnum
            return tabnum

        tabnum = int(tabnum)
        assert tabnum not in assigned, f"table with index {tabnum} already assigned"
        assigned[tabnum] = instrnum
        if tabnum in self._tablePool:
            self._tablePool.remove(tabnum)
        return tabnum

    def _historyDump(self) -> None:
        for chunk in self._history:
            print(chunk)

    def _getInfo(self) -> dict:
        return {'name': self.name, 'uuid': self.uuid, 'oscport': self.oscport}

    def _getInstance(self, instnum:int) -> int:
        # TODO: do something less stupid
        n = self._instanceCounter.get(instnum, 0)
        m = 10 ** self._fracnumdigits - 1
        n = (n + 1) % m
        self._instanceCounter[instnum] = n
        return n

    def _getFractionalInstance(self, num:int, instance:int) -> float:
        frac = (instance / (10**self._fracnumdigits)) % 1
        return num + frac
        
    def _startCsound(self) -> None:
        # print(orc)
        bufferSize = 128
        options = ["-d", "-odac", f"-b{bufferSize}",
                   "-+rtaudio=%s" % self.backend]
        if self.extraOptions:
            options.extend(self.extraOptions)
        if self.backend == 'jack':
            if not _is_jack_running():
                logger.error("jack is not running")
                raise RuntimeError("jack is not running")
            if self.name is not None:
                clientname = self.name.strip().replace(" ", "_")
                options.append(f'-+jack_client=csoundengine.{clientname}')
            self.sr = _get_jack_sr()
        if self.udpport is not None:
            options.append(f"--port={self.udpport}")

        cs = ctcsound.Csound()
        for opt in options:
            cs.setOption(opt)
        orc = self._csdstr.format(sr=self.sr, ksmps=self.ksmps, nchnls=self.nchnls,
                                  backend=self.backend, a4=self.a4,
                                  globalcode=self.globalcode)
        logger.debug("------------------------------------------------------------------")
        logger.debug("  Starting performance thread. ")
        logger.debug(f"     Options: {options}")
        logger.debug(orc)
        logger.debug("------------------------------------------------------------------")
        cs.compileOrc(orc)
        print("starting csound with options: ", options)
        cs.start()
        pt = ctcsound.CsoundPerformanceThread(cs.csound())
        pt.play()
        self._orc = orc
        self._csound = cs
        self._performanceThread = pt
        self._history.append(orc)
        if config['set_sigint_handler']:
            setSigintHandler()
    
    def stop(self):
        if not self.started or self._exited:
            return
        self._performanceThread.stop()
        self._csound.stop()
        self._csound.cleanup()
        self._exited = True
        self._csound = None
        self._performanceThread = None
        self._instanceCounter = {}
        self._instrRegistry = {}
        _engines.pop(self.name, None)
        self.started = False

    def start(self):
        """ Start this engine """
        if self.started:
            logger.error(f"Engine {self.name} already started")
            return
        logger.info(f"Starting engine {self.name}")
        self._startCsound()
        priorengine = _engines.get(self.name)
        if priorengine:
            priorengine.stop()
        _engines[self.name] = self
        self._subgainsTable = self._csound.table(_CSOUND_TABLES['subgains'])
        self.started = True

    def restart(self) -> None:
        """ Restart this engine """
        self.stop()
        time.sleep(2)
        self.start()
        
    def _outcallback(self, _, channelName, valptr, chantypeptr):
        funcOrFuncs = self._outvalueCallbacks.get(channelName)
        if not funcOrFuncs:
            return

        if callable(funcOrFuncs):
            val = _ctypes.cast(valptr, _MYFLTPTR).contents.value
            funcOrFuncs(channelName, val)
            return

        funcs = funcOrFuncs
        unregistered = set()
        assert isinstance(funcs, (tuple, list))
        for i, func in enumerate(funcs):
            val = _ctypes.cast(valptr, _MYFLTPTR).contents.value
            ret = func(channelName, val)
            if ret == "unregister":
                unregistered.add(i)
        if unregistered:
            funcsNext = [func for i, func in enumerate(funcs)
                         if i not in unregistered]
            self._outvalueCallbacks[channelName] = funcsNext

    def _setupCallbacks(self):
        """
        Setup callback system for outvalue calls
        """
        if self._initDone:
            return

        if not self.started:
            self.start()

        def _syncCallback(_, token):
            """ Called with outvalue __sync__, the value is put
            in gi__responses at token idx, then __sync__ is
            called with token to signal that a response is
            waiting. The value can be retrieved via self._responsesTable[token]
            """
            token = int(token)
            callback = self._responseCallbacks.get(token)
            if callback:
                del self._responseCallbacks[token]
                callback(token)
                self._releaseToken(token)
            else:
                print(f"Unknown sync token: {token}")

        self._responsesTable = self._csound.table(_CSOUND_TABLES['responses'])
        self._csound.setOutputChannelCallback(self._outcallback)
        self._initDone = True
        self._outvalueCallbacks[bytes("__sync__", "ascii")] = _syncCallback

    def registerOutvalueCallback(self, chan:str, func) -> None:
        """
        Register a function `func` which will be called whenever a
        channel `chan` is changed in csound via the "outvalue" opcode
        More than one function can be registered for any channel

        Args:

            chan: the name of a channel
            func: a function of the form `func(chan, newvalue) -> msg`
                The function should return "unregister" if it
                wants to be unregistered. Otherwise it remains in the
                resitry

        """
        if not self._initDone:
            self._setupCallbacks()

        key = bytes(chan, "ascii")
        previousCallback = self._outvalueCallbacks.get(key)
        if chan.startswith("__"):
            if previousCallback:
                logger.warning("Attempting to set a reserved callback, but one "
                               "is already present. The new one will replace the old one")
            self._outvalueCallbacks[key] = func
        else:
            if not previousCallback:
                self._outvalueCallbacks[key] = [func]
            else:
                assert isinstance(previousCallback, list)
                previousCallback.append(func)

    def getCsound(self) -> ctcsound.Csound:
        return self._csound

    def defInstr(self, instr:str, name:str = None) -> None:
        """
        Compile a csound instrument

        Args:
            instr : the instrument definition, beginning with 'instr xxx'
            name  : name of the instrument, to keep track of definitions.
        """
        if not self.started: self.start()
        if not name:
            name = _getUUID()
        lines = [line for line in instr.splitlines() if line.strip()]
        instrnum = int(lines[0].split()[1])
        self._instrRegistry[name] = (instrnum, instr)
        logger.debug(f"defInstr (compileOrc): {name}")
        logger.debug(instr)
        self._csound.compileOrc(instr)
        self._history.append(instr)
    
    def evalCode(self, code:str, once=False) -> float:
        if not self.started:
            self.start()
        if once:
            out = self._globalcode.get(code)
            if out is not None:
                return out
        self._history.append(code)
        logger.debug(f"evalCode: \n{code}")
        self._globalcode[code] = out = self._csound.evalCode(code)
        return out

    def tableWrite(self, tabnum:int, idx:int, value:float, delay:float=0.) -> None:
        if not self.started:
            raise RuntimeError("csound not started")
        if delay == 0:
            arr = self.getTableData(tabnum)
            if arr:
                arr[idx] = value
        else:
            pargs = [_CSOUND_INSTRS['tabwrite'], delay, 1, tabnum, idx, value]
            self._performanceThread.scoreEvent(0, "i", pargs)

    def getTableData(self, idx:int, cached=True) -> np.ndarray:
        """"""
        if not cached:
            return self._csound.table(idx)
        arr = self._tableCache.get(idx)
        if arr is not None:
            self._tableCache[idx] = arr
        return arr

    def sched(self, instrnum:int, delay:float = 0, dur:float = -1,
              args:List[float] = None) -> float:
        """
        Schedule an instrument

        Args:

            instrnum : the instrument number
            delay    : time to wait before instrument is started
            dur      : duration of the event
            args     : any other args expected by the instrument

        Returns: 
            the fractional number of the instr started. 
            This can be used to kill the event later on 
            (see unsched)
        """
        if not self.started:
            raise RuntimeError("Engine not started")
        instance = self._getInstance(instrnum)
        instrfrac = self._getFractionalInstance(instrnum, instance)
        pargs = [instrfrac, delay, dur]
        if args:
            pargs.extend(args)
        assert all(isinstance(arg, (int, float)) for arg in pargs), pargs
        logger.debug(f"CsoundEngine.sched: scoreEvent(0, 'i', {pargs})  -> {instrfrac}")
        self._performanceThread.scoreEvent(0, "i", pargs)
        return instrfrac

    def unsched(self, instrfrac:float, delay:float = 0) -> None:
        """
        Args:
            instrfrac: the instrument number to remove
            delay: if 0, remove the instance as soon as possible
        """
        pfields = [_CSOUND_INSTRS['turnoff'], delay, 0.1, instrfrac]
        self._performanceThread.scoreEvent(0, "i", pfields)

    def unschedFuture(self) -> None:
        """
        Remove all future notes (this calles rewindScore)
        """
        self._csound.rewindScore()

    def getManager(self) -> InstrManager:
        return getManager(self.name)

    def makeTable(self, data:List[float] = None, size:int = 0, tabnum:int = None,
                  instrnum=-1, block=True, callback=None
                  ) -> int:
        """
        Create a new ftable and fill it with data.

        Args:
            data: the data used to fill the table, or None if creating an empty table
            size: the size of the table (will only be used if no data is supplied)
            tabnum: the table number. If None, a number is assigned by the engine.
                    If 0, a number is assigned by csound (only possible in
                    block or callback mode)
            instrnum: the instrument this table should be assigned to, if
                applicable
            block: wait until the table is actually created
            callback: call this function when ready - f(token, tablenumber) -> None

        Returns:
            the index of the new table, if wait is True
        """
        if tabnum is None:
            tabnum = self.assignTable(instrnum=instrnum)
        elif tabnum == 0:
            if not callback:
                block = True
        if block or callback:
            return self._makeTableNotify(data=data, size=size, tabnum=tabnum,
                                         callback=callback)
        # Create a table asynchronously
        assert tabnum > 0
        if not data:
            # an empty table
            assert size > 0
            pargs = [tabnum, 0, size, -2, 0]
            self._performanceThread.scoreEvent(0, "f", pargs)
            self._performanceThread.flushMessageQueue()
        elif len(data) < 1900:
            pargs = [tabnum, 0, size, -len(data)]
            if isinstance(data, np.ndarray):
                datalist = data.tolist()
            elif isinstance(data, list):
                datalist = data
            else:
                raise TypeError("data should be a list of floats or a numpa array")
            pargs.extend(datalist)
            self._performanceThread.scoreEvent(0, "f", pargs)
            self._performanceThread.flushMessageQueue()
        else:
            # lots of data
            # make empty table first
            pargs = [tabnum, 0, size, -2, 0]
            self._performanceThread.scoreEvent(0, "f", pargs)
            self._performanceThread.flushMessageQueue()
            self._fillTableViaScore(data, tabnum=tabnum)
            self._performanceThread.flushMessageQueue()
        return int(tabnum)

    def setTableParams(self, tabnum:int, sr:int, numchannels:int = 1) -> None:
        """
        Set metadata for a gen1 table. When csound reads a soundfile into a table,
        it stores some additional data, like samplerate and number of channels.
        A table created by other means and then filled with samples normally does
        not have this information. In most of the times this is ok, but there are
        some opcodes which rely on this information (opcodes like loscil). This
        method allows to set this information for such tables.
        """
        pargs = [_CSOUND_INSTRS['ftsetparams'], 0, -1, tabnum, sr, numchannels]
        self._performanceThread.scoreEvent(0, "i", pargs)

    def makeEmptyTable(self, size, numchannels=1, sr=0, instrnum=-1) -> int:
        """
        Create an empty table, returns the index of the created table
        """
        tabnum = self.assignTable(instrnum=instrnum)
        pargs = [tabnum, 0, size, -2, 0]
        self._performanceThread.scoreEvent(0, "f", pargs)
        self._performanceThread.flushMessageQueue()
        if numchannels > 1 or sr > 0:
            self.setTableParams(tabnum, numchannels=numchannels, sr=sr)
        return tabnum

    def getParamTable(self, idx) -> ParamTable:
        """
        TODO
        """
        return ParamTable(idx=idx, group=self.name)

    def _registerSync(self, token:int) -> Queue:
        q = Queue()
        table = self._responsesTable
        self._responseCallbacks[token] = lambda token, q=q, t=table: q.put(t[token])
        return q

    def _eventNotify(self, token:int, pargs, callback=None, timeout=1) -> Opt[float]:
        """
        Create a csound "i" event with the given pargs.

        The event is passed a token as p4 and can set a return value by:

            itoken = p4
            tabw kreturnValue, itoken, gi__responses
            outvalue "__sync__", itoken

        Args:
            token: a token as returned by self._getToken()
            pargs: the pfields passed to the event (beginning by p1)
            callback: if a callback is not passed, this method will block until a response
                is received from the csound event
            timeout: how long to wait for a response in blocking mode

        Returns:
            in blocking mode, it returns the value returned by the instrument

        """
        self._setupCallbacks()
        assert token == pargs[3]
        if callback:
            self._responseCallbacks[token] = callback
            self._performanceThread.scoreEvent(0, "i", pargs)
        else:
            q = self._registerSync(token)
            self._performanceThread.scoreEvent(0, "i", pargs)
            return q.get(block=True, timeout=timeout)

    def _inputMessageNotify(self, token:int, inputMessage:str, callback=None, timeout=1):
        self._setupCallbacks()
        if callback:
            self._responseCallbacks[token] = callback
            self._performanceThread.inputMessage(inputMessage)
        else:
            q = self._registerSync(token)
            self._performanceThread.inputMessage(inputMessage)
            q.get(block=True, timeout=timeout)

    def _makeTableNotify(self, data=None, size=0, tabnum=0, callback=None,
                         timeout=1, fillmethod='pointer') -> int:
        """
        Create a table with data (or an empty table of the given size).
        Let csound generate a table index if needed

        Args:
            data: the data to put in the table
            tabnum: the table number to create, 0 to let csound generate
                a table number
            callback: a callback of the form (token, value) -> None
                where value will hold the table number
            timeout: how long to wait in blocking mode

        Returns:
            * if set to block, returns the table index
            * if a callback is given, returns -1 and the callback will be
              called when the value is ready
        """
        self._setupCallbacks()
        token = self._getToken()
        instrnum = _CSOUND_INSTRS['maketable']
        if data is None:
            empty = 1
            pargs = [instrnum, 0, 0.01, token, tabnum, size, empty]
            tabnum = self._eventNotify(token, pargs, callback=callback, timeout=timeout)
            return int(tabnum)

        size = len(data)
        assert size > 0
        if size < 1900:
            empty = 0
            pargs = [instrnum, 0, 0.01, token, tabnum, size, empty]
            pargs.extend(data)
            tabnum = self._eventNotify(token, pargs, callback=callback, timeout=timeout)
            return int(tabnum)

        # create an empty table
        empty = 1
        pargs = [instrnum, 0, 0.01, token, tabnum, size, empty]
        tabnum = self._eventNotify(token, pargs)
        self.fillTable(data, tabnum, method=fillmethod, block=True)
        return int(tabnum)

    def setChannel(self, channel:str, value:float) -> None:
        """
        Set the value of a float software channel

        Args:
            channel: the name of the channel
            value: the new value
        """
        instrnum = _CSOUND_INSTRS['chnset']
        s = f'i {instrnum} 0 0.01 "{channel}" {value}'
        self._performanceThread.inputMessage(s)

    def getControlChannel(self, channel:str) -> float:
        """
        Get the value of a channel

        Args:
            channel: the name of the channel

        Returns:
            the value of the channel. Raises KeyError if the channel
            does not exist.
        """
        value, errorCode = self._csound.controlChannel(channel)
        if errorCode != 0:
            raise KeyError(f"control channel {channel} not found")
        return value

    def fillTable(self, tabnum:int, data, method='pointer', block=False) -> None:
        """
        Args:
            data: the data to put into the table
            tabnum: the table number
            method: the method used, one of 'pointer', 'score' or 'api'
            block: this is only used with methods "score" and "api"
        """
        if method == 'pointer':
            numpyptr: np.array = self._csound.table(tabnum)
            if numpyptr is None:
                raise IndexError(f"Table {tabnum} does not exist")
            size = len(numpyptr)
            if size < len(data):
                numpyptr[:] = data[:size]
            else:
                numpyptr[:] = data
        elif method == 'score':
            return self._fillTableViaScore(data, tabnum=tabnum, block=block)
        elif method == 'api':
            return self._fillTableViaAPI(data, tabnum=tabnum, block=block)
        else:
            raise KeyError("method not supported. Must be one of pointer, score, api")

    def _fillTableViaScore(self, data, tabnum:int, block=False) -> None:
        """
        Fill a table through score messages.

        Args:
            data: the data to send to the table
            tabnum: the table index
            block: should we wait until everything is sent?
        """
        chunksize = 1800
        now = 0
        token = 0
        delayBetweenRows = 0
        instrnum= _CSOUND_INSTRS['filltable']
        for idx, numitems in chunks(0, len(data), chunksize):
            if block and numitems < chunksize:
                # last row
                token = self._getToken()
            pargs = [instrnum, token, now, 0.01, tabnum, idx, numitems]
            payload = data[idx: idx+numitems]
            pargs.extend(payload)
            self._performanceThread.scoreEvent(0, "i", pargs)
            now += delayBetweenRows
        if block:
            self._performanceThread.flushMessageQueue()

    def _fillTableViaAPI(self, data:np.ndarray, tabnum:int, block=True) -> None:
        """
        ** NB: don't use this: has a LOT of latency **

        Copy contents of a numpy array to a table. Table must exist

        If data is 2D, it is flattened

        Args:
            data: a numpy array (1D or 2D) of type float64
            tabnum: the table to copy data to. If not given, a table is created
            block: if True, data is copied synchronously

        """
        if not self.started:
            raise RuntimeError("Engine not started")

        if len(data.shape) == 2:
            data = data.flatten()
        else:
            raise ValueError("data should be a 1D or 2D array")

        if block:
            self._csound.tableCopyIn(tabnum, data)
        else:
            self._csound.tableCopyInAsync(tabnum, data)

    def readSoundfile(self, path:str, tabnum:int = None, chan=0) -> int:
        """
        Read a soundfile into a table, returns the table number

        Args:
            path: the path to the soundfile
            tabnum: if given, a table index. If None, an index is
                autoassigned
            chan: the channel to read

        Returns:
            the index of the created table
        """
        if not self.started:
            raise RuntimeError(f"Engine {self.name} not started")
        if tabnum is None:
            tabnum = self.assignTable()
        s = f'f {tabnum} 0 0 -1 "{path}" 0 0 {chan}'
        # s = s.encode('ascii')
        self._performanceThread.inputMessage(s)
        return tabnum

    def getUniqueInstrInstance(self, instr: U[int, str], callback=None) -> float:
        if isinstance(instr, int):
            token = self._getToken()
            pargs = [_CSOUND_INSTRS['uniqinstance'], 0, 0.01, token, instr]
            if not callback:
                uniqinstr = self._eventNotify(token, pargs)
                return uniqinstr
            else:
                self._eventNotify(token, pargs, callback=callback)

    def playSample(self, tabnum:int, delay=0, chan=1, speed=1, gain=1., fade=0, starttime=0., gaingroup=0, dur=-1) -> float:
        return self.sched(_CSOUND_INSTRS['playgen1'], delay=delay, dur=dur,
                          args=[gain, speed, tabnum, chan, fade, starttime, gaingroup])

    def playSoundFromDisc(self, path:str, delay=0., chan=0, speed=1., unique=False
                          ) -> float:
        """
        Play a soundfile from disc.

        Args:
            path: the path to the soundfile
            delay: time offset to start playing
            chan: first channel to output to
            speed: playback speed (2.0 will sound an octave higher)
            unique: if True, a unique p1 is generated, which enables
                to turnoff that instance via unsched.

        Returns:
            the instrument number of the scheduled event
        """
        if not self.started:
            raise RuntimeError(f"Engine {self.name} not started")

        if not unique:
            p1 = _CSOUND_INSTRS['playsndfile']
        else:
            p1 = self.getUniqueInstrInstance(_CSOUND_INSTRS['playsndfile'])
        msg = f"i {p1} {delay} -1 \"{path}\" {chan} {speed}"
        self._performanceThread.inputMessage(msg)
        return p1

    def pwrite(self, p1, pnum, value, delay=0):
        self.sched(_CSOUND_INSTRS['pwrite'], delay=delay, dur=-1, args=[p1, pnum, value])

    def automateTable(self, tabnum:int, idx:int, pairs:Seq[float], 
                      mode='linear', param=0., delay=0.) -> float:
        """
        Write values to table@idx based on a bpf defined by times,values

        Args:
            tabnum: the table to write to
            idx: the index in table to write to
            pairs: a (flat) seq. of float values of the form t0, y0, t1, y1, ...
            mode: one of 'linear', 'cos', 'lag'. 
                For 'lag', a values in the bpf are smoothed via lag. The lag time 
                is controlled via `param`
            param: its meaning depends on mode. For 'lag', defined the lag time
            delay: the time origin for the automation. All breakpoints are relative
                to this time
        """
        assert len(pairs) % 2 == 0
        numpairs = len(pairs) / 2
        imode = {
            'linear': 0,
            'cos': 1,
            'lag': 2
        }.get(mode)
        if imode is None:
            raise ValueError("mode should be one of linear, cos, lag")
        args = [tabnum, idx, numpairs, imode, param]
        if isinstance(pairs, list):
            args += pairs
        else:
            args += list(pairs)
        dur = pairs[-2] + self.ksmps / self.sr
        return self.sched(_CSOUND_INSTRS['automate'], delay=delay, dur=dur, args=args)

    def strSet(self, s:str) -> int:
        if not self.started:
            raise RuntimeError(f"Engine {self.name} not started")

        stringIndex = self._strToIndex.get(s)
        if stringIndex is None:
            stringIndex = self._getStrIndex()
            if not self.started:
                raise CsoundNotStartedError()
            instrnum = _CSOUND_INSTRS['strset']
            msg = f'i {instrnum} 0 -1 "{s}"'
            self._performanceThread.inputMessage(msg)
            self._strToIndex[stringIndex] = s
            self._indexToStr[s] = stringIndex
        return stringIndex

    def strGet(self, index:int) -> Opt[str]:
        return self._indexToStr.get(index)

    def _getStrIndex(self) -> int:
        out = self._strLastIndex
        self._strLastIndex += 1
        return out

    def unassignTable(self, tableindex) -> None:
        """
        Mark the given table as freed, so that it can be assigned
        again. It assumes that the table was deallocated already
        and the index can be assigned again.
        """
        instrnum = self._assignedTables.pop(tableindex, None)
        if instrnum is not None:
            logger.debug(f"Unassigning table {tableindex} for instr {instrnum}")
            # self._tablePool.add(tableindex)
            self._tablePool.append(tableindex)

    def freeTable(self, tableindex, delay=0) -> None:
        logger.debug(f"Freeing table {tableindex}")
        self.unassignTable(tableindex)
        pargs = [_CSOUND_INSTRS['freetable'], delay, 0.01, tableindex]
        self._performanceThread.scoreEvent(0, "i", pargs)

    def testAudio(self, dur:float=4) -> float:
        return self.sched(_CSOUND_INSTRS['testaudio'], dur=dur)


    # ~~~~~~~~~~~~~~~ UDP ~~~~~~~~~~~~~~~~~~

    def udpSend(self, code: str) -> None:
        if not self.udpport:
            logger.warning("This csound instance was started without udp")
            return
        msg = code.encode("ascii")
        self._udpsocket.sendto(msg, self._sendAddr)

    def udpSendScoreline(self, scoreline:str) -> None:
        self.udpSend(f"& {scoreline}\n")

    def udpSetControlChannel(self, channel:str, value:float) -> None:
        self.udpSend(f"@{channel} {value}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def makeInstrTable(self, 
                       instr:CsoundInstr, 
                       overrides: Dict[str, float] = None, 
                       wait=True) -> int:
        """
        Create and initialize the table associated with instr. Returns
        the index 

        Args:
            instr: the instrument to create a table for
            overrides: a dict of the form param:value, which overrides the defaults
                in the table definition of the instrument
            wait: if True, wait until the table has been created

        Returns: 
            the index of the created table
        """
        values = instr.tableinit
        if overrides:
            values = instr.overrideTable(overrides)
        if len(values) < 1:
            logger.warning(f"instr table with no init values (instr={instr})")
            return self.makeTable(size=8, block=wait)
        else:
            logger.debug(f"Making table with init values: {values} ({overrides})")
            return self.makeTable(data=values, block=wait)

    def setSubGain(self, idx: int, gain: float) -> None:
        """
        Sets one of the subgains to the given value

        Each engine has a set of subgains which can be used by instruments
        to set the gain as a group. These gains are stored in a table gi__subgains

        Args:
            idx: the subgain to set
            gain: the value of the subgain

        """
        if not self.started:
            raise RuntimeError("csound not started")
        self._subgainsTable[idx] = gain


def _getUUID() -> str:
    return str(_uuid.uuid4())

def _isUUID(s: str) -> bool:
    if len(s) != 36:
        return False
    parts = s.split("-")
    if len(parts) != 5:
        return False
    if tuple(len(part) for part in parts) != (8, 4, 4, 4, 12):
        return False
    return True


_engines: Dict[str, CsoundEngine]  = {}
_managers: Dict[str, InstrManager] = {}


@_atexit.register
def _cleanup() -> None:
    _managers.clear()
    names = list(_engines.keys())
    for name in names:
        stopEngine(name)


def activeEngines() -> KeysView[str]:
    """
    Returns a list with the names of the active engines
    """
    return _engines.keys()


def getEngine(name="default") -> Opt[CsoundEngine]:
    """
    Get an already created engine (the 'default' engine does not 
    need to be created)

    To create an engine, first call CsoundEngine(name, ...)
    """
    engine = _engines.get(name)
    if engine:
        return engine
    if name == 'default':
        engine = CsoundEngine(name=name)
        return engine
    return None
    

def stopEngine(name="default") -> None:
    engine = _engines.get(name)
    if not engine:
        raise KeyError("engine not found")
    engine.stop()
    

class AbstrSynth:

    def __init__(self, group: str, autostop: bool = False):
        self.group = group
        self.autostop = autostop

    def __del__(self):
        try:
            if self.autostop:
                self.stop(stopParent=False)
        except:
            pass

    def stop(self, delay=0., stopParent=False):
        pass

    def isPlaying(self):
        pass

    def wait(self, pollinterval=None, sleepfunc=None):
        """
        Wait until this synth has stopped

        pollinterval: polling interval in seconds
        sleepfunc: the function to call when sleeping, defaults to time.sleep
        """
        if pollinterval is None: pollinterval = max(0.01, config['wait_poll_interval'])
        if sleepfunc is None: sleepfunc = time.sleep
        with safeSigint():
            while self.isPlaying():
                sleepfunc(pollinterval)

    def set(self, *args, **kws) -> None:
        """
        Set any dynamic value of this synth.

        Expects either

        synth.set(2, 0.5)     # set parameter at index 2 to 0.5
        synth.set('foo', 0.5) # set parameter foo to 0.5
        synth.set(foo=0.5)    # set parameter foo to 0.5

        Multiple values can be set at once

        synth.set(foo=0.5, bar=0.3)
        synth.set('foo', 0.5, 'bar', 0.3)
        """
        pass

    def get(self, slot: U[int, str], default: None) -> Opt[float]:
        pass

    def paramsState(self) -> Dict[str, float]:
        pass

    def paramState(self, param:str) -> Opt[float]:
        pass

    def automate(self, param:str, pairs, mode="linear", modeparam=0., delay=0.):
        pass


class ParamTable:
    def __init__(self,
                 idx: int,
                 group:str = 'default',
                 mapping:Dict[str, int] = None,
                 instrName:str = None,
                 associatedSynth:AbstrSynth = None):
        """
        A ParamTable is a csound table used as a multivalue communication channel
        between a running instrument and the outside world. ParamTables are used
        to communicate with a specific instance of an instrument.
        Channels or globals should be used to address global state.

        Each instrument can define the need of a param table attached at creation time,
        together with initial/default values for all slots. It is also possible
        to assign names to each slot

        The underlying memory can be accessed either directly via the .array
        attribute (a numpy array pointing to the table memory), or via
        table[idx] or table[key]

        A ParamTable does not currently check if the underlying csound table
        exists.

        Args:
            group: the name of the engine/manager
            idx: the index of the table
            mapping: an optional mapping from keys to table indices
                (see setNamed)
            instrName: the name of the instrument to which this table belongs
                (optional, used for the case where a Table is used as
                communication channel)
            associatedSynth: for debugging purposes (not used at the moment)
        """
        self.tableIndex:int = int(idx)
        self.mapping = mapping or {}
        self.groupName:str = group
        self.instrName = instrName
        self.engine = engine = getEngine(group)
        self.csound = engine.getCsound()
        self._array = None
        self.associatedSynth = associatedSynth
        self.deallocated = False
        self._failSilently = config['unknown_parameter_fail_silently']

    def __repr__(self):
        return f"ParamTable(tableIndex={self.tableIndex}, " \
               f"groupName={self.groupName}, associatedSynth={self.associatedSynth})"

    def getSize(self):
        return len(self.array)

    def getParamIndex(self, param:str) -> Opt[int]:
        """
        Returns the index corresponding to the named parameter
        Returns None if the parameter does not exist
        """
        if not self.mapping:
            return None
        return self.mapping.get(param, None)

    def paramState(self, param:str) -> Opt[float]:
        if not self.mapping:
            return None
        idx = self.mapping.get(param, None)
        if idx is not None:
            return self.array[idx]

    @property
    def array(self):
        if self._array is not None:
            return self._array
        self._array = a = self.csound.table(self.tableIndex)
        return a

    def __setitem__(self, idx, value):
        if isinstance(idx, int):
            self.array[idx] = value
        else:
            self._setNamed(idx, value)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.array[idx]
        return self._getNamed(idx)

    def _setNamed(self, key:str, value:float) -> None:
        """
        Set a value via a named index

        Args:

            key: a key as defined in mapping
            value: the value to set the corresponding index to
        """
        idx = self.mapping.get(key, -1)
        if idx < 0:
            if not self._failSilently:
                raise KeyError(f"key {key} not known (keys={list(self.mapping.keys())}")
        else:
            self.array[idx] = value

    def get(self, key:str, default:float = None) -> Opt[float]:
        """
        Get the value of a named slot. If key is not found, return default
        (similar to dict.get)
        """
        idx = self.mapping.get(key, -1)
        if idx == -1:
            return default
        return self.array[idx]

    def mappingRepr(self) -> str:
        values = self.array
        if not self.mapping:
            if values is None:
                return ""
            return str(values)
        if values is None:
            # table is not "live"
            return ", ".join(f"{key}=?" for key in self.mapping.keys())
        keys = list(self.mapping.keys())
        return ", ".join(f"{key}={value}" for key, value in zip(keys, values))
        
    def asDict(self) -> dict:
        """
        Return a dictionary mapping keys to their current value. This is
        only valid if this Table has a mapping, associating keys to indices
        in the table
        """
        if not self.mapping:
            raise ValueError("This Table has no mapping")
        values = self.array
        return {key: values[idx] for key, idx in self.mapping.items()}

    def _getNamed(self, key: U[str, int]) -> float:
        idx: int = self.mapping.get(key, -1)
        if idx < 0:
            raise KeyError(f"key {key} not known (keys={list(self.mapping.keys())}")
        return self.array[idx]

    def free(self, delay=0) -> None:
        if self.deallocated:
            return
        engine = getEngine(self.groupName)
        engine.freeTable(self.tableIndex, delay=delay)
        self.deallocated = True


class Synth(AbstrSynth):
    """
    A user does NOT normally create a Synth. A Synth is created
    when a CsoundInstr is scheduled
    """

    def __init__(self,
                 group:str,
                 synthid:float,
                 starttime:float,
                 dur:float,
                 instrname:str = None,
                 pargs=None,
                 synthgroup:SynthGroup = None,
                 autostop=False,
                 table:ParamTable = None
                 ) -> None:
        """
        Args:
            group: engine/manager name of this synth
            synthid: the synth id inside csound (a fractional instrument number)
            starttime: when was this synth started
            dur: duration of the note (can be -1 for infinite)
            instrname: the name of the instrument used
            pargs: the pargs used to create this synth
            synthgroup: the group this synth belongs to (if any)
            autostop: should this synth autostop? If True, the lifetime of the csound note
                is associated with this Synth object, so if this Synth goes out of
                scope or is deleted, the underlying note is unscheduled
            table: an associated Table (if needed)
        """
        AbstrSynth.__init__(self, group=group, autostop=autostop)
        self.synthid: float = synthid
        self.instrName: str = instrname
        self.startTime: float = starttime
        self.dur: float = dur
        self.pargs: List[float] = pargs
        self.table: Opt[ParamTable] = table
        self.synthGroup = synthgroup
        self._playing: bool = True

    def __repr__(self):
        if self.table:
            tablestr = self.table.mappingRepr()
            return f"Synth({tablestr})"
        return f"Synth(id={self.synthid})"

    def isPlaying(self) -> bool:
        now = time.time()
        if self.dur > 0:
            return self._playing and self.startTime < now < self.startTime+self.dur
        return self._playing and self.startTime < now

    def paramsState(self) -> Opt[Dict[str, float]]:
        """
        Get the state of all named parameters
        """
        if self.table is None:
            return None
        return self.table.asDict()

    def paramState(self, param:str) -> Opt[float]:
        if self.table is None:
            return None
        return self.table.paramState(param)

    def getParams(self) -> Opt[KeysView]:
        """
        Returns a set of named parameters
        """
        if self.table is None:
            return None
        return self.table.mapping.keys()

    def automate(self, param:str, pairs, mode="linear", modeparam=0., delay=0.):
        """
        Automate a named parameter of this synth

        Args:
            param: the named parameter as defined in the CsoundInstr
            pairs: a flat list of pairs of the form [t0, val0, t1, val1, ...]
            mode: one of "linear", "cos", "lag"
            modeparam: a value affecting mode. 
                For "lag", modeparam determines the lag time
            delay: time offset to start automation
        """
        if not self.table:
            logger.warning(f"Synth {self.synthid} has no parameter table")
            return 
        paramidx = self.table.getParamIndex(param)
        if paramidx is None:
            raise KeyError(f"Unknown param {param} for synth {self.synthid}")
        if len(pairs) > 1900:
            raise ValueError(f"pairs is too long (max. pairs = 900, got {len(pairs)/2})")
        engine = getEngine(self.group)
        return engine.automateTable(self.table.tableIndex, paramidx, pairs, mode=mode,
                                    param=modeparam, delay=delay)

    def getManager(self) -> InstrManager:
        return getManager(self.group)

    def stop(self, delay=0., stopParent=False) -> None:
        if self.synthGroup is not None and stopParent:
            self.synthGroup.stop(delay=delay)
        else:
            try:
                self.getManager().unsched(self.synthid, delay=delay)
            except NoEngineError:
                pass

    def set(self, *args, **kws):
        """
        Set a value of an associated array

        Either:

            synth.set('key', value, ...)
            synth.set(2, 0.5)  # set parameter with index 2 to 0.5

        Or:

            synth.set(key=value, [key2=value2, ...])

        """
        if not self._playing:
            logger.info("synth not playing")
            return

        if not self.table:
            logger.info("This synth has no associated table, skipping")

        if args:
            for key, value in pairwise(args):
                self.table[key] = value
        if kws:
            for key, value in kws.items():
                self.table[key] = value
   
    def get(self, slot: U[int, str], default:None) -> Opt[float]:
        if not self._playing:
            return

        if self.table:
            return self.table.get(slot, default)


class SynthGroup(AbstrSynth):
    """
    A SynthGroup is used to control multiple (similar) synths created
    to work together (in additive synthesis, for example)
    """

    def __init__(self, synths: List[AbstrSynth], autostop=False) -> None:
        AbstrSynth.__init__(self, group=synths[0].group, autostop=autostop)
        groupref = weakref.ref(self)
        for synth in synths:
            synth.synthgroup = groupref
        self.synths: List[AbstrSynth] = synths

    def stop(self, delay=0, stopParent=False) -> None:
        for s in self.synths:
            s.stop(stopParent=False, delay=delay)
            
    def isPlaying(self) -> bool:
        return any(s.isPlaying() for s in self.synths)

    def automate(self, param:str, pairs, mode="linear", modeparam=0., delay=0.):
        """
        Automate a named parameter of this synth

        Args:
            param: the named parameter as defined in the CsoundInstr
            pairs: a flat list of pairs of the form [t0, val0, t1, val1, ...]
            mode: one of "linear", "cos", "lag"
            modeparam: a value affecting mode. 
                For "lag", modeparam determines the lag time
            delay: time offset to start automation
        """
        for synth in self.synths:
            if isinstance(synth, Synth):
                if synth.table and param in synth.getParams():
                    synth.automate(param, pairs, mode=mode, modeparam=modeparam, delay=delay)
            elif isinstance(synth, SynthGroup):
                synth.automate(param=param, pairs=pairs, mode=mode, modeparam=modeparam, delay=delay)
    
    def paramsState(self) -> Opt[Dict[str, float]]:
        dicts = [s.paramsState() for s in self.synths]
        dicts = [d for d in dicts if d is not None]
        if not dicts:
            return None
        out = {}
        for d in dicts:
            out.update(d)
        return out

    def paramState(self, param:str) -> Opt[float]:
        for synth in self.synths:
            state = synth.paramState(param)
            if state is not None:
                return state
        return None

    def __repr__(self) -> str:
        lines = [f"SynthGroup(n={len(self.synths)})"]
        for synth in self.synths:
            lines.append("    " + repr(synth))
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.synths)

    def __getitem__(self, idx) -> AbstrSynth:
        return self.synths[idx]

    def __iter__(self):
        return iter(self.synths)

    def set(self, *args, **kws) -> None:
        for synth in self.synths:
            synth.set(*args, **kws)

    def get(self, idx: U[int, str], default=None) -> List[float]:
        return [synth.get(idx, default=default) for synth in self.synths]


def _tabledefGenerateCode(tabledef: dict) -> str:
    #if checkValidKeys:
    #    invalid = _invalidVariables.intersection(tabledef.keys())
    #    if invalid:
    #        raise KeyError(f"The table definition uses invalid variable names: {invalid}")
    lines = []
    idx = 0
    maxidx = len(tabledef)
    _ = lines.append
    _("\n; --- generated table code")
    _("i_params = p4")
    _( 'if ftexists(i_params) == 0 then')
    _( '  throwerror "init", "params table does not exist"')
    _( 'endif')
    _( 'i_paramslen = ftlen(i_params)')
    _(f'if i_paramslen < {maxidx} then')
    _(f'  initerror sprintf("params table is too small (size: %d, needed: {maxidx})", i_paramslen)')
    _( 'endif')
    for key, value in tabledef.items():
        # varname = key if key.startswith("k") else "k" + key
        _(f"k{key} tab {idx}, i_params")
        idx += 1
    _("; --- end generated table code\n")
    return "\n".join(lines)


class CsoundInstr:
    __slots__ = ('body', 'name', 'init', 'group', 'tableinit', 'tablemap', 'tabledef',
                 'numchans', 'mustFreeTable', 'doc',
                 '_numpargs', '_recproc', '_check', '_preschedCallback')

    def __init__(self,
                 name:str,
                 body: str,
                 init: str = None,
                 # tableinit: List[float] = None,
                 # tablemap: Dict[str, int] = None,
                 tabledef: Dict[str, float] = None,
                 numchans: int = 1,
                 group="default",
                 preschedCallback=None,
                 freetable=True,
                 generateTableCode=True,
                 doc: str = ''
                 ) -> None:
        """
        *** A CsoundInstr is created via defInstr, DON'T CREATE IT DIRECTLY ***

        To schedule a Synth using this instrument, call .play

        Args:
            name:
                the name of the instrument, if any. Use None to assign a UUID
            body:
                the body of the instr (the text BETWEEN 'instr' end 'endin')
            init:
                code to be initialized at the instr0 level (tables, reading files, etc.)
            tabledef:
                NB: An instrument can have an associated table to be able to pass
                dynamic parameters which are specific to this note (for example,
                an instrument could define a filter with a dynamic cutoff freq.)
                A tabledef is a dict of the form: {param_name: initial_value}.
                The order of appearence will correspond to the index in the table
            group:
                the name of the group this instrument belongs to. Use None to define
                an abstract instrument, which can be registered at many managers
            preschedCallback:
                a function f(synthid, args) -> args, called before a note is scheduled with
                this instrument. Can be used to allocate a table or a dict and pass
                the resulting index to the instrument as parg
            freetable:
                if True, the associated table is freed when the note is finished
            generateTableCode:
                if True, code is autogenerated to access the associated table
                In particular, all named slots are associated with a k-variable
                For example, if tabledef is {'freq': 440, 'amp': 0.5}, the variables
                kfreq and kamp will be available to the body of the instr. with the
                current values of the slot.
            doc: some documentation describing what this instr does

        """
        def _parseTableDef(t):
            init = list(t.values())
            tabmap = {paramname:idx for idx, paramname in enumerate(t.keys())}
            return init, tabmap

        errmsg = _checkInstr(body)
        if errmsg:
            raise CsoundError(errmsg)
        if tabledef:
            tableinit, tablemap = _parseTableDef(tabledef)
            self.tableinit = tableinit
            self.tablemap = tablemap
        else:
            self.tableinit, self.tablemap = None, None
        self.tabledef = tabledef
        self.group = group
        self.name = name if name is not None else _getUUID()
        if generateTableCode and tabledef:
            body = _tabledefGenerateCode(tabledef) + body
        self.body = body
        self.init = init if init else None
        self.numchans = numchans
        self.doc = doc
        self._numpargs = None
        self._recproc = None
        self._check = config['check_pargs']
        self._preschedCallback = preschedCallback
        if tabledef is None and freetable:
            freetable = False
        self.mustFreeTable = freetable

    def __repr__(self):
        header = f"CsoundInstr({self.name}, group={self.group})"
        sections = ["", header]
        pargs = self.getPargs()
        if pargs:
            pargsStr = ", ".join(f"p{i}: {pname}" for i, pname in sorted(pargs.items()))
            sections.append(pargsStr)
        if self.doc:
            sections.append(f"> doc: {self.doc}")
        if self.init:
            sections.append("> init")
            sections.append(str(self.init))
        if self.tableinit:
            sections.append("> table")
            sections.append(f"    {self.tableinit}, {self.tablemap}")
        sections.append("> body")
        sections.append(self.body)
        return "\n".join(sections)

    def _getManager(self):
        if self.group is None:
            return None
        return getManager(self.group)
    
    def getPargs(self) -> Dict[int, str]:
        """
        Return the name of the pargs defined in the source of this Instr.
        """
        return csound.parg_names(self.body)

    def play(self, dur=-1, args: List[float] = None, priority:int = 1, delay=0.0,
             tabargs: Dict[str, float] = None, whenfinished=None
             ) -> Synth:
        """
        Schedules a Synth with this instrument.

        Args:
            dur: the duration of the synth. -1 = play until stopped
            args: args to be passed to the synth (p values, beginning
                with p5, since p4 is reserved for the associated table
                index)
            priority: a number indicating order of execution. This is
                only important when depending on other synths
            delay: how long to wait to start the synth (this is always
                relative time)
            tabargs: named args passed to the associated table of this
                instrument (if any) (see _InstrManager.sched)
            whenfinished: A function f(synthid) -> None. It will be
                called when the synth is deallocated

        Returns:
            a Synth
        """
        if self.group is None:
            raise InstrumentNotRegistered("This instrument was not registered "
                                          "by any manager")
        if self._check:
            self._checkArgs(args)
        manager = self._getManager()
        return manager.sched(instrname=self.name, priority=priority,
                             delay=delay, dur=dur, pargs=args,
                             tabargs=tabargs, whenfinished=whenfinished)

    def asOrc(self, instrid, sr:int, ksmps:int, nchnls:int = None,
              a4:int = None) -> str:
        """
        Generate a csound orchestra with only this instrument defined

        Args:
            instrid: the id (instr number of name) used for this instrument
            sr: samplerate
            ksmps: ksmps
            nchnls: number of channels
            a4: freq of A4

        Returns:
            The generated csound orchestra
        """
        nchnls = nchnls if nchnls is not None else self._numchannels() 
        a4 = a4 if a4 is not None else config['A4']
        if self.init is None:
            initstr = ""
        else:
            initstr = self.init
        orc = f"""
        sr = {sr}

        ksmps = {ksmps}
        nchnls = {nchnls}
        0dbfs = 1.0
        A4 = {a4}

        {initstr}

        instr {instrid}
        
        {self.body}
        
        endin

        """
        return orc

    def _numchannels(self):
        return 2

    def _numargs(self) -> int:
        if self._numpargs is None:
            self._numpargs = csound.num_pargs(self.body)
        return self._numpargs

    def _checkArgs(self, args) -> bool:
        lenargs = 0 if args is None else len(args)
        numargs = self._numargs()
        ok = numargs == lenargs
        if not ok:
            msg = f"expected {numargs} args, got {lenargs}"
            logger.error(msg)
        return ok

    def rec(self, dur, outfile:str = None, args:List[float] = None, sr=44100, ksmps=64, 
            samplefmt='float', nchnls:int = None, block=True, a4 = None) -> str:
        """
        Args:

            dur: the duration of the recording
            outfile: if given, the path to the generated soundfile.
                If not given, a temporary file will be generated.
            args: the seq. of pargs passed to the instrument (if any),
                beginning with p4
            sr: the sample rate
            ksmps: the number of samples per cycle
            samplefmt: one of 16, 24, 32, or 'float'
            nchnls: the number of channels of the generated soundfile. It defaults to 2
            block: if True, the function blocks until done, otherwise rendering is asynchronous
            a4: the frequency of A4
        """
        event = [0., dur]
        if args:
            event.extend(args)
        return self.recEvents(events=[event], outfile=outfile, sr=sr,
                              ksmps=ksmps, samplefmt=samplefmt, nchnls=nchnls,
                              block=block, a4=a4)

    def recEvents(self, events: List[List[float]], outfile:str = None,
                  sr=44100, ksmps=64, samplefmt='float', nchnls:int = None,
                  block=True, a4=None
                  ) -> str:
        """
        Record the given events with this instrument.

        Args:
            events: a seq. of events, where each event is the list of pargs
                passed to the instrument, as [delay, dur, p4, p5, ...]
                (p1 is omitted)
            outfile: if given, the path to the generated soundfile. If not
                given, a temporary file will be generated.
            sr: the sample rate
            ksmps: number of samples per period
            samplefmt: one of 16, 24, 32, or 'float'
            nchnls: the number of channels of the generated soundfile.
                It defaults to 2
            a4: frequency of A4
            block: if True, the function blocks until done, otherwise rendering
                is asynchronous

        Returns:
            the generated soundfile (if outfile is not given, a temp file
            is created)

        """
        nchnls = nchnls if nchnls is not None else self._numchannels()
        initstr = self.init or ""
        a4 = a4 or config['A4']
        outfile, popen = csound.rec_instr(body=self.body, 
                                          init=initstr, 
                                          outfile=outfile,
                                          events=events, 
                                          sr=sr, 
                                          ksmps=ksmps, 
                                          samplefmt=samplefmt, 
                                          nchnls=nchnls, 
                                          a4=a4)
        if block:
            popen.wait()
        return outfile

    def stop(self):
        """
        Will stop all synths created with this instrument
        """
        self._getManager().unschedByName(self.name)

    def hasExchangeTable(self) -> bool:
        """
        Returns True if this instrument defines an exchange table
        """
        return self.tableinit is not None and len(self.tableinit) > 0

    def overrideTable(self, d:Dict[str, float], **kws) -> List[float]:
        """
        Overrides default values in the exchange table
        Returns the initial values

        Args:
            d: if given, a dictionary of the form {'argname': value}.
                Alternatively key/value pairs can be passed as keywords
            **kws: each key must match a named parameter as defined in
                the tabledef

        Returns:
            A list of floats holding the new initial values of the
            exchange table

        Example:
            instr.overrideTable(param1=value1, param2=value2)

        """
        if self.tableinit is None:
            raise ValueError("This instrument has no associated table")
        if self.tablemap is None:
            raise ValueError("This instrument has no table mapping, so"
                             "named parameters can't be used")
        if d is None and not kws:
            return self.tableinit
        out = self.tableinit.copy()
        if d:
            for key, value in d.items():
                idx = self.tablemap[key]
                out[idx] = value
        if kws:
            for key, value in kws.items():
                idx = self.tablemap[key]
                out[idx] = value
        return out


def _checkInstr(instr: str) -> str:
    """
    Returns an error message if the instrument is not well defined
    """
    lines = [line for line in (line.strip() for line in instr.splitlines()) if line]
    errmsg = ""
    if "instr" in lines[0] or "endin" in lines[-1]:
        errmsg = ("instr should be the body of the instrument,"
                  " without 'instr' and 'endin")
    return errmsg


def _initTable(instr:CsoundInstr,
               engine:CsoundEngine,
               overrides:Dict[str, float] = None,
               wait=True) -> int:
    """
    Create a table with the values from instr.tableinit,
    returns the table index

    Args:
        instr: the instrument to create a table for
        engine: the engine where this instrument is defined
        overrides: a dict of the form param:value, which overrides the defaults
            in the table definition of the instrument
        wait: if True, wait until the table has been created

    Returns: 
        the index of the created table
    """
    values = instr.tableinit
    if overrides:
        values = values.copy()
        for k, v in overrides.items():
            idx = instr.tablemap[k]
            values[idx] = v
    tabnum = engine.makeTable(values, block=wait)
    return tabnum


class InstrManager:
    """
    An InstrManager controls an engine. It has an exclusive associated
    CsoundEngine. A CsoundEngine can be thought of as a low-level
    interface to managing a csound instance, whereas an InstrManager
    allows a higher-level control with less friction

    The user does not normally create an instance of this class directly.
    It is returned by getManager
    """
    
    def __init__(self, name="default") -> None:
        self.name: str = name
        self.instrDefs: Dict[str, CsoundInstr] = {}

        self._bucketsize: int = 1000
        self._numbuckets: int = 10
        self._buckets = [{} for _ in range(self._numbuckets)]  # type: List[Dict[str, int]]
        self._synthdefs = {}                                   # type: Dict[str, Dict[int, _SynthDef]]
        self._synths = {}                                      # type: Dict[float, Synth]
        self._isDeallocCallbackSet = False
        self._whenfinished = {}
        self._initCodes = []
        self._tabnumToTable: Dict[int:TableProxy] = {}
        self._pathToTable: Dict[str:TableProxy] = {}

        if config['define_builtin_instrs']:
            self.defBuiltinInstrs()

    def __repr__(self):
        active = len(self.activeSynths())
        return f"InstrManager({self.name}, active={active})"

    def _deallocSynth(self, synthid: U[int, float], delay=0.) -> None:
        synth = self._synths.pop(synthid, None)
        if synth is None:
            return
        logger.debug(f"Synth({synth.instrName}, id={synthid}) deallocated")
        engine = self.getEngine()
        engine.unsched(synthid, delay)
        synth._playing = False
        if synth.table:
            instr = self.getInstr(synth.instrName)
            engine = self.getEngine()
            if instr.mustFreeTable:
                engine.freeTable(synth.table.tableIndex)
            else:
                engine.unassignTable(synth.table.tableIndex)
        callback = self._whenfinished.pop(synthid, None)
        if callback:
            callback(synthid)
        
    def _deallocCallback(self, _, synthid):
        """ This is called by csound when a synth is deallocated """
        self._deallocSynth(synthid)

    def _ftgenCallback(self, _, ftnum):
        """
        will be called whenever __ftgen__ is changed

        Args:
            _:
            ftnum: the generated table number

        """
        engine = self.getEngine()
        engine.assignTable(tabnum=ftnum)

    def getEngine(self) -> CsoundEngine:
        """ Return the associated engine """
        engine = getEngine(self.name)
        if not self._isDeallocCallbackSet:
            engine.registerOutvalueCallback("__dealloc__", self._deallocCallback)
            engine.registerOutvalueCallback("__ftgen__", self._ftgenCallback)
            self._isDeallocCallbackSet = True
        return engine

    def getInstrNumber(self, instrname:str, priority=1) -> int:
        """
        Get the instrument number corresponding to this name and
        the given priority

        Args:
            instrname: the name of the instr as given to defInstr
            priority: the priority, an int from 1 to 10. Instruments with
                low priority are executed before instruments with high priority

        Returns:
            the instrument number (an integer)
        """
        assert 1 <= priority < self._numbuckets - 1
        bucket = self._buckets[priority]
        instrnum = bucket.get(instrname)
        if instrnum is not None:
            return instrnum
        idx = len(bucket) + 1
        instrnum = self._bucketsize*priority + idx
        bucket[instrname] = instrnum 
        return instrnum

    def defInstr(self,
                 name:str,
                 body:str,
                 init="",
                 tabledef: Dict[str, float] = None,
                 numchans: int = 1,
                 freetable=True,
                 generateTableCode=True
                 ) -> CsoundInstr:
        """

        Args:
            name: a name to identify this instr, or None, in which case a UUID
                is created
            body: the body of the instrument
            init: initialization code for the instr (ftgens, global vars, etc)
            tabledef: a dict {paramname: initial}
            numchans: the number of channels this instrument outputs
            freetable: if True, we take care that the associated table is freed
                at the end of this note. Otherwise the instrument itself should
                call ftfree as part of its body
            generateTableCode: if True (the default), code is generated to read
                the params table as k-variables.

        Returns:
            a CsoundInstr
        """
        if name is None:
            instr = self.findInstrByBody(body=body, init=init)
            name = instr.name if instr else _getUUID()
        else:
            instr = self.instrDefs.get(name)
        if instr:
            if body == instr.body and init == instr.init:
                logger.debug(f"The instruments are identical, reusing old instance")
                return instr
            logger.info("Instruments differ, old definition will be overwritten")
            logger.debug(f"new body: \n{body}\n\nold body:\n{instr.body}")
            logger.debug(f"new init: \n{init}\n\nold init:\n{instr.init}")
            self._resetSynthdefs(name)
        instr = CsoundInstr(name=name, 
                            body=body, 
                            init=init,
                            group=self.name,
                            tabledef=tabledef,
                            numchans=numchans,
                            generateTableCode=generateTableCode,
                            freetable=freetable)
        self.registerInstr(instr)
        return instr

    def findInstrByBody(self, body, init:str = None, onlyunnamed=False) -> Opt[CsoundInstr]:
        for name, instr in self.instrDefs.items():
            if onlyunnamed and not _isUUID(name):
                continue
            if body == instr.body and (not init or init == instr.init):
                return instr

    def registerInstr(self, instr:CsoundInstr, name:str = None) -> None:
        """
        Register the given CsoundInstr in this manager. It evaluates
        any init code, if necessary

        Args:
            instr: the CsoundInstr to register
            name: the name to use (will override the name in the CsoundInstr)

        """
        name = name or instr.name
        self.instrDefs[name] = instr
        if instr.init:
            self._evalInit(instr)
            
    def _evalInit(self, instr:CsoundInstr) -> None:
        """
        Evaluates the init code in the given instrument
        """
        code = instr.init
        logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> evaluating init code: ")
        logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " + code)
        self._initCodes.append(code)
        self.getEngine().evalCode(code)

    def _resetSynthdefs(self, name):
        self._synthdefs[name] = {}

    def _registerSynthdef(self, name: str, priority: int, synthdef: _SynthDef
                          ) -> None:
        registry = self._synthdefs.get(name)
        if registry:
            registry[priority] = synthdef
        else:
            registry = {priority: synthdef}
            self._synthdefs[name] = registry

    def _makeSynthdef(self, name:str, priority:int) -> _SynthDef:
        """
        A SynthDef is a version of an Instrument with a given priority
        While making a SynthDef we send it already to the Engine
        """
        assert isinstance(priority, int) and 1 <= priority <= 10
        qname = _qualifiedName(name, priority)
        instrdef = self.instrDefs.get(name)
        instrnum = self.getInstrNumber(name, priority)
        instrtxt = _instrWrapBody(instrdef.body, instrnum)
        logger.debug(f"_makeSynthdef: {name} --\n\n{instrtxt}")

        engine = self.getEngine()
        if not engine:
            logger.error(f"Engine {self.name} not initialized")
            raise RuntimeError("engine not initialized")
        engine.defInstr(instr=instrtxt, name=name)
        synthdef = _SynthDef(qname, instrnum)
        self._registerSynthdef(name, priority, synthdef)
        return synthdef

    def getInstr(self, name:str) -> CsoundInstr:
        return self.instrDefs.get(name)

    def _getSynthdef(self, name:str, priority:int) -> Opt[_SynthDef]:
        registry = self._synthdefs.get(name)
        if not registry:
            return None
        return registry.get(priority)

    def prepareSched(self, instrname:str, priority:int = 1) -> _SynthDef:
        synthdef = self._getSynthdef(instrname, priority)
        if synthdef is None:
            synthdef = self._makeSynthdef(instrname, priority)
        return synthdef

    def sched(self, 
              instrname:str, 
              priority:int = 1, 
              delay=0., 
              dur=-1.,
              pargs: List[float] = None, 
              tabargs: Dict[str, float] = None,
              whenfinished=None
              ) -> Synth:
        """
        Schedule the instrument identified by 'instrName'

        Args:
            instrname: the name of the instrument, as defined via defInstr
            priority: the priority (1 to 10)
            delay: time offset of the scheduled instrument
            dur: duration (-1 = for ever)
            pargs: pargs passed to the instrument (p5, p6, ...)
            tabargs: args to set the initial state of the associated table. Any
                arguments here will override the defaults in the instrument definition
            whenfinished: a function of the form f(synthid) -> None
                if given, it will be called when this instance stops
        
        Returns:
            a Synth, which is a handle to the instance (can be stopped, etc.)
        """
        assert isinstance(priority, int) and 1 <= priority <= 10
        synthdef = self.prepareSched(instrname, priority)
        instr = self.getInstr(instrname)
        engine = self.getEngine()
        if instr.tableinit is not None:
            # the instruments has an associated table
            tableidx = engine.makeInstrTable(instr, overrides=tabargs, wait=True)
            table = ParamTable(group=self.name, idx=tableidx, mapping=instr.tablemap)
        else:
            tableidx = 0
            table = None
        # tableidx is always p4
        allargs = [tableidx]
        if pargs:
            allargs.extend(pargs)
        synthid = engine.sched(synthdef.instrnum, delay=delay, dur=dur, args=allargs)
        if whenfinished is not None:
            self._whenfinished[synthid] = whenfinished
        synth = Synth(self.name,
                      synthid=synthid,
                      instrname=instrname,
                      starttime=time.time() + delay,
                      dur=dur,
                      table=table,
                      pargs=pargs)
        self._synths[synthid] = synth
        return synth

    def activeSynths(self, sortby="start") -> List[Synth]:
        """
        Returns a list of playing synths

        sortby: either "start" (sort by start time) or None (unsorted)
        """
        synths = [synth for synth in self._synths.values() if synth.isPlaying()]
        if sortby == "start":
            synths.sort(key=lambda synth: synth.startTime)
        return synths

    def scheduledSynths(self) -> List[Synth]:
        """
        Returns all scheduled synths (both active and future)
        """
        return list(self._synths.values())

    def unsched(self, *synthids:float, delay=0.) -> None:
        """
        Stop an already scheduled instrument

        Args:
            synthids: one or many synthids to stop
            delay: how long to wait before stopping them
        """
        engine = self.getEngine()
        for synthid in synthids:
            synth = self._synths.get(synthid)
            if synth.isPlaying() or delay > 0:
                # We just need to unschedule it from csound. If the synth is playing,
                # it will be deallocated and the callback will be fired
                engine.unsched(synthid, delay)
            else:
                self._deallocSynth(synthid, delay)

    def unschedLast(self, n=1, unschedParent=True) -> None:
        """
        Unschedule last synth

        Args:
            n: number of synths to unschedule
            unschedParent: if the synth belongs to a group, unschedule the
                whole group

        """
        activeSynths = self.activeSynths(sortby="start")
        for i in range(n):
            if activeSynths:
                last = activeSynths[-1]
                assert last.synthid in self._synths
                last.stop(stopParent=unschedParent)

    def unschedByName(self, instrname:str) -> None:
        """
        Unschedule all playing synths created from given instr
        (as identified by the name)
        """
        synths = self.findSynthsByName(instrname)
        for synth in synths:
            self.unsched(synth.synthid)

    def unschedAll(self, cancel_future=True, allow_fadeout=0.05) -> None:
        """
        Unschedule all playing synths
        """
        synthids = [synth.synthid for synth in self._synths.values()]
        pendingSynths = [synth for synth in self._synths.values() if not synth.isPlaying()]
        for synthid in synthids:
            self.unsched(synthid, delay=0)
        
        if cancel_future and pendingSynths:
            if allow_fadeout:
                time.sleep(allow_fadeout)
            self.getEngine().unschedFuture()
            self._synths.clear()

    def findSynthsByName(self, instrname:str) -> List[Synth]:
        """
        Return a list of active Synths created from the given instr
        """
        out = []
        for synthid, synth in self._synths.items():
            if synth.instrName == instrname:
                out.append(synth)
        return out

    def restart(self) -> None:
        """
        Restart the associated engine

        Use this when in need of st
        """
        engine = self.getEngine()
        engine.restart()
        for i, initcode in enumerate(self._initCodes):
            print(f"code #{i}: initCode")
            engine.evalCode(initcode)

    def readSoundfile(self, path:str, chan=0, free=False) -> TableProxy:
        """
        Read a soundfile, store its metadata in a TableDef

        Args:
            path: the path to a soundfile
            chan: the channel to read, or 0 to read all channels into a
                (possibly) stereo or multichannel table
            free: free the table when the returned TableDef is itself deallocated

        Returns:
            a TableProxy, holding information like
            .tabnum: the table number
            .path: the path you just passed
            .nchnls: the number of channels in the soundfile
            .sr: the sample rate of the soundfile

        """
        table = self._pathToTable.get(path)
        if table:
            return table
        engine = self.getEngine()
        tabnum = engine.readSoundfile(path=path, chan=chan)
        info = sndinfo(path)
        table = TableProxy(tabnum=tabnum,
                           path=path,
                           sr=info.samplerate,
                           nchnls=info.channels,
                           manager=self,
                           numframes=info.nframes,
                           freeself=free)

        self._registerTable(table)
        return table

    def _registerTable(self, tabproxy: TableProxy) -> None:
        self._tabnumToTable[tabproxy.tabnum] = tabproxy
        if tabproxy.path:
            self._pathToTable[tabproxy.path] = tabproxy

    def playSample(self, sample:U[int, TableProxy, str], chan=1, gain=1.,
                   dur=-1., speed=1., loop=False, delay=0., pan=-1,
                   start=0., fade=-1, gaingroup=0) -> Synth:
        """
        Play a sample

        Args:
            sample: a path to a sample, a TableDef, or a table number
            dur: the duration of playback (-1 to play the whole sample)
            chan: the channel to play the sample to. In the case of multichannel
                  samples, this is the first channel
            pan: a value between 0-1. -1 means default, which is 0 for mono,
                0.5 for stereo. For multichannel (3+) samples, panning is not
                taken into account
            gain: gain factor. See also: gaingroup
            speed: speed of playback
            loop: True/False or -1 to loop as defined in the file itself (not all
                file formats define loop points)
            delay: time to wait before playback starts
            start: the starting playback time (0=play from beginning)
            fade: fade in/out in secods. -1=default
            gaingroup: the idx of a gain group. The gain of all samples routed to the
                same group are scaled by the same value and can be altered as a group
                via CsoundEngine.setSubGain(idx, gain)

        Returns:
            A Synth

        """
        if isinstance(sample, int):
            tabnum = sample
        elif isinstance(sample, TableProxy):
            tabnum = sample.tabnum
        else:
            table = self.readSoundfile(sample)
            tabnum = table.tabnum
        # isndtab, iloop, istart, ifade
        return self.sched('.playgen2',
                          delay=delay,
                          dur=dur,
                          pargs=[tabnum, int(loop), start, fade, gaingroup],
                          tabargs=dict(gain=gain, speed=speed, chan=chan, pan=pan))

    def makeRenderer(self, sr=44100, nchnls=1, ksmps=64, a4=442.) -> Renderer:
        """
        Create a Renderer (to render offline) with the instruments defined
        in this Manager

        To schedule events, use the .sched method of the renderer

        sr, nchnls, ksmps, a4 are passed to the OfflineRenderer
        """
        rendered = Renderer(sr=sr, nchnls=nchnls, ksmps=ksmps, a4=a4)
        for instrname, instrdef in self.instrDefs.items():
            rendered.defInstr(instrname, instrdef.body)
        return rendered

    def defBuiltinInstrs(self):
        for csoundInstr in builtinInstrs:
            self.registerInstr(csoundInstr)


@dataclass
class _OfflineInstrDef:
    """
    Defines an instrument for offline rendering
    """
    body: str
    tabledef: Dict[str, float] = None
    tableinit: List[float] = None
    tablemap: Dict[str, int] = None

    def __post_init__(self):
        if self.tabledef is not None:
            self.tableinit = list(self.tabledef.values())
            self.tablemap = {key:i for i, key in enumerate(self.tabledef.keys())}

    def override(self, **kws):
        if self.tabledef is None:
            raise ValueError("This instrument has no associated table")
        out = self.tableinit.copy()
        for key, value in kws.items():
            idx = self.tablemap[key]
            out[idx] = value
        return out


class Renderer:
    def __init__(self, sr=44100, nchnls=1, ksmps=64, a4=None,
                 maxpriorities=10, bucketsize=100):
        """
        Create an offline renderer.

        Instruments with higher priority are assured to be evaluated later
        in the chain. Instruments within a given priority are evaluated in
        the order they are defined (first defined is evaluated first)

        Args:
            sr: the sampling rate
            nchnls: number of channels
            ksmps: csound ksmps
            a4: reference frequency
            maxpriorities: max. groups
            bucketsize: number of instruments per group
        """
        a4 = a4 or m2f(69)
        self.csd = csound.Csd(sr=sr, nchnls=nchnls, ksmps=ksmps, a4=a4)
        self._name2instrnum = {}
        self._instrnum2name = {}
        self._numbuckets = maxpriorities
        self._bucketCounters = [0] * maxpriorities
        self._bucketSize = bucketsize
        self._instrdefs: Dict[str, CsoundInstr] = {}

        # a list of i events, starting with p1
        self.events: List[List[float]] = []
        self.unscheduledEvents: List[List[float]] = []

    def commitInstrument(self, instrname, priority=1):
        """
        Generates a concrete version of the instrument
        (with the given priority).
        Returns the instr number

        Args:
            instrname: the name of the previously defined instrument to commit
            priority: the priority of this version, will define the order
                of execution (higher priority is evaluated later)

        Returns:
            The instr number (as in "instr xx ... endin" in a csound orc)

        """
        assert 1 <= priority <= self._numbuckets

        instrnum = self._name2instrnum.get((instrname, priority))
        if instrnum is not None:
            return instrnum

        instrdef = self._instrdefs.get(instrname)
        if not instrdef:
            raise KeyError(f"instrument {instrname} is not defined")

        count = self._bucketCounters[priority]
        if count > self._bucketSize:
            raise ValueError(f"Too many instruments ({count}) defined, max. is {self._bucketSize}")

        self._bucketCounters[priority] += 1
        instrnum = priority * self._bucketSize + count
        self._name2instrnum[(instrname, priority)] = instrnum
        self._instrnum2name[instrnum] = (instrname, priority)
        self.csd.add_instr(instrnum, instrdef.body)
        return instrnum

    def defInstr(self,
                 instrname:str,
                 body:str,
                 tabledef:Dict[str, float] = None,
                 generateTableCode = True
                 ) -> None:
        """
        Define an instrument in this Renderer. Only defined instruments
        can be scheduled.

        Any global/init code needed by the instrument can be added via .addGlobal

        Args:
            instrname: the name used to identify this instrument
            body: the body of the instrument
            tabledef: the definition of its attached table, if any. Either a dict
            of the form param:value or a list of values.
            generateTableCode: if True (default), code is generated to handle the params table

        """
        if instrname in self._instrdefs:
            logger.info(f"Instrument {instrname} alread defined")
        
        instr = CsoundInstr(name=instrname, 
                            body=body,
                            tabledef=tabledef,
                            generateTableCode=generateTableCode)

        self._instrdefs[instrname] = instr

    def isInstrDefined(self, instrname:str) -> bool:
        return instrname in self._instrdefs

    def definedInstruments(self) -> List[str]:
        return list(self._instrdefs.keys())

    def addGlobal(self, code: str) -> None:
        """
        Add global code (instr 0)
        """
        self.csd.add_global(code)

    def sched(self, instrname:str, priority=1, delay=0., dur=-1.,
              args: List[float] = None,
              tabargs: Dict[str, float] = None) -> None:
        """

        Args:
            instrname: the name of the already registered instrument
            priority: the priority 1-9, will decide the order of
                execution
            delay: time offset
            dur: duration of this event. -1: endless
            args: pargs beginning with p5
                (p1: instrnum, p2: delay, p3: duration, p4: tabnum)
            tabargs: a dict of the form param: value, to initialize
                values in the exchange table (if defined by the given
                instrument)
        """
        instrdef = self._instrdefs.get(instrname)
        if not instrdef:
            raise KeyError(f"instrument {instrname} is not defined")
        instrnum = self.commitInstrument(instrname, priority)
        if instrdef.hasExchangeTable():
            tableinit = instrdef.overrideTable(tabargs)
            tabnum = self.csd.add_ftable_from_seq(tableinit)
        else:
            tabnum = 0
        pargs4 = [tabnum]
        if args:
            pargs4.extend(args)
        self.csd.add_event(instrnum, start=delay, dur=dur, args=pargs4)

    def render(self, outfile:str, wait=True, quiet=False) -> None:
        """
        Render to a soundfile

        To further customize the render, use the underlying .csd:

        renderer.csd.set_sample_format(24)
        renderer.csd.set_options("--env:VARIABLE=value", "--omacro:XXX=yyy")

        By default, if the output is an uncompressed file (.wav, .aif)
        the sample format is set to float32 (csound defaults to 16 bit pcm)

        Args:
            outfile: the output file to render to. "dac" can be used to
                render in realtime. None will render to a temporary
                .wav file.
            wait: if True, .render will block until the underlying process
                exits
            quiet: if True, all output from the csound subprocess is
                supressed
        """
        if not self.csd.score:
            raise RenderError("score is empty")
        kws = {}
        if quiet:
            kws['supressdisplay'] = True
            kws['piped'] = True
        proc = self.csd.run(output=outfile, **kws)
        if wait:
            proc.wait()

    def generateCsd(self) -> str:
        import io
        stream = io.StringIO()
        self.csd.write_csd(stream)
        return stream.getvalue()


def _qualifiedName(name:str, priority:int) -> str:
    return f"{name}:{priority}"


def _instrWrapBody(body:str, instrnum:int, notify=True) -> str:
    s = """
instr {instrnum}
    {notifystr}
    {body}
endin
    """
    if notify:
        notifystr = 'atstop "_notifyDealloc", 0, -1, p1'
    else:
        notifystr = ""

    s = s.format(instrnum=instrnum, body=body, notifystr=notifystr)
    s = _textwrap.dedent(s)
    return s


def getManager(name="default") -> InstrManager:
    """
    Get a specific Manager. A Manager controls a series of
    instruments and has its own csound engine
    """
    engine = getEngine(name)
    if not engine:
        logger.error(f"engine {name} not created. First create an engine via "
                     f"CsoundEngine(...)")
        raise NoEngineError(f"Engine {name} not active")
    manager = _managers.get(name)
    if not manager:
        manager = InstrManager(name)
        _managers[name] = manager
    return manager 


def unschedAll(instance='default') -> None:
    man = getManager(instance)
    man.unschedAll()


# def evalCode(code:str, group='default', once=False) -> float:
#    return getManager(group).getEngine().evalCode(code, once=once)


# def availableInstrs(group='default'):
#     return getManager(name=group).instrDefs.keys()


# ---------------------------------------------
# -            Builtin Instruments            -
# ---------------------------------------------

# chan = primer canal
# pos (-1, 0, 1) = panning

builtinInstrs = [
    CsoundInstr('.sinegliss', body="""
        ioutchan, iAmp, iPitchStart, iPitchEnd passign 5
        kmidi linseg iPitchStart, p3, iPitchEnd
        kfreq = mtof:k(kmidi)
        aenv linsegr 0, 0.01, 1, 0.05, 0
        a0 oscili iAmp, kfreq
        a0 *= aenv
        outch ioutchan, a0
        """),
    CsoundInstr('.sine', body="""
        iAmp, iFreq, iChan passign 5
        kenv linsegr 0, 0.04, 1, 0.08, 0
        a0 oscil iAmp, iFreq
        a0 *= kenv
        outch iChan, a0
        """),
    CsoundInstr('.playgen2', 
                tabledef=dict(chan=1, speed=1, gain=1, pan=-1),
                body=r"""
        isndtab, iloop, istart, ifade, igaingroup passign 5

        ifade = ifade < 0 ? 0.05 : ifade
        igaingroup = limit(igaingroup, 0, 100)
        inumouts = ftchnls(isndtab)
        inumsamples = nsamp(isndtab)
        isr = ftsr(isndtab)
        idur = inumsamples / isr
        ixfade = 0.005
        know init istart
        ksubgain = table:k(igainGroup, gi__subgains)
                
        if inumouts == 0 then
            ; not a gen1 table, fail
            initerror sprintf("Table %d was not generated via gen1", isndtab)
        endif

        kidx init 0
        aenv = linsegr:a(0, ifade, 1, ifade, 0)
        aenv *= a(ksubgain)
            
        ; if p3 > 0 then
        ;     aenv = linseg(0, ifade, 1, p3-ifade*2, 0)
        ; else
        ;     aenv = linsegr:a(0, ifade, 1, ifade, 0)
        ; endif
        aenv *= interp(kgain)
        if inumouts == 1 then
            ; a1 loscil3 1, kspeed, isndtab, 1, iloop
            ; asig1[,asig2] flooper2 kamp, kpitch, kloopstart, kloopend, kcrossfade, ifn \
            ; [, istart, imode, ifenv, iskip]
            a1 flooper2 1, kspeed, istart, idur, ixfade, isndtab, istart
            a1 *= aenv
            kpan = kpan == -1 ? 0 : kpan
            aL, aR pan2 a1, kpan
            outch kchan, aL, kchan+1, aR
        elseif inumouts == 2 then
            a1, a2 loscil3 1, kspeed, isndtab, 1, iloop
            a1 *= aenv
            a2 *= aenv
            kpan = kpan == -1 ? 0.5 : kpan
            aL, aR _panstereo a1, a2, kpan
            outch kchan, aL, kchan+1, aR
        else
            ; 4: cubic interpolation
            aouts[] loscilx 1, kspeed, isndtab, 4, 0, iloop
            aouts *= aenv
            ichan table 0, i_params
            ichans[] genarray ichan, ichan+inumouts-1
            poly0 inumouts, "outch", ichans, aouts
        endif   
        ionecycle = ksmps/sr
        know += ionecycle * kspeed
        imaxtime = idur - ifade - ionecycle
        if iloop == 0 && know >= imaxtime then
            turnoff
        endif
        """),
    CsoundInstr('.playbuf', body="""
        pset 0, 0, 0, 0, 0, 1, 1, 0
        itabnum, ioutchan, igain, iloop passign 5
        inumsamps ftlen itabnum
        idur = inumsamps / ftsr(itabnum)
        
        if (iloop == 1) then
            iPitch = 1
            iCrossfade = 0.050
            aSig flooper2 igain, iPitch, 0, idur, iCrossfade, itabnum
        else
            iReadfreq = sr / inumsamps; frequency of reading the buffer
            aSig poscil3 igain, iReadfreq, itabnum
            iatt = 1/kr
            irel = 3*iatt
            aEnv linsegr 0, iatt, 1, idur-iatt*2, 1, iatt, 0
            aSig *= aEnv
            if timeinsts() > idur then
                turnoff
            endif
        endif
        outch ioutchan, aSig
        """)
]
