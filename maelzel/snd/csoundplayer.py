import sys
import ctcsound
import re
import uuid
from typing import NamedTuple, Dict, List, Union, Optional as Opt
import ctypes
import atexit
import textwrap
import logging
from configdict import ConfigDict

from maelzel.snd import csoundlib
from functools import lru_cache
from string import Template as _Template

logger = logging.getLogger("maelzel.csoundplayer")

_defaultconfig = {
    'sr': 0,  # 0 indicates the default sr of the backend
    'numchannels': 2,
    'ksmps': 64,
    'linux.backend': 'jack, pulse, pa_cb',
    'A4': 442,
    'multisine.maxosc': 200,
}

_validator = {
    'numchannels::range': (1, 128),
    'sr::choices': [0, 22050, 24000, 44100, 48000, 88200, 96000],
    'ksmps::choices': [16, 32, 64, 128, 256],
    'A4::range': (410, 460)
}

config = ConfigDict("emlib.synthplayer",
                    default=_defaultconfig,
                    validator=_validator)


class CsdInstrError(ValueError):
    pass


class _SynthDef(NamedTuple):
    qname: str
    instrnum: int


_MYFLTPTR = ctypes.POINTER(ctcsound.MYFLT)

_csound_reserved_instrnum = 100
_csound_reserved_instr_turnoff = _csound_reserved_instrnum + 0

_csd: str = _Template("""
sr     = {sr}
ksmps  = {ksmps}
nchnls = {nchnls}
0dbfs  = 1
a4     = {a4}

instr _notifyDealloc
    iwhich = p4
    outvalue "__dealloc__", iwhich
    turnoff
endin

instr ${instr_turnoff}
    iwhich = p4
    turnoff2 iwhich, 4, 1
    turnoff
endin
""").safe_substitute(instr_turnoff=_csound_reserved_instr_turnoff)


@lru_cache(maxsize=1)
def fluidsf2Path():
    """
    Returns the path of the fluid sf2 file
    """
    if sys.platform == 'linux':
        sf2path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    else:
        raise RuntimeError("only works for linux right now")
    return sf2path


def testcsoundapi(dur=20, nchnls=2, backend=None, sr=None):
    backend = backend or csoundlib.get_default_backend()
    sr = sr or csoundlib.get_sr(backend)
    cs = ctcsound.Csound()
    orc = f"""
    sr = {sr}
    ksmps = 128
    nchnls = {nchnls}

    instr 1
        iperiod = 1
        kchn init -1
        ktrig metro 1/idur
        kchn = (kchn + ktrig) % nchnls
        anoise pinker
        outch kchn+1, anoise
        printk2 kchn
    endin

    schedule(1, 0, {dur})
    """
    orc = textwrap.dedent(orc)
    options = ["-d", "-odac", "-+rtaudio=%s" % backend, "-m 0"]
    for opt in options:
        cs.setOption(opt)
    cs.compileOrc(orc)
    cs.start()
    pt = ctcsound.CsoundPerformanceThread(cs.csound())
    pt.play()
    return pt


class CsoundPlayer:
    def __init__(self,
                 sr=None,
                 ksmps=None,
                 backend=None,
                 outdev="dac",
                 a4=None,
                 nchnls=None):
        """
        NB: don't create instances directly, call getPlayer
        """
        cfg = config
        backend = backend if backend is not None else cfg[
            f'{sys.platform}.backend']
        backends = csoundlib.get_audiobackends()
        if backend not in backends:
            raise ValueError(
                f"backend should be one of {backends}, but got {backend}")
        sr = sr if sr is not None else cfg['sr']
        if sr == 0:
            sr = csoundlib.get_sr(backend)
        if a4 is None:
            a4 = cfg['A4']
        if ksmps is None:
            ksmps = cfg['ksmps']
        self.sr = sr
        self.backend = backend
        self.a4 = a4
        self.ksmps = ksmps
        self.nchnls = nchnls or cfg['numchannels']

        self._fracnumdigits = 3
        self._cs = None
        self._pt = None
        self._exited = False
        self._csdstr = _csd
        self._instcounter = {}
        self._instrRegistry = {}
        self._outcallbacks = {}
        self._isOutCallbackSet = False
        self._startCsound()

    def __del__(self):
        self.stop()

    def _getinstance(self, instnum):
        n = self._instcounter.get(instnum, 0)
        n += 1
        self._instcounter[instnum] = n
        return n

    def _getfracinst(self, num, instance):
        frac = instance / (10**self._fracnumdigits)
        return num + frac

    def _startCsound(self):
        cs = ctcsound.Csound()
        orc = self._csdstr.format(sr=self.sr,
                                  ksmps=self.ksmps,
                                  nchnls=2,
                                  backend=self.backend,
                                  a4=self.a4)
        options = ["-d", "-odac", "-+rtaudio=%s" % self.backend, "-m 0"]
        for opt in options:
            cs.setOption(opt)
        logger.debug(orc)
        cs.compileOrc(orc)
        cs.start()
        pt = ctcsound.CsoundPerformanceThread(cs.csound())
        pt.play()
        self._cs = cs
        self._pt = pt

    def stop(self):
        if self._exited:
            return
        self._pt.stop()
        self._cs.stop()
        self._cs.cleanup()
        self._exited = True
        self._cs = None
        self._pt = None
        self._instcounter = {}
        self._instrRegistry = {}

    def restart(self) -> None:
        self.stop()
        self._startCsound()

    def _outcallback(self, _, chan, valptr, chantypeptr):
        func = self._outcallbacks.get(chan)
        if not func:
            return
        val = ctcsound.cast(valptr, _MYFLTPTR).contents.value
        func(chan, val)

    def registerOutvalueCallback(self, chan: str, func) -> None:
        """
        Register a function `func` which will be called whenever a
        channel `chan` is changed in csound via the "outvalue" opcode

        chan: the name of a channel
        func: a function of the form `func(chan, newvalue)`
        """
        if not self._isOutCallbackSet:
            self._isOutCallbackSet = True
            self._cs.setOutputChannelCallback(self._outcallback)
        self._outcallbacks[bytes(chan, "ascii")] = func

    def getCsound(self):
        return self._cs

    def defInstr(self, instr: str, name: str = None) -> None:
        """
        Compile a csound instrument

        instr : the instrument definition, beginning with 'instr xxx'
        name  : name of the instrument, to keep track of definitions.
        """
        if not name:
            name = _getUUID()
        lines = [l for l in instr.splitlines() if l.strip()]
        instrnum = int(lines[0].split()[1])
        self._instrRegistry[name] = (instrnum, instr)
        self._cs.compileOrc(instr)
        logger.debug(f"defInstr: {name}")
        logger.debug(instr)

    def evalCode(self, code: str):
        """
        Evaluates code at instr0 (global code, only i-rate)
        """
        return self._cs.evalCode(code)

    def sched(self, instrnum, delay: float = 0, dur: float = -1, args=[]):
        """
        Schedule an instrument

        instrnum : the instrument number
        delay    : time to wait before instrument is started
        dur      : duration of the event
        args     : any other args expected by the instrument

        Returns:
            the fractional number of the instr started.
            This can be used to kill the event later on
            (see unsched)
        """
        instance = self._getinstance(instrnum)
        instrfrac = self._getfracinst(instrnum, instance)
        pargs = [instrfrac, delay, dur]
        pargs.extend(args)
        self._pt.scoreEvent(0, "i", pargs)
        logger.debug(
            f"CsoundPlayer.sched: scoreEvent(0, 'i', {pargs})  -> {instrfrac}")
        return instrfrac

    def unsched(self, instrfrac: float, delay: float = 0) -> None:
        """
        mode: similar to turnoff2
        """
        self._pt.scoreEvent(
            0, "i", [_csound_reserved_instr_turnoff, 0, 0.1, instrfrac])


def _getUUID() -> str:
    return str(uuid.uuid1())


_players = {}  # type: Dict[str, CsoundPlayer]
_managers = {}  # type: Dict[str, _InstrManager]


@atexit.register
def _cleanup() -> None:
    _managers.clear()
    names = list(_players.keys())
    for name in names:
        stopPlayer(name)


def getPlayer(name: str = "default") -> CsoundPlayer:
    player = _players.get(name)
    if not player:
        player = initPlayer(name=name)
    return player


def initPlayer(name: str = "default",
               sr: int = None,
               backend: str = None,
               outdev: str = "dac",
               a4: float = None) -> CsoundPlayer:
    """
    This routine is only necessary if a csound engine needs to be started
    with specific parameters, which should not be saved for later.
    Otherwise, change the default values in config
    """
    if name in _players:
        raise KeyError(
            f"A Player with name {name} already exists, cannot initialize")
    player = CsoundPlayer(sr=sr, backend=backend, outdev=outdev, a4=a4)
    _players[name] = player
    return player


def stopPlayer(name: str = "default") -> None:
    player = _players.get(name)
    if not player:
        raise KeyError("player not found")
    player.stop()
    del _players[name]


class AbstrSynth:
    def stop(self):
        pass

    def isPlaying(self):
        pass


class Synth(AbstrSynth):
    """
    A user does NOT normally create a Synth. A Synth is created
    when a CsoundInstr is scheduled
    """

    def __init__(self, group: str, synthid: float) -> None:
        self.group = group
        self.synthid = synthid
        self._playing = True

    def isPlaying(self) -> bool:
        return self._playing

    def getManager(self) -> '_InstrManager':
        return getManager(self.group)

    def stop(self, delay=0) -> None:
        self.getManager().unsched(self.synthid, delay=delay)
        self._playing = False


class SynthGroup(AbstrSynth):
    """
    A SynthGroup is used to control multiple (similar) synths created
    to work together (in additive synthesis, for example)
    """

    def __init__(self, synths: List[AbstrSynth]) -> None:
        self.synths = synths

    def stop(self) -> None:
        for s in self.synths:
            s.stop()

    def isPlaying(self) -> bool:
        return any(s.isPlaying() for s in self.synths)


class CsoundInstr:
    __slots__ = ['body', 'name', 'initcode', 'group']

    def __init__(
            self,
            body: str,
            name: str,
            initcode: str = None,
            group: str = "default",
    ) -> None:
        """
        *** A CsoundInstr is created via makeInstr, DON'T CREATE IT DIRECTLY ***

        To schedule a Synth using this instrument, call .play

        body    : the body of the instr (the text BETWEEN 'instr' end 'endin')
        name    : the name of the instrument, if any
        initcode: code to be initialized at the instr0 level (tables, reading files, etc.)
        group   : the name of the group this instrument belongs to
        """
        errmsg = _checkInstr(body)
        if errmsg:
            raise CsdInstrError(errmsg)
        self.group = group
        self.name = name
        self.body = textwrap.dedent(body)
        self.initcode = textwrap.dedent(initcode) if initcode else None

    def __repr__(self):
        header = f"CsoundInstr({self.name}, group={self.group})"
        return "\n".join(
            (header, "> init\n", str(self.initcode), "\n> body", self.body))

    def _getManager(self):
        return getManager(self.group)

    def play(self, dur=-1, args=[], priority=1, delay=0):
        # type: (float, List, int, float) -> Synth
        """
        Schedules a Synth with this instrument.

        dur: the duration of the synth, or -1 to play until stopped
        args: args to be passed to the synth (p values)
        priority: a number indicating order of execution. This is only important
                  when depending on other synths
        delay: how long to wait to start the synth (this is always relative time)
        """
        manager = self._getManager()
        return manager.sched(self.name,
                             priority=priority,
                             delay=delay,
                             dur=dur,
                             args=args)


def _checkInstr(instr: str) -> str:
    """
    Returns an error message if the instrument is not well defined
    """
    lines = [l for l in (l.strip() for l in instr.splitlines()) if l]
    errmsg = ""
    if "instr" in lines[0] or "endin" in lines[-1]:
        errmsg = ("instr should be the body of the instrument,"
                  " without 'instr' and 'endin")
    return errmsg


class _InstrManager:
    """
    An InstrManager controls a series of instruments and represents
    a csound engine. It can have an exclusive CsoundPlayer associated,
    but this is an implementation detail.
    """

    def __init__(self, name: str = "default") -> None:
        self.name: str = name
        self.instrDefs = {}  # type: Dict[str, '_CsoundInstr']

        self._bucketsize: int = 1000
        self._numbuckets: int = 10
        self._buckets = [{} for _ in range(self._numbuckets)
                         ]  # type: List[Dict[str, int]]
        self._synthdefs = {}  # type: Dict[str, _SynthDef]
        self._synths = {}  # type: Dict[float, Synth]
        self._isDeallocCallbackSet = False

    def _deallocCallback(self, _, synthid):
        synth = self._synths.get(synthid)
        if synth is None:
            logger.debug(f"synth {synthid} does not exist!")
            return
        synth._playing = False
        del self._synths[synthid]
        logger.debug(f"instr {synthid} deallocated!")

    def getPlayer(self) -> CsoundPlayer:
        player = getPlayer(self.name)
        if not self._isDeallocCallbackSet:
            player.registerOutvalueCallback("__dealloc__",
                                            self._deallocCallback)
        return player

    def getInstrnum(self, instrname: str, priority: int) -> int:
        assert 1 <= priority < self._numbuckets - 1
        bucket = self._buckets[priority]
        instrnum = bucket.get(instrname)
        if instrnum is not None:
            return instrnum
        idx = len(bucket) + 1
        instrnum = self._bucketsize * priority + idx
        bucket[instrname] = instrnum
        return instrnum

    def defInstr(self, name: str, body: str,
                 initcode: str = None) -> CsoundInstr:
        """
        name     : a name to identify this instr, or None, in which case a UUID is created
        body     : the body of the instrument
        initcode : initialization code for the instr (ftgens, global vars, etc.)
        """
        if name is None:
            name = _getUUID()
        instr = self.instrDefs.get(name)
        if instr:
            logger.debug(f"Instrument already defined in group {self.name}\n"
                         "The previous definition will be used")
            return instr
        instr = CsoundInstr(name=name,
                            body=body,
                            initcode=initcode,
                            group=self.name)
        self.instrDefs[name] = instr
        if initcode:
            self.getPlayer().evalCode(initcode)
        return instr

    def getInstr(self, name: str) -> 'CsoundInstr':
        return self.instrDefs.get(name)

    def sched(self,
              instrname: str,
              priority: int = 1,
              delay: float = 0,
              dur: float = -1,
              args=[]) -> Synth:
        player = self.getPlayer()
        qname = _qualifiedName(instrname, priority)
        synthdef = self._synthdefs.get(qname)
        if synthdef is not None:
            instrnum = synthdef.instrnum
        else:
            instrdef = self.instrDefs.get(instrname)
            if instrdef is None:
                instrs = ", ".join(self.instrDefs.keys())
                raise ValueError(
                    f"sched: instrument {instrname} has not been declared in group {self.name}"
                    f". Delcared instruments are: {instrs}")
            instrnum = self.getInstrnum(instrname, priority)
            instrtxt = _instrWrapBody(instrdef.body, instrnum)
            logger.debug("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
            logger.debug(instrtxt)
            logger.debug("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
            player.defInstr(instrtxt, qname)
            self._synthdefs[qname] = _SynthDef(qname, instrnum)
        synthid = player.sched(instrnum, delay=delay, dur=dur, args=args)
        synth = Synth(self.name, synthid=synthid)
        self._synths[synthid] = synth
        return synth

    def unsched(self, synthid: Union[float, List[float]], delay=0) -> None:
        logger.debug(f"Manager: asking player to unsched {synthid}")
        player = self.getPlayer()
        if isinstance(synthid, float):
            player.unsched(synthid, delay)
        else:
            for sid in synthid:
                player.unsched(sid, delay)

    def unschedAll(self) -> None:
        synthids = [synth.synthid for synth in self._synths.values()]
        self.unsched(synthids, delay=0)


def _qualifiedName(name: str, priority: int) -> str:
    return f"{name}:{priority}"


def _instrWrapBody(body: str, instrnum: int, notify=True, dedent=True) -> str:
    if notify:
        s = """
        instr {instrnum}

        k__release release
        if changed(k__release) == 1 && k__release == 1 then
            event "i", "_notifyDealloc", 0, -1, p1
        endif

        {body}

        endin
        """
    else:
        s = """
        instr {instrnum}

        {body}

        endin
        """
    s = s.format(instrnum=instrnum, body=body)
    if dedent:
        s = textwrap.dedent(s)
    return s


def getManager(name: str = "default") -> _InstrManager:
    """
    Get a specific Manager. A Manager controls a series of
    instruments and normally has its own csound engine
    """
    manager = _managers.get(name)
    if not manager:
        manager = _InstrManager(name)
        _managers[name] = manager
    return manager


def unschedAll(group: str = 'default') -> None:
    man = getManager(group)
    man.unschedAll()


@lru_cache()
def makeInstr(body: str,
              initcode: str = None,
              name: str = None,
              group: str = 'default') -> CsoundInstr:
    """
    Creates a new CsoundInstr as part of group `group`

    To schedule a synth using this instrument use the .play method on the returned CsoundInstr

    See InstrSine for an example

    body    : the body of the instrument (the part between 'instr ...' and 'endin')
    initcode: the init code of the instrument (files, tables, etc.)
    name    : the name of the instrument, or None to assign a unique id
    group   : the group to handle the instrument
    """
    return getManager(group).defInstr(name=name, body=body, initcode=initcode)


def getInstr(name: str, group='default') -> Opt[CsoundInstr]:
    man = getManager(name=group)
    instrdef = man.getInstr(name)
    return makeInstr(body=instrdef.body,
                     initcode=instrdef.initcode,
                     name=instrdef.name,
                     group=group)


def availableInstrs(group='default'):
    man = getManager(name=group)
    return man.instrDefs.keys()


# -----------------------------------------------------------------------------


def InstrSineGliss(name='sinegliss', group='default'):
    body = """
        iDur = p3
        iAmp = p4
        iFreqStart = p5
        iFreqEnd   = p6
        imidi0 = ftom:i(iFreqStart)
        imidi1 = ftom:i(iFreqEnd)
        kmidi linseg imidi0, iDur, imidi1
        kfreq = mtof:k(kmidi)
        aenv linsegr 0, 0.01, 1, 0.05, 0
        a0 oscili iAmp, kfreq
        a0 *= aenv
        outs a0, a0
    """
    return makeInstr(body=body, name=name, group=group)


def InstrSine(name='sine', group='default'):
    body = """
        iDur = p3
        iAmp = p4
        iFreq = p5
        kenv linsegr 0, 0.04, 1, 0.08, 0
        a0 oscil iAmp, iFreq
        a0 *= kenv
        outs a0, a0
    """
    return makeInstr(body=body, name=name, group=group)


def makeDefaultInstrs():
    InstrSine()
