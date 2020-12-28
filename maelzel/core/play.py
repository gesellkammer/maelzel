from __future__ import annotations

"""
This module handles playing of events

Each Note, Chord, Line, etc, can express its playback in terms of _CsoundLine
events.

A _CsoundLine event is a score line with the protocol


The rest of the pfields being a flat list of values representing
the breakpoints

A breakpoint is a tuple of values of the form (offset, pitch [, amp, ...])
The size if each breakpoint and the number of breakpoints are given
by inumbps, ibplen

An instrument to handle playback should be defined with `defPreset` which handles
breakpoints and only needs the audio generating part of the csound code

Whenever a note actually is played with a given preset, this preset is
actually sent to the csound engine and instantiated/evaluated.
"""


import os
import json
import glob

from dataclasses import dataclass
from functools import lru_cache
import textwrap
import fnmatch
from datetime import datetime
from watchdog.observers import Observer as _WatchdogObserver

from maelzel.snd import csoundengine
from typing import List, Dict, Optional as Opt, Union as U, Set, Tuple


from .config import logger, presetsPath, recordPath
from .state import currentConfig, getState, pushState, popState
from .common import CsoundEvent, m2f
from . import tools
from . import presetutils


_INSTR_INDENT = "    "


class PlayEngineNotStarted(Exception): pass


@dataclass
class _InstrDef:
    """
        body: the body of the instrument preset
        name: the name of the preset
        init: any init code (global code)
        includes: #include files
        audiogen: the audio generating code
        tabledef: a dict(param1: default1, param2: default2, ...)
        description: a description of this instr definition

    """
    body: str
    name: str = None
    init: str = None
    includes: List[str] = None
    audiogen: str = None
    tabledef: Dict[U[str, int], float] = None
    userDefined: bool = False
    numsignals: int = 1
    numouts: int = 2
    description: str = ""
    _consolidatedInit: str = None

    def __post_init__(self):
        self.audiogen = self.audiogen.strip()
        
    def __repr__(self):
        lines = []
        descr = f"({self.description})" if self.description else ""
        lines.append(f"Instrdef: {self.name} {descr}")
        if self.tabledef:
            tabstr = ", ".join(f"{key}={value}" for key, value in self.tabledef.items())
            lines.append(f"    tabledef: {tabstr}")
        if self.init:
            lines.append(f"    init: {self.init}")
        if self.includes:
            includesline = ", ".join(self.includes)
            lines.append(f"    includes: {includesline}")
        if self.audiogen:
            lines.append("")
            audiogen = textwrap.indent(self.audiogen, _INSTR_INDENT)
            lines.append(audiogen)
        return "\n".join(lines)

    def globalCode(self) -> str:
        if self._consolidatedInit:
            return self._consolidatedInit
        self._consolidatedInit = init = \
            _consolidateInitCode(self.init, self.includes)
        return init



def _consolidateInitCode(init:str, includes:List[str]) -> str:
    if includes:
        includesCode = _genIncludes(includes)
        init = tools.joinCode((includesCode, init))
    return init


def _parseTableDefinition(d: dict) -> Tuple[List[float], Dict[str, int]]:
    """
    Given a dict of the sort {param: defaultValue}, create a tableinit and
    a tablemap where

    tableinit: a list of default values to populate the table passed to a note
    tablemap: a dict mapping argument name to argument index
    """
    tableinit = []
    tablemap = {}
    idx = 0
    for argname, value in d.items():
        tableinit.append(value)
        tablemap[argname] = idx
        idx += 1
    return tableinit, tablemap


_csoundPrelude = \
"""
opcode _oscsqr, a, kk
    kamp, kfreq xin
    aout vco2, 1, kfreq, 10
    aout *= a(kamp)
    xout aout
endop

opcode passignarr, i[],ii
    istart, iend xin
    idx = 0
    icnt = iend - istart + 1
    iOut[] init icnt
    while idx < icnt do
        iOut[idx] = pindex(idx+istart)
        idx += 1
    od
    xout iOut
endop

opcode sfloadonce, i, S
    Ssf2path xin
    itab chnget Ssf2path
    if (itab == 0) then
        itab sfload Ssf2path
        chnset itab, Ssf2path
    endif
    xout itab
endop

opcode panstereo, aa, aak
    a0, a1, kpos xin
    aL,  aR  pan2 a0, kpos
    aL1, aR1 pan2 a1, kpos
    aL += aL1
    aR += aR1
    xout aL, aR
endop
"""

_invalidVariables = {"kfreq", "kamp", "kpitch"}

def _tabledefGenerateCode(tabledef: dict, checkValidKeys=True) -> str:
    if checkValidKeys:
        invalid = _invalidVariables.intersection(tabledef.keys())
        if invalid:
            raise KeyError(f"The table definition uses invalid variable names: {invalid}")
    lines = []
    idx = 0
    for key, value in tabledef.items():
        varname = key if key.startswith("k") else "k" + key
        line = f"{varname} tab {idx}, itable"
        lines.append(line)
        idx += 1
    return "\n".join(lines)


def _makePresetBody(audiogen:str,
                    tabledef:dict=None,
                    numsignals=1,
                    generateRouting=True):
    template = r"""
;  3  4       5      6      7     8       9       10              11            12       13
idur, itable, igain, ichan, ipos, ifade0, ifade1, i__pitchinterp, i__fadeshape, inumbps, ibplen passign 3
idatalen = inumbps * ibplen
idatastart = 14
iArgs[] passign idatastart, idatastart + idatalen
ilastidx = idatalen - 1
iTimes[]     slicearray iArgs, 0, ilastidx, ibplen
iPitches[]   slicearray iArgs, 1, ilastidx, ibplen
iAmps[]      slicearray iArgs, 2, ilastidx, ibplen

if (itable > 0) then
    ftfree itable, 1
endif

k_time timeinsts

if i__pitchinterp == 0 then      
    ; linear midi interpolation    
    kpitch, kamp bpf k_time, iTimes, iPitches, iAmps
    kfreq mtof kpitch
elseif (i__pitchinterp == 1) then  ; cos midi interpolation
    kpitch = 60
    kamp = 0.5
    kpitch, kamp bpfcos k_time, iTimes, iPitches, iAmps
    kfreq mtof kpitch
elseif (i__pitchinterp == 2) then  ; linear freq interpolation
    iFreqs[] mtof iPitches
    kfreq, kamp bpf k_time, iTimes, iFreqs, iAmps
    kpitch ftom kfreq

elseif (i__pitchinterp == 3) then  ; cos midi interpolation
    kfreq, kamp bpfcos k_time, iTimes, iFreqs, iAmps
    kpitch ftom kfreq
endif

{tabledefstr}

{audiogen}

ifade0 = max:i(ifade0, 1/kr)
ifade1 = max:i(ifade1, 1/kr)
if (i__fadeshape == 0) then
    aenv linsegr 0, ifade0, igain, ifade1, 0
elseif (i__fadeshape == 1) then
    aenv cossegr 0, ifade0, igain, ifade1, 0
endif

{envelope}

{routing}
    """
    envStr = "\n".join(f"a{i} *= aenv" for i in range(numsignals))

    if not generateRouting:
        routingStr = ""
    else:
        if numsignals == 1:
            routingStr = r"""
            if (ipos == 0) then
                outch ichan, a0
            else
                aL, aR pan2 a0, ipos
                outch ichan, aL, ichan+1, aR
            endif
            """
        elif numsignals == 2:
            routingStr = r"""
            aL, aR panstereo a0, a1, ipos
            outch ichan, aL, ichan+1, aR
            """
        else:
            raise ValueError(f"numsignals can be either 1 or 2 at the moment"
                             f"if automatic routing is enabled (got {numsignals})")

    tabledefStr = "" if tabledef is None else _tabledefGenerateCode(tabledef)
    audiogen = textwrap.dedent(audiogen)
    body = template.format(audiogen=tools.reindent(audiogen),
                           envelope=tools.reindent(envStr),
                           tabledefstr=tools.reindent(tabledefStr),
                           routing=tools.reindent(routingStr))
    return body


def makeSoundfontAudiogen(sf2path: str = None, preset=0) -> str:
    """
    Generate audiogen code for a soundfont.
    This can be used as the audiogen parameter to defPreset

    sf2path:
        path to a sf2 soundfont. If None, the default fluidsynth soundfont
        is used
    preset:
        soundfont preset to be loaded. To find the preset number, use
        echo "inst 1" | fluidsynth "path/to/sf2" | egrep '[0-9]{3}-[0-9]{3} '

    banks are not supported
    """
    if sf2path is not None:
        sf2path = os.path.abspath(sf2path)
    else:
        sf2path = csoundengine.fluidsf2Path()
    if not os.path.exists(sf2path):
        raise FileNotFoundError(f"Soundfont file {sf2path} not found")
    audiogen = f"""
    iSfTable sfloadonce "{sf2path}"
    inote0 = p(idatastart + 1)
    ivel = p(idatastart + 2) * 127
    a0, a1  sfinstr ivel, inote0, kamp/16384, mtof:k(kpitch), {preset}, iSfTable, 1
    """
    return audiogen


def _fixNumericKeys(d: dict):
    """
    Transform numeric keys (keys like "10") into numbers, INPLACE
    """
    numericKeys = [k for k in d.keys() if isinstance(k, str) and k.isnumeric()]
    for key in numericKeys:
        v = d.pop(key)
        d[int(key)] = v


def _loadPreset(presetPath: str) -> [str, _InstrDef]:
    """
    load a specific preset.

    Args:
        presetPath: the absolute path to a preset

    Returns:
        A tuple(preset name, _InstrDef)
    """
    ext = os.path.splitext(presetPath)[1]
    if ext == '.json':
        try:
            d = json.load(open(presetPath))
        except json.JSONDecodeError:
            logger.warning(f"Could not load preset {presetPath}")
            return None
    elif ext == '.ini':
        d = presetutils.loadIniPreset(presetPath)
    else:
        raise ValueError("Only .json or .ini presets are supported")
    assert 'name' in d and 'audiogen' in d, d

    presetName = d['name']
    tabledef = d.get('tabledef')
    if tabledef:
        _fixNumericKeys(tabledef)
    audiogen = d.get('audiogen')
    if not audiogen:
        raise ValueError("A preset should define an audiogen")

    audiogenInfo = tools.analyzeAudiogen(audiogen)
    numSignals = audiogenInfo['numSignals']
    body = d.get('body')
    if not body:
        body = _makePresetBody(audiogen, tabledef=tabledef,
                               numsignals=numSignals,
                               generateRouting=audiogenInfo['needsRouting'])

    instrdef = _InstrDef(body=body,
                         name=d.get('name'),
                         includes=d.get('includes'),
                         init=d.get('init'),
                         audiogen=d.get('audiogen'),
                         tabledef=tabledef,
                         numsignals=numSignals,
                         numouts=audiogenInfo['numOutputs']
                         )
    return presetName, instrdef


def _loadPresets() -> Dict[str, _InstrDef]:
    """
    loads all presets from presetsPath, return a dict {presetName:instrdef}
    """
    basepath = presetsPath()
    presets = {}
    if not os.path.exists(basepath):
        logger.debug(f"_loadPresets: presets path does not exist: {basepath}")
        return presets
    patterns = ["preset-*.json", "preset-*.yaml", "preset-*.ini"]
    savedPresets = []
    for pattern in patterns:
        found = glob.glob(os.path.join(basepath, pattern))
        if found:
            savedPresets.extend(found)
    for path in savedPresets:
        presetDef = _loadPreset(path)
        if presetDef:
            presetName, instrdef = presetDef
            presets[presetName] = instrdef
    return presets


class _PresetManager:

    def __init__(self):
        self.instrdefs = {}
        self.presetsPath = presetsPath()
        self._prepareEnvironment()
        self._makeBuiltinPresets()
        self.loadPresets()
        self._watchdog = self._startWatchdog()

    def __del__(self):
        self._watchdog.join()

    def loadPresets(self) -> None:
        presets = _loadPresets()
        if presets:
            for presetName, instrdef in presets.items():
                self._registerPreset(presetName, instrdef)

    def _startWatchdog(self):
        observer = _WatchdogObserver()

        def presetsPathChanged(*args, **kws):
            logger.info(f"presets path changed: {args}, {kws}")
            self.loadPresets()

        observer.schedule(presetsPathChanged, self.presetsPath)
        observer.start()
        return observer

    def _prepareEnvironment(self) -> None:
        path = presetsPath()
        if not os.path.exists(path):
            os.makedirs(path)

    def _makeBuiltinPresets(self) -> None:
        """
        Defines all builtin presets
        """
        from functools import partial
        mkPreset = partial(self.defPreset, _userDefined=False, generateTableCode=True,
                           descr=None)
        builtinSf = partial(self.defPresetSoundfont, _userDefined=False)

        mkPreset('sin',  "a0 oscili a(kamp), mtof(sc_lag(kpitch+ktransp, klag))",
                 tabledef=dict(transp=0, lag=0.1),
                 descr="transposable sine wave")

        mkPreset('tri',
                 """
                 kfreq = mtof:k(lag(kpitch + ktransp, klag))
                 a0 = vco2(1, kfreq,  12) * a(kamp)
                 if kfreqratio > 0 then
                    a0 = K35_lpf(a0, kfreq*kfreqratio, kQ)
                 endif
                 """,
                 tabledef=dict(transp=0, lag=0.1, freqratio=0, Q=3),
                 descr="triangle wave with optional lowpass-filter")

        mkPreset('saw',
                 """
                 kfreq = mtof:k(lag(kpitch + ktransp, klag))
                 a0 = vco2(1, kfreq, 0) * a(kamp)
                 if kfreqratio > 0 then
                    a0 = K35_lpf(a0, kfreq*kfreqratio, kQ)
                 endif
                 """,
                 tabledef = dict(transp=0, lag=0.1, freqratio=0, Q=3),
                 descr="transposable saw with optional low-pass filtering")

        mkPreset('sqr',
                 """
                 a0 = vco2(1, mtof(lag(kpitch+ktransp, klag), 10) * a(kamp)
                 if kcutoff > 0 then
                    a0 moogladder a0, port(kcutoff, 0.05), kresonance
                 endif          
                 """,
                 tabledef=dict(transp=0, lag=0.1, cutoff=0, resonance=0.2),
                 descr="square wave with optional filtering")

        mkPreset('pulse', "a0 vco2 kamp, mtof:k(lag:k(kpitch+ktransp, klag), 2, kpwm",
                 tabledef=dict(transp=0, lag=0.1, pwm=0.5))

        builtinSf('piano',     preset=148)
        builtinSf('clarinet',  preset=61)
        builtinSf('oboe',      preset=58)
        builtinSf('flute',     preset=42)
        builtinSf('violin',    preset=47)
        builtinSf('reedorgan', preset=52)

    def defPreset(self,
                  name: str,
                  audiogen: str,
                  init:str=None,
                  includes: List[str] = None,
                  tabledef: U[List[float], Dict[str, float]] = None,
                  generateTableCode=True,
                  descr: str = None,
                  _userDefined=True,
                  ) -> _InstrDef:
        """
        Define a new instrument preset. The defined preset can be used
        as mynote.play(..., instr='name'), where name is the name of the
        preset.

        A preset is created by defining the audio generating part as
        csound code. Each preset has access to the following variables:

            kpitch: pitch as fractional midi
            kamp  : linear amplitude (0-1)
            kfreq : frequency corresponding to kpitch

        Each preset CAN have an associated ftable, passed as p4.
        If p4 is < 1, then the given preset has no associated ftable

        audiogen should generate an audio output signal named 'a0' for channel 1,
        a1 for channel 2, etc.

        Example

        audiogen = 'a0 oscili a(kamp), kfreq'

        Args:
            name: the name of the preset
            audiogen: audio generating csound code
            init: global code needed by the audiogen part (usually a table definition)
            includes: files to include
            tabledef: either a dict {parameter_name: value} or a list of default
                values (unnamed parameters)
            generateTableCode: should we generate the code reading the associated
                table
            _userDefined: internal parameter, used to identify builtin presets
            descr: a description of what this preset is/does

        Example:

            # create a preset with a dynamic parameter
            manager = getPresetManager()

            audiogen = '''
            kcutoff tab 0, p4
            kq      tab 1, p4
            asig vco2 kamp, kfreq, 10
            asig moogladder a0, sc_lag:k(kcutoff, 0.1), kq
            '''
            manager.defPreset(name='mypreset', audiogen=audiogen,
                              tabledef=dict(cutoff=4000, q=1))

        See Also:
            defPresetSoundfont
        """
        audiogen = textwrap.dedent(audiogen)
        if not generateTableCode:
            tabledef = None
        audiogenInfo = tools.analyzeAudiogen(audiogen)
        numSignals = audiogenInfo['numSignals']
        body = _makePresetBody(audiogen, tabledef,
                               numsignals=numSignals,
                               generateRouting=audiogenInfo['needsRouting'])
        numouts = audiogenInfo['numOutputs']

        instrdef = _InstrDef(name=name,
                             body=body,
                             init=init,
                             audiogen=audiogen,
                             tabledef=tabledef,
                             includes=includes,
                             numsignals=numSignals,
                             numouts=numouts,
                             description=descr,
                             userDefined=_userDefined)

        self._registerPreset(name, instrdef)
        config = currentConfig()
        if _userDefined and config['play.autosavePresets']:
            self.savePreset(name)
        return instrdef

    def defPresetSoundfont(self,
                           name:str,
                           sf2path:str=None,
                           preset=0,
                           init: str = None,
                           includes: List[str] = None,
                           postproc:str=None,
                           tabledef: U[List[float], Dict[str, float]] = None,
                           _userDefined=True) -> _InstrDef:
        """
        Define a new soundfont instrument preset

        Args:
            name: the name of the preset
            sf2path: the path to the soundfont, or None to use the default
                fluidsynth soundfont
            preset: the preset to use
            postproc: any code needed for postprocessing
            init: global code needed by postproc
            includes: files to include (if needed by init or postproc)
            tabledef: mutable values needed by postproc (if any). See defPreset

            _userDefined: internal parameter, used to identify builtin presets
                (builtin presets are not saved)

        NB: to list all presets in a soundfont in linux, use
        $ echo "inst 1" | fluidsynth violin.sf2 2>/dev/null | egrep '[0-9]{3}-[0-9]{3} '
        """
        audiogen = makeSoundfontAudiogen(sf2path=sf2path, preset=preset)
        if postproc:
            audiogen = tools.joinCode((audiogen, postproc))
        return self.defPreset(name=name, audiogen=audiogen, init=init,
                              includes=includes, tabledef=tabledef,
                              _userDefined=_userDefined)

    def _registerPreset(self, name:str, instrdef:_InstrDef) -> None:
        self.instrdefs[name] = instrdef

    def getPreset(self, name:str) -> _InstrDef:
        if name is None:
            name = currentConfig()['play.instr']
        return self.instrdefs.get(name)

    def definedPresets(self) -> Set[str]:
        return set(self.instrdefs.keys())

    def showPresets(self, pattern="*") -> None:
        if "*" not in pattern:
            pattern = f"*{pattern}*"
        for presetName, instrdef in self.instrdefs.items():
            if not fnmatch.fnmatch(presetName, pattern):
                continue
            print("\n")
            print(instrdef)

    def eventMaxNumChannels(self, event: CsoundEvent) -> int:
        """
        Given a CsoundEvent, which defines a base channel (.chan)
        calculate the number of channels needed to render/play
        this event, based on the number of outputs declared by
        the used preset

        Args:
            event: the event to calculate the max. number of channels for

        Returns:
            the max. number of channels needed to play/render this
            event

        """
        instrdef = self.getPreset(event.instr)
        if instrdef is None:
            raise KeyError(f"event has an unknown instr: {event.instr}")
        if event.position == 0:
            maxNumChannels = event.chan - 1
        else:
            maxNumChannels = instrdef.numouts + event.chan - 1
        return maxNumChannels

    def savePresets(self, pattern="*") -> None:
        """
        Saves all presets matching the pattern. Builtin presets are never saved
        """
        import fnmatch
        for name, instrdef in self.instrdefs.items():
            if instrdef.userDefined and fnmatch.fnmatch(name, pattern):
                self.savePreset(name)

    def savePreset(self, name:str) -> str:
        """
        Saves the preset in the presets folder, returns
        the path to the saved file

        Args:
            name: the name of the preset

        Returns:
            the path of the saved preset
        """
        fmt = "ini"
        instrdef = self.getPreset(name)
        if not instrdef.userDefined:
            raise ValueError(f"Can't save a builtin preset: {name}")
        path = presetsPath()

        d = {'name': name,
             'audiogen': instrdef.audiogen or ''}

        def addtags(*keys):
            for key in keys:
                val = getattr(instrdef, key, None)
                if val:
                    d[key] = val

        addtags('init', 'includes', 'numouts', 'tabledef', 'tableinit', 'tablemap')
        outpath = os.path.join(path, f"preset-{name}.{fmt}")
        if fmt == 'json':
            with open(outpath, "w") as f:
                json.dump(d, f, indent=True)
        elif fmt == 'ini':
            presetutils.saveIniPreset(d, outpath)
        else:
            raise KeyError(f"format {fmt} not supported")
        return outpath

    def makeRenderer(self,
                     events: List[CsoundEvent] = None,
                     presetNames:List[str]=None,
                     sr:int=None,
                     ksmps=None
                     ) -> csoundengine.Renderer:
        """
        Make an offline Renderer from instruments defined here

        Args:
            events: if given, schedule the events for offline rendering
            presetNames: a list of instruments to use in the renderer. Leave
                unset to use all or, if events are passed, to use the
                presets defined in the events
            sr: the samplerate of the renderer
            ksmps: if not explicitely set, will use config 'rec.ksmps'

        Returns:
            a csoundengine.Renderer
        """
        config = currentConfig()
        sr = sr or config['rec.samplerate']
        ksmps = ksmps or config['rec.ksmps']
        state = getState()
        renderer = csoundengine.Renderer(sr=sr, nchnls=1, ksmps=ksmps,
                                         a4=state.a4)
        renderer.addGlobal(_csoundPrelude)
        if events is not None:
            presetNames = list(set(ev.instr for ev in events))
        elif presetNames is None:
            presetNames = self.instrdefs.keys()

        # Define all instruments
        for presetName in presetNames:
            instrdef = self.getPreset(presetName)
            if not instrdef:
                logger.error(f"Preset {presetName} not found. "
                             f"Defined presets: {self.definedPresets()}")
                raise KeyError(f"Preset {presetName} not found")
            renderer.defInstr(presetName, instrdef.body, tabledef=instrdef.tabledef)
            globalCode = instrdef.globalCode()
            if globalCode:
                renderer.addGlobal(globalCode)
        if events:
            _schedOffline(renderer, events)

        return renderer

    def makePresetTemplate(presetName: str, edit=False) -> str:
        """
        Create a new preset with the given name. The preset can then be
        edited as a text file. If edit is True, then it is opened to be
        edited right away
        """
        return presetutils.makeIniPresetTemplate(presetName=presetName,
                                                 edit=edit)


class _OfflineRenderer:
    def __init__(self, sr=None, ksmps=64, outfile:str=None):
        self.a4 = m2f(69)
        self.sr = sr or currentConfig()['rec.samplerate']
        self.ksmps = ksmps
        self.outfile = outfile
        self.events: List[CsoundEvent] = []

    def sched(self, event:CsoundEvent) -> None:
        self.events.append(event)

    def schedMany(self, events: List[CsoundEvent]) -> None:
        self.events.extend(events)

    def render(self, outfile:str=None, wait=None, quiet=None) -> None:
        """
        Render the events scheduled until now.

        Args:
            outfile: the soundfile to generate
            wait: if True, wait until rendering is done
            quiet: if True, supress all output generated by csound itself
                (print statements and similar opcodes still produce output)

        """
        quiet = quiet or currentConfig()['rec.quiet']
        outfile = outfile or self.outfile
        recEvents(events=self.events, outfile=outfile, sr=self.sr,
                  wait=wait, quiet=quiet)

    def getCsd(self, outfile:str=None) -> str:
        """
        Generate the .csd representing which would render all
        events scheduled until now

        Args:
            outfile: if given, the .csd is saved to this file

        Returns:
            a string representing the .csd file
        """
        man = getPresetManager()
        renderer = man.makeRenderer(events=self.events, sr=self.sr,
                                    ksmps=self.ksmps)
        csdstr = renderer.generateCsd()
        if outfile:
            with open(outfile, "w") as f:
                f.write(csdstr)
        return csdstr


def recEvents(events: List[CsoundEvent], outfile:str=None,
              sr:int=None, wait:bool=None, ksmps:int=None,
              quiet=None
              ) -> str:
    """
    Record the events to a soundfile

    Args:
        events: a list of events as returned by .events(...)
        outfile: the generated file. If left unset, a temporary file is created
        sr: sample rate of the soundfile
        ksmps: number of samples per cycle (config 'rec.ksmps')
        wait: if True, wait until recording is finished. If None,
            use the config 'rec.block'
        quiet: if True, supress debug information when calling
            the csound subprocess

    Returns:
        the path of the generated soundfile

    Example:

        a = Chord("A4 C5", start=1, dur=2)
        b = Note("G#4", dur=4)
        events = []
        events.extend(a.events(chan=1))
        events.extend(b.events(gain=0.2, chan=2))
        recEvents(events, outfile="out.wav")

        # This is the same as above:

        a = Chord("A4 C5", start=1, dur=2)
        b = Note("G#4", dur=4)
        events = sum([
            a.events(chan=1),
            b.events(chan=2, gain=0.2)
        ], [])
        recEvents(events, outfile="out.wav")
    """
    if outfile is None:
        outfile = makeRecordingFilename(ext=".wav")
    man = getPresetManager()
    renderer = man.makeRenderer(events=events, sr=sr, ksmps=ksmps)
    if quiet is None:
        quiet = currentConfig()['rec.quiet']
    renderer.render(outfile, wait=wait, quiet=quiet)
    return outfile


_presetManager = _PresetManager()
defPreset = _presetManager.defPreset
showPresets = _presetManager.showPresets


def _path2name(path):
    return os.path.splitext(os.path.split(path)[1])[0].replace("-", "_")


def _makeIncludeLine(include: str) -> str:
    if include.startswith('"'):
        return f'#include {include}'
    else:
        return f'#include "{include}"'


def _genIncludes(includes: List[str]) -> str:
    return "\n".join(_makeIncludeLine(inc) for inc in includes)


def makeRecordingFilename(ext=".wav", prefix=""):
    """
    Generate a new filename for a recording.
    This is used when rendering and no outfile is given

    Args:
        ext: the extension of the soundfile (should start with ".")
        prefix: a prefix used to identify this recording

    Returns:
        an absolute path. It is guaranteed that the filename
        does not exist
    """
    path = recordPath()
    assert ext.startswith(".")
    base = datetime.now().isoformat(timespec='milliseconds')
    if prefix:
        base = prefix + base
    out = os.path.join(path, base + ext)
    assert not os.path.exists(out)
    return out


@lru_cache(maxsize=1000)
def makeInstrFromPreset(instrname: str=None) -> csoundengine.CsoundInstr:
    """
    Create a CsoundInstr from a preset

    Args:
        instrname: The name of the preset. If None, use the default
            instrument as defined in config['play.instr']
    """
    config = currentConfig()
    if instrname is None:
        instrname = config['play.instr']
    instrdef = _presetManager.getPreset(instrname)
    if instrdef is None:
        raise KeyError(f"Unknown instrument {instrname}")
    group = config['play.group']
    name = f'{group}.preset.{instrname}'
    startPlayEngine()
    manager = getPlayManager()
    csdinstr = manager.defInstr(name=name,
                                body=instrdef.body,
                                init=instrdef.globalCode(),
                                tabledef=instrdef.tabledef,
                                generateTableCode=False,
                                freetable=False)
    logger.debug(f"Created {csdinstr}")
    return csdinstr


def _soundfontToTabname(sfpath: str) -> str:
    path = os.path.abspath(sfpath)
    return f"gi_sf2func_{hash(path)%100000}"


def _soundfontToChannel(sfpath:str) -> str:
    basename = os.path.split(sfpath)[1]
    return f"_sf:{basename}"


def availableInstrs() -> Set[str]:
    """
    Returns a set of instr presets already defined

    """
    return getPresetManager().definedPresets()


def startPlayEngine(nchnls=None, backend=None) -> None:
    """
    Start the play engine with a given configuration, if necessary.

    If an engine is already active, we do nothing, even if the
    configuration is different. For that case, you need to
    stop the engine first.
    """
    config = currentConfig()
    engineName = config['play.group']
    if engineName in csoundengine.activeEngines():
        return
    nchnls = nchnls or config['play.numChannels']
    backend = backend or config['play.backend']
    logger.info(f"Starting engine {engineName} (nchnls={nchnls})")
    csoundengine.CsoundEngine(name=engineName, nchnls=nchnls,
                              backend=backend,
                              globalcode=_csoundPrelude)


def stopSynths(stop_engine=False, cancel_future=True, allow_fadeout=None):
    """
    Stops all synths (notes, chords, etc) being played

    If stopengine is True, the play engine itself is stopped
    """
    manager = getPlayManager()
    allow_fadeout = (allow_fadeout if allow_fadeout is not None
                     else currentConfig()['play.unschedFadeout'])
    manager.unschedAll(cancel_future=cancel_future, allow_fadeout=allow_fadeout)
    if stop_engine:
        getPlayEngine().stop()


def stopLastSynth(n=1) -> None:
    """
    Stop last active synth
    """
    getPlayManager().unschedLast(n=n)


def getPlayManager() -> csoundengine.InstrManager:
    config = currentConfig()
    group = config['play.group']
    if not isEngineActive():
        if config['play.autostartEngine']:
            startPlayEngine()
        else:
            raise PlayEngineNotStarted("Engine is not running. Call startPlayEngine")
    return csoundengine.getManager(group)


def restart() -> None:
    group = currentConfig()['play.group']
    manager = csoundengine.getManager(group)
    manager.unschedAll()
    manager.restart()


def isEngineActive() -> bool:
    group = currentConfig()['play.group']
    return csoundengine.getEngine(group) is not None


def getPlayEngine() -> Opt[csoundengine.CsoundEngine]:
    engine = csoundengine.getEngine(name=currentConfig()['play.group'])
    if not engine:
        logger.debug("engine not started")
        return
    return engine


def getPresetManager() -> _PresetManager:
    return _presetManager


def getPreset(preset:str) -> Opt[_InstrDef]:
    return getPresetManager().getPreset(preset)


class rendering:
    """
    This is used as a context manager to transform all calls
    to .play to be renderer offline
    """

    def __init__(self, outfile:str=None, wait=True, quiet=None,
                 **kws):
        """
        Args:
            outfile: events played within this context will be rendered
                to this file. If set to None, no rendering will be performed
                and .render needs to be called explicitely
            wait: if True, wait until rendering is done
            quiet: if True, supress any output from the csound
                subprocess (config 'rec.quiet')
            **kws: any other keywords are passed directly to
                getPresetManager().makeRenderer
                Possible keywords: sr, nchnls, ksmps

        Example:

        # this will generate a file foo.wav after leaving the `with` block
        with rendering("foo.wav"):
            chord.play(dur=2)
            note.play(dur=1, fade=0.1, delay=1)

        # You can render manually, if needed
        with rendering() as r:
            chord.play(dur=2)
            ...
            print(r.getCsd())
            r.render("outfile.wav")

        """
        self.kws = kws
        self.outfile = outfile
        self.renderer: Opt[_OfflineRenderer] = None
        self.quiet = quiet or currentConfig()['rec.quiet']
        self.wait = wait

    def __enter__(self):
        self.renderer = _OfflineRenderer(**self.kws)
        pushState(renderer=self.renderer)
        return self.renderer

    def __exit__(self, *args, **kws):
        popState()
        if self.outfile is None:
            return
        try:
            self.renderer.render(outfile=self.outfile, wait=self.wait,
                                 quiet=self.quiet)
        except csoundengine.RenderError as e:
            raise e
        # reraise exceptions if any
        return False


def _schedOffline(renderer: csoundengine.Renderer,
                  events: List[CsoundEvent],
                  _checkNchnls=True
                  ) -> None:
    """
    Schedule the given events for offline rendering. You need
    to call renderer.render(...) to actually render/play the
    scheduled events

    Args:
        renderer: a Renderer as returned by makeRenderer
        events: events as returned by, for example, chord.events(**kws)
        _checkNchnls: (internal parameter)
            if True, will check (and adjust) nchnls in
            the renderer so that it is high enough for all
            events to render properly
    """
    presetManager = getPresetManager()

    if _checkNchnls:
        maxchan = max(presetManager.eventMaxNumChannels(event)
                      for event in events)
        if renderer.csd.nchnls < maxchan:
            logger.warn(f"_schedOffline: the renderer was defined with "
                        f"nchnls={renderer.csd.nchnls}, but {maxchan}"
                        f"are needed to render the given events. "
                        f"Setting nchnls temporarily to {maxchan}")
            renderer.csd.nchnls = maxchan
    for event in events:
        pargs = event.getArgs()
        if pargs[2] != 0:
            logger.warn(f"got an event with a tabnum already set...: {pargs}")
            logger.warn(f"event: {event}")
        instrName = event.instr
        if not renderer.isInstrDefined(event.instr):
            instrdef = presetManager.getPreset(instrName)
            renderer.defInstr(instrName, body=instrdef.body, tabledef=instrdef.tabledef,
                              generateTableCode=False)
        renderer.sched(instrName, delay=pargs[0], dur=pargs[1],
                       args=pargs[3:],
                       tabargs=event.args,
                       priority=event.priority)