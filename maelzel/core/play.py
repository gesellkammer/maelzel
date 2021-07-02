"""
This module handles playing of events

Each Note, Chord, Line, etc, can express its playback in terms of CsoundEvents

A CsoundEvent is a score line with a number of fixed fields,
user-defined fields and a sequence of breakpoints

A breakpoint is a tuple of values of the form (offset, pitch [, amp, ...])
The size if each breakpoint and the number of breakpoints are given
by inumbps, ibplen

An instrument to handle playback should be defined with `defPreset` which handles
breakpoints and only needs the audio generating part of the csound code

Whenever a note actually is played with a given preset, this preset is
 sent to the csound engine and instantiated/evaluated.
"""
from __future__ import annotations
import os
import glob

import textwrap as _textwrap
import fnmatch as _fnmatch
from datetime import datetime

import emlib.misc

import watchdog.events
from watchdog.observers import Observer as _WatchdogObserver

from emlib import textlib as _textlib
from pitchtools import m2f

import csoundengine
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Dict, Optional as Opt, Union as U, Set, Tuple
    from .csoundevent import CsoundEvent

from .config import logger
from .workspace import currentConfig, currentWorkspace, presetsPath, recordPath
from . import tools
from . import presetutils

_INSTR_INDENT = "    "


class PlayEngineNotStarted(Exception): pass

class PresetDef:

    userPargsStart = 15

    """
    An instrument definition

    Attributes:
        body: the body of the instrument preset
        name: the name of the preset
        init: any init code (global code)
        includes: #include files
        audiogen: the audio generating code
        tabledef: a dict(param1: default1, param2: default2, ...). If a named argument
            does not start with a 'k', this 'k' is prepended to it
        description: a description of this instr definition
        priority: if not None, use this priority as default is no other priority
            is given.


    """
    def __init__(self,
                 name: str,
                 body: str,
                 init: str = None,
                 includes: List[str] = None,
                 audiogen: str = None,
                 params: Dict[str, float] = None,
                 userDefined: bool = False,
                 numsignals: int = 1,
                 numouts: int = 1,
                 description: str = "",
                 priority: Opt[int] = None,
                 ):
        self.body = body
        self.name = name
        self.init = init
        self.includes = includes
        self.audiogen = audiogen.strip()
        self.params = params
        self.userDefined = userDefined
        self.numsignals = numsignals
        self.description = description
        self.priority = priority
        self.numouts = numouts
        self._consolidatedInit: str = ''
        self._instr: Opt[csoundengine.instr.Instr] = None

    def __repr__(self):
        lines = []
        descr = f"({self.description})" if self.description else ""
        lines.append(f"Preset: {self.name}  {descr}")
        if self.includes:
            includesline = ", ".join(self.includes)
            lines.append(f"    includes: {includesline}")
        if self.init:
            lines.append(f"    init: {self.init}")
        if self.params:
            tabstr = ", ".join(f"{key}={value}" for key, value in self.params.items())
            lines.append(f"    {{{tabstr}}}")
        if self.audiogen:
            # lines.append("")
            audiogen = _textwrap.indent(self.audiogen, _INSTR_INDENT)
            lines.append(audiogen)
        return "\n".join(lines)

    def _repr_html_(self, theme=None, showGeneratedCode=False):
        if self.description:
            descr = f'(<i>{self.description}</i>)'
        else:
            descr = ''
        ps = [f"Preset: <b>{self.name}</b> {descr}<br>"]
        if not showGeneratedCode:
            body = self.audiogen
        else:
            body = self.makeInstr().body
        if self.params:
            argstr = ", ".join(f"{key}={value}" for key, value in self.params.items())
            argstr = f"|{argstr}|"
            csoundcode = _textlib.joinPreservingIndentation((argstr, body))
        else:
            csoundcode = body
        csoundcode = _textwrap.indent(csoundcode, _INSTR_INDENT)
        htmlcode = csoundengine.csoundlib.highlightCsoundOrc(csoundcode, theme=theme)
        ps.append(htmlcode)
        return "\n".join(ps)

    def instrName(self) -> str:
        """
        Returns the Instr name corresponding to this Preset
        """
        return _instrNameFromPresetName(self.name)

    def makeInstr(self, namedArgsMethod:str=None) -> csoundengine.Instr:
        if self._instr:
            return self._instr
        if namedArgsMethod is None:
            namedArgsMethod = currentConfig()['play.namedArgsMethod']
        instrName = self.instrName()
        if namedArgsMethod == 'table':
            self._instr = csoundengine.Instr(name=instrName,
                                             body=self.body,
                                             init=self.globalCode(),
                                             tabledef=self.params,
                                             numchans=self.numouts)
        elif namedArgsMethod == 'pargs':
            self._instr = csoundengine.Instr(name=instrName,
                                             body=self.body,
                                             init=self.globalCode(),
                                             args=self.params,
                                             userPargsStart=PresetDef.userPargsStart,
                                             numchans=self.numouts)
        else:
            raise ValueError(f"namedArgsMethod expected 'table' or 'pargs', "
                             f"got {namedArgsMethod}")
        return self._instr

    def globalCode(self) -> str:
        if self._consolidatedInit:
            return self._consolidatedInit
        self._consolidatedInit = init = \
            _consolidateInitCode(self.init, self.includes)
        return init


def _consolidateInitCode(init:str, includes:List[str]) -> str:
    if includes:
        includesCode = _genIncludes(includes)
        init = _textlib.joinPreservingIndentation((includesCode, init))
    return init


_csoundPrelude = \
"""
/*
opcode _oscsqr, a, kk
    kamp, kfreq xin
    aout vco2, 1, kfreq, 10
    aout *= a(kamp)
    xout aout
endop
*/

opcode sfloadonce, i, S
    Spath xin
    Skey sprintf "sfloadonce:%s", Spath
    itab chnget Skey
    if (itab == 0) then
        itab sfload Spath
        chnset itab, Skey
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


def _makePresetBody(audiogen:str,
                    numsignals=1,
                    generateRouting=True):
    # TODO: generate user pargs
    template = r"""
;5          6       7      8     9     0    1      2      3          4        
idataidx_,inumbps,ibplen,igain_,ichan_,ipos,ifade0,ifade1,ipchintrp_,ifadekind_ passign 5
idatalen_ = inumbps * ibplen
iArgs[] passign idataidx_, idataidx_ + idatalen_
ilastidx = idatalen_ - 1
iTimes[]     slicearray iArgs, 0, ilastidx, ibplen
iPitches[]   slicearray iArgs, 1, ilastidx, ibplen
iAmps[]      slicearray iArgs, 2, ilastidx, ibplen

k_time timeinsts

if ipchintrp_ == 0 then      
    ; linear midi interpolation    
    kpitch, kamp bpf k_time, iTimes, iPitches, iAmps
    kfreq mtof kpitch
elseif (ipchintrp_ == 1) then  ; cos midi interpolation
    kpitch = 60
    kamp = 0.5
    kidx bisect k_time, iTimes
    kpitch interp1d kidx, iPitches, "cos"
    kamp interp1d kidx, iAmps, "cos"
    kfreq mtof kpitch
elseif (ipchintrp_ == 2) then  ; linear freq interpolation
    iFreqs[] mtof iPitches
    kfreq, kamp bpf k_time, iTimes, iFreqs, iAmps
    kpitch ftom kfreq

elseif (ipchintrp_ == 3) then  ; cos freq interpolation
    kidx bisect k_time, iTimes
    kfreq interp1d kidx, iFreqs, "cos"
    kamp interp1d kidx, iAmps, "cos"
    kpitch ftom kfreq
endif

{audiogen}

ifade0 = max:i(ifade0, 1/kr)
ifade1 = max:i(ifade1, 1/kr)

if (ifadekind_ == 0) then
    aenv_ linseg 0, ifade0, igain_, p3-ifade0-ifade1, igain_, ifade1, 0
elseif (ifadekind_ == 1) then
    aenv_ cosseg 0, ifade0, igain_, p3-ifade0-ifade1, igain_, ifade1, 0
endif
; this envelope makes sure than we fade out even if the note is
; turned off prematurely
aenv_ *= linenr:a(1, 0, ifade1, 0.01)

{envelope}

{routing}
    """
    envStr = "\n".join(f"a{i} *= aenv_" for i in range(numsignals))

    if not generateRouting:
        routingStr = ""
    else:
        if numsignals == 1:
            routingStr = r"""
            if (ipos == 0) then
                outch ichan_, a0
            else
                aL, aR pan2 a0, ipos
                outch ichan_, aL, ichan_+1, aR
            endif
            """
        elif numsignals == 2:
            routingStr = r"""
            aL, aR panstereo a0, a1, ipos
            outch ichan_, aL, ichan_+1, aR
            """
        else:
            raise ValueError(f"numsignals can be either 1 or 2 at the moment"
                             f"if automatic routing is enabled (got {numsignals})")

    audiogen = _textwrap.dedent(audiogen)
    body = template.format(audiogen=_textlib.reindent(audiogen),
                           envelope=_textlib.reindent(envStr),
                           routing=_textlib.reindent(routingStr))
    return body


def _resolveSoundfontPath(path:str=None) -> Opt[str]:
    return (path or
            currentConfig()['play.generalMidiSoundfont'] or
            csoundengine.tools.defaultSoundfontPath() or
            None)

def makeSoundfontAudiogen(sf2path: str = None, instrnum=0) -> str:
    """
    Generate audiogen code for a soundfont.

    This can be used as the audiogen parameter to defPreset
    
    Args:        
        sf2path: path to a sf2 soundfont. If None, the default fluidsynth soundfont
            is used
        instrnum: as returned via `csoundengine.csoundlib.soundfontGetInstruments`

    Examples
    --------
    
        >>> # Add a soundfont preset with transposition
        >>> code = r'''
        ...     kpitch = kpitch + ktransp
        ... '''
        >>> code += makeSoundfontAudiogen("/path/to/soundfont.sf2", instrnum=0)
        >>> defPreset('myinstr', code, params={'ktransp': 0})
        
    """
    sf2path = _resolveSoundfontPath(sf2path)
    ampdiv = currentConfig()['play.soundfontAmpDiv']
    if not sf2path:
        raise ValueError("No soundfont was given and no default soundfont found")
    audiogen = fr'''
    iSfTable sfloadonce "{sf2path}"
    inote0_ = round(p(idataidx_ + 1))
    ivel_ = p(idataidx_ + 2) * 127
    a0, a1  sfinstr ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch), {instrnum}, iSfTable, 1
    kfinished_  trigger detectsilence:k(a0, 0.0001, 0.05), 0.5, 0
    if kfinished_ == 1  then
        turnoff
    endif
    '''
    return audiogen


def _fixNumericKeys(d: dict):
    """
    Transform numeric keys (keys like "10") into numbers, INPLACE
    """
    numericKeys = [k for k in d.keys() if isinstance(k, str) and k.isnumeric()]
    for key in numericKeys:
        v = d.pop(key)
        d[int(key)] = v


def _loadPreset(presetPath: str) -> [str, PresetDef]:
    """
    load a specific preset.

    Args:
        presetPath: the absolute path to a preset

    Returns:
        A tuple(preset name, _InstrDef)
    """
    ext = os.path.splitext(presetPath)[1]
    if ext == '.ini':
        d = presetutils.loadIniPreset(presetPath)
    else:
        raise ValueError("Only .ini presets are supported")
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
        body = _makePresetBody(audiogen,
                               numsignals=numSignals,
                               generateRouting=audiogenInfo['needsRouting'])

    instrdef = PresetDef(body=body,
                         name=d.get('name'),
                         includes=d.get('includes'),
                         init=d.get('init'),
                         audiogen=d.get('audiogen'),
                         params=tabledef,
                         numsignals=numSignals,
                         numouts=audiogenInfo['numOutputs']
                         )
    return presetName, instrdef


def _loadPresets() -> Dict[str, PresetDef]:
    """
    loads all presets from presetsPath, return a dict {presetName:instrdef}
    """
    basepath = presetsPath()
    presets = {}
    if not os.path.exists(basepath):
        logger.debug(f"_loadPresets: presets path does not exist: {basepath}")
        return presets
    patterns = ["*.yaml", "*.ini"]
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


class PresetManager:

    def __init__(self):
        self.instrdefs: Dict[str, PresetDef] = {}
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

        class MyHandler(watchdog.events.FileSystemEventHandler):
            def __init__(self, manager):
                self.manager = manager

            def on_modified(self, event):
                self.manager.loadPresets()

            def on_created(self, event):
                self.manager.loadPresets()

        observer.schedule(MyHandler(self), self.presetsPath)
        observer.start()
        return observer

    def _prepareEnvironment(self) -> None:
        path = presetsPath()
        if not os.path.exists(path):
            os.makedirs(path)

    def _makeBuiltinPresets(self, sf2path:str=None) -> None:
        """
        Defines all builtin presets
        """
        from functools import partial
        mkPreset = partial(self.defPreset, _userDefined=False, descr=None)
        builtinSf = partial(self.defPresetSoundfont, _userDefined=False)

        mkPreset('sin', "a0 oscili a(kamp), mtof(lag(kpitch, 0.01))",
                 descr="simplest sine wave")

        mkPreset('tsin',  "a0 oscili a(kamp), mtof(lag(kpitch+ktransp, klag))",
                 params=dict(ktransp=0, klag=0.1),
                 descr="transposable sine wave")

        mkPreset('tri',
                 r'''
                 kfreq = mtof:k(lag(kpitch, 0.08))
                 a0 = vco2(1, kfreq,  12) * a(kamp)
                 ''',
                 descr="simple triangle wave")

        mkPreset('ttri',
                 r'''
                 kfreq = mtof:k(lag(kpitch + ktransp, klag))
                 a0 = vco2(1, kfreq,  12) * a(kamp)
                 if kfreqratio > 0 then
                    a0 = K35_lpf(a0, kfreq*kfreqratio, kQ)
                 endif
                 ''',
                 params=dict(ktransp=0, klag=0.1, kfreqratio=0, kQ=3),
                 descr="transposable triangle wave with optional lowpass-filter")

        mkPreset('saw',
                 r'''
                 kfreq = mtof:k(lag(kpitch, 0.01))
                 a0 = vco2(1, kfreq, 0) * a(kamp)
                 ''',
                 descr="simple saw-tooth")

        mkPreset('tsaw',
                 r'''
                 kfreq = mtof:k(lag(kpitch + ktransp, klag))
                 a0 = vco2(1, kfreq, 0) * a(kamp)
                 if kfreqratio > 0 then
                    a0 = K35_lpf(a0, kfreq*kfreqratio, kQ)
                 endif
                 ''',
                 params = dict(ktransp=0, klag=0.1, kfreqratio=0, kQ=3),
                 descr="transposable saw with optional low-pass filtering")

        mkPreset('tsqr',
                 r'''
                 a0 = vco2(1, mtof(lag(kpitch+ktransp, klag), 10) * a(kamp)
                 if kcutoff > 0 then
                    a0 moogladder a0, port(kcutoff, 0.05), kresonance
                 endif          
                 ''',
                 params=dict(ktransp=0, klag=0.1, kcutoff=0, kresonance=0.2),
                 descr="square wave with optional filtering")

        mkPreset('tpulse',
                 "a0 vco2 kamp, mtof:k(lag:k(kpitch+ktransp, klag), 2, kpwm",
                 params=dict(ktransp=0, klag=0.1, kpwm=0.5),
                 descr="transposable pulse with moulatable pwm")

        sf2 = _resolveSoundfontPath(path=sf2path)
        if sf2:
            progs = csoundengine.csoundlib.soundfontGetInstruments(sf2)
            prognums = {num for num, name in progs}
            if 147 in prognums:
                builtinSf('piano',     program=147)
            if 61 in prognums:
                builtinSf('clarinet',  program=61)
            if 58 in prognums:
                builtinSf('oboe',      program=58)
            if 42 in prognums:
                builtinSf('flute',     program=42)
            if 47 in prognums:
                builtinSf('violin',    program=47)
            if 52 in prognums:
                builtinSf('reedorgan', program=52)
        else:
            logger.info("No soundfont defined, builtin instruments using soundfonts will"
                        "not be available. Set config['play.generalMidiSoundfont'] to"
                        "the path of an existing soundfont")

    def defPreset(self,
                  name: str,
                  audiogen: str,
                  init:str=None,
                  includes: List[str] = None,
                  params: Dict[str, float] = None,
                  descr: str = None,
                  priority: int = None,
                  temporary = False,
                  _userDefined=True,
                  ) -> PresetDef:
        """
        Define a new instrument preset.

        The defined preset can be used as mynote.play(..., instr='name'), where name
        is the name of the preset. A preset is created by defining the audio generating
        part as csound code. Each preset has access to the following variables:

        - **kpitch**: pitch as fractional midi
        - **kamp**: linear amplitude (0-1)
        - **kfreq**: frequency corresponding to kpitch

        Each preset CAN have an associated ftable, passed as p4.
        If p4 is < 1, then the given preset has no associated ftable

        audiogen should generate an audio output signal named 'a0' for channel 1,
        a1 for channel 2, etc.

        Example::

            audiogen = 'a0 oscili a(kamp), kfreq'

        Args:
            name: the name of the preset
            audiogen: audio generating csound code
            init: global code needed by the audiogen part (usually a table definition)
            includes: files to include
            params: a dict {parameter_name: value}
            _userDefined: internal parameter, used to identify builtin presets
            descr: a description of what this preset is/does
            priority: if given, the instr has this priority as default when scheduled
            temporary: if True, preset will not be saved, eved if
                `config['play.autosavePreset']` is True

        Retursn:
            an InstrDef

        Example
        ~~~~~~~

        .. code-block:: python

            # create a preset with a dynamic parameter
            manager = getPresetManager()

            audiogen = '''
            kcutoff tab 0, p4
            kq      tab 1, p4
            asig vco2 kamp, kfreq, 10
            asig moogladder a0, lag:k(kcutoff, 0.1), kq
            '''
            manager.defPreset(name='mypreset', audiogen=audiogen,
                              params=dict(kcutoff=4000, kq=1))

        See Also:
            defPresetSoundfont
        """
        audiogen = _textwrap.dedent(audiogen)
        audiogenInfo = tools.analyzeAudiogen(audiogen)
        numSignals = audiogenInfo['numSignals']
        body = _makePresetBody(audiogen,
                               numsignals=numSignals,
                               generateRouting=audiogenInfo['needsRouting'])
        numouts = audiogenInfo['numOutputs']

        instrdef = PresetDef(name=name,
                             body=body,
                             init=init,
                             audiogen=audiogen,
                             params=params,
                             includes=includes,
                             numsignals=numSignals,
                             numouts=numouts,
                             description=descr,
                             userDefined=_userDefined,
                             priority=priority)

        self._registerPreset(name, instrdef)
        config = currentConfig()
        if _userDefined and (not temporary or config['play.autosavePresets']):
            self.savePreset(name)
        return instrdef

    def defPresetSoundfont(self,
                           name:str,
                           sf2path:str=None,
                           program: U[int, Tuple[int, int], str]=0,
                           preload=True,
                           init: str = None,
                           includes: List[str] = None,
                           postproc:str=None,
                           params: U[List[float], Dict[str, float]] = None,
                           priority: int = None,
                           _userDefined=True) -> PresetDef:
        """
        Define a new soundfont instrument preset

        Args:
            name: the name of the preset
            sf2path: the path to the soundfont, or None to use the default
                fluidsynth soundfont
            program: the program to use. Either an instr. number, as listed
                via `csoundengine.csoundlib.soundfontGetInstruments`, a
                `(bank, presetnum)` tuplet as returned by
                `csoundengine.csoundlib.soundfontGetPreset`, or an instrument
                name. In this last case a glob pattern can be used, in which case
                the first instrument matching the pattern will be used.
            preload: if True, load the soundfont at the beginning of the session
            postproc: any code needed for postprocessing. Any postprocessing should
                modify the variables a0, a1
            init: global code needed by postproc
            includes: files to include (if needed by init or postproc)
            params: mutable values needed by postproc (if any). See defPreset
            priority: default priority for this preset

            _userDefined: internal parameter, used to identify builtin presets
                (builtin presets are not saved)

        !!! note

            To list all programs in a soundfont, see
            :func:`~maelzel.play.showSoundfontPresets`

        """
        audiogen = makeSoundfontAudiogen(sf2path=sf2path, instrnum=program)
        if preload:
            # We don't need to use a global variable because sfloadonce
            # saved the table num into a channel
            init0 = f'''iSfTable sfloadonce "{sf2path}"'''
            if init:
                init = "\n".join((init0, init))
            else:
                init = init0
        if postproc:
            audiogen = _textlib.joinPreservingIndentation((audiogen, postproc))
        return self.defPreset(name=name, audiogen=audiogen, init=init,
                              includes=includes, params=params,
                              _userDefined=_userDefined,
                              priority=priority)

    def _registerPreset(self, name:str, instrdef:PresetDef) -> None:
        self.instrdefs[name] = instrdef

    def getPreset(self, name:str) -> PresetDef:
        if name is None:
            name = currentConfig()['play.instr']
        preset = self.instrdefs.get(name)
        if preset:
            return preset
        raise ValueError(f"Preset {name} not known. \n"
                         f"Presets: {self.instrdefs.keys()}")


    def definedPresets(self) -> Set[str]:
        return set(self.instrdefs.keys())

    def showPresets(self, pattern="*", showGeneratedCode=False) -> None:
        selectedPresets = [presetName for presetName in self.instrdefs.keys()
                           if _fnmatch.fnmatch(presetName, pattern)]
        if not emlib.misc.inside_jupyter():
            for presetName in selectedPresets:
                presetdef = self.instrdefs[presetName]
                print("")
                if not showGeneratedCode:
                    print(presetdef)
                else:
                    print(presetdef.makeInstr().body)
        else:
            theme = currentConfig()['html.theme']
            htmls = []
            for presetName in selectedPresets:
                presetdef = self.instrdefs[presetName]
                html = presetdef._repr_html_(theme, showGeneratedCode=showGeneratedCode)
                htmls.append(html)
            from IPython.core.display import display, HTML
            display(HTML("\n".join(htmls)))


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
        for name, instrdef in self.instrdefs.items():
            if instrdef.userDefined and _fnmatch.fnmatch(name, pattern):
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
        outpath = os.path.join(path, f"{name}.{fmt}")
        if fmt == 'ini':
            presetutils.saveIniPreset(d, outpath)
        else:
            raise KeyError(f"format {fmt} not supported")
        return outpath

    def makeRenderer(self,
                     events: List[CsoundEvent] = None,
                     presetNames:List[str]=None,
                     sr:int=None,
                     nchnls:int=None,
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
            nchnls: the number of channels
            ksmps: if not explicitely set, will use config 'rec.ksmps'

        Returns:
            a csoundengine.Renderer
        """
        config = currentConfig()
        sr = sr or config['rec.samplerate']
        ksmps = ksmps or config['rec.ksmps']
        nchnls = nchnls or config['rec.nchnls']
        state = currentWorkspace()
        renderer = csoundengine.Renderer(sr=sr, nchnls=nchnls, ksmps=ksmps,
                                         a4=state.a4)
        renderer.addGlobalCode(_csoundPrelude)
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
            renderer.defInstr(presetName, instrdef.body, tabledef=instrdef.params)
            globalCode = instrdef.globalCode()
            if globalCode:
                renderer.addGlobalCode(globalCode)
        if events:
            _schedOffline(renderer, events)

        return renderer

    def makePresetTemplate(presetName: str, edit=False) -> str:
        """
        Create a new preset template with the given name.

        The preset can then be edited as a text file. If edit is True, it is opened
        to be edited right away
        """
        return presetutils.makeIniPresetTemplate(presetName=presetName,
                                                 edit=edit)

    def openPresetsDir(self) -> None:
        """
        Open a file manager at presetsPath
        """
        path = presetsPath()
        emlib.misc.open_with_standard_app(path)

    def removeUserPreset(self, presetName: str) -> bool:
        """
        Remove a user defined preset

        Args:
            presetName: the name of the preset to remove

        Returns:
            True if the preset was removed, False if it did not exist
        """
        extensions = {'.yaml', '.ini'}
        possibleFiles = glob.glob(os.path.join(presetsPath(), f"{presetName}.*"))
        possibleFiles = [f for f in possibleFiles
                         if os.path.splitext(f)[1] in extensions]
        if not possibleFiles:
            return False
        for f in possibleFiles:
            if f.endswith('.ini') or f.endswith('.yaml'):
                os.remove(f)
        return True


class OfflineRenderer:
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
        Generate the .csd which would render all events scheduled until now

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
        outfile: the generated file. If left unset, a file inside the recording
            path is created (see `recordPath`)
        sr: sample rate of the soundfile
        ksmps: number of samples per cycle (config 'rec.ksmps')
        wait: if True, wait until recording is finished. If None,
            use the config 'rec.block'
        quiet: if True, supress debug information when calling
            the csound subprocess

    Returns:
        the path of the generated soundfile

    Example::

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


_presetManager = PresetManager()
defPreset = _presetManager.defPreset
defPresetSoundfont = _presetManager.defPresetSoundfont
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
        an absolute path. It is guaranteed that the filename does not exist.
        The file will be created inside the recording path (see ``state.recordPath``)
    """
    path = recordPath()
    assert ext.startswith(".")
    base = datetime.now().isoformat(timespec='milliseconds')
    if prefix:
        base = prefix + base
    out = os.path.join(path, base + ext)
    assert not os.path.exists(out)
    return out


def _instrNameFromPresetName(presetName: str) -> str:
    # an Instr derived from a PresetDef gets a prefix to prevent collisions
    # with Instrs a user might want to define in the same Session
    return f'preset.{presetName}'


def _registerPresetInSession(preset: PresetDef,
                             session:csoundengine.session.Session
                             ) -> csoundengine.Instr:
    """
    Create and register a :class:`csoundengine.instr.Instr` from a preset

    Args:
        preset: the PresetDef.
        session: the session to manage the instr

    Returns:
        the registered Instr
    """
    # each preset caches the generated instr
    instr = preset.makeInstr()
    # registerInstr checks itself if the instr is already defined
    session.registerInstr(instr)
    return instr


def _soundfontToTabname(sfpath: str) -> str:
    path = os.path.abspath(sfpath)
    return f"gi_sf2func_{hash(path)%100000}"


def _soundfontToChannel(sfpath:str) -> str:
    basename = os.path.split(sfpath)[1]
    return f"_sf:{basename}"


def availablePresets() -> Set[str]:
    """
    Returns the names of instr presets already defined

    """
    return getPresetManager().definedPresets()


def startPlayEngine(numChannels=None, backend=None) -> csoundengine.Engine:
    """
    Start the play engine

    If an engine is already active, nothing happens, even if the
    configuration is different. To start the play engine with a different
    configuration, stop the engine first.

    Args:
        numChannels: the number of output channels, overrides config 'play.numChannels'
        backend: the audio backend used, overrides config 'play.backend'
    """
    config = currentConfig()
    engineName = config['play.engineName']
    if engineName in csoundengine.activeEngines():
        return csoundengine.getEngine(engineName)
    numChannels = numChannels or config['play.numChannels']
    backend = backend or config['play.backend']
    logger.info(f"Starting engine {engineName} (nchnls={numChannels})")
    return csoundengine.Engine(name=engineName, nchnls=numChannels,
                               backend=backend,
                               globalcode=_csoundPrelude)


def stopSynths(stop_engine=False, cancel_future=True):
    """
    Stops all synths (notes, chords, etc) being played

    If stopengine is True, the play engine itself is stopped
    """
    manager = getPlaySession()
    manager.unschedAll(cancel_future=cancel_future)
    if stop_engine:
        getPlayEngine().stop()


def getPlaySession() -> csoundengine.Session:
    config = currentConfig()
    group = config['play.engineName']
    if not isEngineActive():
        if config['play.autostartEngine']:
            startPlayEngine()
        else:
            raise PlayEngineNotStarted("Engine is not running. Call startPlayEngine")
    return csoundengine.getSession(group)


def restart() -> None:
    """
    Restart the sound engine
    """
    group = currentConfig()['play.engineName']
    manager = csoundengine.getSession(group)
    manager.unschedAll()
    manager.restart()


def isEngineActive() -> bool:
    """
    Returns True if the sound engine is active
    """
    group = currentConfig()['play.engineName']
    return csoundengine.getEngine(group) is not None


def getPlayEngine() -> Opt[csoundengine.Engine]:
    """
    Return the sound engine, or None if it has not been started
    """
    engine = csoundengine.getEngine(name=currentConfig()['play.engineName'])
    if not engine:
        logger.debug("engine not started")
        return
    return engine


def getPresetManager() -> PresetManager:
    """
    Get the preset manager
    """
    return _presetManager


def getPreset(preset:str) -> Opt[PresetDef]:
    """
    get a defined preset
    """
    return getPresetManager().getPreset(preset)


class rendering:
    def __init__(self, outfile:str=None, wait=True, quiet=None,
                 **kws):
        """
        Context manager to transform all calls to .play to be renderer offline

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

        Example::

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
        self._oldRenderer: Opt[OfflineRenderer] = None
        self.renderer: Opt[OfflineRenderer] = None
        self.quiet = quiet or currentConfig()['rec.quiet']
        self.wait = wait

    def __enter__(self):
        workspace = currentWorkspace()
        self._oldRenderer = workspace.renderer
        self.renderer = OfflineRenderer(**self.kws)
        workspace.renderer = self.renderer
        return self.renderer

    def __exit__(self, *args, **kws):
        currentWorkspace().renderer = self._oldRenderer
        if self.outfile is None:
            return
        self.renderer.render(outfile=self.outfile, wait=self.wait,
                             quiet=self.quiet)


def _schedOffline(renderer: csoundengine.Renderer,
                  events: List[CsoundEvent],
                  _checkNchnls=True
                  ) -> None:
    """
    Schedule the given events for offline rendering.

    You need to call renderer.render(...) to actually render/play the
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
        if renderer.nchnls < maxchan:
            logger.info(f"_schedOffline: the renderer was defined with "
                        f"nchnls={renderer.csd.nchnls}, but {maxchan} "
                        f"are needed to render the given events. "
                        f"Setting nchnls to {maxchan}")
            renderer.csd.nchnls = maxchan
    for event in events:
        pargs = event.getPfields()
        if pargs[2] != 0:
            logger.warn(f"got an event with a tabnum already set...: {pargs}")
            logger.warn(f"event: {event}")
        instrName = event.instr
        assert instrName is not None
        presetdef = presetManager.getPreset(instrName)
        instr = presetdef.makeInstr()
        if not renderer.isInstrDefined(instr.name):
            renderer.registerInstr(presetdef.makeInstr())
        # renderer.defInstr(instrName, body=presetdef.body, tabledef=presetdef.params)
        renderer.sched(instrName, delay=pargs[0], dur=pargs[1],
                       pargs=pargs[3:],
                       tabargs=event.namedArgs,
                       priority=event.priority)


def playEvents(events: List[CsoundEvent]) -> csoundengine.synth.SynthGroup:
    """
    Play a list of events

    Args:
        events: a list of CsoundEvents

    Returns:
        A SynthGroup

    Example::

        TODO
    """
    synths = []
    session = getPlaySession()
    presetNames = {ev.instr for ev in events}
    p = getPresetManager()
    presetDefs = [p.getPreset(name) for name in presetNames]
    presetToInstr: Dict[str, csoundengine.Instr] = {preset.name:_registerPresetInSession(preset, session)
                                                    for preset in presetDefs}

    for ev in events:
        instr = presetToInstr[ev.instr]
        args = ev.getPfields(numchans=instr.numchans)
        synth = session.sched(instr.name,
                              delay=args[0],
                              dur=args[1],
                              pargs=args[3:],
                              tabargs=ev.namedArgs,
                              priority=ev.priority)
        synths.append(synth)
    # print("finished sched: ", time.time())
    return csoundengine.synth.SynthGroup(synths)
