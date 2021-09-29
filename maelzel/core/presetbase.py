from __future__ import annotations
import dataclasses
import math
import re
import textwrap as _textwrap
import emlib.textlib as _textlib
from .workspace import activeConfig
import csoundengine

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


_INSTR_INDENT = "    "


@dataclasses.dataclass
class AudiogenAnalysis:
    signals: Set[str]
    numSignals: int
    minSignal: int
    maxSignal: int
    numOutchs: int
    needsRouting: bool
    numOutputs: int


def analyzeAudiogen(audiogen:str, check=True) -> AudiogenAnalysis:
    """
    Analyzes the audio generating part of an instrument definition,
    returns the analysis results as a dictionary

    Args:
        audiogen: as passed to play.defPreset
        check: if True, will check that audiogen is well formed

    Returns:
        a dict with keys:
            numSignals (int): number of a_ variables
            minSignal: min. index of a_ variables
            maxSignal: max. index of a_ variables (minSignal+numSignals=maxSignal)
            numOutchs: number of
                (normally minsignal+numsignals = maxsignal)
    """
    audiovarRx = re.compile(r"\baout[1-9]\b")
    outOpcodeRx = re.compile(r"\boutch\b")
    audiovarsList = []
    numOutchs = 0
    for line in audiogen.splitlines():
        foundAudiovars = audiovarRx.findall(line)
        audiovarsList.extend(foundAudiovars)
        outOpcode = outOpcodeRx.fullmatch(line)
        if outOpcode is not None:
            opcode = outOpcode.group(0)
            args = line.split(opcode)[1].split(",")
            assert len(args)%2 == 0
            numOutchs = len(args) // 2

    if not audiovarsList:
        raise ValueError(f"Invalid audiogen: no output audio signals (aoutx): {audiogen}")
    audiovars = set(audiovarsList)
    chans = [int(v[4:]) for v in audiovars]
    maxchan = max(chans)
    # check that there is an audiovar for each channel
    if check:
        if len(audiovars) == 0:
            raise ValueError("audiogen defines no output signals (aout_ variables)")

        for i in range(1, maxchan+1):
            if f"aout{i}" not in audiovars:
                raise ValueError("Not all channels are defined", i, audiovars)

    needsRouting = numOutchs == 0
    numSignals = len(audiovars)
    if needsRouting:
        numOuts = int(math.ceil(numSignals/2))*2
    else:
        numOuts = numOutchs

    return AudiogenAnalysis(signals=audiovars,
                            numSignals=numSignals,
                            minSignal=min(chans),
                            maxSignal=max(chans),
                            numOutchs=numOutchs,
                            needsRouting=needsRouting,
                            numOutputs=numOuts)


def _instrNameFromPresetName(presetName: str) -> str:
    # an Instr derived from a PresetDef gets a prefix to prevent collisions
    # with Instrs a user might want to define in the same Session
    return f'preset.{presetName}'


def _makePresetBody(audiogen:str,
                    numsignals=1,
                    generateRouting=True,
                    epilogue=''):
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

ifade0 = max:i(ifade0, 1/kr)
ifade1 = max:i(ifade1, 1/kr)

if (ifadekind_ == 0) then
    aenv_ linseg 0, ifade0, igain_, p3-ifade0-ifade1, igain_, ifade1, 0
elseif (ifadekind_ == 1) then
    aenv_ cosseg 0, ifade0, igain_, p3-ifade0-ifade1, igain_, ifade1, 0
endif

aenv_ *= linenr:a(1, 0, ifade1, 0.01)

{audiogen}

{envelope}

{routing}

{epilogue}
    """
    envStr = "\n".join(f"aout{i} *= aenv_" for i in range(1, numsignals+1))

    if not generateRouting:
        routingStr = ""
    else:
        if numsignals == 1:
            routingStr = r"""
            if (ipos == 0) then
                outch ichan_, aout1
            else
                aL_, aR_ pan2 aout1, ipos
                outch ichan_, aL_, ichan_+1, aR_
            endif
            """
        elif numsignals == 2:
            routingStr = r"""
            aL_, aR_ panstereo aout1, aout2, ipos
            outch ichan_, aL_, ichan_+1, aR_
            """
        else:
            raise ValueError(f"numsignals can be either 1 or 2 at the moment"
                             f"if automatic routing is enabled (got {numsignals})")

    body = template.format(audiogen=_textlib.reindent(audiogen),
                           envelope=_textlib.reindent(envStr),
                           routing=_textlib.reindent(routingStr),
                           epilogue=epilogue)
    return body



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
        temporary: if True, this PresetDef will not be saved. It can only be saved by 
            calling .save explicitely

    """
    def __init__(self,
                 name: str,
                 audiogen: str = None,
                 init: str = None,
                 includes: List[str] = None,
                 epilogue: str = '',
                 params: Dict[str, float] = None,
                 numsignals: int = None,
                 numouts: int = None,
                 description: str = "",
                 priority: Optional[int] = None,
                 temporary: bool = False,
                 builtin=False
                 ):
        assert isinstance(audiogen, str)

        audiogen = _textwrap.dedent(audiogen)
        audiogenInfo = analyzeAudiogen(audiogen)
        body = _makePresetBody(audiogen,
                               numsignals=audiogenInfo.numSignals,
                               generateRouting=audiogenInfo.needsRouting,
                               epilogue=epilogue)

        self.name = name
        self.init = init
        self.includes = includes
        self.audiogen = audiogen.strip()
        self.epilogue = epilogue
        self.params = params
        self.userDefined = not builtin
        self.numsignals = numsignals if numsignals is not None else audiogenInfo.numSignals
        self.description = description
        self.priority = priority
        self.numouts = numouts if numouts is not None else audiogenInfo.numOutputs
        self.temporary = temporary
        self._consolidatedInit: str = ''
        self._instr: Optional[csoundengine.instr.Instr] = None
        self.body = body

    def __repr__(self):
        lines = []
        descr = f"({self.description})" if self.description else ""
        lines.append(f"Preset: {self.name}  {descr}")
        if self.includes:
            includesline = ", ".join(self.includes)
            lines.append(f"  includes: {includesline}")
        if self.init:
            lines.append(f"  init: {self.init.strip()}")
        if self.params:
            tabstr = ", ".join(f"{key}={value}" for key, value in self.params.items())
            lines.append(f"  {{{tabstr}}}")
        if self.audiogen:
            lines.append(f"  audiogen:")
            audiogen = _textwrap.indent(self.audiogen, _INSTR_INDENT)
            lines.append(audiogen)
        if self.epilogue:
            lines.append(f"  epilogue:")
            lines.append(_textwrap.indent(self.epilogue, "    "))
        return "\n".join(lines)

    def _repr_html_(self, theme=None, showGeneratedCode=False):
        if self.description:
            descr = f'(<i>{self.description}</i>)'
        else:
            descr = ''
        ps = [f"Preset: <b>{self.name}</b> {descr}<br>"]
        if self.init:
            init = _textwrap.indent(self.init, _INSTR_INDENT)
            inithtml = csoundengine.csoundlib.highlightCsoundOrc(init, theme=theme)
            ps.append(rf"init: {inithtml}")
            ps.append("audiogen:")
        body = self.audiogen if not showGeneratedCode else self.makeInstr().body
        if self.params:
            argstr = ", ".join(f"{key}={value}" for key, value in self.params.items())
            argstr = f"{_INSTR_INDENT}|{argstr}|"
            ps.append(csoundengine.csoundlib.highlightCsoundOrc(argstr, theme=theme))
        body = _textwrap.indent(body, _INSTR_INDENT)
        ps.append(csoundengine.csoundlib.highlightCsoundOrc(body, theme=theme))
        if self.epilogue:
            ps.append("epilogue:")
            epilogue = _textwrap.indent(self.epilogue, _INSTR_INDENT)
            ps.append(csoundengine.csoundlib.highlightCsoundOrc(epilogue, theme=theme))
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
            namedArgsMethod = activeConfig()['play.namedArgsMethod']
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
        self._consolidatedInit = _consolidateInitCode(self.init, self.includes)
        return self._consolidatedInit

    def save(self):
        """
        Save this preset to disk
        """
        from . import presetman
        man = presetman.getPresetManager()
        man.savePreset(self.name)


def _consolidateInitCode(init:str, includes:List[str]) -> str:
    if includes:
        includesCode = _genIncludes(includes)
        init = _textlib.joinPreservingIndentation((includesCode, init))
    return init


def _genIncludes(includes: List[str]) -> str:
    return "\n".join(_makeIncludeLine(inc) for inc in includes)


def _makeIncludeLine(include: str) -> str:
    if include.startswith('"'):
        return f'#include {include}'
    else:
        return f'#include "{include}"'
