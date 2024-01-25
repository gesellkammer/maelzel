from __future__ import annotations
import dataclasses
import math
import re
import textwrap as _textwrap
from functools import cache

import emlib.textlib as _textlib
import csoundengine

from . import presetutils
from . import environment
from ._common import logger
from . import _tools

from maelzel.core.workspace import Workspace

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *

_INSTR_INDENT = "  "


@dataclasses.dataclass
class ParsedAudiogen:
    originalAudiogen: str
    signals: Set[str]
    numSignals: int
    minSignal: int
    maxSignal: int
    numOutchs: int
    needsRouting: bool
    numOutputs: int
    audiogen: str
    inlineArgs: dict[str, float] | None = None
    shortdescr: str = ''
    longdescr: str = ''
    argdocs: dict[str, str] | None = None


def _parseAudiogen(code: str, check=False) -> ParsedAudiogen:
    """
    Analyzes the audio generating part of an instrument definition

    Args:
        code: as passed to PresetDef
        check: if True, will check that the code is well formed

    Returns:
        a ParsedAudiogen
    """
    audiovarRx = re.compile(r"\baout[1-9]\b")
    outOpcodeRx = re.compile(r"^.*\b(outch)\b")
    audiovarsList = []
    numOutchs = 0
    audiogenlines = code.splitlines()
    for line in audiogenlines:
        # line = _stripComments(line)
        foundAudiovars = audiovarRx.findall(line)
        audiovarsList.extend(foundAudiovars)
        outOpcode = outOpcodeRx.fullmatch(line)
        if outOpcode is not None:
            opcode = outOpcode.group(0)
            args = line.split(opcode)[1].split(",")
            assert len(args) % 2 == 0
            numOutchs = len(args) // 2

    if not audiovarsList:
        logger.debug(f"Invalid audiogen: no output audio signals (aoutx): {code}")
        needsRouting = False
        audiovars = set()
    else:
        audiovars = set(audiovarsList)
        needsRouting = numOutchs == 0

    chans = [int(v[4:]) for v in audiovars]
    maxchan = max(chans) if chans else 0
    # check that there is an audiovar for each channel
    if check:
        if len(audiovars) == 0:
            raise ValueError("audiogen defines no output signals (aout_ variables)")

        for i in range(1, maxchan+1):
            if f"aout{i}" not in audiovars:
                raise ValueError("Not all channels are defined", i, audiovars)

    numSignals = len(audiovars)
    if needsRouting:
        numOuts = int(math.ceil(numSignals/2))*2
    else:
        numOuts = numOutchs

    inlineargs = csoundengine.instr.instrtools.parseInlineArgs(audiogenlines)
    if inlineargs:
        docstring = csoundengine.instr.instrtools.parseDocstring(audiogenlines[inlineargs.linenum+1:])
    else:
        docstring = None

    return ParsedAudiogen(originalAudiogen=code,
                          signals=audiovars,
                          numSignals=numSignals,
                          minSignal=min(chans) if chans else 0,
                          maxSignal=max(chans) if chans else 0,
                          numOutchs=numOutchs,
                          needsRouting=needsRouting,
                          numOutputs=numOuts,
                          inlineArgs=inlineargs.args if inlineargs else None,
                          audiogen=inlineargs.body if inlineargs else code,
                          shortdescr=docstring.shortdescr if docstring else '',
                          longdescr=docstring.longdescr if docstring else '',
                          argdocs=docstring.args if docstring else None)


def _makePresetBody(audiogen: str,
                    numsignals: int,
                    withEnvelope=True,
                    withOutput=True,
                    epilogue='') -> str:
    """
    Generate the presets body

    Args:
        audiogen: the audio generating part, needs to declare aout1, aout2, ...
        numsignals: the number of audio signals used in augiogen (generaly
            the result of analyzing the audiogen via `parseAudiogen`
        withEnvelope: do we generate envelope code?
        withOutput: do we send the audio to outch? This includes also panning
        epilogue: any code needed **after** output (things like turning off
            the event when silent)

    Returns:
        the presets body
    """
    # TODO: generate user pargs
    prologue = r'''
|kpos, kgain, idataidx_, inumbps, ibplen, ichan, ifadein, ifadeout, ipchintrp_, ifadekind| 

; common case (2 breakpoints is the minimum for a simple note)
if inumbps == 2 && ipchintrp_ == 0 then
    i__t = p(idataidx_ + ibplen)
    i__pitch0 = p(idataidx_ + 1)
    i__pitch1 = p(idataidx_ + 1 + ibplen)
    i__amp0 = p(idataidx_ + 2)
    i__amp1 = p(idataidx_ + 2 + ibplen)
    kamp = linseg:k(i__amp0, i__t, i__amp1)
    kpitch = linseg:k(i__pitch0, i__t, i__pitch1)
    kfreq mtof kpitch
    goto skip_breakpoints
endif

idatalen_ = inumbps * ibplen
iArgs[] passign idataidx_, idataidx_ + idatalen_
ilastidx_ = idatalen_ - 1
iTimes[]     slicearray iArgs, 0, ilastidx_, ibplen
iPitches[]   slicearray iArgs, 1, ilastidx_, ibplen
iAmps[]      slicearray iArgs, 2, ilastidx_, ibplen

k_time = (timeinstk() - 1) * ksmps/sr  ; use eventtime (csound 6.18)

if ipchintrp_ == 0 then      
    ; linear midi interpolation    
    kpitch, kamp bpf k_time, iTimes, iPitches, iAmps
    kfreq mtof kpitch
elseif (ipchintrp_ == 1) then  ; cos midi interpolation
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

skip_breakpoints:

ifadein = max:i(ifadein, 1/kr)
ifadeout = max:i(ifadeout, 1/kr)

    '''
    # makePresetEnvelope is defined in the preset system's prelude (presetmanager.py)
    envelope1 = r"""
    aenv_ = makePresetEnvelope(ifadein, ifadeout, ifadekind)
    aenv_ *= kgain
    """
    parts = [prologue]
    if numsignals == 0:
        withEnvelope = 0
        withOutput = 0

    if withEnvelope:
        parts.append(envelope1)
        audiovars = [f'aout{i}' for i in range(1, numsignals+1)]
        if withOutput:
            # apply envelope at the end
            indentation = _textlib.getIndentation(audiogen)
            prefix = ' ' * indentation
            envlines = [f'{prefix}{audiovar} *= aenv_' for audiovar in audiovars]
            audiogen = '\n'.join([audiogen] + envlines)
        else:
            audiogen = presetutils.embedEnvelope(audiogen, audiovars, envelope="aenv_")

    parts.append(audiogen)
    if withOutput:
        if numsignals == 1:
            routing = r"""
            aL_, aR_ pan2 aout1, kpos
            outch ichan, aL_, ichan+1, aR_
            """
        elif numsignals == 2:
            routing = r"""
            kpos = (kpos == -1) ? 0.5 : kpos
            aL_, aR_ panstereo aout1, aout2, kpos
            outch ichan, aL_, ichan+1, aR_
            """
        else:
            logger.error("Invalid preset. Audiogen:\n")
            logger.error(_textlib.reindent(audiogen, prefix="    "))
            raise ValueError(f"For presets with more than 2 outputs (got {numsignals})" 
                             " the user needs to route these manually, including applying"
                             " any panning/spatialization needed")
        parts.append(routing)
    if epilogue:
        parts.append(epilogue)
    parts = [_textlib.reindent(part) for part in parts]
    body = '\n'.join(parts)
    return body


class PresetDef:

    """
    An instrument preset definition
    
    Normally a user does not create a PresetDef directly. A PresetDef is created
    when calling :func:`~maelzel.core.presetmanager.defPreset` .
    
    A Preset is aware the pitch and amplitude of a SynthEvent and generates all the
    interface code regarding play parameters like panning position, fadetime, 
    fade shape, gain, etc. The user only needs to define the audio generating code and
    any init code needed (global code needed by the instrument, like soundfiles which 
    need to be loaded, buffers which need to be allocated, etc). A Preset can define
    any number of extra parameters (transposition, filter cutoff frequency, etc.). 
    
    Args:
        name: the name of the preset
        code: the audio generating code
        init: any init code (global code)
        includes: #include files
        epilogue: code to include after any other code. Needed when using turnoff,
            since calling turnoff in the middle of an instrument can cause undefined behaviour.    
        args: a dict(arg1: value1, arg2: value2, ...). Parameter names
            need to follow csound's naming: init-only parameters need to start with 'i',
            variable parameters need to start with 'k', string parameters start with 'S'. 
        description: a description of this instr definition
        envelope: If True, apply an envelope as determined by the fadein/fadeout
            play arguments. 
        routing: if True code is generated to output the audio
            to its corresponding channel. If False the audiogen code should
            be responsible for applying panning and sending the audio to 
            an output channel, bus, etc.
        aliases: an optional dict mapping alias parameters to their real name as
            csound variables. This is used, for example, in a Clip to provide coherence
            between names of python parameters ('speed') and their controls
            within the generated synth ('kspeed').
            
    """
    _builtinVariables = ('kfreq', 'kamp', 'kpitch')

    def __init__(self,
                 name: str,
                 code: str,
                 init='',
                 includes: list[str] = None,
                 epilogue: str = '',
                 args: Dict[str, float] = None,
                 numsignals: int = None,
                 numouts: int = None,
                 description="",
                 builtin=False,
                 properties: dict[str, Any] = None,
                 envelope=True,
                 routing=True,
                 aliases: dict[str, str] = None
                 ):
        assert isinstance(code, str)

        code = _textwrap.dedent(code)
        parsedAudiogen = _parseAudiogen(code)
        if parsedAudiogen.numSignals == 0:
            envelope = False
            routing = False

        if args and parsedAudiogen.inlineArgs:
            raise ValueError(f"Inline args ({parsedAudiogen.inlineArgs}) are not supported when "
                             f"defining named args ({args}) for PresetDef '{name}'")

        body = _makePresetBody(parsedAudiogen.audiogen,
                               numsignals=parsedAudiogen.numSignals,
                               withEnvelope=envelope,
                               withOutput=routing,
                               epilogue=epilogue)

        self.name = name
        "Name of this preset"

        self.instrname = self.presetNameToInstrName(name)
        "The name of the corresponding Instrument"

        self.init = init
        "Code run before any instance is created"

        self.includes = includes
        "Include files needed"

        self.parsedAudiogen: ParsedAudiogen = parsedAudiogen
        "The parsed audiogen (a ParsedAudiogen instance)"

        self.code = parsedAudiogen.audiogen
        "The original audio code itself"

        self.epilogue = epilogue
        "Code run after any other code"

        self.args: dict[str, float] | None = args or parsedAudiogen.inlineArgs
        "Named args, if present"

        self.userDefined = not builtin
        "Is this PresetDef user defined?"

        self.numsignals = numsignals if numsignals is not None else parsedAudiogen.numSignals
        "Number of audio signals used in the audiogen (aout1, aout2, ...)"

        self.description = description
        "An optional description"

        self.numouts = numouts if numouts is not None else parsedAudiogen.numOutputs
        "Number of outputs"

        self.body = body
        "The body of the instrument with all generated code"

        self.properties: dict[str, Any] = properties or {}
        "A dict to place user defined properties"

        self.routing = routing
        "Does this PresetDef need routing (panning, output) code to be generated?"

        self.aliases = aliases
        """Dict mapping aliases to real csound parameters"""

        self._consolidatedInit: str = ''
        self._instr: csoundengine.instr.Instr | None = None

        if self.args:
            for arg in self.args.keys():
                if arg in self._builtinVariables:
                    raise ValueError(f"Cannot use builtin variables as arguments "
                                     f"({arg} is a builtin variable)")

    def dynamicParams(self, aliases=True, aliased=False) -> dict[str, float|str]:
        """
        All dynamic params of this preset

        This includes the dynamic arguments of the preset plus the builtin
        arguments common to all presets (position, ...)

        Args:
            aliases: include aliases
            aliased: include aliased names

        Returns:
            a dict of all dynamic params of this preset and their default values
        """
        return self.getInstr().dynamicParams(aliases=aliases, aliased=aliased)

    @staticmethod
    @cache
    def presetNameToInstrName(presetname: str) -> str:
        return f'preset:{presetname}'

    def __repr__(self):
        lines = []
        descr = f"({self.description})" if self.description else ""
        lines.append(f"Preset: {self.name}  {descr}")
        info = [f"routing={self.routing}"]
        if self.properties:
            info.append(f"properties={self.properties}")
        lines.append(_textwrap.indent(', '.join(info), "    "))

        if self.includes:
            includesline = ", ".join(self.includes)
            lines.append(f"  includes: {includesline}")
        if self.init:
            lines.append(f"  init: {self.init.strip()}")
        if self.args:
            tabstr = ", ".join(f"{key}={value}" for key, value in self.args.items())
            lines.append(f"  |{tabstr}|")
        audiogen = _textwrap.indent(self.code, _INSTR_INDENT)
        lines.append(audiogen)
        if self.epilogue:
            lines.append("  epilogue:")
            lines.append(_textwrap.indent(self.epilogue, "    "))
        return "\n".join(lines)

    def isSoundFont(self) -> bool:
        """
        Is this Preset based on a soundfont?

        Returns:
            True if this Preset is based on a soundfont
        """
        return re.search(r"\bsfplay(3m|m|3)?\b", self.body) is not None

    def dump(self):
        if environment.insideJupyter:
            from IPython import display
            display.display(display.HTML(self._repr_html_(showGeneratedCode=True)))

        else:
            print(self.__repr__())

    def _repr_html_(self, theme='', showGeneratedCode=False):
        workspace = Workspace.active
        assert workspace is not None
        if not workspace.config['jupyterHtmlRepr']:
            return f'<pre style="font-size: 0.9em">{self.__repr__()}</pre>'

        span = _tools.htmlSpan
        faintcolor = ':grey2'

        if self.description:
            descr = span(self.description, italic=True, color=faintcolor)
        elif self.parsedAudiogen.shortdescr:
            descr = span(self.parsedAudiogen.shortdescr, italic=True, color=faintcolor)
        else:
            descr = ''
        header = f"Preset: <b>{self.name}</b>"
        if descr:
            header += f' - {descr}'
        ps = [header, '<br>']
        info = []

        if self.routing:
            info.append(f"routing={self.routing}")

        if self.properties:
            info.append(f"properties={self.properties}")
        if self.includes:
            info.append(f"includes={self.includes}")
        info.append(f"numouts={self.numouts}, numsignals={self.numsignals}")

        normalfont = '96%'
        smallfont = '90%'
        headerfont = normalfont
        codefont = smallfont
        argsfont = smallfont
        fontsize = normalfont

        def _header(text):
            return span(text, fontsize=headerfont, bold=True)

        if info:
            infostr = "(" + ', '.join(info) + ")\n"
            ps.append(f'<code style="font-size: {smallfont}">{_INSTR_INDENT}{span(infostr, color=faintcolor, fontsize=fontsize)}</code>')

        if self.parsedAudiogen.argdocs:
            ps.append('<ul style="line-height: 120%">')
            for argname, argdoc in self.parsedAudiogen.argdocs.items():
                ps.append(f'<li>{span(argname, fontsize=argsfont, bold=True)}: {span(argdoc, fontsize=argsfont, italic=True)}</li>')
            ps.append('</ul>')

        if self.init:
            init = _textlib.reindent(self.init, _INSTR_INDENT)
            inithtml = csoundengine.csoundlib.highlightCsoundOrc(init, theme=theme)
            ps.append(_header('init'))
            ps.append(span(inithtml, fontsize=codefont))

        ps.append(_header("code"))
        if self.args:
            argstr = ", ".join(f"{key}={value}" for key, value in self.args.items())
            argstr = f"{_INSTR_INDENT}|{argstr}|"
            arghtml = csoundengine.csoundlib.highlightCsoundOrc(argstr, theme=theme)
            ps.append(span(arghtml, fontsize=codefont))
        # TODO: solve how to generate body at this stage
        instr = self.getInstr()
        body = self.code if not showGeneratedCode else csoundengine.session.Session.defaultInstrBody(instr)
        body = _textwrap.indent(body, _INSTR_INDENT)
        bodyhtml = csoundengine.csoundlib.highlightCsoundOrc(body, theme=theme)
        ps.append(span(bodyhtml, fontsize=codefont))

        if self.epilogue:
            ps.append(_header("epilogue"))
            epilogue = _textwrap.indent(self.epilogue, _INSTR_INDENT)
            html = csoundengine.csoundlib.highlightCsoundOrc(epilogue, theme=theme)
            html = span(html, fontsize=codefont)
            ps.append(html)
        return "\n".join(ps)

    def getInstr(self) -> csoundengine.instr.Instr:
        """
        Returns the csoundengine's Instr corresponding to this PresetDef

        This method is cached, the Instr is constructed only the first time

        Returns:
            the csoundengine.Instr corresponding to this PresetDef

        """
        if self._instr:
            return self._instr

        aliases = {'position': 'kpos', 'gain': 'kgain'}
        if self.aliases:
            aliases |= self.aliases
        instr = csoundengine.Instr(name=self.instrname,
                                   body=self.body,
                                   init=self.init,
                                   includes=self.includes,
                                   args=self.args,
                                   numchans=self.numouts,
                                   aliases=aliases
                                   )

        self._instr = instr
        return instr

    def save(self) -> str:
        """
        Save this preset to disk

        All presets are saved to the presets path. Saved presets will be available
        in a future session

        .. seealso:: :func:`maelzel.core.workspace.presetsPath`
        """
        from . import presetmanager
        savedpath = presetmanager.presetManager.savePreset(self.name)
        return savedpath


def _consolidateInitCode(init: str, includes: list[str]) -> str:
    if includes:
        includesCode = _genIncludes(includes)
        init = _textlib.joinPreservingIndentation((includesCode, init))
    return init


def _genIncludes(includes: list[str]) -> str:
    return "\n".join(csoundengine.csoundlib.makeIncludeLine(inc) for inc in includes)


