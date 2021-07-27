from __future__ import annotations
import textwrap as _textwrap
import emlib.textlib as _textlib
from .workspace import getConfig
import csoundengine

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


_INSTR_INDENT = "    "


def _instrNameFromPresetName(presetName: str) -> str:
    # an Instr derived from a PresetDef gets a prefix to prevent collisions
    # with Instrs a user might want to define in the same Session
    return f'preset.{presetName}'


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
                 priority: Optional[int] = None,
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
        self._instr: Optional[csoundengine.instr.Instr] = None

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
            namedArgsMethod = getConfig()['play.namedArgsMethod']
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


def _genIncludes(includes: List[str]) -> str:
    return "\n".join(_makeIncludeLine(inc) for inc in includes)


def _makeIncludeLine(include: str) -> str:
    if include.startswith('"'):
        return f'#include {include}'
    else:
        return f'#include "{include}"'


