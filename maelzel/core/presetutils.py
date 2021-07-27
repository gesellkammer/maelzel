from __future__ import annotations
import re
import os
import glob
import math
import dataclasses
import textwrap
import csoundengine
from emlib import misc, textlib
from .workspace import presetsPath, getConfig
from . import tools
from ._common import logger
from .presetbase import PresetDef
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *

_removeExtranousCharacters = textlib.makeReplacer({"[":"", "]":"", '"':'', "'":"", "{":"", "}":""})


def _parseIncludeStr(s: str) -> str:
    """ Remove extraneous characters, split and remove quotes """
    includes = _removeExtranousCharacters(s).split(',')
    return includes


def _parseTabledef(s):
    s = _removeExtranousCharacters(s)
    rawpairs = re.split(r",|\n", s)
    out = {}
    for rawpair in rawpairs:
        if ":" not in rawpair:
            continue
        key, value = rawpair.split(":")
        key = key.strip()
        out[key] = float(value)
    return out


_UNSET = object()


def saveYamlPreset(p: PresetDef, outpath: str) -> None:
    with open(outpath, "w") as f:
        f.write(f"name: {p.name}\n")
        if p.description:
            f.write(f"description: {p.description}\n")
        f.write(f"audiogen: |\n")
        audiogen = textwrap.indent(p.audiogen, "    ")
        f.write(audiogen)
        if not audiogen.endswith("\n"):
            f.write("\n")
        if p.params:
            f.write(f"params: {p.params}\n")
        if p.init:
            f.write(f"init: |\n")
            f.write(textwrap.indent(p.init, "    "))
        if p.priority is not None:
            f.write(f"priority: {p.priority}\n")
        if p.includes:
            f.write(f"includes: {p.includes}\n")


def loadYamlPreset(path: str) -> dict:
    import yaml
    d = yaml.safe_load(open(path))
    presetName = d['name']
    params = d.get('params')
    audiogen = d.get('audiogen')
    if not audiogen:
        raise ValueError("A preset should define an audiogen")

    audiogenInfo = analyzeAudiogen(audiogen)
    body = makePresetBody(audiogen,
                          numsignals=audiogenInfo.numSignals,
                          generateRouting=audiogenInfo.needsRouting)
    return PresetDef(body=body,
                     name=d.get('name'),
                     includes=d.get('includes'),
                     init=d.get('init'),
                     audiogen=d.get('audiogen'),
                     params=params,
                     numsignals=audiogenInfo.numSignals,
                     numouts=audiogenInfo.numOutputs)


def makeSoundfontAudiogen(sf2path: str = None, instrnum:int=None,
                          preset:Tuple[int, int]=None,
                          interpolation='linear') -> str:
    """
    Generate audiogen code for a soundfont.

    This can be used as the audiogen parameter to defPreset

    Args:
        sf2path: path to a sf2 soundfont. If None, the default fluidsynth soundfont
            is used
        instrnum: as returned via `csoundengine.csoundlib.soundfontGetInstruments`
        preset: a tuple (bank, presetnumber) as returned via
            `csoundengine.csoundlib.soundfontGetPresets`
        interpolation: refers to the wave interpolation performed on the sample
            data (options: 'linear' or 'cubic')

    .. note::
        Either an instrument number of a preset tuple must be given.

    Examples
    --------

        >>> # Add a soundfont preset with transposition
        >>> code = r'''
        ...     kpitch = kpitch + ktransp
        ... '''
        >>> code += makeSoundfontAudiogen("/path/to/soundfont.sf2", instrnum=0)
        >>> defPreset('myinstr', code, params={'ktransp': 0})

    """
    sf2path = resolveSoundfontPath(sf2path)
    ampdiv = getConfig()['play.soundfontAmpDiv']
    assert bool(instrnum) != bool(preset), "Either instrnum or preset should be given"
    if not sf2path:
        raise ValueError("No soundfont was given and no default soundfont found")
    if instrnum is not None:
        opcode = 'sfinstr' if interpolation == 'linear' else 'sfinstr3'
        audiogen = fr'''
        iSfTable sfloadonce "{sf2path}"
        inote0_ = round(p(idataidx_ + 1))
        ivel_ = p(idataidx_ + 2) * 127
        aout1, aout2  {opcode} ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch), {instrnum}, iSfTable, 1
        kfinished_  trigger detectsilence:k(aout1, 0.0001, 0.05), 0.5, 0
        if kfinished_ == 1  then
          turnoff
        endif
        '''
    else:
        bank, presetnum = preset
        opcode = 'sfplay' if interpolation == 'linear' else 'sfplay3'
        audiogen = fr'''
        ipresetidx sfPresetIndex "{sf2path}", {bank}, {presetnum}
        inote0_ = round(p(idataidx_ + 1))
        ivel_ = p(idataidx_ + 2) * 127
        aout1, aout2 {opcode} ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch), ipresetidx, 1
        kfinished_  trigger detectsilence:k(aout1, 0.0001, 0.05), 0.5, 0
        if kfinished_ == 1  then
          turnoff
        endif
        '''


    return textwrap.dedent(audiogen)


def makePreset(name: str,
               audiogen: str,
               init:str=None,
               includes: List[str] = None,
               params: Dict[str, float] = None,
               descr: str = None,
               priority: int = None,
               temporary = False,
               builtin=False,
               ) -> PresetDef:
    """
    Create a new instrument preset.

    Args:
        name: the name of the preset
        audiogen: audio generating csound code. The audio gen can use the defined
            variables 'kpitch', 'kfreq' and 'kamp' to generate a corresponding signal.
            The output signals must be placed in variables named 'aout1', 'aout2', etc.
        init: global code needed by the audiogen part (usually a table definition)
        includes: files to include
        params: a dict {parameter_name: value}
        builtin: if True, mark this PresetDef as a builtin (always present)
        descr: a description of what this preset is/does
        priority: if given, the instr has this priority as default when scheduled
        temporary: if True, preset will not be saved, eved if
            `config['play.autosavePreset']` is True
    """
    audiogen = textwrap.dedent(audiogen)
    audiogenInfo = analyzeAudiogen(audiogen)
    body = makePresetBody(audiogen,
                          numsignals=audiogenInfo.numSignals,
                          generateRouting=audiogenInfo.needsRouting)

    return PresetDef(name=name,
                     body=body,
                     init=init,
                     audiogen=audiogen,
                     params=params,
                     includes=includes,
                     numsignals=audiogenInfo.numSignals,
                     numouts=audiogenInfo.numOutputs,
                     description=descr,
                     userDefined=not builtin,
                     priority=priority)


def makePresetBody(audiogen:str,
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
    envStr = "\n".join(f"aout{i} *= aenv_" for i in range(1, numsignals+1))

    if not generateRouting:
        routingStr = ""
    else:
        if numsignals == 1:
            routingStr = r"""
            if (ipos == 0) then
                outch ichan_, a0
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

    audiogen = textwrap.dedent(audiogen)
    body = template.format(audiogen=textlib.reindent(audiogen),
                           envelope=textlib.reindent(envStr),
                           routing=textlib.reindent(routingStr))
    return body


def loadPreset(presetPath: str) -> PresetDef:
    """
    load a specific preset.

    Args:
        presetPath: the absolute path to a preset

    Returns:
        a PresetDef

    Raises `ValueError` if the preset cannot be loaded
    """
    ext = os.path.splitext(presetPath)[1]
    if ext == '.yaml' or ext == '.yml':
        return loadYamlPreset(presetPath)
    else:
        raise ValueError("Only .yaml presets are supported")


def loadPresets(skipErrors=True) -> List[PresetDef]:
    """
    loads all presets from presetsPath
    """
    basepath = presetsPath()
    presetdefs = []
    if not os.path.exists(basepath):
        logger.debug(f"Presets path does not exist: {basepath}")
        return presetdefs
    patterns = ["*.yaml", "*.ini"]
    foundpaths = []
    for pattern in patterns:
        paths = glob.glob(os.path.join(basepath, pattern))
        if paths:
            foundpaths.extend(paths)
    for path in foundpaths:
        try:
            presetDef = loadPreset(path)
            presetdefs.append(presetDef)
        except ValueError as e:
            if skipErrors:
                logger.warning(f"Could not load preset {path}:\n{e}")
            else:
                raise e
    return presetdefs


def _fixNumericKeys(d: dict):
    """
    Transform numeric keys (keys like "10") into numbers, INPLACE
    """
    numericKeys = [k for k in d.keys() if isinstance(k, str) and k.isnumeric()]
    for key in numericKeys:
        v = d.pop(key)
        d[int(key)] = v


def resolveSoundfontPath(path:str=None) -> Optional[str]:
    return (path or
            getConfig()['play.generalMidiSoundfont'] or
            csoundengine.tools.defaultSoundfontPath() or
            None)


@dataclasses.dataclass
class AudiogenAnalysis:
    """
    out = {
        'signals': audiovars,
        # number of signals defined in the audiogen
        'numSignals': numSignals,
        'minSignal': min(chans),
        'maxSignal': max(chans),
        # if numOutchs is 0, the audiogen does not implement routing
        'numOutchs': numOutchs,
        'needsRouting': needsRouting,
        'numOutputs': numOuts
    }
    """
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
            raise AudiogenError("audiogen defines no output signals (aout_ variables)")

        for i in range(1, maxchan+1):
            if f"aout{i}" not in audiovars:
                raise AudiogenError("Not all channels are defined", i, audiovars)

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