from __future__ import annotations
import re
import os
import glob
import math
import dataclasses
import textwrap
import csoundengine
import emlib.dialogs
import emlib.textlib
from .workspace import presetsPath, activeConfig
from . import tools
from ._common import logger
from .presetbase import PresetDef, analyzeAudiogen
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


_removeExtranousCharacters = emlib.textlib.makeReplacer({"[":"", "]":"", '"':'', "'":"", "{":"", "}":""})


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
    """
    Serialize a PresetDef to disk as yaml
    """
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
            if not p.init.endswith("\n"):
                f.write("\n")
        if p.epilogue:
            f.write(f"epilogue: |\n")
            f.write(textwrap.indent(p.epilogue, "    "))
            if not p.epilogue.endswith("\n"):
                f.write("\n")
        if p.priority is not None:
            f.write(f"priority: {p.priority}\n")
        if p.includes:
            f.write(f"includes: {p.includes}\n")


def loadYamlPreset(path: str) -> PresetDef:
    """
    Load a PresetDef from a yaml file
    """
    import yaml
    d = yaml.safe_load(open(path))
    presetName = d.get('name')
    if not presetName:
        raise ValueError("A preset should have a name")
    params = d.get('params')
    audiogen = d.get('audiogen')
    if not audiogen:
        raise ValueError("A preset should define an audiogen")
    return PresetDef(name=d.get('name'),
                     audiogen=audiogen,
                     includes=d.get('includes'),
                     init=d.get('init'),
                     epilogue=d.get('epilogue'),
                     params=params)


def makeSoundfontAudiogen(sf2path: str = None, instrnum:int=None,
                          preset:Tuple[int, int]=None,
                          interpolation='linear',
                          ampDivisor:int=None,
                          mono=False) -> str:
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
    ampdiv = ampDivisor or activeConfig()['play.soundfontAmpDiv']
    assert bool(instrnum) != bool(preset), "Either instrnum or preset should be given"
    if not sf2path:
        raise ValueError("No soundfont was given and no default soundfont found")
    if instrnum is not None:
        if not mono:
            opcode = 'sfinstr' if interpolation == 'linear' else 'sfinstr3'
            audiogen = fr'''
            iSfTable sfloadonce "{sf2path}"
            inote0_ = round(p(idataidx_ + 1))
            ivel_ = p(idataidx_ + 2) * 127
            aout1, aout2  {opcode} ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch), {instrnum}, iSfTable, 1
            '''
        else:
            opcode = 'sfinstrm' if interpolation == 'linear' else 'sfinstr3m'
            audiogen = fr'''
            iSfTable sfloadonce "{sf2path}"
            inote0_ = round(p(idataidx_ + 1))
            ivel_ = p(idataidx_ + 2) * 127
            aout1 {opcode} ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch), {instrnum}, iSfTable, 1
            '''

    else:
        bank, presetnum = preset
        if not mono:
            opcode = 'sfplay' if interpolation == 'linear' else 'sfplay3'
            audiogen = fr'''
            ipresetidx sfPresetIndex "{sf2path}", {bank}, {presetnum}
            inote0_ = round(p(idataidx_ + 1))
            ivel_ = p(idataidx_ + 2) * 127
            aout1, aout2 {opcode} ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch), ipresetidx, 1
            '''
        else:
            opcode = 'sfplaym' if interpolation == 'linear' else 'sfplay3m'
            audiogen = fr'''
            ipresetidx sfPresetIndex "{sf2path}", {bank}, {presetnum}
            inote0_ = round(p(idataidx_ + 1))
            ivel_ = p(idataidx_ + 2) * 127
            aout1 {opcode} ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch), ipresetidx, 1
            '''
    return textwrap.dedent(audiogen)


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
            activeConfig()['play.generalMidiSoundfont'] or
            csoundengine.tools.defaultSoundfontPath() or
            None)


def getSoundfontProgram(sf2path: str, presetname: str) -> Tuple[int, int]:
    idx = csoundengine.csoundlib.soundfontIndex(sf2path)
    if presetname not in idx.nameToIndex:
        raise KeyError("fPresetname {presetname} not defined in soundfont {sf2path}"
                       f" Possible presets: {idx.nameToPreset.keys()}")
    return idx.nameToPreset[presetname]


def soundfontSelectProgram(sf2path: str) -> Opt[Tuple[str, int, int]]:
    """
    Select a soundfont program using a gui

    Args:
        sf2path: the path of the soundfont

    Returns:
        a tuple (programname, bank, presetnumber)

    """
    idx = csoundengine.csoundlib.soundfontIndex(sf2path)
    programnames = list(idx.nameToPreset.keys())
    programname = emlib.dialogs.selectFromList(programnames, title="Select Program")
    if programname is None:
        return None
    bank, presetnum = idx.nameToPreset[programname]
    return programname, bank, presetnum


