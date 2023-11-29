from __future__ import annotations
import re
import os
import glob
import textwrap
import csoundengine
import emlib.textlib
from .workspace import getConfig, getWorkspace
from ._common import logger
from . import presetdef


_removeExtranousCharacters = emlib.textlib.makeReplacer({"[":"", "]":"", '"':'', "'":"", "{":"", "}":""})


def _parseIncludeStr(s: str) -> str:
    """ Remove extraneous characters, split and remove quotes """
    includes = _removeExtranousCharacters(s).split(',')
    return includes


def _parseTabledef(s):
    s = _removeExtranousCharacters(s)
    # rawpairs = re.split(r",|\n", s)
    rawpairs = re.split(r"[,\n]", s)
    out = {}
    for rawpair in rawpairs:
        if ":" not in rawpair:
            continue
        key, value = rawpair.split(":")
        key = key.strip()
        out[key] = float(value)
    return out


_UNSET = object()


def saveYamlPreset(p: presetdef.PresetDef, outpath: str) -> None:
    """
    Serialize a presetdef.PresetDef to disk as yaml
    """
    with open(outpath, "w") as f:
        f.write(f"name: {p.name}\n")
        if p.description:
            f.write(f"description: {p.description}\n")
        f.write(f"code: |\n")
        audiogen = textwrap.indent(p.code, "    ")
        f.write(audiogen)
        if not audiogen.endswith("\n"):
            f.write("\n")
        if p.args:
            f.write(f"args: {p.args}\n")
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
        if p.includes:
            f.write(f"includes: {p.includes}\n")
        if p.properties:
            f.write("properties:\n")
            for k, v in p.properties.items():
                f.write(f"    {k}: {v}\n")


def loadYamlPreset(path: str) -> presetdef.PresetDef:
    """
    Load a presetdef.PresetDef from a yaml file
    """
    import yaml
    d = yaml.safe_load(open(path))
    presetName = d.get('name')
    if not presetName:
        raise ValueError("A preset should have a name")
    code = d.get('code')
    if not code:
        raise ValueError("A preset should define an audiogen")
    return presetdef.PresetDef(name=d.get('name'),
                               code=code,
                               includes=d.get('includes'),
                               init=d.get('init'),
                               epilogue=d.get('epilogue'),
                               args=d.get('args'),
                               properties=d.get('properties'))


def makeSoundfontAudiogen(sf2path: str = None,
                          instrnum: int = None,
                          preset: tuple[int, int] = None,
                          interpolation='linear',
                          ampDivisor: int = None,
                          mono=False) -> str:
    """
    Generate audiogen code for a soundfont.

    This can be used as the audiogen parameter to defPreset

    Args:
        sf2path: path to a sf2 soundfont. If None, the default fluidsynth soundfont
            is used
        instrnum: as returned via `csoundengine.csoundlib.soundfontInstruments`
        preset: a tuple (bank, presetnumber) as returned via
            `csoundengine.csoundlib.soundfontPresets`
        interpolation: refers to the wave interpolation performed on the sample
            data (options: 'linear' or 'cubic')

    Returns:
        the audio code for a soundfont preset

    .. note::
        Either an instrument number of a preset tuple must be given.

    Examples
    --------

        >>> # Add a soundfont preset with transposition
        >>> from maelzel.core import *
        >>> code = r'''
        ...     kpitch = kpitch + ktransp
        ... '''
        >>> code += makeSoundfontAudiogen("/path/to/soundfont.sf2", instrnum=0)
        >>> defPreset('myinstr', code, args={'ktransp': 0})

    """
    sf2path = resolveSoundfontPath(sf2path)
    ampdiv = ampDivisor or getConfig()['play.soundfontAmpDiv']
    assert bool(instrnum) != bool(preset), "Either instrnum or preset should be given"
    if not sf2path:
        raise ValueError("No soundfont was given and no default soundfont found")
    if not os.path.exists(sf2path):
        raise OSError(f"Soundfont file not found: '{sf2path}'")
    if instrnum is not None:
        if not mono:
            opcode = 'sfinstr' if interpolation == 'linear' else 'sfinstr3'
            audiogen = fr'''
            iSfTable sfloadonce "{sf2path}"
            inote0_ = round(p(idataidx_ + 1))
            ivel_ = p(idataidx_ + 2) * 127
            kpitch2 = lag:k(kpitch+ktransp, ipitchlag)
            aout1, aout2  {opcode} ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch2), {instrnum}, iSfTable, 1
            '''
        else:
            opcode = 'sfinstrm' if interpolation == 'linear' else 'sfinstr3m'
            audiogen = fr'''
            iSfTable sfloadonce "{sf2path}"
            inote0_ = round(p(idataidx_ + 1))
            ivel_ = p(idataidx_ + 2) * 127
            kpitch2 = lag:k(kpitch+ktransp, ipitchlag)
            aout1 {opcode} ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch2), {instrnum}, iSfTable, 1
            '''

    else:
        presets = csoundengine.csoundlib.soundfontPresets(sf2path)
        if preset is None:
            if not presets:
                raise ValueError(f"The given soundfont has not presets")
            bank, presetnum, presetname = presets[0]
            logger.debug(f"No preset was given. Using first preset found: '{presetname}', "
                         f"bank: {bank}, preset number: {presetnum}")
        else:
            bank, presetnum = preset
            for availablepreset in presets:
                if bank == availablepreset[0] and presetnum == availablepreset[1]:
                    break
            else:
                raise ValueError(f"Preset ({preset}) not found. Available presets: {presets}")

        if not mono:
            opcode = 'sfplay' if interpolation == 'linear' else 'sfplay3'
            audiogen = fr'''
            ipresetidx sfPresetIndex "{sf2path}", {bank}, {presetnum}
            inote0_ = round(p(idataidx_ + 1))
            ivel_ = p(idataidx_ + 2) * 127
            kpitch2 = lag:k(kpitch+ktransp, ipitchlag)
            aout1, aout2 {opcode} ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch2), ipresetidx, 1
            '''
        else:
            opcode = 'sfplaym' if interpolation == 'linear' else 'sfplay3m'
            audiogen = fr'''
            ipresetidx sfPresetIndex "{sf2path}", {bank}, {presetnum}
            inote0_ = round(p(idataidx_ + 1))
            ivel_ = p(idataidx_ + 2) * 127
            kpitch2 = lag:k(kpitch+ktransp, ipitchlag)
            aout1 {opcode} ivel_, inote0_, kamp/{ampdiv}, mtof:k(kpitch2), ipresetidx, 1
            '''
    return textwrap.dedent(audiogen)


def loadPreset(presetPath: str) -> presetdef.PresetDef:
    """
    load a specific preset.

    Args:
        presetPath: the absolute path to a preset

    Returns:
        a presetdef.PresetDef

    Raises `ValueError` if the preset cannot be loaded
    """
    ext = os.path.splitext(presetPath)[1]
    if ext == '.yaml' or ext == '.yml':
        return loadYamlPreset(presetPath)
    else:
        raise ValueError("Only .yaml presets are supported")


def loadPresets(skipErrors=True) -> list[presetdef.PresetDef]:
    """
    loads all presets from the presets path

    To get the current presets' path: ``getWorkspace().presetsPath()``
    """
    basepath = getWorkspace().presetsPath()
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


def resolveSoundfontPath(path: str = None) -> str | None:
    return (path or
            getConfig()['play.generalMidiSoundfont'] or
            csoundengine.tools.defaultSoundfontPath() or
            None)


def getSoundfontProgram(sf2path: str, presetname: str) -> tuple[int, int]:
    idx = csoundengine.csoundlib.soundfontIndex(sf2path)
    if presetname not in idx.nameToIndex:
        raise KeyError("fPresetname {presetname} not defined in soundfont {sf2path}"
                       f" Possible presets: {idx.nameToPreset.keys()}")
    return idx.nameToPreset[presetname]


def soundfontSelectProgram(sf2path: str) -> tuple[str, int, int] | None:
    """
    Select a soundfont program using a gui

    Args:
        sf2path: the path of the soundfont

    Returns:
        a tuple (programname, bank, presetnumber)

    """
    import emlib.dialogs
    idx = csoundengine.csoundlib.soundfontIndex(sf2path)
    programnames = list(idx.nameToPreset.keys())
    programname = emlib.dialogs.selectItem(programnames, title="Select Program")
    if programname is None:
        return None
    bank, presetnum = idx.nameToPreset[programname]
    return programname, bank, presetnum


def findSoundfontInPresetdef(presetdef: presetdef.PresetDef) -> str | None:
    """
    Searched the presetdef for the used soundfont

    Args:
        presetdef: the presetdef.PresetDef

    Returns:
        the path of the used soundfont, if this preset is soundfont based
    """
    assert presetdef.isSoundFont()
    if presetdef.init:
        path = re.search(r'sfloadonce \"(.*)\"', presetdef.init)
        if path:
            return path.group(1)
    return None


def embedEnvelope(audiogen: str, audiovars: list[str], envelope="aenv_"
                  ) -> str:
    """
    Given an audiogen, multiply its audiovars with envelope var
    We assume that there might be code after  the audiogen which is used
    for panning, output, etc. and we want to apply the envelope before.
    For that reason we need to find the line where the audiovars get
    their last value and multiply them by the envelope there

    Args:
        audiogen: audio generating code
        audiovars: list of audio variables
        envelope: the variable holding the event envelope

    Returns:
        the audiogen with the embedded envelope
    """
    lines = audiogen.splitlines()
    for audiovar in audiovars:
        lastassign = csoundengine.csoundlib.lastAssignmentToVariable(audiovar, lines)
        if lastassign is None:
            logger.error(f"Did not find any assignment to variable {audiovar}")
            logger.error("Audiogen:\n")
            logger.error(audiogen)
        else:
            envline = f'{audiovar} *= {envelope}'
            envline = emlib.textlib.matchIndentation(envline, lines[lastassign])
            lines = lines[:lastassign+1] + [envline] + lines[lastassign+1:]
    return '\n'.join(lines)
