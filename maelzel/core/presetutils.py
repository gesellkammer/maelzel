from __future__ import annotations
import re
import os
import sys
import glob
import textwrap
import emlib.textlib
from .workspace import Workspace
from ._common import logger

import typing as _t
if _t.TYPE_CHECKING:
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


def saveYamlPreset(p: presetdef.PresetDef, outpath: str) -> None:
    """
    Serialize a presetdef.PresetDef to disk as yaml
    """
    with open(outpath, "w") as f:
        f.write(f"name: {p.name}\n")
        if p.description:
            f.write(f"description: {p.description}\n")
        f.write("code: |\n")
        audiogen = textwrap.indent(p.code, "    ")
        f.write(audiogen)
        if not audiogen.endswith("\n"):
            f.write("\n")
        if p.args:
            f.write(f"args: {p.args}\n")
        if p.init:
            f.write("init: |\n")
            f.write(textwrap.indent(p.init, "    "))
            if not p.init.endswith("\n"):
                f.write("\n")
        if p.epilogue:
            f.write("epilogue: |\n")
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
    from . import presetdef
    return presetdef.PresetDef(name=d.get('name'),
                               code=code,
                               includes=d.get('includes'),
                               init=d.get('init'),
                               epilogue=d.get('epilogue'),
                               args=d.get('args'),
                               properties=d.get('properties'))


def makeSoundfontAudiogen(sf2path: str,
                          preset: tuple[int, int] | None = None,
                          interpolation='linear',
                          ampDivisor: int | float = 0,
                          normalize=False,
                          velocityCurve: presetdef.GainToVelocityCurve | _t.Sequence[float] = (),
                          # velocityToCutoffMapping: dict[int, int] = None,
                          referencePeakPitch: int = 0,
                          mono=False,
                          reverb=False,
                          reverbChanPrefix='.maelzelreverb'
                          ) -> str:
    """
    Generate audiogen code for a soundfont.

    This can be used as the audiogen parameter to defPreset
    Notice than using instruments directly (also known as layers) bypasses
    the soundfont's own layer triggering (which is preserved when using presets)
    and is only recommended for this very specific objective.

    Args:
        sf2path: path to a sf2 soundfont.
        preset: a tuple (bank, presetnumber) as returned via
            `csoundengine.csoundlib.soundfontPresets`
        interpolation: refers to the wave interpolation performed on the sample
            data (options: 'linear' or 'cubic')
        ampDivisor: a divisor to scale amplitudes down. The soundfont spec says
            that samples should be signed 16 bit, which means that values need
            to be scaled down if using 0dbfs=0, which we use throughout in maelzel.
        normalize: if True, a reference peak of the soundfont is queried and this
            is used as the amplitude divisor
        velocityCurve: a curve mapping ... (#TODO)
        reverb: if True, include code to send audio to a global reverb. At this
            point many aspects of this are hardcoded

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
    ampdiv = ampDivisor or Workspace.active.config['play.soundfontAmpDiv']
    if not os.path.exists(sf2path):
        raise OSError(f"Soundfont file not found: '{sf2path}'")

    import csoundengine.sftools
    presets = csoundengine.sftools.soundfontPresets(sf2path)
    if not presets:
        raise ValueError(f"The given soundfont {sf2path} has no presets")

    if preset is None:
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
    parts = [fr'''
        ipresetidx sfpresetindex "{sf2path}", {bank}, {presetnum}
        iamp0_ = p(idataidx_ + 2)
        inote0_ = round(p(idataidx_ + 1))
        kpitch2 = lag:k(kpitch + ktransp, ipitchlag)
        iampdiv_ = {ampdiv}
        ''']

    if normalize:
        keyrange = csoundengine.sftools.soundfontKeyrange(sf2path, preset=(bank, presetnum))
        if keyrange is None:
            raise RuntimeError(f"No key range found for preset {(bank, presetnum)} "
                               f"for soundfont {sf2path}")
        if referencePeakPitch:
            refpitch1 = referencePeakPitch
            refpitch2 = referencePeakPitch + 12
        else:
            minpitch, maxpitch = keyrange
            refpitch1 = int((maxpitch - minpitch) * 0.2 + minpitch)
            refpitch2 = int((maxpitch - minpitch) * 0.8 + minpitch)

        parts.append(fr'''
        isfpeak_ = dict_get:i(gi__soundfont_peaks, ipresetidx, 0)
        if isfpeak_ > 0 then
            iampdiv_ = isfpeak_
        else
            schedule "_sfpeak", 0, 0.1, ipresetidx, {refpitch1}, {refpitch2}
        endif
        ''')

    if not velocityCurve:
        ivelstr = 'ivel _linexp dbamp(iamp0_), 2.6, -72, 0, 1, 127'
    elif isinstance(velocityCurve, (list, tuple)):
        assert isinstance(velocityCurve, list)
        # breakpoints mapping db to velocity
        valuestr = ', '.join(map(str, velocityCurve))
        ivelstr = f'ivel = bpf(dbamp(iamp0_), {valuestr})'
    else:
        assert isinstance(velocityCurve, presetdef.GainToVelocityCurve)
        ivelstr = f'ivel _linexp dbamp(iamp0_), {velocityCurve.exponent}, {velocityCurve.mindb}, {velocityCurve.maxdb}, {velocityCurve.minvel}, {velocityCurve.maxvel}'

    # ivel is a parameter of the preset, will be -1 by default,
    # indicating that we should derive velocity from amplitude
    # But the user is still able to set a specific velocity
    # for an event, so we only do the calculation if the user
    # has not set an explicit value
    parts.append(f"""
    if ivel < 0 then
        {ivelstr}
    endif
    """)
    if not mono:
        opcode = 'sfplay' if interpolation == 'linear' else 'sfplay3'
        parts.append(f'''\
        aout1, aout2 {opcode} ivel, inote0_, kamp/iampdiv_, mtof:k(kpitch2), ipresetidx, 1
        ''')

    else:
        opcode = 'sfplaym' if interpolation == 'linear' else 'sfplay3m'
        parts.append(fr'''
        aout1 {opcode} ivel, inote0_, kamp/iampdiv_, mtof:k(kpitch2), ipresetidx, 1
        ''')

    if reverb:
        # TODO: add reverb code
        if mono:
            parts.append("kpos = kpos == -1 ? 0 : kpos")
            parts.append("a_outL, a_outR = pan2(aout1, kpos)")
        else:
            parts.append("kpos = kpos == -1 ? 0.5 : kpos")
            parts.append("a_outL, a_outR = panstereo(aout1, aout2, kpos)")
        parts.append(f'''\
        chnmix a_outL * kwet, "{reverbChanPrefix}.1"
        chnmix a_outR * kwet, "{reverbChanPrefix}.2"
        outch ichan, a_outL * (1 - kwet), ichan + 1, a_outR * (1 - kwet)
        ''')
    parts = [emlib.textlib.stripLines(part) for part in parts]
    audiogen = emlib.textlib.joinPreservingIndentation(parts)
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
    basepath = Workspace.active.presetsPath()
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
            logger.debug(f"Loading preset from '{path}'")
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


def defaultSoundfontPath() -> str:
    """
    Returns the path of the fluid sf2 file

    Returns:
        the path of the default soundfont or an empty path if this does not apply
    """
    if sys.platform == 'linux':
        paths = ["/usr/share/sounds/sf2/FluidR3_GM.sf2"]
        path = next((path for path in paths if os.path.exists(path)), '')
    else:
        logger.info("Default path for soundfonts only defined in linux")
        path = ''
    return path


def resolveSoundfontPath(path='') -> str:
    return (path or
            Workspace.active.config['play.generalMidiSoundfont'] or
            defaultSoundfontPath() or
            '')


def getSoundfontProgram(sf2path: str, presetname: str) -> tuple[int, int]:
    import csoundengine.sftools
    idx = csoundengine.sftools.soundfontIndex(sf2path)
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
    import csoundengine.sftools
    idx = csoundengine.sftools.soundfontIndex(sf2path)
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
    import csoundengine.csoundparse
    for audiovar in audiovars:
        lastassign = csoundengine.csoundparse.lastAssignmentToVariable(audiovar, lines)
        if lastassign is None:
            logger.error(f"Did not find any assignment to variable {audiovar}")
            logger.error("Audiogen:\n")
            logger.error(audiogen)
        else:
            envline = f'{audiovar} *= {envelope}'
            envline = emlib.textlib.matchIndentation(envline, lines[lastassign])
            lines = lines[:lastassign+1] + [envline] + lines[lastassign+1:]
    return '\n'.join(lines)
