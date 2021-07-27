from __future__ import annotations
import os
import emlib.textlib
import emlib.misc
import fnmatch
import glob
import csoundengine.csoundlib
from .presetbase import *
from .workspace import presetsPath, getConfig, currentWorkspace
from . import presetutils
from . import playpresets
from ._common import logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from .csoundevent import CsoundEvent

import watchdog.events
from watchdog.observers import Observer as _WatchdogObserver


__all__ = ('csoundPrelude',
           'PresetManager',
           'getPresetManager',
           'showPresets',
           'defPreset',
           'defPresetSoundfont',
           'availablePresets')


csoundPrelude = \
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


class PresetManager:

    def __init__(self):
        self.presetdefs: Dict[str, PresetDef] = {}
        self.presetsPath = presetsPath()
        self._prepareEnvironment()
        self._makeBuiltinPresets()
        self.loadPresets()
        self._watchdog = self._startWatchdog()

    def __del__(self):
        self._watchdog.join()

    def loadPresets(self) -> None:
        presetdefs = presetutils.loadPresets()
        for presetdef in presetdefs:
            self._registerPreset(presetdef.name, presetdef)

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
        for presetdef in playpresets.builtinPresets:
            self._registerPreset(presetdef.name, presetdef)

        sf2 = presetutils.resolveSoundfontPath(path=sf2path)
        if not sf2:
            logger.info("No soundfont defined, builtin instruments using soundfonts will"
                        "not be available. Set config['play.generalMidiSoundfont'] to"
                        "the path of an existing soundfont")

        progs = csoundengine.csoundlib.soundfontGetInstruments(sf2)
        instrnums = {num for num, name in progs}
        return
        for name, instrnum in playpresets.soundfontGeneralMidiInstruments.items():
            if instrnum in instrnums:
                self.defPresetSoundfont(name, sf2path=sf2, instr=instrnum, _builtin=True)

    def defPreset(self,
                  name: str,
                  audiogen: str,
                  init:str=None,
                  includes: List[str] = None,
                  params: Dict[str, float] = None,
                  descr: str = None,
                  priority: int = None,
                  temporary = False,
                  _builtin=False,
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
            _builtin: internal parameter, used to identify builtin presets
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
        presetdef = presetutils.makePreset(name=name,
                                           audiogen=audiogen,
                                           includes=includes,
                                           params=params,
                                           descr=descr,
                                           priority=priority,
                                           builtin=_builtin,
                                           temporary=temporary,
                                           init=init)
        self._registerPreset(name, presetdef)
        return presetdef

    def defPresetSoundfont(self,
                           name:str,
                           sf2path:str=None,
                           preset: Union[int, Tuple[int, int], str]=0,
                           preload=True,
                           init: str = None,
                           includes: List[str] = None,
                           postproc:str=None,
                           params: Union[List[float], Dict[str, float]] = None,
                           priority: int = None,
                           interpolation: str = None,
                           _builtin=False) -> PresetDef:
        """
        Define a new soundfont instrument preset

        Args:
            name: the name of the preset
            sf2path: the path to the soundfont, or None to use the default
                fluidsynth soundfont
            preset: the preset to use. Either a tuple (bank, presetnum) or the name
                of the preset.
            preload: if True, load the soundfont at the beginning of the session
            postproc: any code needed for postprocessing. Any postprocessing should
                modify the variables a0, a1
            init: global code needed by postproc
            includes: files to include (if needed by init or postproc)
            params: mutable values needed by postproc (if any). See defPreset
            priority: default priority for this preset
            interpolation: one of 'linear', 'cubic'. Refers to the interpolation used
                when reading the sample waveform. If None, use the default defined
                in the config (key 'play.soundfontInterpolation')

            _builtin: internal parameter, used to identify builtin presets
                (builtin presets are not saved)

        !!! note

            To list all programs in a soundfont, see
            :func:`~maelzel.play.showSoundfontPresets`

        """
        cfg = getConfig()
        if interpolation is None:
            interpolation = cfg['play.soundfontInterpolation']
        assert interpolation in ('linear', 'cubic')

        if isinstance(preset, str):
            raise TypeError("named presets are not supported yet")
        bank, presetnum = preset
        audiogen = presetutils.makeSoundfontAudiogen(sf2path=sf2path,
                                                     preset=(bank, presetnum),
                                                     interpolation=interpolation)
        if preload:
            # We don't need to use a global variable because sfloadonce
            # saved the table num into a channel
            init0 = f'''iSfTable sfloadonce "{sf2path}"'''
            if init:
                init = "\n".join((init0, init))
            else:
                init = init0
        if postproc:
            audiogen = emlib.textlib.joinPreservingIndentation((audiogen, postproc))
        return self.defPreset(name=name, audiogen=audiogen, init=init,
                              includes=includes, params=params,
                              _builtin=_builtin,
                              priority=priority)

    def _registerPreset(self, name:str, presetdef:PresetDef, temporary=False) -> None:
        self.presetdefs[name] = presetdef
        config = getConfig()
        if presetdef.userDefined and (not temporary or config['play.autosavePresets']):
            self.savePreset(name)

    def getPreset(self, name:str) -> PresetDef:
        if name is None:
            name = getConfig()['play.instr']
        preset = self.presetdefs.get(name)
        if preset:
            return preset
        raise ValueError(f"Preset {name} not known. \n"
                         f"Presets: {self.presetdefs.keys()}")


    def definedPresets(self) -> Set[str]:
        return set(self.presetdefs.keys())

    def showPresets(self, pattern="*", showGeneratedCode=False) -> None:
        selectedPresets = [presetName for presetName in self.presetdefs.keys()
                           if fnmatch.fnmatch(presetName, pattern)]
        if not emlib.misc.inside_jupyter():
            for presetName in selectedPresets:
                presetdef = self.presetdefs[presetName]
                print("")
                if not showGeneratedCode:
                    print(presetdef)
                else:
                    print(presetdef.makeInstr().body)
        else:
            theme = getConfig()['html.theme']
            htmls = []
            for presetName in selectedPresets:
                presetdef = self.presetdefs[presetName]
                html = presetdef._repr_html_(theme, showGeneratedCode=showGeneratedCode)
                htmls.append(html)
                htmls.append("<hr>")
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
        for name, instrdef in self.presetdefs.items():
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
        fmt = "yaml"
        presetdef = self.getPreset(name)
        if not presetdef.userDefined:
            raise ValueError(f"Can't save a builtin preset: {name}")
        path = presetsPath()
        outpath = os.path.join(path, f"{name}.{fmt}")
        if fmt == 'yaml' or fmt == 'yml':
            presetutils.saveYamlPreset(presetdef, outpath)
        else:
            raise KeyError(f"format {fmt} not supported")
        return outpath

    def makeRenderer(self,
                     sr:int=None,
                     nchnls:int=None,
                     ksmps=None
                     ) -> csoundengine.Renderer:
        """
        Make an offline Renderer from instruments defined here

        Args:
            sr: the samplerate of the renderer
            nchnls: the number of channels
            ksmps: if not explicitely set, will use config 'rec.ksmps'

        Returns:
            a csoundengine.Renderer
        """
        config = getConfig()
        sr = sr or config['rec.samplerate']
        ksmps = ksmps or config['rec.ksmps']
        nchnls = nchnls or config['rec.nchnls']
        state = currentWorkspace()
        renderer = csoundengine.Renderer(sr=sr, nchnls=nchnls, ksmps=ksmps,
                                         a4=state.a4)
        renderer.addGlobalCode(csoundPrelude)
        # Define all instruments
        for presetdef in self.presetdefs.values():
            renderer.defInstr(presetdef.name, presetdef.body, tabledef=presetdef.params)
            globalCode = presetdef.globalCode()
            if globalCode:
                renderer.addGlobalCode(globalCode)
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


_presetManager = PresetManager()
defPreset = _presetManager.defPreset
defPresetSoundfont = _presetManager.defPresetSoundfont
showPresets = _presetManager.showPresets


def availablePresets() -> Set[str]:
    """
    Returns the names of instr presets already defined

    """
    return getPresetManager().definedPresets()


def getPresetManager() -> PresetManager:
    """
    Return the active PresetManager
    """
    return _presetManager


def getPreset(preset:str) -> Optional[PresetDef]:
    """
    get a defined preset
    """
    return getPresetManager().getPreset(preset)

