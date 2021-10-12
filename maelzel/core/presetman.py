from __future__ import annotations
import os
import emlib.textlib
import emlib.misc
import fnmatch
import glob
import csoundengine.csoundlib
import emlib.dialogs
from .presetbase import *
from .workspace import presetsPath, activeConfig, activeWorkspace
from . import presetutils
from . import playpresets
from ._common import logger
from . import tools
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from .csoundevent import CsoundEvent

import watchdog.events
from watchdog.observers import Observer as _WatchdogObserver


__all__ = ('csoundPrelude',
           'presetManager',
           'showPresets',
           'defPreset',
           'defPresetSoundfont',
           'PresetManager',
           )


csoundPrelude = r"""
opcode turnoffWhenSilent, 0, a
    asig xin
    ksilent_  trigger detectsilence:k(asig, 0.0001, 0.05), 0.5, 0
    if ksilent_ == 1  then
      turnoff
    endif    
endop
"""


class _WatchdogPresetsHandler(watchdog.events.FileSystemEventHandler):
    def __init__(self, manager: PresetManager):
        self.manager = manager

    def on_modified(self, event):
        self.manager.loadSavedPresets()

    def on_created(self, event):
        self.manager.loadSavedPresets()


class PresetManager:
    _numinstances = 0

    def __init__(self, watchPresets=False):
        if self._numinstances > 0:
            raise RuntimeError("Only one PresetManager should be active")
        self._numinstances = 1
        self.presetdefs: Dict[str, PresetDef] = {}
        self.presetsPath = presetsPath()
        self._prepareEnvironment()
        self._makeBuiltinPresets()
        self.loadSavedPresets()
        self._watchdog = self._startWatchdog() if watchPresets else None

    def __del__(self):
        if self._watchdog:
            self._watchdog.join()

    def loadSavedPresets(self) -> None:
        """
        Loads user-defined presets
        """
        presetdefs = presetutils.loadPresets()
        for presetdef in presetdefs:
            self.registerPreset(presetdef)

    def _startWatchdog(self):
        observer = _WatchdogObserver()
        observer.schedule(_WatchdogPresetsHandler(self), self.presetsPath)
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
            self.registerPreset(presetdef)

        sf2 = presetutils.resolveSoundfontPath(path=sf2path)
        if not sf2:
            logger.info("No soundfont defined, builtin instruments using soundfonts will"
                        "not be available. Set config['play.generalMidiSoundfont'] to"
                        "the path of an existing soundfont")

        for name, preset in playpresets.soundfontGeneralMidiPresets.items():
            self.defPresetSoundfont(name, sf2path=sf2, preset=preset, _builtin=True)

    def defPreset(self,
                  name: str,
                  audiogen: str,
                  init='',
                  epilogue='',
                  includes: List[str] = None,
                  params: Dict[str, float] = None,
                  description: str = None,
                  priority: int = None,
                  temporary = False,
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
        If p4 is 0, then the given preset has no associated ftable

        audiogen should generate an audio output signal named 'aout1' for channel 1,
        'aout2' for channel 2, etc.

        Example::

            audiogen = 'aout1 oscili a(kamp), kfreq'

        Args:
            name: the name of the preset
            audiogen: audio generating csound code
            epilogue: code to include after any other code. Needed when using turnoff,
                since calling turnoff in the middle of an instrument might cause a crash.
            init: global code needed by the audiogen part (usually a table definition)
            includes: files to include
            params: a dict {parameter_name: value}
            description: a description of what this preset is/does
            priority: if given, the instr has this priority as default when scheduled
            temporary: if True, preset will not be saved, even if
                `config['play.autosavePreset']` is True

        Retursn:
            a PresetDef

        Example
        ~~~~~~~

        .. code-block:: python

            # create a preset with a dynamic parameter
            >>> audiogen = r'''
            ... aout1 vco2 kamp, kfreq, 10
            ... aout1 moogladder aout1, lag:k(kcutoff, 0.1), kq
            ... '''
            >>> presetManager.defPreset(name='mypreset', audiogen=audiogen,
            ...                         params=dict(kcutoff=4000, kq=1))

            # Alternatively the parameters can be declared inline:
            >>> presetManager.defPreset('mypreset', r'''
            ... |kcutoff=4000, kq=1|
            ... aout1 vco2 kamp, kfreq, 10
            ... aout1 moogladder aout1, lag:k(kcutoff, 0.1), kq
            ... ''')

        See Also
        ~~~~~~~~

        :meth:`PresetManager.defPresetSoundfont`
        """
        audiogen = emlib.textlib.stripLines(audiogen)
        firstLine = audiogen.splitlines()[0].strip()
        if firstLine[0] == '|' and firstLine[-1] == '|':
            delimiter, inlineParams, audiogen = csoundengine.instr.parseInlineArgs(audiogen)
            if params:
                params.update(inlineParams)
            else:
                params = inlineParams
        presetdef = PresetDef(name=name,
                              audiogen=audiogen,
                              init=init,
                              epilogue=epilogue,
                              includes=includes,
                              params=params,
                              description=description,
                              builtin=False,
                              priority=priority,
                              temporary=temporary)
        self.registerPreset(presetdef)
        return presetdef

    def defPresetSoundfont(self,
                           name:str=None,
                           sf2path:str=None,
                           preset: Union[Tuple[int, int], str]=(0, 0),
                           init: str = None,
                           postproc: str = None,
                           includes: List[str] = None,
                           params: Union[List[float], Dict[str, float]] = None,
                           priority: int = None,
                           interpolation: str = None,
                           temporary = False,
                           mono=False,
                           turnoffWhenSilent=True,
                           _builtin=False) -> PresetDef:
        """
        Define a new soundfont instrument preset

        Args:
            name: the name of the preset. If not given, the name of the preset
                is used
            sf2path: the path to the soundfont, None to use the default soundfont (if
                present in the system) or "?" to open a dialog
            preset: the preset to use. Either a tuple (bank: int, presetnum: int) or the
                name of the preset as string. **Use "?" to select from all available presets
                in the soundfont**
            init: global code needed by postproc
            postproc: code to modify the generated audio before it is sent to the
                outputs
            includes: files to include (if needed by init or postproc)
            params: mutable values needed by postproc (if any). See defPreset
            priority: default priority for this preset
            temporary: if True, preset will not be saved, even if
                `config['play.autosavePreset']` is True
            mono: if True, only the left channel of the soundfont is read
            interpolation: one of 'linear', 'cubic'. Refers to the interpolation used
                when reading the sample waveform. If None, use the default defined
                in the config (key 'play.soundfontInterpolation')
            turnoffWhenSilent: if True, turn a note off when the sample stops (by detecting
                silence for a given amount of time)

            _builtin: internal parameter, used to identify builtin presets
                (builtin presets are not saved)

        !!! note

            To list all programs in a soundfont, see
            :func:`~maelzel.play.showSoundfontPresets`

        """
        if name in self.presetdefs:
            logger.info(f"PresetDef {name} already exists, overwriting")
        if sf2path is None:
            sf2path = presetutils.resolveSoundfontPath()
            if sf2path is None:
                sf2path = "?"
        if sf2path == "?":
            sf2path = tools.selectFileForOpen('soundfontLastDirectory',
                                              filter="*.sf2", prompt="Select Soundfont",
                                              ifcancel="No soundfont selected, aborting")
        cfg = activeConfig()
        if interpolation is None:
            interpolation = cfg['play.soundfontInterpolation']
        assert interpolation in ('linear', 'cubic')

        if isinstance(preset, str):
            if preset == "?":
                result = presetutils.soundfontSelectProgram(sf2path)
                if not result:
                    raise ValueError("No preset selected, aborting")
                progname, bank, presetnum = result
            else:
                bank, presetnum = presetutils.getSoundfontProgram(sf2path, preset)
        else:
            bank, presetnum = preset
        idx = csoundengine.csoundlib.soundfontIndex(sf2path)
        if name is None:
            name = idx.presetToName[(bank, presetnum)]
        if (bank, presetnum) not in idx.presetToName:
            raise ValueError(f"Preset ({bank}:{presetnum}) not found. Possible presets: "
                             f"{idx.presetToName.keys()}")
        audiogen = presetutils.makeSoundfontAudiogen(sf2path=sf2path,
                                                     preset=(bank, presetnum),
                                                     interpolation=interpolation,
                                                     mono=mono)
        # We don't need to use a global variable because sfloadonce
        # saved the table num into a channel
        init0 = f'''iSfTable_ sfloadonce "{sf2path}"'''
        if init:
            init = "\n".join((init0, init))
        else:
            init = init0
        if postproc:
            audiogen = emlib.textlib.joinPreservingIndentation((audiogen, postproc))
        epilogue = "turnoffWhenSilent aout1" if turnoffWhenSilent else ''
        presetdef = PresetDef(name=name,
                              audiogen=audiogen,
                              init=init,
                              epilogue=epilogue,
                              includes=includes,
                              params=params,
                              priority=priority,
                              temporary=temporary,
                              builtin=_builtin)
        self.registerPreset(presetdef)
        return presetdef

    def registerPreset(self, presetdef:PresetDef) -> None:
        """
        Register this PresetDef.

        If the PresetDef is temporary or config['play.autosavePresets'] is False,
        the presetdef will not be saved to disk, otherwise any PresetDef registered
        will be persisted and will be loaded in any new session

        Args:
            presetdef: the PresetDef to register

        """
        self.presetdefs[presetdef.name] = presetdef
        config = activeConfig()
        if presetdef.userDefined and not presetdef.temporary and config['play.autosavePresets']:
            self.savePreset(presetdef.name)

    def getPreset(self, name:str) -> PresetDef:
        """Get a preset by name

        Raises KeyError if no preset with such name is defined

        Args:
            name: the name of the preset to get (use "?" to select from a list
                of defined presets)
        """
        if name is None:
            name = activeConfig()['play.instr']
        elif name == "?":
            name = tools.selectFromList(list(self.presetdefs.keys()),
                                        title="Select Preset",
                                        default=activeConfig()['play.instr'])
        preset = self.presetdefs.get(name)
        if not preset:
            logger.error(f"Preset {name} not known. \n"
                         f"Presets: {self.presetdefs.keys()}")
            raise KeyError(f"Preset {name} not known")
        return preset

    def definedPresets(self) -> List[str]:
        """Returns a list of defined presets"""
        return list(self.presetdefs.keys())

    def showPresets(self, pattern="*", showGeneratedCode=False) -> None:
        """
        Show the selected presets

        Args:
            pattern: a glob pattern to select which presets are shown
            showGeneratedCode: if True, all actual code of an instrument
                is show. Otherwise only the audiogen is shown

        """
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
            theme = activeConfig()['html.theme']
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
        Number of channels needed to play the event

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

    def savePreset(self, preset:Union[str, PresetDef]) -> str:
        """
        Saves the preset in the presets folder, returns the path to the saved file

        Args:
            preset: the name of the preset

        Returns:
            the path of the saved preset
        """
        fmt = "yaml"
        if isinstance(preset, PresetDef):
            presetdef = preset
            if not presetdef.name in self.presetdefs:
                self.registerPreset(presetdef)
            else:
                if presetdef is not self.presetdefs[presetdef.name]:
                    logger.info(f"Updating preset {presetdef.name}")
                    self.registerPreset(presetdef)
        else:
            presetdef = self.getPreset(preset)
        if not presetdef:
            raise ValueError(f"Preset {preset} not found")
        if not presetdef.userDefined:
            raise ValueError(f"Can't save a builtin preset: {preset}")
        path = presetsPath()
        outpath = os.path.join(path, f"{presetdef.name}.{fmt}")
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
            sr: the sr of the renderer
            nchnls: the number of channels
            ksmps: if not explicitely set, will use config 'rec.ksmps'

        Returns:
            a csoundengine.Renderer
        """
        config = activeConfig()
        sr = sr or config['rec.sr']
        ksmps = ksmps or config['rec.ksmps']
        nchnls = nchnls or config['rec.nchnls']
        state = activeWorkspace()
        renderer = csoundengine.Renderer(sr=sr, nchnls=nchnls, ksmps=ksmps,
                                         a4=state.a4)
        renderer.addGlobalCode(csoundPrelude)
        # Define all instruments
        for presetdef in self.presetdefs.values():
            renderer.defInstr(name=presetdef.name, body=presetdef.body, tabledef=presetdef.params)
            globalCode = presetdef.globalCode()
            if globalCode:
                renderer.addGlobalCode(globalCode)
        return renderer

    def openPresetsDir(self) -> None:
        """
        Open a file manager at presetsPath
        """
        path = presetsPath()
        emlib.misc.open_with_app(path)

    def removeUserPreset(self, presetName: str = None) -> bool:
        """
        Remove a user defined preset

        Args:
            presetName: the name of the preset to remove. Use None or "?" to
                select from a list of removable presets

        Returns:
            True if the preset was removed, False if it did not exist
        """
        if presetName is None or presetName == "?":
            saved = self.savedPresets()
            if not saved:
                logger.info("No saved presets, aborting")
                return False
            presetName = emlib.dialogs.selectItem(saved, title="Remove Preset")
            if not presetName:
                return False
        path = os.path.join(presetsPath(), f"{presetName}.yaml")
        if not os.path.exists(path):
            logger.warning(f"Preset {presetName} does not exist (searched: {path})")
            presetnames = self.savedPresets()
            logger.info(f"User defined presets: {presetnames}")
            return False
        os.remove(path)
        return True

    def savedPresets(self) -> List[str]:
        presets = glob.glob(os.path.join(presetsPath(), "*.yaml"))
        return [os.path.splitext(os.path.split(p)[1])[0] for p in presets]

    def selectPreset(self) -> Optional[str]:
        return emlib.dialogs.selectItem(presetManager.definedPresets(),
                                        title="Select Preset")


presetManager = PresetManager()
defPreset = presetManager.defPreset
defPresetSoundfont = presetManager.defPresetSoundfont
showPresets = presetManager.showPresets


