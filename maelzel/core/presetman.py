"""
Implements a :class:`PresetManager`, which is a singleton class in charge
of managing playback presets for a maelzel.core session.


"""
from __future__ import annotations
import os
import emlib.textlib
import emlib.misc
import fnmatch
import glob
import csoundengine.csoundlib
import emlib.dialogs
from .presetbase import *
from .workspace import getConfig, getWorkspace
from . import presetutils
from . import playpresets
from ._common import logger
from . import _dialogs
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from .csoundevent import CsoundEvent

import watchdog.events
from watchdog.observers import Observer as _WatchdogObserver


__all__ = (
    'presetManager',
    'showPresets',
    'defPreset',
    'defPresetSoundfont',
    'definedPresets',
    'PresetManager',
)


_csoundPrelude = r"""
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
    """
    Singleton object, manages all instrument Presets

    Any maelzel.core object can be played with an instrument preset defined
    here. A PresetManager is attached to a running Session as soon as an object
    is scheduled with the given Preset. As such, it acts as a library of Presets
    and any number of such Presets can be created.
    """
    _numinstances = 0

    def __init__(self, watchPresets=False):
        if self._numinstances > 0:
            raise RuntimeError("Only one PresetManager should be active")
        self._numinstances = 1
        self.presetdefs: Dict[str, PresetDef] = {}
        self.presetsPath = getWorkspace().presetsPath()
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
        if not os.path.exists(self.presetsPath):
            os.makedirs(self.presetsPath)

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

        for instr, preset in playpresets.soundfontGeneralMidiPresets.items():
            if sf2 and sf2 != "?":
                presetname = 'gm-' + instr
                descr = f'General MIDI {instr}'
                self.defPresetSoundfont(presetname, sf2path=sf2, preset=preset, _builtin=True,
                                        description=descr)

        for name, (path, preset, descr) in playpresets.builtinSoundfonts().items():
            self.defPresetSoundfont(name, sf2path=path, preset=preset, _builtin=True,
                                    description=descr)

    def defPreset(self,
                  name: str,
                  audiogen: str,
                  init='',
                  epilogue='',
                  includes: list[str] = None,
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

        Similar to csoundengine's `Instr <https://csoundengine.readthedocs.io/en/latest/instr.html>`_ ,
        each preset can have an associated ftable, passed as p4. If p4 is 0, then
        the given preset has no associated ftable

        audiogen should generate an audio output signal named 'aout1' for channel 1,
        'aout2' for channel 2, etc.::

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
                the 'play.autosavePreset' is set to True in the active config

        Returns:
            a PresetDef

        Example
        ~~~~~~~

        Create a preset with dynamic parameters

        .. code-block:: python

            >>> from maelzel.core import *
            >>> audiogen = r'''
            ... aout1 vco2 kamp, kfreq, 10
            ... aout1 moogladder aout1, lag:k(kcutoff, 0.1), kq
            ... '''
            >>> presetManager.defPreset(name='mypreset', audiogen=audiogen,
            ...                         params=dict(kcutoff=4000, kq=1))

        Or simply:

            >>> defPreset('mypreset', r'''
            ... |kcutoff=4000, kq=1|
            ... aout1 vco2 kamp, kfreq, 10
            ... aout1 moogladder aout1, lag:k(kcutoff, 0.1), kq
            ... ''')

        Then, to use the Preset:

            >>> synth = Note("4C", dur=60).play(instr='mypreset', params={'kcutoff': 1000})

        ``.play`` returns a SynthGroup, even if in this case a Note generates only one synth.
        (for example a Chord would generate one synth per note)

        **NB**: Parameters can be modified while the synth is running :

            >>> synth.setp(kcutoff=2000)

        .. admonition:: See Also

            - :func:`defPresetSoundfont`
            - :meth:`maelzel.core.musicobj.MusicObj.play`

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
                           includes: list[str] = None,
                           params: Union[list[float], Dict[str, float]] = None,
                           priority: int = None,
                           interpolation: str = None,
                           temporary = False,
                           mono=False,
                           turnoffWhenSilent=True,
                           description='',
                           _builtin=False) -> PresetDef:
        """
        Define a new instrument preset based on a soundfont

        Once the instr preset is defined the soundfont preset is fixed. To use multiple
        soundfont presets, define one instr preset for each.

        **Use '?' as the preset to select a preset from a dialog**.

        To list all presets in a soundfont, see
        `csoundengine.csoundlib.soundfontGetPresets <https://csoundengine.readthedocs.io/en/latest/api/csoundengine.csoundlib.soundfontGetPresets.html>`_

            >>> import csoundengine
            >>> csoundengine.csoundlib.soundfontGetPresets("~/sf2/Yamaha-C5-Salamander.sf2")
            [(0, 0, 'Yamaha C5 Grand'),
             (0, 1, 'Dynamic Yamaha C5'),
             (0, 2, 'Dark Grand'),
             (0, 3, 'Mellow Grand'),
             (0, 4, 'Bright Grand'),
             (0, 5, 'Very Bright Grand')]


        Args:
            name: the name of the preset. If not given, the name of the preset
                is used
            sf2path: the path to the soundfont; Use "?" open a dialog to select a .sf2 file
                or None to use the default soundfont
            preset: the preset to use. Either a tuple (bank: int, presetnum: int) or the
                name of the preset as string. **Use "?" to select from all available presets
                in the soundfont**.
            init: global code needed by postproc
            postproc: code to modify the generated audio before it is sent to the
                outputs
            includes: files to include (if needed by init or postproc)
            params: mutable values needed by postproc (if any). See :meth:`~PresetManager.defPreset`
            priority: default priority for this preset
            temporary: if True, preset will not be saved, even if
                :ref:`config['play.autosavePreset'] <config_play_autosavepresets>` is True
            mono: if True, only the left channel of the soundfont is read
            interpolation: one of 'linear', 'cubic'. Refers to the interpolation used
                when reading the sample waveform. If None, use the default defined
                in the config (:ref:`key 'play.soundfontInterpolation' <config_play_soundfontinterpolation>`)
            turnoffWhenSilent: if True, turn a note off when the sample stops (by detecting
                silence for a given amount of time)

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> defPresetSoundfont('yamahagrand', '/path/to/yamahapiano.sf2',
            ...                    preset='Yamaha C5 Grand')
            >>> Note("C4", dur=5).play(instr='yamahagrand')


        See Also
        ~~~~~~~~

        :func:`defPreset`
        """
        if name in self.presetdefs:
            logger.info(f"PresetDef {name} already exists, overwriting")
        if sf2path is None:
            sf2path = presetutils.resolveSoundfontPath()
            if sf2path is None:
                sf2path = "?"
        if sf2path == "?":
            sf2path = _dialogs.selectFileForOpen('soundfontLastDirectory',
                                                 filter="*.sf2", prompt="Select Soundfont",
                                                 ifcancel="No soundfont selected, aborting")
        cfg = getConfig()
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
                              builtin=_builtin,
                              description=description,
                              properties={'sfpath': sf2path})
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
        config = getConfig()
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
            name = getConfig()['play.instr']
        elif name == "?":
            name = _dialogs.selectFromList(list(self.presetdefs.keys()),
                                           title="Select Preset",
                                           default=getConfig()['play.instr'])
        preset = self.presetdefs.get(name)
        if not preset:
            raise KeyError(f"Preset {name} not known. Available presets: {self.presetdefs.keys()}")
        return preset

    def definedPresets(self) -> list[str]:
        """Returns a list with the names of all defined presets

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> definedPresets()
            ['sin',
             'saw',
             'pulse',
             '_piano',
             'gm-clarinet',
             'gm-oboe']
            >>> preset = presetManager.getPreset('sin')
            Preset: sin  (transposable sine wave)
              |ktransp=0, klag=0.1|
              aout1 oscili a(kamp), mtof(lag(kpitch+ktransp, klag))


        .. note::

            To show more information about each preset, see
            :func:`~maelzel.core.presetman.showPresets`. You can also access all
            presets via :attr:`PresetManager.presetdefs`
        """
        return list(self.presetdefs.keys())

    def showPresets(self, pattern="*", full=False, showGeneratedCode=False) -> None:
        """
        Show the selected presets

        The output is printed to stdout if inside a terminal or shown as
        html inside of jupyter

        Args:
            pattern: a glob pattern to select which presets are shown
            full: show all attributes of a Preset
            showGeneratedCode: if True, all generated code is shown. Assumes *full*

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> showPresets("piano")
            Preset: piano
              init: iSfTable_ sfloadonce "/home/user/sf2/grand-piano-YDP.sf2"
              audiogen:
                ipresetidx sfPresetIndex "/home/user/sf2/grand-piano-YDP.sf2", 0, 0
                inote0_ = round(p(idataidx_ + 1))
                ivel_ = p(idataidx_ + 2) * 127
                aout1, aout2 sfplay ivel_, inote0_, kamp/16384, mtof:k(kpitch), ipresetidx, 1
              epilogue:
                turnoffWhenSilent aout1

        At init time the samples are loaded. The event turns itself off when the sample
        is silent, so it is possible to use an infinite duration to produce a one-shot
        playback.

        Args:
            pattern: a glob pattern to select which presets are shown
            showGeneratedCode: if True, all actual code of an instrument
                is show. Otherwise only the audiogen is shown

        """
        selectedPresets = [presetName for presetName in self.presetdefs.keys()
                           if fnmatch.fnmatch(presetName, pattern)]
        selectedPresets.sort(key=lambda name: self.presetdefs[name]._sortOrder())
        if showGeneratedCode:
            full = True
        if not emlib.misc.inside_jupyter():
            for presetName in selectedPresets:
                presetdef = self.presetdefs[presetName]
                print("")
                if not showGeneratedCode:
                    print(presetdef)
                else:
                    print(presetdef.getInstr().body)
        else:
            theme = getConfig()['html.theme']
            htmls = []
            if full:
                for presetName in selectedPresets:
                    presetdef = self.presetdefs[presetName]
                    html = presetdef._repr_html_(theme, showGeneratedCode=showGeneratedCode)
                    htmls.append(html)
                    htmls.append("<hr>")
            else:
                # short
                for presetName in selectedPresets:
                    presetdef = self.presetdefs[presetName]
                    l = f"<b>{presetdef.name}</b>"
                    if presetdef.isSoundFont():
                        sfpath = presetdef.properties.get('sfpath')
                        if not sfpath:
                            sfpath = presetutils.findSoundfontInPresetdef(presetdef) or '??'
                        l += f" [sf: {sfpath}]"
                    if presetdef.params:
                        s = ", ".join(f"{k}={v}" for k, v in presetdef.params.items())
                        s = f" <code>({s})</code>"
                        l += s
                    if (descr:=presetdef.description):
                        l += f"<br>&nbsp&nbsp&nbsp&nbsp<i>{descr}</i>"
                    l += "<br>"
                    htmls.append(l)
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
        path = self.presetsPath
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
        config = getConfig()
        sr = sr or config['rec.sr']
        ksmps = ksmps or config['rec.ksmps']
        nchnls = nchnls or config['rec.nchnls']
        state = getWorkspace()
        renderer = csoundengine.Renderer(sr=sr, nchnls=nchnls, ksmps=ksmps,
                                         a4=state.a4)
        renderer.addGlobalCode(_csoundPrelude)
        # Define all instruments
        for presetdef in self.presetdefs.values():
            instr = presetdef.getInstr()
            renderer.registerInstr(instr)
            globalCode = presetdef.globalCode()
            if globalCode:
                logger.debug(f"makeRenderer: adding global code for instr {instr.name}:\n{globalCode}")
                renderer.addGlobalCode(globalCode)
        return renderer

    def openPresetsDir(self) -> None:
        """
        Open a file manager at presetsPath
        """
        path = self.presetsPath
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
        path = os.path.join(self.presetsPath, f"{presetName}.yaml")
        if not os.path.exists(path):
            logger.warning(f"Preset {presetName} does not exist (searched: {path})")
            presetnames = self.savedPresets()
            logger.info(f"User defined presets: {presetnames}")
            return False
        os.remove(path)
        return True

    def savedPresets(self) -> list[str]:
        """
        Returns a list of saved presets

        Returns:
            a list of the names of the presets saved to the presets path

        .. seealso:: :func:`presetsPath`
        """
        presets = glob.glob(os.path.join(self.presetsPath, "*.yaml"))
        return [os.path.splitext(os.path.split(p)[1])[0] for p in presets]

    def selectPreset(self) -> Optional[str]:
        """
        Select one of the available presets via a GUI

        Returns:
            the name of the selected preset, or None if selection was canceled
        """
        return emlib.dialogs.selectItem(presetManager.definedPresets(),
                                        title="Select Preset")


presetManager = PresetManager()
defPresetSoundfont = presetManager.defPresetSoundfont
showPresets = presetManager.showPresets
definedPresets = presetManager.definedPresets
defPreset = presetManager.defPreset