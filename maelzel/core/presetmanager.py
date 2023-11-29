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
import csoundengine

from .presetdef import PresetDef
from .workspace import Workspace
from . import presetutils
from . import builtinpresets
from ._common import logger
from . import _dialogs
from . import environment
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .synthevent import SynthEvent


__all__ = (
    'presetManager',
    'showPresets',
    'defPreset',
    'defPresetSoundfont',
    # 'PresetManager',
    'getPreset'
)


_csoundPrelude = r"""
opcode turnoffWhenSilent, 0, a
    asig xin
    ksilent_  trigger detectsilence:k(asig, 0.0001, 0.05), 0.5, 0
    if ksilent_ == 1  then
      turnoff
    endif    
endop

opcode makePresetEnvelope, a, iii
    ifadein, ifadeout, ifadekind xin
    igain = 1.0
    ifinite = p3 > 0 ? 1 : 0
    if ifinite == 1 then
        if (ifadekind == 0) then
            aenv linseg 0, ifadein, igain, p3-ifadein-ifadeout, igain, ifadeout, 0
        elseif (ifadekind == 1) then
            aenv cosseg 0, ifadein, igain, p3-ifadein-ifadeout, igain, ifadeout, 0
        elseif (ifadekind == 2) then
            aenv transeg 0, ifadein*.5, 2, igain*0.5, ifadein*.5, -2, igain, p3-ifadein-ifadeout, igain, 1, ifadeout*.5, 2, igain*0.5, ifadeout*.5, -2, 0 	
            ; aenv *= linenr:a(1, 0, ifadeout, 0.01)
        endif
        if ifadeout > 0 then
            aenv *= cossegr:a(1, ifadein, 1, ifadeout, 0)
            ; aenv *= transegr:a(1, ifadein, 1, 1, ifadeout, -2, 0)
        endif
    else
        if (ifadekind == 0) then
            aenv linsegr 0, ifadein, igain, ifadeout, 0
        elseif (ifadekind == 1) then
            aenv cossegr 0, ifadein, igain, ifadeout, 0
        elseif (ifadekind == 2) then
            aenv transegr 0, ifadein*.5, 2, igain*0.5, ifadein*.5, -2, igain, ifadeout, -2, 0 	
            aenv *= linenr:a(1, 0, ifadeout, 0.01)
        endif
    endif
    xout aenv
endop
"""


class PresetManager:
    """
    Singleton object, manages all instrument Presets

    Any `maelzel.core` object can be played with an instrument preset defined
    here. A PresetManager is attached to a running Session as soon as an object
    is scheduled with the given Preset. As such, it acts as a library of Presets
    and any number of such Presets can be created.

    """
    _instance: PresetManager | None = None
    csoundPrelude = _csoundPrelude

    def __init__(self):
        if self._instance is not None:
            raise RuntimeError("Only one PresetManager should be active")
        self.presetdefs: dict[str, PresetDef] = {}
        self.presetsPath = Workspace.presetsPath()
        self._prepareEnvironment()
        self._makeBuiltinPresets()
        self.loadPresets()
        self._instance = self

    @staticmethod
    def instance():
        if PresetManager._instance is not None:
            return PresetManager._instance
        return PresetManager()

    def loadPresets(self) -> None:
        """
        Loads user-defined presets
        """
        presetdefs = presetutils.loadPresets()
        for presetdef in presetdefs:
            self.registerPreset(presetdef)

    def _prepareEnvironment(self) -> None:
        if not os.path.exists(self.presetsPath):
            os.makedirs(self.presetsPath)

    def _makeBuiltinPresets(self, sf2path: str = None) -> None:
        """
        Defines all builtin presets
        """
        for presetdef in builtinpresets.builtinPresets:
            self.registerPreset(presetdef)

        sf2 = presetutils.resolveSoundfontPath(path=sf2path)
        if not sf2:
            logger.info("No soundfont defined, builtin instruments using soundfonts will "
                        "not be available. Set config['play.generalMidiSoundfont'] to "
                        "the path of an existing soundfont")
        else:
            for instr, preset in builtinpresets.soundfontGeneralMidiPresets.items():
                if sf2 and sf2 != "?":
                    presetname = 'gm-' + instr
                    descr = f'General MIDI {instr}'
                    self.defPresetSoundfont(presetname, sf2path=sf2, preset=preset,
                                            _builtin=True, description=descr)

        for name, (path, preset, descr) in builtinpresets.builtinSoundfonts().items():
            self.defPresetSoundfont(name, sf2path=path, preset=preset, _builtin=True,
                                    description=descr)

    def defPreset(self,
                  name: str,
                  code: str,
                  init='',
                  post='',
                  includes: list[str] = None,
                  args: dict[str, float] = None,
                  description='',
                  envelope=True,
                  output=True,
                  aliases: dict[str, str] | None = None
                  ) -> PresetDef:
        """
        Define a new instrument preset.

        The defined preset can be used as note.play(instr='name'), where name
        is the name of the preset. A preset is created by defining the audio generating
        part as csound code. Each preset has access to the following variables:

        - **kpitch**: pitch of the event, as fractional midi
        - **kamp**: linear amplitude (0-1)
        - **kfreq**: frequency corresponding to kpitch

        `code` should generate an audio output signal named ``aout1`` for channel 1,
        ``aout2`` for channel 2, etc.::

            code = 'aout1 oscili a(kamp), kfreq'

        Args:
            name: the name of the preset
            code: audio generating csound code
            post: code to include after any other code. Needed when using turnoff,
                since calling turnoff in the middle of an instrument can cause undefined
                behaviour.
            init: global code needed for all instances of this preset (usually a table
                definition, loading samples, etc). It will be run only once before any
                event with this preset is scheduled. Do not confuse this with the init phase
                of an instrument, which runs for every event.
            includes: files to include
            args: a dict ``{parametername: value}`` passed to the instrument. Parameters can
                also be defined inline using the ``|iarg=<default>, karg=<default>|``
                notation
            description: an optional description of the preset. The description can include
                documentation for the parameters (see Example)
            envelope: If True, apply an envelope as determined by the fadein/fadeout
                play arguments. If False, the user is responsible for applying any fadein/fadeout (csound variables:
                ``ifadein``, ``ifadeout``
            output: if True, generate output routing (panning and output) for this
                preset. Otherwise, the user is responsible for applying panning (``kpos``)
                and routing the generated audio to any output channels (``ichan``), buses, etc.
            aliases: if given, a dict mapping alias to real parameter name. This allow to
                use any name for a parameter, instead of a csound variable

        Returns:
            a PresetDef

        Example
        ~~~~~~~

        Create a preset with dynamic parameters

        .. code-block:: python

            >>> from maelzel.core import *
            >>> presetManager.defPreset('mypreset', r'''
            ...     aout1 vco2 kamp, kfreq, 10
            ...     aout1 moogladder aout1, lag:k(kcutoff, 0.1), iq''',
            ...     args={'kcutoff': 4000, 'iq': 1},
            ...     description=r'''
            ...         A filtered saw-tooth
            ...         Args:
            ...             kcutoff: the cutoff frequency of the filter
            ...             iq: the filter resonance
            ...     ''')

        Or simply:

            >>> defPreset('mypreset', r'''
            ... |kcutoff=4000, kq=1|
            ... ; A filtered saw-tooth
            ... ; Args:
            ... ;   kcutoff: cutoff freq. of the filter
            ... ;   kq: filter resonance
            ... aout1 vco2 kamp, kfreq, 10
            ... aout1 moogladder aout1, lag:k(kcutoff, 0.1), kq
            ... ''', aliases={'cutoff': 'kcutoff')

        Then, to use the Preset:

            >>> synth = Note("4C", dur=60).play(instr='mypreset', args={'kcutoff': 1000})

        The :meth:`maelzel.core.mobj.MObj.play` method returns a SynthGroup, even if in
        this case a Note generates only one synth (for example a Chord generates one synth per note)

        **NB**: Parameters can be modified while the synth is running :

            >>> synth.set(kcutoff=2000)

        .. seealso::

            - :func:`defPresetSoundfont`
            - :meth:`PresetManager.getPreset`
            - :meth:`maelzel.core.MObj.play`

        """
        presetdef = PresetDef(name=name,
                              code=code,
                              init=init,
                              epilogue=post,
                              includes=includes,
                              args=args,
                              description=description,
                              builtin=False,
                              envelope=envelope,
                              routing=output,
                              aliases=aliases)
        self.registerPreset(presetdef)
        # NB: before, we would register the preset to the session
        # via playback.playSession().registerInstr(presetdef.getInstr())
        # But it is not necessary, since it will be done the first time
        # the instrument is used
        return presetdef

    def defPresetSoundfont(self,
                           name='',
                           sf2path='',
                           preset: tuple[int, int] | str = (0, 0),
                           init='',
                           postproc='',
                           includes: list[str] = None,
                           args: dict[str, float] | None = None,
                           interpolation='',
                           mono=False,
                           ampDivisor: int = None,
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
                is used.
            sf2path: the path to the soundfont. Use "?" open a dialog to select a .sf2 file
                or None to use the default soundfont
            preset: the preset to use. Either a tuple (bank: int, presetnum: int) or the
                name of the preset as string. **Use "?" to select from all available presets
                in the soundfont**.
            init: global code needed by postproc.
            postproc: code to modify the generated audio before it is sent to the
                outputs. **NB**: the audio is placed in *aout1*, *aout2*, etc. depending
                on the number of channels (normally 2)
            includes: files to include (if needed by init or postproc)
            args: mutable values needed by postproc (if any). See :meth:`~PresetManager.defPreset`
            mono: if True, only the left channel of the soundfont is read
            ampDivisor: most soundfonts are PCM 16bit files and need to be scaled down
                to use them in the range of -1:1. This value is used to scale amp down.
                The default is 16384, but it can be changed in the config
                (:ref:`key 'play.soundfontAmpDiv' <config_play_soundfontampdiv>`)
            interpolation: one of 'linear', 'cubic'. Refers to the interpolation used
                when reading the sample waveform. If None, use the default defined
                in the config (:ref:`key 'play.soundfontInterpolation' <config_play_soundfontinterpolation>`)
            turnoffWhenSilent: if True, turn a note off when the sample stops (by detecting
                silence for a given amount of time)
            description: a short string describing this preset
            _builtin: if True, marks this preset as built-in

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
        if not sf2path:
            sf2path = presetutils.resolveSoundfontPath()
            if sf2path is None:
                sf2path = "?"
        if sf2path == "?":
            sf2path = _dialogs.selectFileForOpen('soundfontLastDirectory',
                                                 filter="*.sf2", prompt="Select Soundfont",
                                                 ifcancel="No soundfont selected, aborting")
            assert sf2path is not None
        cfg = Workspace.getActive().config
        if not interpolation:
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
        if not name:
            name = idx.presetToName[(bank, presetnum)]
        if (bank, presetnum) not in idx.presetToName:
            raise ValueError(f"Preset ({bank}:{presetnum}) not found. Possible presets: "
                             f"{idx.presetToName.keys()}")
        code = presetutils.makeSoundfontAudiogen(sf2path=sf2path,
                                                 preset=(bank, presetnum),
                                                 interpolation=interpolation,
                                                 ampDivisor=ampDivisor,
                                                 mono=mono)
        # We don't actually need the global variable because sfloadonce
        # saves the table number into a channel
        init0 = f'i__SfTable__ sfloadonce "{sf2path}"'
        if init:
            init = "\n".join((init0, init))
        else:
            init = init0
        if postproc:
            code = emlib.textlib.joinPreservingIndentation((code, '\n;; postproc\n', postproc))
        epilogue = "turnoffWhenSilent aout1" if turnoffWhenSilent else ''
        ownargs = {'ktransp': 0., 'ipitchlag': 0.1}
        args = ownargs if not args else args | ownargs
        presetdef = self.defPreset(name=name,
                                   code=code,
                                   init=init,
                                   post=epilogue,
                                   includes=includes,
                                   args=args,
                                   description=description,
                                   aliases={'transpose': 'ktransp'})
        presetdef.userDefined = not _builtin
        presetdef.properties = {'sfpath': sf2path}
        return presetdef

    def registerPreset(self, presetdef: PresetDef) -> None:
        """
        Register this PresetDef.

        Args:
            presetdef: the PresetDef to register

        """
        self.presetdefs[presetdef.name] = presetdef

    def getPreset(self, name: str = '?') -> PresetDef:
        """
        Get a preset by name

        Raises KeyError if no preset with such name is defined

        Args:
            name: the name of the preset to get (use "?" to select from a list
                of defined presets)

        Returns:
            the PresetDef corresponding to the given name
        """
        if name is None:
            name = Workspace.getActive().config['play.instr']
        elif name == "?":
            # Presets starting with _ are private, presets starting with . are builtin
            presets = [name for name in self.presetdefs.keys()
                       if not name.startswith('_')]
            selected = _dialogs.selectFromList(options=presets,
                                               title="Select Preset",
                                               default=Workspace.getActive().config['play.instr'])
            if selected is None:
                raise ValueError("No preset selected")
            name = selected
        preset = self.presetdefs.get(name)
        if not preset:
            raise KeyError(f"Preset '{name}' not known. Available presets: {self.definedPresets()}")
        return preset

    def getInstr(self, presetname: str) -> csoundengine.Instr:
        """
        Get the Instr corresponding to the given presetname

        Args:
            presetname: the name of the preset

        Returns:
            the actual :class:`csoundengine.Instr`

        """
        return self.presetdefs[presetname].getInstr()

    def definedPresets(self) -> list[str]:
        """Returns a list with the names of all defined presets

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> presetManager.definedPresets()
            ['sin',
             'saw',
             'pulse',
             '.piano',
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
        return list(k for k in self.presetdefs.keys() if not k.startswith('_'))

    def showPresets(self, pattern="*", full=False, showGeneratedCode=False) -> None:
        """
        Show the selected presets

        The output is printed to stdout if inside a terminal or shown as
        html inside jupyter

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
              code:
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
                is show. Otherwise, only the audio code is shown

        """
        matchingPresets = [p for name, p in self.presetdefs.items()
                           if fnmatch.fnmatch(name, pattern)]

        def key(p: PresetDef):
            return 1 - int(p.userDefined), 1 - int(p.isSoundFont()), p.name

        matchingPresets.sort(key=key)

        if showGeneratedCode:
            full = True
        if not environment.insideJupyter:
            for preset in matchingPresets:
                print("")
                if not showGeneratedCode:
                    print(preset)
                else:
                    instr = preset.getInstr()
                    print(csoundengine.session.Session.defaultInstrBody(instr))
        else:
            theme = Workspace.getActive().config['htmlTheme']
            htmls = []
            if full:
                for preset in matchingPresets:
                    html = preset._repr_html_(theme, showGeneratedCode=showGeneratedCode)
                    htmls.append(html)
                    htmls.append("<hr>")
            else:
                # short
                for preset in matchingPresets:
                    l = f"<b>{preset.name}</b>"
                    if preset.isSoundFont():
                        sfpath = preset.properties.get('sfpath')
                        if not sfpath:
                            sfpath = presetutils.findSoundfontInPresetdef(preset) or '??'
                        l += f" [sf: {sfpath}]"
                    if preset.args:
                        s = ", ".join(f"{k}={v}" for k, v in preset.args.items())
                        s = f" <code>({s})</code>"
                        l += s
                    if descr := preset.description:
                        l += f"<br>&nbsp&nbsp&nbsp&nbsp<i>{descr}</i>"
                    l += "<br>"
                    htmls.append(l)
            from IPython.core.display import display, HTML
            display(HTML("\n".join(htmls)))

    def eventMaxNumChannels(self, event: SynthEvent) -> int:
        """
        Number of channels needed to play the event

        Given a SynthEvent, which defines a base channel (.chan)
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

    def savePreset(self, preset: str | PresetDef) -> str:
        """
        Saves the preset in the presets' folder, returns the path to the saved file

        Args:
            preset: the name of the preset

        Returns:
            the path of the saved preset
        """
        fmt = "yaml"
        if isinstance(preset, PresetDef):
            presetdef = preset
            if presetdef.name not in self.presetdefs:
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
                     sr: int = None,
                     numChannels: int = None,
                     ksmps: int = None
                     ) -> csoundengine.Renderer:
        """
        Make an offline Renderer from instruments defined here

        Args:
            sr: the sr of the renderer
            numChannels: the number of channels
            ksmps: if not explicitely set, will use config 'rec.ksmps'

        Returns:
            a csoundengine.Renderer
        """
        workspace = Workspace.getActive()
        config = workspace.config
        sr = sr or config['rec.sr']
        ksmps = ksmps or config['rec.ksmps']
        numChannels = numChannels or config['rec.numChannels']
        renderer = csoundengine.Renderer(sr=sr, nchnls=numChannels, ksmps=ksmps,
                                         a4=workspace.a4)
        renderer.addGlobalCode(presetManager.csoundPrelude)
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

            presetName = _dialogs.selectFromList(saved, title="Remove Preset")
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

    def selectPreset(self) -> str | None:
        """
        Select one of the available presets via a GUI

        Returns:
            the name of the selected preset, or None if selection was canceled
        """
        return _dialogs.selectFromList(self.definedPresets(), title="Select Preset")


presetManager = PresetManager.instance()

defPreset = presetManager.defPreset
defPresetSoundfont = presetManager.defPresetSoundfont
getPreset = presetManager.getPreset
showPresets = presetManager.showPresets
