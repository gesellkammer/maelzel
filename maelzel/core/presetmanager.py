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

from . import presetdef as _presetdef
from .workspace import Workspace
from . import presetutils
from . import builtinpresets
from . import environment

from ._common import logger

import typing as _t
if _t.TYPE_CHECKING:
    from .synthevent import SynthEvent
    import csoundengine
    import csoundengine.instr
    import csoundengine.offline


__all__ = (
    'presetManager',
    'showPresets',
    'defPreset',
    'defPresetSoundfont',
    'getPreset'
)


_csoundPrelude = r"""
; dict holding peaks for soundfont normalization
gi__soundfont_peaks dict_new "int:float"

instr _sfpeak
    ipreset = p4
    ipitch1 = p5
    ipitch2 = p6
    kmax0 init 0
    a1 sfplaym 127, ipitch1, 1, 1, ipreset, 0
    a2 sfplaym 127, ipitch2, 1, 1, ipreset, 0
    kmax1 peak a1
    kmax2 peak a2
    kmax = max(kmax1, kmax2)
    if kmax > kmax0 then
        println "sf peak: %f", kmax
        dict_set gi__soundfont_peaks, ipreset, kmax
    endif
    kmax0 = kmax
endin

opcode turnoffWhenSilent, 0, a
    asig xin
    ksilent_  trigger detectsilence:k(asig, 0.0001, 0.05), 0.5, 0
    if ksilent_ == 1  then
      turnoff
    endif
endop

opcode _linexp, i, iiiiii
    ix, iexp, ix0, ix1, iy0, iy1 xin
    idx = (ix - ix0) / (ix1 - ix0)
    iy = (idx ^ iexp) * (iy1 - iy0) + iy0
    ; iy = limit:i(iy, iy0, iy1)
    xout iy
endop

opcode soundfontPlay, aa, ikkpjjppjj
    ; aout1, aout2 soundfontPlay ipreset, kpitch, kamp
    ; i      k       k     p        j        j        p          p          j        j
    ipreset, kpitch, kamp, iinterp, iampdiv, ivelexp, ivelmindb, ivelmaxdb, iminvel, imaxvel xin
    ; iinterp: 1=linear, 3=cubic
    iampdiv = iampdiv > 0 ? iampdiv : 16000
    ivelexp = ivelexp > 0 ? ivelexp : 3
    ivelmindb = ivelmindb <= 0 ? ivelmindb : -60
    ivelmaxdb = ivelmaxdb <= 0 ? ivelmaxdb : -3
    iminvel = iminvel > 0 ? iminvel : 1
    imaxvel = imaxvel > 0 ? imaxvel : 127

    inote0 = round(i(kpitch))
    iamp0 = i(kamp)
    ivel = _linexp(dbamp:i(iamp0), ivelexp, ivelmindb, ivelmaxdb, iminvel, imaxvel)
    if iinterp == 1 then
        aout1, aout2 sfplay ivel, inote0, kamp/iampdiv, mtof:k(kpitch), ipreset, 1
    else
        aout1, aout2 sfplay3 ivel, inote0, kamp/iampdiv, mtof:k(kpitch), ipreset, 1
    endif
    xout aout1, aout2
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
        self.presetdefs: dict[str, _presetdef.PresetDef] = {}
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

    def _makeBuiltinPresets(self) -> None:
        """
        Defines all builtin presets
        """
        for presetdef in builtinpresets.makeBuiltinPresets():
            self.registerPreset(presetdef)

        # sf2 = presetutils.resolveSoundfontPath(path=sf2path)
        # if not sf2:
        #     logger.info("No soundfont defined, builtin instruments using soundfonts will "
        #                 "not be available. Set config['play.generalMidiSoundfont'] to "
        #                 "the path of an existing soundfont")
        # else:
        #     for instr, preset in builtinpresets.soundfontGeneralMidiPresets.items():
        #         if sf2 and sf2 != "?":
        #             presetname = 'gm-' + instr
        #             descr = f'General MIDI {instr}'
        #             self.defPresetSoundfont(presetname, sf2path=sf2, preset=preset,
        #                                     _builtin=True, description=descr)

        for name, info in builtinpresets.builtinSoundfonts().items():
            self.defPresetSoundfont(name,
                                    sf2path=info['sf2path'],
                                    preset=info['preset'],
                                    description=info.get('description', ''),
                                    ampDivisor=info.get('ampDivisor'),
                                    _builtin=True,
                                    )

    def defPreset(self,
                  name: str,
                  code: str,
                  init='',
                  post='',
                  includes: _t.Sequence[str] = (),
                  args: dict[str, float] | None = None,
                  description='',
                  envelope=True,
                  output=True,
                  aliases: dict[str, str] | None = None
                  ) -> _presetdef.PresetDef:
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
                play arguments. If False, the user is responsible for applying any fadein/fadeout
                (the csound variables ``ifadein`` and ``ifadeout`` will be set to the given
                fade times). No envelope code is generated if no output variables are defined
                (``aout1``, ``aout2``, ...)
            output: if True, generate output routing (panning and output) for this
                preset. Otherwise, the user is responsible for applying panning (``kpos``)
                and routing the generated audio to any output channels (``ichan``), buses, etc.
                No output code is generated if the user does not define any output variables
                (``aout1``, ``aout2``, ...)
            aliases: if given, a dict mapping alias to real parameter name. This mechanism
                allows to use any name for a parameter, instead of a csound variable

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

        To output sound to anything different than the hardware output use ``output=False`` and
        implement the output directly within the instr body

        >>> defPreset('mysynth', r'''
        ... asig vco2 kamp, kfreq, 10
        ... asigL, asigR pan2 asig, kpos
        ... chnmix asigL, "left"
        ... chnmix asigR, "right"
        ... ''')
        >>> session = getSession()
        >>> session.defInstr('reverb', r'''
        ...     |kwet=0.8|
        ...     aleft = chnget:a("left")
        ...     aright = chnget:a("right")
        ...     awetL, awetR reverbsc aleft, aright, 0.85, 12000, sr, 0.5, 1
        ...     outch 1, awetL * kwet + aleft * (1 - kwet), 2, awetR * kwet + aright * (1 - kwet)
        ...     chnclear "left", "right"
        ... ''')
        >>> reverb = session.sched('reverb', priority=2)
        >>> synth = Note("4C", dur=1).play(instr='mysynth')

        .. seealso::

            - :func:`defPresetSoundfont`
            - :meth:`PresetManager.getPreset`
            - :meth:`maelzel.core.MObj.play`

        """
        presetdef = _presetdef.PresetDef(name=name,
                              code=code,
                              init=init,
                              epilogue=post,
                              includes=includes,
                              args=args,
                              description=description,
                              envelope=envelope,
                              routing=output,
                              aliases=aliases)
        self.registerPreset(presetdef)
        # NB: before, we would register the preset to the session
        # via playback.getSession().registerInstr(presetdef.getInstr())
        # But it is not necessary, since it will be done the first time
        # the instrument is used
        return presetdef

    def defPresetSoundfont(self,
                           name='',
                           sf2path='',
                           preset: tuple[int, int] | str = (0, 0),
                           init='',
                           postproc='',
                           reverb=False,
                           includes: _t.Sequence[str] = (),
                           args: dict[str, float] | None = None,
                           interpolation='',
                           mono=False,
                           ampDivisor: int | float = 0,
                           turnoffWhenSilent=True,
                           description='',
                           normalize=False,
                           velocityCurve: _t.Sequence[float] | _presetdef.GainToVelocityCurve = (),
                           reverbChanPrefix='',
                           _builtin=False) -> _presetdef.PresetDef:
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
            velocityCurve: either a flat list of pairs of the form [db0, vel0, db1, vel1, ...],
                mapping dB values to velocities, or an instance of GainToVelocityCurve
            ampDivisor: most soundfonts are PCM 16bit files and need to be scaled down
                to use them in the range of -1:1. This value is used to scale amp down.
                The default is 16384 (it can be changed in the config
                (:ref:`key 'play.soundfontAmpDiv' <config_play_soundfontampdiv>`), but
                different soundfonts might need different scaling factors.
            interpolation: one of 'linear', 'cubic'. Refers to the interpolation used
                when reading the sample waveform. If None, use the default defined
                in the config (:ref:`key 'play.soundfontInterpol' <config_play_soundfontInterpol>`)
            turnoffWhenSilent: if True, turn a note off when the sample stops (by detecting
                silence for a given amount of time)
            description: a short string describing this preset
            normalize: if True, queries the amplitude divisor of the soundfont at runtime
                and uses that to scale amplitudes to 0dbfs
            reverbChanPrefix: ???
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
            sf2path = presetutils.resolveSoundfontPath() or '?'
        if sf2path == "?":
            from . import _dialogs
            sf2path = _dialogs.selectFileForOpen('soundfontLastDirectory',
                                                 filter="*.sf2", prompt="Select Soundfont")
            if sf2path is None:
                raise ValueError("No soundfont selected")

        cfg = Workspace.active.config
        if not interpolation:
            interpolation = cfg['play.soundfontInterpol']
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
        from csoundengine import sftools
        idx = sftools.soundfontIndex(sf2path)
        if not name:
            name = idx.presetToName[(bank, presetnum)]
        if (bank, presetnum) not in idx.presetToName:
            raise ValueError(f"Preset ({bank}:{presetnum}) not found. Possible presets: "
                             f"{idx.presetToName.keys()}")
        if normalize and not ampDivisor and cfg['play.soundfontFindPeakAOT']:
            sfpeak = sftools.soundfontPeak(sfpath=sf2path, preset=(bank, presetnum))
            if sfpeak > 0:
                ampDivisor = sfpeak
                normalize = False

        code = presetutils.makeSoundfontAudiogen(sf2path=sf2path,
                                                 preset=(bank, presetnum),
                                                 interpolation=interpolation,
                                                 ampDivisor=ampDivisor,
                                                 mono=mono,
                                                 normalize=normalize,
                                                 reverb=reverb,
                                                 reverbChanPrefix=reverbChanPrefix,
                                                 velocityCurve=velocityCurve)

        # We don't actually need the global variable because sfloadonce
        # saves the table number into a channel
        init0 = f'''i__SfTable__ sfloadonce "{sf2path}"'''
        if init:
            init = "\n".join((init0, init))
        else:
            init = init0
        if postproc:
            code = emlib.textlib.joinPreservingIndentation((code, '\n;; postproc\n', postproc))
        epilogue = "turnoffWhenSilent aout1" if turnoffWhenSilent else ''
        ownargs = {'ktransp': 0., 'ipitchlag': 0.1, 'ivel': -1, 'kwet': 0.}
        args = ownargs if not args else args | ownargs
        presetdef = self.defPreset(name=name,
                                   code=code,
                                   init=init,
                                   post=epilogue,
                                   includes=includes,
                                   args=args,
                                   description=description,
                                   output=not reverb,
                                   aliases={'transpose': 'ktransp'})
        presetdef.userDefined = not _builtin
        presetdef.properties = {'sfpath': sf2path,
                                'ampDivisor': ampDivisor}
        return presetdef

    def registerPreset(self, presetdef: _presetdef.PresetDef) -> None:
        """
        Register this PresetDef.

        Args:
            presetdef: the PresetDef to register

        """
        self.presetdefs[presetdef.name] = presetdef

    def getPreset(self, name: str = '?') -> _presetdef.PresetDef:
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
            name = Workspace.active.config['play.instr']
        elif name == "?":
            # Presets starting with _ are private, presets starting with . are builtin
            presets = [name for name in self.presetdefs.keys()
                       if not name.startswith('_')]
            from . import _dialogs
            selected = _dialogs.selectFromList(options=presets,
                                               title="Select Preset",
                                               default=Workspace.active.config['play.instr'])
            if selected is None:
                raise ValueError("No preset selected")
            name = selected
        preset = self.presetdefs.get(name)
        if not preset:
            raise KeyError(f"Preset '{name}' not known. Available presets: {self.definedPresets()}")
        return preset

    def getInstr(self, presetname: str) -> csoundengine.instr.Instr:
        """
        Get the Instr corresponding to the given presetname

        Args:
            presetname: the name of the preset

        Returns:
            the actual :class:`csoundengine.instr.Instr`

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
                ipresetidx sfpresetindex "/home/user/sf2/grand-piano-YDP.sf2", 0, 0
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

        def key(p: _presetdef.PresetDef):
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
                    import csoundengine.session
                    print(csoundengine.session.Session.defaultInstrBody(instr))
        else:
            theme = Workspace.active.config['htmlTheme']
            htmls = []
            if full:
                for preset in matchingPresets:
                    html = preset._repr_html_(theme, showGeneratedCode=showGeneratedCode)
                    htmls.append(html)
                    htmls.append("<hr>")
            else:
                # short
                for preset in matchingPresets:
                    line = f"<b>{preset.name}</b>"
                    if preset.isSoundFont():
                        sfpath = preset.properties.get('sfpath')
                        if not sfpath:
                            sfpath = presetutils.findSoundfontInPresetdef(preset) or '??'
                        line += f" [sf: {sfpath}]"
                    if preset.args:
                        s = ", ".join(f"{k}={v}" for k, v in preset.args.items())
                        s = f" <code>({s})</code>"
                        line += s
                    if descr := preset.description:
                        line += f"<br>&nbsp&nbsp&nbsp&nbsp<i>{descr}</i>"
                    line += "<br>"
                    htmls.append(line)
            from IPython.display import display, HTML
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

    def savePreset(self, preset: str | _presetdef.PresetDef) -> str:
        """
        Saves the preset in the presets' folder, returns the path to the saved file

        Args:
            preset: the name of the preset

        Returns:
            the path of the saved preset
        """
        fmt = "yaml"
        if isinstance(preset, _presetdef.PresetDef):
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
                     sr=0,
                     numChannels: int | None = None,
                     ksmps=0,
                     ) -> csoundengine.offline.OfflineSession:
        """
        Make an offline Renderer from instruments defined here

        Args:
            sr: if given, the sr of the renderer. Otherwise we use config `rec.sr'
            numChannels: the number of channels, will use config 'rec.numChannels' if not given
            ksmps: if not explicitely set, will use config 'rec.ksmps'

        Returns:
            a csoundengine.OfflineSession
        """
        workspace = Workspace.active
        config = workspace.config
        sr = sr or config['rec.sr']
        ksmps = ksmps or config['rec.ksmps']
        numChannels = numChannels or config['rec.numChannels']
        from csoundengine.offline import OfflineSession
        renderer = OfflineSession(sr=sr, nchnls=numChannels, ksmps=ksmps,
                                  a4=workspace.a4)

        renderer.compile(presetManager.csoundPrelude)
        return renderer

    def openPresetsDir(self) -> None:
        """
        Open a file manager at presetsPath
        """
        path = self.presetsPath
        emlib.misc.open_with_app(path)

    def removeUserPreset(self, presetName: str) -> bool:
        """
        Remove a user defined preset

        Args:
            presetName: the name of the preset to remove. Use "?" to
                select from a list of removable presets

        Returns:
            True if the preset was removed, False if it did not exist
        """
        if presetName == "?":
            saved = self.savedPresets()
            if not saved:
                logger.info("No saved presets, aborting")
                return False
            from . import _dialogs
            selected = _dialogs.selectFromList(saved, title="Remove Preset")
            if not selected:
                return False
            presetName = selected

        path = os.path.join(self.presetsPath, f"{presetName}.yaml")
        if not os.path.exists(path):
            logger.warning(f"Preset {presetName} does not exist (searched: {path})")
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
        from . import _dialogs
        return _dialogs.selectFromList(self.definedPresets(), title="Select Preset")


presetManager = PresetManager.instance()

defPreset = presetManager.defPreset
defPresetSoundfont = presetManager.defPresetSoundfont
getPreset = presetManager.getPreset
showPresets = presetManager.showPresets
