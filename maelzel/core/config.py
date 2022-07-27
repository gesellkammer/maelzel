"""

CoreConfig: Configuration for maelzel.core
==========================================

At any given moment there is one active configuration (an instance of :class:`CoreConfig`,
itself a subclass of `dict`).
The configuration allows to set default values for many settings to customize different
aspects of **maelzel.core**:

* notation (default page size, rendered image scaling, etc). Prefix: *show*
* playback (default audio backend, instrument, etc). Prefix: *play*
* offline rendering. Prefix: *rec*
* etc.

Settings can be modified by simply changing the values of the config dict::

    # Get the active config
    >>> from maelzel.core import *
    >>> config = getConfig()
    >>> config['show.pageSize'] = 'A3'

A config has a :ref:`set of valid keys <coreconfigkeys>`. An attempt to set an unknown key will
result in an error. Values are also validated regarding their type and accepted
values, range, etc.::

    >>> config['foo'] = 'bar'
    KeyError: 'Unknown key foo'
    >>> config['show.pageSize'] = 'Z1'
    ValueError: key show.pageSize should be one of {'a2', 'a4', 'a3'}, got Z1

Persistence
-----------

Modifications to the active configuration can be made persistent by
saving the config.

    >>> from maelzel.core import *
    # Set the reference frequency to 443 for this and all future sessions
    >>> conf = getConfig()
    >>> conf['A4'] = 443
    # Set lilypond as default rendering backend
    >>> conf['show.backend'] = 'lilypond'
    >>> conf.save()

In a future session these changes will be picked up as default:

    >>> from maelzel.core import *
    >>> conf = getConfig()
    >>> conf['A4']
    443


.. seealso::

    :ref:`workspace_mod`

---------------------

.. _activeconfigh:

Active config
-------------

In order to create a configuration specific for a particular task it is possible
to create a new config with :func:`~maelzel.core.workspace.makeConfig` or by
cloning any CoreConfig.

    >>> from maelzel.core import *
    >>> newconfig = makeConfig({'show.pageSize': 'a3'}, active=True)
    # This is the same as
    >>> newconfig = getConfig().clone({'show.pageSize': 'a3'})
    >>> setConfig(newconfig)

Also creating a new :class:`~maelzel.core.workspace.Workspace` will create a new
config:

    >>> from maelzel.core import *
    # Create a config to work with old tuning and display notation using a3 page size
    >>> w = Workspace(updates={'A4': 435, 'show.pageSize': 'a3'}, active=True)
    # do something with this, then deactivate the workspace
    >>> n = Note("4A")
    >>> print(n.freq)
    435
    # Play with default instr
    >>> n.play()
    # When finished, deactivate it to return to previous Workspace
    >>> w.deactivate()
    >>> Note("4A").freq
    442

It is also possible to create a temporary config:

    >>> from maelzel.core import *
    >>> scale = Chain([Note(m, dur=0.5) for m in range(60, 72)])
    >>> with CoreConfig(updates={'show.pageSize':'a3'}):
    ...     scale.show()


"""
from __future__ import annotations
import os
import re
import typing
from maelzel import _state
from maelzel.music import dynamics

from configdict import ConfigDict

if typing.TYPE_CHECKING:
    from typing import Optional, Any
    from maelzel.scoring.render import RenderOptions


__all__ = (
    'rootConfig',
    'CoreConfig'
)


def _validateDecibelRange(cfg, key, val):
    if not isinstance(val, list):
        return "Expected a list"
    if len(val) != 2:
        return "Expected a list of two values"
    if not all(isinstance(_, int) for _ in val):
        return "Expected a list of two insts"
    return True


_default = {
    'A4': 442,
    'splitAcceptableDeviation': 4,
    'show.arpeggiateChord': 'auto',
    'chord.adjustGain': True,
    'm21.displayhook.install': True,
    'm21.displayhook.format': 'xml.png',
    'm21.fixStream': True,
    'repr.showFreq': True,
    'semitoneDivisions': 4,

    'dynamicCurve.shape': 'expon(0.3)',
    'dynamicCurve.mindb': -60,
    'dynamicCurve.maxdb': 0,
    'dynamicCurve.dynamics': 'ppp pp p mp mf f ff fff',

    'show.lastBreakpointDur':1/8,
    'show.cents': True,
    'show.centsFontSize': 8,
    'show.split': True,
    'show.gliss': True,
    'show.centSep': ',',
    'show.scaleFactor': 1.0,
    'show.staffSize': 12.0,
    'show.backend': 'lilypond',
    'show.format': 'png',
    'show.external': False,
    'show.cacheImages': True,
    'show.arpeggioDuration': 0.5,
    'show.labelFontSize': 10.0,
    'show.pageOrientation': 'portrait',
    'show.pageSize': 'a4',
    'show.pageMarginMillimeters': 4,
    'show.glissEndStemless': False,
    'show.glissHideTiedNotes': True,
    'show.lilypondPngStaffsizeScale': 1.5,
    'show.pngResolution': 200,
    'show.measureAnnotationFontSize': 12,
    'show.respellPitches': True,
    'show.horizontalSpacing': 'normal',
    'show.glissandoLineThickness': 2,
    'show.fillDynamicFromAmplitude': False,
    'show.jupyterMaxImageWidth': 1000,
    'show.hideRedundantDynamics': True,

    'app.png': '',
    'musescorepath': '',
    'displayhook.install': True,
    'play.dur': 2.0,
    'play.gain': 1.0,
    'play.chan': 1,
    'play.engineName': 'maelzel.core',
    'play.instr': 'sin',
    'play.fade': 0.02,
    'play.fadeShape': 'cos',
    'play.pitchInterpolation': 'linear',
    'play.numChannels': 2,
    'play.unschedFadeout': 0.05,
    'play.autostartEngine': True,
    'play.backend': 'default',
    'play.presetsPath': '',
    'play.autosavePresets': True,
    'play.defaultAmplitude': 1.0,
    'play.defaultDynamic': 'f',
    'play.generalMidiSoundfont': '',
    'play.namedArgsMethod': 'pargs',
    'play.soundfontAmpDiv': 16384,
    'play.soundfontInterpolation': 'linear',
    'play.schedLatency': 0.2,
    'play.verbose': False,
    'play.useDynamics': True,
    'rec.block': True,
    'rec.sr': 44100,
    'rec.ksmps': 64,
    'rec.nchnls': 2,
    'rec.path': '',
    'rec.quiet': True,
    'rec.compressionBitrate': 224,
    'html.theme': 'light',
    'quant.minBeatFractionAcrossBeats': 1.0,
    'quant.nestedTuples': False,
    'quant.complexity': 'middle',
    'logger.level': 'INFO',
}

_validator = {
    'A4::type': int,
    'A4::range': (10, 10000),
    'play.chan::type': int,
    'play.chan::range': (1, 64),
    'play.backend::choices': {'default', 'jack', 'pulse', 'alsa', 'pa_cb',
                              'auhal', 'portaudio'},
    'semitoneDivisions::choices': {1, 2, 4},
    'm21.displayhook.format::choices': {'xml.png', 'lily.png'},
    'show.backend::choices': {'music21', 'lilypond'},
    'show.format::choices': {'png', 'pdf', 'repr'},
    'show.staffSize::type': float,
    'show.pngResolution::choices': {100, 200, 300, 600, 1200},
    'show.pageSize::choices': {'a3', 'a4', 'a2'},
    'show.arpeggiateChord::choices': {'auto', True, False},
    'play.gain::range': (0, 1),
    'play.fadeShape::choices': {'linear', 'cos', 'scurve'},
    'play.numChannels::type': int,
    'play.soundfontInterpolation::choices': {'linear', 'cubic'},
    'rec.sr::choices': {44100, 48000, 88200, 96000},
    'rec.compressionBitrate::coices': {64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 500},
    'rec.ksmps::choices': {1, 16, 32, 64, 128, 256},    
    'play.defaultAmplitude::range': (0, 1),
    'play.pitchInterpolation::choices': {'linear', 'cos'},
    'app.png::type': str,
    'play.generalMidiSoundfont': lambda cfg, key, val: val == '' or (os.path.exists(val) and os.path.splitext(val)[1] == '.sf2'),
    'play.namedArgsMethod::choices': {'table', 'pargs'},
    'play.defaultDynamic::choices': {'pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff'},
    'html.theme::choices': {'light', 'dark'},
    'show.lastBreakpointDur::range': (1/64., 1),
    'quant.complexity::choices': {'low', 'middle', 'high'},
    'show.pageOrientation::choices': {'portrait', 'landscape'},
    'show.pageMarginMillimeters::range': (0, 1000),
    'show.horizontalSpacing::choices': {'normal', 'medium', 'large', 'xlarge'},
    'show.glissandoLineThickness::choices': {1, 2, 3, 4},
    'logger.level::choices': {'DEBUG', 'INFO', 'WARNING', 'ERROR'},
    'show.jupyterMaxImageWidth::type': int,
    'dynamicCurve.shape': lambda cfg, key, val: val.split("(")[0] in {'linear', 'expon', 'halfcos'},
    'dynamicCurve.mindb::range': (-160, 0),
    'dynamicCurve.maxdb::range': (-160, 0),
    'dynamicCurve.dynamics': lambda cfg, key, val: all(d in dynamics.dynamicSteps
                                                       for d in val.split())
}

_docs = {
    'A4':
        "Freq. of the Kammerton A4. Normal values are 440, 442, 443 or 432 for old tuning, "
        "but any 'fantasy' value can be used",
    'dynamicCurve.shape':
        "The shape used to create the default dynamics curve. The most convenient shape is some"
        " variation of an exponential, given as expon(exp), where exp is the exponential used. "
        "exp < 1 will result in more resolution for soft dynamics",
    'dynamicCurve.mindb':
        "The amplitude (in dB) corresponding to the softest dynamic",
    'dynamicCurve.maxdb':
        "The amplitude (in dB) corresponding to the loudest dynamic",
    'dynamicCurve.dynamics':
        "Possible dynamic steps. A string with all dynamic steps, sorted from softest to loudest",

    'semitoneDivisions':
        "The number of divisions per semitone (2=quarter-tones, 4=eighth-tones)",
    'repr.showFreq':
        "Show frequency when calling printing a Note in the console",
    'show.arpeggiateChord':
        "Arpeggiate notes of a chord when showing. In auto mode, only arpeggiate"
        " when needed",
    'chord.adjustGain':
        "Adjust the gain of a chord according to the number of notes, to prevent "
        "clipping",
    'show.external':
        "Force opening images with an external tool, even when inside a Jupyter "
        "notebook",
    'show.split':
        "Should a voice be split between two staves?. A midinumber can be given "
        "instead",
    'show.lastBreakpointDur':
        "Dur of a note representing the end of a line/gliss, which has "
        "no duration per se",
    'show.scaleFactor':
        "Affects the size of the generated image when using png format",
    'show.staffSize':
        "The size of a staff, in points",
    'show.format':
        "Used when no explicit format is passed to .show",
    'show.gliss':
        "If true, show a glissando line where appropriate",
    'play.numChannels':
        "Default number of channels (channels can be set explicitely when calling"
        " startPlayEngine",
    'play.defaultAmplitude':
        "The amplitude of a Note/Chord when an amplitude is needed and the object "
        "has an undefined amplitude. This is only used if play.useDynamics if False",
    'play.defaultDynamic':
        'THe dynamic of a Note/Chord when a dynamic is needed. This is only used if '
        'play.useDynamics is True. Any event with an amplitude will use that amplitude instead',
    'rec.block':
        "Should recording be blocking or should be done async?",
    'play.engineName':
        "Name of the play engine used",
    'play.chan':
        "Default channel to play to. channels start at 1",
    'm21.fixStream':
        "If True, fix the streams returned by .asmusic21 (see m21fix)",
    'show.label.fontSize':
        "Font size to use for labels",
    'show.centsFontSize':
        "Font size used for cents annotations",
    'show.measureAnnotationFontSize':
        'Font size used for measure annotations',
    'show.glissandoLineThickness':
        'Line thikness when rendering glissandi. The value is abstract and it is'
        'up to the renderer to interpret it',
    'show.glissHideTiedNotes':
        'Hide tied notes which are part of a glissando',
    'show.fillDynamicFromAmplitude':
        'If True, when rendering notation, if an object has an amplitude '
        'and does not have an explicit dynamic, add a dynamic according to the amplitude',
    'play.presetsPath': 'The path were presets are saved',
    'play.autosavePresets':
        'Automatically save user defined presets, so they will be available '
        'for a next session',
    'splitAcceptableDeviation':
        'When splitting notes between staves, notes within this range of the '
        'split point will be grouped together if they all fit',
    'play.autostartEngine':
        'Start play engine if not started manually?',
    'play.schedLatency':
        'Added latency when scheduling events to ensure time precission',
    'rec.quiet':
        'Supress debug output when calling csound as a subprocess',
    'play.dur':
        'Default duration of any play action if the object has no given duration',
    'rec.ksmps':
        'samples per cycle when rendering offline (passed as ksmps to csound)',
    'rec.compressionBitrate':
        'bitrate to use by default when encoding to ogg',
    'play.fade':
        'default fade time',
    'play.unschedFadeout':
        'fade out when stopping a note',
    'play.soundfontInterpolation':
        'Interpolation used when reading sample data from a soundfont.',
    'play.verbose':
        'If True, outputs extra debugging information regarding playback',
    'show.backend':
        'method/backend used when rendering notation',
    'show.cents':
        'show cents deviation as text when rendering notation',
    'show.pageOrientation':
        'Page orientation when rendering to pdf',
    'show.pageSize':
        'The page size when rendering to pdf',
    'show.glissEndStemless':
        'When the end pitch of a gliss. is shown as gracenote, make this stemless',
    'show.pageMarginMillimeters':
        'The page margin in mm',
    'show.lilypondPngStaffsizeScale':
        'A factor applied to the staffsize when rendering to png via lilypond. Useful '
        'if rendered images appear too small in a jupyter notebook',
    'show.pngResolution':
        'DPI used when rendering to png',
    'show.horizontalSpacing':
        'Hint for the renderer to adjust horizontal spacing. The actual result depends '
        'on the backend and the format used',
    'show.jupyterMaxImageWidth':
        'A max. width in pixels for images displayed in a jupyter notebook',
    'show.hideRedundantDynamics':
        'Hide redundant dynamics within a voice',
    'play.backend':
        'backend used for playback',
    'play.useDynamics':
        'If True, any note/chord with a set dynamic will use that to modify its playback '
        'amplitude if no explicit amplitude is set',
    'rec.path':
        'path used to save output files when rendering offline. If '
        'not given the default can be queried via `recordPath`',
    'show.cacheImages':
        'If True, cache rendered images. Set it to False for debugging. '
        'call `resetImageCache()` to reset manually',
    'show.arpeggioDuration':
        'Duration used for individual notes when rendering a chord as arpeggio',
    'rec.sr':
        'Sample rate used when rendering offline',
    'play.fadeShape':
        'Curve-shape used for fading in/out',
    'show.centSep':
        'Separator used when displaying multiple cents deviation (in a chord)',
    'play.instr':
        'Default instrument used for playback. A list of available instruments '
        'can be queried via `availableInstrs`. '
        'New instrument presets can be defined via `defPreset`',
    'play.pitchInterpolation':
        'Curve shape for interpolating between pitches',
    'app.png':
        'Application used when opening .png files externally. If empty, '
        'the platform default is used',
    'play.generalMidiSoundfont':
        'Path to a soundfont (sf2 file) with a general midi mapping',
    'html.theme':
        'Theme used when displaying html inside jupyter',
    'play.namedArgsMethod':
        'Method used to convert named parameters defined in a Preset to their'
        ' corresponding function in a csoundengine.Instr',
    'play.soundfontAmpDiv':
        'A divisor used to scale the amplitude of soundfonts to a range 0-1',
    'quant.complexity':
        'Controls the allowed complexity in the notation. The higher the complexity,'
        ' the more accurate the quantization, at the cost of a more complex notation. ',
    'quant.nestedTuples':
        'Are nested tuples allowed when quantizing? Not all display backends support'
        ' nested tuples (musescore, used to render musicxml '
        ' has no support for nested tuples)',
    'musescorepath':
        'The command to use when calling MuseScore. For macOS users: it must be an '
        'absolute path pointing to the actual binary inside the .app bundle'
}


def _syncCsoundengineTheme(theme:str):
    import csoundengine
    csoundengine.config['html_theme'] = theme


def _resetImageCacheCallback():
    from . import musicobj
    musicobj.resetImageCache()


def _propagateA4(config, a4):
    from . import workspace
    w = workspace.getWorkspace()
    if config is w.config:
        w.a4 = a4


class CoreConfig(ConfigDict):
    """
    A CoreConfig is a ``dict`` like object which controls many aspects of **maelzel.core**

    A **CoreConfig** reads its settings from a persistent copy. This persistent version is
    generated whenever the user calls the method :meth:`CoreConfig.save`.
    When **maelzel.core** is imported it reads this configuration and creates
    the ``rootConfig``, which is an instance of :class:`CoreConfig`.

    Notice that a configuration, in order to modify the behaviour of the environment,
    needs to be either actively used (passed as an argument to any function accepting
    a configuration object) or set as active via :func:`~maelzel.core.workspace.setConfig`
    or by calling its :meth:`CoreConfig.activate` method.

    Args:
        load: if True, the saved version is loaded when creating this CoreConfig
        updates: if given, a dict which will be used to update the newly created instance

    .. admonition:: See Also

        :ref:`Configuration Keys <coreconfigkeys>` for documentation on the keys and
        their possible values

    .. seealso:: :func:`maelzel.core.workspace.makeConfig`

    """
    def __init__(self, load=False, updates: dict[str, Any] = None, **kws):
        super().__init__('maelzel.core', _default, persistent=False,
                         validator=_validator, docs=_docs, load=load)
        self._prevConfig: Optional[CoreConfig] = None
        self.registerCallback(lambda d, k, v: _syncCsoundengineTheme(v), re.escape("html.theme"))
        self.registerCallback(lambda d, k, v: _resetImageCacheCallback(), "show\..+")
        self.registerCallback(lambda d, k, v: _propagateA4(d, v), "A4")
        if updates:
            self.update(updates)
        if kws:
            kws = {k:v for k, v in kws.items() if k in self.keys()}
            self.update(kws)

    def copy(self) -> CoreConfig:
        return CoreConfig(load=False, updates=self)

    def clone(self, updates: dict = None, **kws) -> CoreConfig:
        if updates:
            updates = self | updates
        else:
            updates = self

        if kws:
            kws = self._normalizeDict(kws)
            updates = updates|kws
        out = CoreConfig(load=False, updates=updates)
        return out

    def makeRenderOptions(self) -> RenderOptions:
        """
        Create RenderOptions based on this config

        Returns:
            a RenderOptions instance
        """
        from maelzel.core import notation
        return notation.makeRenderOptionsFromConfig(self)

    def __enter__(self):
        from . import workspace
        w = workspace.getWorkspace()
        self._prevConfig = w.config
        w.config = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from . import workspace
        w = workspace.Workspace.active
        assert self._prevConfig is not None
        w.config = self._prevConfig

    def activate(self) -> None:
        """
        Make this config the active config

        This is just a shortcut for ``setConfig(self)``

        """
        from .workspace import setConfig
        setConfig(self)


def onFirstRun():
    print("*** maelzel.core: first run")
    from maelzel.core import presetman
    if '_piano' in presetman.presetManager.presetdefs:
        print("*** maelzel.core: found builtin piano soundfont; setting default instrument to '_piano'")
        rootConfig['play.instr'] = '_piano'
        rootConfig.save()
    _state.state['first_run'] = False


rootConfig = CoreConfig(load=True)

if _state.state['first_run']:
    onFirstRun()
