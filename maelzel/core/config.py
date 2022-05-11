"""
Configuration
=============

At any given moment there is one active configuration. The configuration
allows to set default values for many settings to customize different
aspects of **maelzel.core**:

* notation (default page size, rendered image scaling, etc)
* playback (default audio backend, instrument, etc)
* offline rendering
* etc.

The active config is an instance of ConfigDict (https://configdict.readthedocs.io/en/latest/),
which is itself a subclass of `dict`. Settings can be modified by simply changing the
values of this dict, like ``config[key] = value``. A config has a **set of valid keys**: an
attempt to set an unknown key will result in an error. Values are also validated regarding
their type and accepted choices, range, etc.

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

See also: :py:module:`csoundengine.workspace`

Active config
-------------

In order to create a configuration specific for a particular task it is possible
to create a new config. This will clone the current workspace with a new config
with the given modifications.

    >>> from maelzel.core import *
    # Create a config to work with old tuning and display notation using a3 page size
    >>> cfg = newWorkspace(updates={'A4': 435, 'show.pageSize': 'a3'}, active=True)
    # do something with this, then deactivate the workspace

-------------------------

Keys
----

A4:
    | Default: **442**  -- `int`
    | Between 415 - 460

defaultDuration:
    | Default: **1.0**  -- `(int, float)`
    | *Value used when a duration is needed and has not been set (Note, Chord). Not the same as play.dur*

splitAcceptableDeviation:
    | Default: **4**  -- `int`
    | *When splitting notes between staves, notes within this range of the split point will be grouped together if they all fit*

chord.arpeggio:
    | Default: **auto**  -- `(str, bool)`
    | Choices: ``False, True, auto``
    | *Arpeggiate notes of a chord when showing. In auto mode, only arpeggiate when needed*

chord.adjustGain:
    | Default: **True**  -- `bool`
    | *Adjust the gain of a chord according to the number of notes, to prevent clipping*

m21.displayhook.install:
    | Default: **True**  -- `bool`

m21.displayhook.format:
    | Default: **xml.png**  -- `str`
    | Choices: ``xml.png, lily.png``

m21.fixStream:
    | Default: **True**  -- `bool`
    | *If True, fix the streams returned by .asmusic21 (see m21fix)*

repr.showFreq:
    | Default: **True**  -- `bool`
    | *Show frequency when calling printing a Note in the console*

semitoneDivisions:
    | Default: **4**  -- `int`
    | Choices: ``1, 2, 4``
    | *The number of divisions per semitone (2=quarter-tones, 4=eighth-tones)*

show.lastBreakpointDur:
    | Default: **0.125**  -- `float`
    | Between 0.015625 - 1
    | *Dur of a note representing the end of a line/gliss, which has no duration per se*

show.cents:
    | Default: **True**  -- `bool`
    | *show cents deviation as text when rendering notation*

show.centsFontSize:
    | Default: **8**  -- `int`
    | *Font size used for cents annotations*

show.split:
    | Default: **True**  -- `bool`
    | *Should a voice be split between two staves?. A midinumber can be given instead*

show.gliss:
    | Default: **True**  -- `bool`
    | *If true, show a glissando line where appropriate*

show.centSep:
    | Default: **,**  -- `str`
    | *Separator used when displaying multiple cents deviation (in a chord)*

show.scaleFactor:
    | Default: **1.0**  -- `float`
    | *Affects the size of the generated image when using png format*

show.staffSize:
    | Default: **12.0**  -- `float`
    | *The size of a staff, in points*

show.backend:
    | Default: **lilypond**  -- `str`
    | Choices: ``lilypond, music21``
    | *method/backend used when rendering notation*

show.format:
    | Default: **png**  -- `str`
    | Choices: ``png, pdf, repr``
    | *Used when no explicit format is passed to .show*

show.external:
    | Default: **False**  -- `bool`
    | *Force opening images with an external tool, even when inside a Jupyter notebook*

show.cacheImages:
    | Default: **True**  -- `bool`
    | *If True, new images are only generated when the object being rendered as notation has changed. Normally this should be left as True but can be deactivated for debugging*

show.arpeggioDuration:
    | Default: **0.5**  -- `float`
    | *Duration used for individual notes when rendering a chord as arpeggio*

show.labelFontSize:
    | Default: **12.0**  -- `float`

show.pageOrientation:
    | Default: **portrait**  -- `str`
    | Choices: ``portrait, landscape``
    | *Page orientation when rendering to pdf*

show.pageSize:
    | Default: **a4**  -- `str`
    | Choices: ``a3, a2, a4``
    | *The page size when rendering to pdf*

show.pageMarginMillimeters:
    | Default: **4**  -- `int`
    | Between 0 - 1000
    | *The page margin in mm*

show.glissEndStemless:
    | Default: **False**  -- `bool`
    | *When the end pitch of a gliss. is shown as gracenote, make this stemless*

show.lilypondPngStaffsizeScale:
    | Default: **1.0**  -- `float`
    | *A factor applied to the staffsize when rendering to png via lilypond. Withoutthis, it might happen that the renderer image is too small*

show.measureAnnotationFontSize:
    | Default: **14**  -- `int`
    | *Font size used for measure annotations*

show.respellPitches:
    | Default: **True**  -- `bool`

show.horizontalSpacing:
    | Default: **normal**  -- `str`
    | Choices: ``normal, medium, large, xlarge``
    | *Hint for the renderer to adjust horizontal spacing. The actual result dependson the backend and the format used*

show.glissandoLineThickness:
    | Default: **2**  -- `int`
    | Choices: ``1, 2, 3, 4``
    | *Line thinkness when rendering glissandi. The value is abstract and it isup to the renderer to interpret it*

show.fillDynamicFromAmplitude:
    | Default: **False**  -- `bool`
    | *If True, when showing a musicobj as notation, if such object has an amplitude and does not  have an explicit dynamic, add a dynamic according to the amplitude*

app.png:
    | Default: ****  -- `str`
    | *Application used when opening .png files externally. If an empty string is set, a suitable default for the platform will be selected*

displayhook.install:
    | Default: **True**  -- `bool`

play.dur:
    | Default: **2.0**  -- `float`
    | *Default duration of any play action if the object has no given duration*

play.gain:
    | Default: **1.0**  -- `float`
    | Between 0 - 1

play.chan:
    | Default: **1**  -- `int`
    | Between 1 - 64
    | *Default channel to play to. channels start at 1*

play.engineName:
    | Default: **maelzel.core**  -- `str`
    | *Name of the play engine used*

play.instr:
    | Default: **sin**  -- `str`
    | *Default instrument used for playback. A list of available instruments can be queried via `availableInstrs`. New instrument presets can be defined via `defPreset`*

play.fade:
    | Default: **0.02**  -- `float`
    | *default fade time*

play.fadeShape:
    | Default: **cos**  -- `str`
    | Choices: ``cos, linear``
    | *Curve-shape used for fading in/out*

play.pitchInterpolation:
    | Default: **linear**  -- `str`
    | Choices: ``cos, linear``
    | *Curve shape for interpolating between pitches*

play.numChannels:
    | Default: **2**  -- `int`
    | *Default number of channels (channels can be set explicitely when calling startPlayEngine*

play.unschedFadeout:
    | Default: **0.05**  -- `float`
    | *fade out when stopping a note*

play.autostartEngine:
    | Default: **True**  -- `bool`
    | *Start play engine if not started manually. This is done when the user performs an action which indirectly needs the engine to be running, like defining an instrument, or calling play.getPlayManager()*

play.backend:
    | Default: **default**  -- `str`
    | Choices: ``default, portaudio, jack, auhal, pa_cb, pulse, alsa``
    | *backend used for playback*

play.presetsPath:
    | Default: ****  -- `str`
    | *The path were presets are saved*

play.autosavePresets:
    | Default: **True**  -- `bool`
    | *Automatically save user defined presets, so they will be available for a next session*

play.defaultAmplitude:
    | Default: **1.0**  -- `float`
    | Between 0 - 1
    | *The amplitude of a Note/Chord when an amplitude is needed and the object has an undefined amplitude*

play.generalMidiSoundfont:
    | Default: ****  -- `str`
    | *Path to a soundfont (sf2 file) with a general midi mapping*

play.namedArgsMethod:
    | Default: **pargs**  -- `str`
    | Choices: ``table, pargs``
    | *Method used to convert named parameters defined in a Preset to their corresponding function in a csoundengine.Instr*

play.soundfontAmpDiv:
    | Default: **16384**  -- `int`

play.soundfontInterpolation:
    | Default: **linear**  -- `str`
    | Choices: ``cubic, linear``
    | *Interpolation used when reading sample data from a soundfont.*

play.schedLatency:
    | Default: **0.2**  -- `float`
    | *Added latency when scheduling events to ensure time precission*

play.verbose:
    | Default: **False**  -- `bool`
    | *If True, outputs extra debugging information regarding playback*

rec.block:
    | Default: **False**  -- `bool`
    | *Should recording be blocking or should be done async?*

rec.sr:
    | Default: **44100**  -- `int`
    | Choices: ``48000, 96000, 44100, 88200``
    | *Sample rate used when rendering offline*

rec.ksmps:
    | Default: **64**  -- `int`
    | Choices: ``32, 1, 64, 128, 256, 16``
    | *samples per cycle when rendering offline (passed as ksmps to csound)*

rec.nchnls:
    | Default: **2**  -- `int`

rec.path:
    | Default: ****  -- `str`
    | *path used to save output files when rendering offline. If not given the default can be queried via `recordPath`*

rec.quiet:
    | Default: **False**  -- `bool`
    | *Supress debug output when calling csound as a subprocess*

html.theme:
    | Default: **light**  -- `str`
    | Choices: ``dark, light``
    | *Theme used when displaying html inside jupyter*

quant.minBeatFractionAcrossBeats:
    | Default: **1.0**  -- `float`

quant.nestedTuples:
    | Default: **False**  -- `bool`
    | *Are nested tuples allowed when quantizing? NB: not all display backends support nested tuples (for example, musescore, which is used to render musicxml to pdf, does not support nested tuples)*

quant.complexity:
    | Default: **middle**  -- `str`
    | Choices: ``low, high, middle``
    | *Controls the allowed complexity in the notation. The higher the complexity, the more accurate the timing of the quantization, at the cost of a more complex notation. The value is used as a preset, controlling aspects like which subdivisions of the beat are allowed at a given tempo, the weighting of each subdivision, etc.*

logger.level:
    | Default: **INFO**  -- `str`
    | Choices: ``WARNING, DEBUG, ERROR, INFO``

"""
from __future__ import annotations
import os
import re

from configdict import ConfigDict
from ._common import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Dict, Optional


_default = {
    'A4': 442,
    'defaultDuration': 1.0,
    'splitAcceptableDeviation': 4,
    'chord.arpeggio': 'auto',
    'chord.adjustGain': True,
    'm21.displayhook.install': True,
    'm21.displayhook.format': 'xml.png',
    'm21.fixStream': True,
    'repr.showFreq': True,
    'semitoneDivisions': 4,

    'dynamicsCurve.shape': 'expon(3.0)',

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
    'show.labelFontSize': 12.0,
    'show.pageOrientation': 'portrait',
    'show.pageSize': 'a4',
    'show.pageMarginMillimeters': 4,
    'show.glissEndStemless': False,
    'show.glissHideTiedNotes': True,
    'show.lilypondPngStaffsizeScale': 1.5,
    'show.measureAnnotationFontSize': 14,
    'show.respellPitches': True,
    'show.horizontalSpacing': 'normal',
    'show.glissandoLineThickness': 2,
    'show.fillDynamicFromAmplitude': False,
    'show.jupyterMaxImageWidth': 1000,

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
    'play.generalMidiSoundfont': '',
    'play.namedArgsMethod': 'pargs',
    'play.soundfontAmpDiv': 16384,
    'play.soundfontInterpolation': 'linear',
    'play.schedLatency': 0.2,
    'play.verbose': False,
    'rec.block': False,
    'rec.sr': 44100,
    'rec.ksmps': 64,
    'rec.nchnls': 2,
    'rec.path': '',
    'rec.quiet': False,
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
    'defaultDuration::type': (int, float),
    'semitoneDivisions::choices': {1, 2, 4},
    'm21.displayhook.format::choices': {'xml.png', 'lily.png'},
    'show.backend::choices': {'music21', 'lilypond'},
    'show.format::choices': {'png', 'pdf', 'repr'},
    'show.staffSize::type': float,
    'show.pageSize::choices': {'a3', 'a4', 'a2'},
    'chord.arpeggio::choices': {'auto', True, False},
    'play.gain::range': (0, 1),
    'play.fadeShape::choices': {'linear', 'cos'},
    'play.numChannels::type': int,
    'play.soundfontInterpolation::choices': {'linear', 'cubic'},
    'rec.sr::choices': {44100, 48000, 88200, 96000},
    'rec.ksmps::choices': {1, 16, 32, 64, 128, 256},    
    'play.defaultAmplitude::range': (0, 1),
    'play.pitchInterpolation::choices': {'linear', 'cos'},
    'app.png::type': str,
    'play.generalMidiSoundfont': lambda cfg, key, val: val == '' or (os.path.exists(val) and os.path.splitext(val)[1] == '.sf2'),
    'play.namedArgsMethod::choices': {'table', 'pargs'},
    'html.theme::choices': {'light', 'dark'},
    'show.lastBreakpointDur::range': (1/64., 1),
    'quant.complexity::choices': {'low', 'middle', 'high'},
    'show.pageOrientation::choices': {'portrait', 'landscape'},
    'show.pageMarginMillimeters::range': (0, 1000),
    'show.horizontalSpacing::choices': {'normal', 'medium', 'large', 'xlarge'},
    'show.glissandoLineThickness::choices': {1, 2, 3, 4},
    'logger.level::choices': {'DEBUG', 'INFO', 'WARNING', 'ERROR'},
    'show.jupyterMaxImageWidth::type': int
}

_docs = {
    'A4':
        "Freq. of the Kammerton A4. Normal values are 440, 442, 443 or 432 for old tuning, "
        "but any 'fantasy' value can be used",
    'defaultDuration':
        "Value used when a duration is needed and has not been set (Note, Chord)."
        " Not the same as play.dur",
    'semitoneDivisions':
        "The number of divisions per semitone (2=quarter-tones, 4=eighth-tones)",
    'repr.showFreq':
        "Show frequency when calling printing a Note in the console",
    'chord.arpeggio':
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
        "has an undefined amplitude",
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
    'show.horizontalSpacing':
        'Hint for the renderer to adjust horizontal spacing. The actual result depends'
        'on the backend and the format used',
    'show.jupyterMaxImageWidth':
        'A max. width in pixels for images displayed in a jupyter notebook',
    'play.backend':
        'backend used for playback',
    'rec.path':
        'path used to save output files when rendering offline. If '
        'not given the default can be queried via `recordPath`',
    'show.cacheImages':
        'If True, cache rendered images. Set it to False for debugging',
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
    'quant.complexity':
        'Controls the allowed complexity in the notation. The higher the complexity,'
        ' the more accurate the quantization, at the cost of a more complex notation. ',
    'quant.nestedTuples':
        'Are nested tuples allowed when quantizing? Not all display backends support'
        ' nested tuples (musescore, used to render musicxml '
        ' has no support for nested tuples)'
}


def checkEnvironment(config:dict=None) -> List[str]:
    """
    Check that we have everything we need

    Returns:
        a list of errors or None if no errors found
    """
    # check that musescore is installed if necessary
    errors = []
    # check if musescore is setup
    import maelzel.music.m21tools
    musescorepath = maelzel.music.m21tools.findMusescore()
    if not musescorepath:
        msg = "In the configuration the key 'show.format' is set to" \
              "'xml.png'. MuseScore is needed to handle this conversion," \
              " and its path must be configured as " \
              "music21.environment.UserSettings()['musescoreDirectPNGPath'] = '/path/to/musescore'"
        logger.error(msg)
        errors.append(f"MuseScore not found")
    else:
        logger.debug("Checked if MuseScore is setup correctly: OK")
    return errors or None


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


def isValidConfig(config: ConfigDict) -> bool:
    """
    Is this a valid config?
    """
    return (config.default == rootConfig.default)


rootConfig = ConfigDict('maelzel.core', _default, persistent=False,
                        validator=_validator, docs=_docs)
rootConfig.registerCallback(lambda d, k, v: _syncCsoundengineTheme(v), re.escape("html.theme"))
rootConfig.registerCallback(lambda d, k, v: _resetImageCacheCallback(), "show\..+")
rootConfig.registerCallback(lambda d, k, v: _propagateA4(d, v), "A4")