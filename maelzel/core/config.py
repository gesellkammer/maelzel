from __future__ import annotations
import os
import sys
import shutil
import music21 as m21
from configdict import ConfigDict
import re
import emlib.misc
from ._common import *


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

    'show.lastBreakpointDur':1/8,
    'show.cents': True,
    'show.centsFontSize': 8,
    'show.split': True,
    'show.gliss': True,
    'show.centSep': ',',
    'show.scaleFactor': 1.0,
    'show.staffSize': 12.0,
    'show.backend': 'music21',
    'show.format': 'png',
    'show.external': False,
    'show.cacheImages': True,
    'show.arpeggioDuration': 0.5,
    'show.labelFontSize': 12.0,
    'show.pageOrientation': 'portrait',
    'show.pageSize': 'a4',
    'show.pageMarginMillimeters': 4,
    'show.glissEndStemless': False,
    'show.lilypondPngStaffsizeScale': 1.0,
    'show.measureAnnotationFontSize': 14,
    'show.respellPitches': True,
    'show.horizontalSpacing': 'normal',
    'show.glissandoLineThickness': 2,
    'show.fillDynamicFromAmplitude': False,

    'app.png': '',
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
    'play.namedArgsMethod': 'table',
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
    'A4::range': (415, 460),
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
    'logger.level::choices': {'DEBUG', 'INFO', 'WARNING', 'ERROR'}
}

_docs = {
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
        'Line thinkness when rendering glissandi. The value is abstract and it is'
        'up to the renderer to interpret it',
    'show.fillDynamicFromAmplitude':
        'If True, when showing a musicobj as notation, if such object has an amplitude '
        'and does not  have an explicit dynamic, add a dynamic according to the amplitude',
    'play.presetsPath': 'The path were presets are saved',
    'play.autosavePresets':
        'Automatically save user defined presets, so they will be available '
        'for a next session',
    'splitAcceptableDeviation':
        'When splitting notes between staves, notes within this range of the '
        'split point will be grouped together if they all fit',
    'play.autostartEngine':
        'Start play engine if not started manually. This is done when the user '
        'performs an action which indirectly needs the engine to be running, '
        'like defining an instrument, or calling play.getPlayManager()',
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
        'A factor applied to the staffsize when rendering to png via lilypond. Without'
        'this, it might happen that the renderer image is too small',
    'show.horizontalSpacing':
        'Hint for the renderer to adjust horizontal spacing. The actual result depends'
        'on the backend and the format used',
    'play.backend':
        'backend used for playback',
    'rec.path':
        'path used to save output files when rendering offline. If '
        'not given the default can be queried via `recordPath`',
    'show.cacheImages':
        'If True, new images are only generated when the object '
        'being rendered as notation has changed. Normally this should '
        'be left as True but can be deactivated for debugging',
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
        'Application used when opening .png files externally. If an empty string '
        'is set, a suitable default for the platform will be selected',
    'play.generalMidiSoundfont':
        'Path to a soundfont (sf2 file) with a general midi mapping',
    'html.theme':
        'Theme used when displaying html inside jupyter',
    'play.namedArgsMethod':
        'Method used to convert named parameters defined in a Preset to their'
        ' corresponding function in a csoundengine.Instr',
    'quant.complexity':
        'Controls the allowed complexity in the notation. The higher the complexity,'
        ' the more accurate the timing of the quantization, at the cost of a more complex'
        ' notation. The value is used as a preset, controlling aspects like which '
        'subdivisions of the beat are allowed at a' 
        ' given tempo, the weighting of each subdivision, etc.',
    'quant.nestedTuples':
        'Are nested tuples allowed when quantizing? NB: not all display backends support'
        ' nested tuples (for example, musescore, which is used to render musicxml to pdf,'
        ' does not support nested tuples)'
}


def checkEnvironment(config:dict, solve=True) -> List[str]:
    """
    Check that we have everything we need

    Returns:
        a list of errors or None if no errors found
    """
    # check that musescore is installed if necessary
    showFormat = config['show.format']
    errors = []
    if showFormat == 'xml.png' or config['m21.displayhook.format'] == 'xml.png':
        us = m21.environment.UserSettings()
        pngapp = us['musescoreDirectPNGPath']
        logger.debug("Checking if MuseScore is setup correctly")
        if not pngapp or not pngapp.exists():
            msg = "In the configuration the key 'show.format' is set to" \
                  "'xml.png'. MuseScore is needed to handle this conversion," \
                  " and its path must be configured as " \
                  "music21.environment.UserSettings()['musescoreDirectPNGPath'] = '/path/to/musescore'"
            logger.error(msg)
            if solve:
                solved = _setupMusescore()
                solvedstr = "solved" if solved else "not solved"
                logger.error("-----> " + solvedstr)
                errors.append("MuseScore not setup within music21 settings: {solvedstr}")
        else:
            logger.debug("Checked if MuseScore is setup correctly: OK")
    return errors or None


def _setupMusescore(force=False) -> bool:
    us = m21.environment.UserSettings()
    pngapp = us['musescoreDirectPNGPath']
    if pngapp is not None and pngapp.exists() and not force:
        return False
    mscore = _multiwhich("musescore", "MuseScore")
    if mscore:
        us['musescoreDirectPNGPath'] = mscore
        return True
    return False


def _multiwhich(*appnames) -> Opt[str]:
    for app in appnames:
        path = shutil.which(app)
        if path and os.path.exists(path):
            return path
    return None


# -----------------

def _syncCsoundengineTheme(theme:str):
    import csoundengine
    csoundengine.config['html_theme'] = theme

def _resetImageCacheCallback():
    from . import musicobj
    musicobj.resetImageCache()

def _propagateA4(config, a4):
    from . import workspace
    w = workspace.activeWorkspace()
    if config is w.config:
        w.a4 = a4

rootConfig = ConfigDict('maelzel.core', _default, validator=_validator,
                        docs=_docs, fmt='yaml')
rootConfig.registerCallback(lambda d, k, v: _syncCsoundengineTheme(v), re.escape("html.theme"))
rootConfig.registerCallback(lambda d, k, v: _resetImageCacheCallback(), "show\..+")
rootConfig.registerCallback(lambda d, k, v: _propagateA4(d, v), "A4")