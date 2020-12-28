from __future__ import annotations
import sys
import os
import shutil
import music21 as m21
import appdirs
from functools import lru_cache
from configdict import ConfigDict
from emlib.misc import dictmerge

from ._base import *


_defaultconfig = {
    'A4': 442,
    'defaultDuration': 1.0,
    'splitAcceptableDeviation': 4,
    'repr.showFreq': True,
    'chord.arpeggio': 'auto',
    'chord.adjustGain': True,
    'm21.displayhook.install': True,
    'm21.displayhook.format': 'xml.png',
    'm21.fixstream': True,
    'show.semitoneDivisions':4,
    'show.lastBreakpointDur':1/8,
    'show.cents': True,
    'show.centsMethod': 'lyric',
    'show.centsFontSize': 8,
    'show.split': True,
    'show.gliss': True,
    'show.centSep': ',',
    'show.scaleFactor': 1.0,
    'show.format': 'xml.png',
    'show.external': False,
    'show.cacheImages': True,
    'show.seqDuration': 1,
    'show.defaultDuration': 1,
    'show.arpeggioDuration': 0.5,
    'show.label.fontSize': 12.0,
    'use_musicxml2ly': True,
    'app.png': 'feh --image-bg white' if sys.platform == 'linux' else '',
    'displayhook.install': True,
    'play.dur': 2.0,
    'play.gain': 0.5,
    'play.chan': None,
    'play.group': 'emlib.mus2',
    'play.instr': 'sin',
    'play.fade': 0.02,
    'play.fadeShape': 'cos',
    'play.pitchInterpolation': 'linear',
    'play.numChannels': 2,
    'play.unschedFadeout': 0.05,
    'play.autostartEngine': True,
    'play.backend': None,
    'play.presetsPath': '',
    'play.autosavePresets': True,
    'play.unknownParameterFailSilently': False,
    'play.defaultAmplitude': 1.0,
    'rec.block': False,
    'rec.gain': 1.0,
    'rec.samplerate': 44100,
    'rec.ksmps': 64,
    'rec.path': '',
    'rec.quiet': False
}

_validator = {
    'defaultDuration::type': (int, float, Fraction),
    'show.semitoneDivisions::choices': {1, 2, 4},
    'm21.displayhook.format::choices': {'xml.png', 'lily.png'},
    'show.format::choices':
        {'xml.png', 'xml.pdf', 'lily.png', 'lily.pdf', 'repr'},
    'chord.arpeggio::choices': {'auto', True, False},
    'play.gain::range': (0, 1),
    'play.fadeShape::choices': {'linear', 'cos'},
    'play.numChannels::type': int,
    'show.centsMethod::choices': {'lyric', 'expression'},
    'rec.samplerate::choices': {44100, 48000, 88200, 96000},
    'rec.ksmps::choices': {1, 16, 32, 64, 128, 256},
    'play.defaultAmplitude::range': (0, 1)
}

_docs = {
    'defaultDuration': 
        "Value used when a duration is needed and has not been set (Note, Chord)."
        " Not the same as play.dur",
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
        "Should a voice be split between two stafs. A midinumber can be given "
        "instead",
    'show.lastBreakpointDur':
        "Dur of a note representing the end of a line/gliss, which has "
        "no duration per se",
    'show.semitoneDivisions':
        "The number of divisions per semitone (2=quarter-tones, 4=eighth-tones)",
    'show.scaleFactor':
        "Affects the size of the generated image",
    'show.format':
        "Used when no explicit format is passed to .show",
    'show.gliss':
        "If true, show a glissando line where appropriate",
    'play.numChannels':
        "Default number of channels (channels can be set explicitely when calling"
        " startPlayEngine",
    'play.defaultAmplitude':
        "The amplitude of a Note/Chord when an amplitude is needed and the object"
        "has an undefined amplitude",
    'rec.block':
        "Default value when calling .rec (True=.rec will block until finished,"
        " otherwise recording is done async)",
    'use_musicxml2ly':
        "Use musicxml2ly when converting xml 2 lily, instead of the builtin"
        " conversion in music21",
    'play.group':
        "Name of the play engine used",
    'm21.fixstream':
        "If True, fix the streams returned by .asmusic21 (see m21fix)",
    'show.seqDuration':
        "Default duration of each element of a NoteSeq or ChordSeq when shown",
    'show.label.fontSize':
        "Font size to use for labels",
    'show.centsFontSize':
        "Font size used for cents annotations",
    'play.presetsPath': 'The path were presets are saved',
    'play.autosavePresets':
        'Automatically save user defined presets, so they will be available '
        'for a next session',
    'play.unknownParameterFailSilently':
        'If True, any attempt to set an unknown parameter will be silently ignored',
    'splitAcceptableDeviation':
        'When splitting notes between staves, notes within this range of the '
        'split point will be grouped together if they all fit',
    'play.autostartEngine':
        'Start play engine if not started manually. This is done when the user'
        'performs an action which indirectly needs the engine to be running,'
        'like defining an instrument, or calling play.getPlayManager()',
    'rec.quiet':
        'Supress debug output when calling csound as a subprocess'
}


def _checkConfig(cfg, key, oldvalue, value):
    if key == 'notation.semitoneDivisions' and value == 4:
        showformat = cfg.get('show.format')
        if showformat and showformat.startswith('lily'):
            newvalue = oldvalue if oldvalue is not None else 2
            msg = ("\nlilypond backend (show.format) does not support 1/8 tones yet.\n"
                   "Either set config['notation.semitoneDivisions'] to 2 or\n"
                   "set config['show.format'] to 'xml.png'."
                   "Setting notation.semitoneDivision to {newvalue}")
            logger.error(msg)
            return newvalue


def makeConfig(temp=False, source=None, **kws) -> ConfigDict:
    if not temp:
        out = ConfigDict(f'maelzel:core', _defaultconfig, validator=_validator, docs=_docs,
                         precallback=_checkConfig, persistent=True)
    else:
        out = ConfigDict(f'maelzel:core', _defaultconfig, validator=_validator, docs=_docs,
                         precallback=_checkConfig, persistent=False)
    if source:
        out.update(source)
    out.update(kws)
    return out


config = ConfigDict(f'maelzel:music_core', _defaultconfig, validator=_validator,
                    docs=_docs, precallback=_checkConfig)


# Environment


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


def checkEnvironment(solve=True) -> List[str]:
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


@lru_cache(maxsize=1)
def _presetsPath() -> str:
    datadirbase = appdirs.user_data_dir("emlib")
    path = os.path.join(datadirbase, "music_core", "presets")
    return path


def presetsPath() -> str:
    """ Returns the path of the presets directory """
    userpath = config['play.presetsPath']
    if userpath:
        return userpath
    return _presetsPath()


def recordPath() -> str:
    """ The path where temporary recordings are saved
    We do not use the temporary folder because it is wiped regularly
    and you might. The returned folder is guaranteed to exist
    """
    userpath = config['rec.path']
    if userpath:
        path = userpath
    else:
        path = appdirs.user_data_dir(appname="emlib", version="recordings")
    if not os.path.exists(path):
        os.makedirs(path)
    return path