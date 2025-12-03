import os
import math
from numbers import Rational
from ._configtools import isValidFraction, isValidStyle


_dynamicSteps = ('pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff')
_dynamicsSet = set(_dynamicSteps)


defaultdict = {
    'A4': 442,
    'chordAdjustGain': True,
    'reprShowFreq': False,
    'reprUnicodeAccidentals': True,
    'reprUnicodeFractions': False,
    'reprDurationAsFraction': True,
    'semitoneDivisions': 4,
    'musescorepath': '',
    'lilypondpath': '',
    'jupyterReprShow': True,
    'fixStringNotenames': False,
    'openImagesInExternalApp': False,
    'soundfilePlotHeight': 3,
    '.soundfilePlotWidth': 24,

    'enharmonic.horizontalWeight': 1,
    'enharmonic.verticalWeight': 0.5,
    '.enharmonic.debug': False,
    '.enharmonic.150centMicroPenalty': 20,

    'show.arpeggiateChord': 'auto',
    'show.centsTextStyle': 'fontsize=6; placement=below',
    'show.cents': True,
    'show.centsTextSnap': 2,
    '.show.centsTextPlusSign': True,
    'show.centSep': ',',
    'show.scaleFactor': 0.75,
    'show.scaleFactorMusicxml': 0.8,
    'show.staffSize': 10.0,
    'show.backend': 'lilypond',
    'show.format': 'png',
    'show.cacheImages': True,
    'show.arpeggioDuration': 0.5,
    'show.labelStyle': 'fontsize=9; placement=above',
    'show.pageOrientation': 'portrait',
    'show.pageSize': 'a4',
    'show.pageMarginMillim': 4,
    'show.glissStemless': True,
    'show.glissParenthesis': False,
    'show.glissHideTiedNotes': True,
    'show.glissLineThickness': 2,
    'show.glissLineType': 'solid',
    'show.lilypondPngStaffsizeScale': 1.5,
    'show.lilypondGlissMinLength': 5,
    'show.pngResolution': 200,
    'show.measureLabelStyle': 'box=rectangle; fontsize=12',
    'show.rehearsalMarkStyle': 'box=rectangle; fontsize=13; bold',
    'show.respellPitches': True,
    'show.horizontalSpace': 'medium',
    'show.dynamicFromAmplitude': False,
    'show.jupyterMaxImageWidth': 1000,
    'show.hideRedundantDynamics': True,
    '.show.dynamicsResetTime': 32,
    '.show.dynamicsResetAfterEmptyMeasure': True,
    '.show.dynamicsResetAfterRest': 1,
    'show.absOffsetWhenDetached': False,
    'show.voiceMaxStaves': 2,
    'show.clipNoteheadShape': 'square',
    'show.referenceStaffsize': 12.0,
    'show.musicxmlFontScaling': 1.0,
    'show.flagStyle': 'straight',
    'show.autoClefChanges': True,
    'show.clefSimplify': 0.,
    'show.spacing': 'normal',
    'show.proportionalDuration': '1/24',
    'show.warnIfEmpty': True,
    'show.clefChangesWindow': 1,
    'show.keepClefBias': 2.0,
    'show.clefTransposingFactor': 0.85,
    'show.pedalStyle': '',
    'show.singleStaffRange': 12,
    'show.explodeStaffPenalty': 1.2,
    'play.gain': 1.0,
    'play.engineName': 'maelzel.core',
    'play.instr': 'sin',
    'play.fade': 0.02,
    'play.fadeShape': 'cos',
    'play.pitchInterpol': 'linear',
    'play.numChannels': 2,
    'play.unschedFadeout': 0.05,
    'play.backend': 'default',
    'play.defaultAmplitude': 1.0,
    'play.defaultDynamic': 'f',
    'play.generalMidiSoundfont': '',
    'play.soundfontAmpDiv': 16384,
    'play.soundfontInterpol': 'linear',
    'play.schedLatency': 0.05,
    'play.verbose': False,
    'play.useDynamics': True,
    'play.graceDuration': '1/14',
    'play.soundfontFindPeakAOT': False,
    
    'reverbInstr': '.zitarev',
    'reverbGaindb': -6,
    'reverbDelayms': 60,
    'reverbDecay': 3,
    'reverbDamp': 0.2,
    'reverbRestart': True,

    'rec.blocking': True,
    'rec.sr': 44100,
    'rec.ksmps': 64,
    'rec.numChannels': 2,
    'rec.path': '',
    'rec.verbose': False,
    '.rec.compressionBitrate': 224,
    'rec.extratime': 0.,

    'htmlTheme': 'light',

    'quant.syncopMinFraction': '1/6',
    'quant.syncopPartMinFraction': '1/10',
    'quant.syncopMaxAsymmetry': 3,
    'quant.nestedTuplets': None,
    'quant.nestedTupletsMusicxml': False,
    'quant.breakBeats': 'weak',
    'quant.complexity': 'high',
    'quant.beatWeightTempoThresh': 52,
    'quant.subdivTempoThresh': 96,
    'quant.gridWeight': None,

    '.quant.divisionWeight': None,
    '.quant.complexityWeight': None,
    '.quant.gridErrorExp': None,
    '.quant.debug': False,
    '.quant.debugShowNumRows': 50,
    '.quant.mergeTupletsDifferentDur': False,

    'dynamicCurveShape': 'expon(0.3)',
    'dynamicCurveMindb': -60,
    'dynamicCurveMaxdb': 0,
    'dynamicCurveDynamics': 'ppp pp p mp mf f ff fff',
}


validator = {
    "lilypondpath": lambda cfg, key, val: val == "" or os.path.exists(val),
    "A4::type": int,
    "A4::range": (10, 10000),
    "play.backend::choices": {
        "default",
        "jack",
        "pulse",
        "alsa",
        "pa_cb",
        "auhal",
        "portaudio",
    },
    "semitoneDivisions::choices": {1, 2, 4},
    "show.backend::choices": {"musicxml", "lilypond"},
    "show.format::choices": {"png", "pdf", "repr"},
    "show.staffSize::type": float,
    "show.staffSize::range": (0.001, 100),
    "show.pngResolution::choices": {100, 200, 300, 600, 1200},
    "show.pageSize::choices": {"a3", "a4", "a2"},
    "show.arpeggiateChord::choices": {"auto", True, False},
    "play.gain::range": (0, 1),
    "play.fadeShape::choices": {"linear", "cos", "scurve"},
    "play.numChannels::type": int,
    "play.numChannels::range": (1, 128),
    "rec.numChannels::range": (1, 128),
    "play.soundfontInterpol::choices": {"linear", "cubic"},
    "reverbInstr::choices": {'.zitarev'},
    "reverbDelayms::type": int,
    "reverbDelayms::range": (1, 500),
    "reverbDecay::range": (0.01, 120),
    "reverbDamp::range": (0, 1),
    "rec.sr::choices": {
        44100,
        48000,
        88200,
        96000,
        192000,
    },
    ".rec.compressionBitrate::coices": {
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
        320,
    },
    "rec.ksmps::choices": {1, 16, 32, 64, 128, 256},
    "rec.extratime::range": (0.0, math.inf),
    "play.defaultAmplitude::range": (0, 1),
    "play.pitchInterpol::choices": {"linear", "cos"},
    "play.generalMidiSoundfont": lambda cfg, key, val: val == ""
    or (os.path.exists(val) and os.path.splitext(val)[1] == ".sf2"),
    "play.defaultDynamic::choices": {
        "pppp",
        "ppp",
        "pp",
        "p",
        "mp",
        "mf",
        "f",
        "ff",
        "fff",
        "ffff",
    },
    "play.graceDuration::type": (int, float, str),
    "htmlTheme::choices": {"light", "dark"},
    "quant.complexity::choices": {"lowest", "low", "medium", "high", "highest"},
    "quant.syncopMinFraction::type": (str, float, Rational),
    "quant.syncopMaxAsymmetry::type": (str, float, Rational),
    "quant.syncopMaxAsymmetry::range": (1, 99),
    "quant.nestedTuplets::choices": {True, False, None},
    # "quant.gridWeight::type": (float, None),
    # "quant.gridWeight::range": (0, 10),
    "show.pageOrientation::choices": {"portrait", "landscape"},
    "show.pageMarginMillim::range": (0, 1000),
    "show.horizontalSpace::choices": (
        "default",
        "small",
        "medium",
        "large",
        "xlarge",
    ),
    "show.glissLineThickness::choices": (1, 2, 3, 4),
    "show.glissLineType::choices": ("solid", "wavy"),
    "show.jupyterMaxImageWidth::type": int,
    "show.voiceMaxStaves::type": int,
    "show.voiceMaxStaves::range": (1, 4),
    "show.labelStyle": isValidStyle,
    "show.measureLabelStyle": isValidStyle,
    "show.centsTextStyle": isValidStyle,
    "show.rehearsalMarkStyle": isValidStyle,
    "show.centsTextSnap::range": (0, 50),
    "show.centsTextSnap::type": int,
    ".show.dynamicsResetTime::range": (0, 999999999),
    "show.clipNoteheadShape::choices": (
        "square",
        "normal",
        "cross",
        "triangle",
        "rhombus",
        "rectangle",
        "slash",
        "cluster",
    ),
    "show.proportionalDuration": isValidFraction,
    "show.spacing::choices": ("normal", "strict", "uniform"),
    "show.flagStyle::choices": ("normal", "straight", "flat"),
    "show.clefSimplify::range": (0, 10000),
    "show.clefTransposingFactor::range": (0, 1),
    "dynamicCurveShape": lambda cfg, key, val: val.split("(")[0] in ("linear", "expon", "halfcos"),
    "dynamicCurveMindb::range": (-160, 0),
    "dynamicCurveMaxdb::range": (-160, 0),
    "dynamicCurveDynamics": lambda cfg, key, val: set(val.split()).issubset(_dynamicsSet),
    ".quant.divisionWeight": lambda cfg, k, v: v is None or 0 <= v <= 1,
    "quant.gridWeight::range": (0, 10),
    "quant.gridWeight": lambda c, k, v: v is None or 0 <= v <= 10,
    ".quant.complexityWeight": lambda cfg, k, v: v is None or 0 <= v <= 10,
    ".quant.complexityWeight::range": (0, 10),
    "quant.breakBeats::choices": ("none", "all", "weak", "strong"),
    "reprUnicodeAccidentals::choices": (False, True, "full", "simple"),
}

docs = {
    'A4':
        "Freq of A4. Normal values are between 440-443, but any value can be used",

    'fixStringNotenames':
        "Fix pitches given as notenames at the spelling given. "
        "False: respell for better readability within the context. "
        "Pitches as midi or frequency are always respelled",

    "jupyterReprShow":
        "Use html as repr within jupyter. If False, .show needs "
        "to be called explicitely to render notation",

    'openImagesInExternalApp':
        "Force opening images with an external tool, even within Jupyter",

    'dynamicCurveShape':
        "Shape used to convert dynamics to amplitudes. "
        "Normally an exponential curve, given as 'expon(exp)'. "
        "exp < 1 results in more resolution for soft dynamics",

    'dynamicCurveMindb':
        "Amplitude in dB corresponding to the softest dynamic",

    'dynamicCurveMaxdb':
        "Amplitude in dB corresponding to the loudest dynamic",

    'dynamicCurveDynamics':
        "Possible dynamic steps. "
        "A str with all dynamic steps, sorted from soft to loud",

    'semitoneDivisions':
        "Number of steps per semitone for notation (2=quarter-tones, 4=eighth-tones)",

    'reprShowFreq':
        "Show freq when printing events",
    
    'reprUnicodeFractions':
        "Show fractions (for durations/offsets) as unicode glyphs. "
        "Not all fonts have support for this",
    
    'reprDurationAsFraction':
        "Show durations as fractions instead of floats",
    
    'reprUnicodeAccidentals':
        "Use unicode accidentals for representation of notes",

    'show.arpeggiateChord':
        "Display chords as an arpeggio. "
        "In auto mode, only arpeggiate when needed",

    'chordAdjustGain':
        "Scale the gain of a chord by the number of notes. "
        "This prevents clipping (used only if notes don't have an explicit amplitude)",

    'show.scaleFactor':
        "Affects the size of the generated image when using png format",

    'show.scaleFactorMusicxml':
        "Apply a scaling factor to images rendered via musicxml",

    'show.staffSize':
        "Size of a staff, in points",

    'show.format':
        "Format used for rendering an obj as an image",

    'play.gain':
        "Default gain for playback",

    'play.numChannels':
        "Number of playback channels",

    'play.defaultAmplitude':
        "Default amplitude for a Note/Chord, used if play.useDynamics is False",

    'play.defaultDynamic':
        'Dynamic of a Note/Chord, only used if play.useDynamics is True. '
        'Any event with an amplitude uses that value instead',

    'rec.blocking':
        "Should recording be blocking or should be done async?",

    'play.engineName':
        "Name of the play engine used",

    'play.graceDuration':
        'Duration assigned to a gracenote for playback (in quarternotes)',

    'play.soundfontFindPeakAOT':
        'Find the peak of a soundfont to adjust its gain, ahead of time',

    'show.labelStyle':
        'Style used for labels. '
        'A list of key=value pairs, '
        'separated by ;. Keys: fontsize, box '
        '(rectangle, square, circle), placement (above, below), italic, '
        'bold. Example: "fontsize=12; italic; box=square"',

    'show.centsTextStyle':
        'Style used for cents annotations. '
        'A list of key=value pairs, '
        'separated by ;. Keys: fontsize, box '
        '(rectangle, square, circle), placement (above, below), italic, '
        'bold. Example: "fontsize=12; italic; box=square"',

    'show.measureLabelStyle':
        'Style for measure annotations. '
        'A list of key=value pairs separated by ;. Keys: fontsize, box '
        '(rectangle, square, circle), placement (above, below), italic, '
        'bold. Example: '
        '"fontsize=12; italic; box=square"',

    'show.rehearsalMarkStyle':
        'Style for rehearsal marks. '
        'A list of key=value pairs, '
        'separated by ";". Keys: fontsize, box '
        '(rectangle, square, circle), placement (above, below), italic, '
        'bold. Example: '
        '"fontsize=12; italic; box=rectangle"',

    'show.glissLineThickness':
        'Line thikness for glissandi. '
        'The value is abstract, it is up to the renderer to interpret it',

    'show.glissHideTiedNotes':
        'Hide tied notes when part of a gliss.',

    'show.glissLineType':
        'Default line type for glissandi',

    'show.dynamicFromAmplitude':
        'Add dynamics according to the amplitude of events when rendering notation',

    'show.absOffsetWhenDetached':
        'Always use the absolute offset of events, even when shown detached from parent',

    'show.respellPitches':
        "Find best enharmonic spelling within the context",
      
    'show.voiceMaxStaves':
        "Max number of staves per voice",

    'show.clipNoteheadShape':
        "Notehead shape to use for clips",

    'show.referenceStaffsize':
        "Staff size used as reference. Allows to use staff size as a generic "
        "indicator for score scale across backends",
    
    'show.clefSimplify':
        "Simplify automatic clef changes. Use higher values to limit clef changes",

    'show.musicxmlFontScaling':
        "Scaling factor applied to font sizes when rendering to musicxml",

    'show.flagStyle':
        "Flag style, at the moment only valid in lilypond",

    'show.warnIfEmpty':
        "Warn if an object did not produce any scoring parts",

    'show.singleStaffRange':
        "A voice/chord which fits within this range is notated as one staff",

    'show.explodeStaffPenalty':
        "Penalty applied when exploding a voice to multiple staves. "
        "A higher value reduces the number of staves",

    '.enharmonic.debug':
        "Print debug info while applying automatic enharmonic spelling",

    'enharmonic.horizontalWeight':
        "Weight of the horizontal dimension (note sequences) for enharmonic spellign",

    'enharmonic.verticalWeight':
        "Weight of the vertical dimension (notes within a chord) for enharmonic spelling",

    'play.schedLatency':
        'Latency when scheduling events to ensure time precission',

    'rec.verbose':
        'Output debug info when rendering offline',

    'rec.ksmps':
        'Samples per cycle when rendering offline (passed as ksmps to csound)',

    '.rec.compressionBitrate':
        'default bitrate to use when encoding to ogg or mp3',

    'rec.numChannels':
        'Number of channels when rendering offline',

    'rec.extratime':
        'Extratime added for things like reverb tails',

    'play.fade':
        'Default fade time',

    'play.unschedFadeout':
        'Fade out when stopping a note',

    'play.soundfontInterpol':
        'Interpolation used when reading sample data from a soundfont',

    'play.verbose':
        'Output extra debugging information regarding playback',

    'show.backend':
        'Method used when rendering notation',

    'show.cents':
        'Show cents deviation as text when rendering notation',

    'show.centsTextSnap':
        'Notes within this number of cents from a quantized pitch'
        'don´t need a text annotation (see `show.cents`)',

    '.show.centsTextPlusSign':
        'Show a plus sign for possitive cents deviations',

    'show.pageOrientation':
        'Page orientation when rendering',

    'show.pageSize':
        'Page size when rendering',

    'show.glissStemless':
        'Make the end note of a gliss stemless when shown as gracenote',

    'show.glissParenthesis':
        'Parenthesize the destination note of a gliss. when shown as gracenote',

    'show.pageMarginMillim':
        'Page margin in mm',

    'show.lilypondPngStaffsizeScale':
        'Scale staffsize when rendering to png via lilypond',

    'show.lilypondGlissMinLength':
        'Minimum length of a glissando in points. Increase it if gliss lines'
        'are hidden or too short',

    'show.pngResolution':
        'DPI used when rendering to png',

    'show.horizontalSpace':
        'Horizontal spacing hint. It depends on format and backend how this'
        ' value is interpreted',

    'show.jupyterMaxImageWidth':
        'Max width in pixels for images in jupyter',

    'show.hideRedundantDynamics':
        'Hide redundant dynamics within a voice',
    
    '.show.dynamicsResetTime': 
        'When removing redundant dynamics, reset after this number of quarters',

    'play.backend':
        'Backend used for playback',

    'play.useDynamics':
        'Use the dynamic of an event for playback if no explicit amplitude is set',

    'rec.path':
        'Path used to save soundfiles when rendering offline.',

    'show.cacheImages':
        'Cache rendered images. Set it to False for debugging or '
        'call `resetImageCache()` to reset manually',

    'show.arpeggioDuration':
        'Duration for individual notes when showing a chord as arpeggio',

    'rec.sr':
        'Sample rate for offline rendering',

    'play.fadeShape':
        'Curve-shape used for fading in/out',

    'show.centSep':
        'Separator used when displaying multiple cents deviations (in a chord)',

    'play.instr':
        'Default instrument used for playback. A list of available instruments '
        'can be queried via `presetManager.definedPresets()`. ',

    'play.pitchInterpol':
        'Curve shape for interpolating between pitches',

    'play.generalMidiSoundfont':
        'Path to a soundfont (sf2 file) with a general midi mapping',

    'reverbInstr':
        'Default instrument used for reverb',

    'reverbGaindb':
        'Main reverb: gain (dB)',

    'reverbDelayms':
        'Main reverb: pre delay (in milliseconds)',

    'reverbDecay':
        'Main reverb: decay time (in seconds)',

    'reverbDamp':
        'Main reverb: damping (range 0-1)',

    'reverbRestart':
        'Restart main reverb',

    'htmlTheme':
        'Theme used when displaying html inside jupyter',

    'play.soundfontAmpDiv':
        'A divisor used to scale the amplitude of soundfonts to a range 0-1',

    'quant.complexity':
        'Complexity used for notation',

    'quant.nestedTuplets':
        'Allow nested tuplets when quantizing. None: follow '
        'the complexity preset (quant.complexity). '
        'The musescore backend cannot parse nested tuplets from musicxml atm',

    'quant.nestedTupletsMusicxml':
        'Allow nested tuplets for musicxml. '
        'The musescore backend cannot parse nested tuplets properly '
        'from mxml at the moment. Nested tuplets are used for other formats if '
        '"quant.nestedTuplets" = True',

    'quant.breakBeats':
        'Level at which to break syncopations. "all": break all syncopations; '
        '"weak": break syncopations over weak beats; "strong": only break '
        'syncopations at strong beats; "none": don´t break syncopations',

    '.quant.divisionWeight':
        'Weight applied to the penalty of the beat division'
        'Higher values result in simpler subdivisions. None to use the '
        'value set by the complexity preset (quant.complexity)',

    'quant.gridWeight':
        'Weight applied to time deviations during quantization. '
        'Higher values result in more accurate quantization, at the cost of complexity. '
        'None sets this value from the complexity preset (quant.complexity)',

    '.quant.complexityWeight':
        'Weight applied to rhythm complexity during '
        'quantization. A higher value results in simpler rhythms. None sets '
        'this value from the complexity preset (quant.complexity)',

    '.quant.gridErrorExp':
        'Exponent applied to the grid error. The error is a value between 0-1 '
        'showing the grid accuracy for a given quantization '
        '(0=perfect timing). A value between 0-1 makes grid errors weight more. '
        'None to use the value set by the complexity preset (quant.complexity)',

    'quant.syncopMinFraction':
        'Min duration of a syncopation as a ratio of the beat. Any syncopation shorter '
        'is broken and its parts tied, to prevent complex syncopations',

    'quant.syncopPartMinFraction':
        'Min duration of any part of a syncopation, as a fraction of the beat. A syncopation '
        'consistings of two parts, one left and one right to the beat boundary',

    'quant.syncopMaxAsymmetry':
        'Max asymmetry of a syncopation. For notes across beats, this sets'
        ' the max. allowed asymmetry across the beat, '
        'as a ratio longest:shortest part across the beat. '
        'A note exactly across the beat has an asymmetry of 1',

    '.quant.debug':
        'Output extra debug info during quantization, showing how different '
        'divisions are evaluated by the quantizer',

    '.quant.debugShowNumRows':
        'Limit the number of possibilities shown during quantization debugging '
        'is turned',

    'musescorepath':
        'Command to use when calling MuseScore. '
        'For macOS users: it must be an '
        'absolute path pointing to the actual binary inside the .app bundle',

    'lilypondpath':
        'Path to the lilypond binary. '
        'If set, it must be an absolute, existing path. '
        'Only needed if using a specific lilypond installation '
        '(lilypond is auto-installed if not found)',

    'show.autoClefChanges':
        'Add clef changes if needed. '
        'Otherwise, one clef is determined for each part and is not changed along the part.',

    'show.spacing':
        'Horizontal spacing used. '
        '"normal": traditional spacing; "uniform": '
        ' proportional spacing with uniform stretching; "strict": proportional '
        'spacing with strict placement (clef changes and bar lines don´t add spacing and'
        'might overlap)',

    'show.proportionalDuration':
        'Space reolution for proportional duration. The lower this value, the longer '
        'the space taken by each note. This mirrors the value used by lilypond. See also: '
        'https://lilypond.org/doc/v2.23/Documentation/notation/proportional-notation',

    'show.keepClefBias':
        'Bias to keep the current clef. A higher value makes it more likely '
        'to keep the previous clef during automatic clef changes',

    'show.clefChangesWindow':
        'Window size for automatic clef changes (number of elements '
        'per evaluation)',

    'show.clefTransposingFactor':
        'Factor applied to a clef fitness if it is a transposing clef. A value < 1'
        'favors non-transposing clefs.',

    'soundfilePlotHeight':
        'Plot height for soundfiles. Used to set the'
        ' figsize in inline plots within Jupyter.',

}
