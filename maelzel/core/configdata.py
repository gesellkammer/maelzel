import os
import math
from maelzel.textstyle import TextStyle
from maelzel import dynamiccurve
from numbers import Rational


defaultdict = {
    'A4': 442,
    'splitAcceptableDeviation': 4,
    'chordAdjustGain': True,
    'reprShowFreq': False,
    'semitoneDivisions': 4,
    'musescorepath': '',
    'lilypondpath': '',
    '.reprShowFractionsAsFloat': True,
    'jupyterHtmlRepr': True,
    'fixStringNotenames': False,
    'openImagesInExternalApp': False,
    'enharmonic.horizontalWeight': 1,
    'enharmonic.verticalWeight': 0.01,
    '.enharmonic.debug': False,
    '.enharmonic.threeQuarterMicrotonePenalty': 20,

    'show.arpeggiateChord': 'auto',
    'show.centsAnnotationStyle': 'fontsize=6; placement=below',
    'show.centsDeviationAsTextAnnotation': True,
    'show.centsAnnotationSnap': 2,
    '.show.centsAnnotationPlusSign': True,
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
    'show.pageMarginMillimeters': 4,
    'show.glissEndStemless': False,
    'show.glissHideTiedNotes': True,
    'show.glissLineThickness': 2,
    'show.glissLineType': 'solid',
    'show.lilypondPngStaffsizeScale': 1.5,
    'show.lilypondGlissandoMinimumLength': 5,
    'show.pngResolution': 200,
    'show.measureAnnotationStyle': 'box=rectangle; fontsize=12',
    'show.rehearsalMarkStyle': 'box=rectangle; fontsize=13; bold',
    'show.respellPitches': True,
    'show.horizontalSpacing': 'medium',
    'show.fillDynamicFromAmplitude': False,
    'show.jupyterMaxImageWidth': 1000,
    'show.hideRedundantDynamics': True,
    'show.asoluteOffsetForDetachedObjects': False,
    'show.voiceMaxStaves': 1,
    'show.clipNoteheadShape': 'square',
    'show.referenceStaffsize': 12.0,
    'show.musicxmlFontScaling': 1.0,
    'show.autoClefChanges': True,
    '.show.autoClefChangesWindow': 1,
    '.show.keepClefBiasFactor': 2.0,

    'play.gain': 1.0,
    'play.engineName': 'maelzel.core',
    'play.instr': 'sin',
    'play.fade': 0.02,
    'play.fadeShape': 'cos',
    'play.pitchInterpolation': 'linear',
    'play.numChannels': 2,
    'play.unschedFadeout': 0.05,
    'play.backend': 'default',
    'play.defaultAmplitude': 1.0,
    'play.defaultDynamic': 'f',
    'play.generalMidiSoundfont': '',
    'play.soundfontAmpDiv': 16384,
    'play.soundfontInterpolation': 'linear',
    'play.schedLatency': 0.05,
    'play.verbose': False,
    'play.useDynamics': True,
    'play.waitAfterStart': 0.5,
    'play.gracenoteDuration': '1/14',

    'rec.blocking': True,
    'rec.sr': 44100,
    'rec.ksmps': 64,
    'rec.numChannels': 2,
    'rec.path': '',
    'rec.verbose': False,
    '.rec.compressionBitrate': 224,
    'rec.extratime': 0.,

    'htmlTheme': 'light',

    'quant.syncopationMinBeatFraction': 1/3,
    'quant.syncopationMaxAsymmetry': 2,
    'quant.nestedTuplets': None,
    'quant.nestedTupletsInMusicxml': False,
    'quant.breakSyncopationsLevel': 'weak',
    'quant.complexity': 'high',
    '.quant.divisionErrorWeight': None,
    '.quant.gridErrorWeight': None,
    '.quant.rhythmComplexityWeight': None,
    '.quant.gridErrorExp': None,
    '.quant.debug': False,
    '.quant.debugShowNumRows': 50,

    'dynamicCurveShape': 'expon(0.3)',
    'dynamicCurveMindb': -60,
    'dynamicCurveMaxdb': 0,
    'dynamicCurveDynamics': 'ppp pp p mp mf f ff fff',
}


validator = {
    'lilypondpath': lambda cfg, key, val: val == '' or os.path.exists(val),
    'A4::type': int,
    'A4::range': (10, 10000),
    'play.backend::choices': {'default', 'jack', 'pulse', 'alsa', 'pa_cb',
                              'auhal', 'portaudio'},
    'semitoneDivisions::choices': {1, 2, 4},
    'show.backend::choices': {'musicxml', 'lilypond'},
    'show.format::choices': {'png', 'pdf', 'repr'},
    'show.staffSize::type': float,
    'show.pngResolution::choices': {100, 200, 300, 600, 1200},
    'show.pageSize::choices': {'a3', 'a4', 'a2'},
    'show.arpeggiateChord::choices': {'auto', True, False},
    'play.gain::range': (0, 1),
    'play.fadeShape::choices': {'linear', 'cos', 'scurve'},
    'play.numChannels::type': int,
    'play.numChannels::range': (1, 128),
    'rec.numChannels::range': (1, 128),
    'play.soundfontInterpolation::choices': {'linear', 'cubic'},
    'rec.sr::choices': {44100, 88200, 176400, 352800, 48000, 96000, 144000, 192000, 384000},
    '.rec.compressionBitrate::coices': {64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 500},
    'rec.ksmps::choices': {1, 16, 32, 64, 128, 256},
    'rec.extratime::range': (0., math.inf),
    'play.defaultAmplitude::range': (0, 1),
    'play.pitchInterpolation::choices': {'linear', 'cos'},
    'play.generalMidiSoundfont': lambda cfg, key, val: val == '' or (os.path.exists(val) and os.path.splitext(val)[1] == '.sf2'),
    'play.defaultDynamic::choices': {'pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff'},
    'play.gracenoteDuration::type': (int, float, str),
    'htmlTheme::choices': {'light', 'dark'},
    'quant.complexity::choices': {'lowest', 'low', 'medium', 'high', 'highest'},
    'quant.syncopationMinBeatFraction::type': (float, Rational),
    'quant.syncopationMaxAsymmetry::type': (float, Rational),
    'quant.nestedTuplets::choices': {True, False, None},
    'show.pageOrientation::choices': {'portrait', 'landscape'},
    'show.pageMarginMillimeters::range': (0, 1000),
    'show.horizontalSpacing::choices': ('default', 'small', 'medium', 'large', 'xlarge'),
    'show.glissLineThickness::choices': (1, 2, 3, 4),
    'show.glissLineType::choices': ('solid', 'wavy'),
    'show.jupyterMaxImageWidth::type': int,
    'show.voiceMaxStaves::type': int,
    'show.voiceMaxStaves::range': (1, 4),
    'show.measureAnnotationStyle': lambda cfg, key, val: TextStyle.validate(val),
    'show.centsAnnotationStyle': lambda cfg, key, val: TextStyle.validate(val),
    'show.centsAnnotationSnap::range': (0, 50),
    'show.centsAnnotationSnap::type': int,
    'show.rehearsalMarkStyle': lambda cfg, key, val: TextStyle.validate(val),
    'show.clipNoteheadShape::choices': ('', 'square', 'normal', 'cross', 'harmonic', 'triangle',
                                        'xcircle', 'rhombus', 'rectangle', 'slash', 'diamond',
                                        'cluster'),
    'dynamicCurveShape': lambda cfg, key, val: val.split("(")[0] in ('linear', 'expon', 'halfcos'),
    'dynamicCurveMindb::range': (-160, 0),
    'dynamicCurveMaxdb::range': (-160, 0),
    'dynamicCurveDynamics': lambda cfg, key, val: all(d in dynamiccurve.dynamicSteps
                                                      for d in val.split()),
    '.quant.divisionErrorWeight': lambda cfg, k, v: v is None or 0 <= v <= 1,
    '.quant.gridErrorWeight': lambda cfg, k, v: v is None or 0 <= v <= 1,
    '.quant.rhythmComplexityWeight': lambda cfg, k, v: v is None or 0 <= v <= 1,
    'quant.breakSyncopationsLevel::choices': ('none', 'all', 'weak', 'strong')

}

docs = {
    'A4':
        "Freq. of the Kammerton A4. Normal values are 440, 442, 443 or 432 for old tuning, "
        "but any 'fantasy' value can be used",

    'fixStringNotenames':
        "If True, pitches given as string notenames are fixed at the spelling given at "
        "creation. Otherwise pitches might be respelled to match their context for better "
        "readability. Pitches given as midi notes or frequencies are always respelled",

    '.reprShowFractionsAsFloat':
        "All time offsets and durations are kept as rational numbers to avoid rounding errors. "
        "If this option is True, these fractions are printed as floats in order to make them "
        "more readable. ",

    "jupyterHtmlRepr":
        "If True, output html inside jupyter as part of the _repr_html_ hook. Under "
        "certain circumstances (for example, when generating documentation from a notebook) "
        "this html might result in style conflict. Setting in False will just output "
        "plain text",

    'openImagesInExternalApp':
        "Force opening images with an external tool, even when inside a Jupyter "
        "notebook",

    'dynamicCurveShape':
        "The shape used to create the default dynamics curve. The most convenient shape is some"
        " variation of an exponential, given as expon(exp), where exp is the exponential used. "
        "exp < 1 will result in more resolution for soft dynamics",

    'dynamicCurveMindb':
        "The amplitude (in dB) corresponding to the softest dynamic",

    'dynamicCurveMaxdb':
        "The amplitude (in dB) corresponding to the loudest dynamic",

    'dynamicCurveDynamics':
        "Possible dynamic steps. A string with all dynamic steps, sorted from softest to loudest",

    'semitoneDivisions':
        "The number of divisions per semitone (2=quarter-tones, 4=eighth-tones)",

    'reprShowFreq':
        "Show frequency when printing a Note in the console",

    'show.arpeggiateChord':
        "Arpeggiate notes of a chord when showing. In auto mode, only arpeggiate"
        " when needed",

    'chordAdjustGain':
        "Adjust the gain of a chord according to the number of notes, to prevent "
        "clipping",

    'show.scaleFactor':
        "Affects the size of the generated image when using png format",

    'show.scaleFactorMusicxml':
        "Apply a scaling factor to images rendered via musicxml",

    'show.staffSize':
        "The size of a staff, in points",

    'show.format':
        "Used when no explicit format is passed to .show",

    'play.gain':
        "Default gain used when playing/recording",

    'play.numChannels':
        "Default number of channels (channels can be set explicitely when calling"
        " startPlayEngine",

    'play.defaultAmplitude':
        "The amplitude of a Note/Chord when an amplitude is needed and the object "
        "has an undefined amplitude. This is only used if play.useDynamics if False",

    'play.defaultDynamic':
        'THe dynamic of a Note/Chord when a dynamic is needed. This is only used if '
        'play.useDynamics is True. Any event with an amplitude will use that amplitude instead',

    'rec.blocking':
        "Should recording be blocking or should be done async?",

    'play.engineName':
        "Name of the play engine used",

    'play.waitAfterStart':
        'How much to wait for the sound engine to be operational after starting it',

    'play.gracenoteDuration':
        'Duration assigned to a gracenote for playback (in quarternotes)',

    'show.labelStyle':
        'Text size used for labels'
        'The format is a list of <key>=<value> pairs, '
        'separated by semicolons. Possible keys are: fontsize, box (choices: '
        'rectangle, square, circle), placement (choices: above, below), italic (flag), '
        'bold (flag). Flag keys do not need any values. Example: '
        '"fontsize=12; italic; box=rectangle"',

    'show.centsAnnotationStyle':
        'Style used for cents annotations. '
        'The format is a list of <key>=<value> pairs, '
        'separated by semicolons. Possible keys are: fontsize, box (choices: '
        'rectangle, square, circle), placement (choices: above, below), italic (flag), '
        'bold (flag). Flag keys do not need any values. Example: '
        '"fontsize=12; italic; box=rectangle"',

    'show.measureAnnotationStyle':
        'Style for measure annotations. '
        'The format is a list of <key>=<value> pairs, '
        'separated by semicolons. Possible keys are: fontsize, box (choices: '
        'rectangle, square, circle), placement (choices: above, below), italic (flag), '
        'bold (flag). Flag keys do not need any values. Example: '
        '"fontsize=12; italic; box=rectangle"',

    'show.rehearsalMarkStyle':
        'Style for rehearsal marks. '
        'The format is a list of <key>=<value> pairs, '
        'separated by semicolons. Possible keys are: fontsize, box (choices: '
        'rectangle, square, circle), placement (choices: above, below), italic (flag), '
        'bold (flag). Flag keys do not need any values. Example: '
        '"fontsize=12; italic; box=rectangle"',

    'show.glissLineThickness':
        'Line thikness when rendering glissandi. The value is abstract and it is'
        'up to the renderer to interpret it',

    'show.glissHideTiedNotes':
        'Hide tied notes which are part of a glissando',

    'show.glissLineType':
        'Default line type for glissandi',

    'show.fillDynamicFromAmplitude':
        'If True, when rendering notation, if an object has an amplitude '
        'and does not have an explicit dynamic, add a dynamic according to the amplitude',

    'show.asoluteOffsetForDetachedObjects':
        'When showing an object which has a parent but is shown detached from it, should'
        'the absolute offset be used?',

    'show.respellPitches':
        "If True, try to find a suitable enharmonic representation of pitches which"
        "have not been fixed already by the user. Otherwise the canonical form of each"
        "pitch is used, independent of the context",

    'show.voiceMaxStaves':
        "The maximum number of staves per voice when showing a Voice as notation. A voice"
        "is a sequence of non-simultaneous events (notes, chords, etc.) but these can"
        "be exploded over multiple staves (for example, a chord might expand across a"
        "wide range and would need multiple extra lines in any clef",

    'show.clipNoteheadShape':
        "Notehead shape to use for clips",

    'show.referenceStaffsize':
        "Staff size used as a reference to convert between staff size and scaling factor. "
        "This allows to use staff size as a general way to indicate the scale of a score, "
        "independent of the backend",

    'show.musicxmlFontScaling':
        "A scaling factor applied to font sizes when rendering to musicxml",

    '.enharmonic.debug':
        "If True, print debug information while calculating automatic enharmonic spelling",

    'enharmonic.horizontalWeight':
        "The weight of the horizontal dimension (note sequences) when evaluating an "
        "enharmonic variant",

    'enharmonic.verticalWeight':
        "The weight of the vertical dimension (chords within a voice) when evaluating "
        "an enharmonic variant",

    'splitAcceptableDeviation':
        'When splitting notes between staves, notes within this range of the '
        'split point will be grouped together if they all fit',

    'play.schedLatency':
        'Added latency when scheduling events to ensure time precission',

    'rec.verbose':
        'Show debug output when calling csound as a subprocess',

    'rec.ksmps':
        'Samples per cycle when rendering offline (passed as ksmps to csound)',

    '.rec.compressionBitrate':
        'default bitrate to use when encoding to ogg or mp3',

    'rec.numChannels':
        'The default number of channels when rendering to disk',

    'rec.extratime':
        'Default extratime added when recording',

    'play.fade':
        'Default fade time',

    'play.unschedFadeout':
        'Fade out when stopping a note',

    'play.soundfontInterpolation':
        'Interpolation used when reading sample data from a soundfont.',

    'play.verbose':
        'If True, outputs extra debugging information regarding playback',

    'show.backend':
        'Method used when rendering notation',

    'show.centsDeviationAsTextAnnotation':
        'Show cents deviation as text when rendering notation',

    'show.centsAnnotationSnap':
        'Pitches which deviate less than this cents from a quantized pitch'
        'do not need a text annotation (see `show.centsDeviationAsTextAnnotation`)',

    '.show.centsAnnotationPlusSign':
        'Show a plus sign for possitive cents deviations',

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

    'show.lilypondGlissandoMinimumLength':
        'The minimum length of a glissando in points. Increase this value if glissando lines'
        'are not shown or are too short (this might be the case within the context of dotted'
        'notes or accidentals)',

    'show.pngResolution':
        'DPI used when rendering to png',

    'show.horizontalSpacing':
        'Hint for the renderer to adjust horizontal spacing. The actual result depends '
        'on the backend and the format used.',

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

    'play.generalMidiSoundfont':
        'Path to a soundfont (sf2 file) with a general midi mapping',

    'htmlTheme':
        'Theme used when displaying html inside jupyter',

    'play.soundfontAmpDiv':
        'A divisor used to scale the amplitude of soundfonts to a range 0-1',

    'quant.complexity':
        'Controls the allowed complexity in the notation. The higher the complexity, '
        'the more accurate the quantization, at the cost of a more complex notation. ',

    'quant.nestedTuplets':
        'Are nested tuples allowed when quantizing? Not all display backends support '
        'nested tuples (musescore, used to render musicxml '
        'has no support for nested tuples). If None, this flag is determined based on '
        'the complexity preset (quant.complexity)',

    'quant.nestedTupletsInMusicxml':
        'If False, nested tuplets default to False when rendering to musicxml. This '
        'is because some musicxml renderers (MuseScore, for example) do not render '
        'nested tuplets properly. Nested tuplets will still be enabled if the config '
        'options "quant.nestedTuplets" is explicitely set to True.',

    'quant.breakSyncopationsLevel':
        'Level at which to break syncopations, one of "all" (break all syncopations), '
        '"weak (break only syncopations over secondary beats)", "strong" (break '
        'syncopations at strong beats) or "none" (do not break any syncopations)',

    '.quant.divisionErrorWeight':
        'A weight (between 0 and 1) applied to the penalty of complex quantization of '
        'the beat. The higher this value is, the simpler the subdivision chosen. '
        'If set to None, this value is derived from the complexity preset '
        '(quant.complexity)',

    '.quant.gridErrorWeight':
        'A weight (between 0 and 1) applied to the deviation of a quantization to the '
        'actual attack times and durations during quantization. The higher this value, '
        'the more accurate the quantization (possibly resulting in more complex '
        'subdivisions of the beat). If None, the value is derived from the complexity '
        'preset (quant.complexity)',

    '.quant.rhythmComplexityWeight':
        'A weight (between 0 and 1) applied to the penalty calculated from the '
        'complexity of the rhythm during quantization. A higher value results in '
        'more complex rhythms being considered for quantization. If None, the value '
        'is derived from the complexity (quant.complexity)',

    '.quant.gridErrorExp':
        'An exponent applied to the grid error. The grid error is a value between 0-1 '
        'which indicates how accurate the grid representation is for a given quantization '
        '(a value of 0 indicates perfect timing). An exponent betwenn 0 < exp <= 1 will '
        'make grid errors weight more dramatically as they diverge from the most accurate '
        'solution. If None, the value is derived from the complexity setting (quant.complexity)',

    'quant.syncopationMinBeatFraction':
        'when merging durations across beats, a merged duration cannot be smaller than this '
        'duration. This is to prevent joining durations across beats which might result in '
        'high rhythmic complexity',

    'quant.syncopationMaxAsymmetry':
        'Max. asymmetry of a syncopation. If a note is placed across a beat, this indicates'
        ' the max. allowed asymettry of such note in respect to the beat. The asymmetry is '
        'calculated as the ratio between the longest and the shortest part of a note placed '
        'across the beat. If the note is placed exactly across the beat, the asymmetrx is 1. '
        'Together with quant.syncopationMinBeatFraction this can be used to control which '
        'kind of syncopations are allowed',

    '.quant.debug':
        'Turns on debugging for the quantization process. This will show how different '
        'divisions of the beat are being evaluated by the quantizer in terms of what '
        'is contributing more to the ranking. With this information it is possible '
        'to adjust the weights (quant.rhythmCompleityWeight, .quant.divisionErrorWeight, '
        'etc)',

    '.quant.debugShowNumRows':
        'When quantization debugging is turned on this setting limits the number '
        'of different quantization possibilities shown',

    'musescorepath':
        'The command to use when calling MuseScore. For macOS users: it must be an '
        'absolute path pointing to the actual binary inside the .app bundle',

    'lilypondpath':
        'The path to the lilypond binary. It must be an absolute, existing path',

    'show.autoClefChanges':
        'If True, add clef changes to a quantized part if needed. Otherwise, one clef '
        'is determined for each part and is not changed along the part.',

    '.show.keepClefBiasFactor':
        'The higher this value, the more priority is  given to keeping the previous '
        'clef during automatic clef changes',

    '.show.autoClefChangesWindow':
        'When adding automatic clef changes, use this window size (number of elements '
        'per evaluation)'


}
