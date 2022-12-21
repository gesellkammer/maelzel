.. _config_a4:

A4:
    | Default: **442**  -- ``int``
    | Between 10 - 10000
    | *Freq. of the Kammerton A4. Normal values are 440, 442, 443 or 432 for old tuning, but any 'fantasy' value can be used*

.. _config_splitacceptabledeviation:

splitAcceptableDeviation:
    | Default: **4**  -- ``int``
    | *When splitting notes between staves, notes within this range of the split point will be grouped together if they all fit*

.. _config_chordadjustgain:

chordAdjustGain:
    | Default: **True**  -- ``bool``
    | *Adjust the gain of a chord according to the number of notes, to prevent clipping*

.. _config_reprshowfreq:

reprShowFreq:
    | Default: **False**  -- ``bool``
    | *Show frequency when printing a Note in the console*

.. _config_semitonedivisions:

semitoneDivisions:
    | Default: **4**  -- ``int``
    | Choices: ``1, 2, 4``
    | *The number of divisions per semitone (2=quarter-tones, 4=eighth-tones)*

.. _config_musescorepath:

musescorepath:
    | Default: **''**  -- ``str``
    | *The command to use when calling MuseScore. For macOS users: it must be an absolute path pointing to the actual binary inside the .app bundle*

.. _config_reprshowfractionsasfloat:

reprShowFractionsAsFloat:
    | Default: **True**  -- ``bool``
    | *All time offsets and durations are kept as rational numbers to avoid rounding errors. If this option is True, these fractions are printed as floats in order to make them more readable. *

.. _config_fixstringnotenames:

fixStringNotenames:
    | Default: **False**  -- ``bool``
    | *If True, pitches given as string notenames are fixed at the spelling given at creation. Otherwise pitches might be respelled to match their context for better readability. Pitches given as midi notes or frequencies are always respelled*

.. _config_openimagesinexternalapp:

openImagesInExternalApp:
    | Default: **False**  -- ``bool``
    | *Force opening images with an external tool, even when inside a Jupyter notebook*

.. _config_enharmonicspellinghorizontalweight:

enharmonicSpellingHorizontalWeight:
    | Default: **1**  -- ``int``
    | *The weight of the horizontal dimension (note sequences) when evaluating an enharmonic variant*

.. _config_enharmonicspellingverticalweight:

enharmonicSpellingVerticalWeight:
    | Default: **0.01**  -- ``float``
    | *The weight of the vertical dimension (chords within a voice) when evaluating an enharmonic variant*

.. _config_enharmonicspellingdebug:

enharmonicSpellingDebug:
    | Default: **False**  -- ``bool``
    | *If True, print debug information while calculating automatic enharmonic spelling*

.. _config_show_arpeggiatechord:

show.arpeggiateChord:
    | Default: **auto**  -- ``(str, bool)``
    | Choices: ``auto, False, True``
    | *Arpeggiate notes of a chord when showing. In auto mode, only arpeggiate when needed*

.. _config_show_lastbreakpointdur:

show.lastBreakpointDur:
    | Default: **0.125**  -- ``float``
    | Between 0.015625 - 1
    | *Dur of a note representing the end of a line/gliss, which has no duration per se*

.. _config_show_centsdeviationastextannotation:

show.centsDeviationAsTextAnnotation:
    | Default: **True**  -- ``bool``
    | *show cents deviation as text when rendering notation*

.. _config_show_centsannotationfontsize:

show.centsAnnotationFontSize:
    | Default: **8**  -- ``int``
    | *Font size used for cents annotations*

.. _config_show_centsep:

show.centSep:
    | Default: **,**  -- ``str``
    | *Separator used when displaying multiple cents deviation (in a chord)*

.. _config_show_scalefactor:

show.scaleFactor:
    | Default: **1.0**  -- ``float``
    | *Affects the size of the generated image when using png format*

.. _config_show_staffsize:

show.staffSize:
    | Default: **12.0**  -- ``float``
    | *The size of a staff, in points*

.. _config_show_backend:

show.backend:
    | Default: **lilypond**  -- ``str``
    | Choices: ``lilypond, music21``
    | *method/backend used when rendering notation*

.. _config_show_format:

show.format:
    | Default: **png**  -- ``str``
    | Choices: ``pdf, png, repr``
    | *Used when no explicit format is passed to .show*

.. _config_show_cacheimages:

show.cacheImages:
    | Default: **True**  -- ``bool``
    | *If True, cache rendered images. Set it to False for debugging. call `resetImageCache()` to reset manually*

.. _config_show_arpeggioduration:

show.arpeggioDuration:
    | Default: **0.5**  -- ``float``
    | *Duration used for individual notes when rendering a chord as arpeggio*

.. _config_show_labelfontsize:

show.labelFontSize:
    | Default: **10.0**  -- ``float``
    | *Font size to use for labels*

.. _config_show_pageorientation:

show.pageOrientation:
    | Default: **portrait**  -- ``str``
    | Choices: ``landscape, portrait``
    | *Page orientation when rendering to pdf*

.. _config_show_pagesize:

show.pageSize:
    | Default: **a4**  -- ``str``
    | Choices: ``a2, a3, a4``
    | *The page size when rendering to pdf*

.. _config_show_pagemarginmillimeters:

show.pageMarginMillimeters:
    | Default: **4**  -- ``int``
    | Between 0 - 1000
    | *The page margin in mm*

.. _config_show_glissendstemless:

show.glissEndStemless:
    | Default: **False**  -- ``bool``
    | *When the end pitch of a gliss. is shown as gracenote, make this stemless*

.. _config_show_glisshidetiednotes:

show.glissHideTiedNotes:
    | Default: **True**  -- ``bool``
    | *Hide tied notes which are part of a glissando*

.. _config_show_glisslinethickness:

show.glissLineThickness:
    | Default: **2**  -- ``int``
    | Choices: ``1, 2, 3, 4``
    | *Line thikness when rendering glissandi. The value is abstract and it isup to the renderer to interpret it*

.. _config_show_lilypondpngstaffsizescale:

show.lilypondPngStaffsizeScale:
    | Default: **1.5**  -- ``float``
    | *A factor applied to the staffsize when rendering to png via lilypond. Useful if rendered images appear too small in a jupyter notebook*

.. _config_show_pngresolution:

show.pngResolution:
    | Default: **200**  -- ``int``
    | Choices: ``100, 200, 300, 600, 1200``
    | *DPI used when rendering to png*

.. _config_show_measureannotationfontsize:

show.measureAnnotationFontSize:
    | Default: **12**  -- ``int``
    | *Font size used for measure annotations*

.. _config_show_respellpitches:

show.respellPitches:
    | Default: **True**  -- ``bool``
    | *If True, try to find a suitable enharmonic representation of pitches whichhave not been fixed already by the user. Otherwise the canonical form of eachpitch is used, independent of the context*

.. _config_show_horizontalspacing:

show.horizontalSpacing:
    | Default: **medium**  -- ``str``
    | Choices: ``default, large, medium, small, xlarge``
    | *Hint for the renderer to adjust horizontal spacing. The actual result depends on the backend and the format used.*

.. _config_show_filldynamicfromamplitude:

show.fillDynamicFromAmplitude:
    | Default: **False**  -- ``bool``
    | *If True, when rendering notation, if an object has an amplitude and does not have an explicit dynamic, add a dynamic according to the amplitude*

.. _config_show_jupytermaximagewidth:

show.jupyterMaxImageWidth:
    | Default: **1000**  -- ``int``
    | *A max. width in pixels for images displayed in a jupyter notebook*

.. _config_show_hideredundantdynamics:

show.hideRedundantDynamics:
    | Default: **True**  -- ``bool``
    | *Hide redundant dynamics within a voice*

.. _config_show_asoluteoffsetfordetachedobjects:

show.asoluteOffsetForDetachedObjects:
    | Default: **False**  -- ``bool``
    | *When showing an object which has a parent but is shown detached from it, shouldthe absolute offset be used?*

.. _config_play_gain:

play.gain:
    | Default: **1.0**  -- ``float``
    | Between 0 - 1
    | *Default gain used when playing/recording*

.. _config_play_enginename:

play.engineName:
    | Default: **maelzel.core**  -- ``str``
    | *Name of the play engine used*

.. _config_play_instr:

play.instr:
    | Default: **sin**  -- ``str``
    | *Default instrument used for playback. A list of available instruments can be queried via `availableInstrs`. New instrument presets can be defined via `defPreset`*

.. _config_play_fade:

play.fade:
    | Default: **0.02**  -- ``float``
    | *default fade time*

.. _config_play_fadeshape:

play.fadeShape:
    | Default: **cos**  -- ``str``
    | Choices: ``cos, linear, scurve``
    | *Curve-shape used for fading in/out*

.. _config_play_pitchinterpolation:

play.pitchInterpolation:
    | Default: **linear**  -- ``str``
    | Choices: ``cos, linear``
    | *Curve shape for interpolating between pitches*

.. _config_play_numchannels:

play.numChannels:
    | Default: **2**  -- ``int``
    | Between 1 - 128
    | *Default number of channels (channels can be set explicitely when calling startPlayEngine*

.. _config_play_unschedfadeout:

play.unschedFadeout:
    | Default: **0.05**  -- ``float``
    | *fade out when stopping a note*

.. _config_play_backend:

play.backend:
    | Default: **default**  -- ``str``
    | Choices: ``alsa, auhal, default, jack, pa_cb, portaudio, pulse``
    | *backend used for playback*

.. _config_play_presetspath:

play.presetsPath:
    | Default: **''**  -- ``str``
    | *The path were presets are saved*

.. _config_play_defaultamplitude:

play.defaultAmplitude:
    | Default: **1.0**  -- ``float``
    | Between 0 - 1
    | *The amplitude of a Note/Chord when an amplitude is needed and the object has an undefined amplitude. This is only used if play.useDynamics if False*

.. _config_play_defaultdynamic:

play.defaultDynamic:
    | Default: **f**  -- ``str``
    | Choices: ``f, ff, fff, ffff, mf, mp, p, pp, ppp, pppp``
    | *THe dynamic of a Note/Chord when a dynamic is needed. This is only used if play.useDynamics is True. Any event with an amplitude will use that amplitude instead*

.. _config_play_generalmidisoundfont:

play.generalMidiSoundfont:
    | Default: **''**  -- ``str``
    | *Path to a soundfont (sf2 file) with a general midi mapping*

.. _config_play_soundfontampdiv:

play.soundfontAmpDiv:
    | Default: **16384**  -- ``int``
    | *A divisor used to scale the amplitude of soundfonts to a range 0-1*

.. _config_play_soundfontinterpolation:

play.soundfontInterpolation:
    | Default: **linear**  -- ``str``
    | Choices: ``cubic, linear``
    | *Interpolation used when reading sample data from a soundfont.*

.. _config_play_schedlatency:

play.schedLatency:
    | Default: **0.05**  -- ``float``
    | *Added latency when scheduling events to ensure time precission*

.. _config_play_verbose:

play.verbose:
    | Default: **False**  -- ``bool``
    | *If True, outputs extra debugging information regarding playback*

.. _config_play_usedynamics:

play.useDynamics:
    | Default: **True**  -- ``bool``
    | *If True, any note/chord with a set dynamic will use that to modify its playback amplitude if no explicit amplitude is set*

.. _config_play_waitafterstart:

play.waitAfterStart:
    | Default: **0.5**  -- ``float``
    | *How much to wait for the sound engine to be operational after starting it*

.. _config_rec_blocking:

rec.blocking:
    | Default: **True**  -- ``bool``
    | *Should recording be blocking or should be done async?*

.. _config_rec_sr:

rec.sr:
    | Default: **44100**  -- ``int``
    | Choices: ``44100, 48000, 88200, 96000, 144000, 176400, 192000, 352800, 384000``
    | *Sample rate used when rendering offline*

.. _config_rec_ksmps:

rec.ksmps:
    | Default: **64**  -- ``int``
    | Choices: ``1, 16, 32, 64, 128, 256``
    | *Samples per cycle when rendering offline (passed as ksmps to csound)*

.. _config_rec_numchannels:

rec.numChannels:
    | Default: **2**  -- ``int``
    | Between 1 - 128
    | *The default number of channels when rendering to disk*

.. _config_rec_path:

rec.path:
    | Default: **''**  -- ``str``
    | *path used to save output files when rendering offline. If not given the default can be queried via `recordPath`*

.. _config_rec_quiet:

rec.quiet:
    | Default: **True**  -- ``bool``
    | *Supress debug output when calling csound as a subprocess*

.. _config_rec_compressionbitrate:

rec.compressionBitrate:
    | Default: **224**  -- ``int``
    | *default bitrate to use when encoding to ogg or mp3*

.. _config_htmltheme:

htmlTheme:
    | Default: **light**  -- ``str``
    | Choices: ``dark, light``
    | *Theme used when displaying html inside jupyter*

.. _config_quant_minbeatfractionacrossbeats:

quant.minBeatFractionAcrossBeats:
    | Default: **1.0**  -- ``float``
    | *when merging durations across beats, a mergef duration cannot be smaller than this duration. This is to prevent joining durations across beats which might result in high rhythmic complexity*

.. _config_quant_nestedtuplets:

quant.nestedTuplets:
    | Default: **None**  -- ``(NoneType, bool)``
    | Choices: ``False, None, True``
    | *Are nested tuples allowed when quantizing? Not all display backends support nested tuples (musescore, used to render musicxml has no support for nested tuples). If None, this flag is determined based on the complexity preset (quant.complexity)*

.. _config_quant_complexity:

quant.complexity:
    | Default: **high**  -- ``str``
    | Choices: ``high, highest, low, medium``
    | *Controls the allowed complexity in the notation. The higher the complexity, the more accurate the quantization, at the cost of a more complex notation. *

.. _config_quant_divisionerrorweight:

quant.divisionErrorWeight:
    | Default: **None**  -- ``NoneType``
    | *A weight (between 0 and 1) applied to the penalty of complex quantization of the beat. The higher this value is, the simpler the subdivision chosen. If set to None, this value is derived from the complexity preset (quant.complexity)*

.. _config_quant_griderrorweight:

quant.gridErrorWeight:
    | Default: **None**  -- ``NoneType``
    | *A weight (between 0 and 1) applied to the deviation of a quantization to the actual attack times and durations during quantization. The higher this value, the more accurate the quantization (possibly resulting in more complex subdivisions of the beat). If None, the value is derived from the complexity preset (quant.complexity)*

.. _config_quant_rhythmcomplexityweight:

quant.rhythmComplexityWeight:
    | Default: **None**  -- ``NoneType``
    | *A weight (between 0 and 1) applied to the penalty calculated from the complexity of the rhythm during quantization. A higher value results in more complex rhythms being considered for quantization. If None, the value is derived from the complexity (quant.complexity)*

.. _config_quant_griderrorexp:

quant.gridErrorExp:
    | Default: **None**  -- ``NoneType``
    | *An exponent applied to the grid error. The grid error is a value between 0-1 which indicates how accurate the grid representation is for a given quantization (a value of 0 indicates perfect timing). An exponent betwenn 0 < exp <= 1 will make grid errors weight more dramatically as they diverge from the most accurate solution. If None, the value is derived from the complexity setting (quant.complexity)*

.. _config_quant_debug:

quant.debug:
    | Default: **False**  -- ``bool``
    | *Turns on debugging for the quantization process. This will show how different divisions of the beat are being evaluated by the quantizer in terms of what is contributing more to the ranking. With this information it is possible to adjust the weights (quant.rhythmCompleityWeight, quant.divisionErrorWeight, etc)*

.. _config_quant_debugshownumrows:

quant.debugShowNumRows:
    | Default: **50**  -- ``int``
    | *When quantization debugging is turned on this setting limits the number of different quantization possibilities shown*

.. _config_dynamiccurveshape:

dynamicCurveShape:
    | Default: **expon(0.3)**  -- ``str``
    | *The shape used to create the default dynamics curve. The most convenient shape is some variation of an exponential, given as expon(exp), where exp is the exponential used. exp < 1 will result in more resolution for soft dynamics*

.. _config_dynamiccurvemindb:

dynamicCurveMindb:
    | Default: **-60**  -- ``int``
    | Between -160 - 0
    | *The amplitude (in dB) corresponding to the softest dynamic*

.. _config_dynamiccurvemaxdb:

dynamicCurveMaxdb:
    | Default: **0**  -- ``int``
    | Between -160 - 0
    | *The amplitude (in dB) corresponding to the loudest dynamic*

.. _config_dynamiccurvedynamics:

dynamicCurveDynamics:
    | Default: **ppp pp p mp mf f ff fff**  -- ``str``
    | *Possible dynamic steps. A string with all dynamic steps, sorted from softest to loudest*
