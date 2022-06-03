maelzel:core
------------

.. _config_a4:

A4:
    | Default: **442**  -- `int`
    | Between 10 - 10000
    | *Freq. of the Kammerton A4. Normal values are 440, 442, 443 or 432 for old tuning, but any 'fantasy' value can be used*

.. _config_defaultduration:

defaultDuration:
    | Default: **1.0**  -- `(int, float)`
    | *Value used when a duration is needed and has not been set (Note, Chord). Not the same as play.dur*

.. _config_splitacceptabledeviation:

splitAcceptableDeviation:
    | Default: **4**  -- `int`
    | *When splitting notes between staves, notes within this range of the split point will be grouped together if they all fit*

.. _config_chord_arpeggio:

chord.arpeggio:
    | Default: **auto**  -- `(str, bool)`
    | Choices: ``auto, False, True``
    | *Arpeggiate notes of a chord when showing. In auto mode, only arpeggiate when needed*

.. _config_chord_adjustgain:

chord.adjustGain:
    | Default: **True**  -- `bool`
    | *Adjust the gain of a chord according to the number of notes, to prevent clipping*

.. _config_m21_displayhook_install:

m21.displayhook.install:
    | Default: **True**  -- `bool`

.. _config_m21_displayhook_format:

m21.displayhook.format:
    | Default: **xml.png**  -- `str`
    | Choices: ``lily.png, xml.png``

.. _config_m21_fixstream:

m21.fixStream:
    | Default: **True**  -- `bool`
    | *If True, fix the streams returned by .asmusic21 (see m21fix)*

.. _config_repr_showfreq:

repr.showFreq:
    | Default: **True**  -- `bool`
    | *Show frequency when calling printing a Note in the console*

.. _config_semitonedivisions:

semitoneDivisions:
    | Default: **4**  -- `int`
    | Choices: ``1, 2, 4``
    | *The number of divisions per semitone (2=quarter-tones, 4=eighth-tones)*

.. _config_dynamicscurve_shape:

dynamicsCurve.shape:
    | Default: **expon(3.0)**  -- `str`

.. _config_show_lastbreakpointdur:

show.lastBreakpointDur:
    | Default: **0.125**  -- `float`
    | Between 0.015625 - 1
    | *Dur of a note representing the end of a line/gliss, which has no duration per se*

.. _config_show_cents:

show.cents:
    | Default: **True**  -- `bool`
    | *show cents deviation as text when rendering notation*

.. _config_show_centsfontsize:

show.centsFontSize:
    | Default: **8**  -- `int`
    | *Font size used for cents annotations*

.. _config_show_split:

show.split:
    | Default: **True**  -- `bool`
    | *Should a voice be split between two staves?. A midinumber can be given instead*

.. _config_show_gliss:

show.gliss:
    | Default: **True**  -- `bool`
    | *If true, show a glissando line where appropriate*

.. _config_show_centsep:

show.centSep:
    | Default: **,**  -- `str`
    | *Separator used when displaying multiple cents deviation (in a chord)*

.. _config_show_scalefactor:

show.scaleFactor:
    | Default: **1.0**  -- `float`
    | *Affects the size of the generated image when using png format*

.. _config_show_staffsize:

show.staffSize:
    | Default: **12.0**  -- `float`
    | *The size of a staff, in points*

.. _config_show_backend:

show.backend:
    | Default: **lilypond**  -- `str`
    | Choices: ``lilypond, music21``
    | *method/backend used when rendering notation*

.. _config_show_format:

show.format:
    | Default: **png**  -- `str`
    | Choices: ``pdf, png, repr``
    | *Used when no explicit format is passed to .show*

.. _config_show_external:

show.external:
    | Default: **False**  -- `bool`
    | *Force opening images with an external tool, even when inside a Jupyter notebook*

.. _config_show_cacheimages:

show.cacheImages:
    | Default: **True**  -- `bool`
    | *If True, cache rendered images. Set it to False for debugging*

.. _config_show_arpeggioduration:

show.arpeggioDuration:
    | Default: **0.5**  -- `float`
    | *Duration used for individual notes when rendering a chord as arpeggio*

.. _config_show_labelfontsize:

show.labelFontSize:
    | Default: **12.0**  -- `float`

.. _config_show_pageorientation:

show.pageOrientation:
    | Default: **portrait**  -- `str`
    | Choices: ``landscape, portrait``
    | *Page orientation when rendering to pdf*

.. _config_show_pagesize:

show.pageSize:
    | Default: **a4**  -- `str`
    | Choices: ``a2, a3, a4``
    | *The page size when rendering to pdf*

.. _config_show_pagemarginmillimeters:

show.pageMarginMillimeters:
    | Default: **4**  -- `int`
    | Between 0 - 1000
    | *The page margin in mm*

.. _config_show_glissendstemless:

show.glissEndStemless:
    | Default: **False**  -- `bool`
    | *When the end pitch of a gliss. is shown as gracenote, make this stemless*

.. _config_show_glisshidetiednotes:

show.glissHideTiedNotes:
    | Default: **True**  -- `bool`
    | *Hide tied notes which are part of a glissando*

.. _config_show_lilypondpngstaffsizescale:

show.lilypondPngStaffsizeScale:
    | Default: **1.5**  -- `float`
    | *A factor applied to the staffsize when rendering to png via lilypond. Useful if rendered images appear too small in a jupyter notebook*

.. _config_show_pngresolution:

show.pngResolution:
    | Default: **200**  -- `int`
    | Choices: ``100, 200, 300, 600, 1200``
    | *DPI used when rendering to png*

.. _config_show_measureannotationfontsize:

show.measureAnnotationFontSize:
    | Default: **14**  -- `int`
    | *Font size used for measure annotations*

.. _config_show_respellpitches:

show.respellPitches:
    | Default: **True**  -- `bool`

.. _config_show_horizontalspacing:

show.horizontalSpacing:
    | Default: **normal**  -- `str`
    | Choices: ``large, medium, normal, xlarge``
    | *Hint for the renderer to adjust horizontal spacing. The actual result dependson the backend and the format used*

.. _config_show_glissandolinethickness:

show.glissandoLineThickness:
    | Default: **2**  -- `int`
    | Choices: ``1, 2, 3, 4``
    | *Line thikness when rendering glissandi. The value is abstract and it isup to the renderer to interpret it*

.. _config_show_filldynamicfromamplitude:

show.fillDynamicFromAmplitude:
    | Default: **False**  -- `bool`
    | *If True, when rendering notation, if an object has an amplitude and does not have an explicit dynamic, add a dynamic according to the amplitude*

.. _config_show_jupytermaximagewidth:

show.jupyterMaxImageWidth:
    | Default: **1000**  -- `int`
    | *A max. width in pixels for images displayed in a jupyter notebook*

.. _config_app_png:

app.png:
    | Default: ****  -- `str`
    | *Application used when opening .png files externally. If empty, the platform default is used*

.. _config_musescorepath:

musescorepath:
    | Default: ****  -- `str`
    | *The command to use when calling MuseScore. For macOS users: it must be an absolute path pointing to the actual binary inside the .app bundle*

.. _config_displayhook_install:

displayhook.install:
    | Default: **True**  -- `bool`

.. _config_play_dur:

play.dur:
    | Default: **2.0**  -- `float`
    | *Default duration of any play action if the object has no given duration*

.. _config_play_gain:

play.gain:
    | Default: **1.0**  -- `float`
    | Between 0 - 1

.. _config_play_chan:

play.chan:
    | Default: **1**  -- `int`
    | Between 1 - 64
    | *Default channel to play to. channels start at 1*

.. _config_play_enginename:

play.engineName:
    | Default: **maelzel.core**  -- `str`
    | *Name of the play engine used*

.. _config_play_instr:

play.instr:
    | Default: **sin**  -- `str`
    | *Default instrument used for playback. A list of available instruments can be queried via `availableInstrs`. New instrument presets can be defined via `defPreset`*

.. _config_play_fade:

play.fade:
    | Default: **0.02**  -- `float`
    | *default fade time*

.. _config_play_fadeshape:

play.fadeShape:
    | Default: **cos**  -- `str`
    | Choices: ``cos, linear``
    | *Curve-shape used for fading in/out*

.. _config_play_pitchinterpolation:

play.pitchInterpolation:
    | Default: **linear**  -- `str`
    | Choices: ``cos, linear``
    | *Curve shape for interpolating between pitches*

.. _config_play_numchannels:

play.numChannels:
    | Default: **2**  -- `int`
    | *Default number of channels (channels can be set explicitely when calling startPlayEngine*

.. _config_play_unschedfadeout:

play.unschedFadeout:
    | Default: **0.05**  -- `float`
    | *fade out when stopping a note*

.. _config_play_autostartengine:

play.autostartEngine:
    | Default: **True**  -- `bool`
    | *Start play engine if not started manually?*

.. _config_play_backend:

play.backend:
    | Default: **default**  -- `str`
    | Choices: ``alsa, auhal, default, jack, pa_cb, portaudio, pulse``
    | *backend used for playback*

.. _config_play_presetspath:

play.presetsPath:
    | Default: ****  -- `str`
    | *The path were presets are saved*

.. _config_play_autosavepresets:

play.autosavePresets:
    | Default: **True**  -- `bool`
    | *Automatically save user defined presets, so they will be available for a next session*

.. _config_play_defaultamplitude:

play.defaultAmplitude:
    | Default: **1.0**  -- `float`
    | Between 0 - 1
    | *The amplitude of a Note/Chord when an amplitude is needed and the object has an undefined amplitude*

.. _config_play_generalmidisoundfont:

play.generalMidiSoundfont:
    | Default: ****  -- `str`
    | *Path to a soundfont (sf2 file) with a general midi mapping*

.. _config_play_namedargsmethod:

play.namedArgsMethod:
    | Default: **pargs**  -- `str`
    | Choices: ``pargs, table``
    | *Method used to convert named parameters defined in a Preset to their corresponding function in a csoundengine.Instr*

.. _config_play_soundfontampdiv:

play.soundfontAmpDiv:
    | Default: **16384**  -- `int`

.. _config_play_soundfontinterpolation:

play.soundfontInterpolation:
    | Default: **linear**  -- `str`
    | Choices: ``cubic, linear``
    | *Interpolation used when reading sample data from a soundfont.*

.. _config_play_schedlatency:

play.schedLatency:
    | Default: **0.2**  -- `float`
    | *Added latency when scheduling events to ensure time precission*

.. _config_play_verbose:

play.verbose:
    | Default: **False**  -- `bool`
    | *If True, outputs extra debugging information regarding playback*

.. _config_rec_block:

rec.block:
    | Default: **False**  -- `bool`
    | *Should recording be blocking or should be done async?*

.. _config_rec_sr:

rec.sr:
    | Default: **44100**  -- `int`
    | Choices: ``44100, 48000, 88200, 96000``
    | *Sample rate used when rendering offline*

.. _config_rec_ksmps:

rec.ksmps:
    | Default: **64**  -- `int`
    | Choices: ``1, 16, 32, 64, 128, 256``
    | *samples per cycle when rendering offline (passed as ksmps to csound)*

.. _config_rec_nchnls:

rec.nchnls:
    | Default: **2**  -- `int`

.. _config_rec_path:

rec.path:
    | Default: ****  -- `str`
    | *path used to save output files when rendering offline. If not given the default can be queried via `recordPath`*

.. _config_rec_quiet:

rec.quiet:
    | Default: **False**  -- `bool`
    | *Supress debug output when calling csound as a subprocess*

.. _config_html_theme:

html.theme:
    | Default: **light**  -- `str`
    | Choices: ``dark, light``
    | *Theme used when displaying html inside jupyter*

.. _config_quant_minbeatfractionacrossbeats:

quant.minBeatFractionAcrossBeats:
    | Default: **1.0**  -- `float`

.. _config_quant_nestedtuples:

quant.nestedTuples:
    | Default: **False**  -- `bool`
    | *Are nested tuples allowed when quantizing? Not all display backends support nested tuples (musescore, used to render musicxml  has no support for nested tuples)*

.. _config_quant_complexity:

quant.complexity:
    | Default: **middle**  -- `str`
    | Choices: ``high, low, middle``
    | *Controls the allowed complexity in the notation. The higher the complexity, the more accurate the quantization, at the cost of a more complex notation. *

.. _config_logger_level:

logger.level:
    | Default: **INFO**  -- `str`
    | Choices: ``DEBUG, ERROR, INFO, WARNING``
