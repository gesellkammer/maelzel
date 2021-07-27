import configdict

config = configdict.CheckedDict()
config.addKey('pageSize', 'a4', type=str, choices=('a3', 'a4'))
config.addKey('renderBackend', 'lilypond', choices=('music21', 'lilypond'))
config.addKey('staffSize', 12.0, type=float, range=(4, 36))
config.addKey('pageOrientation', 'portrait', choices=('portrait', 'landscape'))
config.addKey('divisionsPerSemitone', 4, choices=(1, 2, 4))
config.addKey('centsFontSize', 10, type=int,
              doc="Font used for cent annotations (in points)")
config.addKey('measureAnnotationFontSize', 13, int,
              doc="Font size (in points) for measure annotations")
config.addKey('noteAnnotationFontSize', 10, int,
              doc="Font size (in points) for note annotations")
config.addKey('showCents', False, type=bool,
              doc="Show cents deviations as note annotations")
config.addKey('horizontalSpacing', 'large', choices=('normal', 'medium', 'large', 'xlarge'))
config.addKey('glissandoLineThickness', 1,
              choices=(1, 2, 3, 4))
config.addKey('measureAnnotationBoxed', True,
              doc="Add a square box around a measure label")
config.addKey('removeSuperfluousDynamics', True,
              doc="Remove superfluous dynamics within a part")
config.addKey('respellPitches', True,
              doc='Try make pitch sequences more readable using enharmonic spelling for '
                  'certain pitches')
config.load()