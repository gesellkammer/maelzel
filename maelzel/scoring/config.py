import configdict

config = configdict.ConfigDict('maelzel.scoring')
config.addKey('pageSize', 'a4', type=str, choices=('a3', 'a4'))
config.addKey('renderBackend', 'lilypond', choices=('musicxml', 'lilypond'))
config.addKey('staffSize', 12.0, type=float, range=(4, 36))
config.addKey('pageOrientation', 'portrait', choices=('portrait', 'landscape'))
config.addKey('divisionsPerSemitone', 4, choices=(1, 2, 4))
config.addKey('centsFontSize', 10, type=int)
config.addKey('measureAnnotationFontSize', 12, int)
config.addKey('noteAnnotationFontSize', 10, int)
config.addKey('showCents', False, type=bool)
config.addKey('lilypondHorizontalSpacing', 'large', choices=('normal', 'medium', 'large'))

config.load()
