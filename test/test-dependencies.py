from maelzel import dependencies
from maelzel.music import lilytools

dependencies.checkDependencies()

print(f"lilypond binary: {lilytools.findLilypond()}")

print(f"lilypond version: {lilytools.getLilypondVersion()}")
