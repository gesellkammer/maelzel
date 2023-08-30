from maelzel import dependencies
from maelzel.music import lilytools
import sys

exiterr = 0
errors = dependencies.checkDependencies()
if errors:
    for err in errors:
        print(f"*** ERROR *** : {err}")
    exiterr = 1

lily = lilytools.findLilypond()
if lily is None:
    print("Lilypond not found")
    exiterr = 1
else:
    print(f"lilypond binary: {lily}")
    print(f"lilypond version: {lilytools.getLilypondVersion()}")

sys.exit(exiterr)