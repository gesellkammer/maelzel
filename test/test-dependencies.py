from maelzel import dependencies
from maelzel.music import lilytools
import sys


def testvamp():
    print("Testing included vamp plugins")
    import sndfileio
    import numpy as np
    samples, sr = sndfileio.sndread("snd/sine440.flac")
    dur = len(samples) / sr
    from maelzel.snd import vamptools
    print("  Testing pyin")
    assert vamptools.pyinAvailable()
    arr = vamptools.pyinPitchTrack(samples=samples, sr=sr)
    for row in arr:
        print(f"time={row[0]:.3f}, freq: {row[1]:.3f} Hz, voiced: {row[2]:.3f}")


exiterr = 0
errors = dependencies.checkDependencies(fix=True)
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

testvamp()

dependencies.printReport()


sys.exit(exiterr)
