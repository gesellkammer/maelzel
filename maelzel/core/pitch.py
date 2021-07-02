from __future__ import annotations
import tempfile
import functools

from maelzel import scoring
from pitchtools import m2n, m2f, f2m

from ._common import *
from .workspace import currentConfig
from . import tools
from . import notation


@functools.lru_cache(maxsize=1000)
def _makeImageForPitch(notename: str) -> str:
    cfg = currentConfig()
    n = scoring.makeNote(notename, duration=cfg['defaultDuration'])
    part = scoring.Part([n])
    outfile = tempfile.mktemp(suffix=".png")
    renderer = notation.renderWithCurrentConfig([part])
    renderer.write(outfile)
    return outfile


class Pitch:

    _classInitialized = False

    __slots__ = ("midi")

    def __init__(self, pitch:pitch_t):
        self.midi:float = tools.asmidi(pitch)

    @property
    def name(self) -> str:
        return m2n(self.midi)

    @property
    def freq(self) -> float:
        return m2f(self.midi)

    def __float__(self) -> float:
        return self.midi

    def __add__(self, other) -> Pitch:
        if isNumber(other):
            return Pitch(self.midi + float(other))
        return NotImplemented

    def __mul__(self, other) -> Pitch:
        if isNumber(other):
            return Pitch(f2m(self.freq * float(other)))
        return NotImplemented

    def __truediv__(self, other) -> Pitch:
        if isNumber(other):
            return Pitch(f2m(self.freq/float(other)))
        return NotImplemented

    def __eq__(self, other) -> bool:
        return self.midi == float(other)

    def __gt__(self, other) -> bool:
        return self.midi > float(other)

    def __repr__(self) -> str:
        if currentConfig()['repr.showFreq']:
            return f"<{self.name} {self.freq:.1f}Hz>"
        else:
            return self.name

    def makeImage(self) -> str:
        return _makeImageForPitch(self.name)

    @classmethod
    def setJupyterHook(cls) -> None:
        """
        Sets the jupyter display hook for this class

        """
        if cls._classInitialized:
            return
        from IPython.core.display import Image

        def reprpng(obj: Pitch):
            imgpath = obj.makeImage()
            scaleFactor = currentConfig().get('show.scaleFactor', 1.0)
            if scaleFactor != 1.0:
                imgwidth, imgheight = tools.imgSize(imgpath)
                width = imgwidth*scaleFactor
            else:
                width = None
            return Image(filename=imgpath, embed=True, width=width)._repr_png_()

        tools.setJupyterHookForClass(cls, reprpng, fmt='image/png')