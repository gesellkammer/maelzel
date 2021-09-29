from __future__ import annotations
import tempfile
import functools

from maelzel import scoring
from pitchtools import m2n, m2f, f2m
import emlib.img

from ._common import *
from .workspace import activeConfig
from . import tools
from . import notation


@functools.cache
def _makeImageForPitch(notename: str) -> str:
    cfg = activeConfig()
    n = scoring.makeNote(notename, duration=cfg['defaultDuration'])
    part = scoring.Part([n])
    outfile = tempfile.mktemp(suffix=".png")
    renderer = notation.renderWithCurrentWorkspace([part])
    renderer.write(outfile)
    return outfile


class Pitch:

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
        if activeConfig()['repr.showFreq']:
            return f"<{self.name} {self.freq:.1f}Hz>"
        else:
            return self.name

    def makeImage(self) -> str:
        return _makeImageForPitch(self.name)

    def _htmlImage(self):
        imgpath = self.makeImage()
        scaleFactor = activeConfig().get('show.scaleFactor', 1.0)
        width, height = emlib.img.imgSize(imgpath)
        img = emlib.img.htmlImgBase64(imgpath,
                                      width=f'{int(width*scaleFactor)}px')
        return img

    def _repr_html_(self) -> str:
        img = self._htmlImage()
        return f"<code>{repr(self)}</code><br>"+img