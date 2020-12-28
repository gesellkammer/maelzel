"""
This module declares the basic classes for all renderers.
"""

from __future__ import annotations
from dataclasses import dataclass
import music21 as m21

from maelzel.music import m21tools

from .common import *
from . import quant


@dataclass
class RenderOptions:
    """
    orientation: one of "portrait" or "landscape"
    staffSize: the size of each staff in point
    pageSize: one of "a4", "a3"

    divsPerSemitone: the number of divisions of the semitone
    showCents: should each note/chord have a text label attached
    indicating the cents deviation from the nearest semitone?
    centsPlacement: where to put the cents annotation
    centsFontSize: the font size of the cents annotation

    measureAnnotationFontSize: font size for measure annotations

    glissAllowNonContiguous: if True, allow glissandi between notes which
        have rests between them
    glissHideTiedNotes: if True, hide tied notes which are part of a gliss.

    title: the title of the score
    composer: the composer of the score
    """
    orientation: str = "portrait"
    staffSize: int = 12
    pageSize: str = 'a4'

    divsPerSemitone: int = 4
    showCents: bool = False
    centsPlacement: str = "above"
    centsFontSize: int = 10

    measureAnnotationFontSize: int = 12

    glissAllowNonContiguous: bool = False
    glissHideTiedNotes: bool = False

    title: str = ''
    composer: str = ''


class Renderer:
    def __init__(self, parts: List[quant.QuantizedPart], options:RenderOptions=None):
        assert parts
        assert parts[0].struct is not None
        self.parts = parts
        self.struct = parts[0].struct
        if options is None:
            options = RenderOptions()
        self.options = options
        self._rendered = False

    def render(self) -> None:
        """
        This method should be implemented by the backend
        """
        raise NotImplementedError("Please Implement this method")

    def writeFormats(self) -> List[str]:
        """
        Returns: a list of possible write formats (pdf, xml, musicxml, etc)
        """
        raise NotImplementedError("Please Implement this method")

    def write(self, outfile:str) -> None:
        raise NotImplementedError("Please Implement this method")

    def show(self) -> None:
        raise NotImplementedError("Please Implement this method")

    def musicxml(self) -> Opt[str]:
        m21stream = self.asMusic21()
        return m21tools.getXml(m21stream) if m21stream else None

    def asMusic21(self) -> Opt[m21.stream.Stream]:
        """
        If the renderer can return a music21 stream version of the render,
        return it here, otherwise return None
        """
        return None