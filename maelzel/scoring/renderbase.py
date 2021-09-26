"""
This module declares the basic classes for all renderers.
"""

from __future__ import annotations
import tempfile
from dataclasses import dataclass
import music21 as m21


from maelzel.music import m21tools

from .common import *
from . import quant
from .config import config
import emlib.img
import emlib.misc

@dataclass
class RenderOptions:
    """
    orientation: one of "portrait" or "landscape"
    staffSize: the size of each staff, in points
    pageSize: one of "a1", "a2", "a3", "a4", "a5"
    pageMarginMillimeters: page margin in mm. Only used by some backends

    divsPerSemitone: the number of divisions of the semitone
    showCents: should each note/chord have a text label attached
        indicating the cents deviation from the nearest semitone?
    centsPlacement: where to put the cents annotation
    centsFontSize: the font size of the cents annotation

    measureAnnotationFontSize: font size for measure annotations

    glissAllowNonContiguous: if True, allow glissandi between notes which
        have rests between them
    glissHideTiedNotes: if True, hide tied notes which are part of a gliss.

    lilypondPngSaffsizeScale: a scaling factor applied to staff size when rendering
        to png via lilypond. This is to prevent png images being too small
        or notation in pdfs being too big

    title: the title of the score
    composer: the composer of the score

    opaque: if True, rendered images will be opaque (no transparent
        background)
    """
    orientation: str = config['pageOrientation']
    staffSize: U[int, float] = config['staffSize']
    pageSize: str = config['pageSize']
    pageMarginMillimeters: Opt[int] = 4

    divsPerSemitone: int = config['divisionsPerSemitone']
    showCents: bool = config['showCents']
    centsPlacement: str = "above"
    centsFontSize: U[int, float] = config['centsFontSize']

    measureAnnotationFontSize: U[int, float] = config['measureAnnotationFontSize']
    measureAnnotationBoxed: bool = config['measureAnnotationBoxed']
    noteAnnotationsFontSize: U[int, float] = config['noteAnnotationFontSize']

    glissAllowNonContiguous: bool = False
    glissHideTiedNotes: bool = True

    horizontalSpacing: str = config['horizontalSpacing']
    lilypondPngStaffsizeScale: float = 1.4
    removeSuperfluousDynamics: bool = config['removeSuperfluousDynamics']
    restsResetDynamics: bool = True

    respellPitches: bool = config['respellPitches']
    glissandoLineThickness: int = config['glissandoLineThickness']

    enharmonicsGroupSize: int = 6
    enharmonicsStep: int = 3

    renderFormat: str = ''

    cropToContent: bool = False
    opaque: bool = True

    title: str = ''
    composer: str = ''

    def __post_init__(self):
        self.check()

    def check(self):
        assert self.orientation in ('portrait', 'landscape')
        assert isinstance(self.staffSize, (int, float)) and 2 < self.staffSize < 40, \
            f"Invalid staffSize: {self.staffSize}"
        assert self.pageSize in ('a1', 'a2', 'a3', 'a4', 'a5'), \
            f"Invalid page size, it must be one of a1, a2, ..., a5"
        assert self.divsPerSemitone in (1, 2, 4)
        assert self.centsPlacement in ('above', 'below')
        assert self.horizontalSpacing in ('normal', 'medium', 'large')


class Renderer:

    def __init__(self, score: quant.QuantizedScore, options:RenderOptions=None):
        assert score
        assert score[0].struct is not None
        self.score = score
        self.struct = score[0].struct
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

    def musicxml(self) -> Opt[str]:
        """
        Returns the rendered score as musicxml if supported

        Returns:
            either the musicxml as str, or None if not supported by
            this renderer
        """
        m21stream = self.asMusic21()
        if m21stream is None:
            return None
        return m21tools.getXml(m21stream)

    def asMusic21(self) -> Opt[m21.stream.Stream]:
        """
        If the renderer can return a music21 stream version of the render,
        return it here, otherwise return None
        """
        return None

    def nativeScore(self) -> str:
        raise NotImplementedError("Please Implement this method")

    def show(self, fmt='png', external=None):
        if fmt == 'pdf':
            external = True
        if fmt == 'png' and emlib.misc.inside_jupyter() and not external:
            from IPython.display import display_png
            png = tempfile.mktemp(suffix='.png')
            self.write(png)
            display_png(png)
        else:
            outfile = tempfile.mktemp(suffix=f'.{fmt}')
            self.write(outfile)
            emlib.misc.open_with(outfile)


    def _repr_html_(self) -> str:
        pngfile = tempfile.mktemp(suffix=".png", prefix="render-")
        self.write(pngfile)
        img = emlib.img.htmlImgBase64(pngfile, removeAlpha=True)
        html = f'Renderer({len(self.score)} parts)<br>'+img
        return html
