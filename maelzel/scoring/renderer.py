"""
This module declares the basic classes for all renderers.
"""

from __future__ import annotations
import tempfile
from maelzel.scorestruct import ScoreStruct
from maelzel.scoring.renderoptions import RenderOptions
from maelzel.scoring import quant

import emlib.img
import emlib.misc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import music21


class Renderer:
    """
    Renders a quantizedscore to a given format

    This is an abstract base class for different backend renderers (lilypond, musicxml, etc)
    """

    def __init__(self, score: quant.QuantizedScore, options:RenderOptions):
        assert score
        assert score[0].struct is not None
        self.quantizedScore: quant.QuantizedScore = score
        self.struct: ScoreStruct = score[0].struct
        self.options = options
        self._rendered = False

    def reset(self) -> None:
        """
        Resets the current renderer so that a new render is possible

        A Renderer caches its internal representation and last rendered
        score. This method resets the Renderer to its state after
        construction
        """
        self._rendered = False

    def __hash__(self) -> int:
        return hash((hash(self.quantizedScore), hash(self.struct), hash(self.options)))

    def render(self) -> None:
        """
        Render the quantized score

        .. note::

            The result is internally cached to calling this method multiple times
            only performs the rendering once.
        """
        raise NotImplementedError("Please Implement this method")

    def writeFormats(self) -> list[str]:
        """
        Returns: a list of possible write formats (pdf, xml, musicxml, etc)
        """
        raise NotImplementedError("Please Implement this method")

    def write(self, outfile:str) -> None:
        """Write the rendered score to a file"""
        raise NotImplementedError("Please Implement this method")

    def musicxml(self) -> str | None:
        """
        Returns the rendered score as musicxml if supported

        Returns:
            either the musicxml as str, or None if not supported by
            this renderer
        """
        m21stream = self.asMusic21()
        if m21stream is None:
            return None
        from maelzel.music import m21tools
        return m21tools.getXml(m21stream)

    def asMusic21(self) -> music21.stream.Stream | None:
        """
        If the renderer can return a music21 stream version of the render,
        return it here, otherwise return None
        """
        return None

    def nativeScore(self) -> str:
        """
        Returns the string representation of the rendered score

        This will be backend dependent. For a lilypond renderer this would be
        the actual lilypond score; for a musicxml renderer this would be the
        xml text, etc.

        Returns:
            the actual rendered score, as text (in lilypond format, xml format,
            etc., depending on the backend)
        """
        raise NotImplementedError("Please Implement this method")

    def show(self, fmt='png', external=None) -> None:
        """
        Display the rendered score

        Args:
            fmt: one of 'png', 'pdf'
            external: if True, for the use of an external app to open the rendered result.
                Otherwise, if running inside jupyter this command will try to display
                the result inline
        """
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
            emlib.misc.open_with_app(outfile)

    def _repr_html_(self) -> str:
        pngfile = tempfile.mktemp(suffix=".png", prefix="render-")
        self.write(pngfile)
        img = emlib.img.htmlImgBase64(pngfile, removeAlpha=True)
        parts = "1 part" if len(self.quantizedScore) == 1 else f"{len(self.quantizedScore)} parts"
        html = f'<b>{type(self).__name__}</b> ({parts})<br>'+img
        return html
