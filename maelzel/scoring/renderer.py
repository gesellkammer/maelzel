"""
This module declares the basic classes for all renderers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import tempfile
from maelzel.scorestruct import ScoreStruct
from maelzel.scoring.renderoptions import RenderOptions
from maelzel.scoring import quant
from maelzel.scoring.config import config
from maelzel import _util

import emlib.img
import emlib.misc
import emlib.envir


class Renderer(ABC):
    """
    Renders a quantizedscore to a given format

    This is an abstract base class for different backend renderers (lilypond, musicxml, etc)
    """

    def __init__(self, score: quant.QuantizedScore, options: RenderOptions):
        if not score:
            raise ValueError("The quantized score is empty")

        if score[0].struct is None:
            raise ValueError("The score has no associated ScoreStruct, cannot render")

        self.quantizedScore: quant.QuantizedScore = score
        """The quantized score used for rendering"""

        self.struct: ScoreStruct = score[0].struct
        """The ScoreStruct used for rendering"""

        self.options: RenderOptions = options
        """The render options used"""

    def __hash__(self) -> int:
        return hash((hash(self.quantizedScore), hash(self.struct), hash(self.options)))

    @abstractmethod
    def render(self, options: RenderOptions = None) -> str:
        """
        Render the quantized score

        .. note::

            The result is internally cached so calling this method multiple times
            only performs the rendering once as long as the options used do not
            change

        Args:
            options: if given, these options override the own options

        Returns:
            the rendered score, as text. This will be lilypond text if the
            renderer is a LilypondRenderer, or musicxml text in the case
            of a MusicxmlRenderer
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def writeFormats(self) -> list[str]:
        """
        List of formats supported by this Renderer

        Returns:
            a list of possible write formats (pdf, xml, musicxml, etc)
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def write(self, outfile: str, fmt: str = None, removeTemporaryFiles=False) -> None:
        """
        Write the rendered score to a file

        Args:
            outfile: the path to the written file
            fmt: if given, this will be used as format for the output, independent
                of the extension used in outfile. The possible values depend on the
                formats supported by this Renderer (see :meth:`Renderer.writeFormats`)
            removeTemporaryFiles: if True, removes any temporary files generated during
                the rendering/writing process
        """
        raise NotImplementedError("Please Implement this method")

    def musicxml(self) -> str | None:
        """
        Returns the rendered score as musicxml if supported

        Returns:
            either the musicxml as str, or None if not supported by
            this renderer

        """
        return None

    def show(self, fmt='png', external=False, scalefactor=1.0) -> None:
        """
        Display the rendered score

        Args:
            fmt: one of 'png', 'pdf'
            external: if True, for the use of an external app to open the rendered result.
                Otherwise, if running inside jupyter this command will try to display
                the result inline
            scalefactor: a factor to scale the generated image when shown inline
        """
        if fmt == 'pdf':
            external = True

        if fmt == 'png':
            from maelzel.core import _tools
            png = tempfile.mktemp(suffix='.png')
            self.write(png)
            _util.pngShow(png, forceExternal=external, inlineScale=scalefactor)
        elif fmt == 'pdf':
            outfile = tempfile.mktemp(suffix=f'.{fmt}')
            self.write(outfile)
            emlib.misc.open_with_app(outfile)
        else:
            raise ValueError(f"fmt should be 'png' or 'pdf', got '{fmt}'")

    def _repr_html_(self) -> str:
        scale = config['pngScale']
        pngfile = tempfile.mktemp(suffix=".png", prefix="render-")
        self.write(pngfile)
        w, h = emlib.img.imgSize(pngfile)
        img = emlib.img.htmlImgBase64(pngfile, removeAlpha=True, width=f'{int(w*scale)}px')
        parts = "1 part" if len(self.quantizedScore) == 1 else f"{len(self.quantizedScore)} parts"
        html = f'<b>{type(self).__name__}</b> ({parts})<br>'+img
        return html
