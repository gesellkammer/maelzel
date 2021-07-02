from __future__ import annotations
import logging

from maelzel.scorestruct import *
from . import core
from . import quant
from .renderbase import Renderer, RenderOptions
from . import renderm21
from . import renderlily
from .config import config


logger = logging.getLogger("maelzel.scoring")


def renderQuantizedParts(parts: List[quant.QuantizedPart],
                         options:RenderOptions,
                         backend:str) -> Renderer:
    """
    Render the already quantized parts as notation.

    Args:
        parts: the already quantized parts
        options: the RenderOptions used. A value of None will use default options
        backend: one of {'music21'}

    Returns:
        a Renderer
    """
    if backend == 'musicxml' or backend == 'xml' or backend == 'music21':
        return renderm21.Music21Renderer(parts, options=options)
    elif backend == 'lilypond' or backend == 'lily':
        return renderlily.LilypondRenderer(parts, options=options)
    else:
        raise ValueError(f"Supported backends: 'musicxml', 'lilypond'. Got {backend}")


def renderParts(parts: List[core.Part],
                struct: ScoreStructure=None,
                options: RenderOptions=None,
                backend:str=None,
                quantizationProfile:quant.QuantizationProfile=None
                ) -> Renderer:
    """
    Quantize and render unquantized events organized into parts

    Args:
        parts: a list of Parts, where each part represents a series
            of non-overlapping events which have not yet been quantized
        struct:
            the structure of the resulting score. To create a simple score
            with an anitial time signature and tempo, use something like
            `ScoreStructure.fromTimesig((4, 4), quarterTempo=52)`. If not given,
            defaults to a 4/4 score with tempo 60
        options:
            leave as None to use default render options, or create a
            RenderOptions object to specify things like page size, title,
            pitch resolution, etc.
        backend:
            The backend used for rendering. Supported backends at the
            moment: 'music21'
        quantizationProfile:
            The quantization profile determines how events are quantized,
            which divisions of the beat are possible, how a best division
            is weighted and selected, etc. Not all options in a profile
            are supported by all backends (for example, music21 backend
            does not support nested tuples).
            See quant.presetQuantizationProfiles, which is a dict with
            some predefined profiles

    Returns:
        a Renderer. To produce a pdf or a png call :method:`Renderer.write` on
        the returned Renderer, like `renderer.write('outfile.pdf')`

    """
    if quantizationProfile is None:
        quantizationProfile = quant.QuantizationProfile()
    if backend is None:
        backend = config['renderBackend']
    if backend == 'musicxml':
        quantizationProfile.nestedTuples = False
    if struct is None:
        struct = ScoreStructure.fromTimesig((4,4), quarterTempo=60)
    qparts = []
    for part in parts:
        qpart = quant.quantizePart(struct,
                                   part=part,
                                   profile=quantizationProfile)
        qpart.label = part.label
        qparts.append(qpart)
    logger.info("Using backend", backend)
    renderer = renderQuantizedParts(parts=qparts, options=options, backend=backend)
    renderer.render()
    return renderer


def _asParts(obj: U[core.Part, core.Notation, List[core.Part], List[core.Notation]]
             ) -> List[core.Part]:
    if isinstance(obj, core.Part):
        parts = [obj]
    elif isinstance(obj, list):
        if all(isinstance(item, core.Part) for item in obj):
            parts = obj
        elif all(isinstance(item, core.Notation) for item in obj):
            parts = [core.Part(obj)]
        else:
            raise TypeError(f"Can't show {obj}")
    elif isinstance(obj, core.Notation):
        parts = [core.Part([obj])]
    else:
        raise TypeError(f"Can't convert {obj} to a list of Parts")
    return parts


def render(obj: U[core.Part, core.Notation, List[core.Part], List[core.Notation]],
           struct:ScoreStructure=None,
           options: RenderOptions = None,
           backend:str=None,
           quantizationProfile: quant.QuantizationProfile = None
           ) -> Renderer:
    """
    Render the given object `obj` as notation

    Args:
        obj: the object to render (a Notation, a list thereof, a Part or a list thereof)
        struct: the score structure
        options: the render options
        backend: the render backend to use ('musicxml', 'lilypond')
        quantizationProfile: the quantization profile

    Returns:
        a new Renderer

    """
    parts = _asParts(obj)
    return renderParts(parts, struct=struct, options=options, backend=backend,
                       quantizationProfile=quantizationProfile)









    

