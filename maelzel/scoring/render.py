from __future__ import annotations
import logging
import os
import subprocess
import glob

from maelzel.scorestruct import *
from . import core
from . import quant
from .renderer import Renderer
from .renderoptions import RenderOptions
from . import renderlily

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union


logger = logging.getLogger("maelzel.scoring")


def renderQuantizedScore(score: quant.QuantizedScore,
                         options: RenderOptions
                         ) -> Renderer:
    """
    Render the already quantized parts as notation.

    Args:
        score: the already quantized parts
        options: the RenderOptions used. A value of None will use default options

    Returns:
        a Renderer
    """
    assert isinstance(options, RenderOptions)

    backend = options.backend

    if options.removeSuperfluousDynamics:
        for part in score:
            part.removeUnnecessaryDynamics(tree=True)
    #for part in score:
    #    part.removeUnnecessaryGracenotes()
    if backend == 'music21':
        from . import renderm21
        return renderm21.Music21Renderer(score, options=options)
    elif backend == 'lilypond':
        return renderlily.LilypondRenderer(score, options=options)
    else:
        raise ValueError(f"Supported backends: 'music21', 'lilypond'. Got {backend}")


def _groupNotationsByMeasure(part: core.Part,
                             struct: ScoreStruct
                             ) -> list[list[core.Notation]]:
    currMeasure = -1
    groups = []
    for n in part:
        assert n.offset is not None and n.duration is not None
        loc = struct.beatToLocation(n.offset)
        if loc is None:
            logger.error(f"Offset {n.offset} outside of scorestruct, for {n}")
            logger.error(f"Scorestruct: duration = {struct.totalDurationBeats()} quarters\n{struct.dump()}")
            raise ValueError(f"Offset {float(n.offset):.3f} outside of score structure "
                             f"(max. offset: {float(struct.totalDurationBeats()):.3f})")
        elif loc[0] == currMeasure:
            groups[-1].append(n)
        else:
            # new measure
            currMeasure = loc[0]
            groups.append([n])
    return groups


def quantizeAndRender(parts: list[core.Part],
                      struct: ScoreStruct,
                      options: RenderOptions,
                      quantizationProfile:quant.QuantizationProfile=None,
                      ) -> Renderer:
    """
    Quantize and render unquantized events organized into parts

    Args:
        parts: the parts to render
        struct: the ScoreStruct used
        options: RenderOptions
        backend: the backend to use ('lilypond', 'music21')

    Returns:
        the Renderer object
    """
    enharmonicOptions = options.makeEnharmonicOptions() if options.respellPitches else None
    qscore = quant.quantize(parts,
                            struct=struct,
                            quantizationProfile=quantizationProfile,
                            enharmonicOptions=enharmonicOptions)
    return renderQuantizedScore(score=qscore, options=options)


def _asParts(obj: Union[core.Part, core.Notation, list[core.Part], list[core.Notation]]
             ) -> list[core.Part]:
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


def render(obj: core.Part | core.Notation | list[core.Part] | list[core.Notation],
           struct: ScoreStruct = None,
           options: RenderOptions = None,
           backend='',
           quantizationProfile: quant.QuantizationProfile = None
           ) -> Renderer:
    """
    Quantize and render the given object `obj` to generate musical notation

    Args:
        obj: the object to render
        struct: the structure of the resulting score. To create a simple score
            with an anitial time signature and tempo, use something like
            `ScoreStructure.fromTimesig((4, 4), quarterTempo=52)`. If not given,
            defaults to a 4/4 score with tempo 60
        options: leave as None to use default render options, or create a
            RenderOptions object to specify things like page size, title,
            pitch resolution, etc.
        backend: The backend used for rendering. Supported backends at the
            moment: 'lilypond', 'music21'
        quantizationProfile:
            The quantization preset determines how events are quantized,
            which divisions of the beat are possible, how a best division
            is weighted and selected, etc. Not all options in a preset
            are supported by all backends (for example, music21 backend
            does not support nested tuples).
            See quant.presetQuantizationProfiles, which is a dict with
            some predefined profiles

    Returns:
        a Renderer. To produce a pdf or a png call :method:`Renderer.write` on
        the returned Renderer, like `renderer.write('outfile.pdf')`
    """
    parts = _asParts(obj)
    if struct is None:
        struct = ScoreStruct(timesig=(4, 4), tempo=60)
    if options is None:
        options = RenderOptions()
    if not backend:
        backend = options.backend
    return quantizeAndRender(parts, struct=struct, options=options,
                             quantizationProfile=quantizationProfile)


def renderMusicxml(xmlfile: str, outfile: str, method='', crop: bool = None, pngpage=1
                   ) -> None:
    """
    Convert a saved musicxml file to pdf or png

    Args:
        xmlfile: the musicxml file to convert
        outfile: the output file. The extension determines the output
            format. Possible formats pdf and png
        method: if given, will determine the method used to render. Use
            None to indicate a default method.
            Possible values: 'musescore'
        crop: if True, crop the image to the contents. This defaults to True for
            png and to False for pdf
        pngpage: which page to render if rendering to png


    Supported methods:

    ========  =============
    format    methods
    ========  =============
    pdf       musescore
    png       musescore
    ========  =============
    """
    from maelzel.core import environment
    musescore = environment.findMusescore()
    fmt = os.path.splitext(outfile)[1]
    if fmt == ".pdf":
        method = method or 'musescore'
        if method == 'musescore':
            if musescore is None:
                raise RuntimeError("MuseScore not found")
            subprocess.call([musescore, '--no-webview', '--export-to', outfile, xmlfile],
                            stderr=subprocess.PIPE)
            if not os.path.exists(outfile):
                raise RuntimeError(f"Could not generate pdf file {outfile} from {xmlfile}")
        else:
            raise ValueError(f"Method {method} unknown, possible values: 'musescore'")
    elif fmt == '.png':
        if crop is None:
            crop = True
        method = method or 'musescore'
        if method == 'musescore':
            if musescore:
                _musescoreRenderMusicxmlToPng(xmlfile, outfile, musescorepath=musescore,
                                              page=pngpage, crop=crop)
            else:
                raise RuntimeError("MuseScore not found")
        else:
            raise ValueError(f"method {method} unknown, possible values: 'musescore'")
    else:
        raise ValueError(f"format {fmt} not supported")


def _musescoreRenderMusicxmlToPng(xmlfile: str, outfile: str, musescorepath: str, page=1,
                                  crop=True) -> None:
    """
    Use musescore to render a musicxml file as png

    Args:
        xmlfile: the path to the musicxml file to render
        outfile: the png file to generate
        page: in the case that multiple pages were generated, use the given page
        crop: if true, trim the image to the contents
        musescorepath: if given, the path to the musescore binary

    Raises RuntimeError if the musicxml file could not be rendered
    """
    if not musescorepath:
        from maelzel.core import environment
        musescorepath = environment.findMusescore()
        if not musescorepath:
            raise RuntimeError("MuseScore not found, cannot render musicxml to png")
    assert os.path.exists(xmlfile), f"Musicxml file {xmlfile} not found"
    args = [musescorepath, '--no-webview']
    if crop:
        args.extend(['--trim-image', '10'])
    args.extend(['--export-to', outfile, xmlfile])
    subprocess.call(args, stderr=subprocess.PIPE)
    generatedFiles = glob.glob(os.path.splitext(outfile)[0] + "-*.png")
    if not generatedFiles:
        raise RuntimeError("No output files generated")
    for generatedFile in generatedFiles:
        generatedPage = int(os.path.splitext(generatedFile)[0].split("-")[-1])
        if generatedPage == page:
            os.rename(generatedFile, outfile)
            return
    raise RuntimeError(f"Page not found, generated files: {generatedFiles}")

