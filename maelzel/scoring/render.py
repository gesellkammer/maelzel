from __future__ import annotations
import os
import subprocess
import glob

from maelzel.scorestruct import ScoreStruct
from . import core
from . import quant
from .renderer import Renderer
from .renderoptions import RenderOptions
from . import renderlily
from .common import logger
from . import attachment

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass


__all__ = (
    'renderQuantizedScore',
    'quantizeAndRender',
    'render',
    'renderMusicxml',
    'Renderer',
    'RenderOptions'

)


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
    assert len(score.parts) > 0

    backend = options.backend

    if options.removeRedundantDynamics:
        for part in score:
            part.removeRedundantDynamics(resetTime=options.dynamicsResetTime,
                                         resetAfterEmptyMeasure=options.dynamicsResetAfterEmptyMeasure,
                                         resetAfterRest=options.dynamicsResetAfterRest)

    for i, part in enumerate(score.parts):
        if part.autoClefChanges or (options.autoClefChanges and part.autoClefChanges is None):
            # Do not add if there are manual clefs
            if any(n.findAttachment(attachment.Clef) for n in part.flatNotations()):
                logger.debug(f"Part #{i} (name={part.name}) already has manual clefs set, skipping automatic clefs")
            else:
                part.findClefChanges(apply=True,
                                     biasFactor=options.keepClefBiasFactor,
                                     window=options.autoClefChangesWindow,
                                     simplificationThreshold=options.clefSimplifyThreshold,
                                     propertyKey='')
        # part.repairLinks()

    if backend == 'musicxml':
        from . import rendermusicxml
        return rendermusicxml.MusicxmlRenderer(score=score, options=options)
    elif backend == 'lilypond':
        return renderlily.LilypondRenderer(score, options=options)
    else:
        raise ValueError(f"Supported backends: 'lilypond', 'musicxml'. Got {backend}")


def _groupNotationsByMeasure(part: core.UnquantizedPart,
                             struct: ScoreStruct
                             ) -> list[list[core.Notation]]:
    currMeasure = -1
    groups = []
    for n in part:
        assert n.offset is not None and n.duration is not None
        loc = struct.beatToLocation(n.offset)
        if loc is None:
            logger.error(f"Offset {n.offset} outside of scorestruct, for {n}")
            logger.error(f"Scorestruct: duration = {struct.durationQuarters()} quarters\n{struct.dump()}")
            raise ValueError(f"Offset {float(n.offset):.3f} outside of score structure "
                             f"(max. offset: {float(struct.durationQuarters()):.3f})")
        elif loc[0] == currMeasure:
            groups[-1].append(n)
        else:
            # new measure
            currMeasure = loc[0]
            groups.append([n])
    return groups


def quantizeAndRender(parts: list[core.UnquantizedPart],
                      struct: ScoreStruct,
                      options: RenderOptions,
                      quantizationProfile: quant.QuantizationProfile,
                      ) -> Renderer:
    """
    Quantize and render unquantized events organized into parts

    Args:
        parts: the parts to render
        struct: the ScoreStruct used
        options: RenderOptions
        quantizationProfile: the profile to use for quantization, passed
            to maelzel.scoring.quant.quantize. If not given a default profile
            is used

    Returns:
        the Renderer object
    """
    enharmonicOptions = options.makeEnharmonicOptions() if options.respellPitches else None
    qscore = quant.quantizeParts(parts,
                                 quantizationProfile=quantizationProfile,
                                 struct=struct,
                                 enharmonicOptions=enharmonicOptions)
    return renderQuantizedScore(score=qscore, options=options)


def _asParts(obj: core.UnquantizedPart | core.Notation | list[core.UnquantizedPart] | list[core.Notation]
             ) -> list[core.UnquantizedPart]:
    if isinstance(obj, core.UnquantizedPart):
        return [obj]
    elif isinstance(obj, list):
        if all(isinstance(item, core.UnquantizedPart) for item in obj):
            return obj  # type: ignore
        elif all(isinstance(item, core.Notation) for item in obj):
            return [core.UnquantizedPart(notations=obj)]  # type: ignore
        else:
            raise TypeError(f"Can't show {obj}")
    elif isinstance(obj, core.Notation):
        return [core.UnquantizedPart([obj])]
    else:
        raise TypeError(f"Can't convert {obj} to a list of Parts")


def render(obj: core.UnquantizedPart | core.Notation | list[core.UnquantizedPart] | list[core.Notation],
           struct: ScoreStruct | None = None,
           options: RenderOptions | None = None,
           backend='',
           quantizationProfile: quant.QuantizationProfile | str = 'high'
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
            moment: 'lilypond', 'musicxml'
        quantizationProfile:
            The quantization preset determines how events are quantized,
            which divisions of the beat are possible, how the best division
            is weighted and selected, etc. Not all options in a preset
            are supported by all backends (for example, the musicxml backend
            does not support nested tuples). A preset can also be given
            (see ``maelzel.scoring.quantdata.presets``)
            
    Returns:
        a Renderer. To produce a pdf or a png call :method:`Renderer.write` on
        the returned Renderer, like `renderer.write('outfile.pdf')`
    """
    parts = _asParts(obj)
    if struct is None:
        struct = ScoreStruct((4, 4), tempo=60)
    if options is None:
        options = RenderOptions()
    if backend and options.backend != backend:
        options = options.clone(backend=backend)
    if isinstance(quantizationProfile, str):
        from maelzel.scoring.quantprofile import QuantizationProfile
        quantizationProfile = QuantizationProfile.fromPreset(quantizationProfile)
        
    return quantizeAndRender(parts, 
                             struct=struct, 
                             options=options,
                             quantizationProfile=quantizationProfile)


def renderMusicxml(xmlfile: str, outfile: str, method='musescore', crop: bool | None = None,
                   pngpage=1
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
