from __future__ import annotations
import logging

from emlib import iterlib

from maelzel.scorestruct import *
from . import core
from . import enharmonics
from . import quant
from .renderbase import Renderer, RenderOptions
from . import renderm21
from . import renderlily
from .config import config


logger = logging.getLogger("maelzel.scoring")


def _asQuantizedScore(s: U[quant.QuantizedScore, List[quant.QuantizedPart]],
                      options: RenderOptions) -> quant.QuantizedScore:
    if isinstance(s, quant.QuantizedScore):
        return s
    return quant.QuantizedScore(parts=s,
                                title=options.title,
                                composer=options.composer)


def renderQuantizedScore(score: quant.QuantizedScore,
                         options:RenderOptions,
                         backend:str=None) -> Renderer:
    """
    Render the already quantized parts as notation.

    Args:
        score: the already quantized parts
        options: the RenderOptions used. A value of None will use default options
        backend: one of {'lilypond', 'music21'}

    Returns:
        a Renderer
    """
    if backend is None:
        backend = config['renderBackend']
    if options.removeSuperfluousDynamics:
        for part in score:
            part.removeUnnecessaryDynamics()
    for part in score:
        part.removeUnnecessaryGracenotes()
    if backend == 'music21':
        return renderm21.Music21Renderer(score, options=options)
    elif backend == 'lilypond' or backend == 'lily':
        for part in score:
            _markConsecutiveGracenotes(part)
        return renderlily.LilypondRenderer(score, options=options)
    else:
        raise ValueError(f"Supported backends: 'music21', 'lilypond'. Got {backend}")


def _markConsecutiveGracenotes(part: quant.QuantizedPart):
    for n0, n1 in iterlib.pairwise(part.flatNotations()):
        if n0.isRest:
            continue
        if n0.isGraceNote() and n1.isGraceNote():
            graceGroupState = n0.getProperty("graceGroup")
            if graceGroupState is None:
                n0.setProperty("graceGroup", "start")
                n1.setProperty("graceGroup", "continue")
            elif graceGroupState == 'continue':
                n1.setProperty("graceGroup", 'continue')
        elif n0.isGraceNote() and (n1.isRest or not n1.isGraceNote()) and \
                n0.getProperty('graceGroup') in ('start', 'continue'):
            n0.setProperty("graceGroup", "stop")


def _groupNotationsByMeasure(part:core.Part, struct: ScoreStruct
                             ) -> List[List[core.Notation]]:
    currMeasure = -1
    groups = []
    for n in part:
        assert n.offset is not None and n.duration is not None
        loc = struct.beatToLocation(n.offset)
        if loc.measureNum == currMeasure:
            groups[-1].append(n)
        else:
            # new measure
            currMeasure = loc.measureNum
            groups.append([n])
    return groups


def _fixEnharmonicsInPart(part: core.Part, struct: ScoreStruct,
                          options: enharmonics.EnharmonicOptions) -> None:
    # we split the notations into measures in order to reset
    # the fixed slots (how a specific sounding pitch is spelled)
    # at each measure
    notationGroups = _groupNotationsByMeasure(part, struct)
    for group in notationGroups:
        group[0].setProperty("resetEnharmonicSlots", True)
    enharmonics.fixEnharmonicsInPlace(part, options=options)


def _makeEnharmonicOptionsFromRenderOptions(options: RenderOptions
                                            ) -> enharmonics.EnharmonicOptions:
    return enharmonics.EnharmonicOptions(groupSize=options.enharmonicsGroupSize,
                                         groupStep=options.enharmonicsStep)


def _quantizeAndRender(parts: List[core.Part],
                       struct: ScoreStruct,
                       options: RenderOptions,
                       backend:str,
                       quantizationProfile:quant.QuantizationProfile=None,
                       ) -> Renderer:
    """
    Quantize and render unquantized events organized into parts
    """
    if options.respellPitches:
        enharmonicOptions = _makeEnharmonicOptionsFromRenderOptions(options)
        for part in parts:
            _fixEnharmonicsInPart(part, struct=struct, options=enharmonicOptions)
    qscore = quant.quantize(parts, struct=struct, quantizationProfile=quantizationProfile)
    renderer = renderQuantizedScore(score=qscore, options=options, backend=backend)
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
           struct:ScoreStruct=None,
           options: RenderOptions = None,
           backend:str=None,
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
    parts = _asParts(obj)
    if struct is None:
        struct = ScoreStruct.fromTimesig((4, 4), quarterTempo=60)
    if backend is None:
        backend = config['renderBackend']
    return _quantizeAndRender(parts, struct=struct, options=options, backend=backend,
                              quantizationProfile=quantizationProfile)









    

