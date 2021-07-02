from ._common import *
from .workspace import currentConfig, currentWorkspace

from maelzel import scoring
from maelzel.scorestruct import ScoreStructure
import music21 as m21

import configdict


def scoringPartToMusic21(part: U[ scoring.Part, List[scoring.Notation] ],
                         struct: Opt[ScoreStructure] = None,
                         config: dict=None
                         ) -> U[m21.stream.Score, m21.stream.Part]:
    """
    Creates a m21 Part from the given events according to the config

    Assumes that the events fit in one Part.

    Args:
        part: the events to convert
        struct: the score structure used.
        config: the configuration used. If None, the `currentConfig()` is used

    Returns:
        a music21 Part

    """
    m21score = scoringPartsToMusic21([part], struct=struct,
                                     config=config)
    assert len(m21score.voices) == 1
    return m21score.voices[0]


def scoringPartsToMusic21(parts: List[U[scoring.Part, List[scoring.Notation]]],
                          struct: Opt[ScoreStructure] = None,
                          config:dict=None
                          ) -> U[m21.stream.Score]:
    """
    Render the given scoring Parts as music21

    Args:
        parts: the parts to render
        struct: the score structure
        config ():

    Returns:

    """
    config = config or currentConfig()
    divsPerSemitone = config['show.semitoneDivisions']
    showCents = config['show.cents']
    centsFontSize = config['show.centsFontSize']
    if struct is None:
        struct = currentWorkspace().scorestruct
    renderOptions = scoring.render.RenderOptions(divsPerSemitone=divsPerSemitone,
                                                 showCents=showCents,
                                                 centsFontSize=centsFontSize)
    quantProfile = scoring.quant.QuantizationProfile(nestedTuples=False)
    for part in parts:
        scoring.stackNotationsInPlace(part)
    renderer = scoring.render.renderParts(parts, struct=struct,
                                          options=renderOptions,
                                          backend="music21",
                                          quantizationProfile=quantProfile)
    m21score = renderer.asMusic21()
    return m21score


def makeRenderOptionsFromConfig(cfg: configdict.CheckedDict = None
                                ) -> scoring.render.RenderOptions:
    """
    Generate RenderOptions needed for scoring.render, based
    on the settings in the given config

    Args:
        the config to use. If None, the current config is used

    Returns:
        a scoring.render.RenderOptions used to render parts
        via scoring.render module
    """
    if cfg is None:
        cfg = currentConfig()
    renderOptions = scoring.render.RenderOptions(
            staffSize=cfg['show.staffSize'],
            divsPerSemitone=cfg['semitoneDivisions'],
            showCents=cfg['show.cents'],
            centsFontSize=cfg['show.centsFontSize'],
            noteAnnotationsFontSize=cfg['show.labelFontSize']
    )
    return renderOptions


def makeQuantizationProfileFromConfig(cfg: configdict.CheckedDict = None
                                      ) -> scoring.quant.QuantizationProfile:
    if cfg is None:
        cfg = currentConfig()
    complexity = cfg['quant.complexity']
    preset = scoring.quant.quantdata.complexityPresets[complexity]
    return scoring.quant.QuantizationProfile(
            minBeatFractionAcrossBeats=cfg['quant.minBeatFractionAcrossBeats'],
            nestedTuples=cfg['quant.nestedTuples'],
            possibleDivisionsByTempo=preset['divisionsByTempo'],
            divisionPenaltyMap=preset['divisionPenaltyMap']
    )


def renderWithCurrentConfig(parts: List[scoring.Part], backend: str = None
                            ) -> scoring.render.Renderer:
    """
    Render the given scoring.Parts with the current configuration

    Args:
        parts: the parts to render
        backend: the backend used (see currentConfig/'show.method')

    Returns:
        the rendered Renderer
    """
    state = currentWorkspace()
    cfg = state.config
    options = makeRenderOptionsFromConfig()
    quantizationProfile = makeQuantizationProfileFromConfig()
    return scoring.render.renderParts(parts,
                                      backend=backend or cfg['show.method'],
                                      struct=state.scorestruct,
                                      options=options,
                                      quantizationProfile=quantizationProfile)