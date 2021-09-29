"""
Functionality to interface with maelzel.scoring

"""
from __future__ import annotations
from ._common import *
from .workspace import activeConfig, activeWorkspace

from maelzel import scoring
from maelzel.scorestruct import ScoreStruct
import music21 as m21

import configdict

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


def scoringPartToMusic21(part: Union[ scoring.Part, List[scoring.Notation] ],
                         struct: Optional[ScoreStruct] = None,
                         config: dict=None
                         ) -> Union[m21.stream.Score, m21.stream.Part]:
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


def scoringPartsToMusic21(parts: List[Union[scoring.Part, List[scoring.Notation]]],
                          struct: Optional[ScoreStruct] = None,
                          config:dict=None
                          ) -> Union[m21.stream.Score]:
    """
    Render the given scoring Parts as music21

    Args:
        parts: the parts to render
        struct: the score structure
        config ():

    Returns:

    """
    config = config or activeConfig()
    divsPerSemitone = config['show.semitoneDivisions']
    showCents = config['show.cents']
    centsFontSize = config['show.centsFontSize']
    if struct is None:
        struct = activeWorkspace().scorestruct
    renderOptions = scoring.render.RenderOptions(divsPerSemitone=divsPerSemitone,
                                                 showCents=showCents,
                                                 centsFontSize=centsFontSize)
    quantProfile = scoring.quant.QuantizationProfile(nestedTuples=False)
    for part in parts:
        scoring.stackNotationsInPlace(part)
    renderer = scoring.render._quantizeAndRender(parts, struct=struct,
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
        cfg = activeConfig()
    renderOptions = scoring.render.RenderOptions(
            staffSize=cfg['show.staffSize'],
            divsPerSemitone=cfg['semitoneDivisions'],
            showCents=cfg['show.cents'],
            centsFontSize=cfg['show.centsFontSize'],
            noteAnnotationsFontSize=cfg['show.labelFontSize'],
            pageSize = cfg['show.pageSize'],
            orientation= cfg['show.pageOrientation'],
            pageMarginMillimeters=cfg['show.pageMarginMillimeters'],
            measureAnnotationFontSize=cfg['show.measureAnnotationFontSize'],
            respellPitches=cfg['show.respellPitches'],
            glissandoLineThickness=cfg['show.glissandoLineThickness']
    )
    return renderOptions


def makeQuantizationProfileFromConfig(cfg: configdict.CheckedDict = None
                                      ) -> scoring.quant.QuantizationProfile:
    if cfg is None:
        cfg = activeConfig()
    complexity = cfg['quant.complexity']
    preset = scoring.quant.quantdata.complexityPresets[complexity]
    return scoring.quant.QuantizationProfile(
            minBeatFractionAcrossBeats=cfg['quant.minBeatFractionAcrossBeats'],
            nestedTuples=cfg['quant.nestedTuples'],
            possibleDivisionsByTempo=preset['divisionsByTempo'],
            divisionPenaltyMap=preset['divisionPenaltyMap']
    )


def renderWithCurrentWorkspace(parts: List[scoring.Part],
                               backend: str = None,
                               renderoptions: scoring.render.RenderOptions = None
                               ) -> scoring.render.Renderer:
    """
    Render the given scoring.Parts with the current configuration

    Args:
        parts: the parts to render
        backend: the backend used (see currentConfig/'show.backend')
        renderoptions: if given, will override any option set in the currentConfig

    Returns:
        the rendered Renderer
    """
    workspace = activeWorkspace()
    cfg = workspace.config
    if not renderoptions:
        renderoptions = makeRenderOptionsFromConfig()
    quantizationProfile = makeQuantizationProfileFromConfig()
    backend = backend or cfg['show.backend']
    return scoring.render._quantizeAndRender(parts,
                                             backend=backend,
                                             struct=workspace.scorestruct,
                                             options=renderoptions,
                                             quantizationProfile=quantizationProfile)