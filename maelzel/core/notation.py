"""
Functionality to interface with maelzel.scoring

"""
from __future__ import annotations
from ._common import *
from .config import CoreConfig
from .workspace import getConfig, getWorkspace

from maelzel import scoring
from maelzel.scorestruct import ScoreStruct

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import music21 as m21
    import scoring.enharmonics


def scoringPartToMusic21(part: scoring.Part | list[scoring.Notation],
                         struct: ScoreStruct = None,
                         config: dict = None
                         ) -> m21.stream.Score | m21.stream.Part:
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


def scoringPartsToMusic21(parts: list[scoring.Part | list[scoring.Notation]],
                          struct: ScoreStruct = None,
                          config: dict = None
                          ) -> m21.stream.Score:
    """
    Render the given scoring Parts as music21

    Args:
        parts: the parts to render
        struct: the score structure
        config ():

    Returns:

    """
    config = config or getConfig()
    divsPerSemitone = config['show.semitoneDivisions']
    showCents = config['show.centsDeviationAsTextAnnotation']
    centsFontSize = config['show.centsFontSize']
    if struct is None:
        struct = getWorkspace().scorestruct
    renderOptions = scoring.render.RenderOptions(divsPerSemitone=divsPerSemitone,
                                                 showCents=showCents,
                                                 centsFontSize=centsFontSize)
    quantProfile = scoring.quant.QuantizationProfile(nestedTuplets=False)
    for part in parts:
        scoring.stackNotationsInPlace(part)
    renderer = scoring.render.quantizeAndRender(parts, struct=struct,
                                                options=renderOptions,
                                                backend="music21",
                                                quantizationProfile=quantProfile)
    m21score = renderer.asMusic21()
    return m21score


def makeEnharmonicOptionsFromConfig(cfg: CoreConfig) -> scoring.enharmonics.EnharmonicOptions:
    """
    Generate EnharmonicOptions needed for respelling during quantization

    Respelling happens at the notation level, before quantization

    Args:
        cfg: the CoreConfig

    Returns:
        a scoring.enharmonics.EnharmonicOptions

    """
    from maelzel.scoring.enharmonics import EnharmonicOptions
    return EnharmonicOptions(
        debug=cfg['enharmonicSpellingDebug'],
        horizontalWeight=cfg['enharmonicSpellingHorizontalWeight'],
        verticalWeight=cfg['enharmonicSpellingVerticalWeight']
    )


def makeRenderOptionsFromConfig(cfg: CoreConfig = None,
                                ) -> scoring.render.RenderOptions:
    """
    Generate RenderOptions needed for scoring.render based on the config

    Args:
        the config to use. If None, the current config is used

    Returns:
        a scoring.render.RenderOptions used to render parts
        via scoring.render module
    """
    if cfg is None:
        cfg = getConfig()
    renderOptions = scoring.render.RenderOptions(
        staffSize=cfg['show.staffSize'],
        divsPerSemitone=cfg['semitoneDivisions'],
        showCents=cfg['show.centsDeviationAsTextAnnotation'],
        centsFontSize=cfg['show.centsAnnotationFontSize'],
        noteAnnotationsFontSize=cfg['show.labelFontSize'],
        pageSize = cfg['show.pageSize'],
        orientation= cfg['show.pageOrientation'],
        pageMarginMillimeters=cfg['show.pageMarginMillimeters'],
        measureAnnotationFontSize=cfg['show.measureAnnotationFontSize'],
        respellPitches=cfg['show.respellPitches'],
        glissLineThickness=cfg['show.glissLineThickness'],
        lilypondPngStaffsizeScale=cfg['show.lilypondPngStaffsizeScale'],
        lilypondGlissandoMinimumLength=cfg['show.lilypondGlissandoMinimumLength'],
        glissHideTiedNotes=cfg['show.glissHideTiedNotes'],
        renderFormat=cfg['show.format'],
        pngResolution=cfg['show.pngResolution'],
        horizontalSpacing=cfg['show.horizontalSpacing'],
        enharmonicsDebug=cfg['enharmonicSpellingDebug'],
        enharmonicSpellingHorizontalWeight=cfg['enharmonicSpellingHorizontalWeight'],
        enharmonicSpellingVerticalWeight=cfg['enharmonicSpellingVerticalWeight']
    )
    return renderOptions


def makeQuantizationProfileFromConfig(cfg: CoreConfig = None
                                      ) -> scoring.quant.QuantizationProfile:
    """
    Creates a scoring.quant.QuantizationProfile from a preset

    Args:
        cfg: a CoreConfig

    Returns:
        a scoring.quant.QuantizationProfile
    """
    if cfg is None:
        cfg = getConfig()
    profile = scoring.quant.QuantizationProfile.fromPreset(complexity=cfg['quant.complexity'],
                                                           nestedTuplets=cfg['quant.nestedTuplets'])
    profile.debug = cfg['quant.debug']
    profile.debugMaxDivisions = cfg['quant.debugShowNumRows']

    if (gridWeight:=cfg['quant.gridErrorWeight']) is not None:
        profile.gridErrorWeight = gridWeight
    if (divisionWeight:=cfg['quant.divisionErrorWeight']) is not None:
        profile.divisionErrorWeight = divisionWeight
    if (rhythmWeight:=cfg['quant.rhythmComplexityWeight']) is not None:
        profile.rhythmComplexityWeight = rhythmWeight
    if (gridErrorExp:=cfg['quant.gridErrorExp']) is not None:
        profile.gridErrorExp = gridErrorExp

    profile.minBeatFractionAcrossBeats = cfg['quant.minBeatFractionAcrossBeats']
    return profile


def renderWithActiveWorkspace(parts: list[scoring.Part],
                              backend: str = None,
                              renderoptions: scoring.render.RenderOptions = None,
                              scorestruct: ScoreStruct = None,
                              config: CoreConfig = None,
                              quantizationProfile: scoring.quant.QuantizationProfile = None
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
    workspace = getWorkspace()
    if not config:
        config = workspace.config
    if not renderoptions:
        renderoptions = config.makeRenderOptions()
    if not quantizationProfile:
        quantizationProfile = config.makeQuantizationProfile()
    if backend:
        renderoptions.backend = backend
    if scorestruct is None:
        scorestruct = workspace.scorestruct
    if config['show.hideRedundantDynamics']:
        for part in parts:
            scoring.core.removeRedundantDynamics(part)
    return scoring.render.quantizeAndRender(parts,
                                            struct=scorestruct,
                                            options=renderoptions,
                                            quantizationProfile=quantizationProfile)