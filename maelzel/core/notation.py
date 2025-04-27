"""
Functionality to interface with `maelzel.scoring`

"""
from __future__ import annotations

from functools import cache

from maelzel.common import asF
from maelzel.textstyle import TextStyle

from .workspace import Workspace

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel import scoring
    import maelzel.scoring.enharmonics as enharmonics
    import maelzel.scoring.render as render
    from maelzel.scorestruct import ScoreStruct
    from maelzel.scoring import quant
    from .config import CoreConfig


def makeEnharmonicOptionsFromConfig(cfg: CoreConfig) -> enharmonics.EnharmonicOptions:
    """
    Generate EnharmonicOptions needed for respelling during quantization

    Respelling happens at the notation level, before quantization

    Args:
        cfg: the CoreConfig

    Returns:
        a scoring.enharmonics.EnharmonicOptions

    """
    renderoptions = cfg.makeRenderOptions()
    return renderoptions.makeEnharmonicOptions()


def makeRenderOptionsFromConfig(cfg: CoreConfig = None,
                                ) -> render.RenderOptions:
    """
    Generate RenderOptions needed for `scoring.render` based on the config

    Args:
        the config to use. If None, the current config is used

    Returns:
        a scoring.render.RenderOptions used to render parts
        via `scoring.render` module
    """
    if cfg is None:
        cfg = Workspace.active.config

    centsAnnotationStyle = TextStyle.parse(cfg['show.centsAnnotationStyle'])

    from maelzel.scoring import render
    renderOptions = render.RenderOptions(
        centsAnnotationFontsize=centsAnnotationStyle.fontsize or 8,
        centsAnnotationPlacement=centsAnnotationStyle.placement or 'above',
        centsAnnotationPlusSign=cfg['.show.centsAnnotationPlusSign'],
        divsPerSemitone=cfg['semitoneDivisions'],
        enharmonicDebug=cfg['.enharmonic.debug'],
        enharmonicHorizontalWeight=cfg['enharmonic.horizontalWeight'],
        enharmonicThreeQuarterMicrotonePenalty=cfg['.enharmonic.threeQuarterMicrotonePenalty'],
        enharmonicVerticalWeight=cfg['enharmonic.verticalWeight'],
        glissLineThickness=cfg['show.glissLineThickness'],
        glissHideTiedNotes=cfg['show.glissHideTiedNotes'],
        glissLineType=cfg['show.glissLineType'],
        horizontalSpacing=cfg['show.horizontalSpacing'],
        lilypondBinary=cfg['lilypondpath'],
        lilypondGlissandoMinimumLength=cfg['show.lilypondGlissandoMinimumLength'],
        lilypondPngStaffsizeScale=cfg['show.lilypondPngStaffsizeScale'],
        measureAnnotationStyle=cfg['show.measureAnnotationStyle'],
        musescoreBinary=cfg['musescorepath'],
        noteLabelStyle=cfg['show.labelStyle'],
        orientation=cfg['show.pageOrientation'],
        pageMarginMillimeters=cfg['show.pageMarginMillimeters'],
        pageSize=cfg['show.pageSize'],
        pngResolution=cfg['show.pngResolution'],
        rehearsalMarkStyle=cfg['show.rehearsalMarkStyle'],
        renderFormat=cfg['show.format'],
        respellPitches=cfg['show.respellPitches'],
        showCents=cfg['show.centsDeviationAsTextAnnotation'],
        staffSize=cfg['show.staffSize'],
        referenceStaffsize=cfg['show.referenceStaffsize'],
        autoClefChanges=cfg['show.autoClefChanges'],
        keepClefBiasFactor=cfg['.show.keepClefBiasFactor'],
        autoClefChangesWindow=cfg['.show.autoClefChangesWindow'],
        musicxmlFontScaling=cfg['show.musicxmlFontScaling'],
        centsAnnotationSnap=cfg['show.centsAnnotationSnap'],
        proportionalSpacing=cfg['show.proportionalSpacing'],
        proportionalNotationDuration=cfg['show.proportionalNotationDuration'],
        proportionalSpacingKind=cfg['show.proportionalSpacingKind'],
        flagStyle=cfg['show.flagStyle']
    )
    return renderOptions


@cache
def makeQuantizationProfileFromConfig(cfg: CoreConfig
                                      ) -> quant.QuantizationProfile:
    """
    Creates a scoring.quant.QuantizationProfile from a config

    Args:
        cfg: a CoreConfig

    Returns:
        a scoring.quant.QuantizationProfile
    """
    nestedTuplets = cfg['quant.nestedTuplets']
    if nestedTuplets is None:
        if cfg['show.backend'] == 'musicxml':
            nestedTuplets = cfg['quant.nestedTupletsInMusicxml']
        else:
            nestedTuplets = True

    kws = {}
    if (gridWeight := cfg['.quant.gridErrorWeight']) is not None:
        kws['gridErrorWeight'] = gridWeight
    if (divisionWeight := cfg['.quant.divisionErrorWeight']) is not None:
        kws['divisionErrorWeight'] = divisionWeight
    if (rhythmWeight := cfg['.quant.rhythmComplexityWeight']) is not None:
        kws['rhythmComplexityWeight'] = rhythmWeight
    if (gridErrorExp := cfg['.quant.gridErrorExp']) is not None:
        kws['gridErrorExp'] = gridErrorExp

    from maelzel.scoring import quant
    return quant.QuantizationProfile.fromPreset(
        complexity=cfg['quant.complexity'],
        nestedTuplets=nestedTuplets,
        debug=cfg['.quant.debug'],
        debugMaxDivisions = cfg['.quant.debugShowNumRows'],
        syncopationMinBeatFraction = asF(cfg['quant.syncopationMinBeatFraction']),
        syncopationMaxAsymmetry = cfg['quant.syncopationMaxAsymmetry'],
        breakSyncopationsLevel = cfg['quant.breakSyncopationsLevel'],
        breakLongGlissandi = cfg['show.glissHideTiedNotes'],
        **kws
    )


def renderWithActiveWorkspace(parts: list[scoring.core.UnquantizedPart],
                              backend: str = None,
                              renderoptions: render.RenderOptions = None,
                              scorestruct: ScoreStruct = None,
                              config: CoreConfig = None,
                              quantizationProfile: quant.QuantizationProfile = None
                              ) -> render.Renderer:
    """
    Render the given scoring.UnquantizedParts with the current configuration

    Args:
        parts: the parts to render
        backend: the backend used (see currentConfig/'show.backend')
        renderoptions: if given, will override any option set in the currentConfig
        scorestruct: if given, override the active ScoreStruct
        config: if given, override the ative config
        quantizationProfile: if given, use this for quantization

    Returns:
        the rendered Renderer
    """
    workspace = Workspace.active
    if not config:
        config = workspace.config
    if backend != config['show.backend']:
        config = config.clone({'show.backend': backend})
    if not renderoptions:
        renderoptions = config.makeRenderOptions()
    if not quantizationProfile:
        quantizationProfile = config.makeQuantizationProfile()
    if backend:
        renderoptions.backend = backend
    if scorestruct is None:
        scorestruct = workspace.scorestruct
    from maelzel import scoring
    from maelzel.scoring import render
    if config['show.hideRedundantDynamics']:
        for part in parts:
            scoring.core.removeRedundantDynamics(part.notations)
    return render.quantizeAndRender(parts,
                                    struct=scorestruct,
                                    options=renderoptions,
                                    quantizationProfile=quantizationProfile)
