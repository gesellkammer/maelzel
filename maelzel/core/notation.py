"""
Functionality to interface with `maelzel.scoring`

"""
from __future__ import annotations

from maelzel.common import asF
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


def makeRenderOptionsFromConfig(cfg: CoreConfig | None = None,
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

    from maelzel.textstyle import TextStyle
    centsTextStyle = TextStyle.parse(cfg['show.centsTextStyle'])

    from maelzel.scoring import render
    renderOptions = render.RenderOptions(
        centsAnnotationFontsize=centsTextStyle.fontsize or 8,
        centsAnnotationPlacement=centsTextStyle.placement or 'above',
        centsTextPlusSign=cfg['.show.centsTextPlusSign'],
        divsPerSemitone=cfg['semitoneDivisions'],
        enharmonicDebug=cfg['.enharmonic.debug'],
        enharmonicHorizontalWeight=cfg['enharmonic.horizontalWeight'],
        enharmonicThreeQuarterMicrotonePenalty=cfg['.enharmonic.150centMicroPenalty'],
        enharmonicVerticalWeight=cfg['enharmonic.verticalWeight'],
        glissLineThickness=cfg['show.glissLineThickness'],
        glissHideTiedNotes=cfg['show.glissHideTiedNotes'],
        glissLineType=cfg['show.glissLineType'],
        horizontalSpace=cfg['show.horizontalSpace'],
        lilypondBinary=cfg['lilypondpath'],
        lilypondGlissMinLength=cfg['show.lilypondGlissMinLength'],
        lilypondPngStaffsizeScale=cfg['show.lilypondPngStaffsizeScale'],
        measureLabelStyle=cfg['show.measureLabelStyle'],
        musescoreBinary=cfg['musescorepath'],
        noteLabelStyle=cfg['show.labelStyle'],
        orientation=cfg['show.pageOrientation'],
        pageMarginMillimeters=cfg['show.pageMarginMillim'],
        pageSize=cfg['show.pageSize'],
        pngResolution=cfg['show.pngResolution'],
        rehearsalMarkStyle=cfg['show.rehearsalMarkStyle'],
        renderFormat=cfg['show.format'],
        respellPitches=cfg['show.respellPitches'],
        showCents=cfg['show.cents'],
        staffSize=cfg['show.staffSize'],
        referenceStaffsize=cfg['show.referenceStaffsize'],
        autoClefChanges=cfg['show.autoClefChanges'],
        keepClefBiasFactor=cfg['show.keepClefBias'],
        autoClefChangesWindow=cfg['show.clefChangesWindow'],
        clefSimplifyThreshold=cfg['show.clefSimplify'],
        musicxmlFontScaling=cfg['show.musicxmlFontScaling'],
        centsTextSnap=cfg['show.centsTextSnap'],
        proportionalSpacing=cfg['show.spacing'] != "normal",
        proportionalNotationDuration=cfg['show.proportionalDuration'],
        proportionalSpacingKind=cfg['show.spacing'],
        flagStyle=cfg['show.flagStyle'],
        removeRedundantDynamics=cfg['show.hideRedundantDynamics'],
        dynamicsResetTime=cfg['.show.dynamicsResetTime'],
        dynamicsResetAfterEmptyMeasure=cfg['.show.dynamicsResetAfterEmptyMeasure'],
        dynamicsResetAfterRest=cfg['.show.dynamicsResetAfterRest'],
    )
    return renderOptions


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
    if nestedTuplets is None and cfg['show.backend'] == 'musicxml':
        nestedTuplets = cfg['quant.nestedTupletsMusicxml']

    kws = {}
    if (gridWeight := cfg['quant.gridWeight']) is not None:
        kws['gridErrorWeight'] = gridWeight
    
    if (divisionWeight := cfg['.quant.divisionWeight']) is not None:
        kws['divisionErrorWeight'] = divisionWeight
    if (rhythmWeight := cfg['.quant.complexityWeight']) is not None:
        kws['rhythmComplexityWeight'] = rhythmWeight
    if (gridErrorExp := cfg['.quant.gridErrorExp']) is not None:
        kws['gridErrorExp'] = gridErrorExp

    from maelzel.scoring import quant
    return quant.QuantizationProfile.fromPreset(
        complexity=cfg['quant.complexity'],
        nestedTuplets=nestedTuplets,
        debug=cfg['.quant.debug'],
        debugMaxDivisions = cfg['.quant.debugShowNumRows'],
        syncopMinFraction = asF(cfg['quant.syncopMinFraction']),
        syncopPartMinFraction = asF(cfg['quant.syncopPartMinFraction']),
        syncopMaxAsymmetry = cfg['quant.syncopMaxAsymmetry'],
        syncopExcludeSymDurs = cfg['quant.syncopExcludeSymDurs'],
        breakSyncopationsLevel = cfg['quant.breakBeats'],
        breakLongGlissandi = cfg['show.glissHideTiedNotes'],
        beatWeightTempoThresh = cfg['quant.beatWeightTempoThresh'],
        subdivTempoThresh = cfg['quant.subdivTempoThresh'],
        mergeTupletsDifferentDur = cfg['.quant.mergeTupletsDifferentDur'],
        **kws
    )


def renderWithActiveWorkspace(parts: list[scoring.core.UnquantizedPart],
                              backend='',
                              renderoptions: render.RenderOptions | None = None,
                              scorestruct: ScoreStruct | None = None,
                              config: CoreConfig | None = None,
                              quantizationProfile: quant.QuantizationProfile | None = None
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
    if backend and backend != config['show.backend']:
        config = config.clone({'show.backend': backend})
    if not renderoptions:
        renderoptions = config.makeRenderOptions()
    if not quantizationProfile:
        quantizationProfile = config.makeQuantizationProfile()
    if backend:
        assert renderoptions.backend == backend
    if scorestruct is None:
        scorestruct = workspace.scorestruct
    from maelzel.scoring import render
    return render.quantizeAndRender(parts,
                                    struct=scorestruct,
                                    options=renderoptions,
                                    quantizationProfile=quantizationProfile)
