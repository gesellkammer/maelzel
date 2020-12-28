import music21 as m21
from functools import lru_cache
from .common import *
from maelzel import scoring
from ._base import Opt, List
from .state import currentConfig


@lru_cache()
def makeScoreStructure(timesig=(4, 4), quarterTempo=60) -> scoring.ScoreStructure:
    return scoring.ScoreStructure.fromTimesig(timesig, quarterTempo=quarterTempo)


def scoringPartToMusic21(part: U[ scoring.Part, List[scoring.Notation] ],
                         struct: Opt[scoring.ScoreStructure] = None,
                         showCents=None,
                         divsPerSemitone=None
                         ) -> U[m21.stream.Score, m21.stream.Part]:
    """
    Creates a m21 Part from the given scoring events accoring to
    the options in the configuration. Some options can be overriden
    here.
    Assumes that the events fit in one Part. If you need to split the events across
    multiple staffs, use scoring.s

    Args:
        part: the events to convert
        struct: the score structure used.
        showCents: show cents as text
        divsPerSemitone: divisions of the semitone

    Returns:
        a music21 Part

    """
    m21score = scoringPartsToMusic21([part], struct=struct,
                                     showCents=showCents,
                                     divsPerSemitone=divsPerSemitone)
    assert len(m21score.parts) == 1
    return m21score.parts[0]


def scoringPartsToMusic21(parts: List[U[scoring.Part, List[scoring.Notation]]],
                          struct: Opt[scoring.ScoreStructure] = None,
                          showCents:bool=None,
                          divsPerSemitone:int=None,
                          config:dict=None
                          ) -> U[m21.stream.Score]:
    config = config or currentConfig()
    divsPerSemitone = (divsPerSemitone if divsPerSemitone is not None else
                       config['show.semitoneDivisions'])
    if showCents is None: showCents = config['show.cents']
    centsFontSize = config['show.centsFontSize']
    if struct is None:
        state = getState()
        struct = scoring.ScoreStructure.fromTimesig((4, 4), quarterTempo=state.tempo)
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
    assert isinstance(renderer, scoring.render.Music21Renderer)
    m21score = renderer.nativeScore()
    return m21score


