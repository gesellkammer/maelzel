from __future__ import annotations
from dataclasses import dataclass, replace as _dataclassreplace
from maelzel.scoring import enharmonics
import emlib.misc


@dataclass
class RenderOptions:
    """
    Holds all options needed for rendering

    Attributes:
        orientation: one of "portrait" or "landscape"
        staffSize: the size of each staff, in points
        pageSize: one of "a1", "a2", "a3", "a4", "a5"
        pageMarginMillimeters: page margin in mm. Only used by some backends
        divsPerSemitone: the number of divisions of the semitone
        showCents: should each note/chord have a text label attached indicating
        the cents deviation from the nearest semitone?
        centsPlacement: where to put the cents annotation
        centsFontSizeFactor: the factor used for the font size used in cents annotation
        measureAnnotationFontSize: font size for measure annotations
        glissAllowNonContiguous: if True, allow glissandi between notes which have rests
            between them
        glissHideTiedNotes: if True, hide tied notes which are part of a gliss.
        lilypondPngStaffsizeScale: a scaling factor applied to staff size when rendering
            to png via lilypond.
        pngResolution: dpi used when rendering a lilypond score to png
        title: the title of the score
        composer: the composer of the score
        opaque: if True, rendered images will be opaque (no transparent
            background)
        articulationsWithinTies: if True, include any articulation even if the note if
            tied to a previous note
        dynamicsWithinTies: include dynamics even for notes tied to previous notes
        backend: default rendering backend, one of 'lilypond', 'music21'

    """
    orientation: str = 'portrait'
    staffSize: int | float = 12.0
    pageSize: str = 'a4'
    pageMarginMillimeters: int = 4

    divsPerSemitone: int = 4
    showCents: bool = False
    centsPlacement: str = "above"
    centsFontSize: int | float = 10

    measureAnnotationFontSize: int | float = 12
    measureAnnotationBox: str = 'square'

    noteAnnotationsFontSize: int | float = 10

    glissAllowNonContiguous: bool = False
    glissHideTiedNotes: bool = True

    horizontalSpacing: str = 'large'
    pngResolution: int = 200
    removeSuperfluousDynamics: bool = True
    restsResetDynamics: bool = True

    respellPitches: bool = True
    glissLineThickness: int = 1

    renderFormat: str = ''

    cropToContent: bool = False
    opaque: bool = True

    articulationInsideTie: bool = True
    dynamicInsideTie: bool = True

    rehearsalMarkFontSize: int | float = 13
    rehearsalMarkBoxed: bool = True

    enharmonicGroupSize: int = 6
    enharmonicStep: int = 3
    enharmonicDebug: bool = False
    enharmonicHorizontalWeight: float = 1.
    enharmonicVerticalWeight: float = 0.05
    enharmonicThreeQuarterMicrotonePenalty: float = 100

    backend: str = 'lilypond'

    title: str = ''
    composer: str = ''

    # Options only relevant for lilypond render
    lilypondPngStaffsizeScale: float = 1.4
    lilypondGlissandoMinimumLength: int = 5

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: RenderOptions) -> bool:
        return isinstance(other, RenderOptions) and hash(self) == hash(other)

    def __post_init__(self):
        self.check()

    def clone(self, **changes) -> RenderOptions:
        """
        Clone this RenderOptions with the given changes

        Args:
            **changes: any attribute accepted by this RenderOptions

        Returns:
            a new RenderOptions with the changes applied

        Example
        ~~~~~~~

            >>> defaultoptions = RenderOptions()
            >>> modified = defaultoptions.clone(pageSize='A3')

        """
        out = _dataclassreplace(self, **changes)
        out.check()
        return out

    def check(self) -> None:
        assert self.orientation in ('portrait', 'landscape')
        assert isinstance(self.staffSize, (int, float)) and 2 < self.staffSize < 40, \
            f"Invalid staffSize: {self.staffSize}"
        assert self.pageSize.lower() in ('a1', 'a2', 'a3', 'a4', 'a5'), \
            f"Invalid page size, it must be one of a1, a2, ..., a5"
        heightmm, widthmm = emlib.misc.page_dinsize_to_mm(self.pageSize, self.orientation)
        assert isinstance(self.pageMarginMillimeters, int) and 0 <= self.pageMarginMillimeters <= widthmm
        assert self.divsPerSemitone in (1, 2, 4)
        assert self.centsPlacement in ('above', 'below')
        assert self.horizontalSpacing in ('small', 'medium', 'large', 'xlarge', 'default')
        assert self.backend in ('lilypond', 'musicxml')

    def makeEnharmonicOptions(self) -> enharmonics.EnharmonicOptions:
        return enharmonics.EnharmonicOptions(groupSize=self.enharmonicGroupSize,
                                             groupStep=self.enharmonicStep,
                                             debug=self.enharmonicDebug,
                                             threeQuarterMicrotonePenalty=self.enharmonicThreeQuarterMicrotonePenalty,
                                             horizontalWeight=self.enharmonicHorizontalWeight,
                                             verticalWeight=self.enharmonicVerticalWeight)
