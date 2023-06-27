from __future__ import annotations
from dataclasses import dataclass, replace as _dataclassreplace
from maelzel.scoring import enharmonics
from maelzel import textstyle
import emlib.misc


@dataclass
class RenderOptions:
    """
    Holds all options needed for rendering

    """
    orientation: str = 'portrait'
    """One of portrait, landscape"""

    staffSize: int | float = 12.0
    """The size of each staff, in points"""

    pageSize: str = 'a4'
    """The page size, on of a1, a2, a3, a4, a5"""

    pageMarginMillimeters: int = 4
    """Page margin in mm"""

    divsPerSemitone: int = 4
    """Number of divisions of the semitone"""

    showCents: bool = False
    """Show cents deviation as a text annotation"""

    centsAnnotationPlacement: str = "above"
    """Placement of the cents text annotation (above | below)"""

    centsAnnotationFontsize: int | float = 10
    """Fontsize of the cents text annotation"""

    noteLabelStyle: str = 'fontsize=10'
    """Style applied to labels"""

    glissAllowNonContiguous: bool = False
    """Allow non contiguous gliss"""

    glissHideTiedNotes: bool = True
    """Hide the notehead of intermediate notes within a glissando"""

    horizontalSpacing: str = 'large'
    """The horizontal spacing (large | medium | small). Only used by lilypond backend"""

    pngResolution: int = 200
    """DPI resolution of generated png images"""

    removeSuperfluousDynamics: bool = True
    """If True, remove superfluous dynamics"""

    restsResetDynamics: bool = True
    """A rest resets the dynamic context so that a dynamic which is the same as the current
    dynamic will still be rendered after a rest"""

    respellPitches: bool = True
    """Find the best enharmonic representation"""

    glissLineThickness: int = 1
    """Thickness of the glissando line"""

    glissLineType: str = 'solid'
    """Line type used in a glissando (solid | wavy)"""

    renderFormat: str = ''
    """The format renderer (pdf | png)"""

    cropToContent: bool | None = None
    """Crop the rendered image to the contexts"""

    preview: bool = False
    """Only render a preview (only the first system)"""

    opaque: bool = True
    """Remove any transparency in the background (when rendering to png)"""

    articulationInsideTie: bool = True
    """Render articulations even if the note is tied"""

    dynamicInsideTie: bool = True
    """Render dynamics if the note is tied"""

    rehearsalMarkStyle: str = 'box=square; bold; fontsize=13'
    """Text style used for rehearsal marks"""

    measureAnnotationStyle: str = 'box=rectangle; fontsize=12'
    """Style used for measure annotations"""

    enharmonicGroupSize: int = 6
    """How much horizontal context to take into consideration when finding the best enharmonic
    spelling"""

    enharmonicStep: int = 3
    """The step size of the window when evaluating the best enharmonic spelling"""

    enharmonicDebug: bool = False
    """If True, show debug information regarding the enharmonic spelling process"""

    enharmonicHorizontalWeight: float = 1.
    """The weight of the horizontal dimension during the enharmonic spelling process"""

    enharmonicVerticalWeight: float = 0.05
    """The weight of the vertical dimension during the enharmonic spelling process"""

    enharmonicThreeQuarterMicrotonePenalty: float = 100
    """Penalty for spelling a note using three quarter microtones"""

    backend: str = 'lilypond'
    """Backend used for rendering (lilypond | musicxml)"""

    title: str = ''
    """The title of this score"""

    composer: str = ''
    """The composer of this score"""

    # Options only relevant for lilypond render
    lilypondPngStaffsizeScale: float = 1.4
    """Png staffsize scaling when rendering to lilypond"""

    lilypondGlissandoMinimumLength: int = 5
    """Min. length of the glissando line when rendering with lilypond. This is to avoid
    too short glissandi actually not showing at all"""

    lilypondBinary: str = ''
    """The lilypond binary used"""

    musescoreBinary: str = ''
    """The musescore binary"""

    musicxmlSolidSlideGliss: bool = True
    """If True, use a solid slide line for glissando in musicxml"""

    musicxmlIndent: str = '  '
    """The indentation used when rendering the xml in musicxml"""

    referenceStaffsize: float = 12.0
    """The reference staff size. This is used to convert staffsize
    to a scaling factor"""

    musicxmlTenths: int = 40
    """Tenths used when rendering to musicxml. This is a reference value"""

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: RenderOptions) -> bool:
        return isinstance(other, RenderOptions) and hash(self) == hash(other)

    def __post_init__(self):
        self.check()

    def musicxmlScaling(self) -> tuple[float, int]:
        """
        Maps mm to tenths when rendering musicxml

        The scaling is based on the staff size. This allows to use one
        setting for all backends
        """

        scaling = self.staffSize / 12.
        mm = 6.35 * scaling
        return (mm, self.musicxmlTenths)


    def copy(self) -> RenderOptions:
        """
        Copy this object

        Returns:
            a copy of this RenderOptions
        """
        return _dataclassreplace(self)

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
        assert self.centsAnnotationPlacement in ('above', 'below')
        assert self.horizontalSpacing in ('small', 'medium', 'large', 'xlarge', 'default')
        assert self.backend in ('lilypond', 'musicxml')

    def pageSizeMillimeters(self) -> tuple[float, float]:
        """
        Returns the page size in millimeters

        Returns:
            a tuple (height, width), where both height and width are measured
            in millimeters
        """
        return emlib.misc.page_dinsize_to_mm(self.pageSize, self.orientation)

    @staticmethod
    def parseTextStyle(style: str) -> textstyle.TextStyle:
        return textstyle.parseTextStyle(style)

    @property
    def parsedRehearsalMarkStyle(self) -> textstyle.TextStyle:
        """The style for rehearsal marks, parsed

        Returns:
            a :class:`maelzel.scoring.textstyle.TextStyle`
        """
        return textstyle.parseTextStyle(self.rehearsalMarkStyle)

    @property
    def parsedMeasureAnnotationStyle(self) -> textstyle.TextStyle:
        return textstyle.parseTextStyle(self.measureAnnotationStyle)


    def makeEnharmonicOptions(self) -> enharmonics.EnharmonicOptions:
        return enharmonics.EnharmonicOptions(groupSize=self.enharmonicGroupSize,
                                             groupStep=self.enharmonicStep,
                                             debug=self.enharmonicDebug,
                                             threeQuarterMicrotonePenalty=self.enharmonicThreeQuarterMicrotonePenalty,
                                             horizontalWeight=self.enharmonicHorizontalWeight,
                                             verticalWeight=self.enharmonicVerticalWeight)
