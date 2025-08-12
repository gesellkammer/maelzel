from __future__ import annotations
from dataclasses import dataclass, replace as _replace
from functools import cache

from typing_extensions import Self


@dataclass
class TranscriptionOptions:
    """
    Options used for transcription

    Args:
        addGliss: add a glissando between parts of a same note group
        addAccents: add an accent to breakpoints with detected transient
        addSlurs: add a slur around notes within a group
        unvoicedNotehead: nothead used for unpitched notes or an empty string to
            leave such notes unmodified
        unvoicedPitch: pitch used for note groups where no pitch was detected
        unvoicedMinAmpDb: min. amp for an unvoiced breakpoint to be transcribed
        a4: reference frequency
        simplify: simplify breakpoints, 0 disables simplification
        maxDensity: max breakpoint density, 0 disables simplification

    """

    addGliss: bool = True
    """if True, add a gliss. symbol between parts of a same note group"""

    addAccents: bool = True
    """add an accent symbol to breakpoints with a detected accent"""

    addSlurs: bool = True
    """add a slur encompasing notes within a group"""

    unvoicedNotehead: str = 'x'
    """Notehead used for unpitched notes or an empty string to leave such notes unmodified"""

    unvoicedPitch: str | int = "5C"
    """
    The pitch used for note groups where no pitch was detected. For
    unpitched breakpoints within a group where other pitched breakpoints were
    found the next pitched found will be used.
    """

    unvoicedMinAmpDb: float = -80.
    """
    Breakpoints within an unvoiced group with amp. less than this will not be transcribed.
    """

    a4: float = 442.
    """Reference frequency"""

    simplify: float = 0.
    """
    Simplify breakpoints. 0 disables simplification
    """

    maxDensity: float = 0.
    """
    max. breakpoint density. 0 disables simplification
    """

    debug: bool = False
    """
    Debug transcription process
    """

    def __post_init__(self):
        assert isinstance(self.addGliss, bool)
        assert isinstance(self.addAccents, bool)
        assert isinstance(self.unvoicedNotehead, str)
        assert isinstance(self.unvoicedPitch, (str, int))
        assert isinstance(self.unvoicedMinAmpDb, (int, float))
        assert isinstance(self.a4, (int, float)) and 432 < self.a4 < 460

    def copy(self) -> Self:
        return _replace(self)
    
    @cache
    @staticmethod
    def default() -> TranscriptionOptions:
        return TranscriptionOptions()
