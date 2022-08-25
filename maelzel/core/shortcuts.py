from __future__ import annotations
from maelzel.core.workspace import Workspace

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numbers import Rational
    from maelzel.core._typedefs import time_t

def b2t(beat: time_t) -> Rational:
    return Workspace.active.scorestruct.beatToTime(beat)

def l2b(measureindex: int, beat: time_t = 0.) -> Rational:
    return Workspace.active.scorestruct.locationToBeat(measure=measureindex, beat=beat)
