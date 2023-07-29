from __future__ import annotations
from maelzel.core.workspace import Workspace
from maelzel.core import F, Note, asEvent

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.core._typedefs import time_t


__all__ = (
    'b2t',
    'l2b',
    'b2l',
    'N',
    'Ev'
)


def b2t(beat: time_t) -> F:
    """Shortcut for beatToTime"""
    return Workspace.active.scorestruct.beatToTime(beat)


def l2b(measureindex: int, beat: time_t = 0.) -> F:
    """Shortcut for locationToBeat"""
    return Workspace.active.scorestruct.locationToBeat(measure=measureindex, beat=beat)


def b2l(beat: time_t) -> tuple[int, F]:
    """Shortcut for beatToLocation"""
    return Workspace.active.scorestruct.beatToLocation(beat)


N = Note
Ev = asEvent
