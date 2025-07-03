from __future__ import annotations

from dataclasses import dataclass
from maelzel.common import F

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from maelzel.scoring.common import division_t


@dataclass
class QuantizedBeatDef:
    """
    Represent a quantized beat without the actual data
    """
    offset: F
    duration: F = F(1)
    division: division_t = (1,)
    weight: int = 0

    @property
    def end(self) -> F:
        return self.offset + self.duration
