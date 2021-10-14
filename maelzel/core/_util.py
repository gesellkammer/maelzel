"""
Internal utilities
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from functools import cache
import PIL
import sys
if TYPE_CHECKING:
    from typing import *


@cache
def buildingDocumentation() -> bool:
    return "sphinx" in sys.modules


def checkBuildingDocumentation(logger=None) -> bool:
    building = buildingDocumentation()
    if building:
        msg = "Not available while building documentation"
        if logger:
            logger.error(msg)
        else:
            print(msg)
    return building


def imgSize(path:str) -> Tuple[int, int]:
    """ returns (width, height) """
    im = PIL.Image.open(path)
    return im.size