"""
Utils used by coreconfig. Cannot import from anything within maelzel.core
"""

from maelzel.common import F
from maelzel.textstyle import TextStyle


def isValidFraction(cfg, key, val) -> bool:
    """
    True if val can be interpreted as Fraction
    """
    try:
        _ = F(val)
        return True
    except ValueError:
        return False


def isValidStyle(cfg, key: str, val) -> bool:
    return TextStyle.validate(val) == ''
