"""
Utils used by coreconfig. Cannot import from anything within maelzel.core
"""

from maelzel.common import F


def isValidFraction(obj) -> bool:
    """
    True if obj can be interpreted as Fraction
    """
    try:
        _ = F(obj)
        return True
    except Exception:
        return False
