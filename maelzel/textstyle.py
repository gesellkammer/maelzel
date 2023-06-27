from __future__ import annotations
from dataclasses import dataclass
from functools import cache


@dataclass
class TextStyle:
    """
    A class defining text style

    """
    fontsize: float | None = None
    """The absolute size"""

    box: str = ''
    """Text enclosure, one of '', 'rectangle', 'circle', 'square' """

    bold: bool = False
    """Is this text bold?"""

    italic: bool = False
    """Is the text italic"""

    placement: str = ''

    color: str = ''

    def __post_init__(self):
        if self.box:
            assert self.box in ('square', 'rectangle', 'circle')

        if self.placement:
            assert self.placement in ('above', 'below')


def validateStyle(style: str) -> bool:
    """
    Check that the style is ok

    Args:
        style: the style to check

    Returns:
        True if the style is defined properly
    """
    try:
        _ = parseTextStyle(style)
        return True
    except ValueError:
        return False


@cache
def parseTextStyle(style: str, separator=';') -> TextStyle:
    """
    Parse an ad-hoc format to create a TextStyle

    Possible formats::

        'box=square; bold; italic'
        'bold=true; italic=true'


    Args:
        style: the style to parse
        separator: the separator used

    Returns:
        a TextStyle

    """
    convertions = {
        'fontsize': float,
        'bold': lambda value: value.lower() == 'true',
        'italic': lambda value: value.lower() == 'true',
        'box': None,
        'color': None,
    }
    validkeys = convertions.keys()

    parts = style.split(separator)
    attrs = {}
    for part in parts:
        if '=' in part:
            key, value = part.split('=')
            key = key.strip()
            value = value.strip()
            if (convertfunc := convertions.get(key)) is not None:
                value = convertfunc(value)
        else:
            key, value = part.strip(), True
        if not key in validkeys:
            raise ValueError(f"Invalid key '{key}' in style '{style}'")
        attrs[key] = value
    return TextStyle(**attrs)