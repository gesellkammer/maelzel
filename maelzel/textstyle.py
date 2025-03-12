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

    @staticmethod
    def validate(style: str) -> bool | str:
        return validateStyle(style)

    @staticmethod
    def parse(style: str, separator=';') -> TextStyle:
        return parseStyle(style, separator=separator)


def validateStyle(style: str) -> bool | str:
    """
    Check that the style is ok

    Args:
        style: the style to check

    Returns:
        An error message as string, or an empty string True if the style is defined properly
    """
    try:
        _ = parseStyle(style)
        return True
    except ValueError as e:
        return f"Could not validate style: '{style}', error: '{e}'"


@cache
def parseStyle(style: str, separator=';') -> TextStyle:
    """
    Parse an ad-hoc format to create a TextStyle

    =========  ==========================  ====================
    Key        Possible Values             Example
    =========  ==========================  ====================
    fontsize   The size in some unit       fontsize=8; italic
    bold       **flag**                    bold; color=#ff0000
    italic     **flag**                    bold; italic
    box        rectangle, circle, square   bold; box=square
    color      A CSS color                 italic;color=blue
    placement  above, below                bold; placement=above
    =========  ==========================  ====================


    More examples::

        'box=square; bold; italic'
        'bold=true; italic=true'


    Args:
        style: the style to parse; a set of key-value pairs or flags, separated
            by *separator`. Spaces are stripped
        separator: the separator used

    Returns:
        a TextStyle

    """
    validkeys = ('fontsize', 'bold', 'italic', 'box', 'color', 'placement')

    convertions = {
        'fontsize': float,
        'bold': lambda value: value.lower() == 'true',
        'italic': lambda value: value.lower() == 'true',
    }

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
        if key not in validkeys:
            raise ValueError(f"Invalid key '{key}' in style '{style}'. Valid keys are {'validkeys'}")
        attrs[key] = value
    return TextStyle(**attrs)
