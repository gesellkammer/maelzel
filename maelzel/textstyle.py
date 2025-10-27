from __future__ import annotations
from dataclasses import dataclass
from functools import cache
from maelzel import colortheory


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
    """One of 'above' or 'below'"""

    color: str = ''

    family: str = ''
    """One or multiple font families, separated by a comma"""

    def __post_init__(self):
        if self.box:
            assert self.box in ('square', 'rectangle', 'circle')

        if self.placement:
            assert self.placement in ('above', 'below')

        if self.color:
            assert colortheory.isValidCssColor(self.color)

    @staticmethod
    def validate(style: str) -> str:
        return validateStyle(style)

    @staticmethod
    def parse(style: str, separator=';') -> TextStyle:
        return parseStyle(style, separator=separator)


def validateStyle(style: str) -> str:
    """
    Check that the style is ok

    Args:
        style: the style to check

    Returns:
        An error message as string, or an empty string True if the style is defined properly
    """
    try:
        _ = parseStyle(style)
        return ''
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
    box        rectangle, circle, square   bold; box=square
    color      A CSS color                 italic;color=blue
    placement  above, below                bold; placement=above
    family     Any font name               fontsize=10; family=Helvetica,sans-serif
    bold       **flag**                    bold; color=#ff0000
    italic     **flag**                    bold; italic
    =========  ==========================  ====================

    Args:
        style: the style to parse; a set of key-value pairs or flags, separated
            by *separator`. Spaces are stripped
        separator: the separator used

    Returns:
        a TextStyle

    """
    # Keys need a value
    validkeys = ('fontsize', 'box', 'color', 'placement', 'family')

    # Flags don't
    validflags = ('italic', 'bold')

    convertions = {
        'fontsize': float,
    }

    key2choices = {
        'box': ('rectangle', 'circle', 'square'),
        'placement': ('above', 'below')
    }

    parts = style.split(separator)
    attrs = {}
    for part in parts:
        if '=' in part:
            key, value = part.split('=')
            key = key.strip()
            if key in attrs:
                raise ValueError(f"Duplicate attribute '{key}'")

            value = value.strip()
            if key not in validkeys:
                raise ValueError(f"Invalid key '{key}' in style '{style}'. Valid keys are {validkeys}")
            if (convertfunc := convertions.get(key)) is not None:
                value = convertfunc(value)
            if (choices := key2choices.get(key)) is not None:
                if value not in choices:
                    raise ValueError(f"Invalid value '{value}' for key {key}. Valid values are {choices}")
            elif key == 'color':
                if not colortheory.isValidCssColor(value):
                    raise ValueError(f"Invalid color '{value}'")
            attrs[key] = value
        else:
            flag = part.strip()
            if flag in attrs:
                raise ValueError(f"Duplicate key '{flag}'")
            if flag not in validflags:
                if flag in validkeys:
                    msg = f"Key '{flag}' needs a value in the form {flag}=<value>"
                    if (choices := key2choices.get(flag)) is not None:
                        msg += f". Possible values: {choices}"
                    raise ValueError(msg)
                raise ValueError(f"Invalid flag '{flag}' in style '{style}'. Valid flags are {validflags}")
            attrs[flag] = True
    return TextStyle(**attrs)
