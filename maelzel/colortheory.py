from __future__ import annotations
import colorsys


# Colors which can work with light and dark backgrounds
# 1 is lighter, 2 is darker

safeColors = {
    'blue1': '#9090FF',
    'blue2': '#6666E0',
    'red1': '#FF9090',
    'red2': '#E08080',
    'green1': '#90FF90',
    'green2': '#8080E0',
    'magenta1': '#F090F0',
    'magenta2': '#E080E0',
    'cyan': '#70D0D0',
    'grey1': '#BBBBBB',
    'grey2': '#A0A0A0',
    'grey3': '#909090'
}


def luminosityFactor(color: tuple[float, float, float], factor: float
                     ) -> tuple[float, float, float]:
    """
    Apply a factor to the lumunisity of color, clamps the value between 0 and 1

    Args:
        color: the color as RGB float
        factor: the factor to apply

    Returns:
        the resulting color
    """
    c = colorsys.rgb_to_hls(*color)
    luminosity = c[1] * factor
    luminosity = max(min(luminosity, 1), 0)
    return colorsys.hls_to_rgb(c[0], luminosity, c[2])


def lightenColor(color: tuple[float, float, float], amount: float
                 ) -> tuple[float, float, float]:
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.

    Input must be an RGB tuple where each channel is between 0 and 1

    Args:
        color: the color to lighten
        amount: the amount, a value between 0 and 1

    Returns:
        the resulting color as a RGB float tuple

    Example
    ~~~~~~~

        >>> lighten_color((.3,.55,.1), 0.5)
    """
    c = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def asrgb(color) -> tuple[float, float, float]:
    """
    Convert color to RGB if needed

    Args:
        color: a str color or a tuple like (200, 140, 18). If color already has
            float values between 0 and 1 then it is returned unmodified

    Returns:
        an RGB tuple with float values between 0 and 1

    """
    if isinstance(color, tuple):
        assert len(color) == 3
        if all(isinstance(part, int) for part in color):
            return (color[0] / 255, color[1] / 255, color[2] / 255)
        elif all(isinstance(part, float) and 0 <= part <= 1
                 for part in color):
            return color
        else:
            raise ValueError(f"Not a valid color: {color}")
    elif isinstance(color, str):
        import matplotlib.colors as mplcolors
        return mplcolors.to_rgb(color)
    else:
        raise ValueError(f"Not a valid color: {color}")


def _isValidRGB(color) -> bool:
    return (isinstance(color, tuple) and
            len(color) == 3 and
            all(isinstance(part, float) and 0 <= part <= 1 for part in color))


def desaturate(color: tuple[float, float, float], factor: float
               ) -> tuple[float, float, float]:
    """
    Decrease the saturation channel of a color by some percent.

    Taken from seaborn.desaturate
    (https://github.com/mwaskom/seaborn/blob/master/seaborn/utils.py)

    Args:
        color: the color to desaturate, as RGB float tuple
        factor: the saturation channel will be multiplied by this factor

    Returns:
        the desaturated color as a float rgb tuple

    """
    # Check inputs
    if not 0 <= factor <= 1:
        raise ValueError("factor must be between 0 and 1")

    if not _isValidRGB(color):
        raise ValueError(f"Invalid color: {color}")

    # Short circuit to avoid floating point issues
    if factor == 1:
        return color

    # Convert to hls
    h, l, s = colorsys.rgb_to_hls(*color)

    # Desaturate the saturation channel
    s *= factor

    # Convert back to rgb
    return colorsys.hls_to_rgb(h, l, s)
