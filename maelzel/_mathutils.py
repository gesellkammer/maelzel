from __future__ import annotations
from maelzel.common import F
import math

import typing as _t


def splitInterval(start: F, end: F, offsets: _t.Sequence[F]
                  ) -> list[tuple[F, F]]:
    """
    Split interval (start, end) at the given offsets

    Args:
        start: start of the interval
        end: end of the interval
        offsets: offsets to split the interval at. Must be sorted

    Returns:
        a list of (start, end) segments where no segment extends over any
        of the given offsets
    """
    assert end > start
    assert offsets

    if offsets[0] > end or offsets[-1] < start:
        # no intersection, return the original time range
        return [(start, end)]

    out = []
    for offset in offsets:
        if offset >= end:
            break
        if start < offset:
            out.append((start, offset))
            start = offset
    if start != end:
        out.append((start, end))

    assert len(out) >= 1
    return out


def intersectF(u1: F, u2: F, v1: F, v2: F) -> tuple[F, F] | None:
    """
    return the intersection of (u1, u2) and (v1, v2) or None if no intersection

    Args:
        u1: lower bound of range U
        u2: higher bound of range U
        v1: lower bound of range V
        v2: higher bound of range V

    Returns:
        the intersection between range U and range V as a tuple (start, end).
        If no intersection is found, None is returned

    Example::

        >>> if intersect := intersection(0, 3, 2, 5):
        ...     start, end = intersect
        ...     ...

    """
    x0 = u1 if u1 > v1 else v1
    x1 = u2 if u2 < v2 else v2
    return (x0, x1) if x0 < x1 else None


def linexp(x: float, exp: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """
    Linear to exponential conversion

    Args:
        x: The input value to be converted.
        exp: The exponent to be used in the conversion.
        x0: The lower bound of the input range.
        y0: The lower bound of the output range.
        x1: The upper bound of the input range.
        y1: The upper bound of the output range.

    Returns:
        The converted value.

    """
    dx = (x - x0) / (x1 - x0)
    dx = dx ** exp
    return y0 + (y1 - y0) * dx


def limitDenominator(num: int, den: int, maxden: int, assumeCoprime=False) -> tuple[int, int]:
    """
    Copied from https://github.com/python/cpython/blob/main/Lib/fractions.py

    Args:
        num: The numerator of the fraction.
        den: The denominator of the fraction.
        maxden: The maximum denominator allowed.
        assumeCoprime: Whether to assume the fraction is already in its simplest form.

    Returns:
        The simplified fraction as a tuple of integers.
    """
    if maxden < 1:
        raise ValueError("max_denominator should be at least 1")

    if den <= maxden:
        return num, den

    if not assumeCoprime:
        g = math.gcd(num, den)
        if den < 0:
            g = -g
        num //= g
        den //= g

    p0, q0, p1, q1 = 0, 1, 1, 0
    n, d = num, den
    while True:
        a = n // d
        q2 = q0 + a * q1
        if q2 > maxden:
            break
        p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
        newd = n - a*d
        if newd == 0:
            break
        n, d = d, newd

    k = (maxden - q0) // q1

    # Determine which of the candidates (p0+k*p1)/(q0+k*q1) and p1/q1 is
    # closer to self. The distance between them is 1/(q1*(q0+k*q1)), while
    # the distance from p1/q1 to self is d/(q1*self._denominator). So we
    # need to compare 2*(q0+k*q1) with self._denominator/d.
    if 2 * d * (q0 + k * q1) <= den:
        return p1, q1
    else:
        return p0 + k * p1, q0 + k * q1
