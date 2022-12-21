"""
Rational numbers with float-like repr

    >>> from fractions import Fraction
    >>> from maelzel.rational import Rat
    >>> import math
    >>> pifraction = Fraction.from_float(math.pi)
    >>> pifraction
    Fraction(884279719003555, 281474976710656)
    >>> pirational = Rat.from_float(math.pi)
    >>> pirational
    3.1415927
    >>> pirational == pifraction
    True

To check types, always check against the abstract class. This makes it easy to
use classes like Rat or quicktions.Fraction:

    >>> import numbers
    >>> isinstance(pirational, numbers.Rational)
    True
    >>> isinstance(pirational, Fraction)
    False

"""
from __future__ import annotations
from numbers import Rational
from typing import Tuple, Any

try:
    from quicktions import Fraction as _F
except ImportError:
    from fractions import Fraction as _F


def fractionToDecimal(numerator: int, denominator: int) -> str:
    """
    Converts a fraction to a decimal number with repeating period

    Args:
        numerator: the numerator of the fraction
        denominator: the denominator of the fraction

    Returns:
        the string representation of the resulting decimal. Any repeating
        period will be prefixed with '('

    Example
    ~~~~~~~

        >>> from emlib.mathlib import *
        >>> fraction_to_decimal(1, 3)
        '0.(3'
        >>> fraction_to_decimal(1, 7)
        '0.(142857'
        >>> fraction_to_decimal(100, 7)
        '14.(285714'
        >>> fraction_to_decimal(355, 113)
        '3.(1415929203539823008849557522123893805309734513274336283185840707964601769911504424778761061946902654867256637168'
    """
    result = [str(numerator//denominator) + "."]
    subresults = [numerator % denominator]
    numerator %= denominator
    while numerator != 0:
        numerator *= 10
        result_digit, numerator = divmod(numerator, denominator)
        result.append(str(result_digit))
        if numerator not in subresults:
            subresults.append(numerator)
        else:
            result.insert(subresults.index(numerator) + 1, "(")
            break
    return "".join(result)


class Rat(_F):
    """
    Drop-in replacement to fractions.Fraction with float-like repr

    A rational number used to avoid rounding errors.

    A :class:`maelzel.rational.Rat` is a drop-in replacement for
    :class:`fractions.Fraction` with float-like repr. It can be used
    whenever a Fraction is used to avoid rounding errors, but its ``repr``
    resembles that of a float

    If the package `quicktions` is installed (a fast implementation of
    Fraction in cython), it is used as a base class of Rat. For that,
    to test if a Rat is Fraction-like, avoid using isinstance(x, Fraction)
    but use::

        >>> from numbers import Rational
        >>> from maelzel.rational import Rat
        >>> x = Rat(1, 3)
        >>> isinstance(x, Rational)
        True

    The same is valid when using type annotations::

        import numbers
        def square(a: numbers.Rational) -> numbers.Rational:
            return a*a

    For all other aspects the documentation for python ``fractions.Fraction`` is
    valid for this implementation: https://docs.python.org/3/library/fractions.html

    """
    _reprWithFraction = False
    "If True, add the fraction to the repr"

    _reprElipsisMaxDenominator = 9999
    "A fraction with a denom. higher than this adds a ... to its float repr "

    _reprMaxDenominator = 99999999
    "A fraction with a denom. higher than this is shown as ~num/den, when num/den is rounded"

    _reprShowRepeatingPeriod = False
    "If True, show the repeating period, if any"

    def __repr__(self):
        if self.denominator == 1:
            return str(self.numerator)

        if self.denominator > self._reprElipsisMaxDenominator:
            floatpart = f"{float(self):.8g}…"
        elif self._reprShowRepeatingPeriod:
            floatpart = fractionToDecimal(self.numerator, self.denominator)
            if self.denominator != 1:
                i, rest = floatpart.split(".")
                if len(rest) > 8:
                    rest = rest[:8] + '…'
                    floatpart = f'{i}.{rest}'
        else:
            floatpart = f"{float(self):.8g}"

        if not self._reprWithFraction:
            return floatpart

        if self.denominator > self._reprMaxDenominator:
            f = self.limit_denominator(self._reprMaxDenominator)
            return f'{floatpart} (~{f.numerator}/{f.denominator})'
        else:
            return f'{floatpart} ({self.numerator}/{self.denominator})'


    def __floordiv__(self, other: Any) -> int:
        r = _F.__floordiv__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __format__(self, format_spec) -> str:
        if not format_spec:
            return self.__repr__()
        return float(self).__format__(format_spec)

    def __add__(self, other) -> Rat:
        r = _F.__add__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __radd__(self, other) -> Rat:
        r = _F.__radd__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __round__(self, ndigits: int=None) -> Rat:
        if ndigits is None:
            floor, remainder = divmod(self.numerator, self.denominator)
            if remainder*2<self.denominator:
                return floor
            elif remainder*2>self.denominator:
                return floor+1
            # Deal with the half case:
            elif floor%2 == 0:
                return floor
            else:
                return floor+1
        shift = 10**abs(ndigits)
        if ndigits > 0:
            return Rat(round(self*shift), shift)
        else:
            return Rat(round(self/shift)*shift)

    def __sub__(self, other) -> Rat:
        r = _F.__sub__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __rsub__(self, other) -> Rat:
        r = _F.__rsub__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __mul__(self, other) -> Rat:
        r = _F.__mul__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __divmod__(self, other) -> Tuple[int, Rat]:
        a, b = _F.__divmod__(self, other)
        return (a, Rat(b))

    def __float__(self) -> float:
        return self.numerator/self.denominator

    def __abs__(self) -> Rat:
        return Rat(abs(self.numerator), self.denominator)

    def __mod__(self, other) -> Rat:
        r = _F.__mod__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __neg__(self) -> Rat:
        r = _F.__neg__(self)
        return Rat(r.numerator, r.denominator)

    def __pow__(self, other) -> Rat:
        r = _F.__pow__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __rfloordiv__(self, other) -> Rat:
        r = _F.__rfloordiv__(self, other)
        return Rat(r.numerator, r.denominator)

    def __truediv__(self, other) -> Rat:
        r = _F.__truediv__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __pos__(self) -> Rat:
        return Rat(_F.__pos__(self))

    def __rmod__(self, other) -> Rat:
        r = _F.__rmod__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __rmul__(self, other) -> Rat:
        r = _F.__rmul__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __rpow__(self, other) -> Rat:
        r = _F.__rpow__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    def __rtruediv__(self, other) -> Rat:
        r = _F.__rtruediv__(self, other)
        return Rat(r.numerator, r.denominator) if isinstance(r, _F) else r

    @classmethod
    def from_float(cls, x: float) -> Rat:
        return cls(*x.as_integer_ratio())

    def limit_denominator(self, max_denominator=1000000) -> Rat:
        r = _F.limit_denominator(self, max_denominator)
        return Rat(r.numerator, r.denominator)


def asRat(x) -> Rat:
    if isinstance(x, Rat):
        return x
    elif isinstance(x, Rational):
        return Rat(x.numerator, x.denominator)
    else:
        return Rat(x)