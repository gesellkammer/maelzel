"""
Rational numbers with float-like repr
"""
from __future__ import annotations
from typing import Tuple, Any
try:
    from quicktions import Fraction as _F
except ImportError:
    from fractions import Fraction as _F


class Rat(_F):
    """
    Drop-in replacement to fractions.Fraction with float-like repr

    A rational number used to avoid rounding errors.

    A :class:`maelzel.rational.Rat` is a drop-in replacement for
    :class:`fractions.Fraction` with float-like repr. It can be used
    whenever a Fraction is used to avoid rounding errors, but its ``repr``
    resembles that of a float

    If the package `quicktions` is installed (an fast implementation of
    Fraction in cython), it is used as a base class of Rat. For that,
    to test if a Rat is Fraction-like, avoid using isinstance(x, Fraction)
    but use::

        >>> from numbers import Rational
        >>> from maelzel.rational import Rat
        >>> x = Rat(1, 3)
        >>> isinstance(x, Rational)
        True

    """
    def __floordiv__(self, other: Any) -> int:
        return _F.__floordiv__(self, other)

    def __repr__(self):
        return f"{float(self):.8g}"

    def __add__(self, other) -> Rat:
        return Rat(_F.__add__(self, other))

    def __radd__(self, other) -> Rat:
        return Rat(_F.__radd__(self, other))

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
        return Rat(_F.__sub__(self, other))

    def __rsub__(self, other) -> Rat:
        return Rat(_F.__rsub__(self, other))

    def __mul__(self, other) -> Rat:
        return Rat(_F.__mul__(self, other))

    def __divmod__(self, other) -> Tuple[int, Rat]:
        a, b = _F.__divmod__(self, other)
        return (a, Rat(b))

    def __float__(self) -> float:
        return self.numerator/self.denominator

    def __abs__(self) -> Rat:
        return Rat(_F.__abs__(self))

    def __mod__(self, other) -> Rat:
        return Rat(_F.__mod__(self, other))

    def __neg__(self) -> Rat:
        return Rat(_F.__neg__(self))

    def __pow__(self, other) -> Rat:
        return Rat(_F.__pow__(self, other))

    def __rfloordiv__(self, other) -> Rat:
        return Rat(_F.__rfloordiv__(self, other))

    def __truediv__(self, other) -> Rat:
        return Rat(_F.__truediv__(self, other))

    def __pos__(self) -> Rat:
        return Rat(_F.__pos__(self))

    def __rmod__(self, other) -> Rat:
        return Rat(_F.__rmod__(self, other))

    def __rmul__(self, other) -> Rat:
        return Rat(_F.__rmul__(self, other))

    def __rpow__(self, other) -> Rat:
        return Rat(_F.__rpow__(self, other))

    def __rtruediv__(self, other) -> Rat:
        return Rat(_F.__rtruediv__(self, other))

    @classmethod
    def from_float(cls, x: float) -> Rat:
        return cls(*x.as_integer_ratio())

    def limit_denominator(self, max_denominator=1000000) -> Rat:
        return Rat(_F.limit_denominator(self, max_denominator))