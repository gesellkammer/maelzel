"""
Rational number based on gmpy.mpq
"""
from __future__ import annotations
import numbers
f
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *

from gmpy import mpq

_mpq_class = mpq(1).__class__


class Rat(numbers.Rational):
    def __init__(self, *args):
        if isinstance(args[0], _mpq_class):
            self._val = args[0]
        else:
            self._val = mpq(*args)

    @property
    def numerator(self) -> int:
        return int(self._val.numerator)

    @property
    def denominator(self):
        return int(self._val.denominator)

    def __repr__(self) -> str:
        return f"{float(self._val):.8g}"

    def __add__(self, other) -> Rat:
        if isinstance(other, Rat):
            return Rat(self._val + other._val)
        return Rat(self._val + other)

    def __radd__(self, other) -> Rat:
        if isinstance(other, Rat):
            return Rat(self._val + other._val)
        return Rat(self._val + other)

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
        if isinstance(other, Rat):
            return Rat(self._val - other._val)
        return Rat(self._val - other)

    def __rsub__(self, other) -> Rat:
        if isinstance(other, Rat):
            return Rat(other._val - self._val)
        return Rat(other - self._val)

    def __mul__(self, other) -> Rat:
        if isinstance(other, Rat):
            return Rat(self._val * other._val)
        return Rat(self._val * other)

    def __div__(self, other) -> Rat:
        if isinstance(other, Rat):
            return Rat(self._val / other._val)
        return Rat(self._val / other)

    def __float__(self) -> float:
        return float(self._val)

    def __gt__(self, other) -> bool:
        if isinstance(other, Rat):
            return self._val > other._val
        return self._val > other

    def __ge__(self, other) -> bool:
        if isinstance(other, Rat):
            return self._val >= other._val
        return self._val >= other

    def __lt__(self, other) -> bool:
        if isinstance(other, Rat):
            return self._val < other._val
        return self._val < other

    def __le__(self, other) -> bool:
        if isinstance(other, Rat):
            return self._val <= other._val
        return self._val <= other

    def __eq__(self, other) -> bool:
        if isinstance(other, Rat):
            return self._val == other._val
        return self._val == other

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self._val)

    def __abs__(self) -> Rat:
        return Rat(abs(self._val))

    def __ceil__(self):
        a = self._val
        return Rat(-(-a.numerator//a.denominator))

    def __floor__(self):
        a = self._val
        return Rat(a.numerator // a.denominator)

    def __floordiv__(self, other):
        if isinstance(other, Rat):
            return self._val.__floordiv__(other._val)
        return Rat(self._val.__floordiv__(other))

    def __mod__(self, other) -> Rat:
        if isinstance(other, Rat):
            return self._val.__mod__(other._val)
        return self._val.__mod__(other)

    def __neg__(self) -> Rat:
        return Rat(self._val.__neg__())

    def __pow__(self, other) -> Rat:
        if isinstance(other, Rat):
            return self._val.__pow__(other._val)
        return self._val.__pow__(other)

    def __rfloordiv__(self, other) -> Rat:
        if isinstance(other, Rat):
            return self._val.__rfloordiv__(other._val)
        return self._val.__rfloordiv__(other)

    def __truediv__(self, other) -> Rat:
        if isinstance(other, Rat):
            return self._val.__truevid__(other._val)
        return self._val.__truediv__(other)

    def __pos__(self) -> Rat:
        return Rat(self._val.__pos__())

    def __rmod__(self, other) -> Rat:
        if isinstance(other, Rat):
            return self._val.__rmod__(other._val)
        return Rat(self._val.__rmod__(other))

    def __rmul__(self, other) -> Rat:
        if isinstance(other, Rat):
            return Rat(self._val.__rmul__(other._val))
        return Rat(self._val.__rmul__(other))

    def __rpow__(self, other) -> Rat:
        if isinstance(other, Rat):
            return Rat(self._val.__rpow__(other._val))
        return Rat(self._val.__rpow__(other))

    def __rtruediv__(self, other) -> Rat:
        if isinstance(other, Rat):
            return Rat(self._val.__rtruediv__(other._val))
        return Rat(self._val.__rtruediv__(other))

    def __trunc__(self) -> int:
        """trunc(a)"""
        a = self._val
        if a.numerator<0:
            return -(-a.numerator//a.denominator)
        else:
            return a.numerator//a.denominator

    @classmethod
    def from_float(cls, x: float) -> Rat:
        return cls(*x.as_integer_ratio())

    def limit_denominator(self, max_denominator=1000000) -> Rat:
        return _limit_denom(self.numerator, self.denominator, max_denominator)


def _limit_denom(num: int, den: int, maxden: int) -> Rat:
    """
    Copied from https://github.com/python/cpython/blob/main/Lib/fractions.py
    """
    if maxden < 1:
        raise ValueError("max_denominator should be at least 1")
    if den <= maxden:
        return Rat(num, den)

    p0, q0, p1, q1 = 0, 1, 1, 0


    while True:
        a = num//den
        q2 = q0+a*q1
        if q2>maxden:
            break
        p0, q0, p1, q1 = p1, q1, p0+a*p1, q2
        num, den = den, num-a*den

    k = (maxden-q0) // q1
    bound1 = Rat(p0+k*p1, q0+k*q1)
    bound2 = Rat(p1, q1)
    orig = Rat(num, den)
    if abs(bound2 - orig) <= abs(bound1-orig):
        return bound2
    else:
        return bound1

