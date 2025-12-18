"""
Rational number based on gmpy.mpq

This module should NOT import from anything within maelzel
"""
from __future__ import annotations
import numbers
from gmpy2 import mpq


numbers.Rational.register(mpq)


class Q(numbers.Rational):
    """
    A Rational class with float like repr
    """
    def __init__(self, *args):
        if isinstance(args[0], mpq):
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

    def __add__(self, other) -> Q:
        if isinstance(other, Q):
            return Q(self._val + other._val)
        return Q(self._val + other)

    def __radd__(self, other) -> Q:
        if isinstance(other, Q):
            return Q(self._val + other._val)
        return Q(self._val + other)

    def __round__(self, ndigits: int | None = None) -> Q:
        if ndigits is None:
            floor, remainder = divmod(self.numerator, self.denominator)
            if remainder * 2 < self.denominator:
                return Q(floor, 1)
            elif remainder * 2 > self.denominator:
                return Q(floor + 1, 1)
            # Deal with the half case:
            elif floor % 2 == 0:
                return Q(floor, 1)
            else:
                return Q(floor + 1, 1)
        shift = 10**abs(ndigits)
        if ndigits > 0:
            return Q(round(self * shift), shift)
        else:
            return Q(round(self / shift) * shift)

    def __sub__(self, other) -> Q:
        if isinstance(other, Q):
            return Q(self._val - other._val)
        return Q(self._val - other)

    def __rsub__(self, other) -> Q:
        if isinstance(other, Q):
            return Q(other._val - self._val)
        return Q(other - self._val)

    def __mul__(self, other) -> Q:
        if isinstance(other, Q):
            return Q(self._val * other._val)
        return Q(self._val * other)

    def __div__(self, other) -> Q:
        if isinstance(other, Q):
            return Q(self._val / other._val)
        return Q(self._val / other)

    def __float__(self) -> float:
        return float(self._val)

    def __gt__(self, other) -> bool:
        if isinstance(other, Q):
            return self._val > other._val
        return self._val > other

    def __ge__(self, other) -> bool:
        if isinstance(other, Q):
            return self._val >= other._val
        return self._val >= other

    def __lt__(self, other) -> bool:
        if isinstance(other, Q):
            return self._val < other._val
        return self._val < other

    def __le__(self, other) -> bool:
        if isinstance(other, Q):
            return self._val <= other._val
        return self._val <= other

    def __eq__(self, other) -> bool:
        if isinstance(other, Q):
            return self._val == other._val
        return self._val == other

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self._val)

    def __abs__(self) -> Q:
        return Q(abs(self._val))

    def __ceil__(self):
        a = self._val
        return Q(-(-a.numerator // a.denominator))

    def __floor__(self):
        a = self._val
        return Q(a.numerator // a.denominator)

    def __floordiv__(self, other):
        if isinstance(other, Q):
            return self._val.__floordiv__(other._val)
        return Q(self._val.__floordiv__(other))

    def __mod__(self, other) -> Q:
        if isinstance(other, Q):
            return self._val.__mod__(other._val)
        return self._val.__mod__(other)

    def __neg__(self) -> Q:
        return Q(self._val.__neg__())

    def __pow__(self, other) -> Q:
        if isinstance(other, Q):
            return self._val.__pow__(other._val)
        return self._val.__pow__(other)

    def __rfloordiv__(self, other) -> Q:
        if isinstance(other, Q):
            return self._val.__rfloordiv__(other._val)
        return self._val.__rfloordiv__(other)

    def __truediv__(self, other) -> Q:
        if isinstance(other, Q):
            return self._val.__truevid__(other._val)
        return self._val.__truediv__(other)

    def __pos__(self) -> Q:
        return Q(self._val.__pos__())

    def __rmod__(self, other) -> Q:
        if isinstance(other, Q):
            return self._val.__rmod__(other._val)
        return Q(self._val.__rmod__(other))

    def __rmul__(self, other) -> Q:
        if isinstance(other, Q):
            return Q(self._val.__rmul__(other._val))
        return Q(self._val.__rmul__(other))

    def __rpow__(self, other) -> Q:
        if isinstance(other, Q):
            return Q(self._val.__rpow__(other._val))
        return Q(self._val.__rpow__(other))

    def __rtruediv__(self, other) -> Q:
        if isinstance(other, Q):
            return Q(self._val.__rtruediv__(other._val))
        return Q(self._val.__rtruediv__(other))

    def __trunc__(self) -> int:
        """trunc(a)"""
        a = self._val
        if a.numerator<0:
            return -(-a.numerator//a.denominator)
        else:
            return a.numerator//a.denominator

    @classmethod
    def from_float(cls, x: float) -> Q:
        return cls(*x.as_integer_ratio())

    def limit_denominator(self, max_denominator=1000000) -> Q:
        num, den = _limitDenom(self.numerator, self.denominator, maxden=max_denominator)
        return Q(num, den)


def _limitDenom(num: int, den: int, maxden: int) -> tuple[int, int]:
    """
    Copied from https://github.com/python/cpython/blob/main/Lib/fractions.py
    """
    if maxden < 1:
        raise ValueError("max_denominator should be at least 1")
    if den <= maxden:
        return num, den

    p0, q0, p1, q1 = 0, 1, 1, 0
    n, d = num, den
    while True:
        a = n // d
        q2 = q0 + a * q1
        if q2 > maxden:
            break
        p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
        n, d = d, n - a * d
    k = (maxden - q0) // q1

    # Determine which of the candidates (p0+k*p1)/(q0+k*q1) and p1/q1 is
    # closer to self. The distance between them is 1/(q1*(q0+k*q1)), while
    # the distance from p1/q1 to self is d/(q1*self._denominator). So we
    # need to compare 2*(q0+k*q1) with self._denominator/d.
    if 2 * d * (q0 + k * q1) <= den:
        return p1, q1
    else:
        return p0 + k * p1, q0 + k * q1


