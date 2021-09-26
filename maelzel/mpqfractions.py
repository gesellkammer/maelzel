"""
Module text
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *

from gmpy import mpq


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
        num, den = den, num-a*d

    k = (maxden-q0) // q1
    bound1 = Rat(p0+k*p1, q0+k*q1)
    bound2 = Rat(p1, q1)
    orig = Rat(num, den)
    if abs(bound2 - orig) <= abs(bound1-orig):
        return bound2
    else:
        return bound1


class Rat:
    def __init__(self, *args):
        self._val = mpq(*args)

    @property
    def numerator(self):
        return int(self._val.numerator)

    @property
    def denominator(self):
        return int(self._val.denominator)

    def __repr__(self) -> str:
        return f"{float(self._val):.8g}"

    def __add__(self, other):
        if isinstance(other, Rat):
            return Rat(self._val + other._val)
        return Rat(self._val + other)

    def __radd__(self, other):
        if isinstance(other, Rat):
            return Rat(self._val + other._val)
        return Rat(self._val + other)

    def __sub__(self, other):
        if isinstance(other, Rat):
            return Rat(self._val - other._val)
        return Rat(self._val - other)

    def __rsub__(self, other):
        if isinstance(other, Rat):
            return Rat(other._val - self._val)
        return Rat(other - self._val)

    def __mul__(self, other):
        if isinstance(other, Rat):
            return Rat(self._val * other._val)
        return Rat(self._val * other)

    def __div__(self, other):
        if isinstance(other, Rat):
            return Rat(self._val / other._val)
        return Rat(self._val / other)

    def __float__(self):
        return float(self._val)

    def __gt__(self, other):
        if isinstance(other, Rat):
            return self._val > other._val
        return self._val > other

    def __ge__(self, other):
        if isinstance(other, Rat):
            return self._val >= other._val
        return self._val >= other

    def __lt__(self, other):
        if isinstance(other, Rat):
            return self._val < other._val
        return self._val < other

    def __le__(self, other):
        if isinstance(other, Rat):
            return self._val <= other._val
        return self._val <= other

    def __eq__(self, other):
        if isinstance(other, Rat):
            return self._val == other._val
        return self._val == other

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._val)

    def __abs__(self):
        return Rat(abs(self._val))

    @classmethod
    def from_float(cls, x: float) -> Rat:
        return Rat(x)

    def limit_denominator(self, max_denominator=1000000):
        return _limit_denom(self.numerator, self.denominator, max_denominator)

