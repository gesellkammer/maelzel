from __future__ import annotations
from gmpy import mpq
from fractions import Fraction


def limit_denominator(num, den, maxden):
    f = Fraction(num, den).limit_denominator(maxden)
    return f.numerator, f.denominator


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

    @classmethod
    def from_float(cls, x: float) -> Rat:
        return Rat(x)

    def limit_denominator(self, max_denominator=1000000):
        return Rat(*limit_denominator(self.numerator, self.denominator, max_denominator))



