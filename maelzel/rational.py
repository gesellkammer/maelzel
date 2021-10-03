from __future__ import annotations
from typing import Tuple, Any
try:
    from quicktions import Fraction as F
except ImportError:
    from fractions import Fraction as F


class Rat(F):
    def __floordiv__(self, other: Any) -> int:
        return F.__floordiv__(self, other)

    def __repr__(self):
        return f"{float(self):.8g}"

    def __add__(self, other) -> Rat:
        return Rat(F.__add__(self, other))

    def __radd__(self, other) -> Rat:
        return Rat(F.__radd__(self, other))

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
        return Rat(F.__sub__(self, other))

    def __rsub__(self, other) -> Rat:
        return Rat(F.__rsub__(self, other))

    def __mul__(self, other) -> Rat:
        return Rat(F.__mul__(self, other))

    def __divmod__(self, other) -> Tuple[int, Rat]:
        a, b = F.__divmod__(self, other)
        return (a, Rat(b))

    def __float__(self) -> float:
        return self.numerator/self.denominator

    def __abs__(self) -> Rat:
        return Rat(F.__abs__(self))

    def __mod__(self, other) -> Rat:
        return Rat(F.__mod__(self, other))

    def __neg__(self) -> Rat:
        return Rat(F.__neg__(self))

    def __pow__(self, other) -> Rat:
        return Rat(F.__pow__(self, other))

    def __rfloordiv__(self, other) -> Rat:
        return Rat(F.__rfloordiv__(self, other))

    def __truediv__(self, other) -> Rat:
        return Rat(F.__truediv__(self, other))

    def __pos__(self) -> Rat:
        return Rat(F.__pos__(self))

    def __rmod__(self, other) -> Rat:
        return Rat(F.__rmod__(self, other))

    def __rmul__(self, other) -> Rat:
        return Rat(F.__rmul__(self, other))

    def __rpow__(self, other) -> Rat:
        return Rat(F.__rpow__(self, other))

    def __rtruediv__(self, other) -> Rat:
        return Rat(F.__rtruediv__(self, other))

    @classmethod
    def from_float(cls, x: float) -> Rat:
        return cls(*x.as_integer_ratio())

    def limit_denominator(self, max_denominator=1000000) -> Rat:
        return Rat(F.limit_denominator(self, max_denominator))