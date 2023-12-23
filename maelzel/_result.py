from __future__ import annotations
from typing import Generic, TypeVar


_T = TypeVar('_T')


__all__ = ('Result',)


class Result(Generic[_T]):
    """
    A class to encapsulate the result of an operation

    This is useful for operations which can either return a value
    or an error message.

    Args:
        ok: True if ok, False if failed
        value: the value returned by the operation
        info: an error message if the operation failed

    Example
    -------

    .. code::

        from emlib.result import Result
        import re
        from fractions import Fraction

        def parsefraction(txt: str) -> Result[Fraction]:
            match = re.match(r"([0-9]+)\/([1-9][0-9]*)", txt)
            if not match:
                return Result.Fail(f"Could not parse '{txt}' as fraction")
            num = int(match.group(1))
            den = int(match.group(2))
            return Result.Ok(Fraction(num, den))

        if fraction := parsefraction("4/5"):
            print(f"Fraction ok: {fraction.value})  # prints 'Fraction(4, 5)'

    Typing
    ------

    To make typing analysis work better it is possible to indicate the kind of
    value wrapped by the Result class. See the return type declared in ``parsefraction``
    in the example above

    """

    def __init__(self, ok: bool, value: _T | None = None, info: str = ''):
        self.ok: bool = ok
        self._value: _T | None = value
        self.info: str = info

    @property
    def value(self) -> _T:
        if not self.ok:
            raise ValueError("Cannot access the value of a failed result")
        assert self._value is not None
        return self._value

    def __bool__(self) -> bool:
        return self.ok

    @property
    def failed(self) -> bool:
        """True if operation failed"""
        return not self.ok

    def __repr__(self):
        if self.ok:
            return f"Ok(value={self._value})"
        else:
            return f'Fail(info="{self.info}")'

    @classmethod
    def Fail(cls, info: str) -> Result:
        """Create a Result object for a failed operation."""
        if not isinstance(info, str):
            raise TypeError(f"The info parameter should be a str, got {info}")
        return cls(False, value=None, info=info)

    @classmethod
    def Ok(cls, value: _T | None = None) -> Result:
        """Create a Result object for a successful operation."""
        return cls(True, value=value)