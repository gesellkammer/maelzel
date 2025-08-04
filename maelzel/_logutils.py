from __future__ import annotations
from collections import UserString
import typing as _t


class LazyFmt(UserString):
    """
    A class to do lazy formatting for asserts and exceptions

    .. code-block:: python

        assert all(x % 1 == 0 for x in xs), LazyFmt("Invalid xs: %s", xs)
    """
    def __init__(self, fmt: str, *args):
        self._fmt = fmt
        self._args = args
        self._cached: str | None = None

    @property
    def data(self) -> str:
        if self._cached is not None:
            return self._cached
        self._cached = s = self._fmt % self._args
        return s

    def __getstate__(self) -> tuple[str, tuple]:
        return (self._fmt, self._args)

    def __setstate__(self, state: tuple[str, tuple]) -> None:
        self._fmt, self._args = state

    def __getattr__(self, name: str) -> _t.Any:
        return getattr(self.data, name)

    def __dir__(self) -> list[str]:
        return dir(str)

    def __copy__(self) -> _t.Self:
        return self

    def __repr__(self) -> str:
        try:
            r = repr(str(self.data))
            return f"{self.__class__.__name__}({r})"
        except Exception:
            return "<%s broken>" % self.__class__.__name__



class LazyStr(UserString):
    """
    A string with delayed evaluation.

    To be used mainly when logging big objects

    Args:
        func: a function returning a string
        args: optional arguments passed to func
        kwargs: keyword args passed to func

    Example

        logger.debug("This is wrong: %s", LazyString(myobj.__str__))

    In the example above, the object's str representation is only called
    if debugging actually takes place.

    """
    __slots__ = ("_func", "_args", )

    def __new__(cls, func: _t.Callable | str, *args, **kwargs) -> object:
        if isinstance(func, str):
            # Many UserString's functions like `lower`, `__add__` and so on wrap
            # returned values with a call to `self.__class__(...)` to ensure the
            # result is of the same type as the original class.
            # However, as the result of all of such methods is always a string,
            # there's no need to create a new instance of a `LazyString`
            return func
        return object.__new__(cls)

    def __init__(self, func: _t.Callable[..., str], *args, **kwargs) -> None:
        # we do not want to call super().__init__
        self._func   = func
        self._args   = args
        self._kwargs = kwargs

    @classmethod
    def repr(cls, obj) -> _t.Self:
        return cls(lambda o: repr(obj), obj)

    @classmethod
    def str(cls, obj) -> _t.Self:
        return cls(lambda o: str(obj), obj)

    @property
    def data(self) -> str:
        return self._func(*self._args, **self._kwargs)

    def __getnewargs_ex__(self) -> tuple[tuple, _t.Mapping]:
        args = (self._func, ) + self._args
        return (args, self._kwargs)

    def __getstate__(self) -> tuple[_t.Callable, tuple, _t.Mapping]:
        return (self._func, self._args, self._kwargs)

    def __setstate__(self, state: tuple[_t.Callable, tuple, _t.Mapping]) -> None:
        self._func, self._args, self._kwargs = state

    def __getattr__(self, name: str) -> _t.Any:
        return getattr(self.data, name)

    def __dir__(self) -> list[str]:
        return dir(str)

    def __copy__(self) -> _t.Self:
        return self

    def __repr__(self) -> str:
        try:
            r = repr(str(self.data))
            return f"{self.__class__.__name__}({r})"
        except Exception:
            return "<%s broken>" % self.__class__.__name__
