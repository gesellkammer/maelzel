from __future__ import annotations
import emlib.misc
import warnings
import sys
import os
import weakref
from maelzel.common import F

from typing import Callable, Sequence
from maelzel.common import T


def reprObj(obj,
            filter: dict[str, Callable] = {},
            priorityargs: Sequence[str] = None,
            hideFalse = False,
            hideEmptyStr = False,
            hideFalsy = False,
            quoteStrings=False,
            ) -> str:
    """
    Given an object, generate its repr

    Args:
        obj: the object
        filter: a dict mapping keys to functions deciding if the key:value should
            be shown at all. The default is True, so if a filter function is given
            for a certain key, that key will be shown only if the function returns
            True.
        priorityargs: a list of attributes which are shown first.
        hideFalsy: hide any attr which evaluates to False under bool(obj.attr)
        hideFalse: hide bool attributes which are False.
        hideEmptyStr: hide str attributes which are empty

    Returns:
        a list of strings of the form "{key}={value}" only for those attributes
        which fullfill the given conditions

    """
    attrs = emlib.misc.find_attrs(obj)
    info = []
    attrs.sort()
    if priorityargs:
        attrs.sort(key=lambda attr: 0 if attr in priorityargs else 1)
    for attr in attrs:
        value = getattr(obj, attr)
        if value is None or (not value and hideFalsy) or (value == '' and hideEmptyStr) or (value is False and hideFalse):
            continue
        elif (filterfunc := filter.get(attr)) and not filterfunc(value):
            continue
        elif isinstance(value, weakref.ref):
            refobj = value()
            value = f'ref({type(refobj).__name__})'
        elif quoteStrings and isinstance(value, str):
            value = f'"{value}"'
        info.append(f'{attr}={value}')
    infostr = ', '.join(info)
    cls = type(obj).__name__
    return f"{cls}({infostr})"


def fuzzymatch(query: str, choices: Sequence[str], limit=5) -> list[tuple[str, int]]:
    if 'thefuzz.process' not in sys.modules:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import thefuzz.process
    return thefuzz.process.extract(query, choices=choices, limit=limit)


def checkChoice(name: str, s: str, choices: Sequence[str], threshold=8):
    if s not in choices:
        if len(choices) > threshold:
            matches = fuzzymatch(s, choices, limit=20)
            raise ValueError(f'Invalid value "{s}" for {name}, maybe you meant "{matches[0][0]}"? '
                             f'Other possible choices: {[m[0] for m in matches]}')
        else:
            raise ValueError(f'Invalid value "{s}" for {name}, it should be one of {list(choices)}')


def humanReadableTime(t: float) -> str:
    if t < 1e-6:
        return f"{t*1e9:.1f}ns"

    if t < 1e-3:
        return f"{t*1e6:.1f}µs"

    if t < 1:
        return f"{t*1e3:.1f}ms"

    return f"{t:.6g}s"


def normalizeFilename(path: str) -> str:
    return os.path.expanduser(path)


def showF(f: F, maxdenom=1000) -> str:
    """
    Show a fraction, limit den to *maxdenom*

    Args:
        f: the fraction to show
        maxdenom: the max. denominator to show

    Returns:
        a readable string representation

    """
    if f.denominator > maxdenom:
        f2 = f.limit_denominator(maxdenom)
        return "*%d/%d" % (f2.numerator, f2.denominator)
    return "%d/%d" % (f.numerator, f.denominator)


def showT(f: F | float | None) -> str:
    """Show *f* as time"""
    if f is None:
        return "None"
    if not isinstance(f, float):
        f = float(f)
    return f"{f:.3f}".rstrip('0').rstrip('.')

