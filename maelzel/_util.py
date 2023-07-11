from __future__ import annotations
from typing import Callable, Sequence
import emlib.misc


def reprObj(obj,
            filter: dict[str, Callable] = {},
            priorityargs: Sequence[str] = None,
            hideFalse = False,
            hideEmptyStr = False,
            hideFalsy = False,
            quoteStrings=False
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
        if (filterfunc := filter.get(attr)) and not filterfunc(value):
            continue
        if quoteStrings and isinstance(value, str):
            value = f'"{value}"'
        info.append(f'{attr}={value}')
    infostr = ', '.join(info)
    cls = type(obj).__name__
    return f"{cls}({infostr})"
