from __future__ import annotations
import functools
import emlib.misc
from typing import Callable, Sequence, Any
from collections.abc import Hashable
import weakref


@functools.cache
def _objAttrsCached(obj,
                    exclude: tuple[str, ...] = (),
                    properties: tuple[str, ...] = (),
                    sort=True,
                    first: tuple[str, ...] = ()
                    ) -> list[str]:
    return _objAttrs(obj, exclude=exclude, sort=sort, first=first)


def _objAttrs(obj,
              exclude: tuple[str, ...] = (),
              properties: tuple[str, ...] = (),
              sort=True,
              first: tuple[str, ...] = ()
              ) -> list[str]:
    attrs = emlib.misc.find_attrs(obj)
    if exclude:
        import fnmatch
        attrs = [a for a in attrs if any(fnmatch.fnmatch(a, p) for p in exclude)]
    if properties and any(p not in attrs for p in properties):
        attrs = attrs.copy()
        for p in properties:
            if p not in attrs:
                attrs.append(p)
    if sort:
        attrs.sort()
    if first:
        attrs.sort(key=lambda attr: 0 if attr in first else 1)
    return attrs


def reprObj(obj,
            exclude: tuple[str, ...] = (),
            properties: tuple[str, ...] = (),
            filter: dict[str, Callable] = {},
            first: tuple[str, ...] = (),
            hideFalse=False,
            hideEmptyStr=False,
            hideFalsy=False,
            hideKeys: tuple[str, ...] = (),
            quoteStrings: bool | Sequence[str] = False,
            quoteChar="'",
            sort=True,
            convert: dict[str, Callable[[Any], str]] | None = None,
            ) -> str:
    """
    Given an object, generate its repr

    Args:
        obj: the object
        filter: a dict mapping keys to functions deciding if the key:value should
            be shown at all. The default is True, so if a filter function is given
            for a certain key, that key will be shown only if the function returns
            True.
        sort: if True, sort the keys
        properties: properties to include in the repr
        hideKeys: show the value without the key name. Makes the given key a priority
        quoteChar: char used to quote strings
        exclude: a seq. of attributes to exclude
        first: a list of attributes which are shown first.
        hideFalsy: hide any attr which evaluates to False under bool(obj.attr)
        hideFalse: hide bool attributes which are False.
        hideEmptyStr: hide str attributes which are empty
        quoteStrings: if True, strings are quoted. Alternative, a sequence of
            attributes to quote.
        convert: if given, a dict mapping attr names to a function of the form (value) -> str,
            which returns the string representation of the given value

    Returns:
        a list of strings of the form "{key}={value}" only for those attributes
        which fullfill the given conditions

    """
    if hideKeys:
        if first:
            for key in hideKeys:
                if key not in first:
                    first += (key,)
        else:
            first = hideKeys

    if isinstance(obj, Hashable):
        attrs = _objAttrsCached(obj, sort=sort, exclude=exclude, properties=properties, first=first)
    else:
        attrs = _objAttrs(obj, sort=sort, exclude=exclude, properties=properties, first=first)

    info = []

    for attr in attrs:
        value = getattr(obj, attr)
        if value is None or (hideFalsy and not value) or (hideEmptyStr and value == '') or (hideFalse and value is False):
            continue
        elif convert and attr in convert:
            value = convert[attr](value)
        elif (filterfunc := filter.get(attr)) and not filterfunc(value):
            continue
        elif isinstance(value, weakref.ref):
            refobj = value()
            value = f'ref({type(refobj).__name__})'
        elif isinstance(value, str) and (quoteStrings is True or (isinstance(quoteStrings, (tuple, list)) and attr in quoteStrings)):
            value = f'{quoteChar}{value}{quoteChar}'
        if hideKeys and attr in hideKeys:
            info.append(str(value))
        else:
            info.append(f'{attr}={value}')
    infostr = ', '.join(info)
    cls = type(obj).__name__
    return f"{cls}({infostr})"

