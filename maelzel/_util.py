from __future__ import annotations
import emlib.misc
import warnings
import sys
import os
import weakref
from maelzel.common import F


from typing import Callable, Sequence, TYPE_CHECKING
from maelzel.common import T
if TYPE_CHECKING:
    import PIL.Image


def reprObj(obj,
            filter: dict[str, Callable] = {},
            priorityargs: Sequence[str] = None,
            hideFalse=False,
            hideEmptyStr=False,
            hideFalsy=False,
            quoteStrings=False,
            convert: dict[str, Callable[[Any], str]] = None
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
        quoteStrings: if True, strings are quoted
        convert: if given, a dict mapping attr names to a function of the form (value) -> str,
            which returns the string representation of the given value

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
        elif convert and attr in convert:
            value = convert[attr](obj)
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


def readableTime(t: float) -> str:
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


def hasoverlap(x0: number_t, x1: number_t, y0: number_t, y1: number_t) -> bool:
    """ do (x0, x1) and (y0, y1) overlap? """
    return x1 > y0 if x0 < y0 else y1 > x0


def overlap(u1: number_t, u2: number_t, v1: number_t, v2: number_t) -> tuple[number_t, number_t]:
    """
    The overlap betwen (u1, u2) and (v1, v2)

    If there is no overlap, start > end

    Args:
        u1: start of first interval
        u2: end of first interval
        v1: start of second interval
        v2: end of second interval

    Returns:
        a tuple (overlapstart, overlapend). If no overlap, overlapstart > overlapend

    """
    x1 = u1 if u1 > v1 else v1
    x2 = u2 if u2 < v2 else v2
    return x1, x2


def aslist(seq) -> list:
    if isinstance(seq, list):
        return seq
    return list(seq)


def pngShow(pngpath: str, forceExternal=False, app: str = '',
            wait=False, inlineScale=1.0
            ) -> None:
    """
    Show a png either with an external app or inside jupyter

    Args:
        pngpath: the path to a png file
        forceExternal: if True, it will show in an external app even
            inside jupyter. Otherwise, it will show inside an external
            app if running a normal session and show an embedded
            image if running inside a notebook
        wait: if using an external app, wait until the app exits
        inlineScale: a scale factor to apply when showing inline within a notebook
        app: used if a specific external app is needed. Otherwise, the os
            defined app is used
    """
    if app:
        forceExternal = True

    if not forceExternal and pythonSessionType() == 'jupyter':
        from maelzel.core import jupytertools
        jupytertools.jupyterShowImage(path=pngpath, scalefactor=inlineScale)
    else:
        if app:
            emlib.misc.open_with_app(path=pngpath, app=app, wait=wait)
        else:
            emlib.misc.open_with_app(path=pngpath, wait=wait)


@emlib.misc.runonce
def pythonSessionType() -> str:
    """
    Returns the kind of python session

    .. note::
        See also `is_interactive_session` to check if we are inside a REPL

    Returns:
        Returns one of "jupyter", "ipython-terminal" (if running ipython
        in a terminal), "ipython" (if running ipython outside a terminal),
        "python" if running normal python.

    """
    try:
        # get_ipython should be available within an ipython/jupyter session
        shell = get_ipython().__class__.__name__   # type: ignore
        if shell == 'ZMQInteractiveShell':
            return "jupyter"
        elif shell == 'TerminalInteractiveShell':
            return "ipython-terminal"
        else:
            return "ipython"
    except NameError:
        return "python"


def imageAutocrop(img: PIL.Image.Image | str, bgcolor: str | tuple[int, int, int]
                  ) -> PIL.Image.Image | None:
    from PIL import Image, ImageChops
    imgobj = img if isinstance(img, Image.Image) else Image.open(img)

    if imgobj.mode != "RGB":
        imgobj = imgobj.convert("RGB")
    bg = Image.new("RGB", imgobj.size, bgcolor)
    diff = ImageChops.difference(imgobj, bg)
    bbox = diff.getbbox()
    if bbox:
        return imgobj.crop(bbox)
    return None


def imagefileAutocrop(imgfile: str, outfile: str, bgcolor: str | tuple[int, int, int]) -> bool:
    imgobj = imageAutocrop(imgfile, bgcolor=bgcolor)
    if imgobj is None:
        return False
    imgobj.save(outfile)
    return True

