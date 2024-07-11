from __future__ import annotations

import tempfile
import warnings
import weakref
import sys
import os
import emlib.misc
from maelzel.common import F, getLogger
import functools
import appdirs
import logging


from typing import Callable, Sequence, Any, TYPE_CHECKING
if TYPE_CHECKING:
    import PIL.Image


logger = getLogger('maelzel')


def createTempdir() -> tempfile.TemporaryDirectory:
    """
    Creates a temporary directory within the user space, ensures that it is writable

    Returns:
        the tempfile.TemporaryDirectory object

    Raises IOError if it failes to create the temp directory
    """
    base = appdirs.user_cache_dir(appname='maelzel')
    os.makedirs(base, exist_ok=True)
    if not os.path.exists(base):
        raise IOError(f"Could not create base for temporary folder, tried '{base}'")
    tempdir = tempfile.TemporaryDirectory(dir=appdirs.user_cache_dir())
    if not os.path.exists(tempdir.name):
        raise IOError(f"Could not create temporary directory, '{tempdir.name}' does not exist")

    checkfile = tempfile.mktemp(dir=tempdir.name)
    assert not os.path.exists(checkfile)
    with open(checkfile, "w") as f:
        s = "check"
        numchars = f.write(s)
        assert numchars == len(s)
    if not os.path.exists(checkfile):
        raise IOError(f"Could not create temporary file '{checkfile}' in temporary directory '{tempdir.name}'")
    os.remove(checkfile)
    logger.debug(f"Created temporary directory, ensured it is writable: '{tempdir.name}'")
    return tempdir


_tempdir = createTempdir()


def mktemp(suffix: str, prefix='') -> str:
    """
    Drop-in replacement for tempfile.mktemp, uses a temp dir located under $HOME

    This prevents some errors with processes which do not have access to the
    root folder (and thus to "/tmp/..."), which is where tempfiles are created
    in linux and macos

    Args:
        suffix: a suffix to add to the file
        prefix: prefix to preprend to the file

    Returns:
        the path of the temporary file. Notice that as with tempfile.mktemp, this
        file is not created

    """
    return tempfile.mktemp(suffix=suffix, prefix=prefix, dir=_tempdir.name)


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
        if value is None or (hideFalsy and not value) or (hideEmptyStr and value == '') or (hideFalse and value is False):
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


def fuzzymatch(query: str, choices: Sequence[str], limit=5
               ) -> list[tuple[str, int]]:
    """
    Fuzzy matching

    Args:
        query: query to match
        choices: list of strings
        limit: max. number of matches

    Returns:
        list of tuples (matchingchoice: str, score: int)
    """
    if 'thefuzz.process' not in sys.modules:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import thefuzz.process
    return thefuzz.process.extract(query, choices=choices, limit=limit)


def checkChoice(name: str, s: str, choices: Sequence[str], maxSuggestions=12, throw=True, logger: logging.Logger=None
                ) -> bool:
    """
    Check than `name` is one of `choices`

    Args:
        name: what are we checking, used for error messages
        s: the value to check
        choices: possible choices
        maxSuggestions: possible choices shown when s does not match any
        throw: throw an exception if no match
        logger: if given, any error will be logged using this logger

    Returns:
        True if a match was found, False otherwise
    """
    if s not in choices:
        if logger:
            logger.error(f"Invalud value '{s}' for {name}, possible choices: {sorted(choices)}")

        if not throw:
            return False

        if len(choices) > 8:
            matches = fuzzymatch(s, choices, limit=maxSuggestions)
            raise ValueError(f'Invalid value "{s}" for {name}, maybe you meant "{matches[0][0]}"? '
                             f'Other possible choices: {[m[0] for m in matches]}')
        else:
            raise ValueError(f'Invalid value "{s}" for {name}, it should be one of {list(choices)}')


def readableTime(t: float) -> str:
    """
    Return t as readable time

    Args:
        t: the time in seconds

    Returns:
        a string representation of t with an appropriate unit
    """
    if t < 1e-6:
        return f"{t*1e9:.1f}ns"
    elif t < 1e-3:
        return f"{t*1e6:.1f}µs"
    elif t < 1:
        return f"{t*1e3:.1f}ms"
    else:
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
        num, den = limitDenominator(f.numerator, f.denominator, maxden=maxdenom)
        return f"~{num}/{den}"
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
    if not os.path.exists(pngpath):
        raise IOError(f"PNG does not exist: '{pngpath}'")

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


@functools.cache
def getPlatform(normalize=True) -> tuple[str, str]:
    """
    Return a string with current platform (system and machine architecture).

    Args:
        normalize: if True, architectures are normalized (see below)
    Returns:
        a tuple (osname: str, architecture: str)

    This attempts to improve upon `sysconfig.get_platform` by fixing some
    issues when running a Python interpreter with a different architecture than
    that of the system (e.g. 32bit on 64bit system, or a multiarch build),
    which should return the machine architecture of the currently running
    interpreter rather than that of the system (which didn't seem to work
    properly). The reported machine architectures follow platform-specific
    naming conventions (e.g. "x86_64" on Linux, but "x64" on Windows).
    Use normalize=True to reduce those labels (returns one of 'x86_64', 'arm64', 'x86')
    Example output for common platforms (normalized):

        ('darwin', 'arm64')
        ('darwin', 'x86_64')
        ('linux', 'x86_64')
        ('windows', 'x86_64')
        ...

    Normalizations:

    * aarch64 -> arm64
    * x64 -> x86_64
    * amd64 -> x86_64


    """
    import platform
    import sysconfig

    system = platform.system().lower()
    machine = sysconfig.get_platform().split("-")[-1].lower()
    is_64bit = sys.maxsize > 2 ** 32

    # Normalize system name
    if system == 'win32':
        system = 'windows'

    if system == "darwin":
        # get machine architecture of multiarch binaries
        if any([x in machine for x in ("fat", "intel", "universal")]):
            machine = platform.machine().lower()
        if machine not in ('x86_64', 'arm64'):
            raise RuntimeError(f"Unknown macos architecture '{machine}'")

    elif system == "linux":  # fix running 32bit interpreter on 64bit system
        if not is_64bit and machine == "x86_64":
            machine = "i686"
        elif not is_64bit and machine == "aarch64":
            machine = "armv7l"
        if machine not in ('x86', 'x86_64', 'arm64', 'armv7l', 'i686', 'i386'):
            raise RuntimeError(f"Unknown linux arch '{machine}'")
    elif system == "windows":
        # return more precise machine architecture names
        if machine == "amd64":
            machine = "x86_64"
        elif machine == "win32":
            if is_64bit:
                machine = platform.machine().lower()
            else:
                machine = "x86"
        if machine not in ('x86', 'x86_64', 'arm64', 'i386'):
            raise RuntimeError(f"Unknown windows architecture '{machine}'")
    else:
        raise RuntimeError(f"System '{system}' unknown")

    # some more fixes based on examples in https://en.wikipedia.org/wiki/Uname
    if not is_64bit and machine in ("x86_64", "amd64"):
        if any([x in system for x in ("cygwin", "mingw", "msys")]):
            machine = "i686"
        else:
            machine = "i386"

    if normalize:
        machine = {
            'x64': 'x86_64',
            'aarch64': 'arm64',
            'amd64': 'x86_64'
        }.get(machine, machine)

    return system, machine


def splitInterval(start: F, end: F, offsets: Sequence[F]
                  ) -> list[tuple[F, F]]:
    """
    Split interval (start, end) at the given offsets

    Args:
        start: start of the interval
        end: end of the interval
        offsets: offsets to split the interval at. Must be sorted

    Returns:
        a list of (start, end) segments where no segment extends over any
        of the given offsets
    """
    assert end > start
    assert offsets

    if offsets[0] > end or offsets[-1] < start:
        # no intersection, return the original time range
        return [(start, end)]

    out = []
    for offset in offsets:
        if offset >= end:
            break
        if start < offset:
            out.append((start, offset))
            start = offset
    if start != end:
        out.append((start, end))

    assert len(out) >= 1
    return out


def intersectF(u1: F, u2: F, v1: F, v2: F) -> tuple[F, F] | None:
    """
    return the intersection of (u1, u2) and (v1, v2) or None if no intersection

    Args:
        u1: lower bound of range U
        u2: higher bound of range U
        v1: lower bound of range V
        v2: higher bound of range V

    Returns:
        the intersection between range U and range V as a tuple (start, end).
        If no intersection is found, None is returned

    Example::

        >>> if intersect := intersection(0, 3, 2, 5):
        ...     start, end = intersect
        ...     ...

    """
    x0 = u1 if u1 > v1 else v1
    x1 = u2 if u2 < v2 else v2
    return (x0, x1) if x0 < x1 else None


def limitDenominator(num: int, den: int, maxden: int) -> tuple[int, int]:
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


