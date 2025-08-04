from __future__ import annotations

import functools
import os
import sys
import weakref


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Callable, Sequence
    import logging
    from maelzel.common import F, num_t
    import tempfile


_cache = {}


def createTempdir(check=True) -> tempfile.TemporaryDirectory:
    """
    Creates a temporary directory within the user space, ensures that it is writable

    Returns:
        the tempfile.TemporaryDirectory object

    Raises IOError if it failes to create the temp directory
    """
    import appdirs
    base = appdirs.user_cache_dir(appname='maelzel')
    os.makedirs(base, exist_ok=True)
    if not os.path.exists(base):
        raise IOError(f"Could not create base for temporary folder, tried '{base}'")
    import tempfile
    tempdir = tempfile.TemporaryDirectory(dir=appdirs.user_cache_dir())
    if not os.path.exists(tempdir.name):
        raise IOError(f"Could not create temporary directory, '{tempdir.name}' does not exist")

    checkfile = tempfile.mktemp(dir=tempdir.name)
    assert not os.path.exists(checkfile)
    if check:
        with open(checkfile, "w") as f:
            s = "check"
            numchars = f.write(s)
            assert numchars == len(s)
        if not os.path.exists(checkfile):
            raise IOError(f"Could not create temporary file '{checkfile}' in temporary directory '{tempdir.name}'")
        os.remove(checkfile)
    return tempdir


def sessionTempdir() -> tempfile.TemporaryDirectory:
    """
    Creates a temporary directory within user space, valid for a session

    It is only run once, after the first run it returns always the same dir
    """
    if (tempdir := _cache.get('tempdir')) is not None:
        return tempdir
    _cache['tempdir'] = tempdir = createTempdir()
    return tempdir



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
    tempdir = sessionTempdir()
    import tempfile
    return tempfile.mktemp(suffix=suffix, prefix=prefix, dir=tempdir.name)


def reprObj(obj,
            exclude: Sequence[str] | None = None,
            properties: Sequence[str] | None = None,
            filter: dict[str, Callable] = {},
            priorityargs: Sequence[str] | None = None,
            hideFalse=False,
            hideEmptyStr=False,
            hideFalsy=False,
            hideKeys: Sequence[str] | None = None,
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
        priorityargs: a list of attributes which are shown first.
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
    import emlib.misc
    attrs = emlib.misc.find_attrs(obj)
    if exclude:
        import fnmatch
        attrs = [a for a in attrs if any(fnmatch.fnmatch(a, p) for p in exclude)]
    if properties:
        for p in properties:
            if p not in attrs:
                attrs.append(p)
    info = []
    if hideKeys:
        if priorityargs:
            for key in hideKeys:
                if key not in priorityargs:
                    priorityargs += (key,)
        else:
            priorityargs = hideKeys
    if sort:
        attrs.sort()
    if priorityargs:
        attrs.sort(key=lambda attr: 0 if attr in priorityargs else 1)
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
        import warnings
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
    if s in choices:
        return True

    if logger:
        logger.error(f"Invalid value '{s}' for {name}, possible choices: {sorted(choices)}")

    if not throw:
        return False

    if len(choices) > maxSuggestions:
        matches = fuzzymatch(s, choices, limit=maxSuggestions)
        raise ValueError(f'Invalid value "{s}" for {name}, maybe you meant "{matches[0][0]}"? '
                            f'Other possible choices: {[m[0] for m in matches]}')
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


def showF(f: F, maxdenom=1000, approxAsFloat=False, unicode=False) -> str:
    """
    Show a fraction, limit den to *maxdenom*

    Args:
        f: the fraction to show
        maxdenom: the max. denominator to show

    Returns:
        a readable string representation
    """
    if f.denominator == 1:
        return str(f.numerator)
    if f.denominator > maxdenom:
        if approxAsFloat:
            from . import mathutils
            num, den = mathutils.limitDenominator(f.numerator, f.denominator, maxden=maxdenom, assumeCoprime=True)
            return f"~{num}/{den}"
        else:
            return f"{f:.3f}".rstrip('0').rstrip('.')
    if unicode:
        return unicodeFraction(f.numerator, f.denominator, multi=True)
    return "%d/%d" % (f.numerator, f.denominator)


def showT(f: F | float | None) -> str:
    """Show *f* as time"""
    if f is None:
        return "None"
    if not isinstance(f, float):
        f = float(f)
    return f"{f:.3f}".rstrip('0').rstrip('.')


def hasoverlap(x0: num_t, x1: num_t, y0: num_t, y1: num_t) -> bool:
    """ do (x0, x1) and (y0, y1) overlap? """
    return x1 > y0 if x0 < y0 else y1 > x0


def overlap(u1: num_t, u2: num_t, v1: num_t, v2: num_t) -> tuple[num_t, num_t]:
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
        import emlib.misc
        if app:
            emlib.misc.open_with_app(path=pngpath, app=app, wait=wait)
        else:
            emlib.misc.open_with_app(path=pngpath, wait=wait)


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
    if (session := _cache.get('sessiontype')) is not None:
        return session
    try:
        # get_ipython should be available within an ipython/jupyter session
        shell = get_ipython().__class__.__name__   # type: ignore
        if shell == 'ZMQInteractiveShell':
            session = "jupyter"
        elif shell == 'TerminalInteractiveShell':
            session = "ipython-terminal"
        else:
            session = "ipython"
        _cache['sessiontype'] = session
        return session
    except NameError:
        return "python"


def getPlatform() -> tuple[str, str]:
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
    if (out := _cache.get('getPlatform')) is not None:
        assert isinstance(out, tuple)
        return out

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

    machine = {
        'x64': 'x86_64',
        'aarch64': 'arm64',
        'amd64': 'x86_64'
    }.get(machine, machine)
    _cache['getPlatform'] = out = (system, machine)
    return out


@functools.cache
def _unicodeReplacer(full=True) -> Callable[[str], str]:
    import emlib.textlib
    if full:
        return emlib.textlib.makeReplacer({
            '#>': '𝄰',
            '#<': '𝄱',
            'b<': '𝄭',
            'b>': '𝄬',
            '#': '♯',
            'b': '♭',
            '>': '↑',
            '<': '↓'})
    else:
        return emlib.textlib.makeReplacer({
            # '#>': '𝄰',
            # '#<': '𝄱',
            # 'b<': '𝄭',
            # 'b>': '𝄬',
            '#': '♯',
            'b': '♭',
            '>': '↑',
            '<': '↓'})



@functools.cache
def unicodeNotename(notename: str, full=True) -> str:
    """
    Replace ascii accidentals with unicode accidentals

    C#+45   C♯+45
    Db-15   D♭-15

    Args:
        notename: the note name
        full: replace compound alterations to merged alterations (``#>`` to ``𝄰``)

    Returns:
        the replacement
    """
    return _unicodeReplacer(full=full)(notename)


def unicodeFraction(numerator: int, denominator: int, multi=False) -> str:
    """
    Convert a fraction to its Unicode representation.

    Args:
        numerator (int): The numerator of the fraction
        denominator (int): The denominator of the fraction
        multi: if True, force multigliph output, even for known fractions for which
            a gliph exists

    Returns:
        str: Unicode representation of the fraction

    Examples:
        >>> fraction_to_unicode(1, 2)
        '½'
        >>> fraction_to_unicode(3, 4)
        '¾'
        >>> fraction_to_unicode(22, 7)
        '²²⁄₇'
    """
    # Handle zero numerator
    if numerator == 0:
        return "0"

    # Handle negative fractions
    negative = (numerator < 0) ^ (denominator < 0)
    numerator = abs(numerator)
    denominator = abs(denominator)

    # Common Unicode fractions
    if not multi:
        commonFractions = {
            (1, 2): "½",
            (1, 3): "⅓",
            (2, 3): "⅔",
            (1, 4): "¼",
            (3, 4): "¾",
            (1, 5): "⅕",
            (2, 5): "⅖",
            (3, 5): "⅗",
            (4, 5): "⅘",
            (1, 6): "⅙",
            (5, 6): "⅚",
            (1, 7): "⅐",
            (1, 8): "⅛",
            (3, 8): "⅜",
            (5, 8): "⅝",
            (7, 8): "⅞",
            (1, 9): "⅑",
            (1, 10): "⅒",
        }

        # Check if it's a common fraction
        if (numerator, denominator) in commonFractions:
            result = commonFractions[(numerator, denominator)]
            return f"−{result}" if negative else result

    # Superscript digits for numerator
    superscripts = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }

    # Subscript digits for denominator
    subscripts = {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
    }

    superscript = "".join(superscripts[digit] for digit in str(numerator))
    subscript = "".join(subscripts[digit] for digit in str(denominator))
    # Combine with fraction slash
    result = f"{superscript}⁄{subscript}"
    return f"−{result}" if negative else result


def fileIsLocked(filepath: str) -> bool:
    assert os.path.exists(filepath)
    try:
        bufsize = 8
        # Opening file in append mode and read the first 8 characters.
        fileobj = open(filepath, mode='a', buffering=bufsize)
        if fileobj:
            locked = False
    except IOError:
        locked = True
    finally:
        if fileobj:
            fileobj.close()
    return locked


def waitForFile(filepath: str, period=0.1, timeout=1) -> None:
    import time
    accumtime = 0.
    while fileIsLocked(filepath):
        if accumtime > timeout:
            raise TimeoutError
        time.sleep(period)
        accumtime += period


def htmlImage64(img64: bytes | str, imwidth: int, width: int | str = '', scale=1.,
                maxwidth: int | str = '', margintop='14px', padding='10px') -> str:
    """
    Generate html for displaying an image as base64

    Args:
        img64: an image convertes to bytes, as returned by readImageAsBase64
        imwidth: width of the original image, in pixels
        width: a css width
        scale: alternatively, a scaling width
        maxwidth: a max. width to be applied, either in pixels or as a css rule
        margintop: css top margin
        padding: css padding

    Returns:
        the html generated
    """
    if scale and not width:
        width = int(imwidth * scale)
    attrs = [f'padding:{padding}',
             f'margin-top:{margintop}']
    if maxwidth:
        if isinstance(maxwidth, int):
            maxwidth = f'{maxwidth}px'
        attrs.append(f'max-width: {maxwidth}')
    if width is not None:
        if isinstance(width, int):
            width = f'{width}px'
        attrs.append(f'width:{width}')
    s = img64 if isinstance(img64, str) else img64.decode()
    style = ";\n".join(attrs)
    return fr'''<img style="display:inline; {style}" src="data:image/png;base64,{s}"/>'''

