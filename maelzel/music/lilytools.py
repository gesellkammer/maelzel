import os
import sys
import logging
import subprocess
import tempfile
from typing import List, Tuple as Tup, Optional as Opt, Union as U

logger = logging.getLogger("maelzel.lilytools")

_str = Opt[str]


class PlatformNotSupported(Exception):
    pass


def _logged_call(args: U[str, List[str]], shell=False) -> Tup[str, int, str]:
    """
    Call a subprocess with args

    Returns output, return code, error message
    """
    proc = subprocess.Popen(args,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=shell)
    retcode = proc.wait()
    out = proc.stdout.read().decode("utf-8")
    if retcode == 0:
        error = ""
    else:
        error = (retcode, proc.stderr.read().decode("utf-8"))
    return out, retcode, error


def _check_output(args: List[str], encoding="utf-8") -> _str:
    """
    Like subprocess.check_output, but returns None if failed instead
    of throwing an exeception
    """
    try:
        out = subprocess.check_output(args)
        return out.decode(encoding)
    except subprocess.CalledProcessError:
        return None


def find_lilypond() -> _str:
    """
    Find lilypond binary, or None if not found
    """
    platform = os.uname()[0].lower()
    if platform == 'linux':
        path = _check_output(["which", "lilypond"])
        if path is not None:
            path = path.strip()
            assert os.path.exists(path)
            return path
        paths = ("/usr/bin/lilypond", "/usr/local/bin/lilypond",
                 "~/.local/bin/lilypond")
        paths = [os.path.expanduser(p) for p in paths]
        for path in paths:
            if os.path.exists(path):
                return path
        return None
    elif platform == 'darwin':
        paths = ['/Applications/LilyPond.app/Contents/Resources/bin/lilypond']
        paths = [os.path.expanduser(p) for p in paths]
        for path in paths:
            if os.path.exists(path):
                return path
        return None
    else:
        raise PlatformNotSupported(f"Platform {platform} is not supported")


def musicxml2ly(xmlfile: str, outfile: _str = None) -> str:
    if outfile is None:
        outfile = os.path.splitext(xmlfile)[0] + '.ly'
    subprocess.call(['musicxml2ly', '-o', outfile, xmlfile])
    return outfile


def _lily(lilyfile: str, outfile: _str = None, fmt='pdf'):
    assert fmt in ('pdf', 'png', 'ps')
    assert os.path.exists(lilyfile)
    if outfile is None:
        outfile = lilyfile
    basefile = os.path.splitext(outfile)[0]
    out = f'{basefile}.{fmt}'
    if sys.platform == "win32":
        # in windows we call lilypond through the shell
        s = f'lilypond --{fmt} -o "{out}" "{lilyfile}"'
        output, retcode, error = _logged_call(s, shell=True)
    else:
        # in unix we can find the binary so we can call directly
        lilybinary = find_lilypond()
        output, retcode, error = _logged_call(
            [lilybinary, f'--{fmt}', '-o', basefile, lilyfile])
    if not os.path.exists(out):
        logger.error(f"Failed to produce a {fmt} file: {out}")
        return None
    if error:
        logger.error(f"Error while running lilypond: {output}")
        logger.error(error)
        return None
    return out


def lily2pdf(lilyfile: str, outfile: _str = None) -> _str:
    """
    Call lilypond to generate a pdf file.
    Returns the path to the generated file, or
    None if there was an error
    """
    return _lily(lilyfile, outfile, fmt='pdf')


def lily2png(lilyfile: str, outfile: _str = None, simple=True) -> str:
    if simple:
        tmp = tempfile.mktemp(suffix='.ly')
        postprocess(lilyfile, outfile=tmp, remove_header=True, book=True)
    else:
        tmp = lilyfile
    outfile = _lily(tmp, outfile, fmt='png')
    if simple:
        os.remove(tmp)
    return outfile


def postprocess(lilyfile: str, outfile: str, remove_header=True,
                book=True) -> None:
    import re

    def _remove_header(s):
        header = re.search(r"\\header\s?\{[^\}]+\}", s)
        if header:
            s = s[:header.span()[0]] + '\n\\header {}\n' + s[header.span()[1]:]
        return s

    def _add_preamble(s):
        version = re.search(r"\\version.+", s)
        if version:
            preamble = '\n\\include "lilypond-book-preamble.ly"\n'
            s = s[:version.span()[1]] + preamble + s[version.span()[1]:]
        return s

    s = open(lilyfile).read()
    if remove_header:
        s = _remove_header(s)
    if book:
        s = _add_preamble(s)
    open(outfile, 'w').write(s)
