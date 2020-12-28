from __future__ import annotations

import math as _math
import os 
import sys
import subprocess
import re
from collections import namedtuple
import shutil as _shutil
import logging as _logging
import textwrap as _textwrap
import io
import tempfile
import cachetools
from dataclasses import dataclass
from typing import List, Union as U, Optional as Opt, Generator, Sequence as Seq, Dict, Set, Tuple

import numpy as np

from emlib import misc, filetools


logger = _logging.getLogger("maelzel.csound")


@dataclass
class AudioBackend:
    name: str
    alwaysAvailable: bool
    supportsSystemSr: bool
    needsRealtime: bool
    platforms: List[str]
    dac: str = "dac"  
    adc: str = "adc"
    longname: str = ""

    def __post_init__(self):
        if not self.longname:
            self.longname = self.name

    def isAvailable(self):
        return is_backend_available(self.name)


_backend_jack = AudioBackend('jack',
                             alwaysAvailable=False,
                             supportsSystemSr=True,
                             needsRealtime=False,
                             platforms=['linux', 'darwin', 'win32'],
                             dac="dac:system:playback_",
                             adc="adc:system:capture_")

_backend_pacb = AudioBackend('pa_cb',
                             alwaysAvailable=True,
                             supportsSystemSr=False,
                             needsRealtime=False,
                             longname="portaudio-callback",
                             platforms=['linux', 'darwin', 'win32'])

_backend_pabl = AudioBackend('pa_bl',
                             alwaysAvailable=True,
                             supportsSystemSr=False,
                             needsRealtime=False,
                             longname="portaudio-blocking",
                             platforms=['linux', 'darwin', 'win32'])

_backend_auhal = AudioBackend('auhal',
                              alwaysAvailable=True,
                              supportsSystemSr=True,
                              needsRealtime=False,
                              longname="coreaudio",
                              platforms=['darwin'])

_backend_pulse = AudioBackend('pulse',
                              alwaysAvailable=False,
                              supportsSystemSr=False,
                              needsRealtime=False,
                              longname="pulseaudio",
                              platforms=['linux'])

_backend_alsa = AudioBackend('alsa',
                             alwaysAvailable=True,
                             supportsSystemSr=False,
                             needsRealtime=True,
                             platforms=['linux'])

audio_backends: Dict[str, AudioBackend] = {
    'jack' : _backend_jack,
    'auhal': _backend_auhal,
    'pa_cb': _backend_pacb,
    'pa_bl': _backend_pabl,
    'pulse': _backend_pulse,
    'alsa' : _backend_alsa
}


_platform_backends: Dict[str, List[AudioBackend]] = {
    'linux': [_backend_jack, _backend_pacb, _backend_alsa, _backend_pabl, _backend_pulse],
    'darwin': [_backend_jack, _backend_auhal, _backend_pacb],
    'win32': [_backend_pacb, _backend_pabl]
}

"""
helper functions to work with csound
"""



_csoundbin = None
_OPCODES = None


# --- Exceptions ---

class PlatformNotSupported(Exception): pass
class AudioBackendNotAvailable(Exception): pass


def nextpow2(n:int) -> int:
    return int(2 ** _math.ceil(_math.log(n, 2)))
    

def find_csound() -> Opt[str]:
    global _csoundbin
    if _csoundbin:
        return _csoundbin
    csound = _shutil.which("csound")
    if csound:
        _csoundbin = csound
        return csound
    logger.error("csound is not in the path!")
    if sys.platform.startswith("linux") or sys.platform == 'darwin':
        for path in ['/usr/local/bin/csound', '/usr/bin/csound']:
            if os.path.exists(path) and not os.path.isdir(path):
                _csoundbin = path
                return path
        return None
    elif sys.platform == 'win32':
        return None
    else:
        raise PlatformNotSupported
    

def get_version() -> Tuple[int, int, int]:
    """
    Returns the csound version as tuple (major, minor, patch) so that '6.03.0' is (6, 3, 0)

    Raises IOError if either csound is not present or its version 
    can't be parsed
    """
    csound = find_csound()
    if not csound:
        raise IOError("Csound not found")
    cmd = '{csound} --help'.format(csound=csound).split()
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    proc.wait()
    lines = proc.stderr.readlines()
    if not lines:
        raise IOError("Could not read csounds output")
    for line in lines:
        if line.startswith(b"Csound version"):
            line = line.decode('utf8')
            matches = re.findall(r"(\d+\.\d+(\.\d+)?)", line)
            if matches:
                version = matches[0]
                if isinstance(version, tuple):
                    version = version[0]
                points = version.count(".")
                if points == 1:
                    major, minor = list(map(int, version.split(".")))
                    patch = 0
                else:
                    major, minor, patch = list(map(int, version.split(".")[:3]))
                return (major, minor, patch)
    else:
        raise IOError("Did not found a csound version")


def csound_subproc(args, piped=True):
    """
    Calls csound with the given args in a subprocess, returns
    such subprocess. 

    """
    csound = find_csound()
    if not csound:
        return
    p = subprocess.PIPE if piped else None
    callargs = [csound]
    callargs.extend(args)
    logger.debug(f"csound_subproc> args={callargs}")
    return subprocess.Popen(callargs, stderr=p, stdout=p)
    

def get_default_backend() -> AudioBackend:
    """
    Get the default backend for platform. Check if the backend
    is available and running (in the case of Jack)
    """
    return get_audiobackends()[0]


def run_csd(csdfile:str, 
            output = "", 
            input = "", 
            backend = "",
            supressdisplay = False,
            comment:str = None,
            piped = False,
            extra:List[str] = None) -> subprocess.Popen:
    """
    Args:
        csdfile: the path to a .csd file
        output: "dac" to output to the default device, the label of the
            device (dac0, dac1, ...), or a filename to render offline
            (-o option)
        input: The input to use (for realtime) (-i option)
        backend: The name of the backend to use. If no backend is given,
            the default for the platform is used (this is only meaningful
            if running in realtime)
        supressdisplay: if True, eliminates debugging info from output
        piped: if True, the output of the csound process is piped and can be accessed
            through the Popen object (.stdout, .stderr)
        extra: a list of extra arguments to be passed to csound
        comment: if given, will be added to the generated soundfile
            as comment metadata

    Returns:
        the subprocess.Popen object. In order to wait until
        rendering is finished in offline mode, call .wait on the
        returned process
    """
    args = []
    realtime = False
    if output:
        args.extend(["-o", output])
        if output.startswith("dac"):
            realtime = True
    if realtime and not backend:
        backend = get_default_backend().name
    if backend:
        args.append(f"-+rtaudio={backend}")
    if input:
        args.append(f"-i {input}")
    if supressdisplay:
        args.extend(['-d', '-m', '0'])
    if comment and not realtime:
        args.append(f'-+id_comment="{comment}"')
    if extra:
        args.extend(extra)
    args.append(csdfile)
    return csound_subproc(args, piped=piped)
    

def join_csd(orc: str, sco="", options:List[str] = None) -> str: 
    """
    Join an orc and a score (both as a string) and return a consolidated
    csd structure (as string)

    open("out.csd", "w").write(join_csd(orc, sco, options))
    """
    optionstr = "" if options is None else "\n".join(options)
    csd = r"""
<CsoundSynthesizer>
<CsOptions>
{optionstr}
</CsOptions>
<CsInstruments>

{orc}

</CsInstruments>
<CsScore>

{sco}

</CsScore>
</CsoundSynthesizer>
    """.format(optionstr=optionstr, orc=orc, sco=sco)
    csd = _textwrap.dedent(csd)
    return csd


def test_csound(dur=8, nchnls=2, backend=None, device="dac", sr=None, verbose=True):
    backend = backend or get_default_backend().name
    sr = sr or get_sr(backend)
    printchan = "printk2 kchn" if verbose else ""
    orc = f"""
sr = {sr}
ksmps = 128
nchnls = {nchnls}

instr 1
    iperiod = 1
    kchn init -1
    ktrig metro 1/iperiod
    kchn = (kchn + ktrig) % nchnls 
    anoise pinker
    outch kchn+1, anoise
    {printchan}
endin
    """
    sco = f"i1 0 {dur}"
    orc = _textwrap.dedent(orc)
    logger.debug(orc)
    csd = join_csd(orc, sco=sco)
    tmp = tempfile.mktemp(suffix=".csd")
    open(tmp, "w").write(csd)
    return run_csd(tmp, output=device, backend=backend)
    

ScoreEvent = namedtuple("ScoreEvent", "kind name start dur args")


def parse_sco(sco: str) -> Generator[ScoreEvent]:
    for line in sco.splitlines():
        words = line.split()
        w0 = words[0]
        if w0 == 'i':
            name = words[1]
            t0 = float(words[2])
            dur = float(words[3])
            rest = words[4:]
        elif w0[0] == 'i':
            name = w0[1:]
            t0 = float(words[1])
            dur = float(words[2])
            rest = words[3:]
        else:
            continue
        yield ScoreEvent("i", name, t0, dur, rest)


def get_opcodes(cached=True) -> List[str]:
    """
    Return a list of the opcodes present
    """
    global _OPCODES
    if _OPCODES is not None and cached:
        return _OPCODES
    s = csound_subproc(['-z'])
    lines = s.stderr.readlines()
    allopcodes = []
    for line in lines:
        if line.startswith(b"end of score"):
            break
        opcodes = line.decode('utf8').split()
        if opcodes:
            allopcodes.extend(opcodes)
    _OPCODES = allopcodes
    return allopcodes

   
def save_as_gen23(data: Seq[float], outfile:str, fmt="%.12f", header="") -> None:
    """
    Saves the points to a gen23 table

    NB: gen23 is a 1D list of numbers in text format, sepparated
        by a space

    data: seq
        A 1D sequence (list or array) of floats
    outfile: path
        The path to save the data to. Recommended extension: '.gen23'
    fmt: 
        If saving frequency tables, fmt can be "%.1f" and save space,
        for amplitude the default if "%.12f" is best
    header: str
        If specified it is included as a comment as the first line
        Csound will skip it. It is there just to document what is
        in the table

    Example:

    >>> a = bpf.linear(0, 0, 1, 10, 2, 300)
    >>> sampling_period = 0.01
    >>> points_to_gen23(a[::sampling_period].ys, "out.gen23", header=f"dt={sampling_period}")
    
    In csound

    gi_tab ftgen 0, 0, 0, -23, "out.gen23"
 
    instr 1
      itotaldur = ftlen(gi_tab) * 0.01
      ay poscil 1, 1/itotaldur, gi_tab
    endin
    """
    if header:
        np.savetxt(outfile, data, fmt=fmt, header="# " + header)
    else:
        np.savetxt(outfile, data, fmt=fmt)


def matrix_as_wav(outfile: str, mtx: np.ndarray, dt: float, t0=0.) -> None:
    """
    Save the data in m as a wav file. This is not a real soundfle
    but it is used to transfer the data in binary form to be 
    read in csound

    Format:

        header: headerlength, dt, numcols, numrows
        rows: each row has `numcol` number of items

    outfile: str
        the path where the data is written to (must have a .wav extension)
    mtx: a numpy array of shape (numcols, numsamples)
        a 2D matrix representing a series of streams sampled at a 
        regular period (dt)
    dt: float
        metadata: the sampling period of the matrix
    t0: float
        metadata: the sampling offset of the matrix
        (t = row*dt + t0)
    """
    assert isinstance(outfile, str)
    assert isinstance(dt, float)
    assert isinstance(t0, float)
    import sndfileio
    mtx_flat = mtx.ravel()
    numrows, numcols = mtx.shape
    header = np.array([5, dt, numcols, numrows, t0], dtype=float)
    sndwriter = sndfileio.sndwrite_chunked(sr=44100, outfile=outfile, encoding="flt32")
    sndwriter.write(header)
    sndwriter.write(mtx_flat)
    sndwriter.close()


def matrix_as_gen23(outfile: str, mtx: np.ndarray, dt:float, t0=0, include_header=True) -> None:
    numrows, numcols = mtx.shape
    mtx = mtx.round(6)
    with open(outfile, "w") as f:
        if include_header:
            header = np.array([5, dt, numcols, numrows, t0], dtype=float)
            f.write(" ".join(header.astype(str)))
            f.write("\n")
        for row in mtx:
            rowstr = " ".join(row.astype(str))
            f.write(rowstr)
            f.write("\n")


@dataclass
class AudioDevice:
    index: int
    label: str
    name: str
    kind: str


def get_audiodevices(backend:str=None) -> Tuple[List[AudioDevice], List[AudioDevice]]:
    """
    Returns (indevices, outdevices), where each of these lists 
    is an AudioDevice

    backend: 
        specify a backend supported by your installation of csound
        None to use a default for you OS
    label: 
        is something like 'adc0', 'dac1' and is what you
        need to pass to csound to its -i or -o methods. 
    name: 
        the name of the device. Something like "Built-in Input"

    Backends:

            OSX  Linux  Win   Multiple-Devices    Description
    jack     x      x    -     -                  Jack
    auhal    x      -    -     x                  CoreAudio
    pa_cb    x      x    x     x                  PortAudio (Callback)
    pa_bl    x      x    x     x                  PortAudio (blocking)
    """
    if not backend:
        backend = get_default_backend().name
    indevices, outdevices = [], []
    proc = csound_subproc(['-+rtaudio=%s' % backend, '--devices'])
    proc.wait()
    lines = proc.stderr.readlines()
    # regex_all = r"([0-9]+):\s(adc[0-9]+|dac[0-9]+)\s\((.+)\)"
    regex_all = r"([0-9]+):\s((?:adc|dac).+)\s\((.+)\)"
    for line in lines:
        line = line.decode("ascii")
        match = re.search(regex_all, line)
        if not match:
            continue
        idxstr, devid, devname = match.groups()
        isinput = devid.startswith("adc")
        dev = AudioDevice(int(idxstr), devid, devname, kind="input" if isinput else "output")
        if isinput:
            indevices.append(dev)
        else:
            outdevices.append(dev)
    return indevices, outdevices


def get_sr(backend: U[str, AudioBackend]) -> float:
    """
    Returns the samplerate reported by the given backend, or
    0 if failed

    """
    FAILED = 0

    audiobackend = (backend if isinstance(backend, AudioBackend) else 
                    audio_backends[backend])

    if not audiobackend.isAvailable():
        raise AudioBackendNotAvailable

    if audiobackend.supportsSystemSr:
        return 44100

    if audiobackend.name == 'jack' and _shutil.which('jack_samplerate') is not None:
        sr = int(subprocess.getoutput("jack_samplerate"))
        return sr

    proc = csound_subproc(f"-odac -+rtaudio={backend} --get-system-sr".split())
    proc.wait()
    srlines = [line for line in proc.stdout.readlines() 
               if line.startswith(b"system sr:")]
    if not srlines:
        logger.error(f"get_sr: Failed to get sr with backend {backend}")
        return FAILED
    sr = float(srlines[0].split(b":")[1].strip())
    logger.debug(f"get_sr: sample rate query output: {srlines}")
    return sr if sr > 0 else FAILED


def _jack_is_available() -> bool:
    if sys.platform == 'linux' and _shutil.which('jack_control') is not None:
        status = int(subprocess.getstatusoutput("jack_control status")[0])
        return status == 0
    proc = csound_subproc(['+rtaudio=jack', '--get-system-sr'])
    proc.wait()
    return b'JACK module enabled' in proc.stderr.read()


# cache the result for 30 seconds
@cachetools.cached(cache=cachetools.TTLCache(1, 30))
def is_backend_available(backend: str) -> bool:
    if backend == 'jack':
        return _jack_is_available()
    else:
        indevices, outdevices = get_audiodevices(backend=backend)
        return bool(indevices or outdevices)


@cachetools.cached(cache=cachetools.TTLCache(1, 30))
def get_audiobackends() -> List[AudioBackend]:
    """
    Return a list of supported audio backends as they would be passed
    to -+rtaudio

    Only those backends currently available are returned
    (for example, jack will not be returned in linux if the
    jack server is not running)

    """
    backends = _platform_backends[sys.platform]
    backends = [backend for backend in backends if backend.isAvailable()]
    return backends


def get_audiobackends_names() -> List[str]:
    """
    Similar to get_audiobackends, but returns the names of the
    backends. The AudioBackend class can be retrieved via

    audio_backends[backend_name]

    Returns:
        a list with the names of all available backends for the
        current platform
    """
    backends = get_audiobackends()
    return [backend.name for backend in backends]


def _wrap_string(arg):
    return arg if not isinstance(arg, str) else '"' + arg + '"'


def _event_start(event:tuple) -> float:
    kind = event[0]
    if kind == "e":
        return event[1]
    else:
        return event[2]

_normalizer = misc.make_replacer({".":"_", ":":"_", " ":"_"})

def normalize_instrument_name(name):
    """
    Transform name so that it can be accepted as an instrument name
    """
    return _normalizer(name)


_fmtoptions = {
    16         : '',
    24         : '--format=24bit',
    '24bit'    : '--format=24bit',
    32         : '--format=float',  # also -f
    'float'    : '--format=float',  # also -f
    'double'   : '--format=double'
}


_csound_format_options = {'-3', '-f', '--format=24bit', '--format=float',
                          '--format=double', '--format=long', '--format=vorbis',
                          '--format=short'}


def sample_format_option(fmt) -> str:
    return _fmtoptions.get(fmt)


class Csd:
    def __init__(self, sr=44100, ksmps=64, nchnls=2, a4=442, options=None,
                 supress_display=False):
        self.score: List[U[list, tuple]] = []
        self.instrs: Dict[U[str, int], str] = {}
        self.globalcodes: List[str] = []
        self.options: List[str] = []
        if options:
            self.set_options(*options)
        self.sr = sr
        self.ksmps = ksmps
        self.nchnls = nchnls
        self.a4 = a4
        self._sample_format: Opt[str] = None
        self._defined_ftables = set()
        self._min_ftable_index = 1
        if supress_display:
            pass # TODO
        
    def add_event(self, 
                  instr: U[int, float, str], 
                  start: float, 
                  dur: float, 
                  args: List[float] = None) -> None:
        """
        Add an instrument ("i") event to the score

        Args:

            instr: the instr number or name, as passed to add_instr
            start: the start time
            dur: the duration of the event
            args: pargs beginning at p4
        """
        event = ["i", _wrap_string(instr), start, dur]
        if args:
            event.extend(_wrap_string(arg) for arg in args)
        self.score.append(event)

    def _assign_ftable_index(self, tabnum=0) -> int:
        defined_ftables = self._defined_ftables
        if tabnum > 0:
            if tabnum in defined_ftables:
                raise ValueError(f"ftable {tabnum} already defined")
        else:
            for tabnum in range(self._min_ftable_index, 9999):
                if tabnum not in defined_ftables:
                    break
            else:
                raise IndexError("All possible ftable slots used!")
        defined_ftables.add(tabnum)
        return tabnum


    def _add_ftable(self, pargs) -> int:
        """
        Adds an ftable to the score

        Args:
            pargs: as passed to csound (without the "f")
                p1 can be 0, in which case a table number
                is assigned

        Returns:
            The index of the new ftable
        """
        tabnum = self._assign_ftable_index(pargs[0])
        pargs = ["f", tabnum] + pargs[1:]
        self.score.append(pargs)
        return tabnum

    def add_ftable_from_seq(self, seq: Seq[float], tabnum:int=0, start=0
                            ) -> int:
        """
        Create a ftable, fill it with seq, return the ftable index

        Args:
            seq: a sequence of floats to fill the table. The size of the
                table is determined by the size of the seq.
            tabnum: 0 to auto-assign an index
            start: the same as f 1 2 3

        Returns:

        """
        pargs = [tabnum, start, -len(seq), -2]
        pargs.extend(seq)
        return self._add_ftable(pargs)

    def add_empty_ftable(self, size:int, tabnum: int=0) -> int:
        """

        Args:
            tabnum: use 0 to autoassign an index
            size: the size of the empty table

        Returns:
            The index of the created table
        """
        pargs = (tabnum, 0, -size, -2, 0)
        return self._add_ftable(pargs)

    def add_sndfile(self, sndfile, tabnum=0, start=0):
        tabnum = self._assign_ftable_index(tabnum)
        pargs = [tabnum, start, 0, -1, sndfile, 0, 0, 0]
        self._add_ftable(pargs)
        return tabnum

    def destroy_table(self, tabnum:int, time:float) -> None:
        """
        Schedule ftable with index `tabnum` to be destroyed
        at time `time`

        Args:
            tabnum: the index of the table to be destroyed
            time: the time to destroy it
        """
        pargs = ("f", -tabnum, time)
        self.score.append(pargs)

    def set_end_marker(self, dur: float):
        """
        Add an end marker to the score
        """
        self.score.append(("e", dur))

    def set_comment(self, comment:str):
        self.set_options(f'-+id_comment="{comment}"')

    def set_sample_format(self, fmt) -> None:
        """
        Set the sample format for recording

        :param fmt: one of 16, 24, 32, 'float', 'double'
        :return:
        """
        option = sample_format_option(fmt)
        if option is None:
            fmts = ", ".join(_fmtoptions.keys())
            raise KeyError(f"fmt unknown, should be one of {fmts}")
        if option:
            self.set_options(option)
            self._sample_format = option

    def write_score(self, stream) -> None:
        self.score.sort(key=_event_start)
        for event in self.score:
            line = " ".join(str(arg) for arg in event)
            stream.write(line)
            stream.write("\n")
            
    def add_instr(self, 
                  instr: U[int, float, str], 
                  instrstr: str) -> None:
        self.instrs[instr] = instrstr

    def add_global(self, code: str) -> None:
        self.globalcodes.append(code)

    def set_options(self, *options: str) -> None:
        for opt in options:
            if opt in _csound_format_options:
                self._sample_format = opt
            self.options.append(opt)

    def dump(self) -> str:
        stream = io.StringIO()
        self.write_csd(stream)
        return stream.getvalue()

    def write_csd(self, stream) -> None:
        """
        Args:
            stream: the stream to write to. Either an open file, a io.StringIO stream or a path

        """
        if isinstance(stream, str):
            outfile = stream
            stream = open(outfile, "w")
        write = stream.write
        write("<CsoundSynthesizer>\n")
        
        if self.options:
            stream.write("\n<CsOptions>\n")

            for option in self.options:
                write(option)
                write("\n")
            write("</CsOptions>\n\n")

        srstr = f"sr     = {self.sr}" if self.sr is not None else ""
        
        txt = f"""
            <CsInstruments>

            {srstr}
            ksmps  = {self.ksmps}
            0dbfs  = 1
            nchnls = {self.nchnls}
            A4     = {self.a4}

            """
        txt = _textwrap.dedent(txt)
        write(txt)
        tab = "  "

        for globalcode in self.globalcodes:
            write(globalcode)
            write("\n")
        
        for instr, instrcode in self.instrs.items():
            write(f"instr {instr}\n")
            body = _textwrap.dedent(instrcode)
            body = _textwrap.indent(body, tab)
            write(body)
            write("endin\n")
        
        write("\n</CsInstruments>\n")
        write("\n<CsScore>\n\n")
        
        self.write_score(stream)
        
        write("\n</CsScore>\n")
        write("</CsoundSynthesizer")

    def run(self,
            output:str=None,
            inputdev:str=None,
            backend: str = None,
            supressdisplay=False,
            piped=False,
            extra: List[str] = None) -> subprocess.Popen:
        """
        Run this csd. 
        
        Args:
            output: the file to use as output. This will be passed
                as the -o argument to csound.
            inputdev: the input device to use
            backend: the backend to use
            supressdisplay: if True, debugging information is supressed
            piped: if True, stdout and stderr are piped through
                the Popen object, accessible through .stdout and .stderr
                streams
            extra: any extra args passed to the csound binary

        Returns:
            the subprocess.Popen object

        """
        if self._sample_format is None:
            ext = os.path.splitext(output)[1]
            if ext in {'.wav', '.aif', '.aiff'}:
                self.set_sample_format('float')
            elif ext == '.flac':
                self.set_sample_format('24bit')
        tmp = tempfile.mktemp(suffix=".csd")
        with open(tmp, "w") as f:
            self.write_csd(f)
        logger.debug(f"Csd.run :: tempfile = {tmp}")
        return run_csd(tmp, output=output, input=inputdev,
                       backend=backend, supressdisplay=supressdisplay,
                       piped=piped, extra=extra)

        
class Timeline:
    
    """
    A soundfile timeline
    """

    def __init__(self, sr=None):
        self.events = []
        self.sndfiles = {}
        self._ftablenum = 1
        self.sr = sr

    def _get_ftable_num(self):
        self._ftablenum += 1
        return self._ftablenum

    def add(self, sndfile, time, gain=1, start=0, end=-1, fadein=0, fadeout=0):
        """
        time: start of playback
        start, end: play a slice of the sndfile
        fadein, fadeout: fade time in seconds
        """
        sndfile = os.path.relpath(sndfile)
        if sndfile in self.sndfiles:
            ftable = self.sndfiles[sndfile]["ftable"]
        else:
            ftable = self._get_ftable_num()
            self.sndfiles[sndfile] = {"ftable": ftable}
        if end < 0:
            info = self._sndinfo(sndfile)
            end = info.duration
        event = {
            'time': time,
            'sndfile': sndfile, 
            'start': start, 
            'end': end, 
            'fadein': fadein, 
            'fadeout': fadeout,
            'ftable': ftable,
            'gain': gain

        }
        self.events.append(event)

    def _sndinfo(self, sndfile):
        if sndfile not in self.sndfiles:
            raise ValueError("No event was added witht hthe given sndfile")
        info = self.sndfiles[sndfile].get("info")
        if info is not None:
            return info
        else:
            import sndfileio
            info = sndfileio.sndinfo(sndfile)
            self.sndfiles[sndfile]["info"] = info
        return info

    def _guess_samplerate(self):
        sr = max(self._sndinfo(sndfile).samplerate for sndfile in self.sndfiles)
        return sr

    def total_duration(self):
        return max(event["time"] + (event["end"] - event["start"]) for event in self.events)

    def write_csd(self, outfile, sr=None, ksmps=64):
        self.events.sort(key=lambda event:event['time'])
        orc = """
<CsInstruments>
sr = {sr}
ksmps = {ksmps}
0dbfs = 1
nchnls = 2

instr 1
    Spath, ioffset, igain, ifadein, ifadeout, ienvpow passign  4
    iatt = ifadein > 0.00001 ? ifadein : 0.00001
    irel = ifadeout > 0.00001 ? ifadeout : 0.00001
    ifilelen = filelen(Spath)
    idur = ifilelen < p3 ? ifilelen : (p3 > 0 ? p3 : ifilelen)
    ipow = ienvpow > 0 ? ienvpow : 1
    ichnls = filenchnls(Spath)
    aenv linseg 0.000000001, iatt, 1, idur - (iatt+irel), 1, irel, 0.000000001
    aenv = pow(aenv, ipow) * igain
    ktime = line(0, idur*2, 2)
    if ( ktime > 1 ) then
        turnoff
    endif
    if (ichnls == 1) then
        a0 diskin2 Spath, 1, ioffset, 0, 0, 4
        a0 = a0 * aenv
        a1 = a0
    else
        a0, a1 diskin2 Spath, 1, ioffset, 0, 0, 4
        a0 = a0 * aenv
        a1 = a1 * aenv
    endif
    outs a0, a1
endin
</CsInstruments>
        """
        if sr is None:
            sr = self._guess_samplerate()
        self.sr = sr
        orc = orc.format(sr=sr, ksmps=ksmps)
        scorelines = ["<CsScore>"]
        for event in self.events:
            line = 'i {instr} {time} {dur} "{path}" {offset} {gain} {fadein} {fadeout} 1.5'.format(
                instr=1, time=event["time"], path=event["sndfile"], 
                dur=event['end'] - event['start'],
                gain=event["gain"], offset=event["start"], fadein=event["fadein"],
                fadeout=event["fadeout"]
                )
            scorelines.append(line)
        scorelines.append("f 0 {totalduration}".format(totalduration=self.total_duration()))
        scorelines.append("</CsScore>")
        
        def writeline(f, line):
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")

        header = "<CsoundSynthesizer>"
        footer = "</CsoundSynthesizer>"

        with open(outfile, "w") as out:
            for line in header.splitlines():
                writeline(out, line)
            for line in orc.splitlines():
                writeline(out, line)
            for line in scorelines:
                writeline(out, line)
            for line in footer.splitlines():
                writeline(out, line)        


def mincer(sndfile, timecurve, pitchcurve, outfile=None, dt=0.002, 
           lock=False, fftsize=2048, ksmps=128, debug=False):
    """
    sndfile: the path to a soundfile
    timecurve: a bpf mapping time to playback time or a scalar indicating a timeratio
               (2 means twice as fast)
               1 to leave unmodified
    pitchcurve: a bpf mapping x=time, y=pitchscale. or a scalar indicating a freqratio
                (2 means an octave higher) 
                1 to leave unmodified

    outfile: the path to a resulting outfile

    Returns: a dictionary with information about the process 

    NB: if the mapped time excedes the bounds of the sndfile,
        silence is generated. For example, a negative time
        or a time exceding the duration of the sndfile

    NB2: the samplerate and number of channels of of the generated file matches 
         that of the input file

    NB3: the resulting file is always a 32-bit float .wav file

    ** Example 1: stretch a soundfile 2x

       timecurve = bpf.linear(0, 0, totaldur*2, totaldur)
       outfile = mincer(sndfile, timecurve, 1)
    """
    import bpf4 as bpf
    import sndfileio
    
    if outfile is None:
        outfile = filetools.addSuffix(sndfile, "-mincer")
    info = sndfileio.sndinfo(sndfile)
    sr = info.samplerate
    nchnls = info.channels
    pitchbpf = bpf.asbpf(pitchcurve)
    
    if isinstance(timecurve, (int, float)):
        t0, t1 = 0, info.duration / timecurve
        timebpf = bpf.linear(0, 0, t1, info.duration)
    elif isinstance(timecurve, bpf.core._BpfInterface):
        t0, t1 = timecurve.bounds()
        timebpf = timecurve
    else:
        raise TypeError("timecurve should be either a scalar or a bpf")
    
    assert isinstance(pitchcurve, (int, float, bpf.core._BpfInterface))
    ts = np.arange(t0, t1+dt, dt)
    fmt = "%.12f"
    _, time_gen23 = tempfile.mkstemp(prefix='time-', suffix='.gen23')
    np.savetxt(time_gen23, timebpf.map(ts), fmt=fmt, header=str(dt), comments="")
    _, pitch_gen23 = tempfile.mkstemp(prefix='pitch-', suffix='.gen23')
    np.savetxt(pitch_gen23, pitchbpf.map(ts), fmt=fmt, header=str(dt), comments="")
    if outfile is None:
        outfile = filetools.addSuffix(sndfile, '-mincer')
    csd = f"""
    <CsoundSynthesizer>
    <CsOptions>
    -o {outfile}
    </CsOptions>
    <CsInstruments>

    sr = {sr}
    ksmps = {ksmps}
    nchnls = {nchnls}
    0dbfs = 1.0

    gi_snd   ftgen 0, 0, 0, -1,  "{sndfile}", 0, 0, 0
    gi_time  ftgen 0, 0, 0, -23, "{time_gen23}"
    gi_pitch ftgen 0, 0, 0, -23, "{pitch_gen23}"

    instr vartimepitch
        idt tab_i 0, gi_time
        ilock = {int(lock)}
        ifftsize = {fftsize}
        ikperiod = ksmps/sr
        isndfiledur = ftlen(gi_snd) / ftsr(gi_snd)
        isndchnls = ftchnls(gi_snd)
        ifade = ikperiod*2
        inumsamps = ftlen(gi_time)
        it1 = (inumsamps-2) * idt           ; account for idt and last value
        kt timeinsts
        aidx    linseg 1, it1, inumsamps-1
        at1     tablei aidx, gi_time, 0, 0, 0
        kpitch  tablei k(aidx), gi_pitch, 0, 0, 0
        kat1 = k(at1)
        kgate = (kat1 >= 0 && kat1 <= isndfiledur) ? 1 : 0
        agate = interp(kgate) 
        aenv linseg 0, ifade, 1, it1 - (ifade*2), 1, ifade, 0
        aenv *= agate
        if isndchnls == 1 then
            a0  mincer at1, 1, kpitch, gi_snd, ilock, ifftsize, 8
            outch 1, a0*aenv
        else
            a0, a1   mincer at1, 1, kpitch, gi_snd, ilock, ifftsize, 8
            outs a0*aenv, a1*aenv
        endif
        
      if (kt >= it1 + ikperiod) then
        event "i", "exit", 0.1, 1
            turnoff     
        endif
    endin

    instr exit
        puts "exiting!", 1
        exitnow
    endin

    </CsInstruments>
    <CsScore>
    i "vartimepitch" 0 -1
    f 0 36000

    </CsScore>
    </CsoundSynthesizer>
    """
    _, csdfile = tempfile.mkstemp(suffix=".csd")
    with open(csdfile, "w") as f:
        f.write(csd)
    subprocess.call(["csound", "-f", csdfile])
    if not debug:
        os.remove(time_gen23)
        os.remove(pitch_gen23)
        os.remove(csdfile)
    return {'outfile': outfile, 'csdstr': csd, 'csd': csdfile}


def _instr_as_orc(instrid, body, initstr, sr, ksmps, nchnls):
    orc = """
sr = {sr}
ksmps = {ksmps}
nchnls = {nchnls}
0dbfs = 1

{initstr}

instr {instrid}
    {body}
endin

    """.format(sr=sr, ksmps=ksmps, instrid=instrid, body=body, nchnls=nchnls, initstr=initstr)
    return orc


def extract_pargs(body: str) -> Set[int]:
    regex = r"\bp\d+"
    pargs = re.findall(regex, body)
    nums = [int(parg[1:]) for parg in pargs]
    for line in body.splitlines():
        if not re.search(r"\bpassign\s+\d+", line):
        # if not re.search(r"\bpassign\b", line):
            continue
        left, right = line.split("passign")
        numleft = len(left.split(","))
        pargstart = int(right) if right else 1
        ps = list(range(pargstart, pargstart + numleft))
        nums.extend(ps)
    return set(nums)


def num_pargs(body:str) -> int:
    """
    analyze body to determine the number of pargs needed for this instrument
    """
    try:
        pargs = extract_pargs(body)
    except ValueError:
        pargs = None
    if not pargs:
        return 0
    pargs = [parg for parg in pargs if parg >= 4]
    if not pargs:
        return 0
    maxparg = max(pargs)
    minparg = min(pargs)
    if maxparg - minparg > len(pargs):
        skippedpargs = [n for n in range(minparg, maxparg + 1) if n not in pargs]
        raise ValueError(f"pargs {skippedpargs} skipped")
    return len(pargs)


def parg_names(body:str) -> Dict[int, str]:
    """
    Analyze body to determine the names (if any) of the pargs used

    iname = p6
    kfoo, ibar passign 4
    """
    argnames = {}

    for line in body.splitlines():
        if re.search(r"\bpassign\s+\d+", line):
            names_str, first_idx = line.split("passign")
            first_idx = int(first_idx)
            names: List[str] = names_str.split(",")
            for i, name in enumerate(names):
                argnames[i + first_idx] = name.strip()
        else:
            words = line.split()
            if len(words) == 3 and words[1] == "=":
                w2 = words[2]
                if w2.startswith("p") and all(ch.isdigit() for ch in w2[1:]):
                    idx = int(w2[1:])
                    argnames[idx] = words[0].strip()
    # remove p1, p2 and p3, if present
    for idx in (1, 2, 3):
        argnames.pop(idx, None)
    return argnames


def num_pargs_match_definition(instrbody: str, args: list) -> bool:
    lenargs = 0 if args is None else len(args)
    numargs = num_pargs(instrbody)
    if numargs != lenargs:
        msg = f"Passed {lenargs} pargs, but instrument expected {numargs}"
        logger.error(msg)
        return False
    return True


def rec_instr(body:str, events:list, init="", outfile="",
              sr=44100, ksmps=64, nchnls=2, a4=442, samplefmt='float',
              dur=None, comment=None, quiet=True
              ) -> Tuple[str, subprocess.Popen]:
    """
    Record one instrument for a given duration

    dur:
        the duration of the recording
    body:
        the body of the instrument
    init:
        the initialization code (ftgens, global vars, etc)
    outfile:
        the generated soundfile, or None to generate a temporary file
    events:
        a list of events, where each event is a list of pargs passed 
        to the instrument, beginning with p2: delay, dur, [p4, p5, ...]
    sr, ksmps, nchnls: ...
    samplefmt: defines the sample format used for outfile, one of (16, 24, 32, 'float')
    """
    if not isinstance(events, list) or not all(isinstance(event, (tuple, list)) for event in events):
        raise ValueError("events is a seq., where each item is a seq. of pargs passed to"
                         "the instrument, beginning with p2: [delay, dur, ...]"
                         f"Got {events} instead")

    csd = Csd(sr=sr, ksmps=ksmps, nchnls=nchnls, a4=a4)
    if not outfile:
        outfile = tempfile.mktemp(suffix='.wav', prefix='csdengine-rec-')

    if init:
        csd.add_global(init)
    
    instrnum = 100
    
    csd.add_instr(instrnum, body)
    for event in events:
        start, dur = event[0], event[1]
        csd.add_event(instrnum, start, dur, event[2:])
    
    if dur is not None:
        csd.set_end_marker(dur)

    fmtoption = {16: '', 24: '-3', 32: '-f', 'float': '-f'}.get(samplefmt)
    if fmtoption is None:
        raise ValueError("samplefmt should be one of 16, 24, 32, or 'float'")
    csd.set_options(fmtoption)

    proc = csd.run(output=outfile)
    return outfile, proc


def normalize_path(path:str) -> str:
    return os.path.abspath(os.path.expanduser(path))

    
def gen_body_static_sines(numsines:int, sineinterp=True, attack=0.05, release=0.1,
                          curve='cos', extend=False):
    """
    Generates the body of an instrument for additive synthesis. In order to be
    used it must be wrapped inside "instr xx" and "endin"

    numsines: the number of sines to generate
    sineinterp: if True, the oscilators use interpolation (oscili)
    extend: extend duration for the release, using linsegr (otherwise a fixed envelope is used)

    It takes the following p-fields:

    chan, gain, freq1, amp1, ..., freqN, ampN

    Where:
        N is numsines
        gain is a gain factor affecting all sines
        ampx represent the relative amplitude of each sine

    Example:

    freqs = [440, 660, 880]
    amps = [0.5, 0.3, 0.2]
    body = bodySines(len(freqs))
    rec_instr(dur=2, outfile="out.wav", body=body, args=[1, 1] + list(flatten(zip(freqs, amps))))
    """
    lines = [
        "idur = p3",
        "ichan = p4",
        "igain = p5",
        "aout = 0",
        "itot = 0"
    ]
    sinopcode = "oscili" if sineinterp else "oscil"
    _ = lines.append
    for i in range(numsines):
        _(f"ifreq_{i} = p{i*2+6}")
        _(f"iamp_{i}  = p{i*2+7}")
        _(f"iamp_{i} *= ifreq_{i} > 20 ? 1 : 0")
        _(f"itot += ifreq_{i} > 20 ? 1 : 0")
    for i in range(numsines-1):
        _(f"aout += {sinopcode}:a(iamp_{i}, ifreq_{i})")
        _(f"if itot == {i+1} goto exit")
    _(f"aout += {sinopcode}:a(iamp_{numsines-1}, ifreq_{numsines-1})")
    _("exit:")
    if curve == 'linear':
        if extend:
            env = f"linsegr:a(0, {attack}, igain, {release}, 0)"
        else:
            env = f"linseg:a(0, {attack}, igain, idur-{attack+release}, igain, {release}, 0)"
    elif curve == 'cos':    
        if extend:
            env = f"cossegr:a(0, {attack}, igain, {release}, 0)"
        else:
            env = f"cosseg:a(0, {attack}, igain, idur-{attack+release}, igain, {release}, 0)"
    else:
        raise ValueError(f"curve should be one of 'linear', 'cos', got {curve}")
    _( f"aout *= {env}" )
    _("outch ichan, aout")
    body = "\n".join(lines)
    return body


def _ftsave_read_text(path):
    # a file can have multiple tables saved
    lines = iter(open(path))
    tables = []
    while True:
        tablength = -1
        try:
            headerstart = next(lines)
            if not headerstart.startswith("===="):
                raise IOError(f"Expecting header start, got {headerstart}")
        except StopIteration:
            # no more tables
            break
        # Read header
        for line in lines:
            if line.startswith("flen:"):
                tablength = int(line[5:])
            if 'END OF HEADER' in line:
                break
        if tablength < 0:
            raise IOError("Could not read table length")
        values = np.zeros((tablength+1,), dtype=float)
        # Read data
        for i, line in enumerate(lines):
            if line.startswith("---"):
                break
            values[i] = float(line)
        tables.append(values)
    return tables
 

def ftsave_read(path, mode="text"):
    """
    Read a file saved by ftsave, returns a list of tables
    """
    if mode == "text":
        return _ftsave_read_text(path)
    else:
        raise ValueError("mode not supported")


def get_used_output_channels(instrbody:str) -> Tuple[Set[int], Set[str]]:
    """
    Given the body of an instrument, scan the code
    for output opcodes (outch) to see which output channels
    are used

    Args:
        instrbody: the body of a csound instrument (between instr/endin)

    Returns:
        Two set of channels, one with ints for all cases where the channel
        is a constant, and a second set with variables used as channels,
        for cases like "outch kchn, asignal"

    """
    outregex = re.compile(r"outch\ |outs\ |out\ ")
    outchlines = [line.strip() for line in instrbody.splitlines()
                  if outregex.search(line)]
    outchans = set()
    variables = set()
    for outchline in outchlines:
        if outchline.startswith("outch "):
            opcode, rest = outchline.split("outch")
            words = rest.split(",")
            chns = [w.strip() for w in words[::2]]
            for chn in chns:
                if not chn.isdecimal():
                    variables.add(chn)
                else:
                    outchans.add(int(chn))
        elif outchline.startswith("outs "):
            outchans.add(1)
            outchans.add(2)
        elif outchline.startswith("out "):
            outchans.add(1)
    return outchans, variables
