# depends on JACK-client: https://pypi.python.org/pypi/JACK-Client
import jack
from emlib.iterlib import flatten
import sys
import subprocess
import shutil
import cachetools
from dataclasses import dataclass
from typing import Optional

_jackclient = None


class PlatformNotSupportedError(Exception):
    pass


@cachetools.cached(cache=cachetools.TTLCache(1, 60))
def jack_running() -> bool:
    """
    Returns True if jack is running.

    .. note::
        The result is cached for a certain amount of time. Use `jack_running_check`
        for an uncached version
    """
    return jack_running_check()


def jack_running_check() -> bool:
    """
    Returns True if jack is running.
    """
    if sys.platform == "linux":
        jack_control = shutil.which("jack_control")
        if jack_control:
            proc = subprocess.Popen([jack_control, "status"],
                                    stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            if proc.wait() == 0:
                return True
    try:
        _ = jack.Client("checkjack", no_start_server=True)
    except jack.JackOpenError:
        return False
    return True


def jackd_running() -> bool:
    """ Returns True if jackd is present """
    import psutil
    return any(proc.name == 'jackd' for proc in psutil.process_iter())


def get_client():
    global _jackclient
    if _jackclient is None:
        _jackclient = jack.Client("maelzel.jacktools")
    return _jackclient


@dataclass
class JackInfo:
    running: bool
    samplerate: int
    blocksize: int


def get_sr() -> int:
    if not jack_running():
        return 0
    c = jack.Client("maelzel.jacktools")
    return c.samplerate

def get_info() -> Optional[JackInfo]:
    if not jack_running():
        return None
    c = jack.Client("maelzel.jacktools")

    return JackInfo(running=True,
                    samplerate=c.samplerate,
                    blocksize=c.blocksize)


def clients_connected_to(name_pattern):
    """
    Returns a list of clients connected to the ports matched by name_pattern

    Example
    =======

    .. code::

        clientA:out_1   ----->   system:playback1
        clientB:out_1   ----->   system:playback1
        clientB:out_2   ----->   system:playback2
        clientC:out_1   ----->   clinetA:in_1

    >>> clients_connected_to("system:playback")
    {"clientA", "clientB"}
    """
    myjack = get_client()
    connectedto_ports = myjack.get_ports(name_pattern, is_audio=True, is_input=True)
    connections = flatten(myjack.get_all_connections(p) for p in connectedto_ports)
    clients = set(conn.name.split(":")[0] for conn in connections)
    return clients


def _disconnect_port(port:jack.Port):
    myjack = get_client()
    connections = myjack.get_all_connections(port)
    for conn in connections:
        try:
            myjack.disconnect(port, conn)
        except jack.JackError as e:
            print(f"Failed to disconnect {port} from {conn}")
            print(e)


def connect_client(source: str, dest: str, disconnect=False) -> None:
    """
    Connect all ports of source to matching ports of dest

    Args:
        source: the name of the source client
        dest: the name of the destination client
        disconnect: if True, disconnect source from all other destinations before
            connecting to dest
    """
    myjack = get_client()
    sourceports = myjack.get_ports(source, is_audio=True, is_output=True)
    destports = myjack.get_ports(dest, is_audio=True, is_input=True)
    if disconnect:
        for port in sourceports:
            _disconnect_port(port)

    for sourceport, destport in zip(sourceports, destports):
        try:
            myjack.connect(sourceport, destport)
        except jack.JackError as e:
            print(e)
