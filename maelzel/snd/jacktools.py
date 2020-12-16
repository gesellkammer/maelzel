# # depende de JACK-client: https://pypi.python.org/pypi/JACK-Client
import jack as _jack     
from emlib.iterlib import flatten


_jackclient = None


def get_client():
    global _jackclient
    if _jackclient is None:
        _jackclient = _jack.Client("maelzel.jacktools")
    return _jackclient


def clients_connected_to(name_pattern):
    """
    Returns a list of clients connected to the ports matched by name_pattern

    Example
    =======

    clientA:out_1   ----->   system:playback1
    clientB:out_1   ----->   system:playback1
    clientB:out_2   ----->   system:playback2 
    clientC:out_1   ----->   clinetA:in_1

    clients_connected_to("system:playback") --> {"clientA", "clientB"}
    """
    myjack = get_client()
    connectedto_ports = myjack.get_ports(name_pattern, is_audio=True, is_input=True)
    connections = flatten(myjack.get_all_connections(p) for p in connectedto_ports)
    clients = set(conn.name.split(":")[0] for conn in connections)
    return clients


def _disconnect_port(port:_jack.Port):
    myjack = get_client()
    connections = myjack.get_all_connections(port)
    for conn in connections:
        try:
            myjack.disconnect(port, conn)
        except _jack.JackError as e:
            print(f"Failed to disconnect {port} from {conn}")
            print(e)


def connect_client(source:str, dest:str, disconnect=False):
    """
    Connect all ports of source to matching ports of dest

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
        except _jack.JackError as e:
            print(e)