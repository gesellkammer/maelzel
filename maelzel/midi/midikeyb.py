from collections import namedtuple as _namedtuple
import atexit as _atexit
import fnmatch as _fnmatch
from numbers import Number as _Number
import numpy as _np
import rtmidi2 as rtmidi

_INPORT_GLOBAL = rtmidi.MidiIn(clientname='tmp', queuesize=1000)
_OUTPORT_GLOBAL = rtmidi.MidiOut()

"""
Reference: these values should not be used as globals in a tight loop since there is
a global lookup involved. unless python has a way of inlining. use the value directly
"""

CC = 176
NOTEON = 144
NOTEOFF = 128


def match_port(portname: str):
    """
    return the index of the port matching the given port-name

    Example
    -------

    >>> get_ports()
    ['IAC Driver IAC Bus 1', 'Caps Lock Keyboard']
    >>> match_port('IAC*')
    0
    """
    if not isinstance(portname, str):
        raise TypeError("portname should be a string defining the name or the beginning of the name of a midi port")
    inports = get_ports()
    return _matchfirst(portname, inports)


def _matchfirst(pattern, names):
    matching = [i for i, name in enumerate(names) if _matchstr(name, pattern)]
    if not matching:
        return None
    if len(matching) > 1:
        print("Warning: more than one port matched, using the first match")
    return matching[0]


def _matchall(pattern, names):
    return list(set([i for i, name in enumerate(names) if _matchstr(name, pattern)]))


def match_outport(portname):
    """
    return the index of the port matching the given name, or None if no match

    Example
    =======

    >>> get_outports()
    ['IAC Driver IAC Bus 1', 'BCF2000', 'ipMIDI Port 1']
    >>> match_outport('bcf*')
    'BCF2000'
    """
    outports = get_outports()
    return _matchfirst(portname, outports)


def get_ports():
    return _INPORT_GLOBAL.ports


def get_outports():
    return _OUTPORT_GLOBAL.ports


def _matchstr(s, pattern):
    return _fnmatch.fnmatch(s.lower(), pattern.lower())


class Merged:

    def __init__(self, inports=None, outport='MERGE'):
        """
        merge the inports as one port

        inports: a seq. of strings to match against the inports, or None to match all
        """
        if inports is None:
            inports = get_ports()
        available_inports = [p for p in get_ports() if p != outport]
        indexes = []
        for inport in inports:
            indexes.extend(_matchall(inport, available_inports))
        indexes = sorted(list(set(indexes)))
        self.inport_indexes = indexes
        self.outport_name = outport
        self.inports = []
        self.outport = None
        self.start()
        _atexit.register(self.stop)

    @property
    def inport_names(self):
        names = get_ports()
        return [names[i] for i in self.inport_indexes]

    def start(self):
        self.inports = []
        for index in self.inport_indexes:
            port = rtmidi.MidiIn()
            port.open_port(index)
            port.callback = self.callback
            self.inports.append(port)
        self.outport = rtmidi.MidiOut()
        if self.outport_name:
            self.outport.open_virtual_port(self.outport_name)
        else:
            # check if we are doing a loop
            inportnames = get_ports()
            if get_outports()[0] in inportnames:
                self.outport_name = 'MERGE'
                self.outport.open_virtual_port(self.outport_name)
            else:  # open the default port
                self.outport.open_port()

    def callback(self, msg, time):
        self.outport.send_message(msg)

    def stop(self):
        for port in self.inports:
            port.close_port()
        self.outport.close_port()


class MidiInvert:

    def __init__(self, port):
        self.notes = _np.zeros((127,), dtype=int)
        self.mask = _np.zeros((127,), dtype=int)
        self.mask[60:90] = 1
        self.midiout = rtmidi.MidiOut()
        self.midiout.open_port()
        self.midiin = rtmidi.MidiIn('keyb', 1000)
        if isinstance(port, str):
            port = match_port(port)

        self.midiin.open_port(port)
        self.midiin.callback = self.callback

    def disconnect(self):
        self.midiin.callback = None

    def noteon(self, note, vel):
        notes = self.notes
        notes[note] = 1
        print(self.notes)
        outnotes = (1 - notes) * self.mask
        outnotes = [i for i, note in enumerate(outnotes) if note]
        print("----------")
        print(outnotes)
        vels = [vel] * len(outnotes)
        chs = [0] * len(outnotes)
        self.midiout.send_noteon_many(outnotes, vels, chs)
        self.midiout.send_message([0x80, note, 0])

    def noteoff(self, note, vel):
        print("OFF", note)
        self.notes[note] = 0
        self.midiout.send_messages2(0x80, 0, list(range(60, 90)), [0] * 30)
        # for i in range(60, 90):
        #    self.midiout.send_message([0x80, i, 0])

    def callback(self, msg, time):
        msgtype = msg[0] & 0xF0
        if msgtype == 144:
            vel = msg[2]
            if vel > 0:
                return self.noteon(msg[1], msg[2])
            else:
                return self.noteoff(msg[1], vel)


def _default_outport():
    return get_outports()[0]


def _all_notes_off_bruteforce(midiout, ch):
    for n in range(127):
        midiout.send_noteoff(ch, n)


def _all_notes_off_cc(midiout, ch):
    midiout.send_cc(ch, 127)

all_notes_off = _all_notes_off_bruteforce


class TwoManuals(object):
    def __init__(self, inport, split=62, transpose_lower=12, transpose_upper=0,
                 client_name="keyb", port_name="TwoManuals"):
        self.midiin = rtmidi.MidiIn(client_name, 1000)
        self.split = split
        self.transpose_lower = transpose_lower
        self.transpose_upper = transpose_upper
        if isinstance(inport, str):
            try:
                inport = next(i for i, p in enumerate(self.midiin.ports) if p.startswith(inport))
            except StopIteration:
                raise ValueError("could not find a midi port with the given name")
        else:
            raise ValueError("The inport should be a string indicating the port name to listen to")
        self.inport = inport
        self.port_name = port_name
        self.notes = [0]*127
        self.midiout: rtmidi.MidiOut = None


        self.start()

    def start(self):
        try:
            self.midiout = rtmidi.MidiOut()
            self.midiout.open_virtual_port(self.port_name)
            self.midiin.open_port(self.inport)
            self.midiin.callback = self.callback
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            self.stop()

    def stop(self):
        self.midiin.callback = None
        self.midiout.close_port()
        self.midiin.close_port()

    def noteon(self, note, vel):
        if note < self.split:
            note += self.transpose_lower
        else:
            note += self.transpose_upper  
        self.midiout.send_noteon(note, vel)
        self.notes[note] += 1

    def noteoff(self, note, vel):
        if note < self.split:
            note += self.transpose_lower
        else:
            note += self.transpose_upper
        n = self.notes[note]
        if n == 1:
            self.midiout.send_noteoff(note, vel)
            self.notes[note] = 0
        elif n > 1:
            self.notes[note] -= 1

    def callback(self, msg, time):
        msgtype = msg[0] & 0xF0
        if msgtype == 144:
            vel = msg[2]
            if vel > 0:
                self.noteon(msg[1], vel)
            else:
                self.noteoff(msg[1], vel)
        elif msgtype == 0x80:
            self.noteoff(msg[1], msg[2])


class _Message(object): pass


class NOTEON(_namedtuple("NOTEON", "ch note vel"), _Message):

    def send(self, keyb):
        keyb.midiout.send_noteon(self.ch, self.note, self.vel)
        keyb.notesdown[self.ch * 128 + self.note] = 1

    @staticmethod
    def send_tuple(t, keyb):
        ch, note, vel = t
        keyb.midiout.send_noteon(note, vel, ch)
        keyb.notesdown[t[0] * 128 + t[1]] = 1


class NOTEOFF(_namedtuple("NOTEOFF", "ch note vel"), _Message):

    def send(self, keyb):
        keyb.midiout.send_noteoff(self.note, self.ch)
        keyb.notesdown[self.ch * 128 + self.note] = 0

    @staticmethod
    def send_tuple(t, keyb):
        keyb.midiout.send_noteoff(t[0], t[1])
        keyb.notesdown[t[0] * 128 + t[1]] = 0


class CC(_namedtuple("CC", "ch cc value"), _Message):

    def send(self, keyb):
        keyb.midiout.send_cc(self.cc, self.value, self.ch)

    @staticmethod
    def send_tuple(t, keyb):
        ch, cc, val = t
        keyb.midiout.send_cc(cc, val, ch)


class DummyMidiOut(object):

    def __init__(self, *args, **kws): pass

    def open_virtual_port(self, port): pass

    def close_port(self): pass

    def send_noteoff(self, *args): pass

    def send_noteon(self, *args): pass

    def send_tuple(self, *args): pass

    def send_cc(self, *args): pass

    def send_message(self, *args): pass

    def send_noteon_many(self, *args): pass


class KeyboardPipe(object):

    def __init__(self, inport, outport='KeyboardPipe'):
        """
        Take the input from inport, transform it and output it to outport.

        Outport must be a string and defines the name of the virtual port
        created. 

        Examples
        ========

        >>> keyb = KeyboardPipe("KORG").start()
        >>> keyb.register_noteon(lambda ch, note, vel, k: (ch, note+1, vel))  # transpose one semitone up

        # Use note 36 as a channel switch, channel 0 if not pressed, channel 5 if pressed
        >>> keyb.register_noteon(lambda keyb, ch, note, vel: (keyb.notesdown[36]*5, note, vel))
        >>> keyb.register_noteon((lambda keyb, ch, note, val: None), ch='ALL', note=36)
        """
        self.inport = match_port(inport)
        self.midiin = rtmidi.MidiIn()
        self.midiout = None
        self.outport = outport
        self.noteon_funcs = [None for i in range(128 * 16)]
        self.noteon_funcs_all = [None for i in range(16)]
        self.noteoff_funcs = [None for i in range(128 * 16)]
        self.cc_funcs = [None for i in range(128 * 16)]
        self.notesdown = [0 for i in range(128 * 16)]
        self.callback_noteoff = self.callback_noteoff_route

    def start(self):
        try:   
            if self.outport:   
                self.midiout = rtmidi.MidiOut()
                self.midiout.open_virtual_port(self.outport)
            else:
                self.midiout = DummyMidiOut()
            self.midiin.open_port(self.inport)
            self.midiin.callback = self.callback
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            self.stop()
        return self

    def stop(self):
        self.midiin.callback = None
        self.midiout.close_port()
        self.midiin.close_port()

    def _call_and_send(self, funcs, ch, note, vel, default):
        messages = []
        for func in funcs:
            out = func(self, ch, note, vel)
            if out:
                if isinstance(out, list):
                    messages.extend(out)
                else:
                    messages.append(out)
        for message in messages:
            if isinstance(message, _Message):
                message.send(self)
            else:
                default.send_tuple(message, self)

    def callback_noteoff_mirror(self, ch, note, vel):
        self.midiout.send_noteoff(note, vel, ch)

    def callback_noteoff_route(self, ch, note, vel):
        funcs = self.noteoff_funcs[ch * 128 + note]
        if funcs:
            self._call_and_send(funcs, ch, note, vel, NOTEOFF)

    def callback(self, msg, time):
        msg0 = msg[0]
        msgtype = msg0 & 0xF0
        ch = msg0&0x0F
        if msgtype == 144:
            vel = msg[2]
            note = msg[1]
            if vel == 0:
                return self.callback_noteoff(ch, note, 0)
            # NOTEON
            index = ch * 128 + note
            funcs = self.noteon_funcs[index]
            if funcs:
                self._call_and_send(funcs, ch, note, vel, NOTEON)
        elif msgtype == 176:
            cc = msg[1]
            index = ch * 128 + cc
            funcs = self.cc_funcs[index]
            if funcs:
                self._call_and_send(funcs, ch, cc, msg[2], CC)                
        elif msgtype == 128:
            return self.callback_noteoff(ch, msg[1], msg[2])

    def register_note_transform(self, func, ch='ALL', note='ALL'):
        """
        register a transformation for both noteons and noteoffs
        """
        self.register_noteon(func, ch, note)
        self.register_noteoff(func, ch, note)

    def register_noteon(self, func, ch='ALL', note='ALL'):
        """
        ch can be a number, a tuple or 'ALL'
        note can be a number, a tuple or 'ALL'

        func should be:

        def func(keyb, ch, note, vel):
            ...
            return (ch2, note2, vel2)

        where keyb is this instance of KeyboardPipe (this is necessary if 
        a transformation makes use of its context. watch out for hanging notes)

        a transformation can also return a list of (ch2, note2, vel2), in which case
        they are all sent (this is useful when you want to multiplex messages, 
        for instance, send notes received in one channel to all the channels

        it can also return None, if the action is only there for its side-effects

        it can also return a specific kind of action:
        NOTEON
        NOTEOFF
        CC
        ...

        NB: one note should always be a tuple, a list of notes is a list of tuples
        dont return a list as a note [ch2, note2, vel2] and dont return a tuple of tuples for
        multiple notes
        """
        # TODO: change order of arguments to more specific -> less specific (vel, note, ch, keyb)
        self._register(func, ch, note, funcs_register=self.noteon_funcs)

    def register_noteoff(self, func, ch='ALL', note='ALL'):
        """
        see register_noteon
        """
        self._register(func, ch, note, funcs_register=self.noteoff_funcs)

    def _register(self, func, ch, note, funcs_register):
        if ch == 'ALL':
            ch = list(range(16))
        elif isinstance(ch, _Number):
            ch = [ch]
        if note == 'ALL':
            note = list(range(128))
        elif isinstance(note, _Number):
            note = [note]
        for individual_ch in ch:
            for individual_note in note:
                index = individual_ch * 128 + individual_note
                funcs = funcs_register[index]
                if funcs:
                    funcs.append(func)
                else:
                    funcs = [func]
                funcs_register[index] = funcs

    def noteoff_bypass(self, value=True):
        """
        if True, set this keyboard to mirror the noteoffs
        All noteoffs will be sent through the pipe bypassing
        any transformation registered.
        Set it to false to return to the default behaviour
        """
        self.callback_noteoff = self.callback_noteoff_mirror if value else self.callback_noteoff_route
