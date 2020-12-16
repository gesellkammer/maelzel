from fractions import Fraction as F
from numbers import Number
import os
import warnings as _warnings
from collections import namedtuple
from bpf4 import bpf
from .misc import normalize_frames
from emlib.lib import returns_tuple as _returns_tuple, public, isiterable
from emlib.pitchtools import n2m as _n2m


class Measure(object):
    def __init__(self, num, den, comment=""):
        """
        num: numerator of the time signature
        den: denominator of the time signature
        """
        assert isinstance(den, int) and isinstance(num, int)
        self.num = num
        self.den = den
        self.numbeats = 4 * F(self.num, self.den)
        self.comment = comment

    @property
    def timesig(self):
        return "{num}/{den}".format(num=self.num, den=self.den)

    def __repr__(self):
        comment = " <%s>" % self.comment if self.comment else ""
        return "%d/%d%s" % (self.num, self.den, comment)


class ParseError(BaseException):
    pass

        
def tempos_to_tempocurve(measures, tempos):
    beats = []
    now_beat = 0
    for measure in measures:
        beats.append(now_beat)
        now_beat += measure.numbeats
    return bpf.core.NoInterpol(beats, tempos)


@public
def score_read(scorefile):
    """
    read a text score. For the syntax of a text score, see score_syntax

    See also: Score.fromstring, Score.fromfile
    """
    if isinstance(scorefile, basestring):
        if os.path.exists(scorefile):
            return Score.fromfile(scorefile)


@public
def score_syntax():
    """
    display the syntax for defining a text score
    """
    print("""
Each measure is defined as a line

[TimeSignature] [Tempo] [->]

TimeSignature: 4/4, 3/8, 5/16, etc. (without spaces!)
Tempo        : 4=60, 8=120, 16=54, etc. (without spaces!)
               4 -> quarter note, 8 -> 1/8th note, etc.
->           : means a continuous modification of tempo, until
               a new tempo is found

If a new measure is the same as a previous measure, a "." can 
be used to repeat the previous measure

Example
=======

4/4 4=60     # this is a comment attached to the measure
5/4          # tempo remains the same
.            # another 5/4 bar
3/4 8=100
4/4 4=60 ->  # an accelerando over two bars
.
2/4 100
    """)

_testscore = """
3/4 72
4/8 120
4/4
3/8 90
7/4
4/4
.
"""


class Index:
    def __init__(self, times, beats, measure_indexes):
        self.time2measure = bpf.core.NoInterpol(times, measure_indexes)
        self.beat2measure = bpf.core.NoInterpol(beats, measure_indexes)
        self.measure2time = bpf.core.NoInterpol(measure_indexes, times)
        self.measure2beat = bpf.core.NoInterpol(measure_indexes, beats)
        self.beat2time = bpf.core.Linear(beats, times)
        self.time2beat = bpf.core.Linear(times, beats)
                
@public
class Score(object):
    def __init__(self, measures, tempocurve=None):
        """
        measures   : a list of measures
        tempocurve : a tempo, a list of tempos (one for each measure), 
                     or a bpf defining the tempo (x: beat, y: tempo)
        
        NB the normal way to create a score is to write a txt file defining
        the score and then call Score.fromfile(pathtoscore)

        See Also
        ========

        score_syntax, read_score
        """
        if isinstance(tempocurve, (list, tuple)):
            tempocurve = tempos_to_tempocurve(measures, tempocurve)
        elif isinstance(tempocurve, Number):
            tempocurve = bpf.asbpf(tempocurve)
        self.measures = measures
        self.tempocurve = tempocurve if tempocurve is not None else bpf.asbpf(60)
        self.index = self._build_indexes()
        self._tempos = None

    @property
    def tempos(self):
        if self._tempos is not None:
            return self._tempos
        out = []
        for i in range(len(self.measures)):
            beat = self.index.measure2beat(i)
            tempo = self.tempocurve(beat)
            out.append(tempo)
        self._tempos = out
        return out
    
    def _build_indexes(self):
        now_time = 0
        now_beats = 0
        EPSILON = 1e-12
        times = []
        beats = []
        for measure in self.measures:
            times.append(now_time)
            beats.append(now_beats)
            temponow = self.tempocurve(now_beats)
            if temponow - int(temponow) < EPSILON:
                beatdur = F(60, int(temponow))
            else:
                beatdur = F.from_float(60 / temponow)
            measure_duration = measure.numbeats * beatdur
            now_time += measure_duration
            now_beats += measure.numbeats
        measure_indexes = range(len(self.measures))
        return Index(times, beats, measure_indexes)
        
    @classmethod
    def fromfile(cls, pathtofile):
        f = open(pathtofile)
        measures, tempocurve = parse_lines(f)
        return cls(measures, tempocurve)

    @classmethod
    def fromstring(cls, s):
        lines = s.splitlines()
        return cls.fromlines(lines)

    @classmethod
    def fromlines(cls, lines):
        measures, tempocurve = parse_lines(lines)
        return cls(measures, tempocurve)

    def write_midi(self, midifile, resolution=1920, fillpitch=60):
        """
        Discards any tempo transition, assumes the tempo at the beginning of the measure
        stays the same for the entire measure

        midifile: path of the midi file
        resolution: the ticks per second
        fillpitch: the pitch to be used to fill each measure
        """
        import midi
        t = midi.Track()
        t.tick_relative = False
        sec2tick = lambda sec: int(sec * resolution)
        tempo = -1
        now = 0
        for i, measure in enumerate(self.measures):
            beat = self.index.measure2beat(i)
            assert (now - beat) < 0.0001, \
                "now: {}, beat: {}, timesig: {}, prev: {}".format(
                    now, beat, measure.timesig, self.measures[i - 1])
            temponow = int(self.tempocurve(beat))
            durbeats = measure.numbeats
            t.append(midi.TimeSignatureEvent(tick=sec2tick(now), 
                                             numerator=measure.num,
                                             denominator=measure.den))
            if temponow != tempo:
                tempo = temponow
                t.append(midi.SetTempoEvent(tick=sec2tick(now), bpm=tempo))
            t.append(midi.NoteOnEvent(tick=sec2tick(now),
                                      pitch=fillpitch, velocity=1))
            t.append(midi.NoteOffEvent(tick=sec2tick(now + durbeats), 
                                       pitch=fillpitch, velocity=0))
            now += durbeats
        t.append(midi.EndOfTrackEvent(tick=sec2tick(now) + 1))
        midipattern = midi.Pattern([t], resolution=resolution)
        midipattern.tick_relative = False
        midipattern.make_ticks_rel()
        midi.write_midifile(midifile, midipattern)
        return midipattern

    def write_musicxml(self, path):
        """
        Write the score as musicxml
        """
        import music21 as m21
        s = m21.stream.Stream()
        lasttempo = -1
        for i, (measure, tempo) in enumerate(zip(self.measures, self.tempos)):
            if tempo != lasttempo:
                lasttempo = tempo
                s.append(m21.tempo.MetronomeMark(number=tempo))
            s.append(m21.meter.TimeSignature(measure.timesig))
            s.append(m21.note.Note(pitch=60, duration=m21.duration.Duration(measure.numbeats)))
        s.write("xml", path)
        return s

    def measures_fixed_tempo(self):
        """
        Returns an iterator of (time, timesig, tempo), representing
        each measure in the timeline. It is assumed that each
        measure has a tempo which does not change during the measure
        (no accel or rits)
        """
        now = 0
        for i, measure in enumerate(self.measures):
            beat = self.index.measure2beat(i)
            temponow = self.tempocurve(beat)
            measuredur = measure.numbeats * (60.0 / temponow)
            yield _Measure(time=float(now), timesig=measure.timesig, tempo=temponow)
            now += measuredur

    def dump(self):
        tempo = -1
        now = 0
        for i, measure in enumerate(self.measures):
            beat = self.index.measure2beat(i)
            temponow = self.tempocurve(beat)
            if temponow != tempo:
                tempo = temponow
                tempostr = str(tempo)
            else:
                tempostr = ""
            print("{i} {now} {timesig} {tempo}".format(
                i=i, now=float(now), timesig=measure.timesig, tempo=tempostr))
            measuredur = measure.numbeats * (60.0 / tempo)
            now += measuredur


_Measure = namedtuple("Measure", "time timesig tempo")         


def parse_tempo(s):
    """
    return always a quarter tempo
    """
    if "=" not in s:
        return int(s)
    den, tempo = s.split("=")
    if den == "2":
        return int(tempo) * 2
    elif den == "4":
        return int(tempo)
    elif den == "8":
        return int(tempo) / 2
    elif den == "16":
        return int(tempo) / 4
    else:
        raise ParseError("tempo definition not understood")
    


@_returns_tuple("measures tempocurve")
def parse_lines(lines):
    """
    take an iterator of lines (a file, for examples) and return a tuple (measures, tempocurve)
    
    Example
    =======
    
    score.txt
    ^^^^^^^^^
    
    4/4 72
    3/8 8=120
    4/8 8=120 ->
    4/4 60
        3/4     116
    .       # a point repeats the last measure
    .
    .
    
    format of the txt file:
    
    - each measure in its line
    
    - den/nom   [tempo]
    
    - tempo is optional. if given:
        - one single number: 4/4    120    # in this case, the tempo is 
                                             constant during the measure, 
                                             the value of the quarter note is 120
        - not given, the last tempo is used
        - specify the reference value: 4/4  4=120   # no spaces in the tempo definition.
                                                    4=quarter note, 8=eigth, etc.
        - two tempos indicate a tempo transformation: 4/4 120 -> 160  
        - one tempo with a "->" at the end indicates a transformation to next tempo: 4/4 60 ->
    """
    last_tempo = 0
    default_tempo = 60
    last_measure = None
    now_beat = 0
    measures, tempos, beats, interpolations = [], [], [], []
    transformation_open = False
    
    for line in lines:
        if "#" in line:
            line, comment = line.split("#")
            comment = comment.strip()
        else:
            comment = ""
        words = line.split()
        L = len(words)
        if L == 0:
            _warnings.warn("Found empty line while parsing, skipping")
            continue
        if words[0] == ".":
            num, den = last_measure.num, last_measure.den
        else:
            if "=" in words[0]:
                numden, tempo = words[0].split("=")
                words.insert(1, tempo)
                words[0] = numden
                _warnings.warn("Syntax deprecated. The line should read %s %s" % (numden, tempo))
            num, den = map(int, words[0].split("/"))
        skip = False
        interpolation = 'nointerpol'
        if L == 1:
            if last_tempo == 0:
                # first line
                tempo0 = tempo1 = default_tempo
            else:
                skip = True
                tempo0 = tempo1 = last_tempo
        else:
            if transformation_open:
                transformation_open = False
                interpolation = 'linear'
            else:
                interpolation = 'nointerpol'
            if L == 2:
                tempo0 = tempo1 = parse_tempo(words[1])
            elif L == 3:
                tempo0 = tempo1 = parse_tempo(words[1])
                transformation_open = True
                assert words[2] == "->"
            elif L == 4:
                tempo0, tempo1 = parse_tempo(words[1]), parse_tempo(words[3])
                assert words[2] == "->"
            else:
                raise ParseError("two many elements in measure definition") 
        measure = Measure(num, den, comment=comment)
        if not skip:
            beats.append(now_beat)
            tempos.append(tempo0)
            interpolations.append(interpolation)
            if tempo1 != tempo0:
                beats.append(now_beat + measure.numbeats)
                tempos.append(tempo1)
                interpolations.append('linear')
        measures.append(measure)
        now_beat += measure.numbeats
        last_tempo = tempo1
        last_measure = measure
    interpolations.pop(0)   # discard the first interpolation
    beats_f = [float(b) for b in beats]
    tempos_f = [float(t) for t in tempos]
    tempocurve = bpf.core.Multi(beats_f, tempos_f, interpolations)
    return measures, tempocurve


def asmidi(n):
    """
    return always a numeric midi note

    number (int or midi) --> assumed to be a midinote
    string               --> assumed to be a note, converts to midinote

    Example
    =======

    >>> asmidi(60)
    60
    >>> asmidi("C4")
    60  
    """
    if isinstance(n, Number):
        return n
    elif isinstance(n, basestring):
        return _n2m(n)
    try:
        return float(n)
    except TypeError:
        raise TypeError("could not convert %s to a midi note" % str(n))

    
# #------------------------------------------------------------
# #
# #    MIDI IO    
# #
# #------------------------------------------------------------


def events_to_midi(outfile, starts, durs, pitches, velocities=90, 
                   channel=0, track=0, tempo=60, remove_overlap=True):
    """
    starts    : the start of each note
    durs      : the duration of each note in quarters 
    pitches   : the midi note of each note (or a string representation)
    velocites : a number or a seq. of numbers. The velocity of each note
    channel   : a number or a seq. of numbers. The channel of each event
    track     : a number or a seq. of numbers. The track number of each event
    tempo     : the tempo of the MidiFile
    removeOverlap: reduce the duration of any event which would extend
                    over the next event on the same track. 
                    If False, the next event will be shifted to not collapse
                    with the previous event
    """
    assert isinstance(outfile, str)
    assert isiterable(starts)
    assert isinstance(durs, (int, float)) or isiterable(durs)
    assert isinstance(pitches, (int, float, str)) or isiterable(pitches)
    assert isinstance(velocities, (int, float)) or isiterable(velocities)
    assert isinstance(channel, int) or isiterable(channel)
    assert isinstance(track, int) or isiterable(track)
    assert isinstance(tempo, (int, float))
    return _events_to_midi_multitrack(
        starts, durs, pitches, velocities, 
        tracks=0, channels=channel, tempo=tempo, outfile=outfile)


_Event = namedtuple("_Event", "start dur pitch vel chan")


def _events_to_midi_multitrack(starts, durs, pitches, velocities=90, 
                               tracks=0, channels=0, tempo=60, 
                               resolution=1920, outfile="events.mid"):
    import midi
    N = len(starts)

    as_seq = lambda obj: [obj] * N if isinstance(obj, Number) else obj
    time2tick = lambda time: int(time * resolution)
    
    pitches, durs, velocities, tracks, channels = map(as_seq, 
        [pitches, durs, velocities, tracks, channels])
    numtracks = max(tracks) + 1
    miditracks = []
    pitches = list(map(asmidi, pitches))
    
    for tracknum in range(numtracks):
        t = midi.Track()
        t.make_ticks_abs()
        t.append(midi.SetTempoEvent(tick=0, bpm=tempo))
        miditracks.append(t)
    
    notespertrack = [[] for _ in miditracks]
    for i in range(len(starts)):
        track = tracks[i]
        ev = _Event(starts[i], durs[i], pitches[i], velocities[i], channels[i])
        notespertrack[track].append(ev)
    
    for notes in notespertrack:
        notes.sort()
        for j in range(len(notes) - 1):
            note0, note1 = notes[j], notes[j + 1]
            if note0.start + note0.dur > note1.start:
                notes[j] = note0._replace(dur=note1.start - note0.start)

    for tracknum, notes in enumerate(notespertrack):
        if not notes:
            continue
        t = miditracks[tracknum]
        for note in notes:
            t.append(midi.NoteOnEvent(tick=time2tick(note.start), 
                                      pitch=note.pitch, 
                                      velocity=note.vel))
            t.append(midi.NoteOffEvent(tick=time2tick(note.start + note.dur), 
                                       pitch=note.pitch))
        t.append(midi.EndOfTrackEvent(tick=time2tick(notes[-1].start + notes[-1].dur) + 1))
        
    midipattern = midi.Pattern(miditracks, resolution=resolution)
    midipattern.make_ticks_rel()
    midi.write_midifile(outfile, midipattern)
    return midipattern


def frames_to_midi(frames, outfile, tempo=60):
    """
    Write the frames as a midi-file.

    frames: a list of Frames or a pandas.DataFrame with the columns 
            (start, dur, pitch) defined
    
    Frame = namedtuple('Frame', 'start dur pitch')
    """
    starts, durs, pitches = normalize_frames(frames)
    return events_to_midi(starts=starts, durs=durs, pitches=pitches, 
                          tempo=tempo, outfile=outfile)
