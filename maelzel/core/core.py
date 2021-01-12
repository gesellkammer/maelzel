from __future__ import annotations
import functools as _functools
from math import sqrt
import os
import copy as _copy
import tempfile as _tempfile

import music21 as m21
from emlib import misc
from emlib.mathlib import intersection
from emlib.pitchtools import amp2db, db2amp, m2n, m2f, f2m, r2i, str2midi
from emlib import iterlib

from maelzel.music import m21tools
from maelzel.music import m21fix
from maelzel import scoring
from maelzel.snd import csoundengine

from ._base import *
from .common import CsoundEvent, PlayArgs, astuple, asF
from .config import logger
from .state import getState, currentConfig
from . import m21funcs
from . import play
from . import tools
from . import environment

from emlib.typehints import U, T, Opt, Iter, Seq, List, Dict, Tup


_playkeys = {'delay', 'dur', 'chan', 'gain', 'fade', 'instr', 'pitchinterpol',
             'fadeshape', 'tabargs', 'position'}


def _clone(obj, **kws):
    out = _copy.copy(obj)
    for k, v in kws.items():
        setattr(out, k, v)
    return out


class MusicObj:
    _showableInitialized = False

    __slots__ = ('dur', 'start', 'label', '_playargs', '_hash')

    def __init__(self, label=None, dur:time_t = None, start: time_t = None):
        # A label can be used to identify an object within a group of objects
        self.label: Opt[str] = label

        # A MusicObj can have a duration. A duration can't be 0
        if dur is not None:
            assert dur > 0
        self.dur: Opt[Fraction] = asF(dur) if dur is not None else None

        # start specifies a time offset for this object
        self.start: Opt[Fraction] = F(start) if start is not None else None

        # _playargs are set via .setplay and serve the purpose of
        # attaching playing parameters (like position, instrument)
        # to an object
        self._playargs: Opt[PlayArgs] = None

        # All MusicObjs should be hashable. For the cases where
        # calculating the hash is expensive, we cache that here
        self._hash: int = 0

    def resolvedDuration(self, cfg=None) -> Fraction:
        if self.dur is not None:
            return self.dur
        return (cfg or currentConfig())['defaultDuration']

    @property
    def playargs(self):
        p = self._playargs
        if not p:
            self._playargs = p = PlayArgs()
        return p

    def setplay(self:T, **kws) -> T:
        """
        Pre-set any of the play arguments
         
        Args:
            **kws: any argument passed to .play 

        Returns:
            self
            
        Example:
            
            # a piano note
            note = Note("C4+25", dur=0.5).setplay(instr="piano")
        """
        for k, v in kws.items():
            if k not in _playkeys:
                raise KeyError(f"key {k} not known. "
                               f"Possible keys are {_playkeys}")
            setattr(self.playargs, k, v)
        return self

    def clone(self:T, **kws) -> T:
        """
        Clone this object, changing parameters if needed

        Args:
            **kws: any keywords passed to the constructor

        Returns:
            a clone of this objects, with the given arguments 
            changed
            
        Example:
            
            a = Note("C4+", dur=1)
            b = a.clone(dur=0.5)
        """
        out = _copy.deepcopy(self)
        for key, value in kws.items():
            setattr(out, key, value)
        return out

    def copy(self:T) -> T:
        return _copy.deepcopy(self)

    def timeShift(self:T, timeoffset:time_t) -> T:
        """
        Return a copy of this object with an added time offset

        Example: create a seq. of syncopations

        n = Note("A4", start=0.5, dur=0.5)
        track = Track([n, n.timeShift(1), n.timeShift(2), n.timeShift(3)])

        This is the same as 

        seq = Track([n, n>>1, n>>2, n>>3])
        """
        start = self.start or F(0)
        return self.clone(start=timeoffset + start)

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return hash(self) == hash(other)
        else:
            return False

    def __ne__(self, other) -> bool:
        return not(self == other)

    def __rshift__(self:T, timeoffset:time_t) -> T:
        return self.timeShift(timeoffset)

    def __lshift__(self:T, timeoffset:time_t) -> T:
        return self.timeShift(-timeoffset)

    @property
    def end(self) -> Opt[Fraction]:
        if not self.dur:
            return None
        start = self.start if self.start is not None else 0
        return start + self.dur

    def quantizePitch(self:T, step=1.0) -> T:
        """ Returns a new object, rounded to step """
        raise NotImplementedError()

    def transpose(self:T, step) -> T:
        """ Transpose self by `step` """
        raise NotImplementedError()

    def freqratio(self:T, ratio) -> T:
        """ Transpose this by a given freq. ratio. A ratio of 2 equals
        to transposing an octave higher. """
        return self.transpose(r2i(ratio))

    def show(self, external=None, method:str=None, fmt:str=None) -> None:
        """
        Show this as notation.

        Args:
            external: force opening the image in an external image viewer,
                even when inside a jupyter notebook. Otherwise, show will
                display the image inline
            method: one of 'lilypond', 'musicxml'. None to use default
                (see config['show.method'])
            fmt: one of 'png', 'pdf'. None to use default

        NB: to use the music21 show capabilities, use note.asmusic21().show(...)
        """
        cfg = currentConfig()
        if external is None:
            external = cfg['show.external']
        if method is None:
            method = cfg['show.method']
        if fmt is None:
            fmt = 'png' if environment.insideJupyter else cfg['show.format']
        img = self.makeImage(method=method, fmt=fmt)
        if fmt == 'png':
            tools.pngShow(img, forceExternal=external)
        else:
            misc.open_with_standard_app(img)

    def _changed(self) -> None:
        """
        This method is called whenever the object changes its representation
        (a note changes its pitch inplace, the duration is modified, etc)
        This invalidates, among other things, the image cache for this 
        object
        """
        self._hash = None

    def makeImage(self, method:str=None, outfile:str=None, fmt="png") -> str:
        """
        Creates an image representation, returns the path to the image

        Args:
            method: the rendering backend. One of 'musicxml', 'lilypond'
                None uses the default method (see config['show.method'])
            outfile: the path of the generated file. Use None to generate
                a temporary file.
            fmt: if outfile is None, fmt will determine the format of the
                generated file. Possible values: 'png', 'pdf'.

        Returns:
            the path of the generated file. If outfile was given, the returned
            path will be the same as the outfile.
        """
        # In order to be able to cache the images we put this
        # functionality outside of the class and use lru_cache
        if currentConfig()['show.cacheImages']:
            return renderObject(self, method=method, outfile=outfile, fmt=fmt)
        return _renderObject(self, method=method, outfile=outfile, fmt=fmt)

    def ipythonImage(self):
        """
        Generate a jupyter image from this object, to be used
        within a jupyter notebook

        Returns:
            an IPython.core.display.Image

        """
        from IPython.core.display import Image
        return Image(self.makeImage(fmt='png'), embed=True)

    def scoringEvents(self) -> List[scoring.Notation]:
        """
        Each class should be able to return its notated form as
        an intermediate representation in the form of scoring.Notations.
        These can then be converted into concrete notation via
        musicxml or lilypond

        Returns:
            A list of scoring.Notation which best represent this
            object as notation
        """
        raise NotImplementedError("Subclass should implement this")

    def scoringParts(self) -> List[scoring.Part]:
        notations = self.scoringEvents()
        scoring.stackNotationsInPlace(notations)
        parts = scoring.splitNotationsByClef(notations)
        return parts

    def _scoringAnnotation(self) -> Opt[scoring.Annotation]:
        if not self.label:
            return None
        return scoring.Annotation(self.label, fontSize=currentConfig()['show.label.fontSize'])

    def asmusic21(self, split=None, **options) -> m21.stream.Stream:
        """
        This method is used within .show, to convert this object
        into music notation. When using the musicxml backend
        we first convert our object/s into music21 and
        use the music21 framework to generate an image

        Args:

            split: overrides the 'show.split' setting in the config
            **options: not used here, but classes inheriting from
                this may want to add customization

        Returns:
            a music21 stream which best represent this object as
            notation.

        NB: the music21 representation should be final, not thought to
            be embedded into another stream. For embedding we use
            an abstract representation of scoring objects which can
            be queried via .scoringEvents
        """
        parts = self.scoringParts()
        options = makeRenderOptions()
        renderer = scoring.render.renderParts(parts, backend='musicxml', options=options)
        return renderer.asMusic21()

    def musicxml(self) -> str:
        """
        Return the representation of this object as musicxml. A subclass can
        override this method to provide a way of outputting musicxml which
        bypasses music21
        """
        m = self.asmusic21()
        if currentConfig()['m21.fixstream']:
            m21fix.fixStream(m)
        return m21tools.getXml(m)

    def write(self, outfile:str, backend:str=None) -> None:
        """
        Write this as musicxml, lilypond, pdf, png

        Args:
            outfile: the path of the output file. The extension determines
                the format
            backend: the backend used when writing as pdf or png.
                If not given, the default defined in the current
                configuration is used (this is va
        """
        ext = os.path.splitext(outfile)[1]
        if ext == '.ly':
            backend = 'lilypond'
        elif backend is None:
            cfg = currentConfig()
            backend = cfg['show.method']
        r = scoring.render.renderParts(self.scoringParts(), backend=backend)
        r.write(outfile)

    @classmethod
    def _setJupyterHook(cls) -> None:
        """
        Sets the jupyter display hook for this class

        """
        if cls._showableInitialized:
            return
        from IPython.core.display import Image

        def reprpng(obj):
            imgpath = obj.makeImage()
            scaleFactor = currentConfig().get('show.scaleFactor', 1.0)
            if scaleFactor != 1.0:
                imgwidth, imgheight = tools.imgSize(imgpath)
                width = imgwidth * scaleFactor
            else:
                width = None
            return Image(filename=imgpath, embed=True, width=width)._repr_png_()
            
        tools.setJupyterHookForClass(cls, reprpng, fmt='image/png')

    def csoundEvents(self, playargs:PlayArgs) -> List[CsoundEvent]:
        """
        This should be overriden by each class to generate CsoundEvents

        Args:
            playargs: a PlayArgs, structure, filled with given values,
                own .playargs values and config defaults (in that order)

        Returns:
            a list of CsoundEvents
        """
        raise NotImplementedError("Subclass should implement this")

    def _getDelay(self, delay:time_t=None) -> time_t:
        """
        This is here only to document how delay is calculated

        Args:
            delay: a delay to override playargs['delay']

        Returns:
            the play delay of this object
        """
        return misc.firstval(delay, self.playargs.delay, 0)+(self.start or 0.)

    def _fillPlayArgs(self,
                      delay:float = None,
                      dur:float = None,
                      chan:int = None,
                      gain:float = None,
                      fade=None,
                      instr:str = None,
                      pitchinterpol:str = None,
                      fadeshape:str = None,
                      args: Dict[str, float] = None,
                      position: float = None
                      ) -> PlayArgs:
        """
        Fill playargs with given values and defaults.

        The priority chain is:
            given value as param, prefilled value (playargs), own value, config/default value
        """
        playargs = self.playargs
        config = currentConfig()
        dur = misc.firstval(dur, playargs.dur, self.dur, config['play.dur'])
        if dur < 0:
            dur = MAXDUR
        return PlayArgs(
            dur = dur,
            delay = misc.firstval(delay, playargs.delay, 0)+(self.start or 0.),
            gain = gain or playargs.gain or config['play.gain'],
            instr = instr or playargs.instr or config['play.instr'],
            chan = chan or playargs.chan or config['play.chan'],
            fade = misc.firstval(fade, playargs.fade, config['play.fade']),
            pitchinterpol = pitchinterpol or playargs.pitchinterpol or config['play.pitchInterpolation'],
            fadeshape = fadeshape or playargs.fadeshape or config['play.fadeShape'],
            args = args or playargs.args,
            position = misc.firstval(position, playargs.position, 0)
        )

    def events(self, delay:float=None, dur:float=None, chan:int=None,
               gain:float=None, fade=None, instr:str=None,
               pitchinterpol:str=None, fadeshape:str=None,
               args: Dict[str, float] = None,
               position: float = None
               ) -> List[CsoundEvent]:
        """
        An object always has a start time. It can be unset (None), which defaults to 0
        but can also mean unset for contexts where this is meaningful (a sequence of Notes,
        for example, where they are concatenated one after the other, the start time
        is the end of the previous Note)

        All these attributes here can be set previously via .playargs (or
        using .setplay)

        Params:

            delay: A delay, if defined, is added to the start time.
            dur: play duration
            chan: the chan to play (or rec) this object
            gain: gain modifies .amp
            fade: fadetime or (fadein, fadeout)
            instr: the name of the instrument
            pitchinterpol: 'linear' or 'cos'
            fadeshape: 'linear' or 'cos'
            position: the panning position (0=left, 1=right). The left channel
                is determined by chan

        Returns:
            A list of CsoundEvents

        """
        playargs = self._fillPlayArgs(delay=delay, dur=dur, chan=chan, gain=gain,
                                      fade=fade, instr=instr,
                                      pitchinterpol=pitchinterpol, fadeshape=fadeshape,
                                      args=args,
                                      position=position)
        events = self.csoundEvents(playargs)
        return events

    def play(self, 
             dur: float = None, 
             gain: float = None, 
             delay: float = None, 
             instr: str = None, 
             chan: int = None, 
             pitchinterpol: str = None,
             fade: U[float, Tup[float, float]] = None,
             fadeshape: str = None,
             args: Dict[str, float] = None,
             position: float = None) -> csoundengine.AbstrSynth:
        """
        Plays this object. Play is always asynchronous (to block, use
        some sleep funcion)
        By default, .play schedules this event to be renderer in realtime.
        
        NB: to record multiple events offline, see the example below

        Args:
            dur: the duration of the event
            gain: modifies the own amplitude for playback/recording (0-1)
            delay: delay in seconds, added to the start time of the object
            instr: which instrument to use (see defInstrPreset, availableInstrPresets)
            chan: the channel to output to (an int starting with 1).
            pitchinterpol: 'linear', 'cos', 'freqlinear', 'freqcos'
            fade: fade duration (can be a tuple (fadein, fadeout)
            fadeshape: 'linear' | 'cos'
            args: paramaters passed to the note through an associated table.
                A dict paramName: value
            position: the panning position (0=left, 1=right)

        Returns:
            A SynthGroup

        Example:
            # play a note
            note = Note(60).play(gain=0.1, chan=2)

            # record offline
            with play.rendering(sr=44100, outfile="out.wav"):
                Note(60).play(gain=0.1, chan=2)
                ... other objects.play(...)
        """
        events = self.events(delay=delay, dur=dur, chan=chan,
                             fade=fade, gain=gain, instr=instr,
                             pitchinterpol=pitchinterpol, fadeshape=fadeshape,
                             args=args, position=position)
        renderer = getState().renderer
        if renderer is None:
            return playEvents(events)
        else:
            # schedule offline
            return renderer.schedMany(events)

    def rec(self, outfile:str=None, **kws) -> str:
        """
        Record the output of .play as a soundfile

        Args:
            outfile: the outfile where sound will be recorded. Can be
                None, in which case a filename will be generated
            **kws: any keyword passed to .play

        Returns:
            the path of the generated soundfile
        """
        events = self.events(**kws)
        return play.recEvents(events, outfile)

    def isRest(self) -> bool:
        return False


@_functools.total_ordering
class Note(MusicObj):

    __slots__ = ('midi', 'amp', 'endmidi')

    def __init__(self,
                 pitch: pitch_t,
                 amp:float=None,
                 dur:Fraction=None,
                 start:Fraction=None,
                 endpitch: pitch_t=None,
                 label:str=None, 
                 ):
        """
        In its simple form, a Note is used to represent a Pitch.
        
        A Note must have a pitch. It is possible to specify
        an amplitude, a duration, a time offset (.start), 
        and an endpitch, resulting in a glissando
        
        Args:
            pitch: a midinote or a note as a string
            amp: amplitude 0-1 (optional)
            dur: the duration of this note (optional)
            start: start fot the note (optional)
            endpitch: if given, defines a glissando
            label: a label to identify this note
        """
        MusicObj.__init__(self, label=label, dur=dur, start=start)
        self.midi: float = tools.asmidi(pitch)
        self.amp:  Opt[float] = amp
        self.endmidi: float = tools.asmidi(endpitch) \
            if endpitch is not None else None

    def __hash__(self) -> int:
        return hash((self.midi, self.dur, self.start, self.endmidi, self.label))

    def clone(self, pitch:pitch_t=UNSET, amp:float=UNSET,
              dur:Opt[time_t]=UNSET, start:Opt[time_t]=UNSET, label:str=UNSET,
              endpitch:pitch_t=UNSET
              ) -> Note:
        # we can't use the base .clone method because pitch can be anything
        return Note(pitch=pitch if pitch is not UNSET else self.midi,
                    amp=amp if amp is not UNSET else self.amp,
                    dur=dur if dur is not UNSET else self.dur,
                    start=start if start is not UNSET else self.start, 
                    label=label if label is not UNSET else self.label,
                    endpitch=endpitch if endpitch is not UNSET else self.endmidi)

    def asChord(self) -> Chord:
        endpitches = None if not self.endmidi else [self.endmidi]
        return Chord([self], amp=self.amp, dur=self.dur, start=self.start,
                     endpitches=endpitches, label=self.label)

    def isRest(self) -> bool:
        return self.amp == 0
        
    def freqShift(self, freq:float) -> Note:
        """
        Return a copy of self, shifted in freq.

        C3.shift(C3.freq)
        -> C4
        """
        return self.clone(pitch=f2m(self.freq + freq))

    def transpose(self, interval: float) -> Note:
        """ Return a copy of self, transposed by given `interval`"""
        return self.clone(pitch=self.midi+interval)

    def __lt__(self, other:pitch_t) -> bool:
        if isinstance(other, Note):
            return self.midi < other.midi
        else:
            raise NotImplementedError()

    @property
    def freq(self) -> float: return m2f(self.midi)

    @freq.setter
    def freq(self, value:float) -> None: self.midi = f2m(value)

    @property
    def name(self) -> str: return m2n(self.midi)

    def roundPitch(self, semitoneDivisions:int=0) -> Note:
        divs = semitoneDivisions or currentConfig()['show.semitoneDivisions']
        res = 1 / divs
        return self.quantizePitch(res)
    
    def overtone(self, n:float) -> Note:
        return Note(f2m(self.freq * n))

    @property
    def cents(self) -> int:
        return tools.midicents(self.midi)

    @property
    def centsrepr(self) -> str:
        return tools.centsshown(self.cents,
                                divsPerSemitone=currentConfig()['show.semitoneDivisions'])

    def hasGliss(self) -> bool:
        return self.endmidi is not None and self.endmidi != self.midi

    def scoringEvents(self) -> List[scoring.Notation]:
        config = currentConfig()
        dur = self.dur or config['defaultDuration']
        assert dur is not None
        if self.isRest():
            notes = [scoring.makeRest(self.dur, offset=self.start)]
        else:
            notes = [scoring.makeNote(pitch=self.midi, duration=F(dur), offset=self.start,
                                      gliss=self.hasGliss(),
                                      playbackGain=self.amp)]
            if self.endmidi:
                start = self.end if self.start is not None else None
                notes.append(scoring.makeNote(pitch=self.endmidi, duration=0,
                                              offset=start))

        if self.label:
            annot = self._scoringAnnotation()
            if annot is not None:
                notes[0].addAnnotation(annot)
        return notes

    def _asTableRow(self) -> List[str]:
        if self.isRest():
            elements = ["REST"]
        else:
            elements = [m2n(self.midi)]
            config = currentConfig()
            if config['repr.showFreq']:
                elements.append("%dHz" % int(self.freq))
            if self.amp is not None and self.amp < 1:
                elements.append("%ddB" % round(amp2db(self.amp)))
        if self.dur:
            elements.append(f"dur={tools.showTime(self.dur)}")
        if self.start is not None:
            elements.append(f"start={tools.showTime(self.start)}")
        return elements

    def __repr__(self) -> str:
        elements = self._asTableRow()
        return f'{elements[0].ljust(3)} {" ".join(elements[1:])}'

    def __str__(self) -> str: return self.name

    def __float__(self) -> float: return float(self.midi)

    def __int__(self) -> int: return int(self.midi)

    def __add__(self, other) -> U[Note, Chord]:
        if isinstance(other, (int, float)):
            return self.clone(pitch=self.midi+other,
                              endpitch=self.endmidi+other if self.endmidi else None)
        elif isinstance(other, str):
            return self + asNote(other)
        raise TypeError(f"can't add {other} ({other.__class__}) to a Note")

    def __xor__(self, freq) -> Note: return self.freqShift(freq)

    def __sub__(self, other: U[Note, float, int]) -> Note:
        if isinstance(other, Note):
            raise TypeError("can't substract one note from another")
        elif isinstance(other, (int, float)):
            return self + (-other)
        raise TypeError(f"can't substract {other} ({other.__class__}) from a Note")

    def quantizePitch(self, step=1.0) -> Note:
        """ Returns a new Note, rounded to step """
        return self.clone(pitch=round(self.midi / step) * step)

    def csoundEvents(self, playargs: PlayArgs) -> List[CsoundEvent]:
        amp = 1.0 if self.amp is None else self.amp
        endmidi = self.endmidi or self.midi
        
        bps = [(0.,                  self.midi, amp), 
               (float(playargs.dur), endmidi,   amp)]
        
        return [CsoundEvent.fromPlayArgs(bps=bps, playargs=playargs)]

    def gliss(self, dur:time_t, endpitch:pitch_t, endamp:float=None,
              start:time_t=None) -> Line:
        endnote = asNote(endpitch)
        startamp = self.resolveAmp()
        start = misc.firstval(start, self.start, 0.)
        endamp = misc.firstval(endamp, endnote.amp, self.amp, startamp)
        breakpoints = [(start, self.midi, startamp),
                       (start+dur, endnote.midi, endamp)]
        return Line(breakpoints)

    def resolveAmp(self) -> float:
        """
        Get the amplitude of this object, or a default amplitude
        if no amplitude was defined (self.amp is None)

        Returns:
            the amplitude
        """
        if self.amp is not None:
            return self.amp
        config = currentConfig()
        return config.get('play.defaultAmplitude', 1)


def Rest(dur:Fraction=1, start:Fraction=None) -> Note:
    """
    Create a Rest. A Rest is a Note with pitch 0 and amp 0.
    To test if an item is a rest, call isRest

    Args:
        dur: duration of the Rest
        start: start of the Rest

    Returns:

    """
    assert dur is not None and dur > 0
    return Note(pitch=0, dur=dur, start=start, amp=0)


def asNote(n: U[float, str, Note],
           amp:float=None, dur:time_t=None, start:time_t=None) -> Note:
    """
    Convert n to a Note

    n: str    -> notename
       number -> midinote
       Note   -> Note
    amp: 0-1

    A Note can also be created via `asNote((pitch, amp))`
    """
    if isinstance(n, Note):
        if any(x is not None for x in (amp, dur, start)):
            return n.clone(amp=amp, dur=dur, start=start)
        return n
    elif isinstance(n, (int, float)):
        return Note(n, amp=amp, dur=dur, start=start)
    elif isinstance(n, str):
        midi = str2midi(n)
        return Note(midi, amp=amp, dur=dur, start=start)
    elif isinstance(n, tuple) and len(n) == 2 and amp is None:
        return asNote(*n)
    raise ValueError(f"cannot express this as a Note: {n} ({type(n)})")


class Line(MusicObj):
    """ 
    A Line is a seq. of breakpoints, where each bp is of the form
    (delay, pitch, [amp=1, ...])


    delay: the time offset to the first breakpoint.
    pitch: the pitch as midinote or notename
    amp:   the amplitude (0-1), optional

    pitch, amp and any other following data can be 'carried'

    Line((0, "D4"), (1, "D5", 0.5), ..., fade=0.5)

    also possible:
    bps = [(0, "D4"), (1, "D5"), ...]
    Line(bps)   # without *

    a Line stores its breakpoints as
    [delayFromFirstBreakpoint, pitch, amp, ...]
    """

    __slots__ = ('bps',)

    def __init__(self, *bps, label="", delay:num_t=0, reltime=False):
        """

        Args:
            bps: breakpoints, a tuple of the form (delay, pitch, [amp=1, ...]), where
                delay is the time offset to the beginning of the line
                pitch is the pitch as notename or midinote
                amp is an amplitude between 0-1
            delay: time offset of the line itself
            label: a label to add to the line
            reltime: if True, the first value of each breakpoint is a time offset
                from previous breakpoint
        """
        if len(bps) == 1 and isinstance(bps[0], list):
            bps = bps[0]
        bps = tools.carryColumns(bps)
        
        if len(bps[0]) < 2:
            raise ValueError("A breakpoint should be at least (delay, pitch)", bps)
        
        if len(bps[0]) < 3:
            bps = tools.addColumn(bps, 1)
        
        bps = [(bp[0], tools.asmidi(bp[1])) + astuple(bp[2:])
               for bp in bps]
        
        if reltime:
            now = 0
            absbps = []
            for _delay, *rest in bps:
                now += _delay
                absbps.append((now, *rest))
            bps = absbps
        assert all(all(isinstance(x, (float, int)) for x in bp) for bp in bps)
        assert all(bp1[0]>bp0[0] for bp0, bp1 in iterlib.pairwise(bps))
        super().__init__(dur=bps[-1][0], start=delay, label=label)
        self.bps = bps
        
    def getOffsets(self) -> List[num_t]:
        """ Return absolute offsets of each breakpoint """
        start = self.start
        return [bp[0] + start for bp in self.bps]

    def csoundEvents(self, playargs: PlayArgs) -> List[CsoundEvent]:
        return [CsoundEvent.fromPlayArgs(bps=self.bps, playargs=playargs)]

    def __hash__(self):
        return hash((self.start, *iterlib.flatten(self.bps)))
        
    def __repr__(self):
        return f"Line(start={self.start}, bps={self.bps})"

    def quantizePitch(self, step=1.0) -> Line:
        """ Returns a new object, rounded to step """
        bps = [ (bp[0], tools.quantizeMidi(bp[1]), bp[2:])
                for bp in self.bps ]
        return Line(bps)

    def transpose(self, step: float) -> Line:
        """ Transpose self by `step` """
        bps = [ (bp[0], bp[1] + step, bp[2:])
                for bp in self.bps ]
        return Line(bps)

    def scoringEvents(self) -> List[scoring.Notation]:
        offsets = self.getOffsets()
        group = scoring.makeId()
        notes = []
        for (bp0, bp1), offset in zip(iterlib.pairwise(self.bps), offsets):
            ev = scoring.makeNote(pitch=bp0[1], offset=offset, duration=bp1[0] - bp0[0],
                                  gliss=bp0[1] != bp1[1], group=group)
            notes.append(ev)
        if(self.bps[-1][1] != self.bps[-2][1]):
            # add a last note if last pair needed a gliss (to have a destination note)
            notes.append(scoring.makeNote(pitch=self.bps[-1][1],
                                          offset=offsets[-1],
                                          group=group,
                                          duration=asTime(currentConfig()['show.lastBreakpointDur'])))
        if notes:
            annot = self._scoringAnnotation()
            if annot:
                notes[0].addAnnotation(annot)
        return notes

    def dump(self):
        elems = []
        if self.start:
            elems.append(f"delay={self.start}")
        if self.label:
            elems.append(f"label={self.label}")
        infostr = ", ".join(elems)
        print("Line:", infostr)
        durs = [bp1[0]-bp0[0] for bp0, bp1 in iterlib.pairwise(self.bps)]
        durs.append(0)
        rows = [(offset, offset+dur, dur) + bp
                for offset, dur, bp in zip(self.getOffsets(), durs, self.bps)]
        headers = ("start", "end", "dur", "offset", "pitch", "amp", "p4", "p5", "p6", "p7", "p8")
        misc.print_table(rows, headers=headers)


def mkEvent(pitch, dur:time_t=None, start:time_t=None, endpitch:pitch_t=None,
            amp:float=None, **kws
            ) -> U[Note, Chord]:
    """
    Create a Note or Chord. If pitch is a list of pitches, creates a Chord

    Args:
        pitch: a pitch (as float, int, str) or list of pitches (also a str
            with spaces, like "A4 C5"). If multiple pitches are passed,
            the result is a Chord
        dur: the duration of the note/chord (optional)
        start: the start time of the note/chord (optional)
        endpitch: the end pitch of the note/chord (optional, must match the
            number of pitches passes as start pitch)
        amp: the amplitude of the note/chord (optional)
        kws: any other keywords are passed to the Note or Chord constructor

    Returns:
        a Note or Chord, depending on the number of pitches passed
    """
    if isinstance(pitch, (tuple, list)):
        return Chord(pitch, dur=dur, start=start, endpitches=endpitch, amp=amp, **kws)
    elif isinstance(pitch, str):
        if " " in pitch:
            return Chord(pitch, dur=dur, start=start, endpitches=endpitch, amp=amp, **kws)
        else:
            return Note(pitch, dur=dur, start=start, endpitch=endpitch, amp=amp, **kws)
    else:
        return Note(pitch, dur=dur, start=start, endpitch=endpitch, amp=amp, **kws)


N = mkEvent


class Chord(MusicObj):

    __slots__ = ('amp', 'endchord', 'notes')

    def __init__(self, *notes, amp:float=None,
                 dur:time_t=None, start=None, endpitches=None, label:str=''
                 ) -> None:
        """
        a Chord can be instantiated as:

            Chord(note1, note2, ...) or
            Chord([note1, note2, ...])
            Chord("C4 E4 G4")

        where each note is either a Note, a notename ("C4", "E4+", etc), a midinote
        or a tuple (midinote, amp)

        label: str. If given, it will be used for printing purposes, if possible
        """
        self.amp = amp
        self._hash = None
        if dur is not None:
            assert dur > 0
            dur = F(dur)
        self.notes: List[Note] = []
        if notes:
            # notes might be: Chord([n1, n2, ...]) or Chord(n1, n2, ...)
            if misc.isgenerator(notes):
                notes = list(notes)
            n0 = notes[0]
            if len(notes) == 1:
                if isinstance(n0, (Chord, EventSeq)):
                    notes = list(n0)
                elif isinstance(n0, (list, tuple)):
                    notes = notes[0]
                elif isinstance(n0, str):
                    notes = n0.split()
                    notes = [N(n) for n in notes]
            # determine dur & start
            if dur is None:
                dur = max((n.dur for n in notes if isinstance(n, Note)), default=None)
            if start is None:
                start = min((n.start for n in notes if isinstance(n, Note)), default=None)
            MusicObj.__init__(self, dur=dur, start=start, label=label)

            for note in notes:
                if isinstance(note, Note):
                    # we erase any duration or offset of the individual notes
                    note = note.clone(dur=None, amp=amp, start=None)
                else:
                    note = asNote(note, amp=amp, dur=dur, start=None)
                self.notes.append(note)
            self.sort()
            self.endchord = asChord(endpitches) if endpitches else None

    def __len__(self) -> int:
        return len(self.notes)

    def __getitem__(self, idx) -> U[Note, Chord]:
        out = self.notes.__getitem__(idx)
        if isinstance(out, list):
            out = self.__class__(out)
        return out

    def __iter__(self) -> Iter[Note]:
        return iter(self.notes)

    def scoringEvents(self) -> List[scoring.Notation]:
        config = currentConfig()
        pitches = [note.midi for note in self.notes]
        annot = self._scoringAnnotation()
        endpitches = None if not self.endchord else [note.midi for note in self.endchord]
        dur = self.dur if self.dur is not None else config['defaultDuration']
        chord = scoring.makeChord(pitches=pitches, duration=asF(dur), offset=self.start,
                                  annotation=annot, playbackGain=self.amp)
        events = [chord]
        if endpitches:
            endEvent = chord.clone(duration=0, offset=self.end)
            events.append(endEvent)
        return events

    def _asmusic21(self, **kws) -> m21.stream.Stream:
        config = currentConfig()
        arpeggio = _normalizeChordArpeggio(kws.get('arpeggio', None), self)
        if arpeggio:
            dur = config['show.arpeggioDuration']
            return EventSeq(self.notes, itemDefaultDur=dur).asmusic21()
        events = self.scoringEvents()
        scoring.stackNotationsInPlace(events, start=self.start)
        parts = scoring.splitNotationsByClef(events)
        return scoring.render.renderParts(parts).asMusic21()

    def __hash__(self):
        if self._hash:
            return self._hash
        data = (self.dur, self.start, *(n.midi for n in self.notes))
        if self.endchord:
            data = (data, tuple(n.midi for n in self.endchord))
        self._hash = h = hash(data)
        return h

    def append(self, note:pitch_t) -> None:
        """ append a note to this Chord """
        note = asNote(note)
        if note.freq < 17:
            logger.debug(f"appending a note with very low freq: {note.freq}")
        self.notes.append(note)
        self._changed()

    def extend(self, notes) -> None:
        """ extend this Chord with the given notes """
        for note in notes:
            self.notes.append(asNote(note))
        self._changed()

    def insert(self, index:int, note:pitch_t) -> None:
        self.notes.insert(index, asNote(note))
        self._changed()

    def filter(self, func) -> Chord:
        """
        Example: filter out notes lower than the lowest note of the piano

        return ch.filter(lambda n: n > "A0")
        """
        return Chord([n for n in self if func(n)])
        
    def transpose(self, step:float) -> Chord:
        """
        Return a copy of self, transposed `step` steps
        """
        return Chord([note.transpose(step) for note in self])

    def transposeTo(self, fundamental:pitch_t) -> Chord:
        """
        Return a copy of self, transposed to the new fundamental
        NB: the fundamental is the lowest note in the chord

        Args:
            fundamental: the new lowest note in the chord

        Returns:
            A Chord transposed to the new fundamental
        """
        step = asMidi(fundamental) - self[0].midi
        return self.transpose(step)

    def freqShift(self, freq:float) -> Chord:
        """
        Return a copy of this chord shifted in frequency
        """
        return Chord([note.freqShift(freq) for note in self])

    def roundPitch(self, semitoneDivisions:int=0) -> Chord:
        """
        Returns a copy of this chord, with pitches rounded according
        to semitoneDivisions

        Args:
            semitoneDivisions: if 2, pitches are rounded to the next
                1/4 tone

        Returns:
            the new Chord
        """
        divs = semitoneDivisions or currentConfig()['show.semitoneDivisions']
        notes=[note.roundPitch(divs) for note in self]
        return self._withNewNotes(notes)
    
    def _withNewNotes(self, notes) -> Chord:
        return Chord(notes, start=self.start, dur=self.dur, amp=self.amp)

    def quantizePitch(self, step=1.0) -> Chord:
        """
        Returns a copy of this chord, with the pitches
        quantized. Two notes with the same pitch are considered
        equal if they quantize to the same pitch, independently
        of their amplitude. In the case of two equal notes, only
        the first one is kept.
        """
        seenmidi = set()
        notes = []
        for note in self:
            note2 = note.quantizePitch(step)
            if note2.midi not in seenmidi:
                seenmidi.add(note2.midi)
                notes.append(note2)
        return self._withNewNotes(notes)

    def __setitem__(self, i:int, obj:pitch_t) -> None:
        self.notes.__setitem__(i, asNote(obj))
        self._changed()

    def __add__(self, other:pitch_t) -> Chord:
        if isinstance(other, Note):
            s = Chord(self)
            s.append(other)
            return s
        elif isinstance(other, (int, float)):
            s = [n + other for n in self]
            return Chord(s)
        elif isinstance(other, (Chord, str)):
            return Chord(self.notes + asChord(other).notes)
        raise TypeError("Can't add a Chord to a %s" % other.__class__.__name__)

    def splitByAmp(self, numChords=8, maxNotesPerChord=16) -> List[Chord]:
        """
        Split the notes in this chord into several chords, according
        to their amplitude

        Args:
            numChords: the number of chords to split this chord into
            maxNotesPerChord: max. number of notes per chord

        Returns:
            a list of Chords
        """
        midis = [note.midi for note in self.notes]
        amps = [note.amp for note in self.notes]
        chords = tools.splitByAmp(midis, amps, numGroups=numChords,
                                  maxNotesPerGroup=maxNotesPerChord)
        return [Chord(chord) for chord in chords]

    def loudest(self, n:int) -> Chord:
        """
        Return a new Chord with the loudest `n` notes from this chord
        """
        return self.copy().sort(key='amp', reverse=True)[:n]

    def sort(self, key='pitch', reverse=False) -> Chord:
        """
        Sort INPLACE. If inplace sorting is undesired, use

        sortedchord = chord.copy().sort()

        Args:
            key: either 'pitch' or 'amp'
            reverse: similar as sort

        Returns:
            self
        """
        if key == 'pitch':
            self.notes.sort(key=lambda n: n.midi, reverse=reverse)
        elif key == 'amp':
            self.notes.sort(key=lambda n:n.amp, reverse=reverse)
        return self

    def csoundEvents(self, playargs) -> List[CsoundEvent]:
        gain = playargs.gain
        config = currentConfig()
        if config['chord.adjustGain']:
            gain *= 1/sqrt(len(self))
        if self.endchord is None:
            return sum((note.csoundEvents(playargs) for note in self), [])
        events = []
        for note0, note1 in zip(self.notes, self.endchord):
            bps = [(0, note0.midi, note0.amp*gain),
                   (playargs.dur, note1.midi, note1.amp*gain)]
            events.append(CsoundEvent.fromPlayArgs(bps=bps, playargs=playargs))
        return events

    def asSeq(self, dur=None) -> EventSeq:
        return EventSeq(self.notes, itemDefaultDur=dur or self.dur)

    def __repr__(self):
        lines = []
        justs = [6, -6, -8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

        def justify(s, spaces):
            if spaces > 0:
                return s.ljust(spaces)
            return s.rjust(-spaces)

        cls = self.__class__.__name__
        indent = " " * len(cls)
            
        for i, n in enumerate(sorted(self.notes, key=lambda note:note.midi, reverse=True)):
            elements = n._asTableRow()
            line = " ".join(justify(element, justs[i])
                            for i, element in enumerate(elements))
            if i == 0:
                line = f"{cls} | " + line
            else:
                line = f"{indent} | " + line
            lines.append(line)
        return "\n".join(lines)
        
    def mapAmplitudes(self, curve, db=False) -> Chord:
        """
        Return a new Chord with the amps of the notes modified according to curve
        
        Example #1: compress all amplitudes to 30 dB

        curve = bpf.linear(-90, -30, -30, -12, 0, 0)
        newchord = chord.mapamp(curve, db=True)

        curve:
            a func mapping amp -> amp
        db:
            if True, the value returned by func is interpreted as db
            if False, it is interpreted as amplitude (0-1)
        """
        notes = []
        if db:
            for note in self:
                db = curve(amp2db(note.amp))
                notes.append(note.clone(amp=db2amp(db)))
        else:
            for note in self:
                amp2 = curve(note.amp)
                notes.append(note.clone(amp=amp2))
        return Chord(notes)

    def setAmplitudes(self, amp: float) -> Chord:
        """
        Returns a new Chord where each note has the given amp. 
        This is a shortcut to

        ch2 = Chord([note.clone(amp=amp) for note in ch])

        See also: .scaleamp
        """
        return self.scaleAmpliudes(0, offset=amp)

    def scaleAmpliudes(self, factor:float, offset=0.0) -> Chord:
        """
        Returns a new Chord with the amps scales by the given factor
        """
        return Chord([note.clone(amp=note.amp*factor+offset)
                      for note in self.notes])

    def equalize(self:T, curve) -> T:
        """
        Return a new Chord equalized by curve

        curve: a func(freq) -> gain
        """
        notes = []
        for note in self:
            gain = curve(note.freq)
            notes.append(note.clone(amp=note.amp*gain))
        return self.__class__(notes)

    def gliss(self, dur:float, endnotes, start=None) -> Chord:
        """
        Create a glissando between this chord and the endnotes given

        dur: the dur of the glissando
        endnotes: the end of the gliss, as Chord, list of Notes or string

        Example: semitone glissando in 2 seconds

        ch = Chord("C4", "E4", "G4")
        ch2 = ch.gliss(2, ch.transpose(-1))

        Example: gliss with diminuendo

        Chord("C4 E4", amp=0.5).gliss(5, Chord("E4 G4", amp=0).play()
        """
        endchord = asChord(endnotes)
        if len(endchord) != len(self):
            raise ValueError(f"The number of end notes {len(endnotes)} != the"
                             f"size of this chord {len(self)}")
        startpitches = [note.midi for note in self.notes]
        endpitches = [note.midi for note in endchord]
        assert len(startpitches) == len(endpitches)
        out = Chord(*startpitches, amp=self.amp, label=self.label, endpitches=endpitches)
        out.dur = asTime(dur)
        out.start = None if start is None else asTime(start)
        return out

    def difftones(self) -> Chord:
        """
        Return a Chord representing the difftones between the notes of this chord
        """
        from maelzel.music.combtones import difftones
        return Chord(difftones(*self))

    def isCrowded(self) -> bool:
        return any(abs(n0.midi-n1.midi)<=1 and abs(n1.midi-n2.midi)<=1
                   for n0, n1, n2 in iterlib.window(self, 3))

    def _splitChord(self, splitpoint=60.0, showcents=None, showlabel=True) -> m21.stream.Score:
        config = currentConfig()
        if showcents is None: showcents = config['show.cents']
        parts = splitNotesIfNecessary(self.notes, float(splitpoint))
        score = m21.stream.Score()
        for notes in parts:
            midinotes = [n.midi for n in notes]
            m21chord = m21funcs.m21Chord(midinotes, showcents=showcents)
            part = m21.stream.Part()
            part.append(m21funcs.bestClef(midinotes))
            if showlabel and self.label:
                part.append(m21funcs.m21Label(self.label))
                showlabel = False
            part.append(m21chord)
            if config['show.centsMethod'] == 'expression':
                m21tools.makeExpressionsFromLyrics(part)
            score.insert(0, part)
        return score


def asChord(obj, amp:float=None, dur:float=None) -> Chord:
    """
    Create a Chord from `obj`

    Args:
        obj: a string with spaces in it, a list of notes, a single Note, a Chord
        amp: the amp of the chord
        dur: the duration of the chord

    Returns:
        a Chord
    """
    if isinstance(obj, Chord):
        out = obj
    elif isinstance(obj, (list, tuple, str)):
        out = Chord(obj)
    elif hasattr(obj, "asChord"):
        out = obj.asChord()
        assert isinstance(out, Chord)
    elif isinstance(obj, (int, float)):
        out = Chord(asNote(obj))
    else:
        raise ValueError(f"cannot express this as a Chord: {obj}")
    if amp is not None or dur is not None:
        out = out.clone(amp=amp, dur=dur)
    return out


def asEvent(obj, **kws) -> U[Note, Chord]:
    if isinstance(obj, (Note, Chord)):
        return obj
    return N(obj, **kws)


def _normalizeChordArpeggio(arpeggio: U[str, bool], chord: Chord) -> bool:
    config = currentConfig()
    if arpeggio is None: arpeggio = config['chord.arpeggio']

    if isinstance(arpeggio, bool):
        return arpeggio
    elif arpeggio == 'auto':
        return chord.isCrowded()
    else:
        raise ValueError(f"arpeggio should be True, False, 'auto' (got {arpeggio})")


def stackEvents(events: List[MusicObj], defaultDur:Fraction, start:Fraction=F(0)
                ) -> List[MusicObj]:
    if all(ev.start is not None and ev.dur is not None for ev in events):
        return events
    now = events[0].start if events[0].start is not None else start
    assert now is not None and now >= 0
    out = []
    lasti = len(events) - 1
    for i, ev in enumerate(events):
        if ev.start is None:
            ev = ev.clone(start=now, dur=ev.dur if ev.dur is not None else defaultDur)
        elif ev.dur is None:
            if i == lasti:
                raise ValueError("The last event should have a duration")
            ev = ev.clone(dur=events[i+1].start - ev.start)
        now += ev.dur
        out.append(ev)
    for ev1, ev2 in iterlib.pairwise(out):
        assert ev1.start <= ev2.start
    return out


def stackEventsInPlace(events: List[MusicObj], defaultDur:Fraction, start:Fraction=F(0)
                       ) -> None:
    if all(ev.start is not None and ev.dur is not None for ev in events):
        return
    now = events[0].start if events[0].start is not None else start
    assert now is not None and now >= 0
    lasti = len(events) - 1
    for i, ev in enumerate(events):
        if ev.start is None:
            ev.start = now
            ev.dur = ev.dur if ev.dur is not None else defaultDur
        elif ev.dur is None:
            if i == lasti:
                raise ValueError("The last event should have a duration")
            ev = ev.clone(dur=events[i+1].start - ev.start)
            ev.dur = events[i+1].start - ev.start
        now += ev.dur
    for ev1, ev2 in iterlib.pairwise(events):
        assert ev1.start <= ev2.start


class EventSeq(MusicObj):
    """
    A seq. of Notes or Chords
    """
    __slots__ = ('items', 'itemDefaultDur')

    def __init__(self, items: List[MusicObj], itemDefaultDur:time_t=None, start:time_t=None):
        if itemDefaultDur is None:
            self.itemDefaultDur = currentConfig()['defaultDuration']
        else:
            self.itemDefaultDur = asF(itemDefaultDur)

        super().__init__(dur=None, start=start)
        if items:
            items = [asEvent(item) for item in items]
            items = stackEvents(items, defaultDur=itemDefaultDur, start=start)
            self.items = items
            super().__init__(dur=self.items[0].start, start=self.items[0].start)
        else:
            self.items: List[U[Note, Chord]] = []
            super().__init__(dur=None, start=None)

    def resolvedDuration(self, cfg=None) -> Fraction:
        if not self.items:
            return F(0)
        return self.items[-1].end - self.items[0].start

    def append(self, item:U[Note, Chord]) -> None:
        if item.start is not None and item.dur is not None:
            assert item.start > self.end
            self.items.append(item)
        start = item.start if item.start is not None else (self.end or F(0))
        dur = item.resolvedDuration()
        item = item.clone(start=start, dur=dur)
        self.items.append(item)
        self.dur = item.end - self.start

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iter[Chord]:
        return iter(self.items)

    def __getitem__(self, idx):
        out = self.items.__getitem__(idx)
        if isinstance(out, MusicObj):
            return out
        elif isinstance(out, list):
            return self.__class__(out)
        else:
            raise ValueError(f"__getitem__ returned {out}, expected Chord or list of Chords")

    def scoringEvents(self) -> List[scoring.Notation]:
        events = []
        start = self.start if self.start is not None else F(0)
        for item in self.items:
            scoringEvents = item.scoringEvents()
            for ev in scoringEvents:
                if ev.duration is None and ev.offset is None:
                    ev.duration = self.dur
            events.extend(scoringEvents)
        if events and events[0].offset is None:
            events[0].offset = start
        return events

    def __repr__(self):
        lines = ["EventSeq "]
        for item in self.items:
            sublines = repr(item).splitlines()
            for subline in sublines:
                lines.append("    " + subline)
        return "\n".join(lines)

    def __hash__(self):
        if self._hash:
            return self._hash
        self._hash = hash(tuple(hash(ev) ^ 0x1234 for ev in self.items))
        return self._hash

    def csoundEvents(self, playargs: PlayArgs) -> List[CsoundEvent]:
        allevents = []
        for item in self.items:
            events = item.events()
            # events = item.csoundEvents(playargs)
            allevents.extend(events)
        return allevents

    def cycle(self, dur:float, crop=True) -> EventSeq:
        """
        Cycle the items in this seq. until the given duration is reached

        Args:
            dur: the total duration
            crop: if True, the last event will be cropped to fit
                the given total duration. Otherwise, it will last
                its given duration, even if that would result in
                a total duration longer than the given one
        """
        items = []
        defaultDur = self.dur
        it = iterlib.cycle(self)
        totaldur = 0
        while totaldur < dur:
            item = next(it)
            maxdur = dur - totaldur
            if crop:
                if item.dur is None or item.dur > maxdur:
                    item = item.clone(dur=maxdur)
            elif item.dur is None:
                if crop:
                    item = item.clone(dur=min(defaultDur, maxdur))
                else:
                    item = item.clone(dur=defaultDur)
            totaldur += item.dur
            items.append(item)
        return EventSeq(items, start=self.start)

    def clone(self, items:List[U[Note, Chord]]=None, dur:time_t=None, start:time_t=None
              ) -> EventSeq:
        items = items if items is not None else self.items
        dur = dur if dur is not None else self.dur
        start = start if start is not None else self.start
        return EventSeq(items, itemDefaultDur=dur, start=start)

    def transpose(self:EventSeq, step) -> EventSeq:
        chords = [obj.transpose(step) for obj in self.items]
        return self.clone(chords)

    def quantizePitch(self, step=1.0) -> EventSeq:
        items = [item.quantizePitch(step) for item in self.items]
        return self.clone(items)

    def timeShift(self, timeoffset:time_t) -> EventSeq:
        items = [item.timeShift(timeoffset) for item in self.items]
        return self.clone(items)


class Track(MusicObj):
    """
    A Track is a seq. of non-overlapping objects
    """

    def __init__(self, objs=None, label:str=''):
        self.items: List[MusicObj] = []
        self.instrs: Dict[MusicObj, str] = {}
        self.label = label
        super().__init__()
        if objs:
            for obj in objs:
                self.add(obj)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items.__getitem__(idx)

    def __len__(self):
        return len(self.items)

    def __hash__(self):
        hashes = [hash(item) for item in self.items]
        return hash(tuple(hashes))

    def _changed(self):
        if self.items:
            self.dur = self.items[-1].end-self.items[0].start
        super()._changed()

    def endTime(self) -> Fraction:
        if not self.items:
            return Fraction(0)
        return self.items[-1].end

    def isEmptyBetween(self, start:time_t, end:num_t):
        if not self.items:
            return True
        if start >= self.items[-1].end:
            return True
        if end < self.items[0].start:
            return True
        for item in self.items:
            if intersection(item.start, item.end, start, end):
                return False
        return True

    def needsSplit(self) -> bool:
        pass

    def add(self, obj:MusicObj) -> None:
        """
        Add this object to this Track. If obj has already a given start,
        it will be inserted at that offset, otherwise it will be appended
        to the end of this Track. 

        1) To insert an untimed object (for example, a Note with start=None) to the Track
           at a given offset, set its .start attribute or do track.add(chord.clone(start=...))

        2) To append a timed object at the end of this track (overriding the start
           time of the object), do track.add(obj.clone(start=track.endTime()))

        obj: the object to add (a Note, Chord, Event, etc.)
        """
        if obj.start is None or obj.dur is None:
            obj = _asTimedObj(obj, start=self.endTime(), dur=currentConfig()['defaultDuration'])
        if not self.isEmptyBetween(obj.start, obj.end):
            msg = f"obj {obj} ({obj.start}:{obj.start+obj.dur}) does not fit in track"
            raise ValueError(msg)
        assert obj.start is not None and obj.start >= 0 and obj.dur is not None and obj.dur > 0
        self.items.append(obj)
        self.items.sort(key=lambda obj:obj.start)
        self._changed()

    def extend(self, objs:List[MusicObj]) -> None:
        objs.sort(key=lambda obj:obj.start)
        assert objs[0].start >= self.endTime()
        for obj in objs:
            self.items.append(obj)
        self._changed()

    def scoringEvents(self) -> List[scoring.Notation]:
        return sum((obj.scoringEvents() for obj in self.items), [])
                  
    def csoundEvents(self, playargs: PlayArgs) -> List[CsoundEvent]:
        return sum((obj.csoundEvents(playargs) for obj in self.items), [])

    def play(self, **kws) -> csoundengine.SynthGroup:
        """
        kws: any kws is passed directly to each individual event
        """
        return csoundengine.SynthGroup([obj.play(**kws) for obj in self.items])

    def scoringPart(self) -> scoring.Part:
        return scoring.Part(self.scoringEvents(), label=self.label)

    def transpose(self:Track, step) -> Track:
        return Track([obj.transpose(step) for obj in self.items])

    def quantizePitch(self:Track, step=1.0) -> Track:
        return Track([obj.quantizePitch(step) for obj in self.items])


def _asTimedObj(obj: MusicObj, start, dur) -> MusicObj:
    """
    A TimedObj has a start time and a duration
    """
    if obj.start is not None and obj.dur is not None:
        return obj

    assert (start is not None) and (dur is not None)
    dur = obj.dur if obj.dur is not None else dur
    assert dur > 0
    start = obj.start if obj.start is not None else start
    assert start >= 0
    start = asTime(start)
    dur = asTime(dur)
    obj2 = obj.clone(dur=dur, start=start)
    assert obj2.dur is not None and obj2.start is not None
    return obj2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# notenames
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generateNotes(start=12, end=127) -> Dict[str, Note]:
    """
    Generates all notes for interactive use.

    From an interactive session, 

    locals().update(generate_notes())
    """
    notes = {}
    for i in range(start, end):
        notename = m2n(i)
        octave = notename[0]
        rest = notename[1:]
        rest = rest.replace('#', 'x')
        original_note = rest + str(octave)
        notes[original_note] = Note(i)
        if "x" in rest or "b" in rest:
            enharmonic_note = tools.enharmonic(rest)
            enharmonic_note += str(octave)
            notes[enharmonic_note] = Note(i)
    return notes


def setJupyterHook() -> None:
    MusicObj._setJupyterHook()
    tools.m21JupyterHook()


def _splitNotesOnce(notes: U[Chord, Seq[Note]], splitpoint:float, deviation=None,
                    ) -> Tup[List[Note], List[Note]]:
    """
    Split a list of notes into two lists, one above the splitpoint,
    one below

    Args:
        notes: a seq. of Notes
        splitpoint: the pitch to split the notes
        deviation: an acceptable deviation to fit all notes
            in one group (config: 'splitAcceptableDeviation')

    Returns:
        notes above and below

    """
    deviation = deviation or currentConfig()['splitAcceptableDeviation']
    if all(note.midi>splitpoint-deviation for note in notes):
        above = [n for n in notes]
        below = []
    elif all(note.midi<splitpoint+deviation for note in notes):
        above = []
        below = [n for n in notes]
    else:
        above, below = [], []
        for note in notes:
            (above if note.midi>splitpoint else below).append(note)
    return above, below


def splitNotes(notes: Iter[Note], splitpoints:List[float], deviation=None
               ) -> List[List[Note]]:
    """
    Split notes at given splitpoints. This can be used to split a group of notes
    into multiple staves

    Args:
        notes: the notes to split
        splitpoints: a list of splitpoints
        deviation: an acceptable deviation to fit all notes
            in one group (config: 'splitAcceptableDeviation')

    Returns:
        A list of list of notes, where each list contains notes either above,
        below or between splitpoints
    """
    splitpoints = sorted(splitpoints)
    tracks = []
    above = notes
    for splitpoint in splitpoints:
        above, below = _splitNotesOnce(above, splitpoint=splitpoint, deviation=deviation)
        if below:
            tracks.append(below)
        if not above:
            break
    return tracks


def splitNotesIfNecessary(notes:List[Note], splitpoint:float, deviation=None
                          ) -> List[List[Note]]:
    """
    Like _splitNotesOnce, but returns only groups which have notes in them
    This can be used to split in more than one staves:

    Args:
        notes: the notes to split
        splitpoint: the split point
        deviation: an acceptable deviation, if all notes could fit in one part

    Returns:
        a list of parts (a part is a list of notes)

    """
    return [p for p in _splitNotesOnce(notes, splitpoint, deviation) if p]


def makeRenderOptions() -> scoring.render.RenderOptions:
    config = currentConfig()
    renderOptions = scoring.render.RenderOptions(
            staffSize=config['show.staffSize'],
            divsPerSemitone=config['show.semitoneDivisions'],
            showCents = config['show.cents'],
            centsFontSize=config['show.centsFontSize'],
            noteAnnotationsFontSize=config['show.label.fontSize']
    )
    return renderOptions


def _renderObject(obj: MusicObj, outfile:str=None, method:str=None, fmt='png'
                  ) -> str:
    """
    Given a music object, make an image representation of it.
    NB: we put it here in order to make it easier to cache images

    Args:
        obj: the object to make the image from (a Note, Chord, etc.)
        outfile: the path to be generated. Can be None, in which case a temporary
            file is generated.
        method : one of 'musicxml', 'lilypond'
        fmt: the format of the generated file, if no outfile is given. One
            of 'png', 'pdf' (has no effect if outfile is given, in which case
            the extension determines the format)

    Returns:
        the path of the generated image
    """
    config = currentConfig()
    if method is None:
        method = config['show.method']
    else:
        methods = {'xml', 'musicxml', 'lily', 'lilypond'}
        if method not in methods:
            raise ValueError(f"method {method} not supported. Should be one of {methods}")

    if outfile is None:
        outfile = _tempfile.mktemp(suffix="." + fmt)

    renderOptions = makeRenderOptions()
    logger.debug(f"renderOptions: {renderOptions}")


    logger.debug(f"rendering parts with backend: {method}")
    parts = obj.scoringParts()
    renderer = scoring.render.renderParts(parts, backend=method,
                                          options=renderOptions)
    renderer.write(outfile)
    return outfile


@_functools.lru_cache(maxsize=1000)
def renderObject(obj:MusicObj, outfile:str=None, method:str=None, fmt='png'
                 ) -> str:
    """
    Given a music object, make an image representation of it.
    NB: we put it here in order to make it easier to cache images

    Args:
        obj     : the object to make the image from (a Note, Chord, etc.)
        outfile : the path to be generated. A .png filename
        method  : format used. One of 'musicxml', 'lilypond'
        fmt: the format of the generated object. One of 'png', 'pdf'

    Returns:
        the path of the generated image

    NB: we put it here in order to make it easier to cache images
    """
    return _renderObject(obj=obj, outfile=outfile, method=method, fmt=fmt)


def resetImageCache() -> None:
    """
    Reset the image cache. Useful when changing display format
    """
    renderObject.cache_clear()


def asMusic(obj) -> U[Note, Chord]:
    """
    Convert obj to a Note or Chord, depending on the input itself

    int, float      -> Note
    list (of notes) -> Chord
    "C4"            -> Note
    "C4 E4"         -> Chord
    """
    if isinstance(obj, (Note, Chord)):
        return obj
    elif isinstance(obj, str):
        if " " in obj:
            return Chord(obj.split())
        return Note(obj)
    elif isinstance(obj, (list, tuple)):
        return Chord(obj)
    elif isinstance(obj, (int, float)):
        return Note(obj)


def gliss(a, b, dur:time_t=1, start:time_t=None) -> U[Note, Chord]:
    """
    Create a gliss. between a and b. a should implement
    the method .gliss (either a Note or a Chord)
    Args:
        a: the start object
        b: the end object (should have the same type as obj1)
        dur: the duration of the glissando
        start: the start time of the glissando

    Returns:

    """
    m1 = asMusic(a)
    m2 = asMusic(b)
    assert isinstance(m2, type(m1))
    return m1.gliss(dur, m2, start=start)


class Group(MusicObj):
    """
    A Group represents a group of objects. They can be simultaneous

    a, b = Note(60, dur=2), Note(61, start=2, dur=1)
    h = Group((a, b))

    """

    def __init__(self, items:List[MusicObj], start=0., label:str=None):
        assert isinstance(items, (list, tuple))
        MusicObj.__init__(self, label=label, start=start)
        assert all(item.start is not None and item.dur is not None
                   for item in items)
        self.items: List[MusicObj] = []
        self.items.extend(items)
        self.items.sort(key=lambda item: item.start)
        self.dur = self.end - items[0].start

    @property
    def end(self) -> Fraction:
        return max(item.end for item in self.items)

    def append(self, obj:MusicObj) -> None:
        self.items.append(obj)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iter[MusicObj]:
        return iter(self.items)


    def __getitem__(self, idx) -> U[MusicObj, List[MusicObj]]:
        return self.items[idx]

    def __repr__(self):
        objstr = self.items.__repr__()
        return f"Group({objstr})"

    def __hash__(self):
        hashes = [hash(obj) for obj in self.items]
        return hash(tuple(hashes))

    def rec(self, outfile:str=None, sr:int=None, **kws) -> str:
        return recMany(self.items, outfile=outfile, sr=sr, **kws)

    #def events(self, **kws) -> List[CsoundEvent]:
    #    delay = kws.get('delay', 0)
    #    kws['delay'] = delay + self.start
    #    return getEvents(self.items, **kws)

    def csoundEvents(self, playargs: PlayArgs) -> List[CsoundEvent]:
        return sum((obj.csoundEvents(playargs) for obj in self.items), [])

    def quantizePitch(self, step=1.0) -> Group:
        return Group([obj.quantizePitch(step=step) for obj in self])

    def transpose(self, step) -> Group:
        return Group([obj.transpose(step) for obj in self])

    def scoringEvents(self) -> List[scoring.Notation]:
        events = sum((obj.scoringEvents() for obj in self.items), [])
        if self.start != 0:
            events = [ev.clone(offset=ev.offset+self.start)
                      for ev in events]
        return events

    def scoringParts(self) -> List[scoring.Part]:
        events = self.scoringEvents()
        return scoring.packInParts(events)


def asMidi(obj: U[float, int, str, Note]) -> float:
    """
    Convert obj to a midi note number

    Args:
        obj: a Note, string representation of a pitch or a midinote itself

    Returns:
        The midinote corresponding to obj
    """
    if isinstance(obj, Note):
        return obj.midi
    elif isinstance(obj, str):
        return Note(obj).midi
    elif isinstance(obj, (int, float)):
        return obj
    else:
        raise TypeError(f"Expected a Note, midinote (float) or str, "
                        f"got {obj} ({type(obj)})")


def playEvents(events: List[CsoundEvent]) -> csoundengine.SynthGroup:
    """
    Play a list of events

    Args:
        events: a list of CsoundEvents

    Returns:
        A SynthGroup

    Example:u
        a = Chord("C4 E4 G4", dur=2)
        b = Note("1000hz", dur=4, start=1)
        events = events((a, b))
        playEvents(events)

    """
    synths = []
    for ev in events:
        csdinstr = play.makeInstrFromPreset(ev.instr)
        args = ev.getArgs()
        synth = csdinstr.play(delay=args[0],
                              dur=args[1],
                              args=args[3:],
                              tabargs=ev.args,
                              priority=ev.priority)
        synths.append(synth)
    return csoundengine.SynthGroup(synths)


def getEvents(objs, **kws) -> List[CsoundEvent]:
    """
    Collect events of multiple objects using the same parameters

    Args:
        objs: a seq. of objects
        **kws: keywords passed to play

    Returns:
        a list of the events
    """
    return sum((obj.items(**kws) for obj in objs), [])


def playMany(objs, **kws) -> csoundengine.SynthGroup:
    """
    Play multiple objects with the same parameters

    Args:
        objs: the objects to play
        kws: any keywords passed to play

    """
    return playEvents(getEvents(objs, **kws))


def recMany(objs: List[MusicObj], outfile:str=None, sr:int=None, **kws
            ) -> str:
    """
    Record many objects with the same parameters
    kws: any keywords passed to rec
    """
    allevents = getEvents(objs, **kws)
    return play.recEvents(outfile=outfile, events=allevents, sr=sr)


def trill(note1: U[Note, Chord], note2: U[Note, Chord],
          totaldur: time_t, notedur:time_t=None) -> EventSeq:
    """
    Create a trill

    Args:
        note1: the first note of the trill (can also be a chord)
        note2: the second note of the trill (can also  be a chord)
        totaldur: total duration of the trill
        notedur: duration of each note. This value will only be used
            if the trill notes have an unset duration

    Returns:
        A realisation of the trill as an EventSeq of at least the
        given totaldur (can be longer if totaldur is not a multiple
        of notedur)
    """
    note1 = asChord(note1)
    note2 = asChord(note2)
    note1 = note1.clone(dur=note1.dur or notedur or F(1, 8))
    note2 = note2.clone(dur=note2.dur or notedur or F(1, 8))
    seq = EventSeq([note1, note2])
    return seq.cycle(totaldur)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if environment.insideJupyter:
    setJupyterHook()


