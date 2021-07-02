from __future__ import annotations
import functools
import os
import copy as _copy
import tempfile as _tempfile
from dataclasses import dataclass
import time

import configdict
import music21 as m21

from emlib.misc import firstval
import emlib.misc
import emlib.img
from PIL import Image as _Image

from pitchtools import m2f, m2n, r2i, f2m, amp2db, db2amp, str2midi
import maelzel.music.m21tools as m21tools
from maelzel import scoring

import csoundengine

from ._common import *
from .workspace import currentWorkspace, currentConfig, currentScoreStructure
from . import play
from . import tools
from . import environment
from . import symbols
from . import notation
from .csoundevent import PlayArgs, CsoundEvent

from maelzel.scorestruct import ScoreStructure


_playkeys = PlayArgs.keys()


__all__ = ('cloneObj',
           'renderObject',
           'MusicObj')


def cloneObj(obj: T, **kws) -> T:
    out = _copy.copy(obj)
    for k, v in kws.items():
        setattr(out, k, v)
    return out


@dataclass
class _TimeScale:
    factor: Rat
    offset: Rat

    def __call__(self, t: num_t):
        r = asRat(t)
        return r*self.factor + self.offset


class MusicObj:
    """
    This is the base class for all core objects. A MusicObj can display
    itself via :meth:`show` and play itself via :meth:`play`. It can
    have a duration and a start time.

    A MusicObj has also attributes which are for playback only: :attr:`_playargs`.
    These can be set via XXX

    Args:
        dur: the (optional) duration of this object, in abstract units (beats)
        start: the (optional) time offset of this object, in abstract units (beats)
        label: a string label to identify this object, if necessary
    """
    _showableInitialized = False
    _acceptsNoteAttachedSymbols = True

    __slots__ = ('dur', 'start', 'label', '_playargs', '_hash', '_symbols')

    def __init__(self, dur: time_t = None, start: time_t = None, label: str = ''):

        self.label: Opt[str] = label
        "a label can be used to identify an object within a group of objects"

        # A MusicObj can have a duration. A duration can't be 0
        # A duration of -1 means max. duration.
        if dur is not None:
            if dur == -1:
                dur = Rat(MAXDUR)
            else:
                assert dur > 0
        self.dur: Opt[Rat] = asRat(dur) if dur is not None else None

        self.start: Opt[Rat] = asRat(start) if start is not None else None
        "start specifies a time offset for this object"

        # _playargs are set via .setplay and serve the purpose of
        # attaching playing parameters (like position, instrument)
        # to an object
        self._playargs: Opt[PlayArgs] = None

        # All MusicObjs should be hashable. For the cases where
        # calculating the hash is expensive, we cache that here
        self._hash: int = 0

        self._symbols: Opt[List[symbols.Symbol]] = None

    @property
    def symbols(self) -> List[symbols.Symbol]:
        if self._symbols is None:
            return []
        return self._symbols

    def resolvedDuration(self) -> Rat:
        """
        The explicit duration or a default duration

        If this object has an explicitely set duration, return
        that, otherwise returns a default duration. Child
        classes can override this method to match their behaviour
        """
        return self.dur if self.dur is not None else currentConfig()['defaultDuration']

    def withExplicitTime(self, dur: time_t = None, start: time_t = None):
        """
        Copy of self with start and dur set to explicit values

        Args:
            dur: a duration to fill or override self.dur. If no
                duration is given and this object has no explicit
                duration, a default duration is used
            start: a start time to fill or override self.start

        Returns:
            a clone of self with dur and start set to explicit
            values

        """
        if self.dur is not None and self.start is not None:
            return self
        dur = asRat(dur) if dur is not None else self.resolvedDuration()
        start = asRat(firstval(start, self.start, Rat(0)))
        return self.clone(dur=dur, start=start)

    @property
    def playargs(self) -> PlayArgs:
        if self._playargs is None:
            self._playargs = PlayArgs()
        return self._playargs

    def setplay(self:T, /, **kws) -> T:
        """
        Set any playback attributes, returns self

        Args:
            **kws: any argument passed to .play (delay, dur, chan,
                gain, fade, instr, pitchinterpol, fadeshape, args,
                priority, position).

        Returns:
            self. This allows to chain this to any constructor (see example)

        Example::

            # a piano note
            >>> note = Note("C4+25", dur=0.5).setplay(instr="piano")
        """
        for k, v in kws.items():
            if k not in _playkeys:
                raise KeyError(f"key {k} not known. "
                               f"Possible keys are {_playkeys}")
            setattr(self.playargs, k, v)
        return self

    def clone(self: T, **kws) -> T:
        """
        Clone this object, changing parameters if needed

        Args:
            **kws: any keywords passed to the constructor

        Returns:
            a clone of this objects, with the given arguments
            changed

        Example::

            >>> a = Note("C4+", dur=1)
            >>> b = a.clone(dur=0.5)
        """
        out = self.copy()
        for k, v in kws.items():
            setattr(out, k, v)
        return out

    def copy(self):
        """Returns a copy of this object"""
        return _copy.deepcopy(self)

    def timeShift(self:T, timeoffset: time_t) -> T:
        """
        Return a copy of this object with an added time offset
        """
        start = self.start or Rat(0)
        return self.clone(start=timeoffset + start)

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return hash(self) == hash(other)
        else:
            return False

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __rshift__(self, timeoffset: time_t):
        return self.timeShift(timeoffset)

    def __lshift__(self, timeoffset: time_t):
        return self.timeShift(-timeoffset)

    @property
    def end(self) -> Opt[Rat]:
        """ The end time of this object. Will be None if
        this object has no duration or no start"""
        if self.dur is None or self.start is None:
            return None
        return self.start + self.dur

    def quantizePitch(self: T, step=0.) -> T:
        """ Returns a new object, with pitch rounded to step """
        raise NotImplementedError()

    def transpose(self: T, step: float) -> T:
        """ Transpose self by `step` """
        raise NotImplementedError()

    def freqratio(self: T, ratio: float) -> T:
        """ Transpose this by a given freq. ratio. A ratio of 2 equals
        to transposing an octave higher. """
        return self.transpose(r2i(ratio))

    def show(self, external: bool = None, method: str = None, fmt: str = None) -> None:
        """
        Show this as notation.

        Args:
            external: True to force opening the image in an external image viewer,
                even when inside a jupyter notebook. If False, show will
                display the image inline if inside a notebook environment.
                To change the default, modify ``config['show.external']``
            method: one of 'lilypond', 'musicxml'. None to use default
                (see ``config['show.method']``)
            fmt: one of 'png', 'pdf'. None to use default

        **NB**: to use the music21 show capabilities, use ``note.asmusic21().show(...)``
        """
        cfg = currentConfig()
        if external is None:
            external = cfg['show.external']
        if method is None:
            method = cfg['show.method']
        if fmt is None:
            fmt = 'png' if environment.insideJupyter else cfg['show.format']
        img = self.makeImage(method=method, fmt=fmt, opaque=True)
        if fmt == 'png':
            tools.pngShow(img, forceExternal=external)
        else:
            emlib.misc.open_with_standard_app(img)

    def _changed(self) -> None:
        """
        This method is called whenever the object changes its representation
        (a note changes its pitch inplace, the duration is modified, etc)
        This invalidates, among other things, the image cache for this
        object
        """
        self._hash = 0

    def makeImage(self, method: str = None, outfile: str = None, fmt="png",
                  opaque=True) -> str:
        """
        Creates an image representation, returns the path to the image

        Args:
            method: the rendering backend. One of 'musicxml', 'lilypond'
                None uses the default method (see config['show.method'])
            outfile: the path of the generated file. Use None to generate
                a temporary file.
            fmt: if outfile is None, fmt will determine the format of the
                generated file. Possible values: 'png', 'pdf'.
            opaque: if True, make sure that the background is not transparent

        Returns:
            the path of the generated file. If outfile was given, the returned
            path will be the same as the outfile.
        """
        # In order to be able to cache the images we put this
        # functionality outside of the class and use lru_cache
        if currentConfig()['show.cacheImages']:
            return renderObject(self, method=method, outfile=outfile, fmt=fmt, opaque=opaque)
        return _renderObject(self, method=method, outfile=outfile, fmt=fmt, opaque=opaque)

    def ipythonImage(self):
        """
        Generate a jupyter image from this object

        To be used within a jupyter notebook.

        Returns:
            an IPython.core.display.Image

        """
        from IPython.core.display import Image
        return Image(self.makeImage(fmt='png', opaque=True), embed=True)

    def scoringEvents(self, groupid:str=None) -> List[scoring.Notation]:
        """
        Returns its notated form as scoring.Notations

        These can then be converted into concrete notation via
        musicxml or lilypond

        Args:
            groupid: passed by an object higher in the hierarchy to
                mark this objects as belonging to a group

        Returns:
            A list of scoring.Notation which best represent this
            object as notation
        """
        raise NotImplementedError("Subclass should implement this")

    def scoringParts(self) -> List[scoring.Part]:
        """
        Returns this object as a list of scoring Parts.
        """
        notations = self.scoringEvents()
        scoring.stackNotationsInPlace(notations)
        parts = scoring.distributeNotationsByClef(notations)
        return parts

    def _scoringAnnotation(self) -> Opt[scoring.Annotation]:
        """ Returns owns annotations as a scoring Annotation """
        if not self.label:
            return None
        return scoring.Annotation(self.label,
                                  fontSize=currentConfig()['show.labelFontSize'])

    def asmusic21(self, **kws) -> m21.stream.Stream:
        """
        Used within .show, to convert this object into music21.

        When using the musicxml backend we first convert our object/s into
        music21 and use the music21 framework to generate an image

        Args:

            **kws: not used here, but classes inheriting from
                this may want to add customization

        Returns:
            a music21 stream which best represent this object as
            notation.

        .. note::

            The music21 representation should be final, not thought to be embedded into
            another stream. For embedding we use an abstract representation of scoring
            objects which can be queried via .scoringEvents
        """
        parts = self.scoringParts()
        renderer = notation.renderWithCurrentConfig(parts, backend='musicxml')
        return renderer.asMusic21()

    def musicxml(self) -> str:
        """
        Return the music representation of this object as musicxml.

        A subclass can override this method to provide a way of
        outputting musicxml which bypasses music21
        """
        m = self.asmusic21()
        if currentConfig()['m21.fixStream']:
            m21tools.fixStream(m)
        return m21tools.getXml(m)

    def write(self, outfile: str, backend: str = None) -> None:
        """
        Save the notation representation of self

        Formats supported: musicxml, lilypond, pdf, png

        Args:
            outfile: the path of the output file. The extension determines
                the format
            backend: the backend used when writing as pdf or png.
                If not given, the default defined in the current
                configuration is used (key: 'show.method')
        """
        ext = os.path.splitext(outfile)[1]
        if ext == '.ly':
            backend = 'lilypond'
        elif ext == '.xml' or ext == '.musicxml':
            backend = 'musicxml'
        elif backend is None:
            cfg = currentConfig()
            backend = cfg['show.method']
        r = notation.renderWithCurrentConfig(self.scoringParts(), backend=backend)
        r.write(outfile)

    def _repr_png_(self):
        imgpath = self.makeImage(opaque=True)
        im = _Image.open(imgpath)
        scaleFactor = currentConfig().get('show.scaleFactor', 1.0)
        if scaleFactor == 1:
            return im._repr_png_()
        width, height = im.size
        return im._repr_png_(), {'height': height*scaleFactor, 'width': width*scaleFactor}

    @classmethod
    def setJupyterHook(cls) -> None:
        """
        Sets the jupyter display hook for this class

        """
        if cls._showableInitialized:
            return
        from IPython.core.display import Image

        def reprpng(obj):
            imgpath = obj.makeImage(opaque=True)

            scaleFactor = currentConfig().get('show.scaleFactor', 1.0)
            if scaleFactor != 1.0:
                imgwidth, imgheight = tools.imgSize(imgpath)
                width = imgwidth * scaleFactor
            else:
                width = None
            return Image(filename=imgpath, embed=True, width=width)._repr_png_()

        tools.setJupyterHookForClass(cls, reprpng, fmt='image/png')

    def csoundEvents(self, playargs: PlayArgs, scorestruct: ScoreStructure, conf: dict
                     ) -> List[CsoundEvent]:
        """
        Must be overriden by each class to generate CsoundEvents

        Args:
            playargs: a :class:`PlayArgs`, structure, filled with given values,
                own .playargs values and config defaults (in that order)
            scorestruct: the 'class:`ScoreStructure` used to translate beat-time
                into real-time
            conf: the current config, to fill default values

        Returns:
            a list of :class:`CsoundEvent`s
        """
        raise NotImplementedError("Subclass should implement this")

    def events(self, scorestruct: ScoreStructure=None, config:dict=None, **kws
               ) -> List[CsoundEvent]:
        """
        Returns the CsoundEvents needed to play this object

        An object always has a start time. It can be unset (None), which defaults to 0
        but can also mean unset for contexts where this is meaningful (a sequence of Notes,
        for example, where they are concatenated one after the other, the start time
        is the end of the previous Note)

        All these attributes here can be set previously via .playargs (or
        using .setplay)

        Args:
            scorestruct: the :class:`ScoreStructure` used to map beat-time to
                real-time. If not given the current/default :class:`ScoreStructure`
                is used.

        Keywords:

        - delay: A delay, if defined, is added to the start time.
        - chan: the chan to play (or rec) this object
        - gain: gain modifies .amp
        - fade: fadetime or (fadein, fadeout)
        - instr: the name of the instrument
        - pitchinterpol: 'linear' or 'cos'
        - fadeshape: 'linear' or 'cos'
        - position: the panning position (0=left, 1=right). The left channel
            is determined by chan
        - args: any args needed to pass to the instrument

        Returns:
            A list of :class:`CsoundEvent`s

        Example::

            >>> n = Note(60, dur=1).setplay('instr=piano')
            >>> n.events(gain=0.5)
            [CsoundEvent(delay=0.000, gain=0.5, chan=1, fade=(0.02, 0.02), instr=piano)
             bps 0.000s:  60, 1.000000
                 1.000s:  60, 1.000000]

        """
        playargs = PlayArgs(**kws)
        if scorestruct is None:
            scorestruct = currentScoreStructure()
        events = self.csoundEvents(playargs, scorestruct, config or currentConfig())
        return events

    def play(self,
             instr: str = None,
             delay: float = None,
             args: Dict[str, float] = None,
             gain: float = None,
             chan: int = None,
             pitchinterpol: str = None,
             fade: U[float, Tuple[float, float]] = None,
             fadeshape: str = None,
             position: float = None,
             scorestruct: ScoreStructure = None) -> csoundengine.synth.AbstrSynth:
        """
        Plays this object.

        Play is always asynchronous (to block, use some sleep funcion).
        By default, :meth:`play` schedules this event to be renderer in realtime.

        .. note::
            To record events offline, see the example below

        Args:
            gain: modifies the own amplitude for playback/recording (0-1)
            delay: delay in seconds, added to the start of the object
                As opposed to the .start attribute of each object, which is defined
                in symbolic (beat) time, the delay is always in real (seconds) time
            instr: which instrument to use (see defInstrPreset, availableInstrPresets)
            chan: the channel to output to. **Channels start at 1**
            pitchinterpol: 'linear', 'cos', 'freqlinear', 'freqcos'
            fade: fade duration in seconds, can be a tuple (fadein, fadeout)
            fadeshape: 'linear' | 'cos'
            args: paramaters passed to the note through an associated table.
                A dict paramName: value
            position: the panning position (0=left, 1=right)
            scorestruct: a ScoreStructure to determine the mapping between
                beat-time and real-time. If no scorestruct is given the current/default
                scorestruct is used (see ``pushState``)

        Returns:
            A :class:`~csoundengine.synth.SynthGroup`

        Example::

            >>> from maelzel.core import *
            >>> # play a note
            >>> note = Note(60).play(gain=0.1, chan=2)

            >>> # record offline
            >>> with play.rendering(sr=44100, outfile="out.wav"):
                    Note(60, 5).play(gain=0.1, chan=2)
                    # ... other objects.play(...)
        """
        events = self.events(delay=delay, chan=chan,
                             fade=fade, gain=gain, instr=instr,
                             pitchinterpol=pitchinterpol, fadeshape=fadeshape,
                             args=args, position=position,
                             scorestruct=scorestruct)
        renderer = currentWorkspace().renderer
        if renderer:
            # schedule offline
            return renderer.schedMany(events)
        return play.playEvents(events)

    def rec(self, outfile: str = None, sr: int = None, **kws) -> str:
        """
        Record the output of .play as a soundfile

        Args:
            outfile: the outfile where sound will be recorded. Can be
                None, in which case a filename will be generated
            sr: the sampling rate
            **kws: any keyword passed to .play

        Returns:
            the path of the generated soundfile
        """
        events = self.events(**kws)
        return play.recEvents(events, outfile, sr=sr)

    def isRest(self) -> bool:
        """
        Is this object a Rest?

        Rests are used as separators between objects inside an EventSeq or a Track
        """
        return False

    def setSymbol(self, symbol: U[symbols.Symbol, str], value=None):
        """
        Set a notation symbol in this object

        Either `n.setSymbol(symbols.Dynamic('ff')` or n.setSymbol('Dynamic', 'ff')

        Args:
            symbol ():
            value ():

        Returns:

        """
        # n.setSymbol('Dynamic', 'mf')
        if isinstance(symbol, str):
            symbol = symbols.construct(symbol, value)

        if isinstance(symbol, symbols.NoteAttachedSymbol) and not self._acceptsNoteAttachedSymbols:
            raise ValueError(f"A {type(self)} does not accept note attached symbols")
        if not self._symbols:
            self._symbols = []
            self._symbols.append(symbol)
        else:
            if symbol.exclusive:
                self._symbols = [s for s in self._symbols if type(s) != type(symbol)]
                self._symbols.append(symbol)

    def timeTransform(self:T, timemap: Callable[[float], float]) -> T:
        start = 0. if self.start is None else self.start
        dur = currentConfig()['defaultDur'] if self.dur is None else self.dur
        start2 = timemap(start)
        dur2 = timemap(start+dur) - start2
        return self.clone(start=asRat(start2), dur=asRat(dur2))

    def timeScale(self:T, factor: num_t, offset: num_t = 0) -> T:
        transform = _TimeScale(asRat(factor), offset=asRat(offset))
        return self.timeTransform(transform)


def _renderObject(obj: MusicObj, outfile:str=None, method:str=None, fmt='png',
                  opaque=False
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
        opaque: if True makes sure that the image does not have a transparent
            background

    Returns:
        the path of the generated image
    """
    config = currentConfig()
    if method is None:
        method = config['show.method']
    else:
        methods = {'musicxml', 'lilypond'}
        if method not in methods:
            raise ValueError(f"method {method} not supported. Should be one of {methods}")

    if outfile is None:
        outfile = _tempfile.mktemp(suffix="." + fmt)

    logger.debug(f"rendering parts with backend: {method}")
    parts = obj.scoringParts()
    renderer = notation.renderWithCurrentConfig(parts, backend=method)
    renderer.write(outfile)
    if opaque and method == 'musicxml' and os.path.splitext(outfile)[1].lower() == '.png':
        emlib.img.pngRemoveTransparency(outfile)
    return outfile


@functools.lru_cache(maxsize=1000)
def renderObject(obj:MusicObj, outfile:str=None, method:str=None, fmt='png',
                 opaque=False
                 ) -> str:
    """
    Given a music object, make an image representation of it.

    NB: we put it here in order to make it easier to cache images

    Args:
        obj     : the object to make the image from (a Note, Chord, etc.)
        outfile : the path to be generated. A .png filename
        method  : format used. One of 'musicxml', 'lilypond'
        fmt: the format of the generated object. One of 'png', 'pdf'
        opaque: if True, force the background to white when rendering to png

    Returns:
        the path of the generated image

    NB: we put it here in order to make it easier to cache images
    """
    return _renderObject(obj=obj, outfile=outfile, method=method, fmt=fmt, opaque=opaque)
