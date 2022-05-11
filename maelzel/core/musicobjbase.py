"""
Musical Objects
---------------

Time
~~~~

A MusicObj has always a start and dur attribute. They refer to an abstract time.
When visualizing a MusicObj as musical notation these times are interpreted/converted
to beats and score locations based on a score structure.

Score Structure
~~~~~~~~~~~~~~~

A minimal score structure is a default time-signature (4/4) and a default tempo (60). If
the user does not set a different score structure, an endless score with these default
values will always be used.

"""

from __future__ import annotations
import functools
import os
import copy as _copy
import tempfile as _tempfile
import html as _html
from dataclasses import dataclass

import configdict
import music21 as m21

from emlib.misc import firstval
import emlib.misc
import emlib.img

import pitchtools as pt

import csoundengine

from maelzel.common import asmidi
from ._common import *
from ._typedefs import *
from .workspace import getWorkspace, getConfig, getScoreStruct
from . import play
from . import tools
from . import environment
from . import symbols
from . import notation
import maelzel.music.m21tools as m21tools
from maelzel import scoring

from .csoundevent import PlayArgs, CsoundEvent, cropEvents
from maelzel.scorestruct import ScoreStruct

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *
    from ._typedefs import time_t
    from .play import OfflineRenderer
    _T = TypeVar('_T', bound='MusicObj')

_playkeys = PlayArgs.keys()


__all__ = ('MusicObj',
           'resetImageCache'
           )


@dataclass
class _TimeScale:
    factor: Rat
    offset: Rat

    def __call__(self, t: num_t):
        r = asRat(t)
        return r*self.factor + self.offset


class MusicObj:
    """
    This is the base class for all core objects.

    A MusicObj can display itself via :meth:`show` and play itself via :meth:`play`.
    It can have a duration and a start time.

    **It should not be instantiated by itself**

    A MusicObj has attributes which are for playback only. They can
    be set via :meth:`MusicObj.setPlay` and accessed via the `playargs` property

    A MusicObj can customize its notation via symbols. A symbol is an attribute or
    notation element (like color, size or an attached text expression) which has
    meaning only in the realm of graphical representation. See :meth:`MusicObj.setSymbol`

    Args:
        dur: the (optional) duration of this object, in abstract units (beats)
        start: the (optional) time offset of this object, in abstract units (beats)
        label: a string label to identify this object, if necessary
    """
    _showableInitialized = False
    _acceptsNoteAttachedSymbols = True

    __slots__ = ('dur', 'start', 'label', '_playargs', '_hash', '_symbols')

    def __init__(self, dur: time_t = None, start: time_t = None, label: str = ''):

        self.label: Optional[str] = label
        "a label can be used to identify an object within a group of objects"

        # A MusicObj can have a duration. A duration can't be 0
        # A duration of -1 means max. duration.
        if dur is not None:
            if dur == -1:
                dur = Rat(MAXDUR)
            else:
                assert dur > 0
        self.dur: Optional[Rat] = asRat(dur) if dur is not None else None
        "the duration of this object (can be None, in which case it is unset)"

        self.start: Optional[Rat] = asRat(start) if start is not None else None
        "start specifies a time offset for this object"

        # _playargs are set via .setplay and serve the purpose of
        # attaching playing parameters (like pan position, instrument)
        # to an object
        self._playargs: Optional[PlayArgs] = None

        # All MusicObjs should be hashable. For the cases where
        # calculating the hash is expensive, we cache that here
        self._hash: int = 0

        self._symbols: Optional[List[symbols.Symbol]] = None

    @property
    def symbols(self) -> List[symbols.Symbol]:
        """
        A list of symbols to determine any notation aspect of this object

        Notation elements (Symbols) are those which only make sense in terms
        of notation but have no real correlation regarding computation
        (an accent, or a text-expression, are Symbols)

        Returns:
            a list of symbols
        """
        if self._symbols is None:
            return []
        return self._symbols

    def pitchRange(self) -> Optional[Tuple[float, float]]:
        """
        The pitch range of this object, if applicable

        This is useful to assign a proper Voice when distributing
        objects among voices

        Returns:
            either None or a tuple (lowest pitch, highest pitch)
        """
        return None

    def resolvedDuration(self) -> Rat:
        """
        The explicit duration or a default duration

        If this object has an explicitely set duration, return
        that, otherwise returns a default duration. Child
        classes can override this method to match their behaviour
        """
        return self.dur if self.dur is not None else getConfig()['defaultDuration']

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
        """
        A PlayArgs structure, containing any specification regarding playback

        This is a read-only property. To modify any playback attribute, use
        :meth:`~MusicObj.setplay`
        """
        if self._playargs is None:
            self._playargs = PlayArgs()
        return self._playargs

    def setPlay(self:_T, /, **kws) -> _T:
        """
        Set any playback attributes, returns self

        Args:
            **kws: any argument passed to .play (delay, dur, chan,
                gain, fade, instr, pitchinterpol, fadeshape, params,
                priority, position).

        Returns:
            self. This allows to chain this to any constructor (see example)

        Example::

            # a piano note
            >>> from maelzel.core import *
            >>> note = Note("C4+25", dur=0.5).setPlay(instr="piano")
        """
        playargs = self.playargs
        for k, v in kws.items():
            if k not in _playkeys:
                raise KeyError(f"key {k} not known. "
                               f"Possible keys are {_playkeys}")
            setattr(playargs, k, v)
        assert self._playargs is not None
        return self

    def clone(self: _T, **kws) -> _T:
        """
        Clone this object, changing parameters if needed

        Args:
            **kws: any keywords passed to the constructor

        Returns:
            a clone of this objects, with the given arguments
            changed

        Example::

            >>> from maelzel.core import *
            >>> a = Note("C4+", dur=1)
            >>> b = a.clone(dur=0.5)
        """
        out = self.copy()
        for k, v in kws.items():
            setattr(out, k, v)
        if self._playargs is not None:
            out._playargs = self._playargs.copy()
        out._changed()
        return out

    def copy(self: _T) -> _T:
        """Returns a copy of this object"""
        return _copy.deepcopy(self)

    def moveTo(self, start: time_t) -> None:
        """Move this to the given start time (in place)"""
        self.start = start
        self._changed()

    def timeShift(self:_T, timeoffset: time_t) -> _T:
        """
        Return a copy of this object with an added time offset
        """
        return self.timeTransform(lambda t: t+timeoffset)

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
    def end(self) -> Optional[Rat]:
        """ The end time of this object. Will be None if
        this object has no duration or no start"""
        if self.dur is None or self.start is None:
            return None
        return self.start + self.dur

    def quantizePitch(self: _T, step=0.) -> _T:
        """ Returns a new object, with pitch rounded to step """
        raise NotImplementedError()

    def transposeByRatio(self: _T, ratio: float) -> _T:
        """
        Transpose this by a given freq. ratio.

        A ratio of 2 equals to transposing an octave higher. """
        return self.transpose(pt.r2i(ratio))

    def show(self, fmt: str = None, external: bool = None, backend: str = None,
             scorestruct: ScoreStruct = None,
             config: configdict.ConfigDict = None
             ) -> None:
        """
        Show this as notation.

        Args:
            external: True to force opening the image in an external image viewer,
                even when inside a jupyter notebook. If False, show will
                display the image inline if inside a notebook environment.
                To change the default, modify ``config['show.external']``
            backend: backend used when rendering to png/pdf.
                One of 'lilypond', 'music21'. None to use default
                (see ``config['show.backend']``)
            fmt: one of 'png', 'pdf', 'ly'. None to use default

        **NB**: to use the music21 show capabilities, use ``note.asmusic21().show(...)``
        """
        cfg = config or getConfig()
        if external is None:
            external = cfg['show.external']
        if backend is None:
            backend = cfg['show.backend']
        if fmt is None:
            fmt = 'png' if not external and environment.insideJupyter else cfg['show.format']
        if fmt == 'ly':
            r = self.render(backend='lilypond', scorestruct=scorestruct)
            if external:
                lyfile = _tempfile.mktemp(suffix=".ly")
                r.write(lyfile)
                emlib.misc.open_with_app(lyfile)
            else:
                tools.showLilypondScore(r.nativeScore())
        else:
            img = self.renderImage(backend=backend, fmt=fmt, scorestruct=scorestruct, config=cfg)
            if fmt == 'png':
                tools.pngShow(img, forceExternal=external)
            else:
                emlib.misc.open_with_app(img)

    def _changed(self) -> None:
        """
        This method is called whenever the object changes its representation

        This happens when a note changes its pitch inplace, the duration is modified, etc.
        This invalidates, among other things, the image cache for this object
        """
        self._hash = 0

    def render(self, backend: str = None,
               renderoptions: scoring.render.RenderOptions = None,
               scorestruct: ScoreStruct = None,
               config: configdict.ConfigDict = None) -> scoring.render.Renderer:
        """
        Renders this object as a quantized score

        Args:
            backend: the backend to use, one of 'lilypond', 'music21'. If not given, defaults
                to the active config's ``'show.backend'`` value
            renderoptions: the render options to use. If not given, these are generated from
                the active config
            scorestruct: if given, overrides the scorestruct set within the active Workspace
                and any scorestruct attached to this object
            config: if given, overrides the active config

        Returns:
            a scoring.render.Renderer. This can be used to write the rendered structure
            to an image (png, pdf) or as a musicxml or lilypond file.
        """
        if config is None:
            config = getConfig()
        if not backend:
            backend = config['show.backend']
        if not renderoptions:
            renderoptions = notation.makeRenderOptionsFromConfig(config)
        if not scorestruct:
            scorestruct = self.attachedScoreStruct() or getScoreStruct()
        return _renderObject(self, backend=backend, renderoptions=renderoptions,
                             scorestruct=scorestruct, config=config)

    def renderImage(self, backend: str = None, outfile: str = None, fmt="png",
                    scorestruct: ScoreStruct = None,
                    config: configdict.ConfigDict = None
                    ) -> str:
        """
        Creates an image representation, returns the path to the image

        Args:
            backend: the rendering backend. One of 'music21', 'lilypond'
                None uses the default method (see config['show.backend'])
            outfile: the path of the generated file. Use None to generate
                a temporary file.
            fmt: if outfile is None, fmt will determine the format of the
                generated file. Possible values: 'png', 'pdf'.
            scorestruct: if given will override the active ScoreStruct

        Returns:
            the path of the generated file. If outfile was given, the returned
            path will be the same as the outfile.

        .. seealso:: :meth:`MusicObj.render`
        """
        w = getWorkspace()
        if not config:
            config = w.config
        if backend is None:
            backend = config['show.backend']
        if fmt == 'ly':
            backend = 'lilypond'
        if scorestruct is None:
            scorestruct = self.attachedScoreStruct() or w.scorestruct
        path = _renderImage(self, outfile, fmt=fmt, backend=backend, scorestruct=scorestruct,
                            config=config or getConfig())
        if not os.path.exists(path):
            # cached image does not exist?
            resetImageCache()
            raise RuntimeError("The returned image file does not exist")
        return path


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

    def scoringParts(self, options: scoring.render.RenderOptions=None
                     ) -> List[scoring.Part]:
        """
        Returns this object as a list of scoring Parts.

        This method is used internally to generate the parts which
        constitute a given MusicObj prior to rendering,
        but might be of use itself so it is exposed here.

        A scoring.Part is an intermediate format used by the scoring
        package to represent notated events. In particular, a scoring.Part
        is independent of any score structure and thus it is still not
        quantized/rendered.
        """
        notations = self.scoringEvents()
        scoring.stackNotationsInPlace(notations)
        scoring.enharmonics.fixEnharmonicsInPlace(notations)
        parts = scoring.distributeNotationsByClef(notations)
        return parts

    def scoringArrangement(self, title:str=None) -> scoring.Arrangement:
        """
        Create a notation Score from this object

        Args:
            title: the title of the resulting score (if given)
            scorestruct: a ScoreStruct to use for quantization. If None, use the
                active Workspace's scorestruct

        Returns:
            the Score representation of this object

        """
        parts = self.scoringParts()
        return scoring.Arrangement(parts, title=title)

    def _scoringAnnotation(self) -> Optional[scoring.Annotation]:
        """ Returns owns annotations as a scoring Annotation """
        if not self.label:
            return None
        return scoring.Annotation(self.label,
                                  fontSize=getConfig()['show.labelFontSize'])

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
        renderer = notation.renderWithCurrentWorkspace(parts, backend='music21',
                                                       scorestruct=self.attachedScoreStruct())
        stream = renderer.asMusic21()
        if getConfig()['m21.fixStream']:
            m21tools.m21fix.fixStream(stream, inPlace=True)
        return stream

    def musicxml(self) -> str:
        """
        Return the music representation of this object as musicxml.

        A subclass can override this method to provide a way of
        outputting musicxml which bypasses music21
        """
        stream = self.asmusic21()
        return m21tools.getXml(stream)

    def attachedScoreStruct(self) -> Optional[ScoreStruct]:
        """
        Returns the ScoreStruct attached to this obj, if Any
        """
        return None

    def write(self, outfile: str, backend: str = None,
              scorestruct: ScoreStruct = None) -> None:
        """
        Export to multiple formats

        Formats supported: pdf, png, musicxml (extension: .xml or .musicxml),
        lilypond (.ly), midi (.mid or .midi)

        Args:
            outfile: the path of the output file. The extension determines
                the format
            backend: the backend used when writing as pdf or png. If not given,
                the default defined in the active config is used (key: 'show.backend').
                Possible backends: ``music21`` (uses musescore to render musicxml as image),
                ``lilypond``
        """
        if outfile == '?':
            outfile = tools.selectFileForSave(key="writeLastDir",
                                              filter="All formats (*.pdf, *.png, "
                                                     "*.ly, *.xml, *.mid)")
            if not outfile:
                logger.info("File selection cancelled")
                return
        ext = os.path.splitext(outfile)[1]
        if ext == '.ly' or ext == '.mid' or ext == '.midi':
            backend = 'lilypond'
        elif ext == '.xml' or ext == '.musicxml':
            backend = 'music21'
        elif backend is None:
            cfg = getConfig()
            backend = cfg['show.backend']
        r = notation.renderWithCurrentWorkspace(self.scoringParts(), backend=backend,
                                                scorestruct=scorestruct or self.attachedScoreStruct())
        r.write(outfile)

    def _htmlImage(self) -> str:
        imgpath = self.renderImage()
        scaleFactor = getConfig().get('show.scaleFactor', 1.0)
        width, height = emlib.img.imgSize(imgpath)
        img = emlib.img.htmlImgBase64(imgpath,
                                      width=f'{int(width * scaleFactor)}px')
        return img

    def _repr_html_(self) -> str:
        img = self._htmlImage()
        txt = _html.escape(repr(self))
        return rf"<code>{txt}</code><br>" + img

    def dump(self, indents=0):
        """
        Prints all relevant information about this object
        """
        print(f'{"  "*indents}{repr(self)}')
        if self._playargs:
            print(f'{"  "*(indents+1)}{self.playargs}')

    def csoundEvents(self, playargs: PlayArgs, scorestruct: ScoreStruct, conf: dict,
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

    def events(self, scorestruct: ScoreStruct=None, config:dict=None, instr:str=None,
               **kws
               ) -> List[CsoundEvent]:
        """
        Returns the CsoundEvents needed to play this object

        An object always has a start time. It can be unset (None), which defaults to 0
        but can also mean unset for contexts where this is meaningful (a sequence of
        Notes, for example, where they are concatenated one after the other, the start
        time is the end of the previous Note)

        All these attributes here can be set previously via .playargs (or
        using .setPlay)

        Args:
            scorestruct: the :class:`ScoreStructure` used to map beat-time to
                real-time. If not given the current/default :class:`ScoreStructure`
                is used.
            config: the configuration used (see :func:`maelzel.core.workspace.newConfig`)
            instr: the instrument preset to use, '?' to select from o list or None to use
                the default

        Keywords arguments:

        - delay: A delay, if defined, is added to the start time.
        - chan: the chan to play (or rec) this object
        - gain: gain modifies .amp
        - fade: fadetime or (fadein, fadeout)
        - instr: the name of the instrument
        - pitchinterpol: 'linear' or 'cos'
        - fadeshape: 'linear' or 'cos'
        - position: the panning position (0=left, 1=right). The left channel
            is determined by chan
        - params: any params needed to pass to the instrument

        Returns:
            A list of :class:`CsoundEvent`s

        Example::

            >>> from maelzel.core import *
            >>> n = Note(60, dur=1).setPlay(instr='piano')
            >>> n.events(gain=0.5)
            [CsoundEvent(delay=0.000, gain=0.5, chan=1, fade=(0.02, 0.02), instr=piano)
             bps 0.000s:  60, 1.000000
                 1.000s:  60, 1.000000]

        """
        if instr == "?":
            from .presetman import presetManager
            instr = presetManager.selectPreset()
            if not instr:
                raise ValueError("No preset selected")
        playargs = PlayArgs(instr=instr, **kws)
        if scorestruct is None:
            scorestruct = self.attachedScoreStruct() or getScoreStruct()
        events = self.csoundEvents(playargs, scorestruct, config or getConfig())
        return events

    def play(self,
             instr: str = None,
             delay: float = None,
             params: Dict[str, float] = None,
             gain: float = None,
             chan: int = None,
             pitchinterpol: str = None,
             fade: Union[float, Tuple[float, float]] = None,
             fadeshape: str = None,
             position: float = None,
             scorestruct: ScoreStruct = None,
             start: float = None,
             end: float = None,
             whenfinished: Callable = None) -> csoundengine.synth.AbstrSynth:
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
            instr: which instrument to use (see defPreset, definedPresets). Use "?" to
                select from a list of defined presets.
            chan: the channel to output to. **Channels start at 1**
            pitchinterpol: 'linear', 'cos', 'freqlinear', 'freqcos'
            fade: fade duration in seconds, can be a tuple (fadein, fadeout)
            fadeshape: 'linear' | 'cos'
            params: paramaters passed to the note through an associated table.
                A dict paramName: value
            position: the panning position (0=left, 1=right)
            start: start time of playback. Allows to play a fragment of the object (NB: this trims the playback
                of the object. Use `delay` to offset the playback in time while keeping the playback time
                unmodified)
            end: end time of playback. Allows to play a fragment of the object by trimming the end of the playback
            scorestruct: a ScoreStructure to determine the mapping between
                beat-time and real-time. If no scorestruct is given the current/default
                scorestruct is used (see ``setScoreStructure``)

        Returns:
            A :class:`~csoundengine.synth.SynthGroup`

        Example::

            >>> from maelzel.core import *
            >>> # play a note
            >>> note = Note(60).play(gain=0.1, chan=2)

            >>> # record offline
            >>> with play.OfflineRenderer("out.wav", sr=44100) as r:
            ...     Note(60, 5).play(gain=0.1, chan=2)
            ...     # ... other objects.play(...)
            ...     # r.sched()
        """
        events = self.events(delay=delay,
                             chan=chan,
                             fade=fade,
                             gain=gain,
                             instr=instr,
                             pitchinterpol=pitchinterpol,
                             fadeshape=fadeshape,
                             params=params,
                             position=position,
                             scorestruct=scorestruct)
        if start is not None or end is not None:
            scorestruct = scorestruct or self.attachedScoreStruct() or getScoreStruct()
            starttime = None if start is None else scorestruct.beatToTime(start)
            endtime = None if end is None else scorestruct.beatToTime(end)
            events = cropEvents(events, start=starttime, end=endtime, rewind=True)
        if (renderer:=getWorkspace().renderer) is not None:
            # schedule offline
            for ev in events:
                renderer.schedEvent(ev)
        else:
            return play.playEvents(events, whenfinished=whenfinished)

    def rec(self, outfile: str = None, sr: int = None, quiet: bool = None,
            wait: bool = None, ksmps: int = None, nchnls: int = None,
            **kws
            ) -> str:
        """
        Record the output of .play as a soundfile

        Args:
            outfile: the outfile where sound will be recorded. Can be
                None, in which case a filename will be generated. Use '?'
                to open a save dialog
            sr: the sampling rate (config ``rec.sr``)
            ksmps: the ksmps used when recording (config ``rec.ksmps``)
            wait: if True, the operation blocks until recording is finishes
                (config ``rec.block``)
            nchnls: if given, use this as the number of channels to record.
            **kws: any keyword passed to .play

        Returns:
            the path of the generated soundfile

        See Also
        ~~~~~~~~

        :class:`maelzel.core.play.OfflineRenderer`
        """
        events = self.events(**kws)
        return play.recEvents(events, outfile, sr=sr, ksmps=ksmps, wait=wait, quiet=quiet,
                              nchnls=nchnls)

    def isRest(self) -> bool:
        """
        Is this object a Rest?

        Rests are used as separators between objects inside an Chain or a Track
        """
        return False

    def supportedSymbols(self) -> List[str]:
        """
        Returns a list of supported symbols for this object
        """
        out = []
        if self._acceptsNoteAttachedSymbols:
            out.extend(symbols.noteAttachedSymbols)
        return out

    def getSymbol(self, cls: str) -> Optional[symbols.Symbol]:
        """
        Get a symbol set for the given class

        Args:
            cls: the class name of the symbol (for example, "dynamic", "articulation",
                etc)

        Returns:
            the set symbol, or None

        """
        if not self._symbols:
            return None
        cls = cls.lower()
        for symbol in self._symbols:
            if type(symbol).__name__.lower() == cls:
                return symbol

    def setSymbol(self: _T, symbol: Union[str, symbols.Symbol], *args, **kws) -> _T:
        """
        Set a notation symbol in this object

        Notation symbols are any attributes of this MusicObj which are for
        notation purposes only. Such notation only attributes include dynamics,
        articulations, etc. Each class has a set of symbols which it accepts
        (see :meth:`MusicObj.supportedSymbols`)

        Example
        -------

            >>> from maelzel.core import *
            >>> n = Note(60)
            >>> n.setSymbol('dynamic', 'ff')
            >>> # setSymbol, like setPlay, returns self so it is possible to chain calls:
            >>> n = Note(60).setPlay(instr='piano').setSymbol('dynamic', 'ff')

        Args:
            symbol: the name of a symbol. See :meth:`MusicObj.supportedSymbols` for symbols
                accepted by this object. In general, possible symbols are 'dynamic',
                'articulation', 'notehead', 'expression', but not all objects accept
                all symbols.
            args, keys: passed directly to the class constructor

        Returns:
            self (similar to setPlay, allows to chain calls)

        ============  ==========================================================
        Symbol        Arguments
        ============  ==========================================================
        dynamic       kind: {pppp, ppp, pp, …}
        expression    text: expression text; placement: {above, 'below}
        notehead      kind: {cross, harmonic, triangleup, xcircle,
                      triangle, rhombus, square, rectangle};
                      color: a css color
                      parenthesis: bool
        articulation  kind: {accent, staccato, tenuto, marcato, staccatissimo}
        size          value: int (0=default, 1, 2, …=bigger, -1, -2, … = smaller)
        color         value: a css color
        ============  ==========================================================

        """
        if isinstance(symbol, symbols.Symbol):
            symboldef = symbol
        else:
            symboldef = symbols.construct(symbol, *args, **kws)

        if isinstance(symbol, symbols.NoteAttachedSymbol) \
                and not self._acceptsNoteAttachedSymbols:
            raise ValueError(f"A {type(self)} does not accept note attached symbols")
        if not self._symbols:
            self._symbols = []
            self._symbols.append(symboldef)
        else:
            cls = type(symboldef)
            if symboldef.exclusive and any(isinstance(s, cls) for s in self._symbols):
                self._symbols = [s for s in self._symbols if not isinstance(s, cls)]
            self._symbols.append(symboldef)
        return self

    def timeTransform(self:_T, timemap: Callable[[num_t], num_t]) -> _T:
        """
        Apply a time-transform to this object

        Args:
            timemap: a function mapping old time to new time

        Returns:
            the resulting object

        .. note::

            time is conceived as abstract 'beat' time, measured in quarter-notes.
            The actual time will be also determined by any tempo changes in the
            active score structure.
        """
        start = 0. if self.start is None else self.start
        dur = getConfig()['defaultDur'] if self.dur is None else self.dur
        start2 = timemap(start)
        dur2 = timemap(start+dur) - start2
        return self.clone(start=asRat(start2), dur=asRat(dur2))

    def timeShiftInPlace(self, timeoffset: time_t) -> None:
        """
        Shift the time of this by the given offset (in place)

        Args:
            timeoffset: the time delta
        """
        self.start = self.start + timeoffset
        self._changed()

    def startAbsTime(self) -> Rat:
        """
        Returns the .start of this in absolute time according to the active ScoreStruct

        An Exception is raised if self does not have an start time

        This is equivalent to ``activeScoreStruct().beatToTime(obj.start)``
        """
        if self.start is None:
            raise ValueError(f"The object {self} has no explicit .start")
        s = getScoreStruct()
        timefrac = s.beatToTime(self.start)
        return Rat(timefrac.numerator, timefrac.denominator)

    def endAbsTime(self) -> Rat:
        """
        Returns the .end of this in absolute time according to the active ScoreStruct

        An Exception is raised if self does not have an end time

        This is equivalent to ``activeScoreStruct().beatToTime(obj.end)``
        """
        if self.end is None:
            raise ValueError(f"The object {self} has no explicit .end")
        s = getScoreStruct()
        timefrac = s.beatToTime(self.end)
        return Rat(timefrac.numerator, timefrac.denominator)

    def pitchTransform(self:_T, pitchmap: Callable[[float], float]) -> _T:
        """
        Apply a pitch-transform to this object, returns a copy

        Args:
            pitchmap: a function mapping pitch to pitch

        Returns:
            the object after the transform
        """
        raise NotImplementedError("Subclass should implement this")

    def timeScale(self:_T, factor: num_t, offset: num_t = 0) -> _T:
        """
        Create a copy with modified timing by applying a linear transformation

        Args:
            factor: a factor which multiplies all durations and start times
            offset: an offset added to all start times

        Returns:
            the modified object
        """
        transform = _TimeScale(asRat(factor), offset=asRat(offset))
        return self.timeTransform(transform)

    def invertPitch(self: _T, pivot: pitch_t) -> _T:
        """
        Invert the pitch of this object

        Args:
            pivot: the point around which to invert pitches

        Returns:
            the inverted object
        """
        pivotm = asmidi(pivot)
        func = lambda pitch: pivotm*2 - pitch
        return self.pitchTransform(func)

    def transpose(self:_T, interval: Union[int, float]) -> _T:
        """
        Transpose this object by the given interval

        Args:
            interval: the interval in semitones

        Returns:
            the transposed object
        """
        return self.pitchTransform(lambda pitch: pitch+interval)


@functools.lru_cache(maxsize=1000)
def _renderImage(obj: MusicObj, outfile: Optional[str], fmt, backend,
                  scorestruct: ScoreStruct,
                  config: configdict.ConfigDict):
    renderoptions = notation.makeRenderOptionsFromConfig(config)
    if scorestruct is None:
        scorestruct = getScoreStruct()
    r = obj.render(backend=backend, renderoptions=renderoptions, scorestruct=scorestruct,
                   config=config)
    if not outfile:
        outfile = _tempfile.mktemp(suffix='.' + fmt)
    r.write(outfile)
    return outfile


def _renderObject(obj: MusicObj,
                  backend:str,
                  renderoptions: scoring.render.RenderOptions,
                  scorestruct: ScoreStruct,
                  config: configdict.ConfigDict
                  ) -> scoring.render.Renderer:
    """
    Render an object

    NB: we put it here in order to make it easier to cache rendering, if needed

    All args must be given (not None) so that caching is meaningful.

    Args:
        obj: the object to make the image from (a Note, Chord, etc.)
        outfile: if given, the path to be generated. The extension must be one of
            '.png', '.pdf', '.xml' (for musicxml) or '.ly' (for lilypond)
        backend : one of 'musicxml', 'lilypond'
        scorestruct: if given, this ScoreStruct will be used for rendering. Otherwise
            the scorestruct within the active Workspace is used
        config: if given, this config is used for rendering. Otherwise the config
            within the active Workspace is used

    Returns:
        a scopring.Renderer. The returned object can be used to render (via the ``write()``
        method) or to have access to the generated score (see ``nativeScore()``)

    .. note::

        To render with a temporary Wrokspace (i.e., without modifying the active Workspace),
        use::

        .. code-blocl:: python

            with Workspace(scorestruct=..., config=..., ...) as w:
                renderObject(myobj, "outfile.pdf")
    """
    assert backend and renderoptions and scorestruct and config
    logger.debug(f"rendering parts with backend: {backend}")
    parts = obj.scoringParts()
    renderer = notation.renderWithCurrentWorkspace(parts, backend=backend,
                                                   renderoptions=renderoptions,
                                                   scorestruct=scorestruct,
                                                   config=config)
    return renderer


def resetImageCache() -> None:
    """
    Reset the image cache. Useful when changing display format
    """
    _renderImage.cache_clear()
