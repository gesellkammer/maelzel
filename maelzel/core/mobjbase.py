"""
Musical Objects
---------------

Time
~~~~

A MObj has always a start and dur attribute. They refer to an abstract time.
When visualizing a MObj as musical notation these times are interpreted/converted
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

import music21 as m21

from emlib.misc import firstval
import emlib.misc
import emlib.img

import pitchtools as pt

import csoundengine

from maelzel.common import asmidi
from ._common import *
from .config import CoreConfig
from .workspace import Workspace, getConfig, getScoreStruct
from . import playback
from . import environment
from . import symbols as _symbols
from . import notation
from . import _util
from . import _dialogs
from maelzel.rational import Rat
import maelzel.music.m21tools as m21tools
from maelzel import scoring

from .synthevent import PlayArgs, SynthEvent, cropEvents
from maelzel.scorestruct import ScoreStruct

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Optional, Any, TypeVar, Callable
    from ._typedefs import *
    T = TypeVar('MObjT', bound='MObj')
    from .playback import OfflineRenderer


__all__ = (
    'MObj',
    'resetImageCache'
)


@dataclass
class _TimeScale:
    factor: Rat
    offset: Rat

    def __call__(self, t: num_t):
        r = asRat(t)
        return r*self.factor + self.offset


class MObj:
    """
    This is the base class for all core objects.

    This is an abstract class. **It should not be instantiated by itself**

    A :class:`MObj` can display itself via :meth:`show` and play itself via :meth:`play`.
    It can have a duration and a start time.

    A :class:`MObj` can customize its playback via :meth:`MObj.setPlay`. These attributes can
    be accessed through the `playargs` property

    Elements purely related to notation (text annotations, articulations, etc)
    are added to a :class:`MObj` through the :meth:`addSymbol` and can be accessed
    through the :attr:`symbol` attribute.
    A symbol is an attribute or notation element (like color, size or an attached
    text expression) which has meaning only in the realm of graphical representation.

    Args:
        dur: the (optional) duration of this object, in abstract units (beats)
        start: the (optional) time offset of this object, in abstract units (beats)
        label: a string label to identify this object, if necessary
    """
    _acceptsNoteAttachedSymbols = True
    _isDurationRelative = True

    __slots__ = ('dur', 'start', 'label', '_playargs', 'symbols', '_scorestruct', '_properties')

    def __init__(self, dur: time_t = None, start: time_t = None, label: str = '',
                 properties: dict[str, Any] = None):

        self.label = label
        "a label can be used to identify an object within a group of objects"

        # A MObj can have a duration. A duration can't be 0
        # A duration of -1 means max. duration.

        self.dur: Rat | None = dur
        "the duration of this object (can be None, in which case it is unset)"

        self.start: Rat | None = start
        "start specifies a time offset for this object"

        self.symbols: list[_symbols.Symbol] | None = None
        "A list of all symbols added via addSymbol (can be None)"

        # _playargs are set via .setplay and serve the purpose of
        # attaching playing parameters (like pan position, instrument)
        # to an object
        self._playargs: PlayArgs | None = None

        self._properties: dict[str, Any] | None = properties

        self._scorestruct: ScoreStruct | None = None

    @property
    def properties(self) -> dict[str, Any]:
        """A place to put user-defined properties"""
        if self._properties is None:
            self._properties = {}
        return self._properties

    def pitchRange(self) -> Optional[tuple[float, float]]:
        """
        The pitch range of this object, if applicable

        This is useful to assign a proper Voice when distributing
        objects among voices

        Returns:
            either None or a tuple (lowest pitch, highest pitch)
        """
        return None

    def resolvedStart(self) -> Rat:
        """
        Resolved start of this object

        If the start is unset (``None``), a fallback start is returned
        For non-container objects (:class:`Note`, :class:`Chord`), this is either the
        explicitely set ``.start`` attribute, or 0.

        Derived classes can override this method to match their behaviour

        Returns:
            the resolved start time, in quarter notes
        """
        return self.start if self.start is not None else Rat(0)

    def resolvedDur(self, start: time_t = None) -> Rat:
        """
        The explicit duration or a default duration, in quarternotes

        If this object has an explicitely set duration, this method returns
        that, otherwise returns a default duration. Child
        classes can override this method to match their behaviour

        For non-container objects (:class:`Note`, :class:`Chord`) this is either
        the explicitely set ``.dur`` attribute, or 1.0

        Args:
            start: the start time of the object, overrides the set start time. This
                is relevant for objects with a time defined in seconds, because the
                duration in quarternotes might be dependent on the scorestruct
                so the start time needs to be given
        """
        return self.dur if self.dur is not None else Rat(1)

    def resolved(self, start: time_t = None):
        """
        Copy of self with explicit times

        Args:
            start: a start time to fill or override self.start.

        Returns:
            a clone of self with dur and start set to explicit
            values

        """
        if self.dur is not None and self.start is not None:
            return self
        start = asRat(firstval(start, self.start, Rat(0)))
        dur = self.resolvedDur(start)
        return self.clone(dur=dur, start=start)

    @property
    def playargs(self) -> PlayArgs:
        """
        A PlayArgs structure, containing any specification regarding playback

        .. seealso:: :meth:`~MObj.setPlay`
        """
        if self._playargs is None:
            self._playargs = PlayArgs()
        return self._playargs

    def setPlay(self: T, /, **kws) -> T:
        """
        Set any playback attributes, returns self

        .. note::

            It is possible to access the :attr:`MObj.playargs` attribute directly to set any
            play parameter, like ``note.playargs['instr'] = 'piano'``

            The advantage of :meth:`MObj.setPlay` is that one can set multiple
            parameters simultaneously and the method can be chained with
            the constructor, such as ``note = Note("4C", ...).setPlay(instr=..., gain=...)``

        Args:
            **kws: any argument passed to :meth:`~MObj.play` (delay, dur, chan,
                gain, fade, instr, pitchinterpol, fadeshape, params,
                priority, position).

        Returns:
            self. This allows to chain this to any constructor (see example)

        ============================= =====================================================
        Playback Attribute            Descr
        ============================= =====================================================
        instr: ``str``                The instrument preset to use
        delay: ``float``              Delay in seconds, added to the start of the object
        gain: ``float``               A gain factor applied to the amplitude of this object
        chan: ``int``                 The channel to output to, **channels start at 1**
        pitchinterpol: ``str``        One of 'linear', 'cos', 'freqlinear', 'freqcos'
        fade: ``float``               The fade time; can also be a tuple (fadein, fadeout)
        position: ``float``           Panning position (0=left, 1=right)
        start: ``float``              Start time of playback; allows to play a fragment of the object
        end: ``float``                End time of playback; allow to trim playback of the object
        ============================= =====================================================


        **Example**::

            # a piano note
            >>> from maelzel.core import *
            # Create a note with predetermined instr and panning position
            >>> note = Note("C4+25", dur=0.5).setPlay(instr="piano", position=1)
            # When .play is called, the note will play with the preset instr and position
            >>> note.play()

        .. seealso:: :meth:`~MObj.addSymbol`, :attr:`~MObj.playargs`
        """
        playargs = self.playargs
        for k, v in kws.items():
            playargs[k] = v
        return self

    def clone(self: T,
              start: Union[Rat, None, UNSET] = UNSET,
              **kws) -> T:
        """
        Clone this object, changing parameters if needed

        Args:
            start: the start of this object (use None to erase an already
                set start, leave as UNSET to use the object's start)
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
        if start is not UNSET:
            out.start = None if start is None else asRat(start)

        for k, v in kws.items():
            setattr(out, k, v)

        if self._playargs is not None:
            out._playargs = self._playargs.copy()
        if self._properties is not None:
            out._properties = self._properties.copy()
        return out

    def copy(self: T) -> T:
        """Returns a copy of this object"""
        return _copy.deepcopy(self)

    def moveTo(self, start: time_t) -> None:
        """Move this to the given start time (**in place**)

        Args:
            start: the new start time
        """
        self.start = start
        self._changed()

    def timeShift(self:T, timeoffset: time_t) -> T:
        """
        Return a copy of this object with an added time offset

        Args:
            timeoffset: a delta time added

        Returns:
            a copy of this object shifted in time by the given amount
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
        """
        The end time of this object.

        Will be None if this object has no duration or no start"""
        if self.dur is None or self.start is None:
            return None
        return self.start + self.dur

    def quantizePitch(self: T, step=0.) -> T:
        """ Returns a new object, with pitch rounded to step """
        raise NotImplementedError()

    def transposeByRatio(self: T, ratio: float) -> T:
        """
        Transpose this by a given frequency ratio, if applicable

        A ratio of 2 equals to transposing an octave higher.

        Args:
            ratio: the ratio to transpose by

        Returns:
            a copy of this object, transposed by the given ratio

        Example
        -------

            >>> from maelzel.core import *
            >>> n = Note("4C")
            # A transposition by a ratio of 2 results in a pitch an octave higher
            >>> n.transposeByRatio(2)
            5C
        """
        return self.transpose(pt.r2i(ratio))

    def show(self, fmt: str = None, external: bool = None, backend: str = None,
             scorestruct: ScoreStruct = None,
             config: CoreConfig = None,
             resolution: int = None
             ) -> None:
        """
        Show this as notation.

        Args:
            external: True to force opening the image in an external image viewer,
                even when inside a jupyter notebook. If False, show will
                display the image inline if inside a notebook environment.
                To change the default, modify :ref:`config['show.external'] <config_show_external>`
            backend: backend used when rendering to png/pdf.
                One of 'lilypond', 'music21'. None to use default
                (see :ref:`config['show.backend'] <config_show_backend>`)
            fmt: one of 'png', 'pdf', 'ly'. None to use default.
            scorestruct: if given overrides the current/default score structure
            config: if given overrides the current/default config
            resolution: dpi resolution when rendering to an image, overrides the
                :ref:`config key 'show.pngResolution' <config_show_pngresolution>`
        """
        cfg = config or getConfig()
        if resolution:
            cfg = cfg.clone({'show.pngResolution': resolution})

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
                _util.showLilypondScore(r.nativeScore())
        else:
            img = self.renderImage(backend=backend, fmt=fmt, scorestruct=scorestruct,
                                   config=cfg)
            if fmt == 'png':
                _util.pngShow(img, forceExternal=external)
            else:
                emlib.misc.open_with_app(img)

    def _changed(self) -> None:
        """
        This method is called whenever the object changes its representation

        This happens when a note changes its pitch inplace, the duration is modified, etc.
        This invalidates, among other things, the image cache for this object
        """
        pass

    def render(self, backend: str = None,
               renderoptions: scoring.render.RenderOptions = None,
               scorestruct: ScoreStruct = None,
               config: CoreConfig = None) -> scoring.render.Renderer:
        """
        Renders this object as a quantized score

        Args:
            backend: the backend to use, one of 'lilypond', 'music21'. If not given,
                defaults to the :ref:`config key 'show.backend' <config_show_backend>`
            renderoptions: the render options to use. If not given, these are generated from
                the active config
            scorestruct: if given, overrides the scorestruct set within the active Workspace
                and any scorestruct attached to this object
            config: if given, overrides the active config

        Returns:
            a scoring.render.Renderer. This can be used to write the rendered structure
            to an image (png, pdf) or as a musicxml or lilypond file.
        """
        w = Workspace.active
        if config is None:
            config = w.config
        if not backend:
            backend = config['show.backend']
        if not renderoptions:
            renderoptions = notation.makeRenderOptionsFromConfig(config)
        if not scorestruct:
            scorestruct = self.scorestruct or w.scorestruct
        return _renderObject(self, backend=backend, renderoptions=renderoptions,
                             scorestruct=scorestruct, config=config)

    def renderImage(self, backend: str = None, outfile: str = None, fmt="png",
                    scorestruct: ScoreStruct = None,
                    config: CoreConfig = None
                    ) -> str:
        """
        Creates an image representation, returns the path to the image

        Args:
            backend: the rendering backend. One of 'music21', 'lilypond'
                None uses the default method
                (see :ref:`getConfig()['show.backend'] <config_show_backend>`)
            outfile: the path of the generated file. Use None to generate
                a temporary file.
            fmt: if outfile is None, fmt will determine the format of the
                generated file. Possible values: 'png', 'pdf'.
            scorestruct: if given will override the active ScoreStruct

        Returns:
            the path of the generated file. If outfile was given, the returned
            path will be the same as the outfile.

        .. seealso:: :meth:`MObj.render`
        """
        w = Workspace.active
        if not config:
            config = w.config
        if backend is None:
            backend = config['show.backend']
        if fmt == 'ly':
            backend = 'lilypond'
        if scorestruct is None:
            scorestruct = self.scorestruct or w.scorestruct
        path = _renderImage(self, outfile, fmt=fmt, backend=backend, scorestruct=scorestruct,
                            config=config or getConfig())
        if not os.path.exists(path):
            # cached image does not exist?
            resetImageCache()
            raise RuntimeError("The returned image file does not exist")
        return path


    def scoringEvents(self,
                      groupid: str = None,
                      config: CoreConfig = None,
                      ) -> list[scoring.Notation]:
        """
        Returns its notated form as scoring.Notations

        These can then be converted into notation via some of the available
        backends: musicxml or lilypond

        Args:
            groupid: passed by an object higher in the hierarchy to
                mark this objects as belonging to a group
            config: a configuration to customize rendering

        Returns:
            A list of scoring.Notation which best represent this
            object as notation
        """
        raise NotImplementedError("Subclass should implement this")

    def scoringParts(self, options: scoring.render.RenderOptions=None
                     ) -> list[scoring.Part]:
        """
        Returns this object as a list of scoring Parts.

        Args:
            options: render options used

        Returns:
            a list of scoring.Part

        This method is used internally to generate the parts which
        constitute a given MObj prior to rendering,
        but might be of use itself so it is exposed here.

        A :class:`maelzel.scoring.Part` is an intermediate format used by the scoring
        package to represent notated events. A :class:`maelzel.scoring.Part`
        is unquantized and independent of any score structure
        """
        notations = self.scoringEvents(config=getConfig())
        if not notations:
            return []
        scoring.stackNotationsInPlace(notations)
        # scoring.enharmonics.fixEnharmonicsInPlace(notations)
        parts = scoring.distributeNotationsByClef(notations)
        return parts

    def scoringArrangement(self, title:str=None) -> scoring.Arrangement:
        """
        Create a notation Score from this object

        Args:
            title: the title of the resulting score (if given)

        Returns:
            the Score representation of this object

        """
        parts = self.scoringParts()
        return scoring.Arrangement(parts, title=title)

    def _scoringAnnotation(self) -> Optional[scoring.attachment.Text]:
        """ Returns owns annotations as a scoring Annotation """
        if not self.label:
            return None
        return scoring.attachment.Text(self.label, fontsize=getConfig()['show.labelFontSize'])

    def asmusic21(self, **kws) -> m21.stream.Stream:
        """
        Convert this object to its music21 representation

        Args:

            **kws: not used here, but classes inheriting from
                this may want to add customization

        Returns:
            a music21 stream which best represent this object as
            notation.

        .. note::

            The music21 representation is final, not thought to be embedded into
            another stream. For embedding we use an abstract representation of scoring
            objects which can be queried via .scoringEvents
        """
        parts = self.scoringParts()
        renderer = notation.renderWithActiveWorkspace(parts,
                                                      backend='music21',
                                                      scorestruct=self.scorestruct)
        stream = renderer.asMusic21()
        if getConfig()['m21.fixStream']:
            m21tools.fixStream(stream, inPlace=True)
        return stream

    def musicxml(self) -> str:
        """
        Return the music representation of this object as musicxml.
        """
        stream = self.asmusic21()
        return m21tools.getXml(stream)

    @property
    def scorestruct(self) -> ScoreStruct | None:
        """
        Returns the ScoreStruct attached to this obj, if Any
        """
        return self._scorestruct

    def write(self, outfile: str, backend: str = None,
              scorestruct: ScoreStruct = None,
              resolution: int = None
              ) -> None:
        """
        Export to multiple formats

        Formats supported: pdf, png, musicxml (extension: .xml or .musicxml),
        lilypond (.ly), midi (.mid or .midi)

        Args:
            outfile: the path of the output file. The extension determines
                the format
            backend: the backend used when writing as pdf or png. If not given,
                the default defined in the active config is used
                (:ref:`key: 'show.backend' <config_show_backend>`).
                Possible backends: ``lilypond``; ``music21`` (uses MuseScore to render musicxml as
                image so MuseScore needs to be installed)
            scorestruct: if given it will overwrite the active ScoreStruct
            resolution: image DPI (only valid if rendering to an image) - overrides
                the :ref:`config key 'show.pngResolution' <config_show_pngresolution>`
        """
        if outfile == '?':
            outfile = _dialogs.selectFileForSave(key="writeLastDir",
                                                 filter="All formats (*.pdf, *.png, "
                                                        "*.ly, *.xml, *.mid)")
            if not outfile:
                logger.info("File selection cancelled")
                return
        ext = os.path.splitext(outfile)[1]
        cfg = getConfig()
        if ext == '.ly' or ext == '.mid' or ext == '.midi':
            backend = 'lilypond'
        elif ext == '.xml' or ext == '.musicxml':
            backend = 'music21'
        elif backend is None:
            backend = cfg['show.backend']
        if resolution is not None:
            cfg = cfg.clone(updates={'show.pngResolution': resolution})
        r = notation.renderWithActiveWorkspace(self.scoringParts(),
                                               backend=backend,
                                               scorestruct=scorestruct or self.scorestruct,
                                               config=cfg)
        r.write(outfile)

    def _htmlImage(self) -> str:
        imgpath = self.renderImage()
        if not imgpath:
            return ''
        scaleFactor = getConfig().get('show.scaleFactor', 1.0)
        width, height = emlib.img.imgSize(imgpath)
        img = emlib.img.htmlImgBase64(imgpath,
                                      width=f'{int(width * scaleFactor)}px')
        return img

    def _repr_html_header(self):
        return _html.escape(repr(self))

    def _repr_html_(self) -> str:
        img = self._htmlImage()
        txt = self._repr_html_header()
        return rf'<code style="font-size:0.9em">{txt}</code><br>' + img

    def dump(self, indents=0):
        """
        Prints all relevant information about this object
        """
        print(f'{"  "*indents}{repr(self)}')
        if self._playargs:
            print(f'{"  "*(indents+1)}{self.playargs}')
        if self.symbols:
            print(f'{"  " * (indents + 1)}{self.symbols}')

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace,
                     ) -> list[SynthEvent]:
        """
        Must be overriden by each class to generate SynthEvents

        Args:
            playargs: a :class:`PlayArgs`, structure, filled with given values,
                own .playargs values and config defaults (in that order)
            workspace: a Workspace. This is used to determine the scorestruct, the
                configuration and a mapping between dynamics and amplitudes

        Returns:
            a list of :class:`SynthEvent`s
        """
        raise NotImplementedError("Subclass should implement this")

    def events(self,
               instr: str = None,
               delay: float = None,
               args: dict[str, float] = None,
               gain: float = None,
               chan: int = None,
               pitchinterpol: str = None,
               fade: Union[float, tuple[float, float]] = None,
               fadeshape: str = None,
               position: float = None,
               start: float = None,
               end: float = None,
               sustain: float = None,
               workspace: Workspace = None,
               transpose: float=0,
               **kwargs
               ) -> list[SynthEvent]:
        """
        Returns the SynthEvents needed to play this object

        All these attributes here can be set previously via `playargs` (or
        using :meth:`~maelzel.core.mobj.MObj.setPlay`)

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
            args: named arguments passed to the note. A dict ``{paramName: value}``
            position: the panning position (0=left, 1=right)
            start: start playback at the given offset (in quarternotes). Allows to play
                a fragment of the object (NB: this trims the playback of the object.
                Use `delay` to offset the playback in time while keeping the playback time
                unmodified)
            end: end time of playback, in quarternotes. Allows to play a fragment of the object by trimming the end of the playback
            sustain: a time added to the playback events to facilitate overlapping/legato between
                notes, or to allow one-shot samples to play completely without being cropped.
            workspace: a Workspace. If given, overrides the current workspace. It's scorestruct
                is used to to determine the mapping between beat-time and real-time.
            transpose: an interval to transpose any pitch

        Returns:
            A list of SynthEvents (see :class:`SynthEvent`)

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> n = Note(60, dur=1).setPlay(instr='piano')
            >>> n.events(gain=0.5)
            [SynthEvent(delay=0.000, gain=0.5, chan=1, fade=(0.02, 0.02), instr=piano)
             bps 0.000s:  60, 1.000000
                 1.000s:  60, 1.000000]
            >>> play(n.events(chan=2))

        """
        if instr == "?":
            from .presetman import presetManager
            instr = presetManager.selectPreset()
            if not instr:
                raise ValueError("No preset selected")

        pairs = (
            ('instr', instr),
            ('delay', delay),
            ('gain', gain),
            ('fade', fade),
            ('pitchinterpol', pitchinterpol),
            ('fadeshape', fadeshape),
            ('args', args),
            ('position', position),
            ('chain', chan),
            ('sustain', sustain),
            ('transpose', transpose)
        )

        d = {k: v for k, v in pairs if v is not None}

        if kwargs:
            if args:
                args.update(kwargs)
            else:
                d['args'] = kwargs
        playargs = PlayArgs(d)
        if workspace is None:
            workspace = Workspace.active
        if struct:=self.scorestruct:
            workspace = workspace.clone(scorestruct=struct)
        events = self._synthEvents(playargs, workspace)
        if start is not None or end is not None:
            scorestruct = self.scorestruct or workspace.scorestruct
            starttime = None if start is None else scorestruct.beatToTime(start)
            endtime = None if end is None else scorestruct.beatToTime(end)
            events = cropEvents(events, start=starttime, end=endtime, rewind=True)
        return events

    def play(self,
             instr: str = None,
             delay: float = None,
             args: dict[str, float] = None,
             gain: float = None,
             chan: int = None,
             pitchinterpol: str = None,
             fade: Union[float, tuple[float, float]] = None,
             fadeshape: str = None,
             position: float = None,
             start: float = None,
             end: float = None,
             whenfinished: Callable = None,
             sustain: float = None,
             workspace: Workspace = None,
             transpose: float = 0,
             **kwargs
             ) -> csoundengine.synth.AbstrSynth:
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
            args: arguments passed to the note. A dict ``{paramName: value}``
            position: the panning position (0=left, 1=right)
            start: start time of playback. Allows to play a fragment of the object (NB: this trims the playback
                of the object. Use `delay` to offset the playback in time while keeping the playback time
                unmodified)
            end: end time of playback. Allows to play a fragment of the object by trimming the end of the playback
            sustain: a time added to the playback events to facilitate overlapping/legato between
                notes, or to allow one-shot samples to play completely without being cropped.
            workspace: a Workspace. If given, overrides the current workspace. It's scorestruct
                is used to to determine the mapping between beat-time and real-time. 
                
        Returns:
            A :class:`~csoundengine.synth.SynthGroup`


        .. seealso::
            * :meth:`MObj.events`
            * :meth:`MObj.rec`
            * :func:`~maelzel.core.playback.render`,
            * :class:`~maelzel.core.playbakc.playgroup`


        Example
        ~~~~~~~

        Play a note

            >>> from maelzel.core import *
            >>> note = Note(60).play(gain=0.1, chan=2)

        Play multiple objects synchronised

            >>> with playgroup():
            ...     Note(60, 1.5).play(gain=0.1, position=0.5)
            ...     Chord("4E 4G", 2, start=1.2).play(instr='piano')
            ...     ...


        Render offline

            >>> with render("out.wav", sr=44100) as r:
            ...     Note(60, 5).play(gain=0.1, chan=2)
            ...     Chord("4E 4G", 3).play(instr='piano')
        """
        if workspace is None:
            workspace = Workspace.active

        events = self.events(delay=delay,
                             chan=chan,
                             fade=fade,
                             gain=gain,
                             instr=instr,
                             pitchinterpol=pitchinterpol,
                             fadeshape=fadeshape,
                             args=args,
                             position=position,
                             sustain=sustain,
                             workspace=workspace,
                             start=start,
                             end=end,
                             transpose=transpose,
                             **kwargs)

        if not events:
            return csoundengine.synth.SynthGroup([playback._dummySynth()])

        if (renderer:=workspace.renderer) is not None:
            # schedule offline
            for ev in events:
                renderer.schedEvent(ev)
        else:
            return playback._playFlatEvents(events, whenfinished=whenfinished)

    def rec(self,
            outfile: str = None,
            sr: int = None,
            quiet: bool = None,
            wait: bool = None,
            nchnls: int = None,
            instr: str = None,
            delay: float = None,
            args: dict[str, float] = None,
            gain: float = None,
            position: float = None,
            **kws
            ) -> OfflineRenderer:
        """
        Record the output of .play as a soundfile

        Args:
            outfile: the outfile where sound will be recorded. Can be
                None, in which case a filename will be generated. Use '?'
                to open a save dialog
            sr: the sampling rate (:ref:`config key: 'rec.sr' <config_rec_sr>`)
            wait: if True, the operation blocks until recording is finishes
                (:ref:`config 'rec.block' <config_rec_block>`)
            nchnls: if given, use this as the number of channels to record.

            gain: modifies the own amplitude for playback/recording (0-1)
            delay: delay in seconds, added to the start of the object
                As opposed to the .start attribute of each object, which is defined
                in symbolic (beat) time, the delay is always in real (seconds) time
            instr: which instrument to use (see defPreset, definedPresets). Use "?" to
                select from a list of defined presets.
            args: named arguments passed to the note. A dict ``{paramName: value}``
            position: the panning position (0=left, 1=right)

            **kws: any keyword passed to .play

        Returns:
            the offline renderer used. If no outfile was given it is possible to
            access the renderer soundfile via
            :meth:`OfflineRenderer.lastOutfile() <maelzel.core.playback.OfflineRenderer.lastOutfile>`

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> # a simple note
            >>> chord = Chord("4C 4E 4G", dur=8).setPlay(gain=0.1, instr='piano')
            >>> renderer = chord.rec(wait=True)
            >>> renderer.lastOutfile()
            '/home/testuser/.local/share/maelzel/recordings/tmpashdas.wav'

        See Also
        ~~~~~~~~

        - :class:`~maelzel.core.playback.OfflineRenderer`
        """
        events = self.events(instr=instr, position=position,
                             delay=delay, args=args, gain=gain,
                             **kws)
        return playback.render(outfile=outfile, events=events, r=sr, wait=wait,
                               quiet=quiet, nchnls=nchnls)

    def isRest(self) -> bool:
        """
        Is this object a Rest?

        Rests are used as separators between objects inside an Chain or a Track
        """
        return False

    def addSymbol(self: T, symbol: Union[str, _symbols.Symbol], *args, **kws) -> T:
        """
        Add a notation symbol to this object

        Notation symbols are any attributes which are for
        notation purposes only. Such attributes include dynamics,
        articulations, etc.

        Some symbols are exclusive, meaning that adding a symbol of this kind will
        replace a previously set symbol. Exclusive symbols include any properties
        (color, size, etc) and other customizations like notehead shape,

        .. note::

            Dynamics are not treated as symbols.

        Example
        -------

            >>> from maelzel.core import *
            >>> n = Note(60)
            >>> n.addSymbol('articulation', 'accent')
            >>> n = Note(60).setPlay(instr='piano').addSymbol('text', 'dolce')
            >>> from maelzel.core import symbols
            >>> n2 = Note("4G").addSymbol(symbols.Harmonic(interval=5))


        Args:
            symbol: the name of a symbol. See :meth:`MObj.supportedSymbols` for symbols
                accepted by this object.
            args, keys: passed directly to the class constructor

        Returns:
            self (similar to setPlay, allows to chain calls)

        ============  ==========================================================
        Symbol        Arguments
        ============  ==========================================================
        expression    | text: ``str``
                      | placement: ``str`` {above, below}
        notehead      | kind: ``str`` {cross, harmonic, triangleup, xcircle, triangle, rhombus, square, rectangle}
                      | color: ``str`` (a css color)
                      | parenthesis: ``bool``
        articulation  kind: ``str`` {accent, staccato, tenuto, marcato, staccatissimo}
        size          value: ``int`` (0=default, 1, 2, …=bigger, -1, -2, … = smaller)
        color         value: ``str`` (a css color)
        accidental    | hidden: ``bool``
                      | parenthesis: ``bool``
                      | color:  ``str`` (a css color)
        ============  ==========================================================

        """
        if isinstance(symbol, _symbols.Symbol):
            symboldef = symbol
        else:
            symboldef = _symbols.makeSymbol(symbol, *args, **kws)

        if isinstance(symbol, _symbols.NoteAttachedSymbol) \
                and not self._acceptsNoteAttachedSymbols:
            raise ValueError(f"A {type(self)} does not accept note attached symbols")
        if self.symbols is None:
            self.symbols = [symboldef]
        else:
            if symboldef.exclusive:
                cls = type(symboldef)
                if any(isinstance(s, cls) for s in self.symbols):
                    self.symbols = [s for s in self.symbols if not isinstance(s, cls)]
            self.symbols.append(symboldef)
        if isinstance(symboldef, _symbols.Spanner):
            symboldef.setAnchor(self)
        return self

    def _removeSymbolsOfClass(self, cls: str | type):
        if isinstance(cls, str):
            cls = cls.lower()
            symbols = [s for s in self.symbols if s.name == cls]
        else:
            symbols = [s for s in self.symbols if isinstance(s, cls)]
        for s in symbols:
            self.symbols.remove(s)

    def getSymbol(self, classname: str) -> Optional[_symbols.Symbol]:
        """
        Get a symbol of a given class, if present

        This is only supported for symbol classes which are exclusive
        (notehead, color, ornament, etc.). For symbols like 'articulation',
        which can be present multiple times, query the symbols attribute
        directly (**NB**: symbols might be ``None`` if no symbols have been set):

        .. code::

            if note.symbols:
                articulations = [s for s in note.symbols
                                 if s.name == 'articulation']

        Args:
            classname: the class of the symbol. Possible values are
                'articulation', 'text', 'notehead', 'color', 'ornament',
                'fermata', 'notatedpitch'. See XXX (TODO) for a complete list

        Returns:
            a symbol of the given class, or None
        """

        if not self.symbols:
            return None
        classname = classname.lower()
        return next((s for s in self.symbols if s.name==classname), None)

    def addText(self, text: str, placement='above', fontsize: int = None,
                fontstyle: str = None, box: str = ''
                ) -> None:
        """
        Shortcut to add a text annotation to this object

        This is a shortcut to ``self.setSymbol(symbols.Text(...))``. Use
        that for in-depth customization.

        Args:
            text: the text annotation
            placement: where to place the annotation ('above', 'below')
            fontsize: the size of the annotation
            fontstyle: italic, bold or a comma-separated list thereof ('italic,bold')
            box: the enclosure shape, or '' for no box around the text. Possible shapes
                are 'square', 'circle', 'rounded'

        """
        self.addSymbol(_symbols.Text(text, placement=placement, fontsize=fontsize,
                                     fontstyle=fontstyle, box=box))

    def addSpanner(self,
                   spannercls: str | _symbols.Spanner,
                   endobj: MObj
                   ) -> None:
        """
        Adds a spanner symbol to this object

        A spanner is a slur, line or any other symbol attached to two or more
        objects. A spanner always has a start and an end.

        Args:
            spannercls: one of 'slur', '<', '>', or a Spanner object
            endobj: the object where this spanner ends

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> a = Note("4C")
            >>> b = Note("4E")
            >>> c = Note("4G")
            >>> a.addSpanner('slur', c)
            >>> chain = Chain([a, b, c])

        .. seealso:: :meth:`Spanner.bind() <maelzel.core.symbols.Spanner.bind>`

        """
        spannercls = spannercls.lower()
        spanner = _symbols.makeSpanner(spannercls)
        spanner.bind(self, endobj)

    def timeTransform(self:T, timemap: Callable[[num_t], num_t], inplace=False) -> T:
        """
        Apply a time-transform to this object

        Args:
            timemap: a function mapping old time to new time
            inplace: if True changes are applied in place

        Returns:
            the resulting object (self if inplace)

        .. note::

            time is conceived as abstract 'beat' time, measured in quarter-notes.
            The actual time will be also determined by any tempo changes in the
            active score structure.
        """
        start = 0. if self.start is None else self.start
        dur = 1.0 if self.dur is None else self.dur
        start2 = timemap(start)
        dur2 = timemap(start+dur) - start2
        if inplace:
            self.start = start2
            self.dur = dur2
            self._changed()
            return self
        else:
            return self.clone(start=asRat(start2), dur=asRat(dur2))

    def timeShiftInPlace(self, timeoffset: time_t) -> None:
        """
        Shift the time of this by the given offset (in place)

        Args:
            timeoffset: the time delta (in quarterNotes)
        """
        self.start = self.start + timeoffset
        self._changed()

    def startSecs(self) -> Rat:
        """
        Returns the .start time in seconds according to the score

        The absolute time depends on the active ScoreStruct
        An Exception is raised if self does not have an start time

        This is equivalent to ``activeScoreStruct().beatToTime(obj.start)``
        """
        if self.start is None:
            raise ValueError(f"The object {self} has no explicit .start")
        s = getScoreStruct()
        timefrac = s.beatToTime(self.start)
        return Rat(timefrac.numerator, timefrac.denominator)

    def durSecs(self) -> Rat:
        """
        Returns the duration in seconds according to the active score

        Returns:
            the duration of self in seconds
        """
        s = getScoreStruct()
        start = self.resolvedStart()
        dur = self.resolvedDur(start=start)
        return s.timeDelta(start, start+dur)

    def endSecs(self) -> Rat:
        """
        Returns the end of this in seconds according to the active score

        The absolute end time depends on the active ScoreStruct.
        An Exception is raised if self does not have an end time

        This is equivalent to ``activeScoreStruct().beatToTime(obj.end)``
        """
        return self.startSecs() + self.durSecs()

    def pitchTransform(self:T, pitchmap: Callable[[float], float]) -> T:
        """
        Apply a pitch-transform to this object, returns a copy

        Args:
            pitchmap: a function mapping pitch to pitch

        Returns:
            the object after the transform
        """
        raise NotImplementedError("Subclass should implement this")

    def timeScale(self: T, factor: num_t, offset: num_t = 0) -> T:
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

    def adaptToScoreStruct(self: T, newstruct: ScoreStruct, oldstruct: ScoreStruct=None
                           ) -> T:
        """
        Adapts the time to a new ScoreStruct so that its absolute time is unmodified

        Args:
            newstruct: the new ScoreStruct
            oldstruct: the old ScoreStruct. If not given either the attached or the
                current ScoreStruct is used

        Returns:
            a clone of self with the modified start and duration

        """
        oldstruct = oldstruct or self.scorestruct or getScoreStruct()
        if newstruct == oldstruct:
            logger.warning("The new scorestruct is the same as the old scorestruct")
            return self
        startTime = oldstruct.beatToTime(self.start)
        endTime = oldstruct.beatToTime(self.end)
        return self.clone(start=newstruct.timeToBeat(startTime),
                          dur=newstruct.beatDelta(startTime, endTime))

    def invertPitch(self: T, pivot: pitch_t) -> T:
        """
        Invert the pitch of this object

        Args:
            pivot: the point around which to invert pitches

        Returns:
            the inverted object

        .. code::

            >>> from maelzel.core import *
            >>> series = Chain("4Bb 4E 4F# 4Eb 4F 4A 5D 5C# 4G 4G# 4B 5C".split())
            >>> inverted = series.invertPitch("4F#")
            >>> print(" ".join(_.name.ljust(4) for _ in series))
            ... print(" ".join(_.name.ljust(4) for _ in inverted))
            4A#  4E   4F#  4D#  4F   4A   5D   5C#  4G   4G#  4B   5C
            4D   4G#  4F#  4A   4G   4D#  3A#  3B   4F   4E   4C#  4C
            >>> Score([series, inverted])

        .. image:: ../assets/dodecaphonic-series-1-inverted.png
        """
        pivotm = asmidi(pivot)
        func = lambda pitch: pivotm*2 - pitch
        return self.pitchTransform(func)

    def transpose(self:T, interval: Union[int, float]) -> T:
        """
        Transpose this object by the given interval

        Args:
            interval: the interval in semitones

        Returns:
            the transposed object
        """
        return self.pitchTransform(lambda pitch: pitch+interval)

@functools.lru_cache(maxsize=1000)
def _renderImage(obj: MObj, outfile: Optional[str], fmt, backend,
                 scorestruct: ScoreStruct,
                 config: CoreConfig):
    renderoptions = notation.makeRenderOptionsFromConfig(config)
    if scorestruct is None:
        scorestruct = getScoreStruct()
    r = obj.render(backend=backend, renderoptions=renderoptions, scorestruct=scorestruct,
                   config=config)
    if not outfile:
        outfile = _tempfile.mktemp(suffix='.' + fmt)
    r.write(outfile)
    return outfile


def _renderObject(obj: MObj,
                  backend:str,
                  renderoptions: scoring.render.RenderOptions,
                  scorestruct: ScoreStruct,
                  config: CoreConfig
                  ) -> scoring.render.Renderer:
    """
    Render an object

    NB: we put it here in order to make it easier to cache rendering, if needed

    All args must be given (not None) so that caching is meaningful.

    Args:
        obj: the object to make the image from (a Note, Chord, etc.)
        backend : one of 'musicxml', 'lilypond'
        scorestruct: if given, this ScoreStruct will be used for rendering. Otherwise
            the scorestruct within the active Workspace is used
        config: if given, this config is used for rendering. Otherwise the config
            within the active Workspace is used

    Returns:
        a scoring.Renderer. The returned object can be used to render (via the
        :meth:`~maelzel.scoring.render.Renderer.write` method) or to have access to the
        generated score (see :meth:`~maelzel.scoring.render.Renderer.nativeScore`)

    .. note::

        To render with a temporary Wrokspace (i.e., without modifying the active Workspace),
        use::

        .. code-block:: python

            with Workspace(scorestruct=..., config=..., ...) as w:
                renderObject(myobj, "outfile.pdf")
    """
    assert backend and renderoptions and scorestruct and config
    logger.debug(f"rendering parts with backend: {backend}")
    parts = obj.scoringParts()
    renderer = notation.renderWithActiveWorkspace(parts, backend=backend,
                                                  renderoptions=renderoptions,
                                                  scorestruct=scorestruct,
                                                  config=config)
    return renderer


def resetImageCache() -> None:
    """
    Reset the image cache. Useful when changing display format
    """
    _renderImage.cache_clear()
