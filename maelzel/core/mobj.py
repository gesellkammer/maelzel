"""
Musical Objects
---------------

Time
~~~~

A MObj has always an offset and dur attribute. The offset can be unset (None).
They refer to an abstract time. When visualizing a MObj as musical notation
these times are interpreted/converted to beats and score locations based on
a score structure.

"""

#
# Internal notes
#
# offset
#
# Each object has an offset. This offset can be None if not explicitly set
# by the object itself. A cached object, ._resolvedOffset, can be set by
# either the object itself or by the parent
#
# dur
#
# Each object has a duration (.dur). The duration is always explicit. It is implemented
# as a property since it might be calculated.
#
# TODO: revise this docs
#
# * _calculateDuration: this method should return the duration in beats or None if the
#   object itself cannot determine its own duration
# * the parent should always be able to determine the duration of a child. If the object
#   has no implicit duration, _calculateDuration is called and if this returns None,
#   a default duration is set.
#


from __future__ import annotations
import functools
import os
import tempfile as _tempfile
import html as _html
from dataclasses import dataclass

import emlib.misc
import emlib.img
import pitchtools as pt
import csoundengine

from maelzel.common import asmidi, F, asF, F0, F1
from maelzel.textstyle import TextStyle

from ._common import logger
from ._typedefs import *
from .config import CoreConfig
from .workspace import Workspace
from .synthevent import PlayArgs, SynthEvent

from . import playback
from . import offline
from . import proxysynth
from . import environment
from . import notation
from . import symbols as _symbols
from . import _dialogs
from . import _tools

from maelzel import scoring
from maelzel.scorestruct import ScoreStruct

from typing import TypeVar, Any, Callable

MObjT = TypeVar('MObjT', bound='MObj')


__all__ = (
    'MObj',
    'MContainer',
    'resetImageCache',
)


@dataclass
class _TimeScale:
    factor: F
    offset: F

    def __call__(self, t: num_t):
        r = asF(t)
        return r*self.factor + self.offset


class MContainer:
    """
    An interface for any class which can be a parent

    Implemented downstream by classes like Chain or Score.
    """
    def eventAfter(self, event: MObj) -> MObj | None:
        """
        Returns the next event after *event*

        This method only makes sense when the container is an horizontal
        container (Chain, Voice). *event* and the returned event are
        always some MEvent (see maelzel.core.event)
        """
        raise NotImplementedError

    def childOffset(self, child: MObj) -> F:
        """The offset of child relative to this parent"""
        raise NotImplementedError

    def absOffset(self) -> F:
        """The absolute offset of this container"""
        raise NotImplementedError

    def scorestruct(self) -> ScoreStruct | None:
        """The scorestructure attached to this container (if any)

        This query will be passed along upstream"""
        raise NotImplementedError

    def _childChanged(self, child: MObj) -> None:
        """
        This should be called by a child when changed

        Not all changes are relevant to a parent. In particular only
        changes regarding offset or duration should be signaled

        Args:
            child: the modified child

        """
        pass

    def _update(self) -> None:
        raise NotImplementedError

    def _resolveGlissandi(self) -> None:
        raise NotImplementedError

    def itemAfter(self, item: MObj) -> MObj | None:
        """Returns the item after *item*, if any (None otherwise)"""
        pass



class MObj:
    """
    This is the base class for all core objects.

    This is an abstract class. **It should not be instantiated by itself**.
    A :class:`MObj` can display itself via :meth:`show` and play itself via :meth:`play`.
    Any MObj has a duration (:attr:`dur`) and a time offset (:attr:`offset`). The offset
    can be left as ``None``, indicating that it is not explicitely set, in which case it
    will be calculated from the context. In the case of events or chains, which can be
    contained within other objects, the offset depends on the previous objects. The
    resolved (implicit) offset can be queried via :meth:`MObj.relOffset`. This offset
    is relative to the parent, or an absolute offset if the object has no parent. The absolute
    offset can be queried via :meth:`MObj.absOffset`.

    A :class:`MObj` can customize its playback via :meth:`setPlay`. The playback attributes can
    be accessed through the `playargs` attribute

    Elements purely related to notation (text annotations, articulations, etc)
    are added to a :class:`MObj` through the :meth:`addSymbol` and can be accessed
    through the :attr:`MObj.symbols` attribute. A symbol is an attribute or notation
    element (like color, size or an attached text expression) which has meaning only
    in the realm of graphical representation.

    Args:
        dur: the duration of this object, in quarternotes
        offset: the (optional) time offset of this object, in quarternotes
        label: a string label to identify this object, if necessary
    """
    _acceptsNoteAttachedSymbols = True
    _isDurationRelative = True

    __slots__ = ('_parent', '_dur', 'offset', 'label', 'playargs', 'symbols',
                 '_scorestruct', 'properties', '_resolvedOffset',
                 '__weakref__')

    def __init__(self,
                 dur: time_t = None,
                 offset: time_t = None,
                 label: str = '',
                 parent: MObj = None,
                 properties: dict[str, Any] = None,
                 symbols: list[_symbols.Symbol] | None = None):

        self._parent: MContainer | None = parent
        "The parent of this object (or None if it has no parent)"

        self.label = label
        "a label can be used to identify an object within a group of objects"

        self._dur: F | None = dur
        "the duration of this object (can be None, in which case it is unset)"

        self.offset: F | None = offset
        "offset specifies a time offset for this object"

        self.symbols: list[_symbols.Symbol] | None = symbols
        "A list of all symbols added via :meth:`addSymbol` (None by default)"

        # playargs are set via .setPlay and serve the purpose of
        # attaching playing parameters (like pan position, instrument)
        # to an object
        self.playargs: PlayArgs | None = None
        "playargs are set via :meth:`.setPlay` and are used to customize playback (instr, gain, …). None by default"

        self.properties: dict[str, Any] | None = properties
        """
        User-defined properties as a dict (None by default). Set them via :meth:`~maelzel.core.mobj.MObj.setProperty`
        """

        self._scorestruct: ScoreStruct | None = None
        self._resolvedOffset: F | None = None
        
        assert dur is None or dur >= 0, f"A Duration cannot be negative: {self}"

    @property
    def dur(self) -> F:
        """The duration of this object, in quarternotes"""
        d = self._dur
        return F1 if d is None else d

    @dur.setter
    def dur(self, dur: time_t):
        self._dur = asF(dur)
        self._changed()

    @property
    def parent(self) -> MContainer | None:
        """The parent of this object.

        This attribute is set by the parent when an object is added to it. For
        example, when adding a Note to a Chain, the Chain is set as the parent
        of the Note. This enables the Note to query information about the parent,
        like its absolute position or if a score structure has been set upstream"""
        return self._parent

    @parent.setter
    def parent(self, parent: MContainer):
        self._parent = parent

    def _copyAttributesTo(self, other: MObj) -> None:
        if self.symbols:
            other.symbols = self.symbols.copy()
        if self.playargs:
            other.playargs = self.playargs.copy()
        if self.properties:
            other.properties = self.properties.copy()

    def setProperty(self: MObjT, key: str, value) -> MObjT:
        """
        Set a property, returns self

        Any MObj can have user-defined properties. These properties are optional:
        before any property is created the :attr:`.properties <properties>` attribute
        is ``None``. This method creates the dict if it is still None and
        sets the property.

        Args:
            key: the key to set
            value: the value of the property

        Returns:
            self (similar to setPlay or setSymbol, to allow for chaining calls)

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> n = Note("4C", 1)
            >>> n.setProperty('foo', 'bar')
            4C:1♩
            >>> # To query a property do:
            >>> n.getProperty('foo')
            bar
            >>> # Second Method: query the properties attribute directly.
            >>> # WARNING: if no properties were set, this attribute would be None
            >>> if n.properties:
            ...     foo = n.properties.get('foo')
            ...     print(foo)
            bar

        .. seealso:: :meth:`~MObj.getProperty`, :attr:`~MObj.properties`
        """
        if self.properties is None:
            self.properties = {key: value}
        else:
            self.properties[key] = value
        return self

    def getPlay(self, key: str, default=None):
        """
        Get a playback attribute previously set via :meth:`MObj.setPlay`

        Args:
            key: the key (see  setPlay for possible keys)
            default: the value to return if the given key has not been set

        Returns:
            either the value previously set, or default otherwise.
        """
        if not self.playargs:
            return default
        return self.playargs.get(key, default)

    def getProperty(self, key: str, default=None):
        """
        Get a property of this objects

        Any MObj can have multiple properties. A property is a key:value pair,
        where the key is a string and the value can be anything. Properties can
        be used to attach information to an object, to tag it in any way needed.

        Properties are set via :meth:`MObj.setProperty`. The :attr:`MObj.properties`
        attribute can be queries directly, but bear in mind that **if no properties have
        been set, this attribute is ``None`` by default**.

        Args:
            key: the property to query
            default: returned value if the property has not been set

        Returns:
            the value of the property, or the default value

        .. seealso:: :meth:`~MObj.setProperty`, :attr:`~MObj.properties`
        """
        if not self.properties:
            return default
        return self.properties.get(key, default)

    def meanPitch(self) -> float | None:
        """
        The mean pitch of this object

        Returns:
            The mean pitch of this object
        """
        pitchrange = self.pitchRange()
        if pitchrange is None:
            return None
        minpitch, maxpitch = pitchrange
        return (maxpitch + minpitch) / 2.

    def pitchRange(self) -> tuple[float, float] | None:
        """
        The pitch range of this object, if applicable

        This is useful in order to assign this object to a proper Voice
        when distributing objects among voices

        Returns:
            either None or a tuple (lowest pitch, highest pitch)
        """
        raise NotImplementedError

    def _detachedOffset(self, default=None) -> F | None:
        """
        The explicit or implicit offset (if it has been resolved), or default otherwise

        This method does not call the parent

        Args:
            default: value returned if this object has no explicit or implicit default

        Returns:
             the explicit or implicit offset, or *default* otherwise
        """
        return _ if (_:=self.offset) is not None else _ if (_:=self._resolvedOffset) is not None else default

    def resolveEnd(self) -> F:
        """
        Returns the resolved end of this object

        An object's offset and duration can be explicit (set in their corresponding
        ``.offset`` or ``.dur`` attributes) or implicit, as calculated from the context

        Returns:
            the resolved end of this object, relative to its parent
        """
        return self.relOffset() + self.dur

    def relOffset(self) -> F:
        """
        Resolve the offset of this object, relative to its parent

        If this object has no parent the offset is an absolute offset.

        The ``.offset`` attribute holds the explicit offset. If this attribute
        is unset (``None``) this object might ask its parent to determine the
        offset based on the durations of any previous objects

        Returns:
            the offset, in quarter notes. If no explicit or implicit
            offset and the object has no parent it returns 0.

        .. seealso:: :meth:`MObj.absOffset`
        """
        if (offset := self.offset) is not None:
            return offset
        elif (offset := self._resolvedOffset) is not None:
            return offset
        elif self.parent:
            self._resolvedOffset = offset = self.parent.childOffset(self)
            return offset
        else:
            return F0

    def absOffset(self) -> F:
        """
        Returns the absolute offset of this object in quarternotes

        If this object is embedded (has a parent) in a container,
        its absolute offset depends on the offset of its parent,
        recursively. If the object has no parent then the absolute offset
        is just the resolved offset

        Returns:
            the absolute start position of this object

        """
        offset = self.relOffset()
        return offset + self.parent.absOffset() if self.parent else offset

    def withExplicitTimes(self, forcecopy=False):
        """
        Copy of self with explicit times

        If self already has explicit offset and duration, self itself
        is returned. If you relie on the fact that this method returns
        a copy, use ``forcecopy=True``

        Args:
            forcecopy: if forcecopy, a copy of self will be returned even
                if self already has explicit times

        Returns:
            a clone of self with explicit times

        Example
        ~~~~~~~

            >>> n = None("4C", dur=0.5)
            >>> n
            4C:0.5♩
            >>> n.offset is None
            True
            # An unset offset resolves to 0
            >>> n.withExplicitTimes()
            4C:0.5♩:offset=0
            # An unset duration resolves to 1 quarternote beat
            >>> Note("4C", offset=2.5).withExplicitTimes()
            4C:1♩:offset=2.5

        """
        if self.offset is not None and not forcecopy:
            return self
        return self.clone(offset=self.relOffset())

    def setPlay(self: MObjT, /, **kws) -> MObjT:
        """
        Set any playback attributes, returns self

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
        chan: ``int``                 The channel to output to, **channels start at 1**
        fade: ``float``               The fade time; can also be a tuple (fadein, fadeout)
        fadeshape: ``str``            One of 'linear', 'cos', 'scurve'
        pitchinterpol: ``str``        One of 'linear', 'cos', 'freqlinear', 'freqcos'
        gain: ``float``               A gain factor applied to the amplitud of this object.
                                      **Dynamic argument** (*kgain*)
        position: ``float``           Dynamic argument. Panning position (0=left, 1=right).
                                      **Dynamic argument** (*kpos*)
        skip: ``float``               Skip time of playback; allows to play a fragment of the object.
                                      **NB**: set the delay to the -skip to start playback at the
                                      original time but from the timepoint specified by the skip param
        end: ``float``                End time of playback; counterpart of `skip`, allow to
                                      trim playback of the object
        sustain: ``float``            An extra sustain time. This is useful for sample based
                                      instruments
        transpose: ``float``          Transpose the pitch of this object **only for playback**
        glisstime: ``float``          The duration (in beats) of the glissando for events with
                                      glissando. A short glisstime can be used for legato playback
                                      in non-percusive instruments
        priority: ``int``             The order of evaluation. Events scheduled with a higher
                                      priority are evaluated later in the chain
        args: ``dict``                Named arguments passed to the playback instrument
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
        if playargs is None:
            self.playargs = playargs = PlayArgs()
        if glide := kws.pop('gliss', None):
            glisstime = float(glide) if glide is not True else min(0.1, self.dur*F(3, 4))
            kws['glisstime'] = glisstime
            if not self.gliss:
                self.gliss = True
                self.addSymbol(_symbols.GlissProperties(hidden=True))
        playargs.update(kws)
        return self

    def clone(self: MObjT,
              **kws) -> MObjT:
        """
        Clone this object, changing parameters if needed

        Args:
            **kws: any keywords passed to the constructor

        Returns:
            a clone of this object, with the given arguments changed

        Example::

            >>> from maelzel.core import *
            >>> a = Note("C4+", dur=1)
            >>> b = a.clone(dur=0.5)
        """
        out = self.copy()
        for k, v in kws.items():
            if k == 'offset':
                v = asF(v)
            setattr(out, k, v)

        self._copyAttributesTo(out)
        return out

    def copy(self: MObjT) -> MObjT:
        """Returns a copy of this object"""
        raise NotImplementedError

    def timeShift(self: MObjT, timeoffset: time_t) -> MObjT:
        """
        Return a copy of this object with an added offset

        Args:
            timeoffset: a delta time added

        Returns:
            a copy of this object shifted in time by the given amount
        """
        timeoffset = asF(timeoffset)
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
    def end(self) -> F | None:
        """
        The end time of this object.

        Will be None if this object has no explicit offset
        """
        return None if self.offset is None else self.offset + self.dur

    def quantizePitch(self: MObjT, step=0.) -> MObjT:
        """ Returns a new object, with pitch rounded to step """
        raise NotImplementedError()

    def transposeByRatio(self: MObjT, ratio: float) -> MObjT:
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

    def show(self,
             fmt: str = None,
             external: bool = None,
             backend: str = None,
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
                To change the default, modify :ref:`config['openImagesInExternalApp'] <config_openImagesInExternalApp>`
            backend: backend used when rendering to png/pdf.
                One of 'lilypond', 'musicxml'. None to use default
                (see :ref:`config['show.backend'] <config_show_backend>`)
            fmt: one of 'png', 'pdf', 'ly'. None to use default.
            scorestruct: if given overrides the current/default score structure
            config: if given overrides the current/default config
            resolution: dpi resolution when rendering to an image, overrides the
                :ref:`config key 'show.pngResolution' <config_show_pngresolution>`
        """
        cfg = config or Workspace.active.config
        if resolution:
            cfg = cfg.clone({'show.pngResolution': resolution})

        if external is None:
            external = cfg['openImagesInExternalApp']

        if backend is None:
            backend = cfg['show.backend']

        #elif backend != cfg['show.backend']:
        #    cfg = cfg.clone({'show.backend': backend})

        if fmt is None:
            fmt = 'png' if not external and environment.insideJupyter else cfg['show.format']
        if fmt == 'ly':
            renderer = self.render(backend='lilypond', scorestruct=scorestruct, config=cfg)
            if external:
                lyfile = _tempfile.mktemp(suffix=".ly")
                renderer.write(lyfile)
                emlib.misc.open_with_app(lyfile)
            else:
                _tools.showLilypondScore(renderer.render())
        else:
            img = self._renderImage(backend=backend, fmt=fmt, scorestruct=scorestruct,
                                    config=cfg)
            if fmt == 'png':
                scalefactor = cfg['show.scaleFactor']
                if backend == 'musicxml':
                    scalefactor *= cfg['show.scaleFactorMusicxml']
                _tools.pngShow(img, forceExternal=external, scalefactor=scalefactor)
            else:
                emlib.misc.open_with_app(img)

    def _changed(self) -> None:
        """
        This method is called whenever the object changes its representation

        This happens when a note changes its pitch inplace, the duration is modified, etc.
        """
        if self.parent:
            self.parent._childChanged(self)

    def quantizedScore(self,
                       scorestruct: ScoreStruct = None,
                       config: CoreConfig = None,
                       quantizationProfile: str | scoring.quant.QuantizationProfile = None,
                       enharmonicOptions: scoring.enharmonics.EnharmonicOptions = None,
                       nestedTuplets: bool | None = None
                       ) -> scoring.quant.QuantizedScore:
        """
        Returns a QuantizedScore representing this object

        Args:
            scorestruct: if given it will override the scorestructure active for this object
            config: if given will override the active config
            quantizationProfile: if given it is used to customize the quantization process.
                Otherwise, a profile is constructed based on the config. It is also possible
                to pass the name of a quantization preset (possible values: 'lowest', 'low',
                'medium', 'high', 'highest', see :meth:`maelzel.scoring.quant.QuantizationProfile.fromPreset`)
            enharmonicOptions: if given it is used to customize enharmonic respelling.
                Otherwise, the enharmonic options used for respelling are constructed based
                on the config
            nestedTuplets: if given, overloads the config value 'quant.nestedTuplets'. This is used
                to disallow nested tuplets, mainly for rendering musicxml since there are some music
                editors (MuseScore, for example) which fail to import nested tuplets. This can be set
                at the config level as ``getConfig()['quant.nestedTuplets'] = False``

        Returns:
            a quantized score. To render such a quantized score as notation call
            its :meth:`~maelzel.scoring.quant.QuantizedScore.render` method

        A QuantizedScore contains a list of QuantizedParts, which each consists of
        list of QuantizedMeasures. To access the recursive notation structure of each measure
        call its :meth:`~maelzel.scoring.QuantizedMeasure.asTree` method
        """
        w = Workspace.active
        if config is None:
            config = w.config

        if not scorestruct:
            scorestruct = self.scorestruct() or w.scorestruct
        if quantizationProfile is None:
            quantizationProfile = config.makeQuantizationProfile()
        elif isinstance(quantizationProfile, str):
            quantizationProfile = scoring.quant.QuantizationProfile.fromPreset(quantizationProfile)
        else:
            assert isinstance(quantizationProfile, scoring.quant.QuantizationProfile)

        if nestedTuplets is not None:
            quantizationProfile.nestedTuplets = nestedTuplets

        parts = self.scoringParts()
        if config['show.respellPitches'] and enharmonicOptions is None:
            enharmonicOptions = config.makeEnharmonicOptions()
        qscore = scoring.quant.quantize(parts,
                                        struct=scorestruct,
                                        quantizationProfile=quantizationProfile,
                                        enharmonicOptions=enharmonicOptions)
        return qscore

    def render(self,
               backend: str = None,
               renderoptions: scoring.render.RenderOptions = None,
               scorestruct: ScoreStruct = None,
               config: CoreConfig = None,
               quantizationProfile: str | scoring.quant.QuantizationProfile = None
               ) -> scoring.render.Renderer:
        """
        Renders this object as notation

        First the object is quantized to abstract notation, then it is passed to the backend
        to render it for the specific format (lilypond, musicxml, ...), which is then used
        to generate a document (pdf, png, ...)

        Args:
            backend: the backend to use, one of 'lilypond', 'musicxml'. If not given,
                defaults to the :ref:`config key 'show.backend' <config_show_backend>`
            renderoptions: the render options to use. If not given, these are generated from
                the active config
            scorestruct: if given, overrides the scorestruct set within the active Workspace
                and any scorestruct attached to this object
            config: if given, overrides the active config
            quantizationProfile: if given, it is used to customize the quantization process
                and will override any config option related to quantization.
                A QuantizationProfile can be created from a config via
                :meth:`maelzel.core.config.CoreConfig.makeQuantizationProfileFromPreset`.

        Returns:
            a scoring.render.Renderer. This can be used to write the rendered structure
            to an image (png, pdf) or as a musicxml or lilypond file.

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> voice = Voice(...)
            # Render with the settings defined in the config
            >>> voice.render()
            # Customize the rendering process
            >>> from maelzel.scoring.renderer import RenderOptions
            >>> from maelzel.scoring.quant import QuantizationProfile
            >>> quantprofile = QuantizationProfile.simple(
            ...     possibleDivisionsByTempo={80: []
            ...     })
        """
        w = Workspace.active
        if config is None:
            config = w.config
        if not backend:
            backend = config['show.backend']
        if not renderoptions:
            renderoptions = notation.makeRenderOptionsFromConfig(config)
        if not scorestruct:
            scorestruct = self.scorestruct() or w.scorestruct
        if quantizationProfile is not None:
            if isinstance(quantizationProfile, str):
                quantizationProfile = scoring.quant.QuantizationProfile.fromPreset(quantizationProfile)
            else:
                assert isinstance(quantizationProfile, scoring.quant.QuantizationProfile)

        return _renderObject(self, backend=backend, renderoptions=renderoptions,
                             scorestruct=scorestruct, config=config,
                             quantizationProfile=quantizationProfile)

    def _renderImage(self,
                     backend: str = None,
                     outfile: str = None,
                     fmt="png",
                     scorestruct: ScoreStruct = None,
                     config: CoreConfig = None
                     ) -> str:
        """
        Creates an image representation, returns the path to the image

        Args:
            backend: the rendering backend. One of 'musicxml', 'lilypond'
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

        .. seealso:: :meth:`~maelzel.core.mobj.MObj.render`
        """
        w = Workspace.active
        if not config:
            config = w.config
        if backend is None:
            backend = config['show.backend']
        if fmt == 'ly':
            backend = 'lilypond'
        if scorestruct is None:
            scorestruct = self.scorestruct() or w.scorestruct

        path = _renderImage(self, outfile, fmt=fmt, backend=backend, scorestruct=scorestruct,
                            config=config)
        if not os.path.exists(path):
            # cached image does not exist?
            logger.debug(f"_renderImage did not return an existing path for object {self}, "
                         f"returned path: {path}. This might be a cached path and the cache "
                         f"might be invalid. Resetting the cache and trying again...")
            resetImageCache()
            # Try again, uncached
            path = _renderImage(self, outfile, fmt=fmt, backend=backend, scorestruct=scorestruct,
                                config=config)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Could not generate image, returned image file '{path}' "
                                        f"does not exist")
            else:
                logger.debug(f"... resetting the cache worked, an image file '{path}' "
                             f"was generated")
        return path

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig = None,
                      parentOffset: F | None = None
                      ) -> list[scoring.Notation]:
        """
        Returns its notated form as scoring.Notations

        These can then be converted into notation via some of the available
        backends: musicxml or lilypond

        Args:
            groupid: passed by an object higher in the hierarchy to
                mark this objects as belonging to a group
            config: a configuration to customize rendering
            parentOffset: if given this should be the absolute offset of this object's parent

        Returns:
            A list of scoring.Notation which best represent this
            object as notation
        """
        raise NotImplementedError("Subclass should implement this")

    def scoringParts(self,
                     config: CoreConfig = None
                     ) -> list[scoring.UnquantizedPart]:
        """
        Returns this object as a list of scoring UnquantizedParts.

        Args:
            config: if given, this config instead of the active config will
                be used

        Returns:
            a list of unquantized parts

        This method is used internally to generate the parts which
        constitute a given MObj prior to rendering,
        but might be of use itself so it is exposed here.

        An :class:`maelzel.scoring.UnquantizedPart` is an intermediate format used by the scoring
        package to represent notated events. It represents a list of non-simultaneous Notations,
        unquantized and independent of any score structure
        """
        notations = self.scoringEvents(config=config or Workspace.active.config)
        if not notations:
            return []
        scoring.resolveOffsets(notations)
        parts = scoring.distributeNotationsByClef(notations)
        return parts

    def unquantizedScore(self, title: str = None) -> scoring.UnquantizedScore:
        """
        Create a maelzel.scoring.UnquantizedScore from this object

        Args:
            title: the title of the resulting score (if given)

        Returns:
            the Arrangement representation of this object

        An :class:`~maelzel.scoring.UnquantizedScore` is a list of
        :class:`~maelzel.scoring.UnquantizedPart`, which is itself a list of
        :class:`~maelzel.scoring.Notation`. An :class:`Arrangement` represents
        an **unquantized** score, meaning that the Notations within each part are
        not split into measures, nor organized in beats. To generate a quantized score
        see :meth:`~MObj.quantizedScore`

        This method is mostly used internally when an object is asked to be represented
        as a score. In this case, an Arrangement is created first, which is then quantized,
        generating a :class:`~maelzel.scoring.quant.QuantizedScore`

        .. seealso:: :meth:`~MObj.quantizedScore`, :class:`~maelzel.scoring.quant.QuantizedScore`

        """
        parts = self.scoringParts()
        return scoring.UnquantizedScore(parts, title=title)

    def _scoringAnnotation(self, text: str = None, config: CoreConfig = None
                           ) -> scoring.attachment.Text:
        """ Returns owns annotations as a scoring Annotation """
        if config is None:
            config = Workspace.active.config
        if text is None:
            assert self.label
            text = self.label
        labelstyle = TextStyle.parse(config['show.labelStyle'])
        return scoring.attachment.Text(text,
                                       fontsize=labelstyle.fontsize,
                                       italic=labelstyle.italic,
                                       weight='bold' if labelstyle.bold else '',
                                       color=labelstyle.color)

    def scorestruct(self) -> ScoreStruct | None:
        """
        Returns the ScoreStruct active for this obj or its parent (recursively)

        If this object has no parent ``None`` is returned

        Example
        ~~~~~~~

        .. code-block:: python

            >>> from maelzel.core import *
            >>> n = Note("4C", 1)
            >>> voice = Voice([n])
            >>> score = Score([voice])
            >>> score.setScoreStruct(ScoreStruct(timesig=(3, 4), tempo=72))
            >>> n.scorestruct()
            ScoreStruct(timesig=(3, 4), tempo=72)
        """
        return self._scorestruct or (self.parent.scorestruct() if self.parent else None)

    def write(self,
              outfile: str,
              backend: str = None,
              resolution: int = None
              ) -> None:
        """
        Export to multiple formats

        Formats supported: pdf, png, musicxml (extension: .xml or .musicxml),
        lilypond (.ly), midi (.mid or .midi)

        Args:
            outfile: the path of the output file. The extension determines
                the format. Formats available are pdf, png, lilypond, musicxml,
                midi and csd.
            backend: the backend used when writing as pdf or png. If not given,
                the default defined in the active config is used
                (:ref:`key: 'show.backend' <config_show_backend>`).
                Possible backends: ``lilypond``; ``musicxml`` (uses MuseScore to render musicxml as
                image so MuseScore needs to be installed)
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
        cfg = Workspace.active.config
        if ext == '.ly' or ext == '.mid' or ext == '.midi':
            backend = 'lilypond'
        elif ext == '.xml' or ext == '.musicxml':
            backend = 'musicxml'
        elif ext == '.csd':
            renderer = self._makeOfflineRenderer()
            renderer.writeCsd(outfile)
            return
        elif backend is None:
            backend = cfg['show.backend']
        if resolution is not None:
            cfg = cfg.clone(updates={'show.pngResolution': resolution})
        r = notation.renderWithActiveWorkspace(self.scoringParts(config=cfg),
                                               backend=backend,
                                               scorestruct=self.scorestruct(),
                                               config=cfg)
        r.write(outfile)

    def _htmlImage(self) -> str:
        imgpath = self._renderImage()
        if not imgpath:
            return ''
        scaleFactor = Workspace.active.config.get('show.scaleFactor', 1.0)
        width, height = emlib.img.imgSize(imgpath)
        img = emlib.img.htmlImgBase64(imgpath,
                                      width=f'{int(width * scaleFactor)}px')
        return img

    def _repr_html_header(self):
        return _html.escape(repr(self))

    def _repr_html_(self) -> str:
        img = self._htmlImage()
        txt = self._repr_html_header()
        return rf'<code style="white-space: pre-line; font-size:0.9em;">{txt}</code><br>' + img

    def _makeOfflineRenderer(self,
                             sr: int | None = None,
                             numchannels=2,
                             eventoptions={}
                             ) -> offline.OfflineRenderer:
        r = offline.OfflineRenderer(sr=sr, numchannels=numchannels)
        events = self.events(**eventoptions)
        r.schedEvents(coreevents=events)
        return r

    def dump(self, indents=0, forcetext=False):
        """
        Prints all relevant information about this object

        Args:
            indents: number of indents
            forcetext: if True, force text output via print instead of html
                even when running inside jupyter
        """
        print(f'{"  "*indents}{repr(self)}')
        if self.playargs:
            print(f'{"  "*(indents+1)}{self.playargs}')

    def _synthEvents(self,
                     playargs: PlayArgs,
                     parentOffset: F,
                     workspace: Workspace,
                     ) -> list[SynthEvent]:
        """
        Must be overriden by each class to generate SynthEvents

        Args:
            playargs: a :class:`PlayArgs`, structure, filled with given values,
                own .playargs values and config defaults (in that order)
            parentOffset: the absolute offset of the parent of this object.
                If the object has no parent, this will be F(0)
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
               fade: float | tuple[float, float] = None,
               fadeshape: str = None,
               position: float = None,
               skip: float = None,
               end: float = None,
               sustain: float = None,
               workspace: Workspace = None,
               transpose: float = 0.,
               **kwargs
               ) -> list[SynthEvent]:
        """
        Returns the SynthEvents needed to play this object

        All these attributes here can be set previously via `playargs` (or
        using :meth:`~maelzel.core.mobj.MObj.setPlay`)

        Args:
            gain: modifies the own amplitude for playback/recording (0-1)
            delay: delay in seconds, added to the start of the object
                As opposed to the .offset attribute of each object, which is defined
                in quarternotes, the delay is always in seconds
            instr: which instrument to use (see defPreset, definedPresets). Use "?" to
                select from a list of defined presets.
            chan: the channel to output to. **Channels start at 1**
            pitchinterpol: 'linear', 'cos', 'freqlinear', 'freqcos'
            fade: fade duration in seconds, can be a tuple (fadein, fadeout)
            fadeshape: 'linear' | 'cos'
            args: named arguments passed to the note. A dict ``{paramName: value}``
            position: the panning position (0=left, 1=right)
            skip: start playback at the given offset (in quarternotes), relative
                to the start of the object. Allows to play a fragment of the object
                (NB: this trims the playback of the object. Use `delay` to offset
                the playback in time while keeping the playback time unmodified)
            end: end time of playback, in quarternotes. Allows to play a fragment of the object by trimming the end of the playback
            sustain: a time added to the playback events to facilitate overlapping/legato between
                notes, or to allow one-shot samples to play completely without being cropped.
            workspace: a Workspace. If given, overrides the current workspace. It's scorestruct
                is used to determine the mapping between beat-time and real-time.
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
            from .presetmanager import presetManager
            instr = presetManager.selectPreset()
            if not instr:
                raise ValueError("No preset selected")

        pairs = (
            ('instr', instr),
            ('delay', delay),
            ('args', args),
            ('gain', gain),
            ('chan', chan),
            ('pitchinterpol', pitchinterpol),
            ('fade', fade),
            ('fadeshape', fadeshape),
            ('position', position),
            ('sustain', sustain),
            ('transpose', transpose),
        )

        d = {k: v for k, v in pairs if v is not None}

        if kwargs:
            if args:
                args.update(kwargs)
            else:
                d['args'] = kwargs

        if workspace is None:
            workspace = Workspace.active

        if (struct := self.scorestruct()) is not None:
            workspace = workspace.clone(scorestruct=struct, config=workspace.config)

        playargs = PlayArgs.makeDefault(workspace.config)
        if d:
            playargs.update(d)

        parentOffset = self.parent.absOffset() if self.parent else F(0)

        events = self._synthEvents(playargs=playargs, parentOffset=parentOffset,
                                   workspace=workspace)

        if skip is not None or end is not None:
            delay = playargs.get('delay')
            struct = workspace.scorestruct
            skiptime = 0. if skip is None else struct.beatToTime(skip)
            endtime = float("inf") if end is None else struct.beatToTime(end)
            events = SynthEvent.cropEvents(events, skip=skiptime+delay, end=endtime+delay)

        if event := next((ev for ev in events if ev.delay < 0), None):
            raise ValueError(f"Events cannot have negative delay, event={event}")

        return events

    def play(self,
             instr: str = None,
             delay: float = None,
             args: dict[str, float] = None,
             gain: float = None,
             chan: int = None,
             pitchinterpol: str = None,
             fade: float | tuple[float, float] = None,
             fadeshape: str = None,
             position: float = None,
             skip: float = None,
             end: float = None,
             whenfinished: Callable = None,
             sustain: float = None,
             workspace: Workspace = None,
             transpose: float = 0,
             config: CoreConfig = None,
             display=False,
             **kwargs
             ) -> csoundengine.synth.SynthGroup:

        """
        Plays this object.

        Play is always asynchronous (to block, use some sleep funcion).
        By default, :meth:`play` schedules this event to be renderer in realtime.

        .. note::
            To record events offline, see the example below

        Args:
            gain: modifies the own amplitude for playback/recording (0-1)
            delay: delay in seconds, added to the start of the object
                As opposed to the .offset attribute of each object, which is defined
                in symbolic (beat) time, the delay is always in real (seconds) time
            instr: which instrument to use (see defPreset, definedPresets). Use "?" to
                select from a list of defined presets.
            chan: the channel to output to. **Channels start at 1**
            pitchinterpol: 'linear', 'cos', 'freqlinear', 'freqcos'
            fade: fade duration in seconds, can be a tuple (fadein, fadeout)
            fadeshape: 'linear' | 'cos'
            args: arguments passed to the note. A dict ``{paramName: value}``
            position: the panning position (0=left, 1=right)
            skip: amount of time (in quarternotes) to skip. Allows to play a fragment of
                the object (NB: this trims the playback of the object. Use `delay` to
                offset the playback in time while keeping the playback time unmodified)
            end: end time of playback. Allows to play a fragment of the object by trimming the end of the playback
            sustain: a time added to the playback events to facilitate overlapping/legato between
                notes, or to allow one-shot samples to play completely without being cropped.
            workspace: a Workspace. If given, overrides the current workspace. It's scorestruct
                is used to to determine the mapping between beat-time and real-time. 
            transpose: add a transposition interval to the pitch of this object
            config: if given, overrides the current config
            whenfinished: function to be called when the playback is finished. Only applies to
                realtime rendering
            display: if True and running inside Jupyter, display the resulting synth's html

        Returns:
            A :class:`~csoundengine.synth.SynthGroup`


        .. seealso::
            * :meth:`MObj.events <maelzel.core.mobj.MObj.events>`
            * :meth:`MObj.rec <maelzel.core.mobj.MObj.rec>`
            * :func:`~maelzel.core.playback.render`,
            * :func:`~maelzel.core.playback.play`


        Example
        ~~~~~~~

        Play a note

            >>> from maelzel.core import *
            >>> note = Note(60).play(gain=0.1, chan=2)

        Play multiple objects synchronised

            >>> play(
            ... Note(60, 1.5).events(gain=0.1, position=0.5)
            ... Chord("4E 4G", 2, start=1.2).events(instr='piano')
            ... )

        Or using play as a context managger:

            >>> with play():
            ...     Note(60, 1.5).play(gain=0.1, position=0.5)
            ...     Chord("4E 4G", 2, start=1.2).play(instr='piano')
            ...     ...

        Render offline

            >>> with render("out.wav", sr=44100) as r:
            ...     Note(60, 5).play(gain=0.1, chan=2)
            ...     Chord("4E 4G", 3).play(instr='piano')
        """
        if config is not None:
            workspace = Workspace.active.clone(config=config)
        elif workspace is None:
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
                             skip=skip,
                             end=end,
                             transpose=transpose,
                             **kwargs)

        if not events:
            group = csoundengine.synth.SynthGroup([playback._dummySynth()])
        else:
            renderer = workspace.renderer or playback.RealtimeRenderer()
            group = renderer.schedEvents(events, whenfinished=whenfinished)
            if display and environment.insideJupyter:
                from IPython.display import display
                display(group)
        return group
        # return proxysynth.ProxySynthGroup(group=group)

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
            extratime: float = None,
            workspace: Workspace = None,
            **kws
            ) -> offline.OfflineRenderer:
        """
        Record the output of .play as a soundfile

        Args:
            outfile: the outfile where sound will be recorded. Can be
                None, in which case a filename will be generated. Use '?'
                to open a save dialog
            sr: the sampling rate (:ref:`config key: 'rec.sr' <config_rec_sr>`)
            wait: if True, the operation blocks until recording is finishes
                (:ref:`config 'rec.block' <config_rec_blocking>`)
            nchnls: if given, use this as the number of channels to record.

            gain: modifies the own amplitude for playback/recording (0-1)
            delay: delay in seconds, added to the start of the object
                As opposed to the .start attribute of each object, which is defined
                in symbolic (beat) time, the delay is always in real (seconds) time
            instr: which instrument to use (see defPreset, definedPresets). Use "?" to
                select from a list of defined presets.
            args: named arguments passed to the note. A dict ``{paramName: value}``
            position: the panning position (0=left, 1=right)
            workspace: if given it overrides the active workspace
            extratime: extratime added to the recording (:ref:`config key: 'rec.extratime' <config_rec_extratime>`)
            quiet: if True, do not display any synthesis output

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

        .. seealso:: :class:`~maelzel.core.playback.OfflineRenderer`
        """
        events = self.events(instr=instr, position=position,
                             delay=delay, args=args, gain=gain,
                             workspace=workspace,
                             **kws)
        return offline.render(outfile=outfile, events=events, sr=sr, wait=wait,
                              quiet=quiet, nchnls=nchnls, extratime=extratime)

    def isRest(self) -> bool:
        """
        Is this object a Rest?

        Rests are used as separators between objects inside an Chain or a Track
        """
        return False

    def addSymbol(self: MObjT, *args, **kws) -> MObjT:
        raise NotImplementedError

    def _addSymbol(self, symbol: _symbols.Symbol) -> None:
        if self.symbols is None:
            self.symbols = []

        if self.symbols and symbol.exclusive:
            cls = type(symbol)
            if any(isinstance(s, cls) for s in self.symbols):
                self.symbols = [s for s in self.symbols if not isinstance(s, cls)]
        self.symbols.append(symbol)

    def _removeSymbolsOfClass(self, cls: str | type):
        if isinstance(cls, str):
            cls = cls.lower()
            symbols = [s for s in self.symbols if s.name == cls]
        else:
            symbols = [s for s in self.symbols if isinstance(s, cls)]
        for s in symbols:
            self.symbols.remove(s)

    def getSymbol(self, classname: str) -> _symbols.Symbol | None:
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

    def addText(self: MObjT,
                text: str,
                placement='above',
                italic=False,
                weight='normal',
                fontsize: int = None,
                fontfamily='',
                box=''
                ) -> MObjT:
        """
        Add a text annotation to this object

        This is a shortcut to ``self.addSymbol(symbols.Text(...))``. Use
        that for in-depth customization.

        Args:
            text: the text annotation
            placement: where to place the annotation ('above', 'below')
            italic: if True, use italic as font style
            weight: 'normal' or 'bold'
            fontsize: the size of the annotation
            fontfamily: the font family to use. It is probably best to leave this unset
            box: the enclosure shape, or '' for no box around the text. Possible shapes
                are 'square', 'circle', 'rounded'

        Returns:
            self

        """
        self.addSymbol(_symbols.Text(text, placement=placement, fontsize=fontsize,
                                     italic=italic, weight=weight, fontfamily=fontfamily,
                                     box=box))
        return self

    def timeTransform(self: MObjT, timemap: Callable[[F], F], inplace=False) -> MObjT:
        """
        Apply a time-transform to this object

        Args:
            timemap: a function mapping old time to new time
            inplace: if True changes are applied inplace

        Returns:
            the resulting object (self if inplace)

        .. note::

            time is conceived as abstract 'beat' time, measured in quarter-notes.
            The actual time will be also determined by any tempo changes in the
            active score structure.
        """
        raise NotImplementedError

    def timeShiftInPlace(self, timeoffset: time_t) -> None:
        """
        Shift the time of this by the given offset (inplace)

        Args:
            timeoffset: the time delta (in quarterNotes)
        """
        if self.offset is None:
            raise ValueError("Only objects with an explicit offset can be modified with"
                             "this method")
        self.offset = self.offset + asF(timeoffset)
        self._changed()

    def timeRangeSecs(self,
                      parentOffset: F | None = None,
                      scorestruct: ScoreStruct = None
                      ) -> tuple[F, F]:
        """
        The absolute time range, in seconds

        Args:
            parentOffset: if given, use this offset as parent offset. This is useful
                if the parent's offset has already been calculated
            scorestruct: use this scorestruct to calculate absolute time. This is
                useful if the scorestruct is already known.

        Returns:
            a tuple (absolute start time in seconds, absolute end time in seconds)
        """
        if parentOffset is None:
            absoffset = self.absOffset()
        else:
            absoffset = self._detachedOffset(F0) + parentOffset
        if scorestruct is None:
            scorestruct = self.scorestruct() or Workspace.active.scorestruct
        return scorestruct.beatToTime(absoffset), scorestruct.beatToTime(absoffset + self.dur)

    def durSecs(self) -> F:
        """
        Returns the duration in seconds according to the active score

        Returns:
            the duration of self in seconds
        """
        _, dursecs = self.timeRangeSecs()
        return dursecs

    def pitchTransform(self: MObjT, pitchmap: Callable[[float], float]) -> MObjT:
        """
        Apply a pitch-transform to this object, returns a copy

        Args:
            pitchmap: a function mapping pitch to pitch

        Returns:
            the object after the transform
        """
        raise NotImplementedError("Subclass should implement this")

    def timeScale(self: MObjT, factor: num_t, offset: num_t = 0) -> MObjT:
        """
        Create a copy with modified timing by applying a linear transformation

        Args:
            factor: a factor which multiplies all durations and start times
            offset: an offset added to all start times

        Returns:
            the modified object
        """
        transform = _TimeScale(asF(factor), offset=asF(offset))
        return self.timeTransform(transform)

    def invertPitch(self: MObjT, pivot: pitch_t) -> MObjT:
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

    def transpose(self: MObjT, interval: int | float) -> MObjT:
        """
        Transpose this object by the given interval

        Args:
            interval: the interval in semitones

        Returns:
            the transposed object
        """
        return self.pitchTransform(lambda pitch: pitch+interval)


@functools.lru_cache(maxsize=1000)
def _renderImage(obj: MObj,
                 outfile: str,
                 fmt: str,
                 backend: str,
                 scorestruct: ScoreStruct,
                 config: CoreConfig):
    renderoptions = notation.makeRenderOptionsFromConfig(config)
    if scorestruct is None:
        scorestruct = Workspace.active.scorestruct
    r = obj.render(backend=backend, renderoptions=renderoptions, scorestruct=scorestruct,
                   config=config)
    if not outfile:
        outfile = _tempfile.mktemp(suffix='.' + fmt)
    r.write(outfile)
    return outfile


def _renderObject(obj: MObj,
                  backend: str,
                  renderoptions: scoring.render.RenderOptions,
                  scorestruct: ScoreStruct,
                  config: CoreConfig,
                  quantizationProfile: scoring.quant.QuantizationProfile | None = None
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
    renderer = notation.renderWithActiveWorkspace(parts,
                                                  backend=backend,
                                                  renderoptions=renderoptions,
                                                  scorestruct=scorestruct,
                                                  config=config,
                                                  quantizationProfile=quantizationProfile)
    return renderer


def resetImageCache() -> None:
    """
    Reset the image cache. Useful when changing display format
    """
    _renderImage.cache_clear()
