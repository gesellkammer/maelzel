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
# by the object itself. A cached offset, ._resolvedOffset, can be set by
# either the object itself or by the parent
#
# dur
#
# Each object has a duration (.dur). The duration is always explicit. It is implemented
# as a property since it might be calculated.


from __future__ import annotations
import functools
from abc import ABC, abstractmethod
import os
import math
import re
import html as _html
from dataclasses import dataclass

from maelzel.common import asmidi, F, asF, F0

from maelzel.core._common import logger

from .config import CoreConfig
from .workspace import Workspace
from . import environment
from . import realtimerenderer
from . import notation
from . import _tools
from .synthevent import PlayArgs, SynthEvent

from maelzel import _util
from maelzel import scoring

import typing as _t
if _t.TYPE_CHECKING:
    from typing_extensions import Self
    from matplotlib.axes import Axes
    from . import symbols as _symbols
    from maelzel.common import location_t, beat_t, time_t, num_t
    from maelzel.core import chain
    import maelzel.core.event as _event
    from maelzel.scoring.renderoptions import RenderOptions
    from maelzel.scoring.render import  Renderer
    from maelzel.scoring import quant
    from maelzel.scoring import enharmonics
    import csoundengine
    import csoundengine.synth
    from . import offline
    from maelzel.scorestruct import ScoreStruct



__all__ = (
    'MObj',
    'MContainer',
    'clearImageCache',
)


@dataclass
class _TimeScale:
    factor: F
    offset: F

    def __call__(self, t: num_t):
        r = asF(t)
        return r*self.factor + self.offset

@dataclass
class _PostSymbol:
    symbol: _symbols.Symbol
    """A symbol or spanner"""

    offset: F
    """location to apply the symbol"""

    end: F | None = None
    """Only needed for spanners, end location"""


class MObj(ABC):
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
                 dur: F,
                 offset: F | None = None,
                 label='',
                 parent: MContainer | None = None,
                 properties: dict[str, _t.Any] | None = None,
                 symbols: list[_symbols.Symbol] | None = None):

        if offset is not None and offset < F0:
            raise ValueError(f"Invalid offset: {offset}")

        if dur is None or dur < F0:
            raise ValueError(f"Invalid duration: {dur}")

        self._parent: MContainer | None = parent
        "The parent of this object (or None if it has no parent)"

        self.label: str = label
        "a label can be used to identify an object within a group of objects"

        self._dur: F = dur
        "the duration of this object in quarternotes. It cannot be None"

        self.offset: F | None = offset
        """Optional offset, in quarternotes. Specifies the start time relative to its parent

        It can be None, which indicates that within a container this object would
        start after the previous object. For an object without a parent, the offset
        is an absolute offset. """

        self.symbols: list[_symbols.Symbol] | None = symbols
        "A list of all symbols added via :meth:`addSymbol` (None by default)"

        # playargs are set via .setPlay and serve the purpose of
        # attaching playing parameters (like pan position, instrument)
        # to an object
        self.playargs: PlayArgs | None = None
        "playargs are set via :meth:`.setPlay` and are used to customize playback (instr, gain, …). None by default"

        self.properties: dict[str, _t.Any] | None = properties
        """
        User-defined properties as a dict (None by default). Set them via :meth:`~maelzel.core.mobj.MObj.setProperty`
        """

        self._scorestruct: ScoreStruct | None = None
        self._resolvedOffset: F | None = None

    @abstractmethod
    def __hash__(self) -> int: ...

    @property
    def dur(self) -> F:
        """The duration of this object, in quarternotes"""
        return self._dur

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
        if self._parent is not None and parent is not self._parent:
            if self in self._parent:
                raise ValueError(f"Cannot set the parent for {self}, since "
                                 f"it already is a part of {self._parent}")
        self._parent = parent

    def _copyAttributesTo(self, other: Self) -> None:
        """
        Copy symbols, playargs and properties to other

        Args:
            other: destination object

        """
        if type(other) is not type(self):
            logger.warning(f"Copying attributes to an object of different class, "
                           f"{self=}, {type(self)=}, {other=}, {type(other)=}")
        if self.symbols:
            other.symbols = self.symbols.copy()
        if self.playargs:
            other.playargs = self.playargs.copy()
        if self.properties:
            other.properties = self.properties.copy()

    def setProperty(self, key: str, value) -> Self:
        """
        Set a property, returns self

        An MObj can have user-defined properties. These properties are optional:
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

    def getPlay(self, key: str, default=None, recursive=True):
        """
        Get a playback attribute previously set via :meth:`MObj.setPlay`

        All locally set playback attributes are accessible via the
        :attr:`MEvent.playargs` attribute. This method checks
        not only the locally set attributes, but any attribute set
        by the parent

        Args:
            key: the key (see  setPlay for possible keys)
            default: the value to return if the given key has not been set
            recursive: if True, search the given attribute up the parent chain

        Returns:
            either the value previously set, or default otherwise.
        """
        if self.playargs and (value := self.playargs.db.get(key)) is not None:
            return value
        if not recursive or not self.parent:
            return default
        return self.parent.getPlay(key, default=default, recursive=True)

    def getProperty(self, key: str, default=None):
        """
        Get a property of this objects

        An MObj can have multiple properties. A property is a key:value pair,
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

        .. seealso:: :meth:`setProperty() <maelzel.core.mobj.MObj.setProperty>`, :attr:`properties <maelzel.core.mobj.MObj.properties>`
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

    def _detachedOffset(self, default: F | None = None) -> F | None:
        """
        The explicit or implicit offset (if it has been resolved), or default otherwise

        This method does not call the parent

        Args:
            default: value returned if this object has no explicit or implicit default

        Returns:
             the explicit or implicit offset, or *default* otherwise
        """
        return _ if (_:=self.offset) is not None else _ if (_:=self._resolvedOffset) is not None else default

    def relEnd(self) -> F:
        """
        Resolved end of this object, relative to its parent

        An object's offset can be explicit (set in the ``.offset`` attributes)
        or implicit, as calculated from the context of the parent. For example,
        inside a Chain, the offset of an event depends on the offsets and
        durations of the objects preceding it.

        .. note::

            To calculate the absolute end of an object, use
            ``obj.absOffset() + obj.dur``

        Returns:
            the resolved end of this object, relative to its parent. If this
            object has no parent, the relative end and the absolute end are
            the same

        .. seealso:: :meth:`MObj.relOffset`, :meth:`MObj.absOffset`
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

        .. seealso:: :meth:`absOffset() <maelzel.core.mobj.MObj.absOffset>`
        """

        if (offset := self.offset) is not None:
            return offset
        elif self._resolvedOffset is not None:
            return self._resolvedOffset
        elif self.parent:
            self._resolvedOffset = offset = self.parent._childOffset(self)
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
        return offset if not self.parent else offset + self.parent.absOffset()

    def absEnd(self) -> F:
        """
        Returns the absolute end of this offset, as quarternotes

        If this object is embedded (has a parent) in a container,
        its absolute end depends on the offset of its parent,
        recursively. If the object has no parent then the absolute offset
        is just the resolved offset

        Returns:
            the absolute end position of this object

        """
        return self.absOffset() + self.dur

    def parentAbsOffset(self) -> F:
        """
        The absolute offset of the parent

        Returns:
            the absolute offset of the parent if this object has a parent, else 0
        """
        return self.parent.absOffset() if self.parent else F0

    def withExplicitOffset(self, forcecopy=False) -> Self:
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

    def _asVoices(self) -> list[chain.Voice]:
        raise NotImplementedError

    def plot(self,
             axes: Axes | None = None,
             figsize: tuple[int, int] = (15, 5),
             timeSignatures=True,
             grid=True,
             **kws) -> Axes:
        """
        Plot this object

        To see all supported options, see :func:`maelzel.core.plotting.plotVoices`

        Args:
            axes: use this Axes, if given
            figsize: figure size of the plot, if not axes is given (otherwise
                uses the figure corresponding to the given axes)
            timeSignatures: draw time signatures
            grid: draw a grid
            kws: passed to maelzel.core.plotting.plotVoices

        Returns:
            the axes used
        """
        voices = self._asVoices()
        from maelzel.core import plotting
        return plotting.plotVoices(voices, **kws)

    def setPlay(self, /, **kws) -> Self:
        """
        Set any playback attributes, returns self

        Args:
            **kws: any argument passed to :meth:`~MObj.play` (delay, dur, chan,
                gain, fade, instr, pitchinterpol, fadeshape, params,
                priority, position).

        Returns:
            self. This allows to chain this to any constructor (see example)

        ============== ====== =====================================================
        Attribute      Type   Descr
        ============== ====== =====================================================
        instr          str    The instrument preset to use
        delay          float  Delay in seconds, added to the start of the object
        chan           int    The channel to output to, **channels start at 1**
        fade           float  The fade time; can also be a tuple (fadein, fadeout)
        fadeshape      str    One of 'linear', 'cos', 'scurve'
        pitchinterpol  str    One of 'linear', 'cos', 'freqlinear', 'freqcos'
        gain           float  A gain factor applied to the amplitud of this object.
                              **Dynamic argument** (*kgain*)
        position       float  Dynamic argument. Panning position (0=left, 1=right).
                              **Dynamic argument** (*kpos*)
        skip           float  Skip time of playback; allows to play a fragment of the object.
                              Time in beats relative to the start of the object
                              **NB**: set the delay to the -skip to start playback at the
                              original time but from the timepoint specified by the skip param
        end            float  End time of playback, in beats, relative to the start of the
                              object; counterpart of `skip`, allow to trim playback of the object
        sustain        float  An extra sustain time, in seconds. This is useful for sample
                              based instruments
        transpose      float  Transpose the pitch of this object **only for playback**
        glisstime      float  The duration (in beats) of the glissando for events with
                              glissando. A short glisstime can be used for legato playback
                              in non-percusive instruments. Implies gliss. to the next event.
        priority       int    The order of evaluation. Events scheduled with a higher
                              priority are evaluated later in the chain
        args           dict   Named arguments passed to the playback instrument
        gliss          float/ An object can be set to have a playback only gliss. It
                       bool   is equivalent to having a gliss., but the gliss is not
                              displayed as notation.
        ============== ====== =====================================================

        **Example**::

            # a piano note
            >>> from maelzel.core import *
            # Create a note with predetermined instr and panning position
            >>> note = Note("C4+25", dur=0.5).setPlay(instr="piano", position=1)
            # When .play is called, the note will play with the preset instr and position
            >>> note.play()

        .. seealso:: :meth:`MObj.addSymbol <maelzel.core.mobj.MObj.addSymbol>`, :attr:`MObj.playargs <maelzel.core.mobj.MObj.playargs>`
        """
        playargs = self.playargs
        if playargs is None:
            self.playargs = playargs = PlayArgs()
        extrakeys = kws.keys() - PlayArgs.playkeys

        # set extrakeys as args, without checking the instrument
        if extrakeys:
            # raise ValueError(f"Unknown keys: {extrakeys}, {self=}")
            args = kws.get('args')
            if args:
                for k in extrakeys:
                    args[k] = kws[k]
            else:
                kws['args'] = {k: kws[k] for k in extrakeys}
            for k in extrakeys:
                kws.pop(k)

        playargs.update(kws)
        return self

    def clone(self,
              **kws) -> Self:
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
                offset = asF(v)
                assert offset >= F0, f"Invalid offset for {self}: {offset}"
                out.offset = asF(v)
            else:
                setattr(out, k, v)

        self._copyAttributesTo(out)
        return out

    def remap(self, deststruct: ScoreStruct, sourcestruct: ScoreStruct | None = None
              ) -> Self:
        """
        Creates a clone, remapping times from source scorestruct to destination scorestruct

        The absolute time remains the same

        Args:
            deststruct: the destination scorestruct
            sourcestruct: the source scorestructure, or None to use the resolved scoresturct

        Returns:
            a clone of self remapped to the destination scorestruct

        """
        if sourcestruct is None:
            sourcestruct = self.activeScorestruct()
        offset, dur = deststruct.remapSpan(sourcestruct, self.absOffset(), self.dur)
        return self.clone(offset=offset, dur=dur)

    def copy(self) -> Self:
        """Returns a copy of this object"""
        raise NotImplementedError

    def timeShift(self, offset: time_t) -> Self:
        """
        Return a copy of this object with an added offset

        Args:
            offset: a delta time added

        Returns:
            a copy of this object shifted in time by the given amount
        """
        offset = asF(offset)
        return self.timeTransform(lambda t: t + offset, inplace=False)

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

        Will be None if this object has no explicit offset. Use :meth:`
        """
        return None if self.offset is None else self.offset + self.dur

    def quantizePitch(self, step=0.) -> Self:
        """ Returns a new object, with pitch rounded to step

        Args:
            step: quantization step, in semitones. A value of 0 used the
                default semitone division in the active config (can be
                configured via ``getConfig()['semitoneDivisions']``

        Returns:
            a copy of self with the pitch quantized
        """
        raise NotImplementedError()

    def transposeByRatio(self, ratio: float) -> Self:
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
        return self.transpose(12 * math.log(ratio, 2))

    def getConfig(self, prototype: CoreConfig | None = None) -> CoreConfig | None:
        """
        Returns a CoreConfig overloaded with options set for this object

        Args:
            prototype: the config to use as prototype, falls back to the active config

        Returns:
            A clone of the active config with any customizations made via :meth:`Voice.setConfig` or
            :meth:`Voice.configQuantization`
            If no customizations have been made, None is returned

        .. seealso::
            * :meth:`setConfig() <maelzel.core.chain.Voice.configQuantization>`
            * :meth:`configQuantization() <maelzel.core.chain.Voice.configQuantization>`
        """
        return self.parent.getConfig(prototype) if self.parent else None

    def show(self,
             fmt='',
             external: bool | None = None,
             backend='',
             scorestruct: ScoreStruct | None = None,
             resolution: int = 0,
             pageSize='',
             staffSize: float | None = None,
             cents: bool | None = None,
             voiceMaxStaves: int | None = None,
             ** kws
             ) -> None:
        """
        Show this as notation.

        Args:
            external: True to force opening the image in an external image viewer,
                even when inside a jupyter notebook. If False, show will
                display the image inline if inside a notebook environment.
                To change the default, modify :ref:`config['openImagesInExternalApp'] <config_openImagesInExternalApp>`
            backend: backend used when rendering to png/pdf.
                One of 'lilypond', 'musicxml'. If not given, use default
                (see :ref:`config['show.backend'] <config_show_backend>`)
            fmt: one of 'png', 'pdf', 'ly'. None to use default.
            scorestruct: if given overrides the current/default score structure
            resolution: dpi resolution when rendering to an image, overrides the
                :ref:`config key 'show.pngResolution' <config_show_pngresolution>`
            pageSize: if given, overrides config 'show.pageSize'. One of 'a3', 'a4', ...
            staffSize: if given, overrides config 'show.staffSize'. A value in points
                (default = 10.)
            cents: overrides config 'show.cents'. False to hide cents deviations
                as text annotation
            voiceMaxStaves: overrides config 'show.voiceMaxStaves'. Max. number of
                staves used when expanding a voice to multiple staves
            kws: any keyword is used to override the config. All options starting with
                the 'show.' prefix can be used directly (see below)

        Useful keywords
        ~~~~~~~~~~~~~~~

        ================ ===================== ===============================
        kws              Config Option         Description
        ================ ===================== ===============================
        staffSize        show.staffSize        Size of a staff, in points
        spacing          show.spacing          One of normal (traditional spacing),
                                               strict (proportional), uniform (proportional)
        voiceMaxStaves   show.voiceMaxStaves   Expands any voice to at most
                                               this number of staves
        autoClefChanges  show.autoClefChanges  Adds automatic clef changes when rendering
        clefSimplify     show.clefSimplify     Simplifies automatic clef changes
        cents            show.cents            set to False to avoid showing cents
                                               deviations as text annotation
        glissStemless    show.glissStemless    remove stems from the end note of a gliss
        horizontalSpace  show.horizontalSpace  configure proportional spacing (one of
                                               "default", "small", "medium", "large")
        pageOrientation  show.pageOrientation  one of "landscape", "portrait"
        pageSize         show.pageSize         one of "a4", "a3", ...
        ================ ===================== ===============================

        """
        cfg = self.getConfig() or Workspace.active.config
        cfg = cfg.copy()

        if resolution or kws:
            cfg = cfg.copy()
            if resolution:
                cfg['show.pngResolution'] = resolution
            for kw, value in kws.items():
                if kw in cfg:
                    cfg[kw] = value
                elif (showkw := f"show.{kw}") in cfg:
                    cfg[showkw] = value
                else:
                    matches = cfg._bestMatches(kw, limit=8)
                    logger.error(f'Invalid config keyword {kw} or {showkw} (possible matches: {matches})')

        if staffSize is not None:
            cfg['show.staffSize'] = staffSize
        if cents is not None:
            cfg['show.cents'] = cents
        if voiceMaxStaves is not None:
            cfg['show.voiceMaxStaves'] = voiceMaxStaves
        if pageSize:
            cfg['show.pageSize'] = pageSize

        if external is None:
            external = cfg['openImagesInExternalApp']
            assert isinstance(external, bool)

        if not backend:
            backend = cfg['show.backend']

        if not fmt:
            fmt = 'png' if not external and environment.insideJupyter else cfg['show.format']

        if fmt == 'ly':
            renderer = self.render(backend='lilypond', scorestruct=scorestruct, config=cfg)
            if external:
                lyfile = _util.mktemp(suffix='.ly')
                renderer.write(lyfile)
                import emlib.misc
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
                import emlib.misc
                emlib.misc.open_with_app(img)

    def _changed(self) -> None:
        """
        This method is called whenever the object changes its representation

        This happens when a note changes its pitch inplace, the duration is modified, etc.
        """
        if self.parent:
            self.parent._childChanged(self)

    def quantizedScore(self,
                       scorestruct: ScoreStruct | None = None,
                       config: CoreConfig | None = None,
                       quantizationProfile: str | quant.QuantizationProfile | None = None,
                       enharmonicOptions: enharmonics.EnharmonicOptions | None = None,
                       nestedTuplets: bool | None = None
                       ) -> quant.QuantizedScore:
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

        from maelzel.scoring import quant

        if not scorestruct:
            scorestruct = self.scorestruct() or w.scorestruct
        if quantizationProfile is None:
            quantizationProfile = config.makeQuantizationProfile()
        elif isinstance(quantizationProfile, str):
            quantizationProfile = quant.QuantizationProfile.fromPreset(quantizationProfile)
        else:
            assert isinstance(quantizationProfile, quant.QuantizationProfile)

        if nestedTuplets is not None:
            quantizationProfile.nestedTuplets = nestedTuplets

        parts = self.scoringParts()
        if config['show.respellPitches'] and enharmonicOptions is None:
            enharmonicOptions = config.makeEnharmonicOptions()
        qscore = quant.quantizeParts(parts,
                                     quantizationProfile=quantizationProfile,
                                     struct=scorestruct,
                                     enharmonicOptions=enharmonicOptions)
        return qscore

    def render(self,
               backend='',
               renderoptions: RenderOptions | None = None,
               scorestruct: ScoreStruct | None = None,
               config: CoreConfig | None = None,
               quantizationProfile: str | quant.QuantizationProfile = ''
               ) -> Renderer:
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
            a scoring.Renderer. This can be used to write the rendered structure
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
        if not config:
            config = w.config
        if not backend:
            backend = config['show.backend']
        if not renderoptions:
            renderoptions = notation.makeRenderOptionsFromConfig(config)
        if not scorestruct:
            scorestruct = self.scorestruct() or w.scorestruct

        from maelzel.scoring import quant
        if not quantizationProfile:
            quantizationProfile = config.makeQuantizationProfile()
        elif isinstance(quantizationProfile, str):
            quantizationProfile = quant.QuantizationProfile.fromPreset(quantizationProfile)
        else:
            assert isinstance(quantizationProfile, quant.QuantizationProfile)

        return _renderObject(self, backend=backend, renderoptions=renderoptions,
                             scorestruct=scorestruct, config=config,
                             quantizationProfile=quantizationProfile)

    def _renderImage(self,
                     backend='',
                     outfile='',
                     fmt="png",
                     scorestruct: ScoreStruct | None = None,
                     config: CoreConfig | None = None
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
        if not backend:
            backend = config['show.backend']
        if fmt == 'ly':
            backend = 'lilypond'
        if not outfile:
            assert fmt in ('png', 'pdf')
            outfile = _util.mktemp(suffix='.' + fmt)
        if scorestruct is None:
            scorestruct = self.scorestruct() or w.scorestruct


        _renderImage(obj=self, outfile=outfile, backend=backend,
                     scorestruct=scorestruct, config=config)

        if not os.path.exists(outfile):
            # cached image does not exist?
            logger.debug(f"Error rendering {self}, the rendering process did not generate "
                         f"the expected output file '{outfile}'. This might be a cached "
                         f"path and the cache might be invalid. Resetting the cache and "
                         f"trying again...")
            clearImageCache()
            # Try again, uncached
            _renderImage(self, outfile, backend=backend, scorestruct=scorestruct,
                         config=config)
            if not os.path.exists(outfile):
                raise FileNotFoundError(f"Could not generate image, returned image file '{outfile}' "
                                        f"does not exist")
            else:
                logger.debug(f"... resetting the cache worked, an image file '{outfile}' "
                             f"was generated")
        return outfile

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig | None = None,
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
                     config: CoreConfig | None = None
                     ) -> list[scoring.core.UnquantizedPart]:
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
        notations = self.scoringEvents(config=config or Workspace.getConfig())
        if not notations:
            return []
        scoring.core.resolveOffsets(notations)
        parts = scoring.core.distributeNotationsByClef(notations)
        return parts

    def unquantizedScore(self, title='') -> scoring.core.UnquantizedScore:
        """
        Create a maelzel.scoring.UnquantizedScore from this object

        Args:
            title: the title of the resulting score (if given)

        Returns:
            the Arrangement representation of this object

        An :class:`~maelzel.scoring.UnquantizedScore` is a list of
        :class:`~maelzel.scoring.UnquantizedPart`, which is itself a list of
        :class:`~maelzel.scoring.Notation`. It represents a score in which
        the Notations within each part are not split into measures, nor organized
        in beats. To generate a quantized score see
        :meth:`quantizedScore() <maelzel.core.mobj.MObj.quantizedScore>`

        This method is mostly used internally when an object is asked to be represented
        as a score. In this case, an UnquantizedScore is created first, which is then quantized,
        generating a :class:`~maelzel.scoring.quant.QuantizedScore`

        .. seealso::  :meth:`quantizedScore() <maelzel.core.mobj.MObj.quantizedScore>`, :class:`~maelzel.scoring.quant.QuantizedScore`

        """
        parts = self.scoringParts()
        return scoring.core.UnquantizedScore(parts, title=title)

    def _scoringAnnotation(self, text='', config: CoreConfig | None = None
                           ) -> scoring.attachment.Text:
        """ Returns owns annotations as a scoring Annotation """
        if config is None:
            config = Workspace.active.config
        if not text:
            if not self.label:
                raise ValueError("This object has no label")
            text = self.label
        from maelzel.textstyle import TextStyle
        labelstyle = TextStyle.parse(config['show.labelStyle'])
        return scoring.attachment.Text(text,
                                       fontsize=labelstyle.fontsize,
                                       italic=labelstyle.italic,
                                       weight='bold' if labelstyle.bold else '',
                                       color=labelstyle.color)

    def activeScorestruct(self) -> ScoreStruct:
        """
        Returns the ScoreStruct active for this obj or its parent.

        Otherwise returns the scorestruct for the active workspace

        Returns:
            the active scorestruct for this object

        .. seealso:: :meth:`MObj.scorestruct`
        """
        return self.scorestruct() or Workspace.active.scorestruct

    def _asBeat(self, time: beat_t) -> F:
        if isinstance(time, F):
            return time
        elif isinstance(time, tuple):
            measidx, beat = time
            return self.activeScorestruct().locationToBeat(measidx, beat)
        else:
            return F(time)

    def scorestruct(self) -> ScoreStruct | None:
        """
        Returns the ScoreStruct active for this obj or its parent (recursively)

        If this object has no parent ``None`` is returned. Use
        :meth:`activeScorestruct() <maelzel.core.mobj.MObj.activeScorestruct>`
        to always resolve the active struct for this object

        Returns:
            the associated scorestruct, if set (either directly or through its parent)

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

        .. seealso:: :meth:`activeScorestruct() <maelzel.core.mobj.MObj.activeScorestruct>`
        """
        if self._scorestruct is not None:
            return self._scorestruct
        return self.parent.scorestruct() if self.parent else None

    def write(self,
              outfile: str,
              backend='',
              resolution: int = 0,
              format='',
              ) -> None:
        """
        Export to multiple formats

        Formats supported: pdf, png, musicxml (extension: .xml or .musicxml),
        lilypond (.ly), midi (.mid or .midi) and pickle

        To configure any options either modify the active config or use
        :meth:`.setConfig` for self. You can also use a config
        as context manager to temporary change the active config

        Args:
            outfile: the path of the output file. The extension determines
                the format. Formats available are pdf, png, lilypond, musicxml,
                midi, csd and pickle.
            backend: the backend used when writing as pdf or png. If not given,
                the default defined in the active config is used
                (:ref:`key: 'show.backend' <config_show_backend>`).
                Possible backends: ``lilypond``; ``musicxml`` (uses MuseScore to render musicxml as
                image so MuseScore needs to be installed)
            resolution: image DPI (only valid if rendering to an image) - overrides
                the :ref:`config key 'show.pngResolution' <config_show_pngresolution>`
            format: the format to write to. If not given, the format is inferred from the
                extension of the output file. If the extension is not recognized, an error is raised.
                One of 'pdf', 'png', 'lilypond', 'musicxml', 'midi', 'csd', 'pickle'.

        Formats
        -------

        * pdf, png: will render the object as notation and save that to the given format
        * lilypond: `.ly` extension. Will render the object as notation and save it as lilypond text
        * midi: `.mid` or `.midi` extension. At the moment this is done via lilypond, so the midi
            produced follows the quantization process used for rendering to notation. Notice that
            midi cannot reproduce many features of a maelzel object, like microtones, and many
            complex rhythms will not be translated correctly
        * pickle: the object is serialized using the pickle module. This allows to load it
            via ``pickle.load``: ``myobj = pickle.load(open('myobj.pickle'))``

        Example
        ~~~~~~~

        .. code-block:: python

            chain = Chain(...)
            with CoreConfig({'show.voiceMaxStaves': 2, 'show.staffSize': 12}):
                chain.write('chain.pdf')

        """
        if outfile == '?':
            from . import _dialogs
            selected = _dialogs.selectFileForSave(key="writeLastDir",
                                                  filter="All formats (*.pdf, *.png, "
                                                         "*.ly, *.xml, *.mid)")
            if not selected:
                logger.info("File selection cancelled")
                return
            outfile = selected
        ext = os.path.splitext(outfile)[1]
        cfg = Workspace.getConfig()
        if not format:
            format = {
                '.ly': 'lilypond',
                '.mid': 'midi',
                '.midi': 'midi',
                '.xml': 'musicxml',
                '.musicxml': 'musicxml',
                '.csd': 'csd',
                '.pickle': 'pickle'
            }.get(ext)
        if format == 'lilypond' or format == 'midi':
            backend = 'lilypond'
        elif format == 'musicxml':
            backend = 'musicxml'
        elif format == 'csd':
            renderer = self._makeOfflineRenderer()
            renderer.writeCsd(outfile)
            return
        elif format == 'pickle':
            import pickle
            with open(outfile, 'wb') as f:
                pickle.dump(self, f)
            return
        elif not backend:
            backend = cfg['show.backend']
        elif format not in ('pdf', 'png'):
            raise ValueError(f"Unsupported format: {format}")
        if resolution:
            cfg = cfg.clone(updates={'show.pngResolution': resolution})
        r = notation.renderWithActiveWorkspace(self.scoringParts(config=cfg),
                                               backend=backend,
                                               scorestruct=self.scorestruct(),
                                               config=cfg)
        r.write(outfile)

    def _htmlImage(self, scaleFactor: float = 0.) -> tuple[bytes, str]:
        """
        Returns a tuple of the image as a base64 string and the width and height of the image.

        Args:
            scaleFactor: The scale factor to apply to the image.

        Returns:
            A tuple `(base64 string of the image, html img tag)` representing the base64 string
            of the image and the html img tag.
        """
        imgpath = self._renderImage()
        from maelzel import _imgtools
        img64, width, height = _imgtools.readImageAsBase64(imgpath)
        if scaleFactor == 0.:
            scaleFactor = Workspace.getConfig().get('show.scaleFactor', 1.0)
        return img64, _util.htmlImage64(img64=img64, imwidth=width, width=f'{int(width * scaleFactor)}px')

    def _repr_html_header(self):
        return _html.escape(repr(self))

    def _repr_html_(self) -> str:
        cfg = Workspace.active.config
        txt = self._repr_html_header()
        html = rf'<code style="white-space: pre-line; font-size:0.9em;">{txt}</code><br>'
        if cfg['jupyterReprShow']:
            img64, img = self._htmlImage()
            html += '<br>' + img
        return html

    def _makeOfflineRenderer(self,
                             sr=0,
                             numchannels=2,
                             eventoptions={}
                             ) -> offline.OfflineRenderer:
        from maelzel.core import offline
        r = offline.OfflineRenderer(sr=sr, numchannels=numchannels)
        events = self.synthEvents(**eventoptions)
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

    def synthEvents(self,
                    instr='',
                    delay: float | None = None,
                    args: dict[str, float] | None = None,
                    gain: float | None = None,
                    chan: int | None = None,
                    pitchinterpol='',
                    fade: float | tuple[float, float] | None = None,
                    fadeshape='',
                    position: float | None = None,
                    skip: float | None = None,
                    end: float | None = None,
                    sustain: float | None = None,
                    workspace: Workspace | None = None,
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
                in quarternotes, the delay is always in seconds. It can be negative, in
                which case synth events start the given amount of seconds earlier.
            instr: which instrument to use (see defPreset, definedPresets). Use "?" to
                select from a list of defined presets.
            chan: the channel to output to. **Channels start at 1**
            pitchinterpol: 'linear', 'cos', 'freqlinear', 'freqcos'
            fade: fade duration in seconds, can be a tuple (fadein, fadeout)
            fadeshape: 'linear' | 'cos'
            args: named arguments passed to the note. A dict ``{paramName: value}``
            position: the panning position (0=left, 1=right)
            skip: start playback at the given beat (in quarternotes), relative
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
            >>> n.synthEvnets(gain=0.5)
            [SynthEvent(delay=0.000, gain=0.5, chan=1, fade=(0.02, 0.02), instr=piano)
             bps 0.000s:  60, 1.000000
                 1.000s:  60, 1.000000]
            >>> play(n.synthEvents(chan=2))

        """
        if instr == "?":
            from .presetmanager import presetManager
            instr = presetManager.selectPreset()
            if not instr:
                raise ValueError("No preset selected")

        if kwargs:
            if args:
                args.update(kwargs)
            else:
                args = kwargs

        if workspace is None:
            workspace = Workspace.active

        if (struct := self.scorestruct()) is not None:
            workspace = workspace.clone(scorestruct=struct, config=workspace.config)

        playargs = PlayArgs.makeDefault(workspace.config)
        db = playargs.db

        if instr:
            db['instr'] = instr
        if delay is not None:
            db['delay'] = delay
        if args is not None:
            db['args'] = args
        if gain is not None:
            db['gain'] = gain
        if chan is not None:
            db['chan'] = chan
        if pitchinterpol:
            db['pitchinterpol'] = pitchinterpol
        if fade is not None:
            db['fade'] = fade
        if fadeshape:
            db['fadeshape'] = fadeshape
        if position is not None:
            db['position'] = position
        if sustain is not None:
            db['sustain'] = sustain
        if transpose:
            db['transpose'] = transpose

        parentOffset = self.parent.absOffset() if self.parent else F(0)
        events = self._synthEvents(playargs=playargs,
                                   parentOffset=parentOffset,
                                   workspace=workspace)

        struct = workspace.scorestruct
        playdelay: float = playargs['delay']
        if skip is not None or end is not None or playdelay < 0.:
            skiptime = 0. if skip is None else float(struct.beatToTime(skip))
            endtime = math.inf if end is None else float(struct.beatToTime(end))
            events = SynthEvent.cropEvents(events, start=max(0., skiptime+playdelay), end=endtime + playdelay)

        if any(ev.delay < 0 for ev in events):
            raise ValueError(f"Events cannot have negative delay, events={events}")

        return events

    def play(self,
             instr='',
             delay: float | None = None,
             args: dict[str, float] | None = None,
             gain: float | None = None,
             chan: int | None = None,
             pitchinterpol='',
             fade: float | tuple[float, float] | None = None,
             fadeshape='',
             position: float | None = None,
             skip: float | None = None,
             end: float | None = None,
             whenfinished: _t.Callable | None = None,
             sustain: float | None = None,
             workspace: Workspace | None = None,
             transpose: float = 0,
             config: CoreConfig | None = None,
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
                in symbolic (beat) time, the delay is always in real (seconds) time.
                Delay can be negative, in which case synth events start the given amount
                of seconds earlier.
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
            * :meth:`synthEvents() <maelzel.core.mobj.MObj.synthEvents>`
            * :meth:`MObj.rec() <maelzel.core.mobj.MObj.rec>`
            * :func:`~maelzel.core.offline.render`,
            * :func:`~maelzel.core.playback.play`


        Example
        ~~~~~~~

        Play a note

            >>> from maelzel.core import *
            >>> note = Note(60).play(gain=0.1, chan=2)

        Play multiple objects synchronised

            >>> play(
            ... Note(60, 1.5).synthEvents(gain=0.1, position=0.5)
            ... Chord("4E 4G", 2, start=1.2).synthEvents(instr='piano')
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

        events = self.synthEvents(delay=delay,
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
            import csoundengine.synth
            group = csoundengine.synth.SynthGroup([])
        else:
            renderer = workspace.renderer or realtimerenderer.RealtimeRenderer()
            group = renderer.schedEvents(coreevents=events, whenfinished=whenfinished)
            if display and environment.insideJupyter:
                from IPython.display import display
                display(group)
        return group

    def rec(self,
            outfile='',
            sr: int = 0,
            verbose: bool | None = None,
            wait: bool | None = None,
            nchnls: int | None = None,
            instr='',
            delay: float | None = None,
            args: dict[str, float] | None = None,
            gain: float | None = None,
            position: float | None = None,
            extratime: float | None = None,
            workspace: Workspace | None = None,
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
            verbose: if True, display synthesis output

            **kws: any keyword passed to .play

        Returns:
            the offline renderer used. If no outfile was given it is possible to
            access the renderer soundfile via
            :meth:`OfflineRenderer.lastOutfile() <maelzel.core.offline.OfflineRenderer.lastOutfile>`

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> # a simple note
            >>> chord = Chord("4C 4E 4G", dur=8).setPlay(gain=0.1, instr='piano')
            >>> renderer = chord.rec(wait=True)
            >>> renderer.lastOutfile()
            '/home/testuser/.local/share/maelzel/recordings/tmpashdas.wav'

        .. seealso:: :class:`~maelzel.core.offline.OfflineRenderer`
        """
        events = self.synthEvents(instr=instr, position=position,
                                  delay=delay, args=args, gain=gain,
                                  workspace=workspace,
                                  **kws)

        from maelzel.core import offline
        return offline.render(outfile=outfile, events=events, sr=sr, wait=wait,
                              verbose=verbose, nchnls=nchnls, tail=extratime)

    def isRest(self) -> bool:
        """
        Is this object a Rest?

        Rests are used as separators between objects inside an Chain or a Track
        """
        return False

    def addSymbol(self, *args, **kws) -> Self:
        """
        Add a notation symbol to this object

        Some symbols are exclusive, meaning that adding a symbol of this kind will
        replace a previously set symbol. Exclusive symbols include any properties
        (color, size, etc) and other customizations like notehead shape

        Example
        -------

            >>> from maelzel.core import *
            >>> n = Note(60)
            >>> n.addSymbol(symbols.Articulation('accent'))
            # The same can be achieved via keyword arguments:
            >>> n.addSymbol(articulation='accent')
            # Multiple symbols can be added at once:
            >>> n = Note(60).addSymbol(text='dolce', articulation='tenuto')
            >>> n2 = Note("4G").addSymbol(symbols.Articulation('accent'), symbols.Ornament('mordent'))
            # Known symbols - most common symbols don't actually need keyword arguments:
            >>> n = Note("4Db").addSymbol('accent').addSymbol('fermata')
            # Some symbols can take customizations:
            >>> n3 = Note("4C+:1/3").addSymbol(symbols.Harmonic(interval='4th'))


        Returns:
            self (similar to setPlay, allows to chain calls)

        """
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
        if self.symbols is None:
            return
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

    def addText(self,
                text: str,
                placement='above',
                italic=False,
                weight='normal',
                fontsize: int | None = None,
                fontfamily='',
                box=''
                ) -> Self:
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
            self. This can be used to create an object and add text in one call

        Example
        ~~~~~~~

            >>> chain = Chain([
            ...     Note("4C", 1).addText('do'),
            ...     Note("4D", 1).addText('re')
            ... ])
            >>> chain

        .. image:: assets/event-addText.png
        """
        from . import symbols as _symbols
        self.addSymbol(_symbols.Text(text, placement=placement, fontsize=fontsize,
                                     italic=italic, weight=weight, fontfamily=fontfamily,
                                     box=box))
        return self

    def timeTransform(self, timemap: _t.Callable[[F], F], inplace=False) -> Self:
        """
        Apply a time-transform to this object

        Args:
            timemap: a function mapping old time to new time
            inplace: if True changes are applied inplace

        Returns:
            the resulting object (self if inplace)

        .. note::

            time is conceived as abstract 'beat' time, measured in quarter-notes.
            The actual time in seconds will be also determined by any tempo changes
            in the active score structure.
        """
        raise NotImplementedError

    def timeShiftInPlace(self, offset: time_t) -> None:
        """
        Shift the time of this by the given offset (inplace)

        Args:
            offset: the time delta (in quarterNotes)
        """
        newoffset = self.relOffset() + asF(offset)
        if newoffset < 0:
            raise ValueError(f"This operation would result in a negative offset. "
                             f"Own offset: {self.relOffset()}, resulting offset: {newoffset}, "
                             f"given time shift: {offset}, self: {self}, absolute offset: {self.absOffset()}")
        self.offset = newoffset
        self._changed()

    def timeRange(self) -> tuple[F, F]:
        """
        Returns a tuple (starttime, endtime), in seconds

        Returns:
            a tuple ``(starttime: F, endtime: F)``, where starttime
            and endtime are both absolute times in seconds
        """
        struct = self.activeScorestruct()
        start = self.absOffset()
        return struct.beatToTime(start), struct.beatToTime(start+self.dur)

    def durSecs(self) -> F:
        """
        Returns the duration in seconds according to the active score

        Returns:
            the duration of self in seconds
        """
        startsecs, endsecs = self.timeRange()
        return endsecs - startsecs

    def location(self) -> tuple[location_t, location_t]:
        """
        Returns the location of this object within the active score struct

        Returns:
            a tuple ``(startlocation, endlocation)`` where both ``startlocation``
            are tuples ``(measureindex, beatoffset)`` representing the position
            of this object within the score


        Example
        -------

            >>> setScoreStruct(timesig='3/4')
            >>> note = Note("4C", 1, offset=5)
            >>> note.location()
            ((1, Fraction(2, 1)), (2, Fraction(0, 1)))

        The note starts at measure 1, beat 2 and ends at
        measure 2, beat 0 (both measures and beats start at 0)

        """
        struct = self.activeScorestruct()
        startbeat = self.absOffset()
        return struct.beatToLocation(startbeat), struct.beatToLocation(startbeat + self.dur)

    def pitchTransform(self, pitchmap: _t.Callable[[float], float]) -> Self:
        """
        Apply a pitch-transform to this object, returns a copy

        Args:
            pitchmap: a function mapping pitch to pitch

        Returns:
            the object after the transform
        """
        raise NotImplementedError("Subclass should implement this")

    def timeScale(self, factor: num_t, offset: num_t = F0) -> Self:
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

    def invertPitch(self, pivot: str | float | int) -> Self:
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
        def transform(pitch, pivot=pivotm):
            return pivot * 2 - pitch
        return self.pitchTransform(transform)

    def transpose(self, interval: int | float) -> Self:
        """
        Transpose this object by the given interval

        Args:
            interval: the interval in semitones

        Returns:
            the transposed object
        """
        return self.pitchTransform(lambda pitch: pitch+interval)

# --------------------------------------------------------------------


class MContainer(MObj):
    """
    An interface for any class which can be a parent

    Implemented downstream by classes like Chain or Score.
    """
    _configKeysRegistry = {}
    "A cache for all class config keys"

    __slots__ = ('_config',)

    def __init__(self,
                 offset: F | None = None,
                 label='',
                 parent: MContainer | None = None,
                 properties: dict[str, _t.Any] | None = None):

        super().__init__(offset=offset, dur=F0, label=label,
                         properties=properties, parent=parent)

        self._config: dict[str, _t.Any] = {}
        "Collects customizations to the config specific to this container"

    @abstractmethod
    def __iter__(self) -> _t.Iterator[MObj | MContainer]:
        raise NotImplementedError

    @classmethod
    def _classConfigKeys(cls) -> set[str]:
        # This method can be overloaded to return keys specific to a subclass
        pattern = r'\.?(quant|show)\.\w[a-zA-Z0-9_]*'
        return set(k for k in CoreConfig.root().keys() if re.match(pattern, k))

    @classmethod
    def _configKeys(cls) -> set[str]:
        # This method should probably not be overloaded. It is a workaround
        # to the fact that we want to cache config keys without any subclass
        # needing to worry about what kind of caching we are using
        clsname = cls.__qualname__
        if (keys := cls._configKeysRegistry.get(clsname)) is not None:
            return keys
        configkeys = cls._classConfigKeys()
        cls._configKeysRegistry[clsname] = configkeys
        return configkeys

    def setScoreStruct(self, scorestruct: ScoreStruct | None) -> None:
        """
        Set the ScoreStruct for this object and its children

        This ScoreStruct will be used for any object embedded
        downstream.

        Args:
            scorestruct: the ScoreStruct, or None to remove any scorestruct
                previously set

        """
        if scorestruct is None:
            self._scorestruct = None
            for item in self:
                if isinstance(item, MContainer):
                    item.setScoreStruct(None)
            return

        if self.parent:
            parentstruct = self.parent.scorestruct()
            if parentstruct is None:
                raise ValueError(f"An object cannot promote a scorestruct up in the tree structure. Set the score"
                                 f" structure at the root: ({self.root()})")
            elif scorestruct is not parentstruct:
                raise ValueError(f"This {self.__class__} has a parent with a scorestruct "
                                    f"different than the given one."
                                    f"\nParent struct: {parentstruct}"
                                    f"\nNew struct: {scorestruct}")
        self._scorestruct = scorestruct
        self._changed()

    def _copyAttributesTo(self, other: Self) -> None:
        super()._copyAttributesTo(other)
        if self._scorestruct:
            other.setScoreStruct(self._scorestruct)

    def setConfig(self, *args) -> None:
        """
        Configure this object

        Possible keys are any CoreConfig keys with the prefixes 'quant.' and 'show.'
        and also secondary keys starting with '.quant' and '.show'

        Internal note: any subclass can set the keys accepted by its instances by
        overloading :meth:`MContainer._configKeys`

        Args:
            args: an even number of args of the form key1, value1, key2, value2, ...

        Example
        ~~~~~~~

        Configure the voice to break syncopations at every beat when
        rendered or quantized as a QuantizedScore

            >>> voice = Voice(...)
            >>> voice.setConfig('quant.brakeSyncopationsLevel', 'all')

        Now, whenever the voice is shown all syncopations across beat boundaries
        will be split into tied notes.

        This is the same as:

            >>> voice = Voice(...)
            >>> score = Score([voice])
            >>> quantizedscore = score.quantizedScore()
            >>> quantizedscore.parts[0].brakeSyncopations(level='all')
            >>> quantizedscore.render()
        """
        keys = self._classConfigKeys()
        root = CoreConfig.root()
        assert len(args) % 2 == 0
        kws = args[::2]
        values = args[1::2]
        for key, value in zip(kws, values):
            if key not in keys:
                raise KeyError(f"Invalid key '{key}' for a {self.__class__}. "
                               f"Valid keys are {keys}")
            if errmsg := root.checkValue(key, value):
                raise ValueError(f"Invalid value {value} for key '{key}': {errmsg}")
            self._config[key] = value

    def getConfig(self, prototype: CoreConfig | None = None) -> CoreConfig | None:
        # most common first
        if prototype is None:
            prototype = Workspace.active.config
        if not self.parent:
            return None if not self._config else prototype.clone(self._config)
        if (parentconfig := self.parent.getConfig(prototype)) is None:
            # parent made no changes
            return None if not self._config else prototype.clone(self._config)
        else:
            return parentconfig if not self._config else parentconfig.clone(self._config)

    def _resolveConfig(self, config: CoreConfig | dict | None = None
                       ) -> tuple[CoreConfig, bool]:
        """
        Returns a tuple (resolvedConfig, iscustomized)

        where resolvedConfig is the config for this object, given any customizations,
        and iscustomized is True if self has own customizations

        Args:
            config: a config to use as the active config. Any customizations made
                will have priority over this

        Returns:
            a tuple (resolvedConfig: CoreConfig, iscustomized: bool)
        """
        if config is None:
            activeconfig = Workspace.active.config
        elif not isinstance(config, CoreConfig):
            assert isinstance(config, dict)
            activeconfig = CoreConfig(updates=config)
        else:
            activeconfig = config
        ownconfig = self.getConfig(prototype=activeconfig)
        config = ownconfig or activeconfig
        return config, ownconfig is not None

    def nextEvent(self, event: MObj) -> _event.MEvent | None:
        """
        Returns the next event after *event*

        This method only makes sense when the container is an horizontal
        container (Chain, Voice). *event* and the returned event are
        always some MEvent (see maelzel.core.event)
        """
        return None

    @abstractmethod
    def _childOffset(self, child: MObj) -> F:
        """The offset of child relative to this parent"""
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

    @abstractmethod
    def _update(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _resolveGlissandi(self, force=False) -> None:
        raise NotImplementedError

    def nextItem(self, item: MObj) -> MObj | None:
        """Returns the item after *item*, if any (None otherwise)"""
        return None

    def previousItem(self, item: MObj) -> MObj | None:
        return None

    def previousEvent(self, event: _event.MEvent) -> _event.MEvent | None:
        return None

    def __contains__(self, item: MObj) -> bool:
        raise NotImplementedError

    def root(self) -> MContainer:
        """
        The root of this object

        Objects are organized in a tree structure. For example,
        a note can be embedded in a Chain, which is part
        of a Voice, which is part of a Score. In this case, the
        root of all this objects is the score. A container
        without no parent is its own root.

        Returns:
            the root of this object

        Example
        ~~~~~~~

            >>> voice = Voice([
            ... "4C:1",
            ... Chain("4D 4E 4F")
            ... ])
            >>> score = Score([voice])
            >>> voice[0].root() is score
            True
            >>> score.root() is score
            True

            >>> Note(60).root() is None
            True

            >>> voice2 = voice.copy()
            >>> voice2.parent is None
            True
            >>> voice2.root() is voice2
            True
        """
        return self if self.parent is None else self.parent.root()


# --------------------------------------------------------------------


def _renderImage(obj: MObj,
                 outfile: str,
                 config: CoreConfig,
                 backend: str,
                 scorestruct: ScoreStruct,
                 ) -> Renderer:
    assert outfile and config and backend and scorestruct
    ext = os.path.splitext(outfile)[1].lower()
    if ext not in ('.png', '.pdf'):
        raise ValueError(f"Unknown format '{ext}', possible formats are pdf and png")
    fmt = ext[1:]
    renderoptions = config.makeRenderOptions()
    tmpfile, renderer = _renderImageCached(obj=obj, fmt=fmt, config=config, backend=backend,
                                           scorestruct=scorestruct, renderoptions=renderoptions)
    if not os.path.exists(tmpfile):
        logger.debug(f"Cached file '{tmpfile}' not found, resetting cache, trying again")
        clearImageCache()
        tmpfile, renderer = _renderImageCached(obj=obj, fmt=fmt, config=config, backend=backend,
                                               scorestruct=scorestruct, renderoptions=renderoptions)
        if not os.path.exists(tmpfile):
            raise RuntimeError(f"Could not render {obj} to file '{tmpfile}'")

    import shutil
    shutil.copy(tmpfile, outfile)
    assert os.path.exists(outfile), f"Could not copy {tmpfile} to {outfile}"
    return renderer


@functools.cache
def _renderImageCached(obj: MObj,
                       fmt: str,
                       config: CoreConfig,
                       backend: str,
                       scorestruct: ScoreStruct,
                       renderoptions: RenderOptions
                       ) -> tuple[str, Renderer]:
    assert fmt in ('pdf', 'png')
    renderer = obj.render(backend=backend, renderoptions=renderoptions, scorestruct=scorestruct,
                          config=config)
    outfile = _util.mktemp(suffix="." + fmt)
    renderer.write(outfile)
    if not os.path.exists(outfile):
        raise RuntimeError(f"Error rendering to file '{outfile}', file does not exist")
    return (outfile, renderer)


@functools.cache
def _renderObject(obj: MObj,
                  backend: str,
                  scorestruct: ScoreStruct,
                  config: CoreConfig,
                  renderoptions: RenderOptions,
                  quantizationProfile: quant.QuantizationProfile,
                  check=True
                  ) -> Renderer:
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
        check: if True, check that the generated scoring parts are valid

    Returns:
        a scoring.Renderer. The returned object can be used to render (via the
        :meth:`~maelzel.scoring.Renderer.write` method) or to have access to the
        generated score (see :meth:`~maelzel.scoring.Renderer.nativeScore`)

    .. note::

        To render with a temporary Wrokspace (i.e., without modifying the active Workspace),
        use::

        .. code-block:: python

            with Workspace(scorestruct=..., config=..., ...) as w:
                renderObject(myobj, "outfile.pdf")
    """
    assert scorestruct and config
    assert isinstance(backend, str) and backend in ('musicxml', 'lilypond')
    parts = obj.scoringParts(config=config)
    if not parts:
        if config['show.warnIfEmpty']:
            logger.warning(f"The object {obj} did not produce any scoring parts")
        measure0 = scorestruct.measuredefs[0]
        part = scoring.core.UnquantizedPart(notations=[scoring.Notation.makeRest(measure0.beatStructure()[0].duration)])
        parts = [part]
    elif check:
        for part in parts:
            part.check()
    renderer = notation.renderWithActiveWorkspace(parts,
                                                  backend=backend,
                                                  renderoptions=renderoptions,
                                                  scorestruct=scorestruct,
                                                  config=config,
                                                  quantizationProfile=quantizationProfile)
    return renderer


def clearImageCache() -> None:
    """
    Clear the image cache. Useful when changing display format
    """
    logger.info("Resetting image cache")
    _renderImageCached.cache_clear()
    _renderObject.cache_clear()
