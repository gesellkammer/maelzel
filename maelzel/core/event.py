"""
All individual events inherit from :class:`MEvent`

Events can be chained together (with ties or glissandi )
within a :class:`~maelzel.core.chain.Chain` or a :class:`~maelzel.core.chain.Chain`
to produce more complex horizontal structures. When an event is added
to a chain this chain becomes the event's *parent*.

Absolute / Relative Offset
--------------------------

The :attr:`~maelzel.core.event.MEvent.offset` attribute of any event determines the
start of the event **relative** to the parent container (if the event has not parent
the relative and the absolute offset are the same). The resolved offset can be queried
via :meth:`~maelzel.core.mobj.MObj.relOffset`, the absolute offset via
:meth:`~maelzel.core.mobj.MObj.absOffset`

"""

from __future__ import annotations

import math
import functools

import pitchtools as pt

from maelzel import scoring

from maelzel.core import mobj
from maelzel.common import F, asF, F0, F1, asmidi
from maelzel.common import UNSET
from maelzel.core.workspace import Workspace
from maelzel.core.mevent import MEvent
from maelzel.core._common import MAXDUR, logger
from maelzel.core import synthevent
from maelzel.core import _tools
from . import symbols as _symbols

from maelzel import _util


from typing import TYPE_CHECKING, overload as _overload, cast as _cast

if TYPE_CHECKING:
    from .config import CoreConfig
    from typing import Callable, Any, Sequence, Iterator
    from typing_extensions import Self
    from maelzel.common import time_t, pitch_t, num_t, UnsetType
    from maelzel.dynamiccurve import DynamicCurve


__all__ = (
    'MEvent',
    'Note',
    'Chord',
    'Rest',

    'asEvent',
    'Grace'
)


@functools.total_ordering
class Note(MEvent):
    """
    A Note represents a one-pitch event

    A Note makes a clear division between the value itself and the
    representation as notation or as sound. Playback specific options
    (instrument, pan position, etc) can be set via the
    :meth:`~Note.setPlay` method.

    Any aspects regarding notation (articulation, enharmonic variant, etc)
    can be set via :meth:`~Note.addSymbol`

    The *pitch* parameter can be used to set the pitch and other attributes in
    multiple ways.

    * To set the pitch from a frequency, use `pitchtools.f2m` or use a string as '400hz'.
    * The spelling of the notename can be fixed by suffixing the notename with a '!' sign.
      For example, ``Note('4Db!')`` will fix the spelling of this note to Db
      instead of C#. This is the same as ``Note('4Db', fixed=True)``. It is also possible
      to bind all pitches given as notename, see :ref:`fixStringNotenames <config_fixstringnotenames>`
    * The duration can be set as ``Note('4C#:0.5')`` to set a duration of 0.5, or
      also Note('4C#/8') to set a duration of an 8th note
    * Dynamics can also be set: ``Note('4C#:mf')``. Multiple settings can be combined:
      ``Note('4C#:0.5:mf')``.

    Args:
        pitch: a midinote or a note as a string. A pitch can be a midinote (a float) or a notename (a string).
        dur: the duration of this note (optional)
        amp: amplitude 0-1 (optional)
        offset: offset time fot the note, in quarternotes (optional). If None, the offset time
                will depend on the context (previous notes) where this Note is evaluated.
        gliss: if given, defines a glissando. It can be either the endpitch of the glissando, or
               True, in which case the endpitch remains undefined
        label: a label str to identify this note
        dynamic: allows to attach a dynamic expression to this Note. This dynamic
                 is only for notation purposes, it does not modify playback
        tied: is this Note tied to the next?
        fixed: if True, fix the spelling of the note to the notename given (this is
               only taken into account if the pitch was given as a notename). See also
               the configuration key `fixStringNotenames <config_fixstringnotenames>`.
               **NB**: as a shortcut, it is possible to suffix the notename with a !
               mark in order to lock its spelling, independently of the configuration
               settings
        _init: if True, fast initialization is performed, skipping any checks. This is
               used internally for fast copying/cloning of objects.

    """

    __slots__ = ('pitch',
                 'pitchSpelling',
                 '_gliss'
                 )

    def __init__(self,
                 pitch: pitch_t,
                 dur: time_t | None = None,
                 *,
                 amp: float | None = None,
                 offset: time_t | None = None,
                 gliss: pitch_t | bool = False,
                 label='',
                 dynamic='',
                 tied=False,
                 properties: dict[str, Any] | None = None,
                 symbols: list[_symbols.Symbol] | None = None,
                 fixed=False,
                 _init=True
                 ):
        pitchSpelling = ''
        if not _init:
            midinote = _cast(float, pitch)
            assert offset is None or isinstance(offset, F)
        else:
            if not isinstance(pitch, str):
                assert 0 <= pitch <= 200, f"Expected a midinote (0-127) but got {pitch}"
                midinote = pitch
            else:
                if ":" in pitch:
                    props = _tools.parseNote(pitch)
                    dur = dur if dur is not None else props.dur
                    if isinstance(props.notename, list):
                        raise ValueError(f"Can only accept a single pitch, got {props.notename}")
                    pitch = props.notename
                    if p := props.keywords:
                        offset = offset or p.pop('offset', None)
                        dynamic = dynamic or p.pop('dynamic', '')
                        tied = tied or p.pop('tied', False)
                        gliss = gliss or p.pop('gliss', False)
                        fixed = p.pop('fixPitch', False) or fixed
                        label = label or p.pop('label', '')
                        properties = p if not properties else p | properties
                    if props.symbols:
                        if symbols is None:
                            symbols = props.symbols
                        else:
                            symbols.extend(props.symbols)
                    if props.spanners:
                        for spanner in props.spanners:
                            self.addSpanner(spanner)
                elif "/" in pitch:
                    parsednote = _tools.parseNote(pitch)
                    if not isinstance(parsednote.notename, str):
                        raise ValueError(f"A Note can only have one pitch, got {parsednote.notename}")
                    pitch = parsednote.notename
                    dur = parsednote.dur

                assert isinstance(pitch, str)
                pitch = pitch.lower()

                if pitch == 'rest' or pitch == 'r':
                    midinote, amp = 0, 0
                else:
                    pitch, _tied, _fixed = _tools.parsePitch(pitch)
                    if _tied:
                        tied = _tied
                    if _fixed:
                        fixed = _fixed
                    if pitch.endswith('hz'):
                        pitchSpelling = ''
                    else:
                        pitchSpelling = pt.notename_upper(pitch)
                    midinote = pt.str2midi(pitch)

                if not fixed and Workspace.active.config['fixStringNotenames']:
                    fixed = True

            if offset is not None:
                offset = asF(offset)

            if not isinstance(gliss, bool):
                gliss = asmidi(gliss)

            if amp and amp > 0:
                assert midinote > 0

            assert properties is None or isinstance(properties, dict)

        dur = asF(dur) if dur is not None else F1
        super().__init__(dur=dur, offset=offset, label=label, properties=properties,
                         symbols=symbols, tied=tied, amp=amp, dynamic=dynamic)

        self.pitch: float = midinote
        "The pitch of this note"

        self.pitchSpelling = '' if not fixed else pitchSpelling
        "The pitch representation of this Note. Can be different from the sounding pitch"

        self._gliss: float | bool = gliss  # type: ignore

    @staticmethod
    def makeRest(dur: time_t | str, offset: time_t = None, label='', dynamic='') -> Note:
        """
        Static method to create a Rest

        A Rest is actually a regular Note with pitch == 0.

        Args:
            dur: duration of the rest
            offset: explicit offset of the rest
            label: a label to attach to the rest
            dynamic: a dynamic for this rest (yes, rests can have dynanic...)

        Returns:
            the Note object representing the rest

        """
        return Rest(dur=dur, offset=offset, label=label, dynamic=dynamic)

    def setPlay(self, /, **kws) -> Note:
        if glide := kws.pop('gliss', None) is not None:
            glisstime = float(glide) if glide is not True else min(0.1, self.dur * F(3, 4))
            kws['glisstime'] = glisstime
            if not self._gliss:
                self._gliss = True
                self.addSymbol(_symbols.GlissProperties(hidden=True))
        super().setPlay(**kws)
        return self

    def transpose(self, interval: float, inplace=False) -> Note:
        """
        Transpose this note by the given interval

        If inplace is True the operation is done inplace and the
        returned note can be ignored.

        .. note::
            If this Note has a set spelling, its spelling will also be transposed
            To remove a fixed spelling do ``note.pitchSpelling = ''``

        Args:
            interval: the transposition interval (1=one semitone)
            inplace: if True, the note itself is modified

        Returns:
            The transposed note

        """
        out = self if inplace else self.copy()
        out._setpitch(self.pitch+interval)
        return out

    @classmethod
    def grace(cls,
              pitch: pitch_t,
              stemless=False,
              slash=False,
              value: F | str | float | None = None,
              parenthesis=False,
              hidden=False,
              **kws) -> Self:
        """
        Class method to create a grace note

        Args:
            pitch: the pitch of the grace chord
            stemless: if is stemless?
            slash: slashed stem
            value: the rhythic value of the grace note ("1/2" or F(1, 2)=8th note,
                "1/8"=32nd note, etc.
            parenthesis: is the notehead parenthesized?
            hidden: should the whole note be hidden?
            **kws: keyword args passed to the Note constructor

        Returns:
            the grace note
        """
        note = cls(pitch=pitch, dur=0, **kws)
        _customizeGracenote(note, stemless=stemless, slash=slash,
                            value=value, hidden=hidden, parenthesis=parenthesis)
        return note

    def mergeWith(self, other: Note) -> Note | None:
        """
        Merge this Note with another Note, if possible

        Args:
            other: the other note to merge this with

        Returns:
            the merged note or None if merging is not possible

        """
        if self.isRest() and other.isRest():
            assert self.dur is not None and other.dur is not None
            out = self.clone(dur=self.dur + other.dur)
            return out

        if (not self.tied or
                self.gliss or
                other.isRest() or
                self.isRest() or
                self.pitch != other.pitch or
                self.amp != other.amp or
                self.dynamic != other.dynamic):
            return None

        return self.clone(dur=self.dur + other.dur, tied=other.tied)

    def _setNotatedPitch(self, notename: str) -> None:
        """
        Set the pitch representation when this object is rendered as notation

        This will not change the actual pitch or its playback. The pitch may differ
        from the actual pitch of the object.

        Args:
            notename: the pitch to use to represent this object. Can be set
                to an empty string to remove any pitch previously set

        """
        if not notename:
            self._removeSymbolsOfClass(_symbols.NotatedPitch)
            self.pitchSpelling = ''
        else:
            self.addSymbol(_symbols.NotatedPitch(notename))
            self.pitchSpelling = notename

    def _canBeLinkedTo(self, other: MEvent) -> bool:
        if self.isRest() or not isinstance(other, (Note, Chord)) or other.isRest():
            return False
        if self._gliss is True or (self.playargs and self.playargs.get('glisstime', 0.) > 0.):
            return True
        if isinstance(other, Note):
            if self.tied and self.pitch == other.pitch:
                return True
        elif isinstance(other, Chord):
            if self.tied and self.pitch in other.pitches:
                return True
        return False

    def isGrace(self) -> bool:
        return not self.isRest() and self.dur == 0

    @property
    def gliss(self) -> float | bool:
        """the end pitch (as midinote), True if the gliss extends to the next note, or False"""
        return self._gliss

    @gliss.setter
    def gliss(self, gliss: pitch_t | bool):
        """
        Set the gliss attribute of this Note, inplace
        """
        self._gliss = gliss if isinstance(gliss, bool) else asmidi(gliss)

    def __eq__(self, other: Note | str | float) -> bool:
        if isinstance(other, (int, float)):
            return self.pitch == other
        elif isinstance(other, str):
            try:
                pitch = pt.str2midi(other)
                return self.pitch == pitch
            except ValueError as e:
                raise ValueError(f"Cannot interpret '{other}' as a note: {e}")
        elif isinstance(other, Note):
            return hash(self) == hash(other)
        else:
            return False

    def copy(self) -> Self:
        out = self.__class__(self.pitch, dur=self.dur, amp=self.amp, gliss=self._gliss, tied=self.tied,
                             dynamic=self.dynamic, offset=self.offset, label=self.label,
                             _init=False)
        out.pitchSpelling = self.pitchSpelling
        out._scorestruct = self._scorestruct
        self._copyAttributesTo(out)
        assert out._parent is None
        return out

    def clone(self,
              pitch: pitch_t | None = None,
              dur: time_t | None = None,
              amp: float | None = None,
              offset: time_t | None | UnsetType = UNSET,
              gliss: pitch_t | bool | None = None,
              label='',
              tied: bool | None = None,
              dynamic='') -> Self:
        """
        Clone this note with overridden attributes

        Returns a new note
        """
        offset = self.offset if offset is UNSET else None if offset is None else asF(offset)  # type: ignore
        out = self.__class__(pitch=pitch if pitch is not None else self.pitch,
                             dur=asF(dur) if dur is not None else self.dur,
                             amp=amp if amp is not None else self.amp,
                             offset=offset,
                             gliss=gliss if gliss is not None else self.gliss,
                             label=label or self.label,
                             tied=tied if tied is not None else self.tied,
                             dynamic=dynamic or self.dynamic,
                             _init=False)
        if pitch is not None:
            if self.pitchSpelling:
                out.pitchSpelling = pt.transpose(self.pitchSpelling, out.pitch - self.pitch)
        else:
            out.pitchSpelling = self.pitchSpelling
        self._copyAttributesTo(out)
        return out

    def __hash__(self) -> int:
        hashsymbols = hash(tuple(self.symbols)) if self.symbols else 0
        return hash((self.pitch, self._dur, self.offset, self._gliss, self.label,
                     self.dynamic, self.tied, self.pitchSpelling, hashsymbols))

    def asChord(self, pitches: list[pitch_t] | None = None) -> Chord:
        """ Convert this Note to a Chord of one note """
        gliss: bool | list[float] = self.gliss if isinstance(self.gliss, bool) else [self.gliss]
        properties = self.properties.copy() if self.properties else None
        if pitches:
            notes = pitches
        else:
            notes = [self]
        chord = Chord(notes=notes,
                      dur=self.dur,
                      amp=self.amp,
                      offset=self.offset,
                      gliss=gliss,
                      label=self.label,
                      tied=self.tied,
                      dynamic=self.dynamic,
                      properties=properties,
                      _init=False)

        if self.symbols:
            for s in self.symbols:
                chord.addSymbol(s)
        if self.playargs:
            chord.playargs = self.playargs.copy()
        return chord

    def isRest(self) -> bool:
        """ Is this a Rest? """
        return self.pitch == 0

    def pitchRange(self) -> tuple[float, float] | None:
        return self.pitch, self.pitch

    def meanPitch(self) -> float | None:
        return self.pitch

    def timeShift(self, offset: time_t) -> Self:
        reloffset = self.relOffset()
        reloffset += offset
        if reloffset < 0:
            raise ValueError(f"Cannot shift to a negative offset, {offset=}, "
                             f"relative offset prior to shift: {self.relOffset()}, "
                             f"resulting offset: {reloffset}. ({self=})")
        return self.clone(offset=reloffset)

    def freqShift(self, freq: float) -> Self:
        """
        Return a copy of self, shifted in freq.

        Example::

            # Shifting a note by its own freq. sounds one octave higher
            >>> n = Note("C3")
            >>> n.freqShift(n.freq)
            C4
        """
        out = self.copy()
        out._setpitch(pt.f2m(self.freq + freq))
        return out

    def __lt__(self, other: pitch_t | Note) -> bool:
        if isinstance(other, Note):
            return self.pitch < other.pitch
        elif isinstance(other, (float, int, F)):
            return self.pitch < other
        elif isinstance(other, str):
            return self.pitch < pt.str2midi(other)
        else:
            raise NotImplementedError()

    def __gt__(self, other: pitch_t | Note) -> bool:
        if isinstance(other, Note):
            return self.pitch > other.pitch
        elif isinstance(other, (float, int, F)):
            return self.pitch > other
        elif isinstance(other, str):
            return self.pitch > pt.str2midi(other)
        else:
            raise NotImplementedError()

    def __abs__(self) -> Self:
        if self.pitch >= 0:
            return self
        return self.clone(pitch=-self.pitch)

    def _setpitch(self, pitch: float) -> None:
        interval = pitch - self.pitch
        self.pitch = pitch
        if self.pitchSpelling:
            self.pitchSpelling = pt.transpose(self.pitchSpelling, interval)

    @property
    def freq(self) -> float:
        """The frequency of this Note (according to the current A4 value)"""
        return pt.m2f(self.pitch)

    @freq.setter
    def freq(self, value: float) -> None:
        self.pitch = pt.f2m(value)

    @property
    def name(self) -> str:
        """The notename of this Note"""
        if self.isRest():
            return 'Rest'
        return self.pitchSpelling or pt.m2n(self.pitch)

    @name.setter
    def name(self, notename: str):
        self.pitchSpelling = notename
        self.pitch = pt.n2m(notename)

    @property
    def pitchclass(self) -> int:
        """The pitch-class of this Note (an int between 0-11)"""
        return round(self.pitch) % 12

    @property
    def cents(self) -> int:
        """The fractional part of this pitch, rounded to the cent"""
        return _tools.midicents(self.pitch)

    def scoringParts(self, config: CoreConfig | None = None) -> list[scoring.core.UnquantizedPart]:
        if self.isRest():
            notations = self.scoringEvents(config=config)
            assert len(notations) == 1
            n = notations[0]
            n.mergeableNext = False
            n.mergeablePrev = False
            from maelzel.scoring import attachment
            n.addAttachment(attachment.Breath(visible=False, horizontalPlacement='post'))
            return [scoring.core.UnquantizedPart(notations)]
        else:
            return super().scoringParts(config)


    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig | None = None,
                      parentOffset: F | None = None
                      ) -> list[scoring.Notation]:
        if not config:
            config = Workspace.active.config
        offset = self.absOffset() if parentOffset is None else self.relOffset() + parentOffset
        dur = self.dur

        def _mergeOptLists(a: list | None, b: list | None) -> list | None:
            return a+b if (a and b) else a or b

        if self.isRest():
            rest = scoring.Notation.makeRest(dur, offset=offset, dynamic=self.dynamic)
            if self.label:
                rest.addText(self.label, role='label')
            tempsymbols = self.properties.pop('.tempsymbols', None) if self.properties else None
            if (symbols := _mergeOptLists(tempsymbols, self.symbols)) is not None:
                for symbol in symbols:
                    if symbol.appliesToRests:
                        symbol.applyToNotation(rest, parent=self)
            return [rest]

        notation = scoring.Notation.makeNote(pitch=self.pitch,
                                             duration=asF(dur),
                                             offset=offset,
                                             gliss=bool(self.gliss),
                                             dynamic=self.dynamic,
                                             group=groupid)
        if self.pitchSpelling:
            notation.fixNotename(self.pitchSpelling, index=0)

        if self.tied:
            notation.tiedNext = True

        notes = [notation]
        if self.gliss and not isinstance(self.gliss, bool):
            offset = self.end if self.end is not None else None
            groupid = groupid or str(hash(self))
            notes[0].groupid = groupid
            assert self.gliss >= 12, f"self.gliss = {self.gliss}"
            notes.append(scoring.Notation.makeNote(pitch=self.gliss,
                                                   duration=0,
                                                   offset=offset,
                                                   group=groupid))
            if config['show.glissStemless']:
                from maelzel.scoring import attachment
                notes[-1].addAttachment(attachment.StemTraits(hidden=True))

        if self.label:
            notes[0].addText(self._scoringAnnotation(config=config))
        elif chainlabel := self.getProperty('.chainlabel'):
            notes[0].addText(self._scoringAnnotation(text=chainlabel, config=config))

        tempsymbols = self.properties.pop('.tempsymbols', None) if self.properties else None
        if (symbols := _mergeOptLists(tempsymbols, self.symbols)) is not None:
            for symbol in symbols:
                symbol.applyToTiedGroup(notes, parent=self)
        return notes

    def _asTableRow(self, config: CoreConfig | None = None) -> list[str]:
        config = config or Workspace.active.config

        if self.isRest():
            elements = ["REST"]
        else:
            notename = self.name
            if (unicodeaccidentals := config['reprUnicodeAccidentals']):
                full = unicodeaccidentals == 'full'
                notename = _util.unicodeNotename(notename, full=full)
            if self.tied:
                notename += "~"

            if self.symbols:
                notatedPitch = next((s for s in self.symbols
                                     if isinstance(s, _symbols.NotatedPitch)), None)
                if notatedPitch and notatedPitch.pitch != notename:
                    notename = f'{notename} ({notatedPitch.pitch})'

            elements = [notename]
            if config['reprShowFreq']:
                elements.append("%dHz" % int(self.freq))
            if self.amp is not None and self.amp < 1:
                elements.append("%ddB" % round(pt.amp2db(self.amp)))

        if self.dur:
            if self.dur >= MAXDUR:
                elements.append("dur=inf")
            elif config['reprDurationAsFraction']:
                elements.append(f"{_util.showF(self.dur, maxdenom=32, approxAsFloat=True, unicode=config['reprUnicodeFractions'])}♩")
            else:
                elements.append(f"{_util.showT(self.dur)}")
        if self.offset is not None:
            elements.append(f"offset={_util.showT(self.offset)}")

        if self.gliss:
            if isinstance(self.gliss, bool):
                elements.append(f"gliss={self.gliss}")
            else:
                elements.append(f"gliss={pt.m2n(self.gliss)}")

        if self.symbols:
            elements.append(f"symbols={self.symbols}")

        return elements

    def __repr__(self) -> str:
        if self.isRest():
            cfg = Workspace.active.config
            if cfg['reprDurationAsFraction']:
                parts = [f"{_util.showF(self.dur, maxdenom=32, approxAsFloat=True)}♩"]
            else:
                parts = [f"{_util.showT(self.dur)}"]
            if self.offset is not None:
                parts.append(f"offset={_util.showT(self.offset)}")
            if self.symbols:
                parts.append(f"symbols={(self.symbols)}")
            return "R:" + ":".join(parts)
        elements = self._asTableRow()
        if len(elements) == 1:
            return elements[0]
        else:
            s = ":".join(elements)
            return s

    def __float__(self) -> float: return float(self.pitch)

    def __int__(self) -> int: return int(self.pitch)

    def __add__(self, other: num_t) -> Self:
        if isinstance(other, (int, float)):
            out = self.copy()
            out._setpitch(self.pitch + other)
            out._gliss = self.gliss if isinstance(self.gliss, bool) else self.gliss + other
            return out
        raise TypeError(f"can't add {other} ({other.__class__}) to a Note")

    def __mul__(self, other: num_t) -> Self:
        # modify the duration
        if isinstance(other, (int, float, F)):
            return self.clone(dur=self.dur * other)
        raise TypeError(f"Can't multiply {other} ({other.__class__}) to a {type(self)}")

    def __xor__(self, freq) -> Self: return self.freqShift(freq)

    def __sub__(self, other: num_t) -> Self:
        return self + (-other)

    def quantizePitch(self, step=0.) -> Self:
        """
        Returns a new Note, rounded to step.

        If step is 0, the default quantization value is used (this can be
        configured via ``getConfig()['semitoneDivisions']``)

        .. note::
            - If this note has a gliss, the target pitch is also quantized
            - Any set pitch spelling is deleted
        """
        if self.isRest():
            # When an event is already part of a container (chain/voice), we
            # need to return a copy since otherwise it cannot be made part of
            # a different container. For example, if .quantizedPitch is
            # called on a chain, the returned events will be made part of
            # a newly constructed chain
            return self.copy() if self.parent else self

        if step == 0:
            step = 1 / Workspace.active.config['semitoneDivisions']
        out = self.clone(pitch=round(self.pitch / step) * step)
        if isinstance(self._gliss, (int, float)):
            out._gliss = round(out._gliss / step) * step
        out.pitchSpelling = ''
        return out

    def _resolveAmp(self,
                    config: CoreConfig,
                    dyncurve: DynamicCurve
                    ) -> float:
        """
        Resolves the amplitude of this event.

        This is mostly used internally to determine the amplitude corresponding
        to a given event.

        Args:
            config: the active config
            dyncurve: the active dynamic curve

        Returns:
            the playback amplitude as a float between 0 and 1
        """
        if self.amp is not None:
            return self.amp
        else:
            if config['play.useDynamics']:
                dyn = self.dynamic or config['play.defaultDynamic']
                return dyncurve.dyn2amp(dyn)
            return config['play.defaultAmplitude']

    def resolveAmp(self) -> float:
        """
        Resolves the amplitude of this event.

        The amplitude can be set explicitely (the ``.amp`` attribute) or can
        be set via the dynamic if ``config['play.useDynamics']`` is True.
        If no amplitude is set, a default amplitude via ``config['play.defaultAmplitude']``
        is used as fallback

        Returns:
            the playback amplitude as a float between 0 and 1
        """
        w = Workspace.active
        return self._resolveAmp(config=w.config, dyncurve=w.dynamicCurve)

    def _synthEvents(self,
                     playargs: synthevent.PlayArgs,
                     parentOffset: F,
                     workspace: Workspace,
                     ) -> list[synthevent.SynthEvent]:
        if self.isRest():
            return []
        conf = workspace.config
        struct = workspace.scorestruct
        if self.playargs is not None:
            playargs = playargs.updated(self.playargs)

        amp = self._resolveAmp(config=conf, dyncurve=workspace.dynamicCurve)
        glissdur = playargs.get('glisstime', 0.)
        linkednext = glissdur or self.gliss
        endpitch = self.resolveGliss() if linkednext else self.pitch
        absoffset = self.relOffset() + parentOffset
        startbeat = absoffset + playargs.get('skip', 0)
        endbeat = absoffset + playargs.get('end', self.dur)
        startsecs = float(struct.beatToTime(startbeat))
        endsecs = float(struct.beatToTime(endbeat))
        if startsecs >= endsecs:
            raise ValueError(f"Trying to play an event with 0 or negative duration: {endsecs-startsecs}. "
                             f"Object: {self}, {startbeat=}, {endbeat=}, {playargs=}")
        transp = playargs.get('transpose', 0.)
        startpitch = self.pitch + transp
        if glissdur > 0.:
            glissstart = float(struct.beatToTime(max(startbeat, endbeat - glissdur)))
            bps = [(startsecs,  startpitch,        amp),
                   (glissstart, startpitch,        amp),
                   (endsecs,    endpitch + transp, amp)]
        else:
            bps = [(startsecs, self.pitch + transp, amp),
                   (endsecs,   endpitch + transp,   amp)]

        event = synthevent.SynthEvent.fromPlayArgs(bps=bps, playargs=playargs)
        if playargs.automations:
            event.addAutomationsFromPlayArgs(playargs, scorestruct=struct)

        if self.tied or linkednext:
            event.linkednext = True
        return [event]

    def resolveGliss(self) -> float:
        """
        Resolve the target pitch for this note's glissando

        Returns:
            the target pitch or this note's own pitch if its target
            pitch cannot be determined
        """
        if not isinstance(self._gliss, bool):
            # .gliss is already a pitch, return that
            return self._gliss
        elif self._glissTarget:
            return self._glissTarget
        elif not self.parent:
            # .gliss is a bool, so we need to know the next event, but we are parentless
            return self.pitch

        self.parent._resolveGlissandi()
        return self._glissTarget or self.pitch

    def glissTarget(self) -> str:
        """
        The gliss target as notename

        Raises ValueError if this Note does not have a gliss

        Returns:
            The gliss target as notename
        """
        if not self.gliss:
            raise ValueError("This note does not have a glissando")

        if self._glissTarget:
            return pt.m2n(self._glissTarget)
        elif not isinstance(self._gliss, bool):
            return pt.m2n(self._gliss)
        elif self.parent:
            self.parent._resolveGlissandi()
            target = self._glissTarget
            return target if isinstance(target, str) else pt.m2n(target)
        else:
            return self.name

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Self:
        """
        A copy of self with the pitch transformed by the given callable

        Args:
            pitchmap: a function mapping ``pitch -> pitch``

        Returns:
            A copy of self with the pitch transformed
        """
        if self.isRest():
            return self
        pitch = pitchmap(self.pitch)
        gliss = self.gliss if isinstance(self.gliss, bool) else pitchmap(self.gliss)
        out = self.copy()
        out._setpitch(pitch)
        out._gliss = gliss
        return out


def Rest(dur: time_t | str, offset: time_t | None = None, label='', dynamic='') -> Note:
    """
    Create a Rest

    Args:
        dur: duration in beats
        offset:
        label:
        dynamic:

    Returns:

    """
    fdur = _tools.parseDuration(dur) if isinstance(dur, str) else asF(dur)
    if fdur <= 0:
        raise ValueError(f"A rest must have a possitive duration, got {dur}")
    return Note(pitch=0, dur=fdur,
                offset=None if offset is None else asF(offset),
                amp=0, label=label, dynamic=dynamic,
                _init=False)



class Chord(MEvent):
    """
    A Chord is a stack of Notes

    a Chord can be instantiated as::

        Chord(note1, note2, ...)
        Chord([note1, note2, ...])
        Chord("C4,E4,G4", ...)
        Chord("C4 E4 G4", ...)


    Where each note is either a Note, a notename ("C4", "E4+", etc) or a midinote

    Args:
        notes: the notes of this chord. Can be a list of pitches, where each
            pitch is either a fractional midinote or a notename (str); notes
            can be already created ``Note`` instances; or a string with multiple
            notes separated by a space
        amp: the amplitude of this chord. To specify a different amplitude for each
            pitch within the chord, first create a Note for each pitch with its
            corresponding amplitude and use that list as the *notes* argument
        dur: the duration of this chord (in quarternotes)
        offset: the offset time (in quarternotes)
        gliss: either a list of end pitches (with the same size as the chord), or
            True to leave the end pitches unspecified (a gliss to the next chord)
        label: if given, it will be used for printing purposes
        tied: if True, this chord should be tied to a following chord, if possible
        dynamic: a dynamic for this chord ('pp', 'mf', 'ff', etc). Dynamics range from
            'pppp' to 'ffff'
        fixed: if True, fix the spelling of the note to the notename given (this is
            only taken into account if the pitch was given as a notename). See also
            the configuration key `fixStringNotenames <config_fixstringnotenames>`.
            **NB**: it is possible to suffix the notename with a '!'
            mark in order to lock its spelling, independently of the configuration
            settings
        properties: properties are a space given to the user to attach any information to
            this object

    Attributes:
        amp: the amplitude of the chord itself (each note can have an individual amp)
        notes: the notes which build this chord
        tied: is this Chord tied to another Chord?
    """

    __slots__ = (
         'notes',
         'dynamic',
         '_gliss',
         '_glissTarget',
         '_notatedPitches',
    )

    def __init__(self,
                 notes: str | Sequence[Note | int | float | str],
                 dur: time_t | None = None,
                 *,
                 amp: float | None = None,
                 offset: time_t | None = None,
                 gliss: str | bool | Sequence[pitch_t] = False,
                 label='',
                 tied=False,
                 dynamic='',
                 properties: dict[str, Any] | None = None,
                 fixed=False,
                 _init=True
                 ) -> None:
        """
        a Chord can be instantiated as:

            Chord(note1, note2, ...) or
            Chord([note1, note2, ...])
            Chord("C4 E4 G4")

        where each note is either a Note, a notename ("C4", "E4+", etc), a midinote
        or a tuple (midinote, amp)

        Args:
            amp: the amplitude (volume) of this chord. This applies to all the notes in the chord,
                but each note can have its own amplitude, which is then multiplied by this factor
            dur: the duration of this chord (in quarternotes)
            offset: the offset time (in quarternotes)
            gliss: either a list of end pitches (with the same size as the chord), or
                True to leave the end pitches unspecified (a gliss to the next chord)
            label: if given, it will be used for printing purposes
            fixed: if True, any note in this chord which was given as a notename (str)
                will be fixed in the given spelling (**NB**: the spelling can also
                be fixed by suffixing any notename with a '!' sign)

        """
        self.amp: float | None = amp

        if dur is not None:
            dur = asF(dur)

        if not notes:
            super().__init__(dur=dur if dur is not None else F1, offset=None, label=label)
            return

        symbols = None
        # notes might be: Chord([n1, n2, ...]) or Chord("4c 4e 4g", ...)
        if isinstance(notes, str):
            if ':' in notes:
                props = _tools.parseNote(notes)
                dur = dur if dur is not None else props.dur
                pitches = props.notename if isinstance(props.notename, list) else [props.notename]
                if p := props.keywords:
                    offset = offset or p.pop('offset', None)
                    dynamic = dynamic or p.pop('dynamic', '')
                    tied = tied or p.pop('tied', False)
                    gliss = gliss or p.pop('gliss', False)
                    fixed = p.pop('fixPitch', False) or fixed
                    label = label or p.pop('label', '')
                    properties = p if not properties else p | properties
                if props.symbols:
                    symbols = props.symbols
                if props.spanners:
                    for spanner in props.spanners:
                        self.addSpanner(spanner)
                notes = [Note(pitch, amp=amp, fixed=fixed) for pitch in pitches]
            elif ',' in notes:
                notes = [Note(n.strip(), amp=amp, fixed=fixed) for n in notes.split(',')]
            else:
                notes = [Note(n.strip(), amp=amp, fixed=fixed) for n in notes.split()]

        if not _init:
            notes2 = _cast(list[Note], notes)
        else:
            notes2: list[Note] = []
            for n in notes:
                if isinstance(n, Note):
                    if n.offset is None:
                        notes2.append(n)
                    else:
                        notes2.append(n.clone(offset=None))
                elif isinstance(n, (int, float, str)):
                    notes2.append(Note(n, amp=amp))
                else:
                    raise TypeError(f"Expected a Note or a pitch, got {n}")
            notes2.sort(key=lambda n: n.pitch)
            if any(n.tied for n in notes2):
                tied = True

            if not isinstance(gliss, bool):
                gliss = pt.as_midinotes(gliss)
                if not len(gliss) == len(notes):
                    raise ValueError(f"The destination chord of the gliss should have "
                                     f"the same length as the chord itself, {notes=}, {gliss=}")

        assert offset is None or isinstance(offset, F)
        super().__init__(dur=dur if dur is not None else F1,
                         offset=offset, label=label, properties=properties,
                         tied=tied, amp=amp, dynamic=dynamic)
        self.notes: list[Note] = notes2
        """The notes in this chord, each an instance of Note"""

        self._gliss: bool | list[float] = gliss  # type: ignore
        self._glissTarget: list[float] | None = None
        if symbols:
            for symbol in symbols:
                self.addSymbol(symbol)

    @property
    def gliss(self) -> list[float] | bool:
        return self._gliss

    @gliss.setter
    def gliss(self, gliss: bool | list[pitch_t]):
        """
        Set the gliss attribute of this Note, inplace
        """
        if isinstance(gliss, bool):
            self._gliss = gliss
            self._glissTarget = None
        else:
            if not isinstance(gliss, (list, tuple)):
                raise TypeError(f"Expected a list/tuple of pitches, got {gliss}")
            if len(gliss) != len(self.notes):
                raise ValueError(f"The number of pitches for the target of the glissando "
                                 f"should match the number of pitches in this chord, "
                                 f"{gliss=}, {self=}")
            self._gliss = [asmidi(pitch) for pitch in gliss]

    def copy(self):
        notes = [n.copy() for n in self.notes]
        out = self.__class__(notes=notes, dur=self.dur, amp=self.amp, offset=self.offset,
                             gliss=self.gliss, label=self.label, tied=self.tied,
                             dynamic=self.dynamic,
                             _init=False)
        self._copyAttributesTo(out)
        return out

    def __len__(self) -> int:
        return len(self.notes)

    @_overload
    def __getitem__(self, idx: int) -> Note: ...

    @_overload
    def __getitem__(self, idx: slice) -> Chord: ...

    def __getitem__(self, idx):
        out = self.notes.__getitem__(idx)
        if isinstance(out, list):
            out = self.__class__(out)
        return out

    def __iter__(self) -> Iterator[Note]:
        return iter(self.notes)

    @property
    def name(self) -> str:
        return ",".join(self._bestSpelling())

    def setNotatedPitch(self, notenames: str | list[str]) -> None:
        if isinstance(notenames, str):
            return self.setNotatedPitch(notenames.split())
        if not len(notenames) == len(self.notes):
            raise ValueError(f"The number of given fixed spellings ({notenames}) does not correspond"
                             f"to the number of notes in this chord ({self._bestSpelling()})")
        for notename, n in zip(notenames, self.notes):
            if notename:
                n.pitchSpelling = notename

    def _canBeLinkedTo(self, other: mobj.MObj) -> bool:
        if other.isRest():
            return False
        if self._gliss is True:
            return True
        if isinstance(other, Note):
            if self.tied and any(p == other.pitch for p in self.pitches):
                return True
            else:
                logger.debug(f"Chord {self} is tied, but {other} has no pitches in common")
        elif isinstance(other, Chord):
            if self.tied and any(p in other.pitches for p in self.pitches):
                return True
            else:
                logger.debug(f"Chord {self} is tied, but {other} has no pitches in common")
        return False

    @classmethod
    def grace(cls,
              notes: str | Sequence[int | float | str],
              stemless=False,
              slash=False,
              value: F | str | float | None = None,
              parenthesis=False,
              hidden=False,
              **kws) -> Self:
        """
        Class method to create a grace chord

        Args:
            notes: the pitches of the grace chord
            stemless: if is stemless?
            slash: slashed stem
            value: the rhythic value of the grace chord ("1/2" or F(1, 2)=8th note,
                "1/8"=32nd note, etc.
            parenthesis: are noteheads parenthesized?
            hidden: should the whole note be hidden?
            **kws: keyword args passed to the Chord constructor

        Returns:
            the grace chord
        """
        chord = cls(notes=notes, dur=0, **kws)
        _customizeGracenote(chord,
                            stemless=stemless,
                            slash=slash,
                            value=value,
                            parenthesis=parenthesis, hidden=hidden)
        return chord

    def mergeWith(self, other: Chord) -> Chord | None:
        if not isinstance(other, Chord):
            return None

        if not self.tied or other.gliss or self.pitches != other.pitches:
            return None

        if any(n1 != n2 for n1, n2 in zip(self.notes, other.notes)):
            return None

        return self.clone(dur=self.dur + other.dur)

    def pitchRange(self) -> tuple[float, float] | None:
        return min(n.pitch for n in self.notes), max(n.pitch for n in self.notes)

    def meanPitch(self) -> float | None:
        return sum(n.pitch for n in self.notes) / len(self.notes)

    def glissTarget(self) -> list[str]:
        """
        The gliss targets as a list of notenames

        Raised ValueError if this chord does not have a gliss

        Returns:
            The gliss target as notename
        """
        if not self.gliss:
            raise ValueError("This Chord does not have a glissando")
        elif self._glissTarget:
            return [pt.m2n(pitch) for pitch in self._glissTarget]
        elif not isinstance(self._gliss, bool):
            return [pt.m2n(pitch) for pitch in self._gliss]
        elif self.parent:
            self.parent._resolveGlissandi()
            assert self._glissTarget is not None
            return [pt.m2n(pitch) for pitch in self._glissTarget]
        else:
            return [note.name for note in self.notes]

    def resolveGliss(self) -> list[float]:
        """
        Resolve the target pitch for this chord's glissando

        Returns:
            the target pitch or this note's own pitch if its target
            pitch cannot be determined
        """
        if not self.gliss:
            raise ValueError("This Chord does not have a glissando")

        if not isinstance(self.gliss, bool):
            assert all(isinstance(_, (int, float)) for _ in self.gliss)
            return self.gliss

        if self._glissTarget:
            return self._glissTarget

        if not self.parent:
            # .gliss is a bool, so we need to know the next event, but we are parentless
            return self.pitches

        self.parent._update()
        if self._glissTarget is None:
            self._glissTarget = self.pitches
        return self._glissTarget

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig | None = None,
                      parentOffset: F | None = None
                      ) -> list[scoring.Notation]:
        if not config:
            config = Workspace.active.config
        notenames = [note.name for note in self.notes]
        annot = '' if not self.label else self._scoringAnnotation(config=config)
        dur = self.dur
        offset = self.absOffset() if parentOffset is None else self.relOffset() + parentOffset
        notation = scoring.Notation.makeChord(pitches=notenames,
                                              duration=dur,
                                              offset=offset,
                                              annotation=annot,
                                              group=groupid,
                                              dynamic=self.dynamic,
                                              tiedNext=self.tied)
        if chainlabel := self.getProperty('.chainlabel'):
            notation.addText(self._scoringAnnotation(chainlabel, config=config))

        # Transfer any pitch spelling
        for i, note in enumerate(self.notes):
            if note.pitchSpelling:
                notation.fixNotename(note.pitchSpelling, i)

        # Add gliss.
        notations = [notation]
        if self.gliss:
            notation.gliss = True
            if not isinstance(self.gliss, bool):
                groupid = scoring.core.makeGroupId(groupid)
                notation.groupid = groupid
                endEvent = scoring.Notation.makeChord(pitches=self.gliss, duration=0,
                                                      offset=self.end, group=groupid)
                if config['show.glissStemless']:
                    from maelzel.scoring import attachment
                    endEvent.addAttachment(attachment.StemTraits(hidden=True))
                    # endEvent.stem = 'hidden'
                notations.append(endEvent)

        if self.symbols:
            for s in self.symbols:
                s.applyToNotation(notation, parent=self)

        # Transfer note symbols
        for i, n in enumerate(self.notes):
            if n.symbols:
                for symbol in n.symbols:
                    if isinstance(symbol, _symbols.NoteheadSymbol):
                        symbol.applyToPitch(notation, idx=i, parent=n)
                    else:
                        logger.debug(f"Cannot apply symbol {symbol} to a pitch inside chord {self}")

        return notations

    def __hash__(self):
        if isinstance(self.gliss, bool):
            glisshash = int(self.gliss)
        elif isinstance(self.gliss, list):
            glisshash = hash(tuple(self.gliss))
        else:
            glisshash = hash(self.gliss)
        symbolshash = hash(tuple(self.symbols)) if self.symbols else 0

        data = (self.dur, self.offset, self.label, glisshash, self.dynamic,
                self.tied, symbolshash,
                *(hash(n) for n in self.notes))
        return hash(data)

    def append(self, note: float | str | Note) -> None:
        """ append a note to this Chord """
        note = note if isinstance(note, Note) else Note(note)
        if note.freq < 17:
            logger.debug(f"appending a note with very low freq: {note.freq}")
        self.notes.append(note)
        self._changed()

    def extend(self, notes) -> None:
        """ extend this Chord with the given notes """
        for note in notes:
            self.notes.append(note if isinstance(note, Note) else Note(note))
        self._changed()

    def insert(self, index: int, note: pitch_t) -> None:
        """
        Insert a note in this chord, in place

        Args:
            index: where to insert it
            note: a Note or a pitch
        """
        self.notes.insert(index, note if isinstance(note, Note) else Note(note))
        self._changed()

    def transposeTo(self, fundamental: pitch_t) -> Chord:
        """
        Return a copy of self, transposed to the new fundamental

        .. note::
            the fundamental is the lowest note in the chord

        Args:
            fundamental: the new lowest note in the chord

        Returns:
            A Chord transposed to the new fundamental
        """
        step = asmidi(fundamental) - self[0].pitch
        return self.transpose(step)

    def freqShift(self, freq: float) -> Chord:
        """
        Return a copy of this chord shifted in frequency
        """
        return Chord([note.freqShift(freq) for note in self])

    def quantizePitch(self, step=0.) -> Chord:
        """
        Returns a copy of this chord, with the pitches quantized.

        Two notes with the same pitch are considered equal if they quantize to the same
        pitch, independently of their amplitude. Amplitudes of equal notes are accumulated
        """
        if step == 0:
            step = 1 / Workspace.active.config['semitoneDivisions']
        notes = {}
        for note in self:
            note2 = note.quantizePitch(step)
            if accumnote := notes.get(note2.pitch):
                accumnote.amp += note2.amp
            else:
                notes[note2.pitch] = note2
        return self.clone(notes=notes.values())

    def __setitem__(self, i: int, note: pitch_t) -> None:
        self.notes.__setitem__(i, note if isinstance(note, Note) else Note(note))
        self._changed()

    def __add__(self, other: pitch_t) -> Chord:
        if isinstance(other, Note):
            # append the note
            s = self.copy()
            s.append(other)
            return s
        elif isinstance(other, (int, float)):
            # transpose
            s = [n + other for n in self]
            return Chord(s)
        elif isinstance(other, (Chord, str)):
            # join chords together
            return Chord(self.notes + _asChord(other).notes)
        raise TypeError(f"Can't add a Chord to a {other.__class__.__name__}")

    def loudest(self, n: int) -> Chord:
        """
        Return a new Chord with the loudest `n` notes from this chord
        """
        out = self.copy()
        out.sort(key='amp', reverse=True)
        return out[:n]

    def sort(self, key='pitch', reverse=False) -> None:
        """
        Sort **INPLACE**.

        If inplace sorting is undesired, use ``x = chord.copy(); x.sort()``
        By default, chords are sorted by pitch, from low to high

        Args:
            key: either 'pitch' or 'amp'
            reverse: similar as sort
        """
        if key == 'pitch':
            self.notes.sort(key=lambda n: n.pitch, reverse=reverse)
        elif key == 'amp':
            self.notes.sort(key=lambda n: n.amp if n.amp is not None else 1.0, reverse=reverse)
        else:
            raise KeyError(f"Unknown sort key {key}. Options: 'pitch', 'amp'")

    @property
    def pitches(self) -> list[float]:
        """
        The pitches of this chord (a list of midinotes)
        """
        return [n.pitch for n in self.notes]

    def resolveAmps(self,
                    config: CoreConfig,
                    dyncurve: DynamicCurve,
                    ) -> list[float]:
        useDynamics = config['play.useDynamics']
        if self.amp is not None:
            chordamp = self.amp
        else:
            if not useDynamics:
                chordamp = config['play.defaultAmplitude']
            else:
                dyn = self.dynamic or config['play.defaultDynamic']
                chordamp = dyncurve.dyn2amp(dyn)
        amps = []
        for n in self.notes:
            if n.amp:
                amps.append(n.amp)
            elif n.dynamic and useDynamics:
                amps.append(dyncurve.dyn2amp(n.dynamic))
            else:
                amps.append(chordamp)
        return amps

    def _synthEvents(self,
                     playargs: synthevent.PlayArgs,
                     parentOffset: F,
                     workspace: Workspace
                     ) -> list[synthevent.SynthEvent]:
        if not self.notes:
            return []
        conf = workspace.config
        struct = workspace.scorestruct
        if self.playargs:
            playargs = playargs.updated(self.playargs)

        if conf['chordAdjustGain'] and all(n.amp is None for n in self.notes):
            globalgain = 1/math.sqrt(len(self.notes))
        else:
            globalgain = 1.

        startbeat = self.relOffset() + parentOffset
        startbeat = max(startbeat, playargs.get('skip', F0))
        endbeat = min(startbeat + self.dur, playargs.get('end', float('inf')))
        startsecs = float(struct.beatToTime(startbeat))
        endsecs = float(struct.beatToTime(endbeat))
        endpitches = self.pitches if not self.gliss else self.resolveGliss()
        amps = self.resolveAmps(config=conf, dyncurve=workspace.dynamicCurve)
        transpose = playargs.get('transpose', 0.)
        glissdur = playargs.get('glisstime', 0)
        linkednext = self.gliss or glissdur > 0
        if glissdur > endbeat - startbeat:
            glissdur = endbeat - startbeat

        synthevents = []
        for note, endpitch, amp in zip(self.notes, endpitches, amps):
            startpitch = note.pitch + transpose
            amp *= globalgain
            bps = [(float(startsecs), startpitch, amp)]
            if glissdur:
                glissabstime = float(struct.beatToTime(endbeat - glissdur))
                bps.append((glissabstime, startpitch, amp))
            bps.append((float(endsecs), endpitch+transpose, amp))
            event = synthevent.SynthEvent.fromPlayArgs(bps=bps, playargs=playargs)
            if playargs.automations:
                event.addAutomationsFromPlayArgs(playargs, scorestruct=struct)
            if linkednext or self._isNoteTied(note):
                event.linkednext = True
            synthevents.append(event)
        return synthevents

    def tieNotes(self, notes: Sequence[Note | pitch_t]) -> None:
        """
        Marks the given notes as tied, in place

        Raises ValueError if any note/pitch is not present in the given chord a
        """
        changed = False
        for n in notes:
            pitch = n.pitch if isinstance(n, Note) else asmidi(n)
            if note := self.findNote(pitch):
                note.tied = True
                changed = True
            else:
                raise ValueError(f"Pitch {n} not present in this chord: {self}")
        if changed:
            self.tied = True
            self._changed()

    def tieCommonNotes(self, other: Chord, tolerance=0.) -> list[Note]:
        notes = self.commonNotes(other, tolerance=tolerance)
        if not notes:
            return []
        for note in notes:
            note.tied = True
        self.tied = True
        self._changed()
        return notes

    def commonNotes(self, other: Chord, tolerance=0.) -> list[Note]:
        """

        """
        otherpitches = other.pitches
        common = [n for n in self.notes
            if any(abs(n.pitch - p) <= tolerance for p in otherpitches)]
        return common

    def _isNoteTied(self, note: Note) -> bool:
        """
        Query if the given note within this chord is tied to the following note/chord

        For a note within a chord to be tied the chord needs to be tied and the
        next event needs to have a pitch equal to the pitch of that note

        Args:
            note: a note within this chord

        Returns:
            True if this note is tied to the next chord/note

        """
        if not self.tied:
            return False

        if not self.parent:
            return True

        nextitem = self.parent.nextItem(self)
        if not nextitem:
            return False
        elif isinstance(nextitem, Chord):
            istied = next((candidate.pitch == note.pitch for candidate in nextitem), None) is not None
        elif isinstance(nextitem, Note):
            istied = nextitem.pitch == note.pitch
        else:
            return False
        if istied and not self.tied:
            logger.warning(f"A note ({note}) in this chord is tied, but the chord itself is "
                           f"not tied (chord={self})")
        return istied

    def _bestSpelling(self) -> tuple[str, ...]:
        notenames = [n.pitchSpelling + '!' if n.pitchSpelling else n.name
                     for n in self.notes]
        from maelzel.scoring import enharmonics
        return enharmonics.bestChordSpelling(notenames)

    def __repr__(self):
        # «4C+14,4A 0.1q -50dB»
        elements = [" ".join(self._bestSpelling())]
        if self.dur:
            if self.dur >= MAXDUR:
                elements.append("dur=inf")
            else:
                elements.append(f"{float(self.dur):.3g}♩")
        if self.offset is not None:
            elements.append(f'offset={float(self.offset):.3g}')
        if self.gliss:
            if isinstance(self.gliss, bool):
                elements.append("gliss")
            else:
                endpitches = ','.join([pt.m2n(_) for _ in self.gliss])
                elements.append(f"gliss={endpitches}")
        if self.dynamic:
            elements.append(self.dynamic)

        if len(elements) == 1:
            return f'‹{elements[0]}›'
        else:
            return f'‹{elements[0].ljust(3)} {" ".join(elements[1:])}›'

    def dump(self, indents=0, forcetext=False):
        elements = f'offset={self.offset}, dur={self.dur}, gliss={self.gliss}'
        print(f"{'  '*indents}Chord({elements})")
        if self.playargs:
            print("  "*(indents+1), self.playargs)
        if self.symbols:
            print("  "*(indents+1), "symbols:", self.symbols)
        for n in reversed(self.notes):
            print("  "*(indents+2), repr(n))

    def mappedAmplitudes(self, curve, db=False) -> Chord:
        """
        Return new Chord with the amps of the notes modified according to curve

        Example::

            # compress all amplitudes to 30 dB
            >>> import bpf4 as bpf
            >>> chord = Chord([Note(60, amp=0.5), Note(65, amp=0.1)])
            >>> curve = bpf.linear(-90, -30, -30, -12, 0, 0)
            >>> chord2 = chord.mappedAmplitudes(curve, db=True)

        Args:
            curve: a func mapping ``amp -> amp``
            db: if True, the value returned by func is interpreted as dB
                if False, it is interpreted as amplitude (0-1)

        Returns:
            the resulting chord
        """
        notes = []
        if db:
            for note in self:
                db = curve(pt.amp2db(note.amp if note.amp is not None else 1.0))
                notes.append(note.clone(amp=pt.db2amp(db)))
        else:
            for note in self:
                amp2 = curve(note.amp if note.amp is not None else 1.0)
                notes.append(note.clone(amp=amp2))
        return Chord(notes)

    def setAmplitudes(self, amp: float) -> None:
        """
        Set the amplitudes of the notes in this chord to `amp` (inplace)

        This modifies the Note objects within this chord, without modifying
        the amplitude of the chord itself

        .. note::

            Each note (instance of Note) within a Chord can have its own ampltidue.
            The Chord itself can also have an amplitude, which is used as fallback
            for any note with unset amplitude. These two amplitudes do not multiply

        Args:
            amp: the new amplitude for the notes of this chord

        Example
        ~~~~~~~

            >>> chord = Chord("3f 3b 4d# 4g#", dur=4)
            >>> chord.amp = 0.5                 # Fallback amplitude
            >>> chord[0:-1].setAmplitude(0.01)  # Sets the amp of all notes but the last
            >>> chord.synthEvents()
            [SynthEvent(delay=0, dur=4, gain=0.5, chan=1, fade=(0.02, 0.02), instr=_piano)
             bps 0.000s: 53       0.01
                 4.000s: 53       0.01    ,
             SynthEvent(delay=0, dur=4, gain=0.5, chan=1, fade=(0.02, 0.02), instr=_piano)
             bps 0.000s: 59       0.01
                 4.000s: 59       0.01    ,
             SynthEvent(delay=0, dur=4, gain=0.5, chan=1, fade=(0.02, 0.02), instr=_piano)
             bps 0.000s: 63       0.01
                 4.000s: 63       0.01    ,
             SynthEvent(delay=0, dur=4, gain=0.5, chan=1, fade=(0.02, 0.02), instr=_piano)
             bps 0.000s: 68       0.5
                 4.000s: 68       0.5     ]

        Notice that the last event, corresponding to the highest note, has taken the ampltidue
        of the chord (0.5) as default, since it was unset. The other notes have an amplitude
        of 0.01, as set via :meth:`~Chord.setAmplitudes`
        """
        return self.scaleAmplitudes(factor=0., offset=amp)

    def scaleAmplitudes(self, factor: float, offset=0.0) -> None:
        """
        Scale the amplitudes of the notes within this chord **inplace**

        .. note::

            Each note (instance of Note) within a Chord can have its own ampltidue.
            The Chord itself can also have an amplitude, which is used as fallback
            for any note with unset amplitude. These two amplitudes do not multiply

        Args:
            factor: a factor to multiply the amp of each note
            offset: a value to add to the amp of each note
        """
        for n in self.notes:
            amp = n.amp if n.amp is not None else self.amp if self.amp is not None else 1.0
            n.amp = amp * factor + offset

    def equalize(self, curve: Callable[[float], float]) -> None:
        """
        Scale the amplitude of the notes according to their frequency, **inplace**

        Args:
            curve: a func mapping freq to gain
        """
        for note in self:
            gain = curve(note.freq)
            if note.amp is None:
                note.amp = gain
            else:
                note.amp *= gain

    def _isTooCrowded(self) -> bool:
        """
        Is this chord two dense that it needs to be arpeggiated when shown?
        """
        if len(self.notes) < 3:
            return False
        n0, n1 = self.notes[0], self.notes[1]
        for n2 in self.notes[2:]:
            if abs(n0.pitch - n1.pitch) <= 1 and abs(n1.pitch - n2.pitch) <= 1:
                return True
            n0, n1 = n1, n2
        return False

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Chord:
        transpositions = [pitchmap(note.pitch) - note.pitch for note in self.notes]
        newnotes = [n.transpose(interval) for n, interval in zip(self.notes, transpositions)]
        return self.clone(notes=newnotes,
                          gliss=self.gliss if isinstance(self.gliss, bool) else list(map(pitchmap, self.gliss)))

    def findNote(self, pitch: pitch_t) -> Note | None:
        """
        Find a note within this Chord

        Args:
            pitch: the pitch to match (a midinote or a notename)

        Returns:
            the matched note or None. If multiple notes within this chord share
            the same pitch only one is returned

        """
        midi = pt.str2midi(pitch) if isinstance(pitch, str) else pitch
        return next((n for n in self.notes if n.pitch == midi), None)



def _asChord(obj, amp: float | None = None, dur: float | None = None) -> Chord:
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
    elif isinstance(obj, (int, float)):
        out = Chord([Note(obj)])
    else:
        raise ValueError(f"cannot express this as a Chord: {obj}")
    return out if amp is None and dur is None else out.clone(amp=amp, dur=dur)


@_overload
def Grace(pitch: pitch_t,
          slash=False,
          stemless=False,
          offset: time_t | None = None,
          value: F | None = None,
          hidden=False,
          parenthesis=False,
          **kws
          ) -> Note:
    ...


@_overload
def Grace(pitch: Sequence[pitch_t],
          slash=False,
          stemless=False,
          offset: time_t | None = None,
          value: F | None = None,
          hidden=False,
          parenthesis=False,
          **kws
          ) -> Note | Chord:
    ...


def Grace(pitch: pitch_t | Sequence[pitch_t],
          slash=False,
          stemless=False,
          offset: time_t | None = None,
          value: F | None = None,
          hidden=False,
          parenthesis=False,
          **kws
          ) -> Note | Chord:
    """
    Create a gracenote (a note or chord)

    The resulting gracenote will be a Note or a Chord, depending on pitch.

    .. note::
        A gracenote is just a regular note or chord with a duration of 0.
        This function is here for visibility and to allow to customize
        attributes specific to a gracenote

    Args:
        pitch: a single pitch (as midinote, notename, etc), a list of pitches or string
            representing one or more pitches
        offset: the offset of this gracenote. Normally a gracenote should not have an explicit offset
        slash: if True, the gracenote will be marked as slashed
        stemless: if True, hide the stem of this gracenote
        value: the rhythmic value to use (1/2=eighth note, 1/4=sixteenth note, etc.)

    Returns:
        a Note if one pitch is given, a Chord if a list of pitches are passed instead.
        A gracenote is basically a note/chord with 0 duration

    Example
    ~~~~~~~

        >>> grace = Grace('4F')
        >>> grace2 = Note('4F', dur=0)
        >>> grace == grace2
        True
        >>> gracechord = Grace('4F 4A')
        >>> gracechord2 = Chord('4F 4A', dur=0)
        >>> gracechord == gracechord2
        True

    """
    out = asEvent(pitch, dur=0, offset=offset, **kws)
    assert isinstance(out, (Note, Chord))
    _customizeGracenote(out, slash=slash, stemless=stemless, value=value, hidden=hidden, parenthesis=parenthesis)
    return out


def _customizeGracenote(grace: Note | Chord,
                        slash=False,
                        stemless=False,
                        value: F | str | int | float | None = None,
                        hidden=False,
                        parenthesis=False,
                        ) -> None:
    if hidden:
        grace.addSymbol(_symbols.Hidden())
    else:
        if stemless and (slash or value):
            logger.debug("A gracenote cannot be stemless and have a slashed stem or "
                         "a custom value...")
            slash = False
            value = None
        if stemless:
            grace.addSymbol(_symbols.Stem(hidden=True))
        elif slash or value:
            grace.addSymbol(_symbols.Gracenote(slash=True,
                                               value=F(1, 2) if not value else asF(value)))
        if parenthesis:
            grace.addSymbol(_symbols.Notehead(parenthesis=True))


def asEvent(obj, **kws) -> MEvent:
    """
    Convert obj to a Note or Chord, depending on the input itself

    =============================  ==========
    Input                          Output
    =============================  ==========
    int, float (midinote)          Note
    list (of notes)                Chord
    notename as string ("C4")      Note
    str with notenames ("C4 E4")   Chord
    =============================  ==========

    Args:
        obj: the object to convert
        kws: any keyword passed to the constructor (Note, Chord)

    Returns:
        the resulting object, either a Note or a Chord

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> asEvent("4C")     # a Note
        4C
        >>> asEvent("4C E4")  # a Chord
        ‹4C 4E›
        >>> asEvent("4C:1/3:accent")
        4C:0.333♩:symbols=[Articulation(kind=accent)]
        # Internally this note has a duration of 1/3
        >>> asEvent("4C,4E:0.5:mf")
        ‹4C 4E 0.5♩ mf›

    """
    if isinstance(obj, MEvent):
        return obj
    elif isinstance(obj, str):
        if " " in obj:
            return Chord(obj.split(), **kws)
        elif ":" or "," in obj:
            notedef = _tools.parseNote(obj)
            dur = kws.pop('dur', None) or notedef.dur

            if notedef.keywords:
                kws = notedef.keywords | kws
            if isinstance(notedef.notename, list):
                if len(notedef.notename) > 1:
                    out = Chord(notedef.notename, dur=dur, **kws)
                else:
                    out = Note(notedef.notename[0], dur=dur, **kws)
            elif notedef.notename == 'rest':
                if dur is None:
                    raise ValueError(f"A rest needs a duration, got {obj}")
                out = Rest(dur=dur, **kws)
            else:
                out = Note(notedef.notename, dur=dur, **kws)
            if notedef.symbols:
                for symbol in notedef.symbols:
                    out.addSymbol(symbol)
            if notedef.spanners:
                for spanner in notedef.spanners:
                    out.addSpanner(spanner)

        elif " " in obj:
            out = Chord(obj.split(), **kws)
        else:
            out = Note(obj, **kws)
    elif isinstance(obj, (list, tuple)):
        out = Chord(obj, **kws)
    elif isinstance(obj, (int, float)):
        out = Note(obj, **kws)
    else:
        raise TypeError(f"Cannot convert {obj} to a Note or Chord")

    return out
