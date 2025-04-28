from __future__ import annotations

from maelzel.common import F
from maelzel.core.mobj import MObj
import maelzel.core.symbols as _symbols
from maelzel.core import synthevent
from maelzel.scoring import definitions
from maelzel import mathutils


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing_extensions import Self
    from maelzel.core import chain
    from typing import Any, Callable
    from maelzel.common import time_t, location_t, num_t, beat_t
    from maelzel.core.mobj import MContainer


class MEvent(MObj):
    """
    A discrete event in time (a Note, Chord, etc)

    Args:
        dur: the duration of the object, in beats
        offset: an explicit offset (start time) in beats
        amp: an amplitude value
        parent: the parent of this object, if any
        properties: user-defined properties
        symbols: notations symbols attached to this event
        label: a label for this object
        dynamic: a dynamic as string, if applicable
        tied: is this event tied to the next event?
    """
    __slots__ = ('tied', 'amp', 'dynamic', '_glissTarget')

    def __init__(self,
                 dur: F,
                 offset: F = None,
                 amp: float | None = None,
                 parent: MContainer = None,
                 properties: dict[str, Any] = None,
                 symbols: list[_symbols.Symbol] = None,
                 label='',
                 dynamic='',
                 tied=False):
        if not isinstance(dur, F):
            raise ValueError(f"Invalid (None) duration for {self}")
        super().__init__(dur=dur, offset=offset, label=label, parent=parent,
                         properties=properties, symbols=symbols)
        self.tied: bool = tied
        """Is this event tied?"""

        self.amp: float | None = amp
        "The playback amplitude 0-1 of this note"

        if dynamic:
            if dynamic.endswith('!'):
                dynamic = dynamic[:-1]
                self.addSymbol(_symbols.Dynamic(dynamic, force=True))
            assert dynamic in definitions.dynamicLevels

        self.dynamic: str = dynamic
        """A musical dynamic (*pppp, ppp, ..., mp, mf, f, ..., ffff*)"""

        self._glissTarget: float = 0.

    def linkedNext(self) -> bool:
        """
        Is this event linked to the next?

        An event is linked to a next event if it is tied or has
        glissando set to True where this applies. This is not the
        case if the event has a gliss value set to other than True
        """
        return self.tied or (self.gliss is True)

    def linkedPrev(self) -> bool:
        """
        Is this event linked to the previous?
        """
        if not self.parent:
            return False
        prev = self.parent.previousEvent(self)
        if not prev:
            return False
        assert isinstance(prev, MEvent)
        return prev.linkedNext()

    def root(self) -> MContainer | None:
        """
        The root of this object or None if this object has no parent

        Returns:
            the root which contains this object
        """
        return None if self.parent is None else self.parent.root()

    @property
    def gliss(self):
        """The end target of this event, if any"""
        return False

    @gliss.setter
    def gliss(self, gliss):
        """
        Set the gliss attribute of this Note, inplace
        """
        raise NotImplementedError("gliss setter not implemented")

    def isRest(self) -> bool:
        """Is this a rest?"""
        return False

    def isGracenote(self) -> bool:
        """
        Is this a grace note?

        A grace note has a pitch but no duration

        Returns:
            True if this can be considered a grace note
        """
        return not self.isRest() and self.dur == 0

    def _asVoices(self) -> list[chain.Voice]:
        from maelzel.core import chain
        return [chain.Voice([self])]

    def addSymbol(self, *args, **kws) -> Self:
        """
        Add a notation symbol to this object

        Notation symbols are attributes attached to **one event**
        and are intended for **notation only**. Such symbols include articulations,
        ornaments, fermatas but also properties, like color, size, etc.
        Also customizations like notehead shape, bend signs, all are
        considered symbols. Notation symbols spanning across multiple events
        (like slurs, crescendo hairpins, lines, etc.) are considered *spanners* and
        are added via :meth:`~MObj.addSpanner`

        Some symbols are exclusive, meaning that adding a symbol of this kind will
        replace a previously set symbol. Exclusive symbols include any properties
        (color, size, etc) and other customizations like notehead shape

        .. note::

            Dynamics are not treated as symbols since they can also be used for playback

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

        ============  ==========================================================
        Symbol        Possible Values
        ============  ==========================================================
        text          any text
        notehead      cross, harmonic, triangleup, xcircle, triangle, rhombus,
                      square, rectangle
        articulation  accent, staccato, tenuto, marcato, staccatissimo, etc.
        size          A relative size (0=default, 1, 2, …=bigger, -1, -2, … = smaller)
        color         a css color
        ============  ==========================================================

        """
        symbol = _symbols.parseAddSymbol(args, kws)
        self._addSymbol(symbol)
        if isinstance(symbol, _symbols.Spanner):
            symbol.setAnchor(self)
        elif isinstance(symbol, _symbols.EventSymbol):
            if errormsg := symbol.checkAnchor(self):
                raise ValueError(f"Cannot add this symbol to {self}: {errormsg}")
        return self

    def _canBeLinkedTo(self, other: MEvent) -> bool:
        """
        Can self be linked to *other* within a playback line, assuming other follows self?

        A line is a sequence of events (notes, chords) where
        one is linked to the next by either being tied, a gliss
        leading to the next pitch, or a portamento (an implicit glissano)

        This method should not take offset time into account: it should
        simply return if self can be linked to other assuming that
        other follows self
        """
        raise NotImplementedError

    def mergeWith(self, other: MEvent) -> Self | None:
        """
        Merge this with other, return None if not possible

        Args:
            other: the event to concatenato to this. Only events of the same type
                can be merged (a Note with a Note, a Chord with a Chord)

        Returns:
            the merged event, or None

        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """A string representing this event"""
        raise NotImplementedError('Subclass should implement this')

    def cropped(self, start: beat_t, end: beat_t
                ) -> Self:
        """
        A copy of Self, cropped to the given time range

        The returned event will have an explicit offset set to its
        absolute offset. It is parentless

        Args:
            start: the start location to crop this event
            end: the end location to crop this event

        Returns:
            the cropped event

        """
        scorestruct = self.activeScorestruct()
        startbeat = start if isinstance(start, F) else scorestruct.asBeat(start)
        endbeat = end if isinstance(end, F) else scorestruct.asBeat(end)
        absoffset = self.absOffset()
        if intersect := mathutils.intersectF(startbeat, endbeat, absoffset, absoffset + self.dur):
            intersect0, intersect1 = intersect
            return self.clone(offset=intersect0, dur=intersect1 - intersect0)
        else:
            raise ValueError(f"No intersection between {self} and the given time range "
                             f"({start=}, {end=}")

    def splitAt(self, offset: beat_t, tie=True, nomerge=False
                ) -> tuple[Self, ...]:
        """
        Split this event at the given absolute offset

        Args:
            offset: the absolute offset (in beats) at which to split this event.
                Can be a beat or a location as a tuple ``(measureindex, beatoffset)``.
            tie: tie the parts. The returned events are tied.
            nomerge: if True, adds a break symbol to the events resulted in the split
                operation to prevent them from being merged when converted to notation

        Returns:
            a tuple with the parts. If the offset lies perfectly at the start or
            end of this event, only one part will be returned. If the offset does
            not intersect the event, ValueError is raised. The returned events are
            parentless.

        Example
        -------

            >>> n = Note(60, 4)
            >>> n.splitAt(2)
            (4C~:2♩, 4C:2♩)

            >>> n = Note(60, 3.5)
            >>> notes = n.splitAt(1.8)
            >>> notes[1].addSymbol('>')
            >>> Chain(notes)

        .. image:: ../assets/note-splitat.png
        """
        parts = self.splitAtOffsets([offset], tie=tie, nomerge=nomerge)
        if not parts:
            raise ValueError(f"Offset {offset} does not intersect {self}")
        assert len(parts) <= 2
        return tuple(parts)

    def _splitAtOffsets(self, offsets: list[F], tie=True, nomerge=False
                       ) -> list[Self]:
        """
        Internal and efficient version of splitAtOffsets which only accepts absolute offsets as F

        Args:
            offsets: absolute offsets to split this event at
            tie: if True, tie the parts
            nomerge: if True, adds a break symbol to the events resulted in the split
                operation to prevent them from being merged when converted to notation

        Returns:
            the parts. The total duration of the parts should sum up to the
            duration of self
        """
        if not offsets:
            return []
        offset = self.absOffset()
        dur = self.dur
        if offset >= offsets[-1] or offset + dur <= offsets[0]:
            return [self]

        intervals = mathutils.splitInterval(offset, offset + dur, offsets)
        events = [self.clone(dur=end-start, offset=None)
                  for start, end in intervals]
        events[0].offset = self.offset
        if tie and len(events) > 1:
            for event in events[:-1]:
                event.tied = True
            if nomerge:
                for event in events[1:]:
                    event.addSymbol(_symbols.NoMerge())
        return events

    def splitAtOffsets(self, offsets: list[beat_t], tie=True, nomerge=False
                       ) -> list[Self]:
        """
        Split this event at the given offsets

        Args:
            offsets: absolute offsets. To use score locations, convert those to absolute
                offsets via :meth:`scorestruct.locationToBeat <maelzel.scorestruct.ScoreStruct.locationToBeat>`
            tie: if True, tie the parts
            nomerge: if True, adds a break symbol to the events resulted in the split
                operation to prevent them from being merged when converted to notation

        Returns:
            the parts. The total duration of the parts should sum up to the
            duration of self
        """
        sco = self.activeScorestruct()
        beats = [sco.asBeat(offset) for offset in offsets]
        return self._splitAtOffsets(beats, tie=tie, nomerge=nomerge)

    def addSpanner(self,
                   spanner: str | _symbols.Spanner,
                   endobj: MEvent = None
                   ) -> Self:
        """
        Adds a spanner symbol to this object

        A spanner is a slur, line or any other symbol attached to two or more
        objects. A spanner always has a start and an end.

        Args:
            spanner: a Spanner object or a spanner description (one of 'slur', '<', '>',
                'trill', 'bracket', etc. - see :func:`maelzel.core.symbols.makeSpanner`
                When passing a string description, prepend it with '~' to create an end spanner
            endobj: the object where this spanner ends, if known

        Returns:
            self (allows to chain calls)

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> a = Note("4C")
            >>> b = Note("4E")
            >>> c = Note("4G")
            >>> a.addSpanner('slur', c)
            >>> chain = Chain([a, b, c])

        .. seealso:: :meth:`Spanner.bind() <maelzel.core.symbols.Spanner.bind>`

        In some cases the end target can be inferred:

            >>> chain = Chain([
            ... Note("4C", 1, dynamic='p').addSpanner("<"),
            ... Note("4D", 0.5),
            ... Note("4E", dynamic='f')   # This ends the hairpin spanner
            ... ])

        Or it can be set later

            >>> chain = Chain([
            ... Note("4C", 1).addSpanner("slur"),
            ... Note("4D", 0.5),
            ... Note("4E").addSpanner("~slur")   # This ends the last slur spanner
            ... ])

        """
        if isinstance(spanner, str):
            if spanner.startswith('~'):
                spanner = spanner[1:].lower()
                kind = 'end'
            else:
                kind = 'start'
            spanner = _symbols.makeSpanner(spanner.lower(), kind=kind)
        assert isinstance(spanner, _symbols.Spanner)

        if endobj is not None:
            assert spanner.kind == 'start'
            spanner.bind(self, endobj)
        else:
            self.addSymbol(spanner)
        return self

    def timeTransform(self, timemap: Callable[[F], F], inplace=False
                      ) -> Self:
        """
        Apply a transformation to the time axes of this event

        Args:
            timemap: a callable mapping time (in quarterbeats) to time (in quarterbeats)
            inplace: transform the object in place

        Returns:
            the transformed object, or self if inplace is True

        """
        offset = self.relOffset()
        dur = self.dur
        offset2 = timemap(offset)
        dur2 = timemap(offset + dur) - offset2
        if inplace:
            self.offset = offset2
            self.dur = dur2
            self._changed()
            return self
        else:
            return self.clone(offset=offset2, dur=dur2)

    def automate(self,
                 param: str,
                 breakpoints: list[tuple[time_t | location_t, float]] | list[tuple[time_t|location_t, float, str]] | list[num_t],
                 interpolation='linear',
                 relative=True,
                 ) -> None:
        """
        Add an automation action to this event

        Args:
            param: the playback parameter to modify, either a builtin parameter like
                position, or an instrument defined parameter (for example, an instrument
                based on substractive synthesis could define a 'filterattack' parameter,
                or a vocal synthesis instrument could define a 'vibratoamount' parameter)
            breakpoints: the data, a list of pairs in the form (time, value),
                or (time, value, interpolation). time is given in quarternotes or as a
                location (measure, beatoffset); value is any valid valud for the given
                parameter; interpolation is one of 'linear', 'cos'. As a shortcut
                it is possible to also pass a flat list of the form
                [time0, value0, time1, value1, ...]. A single point is also possible. This
                sets the value for the given param at the specified time
            interpolation: default interpolation used for breakpoints without interpolation
            relative: if True, the time positions are relative to the absolute offset
                of this event. If False, these times are absolute times

        Example
        ~~~~~~~

        .. code::

            note = Note("4c", 10)

            # Automate position starting at beat 5 after this event has started
            note.automate('position', [(5, 0.), (6, 1.)], relative=True)

            # The same data can be given as a flat list
            note.automate('position', [5, 0., 6, 1.])

            # Time position can be also given as a tuple (measure num, beat offset),
            # and the time mode can be set to absolute
            # In this case, this automation indicates a modification of the
            # pan position, from 0 to 1 starting at the 4th measure (index 3) and
            # ending at the 5th measure (index 4)
            note.automate('position', [(3, 0), 0., (4, 0), 1.], relative=False)

        Any dynamic parameter can be automated:

        .. code::

            # Define a preset with some dynamic parameter

            defPreset('detuned', '''
            |kdetune=2|
                aout1 = oscili:a(kamp, kfreq) + oscili:a(kamp, kfreq+kdetune)
            ''')

            # Automate the kdetune param. When automating a Note/Chord, the times
            # are given in quarterbeats, which means that the real time in seconds
            # depend on the tempo structure. In this case the kdetune param will
            # be shifted from 0 to 20 starting at the moment the note is played
            # and ending 4 **quarterbeats** after.
            note.automate('kdetune', (0, 0, 4, 20))

            # When the note is actually played the automation takes effect
            synth = note.play(instr='detuned')

            # The synth itself can be automated. In this case, we are already
            # in the real-time realm and any times are given in seconds. If
            # needed the active scorestruct can be used to convert between
            # quarterbeats and seconds
            synth.automate('position', (2, 0.5, 4, 1))

        """
        if self.playargs is None:
            self.playargs = synthevent.PlayArgs()
        self.playargs.addAutomation(param=param, breakpoints=breakpoints,
                                    interpolation=interpolation, relative=relative)
