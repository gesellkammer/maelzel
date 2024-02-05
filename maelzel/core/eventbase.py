from __future__ import annotations

from emlib import mathlib
from maelzel.common import F
from maelzel.core.mobj import MObj, MContainer
import maelzel.core.symbols as _symbols
from maelzel.core.synthevent import PlayArgs

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import TypeVar, Any, Callable
    from ._typedefs import time_t, location_t, num_t
    MEventT = TypeVar("MEventT", bound="MEvent")


class MEvent(MObj):
    """
    A discrete event in time (a Note, Chord, etc)
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
        super().__init__(dur=dur, offset=offset, label=label, parent=parent,
                         properties=properties, symbols=symbols)
        self.tied: bool = tied
        """Is this event tied?"""

        self.amp: float | None = amp
        "The playback amplitude 0-1 of this note"

        self.dynamic: str = dynamic

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
        return prev.linkedNext() if prev else False

    @property
    def gliss(self):
        """The end target of this event, if any"""
        return False

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

    def addSymbol(self: MEventT, *args, **kws) -> MEventT:
        """
        Add a notation symbol to this object

        Notation symbols are any attributes which are attached to **one event**
        and are intended for **notation only**. Such attributes include articulations,
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
        elif isinstance(symbol, _symbols.NoteSymbol):
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

    def mergeWith(self: MEventT, other: MEventT) -> MEventT | None:
        """
        Merge this with other, return None if not possible

        Args:
            other: the event to concatenato to this. Only events of the same type
                can be merged (a Note with a Note, a Chord with a Chord)

        Returns:
            the merged event, or None

        """
        raise NotImplemented

    @property
    def name(self) -> str:
        """A string representing this event"""
        raise NotImplementedError('Subclass should implement this')

    def splitAtOffsets(self: MEventT, offsets: list[time_t], tie=True, absolute=True
                       ) -> list[MEventT]:
        """
        Split this event at the given offsets

        Args:
            offsets: absolute offsets. To use score locations, convert those to absolute
                offsets via :meth:`scorestruct.locationToBeat <maelzel.scorestruct.ScoreStruct.locationToBeat>`
            tie: if True, tie the parts
            absolute: if True, the offsets are interpreted as absolute offsets

        Returns:
            the parts. The total duration of the parts should sum up to the
            duration of self
        """
        if not offsets:
            raise ValueError("No offsets given")

        offset = self.absOffset() if absolute else self.relOffset()
        dur = self.dur
        intervals = mathlib.split_interval_at_values(offset, offset + dur, offsets)
        events = [self.clone(offset=intervalstart, dur=intervalend-intervalstart)
                  for intervalstart, intervalend in intervals]
        if tie:
            for event in events[:-1]:
                event.tied = True
        return events

    def addSpanner(self: MEventT,
                   spanner: str | _symbols.Spanner,
                   endobj: MEvent = None
                   ) -> MEventT:
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
            spanner.setAnchor(self)
        return self

    def timeTransform(self: MEventT, timemap: Callable[[F], F], inplace=False
                      ) -> MEventT:
        """
        Apply a transformation to the time axes of this

        Args:
            timemap: a callable mapping time (in quarterbeats) to time (in quarterbeats)
            inplace:

        Returns:

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
            self.playargs = PlayArgs()
        self.playargs.addAutomation(param=param, breakpoints=breakpoints,
                                    interpolation=interpolation, relative=relative)

