from __future__ import annotations
from ._common import UNSET
from maelzel.common import F, asF
from .event import MObj, MEvent, asEvent, Note, Chord
from . import _mobjtools
from . import symbols
from . import _util
from . import environment
from .workspace import getConfig, Workspace
from .synthevent import PlayArgs, SynthEvent

from maelzel import scoring
from maelzel.colortheory import safeColors

from emlib import iterlib
from emlib import misc

from typing import TYPE_CHECKING, overload
if TYPE_CHECKING:
    from typing import Any, Iterator, overload, TypeVar, Callable
    from numbers import Rational
    from ._typedefs import time_t
    from .config import CoreConfig
    from maelzel.scorestruct import ScoreStruct
    ChainT = TypeVar("ChainT", bound="Chain")


__all__ = (
    'Chain',
    'Voice',
    'stackEvents',
    '_resolvedTimes',
    '_flattenObjs'
)


def _itemsAreStacked(items: list[MObj]) -> bool:
    for item in items:
        if isinstance(item, MEvent):
            if item.offset is None or item.dur is None:
                return False
        elif isinstance(item, Chain):
            if item.offset is None or not _itemsAreStacked(item.items):
                return False
        else:
            raise TypeError(f"{item} ({type(item).__name__}) cannot be stacked")
    return True


def stackEvents(events: list[MEvent | Chain],
                defaultDur: time_t = F(1),
                offset: time_t = F(0),
                inplace=False,
                check=False
                ) -> list[MEvent | Chain]:
    """
    Stack events to the left, making any unset offset and duration explicit

    After fixing all offset times and durations an extra offset can be added,
    if given

    Args:
        events: the events to modify, either in place or as a copy
        defaultDur: the default duration used when an event has no duration and
            the next event does not have an explicit offset
        inplace: if True, events are modified in place
        offset: an extra offset to add to all offset times after left-stacking them
        recurse: if True, stack also events inside subchains

    Returns:
        the modified events. If inplace is True, the returned events are the
        same as the events passed as input

    """

    if not events:
        raise ValueError("no events given")

    if check and _itemsAreStacked(events):
        return events

    if not inplace:
        events = [ev.copy() for ev in events]
        stackEvents(events=events, defaultDur=defaultDur, offset=offset,
                    inplace=True)
        return events

    # All offset times given in the events are relative to the start of the chain
    now = F(0)
    lasti = len(events) - 1
    for i, ev in enumerate(events):
        if ev.offset is None:
            ev.offset = now
        if isinstance(ev, MEvent):
            if ev.dur is None:
                if i == lasti:
                    ev.dur = defaultDur
                else:
                    nextev = events[i+1]
                    if nextev.offset is None:
                        ev.dur = defaultDur
                    else:
                        ev.dur = nextev.offset - ev.offset
            now = ev.end
        elif isinstance(ev, Chain):
            ev.stack()
            dur = ev.resolvedDur()
            now = ev.offset + dur

    if offset:
        for ev in events:
            ev.offset += offset
    assert all(ev.offset is not None for ev in events)
    assert all(ev.dur is not None for ev in events
               if isinstance(ev, MEvent))
    return events


class Chain(MObj):
    """
    A Chain is a sequence of Notes, Chords or other Chains

    Args:
        items: the items of this Chain. The start time of any object, if given, is
            interpreted as relative to the start of the chain.
        offset: offset of the chain itself relative to its parent
        label: a label for this chain
        properties: any properties for this chain. Properties can be anything,
            they are a way for the user to attach data to an object
    """
    _acceptsNoteAttachedSymbols = False

    __slots__ = ('items',)

    def __init__(self,
                 items: list[MEvent | Chain | str] = None,
                 offset: time_t = None,
                 label: str = '',
                 properties: dict[str, Any] = None):
        if offset is not None:
            offset = asF(offset)
        if items is not None:
            items = [item if isinstance(item, (MEvent, Chain)) else asEvent(item)
                     for item in items]
            for item in items:
                item.parent = self

            for i0, i1 in iterlib.pairwise(items):
                assert i0.offset is None or i1.offset is None or i0.offset <= i1.offset, f'{i0 = }, {i1 = }'
        else:
            items = []

        super().__init__(offset=offset, dur=None, label=label, properties=properties)
        self.items: list[MEvent | 'Chain'] = items
        self._changed()

    def __hash__(self):
        items = [type(self).__name__, self.label, self.offset, len(self.items)]
        if self.symbols:
            items.extend(self.symbols)
        items.extend(self.items)
        out = hash(tuple(items))
        return out

    def clone(self, items=UNSET, offset=UNSET, label='', properties=UNSET) -> Chain:
        return Chain(items=self.items if items is UNSET else items,
                     offset=self.offset if offset is UNSET else offset,
                     label=self.label if label is UNSET else label,
                     properties=self.properties if properties is UNSET else properties)

    def copy(self) -> Chain:
        items = [item.copy() for item in self.items]
        return Chain(items=items, offset=self.offset, label=self.label, properties=self._properties)

    def isStacked(self) -> bool:
        """
        True if items in this chain have a defined offset and duration
        """
        return self.offset is not None and _itemsAreStacked(self.items)

    def stack(self, offset: time_t = F(0)) -> None:
        """
        Stack events to the left (in place), making any unset offset and duration explicit

        After setting all start times and durations an offset is added, if given

        Args:
            offset: an offset to add to all offset times after stacking them

        """
        stackEvents(self.items, offset=offset, inplace=True)

    def fillGapsWithRests(self, recurse=True) -> None:
        """
        Fill any gaps with rests, in place

        A gap is produced when an event within a chain has an explicit offset
        later than the offset calculated by stacking the previous objects in terms
        of their duration

        Args:
            recurse: if True, fill gaps within subchains
        """
        now = F(0)
        items = []
        for item in self.items:
            if item.offset is None or item.dur is None:
                raise ValueError(f"This operation can only be performed if all items have "
                                 f"an explicit offset time ({item=}")
            if item.offset > now:
                r = Note.makeRest(item.offset - now, offset=now)
                items.append(r)
            items.append(item)
            if isinstance(item, Chain) and recurse:
                item.fillGapsWithRests(recurse=True)
            now = item.end
        self.items = items
        self._changed()

    def itemAfter(self, item: MEvent) -> MEvent | Chain | None:
        """
        Returns the next item after *item*

        Args:
            item: the item to find its next item

        Returns:
            the item following *item* or None if the given item is not
            in this container or it has no item after it

        """
        for ev0, ev1 in iterlib.pairwise(self.items):
            if ev0 == item:
                return ev1
        return None

    def flat(self, removeRedundantOffsets=False, offset: time_t = None) -> Chain:
        """
        A flat version of this Chain

        A Chain can contain other Chains. This method serializes all objects inside
        this Chain and any sub-chains to a flat chain of notes/chords.

        If this Chain is already flat, meaning that it does not contain any
        Chains, self is returned unmodified.

        As a side-effect all offsets (start times) are made explicit

        Args:
            removeRedundantOffsets: remove any redundant start times. A start time is
                redundant if it merely confirms the time offset of an object as
                determined by the durations of the previous objects.
            offset: a start time to fill or override self.start.

        Returns:
            a chain with exclusively Notes and/or Chords
        """
        if all(isinstance(item, MEvent) for item in self.items) and offset == self.offset:
            return self
        chain = self.resolved(offset=offset)
        if offset is not None:
            assert chain.offset == offset
        offset = chain.offset if chain.offset is not None else F(0)
        items = _flattenObjs(chain.items, offset)
        if chain.offset is not None:
            for item in items:
                item.offset -= chain.offset
        out = chain.clone(items=items)
        assert out.offset == chain.offset
        if offset is not None:
            assert out.offset == offset

        if removeRedundantOffsets:
            out.removeRedundantOffsets()
        return out

    def pitchRange(self) -> tuple[float, float] | None:
        pitchRanges = [item.pitchRange() for item in self.items]
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

    def resolved(self, offset: time_t = None) -> Chain:
        """
        Copy of self with explicit times

        The items in the returned object have an explicit start and
        duration.

        .. note:: use a start time of 0 to have an absolute start
            time set for each item.

        Args:
            offset: a start time to fill or override self.start.

        Returns:
            a clone of self with dur and start set to explicit
            values

        """
        if offset is not None:
            offset = self.resolvedOffset() - offset
            if offset < 0:
                raise ValueError(f"This would result in a negative offset: {offset}")
            clonedOffset = offset
        else:
            offset = 0
            clonedOffset = self.offset
        if self.isStacked():
            return self
        items = stackEvents(self.items, offset=offset, inplace=False)
        return self.clone(items=items, offset=clonedOffset)

    def resolvedOffset(self) -> F:
        ownstart = self.offset or F(0)
        if not self.items:
            return ownstart
        item = self.items[0]
        return ownstart if item.offset is None else ownstart + item.offset

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        chain = self.flat(removeRedundantOffsets=False)
        conf = workspace.config
        if self.playargs:
            playargs.overwriteWith(self.playargs)
        items = stackEvents(chain.items, inplace=True, offset=self.offset)
        if any(n.isGracenote() for n in self.items
               if isinstance(n, (Note, Chord))):
            _mobjtools.addDurationToGracenotes(items, F(1, 14))
        if conf['play.useDynamics']:
            _mobjtools.fillTempDynamics(items, initialDynamic=conf['play.defaultDynamic'])
        return _mobjtools.chainSynthEvents(items, playargs=playargs, workspace=workspace)

    def mergeTiedEvents(self) -> None:
        """
        Merge tied events in place

        Two events can be merged if they are tied and the second second
        event does not provide any extra information (does not have
        an individual amplitude, dynamic, does not start a gliss, etc)
        """
        out = []
        last = None
        lastidx = len(self.items) - 1
        for i, item in enumerate(self.items):
            if isinstance(item, Chain):
                item.mergeTiedEvents()
                out.append(item)
                last = None
            elif type(last) == type(item):
                merged = last.mergeWith(item)
                if merged is None:
                    if last is not None:
                        out.append(last)
                    last = item
                else:
                    if i < lastidx:
                        last = merged
                    else:
                        out.append(merged)

            else:
                if last is not None:
                    out.append(last)
                last = item
                if i == lastidx:
                    out.append(item)
        self.items = out
        self._changed()

    def timeShiftInPlace(self, timeoffset):
        if any(item.offset is None for item in self.items):
            stackEvents(self.items, inplace=True)
        for item in self.items:
            item.offset += timeoffset
        self._changed()

    def movedTo(self, start: time_t):
        offset = start - self.items[0].offset
        return self.timeShift(offset)

    def moveTo(self, start: time_t):
        offset = start - self.items[0].offset
        self.timeShiftInPlace(offset)

    def resolvedDur(self, offset: time_t = None) -> F:
        if not self.items:
            return F(0)

        defaultDur = F(1)
        accum = F(0)
        items = self.items
        lasti = len(items) - 1

        for i, ev in enumerate(items):
            if ev.offset is not None:
                accum = ev.offset
            if isinstance(ev, MEvent):
                if ev.dur:
                    accum += ev.dur
                elif i == lasti:
                    accum += defaultDur
                else:
                    nextev = items[i + 1]
                    accum += defaultDur if nextev.offset is None else nextev.offset - accum
            else:
                # a Chain
                accum += ev.resolvedDur()

        return accum

    def append(self, item: MEvent) -> None:
        """
        Append an item to this chain

        Args:
            item: the item to add
        """
        item.parent = self
        self.items.append(item)
        if len(self.items) > 1:
            butlast = self.items[-2]
            last = self.items[-1]
            if isinstance(butlast, Note) and butlast.gliss is True and isinstance(last, Note):
                butlast.gliss = last.pitch
        self._changed()

    def extend(self, items: list[MEvent]) -> None:
        """
        Extend this chain with items

        Args:
            items: a list of items to append to this chain
        """
        self.items.extend(items)
        self._changed()

    def _changed(self):
        if self.items:
            self.dur = self.resolvedDur()
        else:
            self.offset = None
            self.dur = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[MEvent]:
        return iter(self.items)

    @overload
    def __getitem__(self, idx: int) -> MEvent: ...

    @overload
    def __getitem__(self, slice_: slice) -> Chain: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.items[idx]
        else:
            return self.__class__(self.items.__getitem__(idx))

    def _dumpRows(self, indents=0) -> list[str]:
        fontsize = '80%'
        durwidth = 7
        selfstart = round(float(self.offset.limit_denominator(1000)), 3) if self.offset is not None else None
        if environment.insideJupyter:
            namew = max((sum(len(n.name) for n in event.notes)+len(event.notes)
                         for event in self.recurse()
                         if isinstance(event, Chord)),
                        default=10)
            header = f"<code>{'  '*indents}</code><strong>{type(self).__name__}</strong> &nbsp;" \
                     f'start: <code>{_util.htmlSpan(selfstart, color=":blue1")}</code>'
            if self.label:
                header += f', label: <code>{_util.htmlSpan(self.label, color=":blue1")}</code>'
            rows = [header]
            columnnames = f"{'  ' * indents}{'start'.ljust(6)}{'dur'.ljust(durwidth)}{'name'.ljust(namew)}{'gliss'.ljust(6)}{'dyn'.ljust(5)}playargs"
            row = f"<code>  {_util.htmlSpan(columnnames, ':grey1', fontsize=fontsize)}</code>"
            rows.append(row)
            for item in self.items:
                if isinstance(item, MEvent):
                    if item.isRest():
                        name = "Rest"
                    elif isinstance(item, Note):
                        name = item.name
                    elif isinstance(item, Chord):
                        name = ",".join(item._bestSpelling())
                    else:
                        raise TypeError(f"Expected Note or Chord, got {item}")

                    if item.tied:
                        name += "~"
                    start = f"{float(item.offset):.3g}" if item.offset is not None else "None"
                    dur = f"{float(item.dur):.3g}" if item.dur is not None else "None"
                    rowtxt = f"{'  '*indents}{start.ljust(6)}{dur.ljust(durwidth)}{name.ljust(namew)}{str(item.gliss).ljust(6)}{str(item.dynamic).ljust(5)}{self.playargs}</code>"
                    row = f"<code>  {_util.htmlSpan(rowtxt, ':blue1', fontsize=fontsize)}</code>"
                    rows.append(row)
                    if item.symbols:
                        symbolstr = str(item.symbols)
                        row = f"<code>      {_util.htmlSpan(symbolstr, ':green2', fontsize=fontsize)}</code>"
                        rows.append(row)
                elif isinstance(item, Chain):
                    rows.extend(item._dumpRows(indents=indents+1))
            return rows
        else:
            rows = [f"{' ' * indents}Chain"]
            for item in self.items:
                if isinstance(item, MEvent):
                    sublines = repr(item).splitlines()
                    for subline in sublines:
                        rows.append(f"{'  ' * (indents + 1)}{subline}")
                else:
                    rows.extend(item._dumpRows(indents=indents+1))
            return rows

    def dump(self, indents=0):
        rows = self._dumpRows(indents=indents)
        if environment.insideJupyter:
            html = '<br>'.join(rows)
            from IPython.display import HTML, display
            display(HTML(html))
        else:
            for row in rows:
                print(row)

    def __repr__(self):
        if len(self.items) < 10:
            itemstr = ", ".join(repr(_) for _ in self.items)
        else:
            itemstr = ", ".join(repr(_) for _ in self.items[:10]) + ", …"
        cls = self.__class__.__name__
        namedargs = []
        if self.offset is not None:
            namedargs.append(f'offset={self.offset}')
        if namedargs:
            info = ', ' + ', '.join(namedargs)
        else:
            info = ''
        return f'{cls}([{itemstr}]{info})'

    def _repr_html_header(self):
        itemcolor = safeColors['blue2']
        items = self.items if len(self.items) < 10 else self.items[:10]
        itemstr = ", ".join(f'<span style="color:{itemcolor}">{repr(_)}</span>'
                            for _ in items)
        if len(self.items) >= 10:
            itemstr += ", …"
        cls = self.__class__.__name__
        namedargs = []
        if self.offset is not None:
            namedargs.append(f'start={self.offset}')
        if namedargs:
            info = ', ' + ', '.join(namedargs)
        else:
            info = ''
        return f'{cls}([{itemstr}]{info})'

    def cycle(self, dur: time_t, crop=True):
        """
        Cycle the items in this chain until the given duration is reached

        Args:
            dur: the total duration
            crop: if True, the last event will be cropped to fit
                the given total duration. Otherwise, it will last
                its given duration, even if that would result in
                a total duration longer than the given one

        Returns:
            the resulting Chain
        """
        defaultDur = F(1)
        accumDur = F(0)
        maxDur = asF(dur)
        items: list[MEvent] = []
        ownitems = stackEvents(self.items)
        for item in iterlib.cycle(ownitems):
            dur = item.dur if item.dur else defaultDur
            if dur > maxDur - accumDur:
                if crop:
                    dur = maxDur - accumDur
                else:
                    break
            if item.dur is None or item.offset is not None:
                item = item.clone(dur=dur, start=None)
            assert isinstance(item, MEvent)
            items.append(item)
            accumDur += item.dur
            if accumDur == maxDur:
                break
        return self.__class__(items, offset=self.offset)

    def removeRedundantOffsets(self):
        """
        Remove over-secified start times in this Chain **inplace**
        """
        # This is the relative position (independent of the chain's start)
        now = F(0)
        for item in self.items:
            if isinstance(item, MEvent):
                if item.dur is None:
                    raise ValueError(f"This Chain contains events with unspecified duration: {item}")
                if item.offset is None:
                    now += item.dur
                else:
                    if item.offset < now:
                        raise ValueError(f"Items overlap: {item}, {now=}")
                    elif item.offset > now:
                        now = item.end
                    else:
                        # item.start == now
                        item.offset = None
                        now += item.dur
            elif isinstance(item, Chain):
                item.removeRedundantOffsets()
        if self.offset == 0:
            self.offset = None

    def asVoice(self) -> Voice:
        """Convert this Chain to a Voice"""
        resolved = self.resolved(offset=0)
        resolved.removeRedundantOffsets()
        return Voice(resolved.items, label=self.label)

    def makeVoices(self) -> list[Voice]:
        return [self.asVoice()]

    def scoringEvents(self,
                      groupid: str = None,
                      config: CoreConfig = None,
                      ) -> list[scoring.Notation]:
        """
        Returns the scoring events corresponding to this object

        Args:
            groupid: if given, all events are given this groupid
            config: the configuration used (None to use the active config)

        Returns:
            the scoring notations representing this object
        """
        if config is None:
            config = getConfig()
        items = self.flat(removeRedundantOffsets=False)
        notations: list[scoring.Notation] = []
        for item in items:
            notations.extend(item.scoringEvents(groupid=groupid, config=config))
        scoring.stackNotationsInPlace(notations)
        if self.offset is not None and self.offset > 0:
            for notation in notations:
                notation.offset += self.offset

        for n0, n1 in iterlib.pairwise(notations):
            if n0.tiedNext and not n1.isRest:
                n1.tiedPrev = True

        if self.symbols:
            for s in self.symbols:
                for n in notations:
                    s.applyTo(n)

        openSpanners: list[scoring.spanner.Spanner] = []
        lastHairpin: scoring.spanner.Hairpin | None = None
        for n in notations:
            if n.spanners:
                for spanner in n.spanners:
                    if spanner.kind == 'start':
                        if isinstance(spanner, symbols.Hairpin):
                            lastHairpin = spanner
                        else:
                            openSpanners.append(spanner)
                    else:
                        if isinstance(spanner, symbols.Hairpin):
                            lastHairpin = None
                        else:
                            misc.remove_last_matching(openSpanners, lambda s: type(s) == type(spanner))
            if n.dynamic and lastHairpin is not None:
                n.addSpanner(lastHairpin.endSpanner())

        if openSpanners:
            for spanner in openSpanners:
                notations[-1].addSpanner(spanner.endSpanner())

        return notations

    def scoringParts(self, config: CoreConfig = None
                     ) -> list[scoring.Part]:
        notations = self.scoringEvents(config=config or getConfig())
        if not notations:
            return []
        scoring.stackNotationsInPlace(notations)
        part = scoring.Part(notations, name=self.label)
        return [part]

    def quantizePitch(self, step=0.):
        if step == 0:
            step = 1 / getConfig()['semitoneDivisions']
        items = [i.quantizePitch(step) for i in self.items]
        return self.clone(items=items)

    def timeShift(self, timeoffset: time_t) -> Chain:
        if self.offset is not None:
            return self.clone(offset=self.offset+timeoffset)
        items = stackEvents(self.items, offset=timeoffset)
        return self.clone(items=items)

    def timeTransform(self, timemap: Callable[[F], F], inplace=False) -> Chain:
        offset = self.resolvedOffset()
        offset2 = timemap(offset)
        if inplace:
            stackEvents(self.items, inplace=True)
            for item in self.items:
                item.offset = timemap(item.offset + offset) - offset2
                item.dur = timemap(item.end + offset) - offset2 - item.offset
            self.offset = offset2
            return self
        else:
            items = stackEvents(self.items, inplace=False)
            for item in items:
                item.offset = timemap(item.offset + offset) - offset2
                item.dur = timemap(item.end + offset) - offset2 - item.offset
            return self.clone(items=items, offset=offset2)

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Chain:
        newitems = [item.pitchTransform(pitchmap) for item in self.items]
        return self.clone(items=newitems)

    def recurse(self, reverse=False) -> Iterator[MEvent]:
        """
        Yields all Notes/Chords in this chain, recursing through sub-chains if needed

        This method guarantees that the yielded events are the actual objects included
        in this chain or its sub-chains. This is usefull when used in combination with
        methods like addSpanner, which modify the objects themselves.

        Args:
            reverse: if True, recurse the chain in reverse

        Returns:
            an iterator over all notes/chords within this chain and its sub-chains

        """
        if not reverse:
            for item in self.items:
                if isinstance(item, MEvent):
                    yield item
                elif isinstance(item, Chain):
                    yield from item.recurse(reverse=False)
        else:
            for item in reversed(self.items):
                if isinstance(item, MEvent):
                    yield item
                else:
                    yield from item.recurse(reverse=True)

    def addSpanner(self, spanner: str | symbols.Spanner, endobj: MObj = None) -> None:
        first = next(self.recurse())
        last = next(self.recurse(reverse=True))
        if isinstance(spanner, str):
            spanner = symbols.makeSpanner(spanner)
        assert isinstance(first, (Note, Chord)) and isinstance(last, (Note, Chord))
        spanner.bind(first, last)

    def resolvedTimes(self, defaultDur=F(1)) -> list[tuple[MEvent, F, F] | list]:
        """
        Resolves the times of the events without modifying them

        For each event it returns a tuple [event, absolute start, duration]. Nested
        subchains result in nested lists, but the times are still absolute. All
        times are in beats

        Args:
            defaultDur: the default duration used when an event has no duration and
                the next event does not have an explicit offset

        Returns:
            a list of tuples, one tuple for each event.
        """
        return _resolvedTimes(self.items, offset=self.offset, defaultDur=defaultDur)




class Voice(Chain):
    """
    A Voice is a sequence of non-overlapping objects

    It is **very** similar to a Chain, the only difference being that its offset
    is always 0.


    Voice vs Chain
    ~~~~~~~~~~~~~~

    * A Voice can contain a Chain, but not vice versa.
    * A Voice does not have a time offset, its offset is always 0.
    """

    _acceptsNoteAttachedSymbols = False

    def __init__(self,
                 items: list[MEvent | str] = None,
                 label='',
                 shortname=''):
        super().__init__(items=items, label=label, offset=F(0))
        self.shortname = shortname

    def scoringParts(self, config: CoreConfig = None
                     ) -> list[scoring.Part]:
        parts = super().scoringParts(config=config)
        for part in parts:
            part.shortname = self.shortname
        return parts

    def setScoreStruct(self, scorestruct: ScoreStruct | None) -> None:
        self._scorestruct = scorestruct


def _resolvedTimes(events: list[MEvent | Chain],
                   offset: F = F(0),
                   defaultDur = F(1),
                   ) -> list[tuple[MEvent, F, F] | list]:
    if not events:
        raise ValueError("no events given")

    now = asF(offset)
    lasti = len(events) - 1
    out = []
    for i, ev in enumerate(events):
        if ev.offset is not None:
            now = offset + ev.offset
        dur = defaultDur
        if isinstance(ev, MEvent):
            if ev.dur is not None:
                dur = ev.dur
            elif i < lasti:
                nextev = events[i+1]
                if nextev.offset is not None:
                    dur = offset + nextev.offset - now
            out.append((ev, now, dur))
            now += dur
        elif isinstance(ev, Chain):
            subitems = _resolvedTimes(ev.items, offset=now)
            out.append(subitems)
            dur = _resolvedTimesDur(ev.items)
            now += dur
        else:
            raise TypeError(f"Expected a MEvent of a Chain, got {ev}")
    return out


def _resolvedTimesDur(items: list[MEvent | Chain]) -> Rational:
    return sum(_resolvedTimesDur(x.items) if isinstance(x, Chain) else x.dur
               for x in items)


def _flattenObjs(objs: list[MEvent | Chain], offset=F(0)) -> list[MEvent]:
    collected = []
    for obj in objs:
        assert obj.offset is not None, \
            f"This function should be called with objects with resolved start, got {obj}"
        if isinstance(obj, MEvent):
            assert obj.dur is not None
            collected.append(obj.clone(offset=obj.offset + offset))
        elif isinstance(obj, Chain) and obj.items:
            collected.extend(_flattenObjs(obj.items, offset=offset + obj.offset))
        else:
            raise TypeError(f"Expected a Note/Chord or a Chain, got {obj} ({type(obj)})")
    return collected

