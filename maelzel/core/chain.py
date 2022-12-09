from __future__ import annotations
from ._common import UNSET
from maelzel.common import F, asF
from .mobj import MObj, MContainer
from .event import MEvent, asEvent, Note, Chord
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
    from ._typedefs import time_t
    from .config import CoreConfig
    from maelzel.scorestruct import ScoreStruct
    ChainT = TypeVar("ChainT", bound="Chain")


__all__ = (
    'Chain',
    'Voice',
    'stackEvents',
)


def _stacked(items: list[MEvent | Chain], now=F(0), ensureOffsets=False) -> bool:
    frame = now
    for item in items:
        if item.offset is not None:
            if item.offset + frame < now:
                raise ValueError(f"items overlap... now={_util.showT(now)}, "
                                 f"{item=} (offset={_util.showT(item.offset)})")
            now = item.offset + frame
        elif ensureOffsets:
            return False
        if isinstance(item, MEvent):
            if item.dur is None:
                return False
            now += item.dur
        else:
            if not _stacked(item.items, now=now):
                return False
            now += item.resolvedDur()
    return True


def stackEvents(events: list[MEvent | Chain],
                explicitOffsets=True,
                explicitDurations=True,
                defaultDur=F(1),
                offset=F(0)
                ) -> F:
    """
    Stack events to the left **in place**, making any unset offset and duration explicit

    Args:
        events: the events to modify, either in place or as a copy
        explicitOffsets: if True, all offsets are made explicit, recursively
        explicitDurations: if True, all durations are made explicit, recursively
        defaultDur: the default duration used when an event has no duration and
            the next event does not have an explicit offset
        offset: the offset of the events given

    Returns:
        the accumulated duration of all events

    """
    if not events:
        raise ValueError("No events given")

    # All offset times given in the events are relative to the start of the chain
    now = F(0)
    lasti = len(events) - 1
    for i, ev in enumerate(events):
        if ev.offset is not None:
            assert ev.offset >= now, (f'{ev} has an explicit offset={ev.offset}, but it overlaps with '
                                      f'the calculated offset ({now})')
            now = ev.offset
        elif explicitOffsets:
            ev.offset = now

        ev._resolvedOffset = now
        if isinstance(ev, MEvent):
            if (_ := ev._calculateDuration(relativeOffset=now, parentOffset=offset)) is not None:
                dur = _
            elif i == lasti:
                dur = defaultDur
            else:
                nextev = events[i+1]
                if nextev.offset is not None:
                    dur = nextev.offset - now
                elif nextev._resolvedOffset is not None:
                    dur = nextev._resolvedOffset - now
                else:
                    dur = defaultDur
            now += dur
            ev._resolvedDur = dur
            if explicitDurations:
                ev.dur = dur
        elif isinstance(ev, Chain):
            stackeddur = stackEvents(ev.items,
                                     explicitOffsets=explicitOffsets,
                                     explicitDurations=explicitDurations,
                                     offset=now)
            now = ev._resolvedOffset + stackeddur
            ev.dur = stackeddur
        else:
            raise TypeError(f"Expected an MEvent (Note, Chord, ...) or a Chain, got {ev}")

    if explicitDurations:
        assert all(ev.dur is not None for ev in events if isinstance(ev, MEvent))
    return now


def _removeRedundantOffsets(items: list[MEvent | Chain],
                            fillGaps=False,
                            frame=F(0)
                            ) -> F:
    """
    Remove over-secified start times in this Chain

    Args:
        fillGaps: if True, any gap resulting of an event's starting
            after the end  of the previous event will be filled with
            a Rest (and the event's offset will be removed since it
            becomes redundant)

    Returns:
        the total duration of *items*

    """
    # This is the relative position (independent of the chain's start)
    assert isinstance(items, list) and all(isinstance(item, (MEvent, Chain)) for item in items)
    now = frame
    out = []
    for item in items:
        itemoffset = item._detachedOffset()
        absoffset = itemoffset + frame if itemoffset is not None else None
        if itemoffset is not None:
            if absoffset == now:
                item.offset = None
            elif absoffset < now:
                raise ValueError(f"Items overlap: {item} (offset={_util.showT(absoffset)}) "
                                 f"starts before current time ({_util.showT(now)})")
            else:
                if fillGaps:
                    out.append(Note.makeRest(absoffset - now))
                now = absoffset

        if isinstance(item, MEvent):
            dur = dur if (dur := item.dur) is not None else item._resolvedDur
            if dur is None:
                raise ValueError(f"This Chain contains events with unspecified duration: {item}")
            now += dur
        elif isinstance(item, Chain):  # a Chain
            dur = _removeRedundantOffsets(item.items, fillGaps=fillGaps, frame=now)
            item._modified = True
            now += dur
        else:
            raise TypeError(f"Expected a Note, Chord or Chain, got {item}")
        out.append(item)
    items[:] = out
    return now - frame


class Chain(MObj, MContainer):
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

    __slots__ = ('items', '_modified')

    def __init__(self,
                 items: list[MEvent | Chain | str] = None,
                 offset: time_t = None,
                 label: str = '',
                 properties: dict[str, Any] = None,
                 parent: MObj = None,
                 _init=True):
        if _init:
            if offset is not None:
                offset = asF(offset)
            if items is not None:
                items = [item if isinstance(item, (MEvent, Chain)) else asEvent(item)
                         for item in items]

        super().__init__(offset=offset, dur=None, label=label,
                         properties=properties, parent=parent)
        if items is not None:
            for item in items:
                item.parent = self
        else:
            items = []
        assert isinstance(items, list)
        self.items: list[MEvent | Chain] = items
        self._modified = items is not None

    def __hash__(self):
        items = [type(self).__name__, self.label, self.offset, len(self.items)]
        if self.symbols:
            items.extend(self.symbols)
        items.extend(self.items)
        out = hash(tuple(items))
        return out

    def clone(self, items=UNSET, offset=UNSET, label=UNSET, properties=UNSET) -> Chain:
        # parent is not cloned
        return Chain(items=self.items.copy() if items is UNSET else items,
                     offset=self.offset if offset is UNSET else asF(offset),
                     label=self.label if label is UNSET else label,
                     properties=self.properties if properties is UNSET else properties,
                     _init=False)

    def copy(self) -> Chain:
        # chain's parent is not copied
        items = [item.copy() for item in self.items]
        return Chain(items=items, offset=self.offset, label=self.label,
                     properties=self.properties, _init=False)

    def isResolved(self) -> bool:
        """
        True if all items in this chain have an offset and duration
        """
        self._update()
        for item in self.items:
            if isinstance(item, MEvent):
                if item.offset is None and item._resolvedOffset is None:
                    return False
                if item.dur is None and item._resolvedDur is None:
                    return False
            else:
                if not item.isResolved():
                    return False
        return True

    def stack(self, explicitOffsets=True, explicitDurations=True) -> F:
        """
        Stack events to the left **INPLACE**, optionally making offset/duration explicit

        Args:
            explicitDurations: make all durations explicit (sets the .dur attribute of all
                items in this chain, recursively)
            explicitOffsets: make all offset explicit (sets the .offset attribute of
                all items in this chain, recursively)

        Returns:
            the total duration of self
        """
        dur = stackEvents(self.items, explicitDurations=explicitDurations, explicitOffsets=explicitOffsets)
        self.dur = dur
        return dur

    def fillGaps(self, recurse=True) -> None:
        """
        Fill any gaps with rests, in place

        A gap is produced when an event within a chain has an explicit offset
        later than the offset calculated by stacking the previous objects in terms
        of their duration

        Args:
            recurse: if True, fill gaps within subchains
        """
        self._update()
        now = F(0)
        items = []
        for item in self.items:
            if item.offset is not None and item.offset > now:
                gapdur = item.offset - now
                r = Note.makeRest(gapdur)
                items.append(r)
                now += gapdur
            items.append(item)
            if isinstance(item, Chain) and recurse:
                item.fillGaps(recurse=True)
            dur = item.dur if item.dur is not None else item._resolvedDur
            assert dur is not None
            now += dur
        self.items = items
        self._modified = True

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

    def flat(self, forcecopy=False) -> Chain:
        """
        A flat version of this Chain

        A Chain can contain other Chains. This method flattens all objects inside
        this Chain and any sub-chains to a flat chain of notes/chords.

        If this Chain is already flat (it does not contain any
        Chains), self is returned unmodified.

        Args:
            forcecopy: if True the returned Chain is completely independent
                of self, even if self is already flat

        Returns:
            a chain with exclusively Notes and/or Chords.
        """
        self._update()

        if all(isinstance(item, MEvent) for item in self.items) and not forcecopy and self.isResolved():
            return self

        items = []
        for event, offset, eventdur in self.recurseWithTimes():
            if forcecopy or event.offset != offset or event.dur != eventdur:
                event = event.copy()
            event.offset = offset
            event.dur = eventdur
            items.append(event)
        return self.clone(items=items)

    def pitchRange(self) -> tuple[float, float] | None:
        pitchRanges = [item.pitchRange() for item in self.items]
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

    def resolved(self) -> Chain:
        """
        Copy of self with explicit times

        The items in the returned object have an explicit start and
        duration.

        Returns:
            a clone of self with dur and offset set to explicit
            values

        """
        if self.isResolved():
            return self
        out = self.copy()
        out.stack(explicitOffsets=True)
        return out

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        chain = self.flat(forcecopy=True)
        chain.stack()
        assert all(item.offset is not None and item.dur is not None for item  in chain)

        conf = workspace.config
        if self.playargs:
            playargs = playargs.overwrittenWith(self.playargs)

        if self.offset:
            for item in chain.items:
                item.offset += self.offset

        if any(n.isGracenote() for n in chain.items
               if isinstance(n, (Note, Chord))):
            _mobjtools.addDurationToGracenotes(chain.items, F(1, 14))
        if conf['play.useDynamics']:
            _mobjtools.fillTempDynamics(chain.items, initialDynamic=conf['play.defaultDynamic'])
        return _mobjtools.chainSynthEvents(chain.items, playargs=playargs, workspace=workspace)

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

    def childDuration(self, child: MObj) -> F:
        if child.dur is not None:
            assert child in self.items
            return child.dur

        self._update()
        dur = child.dur if child.dur is not None else child._resolvedDur
        assert dur is not None
        return dur

    def childOffset(self, child: MObj) -> F:
        """
        Returns the offset of child within this chain

        Args:
            child: the object whose offset is to be determined

        Returns:
            The offset of this child within this chain
        """
        if child.offset is not None:
            assert child in self.items
            return child.offset

        self._update()
        for item in self.items:
            if item is child:
                return item.offset if item.offset is not None else item._resolvedOffset

        raise ValueError(f"The item {child} is not a child of {self}")

    def resolvedDur(self) -> F:
        if self.dur is not None and not self._modified:
            return self.dur

        if not self.items:
            self.dur = F(0)
            return self.dur

        self._update()
        assert self.dur is not None
        return self.dur

    def append(self, item: MEvent) -> None:
        """
        Append an item to this chain

        Args:
            item: the item to add
        """
        item.parent = self
        self.items.append(item)
        self._changed()

    def extend(self, items: list[MEvent]) -> None:
        """
        Extend this chain with items

        Args:
            items: a list of items to append to this chain

        .. note::

            Items passed are marked as children of this chain (their *.parent* attribute
            is modified)
        """
        for item in items:
            item.parent = self
        self.items.extend(items)
        self._changed()

    def _update(self):
        if not self._modified and self.dur is not None:
            return
        self.stack(explicitOffsets=False, explicitDurations=False)
        self._modified = False

    def _changed(self) -> None:
        self._modified = True
        self.dur = None

    def childChanged(self, child: MObj) -> None:
        self._changed()

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[MEvent]:
        return iter(self.items)

    @overload
    def __getitem__(self, idx: int) -> MEvent: ...

    @overload
    def __getitem__(self, slice_: slice) -> list[MEvent | Chain]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.items[idx]
        else:
            return self.items.__getitem__(idx)

    def _dumpRows(self, indents=0, now=F(0)) -> list[str]:
        fontsize = '85%'
        IND = '  '
        selfstart = f"{float(self.offset):.3g}" if self.offset is not None else 'None'
        namew = max((sum(len(n.name) for n in event.notes) + len(event.notes)
                     for event in self.recurse()
                     if isinstance(event, Chord)),
                    default=10)

        widths = {
            'beat': 7,
            'offset': 12,
            'dur': 12,
            'name': namew,
            'gliss': 6,
            'dyn': 5,
            'playargs': 20,
            'info': 20
        }

        self._update()

        if environment.insideJupyter:
            _ = _util.htmlSpan
            r = type(self).__name__

            header = (f'<code><span style="font-size: {fontsize}">{IND*indents}<b>{r}</b> - '
                      f'beat: {self.absoluteOffset()}, offset: {selfstart}, dur: {_util.showT(self.resolvedDur())}'
                      )
            if self.label:
                header += f', label: {self.label}'
            header += '</span></code>'
            rows = [header]
            columnparts = [IND*(indents+1)]
            for k, width in widths.items():
                columnparts.append(k.ljust(width))
            columnnames = ''.join(columnparts)
            row = f"<code>{_util.htmlSpan(columnnames, ':grey1', fontsize=fontsize)}</code>"
            rows.append(row)

            items, itemsdur = self._iterateWithTimes(recurse=False, frame=F(0))
            for item, itemoffset, itemdur in items:
                infoparts = []
                if item.label:
                    infoparts.append(f'label: {item.label}')

                if isinstance(item, MEvent):
                    name = item.name
                    if isinstance(item, (Note, Chord)) and item.tied:
                        name += "~"
                    if item.offset is not None:
                        offsetstr = _util.showT(item.offset).ljust(widths['offset'])
                    else:
                        offsetstr = f'({itemoffset})'.ljust(widths['offset'])
                    if item.dur is not None:
                        durstr = _util.showT(item.dur).ljust(widths['dur'])
                    else:
                        durstr = f'({itemdur})'.ljust(widths['dur'])
                    rowparts = [IND*(indents+1),
                                _util.showT(now + itemoffset).ljust(widths['beat']),
                                offsetstr,
                                durstr,
                                name.ljust(widths['name']),
                                str(item.gliss).ljust(widths['gliss']),
                                str(item.dynamic).ljust(widths['dyn']),
                                str(self.playargs).ljust(widths['playargs']),
                                ' '.join(infoparts) if infoparts else '-'
                                ]
                    row = f"<code>{_util.htmlSpan(''.join(rowparts), ':blue1', fontsize=fontsize)}</code>"
                    rows.append(row)
                    if item.symbols:
                        row = f"<code>      {_util.htmlSpan(str(item.symbols), ':green2', fontsize=fontsize)}</code>"
                        rows.append(row)

                elif isinstance(item, Chain):
                    rows.extend(item._dumpRows(indents=indents+1, now=now+itemoffset))
            return rows
        else:
            rows = [f"{' ' * indents}Chain"]
            for item, itemoffset, itemdur in self._iterateWithTimes(recurse=False, frame=now):
                if isinstance(item, MEvent):
                    rows.append(f'{IND}{item} resolved offset: {itemoffset}, resolved dur: {itemdur}')
                    sublines = repr(item).splitlines()
                    for subline in sublines:
                        rows.append(f"{'  ' * (indents + 1)}{subline}")
                else:
                    rows.extend(item._dumpRows(indents=indents+1))
            return rows

    def dump(self, indents=0) -> None:
        """
        Dump this chain, recursively

        Values inside parenthesis are implicit. For example if an object inside
        this chain does not have an explicit .offset, its resolved offset will
        be shown within parenthesis
        """
        rows = self._dumpRows(indents=indents, now=self.offset or F(0))
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

    def _repr_html_header(self) -> str:
        itemcolor = safeColors['blue2']
        items = self.items if len(self.items) < 10 else self.items[:10]
        itemstr = ", ".join(f'<span style="color:{itemcolor}">{repr(_)}</span>'
                            for _ in items)
        if len(self.items) >= 10:
            itemstr += ", …"
        cls = self.__class__.__name__
        namedargs = [f'dur={_util.showT(self.resolvedDur())}']
        if self.offset is not None:
            namedargs.append(f'offset={self.offset}')
        info = ', ' + ', '.join(namedargs)
        return f'{cls}([{itemstr}]{info})'

    def removeRedundantOffsets(self, fillGaps=False) -> None:
        """
        Remove over-secified start times in this Chain

        Args:
            fillGaps: if True, any gap resulting of an event's starting
                after the end  of the previous event will be filled with
                a Rest (and the event's offset will be removed since it
                becomes redundant)

        Returns:
            self if inplace=True, else the modified copy

        """
        # This is the relative position (independent of the chain's start)
        self._update()
        _removeRedundantOffsets(self.items, fillGaps=fillGaps, frame=self.resolvedOffset())
        self._modified = True

    def asVoice(self) -> Voice:
        """
        Convert this Chain to a Voice
        """
        items, itemsdur = stackEvents(self.items)
        for item in items:
            item.offset += self.offset
        voice = Voice(items, label=self.label)
        voice.removeRedundantOffsets()
        return voice

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
        if not self.items:
            return []

        if config is None:
            config = getConfig()

        chain = self.flat()
        notations: list[scoring.Notation] = []
        for item in chain.items:
            notations.extend(item.scoringEvents(groupid=groupid, config=config))
        if self.label:
            annot = self._scoringAnnotation()
            annot.instancePriority = -1
            notations[0].addAttachment(annot)
        scoring.stackNotationsInPlace(notations)
        if self.offset is not None and self.offset > 0 and not config['show.asoluteOffsetForDetachedObjects']:
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

    def quantizePitch(self, step=0.25):
        if step == 0:
            raise ValueError("Step should be possitive")
        items = [i.quantizePitch(step) for i in self.items]
        return self.clone(items=items)

    def timeShift(self, timeoffset: time_t) -> Chain:
        if self.offset is not None:
            return self.clone(offset=self.offset+timeoffset)
        return self.clone(offset=timeoffset)

    def firstOffset(self) -> F | None:
        """
        Returns the offset (relative to the start of this chain) of the first event in this chain
        """
        if not self.items:
            return None
        item = next(item for item in self.recurse())
        return item.absoluteOffset() - self.absoluteOffset()

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
            an iterator over all notes/chords within this chain and its sub-chains, where
            for each event a tuple (event: MEvent, offset: F) is returned. The offset is
            relative to the offset of this chain, so in order to determine the absolute
            offset for each returned event one needs to add the absolute offset of this
            chain

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

    def _iterateWithTimes(self,
                          recurse: bool,
                          frame: F,
                          ) -> tuple[list[tuple[MEvent | list, F, F]], F]:
        """
        For each item returns a tuple (item, offset, duration)

        Each event is represented as a tuple (event, offset, dur), a chain
        is represented as a list of such tuples

        Args:
            recurse: if True, traverse any subchain
            frame: the frame of reference

        Returns:
            a tuple (eventtuples, duration) where eventtuples is a list of
            tuples (event, offset, dur). If recurse is True,
            any subchain is returned as a list of eventtuples. Otherwise
            a flat list is returned. In each eventtuple, the offset is relative
            to the first frame passed, so if the first offset was 0
            the offsets will hold the absolute offset of each event. Duration
            is the total duration of the items in
            the chain (not including its own offset)

        """
        now = frame
        out = []
        for i, item in enumerate(self.items):
            if item.offset is not None:
                t = frame + item.offset
                assert t >= now
                now = t
            if isinstance(item, MEvent):
                if item.dur is not None:
                    dur = item.dur
                elif i == len(self.items) - 1:
                    dur = F(1)
                else:
                    nextobj = self.items[i + 1]
                    if nextobj.offset is not None:
                        dur = nextobj.offset + frame - now
                    else:
                        dur = F(1)
                out.append((item, now, dur))
                item._resolvedDur = dur
                item._resolvedOffset = now - frame
                now += dur
            else:
                if recurse:
                    subitems, subdur = item._iterateWithTimes(frame=now, recurse=True)
                    item.dur = subdur
                    item._resolvedOffset = now - frame
                    out.append((subitems, now, subdur))
                else:
                    subdur = item.resolvedDur()
                    out.append((item, now, subdur))
                now += subdur
        return out, now - frame

    def recurseWithTimes(self, absoluteTimes=False) -> Iterator[tuple[MEvent, F, F]]:
        """
        Recurse the events in self, yields a tuple (event, offset, duration) for each event

        Args:
            absoluteTimes: if True, the offset returned for each item will be its
                absolute offset; otherwise the offsets are relative to the start of self

        Returns:
            a generator of tuples (event, offset, duration), where offset is either the
            offset relative to the start of self, or absolute if absoluteTimes is True

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> chain = Chain([
            ... "4C:0.5",
            ... "4D",
            ... Chain(["4E:0.5"], offset=2)
            ... ], offset=1)

        """
        frame = self.absoluteOffset() if absoluteTimes else F(0)
        itemtuples, totaldur = self._iterateWithTimes(frame=frame, recurse=True)
        for itemtuple in itemtuples:
            item = itemtuple[0]
            if isinstance(item, MEvent):
                yield itemtuple
            else:
                yield from item

    def resolvedTimes(self, absoluteTimes=False
                      ) -> list[tuple[MEvent, F, F] | list]:
        """
        Resolves the times of the events without modifying them

        For each event it returns a tuple (event, offset, duration), where the
        event is unmodified, the offset is the resolved offset relative to its parent
        and the duration is the resolved duration. Nested
        subchains result in nested lists. All times are in beats

        Args:
            absoluteTimes: if True, offsets are returned as absolute offsets

        Returns:
            a list of tuples, one tuple for each event.
        """
        offset = self.absoluteOffset() if absoluteTimes else F(0)
        items, itemsdur = self._iterateWithTimes(frame=offset, recurse=True)
        self.dur = itemsdur
        return items

    def addSpanner(self, spanner: str | symbols.Spanner, endobj: MObj = None
                   ) -> None:
        first = next(self.recurse())
        last = next(self.recurse(reverse=True))
        if isinstance(spanner, str):
            spanner = symbols.makeSpanner(spanner)
        assert isinstance(first, (Note, Chord)) and isinstance(last, (Note, Chord))
        spanner.bind(first, last)


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

