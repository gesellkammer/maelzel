from __future__ import annotations
from .mobj import MObj, MEvent, asEvent, stackEvents, Note, Chord
from . import _mobjtools
from . import symbols
from ._common import UNSET, asRat, Rat
from . import _util
from .synthevent import PlayArgs, SynthEvent
from . import environment
from .workspace import getConfig, Workspace

from maelzel import scoring

from maelzel.colortheory import safeColors

from emlib import iterlib
from typing import TYPE_CHECKING, overload
if TYPE_CHECKING:
    from typing import Any, Iterator, overload, TypeVar, Callable
    from ._typedefs import time_t
    from .config import CoreConfig
    from maelzel.scorestruct import ScoreStruct
    ChainT = TypeVar("ChainT", bound="Chain")

__all__ = (
    'Chain',
    'Voice'
)


def _itemsAreStacked(items: list[MEvent | Chain]) -> bool:
    for item in items:
        if isinstance(item, MEvent):
            if item.start is None or item.dur is None:
                return False
        elif isinstance(item, Chain):
            if item.start is None or not _itemsAreStacked(item.items):
                return False
    return True


class Chain(MObj):
    """
    A Chain is a sequence of Notes, Chords or other Chains

    Attributes:
        items: the items of this Chain. Each item is either a MEvent (a Note or Chord)
            or a subchain.
        start: the offset of this chain or None if the start time depends on the
            position of the chain within another chain

    Args:
        items: the items of this Chain. The start time of any object, if given, is
            interpreted as relative to the start of the chain.
        start: start time of the chain itself
        label: a label for this chain
        properties: any properties for this chain. Properties can be anything,
            they are a way for the user to attach data to an object
    """
    _acceptsNoteAttachedSymbols = False

    def __init__(self,
                 items: list[MEvent | Chain | str] = None,
                 start: time_t = None,
                 label: str = '',
                 properties: dict[str, Any] = None):
        if start is not None:
            start = asRat(start)
        if items is not None:
            items = [item if isinstance(item, (MEvent, Chain)) else asEvent(item)
                     for item in items]
            for i0, i1 in iterlib.pairwise(items):
                assert i0.start is None or i1.start is None or i0.start <= i1.start, f'{i0 = }, {i1 = }'
        else:
            items = []

        super().__init__(start=start, dur=None, label=label, properties=properties)
        self.items: list[MEvent | 'Chain'] = items

    def __hash__(self):
        items = [type(self).__name__, self.label, self.start, len(self.items)]
        if self.symbols:
            items.extend(self.symbols)
        items.extend(self.items)
        out = hash(tuple(items))
        return out

    def clone(self, items=UNSET, start=UNSET, label='', properties=UNSET) -> Chain:
        return Chain(items=self.items if items is UNSET else items,
                     start=self.start if start is UNSET else start,
                     label=self.label if label is UNSET else label,
                     properties=self.properties if properties is UNSET else properties)

    def copy(self) -> Chain:
        items = [item.copy() for item in self.items]
        return Chain(items=items, start=self.start, label=self.label, properties=self._properties)

    def isStacked(self) -> bool:
        """
        True if items in this chain have a defined offset and duration
        """
        return self.start is not None and _itemsAreStacked(self.items)

    def fillGapsWithRests(self) -> None:
        """
        Fill any gaps with rests

        A gap is produced when an event within a chain has an explicit start time
        later than the offset calculated by stacking the previous objects in terms
        of their duration
        """
        # TODO
        pass

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


    def flat(self, removeRedundantOffsets=True) -> Chain:
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

        Returns:
            a chain with exclusively Notes and/or Chords
        """
        if all(isinstance(item, MEvent) for item in self.items):
            return self
        chain = self.resolved()
        offset = chain.start if chain.start is not None else Rat(0)
        items = _mobjtools.flattenObjs(chain.items, offset)
        if chain.start is not None:
            for item in items:
                item.start -= chain.start
        out = self.clone(items=items)
        if removeRedundantOffsets:
            out.removeRedundantOffsets()
        return out

    def pitchRange(self) -> tuple[float, float] | None:
        pitchRanges = [item.pitchRange() for item in self.items]
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

    def resolved(self, start: time_t = None) -> Chain:
        """
        Copy of self with explicit times

        The items in the returned object have an explicit start and
        duration.

        .. note:: use a start time of 0 to have an absolute start
            time set for each item.

        Args:
            start: a start time to fill or override self.start.

        Returns:
            a clone of self with dur and start set to explicit
            values

        """
        if start is not None:
            offset = self.resolvedStart() - start
            if offset < 0:
                raise ValueError(f"This would result in a negative offset: {offset}")
            clonedStart = start
        else:
            offset = 0
            clonedStart = self.start
        if self.isStacked():
            return self
        items = stackEvents(self.items, offset=offset, recurse=True)
        return self.clone(items=items, start=clonedStart)

    def resolvedStart(self) -> Rat:
        ownstart = self.start or Rat(0)
        if not self.items:
            return ownstart
        item = self.items[0]
        return ownstart if item.start is None else ownstart + item.start

    def _synthEvents(self, playargs: PlayArgs, workspace: Workspace
                     ) -> list[SynthEvent]:
        chain = self.flat(removeRedundantOffsets=False)
        conf = workspace.config
        if self._playargs:
            playargs.overwriteWith(self._playargs)
        items = stackEvents(chain.items, inplace=True, offset=self.start)
        if any(n.isGracenote() for n in self.items
               if isinstance(n, (Note, Chord))):
            _mobjtools.addDurationToGracenotes(items, Rat(1, 14))
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
                merged = _mobjtools.mergeIfPossible(last, item)
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
        if any(item.start is None for item in self.items):
            stackEvents(self.items, inplace=True)
        for item in self.items:
            item.start += timeoffset
        self._changed()

    def movedTo(self, start: time_t):
        offset = start - self.items[0].start
        return self.timeShift(offset)

    def moveTo(self, start: time_t):
        offset = start - self.items[0].start
        self.timeShiftInPlace(offset)

    def resolvedDur(self, start: time_t = None) -> Rat:
        if not self.items:
            return Rat(0)

        defaultDur = Rat(1)
        accum = Rat(0)
        items = self.items
        lasti = len(items) - 1
        if start is None:
            start = self.resolvedStart()

        for i, ev in enumerate(items):
            if ev.start is not None:
                accum = ev.start
            if isinstance(ev, MEvent):
                if ev.dur:
                    accum += ev.dur
                elif i == lasti:
                    accum += defaultDur
                else:
                    nextev = items[i + 1]
                    accum += defaultDur if nextev.start is None else nextev.start - accum
            else:
                # a Chain
                accum += ev.resolvedDur()

        return accum

    def append(self, item: Note|Chord) -> None:
        """
        Append an item to this chain

        Args:
            item: the item to add
        """
        self.items.append(item)
        if len(self.items) > 1:
            butlast = self.items[-2]
            last = self.items[-1]
            if isinstance(butlast, Note) and butlast.gliss is True and isinstance(last, Note):
                butlast.gliss = last.pitch
        self._changed()

    def _changed(self):
        if self.items:
            self.dur = self.resolvedDur()
        else:
            self.start = None
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
        selfstart = round(float(self.start.limit_denominator(1000)), 3) if self.start is not None else None
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
                    start = f"{float(item.start):.3g}" if item.start is not None else "None"
                    dur = f"{float(item.dur):.3g}" if item.dur is not None else "None"
                    rowtxt = f"{'  '*indents}{start.ljust(6)}{dur.ljust(durwidth)}{name.ljust(namew)}{str(item.gliss).ljust(6)}{str(item.dynamic).ljust(5)}{self._playargs}</code>"
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
        if self.start is not None:
            namedargs.append(f'start={self.start}')
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
        if self.start is not None:
            namedargs.append(f'start={self.start}')
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
        defaultDur = Rat(1)
        accumDur = Rat(0)
        maxDur = asRat(dur)
        items: list[MEvent] = []
        ownitems = stackEvents(self.items)
        for item in iterlib.cycle(ownitems):
            dur = item.dur if item.dur else defaultDur
            if dur > maxDur - accumDur:
                if crop:
                    dur = maxDur - accumDur
                else:
                    break
            if item.dur is None or item.start is not None:
                item = item.clone(dur=dur, start=None)
            assert isinstance(item, MEvent)
            items.append(item)
            accumDur += item.dur
            if accumDur == maxDur:
                break
        return self.__class__(items, start=self.start)

    def removeRedundantOffsets(self):
        """
        Remove over-secified start times in this Chain **inplace**
        """
        # This is the relative position (independent of the chain's start)
        now = Rat(0)
        for item in self.items:
            if isinstance(item, MEvent):
                if item.dur is None:
                    raise ValueError(f"This Chain contains events with unspecified duration: {item}")
                if item.start is None:
                    now += item.dur
                else:
                    if item.start < now:
                        raise ValueError(f"Items overlap: {item}, {now=}")
                    elif item.start > now:
                        now = item.end
                    else:
                        # item.start == now
                        item.start = None
                        now += item.dur
            elif isinstance(item, Chain):
                item.removeRedundantOffsets()
        if self.start == 0:
            self.start = None

    def asVoice(self) -> Voice:
        """Convert this Chain to a Voice"""
        resolved = self.resolved(start=0)
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
        if self.start is not None and self.start > 0:
            for notation in notations:
                notation.offset += self.start

        for n0, n1 in iterlib.pairwise(notations):
            if n0.tiedNext and not n1.isRest:
                n1.tiedPrev = True

        if self.symbols:
            for s in self.symbols:
                for n in notations:
                    s.applyTo(n)
        return notations

    def scoringParts(self, options: scoring.render.RenderOptions = None
                     ) -> list[scoring.Part]:
        notations = self.scoringEvents(config=getConfig())
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
        if self.start is not None:
            return self.clone(start=self.start+timeoffset)
        items = stackEvents(self.items, offset=timeoffset)
        return self.clone(items=items)

    def timeTransform(self, timemap: Callable[[Rat], Rat], inplace=False) -> Chain:
        start = self.resolvedStart()
        start2 = timemap(start)
        if inplace:
            stackEvents(self.items, inplace=True)
            for item in self.items:
                item.start = timemap(item.start + start) - start2
                item.dur = timemap(item.end + start) - start2 - item.start
            self.start = start2
            return self
        else:
            items = stackEvents(self.items, inplace=False)
            for item in items:
                item.start = timemap(item.start + start) - start2
                item.dur = timemap(item.end + start) - start2 - item.start
            return self.clone(items=items, start=start2)

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


class Voice(Chain):
    """
    A Voice is a sequence of non-overlapping objects

    It is **very** similar to a Chain, the only difference being that its start
    is always 0.


    Voice vs Chain
    ~~~~~~~~~~~~~~

    * A Voice can contain a Chain, but not vice versa.
    * A Voice does not have a start offset, its start is always 0.
    """

    _acceptsNoteAttachedSymbols = False

    def __init__(self,
                 items: list[MEvent|str] = None,
                 label='',
                 shortname=''):
        super().__init__(items=items, label=label, start=Rat(0))
        self.shortname = shortname

    def scoringParts(self, options: scoring.render.RenderOptions = None
                     ) -> list[scoring.Part]:
        parts = super().scoringParts(options=options)
        for part in parts:
            part.shortname = self.shortname
        return parts

    def setScoreStruct(self, scorestruct: ScoreStruct | None) -> None:
        self._scorestruct = scorestruct

