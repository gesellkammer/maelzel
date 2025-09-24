from __future__ import annotations

import sys
import itertools

from maelzel.common import F, asF, F0
from maelzel import _util
from maelzel.core.config import CoreConfig
from .mobj import MObj, MContainer
from .event import MEvent, asEvent, Note, Chord
from .workspace import Workspace
from .synthevent import PlayArgs, SynthEvent
from . import symbols
from . import environment
from . import _mobjtools
from . import _tools
from ._common import logger

from maelzel import scoring


from typing import TYPE_CHECKING, overload
if TYPE_CHECKING:
    from typing_extensions import Self
    from typing import Any, Iterable, Iterator, Callable, Sequence
    from maelzel.common import time_t, location_t, num_t, beat_t
    from maelzel.scoring import quant
    from maelzel.scorestruct import ScoreStruct


__all__ = (
    'Chain',
    'Voice'
)


def _removeRedundantOffsets(items: list[MEvent | Chain],
                            frame=F0
                            ) -> tuple[F, bool]:
    """
    Remove over-secified start times in this Chain

    Args:
        items: the items to process
        frame: the frame of reference

    Returns:
        a tuple (total duration of *items*, True if items was modified)

    """
    # This is the relative position (independent of the chain's start)
    now = frame
    modified = False
    for i, item in enumerate(items):
        if (itemoffset := item._detachedOffset()) is not None:
            absoffset = itemoffset + frame
            if absoffset == now and (i == 0 or items[i-1].dur is not None):
                if item.offset is not None:
                    modified = True
                    item.offset = None
            elif absoffset < now:
                raise ValueError(f"Items overlap: {item} (abs. offset={_util.showT(absoffset)}) "
                                 f"starts before current time ({_util.showT(now)})")
            else:
                now = absoffset

        if isinstance(item, MEvent):
            now += item.dur
        elif isinstance(item, Chain):
            dur, submodified = _removeRedundantOffsets(item.items, frame=now)
            if submodified:
                item._changed()
                modified = True
            now += dur
        else:
            raise TypeError(f"Expected an MEvent (Note, Chord, etc.) or a Chain, got {item}")

    return now - frame, modified


class Chain(MContainer):
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

    __slots__ = ('items', '_modified', '_cachedEventsWithOffset', '_postSymbols',
                 '_absOffset', '_hasRedundantOffsets')

    def __init__(self,
                 items: Sequence[MEvent | Chain | str] | str = (),
                 offset: time_t | None = None,
                 label: str = '',
                 properties: dict[str, Any] | None = None,
                 parent: MContainer | None = None,
                 _init=True):
        self._modified = bool(items)
        """True if this object was modified and needs to be updated"""

        self._cachedEventsWithOffset: list[tuple[MEvent, F]] | None = None

        self._postSymbols: list[tuple[symbols.Symbol, time_t, time_t | None]] = []
        """Symbols to apply a posteriory, a tuple (symbol, abs. offset, end=None)"""

        self._absOffset: F | None = None
        """Cached absolute offset"""

        self._hasRedundantOffsets = True
        """Assume redundant offsets at creation"""

        if isinstance(items,  str):
            _init = True

        if _init:
            if offset is not None:
                offset = offset if isinstance(offset, F) else asF(offset)
            if items is None:
                items = []
            else:
                if isinstance(items, str):
                    # split using new lines and semicolons as separators
                    tokens = _tools.regexSplit('[\n;]', items, strip=True, removeEmpty=True)
                    tokens2 = [_tools.stripNoteComments(token) for token in tokens]
                    items = [asEvent(tok) for tok in tokens2 if tok]

                else:
                    uniqueitems = []
                    for item in items:
                        if isinstance(item, MEvent):
                            uniqueitems.append(item if item.parent is None or item.parent is self else item.copy())
                        elif isinstance(item, Chain):
                            uniqueitems.append(item)
                        else:
                            uniqueitems.append(asEvent(item))
                    items = uniqueitems
                for item in items:
                    item.parent = self
        else:
            assert offset is None or isinstance(offset, F), f"Expected a Fraction, got {offset}"
            if items:
                for item in items:
                    assert isinstance(item, (MEvent, Chain)), f"Expected an MEvent or Chain, got {type(item)}"
                    item.parent = self

        super().__init__(offset=offset,
                         label=label,
                         properties=properties,
                         parent=parent)

        if items is None:
            items = []
        elif items and not isinstance(items, list):
            items = list(items)

        self.items: list[MEvent | Chain] = items  # type: ignore
        """The items in this chain, a list of events of other chains"""


    def _check(self):
        for item in self.items:
            assert item.parent is self
            if isinstance(item, Chain):
                item._check()

    def __hash__(self):
        items = [type(self).__name__, self.label, self.offset, len(self.items)]
        if self.symbols:
            items.extend(self.symbols)
        if self._postSymbols:
            items.extend(self._postSymbols)
        if self._config:
            items.extend((k, self._config[k]) for k in sorted(self._config.keys()))
        items.extend(self.items)
        out = hash(tuple(items))
        return out

    def clone(self,
              items: Sequence[MEvent | Chain] | None = None,
              offset: time_t | None = -1,
              label: str | None = None,
              properties: dict | None = None
              ) -> Self:
        offset = None if offset is None else asF(offset) if offset >= 0 else self.offset
        if items is None:
            items = [item.copy() for item in self.items]
        out = self.__class__(items=items,
                             offset=offset,
                             label=self.label if label is None else label,
                             _init=False)
        self._copyAttributesTo(out)
        return out

    def __copy__(self) -> Self:
        out = self.__class__(self.items.copy(), offset=self.offset, label=self.label,
                             properties=self.properties, _init=False)
        self._copyAttributesTo(out)
        return out

    def __deepcopy__(self, memodict={}) -> Self:
        items = [item.copy() for item in self.items]
        out = self.__class__(items=items, offset=self.offset, label=self.label, _init=False)
        self._copyAttributesTo(out)
        return out

    def _copyAttributesTo(self, other: Self) -> None:
        super()._copyAttributesTo(other)
        if self._postSymbols:
            other._postSymbols = self._postSymbols.copy()

    def copy(self) -> Self:
        return self.__deepcopy__()

    def stack(self) -> None:
        """
        Stack events to the left **INPLACE**, making offsets explicit

        This method modifies the items within this object.

        Example
        ~~~~~~~

            >>> chain = Chain([Note("4C", dur=0.5),
            ...                Note("4D", dur=1, offset=4),
            ...                Note("4E", dur=0.5)])
            >>> chain.dump()
            Chain -- beat: 0, offset: None, dur: 11/2
              beat   offset  dur    item
              0      None    0.5    4C:0.5♩
              4      4       1      4D:1♩:offset=4
              5      None    0.5    4E:0.5♩
            >>> chain.stack()
            >>> chain.dump()  # Notice how all offsets are now explicit (they are no longer None)
            Chain -- beat: 0, offset: None, dur: 11/2
              beat   offset  dur    item
              0      0       0.5    4C:0.5♩:offset=0
              4      4       1      4D:1♩:offset=4
              5      5       0.5    4E:0.5♩:offset=5

        """
        dur = _stackEvents(self.items, explicitOffsets=True)
        self._dur = dur
        self._cachedEventsWithOffset = None

    def fillGaps(self, recurse=True) -> None:
        """
        Fill any gaps with rests, inplace

        A gap is produced when an event within a chain has an explicit offset
        later than the offset calculated by stacking the previous objects in terms
        of their duration

        Args:
            recurse: if True, fill gaps within subchains
        """
        self._update()
        now = F0
        items = []
        for item in self.items:
            if item.offset is not None and item.offset > now:
                gapdur = item.offset - now
                r = Note.makeRest(gapdur)
                items.append(r)
                now += gapdur
            items.append(item)
            if recurse and isinstance(item, Chain):
                item.fillGaps(recurse=True)
            now += item.dur
        self.setItems(items)

    def nextItem(self, item: MEvent | Chain) -> MEvent | Chain | None:
        """
        Returns the next item after *item*

        An item can be an event (note, chord) or another chain

        Args:
            item: the item to find its next item

        Returns:
            the item following *item* or None if the given item is not
            in this container, or it has no item after it

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> chain = Chain(['4C', '4D', Chain(['4E', '4F'])])
            >>> chain.eventAfter(chain[1])
            4E
            >>> chain.itemAfter(chain[1])
            Chain([4E, 4F])

        .. seealso:: :meth:`Chain.nextEvent`

        """
        idx = self.items.index(item)
        return self.items[idx + 1] if idx < len(self.items) - 2 else None

    def nextEvent(self, event: MEvent) -> MEvent | None:
        """
        Returns the next event after *event*

        Args:
            event: the start event

        Returns:
            The event following the given event, even if this
            is part of a chain, or None if no event exists after
            the given event

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> chain = Chain(['4C', '4D', Chain(['4E', '4F'])])
            # Notice how this method returns the event within the sub-chain
            >>> chain.nextEvent(chain[1])
            4E
            # In this case the next item is the entire chain
            >>> chain.nextItem(chain[1])
            Chain([4E, 4F])
        """
        idx = self.items.index(event)
        if idx >= len(self.items) - 1:
            return None
        nextitem = self.items[idx+1]
        return nextitem if isinstance(nextitem, MEvent) else nextitem.firstEvent()

    def previousItem(self, item: MEvent | Chain) -> MEvent | Chain | None:
        """
        Returns the item (an event or a chain) previous to the given one

        Args:
            item: the item to query.

        Returns:
            the item previous to *item*

        .. seealso:: :meth:`Chain.previousEvent`
        """
        try:
            idx = self.items.index(item)
            return None if idx == 0 else self.items[idx - 1]
        except ValueError as e:
            raise ValueError(f"The item {item} is not a part of {self}: {e}")

    def previousEvent(self, event: MEvent) -> MEvent | None:
        """
        Returns the event before the given event

        Args:
            event: the event to query

        Returns:
            the event before the given event, or None if no event is found. Raises
            ValueError if event is not part of this container

        """
        try:
            idx = self.items.index(event)
            if idx == 0:
                # This is the first event, so no previous event
                return None
            previtem = self.items[idx - 1]
            return previtem if isinstance(previtem, MEvent) else previtem.lastEvent()
        except ValueError as e:
            raise ValueError(f"event {event} not part of {self}, exception: {e}")

    def isFlat(self) -> bool:
        """
        Is self flat?

        A flat chain/voice contains only events, not other containers
        """
        return all(isinstance(item, MEvent) for item in self.items)

    def flatEvents(self, forcecopy=False) -> list[MEvent]:
        """
        A list of flat events, with explicit absolute offsets set

        The returned events are a clone of the events in this chain,
        not the actual events themselves

        Args:
            forcecopy: if True, all returned events are copy of events
                within self, even if they have an explicit absolute offset

        Returns:
            a list of events (Notes, Chords, Clips, ...) with explicit
            absolute offset            offset

        .. seealso:: :meth:`Chain.eventsWithOffset`

        """

        if not self.items:
            return []
        self._update()
        if forcecopy:
            flatitems = [ev.clone(offset=evoffset) if ev.offset != evoffset else ev.copy()
                         for ev, evoffset in self.eventsWithOffset()]
        else:
            flatitems = [ev.clone(offset=evoffset) if ev.offset != evoffset else ev
                         for ev, evoffset in self.eventsWithOffset()]
        _resolveGlissandi(flatitems)
        return flatitems

    def flat(self, forcecopy=False) -> Self:
        """
        A flat version of this Chain

        A Chain can contain other Chains. This method flattens all objects inside
        this Chain and any sub-chains to a flat chain of events (notes/chords/clips).

        If this Chain is already flat (it does not contain any
        Chains), self is returned unmodified (unless forcecopy=True).

        .. note::

            All items in the returned Chain will have an explicit ``.offset`` attribute.
            To remove any redundant .offset call :meth:`Chain.removeRedundantOffsets`

        Args:
            forcecopy: return a new chain, even if self is already flat

        Returns:
            a flat chain

        .. seealso:: :meth:`Chain.isFlat`, :meth:`Chain.recurse`
        """
        self._update()

        if not forcecopy and all(isinstance(item, MEvent) for item in self.items) and self.hasOffsets():
            return self

        ownoffset = self.absOffset()
        events = [ev.clone(offset=offset - ownoffset) for ev, offset in self.eventsWithOffset()]
        _resolveGlissandi(events)
        postsymbols = self._flatPostSymbols(self.getConfig() or Workspace.active.config)
        out = self.clone(items=events)
        if postsymbols:
            out._postSymbols.extend(postsymbols)
            out._postSymbols.sort(key=lambda entry: entry[1])
        return out

    def pitchRange(self) -> tuple[float, float] | None:
        pitchRanges = [pitchrange for item in self.items
                       if (pitchrange := item.pitchRange()) is not None]
        if not pitchRanges:
            return None
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

    def meanPitch(self) -> float:
        items = [item for item in self.items if not item.isRest()]
        graceDur = F(1, 16)
        sumpitch, sumdur = 0., 0.
        for item in items:
            pitch = item.meanPitch()
            if pitch:
                dur = item.dur if item.dur > 0 else graceDur
                sumpitch += pitch * dur
                sumdur += float(dur)
        if sumdur == 0:
            return 0.0
        return sumpitch / sumdur

    def withExplicitOffset(self, forcecopy=False) -> Self:
        """
        Copy of self with explicit offset

        If self already has explicit offset, self itself
        is returned.

        Args:
            forcecopy: if forcecopy, a copy of self will be returned even
                if self already has explicit times

        Returns:
            a clone of self with explicit times

        Example
        ~~~~~~~

        The offset and dur shown as the first two columns are the resolved
        times. When an event has an explicit offset, these are
        shown as part of the event repr. See for example the second note, 4C,
        which in the first version does not have any explicit times and is shown
        as "4C" and in the second version it appears as "4C:2.5♩:offset=0.5"

            >>> from maelzel.core import *
            >>> chain = Chain([Rest(0.5), Note("4C"), Chord("4D 4E", offset=3)])
            >>> chain.dump()
            Chain
              offset: 0      dur: 0.5    | Rest:0.5♩
              offset: 0.5    dur: 2.5    | 4C
              offset: 3      dur: 1      | ‹4D 4E offset=3›
            >>> chain.withExplicitTimes().dump()
            Chain
              offset: 0      dur: 0.5    | Rest:0.5♩:offset=0
              offset: 0.5    dur: 2.5    | 4C:2.5♩:offset=0.5
              offset: 3      dur: 1      | ‹4D 4E 1♩ offset=3›


        """
        if self.hasOffsets() and not forcecopy:
            return self
        out = self.copy()
        out.stack()
        return out

    def hasOffsets(self) -> bool:
        """
        True if self has an explicit offset and all items as well (recursively)

        Returns:
            True if all items in self have explicit offsets
        """
        if self._parent and self.offset is None:
            return False

        return all(item.offset is not None if isinstance(item, MEvent) else item.hasOffsets()
                   for item in self.items)

    def dynamicAt(self, absoffset: F, fallback='') -> str:
        if self._parent and isinstance(self._parent, Chain):
            return self._parent.dynamicAt(absoffset, fallback=fallback)
        dyn = ''
        for event, offset in self.eventsWithOffset():
            if offset > absoffset:
                break
            if event.dynamic and event.dynamic != dyn:
                dyn = event.dynamic
        return dyn or fallback

    def _recurse(self) -> Iterator[MEvent]:
        for item in self.items:
            if isinstance(item, MEvent):
                yield item
            else:
                yield from item._recurse()

    def _resolveGlissandi(self, force=False) -> None:
        """
        Set the _glissTarget attribute with the pitch of the gliss target
        if a note or chord has an unset gliss target

        Args:
            force: if True, calculate/update all glissando targets

        """
        _resolveGlissandi(self._recurse(), force=force)

    def _synthEvents(self,
                     playargs: PlayArgs,
                     parentOffset: F,
                     workspace: Workspace
                     ) -> list[SynthEvent]:
        # TODO: add playback for crescendi (hairpins)
        conf = workspace.config
        if self.playargs:
            # We don't include the chain's automations since these are added
            # later, after events have been merged.
            playargs = playargs.updated(self.playargs, automations=False)

        flatitems = self.flatEvents()
        assert all(item.offset is not None and item.dur >= 0 for item in flatitems)

        if any(n.isGrace() for n in flatitems):
            graceDur = F(conf['play.graceDuration'])
            _mobjtools.addDurationToGracenotes(flatitems, graceDur)

        if conf['play.useDynamics']:
            _mobjtools.fillTempDynamics(flatitems, initialDynamic=conf['play.defaultDynamic'])

        synthevents = []
        offset = parentOffset + self.relOffset()
        groups = _mobjtools.groupLinkedEvents(flatitems)
        for item in groups:
            if isinstance(item, MEvent):
                # item has absolute timing so parent offset is 0
                events = item._synthEvents(playargs, parentOffset=F0, workspace=workspace)
                synthevents.extend(events)
            elif isinstance(item, list):
                synthgroups = [event._synthEvents(playargs, parentOffset=F0, workspace=workspace)
                               for event in item]
                synthlines = _splitSynthGroupsIntoLines(synthgroups)
                for synthline in synthlines:
                    if isinstance(synthline, SynthEvent):
                        synthevent = synthline
                    elif isinstance(synthline, list):
                        if len(synthline) == 1:
                            synthevent = synthline[0]
                        else:
                            synthevent = SynthEvent.mergeEvents(synthline)
                    else:
                        raise TypeError(f"Expected a SynthEvent or a list thereof, got {synthline}")
                    synthevents.append(synthevent)
                    # TODO: fix / add playargs
            else:
                raise TypeError(f"Did not expect {item}")

        if self.playargs and self.playargs.automations:
            scorestruct = self.scorestruct() or workspace.scorestruct
            for automation in self.playargs.automations:

                startsecs, endsecs = automation.absTimeRange(parentOffset=offset, scorestruct=scorestruct)
                presetman = Workspace.active.presetManager()
                for ev in synthevents:
                    preset = presetman.getPreset(ev.instr)
                    if automation.param not in preset.dynamicParams(aliases=True, aliased=True):
                        continue
                    overlap0, overlap1 = _util.overlap(float(startsecs), float(endsecs), ev.delay, ev.end)
                    if overlap0 > overlap1:
                        continue
                    synthautom = automation.makeSynthAutomation(scorestruct=scorestruct, parentOffset=offset)
                    ev.addAutomation(synthautom.cropped(float(overlap0), float(overlap1)))

        return synthevents

    def mergeTiedEvents(self) -> None:
        """
        Merge tied events **inplace**

        Two events can be merged if they are tied and the second
        event does not provide any extra information (does not have
        an individual amplitude, dynamic, does not start a gliss, etc.)

        """
        out = []
        last = None
        lastidx = len(self.items) - 1
        for i, item in enumerate(self.items):
            if isinstance(item, Chain):
                item.mergeTiedEvents()
                out.append(item)
                last = None
            elif last is not None and type(last) is type(item):
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

    def convertGlissTargetsToGracenotes(self) -> list[MEvent]:
        """
        Convert gliss. end pitches within events as gracenotes, in place

        Returns:
            the list of newly created gracenotes (an empty list if no changes
            where performed). The returned gracenotes are part of self or
            any corresponding subchain
        """
        items = []
        gracenotes = []
        changed = False
        for item in self.items:
            items.append(item)
            if isinstance(item, Chain):
                gracenotes.extend(item.convertGlissTargetsToGracenotes())
            else:
                assert isinstance(item, MEvent)
                if item.gliss and not isinstance(item.gliss, bool):
                    try:
                        grace = asEvent(item.gliss, dur=0)
                        gracenotes.append(grace)
                        item.gliss = True
                        changed = True
                        if item.symbols:
                            # transfer end spanners to the gracenote
                            for s in item.symbols.copy():
                                if isinstance(s, symbols.Spanner) and s.kind == 'end':
                                    grace.addSymbol(s)
                                    item.symbols.remove(s)
                        items.append(grace)
                    except NotImplementedError:
                        # item does not implement gracenotes, don't do anything
                        pass
        self.items = items
        if changed:
            self._changed()
        return gracenotes

    def updateChildrenOffsets(self) -> None:
        self._update()

    def __contains__(self, item: MObj) -> bool:
        return item in self.items

    def _childOffset(self, child: MObj) -> F:
        """
        Returns the offset of child within this chain

        raises ValueError if self is not a parent of child

        Args:
            child: the object whose offset is to be determined

        Returns:
            The offset of this child within this chain
        """
        if not any(item is child for item in self.items):
            raise ValueError(f"The item {child} is not a child of {self}")

        if child.offset is not None:
            return child.offset

        self._update()
        offset = child._resolvedOffset
        assert offset is not None
        return offset

    @property
    def dur(self) -> F:
        """The duration of this sequence"""
        if not self._modified:
            return self._dur

        if not self.items:
            self._dur = F0
            return self._dur

        self._update()
        return self._dur

    @dur.setter
    def dur(self, value):
        raise AttributeError(f"Duration is readonly for instances of {type(self)}")

    def append(self, item: MEvent) -> None:
        """
        Append an item to this chain

        Args:
            item: the item to add
        """
        item.parent = self
        self.items.append(item)
        self._changed()

    def extend(self, items: Sequence[MEvent | Chain]) -> None:
        """
        Extend self with items

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
        if not self._modified and self._dur > 0:
            return
        self._dur = _stackEvents(self.items, explicitOffsets=False)
        self._resolveGlissandi()
        self._modified = False
        self._absOffset = None
        self._hasRedundantOffsets = True

    def _changed(self) -> None:
        if self._modified:
            return
        self._modified = True
        self._dur = F0
        self._cachedEventsWithOffset = None
        self._absOffset = None
        if self.parent:
            self.parent._childChanged(self)

    def _childChanged(self, child: MObj) -> None:
        if not self._modified:
            self._changed()

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[MEvent | Chain]:
        return iter(self.items)

    @overload
    def __getitem__(self, idx: int) -> MEvent: ...

    @overload
    def __getitem__(self, idx: slice) -> list[MEvent | Chain]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.items[idx]
        else:
            return self.items.__getitem__(idx)

    def _dumpRows(self, indents=0, now=F0, forcetext=False, struct: ScoreStruct|None = None
                  ) -> list[str]:
        fontsize = '85%'
        IND = '  '
        selfstart = f"{float(self.offset):.3g}" if self.offset is not None else 'None'
        namew = max((sum(len(n.name) for n in event.notes) + len(event.notes)
                     for event in self.recurse()
                     if isinstance(event, Chord)),
                    default=10)

        widths = {
            'location': 10,
            'beat': 7,
            'offset': 12,
            'dur': 12,
            'name': namew,
            'gliss': 6,
            'dyn': 5,
            'playargs': 20,
            'info': 20
        }
        T = _util.showT

        if struct is None:
            struct = self.scorestruct() or Workspace.active.scorestruct

        if environment.insideJupyter and not forcetext:
            r = type(self).__name__
            header = (f'<code><span style="font-size: {fontsize}">{IND*indents}<b>{r}</b> - '
                      f'beat: {T(self.absOffset())}, offset: {selfstart}, '
                      f'dur: {T(self.dur)}'
                      )
            if self.label:
                header += f', label: {self.label}'
            if self._postSymbols:
                header += f', postsymbols: {self._postSymbols}'
            header += '</span></code>'
            rows = [header]

            columnparts = [IND*(indents+1)]
            for k, width in widths.items():
                columnparts.append(k.ljust(width))
            columnnames = ''.join(columnparts)
            row = f"<code>{_tools.htmlSpan(columnnames, ':grey1', fontsize=fontsize)}</code>"
            rows.append(row)

            items, itemsdur = self._iterateWithTimes(recurse=False, frame=F0)
            for item, itemoffset, itemdur in items:
                infoparts = []
                assert isinstance(item, (MEvent, Chain))
                if item.label:
                    infoparts.append(f'label: {item.label}')
                if item.properties:
                    infoparts.append(f'properties: {item.properties}')

                if isinstance(item, MEvent):
                    name = item.name
                    if isinstance(item, (Note, Chord)) and item.tied:
                        name += "~"
                    offsetstr = T(item.offset) if item.offset is not None else f'({T(itemoffset)})'
                    offsetstr = offsetstr.ljust(widths['dur'])
                    durstr = T(item.dur).ljust(widths['dur'])
                    measureidx, measurebeat = struct.beatToLocation(now + itemoffset)
                    locationstr = f'{measureidx}:{T(measurebeat)}'.ljust(widths['location'])
                    playargs = 'None' if not item.playargs else ', '.join(f'{k}={v}' for k, v in item.playargs.db.items())
                    if isinstance(item, (Note, Chord)):
                        glissstr = 'F' if not item.gliss else f'T ({item.resolveGliss()})' if isinstance(item.gliss, bool) else str(item.gliss)
                    else:
                        glissstr = '-'
                    rowparts = [IND*(indents+1),
                                locationstr,
                                T(now + itemoffset).ljust(widths['beat']),
                                offsetstr,
                                durstr,
                                name.ljust(widths['name']),
                                glissstr.ljust(widths['gliss']),
                                str(item.dynamic).ljust(widths['dyn']),
                                playargs.ljust(widths['playargs']),
                                ' '.join(infoparts) if infoparts else '-'
                                ]
                    row = f"<code>{_tools.htmlSpan(''.join(rowparts), ':blue1', fontsize=fontsize)}</code>"
                    rows.append(row)
                    if item.symbols:
                        row = f"<code>      {_tools.htmlSpan(str(item.symbols), ':green2', fontsize=fontsize)}</code>"
                        rows.append(row)

                elif isinstance(item, Chain):
                    rows.extend(item._dumpRows(indents=indents+1, now=now+itemoffset, forcetext=forcetext, struct=struct))
            return rows
        else:
            rows = [f"{IND * indents}Chain -- beat: {T(self.absOffset())}, offset: {selfstart}, dur: {T(self.dur)}",
                    f'{IND * (indents + 1)}beat   offset  dur    item']
            items, itemsdur = self._iterateWithTimes(recurse=False, frame=F0)
            for item, itemoffset, itemdur in items:
                if isinstance(item, MEvent):
                    itemoffsetstr = T(item.offset) if item.offset is not None else 'None'
                    rows.append(f'{IND * (indents+1)}'
                                f'{T(now + itemoffset).ljust(7)}'
                                f'{itemoffsetstr.ljust(7)} '
                                f'{T(itemdur).ljust(7)}'
                                f'{item}')
                elif isinstance(item, Chain):
                    rows.extend(item._dumpRows(indents=indents+1, forcetext=forcetext, now=now+itemoffset, struct=struct))
                else:
                    raise TypeError(f"Expected an MEvent or a Chain, got {item}")
            return rows

    def dump(self, indents=0, forcetext=False) -> None:
        """
        Dump this chain, recursively

        Values inside parenthesis are implicit. For example if an object inside
        this chain does not have an explicit .offset, its withExplicitTimes offset will
        be shown within parenthesis

        Args:
            indents: the number of indents to use
            forcetext: if True, force print output instea of html, even when running
                inside jupyter
        """
        self._update()
        rows = self._dumpRows(indents=indents, now=self.offset or F0, forcetext=forcetext)
        if environment.insideJupyter and not forcetext:
            html = '<br>'.join(rows)
            from IPython.display import HTML, display
            display(HTML(html))
        else:
            for row in rows:
                print(row)

    def __repr__(self):
        self._update()
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
        self._update()
        from maelzel.colortheory import safeColors
        itemcolor = safeColors['blue2']
        items = self.items if len(self.items) < 10 else self.items[:10]
        itemstr = ", ".join(f'<span style="color:{itemcolor}">{repr(_)}</span>'
                            for _ in items)
        if len(self.items) >= 10:
            itemstr += ", …"
        cls = self.__class__.__name__
        namedargs = [f'dur={_util.showT(self.dur)}']
        if self.offset:
            namedargs.append(f'offset={_util.showT(self.offset)}')
        info = ', ' + ', '.join(namedargs)
        return f'{cls}([{itemstr}]{info})'

    def removeRedundantOffsets(self) -> None:
        """
        Remove over-specified start times in this Chain (in place)
        """
        # This is the relative position (independent of the chain's start)
        if not self._hasRedundantOffsets and not self._modified:
            return

        self._update()

        _, modified = _removeRedundantOffsets(self.items, frame=F0)
        if self.offset == F0:
            self.offset = None
            modified = True
        if modified:
            self._changed()
            self._hasRedundantOffsets = False

    def asVoice(self, removeOffsets=True) -> Voice:
        """
        Create a Voice as a copy of this Chain

        Args:
            removeOffsets: if True, remove any redundant offsets in the returned voice

        Returns:
            this chain as a Voice
        """
        self._update()
        items = [item.copy() for item in self.items]
        _ = _stackEvents(items, explicitOffsets=True)
        if self.offset and self.offset > F0:
            for item in items:
                assert item.offset is not None
                item.offset += self.offset
        voice = Voice(items, name=self.label)
        if removeOffsets:
            voice.removeRedundantOffsets()
        if self.symbols:
            for symbol in self.symbols:
                voice.addSymbol(symbol)
        if self.playargs:
            voice.playargs = self.playargs.copy()
        if self._scorestruct:
            voice.setScoreStruct(self._scorestruct)
        if self._config:
            for k, v in self._config.items():
                voice.setConfig(k, v)
        return voice

    def _asVoices(self) -> list[Voice]:
        return [self.asVoice()]

    def timeTransform(self, timemap: Callable[[F], F], inplace=False
                      ) -> Self:
        items = []
        for item in self.items:
            items.append(item.timeTransform(timemap, inplace=inplace))
        return self if inplace else self.clone(items=items)

    @classmethod
    def _labelSymbol(cls, label: str, config: CoreConfig | None = None):
        if config is None:
            config = Workspace.active.config
        from maelzel.textstyle import TextStyle
        labelstyle = TextStyle.parse(config['show.labelStyle'])
        return symbols.Text(text=label, fontsize=labelstyle.fontsize, italic=labelstyle.italic, weight="bold" if labelstyle.bold else "normal", color=labelstyle.color)

    def _applyChainSymbols(self):
        for item in self.items:
            if isinstance(item, Chain):
                item._applyChainSymbols()
            elif self.symbols:
                for symbol in self.symbols:
                    if isinstance(symbol, symbols.EventSymbol):
                        if item.properties is None:
                            item.properties = {}
                        item.properties.setdefault('.tempsymbols', []).append(symbol)

    def _collectSubLabels(self, config: CoreConfig) -> Iterator[tuple[F, str]]:
        for item in self.items:
            if isinstance(item, Chain):
                if item.label:
                    yield (item.absOffset(), item.label)
                yield from item._collectSubLabels(config)

    def _flatPostSymbols(self, config: CoreConfig) -> list[tuple[symbols.Symbol, time_t, time_t | None]]:
        postsymbols = self._postSymbols.copy() if self._postSymbols else []
        sublabels = list(self._collectSubLabels(config))
        for offset, label in sublabels:
            postsymbols.append((Chain._labelSymbol(label, config), offset, None))
        postsymbols.sort(key=lambda x: x[1])
        return postsymbols

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig | None = None,
                      parentOffset: F | None = None
                      ) -> list[scoring.Notation]:
        """
        Returns the scoring events corresponding to this object.

        The scoring events returned always have an absolute offset

        Args:
            groupid: if given, all events are given this groupid
            config: the configuration used (None to use the active config)
            parentOffset: if given will override the parent's offset

        Returns:
            the scoring notations representing this object
        """
        if not self.items:
            return []

        if config is None:
            config = Workspace.active.config

        config = self.getConfig(prototype=config) or config
        self._applyChainSymbols()

        allns: list[scoring.Notation] = []
        for event, offset in self.eventsWithOffset():
            allns.extend(event.scoringEvents(groupid=groupid, config=config))

        if len(allns) > 1:
            n0 = allns[0]
            for n1 in allns[1:]:
                if n0.tiedNext and not n1.isRest:
                    n1.tiedPrev = True
                n0 = n1

        if postsymbols := self._flatPostSymbols(config):
            from maelzel.scoring import quantutils
            sco = self.activeScorestruct()
            postsymbols = [(symbol, sco.asBeat(offset), sco.asBeat(end) if end else None)
                            for symbol, offset, end in postsymbols]
            if self.label:
                postsymbols.append((Chain._labelSymbol(self.label, config), self.absOffset(), None))
            postsymbols.sort(key=lambda row: (isinstance(row[0], symbols.Spanner), row[1]))
            symbolsMaxBeat = max(end if end is not None else offset for _, offset, end in postsymbols)

            if symbolsMaxBeat > allns[-1].end:
                allns.append(scoring.Notation.makeRest(symbolsMaxBeat - allns[-1].end))

            allns = quantutils.fillSpan(allns, allns[0].offset, allns[-1].end)
            splitpoints = []
            for _, offset, end in postsymbols:
                splitpoints.append(offset)
                if end is not None:
                    splitpoints.append(end)
            splitpoints.sort()
            allns = scoring.Notation.splitNotationsAtOffsets(allns, splitpoints, nomerge=True)
            for symbol, offset, end in postsymbols:
                if isinstance(symbol, symbols.Spanner):
                    assert end is not None
                    startobj = next((n for n in allns if n.offset == offset), None)
                    endobj = next((n for n in allns if n.end == end), None)
                    if not symbol.appliesToRests and (startobj is None or startobj.isRest or endobj is None or endobj.isRest):
                        logger.info(f"{symbol} can't be applied, no notations found for {offset} or {end}, notations={allns}")
                        continue
                    if startobj is None:
                        startobj = quantutils.insertRestAt(offset, allns)
                    endobj = next((n for n in allns if n.end == end), None)
                    if endobj is None:
                        endobj = quantutils.insertRestEndingAt(end, allns)
                        assert endobj is not None
                        allns.append(endobj)
                    symbol.applyToPair(startobj, endobj)

                elif isinstance(symbol, symbols.EventSymbol):
                    assert end is None
                    nindex = quantutils.notationAtOffset(allns, offset, exact=True)
                    if nindex is not None:
                        n = allns[nindex]
                        symbol.applyToNotation(n, parent=None)
                    
                    elif symbol.appliesToRests:
                        # we need to create rest starting from this offset and ending at the next notation
                        rest = quantutils.insertRestAt(offset, allns)
                        symbol.applyToNotation(rest, parent=None)
                    else:
                        raise RuntimeError(f"Could not apply post symbol {symbol} at {offset=}, no"
                                           f" notation found.\nNotations: {allns}")
                else:
                    raise TypeError(f"Symbol {symbol} not supported as postsymbol")
        return allns

    def _solveOrfanHairpins(self, currentDynamic='mf'):
        lastHairpin: symbols.Hairpin | None = None
        for n in self.recurse():
            if not isinstance(n, (Chord, Note)):
                continue
            if n.dynamic and n.dynamic != currentDynamic:
                if lastHairpin:
                    n.addSpanner(lastHairpin.makePartnerSpanner())
                    lastHairpin = None
                currentDynamic = n.dynamic

            if n.symbols:
                for s in n.symbols:
                    if isinstance(s, symbols.Hairpin) and s.kind == 'start' and not s.partner:
                        lastHairpin = s

    def _scoringParts(self,
                      config: CoreConfig,
                      maxstaves=0,
                      name='',
                      shortname='',
                      groupParts=False,
                      addQuantizationProfile=False) -> list[scoring.core.UnquantizedPart]:
        self._update()
        notations = self.scoringEvents(config=config)
        if not notations:
            return []
        scoring.core.resolveOffsets(notations)
        config = self.getConfig(config) or config
        maxstaves = maxstaves or config['show.voiceMaxStaves']

        # Until we support cross staffs, a chain/voice with spanners spanning
        # across multiple staffs is prone to confussion.
        if maxstaves > 1 and any(n.spanners for n in notations):
            logger.info("Spanners across multiple staves are not supported")

        if maxstaves == 1:
            parts = [scoring.core.UnquantizedPart(notations, name=name, shortname=shortname)]
        else:
            parts = scoring.core.distributeNotationsByClef(notations, name=name, shortname=shortname,
                                                           maxstaves=maxstaves)
            parts.reverse()
            if len(parts) > 1 and groupParts:
                scoring.core.UnquantizedPart.groupParts(parts, name=name, shortname=shortname)

        if addQuantizationProfile:
            quantProfile = config.makeQuantizationProfile()
            for part in parts:
                part.quantProfile = quantProfile
        return parts

    def scoringParts(self,
                     config: CoreConfig | None = None
                     ) -> list[scoring.core.UnquantizedPart]:
        config, iscustomized = self._resolveConfig(config)
        parts = self._scoringParts(
            config=config,
            maxstaves=config["show.voiceMaxStaves"],
            name=self.label,
            addQuantizationProfile=iscustomized)
        return parts

    def quantizePitch(self, step=0.):
        if step < 0:
            raise ValueError(f"Step should be possitive, got {step}")
        items = [i.quantizePitch(step) for i in self.items]
        return self.clone(items=items)

    def setItems(self, items: list[MEvent|Chain]) -> None:
        """
        Set the items of this chain/voice, inplace

        Args:
            items: the new items

        Setting the ``.items`` attribute directly will result
        in errors, since the given items need to be modified in order
        to have their ``.parent`` attribute set and the cache for this
        container and its parents, if any, need to be invalidated
        """
        for item in items:
            item.parent = self
        self.items = items
        self._changed()

    def timeShift(self, offset: time_t) -> Self:
        self._update()
        if offset == 0:
            return self
        reloffset = self.relOffset()
        if offset > 0:
            return self.clone(offset=reloffset + offset)

        if reloffset + offset >= 0:
            return self.clone(offset=reloffset + offset)

        out = self.copy()
        out.timeShiftInPlace(offset)
        return out

    def timeShiftInPlace(self, offset: time_t) -> None:
        """
        Shift the time of this by the given offset (inplace)

        Args:
            offset: the time delta (in quarterNotes)
        """

        offset = asF(offset)
        if offset == 0:
            return

        self._update()
        reloffset = self.relOffset()
        if offset > 0:
            self.offset = reloffset + offset
            self._changed()
            return

        # Negative offset. First decrease the offset to the first event.
        firstoffset = self.firstOffset()
        assert firstoffset is not None
        newfirstoffset = max(F0, firstoffset + offset)
        itemshift = newfirstoffset - firstoffset
        for item in self.items:
            item.timeShiftInPlace(itemshift)

        # Remaining shift
        restshift = offset + firstoffset - newfirstoffset
        if restshift:
            newreloffset = reloffset + restshift
            if newreloffset < 0:
                raise ValueError(f"The shift would result in negative time. "
                                 f"Resulting offset: {newreloffset}, current "
                                 f"offset: {reloffset}, self: {self}")
            if not self.parent:
                self.offset = newreloffset
            else:
                previtem = self.parent.previousItem(self)
                if previtem is None:
                    # No previous item, so can just adjust own offset
                    self.offset = newreloffset
                else:
                    assert isinstance(previtem, (MEvent, Chain))
                    if newreloffset < previtem.relEnd():
                        raise ValueError("The shift would result in negative time")
                    self.offset = newreloffset
        self._changed()

    def firstOffset(self) -> F | None:
        """
        The offset (relative to the start of this chain) of the first event

        The offset returned might refer to an item of this chain or any
        subchain, recursively.

        Returns:
            the offset of the first event, relative to self, None if empty.
        """
        event = self.firstEvent()
        return None if not event else event.absOffset() - self.absOffset()

    def pitchTransform(self, pitchmap: Callable[[float], float]) -> Self:
        newitems = [item.pitchTransform(pitchmap) for item in self.items]
        return self.clone(items=newitems)

    def recurse(self, reverse=False) -> Iterator[MEvent]:
        """
        Yields all events (Notes/Chords) in this chain, recursively

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

        .. seealso:: :meth:`Chain.eventsWithOffset`, :meth:`Chain.itemsWithOffset`, :meth:`Chain.flatEvents`
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

    def eventsWithOffset(self,
                         start: beat_t | None = None,
                         end: beat_t | None = None,
                         partial=True) -> list[tuple[MEvent, F]]:
        """
        Recurse the events in self and resolves each event's offset

        Args:
            start: absolute start beat/location. Filters the returned
                event pairs to events within this time range
            end: absolute end beat/location. Filters the returned event
                pairs to events within the given range
            partial: only used if either start or end are given, this controls
                how events are matched. If True, events only need to be
                partially defined within the time range. Otherwise, they need
                to be fully included within the time range

        Returns:
            a list of pairs, where each pair has the form (event, offset), the offset being
             the **absolute** offset of the event. Event themselves are not modified

        .. seealso:: :meth:`Chain.recurse`, :meth:`Chain.itemsWithOffset`

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> chain = Chain([
            ... "4C:0.5",
            ... "4D",
            ... Chain(["4E:0.5"], offset=2)
            ... ], offset=1)
            >>> chain.eventsWithOffset()
            [(4C:0.5♩, Fraction(1, 1)), (4D:1♩, Fraction(3, 2)), (4E:0.5♩, Fraction(3, 1))]

        """
        self._update()
        if self._cachedEventsWithOffset is not None:
            eventpairs = self._cachedEventsWithOffset
        else:
            eventpairs, totaldur = self._eventsWithOffset(frame=self.absOffset())
            self._dur = totaldur
            self._cachedEventsWithOffset = eventpairs

        if start is not None or end is not None:
            struct = self.activeScorestruct()
            start = struct.asBeat(start) if start else F0
            end = struct.asBeat(end) if end else F(sys.maxsize)
            eventpairs = _eventPairsBetween(eventpairs,
                                            start=start,
                                            end=end,
                                            partial=partial)
        return eventpairs

    def itemsWithOffset(self) -> Iterator[tuple[MEvent|Chain, F]]:
        """
        Iterate over the items of this chain with their absolute offset

        Returns:
            an iterator over tuple[item, offset], where an item can be
            an event or a Chain, and offset is the absolute offset

        .. seealso:: :meth:`Chain.eventsWithOffset`
        """
        self._update()
        parentOffset = self.absOffset()
        for item in self.items:
            yield item, item.relOffset() + parentOffset

    def _eventsWithOffset(self,
                          frame: F
                          ) -> tuple[list[tuple[MEvent, F]], F]:
        events = []
        now = frame
        for item in self.items:
            if item.offset:
                now = frame + item.offset
            if isinstance(item, MEvent):
                events.append((item, now))
                now += item.dur
            else:
                subitems, subdur = item._eventsWithOffset(frame=now)
                events.extend(subitems)
                now += subdur
        return events, now - frame

    def _iterateWithTimes(self,
                          recurse: bool,
                          frame: F,
                          ) -> tuple[list[tuple[MEvent | list, F, F]], F]:
        """
        For each item returns a tuple (item, offset, dur)

        Each event is represented as a tuple (event, offset, dur), a chain
        is represented as a list of such tuples

        Args:
            recurse: if True, traverse any subchain
            frame: the frame of reference

        Returns:
            a tuple (eventtuples, duration) where eventtuples is a list of
            tuples (event, offset, dur). If recurse is True,
            any subchain is returned as a list of eventtuples. Otherwise,
            a flat list is returned. In each eventtuple, the offset is relative
            to the first frame passed, so if the first offset was 0
            the offsets will hold the absolute offset of each event. Duration
            is the total duration of the items in
            the chain (not including its own offset)

        """
        assert isinstance(frame, F)
        now = frame
        out = []
        for i, item in enumerate(self.items):
            if item.offset is not None:
                t = frame + item.offset
                assert t >= now, f"Invalid time: {now=}, {t=}, {frame=}, {item.offset=}"
                now = t
            if isinstance(item, MEvent):
                dur = item.dur
                out.append((item, now, dur))
                item._resolvedOffset = now - frame
                # if i == 0 and self.label:
                #     item.setProperty('.chainlabel', self.label)
                now += dur
            else:
                # a Chain
                if recurse:
                    subitems, subdur = item._iterateWithTimes(frame=now, recurse=True)
                    item._dur = subdur
                    item._resolvedOffset = now - frame
                    out.append((subitems, now, subdur))
                else:
                    subdur = item.dur
                    out.append((item, now, subdur))
                now += subdur
        return out, now - frame

    def addSpanner(self,
                   spanner: str | symbols.Spanner,
                   start: location_t | MEvent | None = None,
                   end: location_t | MEvent | None = None,
                   post=False
                   ) -> Self:
        """
        Adds a spanner symbol across this object

        A spanner is a slur, line or any other symbol attached to two or more
        objects. A spanner always has a start and an end.

        Args:
            spanner: a Spanner object or a spanner description (one of 'slur', '<', '>',
                'trill', 'bracket', etc. - see :func:`maelzel.core.symbols.makeSpanner`
                When passing a string description, prepend it with '~' to create an end spanner
            start: start location or event
            end: end location or event

        Returns:
            self (allows to chain calls)

        Example
        ~~~~~~~

            >>> chain = Chain([
            ... Note("4C", 1),
            ... Note("4D", 0.5),
            ... Note("4E")   # This ends the hairpin spanner
            ... ])
            >>> chain.addSpanner('slur')

        This is the same as:

            >>> chain[0].addSpanner('slur', chain[-1])

        """
        if isinstance(spanner, str):
            spanner = symbols.makeSpanner(spanner)
        if start is None:
            start = self.firstEvent(acceptRest=spanner.appliesToRests)
            if start is None:
                raise RuntimeError(f"This {type(self)} has no event to apply {spanner} to")
        if end is None:
            end = self.lastEvent(acceptRest=spanner.appliesToRests)
            if end is None:
                raise RuntimeError(f"This {type(self)} has no event to apply {spanner} to")
        if not post and isinstance(start, MEvent) and isinstance(end, MEvent):
            start.addSpanner(spanner=spanner, endobj=end)
        else:
            startloc = start.absOffset() if isinstance(start, MEvent) else start
            endloc = end.absOffset() if isinstance(end, MEvent) else end
            self._postSymbols.append((spanner, startloc, endloc))

        return self

    def addSymbolAt(self,
                    symbol: symbols.EventSymbol | str,
                    offset: beat_t,
                    post=False,
                    skipGrace=False
                    ) -> Self:
        """
        Adds a symbol at the given location

        If there is no event starting at the given location, the quantized part is split at
        the location when rendering and the symbol is added to the event. This allows to add
        'soft' symbols at any location without the need to modify the events themselves.
        If there actually is an event **starting** at the given offset, the symbol
        is added to the event directly.

        Args:
            symbol: the symbol to add
            offset: the location to add the symbol as an absolute beat or a location (measureindex, measureoffset)
            post: if True, add this symbol at quantization, not to the event itself. If there
                is no event starting at the given offset, post is enforced even if false
            skipGrace: If there are multiple notes found at the location (this
                happens only if a note starts at the location, preceded
                by one or many gracenotes), a True value would apply
                the symbol to the "real" event, skipping the gracenotes

        Returns:
            self

        Example
        -------

            >>> chain = Chain([
            ... Note("4C", 2),
            ... Note("4E", 1)
            ])
            >>> chain.addSymbolAt(symbols.Fermata(), 1)


        .. seealso:: :meth:`Chain.addBreak`
        """
        absoffset = self._locationToAbsOffset(offset)
        events = self.eventsAt(absoffset)
        if not events:
            event = None
            post = True
        elif skipGrace:
            event = next((ev for ev in events if not ev.isGrace()), None)
        else:
            event = events[0]
        if isinstance(symbol, str):
            _symbol = symbols.makeKnownSymbol(symbol)
            if _symbol is None:
                raise ValueError(f"Could not create a symbol from {symbol}")
            symbol = _symbol  # type: ignore
        if not post and event and event.absOffset() == absoffset:
            event.addSymbol(symbol)
        else:
            self._postSymbols.append((symbol, absoffset, None))
        return self

    def addSymbol(self, *args, **kws) -> Self:
        symbol = symbols.parseAddSymbol(args, kws)
        if not isinstance(symbol, symbols.PartMixin):
            partsymbols = [cls.__name__ for cls in symbols._voiceSymbols]
            raise TypeError(f"Cannot add {symbol} to a {type(self)}. Possible symbols: {partsymbols}")
        assert isinstance(symbol, symbols.Symbol)
        self._addSymbol(symbol)
        return self

    def firstEvent(self, acceptRest=True) -> MEvent | None:
        """
        The first event in this chain, recursively

        .. seealso:: `:meth:`Chain.recurse`

        Example
        ~~~~~~~

            >>> chain = Chain([
            ...     Chain("4E:0.5", offset=2),
            ...     "4C:0.5",
            ...     "4D"])
            >>> chain.firstEvent()
            4E:0.5♩

        The returned event is the actual event enclosed in this chain/voice.
        As seen in the example, it can be an event enclosed in a subchain
        """
        if acceptRest:
            return next(self.recurse(), None)
        return next((ev for ev in self.recurse() if not ev.isRest()), None)

    def lastEvent(self, acceptRest=True) -> MEvent | None:
        """
        The last event in this chain, recursively

        .. seealso:: `:meth:`Chain.recurse`

        Example
        ~~~~~~~

            >>> chain = Chain([
            ...     "4C:0.5",
            ...     "4D",
            ...     Chain(["4E:0.5", "4F:0.5"])
            ...     ])
            >>> chain.lastEvent()
            4F:0.5♩

        The returned event is the actual event enclosed in this chain/voice.
        As seen in the example, it can be an event enclosed in a subchain

        """
        if acceptRest:
            return next(self.recurse(reverse=True))
        return next((ev for ev in self.recurse(reverse=True) if not ev.isRest()), None)

    def eventAt(self, location: beat_t, split=False, margin: F = F0, start=False
                ) -> MEvent | None:
        """
        The event present at the given location

        .. note::

            If there are multiple events at the given location (gracenotes have a duration
            if 0 and thus can share a location with other gracenotes and with an event
            starting at that location) only the first event will be returned.
            Use :meth:`Chain.eventsAt` to return all events at a given location

        Args:
            location: the beat or a tuple (measureindex, beatoffset). If a beat is given,
                it is interpreted as an absoute offset
            split: if the offset lies within an event, splits the event at the given offset,
                returns the right part of the event (the part starting at the offset), as
                a tied event. If the returned event is not modified (no symbol is added or
                any other property is changed) it might be remerged when shown as notation.
                To prevent merging without any other visible side-effects you can add
                a NoMerge symbol to the returned event
            margin: if given, the first event within location and location+margin will be
                returned
            start: if True, an event will be returned only if it starts at the given offset

        Returns:
            the event present at the given location, or None if no event was found. An
            explicit rest will be returned if found but empty space will return None. If there
            are multiple events at the given location (due to gracenotes having 0 duration),
            the first event will be returned.

        .. seealso:: :meth:`Chain.eventsBetween`, :meth:`Chain.eventsAt`
        """
        eps = margin if margin else F(1, 10000)
        start = self._locationToAbsOffset(location)
        end = start + eps
        events = self.eventsBetween(start, end)
        if not events:
            return None
        if not start:
            event = events[0]
        else:
            event = next((ev for ev in events if ev.absOffset() == start), None)
            if event is None:
                return None
        if split:
            eventoffset = event.absOffset()
            if eventoffset < start < eventoffset + event.dur:
                event = self.splitAt(start, beambreak=False, nomerge=False)
        return event

    def _locationToAbsOffset(self, location: beat_t) -> F:
        if isinstance(location, tuple):
            struct = self.scorestruct() or Workspace.active.scorestruct
            measidx, beat = location
            offset = struct.locationToBeat(measidx, beat)
        else:
            offset = asF(location)  # type: ignore
        return offset

    def eventsAt(self, location: beat_t, start=False) -> list[MEvent]:
        """
        Returns all events present at the given location

        Args:
            location: the beat or a tuple (measureindex, beatoffset). If a beat is given,
                it is interpreted as an absoute offset
            start: if True, only events **starting** at the given location are returned

        Returns:
            the events present or starting at the given location
        """
        offset = self._locationToAbsOffset(location)
        EPS = F(1, 100000)
        if start:
            events = [ev for ev, offset in self.eventsWithOffset(start=offset, end=offset+EPS)
                      if offset == start]
        else:
            events = [ev for ev, _ in self.eventsWithOffset(start=offset, end=offset + EPS)]
        return events

    def eventsBetween(self,
                      start: beat_t,
                      end: beat_t,
                      partial=True,
                      ) -> list[MEvent]:
        """
        Events between the given time range

        If ``partial`` is false, only events which lie completey within
        the given range are included. Gracenotes at the edges are always
        included.

        .. note::

            The returned events are the actual events in this
            Chain or subchains: they are NOT copies. If these events
            do not have an `.offset` set or they are nested, their
            resulting offset when used parentless will differ. To force every
            event having an explicit offset use :meth:`.stack() <maelzel.core.chain.Chain.stack>`

        Args:
            start: **absolute** start location (a beat or a score location)
            end: **absolute** end location (a beat or score location)
            partial: include also events wich are partially included within
                the given time range

        Returns:
            a list of the events within the given time range. The actual events
            are returned, so modifying the returned events will modify
            self

        .. seealso:: :meth:`Chain.eventsWithOffset`, :meth:`Chain.itemsBetween`
        """
        eventpairs = self.eventsWithOffset(start=start, end=end, partial=partial)
        return [event for event, offset in eventpairs]

    def eventAfter(self, offset: beat_t) -> MEvent | None:
        """
        First event starting at or after offset

        If you want events strictly after offset, add an epsilon to offset

        Args:
            offset: absolute start location (a beat or score location)

        Returns:
            the first event starting at or after offset, if exists, None otherwise
        """
        for ev, evoffset in self.eventsWithOffset(start=offset):
            if evoffset >= offset:
                return ev
        return None

    def eventBefore(self, offset: beat_t) -> MEvent | None:
        """
        Last event ending before or at the offset

        If you want an event ending strictly before the given offset, substract
        an epsilon to the offset

        Args:
            offset: absolute end location (a beat or score location)

        Returns:
            the last event ending before or at the offset, if exists, None otherwise

        """
        # first(ev for ev in self.recurse(reverse=True) if ev.end <= offset)
        pairs = self.eventsWithOffset(end=offset+F(1, 100))
        for ev, evoffset in reversed(pairs):
            if evoffset + ev.dur <= offset:
                return ev
        return None

    def itemsBetween(self,
                     start: beat_t,
                     end: beat_t,
                     partial=True
                     ) -> list[MEvent | Chain]:
        """
        Items between the given time range

        An item is either an event (Note, Chord, Clip, etc.) or another Chain.

        If ``partial`` is false, only items which lie completey within
        the given range are included. Gracenotes at the edges are always
        included

        Args:
            start: absolute start location (a beat or a score location)
            end: absolute end location (a beat or score location)
            partial: include also events wich are partially included within
                the given time range

        Returns:
            a list of the items within the given time range. The actual items
            are returned

        .. seealso:: :meth:`Chain.itemsWithOffset`, :meth:`Chain.eventsBetween`

        """
        sco = self.scorestruct() or Workspace.active.scorestruct
        startbeat = sco.asBeat(start)
        endbeat = sco.asBeat(end)
        out = []
        if partial:
            for item, offset in self.itemsWithOffset():
                if offset > endbeat or (offset == endbeat and item.dur > 0):
                    break
                if offset + item.dur >= startbeat:
                    out.append(item)
        else:
            for item, offset in self.eventsWithOffset():
                if offset > endbeat:
                    break
                if startbeat <= offset and offset + item.dur <= endbeat:
                    out.append(item)
        return out

    def splitEventsAtMeasures(self,
                              scorestruct: ScoreStruct | None = None,
                              startindex=0,
                              stopindex=0
                              ) -> None:
        """
        Splits items in self at measure offsets, **inplace** (recursively)

        After this method is called, no event extends for longer than a measure,
        as defined in the given scorestruct or the active scorestruct.

        .. note::

            To avoid modifying self, create a copy first:
            ``newchain = self.copy(); newchain.splitEventsAtMeasure(...)``

        Args:
            scorestruct: if given, overrides any active scorestruct for this object
            startindex: the first measure index to use
            stopindex: the last measure index to use. 0=len(measures). The stopindex is not
                included (similar to how python's builtin `range` behaves`

        .. seealso:: :meth:`Chain.splitAt`
        """
        if scorestruct is None:
            scorestruct = self.activeScorestruct()
        else:
            if self.scorestruct():
                clsname = type(self).__name__
                logger.warning(f"This {clsname} has already an active ScoreStruct "
                               f"via its parent. "
                               f"Passing an ad-hoc scorestruct might cause problems...")
        offsets = scorestruct.measureOffsets(startIndex=startindex, stopIndex=stopindex)
        self.splitEventsAtOffsets(offsets, tie=True)

    def splitAt(self,
                location: beat_t,
                tie=True,
                beambreak=False,
                nomerge=False,
                ) -> MEvent | None:
        """
        Split any event present at the given absolute offset (in place)

        The parts resulting from the split operation will be part of this chain/voice.

        To split at a relative offset, substract the absolute offset of this Chain
        from the given offset

        Args:
            location: the absolute offset to split at, or a score location (measureindex, measureoffset)
            tie: tie the parts of an event together if the split intersects an event
            beambreak: if True, add a BeamBreak symbol to the given event
            nomerge: if True, enforce that the items splitted cannot be
                merged at a later stage (they are marked with a NoMerge symbol)

        Returns:
            Returns the event starting at the given offset, or None if no event found at
            the given offset. The returned event can be a part of a previous event spanning
            across the given offset, or an event starting exactly at the given offset.

        """
        absoffset = asF(location) if not isinstance(location, tuple) else self.activeScorestruct().locationToBeat(*location)  # type: ignore
        self.splitEventsAtOffsets([absoffset], tie=tie)
        ev = self.eventAt(absoffset)
        if not ev:
            return None
        assert ev.absOffset() == absoffset, f"Failed to split correctly? {ev=}, event offset: {ev.absOffset()}, offset should be {absoffset}"
        if beambreak:
            ev.addSymbol(symbols.BeamBreak())
        if nomerge:
            ev.addSymbol(symbols.NoMerge())
        return ev

    def splitEventsAtOffsets(self,
                             offsets: Sequence[beat_t],
                             tie=True,
                             nomerge=False
                             ) -> None:
        """
        Splits events in self at the given offsets, **inplace** (recursively)

        The offsets are absolute. Split events are by default tied together.
        This method is useful for the case where a part of an event needs
        to be adressed in some way. For example, a symbol needs to be
        added to a part of a note (a crescendo hairpin which starts in the
        middle of an event).

        Args:
            offsets: the offsets to split items at (either absolute offsets or
                score locations as tuple (measureindex, measureoffset)
            tie: if True, parts of an item are tied together
            nomerge: add a break to prevent events from being
                merged
        """
        if not offsets:
            raise ValueError("No locations given")
        items = []
        sco = self.activeScorestruct()
        absoffsets = [sco.asBeat(offset) for offset in offsets]
        for item, offset in self.itemsWithOffset():
            if isinstance(item, Chain):
                item.splitEventsAtOffsets(absoffsets, tie=tie, nomerge=nomerge)
                items.append(item)
            else:
                parts = item._splitAtOffsets(absoffsets, tie=tie, nomerge=nomerge)
                for part in parts:
                    part.parent = self
                items.extend(parts)
        self.items = items
        self._changed()

    def cycle(self, totaldur: F, crop=False) -> Self:
        """
        Cycle over the items of self for the given total duration

        Args:
            totaldur: the total duration of the resulting sequence
            crop: if True, crop last item if it exceeds the given
                total duration

        Returns:
            a copy of self representing cycles of its items

        """
        filled = self.copy()
        filled.fillGaps()
        filled.removeRedundantOffsets()
        flatitems = list(filled.recurse())
        items: list[MEvent] = []
        accum = F0
        for item in itertools.cycle(flatitems):
            items.append(item.copy())
            accum += item.dur
            if accum >= totaldur:
                break
        if crop and accum > totaldur:
            diff = accum - totaldur
            lastitem = items[-1]
            assert diff < lastitem.dur
            lastitem = lastitem.clone(dur=lastitem.dur - diff)
            items[-1] = lastitem
        return self.clone(items=items)

    def matchOrfanSpanners(self, removeUnmatched=False) -> None:
        """
        Match unmatched spanners

        When adding spanners to objects, it is possible to create a spanner
        without a partner spanner. As long as there are as many start spanners
        as end spanners for a specific spanner class, these "orfan" spanners
        are matched. This method makes the matches explicit, as if they had
        been created with a partner spanner.

        Args:
            removeUnmatched: if True, any spanners which cannot be matched will
                be removed
        """
        unmatched: list[symbols.Spanner] = []
        for event in self.recurse():
            if event.symbols:
                for symbol in event.symbols:
                    if isinstance(symbol, symbols.Spanner) and symbol.partner is None:
                        unmatched.append(symbol)
        if not unmatched:
            return
        # sort by class
        byclass: dict[type, list[symbols.Spanner]] = {}
        for spanner in unmatched:
            byclass.setdefault(type(spanner), []).append(spanner)
        for cls, spanners in byclass.items():
            stack: list[symbols.Spanner] = []
            for spanner in spanners:
                if spanner.kind == 'start':
                    stack.append(spanner)
                else:
                    assert spanner.kind == 'end'
                    if stack:
                        startspanner = stack.pop()
                        startspanner.setPartner(spanner)
                    elif removeUnmatched:
                        assert spanner.anchor is not None
                        obj = spanner.anchor
                        if obj is None:
                            logger.error(f"The spanner has no anchor ({spanner=})")
                        elif obj.symbols is None:
                            logger.error(f"Invalid spanner anchor, {spanner=}, anchor={obj}")
                        else:
                            logger.debug(f"Removing spanner {spanner} from {obj}")
                            obj.symbols.remove(spanner)
                            spanner._anchor = None

    def remap(self, deststruct: ScoreStruct, sourcestruct: ScoreStruct | None = None,
              setStruct=True
              ) -> Self:
        """
        Creates a clone, remapping times from source scorestruct to destination scorestruct

        The absolute time remains the same

        Args:
            deststruct: the destination scorestruct
            sourcestruct: the source scorestructure, or None to use the resolved scoresturct
            setStruct: if True, explicitely sets deststruct as the score structure for
                this chain/voice

        Returns:
            a clone of self remapped to the destination scorestruct

        """
        remappedEvents = [ev.remap(deststruct, sourcestruct=sourcestruct or self.activeScorestruct())
                          for ev in self]
        out = self.clone(items=remappedEvents)
        if setStruct:
            out.setScoreStruct(deststruct)
        return out

    def automate(self,
                 param: str,
                 breakpoints: list[tuple[time_t|location_t, float]] | list[num_t],
                 relative=True,
                 interpolation='linear'
                 ) -> None:
        if self.playargs is None:
            self.playargs = PlayArgs()
        self.playargs.addAutomation(param=param, breakpoints=breakpoints,
                                    interpolation=interpolation, relative=relative)

    def absorbInitialOffset(self, removeRedundantOffsets=True):
        """
        Moves the offset of the first event to the offset of the chain itself

        Args:
            removeRedundantOffsets: remove redundant offsets.

        Example
        ~~~~~~~

        Notice how the offset of the first note is now None and the chain
        itself has an offset of 0.5

            >>> ch = Chain([
            ...     "4C:1:offset=0.5",
            ...     "4E:1",
            ...     "4G:1"
            ... ])
            >>> ch.dump()
            Chain - beat: 0, offset: None, dur: 3.5
            location  beat   offset      dur         name
            0:0.5     0.5    0.5         1           4C
            0:1.5     1.5    (1.5)       1           4E
            0:2.5     2.5    (2.5)       1           4G
            >>> ch._absorbInternalOffset()
            >>> ch.dump()
            Chain - beat: 1/2, offset: 0.5, dur: 3
            location  beat   offset      dur         name
            0:0.5     0.5    (0)         1           4C
            0:1.5     1.5    (1)         1           4E
            0:2.5     2.5    (2)         1           4G

        """
        firstoffset = self.firstOffset()
        if firstoffset is not None and firstoffset > 0:
            self._update()
            for item in self.items:
                item.timeShiftInPlace(-firstoffset)
            self.offset = self.relOffset() + firstoffset
            if removeRedundantOffsets:
                self.removeRedundantOffsets()
            self._changed()

    def _cropped(self, startbeat: F, endbeat: F, absorbOffset=False
                 ) -> Self:
        items = []
        # absoffset = self.absOffset()
        for item, offset in self.itemsWithOffset():
            if offset > endbeat or (offset == endbeat and item.dur > 0):
                break

            if item.dur == 0 and startbeat <= offset:
                items.append(item.clone(offset=offset - startbeat))
            elif offset + item.dur > startbeat:
                # Add a cropped part or the entire item?
                if startbeat <= offset and offset + item.dur <= endbeat:
                    items.append(item.clone(offset=offset - startbeat))
                else:
                    if isinstance(item, MEvent):
                        item2 = item.cropped(startbeat, endbeat)
                        items.append(item2.clone(offset=item2.absOffset() - startbeat))
                    else:
                        # TODO: combine these two operations, if needed
                        subchain = item._cropped(startbeat, endbeat, absorbOffset=True)
                        items.append(subchain.clone(offset=subchain.absOffset() - startbeat))
        out = self.clone(items=items, offset=startbeat)
        if absorbOffset:
            out.absorbInitialOffset()
        return out

    def addBreak(self, location: F | tuple[int, F]) -> None:
        """
        Adds a symbolic break at the given location.

        This only modifies the representation as notation, it does not
        split any note/chord within this Chain. To actually split any
        item at the given location, use :meth:`Chain.splitAt`

        Args:
            location: the absolute location to break the beam (a beat or a tuple (measureidx, beat))

        """
        self.addSymbolAt(offset=location, symbol=symbols.BeamBreak(), post=True)

    def cropped(self, start: beat_t, end: beat_t) -> Self | None:
        """
        Returns a copy of this chain, cropped to the given beat range

        Returns None if there are no events in this chain within
        the given time range

        Args:
            start: absolute start of the beat range
            end: absolute end of the beat range

        Returns:
            a Chain cropped at the given beat range
        """
        sco = self.scorestruct() or Workspace.active.scorestruct
        startbeat = sco.asBeat(start)
        endbeat = sco.asBeat(end)
        cropped = self._cropped(startbeat=startbeat, endbeat=endbeat)
        if not cropped.items:
            return None
        if any(item.offset is None for item in self.items):
            cropped.removeRedundantOffsets()
        return cropped


class PartGroup:
    """
    This class represents a group of parts

    It is used to indicate that a group of parts are to be notated
    within a staff group, sharing a name/shortname if given. This is
    usefull for things like piano scores, for example

    A PartGroup is immutable

    Args:
        parts: the parts inside this group
        name: the name of the group
        shortname: a shortname to use in systems other than the first
        showPartNames: if True, the name of each part will still be shown in notation.
            Otherwise, it is hidden and only the group name appears
    """
    def __init__(self, parts: list[Voice], name='', shortname='', showPartNames=False):
        for part in parts:
            part._group = self

        self.parts = parts
        """The parts in this group"""

        self.name = name
        """The name of the group"""

        self.shortname = shortname
        """A short name for the group"""

        self.groupid = scoring.core.makeGroupId()
        """A group ID"""

        self.showPartNames = showPartNames
        """Show the names of the individual parts?"""

    def __hash__(self) -> int:
        partshash = hash(tuple(id(part) for part in self.parts))
        return hash((self.name, len(self.parts), partshash))


class Voice(Chain):
    """
    A Voice is a sequence of non-overlapping objects.

    It is **very** similar to a Chain, the only difference being that its offset
    is always 0.


    Voice vs Chain
    ~~~~~~~~~~~~~~

    * A Voice can contain a Chain, but not vice versa.
    * A Voice does not have a time offset, its offset is always 0.

    Args:
        items: the items in this voice. Items can also be added later via :meth:`Voice.append`
        name: the name of this voice. This will be interpreted as the staff name
            when shown as notation
        shortname: optionally a shortname can be given, it will be used for subsequent
            systems when shown as notation
        maxstaves: if given, a max. number of staves to explode this voice when shown
            as notation. If not given the config key 'show.voiceMaxStaves' is used
    """

    def __init__(self,
                 items: Sequence[MEvent | str | Chain] | Chain | str = (),
                 name='',
                 shortname='',
                 maxstaves=0,
                 minstaves=1
                 ):
        if isinstance(items, Chain):
            chain = items
            if chain.offset and chain.offset > 0:
                events = chain.timeShift(chain.offset).items
            else:
                events = chain.items
        else:
            events = items

        super().__init__(items=events, offset=F0)
        self.name = name
        """The name of this voice/staff"""

        self.shortname = shortname
        """A shortname to display as abbreviation after the first system"""

        self._config: dict[str, Any] = {}
        """Any key set here will override keys from the coreconfig for rendering
        Any key in CoreConfig is supported"""

        self._group: PartGroup | None = None
        """A part group is created via Score.makeGroup"""

        if maxstaves:
            self.configNotation(maxStaves=maxstaves)

    def __repr__(self):
        if len(self.items) < 10:
            itemstr = ", ".join(repr(_) for _ in self.items)
        else:
            itemstr = ", ".join(repr(_) for _ in self.items[:10]) + ", …"
        cls = self.__class__.__name__
        namedargs = []
        if namedargs:
            info = ', ' + ', '.join(namedargs)
        else:
            info = ''
        return f'{cls}([{itemstr}]{info})'

    def __hash__(self):
        superhash = super().__hash__()
        return hash((superhash, self.name, self.shortname, id(self._group)))

    def _copyAttributesTo(self, other: Self) -> None:
        super()._copyAttributesTo(other)
        if self._config:
            other._config = self._config.copy()
        if self._scorestruct:
            other.setScoreStruct(self._scorestruct)
            # other._scorestruct = self._scorestruct

    def __copy__(self) -> Self:
        # always a deep copy
        voice = self.__class__(name=self.name,
                               shortname=self.shortname)
        voice.items = [item.copy() for item in self.items]
        self._copyAttributesTo(voice)
        return voice

    def __deepcopy__(self, memodict={}) -> Self:
        return self.__copy__()

    @property
    def group(self) -> PartGroup | None:
        return self._group

    def parentAbsOffset(self) -> F:
        return F0

    def configNotation(self,
                       autoClefChanges=True,
                       staffSize=0.,
                       maxStaves=0
                       ) -> None:
        """
        Customize options for rendering this voice as notation

        Each of these options corresponds to a setting in the config

        Args:
            autoClefChanges: add clef changes to a quantized part if needed.
                Otherwise one clef is determined for each part
                (see config key `show.autoClefChanges <config_show_autoclefchanges>`).
                **NB**: clef changes can be added manually via ``Voice.eventAt(...).addSymbol(symbols.Clef(...))``
            staffSize: the size of a staff, in points (see config key `show.staffSize` <config_show_staffsize>`)
            maxStaves: the max. number of staves per voice when showing a
                Voice as notation (see config `show.voiceMaxStaves <config_show_voicemaxstaves>`)

        .. seealso:: :meth:`~Voice.configQuantization`
        """
        if staffSize is not None:
            self.setConfig('show.staffSize', staffSize)
        if autoClefChanges is not None:
            self.setConfig('show.autoClefChanges', autoClefChanges)
        if maxStaves:
            self.setConfig('show.voiceMaxStaves', maxStaves)

    def configQuantization(self,
                           breakSyncopationsLevel='',
                           complexity='',
                           nestedTuplets: bool | None = None,
                           syncopMinFraction: F | None = None,
                           debug=False
                           ) -> None:
        """
        Customize the quantization process for this Voice

        Args:
            breakSyncopationsLevel: one of 'all', 'weak', 'strong' (see
                config key `quant.breakBeats <config_quant_breaksyncopationslevel>`).
                Factory default: 'weak'
            complexity: the quantization complexity, one of 'lowest', 'low', 'medium', 'high', 'highest'
                (see config key `quant.complexity <config_quant_complexity>`). Default: 'high'
            nestedTuplets: if False, nested tuplets are disabled. (see config key `quant.nestedTuplets <config_quant_nestedtuplets>`)
            syncopMinFraction: a merged duration across beats cannot be smaller than this. Setting it too low
                can result in very complex rhythms (see config key `quant.syncopMinFraction <config_quant_syncopMinFraction>`)
            debug: if True, display debugging information when quantizing this voice

        .. seealso:: :meth:`~Voice.configNotation`, :meth:`setConfig() <maelzel.core.chain.Voice.setConfig>`

        """
        config = Workspace.active.config
        if breakSyncopationsLevel:
            self.setConfig('quant.breakBeats', breakSyncopationsLevel)
        if complexity:
            self.setConfig('quant.complexity', complexity)
        if nestedTuplets is not None:
            self.setConfig('quant.nestedTuplets', nestedTuplets)
        if syncopMinFraction is not None:
            self.setConfig('quant.syncopMinFraction', asF(syncopMinFraction))
        if debug != config['.quant.debug']:
            self.setConfig('.quant.debug', debug)

    def clone(self, **kws) -> Self:
        if 'items' not in kws:
            kws['items'] = self.items.copy()
        if 'shortname' not in kws:
            kws['shortname'] = self.shortname
        if 'name' not in kws:
            kws['name'] = self.name

        offset = kws.pop('offset', F0)
        out = self.__class__(**kws)
        if self.label:
            out.label = self.label
        if offset:
            out.timeShiftInPlace(offset)
        if self._scorestruct:
            out.setScoreStruct(self._scorestruct)
        return out

    def quantizedPart(self, **kws) -> quant.QuantizedPart:
        cfg = (self.getConfig() or Workspace.active.config).clone(showVoiceMaxStaves=1)
        qscore = self.quantizedScore(config=cfg, **kws)
        assert len(qscore.parts) == 1
        return qscore.parts[0]

    def scoringParts(self, config: CoreConfig | None = None
                     ) -> list[scoring.core.UnquantizedPart]:
        config, iscustomized = self._resolveConfig(config)
        parts = self._scoringParts(config=config,
                                   maxstaves=config['show.voiceMaxStaves'],
                                   name=self.name or self.label,
                                   shortname=self.shortname,
                                   groupParts=self._group is None,
                                   addQuantizationProfile=iscustomized)
        if self.symbols:
            for symbol in self.symbols:
                if isinstance(symbol, symbols.PartMixin):
                    if symbol.applyToAllParts:
                        for part in parts:
                            symbol.applyToPart(part)
                    else:
                        symbol.applyToPart(parts[0])

        if self._group and parts:
            parts[0].groupParts(parts,
                                groupid=self._group.groupid,
                                name=self._group.name,
                                shortname=self._group.shortname,
                                showPartNames=self._group.showPartNames)
        return parts

    def relOffset(self) -> F:
        # A voice always starts at 0
        return F0

    def absOffset(self) -> F:
        # A voice always starts at 0
        return F0

    def _asVoices(self) -> list[Voice]:
        return [self]

    def timeShift(self, offset: time_t) -> Self:
        out = self.copy()
        if offset != 0:
            out.timeShiftInPlace(offset)
        return out

    def timeShiftInPlace(self, offset: time_t) -> None:
        """
        Shift the time of this by the given offset (inplace)

        Args:
            offset: the time delta (in quarterNotes)
        """
        if offset == 0:
            return

        self._update()
        offset = asF(offset)
        if offset < 0:
            firstoffset = self.firstOffset()
            assert firstoffset is not None
            if firstoffset + offset < 0:
                raise ValueError(f"Cannot shift to negative time: first item "
                                 f"starts at {firstoffset}, cannot shift by {offset}")
        for item in self.items:
            item.timeShiftInPlace(offset)
        self._changed()


def _splitSynthGroupsIntoLines(groups: list[list[SynthEvent]]
                               ) -> list[SynthEvent | list[SynthEvent]]:
    """
    Split synthevent groups into individual lines

    When resolving the synthevents of a chain, each item in the chain is asked to
    deliver its synthevents. For an individual item which is neither tied to a
    following item nor makes a glissando the result are one or multiple synthevents
    which are independent from any other. Such synthevents are placed in the output
    list as is, flattened. Any tied events are packed inside a list

    .. code::

        C4 --gliss-- D4 --tied-- D4
                                 G3

     This results in the list [[C4, D4, D4], G3]

    Args:
        groups: A list of synthevents. Each synthevent group corresponds
            to the synthevents returned by a note/chord

    Returns:
        a list of either a single SynthEvent or a list thereof, in which case
        these enclosed synthevents build together a line


    **Algorithm**

    TODO
    """
    def matchNext(event: SynthEvent, group: list[SynthEvent], availableNodes: set[int]) -> int | None:
        pitch = event.bps[-1][1]
        for idx in availableNodes:
            candidate = group[idx]
            if abs(pitch - candidate.bps[0][1]) < 1e-6:
                return idx
        return None

    def makeLine(nodeindex: int, groupindex: int, availableNodesPerGroup: list[set[int]]
                 ) -> list[SynthEvent]:
        event = groups[groupindex][nodeindex]
        out = [event]
        if not event.linkednext or groupindex == len(groups) - 1:
            return out
        availableNodes = availableNodesPerGroup[groupindex + 1]
        if not availableNodes:
            return out
        nextEventIndex = matchNext(event, group=groups[groupindex+1], availableNodes=availableNodes)
        if nextEventIndex is None:
            return out
        availableNodes.discard(nextEventIndex)
        continuationLine = makeLine(nextEventIndex, groupindex + 1,
                                    availableNodesPerGroup=availableNodesPerGroup)
        out.extend(continuationLine)
        return out

    out: list[SynthEvent | list[SynthEvent]] = []
    availableNodesPerGroup: list[set[int]] = [set(range(len(group))) for group in groups]
    # Iterate over each group. A group is just the list of events generated by a given chord
    # Within a group, iterate over the _beatNodes of each group
    for groupindex in range(len(groups)):
        for nodeindex in availableNodesPerGroup[groupindex]:
            line = makeLine(nodeindex, groupindex=groupindex,
                            availableNodesPerGroup=availableNodesPerGroup)
            line[-1].linkednext = False
            assert isinstance(line, list) and len(line) >= 1, f"{nodeindex=}, event={groups[groupindex][nodeindex]}"
            if len(line) == 1:
                out.append(line[0])
            else:
                out.append(line)

    # last group
    if len(groups) > 1 and (lastGroupIndexes := availableNodesPerGroup[-1]):
        lastGroup = groups[-1]
        out.extend(lastGroup[idx] for idx in lastGroupIndexes)

    return out


def _resolveGlissandi(flatevents: Iterable[MEvent], force=False) -> None:
    """
    Set the _glissTarget attribute with the pitch of the gliss target
    if a note or chord has an unset gliss target (in place)

    Args:
        flatevents: subsequent events
        force: if True, calculate/update all glissando targets

    """
    ev2 = None
    for ev1, ev2 in itertools.pairwise(flatevents):
        if ev1.isRest() or ev2.isRest():
            continue
        if ev1.gliss or (ev1.playargs and ev1.playargs.get('glisstime', 0.) > 0):
            # Only calculate glissTarget if gliss is True
            if not force and ev1._glissTarget:
                continue
            if isinstance(ev1, Note):
                if isinstance(ev2, Note):
                    ev1._glissTarget = ev2.pitch
                elif isinstance(ev2, Chord):
                    ev1._glissTarget = max(n.pitch for n in ev2.notes)
                else:
                    ev1._glissTarget = ev1.pitch
            elif isinstance(ev1, Chord):
                if isinstance(ev2, Chord):
                    ev2pitches = ev2.pitches
                    if len(ev2pitches) > len(ev1.notes):
                        ev2pitches = ev2pitches[-len(ev1.notes):]
                    ev1._glissTarget = ev2pitches
                elif isinstance(ev2, Note):
                    ev1._glissTarget = [ev2.pitch] * len(ev1.notes)
                else:
                    ev1._glissTarget = ev1.pitches

    # last event
    if ev2 and ev2.gliss:
        if isinstance(ev2, Chord):
            ev2._glissTarget = ev2.pitches
        elif isinstance(ev2, Note):
            ev2._glissTarget = ev2.pitch


def _eventPairsBetween(eventpairs: list[tuple[MEvent, F]],
                       start: F,
                       end: F,
                       partial=True,
                       ) -> list[tuple[MEvent, F]]:
    """
    Events between the given time range

    If ``partial`` is false, only events which lie completey within
    the given range are included. Gracenotes at the edges are always
    included

    Args:
        eventpairs: list of pairs (event, absoluteOffset)
        start: absolute start location in beats
        end: absolute end location in beats
        partial: include also events wich are partially included within
            the given time range

    Returns:
        a list pairs (event: MEvent, absoluteoffset: F)
    """
    out = []
    if partial:
        for event, offset in eventpairs:
            if offset > end:
                break
            if event.dur > 0:
                if offset < end and offset + event.dur > start:
                    out.append((event, offset))
            elif start <= offset <= end:
                # A gracenote
                out.append((event, offset))
    else:

        for event, offset in eventpairs:
            if offset > end:
                break
            if start <= offset and offset + event.dur <= end:
                out.append((event, offset))
    return out


def _stackEvents(events: list[MEvent | Chain],
                 explicitOffsets=True,
                 ) -> F:
    """
    Stack events to the left **inplace**, making any unset offset explicit

    Args:
        events: the events to modify
        explicitOffsets: if True, all offsets are made explicit, recursively

    Returns:
        the accumulated duration of all events

    """
    # All offset times given in the events are relative to the start of the chain
    now = F0
    for ev in events:
        if ev.offset is not None:
            now = ev.offset
        elif explicitOffsets:
            ev.offset = now
        ev._resolvedOffset = now
        if isinstance(ev, Chain):
            stackeddur = _stackEvents(ev.items, explicitOffsets=explicitOffsets)
            ev._dur = stackeddur
            now = ev._resolvedOffset + stackeddur
        else:
            # An event
            now += ev.dur
    return now
