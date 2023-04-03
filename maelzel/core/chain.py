from __future__ import annotations

import re
from maelzel.common import F, asF
from maelzel.core.config import CoreConfig
from .mobj import MObj, MContainer
from .event import MEvent, asEvent, Note, Chord
from .workspace import Workspace
from .synthevent import PlayArgs, SynthEvent
from . import symbols
from . import environment
from . import _mobjtools
from . import _util
from ._common import UNSET

from maelzel import scoring
from maelzel.colortheory import safeColors

from emlib import iterlib
from emlib import misc

from typing import TYPE_CHECKING, overload
if TYPE_CHECKING:
    from typing import Any, Iterator, overload, TypeVar, Callable
    from ._typedefs import time_t
    ChainT = TypeVar("ChainT", bound="Chain")


__all__ = (
    'Chain',
    'Voice',
)


def _stackEvents(events: list[MEvent | Chain],
                 explicitOffsets=True,
                 ) -> F:
    """
    Stack events to the left **in place**, making any unset offset explicit

    Args:
        events: the events to modify
        explicitOffsets: if True, all offsets are made explicit, recursively

    Returns:
        the accumulated totalDuration of all events

    """
    if not events:
        return F(0)

    # All offset times given in the events are relative to the start of the chain
    now = F(0)
    for i, ev in enumerate(events):
        if ev.offset is not None:
            if ev.offset < now:
                if now - ev.offset < 1e-10:
                    ev.offset = now
                else:
                    raise ValueError(f'{ev} (#{i}) has an explicit offset={ev.offset}, but it overlaps with '
                                     f'the calculated offset by {now - ev.offset})')
            else:
                now = ev.offset
        elif explicitOffsets:
            ev.offset = now

        ev._resolvedOffset = now
        if isinstance(ev, MEvent):
            now += ev.dur
        elif isinstance(ev, Chain):
            stackeddur = _stackEvents(ev.items, explicitOffsets=explicitOffsets)
            ev._dur = stackeddur
            now = ev._resolvedOffset + stackeddur
        else:
            raise TypeError(f"Expected an MEvent (Note, Chord, ...) or a Chain, got {ev}")
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
        the total totalDuration of *items*

    """
    # This is the relative position (independent of the chain's start)
    now = frame
    out = []
    for i, item in enumerate(items):
        itemoffset = item._detachedOffset()
        absoffset = itemoffset + frame if itemoffset is not None else None
        if itemoffset is not None:
            if absoffset == now and (i == 0 or items[i-1].dur is not None):
                item.offset = None
            elif absoffset < now:
                raise ValueError(f"Items overlap: {item} (offset={_util.showT(absoffset)}) "
                                 f"starts before current time ({_util.showT(now)})")
            else:
                if fillGaps:
                    out.append(Note.makeRest(absoffset - now))
                now = absoffset

        if isinstance(item, MEvent):
            now += item.dur
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
                 items: list[MEvent | Chain | str] | str | None = None,
                 offset: time_t = None,
                 label: str = '',
                 properties: dict[str, Any] = None,
                 parent: MObj = None,
                 _init=True):
        if _init:
            if offset is not None:
                offset = asF(offset)
            if isinstance(items, str):
                items = _parseMultiLineChain(items)
            elif items is not None:
                items = [item if isinstance(item, (MEvent, Chain)) else asEvent(item)
                         for item in items]

        if items is not None:
            for item in items:
                item.parent = self
        else:
            items = []
        assert isinstance(items, list)
        super().__init__(offset=offset, dur=None, label=label,
                         properties=properties, parent=parent)

        self.items: list[MEvent | Chain] = items
        """The items in this chain, a list of events of other chains"""

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

    def __copy__(self) -> Chain:
        return Chain(self.items.copy(), offset=self.offset, label=self.label,
                     properties=self.properties, _init=False)

    def __deepcopy__(self, memodict={}) -> Chain:
        items = [item.copy() for item in self.items]
        properties = None if not self.properties else self.properties.copy()
        out = Chain(items=items, offset=self.offset, label=self.label,
                    properties=properties, _init=False)
        self._copyAttributesTo(out)
        return out

    def copy(self) -> Chain:
        return self.__deepcopy__()

    def stack(self, explicitOffsets=True) -> F:
        """
        Stack events to the left **INPLACE**, optionally making offsets explicit

        Args:
            explicitOffsets: make all offset explicit (sets the .offset attribute of
                all items in this chain, recursively)

        Returns:
            the total totalDuration of self
        """
        dur = _stackEvents(self.items, explicitOffsets=explicitOffsets)
        self._dur = dur
        self._modified = False
        return dur

    def fillGaps(self, recurse=True) -> None:
        """
        Fill any gaps with rests, in place

        A gap is produced when an event within a chain has an explicit offset
        later than the offset calculated by stacking the previous objects in terms
        of their totalDuration

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
            now += item.dur
        self.items = items
        self._changed()

    def itemAfter(self, item: MEvent | Chain) -> MEvent | Chain | None:
        """
        Returns the next item after *item*

        Args:
            item: the item to find its next item

        Returns:
            the item following *item* or None if the given item is not
            in this container or it has no item after it

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> chain = Chain(['4C', '4D', Chain(['4E', '4F'])])
            >>> chain.eventAfter(chain[1])
            4E
            >>> chain.itemAfter(chain[1])
            Chain([4E, 4F])

        """
        idx = self.items.index(item)
        return self.items[idx + 1] if idx < len(self.items) - 2 else None

    def eventAfter(self, event: MEvent) -> MEvent | None:
        """
        Returns the next event after *event*

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> chain = Chain(['4C', '4D', Chain(['4E', '4F'])])
            >>> chain.eventAfter(chain[1])
            4E
            >>> chain.itemAfter(chain[1])
            Chain([4E, 4F])
        """
        idx = self.items.index(event)
        if idx >= len(self.items) - 1:
            return None
        item = self.items[idx]
        return item if isinstance(item, MEvent) else next(item.recurse())

    def isFlat(self) -> bool:
        """Is self flat?

        A flat sequence contains only events, not other containers
        """
        return all(isinstance(item, MEvent) for item in self.items)

    def flat(self, forcecopy=False) -> Chain:
        """
        A flat version of this Chain

        A Chain can contain other Chains. This method flattens all objects inside
        this Chain and any sub-chains to a flat chain of notes/chords.

        If this Chain is already flat (it does not contain any
        Chains), self is returned unmodified. **NB**: all items in the returned
        Chain have an explicit ``.offset``

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
        for event, offset in self.recurseWithOffset():
            if forcecopy or event.offset != offset:
                event = event.copy()
            event.offset = offset
            items.append(event)
        assert all(item.offset is not None for item in items)
        return self.clone(items=items)

    def pitchRange(self) -> tuple[float, float] | None:
        pitchRanges = [item.pitchRange() for item in self.items]
        return min(p[0] for p in pitchRanges), max(p[1] for p in pitchRanges)

    def withExplicitTimes(self, forcecopy=False) -> Chain:
        """
        Copy of self with explicit times

        If self already has explicit offset, self itself
        is returned. If you relie on the fact that this method returns
        a copy, use ``forcecopy=True``

        Args:
            forcecopy: if forcecopy, a copy of self will be returned even
                if self already has explicit times

        Returns:
            a clone of self with explicit times

        Example
        ~~~~~~~

        The offset and dur shown as the first two columns are the resolved
        times. When an event has an explicit offset or totalDuration these are
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
        if self.isResolved() and not forcecopy:
            return self
        out = self.copy()
        out.stack(explicitOffsets=True)
        return out

    def isResolved(self) -> bool:
        """
        True if self has an explicit offset and all items as well (recursively)

        Returns:
            True if all items in self have explicit offsets
        """
        if self.offset is None:
            return False

        return all(item.offset is not None if isinstance(item, MEvent) else item.isResolved()
                   for item in self.items)

    def _resolveGlissandi(self, force=False) -> None:
        """
        Set the .glisstarget property with the pitch of the gliss target
        if a note or chord has an unset gliss target

        Args:
            force: if True, calculate/update all glissando targets

        """
        for ev1, ev2 in iterlib.pairwise(self.recurse()):
            if not ev1.gliss or not isinstance(ev1.gliss, bool):
                continue
            if not force and ev1.properties and ev1.properties.get('.glisstarget'):
                continue
            if isinstance(ev1, Note):
                if isinstance(ev2, Note):
                    ev1.setProperty('.glisstarget', ev2.pitch)
                elif isinstance(ev2, Chord):
                    ev1.setProperty('.glisstarget', max(n.pitch for n in ev2.notes))
            elif isinstance(ev1, Chord):
                if isinstance(ev2, Note):
                    ev1.setProperty('.glisstarget', [ev2.pitch] * len(ev1.notes))
                elif isinstance(ev2, Chord):
                    ev2pitches = ev2.pitches
                    if len(ev2pitches) > len(ev1.notes):
                        ev2pitches = ev2pitches[-len(ev1.notes):]
                    ev1.setProperty('.glisstarget', ev2pitches)

    def _synthEvents(self,
                     playargs: PlayArgs,
                     parentOffset: F,
                     workspace: Workspace
                     ) -> list[SynthEvent]:
        conf = workspace.config
        if self.playargs:
            playargs = playargs.overwrittenWith(self.playargs)

        if self._modified:
            self._update()
        self._resolveGlissandi()

        chain = self.flat(forcecopy=True)
        chain.stack()

        for item in chain.items:
            assert item.dur >= 0, f"{item=}"

        if self.offset:
            for item in chain.items:
                item.offset += self.offset

        if any(n.isGracenote() for n in chain.items):
            gracenoteDur = F(conf['play.gracenoteDuration'])
            _mobjtools.addDurationToGracenotes(chain.items, gracenoteDur)

        if conf['play.useDynamics']:
            _mobjtools.fillTempDynamics(chain.items, initialDynamic=conf['play.defaultDynamic'])

        synthevents = []
        offset = parentOffset + self.resolveOffset()
        groups = _mobjtools.groupLinkedEvents(chain.items)
        for group in groups:
            if isinstance(group, MEvent):
                events = group._synthEvents(playargs,   # should we copy playargs??
                                            parentOffset=offset,
                                            workspace=workspace)
                synthevents.extend(events)
            elif isinstance(group, list):
                synthgroups = [event._synthEvents(playargs, parentOffset=offset, workspace=workspace)
                               for event in group]
                synthlines = _splitSynthGroupsIntoLines(synthgroups)
                for synthline in synthlines:
                    if isinstance(synthline, SynthEvent):
                        synthevent = synthline
                    elif isinstance(synthline, list):
                        if len(synthline) == 1:
                            synthevent = synthline[0]
                        else:
                            for ev1, ev2 in iterlib.pairwise(synthline):
                                assert abs(ev1.end - ev2.delay) < 1e-6, f"gap={ev1.end - ev1.delay}, {ev1=}, {ev2=}"

                            synthevent = SynthEvent.mergeTiedEvents(synthline)
                    else:
                        raise TypeError(f"Expected a SynthEvent or a list thereof, got {synthline}")
                    synthevents.append(synthevent)
                    # TODO: fix / add playargs
            else:
                raise TypeError(f"Did not expect {group}")
        for event in synthevents:
            event.linkednext = False
        return synthevents

    def mergeTiedEvents(self) -> None:
        """
        Merge tied events in place

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

    def childOffset(self, child: MObj) -> F:
        """
        Returns the offset of child within this chain

        Args:
            child: the object whose offset is to be determined

        Returns:
            The offset of this child within this chain
        """
        if child not in self.items:
            raise ValueError(f"The item {child} is not a child of {self}")

        if child.offset is not None:
            return child.offset

        self._update()
        return child._resolvedOffset

    @property
    def dur(self) -> F:
        """The duration of this sequence"""
        if self._dur is not None and not self._modified:
            return self._dur

        if not self.items:
            self._dur = F(0)
            return self._dur

        self._update()
        assert self._dur is not None
        return self._dur

    @dur.setter
    def dur(self, value):
        cls = type(self).__name__
        raise AttributeError(f"Objects of class {cls} cannot set their duration")

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
        self._dur = _stackEvents(self.items, explicitOffsets=False)
        self._modified = False
        for item in self.items:
            if item.properties and '.glisstarget' in item.properties:
                del item.properties['.glisstarget']

    def _changed(self) -> None:
        self._modified = True
        self._dur = None
        if self.parent:
            self.parent.childChanged(self)

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

    def _dumpRows(self, indents=0, now=F(0), forcetext=False) -> list[str]:
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

        if environment.insideJupyter and not forcetext:
            _ = _util.htmlSpan
            r = type(self).__name__

            header = (f'<code><span style="font-size: {fontsize}">{IND*indents}<b>{r}</b> - '
                      f'beat: {self.absoluteOffset()}, offset: {selfstart}, dur: {_util.showT(self.dur)}'
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
                if item.properties:
                    infoparts.append(f'properties: {item.properties}')

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
                    playargs = 'None' if not item.playargs else ', '.join(f'{k}={v}' for k, v in item.playargs.db.items())
                    rowparts = [IND*(indents+1),
                                _util.showT(now + itemoffset).ljust(widths['beat']),
                                offsetstr,
                                durstr,
                                name.ljust(widths['name']),
                                str(item.gliss).ljust(widths['gliss']),
                                str(item.dynamic).ljust(widths['dyn']),
                                playargs.ljust(widths['playargs']),
                                ' '.join(infoparts) if infoparts else '-'
                                ]
                    row = f"<code>{_util.htmlSpan(''.join(rowparts), ':blue1', fontsize=fontsize)}</code>"
                    rows.append(row)
                    if item.symbols:
                        row = f"<code>      {_util.htmlSpan(str(item.symbols), ':green2', fontsize=fontsize)}</code>"
                        rows.append(row)

                elif isinstance(item, Chain):
                    rows.extend(item._dumpRows(indents=indents+1, now=now+itemoffset, forcetext=forcetext))
            return rows
        else:
            rows = [f"{IND * indents}Chain -- beat: {self.absoluteOffset()}, offset: {selfstart}, dur: {self.dur}",
                    f'{IND * (indents + 1)}beat   offset  dur    item']
            items, itemsdur = self._iterateWithTimes(recurse=False, frame=F(0))
            for item, itemoffset, itemdur in items:
                if isinstance(item, MEvent):
                    rows.append(f'{IND * (indents+1)}'
                                f'{repr(now + itemoffset).ljust(7)}'
                                f'{repr(itemoffset).ljust(7)} '
                                f'{repr(itemdur).ljust(7)}'
                                f'{item}')
                elif isinstance(item, Chain):
                    rows.extend(item._dumpRows(indents=indents+1, forcetext=forcetext, now=now+itemoffset))
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
        rows = self._dumpRows(indents=indents, now=self.offset or F(0), forcetext=forcetext)
        if environment.insideJupyter and not forcetext:
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
        self._update()
        itemcolor = safeColors['blue2']
        items = self.items if len(self.items) < 10 else self.items[:10]
        itemstr = ", ".join(f'<span style="color:{itemcolor}">{repr(_)}</span>'
                            for _ in items)
        if len(self.items) >= 10:
            itemstr += ", …"
        cls = self.__class__.__name__
        namedargs = [f'dur={_util.showT(self.dur)}']
        if self.offset is not None:
            namedargs.append(f'offset={self.offset}')
        info = ', ' + ', '.join(namedargs)
        return f'{cls}([{itemstr}]{info})'

    def removeRedundantOffsets(self, fillGaps=False) -> None:
        """
        Remove over-specified start times in this Chain

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
        _removeRedundantOffsets(self.items, fillGaps=fillGaps, frame=F(0))
        self._modified = True

    def asVoice(self) -> Voice:
        """
        Convert this Chain to a Voice
        """
        items = self.copy().items
        _ = _stackEvents(items, explicitOffsets=True)
        if self.offset:
            for item in items:
                item.offset += self.offset
        voice = Voice(items, name=self.label)
        voice.removeRedundantOffsets()
        return voice

    def timeTransform(self: ChainT, timemap: Callable[[F], F], inplace=False
                      ) -> ChainT:
        items = []
        for item in self.items:
            items.append(item.timeTransform(timemap, inplace=inplace))
        return self if inplace else self.clone(items=items)

    def scoringEvents(self,
                      groupid='',
                      config: CoreConfig = None,
                      parentOffset: F | None = None
                      ) -> list[scoring.Notation]:
        """
        Returns the scoring events corresponding to this object

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

        if parentOffset is None:
            parentOffset = self.parent.absoluteOffset() if self.parent else F(0)

        offset = parentOffset + self.resolveOffset()
        chain = self.flat()
        notations: list[scoring.Notation] = []
        if self.label and chain and chain[0].resolveOffset() > 0:
            notations.append(scoring.makeRest(duration=chain[0].resolveOffset()))
        for item in chain.items:
            notations.extend(item.scoringEvents(groupid=groupid, config=config, parentOffset=offset))
        #if self.label:
        #y    notations[0].addText(self._scoringAnnotation(text=self.label))
        scoring.stackNotationsInPlace(notations)

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

    def scoringParts(self,
                     config: CoreConfig = None
                     ) -> list[scoring.Part]:
        if config is None:
            config = Workspace.active.config
        notations = self.scoringEvents(config=config)
        if not notations:
            return []
        scoring.stackNotationsInPlace(notations)
        if config['show.voiceMaxStaves'] == 1:
            parts = [scoring.Part(notations, name=self.label)]
        else:
            groupid = scoring.makeGroupId()
            parts = scoring.distributeNotationsByClef(notations, groupid=groupid)
        return parts

    def quantizePitch(self, step=0.25):
        if step <= 0:
            raise ValueError(f"Step should be possitive, got {step}")
        items = [i.quantizePitch(step) for i in self.items]
        return self.clone(items=items)

    def timeShift(self: ChainT, timeoffset: time_t) -> ChainT:
        items = [item.timeShift(timeoffset) for item in self.items]
        return self.clone(items=items)

    def firstOffset(self) -> F | None:
        """
        Returns the offset (relative to the start of this chain) of the first event in this chain
        """
        event  = self.firstEvent()
        return None if not event else event.absoluteOffset() - self.absoluteOffset()

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
        For each item returns a tuple (item, dur)

        Each event is represented as a tuple (event, offset, dur), a chain
        is represented as a list of such tuples

        Args:
            recurse: if True, traverse any subchain
            frame: the frame of reference

        Returns:
            a tuple (eventtuples, totalDuration) where eventtuples is a list of
            tuples (event, offset, dur). If recurse is True,
            any subchain is returned as a list of eventtuples. Otherwise,
            a flat list is returned. In each eventtuple, the offset is relative
            to the first frame passed, so if the first offset was 0
            the offsets will hold the absolute offset of each event. Duration
            is the total totalDuration of the items in
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
                dur = item.dur
                out.append((item, now, dur))
                item._resolvedOffset = now - frame
                if i == 0 and self.label:
                    item.setProperty('.chainlabel', self.label)
                now += dur
            else:
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

    def recurseWithOffset(self, absolute=False) -> Iterator[tuple[MEvent, F]]:
        """
        Recurse the events in self, yields a tuple (event, offset) for each event

        Args:
            absolute: if True, the offset returned for each item will be its
                absolute offset; otherwise the offsets are relative to the start of self

        Returns:
            a generator of tuples (event, offset), where offset is either the
            offset relative to the start of self, or absolute if absolute is True

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> chain = Chain([
            ... "4C:0.5",
            ... "4D",
            ... Chain(["4E:0.5"], offset=2)
            ... ], offset=1)

        """
        frame = self.absoluteOffset() if absolute else F(0)
        itemtuples, totaldur = self._iterateWithTimes(frame=frame, recurse=True)
        for itemtuple in itemtuples:
            item = itemtuple[0]
            if isinstance(item, MEvent):
                item, offset, dur = itemtuple
                yield item, offset
            else:
                for subitem, offset, dur in item:
                    yield item, offset

    def iterateWithTimes(self, absolute=False
                         ) -> list[tuple[MEvent, F, F] | list]:
        """
        Iterates over the items in self, returns a tuple (event, offset, duration) for each

        The explicit times (offset, duration) of the events are not modified. For each event
        it returns a tuple (event, offset, duration), where the event is unmodified, the
        offset is the withExplicitTimes offset relative to its parent and the totalDuration is the withExplicitTimes
        totalDuration. Nested subchains result in nested lists. All times are in beats

        Args:
            absolute: if True, offsets are returned as absolute offsets

        Returns:
            a list of tuples, one tuple for each event.
            Each tuple has the form (event: MEvent, offset: F, duration: F). The
            explicit times in the events themselves are never modified. *offset* will
            be the offset to self if ``absolute=False`` or the absolute offset otherwise
        """
        offset = self.absoluteOffset() if absolute else F(0)
        items, itemsdur = self._iterateWithTimes(frame=offset, recurse=True)
        self._dur = itemsdur
        return items

    def addSpanner(self, spanner: str | symbols.Spanner
                   ) -> None:
        """
        Adds a spanner symbol across this object

        A spanner is a slur, line or any other symbol attached to two or more
        objects. A spanner always has a start and an end.

        Args:
            spanner: a Spanner object or a spanner description (one of 'slur', '<', '>',
                'trill', 'bracket', etc. - see :func:`maelzel.core.symbols.makeSpanner`
                When passing a string description, prepend it with '~' to create an end spanner

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
        first = next(self.recurse())
        last = next(self.recurse(reverse=True))
        if isinstance(spanner, str):
            spanner = symbols.makeSpanner(spanner)
        assert isinstance(first, (Note, Chord)) and isinstance(last, (Note, Chord))
        spanner.bind(first, last)

    def firstEvent(self) -> MEvent | None:
        """The first event in this chain, recursively"""
        return next(self.recurse(), None)

    def lastEvent(self) -> MEvent | None:
        """The last event in this chain, recursively"""
        return next(self.recurse(reverse=True))

    def eventsBetween(self, startbeat: time_t, endbeat: time_t, absolute=False
                      ) -> list[MEvent]:
        """
        Returns the events which are **included** by the given times in quarternotes

        Events which start before *startbeat* or end after *endbeat* are not
        included, even if parts of them might lie between the given time interval.
        Chains are treated as one item. To access sub-chains, first flatten self.

        Args:
            startbeat: the start time, relative to the start of self
            endbeat: end time, relative to the start of self
            absolute: if True, interpret startbeat and endbeat as absolute

        Returns:
            a list of events located between the given time-interval

        """
        startbeat = asF(startbeat)
        endbeat = asF(endbeat)
        items = []
        for item, itemoffset, itemdur in self.iterateWithTimes(absolute=absolute):
            if startbeat <= itemoffset and itemoffset+itemdur <= endbeat:
                items.append(item)
        return items

    def splitEventsAtOffsets(self, offsets: list[F], tie=True) -> None:
        """
        Splits items in self at the given offsets, **inplace**

        The offsets are absolute. Split items are by default tied together

        Args:
            offsets: the offsets to split items at.
            tie: if True, parts of an item are tied together
        """
        items = []
        for item, itemoffset, itemdur in self.iterateWithTimes(absolute=True):
            if isinstance(item, Chain):
                item.splitEventsAtOffsets(offsets, tie=tie)
                items.append(item)
            else:
                parts = item.splitAtOffsets(offsets, tie=tie)
                items.extend(parts)
        self.items = items


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
    _configKeys: set[str] | None = None

    def __init__(self,
                 items: list[MEvent | str] | Chain = None,
                 name='',
                 shortname='',
                 maxstaves: int = None,
                 clef: str = ''):
        if isinstance(items, Chain):
            items = items.items
        super().__init__(items=items, offset=F(0))
        self.name = name
        """The name of this voice/staff"""

        self.shortname = shortname
        """A shortname to display as abbreviation after the first system"""

        self.maxstaves = maxstaves if maxstaves is not None else Workspace.active.config['show.voiceMaxStaves']
        """The max. number of staves this voice can be expanded to"""

        self.clef = clef
        """If given, force this voice to use this clef for the case of a single staff"""

        self._config: dict[str, Any] = {}
        """Any key set here will override keys from the coreconfig for rendering
        Any key in CoreConfig is supported"""

    def setConfig(self, key: str, value) -> Voice:
        """
        Set a configuration key for this object.

        Possible keys are any CoreConfig keys with the prefixes 'quant.' and 'show.'

        Args:
            key: the key to set
            value: the value. It will be validated via CoreConfig

        Returns:
            self. This allows multiple calls to be chained

        Example
        ~~~~~~~

        Configure the voice to break syncopations at every beat when
        rendered or quantized as a QuantizedScore

            >>> voice = Voice(...)
            >>> voice.setConfig('quant.brakeSyncopationsLevel', 'all')

        Now, whenever the voice is shown in itself or as part of a score
        all syncopations across beat boundaries will be split into tied notes.

        This is the same as:

            >>> voice = Voice(...)
            >>> score = Score([voice])
            >>> quantizedscore = score.quantizedScore()
            >>> quantizedscore.parts[0].brakeSyncopations(level='all')
            >>> quantizedscore.render()
        """
        configkeys = Voice._configKeys
        if not configkeys:
            pattern = r'(quant|show)\..+'
            corekeys = CoreConfig.root.keys()
            configkeys= set(k for k in corekeys if re.match(pattern, k))
            Voice._configKeys = configkeys
        if key not in configkeys:
            raise KeyError(f"Key {key} not known. Possible keys: {configkeys}")
        if errormsg := CoreConfig.root.checkValue(key, value):
            raise ValueError(f"Cannot set {key} to {value}: {errormsg}")
        self._config[key] = value
        return self

    def getConfig(self, prototype: CoreConfig = None) -> CoreConfig | None:
        """
        Returns a CoreConfig overloaded with any option set via :meth:`~Voice.setConfig`

        Args:
            prototype: the config to use as prototype. If not given, the active config
                will be used

        Returns:
            the resulting CoreConfig set via :meth:`Voice.setConfig`. If not prototype
            and no changes have been made, None is returned

        """
        if not self._config:
            return prototype
        return (prototype or Workspace.active.config).clone(self._config)

    def configQuantization(self,
                           breakSyncopationsLevel: str = None,
                           ) -> None:
        """
        Customize the quantization process for this Voice

        Args:
            breakSyncopationsLevel: one of 'all', 'weak', 'strong' (see
                config key `quant.breakSyncopationsLevel <config_quant_breaksyncopationslevel>`)

        """
        if breakSyncopationsLevel is not None:
            self.setConfig('quant.breakSyncopationsLevel', breakSyncopationsLevel)

    def clone(self, **kws) -> Voice:
        if not 'items' in kws:
            kws['items'] = self.items.copy()
        if not 'shortname' in kws:
            kws['shortname'] = self.shortname
        if not 'name' in kws:
            kws['name'] = self.name
        if not 'clef' in kws:
            kws['clef'] = self.clef
        out = Voice(**kws)
        if self.label:
            out.label = self.label
        return out

    def scoringParts(self, config: CoreConfig = None
                     ) -> list[scoring.Part]:
        config = self.getConfig(config)
        parts = super().scoringParts(config=config)
        if len(parts) == 1:
            part = parts[0]
            part.name = self.name
            part.shortname = self.shortname
            part.firstclef = self.clef
        else:
            for i, part in enumerate(parts):
                part.name = f'{self.name}-{i+1}'
                part.shortname = f'{self.shortname}{i+1}'
        if config:
            quantProfile = config.makeQuantizationProfile()
            for part in parts:
                part.quantProfile = quantProfile
        return parts


def _parseMultiLineChain(s: str) -> list[MEvent]:
    s = s.replace(';', '\n')
    events = [asEvent(line.strip()) for line in s.splitlines() if line]
    return events


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
        groups: A list of synthevents. Each synthevent tree corresponds
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
        # tree = groups[groupindex]
        # event = tree[nodeindex]
        event = groups[groupindex][nodeindex]
        out = [event]
        if not event.linkednext or groupindex == len(groups) - 1:
            return out
        availableNodes = availableNodesPerGroup[groupindex + 1]
        if not availableNodes:
            return out
        # nextGroup = groups[groupindex + 1]
        # candidates = [nextGroup[index] for index in availableNodes]
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
    # Iterate over each tree. A tree is just the list of events generated by a given chord
    # Within a tree, iterate over the nodes of each tree
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

    # last tree
    if len(groups) > 1 and (lastGroupIndexes := availableNodesPerGroup[-1]):
        lastGroup = groups[-1]
        out.extend(lastGroup[idx] for idx in lastGroupIndexes)

    return out
