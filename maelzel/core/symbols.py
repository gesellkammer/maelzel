"""
Notation can be customized by adding so called :class:`Symbols <Symbol>` to
an object. Symbols are objects which can be attached to a note, chord, voice, etc.
to modify its representation as musical notation.

Most symbols do not have any other meaning than to indicate how a certain object
should be displayed as notation. In particular, they do not affect
playback in any way. For example a Notehead symbol can be attached to a note
to modify the notehead shape used, a Color can be attached to a voice to modify
the color of all its items, a Clef can be added to a note to force a clef change
at that particular moment, etc.

There are basically three kind of symbols: properties, note-attached symbols
and spanners.

Property
    A **property** modifies an attribute of the object it is attached to.
    For example, a SizeFactor property modifies the size of the object and, if the
    object is a container (a voice, for example), then all elements within that
    container are modified.

"""
# This module cannot import from maelzel.core

from __future__ import annotations
from abc import abstractmethod, ABC
import random
import copy
from functools import cache

from maelzel import _util
from maelzel import colortheory
from maelzel.common import F

from maelzel import scoring
import maelzel.scoring.spanner as _spanner
import maelzel.scoring.attachment as _attachment

from ._common import logger
import pitchtools as pt
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self
    from typing import Sequence, Any
    from maelzel.core import mobj
    from maelzel.core import mevent
    
_uuid_alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'


def _makeuuid(size=8) -> str:
    return ''.join(random.choices(_uuid_alphabet, k=size))


class Symbol(ABC):
    """Base class for all symbols"""

    exclusive = False
    """Only one of a given class"""

    applyToTieStrategy = 'first'
    """One of 'first', or 'all'"""

    def __init__(self):
        self.properties: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return _util.reprObj(self, hideFalsy=True)

    def getProperty(self, key: str, default=None) -> Any:
        """
        Get a property value by key, or return the default value if not found.
        """
        return default if not self.properties else self.properties.get(key, default)

    def setProperty(self, key: str, value) -> None:
        """
        Set a property value by key.
        """
        if self.properties is None:
            self.properties = {}
        self.properties[key] = value

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @abstractmethod
    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        """Apply this symbol to the given notation, **inplace**"""
        raise NotImplementedError

    def applyToTiedGroup(self, notations: Sequence[scoring.Notation], parent: mobj.MObj | None
                         ) -> None:
        """
        Apply this symbol to a group of tied notations, **inplace**.
        """
        if self.applyToTieStrategy == 'all':
            for n in notations:
                self.applyToNotation(n, parent=parent)
        elif self.applyToTieStrategy == 'first':
            self.applyToNotation(notations[0], parent=parent)


class Spanner(Symbol):
    """
    A Spanner is a line/curve between two anchors (notes/chords/rests)

    Spanners always come in pairs start/end.

    Example
    ~~~~~~~

        >>> from maelzel.core import *
        >>> from maelzel.core import symbols
        >>> chain = Chain(["C4:0.5", "D4:1", "E4:0.5"])
        >>> slur = symbols.Slur()
        >>> chain[0].addSymbol(slur)
        >>> chain[2].addSymbol(slur.makeEndSpanner())

    Or you can use the ``.bind`` method which is a shortcut:

        >>> symbols.Slur().bind(chain[0], chain[2])
    """
    exclusive = False
    appliesToRests = True

    def __init__(self,
                 kind='start',
                 uuid='',
                 linetype='solid',
                 placement='',
                 color='',
                 anchor: mevent.MEvent | None = None):
        super().__init__()
        assert not kind or kind == 'start' or kind == 'end', f"got kind={kind}"
        assert linetype in {'', 'solid', 'dashed', 'dotted', 'wavy', 'trill', 'zigzag'}, f"got {linetype}"
        if placement:
            assert placement == 'above' or placement == 'below'
        self.kind = kind
        """The kind of spanner. Either unset or 'start' or 'end'"""

        self.uuid = uuid or _makeuuid(8)
        """An id identifying this spanner"""

        self.linetype = linetype
        """Linetype, one of solid, dahsed, dotted, wavy, trill, zigzag"""

        self.placement = placement
        """If given, one of above or below"""

        self.color = color
        """Color, any hex or css color"""

        self._anchor: mevent.MEvent | None = anchor
        """The event to which this spanner is anchored to, if known"""

        self._partner: Spanner | None = None
        """The partner spanner"""

    @property
    def anchor(self) -> mevent.MEvent | None:
        return self._anchor

    @property
    def partner(self) -> Spanner | None:
        return self._partner

    def scoringSpanner(self) -> _spanner.Spanner:
        raise NotImplementedError

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        spanner = self.scoringSpanner()
        n.addSpanner(spanner)

    def applyToPair(self, start: scoring.Notation, end: scoring.Notation) -> None:
        assert isinstance(start, scoring.Notation) and isinstance(end, scoring.Notation)
        partner = self.makePartnerSpanner()
        self.applyToNotation(start, parent=None)
        partner.applyToNotation(end, parent=None)

    def applyToTiedGroup(self, notations: Sequence[scoring.Notation], parent: mobj.MObj | None
                         ) -> None:
        if self.kind == 'start':
            self.applyToNotation(notations[0], parent=parent)
        elif self.kind == 'end':
            self.applyToNotation(notations[-1], parent=parent)
        else:
            logger.warning("Unknown kind '%s' for spanner %s", self.kind, str(self))



    def __repr__(self) -> str:
        return _util.reprObj(self,
                             hideFalsy=True,
                             properties=('anchor',),
                             filter={'linetype': lambda val: val!='solid'},
                             convert={'anchor': lambda ev: ev.name,
                                      'partnerSpanner': lambda val: f"{type(val).__name__}"},)

    def _attrs(self) -> dict:
        keys = ('kind', 'uuid', 'linetype', 'placement', 'color')
        return {k: v for k in keys
                if (v := getattr(self, k))}

    def setAnchor(self, obj: mevent.MEvent) -> None:
        """
        Set the anchor for this spanner.

        This is called by :meth:``MusicObj.setSymbol`` or by :meth:``Spanner.bind`` to
        set the anchor of this spanner. A User should not normally call this method

        Args:
            obj: the object this spanner is anchored to, either as start or end
        """
        self._anchor = obj

    def bind(self, startobj: mevent.MEvent, endobj: mevent.MEvent) -> None:
        """
        Bind a Spanner to two notes/chords

        Args:
            startobj: start anchor object
            endobj: end anchor object

        Example
        ~~~~~~~

            >>> from maelzel.core import *
            >>> chain = Chain(["4E:1", "4F:1", "4G:1"])
            # slur the first two notes of the chain, customizing the line-type
            >>> Slur(linetype='dashed').bind(chain[0], chain[1])
            # The same can be achieved as:
            >>> chain[0].addSpanner(Slur(linetype='dashed'), chain[1])

        """
        if startobj is endobj:
            raise ValueError("Cannot bind a spanner to the same object as start and end anchor")

        startobj.addSymbol(self)
        if self.anchor is None:
            self.setAnchor(startobj)
        self.makePartnerSpanner(anchor=endobj)
        assert self.partner is not None

    def makePartnerSpanner(self, anchor: mevent.MEvent | None = None) -> Self:
        """
        Creates the partner spanner for an already existing spanner

        start and end spanner share the same uuid and have a ref to
        each other, allowing each of the spanners to access their
        twin. As each spanner also holds a ref to their anchor,
        the anchored events can be made aware of each other.

        Args:
            anchor: the event to which the end spanner is anchored to.

        Returns:
            the created spanner. This is a copy of the start spanner in every
            way with the only difference that ``kind='end'``

        """
        if anchor and self.anchor is anchor:
            raise ValueError("Start anchor and end anchor cannot be the same object")
        if not self.kind:
            raise ValueError("This spanner is not a start or end spanner")
        endSpanner = copy.copy(self)
        self.setPartner(endSpanner)
        if anchor:
            anchor.addSymbol(endSpanner)
        return endSpanner

    def setPartner(self, partner: Spanner) -> None:
        """
        Set the given spanner as the partner spanner of self (and self as partner of other)

        Args:
            partner: the partner spanner

        """
        partner._partner = self
        self._partner = partner
        if self.kind == 'start':
            partner.kind = 'end'
            partner.uuid = self.uuid
        else:
            partner.kind = 'start'
            self.uuid = partner.uuid


class TrillLine(Spanner):
    """
    Trill spanner between two notes.

    Args:
        kind: start | end
        startmark: one of 'trill' or 'bisb' (bisbigliando)
        trillpitch: if given, the pitch to trill with
        alteration: if given, add an alteration to the trill. Not compatible with
            trillpitch
        placement: 'above' | 'below'

    """

    def __init__(self,
                 kind='start',
                 startmark='trill',
                 trillpitch='',
                 alteration='',
                 placement='above',
                 uuid='',
                 **kws):
        super().__init__(kind=kind, placement=placement, uuid=uuid, **kws)
        self.startmark = startmark
        self.trillpitch = trillpitch
        self.alteration = alteration

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        spanner = _spanner.TrillLine(kind=self.kind, uuid=self.uuid,
                                     startmark=self.startmark,
                                     alteration=self.alteration,
                                     trillpitch=self.trillpitch,
                                     placement=self.placement)
        n.addSpanner(spanner)


class NoteheadLine(Spanner):
    """
    A line conecting two noteheads

    This results in a glissando only when rendered
    as notation

    Args:
        kind: one of 'start', 'end'
        uuid: used to match the start spanner
        color: a css color
        linetype: a valid linetype
        text: a text attached to the line
    """

    appliesToRests = False

    def __init__(self, kind='start', uuid='', color='', linetype='solid', text=''):
        super().__init__(kind=kind, uuid=uuid, color=color, linetype=linetype)
        self.text = text

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        spanner = _spanner.Slide(kind=self.kind, uuid=self.uuid,
                                 color=self.color, linetype=self.linetype,
                                 text=self.text)
        n.addSpanner(spanner)


class OctaveShift(Spanner):
    """
    An octave shift

    Args:
        kind: the kind of spanner, one of 'start', 'end'
        octaves: number of octave to shift. Can be negative
        uuid: used to match start and end spanner
    """

    def __init__(self, kind='start', octaves=1, uuid=''):
        assert octaves != 0 and abs(octaves) <= 3
        super().__init__(kind=kind, placement='above' if octaves >= 0 else 'below', uuid=uuid)
        self.octaves = octaves

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        spanner = _spanner.OctaveShift(kind=self.kind, octaves=self.octaves,
                                       uuid=self.uuid)
        n.addSpanner(spanner)


class Slur(Spanner):
    """
    A slur spanner between two notes
    """

    def scoringSpanner(self) -> _spanner.Spanner:
        return _spanner.Slur(kind=self.kind, uuid=self.uuid, linetype=self.linetype,
                             placement=self.placement, color=self.color)



class Beam(Spanner):
    """
    Notes within a Beam spanner are beamed together
    """
    appliesToRests = True

    def scoringSpanner(self) -> _spanner.Spanner:
        return _spanner.Beam(kind=self.kind, uuid=self.uuid)


class Hairpin(Spanner):
    """
    A hairpin crescendo or decrescendo

    Args:
        direction: one of "<" or ">"
        niente: if True, add a niente 'o' to the start or end of the hairpin
    """
    def __init__(self, direction: str, niente=False, kind='start', uuid='', placement='', linetype=''):
        super().__init__(kind=kind, uuid=uuid, placement=placement, linetype=linetype)
        assert direction == "<" or direction == ">"
        self.direction = direction
        self.niente = niente

    def _attrs(self):
        attrs = {'direction': self.direction}
        attrs.update(super()._attrs())
        return attrs

    def scoringSpanner(self) -> _spanner.Spanner:
        return _spanner.Hairpin(kind=self.kind, uuid=self.uuid,
                                direction=self.direction,
                                niente=self.niente,
                                placement=self.placement)


class Bracket(Spanner):
    def __init__(self, kind='start', uuid='', linetype='solid', placement='',
                 text=''):
        super().__init__(kind=kind, uuid=uuid, linetype=linetype, placement=placement)
        self.text = text

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        bracket = _spanner.Bracket(kind=self.kind, uuid=self.uuid,
                                   text=self.text, placement=self.placement,
                                   linetype=self.linetype)
        n.addSpanner(bracket)


class LineSpan(Spanner):
    """
    A line spanner

    Args:
        kind: start or end
        uuid: the uuid, will be autogenerated if not given.
        linetype: one of solid, dashed, dotted, wavy, zigzag
        placement: unset to use default, otherwise above or below
        starttext: a text to add at the begining of the line
        endtext: a text to add at the end. If an endtext is specified
            an endhook is not allowed.
        middletext: a text to add at the middle (not always supported)
        verticalAlign: alignment of the text (one of up, down, center)
        endhook: if true, draw a hook segment at the end of the line.
            Direction of the hook is opposito to placement. An endhook
            is mutually exclusive with an endtext
    """
    def __init__(self, kind='start', uuid='', linetype='solid', placement='',
                 starttext='', endtext='', middletext='', verticalAlign='',
                 starthook=False, endhook=False):
        super().__init__(kind=kind, uuid=uuid, linetype=linetype, placement=placement)
        if verticalAlign:
            assert verticalAlign in {'up', 'down', 'center'}
        assert not (endtext and endhook), "An endtext and an endhook are mutually exclusive"
        self.starttext = starttext
        self.endtext = endtext
        self.middletext = middletext
        self.verticalAlign = verticalAlign
        self.starthook = starthook
        self.endhook = endhook

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        spanner = _spanner.LineSpan(kind=self.kind, uuid=self.uuid,
                                    placement=self.placement, linetype=self.linetype,
                                    starttext=self.starttext, endtext=self.endtext,
                                    middletext=self.middletext,
                                    verticalAlign=self.verticalAlign,
                                    starthook=self.starthook,
                                    endhook=self.endhook)
        n.addSpanner(spanner)



_spannerNameToConstructor: dict[str, Any] = {
    'slur': Slur,
    'line': LineSpan,
    'linespan': LineSpan,
    'trill': TrillLine,
    'tr': TrillLine,
    'bracket': Bracket,
    '<': lambda **kws: Hairpin(direction='<', **kws),
    'cresc': lambda **kws: Hairpin(direction='<', **kws),
    '>': lambda **kws: Hairpin(direction='>', **kws),
    'decresc': lambda **kws: Hairpin(direction='>', **kws),
    'dim': lambda **kws: Hairpin(direction=">", **kws),
    'hairpin': Hairpin,
    'beam': Beam
}


def makeSpanner(descr: str, kind='start', linetype='', placement='', color=''
                ) -> Spanner:
    """
    Create a spanner from a descriptor

    This is mostly used within a note/chord defined as string. For example, within a
    Voice/Chain, a note could be defined as "C4:1/2:slur"; this will create a C4 eighth
    note which starts a slur. The slur will be extended until the end of the chain/voice or
    until another note defines a '~slur' attribute, which ends the slur (a '~' sign
    ends a previously open spanner of the same kind).

    Possible descriptors:

    * 'slur:dashed'
    * 'bracket:text=foo'
    * '<'
    * '>'
    * 'linespan:dotted:starttext=foo'
    * 'trill'
    * etc.

    Args:
        descr: a descriptor string

    Returns:
        the spanner

    Example
    -------

        >>> from maelzel.core import *
        >>> chain = Chain(...)
        >>> spanner = makeSpanner("trill")
        >>> spanner.bind(chain[0], chain[-1])

    """
    name, *rest = descr.split(":")
    name = name.lower()
    if name.startswith("~"):
        name = name[1:]
        kind = 'end'
    cls = _spannerNameToConstructor.get(name)
    if cls is None:
        raise ValueError(f"Spanner class {name} not understood. "
                         f"Possible spanners are {_spannerNameToConstructor.keys()}")
    kws = {}
    for part in rest:
        if part in {'solid', 'dashed', 'dotted', 'zigzag', 'trill', 'wavy'}:
            kws['linetype'] = part
        elif part in {'above', 'below'}:
            kws['placement'] = part
        elif '=' in part:
            k, v = part.split('=', maxsplit=2)
            if v == 'True' or v == 'False':
                v = bool(v)
            kws[k] = v
        else:
            raise ValueError(f"Spanner descriptor not understood: {part} ({descr})")
    if placement:
        kws['placement'] = placement
    if color:
        kws['color'] = color
    if linetype:
        kws['linetype'] = linetype
    spanner = cls(kind=kind, **kws)
    return spanner

# --------------------------------



class EventSymbol(Symbol):
    """
    Base-class for all event-attached symbols

    These are symbols attached to one event (note, chord, rest, clip, ...).
    The color and placement attributes do not apply for all symbols of this
    kind but we include it at this level to make the structure simpler

    Args:
        color (str): The color of the symbol.
        placement (str): The placement of the symbol. One of 'above', 'below' or ''
            to use the default placement
        noMergeNext: if True, do not allow an event with this symbol to be merged
            with another event
    """
    appliesToRests = True

    def __init__(self, color='', placement='', noMergeNext=False):
        super().__init__()
        assert not placement or placement in ('above', 'below')
        self.placement = placement
        self.noMergeNext = noMergeNext
        self.color = color

    def checkAnchor(self, anchor: mevent.MEvent) -> str:
        """Returns an error message if the event cannot add this symbol"""
        return ''

    def scoringAttachment(self) -> _attachment.Attachment:
        raise NotImplementedError("Class should implement this method or "
                                  "override applyToNotation")

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        if self.noMergeNext:
            n.mergeableNext = False
        attachment = self.scoringAttachment()
        n.addAttachment(attachment)


class Hidden(EventSymbol):
    """A hidden property can be attached to note to hide it"""
    exclusive = True
    applyToTieStrategy = 'all'

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Hidden()


class SizeFactor(EventSymbol):
    """Sets the size of an object (as a factor of default size)"""
    applyToTieStrategy = 'all'
    exclusive = True

    def __init__(self, size: float):
        super().__init__()
        self.size = size

    def __repr__(self):
        return f"SizeFactor({self.size})"

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.SizeFactor(size=self.size)


class Color(EventSymbol):
    """Customizes the color of an object"""
    exclusive = True
    applyToTieStrategy = 'all'

    def __init__(self, color: str):
        super().__init__()
        self.color = color

    def __repr__(self):
        return f"Color({self.color})"

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Color(self.color)


class NoteheadSymbol(Symbol):
    """Symbols attached to a notehead (a pitch)"""
    appliesToRests = False

    @abstractmethod
    def applyToPitch(self, n: scoring.Notation, idx: int | None, parent: mobj.MObj | None
                     ) -> None:
        raise NotImplementedError

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        for idx in range(len(n.pitches)):
            self.applyToPitch(n, idx=idx, parent=parent)


class PartMixin:
    """Symbols which can be attached to a voice"""

    applyToAllParts = True
    """Within a multipart voice, apply this symbol to all parts"""

    @abstractmethod
    def applyToPart(self, part: scoring.core.UnquantizedPart) -> None:
        raise NotImplementedError


# ----------------------------------------------------------

class BeamSubdivision(EventSymbol, PartMixin):
    """
    Customize beam subdivision
    
    The customization is applied to the beat starting at this event

    The beams of consecutive 16th (or shorter) notes are, by default, not subdivided.
    That is, the beams of more than two stems stretch unbroken over entire groups of notes.
    This behavior can be modified to subdivide the beams into sub-groups. Beams will be
    subdivided at intervals to match the metric value of the subdivision.

    .. note:: At the moment this is only supported by the lilypond backend

    Args:
        minimum: minimum limit of beam subdivision. A fraction or simply the denominator.
            1/8 indicates an eighth note, 16 indicates a 16th note
        maximum: maximum limit of beam subdivision. Similar to minimum
        
    """
    exclusive = True
    appliesToRests = True
    
    def __init__(self, minimum: int | F = 0, maximum: int | F = 0):
        super().__init__()
        self.minimum: F = minimum if isinstance(minimum, F) else F(1, minimum) if minimum > 0 else F(0)
        self.maximum: F = maximum if isinstance(maximum, F) else F(1, maximum) if maximum > 0 else F(0)
        
    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.BeamSubdivisionHint(minimum=self.minimum, maximum=self.maximum)
    
    def applyToPart(self, part: scoring.core.UnquantizedPart) -> None:
        att = _attachment.BeamSubdivisionHint(minimum=self.minimum, 
                                                     maximum=self.maximum, 
                                                     once=False) 
        part.notations[0].addAttachment(att)
        
        
class Clef(EventSymbol):
    """
    An explicit clef sign, applied prior to the event attached

    Args:
        kind: one of 'treble15', 'treble8', 'treble', 'bass', 'alto', 'bass8', 'bass15'

    Some clefs have alternate names:

    ========= ==================
    Clef       Aliases
    ========= ==================
    treble     violin, g
    bass       f
    alto       viola
    ========= ==================

    """
    exclusive = True
    appliesToRests = True

    def __init__(self, kind: str, color=''):
        super().__init__(color=color)
        clef = scoring.definitions.clefs.get(kind)
        if clef is None:
            raise ValueError(f"Clef {kind} unknown. Possible values: {scoring.definitions.clefs}")
        self.kind = clef

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Clef(self.kind, color=self.color)

    def __repr__(self):
        return _util.reprObj(self, priorityargs=('kind',), hideFalsy=True)


class Ornament(EventSymbol):
    """
    Note-attached ornament (trill, mordent, prall, etc.)

    Args:
        kind: one of 'trill', 'mordent', 'prall', 'turn'
            (see ``maelzel.scoring.definitions.avialableOrnaments``)

    """
    exclusive = False
    appliesToRests = False

    def __init__(self, kind: str, color=''):
        super().__init__(color=color)
        if kind not in scoring.definitions.availableOrnaments:
            raise ValueError(f"Ornament {kind} unknown. "
                             f"Possible values: {scoring.definitions.availableOrnaments}")
        self.kind = kind

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Ornament(self.kind, color=self.color)


class Tremolo(EventSymbol):
    """
    A stem-attached tremolo sign
    """

    exclusive = True
    appliesToRests = False

    def __init__(self, tremtype='single', nummarks: int = 2, relative=False, color=''):
        """
        Args:
            tremtype: the type of tremolo. 'single' indicates a repeated note/chord,
                'start' indicates the first of two alternating notes/chords,
                'end' indicates the second of two alternating notes/chords
            nummarks: how many tremolo marks (2=16th tremolo, 3=32nd tremolo, ...)
            relative: if True, the number of marks depends on the rhythmic figure
                to which the tremolo is attached. For example, if relative is True,
                a tremolo with nummarks 2 attached to an 8th note would result
                in a single-beam tremolo. If relative is False, nummarks will
                always determine the number of beams
        """
        super().__init__(color=color)
        assert tremtype in {'single', 'start', 'end'}, f'Unknown tremolo type: {tremtype}'
        self.tremtype = tremtype
        self.nummarks = nummarks
        self.relative = relative

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Tremolo(tremtype=self.tremtype, nummarks=self.nummarks, relative=self.relative, color=self.color)



class Fermata(EventSymbol):
    """A Fermata sign over a note"""
    exclusive = True
    appliesToRests = True

    def __init__(self, kind='normal', mergenext=True, color=''):
        """
        A fermata symbol over an event (note, rest, chord, ...)

        Args:
            kind: one of 'normal', 'square', 'angled', 'double-angled', 'double-square'
            mergenext: whether an event with a fermata can be merged with
                its right neighbour (if tied or two rests)
        """
        super().__init__(color=color, noMergeNext=not mergenext)
        self.kind = kind

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Fermata(kind=self.kind, color=self.color)


class Breath(EventSymbol):
    """
    A breathmark symbol, will also break the beam at the given instant

    The breathmark is applied prior to the event

    Args:
        kind: one of 'comma', 'varcomma', 'upbow', 'outsidecomma', 'caesura', 'chant'
            (see maelzel.scoring.definitions.breathMarks)
        visible: if False, the mark will not be shown in notation but will still have
            an effect on beaming
        horizontalPlacement: one of 'pre', 'post'. Indicates whether the break
            should be placed before or after the event
    """
    exclusive = True
    appliesToRests = True

    def __init__(self, kind='', visible=True, horizontalPlacement='pre'):
        super().__init__()
        self.visible = visible
        self.kind = kind
        self.horizontalPlacement = horizontalPlacement

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Breath(kind=self.kind, visible=self.visible, horizontalPlacement=self.horizontalPlacement)


class Text(EventSymbol):
    """
    A note attached text expression

    Args:
        text: the text
        placement: 'above', 'below' or None to leave it undetermined
        fontsize: the size of the text. The actual resulting size will depend
            on the backend used
        weight: one of 'normal', 'bold'
        italic: should this text be italic?
        color: a valid css color
        box: one of 'square', 'rectangle', 'circle' or '' to disable
        force: force the text to be displayed even if the event is tied
    """
    exclusive = False
    appliesToRests = True

    def __init__(self, text: str, placement='above', fontsize: float | None = None,
                 italic=False, weight='normal', box='',
                 color='', fontfamily='', force=False):
        assert fontsize is None or isinstance(fontsize, (int, float)), \
            f"Invalid fontsize: {fontsize}, type: {type(fontsize)}"
        _util.checkChoice('box', box, ('', 'square', 'rectangle', 'circle'))
        _util.checkChoice('weight', weight, ('normal', 'bold'))
        super().__init__(color=color, placement=placement)
        self.text = text
        self.fontsize = fontsize
        self.italic = italic
        self.weight = weight
        self.fontfamily = fontfamily
        self.box = box
        self.force = force

    def __repr__(self):
        return _util.reprObj(self, priorityargs=('text',), hideFalsy=True,
                             quoteStrings=True,
                             filter={'italic': lambda val: val,
                                     'weight': lambda val: val != 'normal'})

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Text(text=self.text, placement=self.placement,
                                       fontsize=self.fontsize, italic=self.italic,
                                       weight=self.weight, box=self.box,
                                       fontfamily=self.fontfamily)

    def __hash__(self):
        return hash((type(self).__name__, self.text, self.placement, self.fontsize, self.italic,
                     self.weight, self.fontfamily))


class Transpose(EventSymbol, PartMixin):
    """
    Apply a transposition for notation only

    Notice that this will not have any effect other than modify the pitch
    used for notation.

    .. seealso:: :class:`NotatedPitch`
    """

    exclusive = True
    applyToTieStrategy = 'all'
    appliesToRests = False

    def __init__(self, interval: str | float):
        super().__init__()
        if isinstance(interval, str):
            if interval[0] in '+-':
                semitones = _intervalToSemitones.get(interval[1:])
                if semitones is None:
                    raise ValueError(f"Invalid interval '{interval[1:]}', "
                                     f"possible intervals: {_intervalToSemitones.keys()}")
                if interval[0] == '-':
                    semitones = -semitones
                interval = semitones
            elif interval in _intervalToSemitones:
                interval = _intervalToSemitones[interval]
            else:
                raise ValueError(f"Invalid interval '{interval}', Valid transpositions "
                                 f"are {_intervalToSemitones.keys()}")
        if interval == 0:
            raise ValueError(f"Invalid transposition interval: {interval}")
        self.interval: float = interval

    def __repr__(self):
        return _util.reprObj(self, priorityargs=('interval',))

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        pitches = [pitch + self.interval for pitch in n.pitches]
        fixed = n.fixedNotenames.copy() if n.fixedNotenames else None
        n.setPitches(pitches)
        if fixed:
            for i, pitch in fixed.items():
                n.fixNotename(pt.transpose(pitch, self.interval), index=i)

    def applyToPart(self, part: scoring.core.UnquantizedPart) -> None:
        for notation in part.notations:
            if not notation.isRest:
                self.applyToNotation(notation, parent=None)


class NotatedPitch(NoteheadSymbol):
    """
    Allows to customize the notated pitch of a note

    This can be used both to fix the enharmonic variant or to display a different
    pitch altogether.

    Notice that this will not have any effect other than modify the pitch
    used for notation

    To apply a relative transposition, see :class:`Transpose`

    .. seealso:: :class:`Transpose`
    """
    exclusive = True
    applyToTieStrategy = 'all'
    appliesToRests = False

    def __init__(self, pitch: str):
        super().__init__()
        self.pitch = pitch

    def __repr__(self):
        return _util.reprObj(self, priorityargs=('pitch',), hideEmptyStr=True)

    def applyToPitch(self, n: scoring.Notation, idx: int | None, parent: mobj.MObj | None
                     ) -> None:
        #if type(parent).__name__ != 'Note':
        #    raise TypeError(f"Expected a Note, got {parent}")

        if idx is None:
            from maelzel.core import event
            assert isinstance(parent, event.Note)
            pitch = parent.pitch
            idx = min(range(len(n.pitches)), key=lambda idx: abs(pitch - n.pitches[idx]))

        n.fixNotename(notename=self.pitch, index=idx)


_intervalToSemitones = {
    '2nd': 2,
    '2m': 1,
    '2M': 2,
    '3rd': 4,
    '3m': 3,
    '3M': 4,
    '4': 5,
    '4th': 5,
    '5': 7,
    '5th': 7,
    '6m': 8,
    '6M': 9,
    '7m': 10,
    '7M': 11,
    '8': 12,
    '8a': 12,
    '8va': 12,
    '15': 24,
    '15a': 24,
    '15va': 24,
    '8b': -12,
    '15b': -24,
}


class Harmonic(EventSymbol):
    """
    A natural/artificial harmonic, or a sound-pitch flageolet

    In the case of a natural or artificial harmonic, the notated pitch
    is the "action" pitch. In the case of a natural harmonic, for a string
    instrument this means the pitch of the node to lightly touch.
    Fon an attificial harmonic the note to which this symbol is attached
    identifies the pressed pitch, the interval will determine the
    node to touch (for a 4th harmonic the interval should be 5 since the
    4th is 5 semitones above of the pressed pitch)
    A flageolet is a harmonic where the written pitch indicates the
    sounding pitch, and the means of execution is left for the player
    to determine.

    If the interval is given then an artificial harmonic is assumed.

    Args:
        kind: either 'natural', 'artificial' or an interval as string
            ('4th' is a perfect fourth, '3M' is a major third). In this last
            case the kind is set to artificial and the interval is set
            to this value
        interval: the interval for artificial harmonics. If set, the
            kind is set to 'artificial'


    =============  ========== =========
    Interval        semitones  String
    =============  ========== =========
    Perfect fifth   7           5th
    Perfect fourth  5           4th
    Major third     4           3M
    Minor third     3           3m
    Major second    2           2 or 2M
    =============  ========== =========

    Example
    ~~~~~~~

        # A string 4th (2 octave higher) harmonic
        >>> symbols.Harmonic('4th')
        # A flageolet, where the written note indicates the sounding pitch
        >>> symbols.Harmonic('natural')
        # A touch harmonic, where the written note indicates where to slightly
        # touch the string. The interval is left as 0
        >>> symbols.Harmonic('artificial')
    """
    applyToRests = False

    def __init__(self, kind='natural', interval: int | str = 0):
        super().__init__()

        if kind in _intervalToSemitones:
            assert interval == 0
            semitone = _intervalToSemitones[kind]
            kind = 'artificial'
        elif interval == 0:
            semitone = 0
        else:
            kind = 'artificial'
            if isinstance(interval, str):
                semitone = _intervalToSemitones.get(kind)
                if semitone is None:
                    raise ValueError(f"Invalid interval, expected one of {_intervalToSemitones.keys()}")
            else:
                semitone = interval

        assert kind in ('natural', 'artificial')
        self.kind: str = kind
        self.interval: int = semitone

    def scoringAttachment(self) -> _attachment.Attachment:
        if self.kind == 'natural':
            return _attachment.Articulation('flageolet')
        else:
            return _attachment.Harmonic(self.interval)

    def applyToTiedGroup(self, notations: Sequence[scoring.Notation], parent: mobj.MObj | None
                         ) -> None:
        if self.kind == 'natural':
            self.applyToNotation(notations[0], parent=parent)
        else:
            for n in notations:
                n.addAttachment(_attachment.Harmonic(self.interval))


class Notehead(NoteheadSymbol):
    """
    Customizes the notehead shape, color, parenthesis and size

    Args:
        shape: one of 'cross', 'harmonic', 'triangleup', 'xcircle',
              'triangle', 'rhombus', 'square', 'rectangle'. You can also add parenthesis
              to the shape, as  '(x)' or even '()' to indicate a normal, parenthesized
              notehead
        color: a css color (str)
        parenthesis: if True, parenthesize the notehead
        size: a size factor (1.0 means the size corresponding to the staff size, 2. indicates
            a notehead twice as big)
    """

    exclusive = False
    applyToTieStrategy = 'all'
    appliesToRests = False

    def __init__(self, shape='', color='', parenthesis=False,
                 size: float | None = None, hidden=False):
        super().__init__()
        self.hidden = hidden
        if shape and shape.endswith('?'):
            parenthesis = True
            shape = shape[:-1] if len(shape) > 1 else 'normal'
        elif shape and shape[0] == '(' and shape[-1] == ')':
            shape = shape[1:-1]
            if not shape:
                shape = 'normal'
            parenthesis = True
        elif shape == 'hidden':
            shape = ''
            self.hidden = True
        if shape:
            shape2 = scoring.definitions.normalizeNoteheadShape(shape)
            assert shape2, (f"Notehead '{shape}' unknown. Possible noteheads: "
                            f"{scoring.definitions.noteheadShapes}")
            shape = shape2
        self.shape = shape
        self.color = color
        self.parenthesis = parenthesis
        self.size = size

    def __hash__(self):
        return hash((type(self).__name__, self.shape, self.color, self.parenthesis, self.size))

    def __repr__(self):
        return _util.reprObj(self, priorityargs=('shape,'), hideFalsy=True)

    def asScoringNotehead(self) -> scoring.definitions.Notehead:
        return scoring.definitions.Notehead(shape=self.shape, color=self.color, size=self.size,
                                            parenthesis=self.parenthesis, hidden=self.hidden)

    def applyToPitch(self, n: scoring.Notation, idx: int | None, parent: mobj.MObj | None
                     ) -> None:
        # if idx is None, apply to all noteheads
        scoringNotehead = self.asScoringNotehead()
        n.setNotehead(scoringNotehead, index=idx, merge=True)


def iscolor(s: str) -> bool:
    return (re.match(r"^#([a-fA-F0-9]{6}|[a-fA-F0-9]{3})$", s)) is not None or s in colortheory.cssColors()


def makeKnownSymbol(name: str) -> Symbol | None:
    """
    Create a symbol from a known name

    Args:
        name: the name of the symbol ('accent', 'fermata', 'mordent', etc.). Supported
            are all articulations, ornaments, breath marks ('comma', 'caesura'),
            colors, css colors, 'fermata'.

    Returns:
        the created Symbol, or None if the name cannot be interpreted
        as a Symbol. No Exceptions are thrown

    This is mostly used internally to add symbols to notes/chords which are defined
    as strings. For example, a note defined as "C4:1:accent", will result in a
    C4 note with a duration of 1 (quarternotes) and an Accent symbol.

    """
    name = name.lower()
    if name in scoring.definitions.allArticulations():
        return Articulation(name)

    if name in scoring.definitions.availableOrnaments:
        return Ornament(name)

    if name in ('comma', 'caesura'):
        return Breath(name)

    if name == 'break':
        return BeamBreak()

    if name == 'tremolo':
        return Tremolo()

    if iscolor(name):
        return Color(name)

    if name == 'fermata':
        return Fermata()

    if name == 'nomerge':
        return NoMerge()

    return None


@cache
def knownSymbols() -> set[str]:
    out = set()
    out |= scoring.definitions.allArticulations()
    out |= scoring.definitions.availableOrnaments
    out |= {'comma', 'caesura', 'fermata', 'break'}
    return out


class Dynamic(EventSymbol):
    """
    A notation only dynamic

    This should only be used for the case where a note should
    have a dynamic only for display
    """
    exclusive = True
    appliesToRests = False

    def __init__(self, kind: str, force=False, placement=''):
        if kind not in scoring.definitions.validDynamics():
            raise ValueError(f"Invalid dynamic '{kind}', "
                             f"possible dynamics: {scoring.definitions.validDynamics()}")
        super().__init__(placement=placement)
        self.kind = kind
        self.force = force

    def __hash__(self):
        return hash((type(self).__name__, self.kind, self.placement))
    
    def __repr__(self):
        return _util.reprObj(self, hideKeys=('kind',), hideFalsy=True)

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        n.dynamic = self.kind
        if self.force:
            n.dynamic += '!'


class Articulation(EventSymbol):
    """
    Represents a note attached articulation
    """
    exclusive = False
    appliesToRests = False

    def __init__(self, kind: str, color='', placement=''):
        super().__init__(color=color, placement=placement)

        normalized = scoring.definitions.normalizeArticulation(kind)
        assert normalized, f"Articulation {kind} unknown. Possible values: " \
                           f"{scoring.definitions.articulations}"
        self.kind = normalized

    def __hash__(self):
        return hash((type(self).__name__, self.kind, self.color, self.placement))

    def __repr__(self):
        return _util.reprObj(self, hideKeys=('kind',), hideFalsy=True)

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Articulation(kind=self.kind, color=self.color, placement=self.placement)

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        if n.isRest:
            logger.warning(f"Cannot apply {self} to a rest: {n}")
        else:
            if n.tiedPrev:
                logger.debug(f"Applying articulation {self} to {n}, even if it is tied to prev")
            super().applyToNotation(n=n, parent=parent)


class Fingering(EventSymbol):
    """
    A fingering attached to a note
    """
    exclusive = True
    appliesToRests = False

    def __init__(self, finger: str):
        super().__init__()
        self.finger = finger

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Fingering(self.finger)


class Bend(EventSymbol):
    """
    A (post) bend with an alteration up or down

    A bend is a modification of the pitch starting at the pitch
    of the note and going in a curve up or down a number of semitones

    Attributes:
        alter: the alteration of the bend, in semitones (> 0 is an upward bend)
    """
    exclusive = True
    appliesToRests = False

    def __init__(self, alter: float):
        super().__init__()
        self.alter = alter

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Bend(self.alter)


class NoMerge(EventSymbol):
    """
    A note marked with this symbol will not be merged to a previous note

    This is true even if the note is tied to the previous and the
    symbolic durations, after quantization, are compatible to be
    merged into a longer duration
    
    Args:
        prev: if True, this cannot be merged to the previous note
        next: if True, this cannot be merged to the next note
    """
    appliesToRests = True

    def __init__(self, prev=True, next=False):
        super().__init__()
        self.prev = prev
        self.next = next

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        if self.prev:
            n.mergeablePrev = False
        if self.next:
            n.mergeableNext = False


class Stem(EventSymbol):
    """
    Customize the stem of a note

    Attributes:
        hidden: if True, the stem will be hidden
    """
    appliesToRests = False

    def __init__(self, hidden: bool = False, color=''):
        super().__init__(color=color)
        self.hidden = hidden

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        if self.hidden or self.color:
            n.addAttachment(_attachment.StemTraits(hidden=self.hidden, color=self.color))


class Gracenote(EventSymbol):
    """
    Customizes properties of a gracenote
    """

    def __init__(self, slash=False, value=F(1, 2)):
        super().__init__()
        self.slash = slash
        self.value = value

    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.GracenoteProperties(slash=self.slash, value=self.value)

    def checkAnchor(self, anchor: mevent.MEvent) -> str:
        if anchor.dur != 0:
            return f'A {type(self)} can only be added to a gracenote, got {anchor}'
        return ''


class GlissProperties(EventSymbol):
    """
    Customizes properties of a glissando

    It can only be attached to an event which starts a glissando

    Args:
        linetype: the linetype to use, one of 'solid', 'dashed', 'dotted', 'trill'
        color: the color of the line
        hidden: if True the glissando is not represented as notation. When
            applied to a Notation, the .gliss attribute of the Notation is
            set to false.

    """
    exclusive = True
    appliesToRests = False

    def __init__(self, linetype='solid', color='', hidden=False):
        super().__init__()
        _util.checkChoice('linetype', linetype, _attachment.GlissProperties.linetypes)
        self.linetype = linetype
        self.color = color
        self.hidden = hidden

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        if self.hidden:
            n.gliss = False
            return
        elif not n.gliss and not (n.tiedPrev and n.tiedNext):
            raise ValueError("Cannot apply GlissProperties to a Notation without glissando,"
                             f"(destination: {n})")
        n.addAttachment(_attachment.GlissProperties(linetype=self.linetype, color=self.color))

    def checkAnchor(self, anchor: mevent.MEvent) -> str:
        return f'This event ({self}) has no glissando' if not anchor.gliss else ''


class Accidental(NoteheadSymbol):
    """
    Customizes the accidental of a note

    An accidental applies to a note/chord and is bound to the note/chord. Within
    a chord any accidental attached to its individual notes is ...

    TODO: clear what happends with symbols within a chord

    Args:
        hidden: is this accidental hidden?
        parenthesis: put this accidental between parenthesis?
        color: the color of this accidental
        force: if True, force the accidental to be shown
        size: a size factor
    """
    exclusive = True
    appliesToRests = False

    def __init__(self,
                 hidden=False,
                 parenthesis=False,
                 color='',
                 force=False,
                 size: float | None = None):
        super().__init__()

        self.hidden: bool = hidden
        self.parenthesis: bool = parenthesis
        self.color: str = color
        self.force: bool = force
        self.size: float | None = size

    def __repr__(self):
        return _util.reprObj(self, hideFalse=True, hideEmptyStr=True)

    def __hash__(self):
        return hash((type(self).__name__, self.hidden, self.parenthesis, self.color, self.force))

    def applyToPitch(self, n: scoring.Notation, idx: int | None, parent: mobj.MObj | None
                     ) -> None:
        if n.isRest:
            return
        attachment = _attachment.AccidentalTraits(color=self.color,
                                                         hidden=self.hidden,
                                                         parenthesis=self.parenthesis,
                                                         force=self.force,
                                                         size=self.size)
        n.addAttachment(attachment, pitchanchor=idx)


# -------------------------------------------------------------------


class QuantHint(EventSymbol):
    def __init__(self, division: tuple[int, ...], strength=100.):
        super().__init__()
        self.division = division
        self.strength = strength

    def applyToNotation(self, n: scoring.Notation, parent: mobj.MObj | None) -> None:
        n.addAttachment(_attachment.QuantHint(division=self.division, strength=self.strength))


class BeamBreak(EventSymbol):
    """
    Symbolizes a beam break at the given location 

    A BeamBreak can be added both to a Note/Chord or to a Voice via ``.addSymbolAt``. 
    
    .. note::

        When a BeamBreak is added to a Part this does not modify in any way the
        contents of the Voice. The modification takes place at quantization. 
        The resulting Notation present at the given location is broken at that
        point (adding a tie if needed)
    """
    def scoringAttachment(self) -> _attachment.Attachment:
        return _attachment.Breath(visible=False)


def parseAddSymbol(args: tuple, kws: dict) -> Symbol:
    """
    Parses the input of addSymbol

    .. seealso:: :meth:`maelzel.core.mobj.mobj.MObj.addsymbol`

    Args:
        args: args as passed to event.addSymbol
        kws: keywords as passed to event.addSymbol

    Returns:
        a list of Symbols

    """
    if len(args) == 2 and all(isinstance(arg, str) for arg in args) and not kws:
        symbolclass, kind = args
        symboldef = makeSymbol(symbolclass, kind)
    elif len(args) == 1:
        arg = args[0]
        if isinstance(arg, str):
            if not kws:
                symboldef = makeKnownSymbol(arg)
            else:
                symboldef = makeSymbol(arg, **kws)
            if not symboldef:
                raise ValueError(f"{arg} is not a known symbol. Known symbols are: {knownSymbols()}")
        elif not kws and isinstance(arg, Symbol):
            symboldef = arg
        elif isinstance(arg, type) and issubclass(arg, Symbol):
            symboldef = arg()
        else:
            raise ValueError(f"Could not add a symbol with {args=}, {kws=}")
    elif kws:
        symboldef = next(makeSymbol(key, value) for key, value in kws.items())
    else:
        raise ValueError(f"Could not add a symbol with {args=}, {kws=}")
    return symboldef


# -------------------------------

_symbolClasses = (
    Notehead,
    Articulation,
    Text,
    SizeFactor,
    Color,
    Accidental,
    Ornament,
    Fermata,
    NotatedPitch,
    Slur,
    Hairpin,
    LineSpan,
    TrillLine,
    NoteheadLine,
    Bend,
    Fingering,
    Harmonic,
    Breath,
    BeamBreak,
    Stem,
    GlissProperties,
    Gracenote,
    Transpose
)

_voiceSymbols = (
    Transpose,
    
)


_symbolNameToClass = {cls.__name__.lower(): cls for cls in _symbolClasses}


def makeSymbol(clsname: str, *args, **kws) -> Symbol:
    """
    Construct a Symbol from the symbol name and any values and/or keywords passed

    Args:
        clsname: the symbol name (case independent)
        *args: any other arguments passed to the constructor
        **kws: any keywords passed to the constructor

    Returns:
        the created Symbol

    """
    cls = _symbolNameToClass.get(clsname.lower())
    if cls is None:
        raise ValueError(f"Class '{clsname}' unknown. "
                         f"Known symbol names: {list(_symbolNameToClass.keys())}")
    return cls(*args, **kws)
