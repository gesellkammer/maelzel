"""
Symbols are objects which can be attached to a note/chord to modify its notation

Most symbols do not have any other meaning than to hint the backend used for
notation to display the object in a certain way. For example a Notehead symbol
can be attached to a note to modify the notehead shape used.

Dynamics are included as Symbols but they are deprecated, since dynamics
can be used for playback (see :ref:`config_play_usedynamics`)

"""
from __future__ import annotations
import random
import copy
import weakref
# from emlib.misc import ReprMixin
from maelzel import scoring
from ._common import logger
import pitchtools as pt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Set, Sequence, Any, TypeVar
    TSpanner = TypeVar('TSpanner', bound='Spanner')
    from maelzel.core import event

_uuid_alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'


def makeuuid(size=8) -> str:
    return ''.join(random.choices(_uuid_alphabet, k=size))


class Symbol:
    """Base class for all symbols"""
    exclusive = False
    applyToTieStrategy = 'first'
    appliesToRests = True
    modifiesScoringContext = False

    def __init__(self):
        self.properties: dict[str, Any] | None = None

    def getProperty(self, key):
        return None if not self.properties else self.properties.get(key)

    def setProperty(self, key: str, value):
        if self.properties is None:
            self.properties = {}
        self.properties[key] = value

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    def applyTo(self, n: scoring.Notation) -> None:
        """Apply this symbol to the given notation, in place"""
        raise NotImplementedError

    def applyToTiedGroup(self, notations: Sequence[scoring.Notation]) -> None:
        if self.applyToTieStrategy == 'all':
            for n in notations:
                self.applyTo(n)
        elif self.applyToTieStrategy == 'first':
            self.applyTo(notations[0])


class Property(Symbol):
    """
    A property is a modifier of an object itself (like a color or a size)
    """
    exclusive = True
    scopes = set()

    def __init__(self, value, scope=''):
        super().__init__()
        self.value = value
        if self.scopes:
            assert scope in self.scopes
        self.scope = scope

    def __hash__(self):
        return hash((type(self).__name__, self.value))

    def __repr__(self):
        cls = type(self).__name__
        return f'{cls}(value={self.value})'


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

    def __init__(self, kind='start', uuid: str = '', linetype='solid',
                 placement='', color=''):
        super().__init__()
        assert kind == 'start' or kind == 'end', f"got kind={kind}"
        assert linetype in {'', 'solid', 'dashed', 'dotted', 'wavy', 'trill', 'zigzag'}, f"got {linetype}"
        if placement:
            assert placement == 'above' or placement == 'below'
        self.kind = kind
        self.uuid = uuid or makeuuid(8)
        self.linetype = linetype
        self.placement = placement
        self.color = color

        self.anchor: weakref.ReferenceType[event.MEvent] | None = None
        """The event to which this spanner is anchored to"""

        self.partnerSpanner: weakref.ReferenceType[Spanner] | None = None
        """The partner spanner"""

    def _attrs(self) -> dict:
        keys = ('kind', 'uuid', 'linetype', 'placement', 'color')
        return {k: v for k in keys
                if (v := getattr(self, k))}

    def __repr__(self) -> str:
        cls = type(self).__qualname__
        attrstr = ', '.join(f'{k}={v}' for k, v in self._attrs().items())
        return f'{cls}({attrstr})'

    def setAnchor(self, obj: event.MEvent) -> None:
        """
        Set the anchor for this spanner.

        This is called by :meth:``MusicObj.setSymbol`` or by :meth:``Spanner.bind`` to
        set the anchor of this spanner. A User should not normally call this method

        Args:
            obj: the object this spanner is anchored to, either as start or end
        """
        self.anchor = weakref.ref(obj)

    def bind(self, startobj: event.MEvent, endobj: event.MEvent) -> None:
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
        self.makeEndSpanner(anchor=endobj)
        assert self.partnerSpanner is not None

    def makeEndSpanner(self: TSpanner, anchor: event.MEvent = None) -> TSpanner:
        """
        Creates the end spanner for an already existing start spanner

        start and end spanner share the same uuid and have a weakref to
        each other, allowing each of the spanners to access their
        twin. As each spanner also holds a weak ref to their anchor,
        the anchored events can be made aware of the each other.

        Args:
            anchor: the event to which the end spanner is anchored to.

        Returns:
            the created spanner. This is a copy of the start spanner in every
            way with the only difference that ``kind='end'``

        """
        if anchor and self.anchor is anchor:
            raise ValueError("Start anchor and end anchor cannot be the same object")
        endSpanner = copy.copy(self)
        endSpanner.kind = 'end'
        endSpanner.partnerSpanner = weakref.ref(self)
        self.partnerSpanner = weakref.ref(endSpanner)
        if anchor:
            anchor.addSymbol(endSpanner)
        return endSpanner


class TrillLine(Spanner):

    def __init__(self,
                 kind='start',
                 startmark='trill',
                 trillpitch='',
                 alteration='',
                 placement='above', uuid=''):
        super().__init__(kind=kind, placement=placement, uuid=uuid)
        self.startmark = startmark
        self.trillpitch = trillpitch
        self.alteration = alteration

    def applyTo(self, n: scoring.Notation) -> None:
        spanner = scoring.spanner.TrillLine(kind=self.kind, uuid=self.uuid,
                                            startmark=self.startmark,
                                            alteration=self.alteration,
                                            trillpitch=self.trillpitch,
                                            placement=self.placement)
        n.addSpanner(spanner)


class NoteheadLine(Spanner):
    appliesToRests = False

    """
    A line conecting two noteheads
    """
    def __init__(self, kind='start', uuid='', color='', linetype='solid', text=''):
        super().__init__(kind=kind, uuid=uuid, color=color, linetype=linetype)
        self.text = text

    def applyTo(self, n: scoring.Notation) -> None:
        spanner = scoring.spanner.Slide(kind=self.kind, uuid=self.uuid,
                                        color=self.color, linetype=self.linetype,
                                        text=self.text)
        n.addSpanner(spanner)


class OctaveShift(Spanner):
    modifiesScoringContext = True

    def __init__(self, kind='start', octaves=1, uuid=''):
        assert octaves != 0 and abs(octaves) <= 3
        super().__init__(kind=kind, placement='above' if octaves >= 0 else 'below', uuid=uuid)
        self.octaves = octaves

    def applyTo(self, n: scoring.Notation) -> None:
        spanner = scoring.spanner.OctaveShift(kind=self.kind, octaves=self.octaves,
                                              uuid=self.uuid)
        n.addSpanner(spanner)


class Slur(Spanner):

    def applyTo(self, n: scoring.Notation) -> None:
        slur = scoring.spanner.Slur(kind=self.kind, uuid=self.uuid, linetype=self.linetype)
        n.addSpanner(slur)


class Beam(Spanner):
    appliesToRests = False

    def applyTo(self, n: scoring.Notation) -> None:
        spanner = scoring.spanner.Beam(kind=self.kind, uuid=self.uuid)
        n.addSpanner(spanner)


class Hairpin(Spanner):
    """
    A hairpin crescendo or decrescendo

    Args:
        direction: one of "<" or ">"
        niente: if True, add a niente 'o' to the start or end of the hairpin
    """
    def __init__(self, direction, niente=False, kind='start', uuid='', placement='', linetype=''):
        super().__init__(kind=kind, uuid=uuid, placement=placement, linetype=linetype)
        assert direction == "<" or direction == ">"
        self.direction = direction
        self.niente = niente

    def _attrs(self):
        attrs = {'direction': self.direction}
        attrs.update(super()._attrs())
        return attrs

    def applyTo(self, n: scoring.Notation) -> None:
        hairpin = scoring.spanner.Hairpin(kind=self.kind, uuid=self.uuid,
                                          direction=self.direction,
                                          niente=self.niente,
                                          placement=self.placement)
        n.addSpanner(hairpin)


class Bracket(Spanner):
    def __init__(self, kind='start', uuid: str = '', linetype='solid', placement='',
                 text=''):
        super().__init__(kind=kind, uuid=uuid, linetype=linetype, placement=placement)
        self.text = text

    def applyTo(self, n: scoring.Notation) -> None:
        bracket = scoring.spanner.Bracket(kind=self.kind, uuid=self.uuid,
                                          text=self.text, placement=self.placement,
                                          linetype=self.linetype)
        n.addSpanner(bracket)


class LineSpan(Spanner):

    def __init__(self, kind='start', uuid: str = '', linetype='solid', placement='',
                 starttext='', endtext='', middletext='', verticalAlign='',
                 starthook=False, endhook=False):
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

    def applyTo(self, n: scoring.Notation) -> None:
        spanner = scoring.spanner.LineSpan(kind=self.kind, uuid=self.uuid,
                                           placement=self.placement, linetype=self.linetype,
                                           starttext=self.starttext, endtext=self.endtext,
                                           middletext=self.middletext,
                                           verticalAlign=self.verticalAlign,
                                           starthook=self.starthook,
                                           endhook=self.endhook)
        n.addSpanner(spanner)


_spannerNameToConstructor: dict[str] = {
    'slur': Slur,
    'line': LineSpan,
    'linespan': LineSpan,
    'trill': TrillLine,
    'tr': TrillLine,
    'bracket': Bracket,
    '<': lambda **kws: Hairpin(direction='<', **kws),
    '>': lambda **kws: Hairpin(direction='>', **kws),
    'hairpin': Hairpin,
    'beam': Beam
}


def makeSpanner(descr: str, kind='start') -> Spanner:
    """
    Create a spanner from a descriptor

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
    cls = _spannerNameToConstructor.get(name)
    if cls is None:
        raise ValueError(f"Spanner class {cls} not understood. "
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
    spanner = cls(kind=kind, **kws)
    return spanner

# --------------------------------



class SizeFactor(Property):
    """Sets the size of an object (as a factor of default size)"""
    scopes = {'note'}
    applyToTieStrategy = 'all'

    def __init__(self, size: float, scope='note'):
        super().__init__(value=size, scope=scope)

    def applyTo(self, n: scoring.Notation) -> None:
        if self.scope == 'note':
            n.sizeFactor = self.value


class Color(Property):
    """Customizes the color of an object"""
    scopes = {'note'}

    def __init__(self, color: str, scope='note'):
        super().__init__(value=color, scope=scope)

    def applyTo(self, n: scoring.Notation) -> None:
        if self.scope == 'note':
            n.color = self.value


class Hidden(Property):
    def __init__(self, scope='note'):
        super().__init__(value=True, scope=scope)

    def applyTo(self, n: scoring.Notation) -> None:
        n.addAttachment(scoring.attachment.Property('hidden', value=True))


class NoteAttachedSymbol(Symbol):
    """Base-class for all note attached symbols"""

    @classmethod
    def possibleValues(cls, key: str = None) -> Set[str] | None:
        return None


class PitchAttachedSymbol(NoteAttachedSymbol):
    appliesToRests = False

    def applyToPitch(self, n: scoring.Notation, idx: int = None):
        raise NotImplementedError

    def applyTo(self, n: scoring.Notation) -> None:
        self.applyToPitch(n, idx=None)


class Ornament(NoteAttachedSymbol):
    exclusive = False
    appliesToRests = False

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    @classmethod
    def possibleValues(cls, key: str = None) -> Set[str] | None:
        if key is None or key == 'kind':
            return scoring.definitions.availableOrnaments

    def applyTo(self, n: scoring.Notation) -> None:
        n.addAttachment(scoring.attachment.Ornament(self.kind))


class Tremolo(Ornament):
    exclusive = True
    appliesToRests = False

    def __init__(self, tremtype='single', nummarks: int = 2):
        """
        Args:
            tremtype: the type of tremolo. 'single' indicates a repeated note/chord,
                'start' indicates the first of two alternating notes/chords,
                'end' indicates the second of two alternating notes/chords
            nummarks: how many tremolo marks (2=16th tremolo, 3=32nd tremolo, ...)
        """
        super().__init__(kind='tremolo')
        assert tremtype in {'single', 'start', 'end'}, f'Unknown tremolo type: {tremtype}'
        self.tremtype = tremtype
        self.nummarks = nummarks

    def applyTo(self, n: scoring.Notation) -> None:
        n.addAttachment(scoring.attachment.Tremolo(tremtype=self.tremtype,
                                                   nummarks=self.nummarks))


class Fermata(NoteAttachedSymbol):
    exclusive = True
    appliesToRests = True

    def __init__(self, kind: str = 'normal'):
        super().__init__()
        self.kind = kind

    @classmethod
    def possibleValues(cls, key: str = None) -> Set[str] | None:
        if key is None or key == 'kind':
            return scoring.definitions.availableFermatas

    def applyTo(self, n: scoring.Notation) -> None:
        n.addAttachment(scoring.attachment.Fermata(kind=self.kind))


class BeamBreak(NoteAttachedSymbol):
    """
    An invisible breathmark, only useful to break a beam
    """
    def applyTo(self, n: scoring.Notation) -> None:
        n.addAttachment(scoring.attachment.Breath(visible=False))


class Breath(NoteAttachedSymbol):
    """
    A breathmark symbol, will also break the beam at the given instant

    Args:
        kind: one of 'comma', 'varcomma', 'upbow', 'outsidecomma', 'caesura', 'chant'
            (see maelzel.scoring.definitions.breathMarks)
        visible: if False, the mark will not be shown in notation but will still have
            an effect on beaming
    """
    exclusive = True
    appliesToRests = True

    def __init__(self, kind='', visible=True):
        super().__init__()
        self.visible = visible
        self.kind = kind

    def applyTo(self, n: scoring.Notation) -> None:
        n.addAttachment(scoring.attachment.Breath(kind=self.kind, visible=self.visible))


class Text(NoteAttachedSymbol):
    """
    A note attached text expression

    Args:
        text: the text
        placement: 'above', 'below' or None to leave it undetermined
        fontsize: the size of the text. The actual resulting size will depend
            on the backend used
        fontstyle: None or one of 'italic', 'bold' or a comma separated string such
            as 'italic,bold'

    """
    exclusive = False
    appliesToRests = True

    def __init__(self, text: str, placement='above', fontsize: float = None,
                 fontstyle: str = None, box: str | bool = False):
        assert fontsize is None or isinstance(fontsize, (int, float)), \
            f"Invalid fontsize: {fontsize}, type: {type(fontsize)}"
        super().__init__()
        self.text = text
        self.placement = placement
        self.fontsize = fontsize
        self.fontstyle = fontstyle
        self.box = box

    def __repr__(self):
        return f"Text('{self.text}', placement={self.placement})"

    def applyTo(self, n: scoring.Notation) -> None:
        if not n.tiedPrev:
            # TODO: add fontsize
            n.addText(self.text, placement=self.placement, fontsize=self.fontsize,
                      fontstyle=self.fontstyle, box=self.box)

    def __hash__(self):
        return hash((type(self).__name__, self.text, self.placement, self.fontsize, self.fontstyle))


class NotatedPitch(NoteAttachedSymbol):
    """
    Allows to customize the notated pitch of a note
    """
    exclusive = True
    applyToTieStrategy = 'all'
    appliesToRests = False

    def __init__(self, notename: str, realpitch: float = 0):
        super().__init__()

        self.notename = notename
        self.realpitch = realpitch

    def __repr__(self):
        return f'NotatedPitch(notename={self.notename}, realpitch={self.realpitch})'

    def applyTo(self, n: scoring.Notation) -> None:
        if len(n.pitches) == 1:
            n.fixNotename(self.notename)
        else:
            pitchthreshold = 0.04
            if not self.realpitch:
                m = pt.n2m(self.notename)
                nearestidx = min(range(len(n.pitches)),
                                 key=lambda idx: abs(m - n.pitches[idx]))
                if abs(m - n.pitches[nearestidx]) < pitchthreshold:
                    n.fixNotename(self.notename, nearestidx)
                else:
                    logger.error(f"Could not fix notename {self.notename} for chord {n.pitches}")

            for i, pitch in enumerate(n.pitches):
                if abs(pitch - self.realpitch) < pitchthreshold:
                    n.fixNotename(self.notename, i)


class Harmonic(NoteAttachedSymbol):
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
        kind: one of natural, artificial or sounding. A sounding harmonic
            (flageolet) will be notated with a small 'o'.
        interval: the interval between the node touched and the pitch
            depressed, only needed for artificial harmonics
    """
    applyToRests = False

    def __init__(self, kind='natural', interval=0):
        super().__init__()
        if interval > 0:
            kind = 'artificial'

        assert kind in {'natural', 'artificial', 'sounding'}
        if kind == 'artificial':
            assert interval >= 1
        else:
            interval = 0
        self.kind = kind
        self.interval = interval

    def __repr__(self):
        return f'Harmonic(kind={self.kind}, interval={self.interval})'

    def applyTo(self, n: scoring.Notation) -> None:
        if self.kind == 'sounding':
            if not n.tiedPrev:
                n.addArticulation('flageolet')
        else:
            n.addAttachment(scoring.attachment.Harmonic(self.interval))
            # notation.addHarmonic(self.kind, interval=self.interval)

    def applyToTiedGroup(self, notations: Sequence[scoring.Notation]) -> None:
        if self.kind == 'sounding':
            self.applyTo(notations[0])
        else:
            for n in notations:
                n.addAttachment(scoring.attachment.Harmonic(self.interval))


class Notehead(PitchAttachedSymbol):
    """
    Customizes the notehead shape, color, parenthesis and size

    Args:
        shape: one of 'cross', 'harmonic', 'triangleup', 'xcircle',
              'triangle', 'rhombus', 'square', 'rectangle'
        color: a css color (str)
        parenthesis: if True, parenthesize the notehead
        size: a size factor (1.0 means the size corresponding to the staff size, 2. indicates
            a notehead twice as big)
    """
    exclusive = False
    applyToTieStrategy = 'all'
    appliesToRests = False

    def __init__(self, shape='', color='', parenthesis=False,
                 size: float = None, hidden=False):
        super().__init__()
        self.hidden = hidden
        if shape and shape.endswith('?'):
            parenthesis = True
            shape = shape[:-1]
        elif shape == 'hidden':
            shape = ''
            self.hidden = True
        if shape:
            shape2 = scoring.definitions.normalizeNoteheadShape(shape)
            assert shape2, f"Notehead '{shape}' unknown. Possible noteheads: " \
                          f"{scoring.definitions.noteheadShapes}"
            shape = shape2
        self.shape = shape
        self.color = color
        self.parenthesis = parenthesis
        self.size = size

    def __hash__(self):
        return hash((type(self).__name__, self.shape, self.color, self.parenthesis, self.size))

    @classmethod
    def possibleValues(cls, key: str = None) -> Set[str] | None:
        if key is None or key == 'shape':
            return scoring.definitions.noteheadShapes

    def __repr__(self):
        parts = []
        if self.shape:
            parts.append(self.shape)
        if self.color:
            parts.append(f'color={self.color}')
        if self.parenthesis:
            parts.append(f'parenthesis=True')
        if self.size is not None and self.size != 1.0:
            parts.append(f'size={self.size:.6g}')
        return f"Notehead({', '.join(parts)})"

    def asScoringNotehead(self) -> scoring.definitions.Notehead:
        return scoring.definitions.Notehead(shape=self.shape, color=self.color, size=self.size,
                                            parenthesis=self.parenthesis, hidden=self.hidden)

    def applyToPitch(self, n: scoring.Notation, idx: int = None) -> None:
        # if idx is None, apply to all noteheads
        scoringNotehead = self.asScoringNotehead()
        n.setNotehead(scoringNotehead, idx=idx, merge=True)



class Articulation(NoteAttachedSymbol):
    """
    Represents a note attached articulation
    """
    exclusive = False
    appliesToRests = False

    def __init__(self, kind: str):
        super().__init__()

        normalized = scoring.definitions.normalizeArticulation(kind)
        assert normalized, f"Articulation {kind} unknown. Possible values: " \
                           f"{scoring.definitions.articulations}"
        self.kind = normalized

    def __hash__(self):
        return hash((type(self).__name__, self.kind))

    def __repr__(self):
        return f"Articulation(kind={self.kind})"

    @classmethod
    def possibleValues(cls, key: str = None) -> Set[str] | None:
        if key is None or key == 'kind':
            return scoring.definitions.articulations

    def applyTo(self, n: scoring.Notation) -> None:
        if not n.tiedPrev and not n.isRest:
            n.addArticulation(self.kind)


class Fingering(NoteAttachedSymbol):
    exclusive = True
    appliesToRests = False

    def __init__(self, finger: str):
        super().__init__()

        self.finger = finger

    def applyTo(self, n: scoring.Notation) -> None:
        n.addAttachment(scoring.attachment.Fingering(self.finger))


class Bend(NoteAttachedSymbol):
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

    def applyTo(self, n: scoring.Notation) -> None:
        n.addAttachment(scoring.attachment.Bend(self.alter))


class Stem(NoteAttachedSymbol):
    """
    Customize the stem of a note

    Attributes:
        hidden: if True, the stem will be hidden
    """
    appliesToRests = False

    def __init__(self, hidden: bool = False):
        super().__init__()
        self.hidden = hidden

    def applyTo(self, n: scoring.Notation) -> None:
        if self.hidden:
            n.stem = 'hidden'


class Accidental(PitchAttachedSymbol):
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
                 size: float = None):
        super().__init__()

        self.hidden = hidden
        self.parenthesis = parenthesis
        self.color = color
        self.force = force
        self.size = size

    def __repr__(self):
        parts = []
        if self.hidden:
            parts.append('hidden=True')
        if self.parenthesis:
            parts.append('parenthesis=True')
        if self.color:
            parts.append(f'color={self.color}')
        if self.size is not None:
            parts.append(f'size={self.size}')
        if self.force:
            parts.append('force=True')
        return f'Accidental({", ".join(parts)})'

    def __hash__(self):
        return hash((type(self).__name__, self.hidden, self.parenthesis, self.color, self.force))

    def applyToPitch(self, n: scoring.Notation, idx: int = None) -> None:
        if n.isRest:
            return
        attachment = scoring.attachment.AccidentalTraits(color=self.color,
                                                         hidden=self.hidden,
                                                         parenthesis=self.parenthesis,
                                                         force=self.force,
                                                         size=self.size)
        attachment.anchor = idx
        n.addAttachment(attachment)


# -------------------------------------------------------------------

_symbols = (
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
    BeamBreak
)


_symbolNameToClass = {cls.__name__.lower(): cls for cls in _symbols}


def symbolnameToClass(name: str) -> type:
    return _symbolNameToClass[name]


def makeSymbol(clsname: str, *args, **kws) -> Symbol:
    """
    Construct a Symbol from the symbol name and any values and/or keywords passed

    A parameter can have '?' as a value, in which case a selection dialog will
    be open to select one of such values. Not all parameters support this (for example,
    any size or color or boolean will not support such selection). At the moment, dynamics,
    notehead shapes, articulations all suport this.

    Args:
        clsname: the symbol name (case independent)
        *args: any other arguments passed to the constructor
        **kws: any keywords passed to the constructor

    Returns:
        the created Symbol

    ============  ============================================================
    Symbol        Possible Values
    ============  ============================================================
    dynamic       pppp, ppp, pp, p, mp, mf, f, ff, fff, fff
    articulation  accent, staccato, tenuto, marcato, staccatissimo
    notehead      cross, harmonic, triangleup, xcircle, triangle, rhombus,
                  square, rectangle
    expression    Any text expression
    sizefactor    value (float): 1.0=default size, 2=doubles the size, etc.
    ============  ============================================================

    - **dynamics** are defined in ``maelzel.scoring.definitions.availableDynamics``
    - **articulations** are defined in ``maelzel.scoring.definitions.availableArticulations``
    - **noteheads** are defined in ``maelzel.scoring.definitions.noteheadShapes``

    """
    cls = _symbolNameToClass.get(clsname.lower())
    if cls is None:
        raise ValueError(f"Class '{clsname}' unknown. "
                         f"Known symbol names: {list(_symbolNameToClass.keys())}")
    if args and args[0] == '?':
        possibleValues = cls.possibleValues()
        if possibleValues is None:
            raise ValueError(f"Class {cls.__name__} does not define a set of possible values")
        import emlib.dialogs
        value = emlib.dialogs.selectItem(sorted(possibleValues), ensureSelection=True)
        args = (value,) + args[1:]
    for k, v in kws.items():
        if v == '?':
            possibleValues = cls.possibleValues(k)
            if possibleValues is None:
                raise ValueError(f"Class {cls.__name__} does not define possible values for param {k}")
            import emlib.dialogs
            v = emlib.dialogs.selectItem(sorted(possibleValues), ensureSelection=True)
            kws[k] = v
    return cls(*args, **kws)


def applyGroup(symbols: list[NoteAttachedSymbol], notation: scoring.Notation) -> None:
    cls = next(type(s) for s in symbols if s is not None)
    assert all(isinstance(s, cls) or s is None for s in symbols)


