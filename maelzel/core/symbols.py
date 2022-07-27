"""
Symbols are objects which can be attached to a note/chord to modify its notation

Most symbols do not have any other meaning than to hint the backend used for
notation to display the object in a certain way. For example a Notehead symbol
can be attached to a note to modify the notehead shape used.

Dynamics are included as Symbols but they are deprecated, since dynamics
can be used for playback (see :ref:`config_usedynamics`)

"""
from __future__ import annotations
import random
from maelzel import scoring
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Set, Sequence, Union
    from maelzel.core import musicobj
    MusicEvent = Union[musicobj.Note, musicobj.Chord]


# These are abstract notations used to later convert
# them to concrete scoring indications

noteAttachedSymbols = [
    'expression',
    'notehead',
    'articulation',
    'accidental'
]


_uuid_alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'


def makeuuid(size=8) -> str:
    return ''.join(random.choices(_uuid_alphabet, k=size))


class Symbol:
    """Base class for all symbols"""
    exclusive = False
    applyToTieStrategy = 'first'
    appliesToRests = False

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    def applyTo(self, notation: scoring.Notation) -> None:
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

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash((type(self).__name__, self.value))

    def __repr__(self):
        cls = type(self).__name__
        return f'{cls}(value={self.value})'


class Spanner(Symbol):
    """
    A Spanner is a line/curve between two locations (notes/chords/rests)

    """
    exclusive = False
    appliesToRests = True

    def __init__(self, kind='start', uuid: str = ''):
        self.kind = kind
        self.uuid = uuid or makeuuid(8)

    def __repr__(self) -> str:
        cls = type(self).__qualname__
        return f'{cls}(kind={self.kind})'

    @classmethod
    def bind(cls, startobj, endobj, **kws) -> None:
        """
        Bind a Spanner to two notes/chords

        Args:
            startobj: start Note / Chord
            endobj: end Note / Chord

        Returns:

        """
        spanstart = cls(kind='start', **kws)
        startobj.setSymbol(spanstart)
        spanend = cls(kind='end', uuid=spanstart.uuid)
        endobj.setSymbol(spanend)


class Slur(Spanner):

    def applyTo(self, notation: scoring.Notation) -> None:
        slur = scoring.spanner.Slur(kind=self.kind, uuid=self.uuid)
        notation.addSpanner(slur)


class HairpinCresc(Spanner):

    def applyTo(self, notation: scoring.Notation) -> None:
        hairpin = scoring.spanner.Hairpin(kind=self.kind, uuid=self.uuid, direction='<')
        notation.addSpanner(hairpin)


class HairpinDecr(Spanner):

    def applyTo(self, notation: scoring.Notation) -> None:
        hairpin = scoring.spanner.Hairpin(kind=self.kind, uuid=self.uuid, direction='>')
        notation.addSpanner(hairpin)



class SizeFactor(Property):
    """Sets the size of an object (as a factor of default size)"""
    applyToTieStrategy = 'all'

    def applyTo(self, notation: scoring.Notation) -> None:
        notation.sizeFactor = self.value


class Color(Property):

    """Customizes the color of a MusicObj"""
    def applyTo(self, notation: scoring.Notation) -> None:
        notation.color = self.value


class NoteAttachedSymbol(Symbol):
    """Base-class for all note attached symbols"""
    noteheadAttached = False

    @classmethod
    def possibleValues(cls, key: str = None) -> Optional[Set[str]]:
        return None


class Dynamic(NoteAttachedSymbol):
    """A dynamic attached to a note/chord"""
    exclusive = True

    def __init__(self, kind: str):
        assert kind in scoring.definitions.availableDynamics, \
            f"Dynamic {kind} unknown, possible values: {scoring.definitions.availableDynamics}"
        self.kind = kind

    def __hash__(self):
        return hash((type(self).__name__, self.kind))

    @classmethod
    def possibleValues(self, key: str = None) -> Optional[Set[str]]:
        if key is None or key == 'kind':
            return scoring.definitions.availableDynamics

    def __str__(self):
        return self.kind

    def __repr__(self):
        return f"Dynamic({self.kind})"

    def applyTo(self, n: scoring.Notation) -> None:
        if not n.tiedPrev:
            n.dynamic = self.kind


class Expression(NoteAttachedSymbol):

    """A note attached expression """
    exclusive = False

    def __init__(self, text: str, placement='above'):
        self.text = text
        self.placement = placement

    def __repr__(self):
        return f"Expression({self.text})"

    def applyTo(self, n: scoring.Notation) -> None:
        if not n.tiedPrev:
            n.addAnnotation(self.text)

    def __hash__(self):
        return hash((type(self).__name__, self.text, self.placement))


class Notehead(NoteAttachedSymbol):
    """
    Customizes the notehead shape, color, parenthesis and size

    Args:
        kind: one of 'cross', 'harmonic', 'triangleup', 'xcircle',
              'triangle', 'rhombus', 'square', 'rectangle'
        color: a css color (str)
        parenthesis: if True, parenthesize the notehead
        size: a size factor (1.0 means the size corresponding to the staff size, 2. indicates
            a notehead twice as big)
    """
    noteheadAttached = True
    exclusive = True
    applyToTieStrategy = 'all'
    appliesToRests = False


    def __init__(self, kind: str = None, color: str = None, parenthesis = False,
                 size: float = None):
        self.hidden = False
        if kind and kind.endswith('?'):
            parenthesis = True
            kind = kind[:-1]
        elif kind == 'hidden':
            kind = ''
            self.hidden = True
        assert not kind or kind in scoring.definitions.noteheadShapes, \
            f"Notehead {kind} unknown. Possible noteheads: " \
            f"{scoring.definitions.noteheadShapes}"
        self.kind = kind
        self.color = color
        self.parenthesis = parenthesis
        self.size = size

    def asScoringNotehead(self) -> str:
        if self.hidden:
            return 'hidden'
        kind = self.kind or ''
        if self.parenthesis:
            kind += '?'
        parts = [kind]
        if self.color:
            parts.append(f"color={self.color}")
        if self.size:
            parts.append(f"size={self.size}")
        return ";".join(parts)

    def __hash__(self):
        return hash((type(self).__name__, self.kind, self.color, self.parenthesis, self.size))

    @classmethod
    def possibleValues(cls, key: str = None) -> Optional[Set[str]]:
        if key is None or key == 'kind':
            return scoring.definitions.noteheadShapes

    def __repr__(self):
        parts = [self.kind or 'default']
        if self.color:
            parts.append(f'color={self.color}')
        if self.parenthesis:
            parts.append(f'parenthesis=True')
        if self.size is not None and self.size != 1.0:
            parts.append(f'size={self.size}')
        return f"Notehead({', '.join(parts)})"

    def applyToNotehead(self, notation: scoring.Notation, idx: int=None) -> None:
        notehead = self.kind if self.kind else ''
        if self.parenthesis:
            notehead += '?'

        if idx is None:
            notation.notehead = self.asScoringNotehead()
            #if self.color:
            #    notation.setProperty('noteheadColor', self.color)
            #if self.size:
            #    notation.setProperty('noteheadSizeFactor', self.size)
        else:
            if isinstance(notation.notehead, str):
                notation.notehead = [''] * len(notation.pitches)
            notation.notehead[idx] = self.asScoringNotehead()

    def applyTo(self, notation: scoring.Notation) -> None:
        self.applyToNotehead(notation)


class Articulation(NoteAttachedSymbol):
    """
    Represents a note attached articulation
    """
    exclusive = True
    appliesToRests = False

    def __init__(self, kind: str):
        assert kind in scoring.definitions.availableArticulations, \
            f"Articulation {kind} unknown. Possible values: " \
            f"{scoring.definitions.availableArticulations}"
        self.kind = kind

    def __hash__(self):
        return hash((type(self).__name__, self.kind))

    def __repr__(self):
        return f"Articulation(kind={self.kind})"

    @classmethod
    def possibleValues(cls, key: str = None) -> Optional[Set[str]]:
        if key is None or key == 'kind':
            return scoring.definitions.availableArticulations

    def applyTo(self, n: scoring.Notation) -> None:
        if not n.tiedPrev and not n.isRest:
            n.articulation = self.kind


class Accidental(NoteAttachedSymbol):
    """Customizes the accidental of a note"""
    exclusive = True
    appliesToRests = False

    def __init__(self, hidden=False, parenthesis=False, color: str = None):
        self.hidden = hidden
        self.parenthesis = parenthesis
        self.color = color

    def __repr__(self):
        parts = []
        if self.hidden:
            parts.append('hidden=True')
        if self.parenthesis:
            parts.append('parenthesis=True')
        if self.color:
            parts.append(f'color={self.color}')
        return f'Accidental({", ".join(parts)})'

    def __hash__(self):
        return hash((type(self).__name__, self.hidden, self.parenthesis, self.color))

    def applyTo(self, n: scoring.Notation) -> None:
        if n.isRest:
            return
        n.accidentalHidden = self.hidden
        if self.parenthesis:
            n.setProperty('accidentalParenthesis', self.parenthesis)
        if self.color:
            n.setProperty('accidentalColor', self.parenthesis)


_symbols = (Dynamic, Notehead, Articulation, Expression, SizeFactor, Color, Accidental)

_symbolNameToClass = {cls.__name__.lower(): cls for cls in _symbols}


def symbolnameToClass(name: str) -> type:
    return _symbolNameToClass[name]


def makeSymbol(clsname:str, *args, **kws) -> Symbol:
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

