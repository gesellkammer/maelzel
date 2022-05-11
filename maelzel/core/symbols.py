from __future__ import annotations
from maelzel import scoring
from typing import Optional, Set, Sequence


# These are abstract notations used to later convert
# them to concrete scoging indications


class Symbol:
    exclusive = False
    applyToTieStrategy = 'first'

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
    exclusive = True

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash((type(self).__name__, self.value))


class SizeFactor(Property):
    applyToTieStrategy = 'all'

    def applyTo(self, notation: scoring.Notation) -> None:
        notation.sizeFactor = self.value


class Color(Property):
    def applyTo(self, notation: scoring.Notation) -> None:
        notation.color = self.value


class NoteAttachedSymbol(Symbol):

    @classmethod
    def possibleValues(cls, key: str = None) -> Optional[Set[str]]:
        return None


class Dynamic(NoteAttachedSymbol):
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
    Args:
        kind: one of 'cross', 'harmonic', 'triangleup', 'xcircle',
              'triangle', 'rhombus', 'square', 'rectangle'
        color: a css color
        parenthesis: if True, parenthesize the notehead
        size: a size factor (1.0 means the size corresponding to the staff size)
    """

    exclusive = True
    applyToTieStrategy = 'all'


    def __init__(self, kind: str = None, color: str = None, parenthesis = False,
                 size: float = None):
        assert kind is None or kind in scoring.definitions.noteheadShapes, \
            f"Notehead {kind} unknown. Possible noteheads: " \
            f"{scoring.definitions.noteheadShapes}"
        self.kind = kind
        self.color = color
        self.parenthesis = parenthesis
        self.size = size

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
            parts.append(f'parenthesize=True')
        if self.size is not None and self.size != 1.0:
            parts.append(f'size={self.size}')
        return f"Notehead({', '.join(parts)})"

    def applyTo(self, notation: scoring.Notation) -> None:
        if self.kind:
            notation.notehead = self.kind
        if self.color:
            notation.setProperty('noteheadColor', self.color)
        if self.size:
            notation.setProperty('noteheadSizeFactor', self.size)
        notation.noteheadParenthesis = self.parenthesis


class Articulation(NoteAttachedSymbol):
    exclusive = True

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
        if not n.tiedPrev:
            n.articulation = self.kind


class Accidental(NoteAttachedSymbol):
    exclusive = True

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
        n.accidentalHidden = self.hidden
        if self.parenthesis:
            n.setProperty('accidentalParenthesis', self.parenthesis)
        if self.color:
            n.setProperty('accidentalColor', self.parenthesis)


_symbols = (Dynamic, Notehead, Articulation, Expression, SizeFactor, Color, Accidental)

_symbolNameToClass = {cls.__name__.lower(): cls for cls in _symbols}


def construct(clsname:str, *args, **kws) -> Symbol:
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