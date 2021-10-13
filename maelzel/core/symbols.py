from maelzel import scoring

# These are abstract notations used to later convert
# them to concrete scoging indications

class Symbol:
    exclusive = False


class NoteAttachedSymbol(Symbol):
    pass


class Dynamic(NoteAttachedSymbol):
    exclusive = True

    def __init__(self, kind: str):
        assert kind in scoring.definitions.availableDynamics, \
            f"Dynamic {kind} unknown, possible values: {scoring.definitions.availableDynamics}"
        self.kind = kind

    def __str__(self):
        return self.kind

    def __repr__(self):
        return f"Dynamic({self.kind})"


class Expression(NoteAttachedSymbol):
    exclusive = False

    def __init__(self, text: str):
        self.text = text


class Notehead(NoteAttachedSymbol):
    exclusive = True

    def __init__(self, kind: str, color: str = None):
        assert kind in scoring.definitions.noteheadShapes, \
            f"Notehead {kind} unknown. Possible noteheads: " \
            f"{scoring.definitions.noteheadShapes}"
        self.kind = kind
        self.color = color


class Articulation(NoteAttachedSymbol):
    exclusive = True

    def __init__(self, kind: str):
        assert kind in scoring.definitions.availableArticulations, \
            f"Articulation {kind} unknown. Possible values: " \
            f"{scoring.definitions.availableArticulations}"
        self.kind = kind


noteAttachedSymbols = [
    'Articulation',
    'Dynamic',
    'Notehead',
    'Expression'
]

_clsnameToClass = {
    'dynamic': Dynamic,
    'notehead': Notehead,
    'articulation': Articulation,
    'expression': Expression
}


def construct(clsname:str, *args, **kws) -> Symbol:
    cls = _clsnameToClass.get(clsname.lower())
    if cls is None:
        raise ValueError(f"Class {clsname} unknown")
    return cls(*args, **kws)