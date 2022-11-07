from __future__ import annotations
from emlib.misc import ReprMixin
from . import definitions
from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class AcciddentalTraits:
    color: str = ''
    hidden: bool = False
    parenthesis: bool = False
    brackets: bool = False


class Attachment(ReprMixin):
    exclusive = False

    def __init__(self, color=''):
        self.color = color


class Fingering(Attachment):
    exclusive = True

    def __init__(self, fingering: str):
        super().__init__()
        self.fingering = fingering

    def __hash__(self):
        return hash((type(self).__name__, self.fingering))


class Articulation(Attachment):

    def __init__(self, kind: str, color: str = '', **kws):
        assert kind in definitions.articulations
        super().__init__(color=color)
        self.kind = kind
        self.properties: dict = kws

    def __hash__(self):
        props = tuple(self.properties.items())
        return hash((type(self).__name__, self.kind, hash(props)))


class Ornament(Attachment):
    exclusive = True

    def __init__(self, kind: str, color: str = ''):
        assert kind in definitions.availableOrnaments
        super().__init__(color=color)
        self.kind = kind

    def __hash__(self):
        return hash(('Ornament', self.kind, self.color))

class Tremolo(Attachment):
    def __init__(self, tremtype='single', nummarks=2, **kws):
        assert tremtype in {'single', 'start', 'end'}
        super().__init__(**kws)
        self.tremtype = tremtype
        self.nummarks = nummarks

    def singleDuration(self) -> int:
        # 1:8, 2:16, 3:32,...
        return 2**(2+self.nummarks)

    def __hash__(self):
        return hash(('Tremolo', self.tremtype, self.nummarks))


class Fermata(Attachment):
    exclusive = True

    def __init__(self, kind='normal'):
        super().__init__()
        assert kind in definitions.availableFermatas
        self.kind = kind

    def __hash__(self):
        return hash(('Fermata', self.kind))



class Bend(Attachment):
    exclusive = True

    def __init__(self, interval: float):
        super().__init__()
        self.interval = interval

    def __hash__(self):
        return hash(('Bend', self.interval))


class Harmonic(Attachment):
    """
    A natural or artificial harmonic

    A harmonic is not a flageolet: here we use the term flageolet to identify a
    sounding pitch which is to be produced as a harmonic without indicating
    how this is to be achieved.

    Args:
        interval: the interval above the string/pressed pitch. A natural harmonic
            has an interval of 0
    """
    exclusive = True

    def __init__(self, interval: int=0):
        super().__init__()
        self.interval = interval

    def __hash__(self):
        return hash(('Harmonic', self.interval))


class Text(Attachment):
    """
    A text annotation which can be added to a Notation
    """
    __slots__ = ('text', 'placement', 'fontsize', 'fontstyles', 'box')

    def __init__(self, text: str, placement='above', fontsize: float = None, fontstyle='',
                 box: str | bool = False, color=''):
        super().__init__(color=color)
        assert not text.isspace()
        if fontsize is not None:
            assert isinstance(fontsize, (int, float))
        self.text = text
        self.placement = placement
        self.fontsize = fontsize
        self.box: str = box if isinstance(box, str) else 'square' if box else ''
        if not fontstyle:
            self.fontstyles = None
        else:
            styles = fontstyle.split(',')
            for style in styles:
                assert style in {'italic', 'bold'}, f'Style {style} not supported'
            self.fontstyles = styles

    def __repr__(self):
        elements = [f'text={self.text}', f'placement={self.placement}']
        if self.fontsize:
            elements.append(f'fontsize={self.fontsize}')
        if self.fontstyles:
            elements.append(f'fontstyles={self.fontstyles}')
        if self.box:
            elements.append(f'box={self.box}')
        return f'{type(self).__name__}({", ".join(elements)})'

    def __hash__(self) -> int:
        return hash(('Text', self.text, self.placement, self.fontsize, self.box))

    def __eq__(self, other: 'Text') -> bool:
        return hash(self) == hash(other)

    def isItalic(self):
        return self.fontstyles and 'italic' in self.fontstyles

    def isBold(self):
        return self.fontstyles and 'bold' in self.fontstyles
