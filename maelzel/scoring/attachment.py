from __future__ import annotations
from emlib.misc import ReprMixin
from . import definitions


class Attachment(ReprMixin):
    """
    An Attachment is any kind of symbol added to a Notation or a pitch thereof

    Args:
        color: the color of the attachment
        instancePriority: a priority for this attachment in relation to other attachments
            of the same kind. A negative priority will place this attachemnt nearer to
            the note/notehead
        anchor: if given, an index corresponding to the notehead this attachment should be
            anchored to. Noteheads are sorted by pitch, from low to high. A value of None
            anchors the attachment to the whole note/chord
    """
    exclusive = False
    priority = 100

    def __init__(self, color='', instancePriority=0, anchor: int = None):
        self.color: str = color
        """The color of this attachment, if applicable"""

        self.instancePriority: int = instancePriority
        """A priority for this attachment in relation to others of the same kind. A 
        negative priority will place this attachment nearer to the note/notehead"""

        self.anchor: int | None = anchor
        """if given, in index corresponding to the notehead this attachment should be
        anchored to."""

    def getPriority(self) -> int:
        return self.priority + self.instancePriority


class Property(Attachment):

    def __init__(self, key: str, value=True, anchor: int = None):
        super().__init__(anchor=anchor)
        self.key = key
        self.value = value


class AccidentalTraits(Attachment):
    _default: AccidentalTraits = None

    def __init__(self, color='', hidden=False, parenthesis=False,
                 brackets=False, force=False, size: float = None):
        super().__init__(color=color)

        self.hidden = hidden
        """Hide accidental"""

        self.parenthesis = parenthesis
        """Parenthesize accidental"""

        self.brackets = brackets
        """Place brackets around accidental"""

        self.force = force
        """If True, the accidental should be forced to be shown"""

        self.size = size
        """A size factor applied to the accidental (1=normal)"""

    @classmethod
    def default(cls) -> AccidentalTraits:
        if cls._default is None:
            cls._default = cls()
        return cls._default


class Ornament(Attachment):
    exclusive = True
    priority = 1

    def __init__(self, kind: str, color: str = ''):
        assert kind in definitions.availableOrnaments
        super().__init__(color=color)
        self.kind = kind

    def __hash__(self):
        return hash(('Ornament', self.kind, self.color))


class Fingering(Attachment):
    exclusive = True
    priority = 30

    def __init__(self, fingering: str):
        super().__init__()
        self.fingering = fingering

    def __hash__(self):
        return hash((type(self).__name__, self.fingering))


class Articulation(Attachment):
    priority = 20
    extraPriority = {
        'flageolet': -1
    }

    def __init__(self, kind: str, color: str = '', **kws):
        assert kind in definitions.articulations
        super().__init__(color=color)
        self.kind = kind
        self.properties: dict = kws

    def __hash__(self):
        props = tuple(self.properties.items())
        return hash((type(self).__name__, self.kind, hash(props)))

    def getPriority(self):
        return super().getPriority() + self.extraPriority.get(self.kind, 0)


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


class Breath(Attachment):
    exclusive = True

    def __init__(self, kind='', visible=True):
        super().__init__()
        if kind:
            assert kind in definitions.breathMarks, f'Kind unknown, supported values are {definitions.breathMarks}'
        self.kind = kind
        self.visible = visible

    def __hash__(self):
        return hash(('Breath', self.kind, self.visible))


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
    priority = 10

    def __init__(self, interval: int=0):
        super().__init__()
        self.interval = interval

    def __hash__(self):
        return hash(('Harmonic', self.interval))


class Text(Attachment):
    """
    A text annotation which can be added to a Notation
    """
    priority = 100

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


class Clef(Attachment):

    def __init__(self, kind: str, color=''):
        super().__init__(color=color)
        self.kind = kind

    def __hash__(self):
        return hash(('Clef', self.kind))




