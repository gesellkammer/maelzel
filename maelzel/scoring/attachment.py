from __future__ import annotations
from emlib.misc import ReprMixin
from . import definitions
from typing import TypeVar
import copy

_AttachmentT = TypeVar('_AttachmentT', bound='Attachment')


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
    copyToSplitNotation = False

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

    def copy(self: _AttachmentT) -> _AttachmentT:
        return copy.deepcopy(self)


class Property(Attachment):

    def __init__(self, key: str, value=True, anchor: int = None):
        super().__init__(anchor=anchor)
        self.key = key
        self.value = value


class GlissandoProperties(Attachment):
    copyToSplitNotation = True

    def __init__(self, linetype='solid', color=''):
        super().__init__(color=color)
        if not linetype in ('solid', 'wavy', 'dotted', 'dashed'):
            raise ValueError(f"linetype should be one of 'solid', 'wavy', 'dotted', 'dashed', "
                             f"got {linetype}")
        self.linetype = linetype
        """The line type, one of 'solid', 'wavy', 'dotted', 'dashed'"""


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

    def __init__(self, kind: str, color: str = '', placement: str = '', **kws):
        assert kind in definitions.articulations
        assert not placement or placement in ('above', 'below')
        super().__init__(color=color)
        self.kind = kind
        self.placement = placement
        self.properties: dict = kws

    def __hash__(self):
        props = tuple(self.properties.items())
        return hash((type(self).__name__, self.kind, hash(props)))

    def getPriority(self):
        return super().getPriority() + self.extraPriority.get(self.kind, 0)


class Tremolo(Attachment):
    copyToSplitNotation = True

    def __init__(self, tremtype='single', nummarks=2, relative=True, **kws):
        assert tremtype in {'single', 'start', 'end'}
        super().__init__(**kws)
        self.tremtype = tremtype
        self.nummarks = nummarks
        self.relative = relative

    def singleDuration(self) -> int:
        # 1:8, 2:16, 3:32,...
        return 2**(2+self.nummarks)

    def __hash__(self):
        return hash(('Tremolo', self.tremtype, self.nummarks, self.relative))


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

    def __init__(self, kind='', visible=True, placement='above'):
        super().__init__()
        if kind:
            assert kind in definitions.breathMarks, f'Kind unknown, supported values are {definitions.breathMarks}'
        self.kind = kind
        self.visible = visible
        self.placement = placement

    def __hash__(self):
        return hash(('Breath', self.kind, self.visible))


class Harmonic(Attachment):

    copyToSplitNotation = True

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

    def __init__(self, interval: int = 0):
        super().__init__()
        self.interval = interval
        self.kind = 'artificial' if interval > 0 else 'natural'

    def __hash__(self):
        return hash(('Harmonic', self.interval))



class Text(Attachment):
    """
    A text annotation which can be added to a Notation

    Args:
        text: the text
        placement: if given, one of 'above', 'below'
        fontsize: the absolute font size
        italic: if True, use italic style
        weight: one of 'normal', 'bold'
        fontfamily: a font family or a comma separated list thereof
        box: one of '' (no enclosure), 'square', 'rectangle', 'circle'
        role: the role of the text. This is a hint as to what the text is
            intended for. At the moment possible values are '' (no role),
            'measure' (text attached to the measure itself)
    """
    priority = 100

    __slots__ = ('text', 'placement', 'fontsize', 'italic', 'weight', 'fontfamily', 'box')

    def __init__(self,
                 text: str,
                 placement='',
                 fontsize: float = None,
                 italic=False,
                 weight='',
                 fontfamily: str = '',
                 box='',
                 color='',
                 role=''):
        super().__init__(color=color)
        assert not text.isspace()
        if fontsize is not None:
            assert isinstance(fontsize, (int, float))
        assert weight in ('', 'normal', 'bold')
        assert box in ('', 'square', 'rectangle', 'circle')
        self.text = text
        self.placement = placement
        self.fontsize = fontsize
        self.box = box
        self.italic = italic
        self.weight = weight
        self.fontfamily = fontfamily
        self.role = role

    def __repr__(self):
        elements = [f'text={self.text}', f'placement={self.placement}']
        if self.fontsize:
            elements.append(f'fontsize={self.fontsize}')
        if self.italic:
            elements.append(f'italic=True')
        if self.weight and self.weight != 'normal':
            elements.append(f'weight={self.weight}')
        if self.box:
            elements.append(f'box={self.box}')
        return f'{type(self).__name__}({", ".join(elements)})'

    def __hash__(self) -> int:
        return hash(('Text', self.text, self.placement, self.fontsize, self.box))

    def __eq__(self, other: 'Text') -> bool:
        return hash(self) == hash(other)

    def isBold(self):
        return self.weight == 'bold'


class Clef(Attachment):

    def __init__(self, kind: str, color=''):
        super().__init__(color=color)
        self.kind = kind

    def __hash__(self):
        return hash(('Clef', self.kind))

