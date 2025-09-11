from __future__ import annotations
import copy

from maelzel import _util
from . import definitions
from maelzel.common import F, F0

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from typing_extensions import Self
    import maelzel.scoring.quant as quant


class Attachment:
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
    """If True, allow only one attachment of a given class to be added to a Notation"""
    
    priority = 100
    """???"""
    
    copyToSplitNotation = False
    """Should an attachment of this class be copied to parts of a notation if it is split?"""

    def __init__(self, 
                 color='',
                 instancePriority=0,
                 anchor: int | None = None,
                 horizontalPlacement=''):
        self.color: str = color
        """The color of this attachment, if applicable"""

        self.instancePriority: int = instancePriority
        """A priority for this attachment in relation to others of the same kind. A
        negative priority will place this attachment nearer to the note/notehead"""

        self.anchor: int | None = anchor
        """if given, an index corresponding to the notehead this attachment should be
        anchored to."""

        self.horizontalPlacement: str = horizontalPlacement
        """The horizontal placement of the attachment, one of '', 'pre' or 'post """

    def getPriority(self) -> int:
        return self.priority + self.instancePriority

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def __repr__(self):
        return _util.reprObj(self, hideFalsy=True)


class GlissProperties(Attachment):
    copyToSplitNotation = True
    linetypes = ('solid', 'zigzag', 'dotted', 'dashed', 'trill')

    def __init__(self, linetype='solid', color=''):
        super().__init__(color=color)
        _util.checkChoice('linetype', linetype, GlissProperties.linetypes)
        self.linetype = linetype
        """The line type, one of 'solid', 'wavy', 'dotted', 'dashed'"""


class Color(Attachment):
    exclusive = True
    copyToSplitNotation = True
    
    def __init__(self, color: str):
        super().__init__(color=color)
        

class Hidden(Attachment):
    exclusive = True
    copyToSplitNotation = True


class SizeFactor(Attachment):
    exclusive = True
    copyToSplitNotation = True

    def __init__(self, size: float):
        """
        Size Factor applied to a Notation

        Args:
            size: relative size. 0 is normal, +1 is bigger, -1 is smaller
        """
        super().__init__()
        self.size = size


class GracenoteProperties(Attachment):

    def __init__(self, slash: bool, value=F(1, 2)):
        super().__init__()
        self.slash = slash
        self.value = value


class StemTraits(Attachment):
    exclusive = True
    copyToSplitNotation = True

    def __init__(self, color='', hidden=False):
        super().__init__(color=color)
        self.hidden = hidden
        """Hide stem"""


class AccidentalTraits(Attachment):
    """Traits for accidentals"""
    _default: AccidentalTraits | None = None

    def __init__(self, color='', hidden=False, parenthesis=False,
                 brackets=False, force=False, size: float | None = None,
                 anchor: int | None = None):
        super().__init__(color=color, anchor=anchor)

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

    def __init__(self, kind: str, color=''):
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

    def __init__(self, kind: str, color='', placement='', **kws):
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
    exclusive = True

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

    def __init__(self, kind='normal', color=''):
        super().__init__(color=color)
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

    def __init__(self, kind='', visible=True, placement='above', horizontalPlacement='pre'):
        super().__init__()
        if kind:
            assert kind in definitions.breathMarks, f'Kind unknown, supported values are {definitions.breathMarks}'
        self.kind = kind
        self.visible = visible
        self.placement = placement
        self.horizontalPlacement = horizontalPlacement

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

    __slots__ = ('text', 'placement', 'fontsize', 'italic', 'weight', 'fontfamily', 'box', 'relativeSize')

    def __init__(self,
                 text: str,
                 placement='',
                 fontsize: int | float | None = None,
                 italic=False,
                 weight='',
                 fontfamily='',
                 box: str | bool = '',
                 color='',
                 role='',
                 relativeSize=False):
        super().__init__(color=color)
        assert not text.isspace()
        if fontsize is not None:
            assert isinstance(fontsize, (int, float))
        assert weight in ('', 'normal', 'bold')
        assert box in (True, False, '', 'square', 'rectangle', 'circle')
        if isinstance(box, bool):
            boxshape = 'rectangle' if box else ''
        else:
            boxshape = box
        self.text = text
        self.placement = placement
        self.fontsize = fontsize
        self.box = boxshape
        self.italic = italic
        self.weight = weight if weight != 'normal' else ''
        self.fontfamily = fontfamily
        self.role = role
        self.relativeSize = relativeSize

    def __repr__(self):
        return _util.reprObj(self, hideFalsy=True, hideEmptyStr=True, priorityargs=('text',), quoteStrings=('text',), hideKeys=('text',))

    def __hash__(self) -> int:
        return hash(('Text', self.text, self.placement, self.fontsize, self.box))

    def __eq__(self, other: 'Text') -> bool:
        return hash(self) == hash(other)

    def isBold(self):
        return self.weight == 'bold'


class Clef(Attachment):
    exclusive = True

    def __init__(self, kind: str, color=''):
        super().__init__(color=color)
        if kind not in definitions.clefs:
            raise ValueError(f"Clef '{kind}' unknown. Possible clefs: {definitions.clefs}")
        self.kind = kind

    def __hash__(self):
        return hash(('Clef', self.kind))


class BeamSubdivisionHint(Attachment):
    exclusive = True

    def __init__(self, minimum: F = F0, maximum: F = F0, once=True):
        """
        
        Args:
            minimum: beam minimum subdivision 
            maximum: beam maximum subdivision 
            once: if True, the hint is applied only once. Otherwise, it is applied at
                the event attached and stays active
        """
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.once = once


class QuantHint(Attachment):
    exclusive = True

    def __init__(self, division: tuple[int, ...], strength: float):
        super().__init__()
        self.division = division
        self.strength = strength


class Hook:
    """
    A Hook is a wrapper around a function, triggered at different situations

    """
    def __init__(self, func: Callable[[quant.QuantizedPart], None]):
        """
        
        Args:
            func: a callable of the form ``(part: QuantizedPart) -> None``

        """
        self._func = func

    def __call__(self, obj):
        self._func(obj)


class PostPartQuantHook(Hook):
    """
    Hook called after part quantization

    The function will be called with a QuantizedPart after it has been
    quantized. This can be used to apply beam breaking strategies,
    etc. For an example usage see :class:`maelzel.core.symbols.BeamBreak`


    """

    def __call__(self, part: quant.QuantizedPart) -> None:
        from maelzel.scoring.quant import QuantizedPart
        assert isinstance(part, QuantizedPart)
        self._func(part)
