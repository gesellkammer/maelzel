"""
scoring
=======

The scoring package provides functionality for representing a musical score
and rendering to pdf or an image via multiple backends (lilypond, MuseScore)

scoring.core
------------

.. seealso:: :py:mod:`maelzel.scoring.core`

The **scoring.core** provides the basic building blocks to define an abstract
score. For scoring purposes the most basic element is the
:class:`maelzel.scoring.notation.Notation`. A Notation can be used to represent
a Note, a Chord or a Rest. Notations can be grouped together in a
:class:`maelzel.scoring.core.Part` and multiple Parts form an
:class:`maelzel.scoring.core.UnquantizedScore`

Quantized / Unquantized
-----------------------

.. seealso:: :py:mod:`maelzel.scoring.quant`

In all these cases durations are abstract and measured in quarter notes.
The generated UnquantizedPart/UnquantizedScore do not have any measures or
scorestructure and are not quantized to musically meaningful durations.

For quantization purposes a :class:`~maelzel.scorestruct.ScoreStruct` is needed, where
the structure of the score is defined (time signatures, tempi, metadata). In order
to fine-tune the quantization a :class:`~maelzel.scoring.quant.QuantizationProfile` can
be defined, or the default preset can be used. Such a ``QuantizationProfile`` allows to
determine the complexity of the resulting rhythms, kinds of tuplets allowed, when to use
grace-notes, etc.

Rendering
---------

.. seealso:: :py:mod:`maelzel.scoring.render`

Only when an Arrangement has been quantized to a :class:`~maelzel.scoring.quant.QuantizedScore`
it is possible to render this as pdf. The rendering process is done by first converting
the quantized intermediate representation to some of the supported backend formats:
**lilypond** or **musicxml** (rendered via MuseScore). Then the rendering backend is called
to generate either a ``pdf`` or a ``png``. It is also possible to export the rendered score
as ``musicxml`` or even as ``midi``.

.. note::

    The external programs used for rendering (**lilypond** or **MuseScore**) need to be
    installed and present in the path. It is possible to customize the binary called when
    rendering in the general *maelzel* configuration. See :ref:`config`

"""
from .notation import *
from .core import *
from .common import logger
from . import quant
from . import render
from . import definitions
from . import spanner
from . import attachment
from . import enharmonics



