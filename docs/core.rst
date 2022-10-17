.. py:currentmodule:: maelzel.core.mobj

.. _core:

Core
====

**maelzel.core** provides a set of classes to define notes, chords,
lines, sequences, voices or entire scores. Any of these objects can
be played in real time, recorded and displayed as notation. When
converting to notation, objects are quantized following a score
structure.

**maelzel.core**'s main purpose is to represent musical ideas, reason
about aspects of pitch, rhythms, etc., and be of help while composing,
analyzing or preparing a performance.

.. admonition:: Notation as display vs engraving

    The objects defined in **maelzel.core** can have many attributes
    within the realm of notation which do not have a well defined
    acoustic translation: articulations, text labels, size. Although
    such symbolic attributes are very important for composing or analyzing
    **maelzel.core** does not seek to provide the level of customization
    needed to produce a finished engraved score.
    **The notation produced is to be understood as a means of visualization**
    of musical and / or acoustic processes.

Key Concepts
------------

MObj
~~~~

All classes defined in **maelzel.core** inherit from :class:`MObj` (*Maelzel Object*, or
*Music Object*).
A :class:`MObj` **exists in time** (in has a start and duration attribute),
it **can be displayed as notation** and, if appropriate, **played as audio**.

Implicit / Explicit Time
~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`MObj` has always a *start* and *dur* attributes.
These can be unset (``None``), meaning that they are not explicitely determined and depend
on the context. For example, a note might have no start or duration set. When adding such a note
to a sequence of notes (a :class:`Chain`) its start time will be set to the end of the previous
note/chord in the chain, or 0 if this is the first note.

Real Time / Beat (symbolic) Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The time attributes (*start*, *dur*, *end*) of a :class:`MObj` refer to a relative time (a *beats*),
measured in quarternotes. This *quarternote time* depends on the tempo at a given
moment. To map from *relative* time (in *quarternotes*) to *absolute* time (in *seconds*) a score structure
(:class:`~maelzel.scorestruct.ScoreStruct`) is needed

Score Structure
~~~~~~~~~~~~~~~

A Score Structure (:class:`~maelzel.scorestruct.ScoreStruct`) is a timeline built from
a sequence of measure definitions. Each measure defines a time signature and tempo.
A :class:`~maelzel.scorestruct.ScoreStruct` **does not contain any material itself**: it is only
the "skeleton" of a score.

At any moment there is always an **active score structure**
(:func:`~maelzel.core.workspace.getScoreStruct`, :func:`~maelzel.core.workspace.setScoreStruct`),
the default being an endless score with a *4/4* time-signature and a tempo of *60 bpm*.

Configuration - Workspace
~~~~~~~~~~~~~~~~~~~~~~~~~

Just as there is an active ScoreStruct, there is at any moment an active
:class:`Configuration <maelzel.core.config.CoreConfig>` (see :ref:`config`) which controls multiple
aspects of **maelzel.core** and enables the user to customize multiple aspects of
the rendering process, quantization, playback, etc. Both
score structure and config are contained within a :ref:`Workspace <workspace_mod>`


Playback
~~~~~~~~

For playback **maelzel** uses `csound <https://csound.com/>`_ as an audio engine embedded
in python (see `csoundengine <https://csoundengine.readthedocs.io>`_)

When the :meth:`~maelzel.core.MObj.play` method is called, a
:class:`~maelzel.core.MObj` generates a list of
:class:`~maelzel.core.synthevent.SynthEvent`, which tell *csound* how
to play a :class:`Note`, :class:`Chord`, or an entire :class:`Score`. Using csound it is
possible to define instrumental presets using any kind of synthesis or by simply loading
a set of samples or a soundfont.

.. admonition:: See Also

    - :py:mod:`The maelzel.core.play module <maelzel.core.play>`

----------------


----------------

Table of Contents
=================

.. toctree::
    :maxdepth: 4

    Musical Objects: Note, Chord, Chain, Voice <mobj>
    Score Structure: interfacing symbolic and real time <scorestruct>
    Score <score>
    coreplayintro
    config
    workspace
    coretools
    symbols

