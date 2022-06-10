.. py:currentmodule:: maelzel.core.musicobj

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

    **maelzel.core** defines music objects which can have many attributes
    within the realm of notation and do not have a well defined
    purely acoustic meaning: dynamics, articulations, text labels, etc. These
    "notation" attributes are very important for composing or analyzing.
    Nevertheless **maelzel.core** does not seek to provide the
    level of customization needed to produce a finished engraved score.
    **The notation produced is to be understood as a means of visualization**
    of musical and / or acoustic processes.

Key Concepts
------------

MusicObj
~~~~~~~~

All classes defined in **maelzel.core** inherit from :class:`MusicObj`.
A :class:`MusicObj` **exists in time** (in has a start and duration attribute),
it **can display itself as notation** and, if appropriate, **play itself as audio**.

Implicit / Explicit Time
~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`MusicObj` has always a *start* and *dur* attributes.
These can be unset (``None``), meaning that they are not explicitely determined and depend
on the context. For example, a note might have no start or duration set. When adding such a note
to a sequence of notes (a :class:`Chain`) its start time will be set to the end of the previous
note/chord in the chain, or 0 if this is the first note.

Real Time / Beat (symbolic) Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The time attributes (*start*, *dur*, *end*) of a :class:`MusicObj` refer to a symbolic,
*beat* time, measured in quarternotes. This *quarternote time* depends on the tempo. To map
from *quarternote* time to *absolute* time a score structure
(:class:`~maelzel.scorestruct.ScoreStruct`) is needed, which sets the tempo at any moment in time

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

Just as there is an active ScoreStruct, there is at any moment at active
:class:`Configuration <maelzel.core.config.CoreConfig>` (see :ref:`config`) which controls multiple
aspects of **maelzel.core** and enables the user to customize multiple aspects of
the rendering process, quantization, playback, etc. Both
score structure and config are contained within a :ref:`Workspace <workspace_mod>`


Playback
~~~~~~~~

For playback **maelzel** uses `csound <https://csound.com/>`_ as an audio engine embedded
in python (see `csoundengine <https://csoundengine.readthedocs.io>`_)

When the :meth:`~maelzel.core.musicobjbase.MusicObj.play` method is called, a
:class:`~maelzel.core.musicobjbase.MusicObj` generates a list of
:class:`~maelzel.core.csoundevent.CsoundEvent`, which tell *csound* how
to play a :class:`Note`, :class:`Chord`, or an entire :class:`Score`. Using csound it is
possible to define instrumental presets using any kind of synthesis or by simply loading
a set of samples or a soundfont.

.. admonition:: See Also

    - :py:mod:`The maelzel.core.play module <maelzel.core.play>`

----------------

Tutorial (Jupyter Notebooks)
============================

#. `First Steps <https://nbviewer.org/github/gesellkammer/maelzel/blob/master/notebooks/firststeps.ipynb>`_
#. `Durations: Ockeghem's Missa Prolationum <https://nbviewer.org/github/gesellkammer/maelzel/blob/master/notebooks/ockeghem.ipynb>`_
#. `Config and Workspace <https://nbviewer.org/github/gesellkammer/maelzel/blob/master/notebooks/maelzel.core%20-%20Config%20and%20Workspace.ipynb>`_
#. `Score Structure <https://nbviewer.org/github/gesellkammer/maelzel/blob/master/notebooks/maelzel.core%20-%20Score%20Structure.ipynb>`_
#. `Notation Refinements <https://nbviewer.org/github/gesellkammer/maelzel/blob/master/notebooks/maelzel.core%20-%20Symbols.ipynb>`_

----------------

Table of Contents
=================

.. toctree::
    :maxdepth: 4

    Musical Objects: Note, Chord, Line, Chain, Voice <musicobj>
    Score Structure: interfacing symbolic and real time <scorestruct>
    coreplayintro
    config
    workspace
    coretools



