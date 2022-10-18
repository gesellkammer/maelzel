.. py:currentmodule:: maelzel.core.mobj

MObj
====

- All classes defined in **maelzel.core** inherit from :class:`MObj`.
- A :class:`MObj` can either be an event (like a :class:`Note` or a :class:`Chord`, all inherit from
  :class:`MEvent`) or a container (a :class:`Chain` or a :class:`~maelzel.core.score.Score`)

MEvents: Note, Chord
--------------------

- There are two basic kinds of :class:`MEvent`: :class:`Note`, :class:`Chord`.
- A :class:`Chord` contains a list of duration-less :class:`Note`s. Those notes
  can be used to customized individual attributes for each of the components of the chord (they
  can have different amplitudes, notehead shapes and other notation-specific customizations)
- Rests are a kind of :class:`Note`. Rests can be identified via the :meth:`~Note.isRest` method
- A :class:`~maelzel.core.clip.Clip` is a special kind of MEvent: its duration is defined in absolute terms,
  its relative duration (in beats) depends on the tempo


.. code-block:: python

    >>> from maelzel.core import *
    >>> note = Note("4C", dur=1)
    >>> rest = Rest(dur=0.5)
    >>> note.isRest()
    False
    >>> rest.isRest()
    True


Containers
----------

Notes and Chords can be grouped in different containers.

- :class:`Chain`: a sequence of notes, chords or other Chains, **without overlap**.
- :class:`Voice`: a Voice is a special kind of Chain used to represent a voice within a Score.
- :class:`Score`: groups multiple Voices to represent a Score. It can have an attached :class:`~maelzel.scorestruct.ScoreStruct`

----------------


.. automodapi:: maelzel.core.mobj
    :no-main-docstr:
    :allowed-package-names: maelzel.core.mobj,maelzel.core.mobjbase,maelzel.core.synthevent

