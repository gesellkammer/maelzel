.. py:currentmodule:: maelzel.core.musicobj

MusicObj
========

Events: Note, Chord
-------------------

- All classes defined in **maelzel.core** inherit from :class:`MusicObj`.
- There are three basic kinds of :class:`MusicObj`: :class:`Note`, :class:`Chord`
  and :class:`Line`
- Rests are a kind of :class:`Note`. Rests can be identified via the :meth:`Note.isRest` method:

.. code-block:: python

    >>> from maelzel.core import *
    >>> note = Note("4C", dur=1)
    >>> rest = Rest(dur=0.5)
    >>> note.isRest()
    False
    >>> rest.isRest()
    True


Containers (MusicObjList)
-------------------------

Notes and Chords can be grouped in different containers. All inherit from :class:`MusicObjList`:

- :class:`Chain`: a sequence of notes/chords, without overlap.
- :class:`Voice`: a group of notes/chords/lines/chains (no overlap). Represents a voice within a :class:`Score`
- :class:`Score`: groups multiple Voices to represent a Score. It can have an attached :class:`~maelzel.scorestruct.ScoreStruct`

----------------


.. automodapi:: maelzel.core.musicobj
    :no-main-docstr:
    :allowed-package-names: maelzel.core.musicobj,maelzel.core.musicobjbase,maelzel.core.csoundevent

