.. _chainmod:

.. py:currentmodule:: maelzel.core.chain


Horizontal Containers
=====================

Chain
-----

A :class:`Chain` is used to represent a sequence of notes or chords. It can contain other
Chains. Within a :class:`Chain` there are **no simultaneous events**: events cannot overlap.
Events in a :class:`Chain` have an offset, relative to the start of the :class:`Chain`.
This offset can be explicitely set (meaning ``someNote.offset`` is not :class:`None`) or
inferred to start exactly at the end of the previous item in the :class:`Chain`.

When an item is appended to a :class:`Chain`, the item becomes a *child* of the chain and
the chain becomes its *parent* (see :attr:`MEvent.parent <maelzel.core.event.MEvent.parent>`)

.. toctree::
    :hidden:

    Chain Referene <notebooks/reference-chain>

.. grid:: 1 1 2 2

    .. grid-item-card:: Chain Reference
        :link: notebooks/reference-chain
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/notebook-reference-chain.png
            :height: 300px

        Reference Notebook for the Chain class

-----------------

Voice
~~~~~

A :class:`Voice` is a subclass of :class:`Chain` and all said before applies to it, with
the following differences:

* A :class:`Voice` can contain a Chain, but **a Voice cannot contain another Voice**
* A :class:`Voice` does not have a time offset, **its offset is always 0**
* A :class:`Voice` is used to represent a *UnquantizedPart* or *Instrument* within a score.
  It includes multiple attributes and methods to customize its representation and playback.

.. automodapi:: maelzel.core.chain

