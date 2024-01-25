MObj
====

A :class:`~maelzel.core.mobj.MObj` is an abstract class (it cannot be created as is)
at the base of the class structure within **maelzel.core**. All classes defined in
**maelzel.core** inherit from :class:`~maelzel.core.mobj.MObj`.

A :class:`~maelzel.core.mobj.MObj` can either be an :ref:`event <mevent>`, like a :class:`~maelzel.core.event.Note` or
a :class:`~maelzel.core.event.Chord` (both inherit from :class:`~maelzel.core.event.MEvent`) or a **container**
(a :class:`~maelzel.core.chain.Chain`, a :class:`~maelzel.core.chain.Voice` or a
:class:`~maelzel.core.score.Score`)

Any :class:`~maelzel.core.mobj.MObj` can be played (:meth:`~maelzel.core.mobj.MObj.play`),
shown as notation (:meth:`~maelzel.core.mobj.MObj.show`), saved as image, pdf,
musicxml, lilypond or midi (:meth:`~maelzel.core.mobj.MObj.write`).


----------------


.. automodapi:: maelzel.core.mobj
    :no-main-docstr:
    :allowed-package-names: maelzel.core.mobj,maelzel.core.synthevent
    :no-inheritance-diagram:
    :skip: MContainer





