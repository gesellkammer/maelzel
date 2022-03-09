.. _core:

Core - Computation / Notation / Playback
----------------------------------------

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
    corresponding to the realm of notation and do not have a well defined
    purely acoustic meaning: dynamics, articulations, text labels, etc. These
    "notation" attributes are very important for composing or analyzing.
    Nevertheless **maelzel.core** does not seek to provide the
    level of customization needed to produce a finished engraved score.
    The notation produced is to be understood as a means of visualization
    of musical and / or acoustic processes.

----------------

Contents
--------

.. toctree::
    :maxdepth: 2


    musicobj
    scorestruct
    config
    workspace


