
Reference 
=========

maelzel.core
------------

**maelzel.core** provides a set of classes to define notes, chords, sequences, voices or
entire scores. These musical objects can be used for composition, analysis or any kind of
computation related to music/sound. They can be played in real time, recorded
and displayed as notation.

.. toctree::
    :maxdepth: 2

    core

---------------------

maelzel.scoring
---------------

The **maelzel.scoring** package can be used for general purpose musical notation. It provides
the notation functionality used by **maelzel.core** but it can be used independently. 
It implements a very flexible and robust quantization engine (see :py:mod:`maelzel.scoring.quant`)
and can output *musicxml*, *lilypond* and *midi* as well as render to *pdf* and *png* via 
multiple backends (*lilypond*, *musescore*) 

.. toctree::
    :maxdepth: 1

    scoringcore
    scoringquant

--------------------

maelzel.snd
-----------

All functionality related to *sound* is contained within this package

.. toctree::
    :maxdepth: 1
    :titlesonly:
    :glob:

    snd-*

--------------------

Miscellaneous Functionality
---------------------------

.. toctree::
    :maxdepth: 1

    rationalnumbers
    distribute


