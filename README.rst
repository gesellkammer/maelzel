.. image:: docs/assets/maelzel-logo.png
	:width: 320px
	:align: center
	

maelzel
=======

**maelzel** is a framework for computer music in python. It provides
datatypes to define notes, chords, sequences, lines, voices, scores, etc.
These objects can be combined with soundfiles and samples, rendered as notation
and synthesized in realtime.

It provides functionality for composition, music notation, sound analysis
and synthesis, feature extraction, transcription and machine learning


Documentation
-------------

https://maelzel.readthedocs.io/en/latest/index.html

----------------


Structure
---------
 

- **core**: core module defining basic classes for computer music
  (Note, Chord, Voice, Score, etc). Any of these objects can be
  representated as notation and played back. 
- **snd**: read/write soundfiles, process samples, apply filters,
  fades, etc.
- **music**: music representation, algorithmic music, etc
- **scoring**: music representation. Offers support for microtones (up
  to 1/8 tones or any cent deviation), very refined cotrol of
  automatic quantization, glissandi, grace-notes, annotations,
  etc. Supports multiple backends (lilypond and musicxml at the
  moment).
- **acoustics**: formulas for helmholtz resonators, speed of sound, etc.
- **midi**: read/write midi files, general midi
- **ext**: wrappers around external software, like sonic-visualizer,
  audacity, reaper


--------------


Installation
------------

::

    pip install maelzel


Dependencies
~~~~~~~~~~~~

- csound (>= 6.18)
- lilypond


-------------


Demos
-----

https://maelzel.readthedocs.io/en/latest/Introduction.html#demo-notebooks

.. figure:: docs/assets/maelzel-demos.png
  :target: https://maelzel.readthedocs.io/en/latest/Introduction.html#demo-notebooks

