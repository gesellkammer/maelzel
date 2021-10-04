maelzel
=======

A set of modules for computer music

- core: core module defining basic classes for computer music (Note, Chord, Voice, Score, etc). Any of these classes can be representation as notation and played back.
- music: music representation, algorithmic music, etc
- scoring: music representation, quantization
- midi: read/write midi files, general midi
- snd: read/write soundfiles, process samples, apply filters, fades, etc.
- ext: wrappers around external software, like sonic-visualizer, audacity, reaper
- acoustics: formulas for helmholtz resonators, speed of sound, etc.


Installation
------------

::

    pip install maelzel


External Dependencies
~~~~~~~~~~~~~~~~~~~~~

- csound (>= 6.16)


Examples
--------

Messiaen - Quatour pour la fin du temps - La liturgie de Cristal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: docs/assets/messiaen-notebook.jpg
  :target: https://nbviewer.jupyter.org/github/gesellkammer/maelzel/blob/master/examples/Messiaen-La%20Liturgie%20de%20Cristal.ipynb
  :alt: Screenshot of jupyter notebook

`Notebook: Analysis and reconstruction of the piano and cello isorhythmic structures <https://nbviewer.jupyter.org/github/gesellkammer/maelzel/blob/master/examples/Messiaen-La%20Liturgie%20de%20Cristal.ipynb>`_

