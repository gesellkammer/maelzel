.. maelzel documentation master file, created by
   sphinx-quickstart on Sun Jan 31 20:13:32 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


maelzel - Documentation
=======================

**maelzel** is a framework for computer music in python. It is
based on a :ref:`core<core>` package, which provides datatypes to define *notes*,
*chords*, *sequences*, *lines*, *voices*, *scores*, etc. Other modules 
provide functionality for working with sound (loading soundfiles, working
with samples, applying filters, fading, mixing, playback), 
sound analysis, music, acoustics, music notation, etc.

Demo Notebook
-------------

`Analysis / Recreation of Messiaen's *Liturgie de Cristal* <https://nbviewer.jupyter.org/github/gesellkammer/maelzel/blob/master/examples/Messiaen-La%20Liturgie%20de%20Cristal.ipynb>`_

.. image:: assets/messiaen-notebook.jpg
  :target: https://nbviewer.jupyter.org/github/gesellkammer/maelzel/blob/master/examples/Messiaen-La%20Liturgie%20de%20Cristal.ipynb
  :alt: Jupyter Session 1

----------------

Quick Start
-----------

**maelzel** needs a python version >= 3.9

    pip install maelzel

There are some **external dependencies** which need to be installed manually:

* **csound** >= 6.16. (https://csound.com/download.html).
* **lilypond** (https://lilypond.org/download.html)
* **MuseScore** (https://musescore.org) - Optional - useful for quick rendering to png

For more information see :ref:`installation`

-----------------

.. include:: Tutorial.rst


-----------------

.. toctree::
   :maxdepth: 3

   Introduction
   Tutorial
   Reference
   Installation


