==============
Input / Output
==============

.. toctree::
    :maxdepth: 1

    musicxmlio <notebooks/musicxmlio>


Musicxml
========

**maelzel.core** can input and output musicxml

.. grid:: 1 1 2 2

    .. grid-item-card:: MusicXML Input / Output
        :link: notebooks/musicxmlio
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/notebook-musicxmlio.png
            :height: 300px

        Introduction to notes, chords, voices, etc. Notation and playback

#.. figure:: assets/notebook-musicxmlio.png
#    :height: 300px
#    :target: notebooks/musicxmlio


Lilypond
========

Lilypond can output lilypond files for further modification or use lilypond to
render scores to pdf / png / svg.

Lilypond input, while not directly supported, can be achieved via
``lilypond -> musicxml -> maelzel``, see https://python-ly.readthedocs.io/en/latest/ly.musicxml.html

MIDI
====

A :class:`~maelzel.core.score.Score` can be written as a MIDI file:

TODO: example with screenshot

Direct MIDI input is not supported at the moment but you can convert any MIDI
file to musicxml via a wide number of applications (MuseScore, for example)
and use that as an input.
