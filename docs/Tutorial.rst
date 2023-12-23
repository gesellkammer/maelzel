.. _core_jupyter_tutorial:

Tutorial (Jupyter Notebooks)
============================

.. toctree::
    :maxdepth: 1

    First Steps <notebooks/firststeps>
    Durations: Ockeghem's Missa Prolationum <notebooks/ockeghem>
    Configuration <notebooks/maelzel-core-configuration>
    Workspace <notebooks/maelzel-core-workspace>
    Score Structure <notebooks/maelzel-core-scorestruct>
    Notation Refinements <notebooks/maelzel-core-symbols>
    Input / Output <notebooks/musicxmlio>
    Clicktrack <notebooks/clicktrack>


-------------------


.. grid:: 1 1 2 2

    .. grid-item-card:: First Steps
        :link: notebooks/firststeps
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/notebook-firststeps.png
            :height: 300px

        Introduction to notes, chords, voices, etc. Notation and playback


    .. grid-item-card:: Durations: Ockeghem's Missa Prolationum
        :link: notebooks/ockeghem
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/notebook-durations.png
           :height: 300px

        A recreation of Ockeghem's Missa Prolationum showcasing how durations work

.. grid:: 1 1 2 2

    .. grid-item-card:: Configuration
        :link: notebooks/maelzel-core-configuration
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/notebook-configuration.png
            :height: 300px

        Most aspects of how **maelzel** handles notation and playback can be customized by modifying or creating
        a configuration object (an instance of :class:`~maelzel.core.config.CoreConfig`)


    .. grid-item-card:: Workspace
        :link: notebooks/maelzel-core-workspace
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/notebook-workspace.png
            :height: 300px

        A workspace contains the current state: the active config, the active scorestrucutre, a playback engine, etc.
        Many actions, like note playback, notation, etc., use the active workspace to determine tempo, score structure,
        default playback instrument, etc.

.. grid:: 1 1 2 2

    .. grid-item-card:: Score Structure
        :link: notebooks/maelzel-core-scorestruct
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/notebook-scorestruct.png
            :height: 300px

        In maelzel.core there is a division of concerns between music data (notes, chords, lines, voices, etc) and a
        score structure (:class:`~maelzel.scorestruct.ScoreStruct`). The score structure consists of a set of measure
        definitions (time signature, tempo).


    .. grid-item-card:: Notation Refinements
        :link: notebooks/maelzel-core-symbols
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/notebook-symbols.png
            :height: 300px

        Most aspects of **maelzel.core**'s notation output can be modified. Not only is it possible to add spanners
        (slurs, brackets, lines), articulations, dynamics and many other symbols to notes and chords. Also color,
        size, text style, etc. can be customized.

.. grid:: 1 1 2 2

    .. grid-item-card:: Input / Output
        :link: notebooks/musicxmlio
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/notebook-musicxmlio.png
            :height: 300px

        Shows the different output (pdf, png, midi, ...) and input formats

    .. grid-item-card:: Clicktrack
        :link: notebooks/clicktrack
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/notebook-clicktrack.jpg
            :height: 300px

        **maelzel** can be used to create click-track and export is as audio and / or
        MIDI to any DAW to use it for performance

