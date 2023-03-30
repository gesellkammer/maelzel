.. py:currentmodule:: maelzel.snd.audiosample

Sample: working with audio data
===============================

:class:`Sample`: a class representing **audio data** (wraps a *numpy* float array).

A Sample contains the audio (read from a soundfile or synthesized) as a numpy float array.
It aware of its sr, original format and encoding, etc. It can also perform many operations
on the sample data (fade-in/out, cut, insert, reverse, normalize, etc) and implements most
math operations valid for audio data (``+``, ``-``, ``*``, ``/``).

.. note::

    Some operations are performed *in place* while others return a copy. In general,
    **whenever the number of samples or the number of channels is modified (e.g. `mixdown`
    or `stripLeft`), a copy is returned**. Otherwise the operation is performed *in-place*

    * **In place**: `fade`, `normalize`, `reverse`
    * **Copy**: all others (`appendSilence`, `prependSilence`, `mixdown`, `stripLeft`, ...)


Demo Notebooks
--------------


.. grid:: 1 1 2 2
    :margin: 4
    :gutter: 3

    .. grid-item-card:: Sample - Basic Operations
        :link: notebooks/maelzel-snd-audiosample
        :link-type: doc

        .. figure:: assets/audiosample-demo-notebook.png
            :height: 300px

        * Loading a soundfile from disk
        * Plotting / Spectrogram
        * Basic operations (fade, normalization, mixing, etc.)
        * Playback

    .. grid-item-card:: Sample - Feature Extraction
        :link: notebooks/audiosample-feature-extraction
        :link-type: doc

        .. figure:: assets/audiosample-feature-extraction.png
            :height: 300px

        A :class:`Sample` implements multiple feature extraction algorithms, like
        **onset detection**, **fundamental tracking**, etc.

-----

.. automodapi:: maelzel.snd.audiosample
    :allowed-package-names: maelzel.snd.audiosample
    :no-inheritance-diagram:
    :skip: Path,amp2db