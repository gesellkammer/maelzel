Introduction
------------

**maelzel** provides a framework to work with both *symbolic* (music) and *concrete*
(audio) sound.


Installation
~~~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    Installation


Demo Notebooks
--------------

Here is a selection of short demonstrations showcasing some of its features:


.. toctree::
    :maxdepth: 1
    :numbered:

    Messiaen - Quatour pour la fin du temps <messiaen-la-liturgie>
    Speech Transcription <demo-transcribe>
    Complex Rhythms (Ferneyhough's 3rd String Quartet) <demo-complex-rhythms>
    Soundfile chords <clip-chords>


.. grid:: 1 1 2 2
    :margin: 3
    :gutter: 3

    .. grid-item-card:: Messiaen - Quatour pour la fin du temps
        :link: messiaen-la-liturgie
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/messiaen-notebook.jpg
            :height: 300px

        * **Analysis and reconstruction** of the rhythmic ostinati in Messiaen's
          *La liturgie de Cristal*
        * **Isorhythmic structures**
        * **Resynthesis**. Interaction with the *csound* engine (routing events through a reverb)

    .. grid-item-card:: Speech Transcription
        :link: demo-transcribe
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/demo-transcribe-notebook.png
            :height: 300px

        **Transcription** of a monophonic speaking voice: intonation inflection, simplification
        strategies (the *Visvalingam-Wyatt* algorithm), dealing with consonants. Feature extraction:
        **fundamental tracking** via **pYin**, **onset** prediction, **voiced/unvoiced** analysis,
        etc. **Resynthesis**.

.. grid:: 1 1 2 2
    :margin: 3
    :gutter: 3

    .. grid-item-card:: Complex Rhythms and Transformations
        :link: demo-complex-rhythms
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/demo-complex-rhythms-notebook.png
          :height: 300px

        * Reconstructing the beginning of **Ferneyhough's 3rd String Quartet**.
        * Complex rhythms and nested tuplets
        * Rhythm and pitch transformations


    .. grid-item-card:: Soundfile chords
        :link: clip-chords
        :link-type: doc
        :class-title: cardtitle

        .. figure:: assets/demo-clip-chords.png
          :height: 300px

        * **Spectral analysis / resynthesis** of a soundfile using chords
        * Partial tracking to extract sinusoidal and noise components over time (Haken's
          *loris* algorithm)
        * Multiple resolutions and pixelation effects (Ablinger's *Quadraturen*).
        * Reconstruction / resynthesis with sine-tones and instrumental samples.