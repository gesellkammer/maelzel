Playback
========

The play method
---------------

Each :class:`~maelzel.core.musicobj.Note`, :class:`~maelzel.core.musicobj.Chord`,
:class:`~maelzel.core.musicobj.Line`, can play itself by calling its
:meth:`~maelzel.core.musicobj.MusicObj.play` method. Internally playback relies
on csound for realtime and offline audio generation, via
`csoundengine <https://github.com/gesellkammer/csoundengine>`_. All audio
playback within **maelzel** is implemented using *csoundengine*. This makes it possible
to interact between the *core* classes, like :class:`Note` or :class:`Voice` and other
"unrelated" parts of *maelzel*, like the :class:`maelzel.snd.audiosample.Sample` class.

Internally each :class:`~maelzel.core.musicobj.MusicObj` expresses its playback in
terms of a list of :class:`~maelzel.core.synthevent.SynthEvent`.

.. admonition:: synthesis events

    A SynthEvent is a score line with a number of fixed fields,
    user-defined fields and a sequence of breakpoints to define pitch, amplitude
    and any other parameter over time. For more information
    see :class:`~maelzel.core.synthevent.SynthEvent`

To play any :class:`~maelzel.core.musicobj.MusicObj` (a Note, a Voice, a Score), call its
:meth:`~maelzel.core.musicobj.MusicObj.play` method. Within this method a number of parameters
regarding playback can be determined. Such parameters are common to all objects. The most
relevant of these playback parameters are:

instr
  which instrument ``Preset`` (see :class:`~maelzel.core.presetbase.PresetDef` for more
  information). If not given the default preset is used, as determined in the configuration
  (see :ref:`play.instr <config_play_instr>`)

delay
  delay in seconds, added to the start of the object
  As opposed to the ``.start`` attribute of each object, which is defined
  in *symbolic time* (quarternote-time), the delay is always in *absolute time* (seconds)

gain
  modifies the own amplitude for playback/recording (0-1)

chan
  the channel to output to. **Channels start at 1**

pitchinterpol
  'linear', 'cos', 'freqlinear', 'freqcos'

fade
  fade duration in seconds, can be a tuple (fadein, fadeout)

position
  the panning position (0=left, 1=right)

params
  a Preset can define custom parameters, like a cutoff frequency for a filter
  or a modulation ratio for FM synthesis, etc.


Presets - Introduction
----------------------

**maelzel** uses **csound** for audio synthesis in realtime and offline. Within an
instrument preset (a :class:`~maelzel.core.presetbase.PresetDef`) the user is given
three variables: *kfreq* (the current frequency of the event), *kamp* (the current
amplitude of the event) and *kpitch* (the current midinote of the event, corresponds
to *kfreq*). With this information the user needs to provide the audio-generating part
using csound code, by assigning the audio output to the variable *aout1* for channel 1,
*aout2* for channel 2, etc.

There are a number of built-in Presets, but it is very easy to define new Presets. A new
Preset is defined via :func:`~maelzel.core.presetman.defPreset` or
:func:`~maelzel.core.presetman.defPresetSoundfont` (a shortcut to define a preset by
just pointing to a soundfont). Whenever a note is actually played with a
given preset, this preset is sent to the csound engine and instantiated/evaluated.


Example
~~~~~~~

.. code-block:: python

    from maelzel.core import *
    f0 = Note("1E")
    notes = Chain([Note(f2m(f0.freq*i), dur=0.5) for i in range(20)])
    play.defPreset("detuned", r'''
        ; kfreq and kamp are always available
        a0 = vco2(kamp/3, kfreq)
        a1 = vco2(kamp/3, kfreq+2)
        a2 = vco2(kamp/3, kfreq-3)
        ; aout1 is assigned the audio output (this is a mono-preset)
        aout1 = sum(a0, a1, a2)
    ''')
    notes.play(instr='detuned')

.. admonition:: See Also

    - :func:`~maelzel.core.presetman.defPreset`
    - :py:mod:`maelzel.core.presetman`


Playback attributes (setPlay)
-----------------------------

Any playback attribute determined via the :meth:`~maelzel.core.musicobj.MusicObj.play`
method can be set beforehand for each object individually, using
:meth:`~maelzel.core.musicobj.MusicObj.setPlay`. Then, when this object is played
any play settings fixed via ``.setPlay`` is used as if it was passed
to :meth:`~maelzel.core.musicobj.MusicObj.play`.

In the following example we fix the *instr* of 1 every 4 notes to 'mypiano'. For the
rest of the notes this settings stays undetermined. When ``.play`` is called on the
:class:`~maelzel.core.musicobj.Chain`, ``'saw'`` (which is a built-in preset) is set as the
default *instr*. Any fixed playarg will be used for playback, otherwise a group default
is used (the *instr* set for :class:`Chain`) or, as a fallback, the default *instr* as
defined in the config (:ref:`key: 'play.instr' <config_play_instr>`).

    .play param  →  param set via .setPlay  →  .play param for the group  →  default value

Example
~~~~~~~

.. code-block:: python

    from maelzel.core import *
    defPresetSoundfont('mypiano', '/path/to/mypiano.sf2')
    notes = Chain([Note(m, dur=1) for m in range(48, 72)])
    for i, note in enumerate(notes):
        if i%4 == 0:
            note.setPlay(instr='mypiano')
    notes.play(instr='saw')


Preset parameters
-----------------

A Preset can have parameters which modify its behaviour. For example, an FM-synth preset
might allow to define modulation ratio or depth, or a subtractive-synth preset could define
parameters for its filter. Any custom parameter must define a default value

.. code-block:: python

    from maelzel.core import *
    defPreset('substractive', r'''
        a0 vco2 kamp, kfreq, 10
        aout1 moogladder a0, lag:k(kcutoff, 0.1), kq
    ''', params={'kcutoff': 3000, 'kQ': 0.9})
    Chord("C4 E4 G4", dur=10).play(instr='substractive', params={'kcutoff': 1000})

Preset parameters can also be defined inline, following the syntax for csoundengine's
`Instr <https://csoundengine.readthedocs.io/en/latest/instr.html>`_

.. code-block:: python

    from maelzel.core import *
    defPreset('substractive', r'''
        |kcutoff=3000, kQ=0.9|
        a0 vco2 kamp, kfreq, 10
        aout1 moogladder a0, lag:k(kcutoff, 0.1), kq
    ''')
    Chord("C4 E4 G4", dur=10).play(instr='substractive', params=dict(kcutoff=1000))



Recording / Offline Rendering
-----------------------------

By replacing a call to :meth:`~maelzel.core.musicobj.MusicObj.play` with a call to
:meth:`~maelzel.core.musicobj.MusicObj.rec` it is possible to render any
:class:`~maelzel.core.musicobj.MusicObj` as a soundfile (offline rendering). To
render multiple objects to the same soundfile you can either group them in a container
(for example, place multiple notes inside a :class:`~maelzel.core.musicobj.Voice` and
render that) or render them via :func:`maelzel.core.musicobj.recObjects`

.. code-block:: python

    from maelzel.core import *
    voice1 = Voice([Note(m, dur=1)
                    for m in range(60, 72)])
    voice2 = Voice([Note(m+0.5, dur=1.75)
                    for m in range(60, 72)])
    voice1.setPlay(instr='saw')
    voice2.setPlay(instr='tri')
    outfile = recObjects([voice1, voice2])

Notice that in this way, any parameter normally passed to ``.rec`` needs to be fixed
via ``.setPlay``.

Using the ``rendering`` context manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simpler method is to use the :func:`~maelzel.core.play.rendering` context
manager:

.. code-block:: python

    from maelzel.core import *
    voice1 = Voice([Note(m, dur=1)
                    for m in range(60, 72)])
    voice2 = Voice([Note(m+0.5, dur=1.75)
                    for m in range(60, 72)])
    with rendering('out.wav'):
        voice1.play(instr='saw')
        voice2.play(instr='tri')

With the context manager the same code used for playing in realtime can be used to render offline
(see also: :class:`~maelzel.core.play.OfflineRenderer`)

-----------------

Playback API
------------

.. toctree::

    coreplay
    presetman
    presetdef

