The Workspace Model
-------------------

maelzel.core is organized on the idea of a :class:`~maelzel.core.workspace.Workspace`.
A workspace contains the current state: an active config, an active scorestrucutre,
a playback engine, etc. Many actions, like note playback, notation rendering, etc.,
use the active workspace to determine tempo, score structure, default playback instrument,
etc.

At any moment there is always an active workspace. This can be accessed via
:func:`~maelzel.core.workspace.getWorkspace`.
At the start of a session a workspace (the *'root'* workspace) is created, based on the default
config and a default score structure.

Configuration
~~~~~~~~~~~~~

The active configuration determines many aspects of **maelzel**, like the default
instrument used for playback, the audio backend used, the quantization complexity used
when rendering a score, etc.

At module initialization the ``rootConfig`` is created. This is an instance of
:class:`~maelzel.core.config.CoreConfig`, updated with any persisted settings
saved from any previous version (see :meth:`CoreConfig.save() <maelzel.core.config.CoreConfig.save>`). To create
a new config based on this rootConfig use :func:`~maelzel.core.workspace.makeConfig`

.. admonition:: See Also

    :ref:`config`

-----------------

.. automodapi:: maelzel.core.workspace
    :no-heading:
    :allowed-package-names: maelzel.core.workspace,maelzel.core.config,maelzel.music.dynamics

