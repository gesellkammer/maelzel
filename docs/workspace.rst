The Workspace Model
-------------------

maelzel.core is organized on the idea of a :class:`~maelzel.core.workspace.Workspace`. A workspace contains
the current state: an active config, an active scorestrucutre, a playback engine, etc. Many actions, like note
playback, notation rendering, etc., use the active workspace to determine tempo, score structure,
default playback instrument, etc.

At any moment there is always an active workspace. This can be accessed via
:func:`~maelzel.core.workspace.activeWorkspace`.
At the start of a session a workspace (the 'root' workspace) is created, based on the default
config and a default score structure.

Configuration
~~~~~~~~~~~~~

The active config determines many aspects of **maelzel**, like XXX

Root Config
~~~~~~~~~~~

The configuration (a ``configdict.ConfigDict``, see `<https://configdict.readthedocs.io>`_ ) associated with the root Workspace is
persistent: any changes to it will be saved to disk and loaded in a next session. To make
changes to the configuration without persisting it call :func:`~maelzel.core.workspace.newConfig`

-----------------

.. automodapi:: maelzel.core.workspace
    :no-heading:
