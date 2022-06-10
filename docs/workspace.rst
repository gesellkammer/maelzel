.. _workspace_mod:

The Workspace Model
===================

Workspace
---------

maelzel.core is organized on the idea of a :class:`~maelzel.core.workspace.Workspace`.
A workspace contains the current state: the *active config*, the *active scorestrucutre*,
a playback engine, etc. Many actions, like note playback, notation rendering, etc.,
use the active workspace to determine tempo, score structure, default playback instrument,
etc.

At any moment there is always an active workspace. This can be accessed via
:func:`~maelzel.core.workspace.getWorkspace`.
At the start of a session a workspace (the *'root'* workspace) is created, based on the root
config and a default score structure.

config: Active Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The active configuration determines many aspects of **maelzel**, like the default
instrument used for playback, the audio backend used, the quantization complexity used
when rendering a score, etc.

At module initialization a ``rootConfig`` is created. This is an instance of
:class:`~maelzel.core.config.CoreConfig`, updated with any customizations persisted
from previous sessions (see :meth:`CoreConfig.save() <maelzel.core.config.CoreConfig.save>`).

.. admonition:: See Also

    - :func:`~maelzel.core.workspace.makeConfig`
    - :ref:`config`

Example
~~~~~~~

.. code-block:: python

    from maelzel.core import *
    notes = Chain([Note(m, start=i) for i, m in enumerate(range(60, 72))])
    notes.show()

.. image:: assets/workspace-chromaticscale.png

Create a Workpsace with a different default instrument

.. code-block:: python

    w = Workspace(updates={'play.instr': 'piano'}, active=True)
    notes.play()
    # Deactivate will set the previous Workspace as active
    w.deactivate()

A temporary Workpsace can be created as a context manager.
In this case we use an ad-hoc score struct and a copy of the
active config:

.. code-block:: python

    scorestruct = ScoreStruct.fromString(r'''
    4/4, 60
    3/4
    5/8, 72
    6/8
    ''')
    with Workspace(scorestruct=scorestruct, config=getConfig()) as w:
        notes.show()

.. image:: assets/workspace-temporaryworkspace.png


-----------------

.. automodapi:: maelzel.core.workspace
    :no-heading:
    :allowed-package-names: maelzel.core.workspace,maelzel.core.config,maelzel.music.dynamics
    :no-inheritance-diagram:

