.. _magics-label:

Core
====

**csoundengine** defines a set of ipython/jupyter ``%magic`` / ``%%magic`` commands

%csound
-------

Syntax::

    %csound setengine <enginename>         : Sets the default Engine


%%csound
--------

Compile the code in this cell within the current Engine. The current engine
can be explicitely selected via ``%csound setengine <enginename>``. Otherwise
the last started Engine will be used.


.. code-block:: python

    from csoundengine import *
    e = Engine()

.. code-block:: csound

    %%csound
    instr 10
        kamp = p4
        kmidi = p5
        a0 oscili kamp, mtof:k(kmidi)
        aenv linsegr 0, 0.01, 1, 0.01, 0
        a0 *= aenv
        outch 1, a0
    endin

.. code-block:: python

    event = e.sched(10, args=[0.1, 67])

%%definstr
----------

Defines a new Instr inside the current Session

.. code-block:: python

    from csoundengine import *
    e = Engine()

.. code-block:: csound

    %%definstr foo
    |kamp=0.5, kfreq=1000|
    a0 oscili kamp, kfreq
    aenv linsegr 0, 0.01, 1, 0.01, 0
    a0 *= aenv
    outch 1, a0

The last block is equivalent to

.. code-block:: python

    s = e.session()
    s.defInstr("foo", r'''
        |kamp=0.5, kfreq=1000|
        a0 oscili kamp, kfreq
        aenv linsegr 0, 0.01, 1, 0.01, 0
        a0 *= aenv
        outch 1, a0
    ''')
