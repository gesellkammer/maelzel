.. _installation:

************
Installation
************

**maelzel** needs a python version >= 3.9. For all platforms, the installation is
the same::

    pip install maelzel


There are some **external dependencies** which need to be installed manually:

* **csound** >= 6.17. (https://csound.com/download.html). Csound 7 is not supported yet.
* **lilypond** (https://lilypond.org/download.html)
* **MuseScore** (https://musescore.org/en) - **Optional** - Useful when rendering musicxml

----------------

Dependencies
============

MacOS
-----

- Install csound: https://csound.com/download.html
- Install lilypond: http://lilypond.org/macos-x.html
- Install MuseScore: https://musescore.org/en/download

Windows
-------

- Install csound: https://csound.com/download.html
- Install lilypond: http://lilypond.org/windows.html
- Install MuseScore: https://musescore.org/en/download

Linux
-----

Arch
~~~~

.. code-block:: bash

    sudo pacman -S csound lilypond musescore

Debian / Ubuntu
~~~~~~~~~~~~~~~

**csound**: The *csound* package provided is probably too old. *csound* can be easily installed
from source (see also the
`official instructions <https://github.com/csound/csound/blob/develop/BUILD.md#debian>`_)::

    sudo apt-get build-dep csound
    sudo apt-get install cmake
    git clone -b csound6 https://github.com/csound/csound.git csound
    cd csound
    mkdir build
    cd build
    cmake ..
    make -j $(nproc)
    sudo make install
    sudo ldconfig

**lilypond**: install from official repositories::

    sudo apt-get install lilypond

**MuseScore**: install it via their distributed AppImages:
https://musescore.org/en/download
