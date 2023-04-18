.. _installation:

************
Installation
************

**maelzel** needs a python version >= 3.9. For all platforms, the installation is
the same::

    pip install maelzel


There are some **external dependencies** which need to be installed manually:

* **csound** >= 6.17. (https://csound.com/download.html).
* **lilypond** (https://lilypond.org/download.html)

----------------

Dependencies
============

MacOS
-----

**Install Csound**
    Download the ``.dmg`` package from https://csound.com/download.html (latest release:
    https://github.com/csound/csound/releases/download/6.18.1/Csound-MacOS-universal-6.18.1.dmg)

**Install Lilypond**
    At the moment the recommended method to install *lilypond* is via *homebrew*:
    ``brew install lilypond``. This will install the last version for your platform
    (both intel and arm64 architectures are supported) and add it to your ``PATH``.

    For more information visit http://lilypond.org

--------------

Windows
-------

**Install Csound**
    Install csound from https://csound.com/download.html. Use the installer: this will take care
    of setting up the ``PATH`` and setup csound so that it can be found by *maelzel*

**Install lilypond**
    1. Download https://gitlab.com/lilypond/lilypond/-/releases/v2.24.1/downloads/lilypond-2.24.1-mingw-x86_64.zip.
    2. Unzip it and place the folder ``lilypond-2.X.Y`` in ``C:\Program Files``.
    3. Add the path ``C:\Program Files\lilypond-2.X.Y\bin`` to your ``PATH`` environmental
       variable (as "System Variable")

----------------

Linux
-----

Arch
~~~~

.. code-block:: bash

    sudo pacman -S csound lilypond

Debian / Ubuntu
~~~~~~~~~~~~~~~

It is possible to install *csound* via the package manager::

    sudo apt install csound libcsnd-dev

However it is highly recommended to install csound from sources (see also the
`official instructions <https://github.com/csound/csound/blob/develop/BUILD.md#debian>`_)::

    sudo apt-get build-dep csound
    sudo apt-get install cmake
    git clone -b csound6 https://github.com/csound/csound.git csound
    cd csound
    mkdir build && cd build
    cmake ..
    cmake --build . --parallel
    sudo cmake --install .
    sudo ldconfig

**lilypond**: install from official repositories::

    sudo apt-get install lilypond

