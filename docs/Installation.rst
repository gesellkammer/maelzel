.. _installation:

************
Installation
************

**maelzel** needs a python version >= 3.9. For all platforms, the installation is
the same::

    pip install maelzel


----------------

Dependencies
============

**csound** needs to be installed manually if not already installed. Any version of csound >= 6.18
is supported (csound 7 is explicitely supported and recommended)

MacOS
-----

* **csound 7** (recommended): https://github.com/csound/csound/releases/download/7.0.0-beta.10/csound-macos-7.0.0-beta.10.zip
* **csound 6**: https://github.com/csound/csound/releases/download/6.18.1/Csound-MacOS-universal-6.18.1.dmg

--------------

Windows
-------

* **csound 7** (recommended): https://github.com/csound/csound/releases/download/7.0.0-beta.10/csound-windows-7.0.0-beta.10.zip
* **csound 6**: download the installed from https://csound.com/download.html

----------------

Linux
-----


Csound 7
~~~~~~~~

At the moment there are no packages for csound 7 for linux.

.. code-block:: bash

    sudo apt-get build-dep csound
    sudo apt-get install cmake
    git clone -b develop https://github.com/csound/csound.git csound
    cd csound
    mkdir build && cd build
    cmake ..
    cmake --build . --parallel
    sudo cmake --install .
    sudo ldconfig

Csound 6
~~~~~~~~

Alternatively *maelzel* works also with csound 6.

.. tab-set::

    .. tab-item:: Ubuntu / Debian

        .. code-block:: bash

            sudo apt install csound libcsnd-dev

    .. tab-item:: Arch

        .. code-block:: bash

            sudo pacman -S csound
