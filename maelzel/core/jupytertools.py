from __future__ import annotations
from . import environment

if not environment.insideJupyter:
    raise ImportError("This module is only available inside a jupyter session")

import os
from ._common import logger

import IPython
from IPython.core.display import display, Image

import emlib.img
from .workspace import getConfig

# ipywidgets is a dependency of jupyter so it should be available
import ipywidgets


def setJupyterHookForClass(cls, func, fmt='image/png') -> None:
    """
    Register func as a displayhook for class `cls`
    """
    if not environment.insideJupyter:
        logger.debug("_setJupyterHookForClass: not inside IPython/jupyter, skipping")
        return
    ip = IPython.get_ipython()
    formatter = ip.display_formatter.formatters[fmt]
    return formatter.for_type(cls, func)


def jupyterMakeImage(path: str, scalefactor=1.0) -> Image:
    """
    Makes a jupyter Image, which can be displayed inline inside a notebook

    Args:
        path: the path to the image file
        scalefactor: a factor to scale the image

    Returns:
        an IPython.core.display.Image

    """
    if not environment.insideJupyter:
        raise RuntimeError("Not inside a Jupyter session")

    width, height = emlib.img.imgSize(path)
    if scalefactor != 1.0:
        width *= scalefactor
    return Image(filename=path, embed=True, width=width)


def jupyterShowImage(path: str, scalefactor=1.0, maxwidth: int = None):
    """
    Show an image inside (inline) of a jupyter notebook

    Args:
        path: the path to the image file
        scalefactor: a factor to scale the image
        maxwidth: max. width of the image, in pixels

    """
    img = jupyterMakeImage(path, scalefactor=scalefactor)
    if maxwidth is not None and img.width > maxwidth:
        img.width = maxwidth
    return display(img)


def showPng(pngpath: str, forceExternal=False, app='', scalefactor=1.0,
            maxwidth: int | None = None
            ) -> None:
    """
    Show a png either inside jupyter or with an external app

    Args:
        pngpath: the path to a png file
        forceExternal: if True, it will show in an external app even
            inside jupyter. Otherwise it will show inside an external
            app if running a normal session and show an embedded
            image if running inside a notebook
        scalefactor: a factor to apply when showing a png inline
        app: used if a specific external app is needed. Otherwise the os
            defined app is used
    """
    if environment.insideJupyter and not forceExternal:
        jupyterShowImage(pngpath, scalefactor=scalefactor, maxwidth=maxwidth)
    else:
        environment.openPngWithExternalApplication(pngpath, app=app)


def displayButton(buttonText: str, callback: Callable[[], None]
                  ) -> None:
    """
    Create and display an html button inside a jupyter notebook

    If not inside a jupyter notebook this function will raise RuntimeError

    Args:
        buttonText: the text of the button
        callback: the function to call when the button is pressed. This function
            takes no arguments and should not return anything
    """
    if not environment.insideJupyter:
        raise RuntimeError("This function is only available when running inside a jupyter notebook")

    button = ipywidgets.Button(description=buttonText)
    output = ipywidgets.Output()

    def clicked(b):
        with output:
            callback()

    button.on_click(clicked)
    display(button, output)
