from __future__ import annotations
import emlib.img
from ._common import logger
from .workspace import activeConfig

from . import environment


if environment.insideJupyter:
    from IPython.core.display import (display as jupyterDisplay, HTML as JupyterHTML,
                                      Image as JupyterImage)


def setJupyterHookForClass(cls, func, fmt='image/png'):
    """
    Register func as a displayhook for class `cls`
    """
    if not environment.insideJupyter:
        logger.debug("_setJupyterHookForClass: not inside IPython/jupyter, skipping")
        return
    import IPython
    ip = IPython.get_ipython()
    formatter = ip.display_formatter.formatters[fmt]
    return formatter.for_type(cls, func)


def jupyterMakeImage(path: str, scalefactor:float = None) -> JupyterImage:
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

    scalefactor = scalefactor if scalefactor is not None else activeConfig()['show.scaleFactor']
    if scalefactor != 1.0:
        imgwidth, imgheight = emlib.img.imgSize(path)
        width = imgwidth * scalefactor
    else:
        width = None
    return JupyterImage(filename=path, embed=True, width=width)


def jupyterShowImage(path: str, scalefactor:float = None):
    """
    Show an image inside (inline) of a jupyter notebook

    Args:
        path: the path to the image file
        scalefactor: a factor to scale the image

    """
    if not environment.insideJupyter:
        logger.error("jupyter is not available")
        return

    img = jupyterMakeImage(path, scalefactor=scalefactor)
    return jupyterDisplay(img)


def pngShow(pngpath:str, forceExternal=False, app:str='') -> None:
    """
    Show a png either inside jupyter or with an external app

    Args:
        pngpath: the path to a png file
        forceExternal: if True, it will show in an external app even
            inside jupyter. Otherwise it will show inside an external
            app if running a normal session and show an embedded
            image if running inside a notebook
        app: used if a specific external app is needed. Otherwise the os
            defined app is used
    """
    if environment.insideJupyter and not forceExternal:
        jupyterShowImage(pngpath)
    else:
        environment.openPngWithExternalApplication(pngpath, app=app)


def m21JupyterHook(enable=True) -> None:
    """
    Set an ipython-hook to display music21 objects inline on the
    ipython notebook

    Args:
        enable: if True, the hook will be set up and enabled
            if False, the hook is removed
    """
    if not environment.insideJupyter:
        logger.debug("m21JupyterHook: not inside ipython/jupyter, skipping")
        return
    from IPython.core.getipython import get_ipython
    from IPython.core import display
    from IPython.display import Image, display
    import music21 as m21
    ip = get_ipython()
    formatter = ip.display_formatter.formatters['image/png']
    if enable:
        def showm21(stream: m21.stream.Stream):
            cfg = activeConfig()
            fmt = cfg['m21.displayhook.format']
            filename = str(stream.write(fmt))
            if fmt.endswith(".png") and cfg['html.theme'] == 'dark':
                emlib.img.pngRemoveTransparency(filename)
            return display(Image(filename=filename))
            # return display.Image(filename=filename)._repr_png_()

        dpi = formatter.for_type(m21.Music21Object, showm21)
        return dpi
    else:
        logger.debug("disabling display hook")
        formatter.for_type(m21.Music21Object, None)
