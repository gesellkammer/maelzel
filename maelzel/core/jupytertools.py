from __future__ import annotations
import emlib.img
from ._common import logger
from .workspace import getConfig
import tempfile
import os
from . import environment


if environment.insideJupyter:
    from IPython.core.display import (display as jupyterDisplay, Image as JupyterImage)


def setJupyterHookForClass(cls, func, fmt='image/png') -> None:
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


def jupyterMakeImage(path: str, scalefactor=1.0) -> JupyterImage:
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
    return JupyterImage(filename=path, embed=True, width=width)


def jupyterShowImage(path: str, scalefactor=1.0, maxwidth: int = None):
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
    if maxwidth is not None and img.width > maxwidth:
        img.width = maxwidth
    return jupyterDisplay(img)


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

    try:
        import music21 as m21
    except ImportError:
        raise ImportError("Cannot set the jupyter hook for music21 since it is not installed. "
                          "Install it via 'pip install music21'")
    
    from IPython.core.getipython import get_ipython
    from IPython.core import display
    from IPython.display import Image, display
    import maelzel.scoring
    ip = get_ipython()
    formatter = ip.display_formatter.formatters['image/png']
    if enable:
        def showm21(stream: m21.stream.Stream):
            cfg = getConfig()
            xmlfile = tempfile.mktemp(suffix='.musicxml', dir='.')
            outfile = tempfile.mktemp(suffix='.png', dir='.')
            stream.write('musicxml', xmlfile)
            assert os.path.exists(xmlfile), f"Failed to write {xmlfile}!"
            maelzel.scoring.render.renderMusicxml(xmlfile, outfile=outfile)
            if cfg['htmlTheme'] == 'dark':
                emlib.img.pngRemoveTransparency(outfile)
            imgwidth, imgheight = emlib.img.imgSize(outfile)
            scaleFactor = cfg['show.scaleFactor']
            width = min(cfg['show.jupyterMaxImageWidth'], imgwidth*scaleFactor)
            img = Image(filename=outfile, width=width)
            if os.path.exists(xmlfile):
                os.remove(xmlfile)
            if os.path.exists(outfile):
                os.remove(outfile)
            display(img)

        dpi = formatter.for_type(m21.Music21Object, showm21)
        return dpi
    else:
        logger.debug("disabling display hook")
        formatter.for_type(m21.Music21Object, None)
