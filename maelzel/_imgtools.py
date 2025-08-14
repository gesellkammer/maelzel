from __future__ import annotations

import os


import typing
if typing.TYPE_CHECKING:
    from PIL import Image


def imageAutocrop(img: Image.Image | str, bgcolor: str | tuple[int, int, int]
                  ) -> Image.Image | None:
    """
    Crop an image to its content, automatically

    Args:
        img: PIL image to crop
        bgcolor: background color

    Returns:
        cropped image or None if the operation was not successful

    """
    from PIL import Image, ImageChops
    imgobj = img if isinstance(img, Image.Image) else Image.open(img)
    if imgobj.mode != "RGB":
        imgobj = imgobj.convert("RGB")
    bg = Image.new("RGB", imgobj.size, bgcolor)
    diff = ImageChops.difference(imgobj, bg)
    bbox = diff.getbbox()
    if bbox:
        return imgobj.crop(bbox)
    return None


def imagefileAutocrop(imgfile: str, outfile: str, bgcolor: str | tuple[int, int, int]
                      ) -> bool:
    """
    Crop an image in a file to its content, save it to another file

    Args:
        imgfile: original image file to crop 
        outfile: output file
        bgcolor: background color

    Returns:
        True if OK, False otherwise

    """
    imgobj = imageAutocrop(imgfile, bgcolor=bgcolor)
    if imgobj is None:
        return False
    imgobj.save(outfile)
    return True


def imgSize(imgfile: str) -> tuple[int, int]:
    """
    Size of the image

    Returns:
        a tuple (width, height)
    """
    import imagesize
    width, height = imagesize.get(imgfile)
    assert isinstance(width, int) and width >= 0
    assert isinstance(height, int) and height >= 0
    return width, height

    # ext = os.path.splitext(imgfile)[-1]
    # if ext == '.png':
    #     import png
    #     r = png.Reader(imgfile)
    #     r.preamble()
    #     return r.width, r.height
    # else:
    #     import emlib.img
    #     return emlib.img.imgSize(imgfile)


def _pypngReadImageAsBase64(imgpath: str) -> tuple[bytes, int, int]:
    import base64
    import png
    r = png.Reader(imgpath)
    width, height, pixels, info = r.read_flat()
    img64 = base64.b64encode(pixels.tobytes())
    return img64, width, height


def _pyllowReadAsBase64(imgpath: str) -> tuple[bytes, int, int]:
    import io
    import PIL.Image
    try:
        import pybase64 as base64
    except ImportError:
        import base64
    im = PIL.Image.open(imgpath)
    buffer = io.BytesIO()
    im.save(buffer, format='PNG')
    imgbytes = base64.b64encode(buffer.getvalue())
    width, height = im.size
    return imgbytes, width, height


def readImageAsBase64(imgpath: str) -> tuple[bytes, int, int]:
    """
    Read an image as base64

    Args:
        imgpath: the path to the image

    Returns:
        a tuple ``(imagebytes: bytes, width: int, height: int)``

    .. seealso:: :func:`htmlImage64`
    """
    return _pyllowReadAsBase64(imgpath)
