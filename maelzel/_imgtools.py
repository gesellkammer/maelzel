from __future__ import annotations
from PIL import Image, ImageChops


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
