from typing import List, Tuple
from credo_cf.commons.utils import get_and_set
from credo_cf.commons.consts import BRIGHT_PIXELS,GOOD_BRIGHT, IMAGE
from credo_cf.image.image_utils import measure_darkness_brightest

"""
    Analysis of the detection slice by checking the brightness of the pixels

    :param detections: list of detections
    :param bright_pixels (int) : maximum number of bright pixels on the slice (wycinek)
    :paramthreshold (int) – value of pixels for the bright_pixel (i.e bright pixel is when pixel have value >70)


    The code from "frame_content" is downloaded and saved as a disk image, loaded by functions from the "Image" library 
    and converted to gray scale LA (convert 'LA') - (8-bit pixels, black and white) with ALPHA. 
    After conversion, we analyze pixel by pixel counting bright pixels according to the given threshold, 
    at the end we count how many bright pixels were checking for more than the number of bright pixels, 
    if we classify more as an artifact.


    Required keys:
      * ``frame_content``: base64 image our detection

    Keys will be add:
      * ``BRIGHT_PIXELS``: number of bright pixels
      * ``GOOD_BRIGH``: True,or False.

    gray change used:
    https://pillow.readthedocs.io/en/stable/reference/Image.html

    Example::

      good,bad =  bright_detections(detections,70,70)

    :return: tuple of (list of good detections, list of bad detections)
    Return type
    Tuple[List[dict], List[dict]]
"""

def bright_detections(detections: List[dict], bright_pixels=70, threshold=70) -> Tuple[List[dict], List[dict]]:
    good = []
    bad = []
    for image in detections:
        assert image.get(IMAGE) is not None
        measure_darkness_brightest(image)

        hit_img = image.get(IMAGE).convert('LA')
        width, height = hit_img.size
        pixelMap = hit_img.load()
        bright_point = 0
        for cy in range(height):
            for cx in range(width):
                a = pixelMap[cx,cy]  # a -pixel table ((p), (t)) we are only interested in p; t always == 255
                a0 = a[0]
                if threshold < a0:
                    bright_point += 1
        get_and_set(image, BRIGHT_PIXELS, bright_point)
        get_and_set(image, GOOD_BRIGHT, "False")

        if (bright_point > 0 and bright_point < bright_pixels):
            image[GOOD_BRIGHT] = "True"
            good.append(image)
        else:
            bad.append(image)
    return good, bad