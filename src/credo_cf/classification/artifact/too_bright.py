from typing import List, Tuple
from credo_cf.commons.utils import get_and_set
from credo_cf.commons.consts import BRIGHT_PIXELS,GOOD_BRIGHT, IMAGE
from credo_cf.image.image_utils import measure_darkness_brightest

"""
too_bright(detections,number,time)
detections        -   dict with good time detection in one device
bright_pixels     -   maximum number of bright pixels on the slice
threshold         -   the bright pixel has a brightness greater than the threshold (range 0 - 255)
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
