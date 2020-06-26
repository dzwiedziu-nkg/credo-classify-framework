from typing import List, Tuple
from credo_cf.commons.utils import get_and_set
from credo_cf.commons.consts import GOOD_BRIGHT, IMAGE,DARKNESS, BRIGHTEST
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

        assert image.get(DARKNESS) is not None
        assert image.get(BRIGHTEST) is not None

        brightest = image.get(BRIGHTEST)

        get_and_set(image, GOOD_BRIGHT, "False")

        if (brightest > 0 and brightest < bright_pixels):
            image[GOOD_BRIGHT] = "True"
            good.append(image)
        else:
            bad.append(image)
    return good, bad
