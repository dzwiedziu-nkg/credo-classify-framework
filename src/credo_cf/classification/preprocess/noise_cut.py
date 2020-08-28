from typing import List
import numpy as np

from credo_cf import GRAY, NOISE_THRESHOLD
from credo_cf.commons.filters import base_xor_filter


def noise_cut_preprocess(detection: dict, rows: int = 3, cols: int = 3) -> dict:
    """
    Search for upper noise threshold by:

      1. Divide image per rows and columns.
      2. Search for brightest pixel in division.
      3. Return darkness of division.

    Note: rows and cols should dividing completely accordingly height and width of image.

    :param detection: detection with ``gray`` key
    :param rows: divides in horizontal
    :param cols: divides in vertical

    Required keys:
      * ``gray``: numpy grayscale image.

    Keys will be add:
      * ``noise_threshold``: boolean flag marked that filter has been executed and returned the result.

    Example::

      for detection in detections:
        noise_cut_preprocess(detection, 3, 3)

    :return: detection with added ``noise_threshold`` key
    """

    gray = detection.get(GRAY)
    h, w = gray.shape
    h_step = h // rows
    w_step = w // cols

    noise_threshold = 999999999

    h_pos = 0
    while h_pos < h:
        w_pos = 0
        while w_pos < w:
            sub_gray = gray[w_pos:w_pos + w_step, h_pos:h_step + h_step]
            local_max = np.max(sub_gray)
            noise_threshold = min(noise_threshold, local_max)
            w_pos += w_step
        h_pos += h_step

    detection[NOISE_THRESHOLD] = noise_threshold

    return detection
