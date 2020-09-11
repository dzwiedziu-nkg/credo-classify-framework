import numpy as np

from credo_cf import GRAY, NOISE_THRESHOLD, CLEARLY


def noise_measure_preprocess(detection: dict, rows: int = 3, cols: int = 3, div: float = 4) -> dict:
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

    min_of_max = 999999999
    global_max = 0

    h_pos = 0
    while h_pos < h:
        w_pos = 0
        while w_pos < w:
            sub_gray = gray[w_pos:w_pos + w_step, h_pos:h_pos + h_step]
            local_max = np.max(sub_gray)
            min_of_max = min(min_of_max, local_max)
            w_pos += w_step
            global_max = max(local_max, global_max)
        h_pos += h_step

    clearly = global_max - min_of_max
    detection[CLEARLY] = clearly
    detection[NOISE_THRESHOLD] = clearly / div + min_of_max

    return detection
