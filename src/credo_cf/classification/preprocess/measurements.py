import math

import numpy as np
import cv2

from credo_cf import GRAY, NOISE_THRESHOLD, WHOLE_AREA, FLOOD_AREA, WHOLE_DIAGONAL, FLOOD_DIAGONAL


def get_bound_size(a, s: int = 0):
    w = np.max(a[0]) - np.min(a[0])
    h = np.max(a[1]) - np.min(a[1])
    return math.sqrt(((w - 2*s)**2)+((h - 2*s)**2))


def measurements(detection: dict, x: int = 30, y: int = 30, dilate_size: int = 3, threshold: int = None) -> dict:
    matrix = detection.get(GRAY)
    threshold = detection.get(NOISE_THRESHOLD) if threshold is None else threshold

    whole_area = np.count_nonzero(matrix >= threshold)

    marked = np.where(matrix >= threshold, 255, 0).astype(np.uint8)

    k = dilate_size * 2 + 1
    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(marked, kernel, iterations=1)
    dilate_mask = dilated - marked

    mask = np.zeros(np.asarray(dilated.shape) + 2, dtype=np.uint8)
    start_pt = (y, x)
    cv2.floodFill(dilated, mask, start_pt, 255, flags=4)
    mask = mask[1:-1, 1:-1] - (dilate_mask / 255)

    flood_area = np.count_nonzero(mask == 1)

    whole_diagonal = get_bound_size(np.where(matrix >= threshold))
    flood_diagonal = get_bound_size(np.where(mask == 1))

    detection[WHOLE_AREA] = whole_area
    detection[FLOOD_AREA] = flood_area
    detection[WHOLE_DIAGONAL] = whole_diagonal
    detection[FLOOD_DIAGONAL] = flood_diagonal

    return detection
