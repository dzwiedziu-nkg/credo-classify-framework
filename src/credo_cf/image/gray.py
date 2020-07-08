from typing import List, Callable

import numpy as np

from credo_cf import IMAGE, GRAY


def gray_max(colors: List[int]) -> float:
    """
    Convert to grayscale: get brightest value from channels.

    :param colors: [R, G, B] values
    :return: one grayscale color
    """
    return max(colors)


def gray_min(colors: List[int]) -> float:
    """
    Convert to grayscale: get darkness value from channels.

    :param colors: [R, G, B] values
    :return: one grayscale color
    """
    return min(colors)


def gray_avg(colors: List[int]) -> float:
    """
    Convert to grayscale: make average value of channels.

    :param colors: [R, G, B] values
    :return: one grayscale color
    """
    return sum(colors) / len(colors)


def gray_rgb(colors: List[int]) -> float:
    """
    Standard RGB -> grayscale transformation.

    :param colors: [R, G, B] values
    :return: one grayscale color
    """
    r, g, b = colors
    return 0.07 * r + 0.72 * g + 0.21 * b


def convert_to_gray(detection: dict, grayscale: Callable[[List[int]], float] = gray_rgb):
    """
    Convert RGB image to grayscale.

    Required keys:
      * ``image``: RGB bitmap in numpy

    Keys will be add:
      * ``gray``: grayscale bitmap in numpy

    :param detection: detection object with required keys, new keys will be add or overwrite
    :param grayscale: method used to transform RGB channels to one grayscale channel
    :return: the same detection object from param, usable for lambda chain
    """
    img = detection[IMAGE]
    detection[GRAY] = np.apply_along_axis(grayscale, 2, img)
    return detection
