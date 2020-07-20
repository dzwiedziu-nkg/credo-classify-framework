from typing import List, Callable

import numpy as np
from PIL import Image
from credo_cf.io.io_utils import decode_base64
from io import BytesIO
from credo_cf import IMAGE,GRAY
from credo_cf.commons.utils import get_and_set

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

def gray_la(colors: List[int]) -> float:
    """
    Standard LA form PIL.Image -> grayscale transformation.

    :param colors: [R, G, B] values
    :return: one grayscale color
    """
    r, g, b = colors
    return 0.299 * r + 0.587 * g + 0.114 * b

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



def rgb2gray(rgb, coeff=(1.0, 1.0, 1.0), normalize=True):
    """
    Conversion from RGB (3-channels) image to grayscale image.
    params rgb: numpy image
    params coeff: vector of weight values
    params normalize: flag, if True the grayscale image will be normalized if False will not

    return: grayscale numpy image
    """
    dot_product = np.dot(rgb[..., :3].astype('int64'), coeff)
    gray = (dot_product / sum(coeff)).astype('uint8') if (normalize == True) else dot_product.astype('int64')

    return gray


def convert_to_gray_scale(detection: dict, grayscale = "LA"):
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
    scala = (1.0, 1.0, 1.0)
    if grayscale == "LA":
        scala = (0.299, 0.587, 0.114)
    if grayscale == "RGB":
        scala = (0.07,0.72,0.21)


    img = detection["frame_content"]
    decode = decode_base64(img)
    imga = np.array(Image.open(BytesIO(decode)))
    img = rgb2gray(imga, scala)
    detection[GRAY]=img
    return detection