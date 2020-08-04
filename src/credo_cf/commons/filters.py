import numpy as np


def base_xor_filter(im: np.float, ref: np.float, var: int = 70) -> np.float:
    """
    Tool function implementing the core XOR filter operation on pair of images, where first image 
    is considered as processed one and second image provides reference background.

    :param im: numpy image needs to be filtered
    :param ref: numpy image which provides reference background
    :param var: threshold value indicating the noise level

    :return: filtered image
    """

    diff = (np.abs(im - ref) <= var)

    mask = np.ones(len(im)).astype(bool)  # same shape as the array
    mask = np.logical_and(mask, diff)
    res = im.copy()
    res[mask] = 0  # replace 0's with whatever value

    return res
