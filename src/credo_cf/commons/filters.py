from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np


def base_xor_filter(image: object, reference: object, var: int = 70) -> object:
    """
    Tool function implementing the core XOR filter operation on pair of images, where first image 
    is considered as processed one and second image provides reference background
    :param image: numpy image needs to be filtered
    :param reference: numpy image which provides reference background
    :param var: threshold value indicating the noise level, ``uint8`` type number 

    :return: filtered image
    """

    im = np.array(image)
    ref = np.array(reference)

    im = np.dot(im[...,:3], (0.2989, 0.5870, 0.1140))
    ref = np.dot(ref[...,:3], (0.2989, 0.5870, 0.1140))

    diff = (np.abs(im.astype('int64') -ref.astype('int64')) <= var)

    mask = np.ones(len(im)).astype(bool) #same shape as the array
    mask = np.logical_and(mask, diff)
    res = im.copy()
    res[mask] = 0 # replace 0's with whatever value

    return res.astype('uint8')
