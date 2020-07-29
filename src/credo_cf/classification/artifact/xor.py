from typing import List
import numpy as np

from credo_cf.commons.filters import base_xor_filter


def xor_filter(detections: List[dict], mode: str = 'pair', var: int = 30) -> List[dict]:
    """
    The function runs a discrete XOR filter what enables to remove some noise existing on image. The filter works on pairs of images
    where one of them is considered as filtered and the other one provide the reference background.

    Note: detections should be grouped by ``device_id`` and ``resolution``.
    See: ``group_by_device_id()`` and ``group_by_resolution()``.

    :param detections: list of detections
    :param mode: decision key
      * ``pair``: defines pairwise process where adjacent images are agruments
      * ``avg`` means that the reference image is created  by averenge over the whole input collections of images
    :param var: threshold value indicating the noise level, ``uint8`` type number

    Required keys:
      * ``gray``: numpy grayscale image.

    Keys will be add:
      * ``xor_filter_passed``: boolean flag marked that filter has been executed and returned the result.

    Keys will be modified:
      * ``gray``: set to new version of numpy grayscale image when the XOR filter has been applied.

    Example::

      for by_device_id in group_by_device_id(detections):
        for by_resolution in group_by_resolution(by_device_id)
          xor_filter(by_resolution)

    :return: list of detections where the XOR filter has been applied.
    """

    size = len(detections)

    if mode == 'pair':

        for i in range(1, size):
            detections[i]['xor_filter_passed'] = True
            detections[i]['gray'] = base_xor_filter(detections[i]['gray'], detections[i-1]['gray'], var)

        detections[0]['xor_filter_passed'] = True
        detections[0]['gray'] = base_xor_filter(detections[0]['gray'], detections[1]['gray'], var)

    elif mode == 'avg':

        avg_image = np.zeros_like(detections[0]['gray'])
        for d in detections:
            avg_image += d['gray']
        avg_image = avg_image / size

        for i in range(size):
            detections[i]['xor_filter_passed'] = True
            detections[i]['gray'] = base_xor_filter(detections[i]['gray'], avg_image, var)

    return detections
