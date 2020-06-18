import itertools
from typing import List, Tuple

from credo_cf.commons.classify import classify_by_lambda
from credo_cf.commons.consts import X, Y, ARTIFACT_NEAR_HOT_PIXEL2
from credo_cf.commons.utils import point_to_point_distance, get_and_set


def near_hot_pixel2(detections: List[dict], often: int = 3, distance: float = 5) -> Tuple[List[dict], List[dict]]:
    """
    Analyse by near hot pixel v2 classifier.

    Note: detections should be grouped by ``device_id`` and ``resolution``.
    See: ``group_by_device_id()`` and ``group_by_resolution()``.

    :param detections: list of detections
    :param often: classified threshold
    :param distance: distance in px to group as one hot pixel

    It is extension of hot pixel filter. In hot pixel we get ``(X, Y)`` as key of group.
    In near hot pixel v2 all other detections who distance is less than ``distance`` are counted to ``artifact_near_hot_pixel2`` object's key.

    The distance measurement of keys is the Euclidean distance between ``(X, Y)`` and ``(X', Y')`` on 2D plane.

    When in one key we have more than ``often`` detections, we classify all as near_hot_pixel2 artifact.

    Required keys:
      * ``X`` and ``Y``: coordinates of detection on original frame

    Keys will be add:
      * ``artifact_near_hot_pixel2``: count of detections in near distance.
      * ``classified``: set to ``artifact`` when detection will be classified as near_hot_pixel artifact.

    Example::

      for by_device_id in group_by_device_id(detections):
        for by_resolution in group_by_resolution(by_device_id)
          near_hot_pixel2(by_resolution)

    :return: tuple of (list of classified, list of no classified)
    """
    to_compare = itertools.combinations_with_replacement(detections, 2)
    for d, d_prim in to_compare:
        key = (d.get(X), d.get(Y))
        key_prim = (d_prim.get(X), d_prim.get(Y))

        get_and_set(d, ARTIFACT_NEAR_HOT_PIXEL2, 0)
        get_and_set(d_prim, ARTIFACT_NEAR_HOT_PIXEL2, 0)

        if point_to_point_distance(key, key_prim) < distance:
            d[ARTIFACT_NEAR_HOT_PIXEL2] += 1
            d_prim[ARTIFACT_NEAR_HOT_PIXEL2] += 1

    return classify_by_lambda(detections, lambda x: x.get(ARTIFACT_NEAR_HOT_PIXEL2) >= often)
