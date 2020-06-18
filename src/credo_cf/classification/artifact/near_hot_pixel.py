from typing import List, Tuple

from credo_cf.commons.classify import classify_by_count_in_group
from credo_cf.commons.consts import CLASSIFIED, CLASS_ARTIFACT, X, Y, ARTIFACT_NEAR_HOT_PIXEL, ARTIFACT_NEAR_HOT_PIXEL_REFXY
from credo_cf.commons.utils import point_to_point_distance, get_and_set


def near_hot_pixel(detections: List[dict], often: int = 3, distance: float = 5) -> Tuple[List[dict], List[dict]]:
    """
    Analyse by near hot pixel classifier.

    Note: detections should be grouped by ``device_id`` and ``resolution``.
    See: ``group_by_device_id()`` and ``group_by_resolution()``.

    :param detections: list of detections
    :param often: classified threshold
    :param distance: distance in px to group as one hot pixel

    It is extension of hot pixel filter. In hot pixel we get ``(X, Y)`` as key of group.
    In near hot pixel the first detection provide key from ``(X, Y)``
    but ``(X', Y')`` from next detection are compared to previous keys
    and when distance is less than ``distance`` arg then append to key and break loop.
    When loop was end and near keys not found the ``(X', Y')`` make new key.

    The distance measurement of keys is the Euclidean distance between ``(X, Y)`` and ``(X', Y')`` on 2D plane.

    When in one key we have more than ``often`` detections, we classify all as near_hot_pixel artifact.

    Required keys:
      * ``X`` and ``Y``: coordinates of detection on original frame

    Keys will be add:
      * ``artifact_near_hot_pixel``: count of detections in the same key.
      * ``artifact_near_hot_pixel_refxy``: value of the used key.
      * ``classified``: set to ``artifact`` when detection will be classified as near_hot_pixel artifact.

    Example::

      for by_device_id in group_by_device_id(detections):
        for by_resolution in group_by_resolution(by_device_id)
          near_hot_pixel(by_resolution)

    :return: tuple of (list of classified, list of no classified)
    """
    grouped = {}
    for detection in detections:
        key_prim = (detection.get(X), detection.get(Y))
        for key in grouped.keys():
            if point_to_point_distance(key, key_prim) < distance:
                key_prim = key
                break
        detection[ARTIFACT_NEAR_HOT_PIXEL_REFXY] = key_prim
        get_and_set(grouped, key_prim, []).append(detection)

    return classify_by_count_in_group(grouped, often, ARTIFACT_NEAR_HOT_PIXEL)
