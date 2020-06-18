from typing import List, Tuple

from credo_cf.commons.classify import classify_by_count_in_group
from credo_cf.commons.consts import CLASSIFIED, CLASS_ARTIFACT, ARTIFACT_HOT_PIXEL, X, Y
from credo_cf.commons.grouping import group_by_lambda


def hot_pixel(detections: List[dict], often: int = 3) -> Tuple[List[dict], List[dict]]:
    """
    Analyse by hot pixel classifier.

    Note: detections should be grouped by ``device_id`` and ``resolution``.
    See: ``group_by_device_id()`` and ``group_by_resolution()``.

    :param detections: list of detections
    :param often: classified threshold

    When in one ``(X, Y)`` coordinates on original frame we have more than ``often`` detections, we classify all as hot_pixel artifact.

    Required keys:
      * ``X`` and ``Y``: coordinates of detection on original frame

    Keys will be add:
      * ``artifact_hot_pixel``: count of detections in the same ``(X, Y)`` coordinates on original frame.
      * ``classified``: set to ``artifact`` when detection will be classified as hot_pixel artifact.

    Example::

      for by_device_id in group_by_device_id(detections):
        for by_resolution in group_by_resolution(by_device_id)
          hot_pixel(by_resolution)

    :return: tuple of (list of classified, list of no classified)
    """
    grouped = group_by_lambda(detections, lambda x, ret: (x.get(X), x.get(Y)))
    return classify_by_count_in_group(grouped, often, ARTIFACT_HOT_PIXEL)
