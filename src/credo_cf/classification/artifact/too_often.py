import itertools
from typing import List, Tuple

from credo_cf.commons.classify import classify_by_lambda
from credo_cf.commons.consts import ARTIFACT_TOO_OFTEN
from credo_cf.commons.grouping import group_by_timestamp_division
from credo_cf.commons.utils import get_and_set


def too_often(detections: List[dict], often: int = 10, time_window: int = 60000) -> Tuple[List[dict], List[dict]]:
    """
    Analyse by too often classifier.

    Note: detections should be grouped by ``device_id``.
    See: ``group_by_device_id()``.
    The additional group by ``resolution`` is not required, but is not prohibited.
    So it may be work in the same grouped detections than for ``(near_)hot_pixel(2)`` classifiers.

    :param detections: list of detections
    :param often: classified threshold
    :param time_window: timestamp distance

    Classifier work similar to ``near_hot_pixel2`` classifier but in this we use ``timestamp`` object's key as group key.
    At first, te detections from the same original image frame (with the same ``timestamp`` value) are counted as one detection.
    At second, all other detections who distance is less than ``time_window`` are counted to ``artifact_too_often`` object's key.

    The distance measurement of keys is the Euclidean distance between ``timestamp`` and ``timestamp'`` in 1D space.

    Required keys:
      * ``timestamp``: for group by the same original image frame, and count of detections in near

    Keys will be add:
      * ``artifact_hot_pixel``: count of detections in near ``timestamp``.
      * ``classified``: set to ``artifact`` when detection will be classified as too_often artifact.

    Example::

      for by_device_id in group_by_device_id(detections):
        too_often(by_device_id)

    :return: tuple of (list of classified, list of no classified)
    """
    grouped = group_by_timestamp_division(detections, 1)
    if len(grouped.keys()) == 1:
        # zero value guard (fill 0 when only one group of detections on the provided detections list)
        for group in grouped.values():
            for d in group:
                get_and_set(d, ARTIFACT_TOO_OFTEN, 0)
    else:
        to_compare = itertools.combinations(grouped.keys(), 2)
        for key, key_prim in to_compare:
            for d in [*grouped.get(key), *grouped.get(key_prim)]:
                get_and_set(d, ARTIFACT_TOO_OFTEN, 0)
                if abs(key - key_prim) < time_window:
                    d[ARTIFACT_TOO_OFTEN] += 1
    return classify_by_lambda(detections, lambda x: x.get(ARTIFACT_TOO_OFTEN) >= often)
