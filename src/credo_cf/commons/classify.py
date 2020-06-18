from typing import Dict, Any, List, Tuple, Callable, Optional

from credo_cf.commons.consts import CLASSIFIED, CLASS_ARTIFACT


ClassifyLambda = Callable[[dict], Optional[bool]]


def classify_by_lambda(detections: List[dict], func: ClassifyLambda, c: str = CLASS_ARTIFACT) -> Tuple[List[dict], List[dict]]:
    """
    Classify by simple callback function in ``func`` arg.
    :param detections: list of detections to classify
    :param func: callback function for classify
    :param c: the class will be set to ``classified`` object's key when func return True

    The ``func(obj)`` callback provided as arg:
      Should return logic value and set additional object's keys

      Args:
        * ``obj``: next element from ``detections``

      Return effect:
        * ``None``: object will not be added anywhere
        * ``True``: object will be append to list of classified, and set ``c`` to ``classified`` object's key
        * ``False``: object will be append to list of no classified

    :return: tuple of (list of classified, list of no classified)
    """
    ret_yes = []
    ret_no = []
    for d in detections:
        ret = func(d)
        if ret is None:
            pass
        elif ret:
            d[CLASSIFIED] = c
            ret_yes.append(d)
        else:
            ret_no.append(d)
    return ret_yes, ret_no


def classify_by_count_in_group(grouped: Dict[Any, List[dict]], often: int, obj_key: str) -> Tuple[List[dict], List[dict]]:
    """
    Classify helper for count in group based classifiers like (near_)hot_pixel(2) and too_often,
    :param grouped: detections grouped by key
    :param often: count in key threshold to classify as artifact
    :param obj_key: key in object to store the count in group value
    :return: tuple of (list of classified, list of no classified)
    """
    ret_yes = []
    ret_no = []
    for ds in grouped.values():
        for d in ds:
            count = len(ds)
            if count >= often:
                d[CLASSIFIED] = CLASS_ARTIFACT
                ret_yes.append(d)
            else:
                ret_no.append(d)
            d[obj_key] = count
    return ret_yes, ret_no
