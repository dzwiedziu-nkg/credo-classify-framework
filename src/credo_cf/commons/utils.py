from typing import Optional, Any, Tuple
from time import time


def print_log(content: str, timer: Optional[float] = None) -> float:
    t = time()
    if timer is not None:
        print('%s (time: %.3fs)' % (content, t - timer))
    else:
        print(content)
    return t


def get_and_set(obj: dict, key: Any, default: Any) -> Any:
    """
    Dict's helper function: get value from key.
    When key value is not exists then will be set by default value before get.

    :param obj: dict object, may be modified
    :param key: key name
    :param default: default value
    :return: key value or default value
    """
    o = obj.get(key, default)
    obj[key] = o if o is not None else default
    return o


def get_and_add(obj: dict, key: Any, add: int or float, default: Any = 0) -> Any:
    """
    Dict's helper function: add value to key.
    When key value is not exists then will be set by default value before add.

    :param obj: dict object, will be modified
    :param key: key name of numeric value (int or float)
    :param add: value to add
    :param default: default value
    :return: value after add
    """
    get_and_set(obj, key, default)
    obj[key] += add
    return obj[key]


def join_tuple(tpl: Tuple[int, int], separator: str = 'x') -> str:
    return '%d%s%d' % (tpl[0], separator, tpl[1])


def get_resolution_key(d: dict) -> Tuple[int, int]:
    return d.get('width'), d.get('height')


def get_xy_key(d: dict) -> Tuple[int, int]:
    return d.get('x'), d.get('y')


def point_to_point_distance(p1: Tuple[int, int], p2: Tuple[int, int]):
    return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
