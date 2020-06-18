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
    o = obj.get(key, default)
    obj[key] = o
    return o


def join_tuple(tpl: Tuple[int, int], separator: str = 'x') -> str:
    return '%d%s%d' % (tpl[0], separator, tpl[1])


def get_resolution_key(d: dict) -> Tuple[int, int]:
    return d.get('width'), d.get('height')


def get_xy_key(d: dict) -> Tuple[int, int]:
    return d.get('x'), d.get('y')


def point_to_point_distance(p1: Tuple[int, int], p2: Tuple[int, int]):
    return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
