from typing import List, Tuple

import math
import itertools

import numpy as np
from numpy import unravel_index
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra

from credo_cf.commons.consts import GRAY, NKG_MASK, NKG_CORES, NKG_PATH, NKG_DERIVATIVE, NKG_DIRECTION, NKG_VALUES

KERNEL_HV = [
    (-1, 0),
    (0, -1),
    (1, 0),
    (0, 1)
]


KERNEL_HVD = [
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1)
]


ANGLES = {
    (-1, 0): -90,
    (-1, -1): -45,
    (0, -1): 0,
    (1, -1): 45,
    (1, 0): 90,
    (1, 1): 135,
    (0, 1): 180,
    (-1, 1): -135
}


def marking_recursion(img: np.ndarray, mark: np.ndarray, color: int, point: Tuple[int, int], kernel: List[Tuple[int, int]], spread_on_flat: bool = False, threshold: int = 0, local_mark: np.ndarray = None):
    """
    Recursion marking from local maximum to down.

    :param img: input image, warning! marked pixels will be filled by 0
    :param mark: marked area, warning! marked pixels will be filled by color value
    :param color: color in marked area
    :param point: start of point (the local maximum)
    :param kernel: kernel of recursion spreading
    :param spread_on_flat: continue recursion spreading on flat
    :param threshold: recursion force to stop at threshold
    :param local_mark: local mark, use at start of recursion
    :return:
    """

    _lm = local_mark if local_mark is not None else np.zeros(img.shape, dtype=np.bool)

    center_color = img[point]
    img[point] = 0
    mark[point] = color or center_color
    _lm[point] = True

    nexts = []

    for k in kernel:
        p2 = (point[0] + k[0], point[1] + k[1])

        # no enter recursion to out of image boundary
        if not (0 <= p2[0] < np.size(img, 0) and 0 <= p2[1] < np.size(img, 1)):
            continue

        next_pixel_marked = _lm[p2]

        # no enter recursion to just marked pixels
        if next_pixel_marked:
            continue

        # no enter recursion
        next_pixel_color = img[p2]
        if next_pixel_color <= threshold:
            continue

        if spread_on_flat:
            if center_color < next_pixel_color:
                continue
        else:
            if center_color <= next_pixel_color:
                continue

        mark[p2] = color
        _lm[p2] = True
        nexts.append((img, mark, color, p2, kernel, spread_on_flat, threshold, _lm))

    for n in nexts:
        marking_recursion(*n)


def find_all_maximums(img: np.ndarray, max_maximums: int = 100, recursion_func=None, kernel: List[Tuple[int, int]] = None, spread_on_flat: bool = False, threshold: int = 0):
    _kernel = kernel or KERNEL_HVD
    _recursion = recursion_func or marking_recursion

    _img = img.copy()
    mask = np.zeros(img.shape, dtype=np.uint8)
    cores = []

    values = []

    for i in range(1, max_maximums):
        core = unravel_index(_img.argmax(), _img.shape)
        core_value = _img[core]
        if core_value < threshold:
            break
        cores.append(core)
        values.append(core_value)
        _recursion(_img, mask, core_value, core, _kernel, spread_on_flat, threshold=threshold)
    return {'cores': cores, 'values': values, 'mask': mask}


def nkg_mark_hit_area(detection: dict, recursion_func=None, kernel: List[Tuple[int, int]] = None):
    ret = find_all_maximums(detection.get(GRAY), recursion_func=recursion_func, kernel=kernel, spread_on_flat=False)

    find_median = ret['values']

    core_median = find_median[len(find_median) // 2]
    # threshold = (find_median[0] - core_median) // 10 + core_median
    threshold = core_median + 5

    ret2 = find_all_maximums(detection.get(GRAY), recursion_func=recursion_func, kernel=kernel, spread_on_flat=True, threshold=threshold)
    detection[NKG_MASK] = ret2['mask']
    detection[NKG_VALUES] = ret2['values']
    detection[NKG_CORES] = ret2['cores']

    return detection


# Defines a translation from 2 coordinates to a single number
def to_index(img, y, x):
    return y * img.shape[1] + x


# Defines a reversed translation from index to 2 coordinates
def to_coordinates(img, index):
    return index // img.shape[1], index % img.shape[1]


# make path by dijkstry
def bitmap_to_graph(img, mask):
    # graph = csr_matrix(img)
    am_ind = unravel_index(img.argmax(), img.shape)
    am = img[am_ind]

    # A sparse adjacency matrix.
    # Two pixels are adjacent in the graph if both are painted.
    adjacency = dok_matrix((img.shape[0] * img.shape[1],
                            img.shape[0] * img.shape[1]))

    # The following lines fills the adjacency matrix by
    directions = list(itertools.product([0, 1, -1], [0, 1, -1]))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            pix1 = img[i, j]
            if not mask[i, j]:
                continue
            for y_diff, x_diff in directions:
                pix2 = img[i + y_diff, j + x_diff]
                if not mask[i + y_diff, j + x_diff]:
                    continue
                adjacency[to_index(img, i, j), to_index(img, i + y_diff, j + x_diff)] = float(am * 2 - pix1 - pix2)**16 + 1  #abs(int(pix2) - int(pix1))*2 + 1
    return adjacency


def search_path_dijkstra(img, adjacency, src, dst):
    # We chose two arbitrary points, which we know are connected
    source = to_index(img, *src)
    target = to_index(img, *dst)

    # Compute the shortest path between the source and all other points in the image
    _, predecessors = dijkstra(adjacency, directed=True, indices=[source],
                               unweighted=False, return_predecessors=True)

    # Constructs the path between source and target
    pixel_index = target
    pixels_path = []
    if math.isinf(_[0][target]):
        return {'dist': math.inf }
    # print('source: ' + str(_[0][source]**(1.0/16.0)))
    # print('target: ' + str(_[0][target]**(1.0/16.0)))
    i = 0
    #pixels_path.append(target)
    while pixel_index != source:
        i += 1
        pixels_path.append(pixel_index)
        pixel_index = predecessors[0, pixel_index]
        # print(str(i) + '.: ' + str(_[0][pixel_index]**(1.0/16.0)))
    pixels_path.append(source)
    return {'dist': _[0][target]**(1.0/16.0), 'path': pixels_path}


def unraveled_to_alphabet(unraveled):
    ret = []
    old = None
    for u in unraveled:
        if old != None:
            d = (old[0] - u[0], old[1] - u[1])
            ret.append(ANGLES[d])
        old = u
    return ret


def unraveled_to_alphabet2(unraveled):
    ret = []
    old2 = None
    old = None
    for u in unraveled:
        if old is not None and old2 is not None:
            a = ANGLES[(old2[0] - old[0], old2[1] - old[1])]
            b = ANGLES[(old[0] - u[0], old[1] - u[1])]
            d = b - a
            d = ((d + 180) % 360) - 180
            ret.append(d)
        old2 = old
        old = u
    return ret


def nkg_make_track(detection: dict):
    img = detection.get(GRAY)

    mask = detection.get(NKG_MASK)
    found_cores = detection.get(NKG_CORES)

    adjacency = bitmap_to_graph(img, mask)

    pairs = itertools.combinations(found_cores, 2)
    dists = []
    for p in pairs:
        c1 = p[0]
        c2 = p[1]

        p = search_path_dijkstra(img, adjacency, c1, c2)
        dist = p['dist']
        if not math.isinf(dist):
            dists.append({'pair': p, 'c1': c1, 'c2': c2, 'dist': dist, 'path': p['path']})

    if len(dists) == 0:
        return

    # get longest path
    max_dist = sorted(dists, key=lambda x: len(x['path']), reverse=True)[0]
    unraveled = []
    for v in max_dist['path']:
        u = unravel_index(v, img.shape)
        unraveled.append(u)

    alphabeted = unraveled_to_alphabet(unraveled)
    alphabeted2 = unraveled_to_alphabet2(unraveled)

    detection[NKG_PATH] = unraveled
    detection[NKG_DIRECTION] = alphabeted
    detection[NKG_DERIVATIVE] = alphabeted2

    return detection
