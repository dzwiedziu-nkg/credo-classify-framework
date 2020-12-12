from typing import List, Tuple, Union, Callable

import math
import itertools

import numpy as np
from numpy import unravel_index
from scipy.interpolate import interp2d
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import medial_axis, skeletonize

from credo_cf.commons.consts import GRAY, NKG_MASK, NKG_CORES, NKG_PATH, NKG_DERIVATIVE, NKG_DIRECTION, NKG_VALUES, NKG_THRESHOLD, NKG_UPSCALE, \
    NKG_UPSCALE_MASK, NKG_SKELETON

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


def nkg_mark_hit_area(detection: dict, recursion_func=None, kernel: List[Tuple[int, int]] = None, mask_method: Union[str, Callable[[List[int]], np.ndarray]] = 'derivate'):
    ret = find_all_maximums(detection.get(GRAY), recursion_func=recursion_func, kernel=kernel, spread_on_flat=False)

    find_median = ret['values']

    if mask_method in ['derivate', 'derivate2']:
        fm = list(reversed(find_median))
        derivates = []
        for i in range(1, len(fm)):
            derivates.append(fm[i] - fm[i-1])

        derivates2 = []
        for i in range(1, len(derivates)):
            derivates2.append(int(derivates[i]) - int(derivates[i-1]))

        for i in range(0, len(derivates)):
            if derivates[i] > 2:
                threshold = fm[i] + 1
                break

        # row1 = []
        # row2 = ['----']
        # row3 = ['----', '----']
        # for d in fm:
        #     row1.append('%4d' % d)
        # for d in derivates:
        #     row2.append('%4d' % d)
        # for d in derivates2:
        #     row3.append('%4d' % d)
        #
        # print(', '.join(row1))
        # print(', '.join(row2))
        # print(', '.join(row3))
    elif mask_method == 'median':
        core_median = find_median[len(find_median) // 2]
        # threshold = (find_median[0] - core_median) // 10 + core_median
        threshold = core_median + 5

    # print(threshold)
    # print('-----')

    ret2 = find_all_maximums(detection.get(GRAY), recursion_func=recursion_func, kernel=kernel, spread_on_flat=True, threshold=threshold)
    detection[NKG_MASK] = np.where(ret2['mask'] > 0, 1, 0)
    detection[NKG_VALUES] = ret2['values']
    detection[NKG_CORES] = ret2['cores']
    detection[NKG_THRESHOLD] = threshold

    return detection


# Defines a translation from 2 coordinates to a single number
def to_index(img, y, x):
    return y * img.shape[1] + x


# Defines a reversed translation from index to 2 coordinates
def to_coordinates(img, index):
    return index // img.shape[1], index % img.shape[1]


# make path by dijkstry
def bitmap_to_graph(img, mask, y=0, p=16.0):
    # graph = csr_matrix(img)
    am_ind = unravel_index(img.argmax(), img.shape)
    am = img[am_ind] + y

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
                weight = float(am * 2 - pix1 - pix2)**p + 1  # abs(int(pix2) - int(pix1))*2 + 1
                adjacency[to_index(img, i, j), to_index(img, i + y_diff, j + x_diff)] = weight
    return adjacency


def search_path_dijkstra(img, adjacency, src, dst) -> dict:
    # We chose two arbitrary points, which we know are connected
    source = to_index(img, *src)
    target = to_index(img, *dst)

    # Compute the shortest path between the source and all other points in the image
    _, predecessors = dijkstra(adjacency, directed=True, indices=[source], unweighted=False, return_predecessors=True)

    # Constructs the path between source and target
    pixel_index = target
    pixels_path = []
    if math.isinf(_[0][target]):
        return {'dist': math.inf, 'path': []}
    # print('source: ' + str(_[0][source]**(1.0/16.0)))
    # print('target: ' + str(_[0][target]**(1.0/16.0)))
    #i = 0
    #pixels_path.append(target)
    while pixel_index != source:
        pixels_path.append(pixel_index)
        pixel_index = predecessors[0, pixel_index]
        #i += 1
        # print(str(i) + '.: ' + str(_[0][pixel_index]**(1.0/16.0)))
    pixels_path.append(source)
    return {'dist': _[0][target]**(1.0/16.0), 'path': pixels_path}


def search_longest_path_dijkstra(img, mask, src, y=1000, p=0.0625, adjacency=None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    _adjacency = adjacency if adjacency is not None else bitmap_to_graph(img, mask, y, p)

    # We chose two arbitrary points, which we know are connected
    source = to_index(img, *src)

    # Compute the shortest path between the source and all other points in the image
    distances, predecessors = dijkstra(_adjacency, directed=True, indices=[source], unweighted=False, return_predecessors=True)

    distances2 = distances.reshape(img.shape)
    distances2[distances2 == np.inf] = 0
    farest = unravel_index(distances2.argmax(), distances2.shape)

    source = to_index(img, *farest)
    _, predecessors = dijkstra(_adjacency, directed=True, indices=[source], unweighted=False, return_predecessors=True)

    _2 = _.reshape(img.shape)
    _2[_2 == np.inf] = 0
    farest2 = unravel_index(_2.argmax(), _2.shape)

    return farest, farest2


def upscale_image(img: np.ndarray, scale=10, kind='linear') -> np.ndarray:
    xrange = lambda x: np.linspace(0, 1, x)
    fimg = interp2d(xrange(img.shape[0]), xrange(img.shape[1]), img, kind=kind)
    img = fimg(xrange(img.shape[0]*scale), xrange(img.shape[0]*scale))
    return img


def do_skeletons(img: np.ndarray, method='medial_axis') -> np.ndarray:
    if method == 'medial_axis':
        skel, distance = medial_axis(img, return_distance=True)
        return skel
    elif method == 'zhang':
        return skeletonize(np.where(img > 0, 1, 0), method='zhang')
    elif method == 'lee':
        return skeletonize(np.where(img > 0, 1, 0), method='lee')


def find_start_stop_point(img: np.ndarray, mask: np.ndarray, skeleton: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:

    ms = mask * skeleton
    adjacency = bitmap_to_graph(img, ms, p=1)

    paths = img * ms
    start = unravel_index(paths.argmax(), paths.shape)
    return search_longest_path_dijkstra(img, ms, start, adjacency=adjacency)


def rescale_path(path: List[Tuple[int, int]], scale: float,  do_round=False) -> List[Tuple[float, float]]:
    path2 = []
    for p in path:
        np = p[0] * scale, p[1] * scale
        if do_round:
            np = round(np[0]), round(np[1])
        if np not in path2:
            path2.append(np)
    return path2


def unravel_path(path: List[int], shape) -> List[Tuple[int, int]]:
    unraveled = []
    for v in path:
        u = unravel_index(v, shape)
        unraveled.append(u)
    return unraveled


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


def nkg_make_track(detection: dict, scale=1, downscale='round', upscale_kind='linear', skeleton_method='medial_axis'):
    img = detection.get(GRAY)
    mask = detection.get(NKG_MASK)

    if scale != 1:
        img = upscale_image(img, scale=scale, kind=upscale_kind)
        mask = np.where(img > detection.get(NKG_THRESHOLD), 1, 0)
    detection[NKG_UPSCALE] = img
    detection[NKG_UPSCALE_MASK] = mask

    skeleton = do_skeletons(img*mask, method=skeleton_method)
    detection[NKG_SKELETON] = skeleton

    start, strop = find_start_stop_point(img, mask, skeleton)
    adjacency = bitmap_to_graph(img, mask)
    ret = search_path_dijkstra(img, adjacency, start, strop)
    path = unravel_path(ret['path'], img.shape)
    if scale != 1 and downscale:
        path = rescale_path(path, 1.0/scale, downscale == 'round')

    alphabeted = unraveled_to_alphabet(path)
    alphabeted2 = unraveled_to_alphabet2(path)

    detection[NKG_PATH] = path
    detection[NKG_DIRECTION] = alphabeted
    detection[NKG_DERIVATIVE] = alphabeted2

    return detection


def angle_3points(a, b, c):
    a = np.array(list(a))
    b = np.array(list(b))
    c = np.array(list(c))

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


def analyse_path(path: List[Tuple[float, float]], cut_first=1, cut_latest=1, divide=3) -> List[float]:
    if len(path) - cut_first - cut_latest < divide:
        return []

    new_path = path[cut_first:len(path)-cut_latest]
    step = len(new_path) // (divide - 1)

    points = []
    for i in range(0, divide):
        if i == divide - 1:
            points.append(new_path[-1])
        else:
            points.append(new_path[i*step])

    ret = []

    for i in range(2, divide):
        angle = angle_3points(points[i-2], points[i-1], points[i]) / math.pi * 180.0
        if angle > 180:
            angle -= 360
        ret.append(angle)
    return ret
