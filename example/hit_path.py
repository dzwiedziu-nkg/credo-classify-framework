import math

from scipy.interpolate import interp2d
from scipy.ndimage import rotate
from skimage.feature import canny
from skimage.morphology import medial_axis, square, erosion
from skimage.transform import probabilistic_hough_line

from credo_cf import load_json, progress_and_process_image, group_by_id, GRAY, nkg_mark_hit_area, NKG_MASK, nkg_make_track, NKG_PATH, NKG_DIRECTION, \
    NKG_DERIVATIVE, ID, NKG_THRESHOLD, NKG_UPSCALE, NKG_SKELETON
import matplotlib.pyplot as plt
from numpy import unravel_index, ma
import numpy as np
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra


# prepare dataset: hits - JSON's objects, and grays - numpy grayscale images 60x60
from credo_cf.classification.preprocess.nkg_processings import search_longest_path_dijkstra, bitmap_to_graph, analyse_path

objects, count, errors = load_json('../data/manual.json', progress_and_process_image)
by_id = group_by_id(objects)
used_hits1 = {4711435, 6234182, 9152349, 4913621, 5468291, 7097636, 4976474, 5206452, 4876475, 5951007, 4714801, 4819239, 4660572, 4705446, 8280225, 8459656,
             8471578, 9124308, 9314789, 4813841}
used_hits2 = [7741225, 7238971, 5973441, 4892405, 17432760,
             17432645, 4731298, 6229582, 17571002, 17368987,
             7148947, 4899235, 18349704, 18250739, 6908292,
             9129139, 17771578, 17861029, 17337669, 7470695]

used_hits = used_hits1

hits = []
for u in used_hits:
    hits.append(by_id[u][0])
grays = list(map(lambda x: x['gray'], hits))


# utils
def display(img):
    plt.matshow(img)
    plt.colorbar()
    plt.show()


def display_all(values):
    f, axs = plt.subplots(4, 5, constrained_layout=True, figsize=(32, 24))
    i = 0
    for ax in axs.flat:
        im = ax.matshow(values[i])
        i += 1
    # f.colorbar(im, ax=axs.flat)
    plt.show()


def display_all_from(hits, _from):
    f, axs = plt.subplots(4, 5, constrained_layout=True, figsize=(32, 24))
    i = 0
    for ax in axs.flat:
        im = ax.matshow(hits[i].get(_from))
        i += 1
    # f.colorbar(im, ax=axs.flat)
    plt.show()


# wycinek 1/8 drogi promienia sprawdzania
ray_way_octet = [
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ],
]


def build_ray_way(octet):
    ret = []

    for r in range(0, 4):
        for i in range(0, len(octet)):
            o = octet[i]
            oct = np.array(o)
            angle = r * 90.0 + 45.0 / (len(octet) - 1) * i - 180
            ret.append({'way': np.rot90(oct, r), 'angle': angle})

        for i in range(1, len(octet) - 1):
            o = octet[-(i + 1)]
            oct = np.array(o)
            fl = np.flip(oct, axis=0)
            angle = r * 90.0 + 45.0 / (len(octet) - 1) * i + 45 - 180
            ret.append({'way': np.rot90(rotate(fl, angle=90), r), 'angle': angle})

    return ret


def way_next_point(way, step=1):
    w, h = way.shape
    a = np.zeros(way.shape)
    a[step:w-step, step:h-step] = way[step:w-step, step:h-step]
    a[step+1:w-step-1, step+1:h-step-1] = 0
    next = unravel_index(a.argmax(), a.shape)
    return next[0] - (w - 1) / 2, next[1] - (h - 1) / 2


def calc_ways(img, pos, ways):
    w = int((ways[0]['way'].shape[0] - 1) / 2)
    cut = img[pos[0]-w:pos[0]+w+1, pos[1]-w:pos[1]+w+1]
    sums = []
    for way in ways:
        calc = cut * way['way']
        s = np.sum(calc)
        sums.append({**way, 'value': s})
    return sums


def calc_pos(a, b):
    return int(a[0]+b[0]), int(a[1]+b[1])


def normalize_angle(angle):
    n = angle % 360
    return n if n <= 180 else n - 360


def in_angle(a, b, v):
    pa = normalize_angle(a) + 360
    pb = normalize_angle(b) + 360
    pv = normalize_angle(v) + 360
    if pa <= pb:
        return pa <= pv <= pb
    else:
        return not (pa >= pv >= pb)


ray_way = build_ray_way(ray_way_octet)
www = way_next_point(ray_way[0]['way'])

# mark_all(vs, used_kernel, requrence_mark)
# display_all(grays)
display_all_from(hits, GRAY)
for h in hits:
    nkg_mark_hit_area(h)
display_all_from(hits, NKG_MASK)


for h in hits:
    step = 1
    fov = 90

    img = h.get(GRAY).copy()
    mask = np.zeros(img.shape)
    start = unravel_index(img.argmax(), img.shape)
    calc = calc_ways(img, start, ray_way)
    direction = max(calc, key=lambda x:x['value'])
    next_pos = calc_pos(start, way_next_point(direction['way'], step))

    angle = direction['angle']
    next_angle = direction['angle'] - 180
    # img[start] = 0

    while img[next_pos] > h[NKG_THRESHOLD]:
        mask[next_pos] = 1
        img[next_pos] = 0

        try:
            calc = calc_ways(img, next_pos, ray_way)
        except:
            # edge achieved
            break

        filtered = list(filter(lambda x: in_angle(angle - fov / 2, angle + fov / 2, x['angle']), calc))
        direction = max(filtered, key=lambda x: x['value'])
        next_pos = calc_pos(next_pos, way_next_point(direction['way'], step))
        angle = direction['angle']

    angle = next_angle
    next_pos = start
    while img[next_pos] > h[NKG_THRESHOLD]:
        mask[next_pos] = 1
        img[next_pos] = 0

        try:
            calc = calc_ways(img, next_pos, ray_way)
        except:
            # edge achieved
            break

        filtered = list(filter(lambda x: in_angle(angle - fov / 2, angle + fov / 2, x['angle']), calc))
        direction = max(filtered, key=lambda x: x['value'])
        next_pos = calc_pos(next_pos, way_next_point(direction['way'], step))
        angle = direction['angle']

    masked = ma.masked_array(h.get(GRAY), mask=mask)
    h['masked'] = masked

    # edges = canny(img, 2, 1, 25)
    # h['edges'] = edges
    # lines = probabilistic_hough_line(img, threshold=10, line_length=5, line_gap=3)

display_all_from(hits, 'masked')
