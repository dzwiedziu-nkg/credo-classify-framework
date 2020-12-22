import math
import sys

from scipy.interpolate import interp2d
from scipy.ndimage import rotate, center_of_mass
from scipy.spatial import distance
from skimage.feature import canny
from skimage.filters import rank, gaussian
from skimage.measure import subdivide_polygon
from skimage.morphology import medial_axis, square, erosion, disk
from skimage.segmentation import active_contour
from skimage.transform import probabilistic_hough_line, rescale
from sklearn.linear_model import LinearRegression

from credo_cf import load_json, progress_and_process_image, group_by_id, GRAY, nkg_mark_hit_area, NKG_MASK, nkg_make_track, NKG_PATH, NKG_DIRECTION, \
    NKG_DERIVATIVE, ID, NKG_THRESHOLD, NKG_UPSCALE, NKG_SKELETON, point_to_point_distance, center_of_points, NKG_MASKED, NKG_REGRESSION, NKG_PATH_FIT, \
    store_png, IMAGE
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

used_hits3 = [7741225, 4580817, 5973441, 4892405, 17432760,
              17432645, 4731298, 6229582, 17571002, 17368987,
              7148947, 4899235, 18349704, 18250739, 6908292,
              9129139, 17771578, 17861029, 17337669, 7470695,
              4711435, 6234182, 9152349, 4913621, 5468291,
              7097636, 4976474, 5206452, 4876475, 5951007,
              4714801, 4819239, 4660572, 4705446, 8280225,
              8459656, 8471578, 9124308, 9314789, 4813841]

used_hits = used_hits3

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


def display_all_from(hits, _from, title_func=None, scale=6):
    cols = 5
    rows = int(math.ceil(len(hits) / cols))
    f, axs = plt.subplots(rows, cols, constrained_layout=True, figsize=(4*scale, 3*scale*rows/4))
    i = 0
    for ax in axs.flat:
        if len(hits) <= i:
            break
        im = ax.matshow(hits[i].get(_from))
        if title_func is not None:
            ax.title.set_text(title_func(hits[i]))
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

ray_way_octet2 = [
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ],
]


fit_mask = [
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
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

        dw = np.ones(calc.shape)
        dw[1:4, 1:4] = calc[1:4, 1:4]
        calc = calc + dw  # najbliższe srodka są 2x
        # calc = calc * dw  # najbliższe środka są x^2

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


def nkg_pather_step(img, next_pos, angle, threshold, fov, step):
    path = []
    while img[next_pos] > threshold:
        path.append(next_pos)
        img[next_pos] = 0

        try:
            calc = calc_ways(img, next_pos, ray_way)
        except:
            # edge achieved
            break

        filtered = list(filter(lambda x: in_angle(angle - fov / 2, angle + fov / 2, x['angle']) and img[calc_pos(next_pos, way_next_point(x['way'], step))] > threshold, calc))
        if len(filtered) == 0:
            break
        direction = max(filtered, key=lambda x: x['value'])
        next_pos = calc_pos(next_pos, way_next_point(direction['way'], step))
        angle = direction['angle']
    return path


def nkg_pather(img, threshold, fov=90, step=1):
    # mask = np.zeros(img.shape)
    img = img.copy()
    start = unravel_index(img.argmax(), img.shape)
    try:
        calc = calc_ways(h['smooth'], start, ray_way)
    except:
        return np.array([])
    direction = max(calc, key=lambda x: x['value'])
    next_pos = calc_pos(start, way_next_point(direction['way'], step))

    angle = direction['angle']
    next_angle = direction['angle'] - 180
    path = nkg_pather_step(img, next_pos, angle, threshold, fov, step)

    angle = next_angle
    next_pos = start
    path2 = nkg_pather_step(img, next_pos, angle, threshold, fov, step)

    return np.array([*reversed(path2), *path])


def line_to_mask(img, path, scale=1, value=1, create_new_mask=False):
    if create_new_mask:
        mask = np.zeros(img.shape)
    else:
        mask = img

    for a in path:
        if scale > 1:
            mask[round(a[0] * scale + scale/2.0), round(a[1] * scale + scale/2.0)] = value
        else:
            mask[round(a[0]), round(a[1])] = value

    return ma.masked_array(img, mask)


def path_to_center_of_weight(img, fm, path):
    path2 = []
    fit_mask = np.array(fm)
    w = fit_mask.shape[0]
    h = fit_mask.shape[1]

    fit = img.copy()
    for i in path:
        x1 = int(i[0] - (w - 1) / 2)
        x2 = int(i[0] + (w + 1) / 2)
        y1 = int(i[1] - (h - 1) / 2)
        y2 = int(i[1] + (h + 1) / 2)
        # cut = fit[i[0]-2:i[0]+3, i[1]-2:i[1]+3]
        cut = fit[x1:x2, y1:y2]
        if cut.shape[0] != 5 or cut.shape[1] != 5:
            continue
        m = cut * fit_mask

        new_i = center_of_mass(m)
        path2.append([new_i[0] + x1, new_i[1] + y1])

    path2 = optimize_path(path2, 0.5)
    path2 = np.array(path2)

    # if path2.shape[0] > 1:
    #     return subdivide_polygon(path2)
    return path2


def optimize_path(path, max_distance, max_passes=20):
    working = path
    for i in range(0, max_passes):
        used = False
        path2 = [working[0]]
        for pos in range(1, len(working)):
            dist = point_to_point_distance(working[pos - 1], working[pos])
            if dist <= max_distance:
                new_point = center_of_points([working[pos - 1], working[pos]])
                if path2[-1][0] == working[pos - 1][0] and path2[-1][1] == working[pos - 1][1]:
                    path2[-1] = new_point
                else:
                    path2.append(new_point)
                used = True
            else:
                path2.append(working[pos])
        working = path2
        if not used:
            break
    return working


def nkg_path_analysis(detection: dict, fov=90, step=1):
    h = detection
    path = nkg_pather(h.get(GRAY), h.get(NKG_THRESHOLD), fov, step)
    h[NKG_PATH] = path
    h[NKG_MASKED] = line_to_mask(h.get(GRAY), path, create_new_mask=True)

    path_fit = path_to_center_of_weight(h.get(GRAY), fit_mask, path) if len(path) else []
    h[NKG_PATH_FIT] = path_fit

    if len(path_fit) == 0:
        h[NKG_REGRESSION] = 0
        return h

    X = path_fit[:,0].reshape(-1, 1)
    y = path_fit[:,1].reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    h[NKG_REGRESSION] = score
    return h


ray_way = build_ray_way(ray_way_octet)
www = way_next_point(ray_way[0]['way'])

# mark_all(vs, used_kernel, requrence_mark)
# display_all(grays)
display_all_from(hits, GRAY)
for h in hits:
    nkg_mark_hit_area(h)
display_all_from(hits, NKG_MASK)

for h in hits:
    img = h.get(GRAY).copy()

    img = gaussian(img, 0.5)  # rank.mean(img, selem=disk(1))
    h['smooth'] = img


display_all_from(hits, 'smooth')


for h in hits:
    nkg_path_analysis(h, 90)
    h['masked'] = line_to_mask(h.get(GRAY), h[NKG_PATH], create_new_mask=True)
    h['score'] = '%s: %0.3f/%d' % (str(h.get(ID)), h[NKG_REGRESSION], len(h[NKG_PATH_FIT]))

    img = rescale(h.get(GRAY), 8, order=0, preserve_range=True, anti_aliasing=False)
    mask = np.zeros(img.shape)
    h['masked2'] = line_to_mask(img, h.get(NKG_PATH_FIT), scale=8, create_new_mask=True)


display_all_from(hits, 'masked', lambda x:str(x['score']))
display_all_from(hits, 'masked2', lambda x:str(x['score']), scale=10)


#def measure_angle(fn: str):
#    hits, count, errors = load_json('../data/%s' % fn, progress_and_process_image)
#    for h in hits:
#        nkg_mark_hit_area(h)
#        nkg_path_analysis(h)

#        store_png('/tmp/credo', [fn], '%0.3f_%s' % (h.get(NKG_REGRESSION), str(h.get(ID))), h.get(IMAGE))


#def main():
#    measure_angle('hits_votes_4_class_2.json')
#    measure_angle('hits_votes_4_class_3.json')


#if __name__ == '__main__':
#    main()
#    sys.exit(0)  # not always close

# for o in objects:
#     print('%s;%f' % (str(o.get(ID)), o.get(NKG_REGRESSION)))
