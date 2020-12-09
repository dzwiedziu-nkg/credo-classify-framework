import math

from credo_cf import load_json, progress_and_process_image, group_by_id, GRAY, nkg_mark_hit_area, NKG_MASK, nkg_make_track, NKG_PATH, NKG_DIRECTION, \
    NKG_DERIVATIVE, ID
import matplotlib.pyplot as plt
from numpy import unravel_index
import numpy as np
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra


# prepare dataset: hits - JSON's objects, and grays - numpy grayscale images 60x60
objects, count, errors = load_json('../data/manual.json', progress_and_process_image)
by_id = group_by_id(objects)
used_hits = {4711435, 6234182, 9152349, 4913621, 5468291, 7097636, 4976474, 5206452, 4876475, 5951007, 4714801, 4819239, 4660572, 4705446, 8280225, 8459656,
             8471578, 9124308, 9314789, 4813841}
hits = []
for u in used_hits:
    hits.append(by_id[u][0])
grays = list(map(lambda x: x['gray'], hits))


# settings
used_kernel = [
    [-1, 0],
    [0, -1],
    [1, 0],
    [0, 1]
]

used_kernel2 = [
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1]
]

alphabet = {
    (-1, 0): 'A',
    (-1, -1): 'B',
    (0, -1): 'C',
    (1, -1): 'D',
    (1, 0): 'E',
    (1, 1): 'F',
    (0, 1): 'G',
    (-1, 1): 'H'
}

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


# mark the hits and locals maximums
def requrence_mark(inp, mark, color, point, kernel, threshold):
    cv = inp[point]
    mark[point] = color
    inp[point] = 0
    for k in kernel:
        try:
            p2 = (point[0] + k[0], point[1] + k[1])
            if p2[0] < 0 or p2[1] < 0 or p2[0] >= np.size(inp, 0) or p2[1] >= np.size(inp, 1):
                return

            cv2 = inp[p2]
            m2 = mark[p2]
            if m2 == 0 and cv2 < cv and cv2 > threshold:
                requrence_mark(inp, mark, color, p2, kernel, threshold)
        except IndexError:
            pass


def requrence_mark2(inp, mark, color, point, kernel, threshold):
    cv = inp[point]
    mark[point] = color
    inp[point] = 0
    nexts = []

    for k in kernel:
        try:
            p2 = (point[0] + k[0], point[1] + k[1])
            if p2[0] < 0 or p2[1] < 0 or p2[0] >= np.size(inp, 0) or p2[1] >= np.size(inp, 1):
                return

            cv2 = inp[p2]
            m2 = mark[p2]

            if m2 == 0 and cv2 < cv and cv2 > threshold:
                mark[p2] = color
                # inp[p2] = 0
                nexts.append((inp, mark, color, p2, kernel, threshold))
        except IndexError:
            pass

    for n in nexts:
        try:
            requrence_mark2(*n)
        except IndexError:
            pass


def find_all_maximums(img, max_maximums, kernel, recursion_func):
    a = img.copy()
    b = np.zeros(a.shape)
    cores = []

    find_median = []

    for i in range(1, max_maximums):
        core = unravel_index(a.argmax(), a.shape)
        core_value = a[core]
        cores.append(core_value)
        find_median.append(core_value)
        recursion_func(a, b, core_value, core, kernel, 0)
        b[core] = cores[0]

    core_median = find_median[len(find_median) // 2]
    # threshold = (find_median[0] - core_median) // 10 + core_median
    threshold = core_median + 5
    return {'threshold': threshold, 'mask': b}


def find_maximums_bounded(img, threshold, kernel, recursion_func):
    a = img.copy()
    b = np.zeros(a.shape)
    cores = []
    found_cores = []

    while True:
        core = unravel_index(a.argmax(), a.shape)
        core_value = a[core]
        if core_value <= threshold:
            break

        cores.append(core_value)
        found_cores.append({'core': core, 'value': core_value})
        recursion_func(a, b, core_value, core, kernel, threshold)
        b[core] = -100

    return {'found_cores': found_cores, 'mask': b}


def mark_all(hits, kernel, recursion_func):
    maps1 = []
    maps2 = []
    img_pos = 0
    img_found_cores = []

    for hit in hits:
        # first stage: detect all local maximums
        gray = hit.get(GRAY)
        img_pos += 1

        ret = find_all_maximums(gray, 100, kernel, recursion_func)
        threshold = ret['threshold']
        maps1.append(ret['mask'])

        # second stage: cutoff maximums from noise and cur
        ret = find_maximums_bounded(gray, threshold, kernel, recursion_func)
        found_cores = ret['found_cores']
        maps2.append(ret['mask'])

        img_found_cores.append({'hit': hit, 'found_cores': found_cores, 'mask': ret['mask']})

    display_all(maps1)
    print('Threshold')
    display_all(maps2)
    return img_found_cores


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
    print('source: ' + str(_[0][source]**(1.0/16.0)))
    print('target: ' + str(_[0][target]**(1.0/16.0)))
    i = 0
    #pixels_path.append(target)
    while pixel_index != source:
        i += 1
        pixels_path.append(pixel_index)
        pixel_index = predecessors[0, pixel_index]
        print(str(i) + '.: ' + str(_[0][pixel_index]**(1.0/16.0)))
    pixels_path.append(source)
    return {'dist': _[0][target]**(1.0/16.0), 'path': pixels_path}



    # disp = img.copy()
    # disp[39, 23] = am
    # disp[30, 30] = am
    # for pixel_index in pixels_path:
    #     i, j = to_coordinates(pixel_index)
    #     disp[i, j] = am

    #plt.imshow(disp)
    #plt.show()
    #display(disp)

    #dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=0, return_predecessors=True)
    #print(dist_matrix)


def unraveled_to_alphabet(unraveled):
    ret = []
    old = None
    for u in unraveled:
        if old != None:
            d = (old[0] - u[0], old[1] - u[1])
            ret.append(alphabet[d])
        old = u
    return ret


def unraveled_to_alphabet2(unraveled):
    ret = []
    old2 = None
    old = None
    for u in unraveled:
        if old != None and old2 != None:
            a = alphabet[(old2[0] - old[0], old2[1] - old[1])]
            b = alphabet[(old[0] - u[0], old[1] - u[1])]
            d = ord(b) - ord(a)
            if d < 0:
                d += 8
            d = chr(d + ord('A'))
            ret.append(d)
        old2 = old
        old = u
    return ret

def list_to_str(l):
    return ', '.join(map(lambda x: str(x), l))

# mark_all(vs, used_kernel, requrence_mark)
# display_all(grays)
display_all_from(hits, GRAY)
for h in hits:
    nkg_mark_hit_area(h)
display_all_from(hits, NKG_MASK)

for h in hits:
    nkg_make_track(h)
    h['path_preview'] = h.get(GRAY).copy()
    print(h.get(ID))
    if h.get(NKG_PATH):
        for i in h.get(NKG_PATH):
            h['path_preview'][i] = -50
        print(list_to_str(h.get(NKG_DIRECTION)) + '|\t|' + list_to_str(h.get(NKG_DERIVATIVE)) + '\n')
    else:
        print('it is spot')
    print('---')

display_all_from(hits, 'path_preview')
#
# img_found_cores = mark_all(hits, used_kernel2, requrence_mark2)
#
# to_draws = []
#
# for ifc in img_found_cores:
#     hit = ifc['hit']
#     img = hit.get(GRAY)
#     mask = ifc['mask']
#     found_cores = ifc['found_cores']
#
#     adjacency = bitmap_to_graph(img, mask)
#
#     pairs = itertools.combinations(found_cores, 2)
#     dists = []
#     for p in pairs:
#         c1 = p[0]['core']
#         c2 = p[1]['core']
#
#         p = search_path_dijkstra(img, adjacency, c1, c2)
#         dist = p['dist']
#         if not math.isinf(dist):
#             dists.append({'pair': p, 'c1': c1, 'c2': c2, 'dist': dist, 'path': p['path']})
#     print(dists)
#
#     to_draw = img.copy()
#     if len(dists) > 0:
#         sd = sorted(dists, key=lambda x: x['dist'], reverse=True)
#         max_dist = sd[0]
#         print(max_dist)
#         max_value = to_draw[unravel_index(to_draw.argmax(), to_draw.shape)]
#         unraveled = []
#         for v in max_dist['path']:
#             u = unravel_index(v, to_draw.shape)
#             to_draw[u] = max_value
#             unraveled.append(u)
#         alphabeted = unraveled_to_alphabet(unraveled)
#         alphabeted2 = unraveled_to_alphabet2(unraveled)
#         hit['dir_src'] = ''.join(alphabeted)
#         hit['dir_der'] = ''.join(alphabeted2)
#         #to_draw[max_dist['c1']] = max_value
#         #to_draw[max_dist['c2']] = max_value
#
#     to_draws.append(to_draw)
#
# display_all(to_draws)
#
# i = 0
# for h in hits:
#     if i >= 5:
#         i = 0
#         print('\n')
#     i += 1
#     print('%s\t%s' % (h.get('dir_src', 'to_small'), h.get('dir_der', 'to_small')))
#

#display(grays[0])
#bitmap_to_graph(grays[0])
