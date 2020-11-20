from credo_cf import load_json, progress_and_process_image, group_by_id, GRAY
import matplotlib.pyplot as plt
from numpy import unravel_index
import numpy as np


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


# make path by dijkstry




# mark_all(vs, used_kernel, requrence_mark)
display_all(grays)
img_found_cores = mark_all(hits, used_kernel2, requrence_mark2)
