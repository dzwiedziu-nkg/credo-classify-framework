import os
from typing import Optional, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from credo_cf import load_json, progress_load_filter, load_image, GRAY, ID, print_log, deserialize_or_run, store_png, deserialize, serialize, FRAME_CONTENT

OUTPUT_DIR = '/tmp/credo'
WORKING_SET = '/tmp/16.json'
STORE_GRAY_CLUSTER = False


html_head = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Title</title>
</head>

<style>
table, th, td {
  border: 1px black solid;
}

th, td {
  padding: 0.5em;
}
</style>

<body>
<table>
  <tr>
    <th>Cluster</th>
    <th>Count</th>
    <th>Sample images</th>
  </tr>
'''

html_foot = '''
</table>
</body>
</html>'''


def get_working_set_file(stage: int):
    return '%s/loaded-%02d.dat' % (OUTPUT_DIR, stage)


def get_kmeans_file(stage: int):
    return '%s/kmeans-%02d.dat' % (OUTPUT_DIR, stage)


def get_labels_file(stage: int):
    return '%s/labels-%02d.dat' % (OUTPUT_DIR, stage)


def load_data(fn: str):
    """
    Load data from prepared working set in JSON from ``fn`` file.

    For each image:
    1. Load from JSON
    2. Convert to grayscale

    :param fn: JSON file with
    :return: two tables in dict: ``id_array`` with hit ID's and `bitmap_array`` with array of gray bitmaps
    """
    id_array = []
    bitmap_array = []
    stored = {}

    def load_parser(obj: dict, count: int, ret: List[dict]) -> Optional[bool]:
        progress_load_filter(obj, count, ret)
        load_image(obj, False)
        st = '%03d' % ((count - 1) // 1000)
        stored[obj[ID]] = st
        store_png(OUTPUT_DIR, ['images', st], str(obj[ID]), obj[FRAME_CONTENT])

        id_array.append(obj[ID])
        bitmap_array.append([obj[GRAY]])

        return False

    load_json(fn, load_parser)

    return {
        'id_array': id_array,
        'bitmap_array': bitmap_array,
        'stored': stored
    }


def clustering(data: dict):
    # Stacking array of 2D bitmaps to 3D stack
    start_time = print_log('Start vstack...')
    bitmap_array = data['bitmap_array']
    stacked = np.vstack(bitmap_array)
    print_log('  ... finish', start_time)

    # Reshape of bitmaps to 1D, so we get 2D in result
    start_time = print_log('Start reshape...')
    stacked_flat = stacked.reshape(len(stacked), -1)
    print_log('  ... finish', start_time)

    # Sort
    start_time = print_log('Start sort...')
    sorted = np.sort(stacked_flat, axis=1)
    print_log('  ... finish', start_time)


    # Clustering
    start_time = print_log('Start clustering...')

    n_clusters = 20
    n_init = 20
    max_iter = 300
    random_state = 0

    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, verbose=1, algorithm='elkan', random_state=random_state)
    kmeans.fit(sorted)
    print_log('  ... finish', start_time)

    return kmeans


def save_html_and_pngs_and_labels(stage: int, kmeans: KMeans, data: dict):
    sn = get_labels_file(stage)
    if os.path.exists(sn):
        return

    # Save PNG files in file system
    labels = {}
    for i in range(0, len(kmeans.labels_)):
        label = kmeans.labels_[i]
        _id = data['id_array'][i]
        image = data['bitmap_array'][i][0]

        if STORE_GRAY_CLUSTER:
            store_png(OUTPUT_DIR, ['%02d' % stage, '%03d' % label], str(_id), image)
        in_label = labels.get(label, [])
        in_label.append(_id)
        labels[label] = in_label

    # make normalized sum
    for k, a in labels.items():
        stacked = np.vstack(a)
        summed = np.sum(stacked, 0)



    # Save HTML for preview clusters
    max_files_per_cluster = 65
    with open('%s/%02d.html' % (OUTPUT_DIR, stage), 'wt') as html:
        html.write(html_head)
        for label in sorted(labels.keys()):
            html.write('  <tr><th>%d</th><th>%d</th><td>\n' % (label, len(labels[label])))
            used = 0
            for _id in labels[label]:
                html.write('    <img src="images/%s/%s.png"/>\n' % (data['stored'][_id], str(_id)))
                used += 1
                if used >= max_files_per_cluster:
                    break
            html.write('  </td></tr>\n')
        html.write(html_foot)

    serialize(sn, labels)


def exclude_hits(stage: int, excludes: List[int]) -> dict:
    data = deserialize(get_working_set_file(stage - 1))
    labels = deserialize(get_labels_file(stage - 1))

    # prepare excluded working set
    to_exclude = set()
    for ex in excludes:
        to_exclude |= set(labels[ex])

    new_data = {
        'id_array': [],
        'bitmap_array': [],
        'stored': data['stored']
    }

    for i in range(0, len(data['id_array'])):
        _id = data['id_array'][i]
        if _id not in to_exclude:
            image = data['bitmap_array'][i]
            new_data['id_array'].append(_id)
            new_data['bitmap_array'].append(image)
    return new_data


def do_compute(data: dict, stage: int):
    # Deserialize or compute clustering ``data`` when serialized file is not exists. See: clustering
    start_time = print_log('Clustering %d stage...' % stage)
    kmeans = deserialize_or_run(get_kmeans_file(stage), clustering, data)
    print_log('  ... finish', start_time)

    save_html_and_pngs_and_labels(stage, kmeans, data)


def do_compute_first_stage(fn: str):
    """
    Clustering first stage.

    1. Load full working set.
    2. Compute kmeans for stage 1.
    3. Save to PNG and html for human preview.

    :param fn: input working set in JSON
    """
    stage = 1

    # Deserialize or load from ``fn`` JSON file when serialized file is not exists. See: load_data
    start_time = print_log('Load from JSON...')
    data = deserialize_or_run(get_working_set_file(stage), load_data, fn)
    print_log('  ... finish', start_time)

    do_compute(data, stage)


def do_compute_nth_stage(stage: int, excludes: List[int]):
    """
    Clustering nth stage.

    1. Load working set and labels from previous stage.
    2. Exclude hits from excluded labels.
    3. Save working set for current stage.
    4. Compute kmeans for current stage.

    :param stage: current stage of classification
    :param excludes: cut off hits from these clusters from previous stage
    """

    # Deserialize or load working set and labels from previous stage and Exclude hits from excluded labels when serialized file is not exists. See: exclude_hits
    start_time = print_log('Exclude hits...')
    data = deserialize_or_run(get_working_set_file(stage), exclude_hits, stage, excludes)
    print_log('  ... finish', start_time)

    do_compute(data, stage)


def main():
    do_compute_first_stage(WORKING_SET)
    do_compute_nth_stage(2, [2, 3, 5, 12, 13, 15, 18])  # this is sample of clusters to eliminate
    do_compute_nth_stage(3, [4, 8, 10, 14, 18])
    do_compute_nth_stage(4, [6, 11, 12, 16, 17, 19])
    do_compute_nth_stage(5, [2, 4, 15, 18])


if __name__ == '__main__':
    main()
