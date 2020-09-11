import os
from typing import Optional, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from credo_cf import load_json, progress_load_filter, load_image, GRAY, ID, print_log, deserialize_or_run, store_png, deserialize, serialize

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


def get_working_set_file(dir: str, stage: int):
    return '%s/loaded-%02d.dat' % (dir, stage)


def get_kmeans_file(dir: str, stage: int):
    return '%s/kmeans-%02d.dat' % (dir, stage)


def get_labels_file(dir: str, stage: int):
    return '%s/labels-%02d.dat' % (dir, stage)


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

    def load_parser(obj: dict, count: int, ret: List[dict]) -> Optional[bool]:
        progress_load_filter(obj, count, ret)
        load_image(obj, False)

        id_array.append(obj[ID])
        bitmap_array.append([obj[GRAY]])

        return False

    load_json(fn, load_parser)

    return {
        'id_array': id_array,
        'bitmap_array': bitmap_array
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


def save_html_and_pngs_and_labels(dir: str, stage: int, kmeans: KMeans, data: dict):
    sn = get_labels_file(dir, stage)
    if os.path.exists(sn):
        return

    # Save PNG files in file system
    labels = {}
    for i in range(0, len(kmeans.labels_)):
        label = kmeans.labels_[i]
        _id = data['id_array'][i]
        image = data['bitmap_array'][i][0]

        store_png(dir, ['%02d' % stage, '%03d' % label], str(_id), image)
        in_label = labels.get(label, [])
        in_label.append(_id)
        labels[label] = in_label

    # Save HTML for preview clusters
    max_files_per_cluster = 75
    with open('%s/%02d.html' % (dir, stage), 'wt') as html:
        html.write(html_head)
        for label in sorted(labels.keys()):
            html.write('  <tr><th>%d</th><th>%d</th><td>\n' % (label, len(labels[label])))
            used = 0
            for _id in labels[label]:
                html.write('    <img src="%02d/%03d/%s.png"/>\n' % (stage, label, str(_id)))
                used += 1
                if used >= max_files_per_cluster:
                    break
            html.write('  </td></tr>\n')
        html.write(html_foot)

    serialize(sn, labels)


def exclude_hits(dir: str, stage: int, excludes: List[int]) -> dict:
    data = deserialize(get_working_set_file(dir, stage - 1))
    labels = deserialize(get_labels_file(dir, stage - 1))

    # prepare excluded working set
    to_exclude = set()
    for ex in excludes:
        to_exclude |= set(labels[ex])

    new_data = {
        'id_array': [],
        'bitmap_array': []
    }

    for i in range(0, len(data['id_array'])):
        _id = data['id_array'][i]
        if _id not in to_exclude:
            image = data['bitmap_array'][i]
            new_data['id_array'].append(_id)
            new_data['bitmap_array'].append(image)
    return new_data


def do_compute(dir: str, data: dict, stage: int):
    # Deserialize or compute clustering ``data`` when serialized file is not exists. See: clustering
    start_time = print_log('Clustering %d stage...' % stage)
    kmeans = deserialize_or_run(get_kmeans_file(dir, stage), clustering, data)
    print_log('  ... finish', start_time)

    save_html_and_pngs_and_labels(dir, stage, kmeans, data)


def do_compute_first_stage(dir: str, fn: str):
    """
    Clustering first stage.

    1. Load full working set.
    2. Compute kmeans for stage 1.
    3. Save to PNG and html for human preview.

    :param dir: path to store data
    :param fn: input working set in JSON
    """
    stage = 1

    # Deserialize or load from ``fn`` JSON file when serialized file is not exists. See: load_data
    start_time = print_log('Load from JSON...')
    data = deserialize_or_run(get_working_set_file(dir, stage), load_data, fn)
    print_log('  ... finish', start_time)

    do_compute(dir, data, stage)


def do_compute_nth_stage(dir: str, stage: int, excludes: List[int]):
    """
    Clustering nth stage.

    1. Load working set and labels from previous stage.
    2. Exclude hits from excluded labels.
    3. Save working set for current stage.
    4. Compute kmeans for current stage.

    :param dir: path to store data
    :param stage: current stage of classification
    :param excludes: cut off hits from these clusters from previous stage
    """

    # Deserialize or load working set and labels from previous stage and Exclude hits from excluded labels when serialized file is not exists. See: exclude_hits
    start_time = print_log('Exclude hits...')
    data = deserialize_or_run(get_working_set_file(dir, stage), exclude_hits, dir, stage, excludes)
    print_log('  ... finish', start_time)

    do_compute(dir, data, stage)


def main():
    do_compute_first_stage('/tmp/credo', '/tmp/16.json')
    do_compute_nth_stage('/tmp/credo', 2, [2, 3, 5, 12, 13, 15, 18])
    # do_compute_nth_stage('/tmp/credo', 3, [???])
    # do_compute_nth_stage('/tmp/credo', 4, [???])
    # do_compute_nth_stage('/tmp/credo', 5, [???])


if __name__ == '__main__':
    main()
