import json
import sys
from copy import deepcopy
from typing import List, Union, Iterable

from credo_cf import load_json, progress_load_filter, FRAME_CONTENT, load_image, ID, EDGE, CROP_SIZE, FRAME_DECODED


def prepare_working_set(input_file: Union[str, List[str]], output_file: str = None, exclude_edge: bool = True) -> Iterable[dict]:
    """
    Prepare detections set for working. Function save output as JSON file or return as iterable of detections.

    The detections are filter by follow rules:
      1. Has image (exclude i.e. CosmicWatch detections).
      2. Image is not corrupted.
      3. Image has size 60px x 60px.
      4. (optional) Image has not ``edge`` key.

    Note: function print progress and result logs to stderr

    Note: when ``input_file`` (or element in ``input_file`` when is a list) is the ``"-"`` string then input will be read from ``stdin``.
    Otherwise the file will be open as input text stream.

    Note: the return is the lazy evaluated iterable object. Because is need to compute. Do when you not use it then not be computed.
    If you want get true list please use::

        list(prepare_working_set(...))

    :param input_file: file or list of files
    :param output_file: output file name, when is None then output will not be written
    :param exclude_edge: exclude image with EDGE key, default: True
    :return: iterable with filtered object
    """

    objs = []
    count = 0

    # load detection from file or list of files
    if isinstance(input_file, list):
        for fn in input_file:
            print('Load file: %s' % fn, file=sys.stderr)
            os, c = load_json(fn, progress_load_filter)
            objs.extend(os)
            count += c
    else:
        print('Load file: %s' % input_file, file=sys.stderr)
        objs, count = load_json(input_file, progress_load_filter)

    # set of ID of detections without
    non_image_count = 0
    corrupted_count = 0
    other_size_count = 0
    edge_count = 0

    # filtered will be save from original JSON objects
    detections = deepcopy(objs)

    # make set of detections to save
    to_save = set()
    for d in detections:
        if not d.get(FRAME_CONTENT):
            non_image_count += 1
            continue

        try:
            load_image(d)

            # free memory: clean original base64 frame contend and byte array with decoded it, save only PIL.Image object
            d.pop(FRAME_CONTENT)
            d.pop(FRAME_DECODED)

            if not d.get(CROP_SIZE) == (60, 60):
                other_size_count += 1
                continue

            if exclude_edge and d.get(EDGE):
                edge_count += 1
                continue

            to_save.add(d.get(ID))
        except Exception as e:
            print('Fail of load image in object with ID: %d, error: %s' % (d.get(ID), str(e)), file=sys.stderr)
            corrupted_count += 1
            continue

    # print logs
    print('The results count: %d and the excluded counts:' % len(to_save), file=sys.stderr)
    print('- whole detections: %d' % count, file=sys.stderr)
    print('- non image count: %d' % non_image_count, file=sys.stderr)
    print('- corrupted image count: %d' % corrupted_count, file=sys.stderr)
    print('- other size count: %d' % other_size_count, file=sys.stderr)
    if exclude_edge:
        print('- detections in edge count: %d' % edge_count, file=sys.stderr)

    # filter detections for save from original JSON objects
    objs_to_save = filter(lambda obj: obj.get(ID) in to_save, objs)

    # (optional) write filtered detections to JSON file
    if output_file is not None:
        out = sys.stdout if output_file == '-' else open(output_file, 'w')

        json.dump({'detections': list(objs_to_save)}, out)

        if output_file != '-':
            out.close()
        print('Saved to: %s' % output_file, file=sys.stderr)

    # return lazy iterable of detections with additional keys
    return filter(lambda obj: obj.get(ID) in to_save, detections)
