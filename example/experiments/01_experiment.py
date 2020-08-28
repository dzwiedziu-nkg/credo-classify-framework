import bz2
import time
import urllib.request
import io
from typing import List, Tuple

from credo_cf import load_json_from_stream, progress_and_process_image, group_by_device_id, group_by_resolution, too_often, near_hot_pixel2, \
    too_bright
from credo_cf import xor_preprocess
from credo_cf.commons.utils import get_and_add

WORKING_SET = 'http://mars.iti.pk.edu.pl/~nkg/credo/working_set.json.bz2'

time_profile = {}


def download_working_set(url: str) -> Tuple[List[dict], int]:
    print('Download working set...')
    data = urllib.request.urlopen(url).read()

    print('Decompress...')
    json_content = bz2.decompress(data).decode("utf-8")

    print('Prase JSON...')
    objs, count = load_json_from_stream(io.StringIO(json_content), progress_and_process_image)
    print('Parsed %d, skipped %d' % (count, count - len(objs)))
    return objs, count


def start_analyze(all_detections):
    # print('Make custom grayscale conversion...')
    # for d in all_detections:
    #     convert_to_gray(d)

    ts_load = time.time()
    print('Group by devices...')
    by_devices = group_by_device_id(all_detections)
    get_and_add(time_profile, 'grouping', time.time() - ts_load)

    drop_counts = {}
    leave_good = 0

    print('Run experiment...')
    dev_no = 0
    dev_count = len(by_devices.keys())
    for device_id, device_detections in by_devices.items():

        ts_load = time.time()
        by_resolution = group_by_resolution(device_detections)
        get_and_add(time_profile, 'grouping', time.time() - ts_load)

        for resolution, detections in by_resolution.items():
            dev_no += 1
            print('Start device %d of %d, detectons count: %d' % (dev_no, dev_count, len(detections)))

            # too_often
            ts_load = time.time()

            goods = detections
            goods, bads = too_often(goods)
            get_and_add(drop_counts, 'too_often', len(bads))

            get_and_add(time_profile, 'too_often', time.time() - ts_load)

            # too_bright
            ts_load = time.time()
            goods, bads = too_bright(goods, 70, 70)
            get_and_add(time_profile, 'too_bright', time.time() - ts_load)
            get_and_add(drop_counts, 'too_bright', len(bads))

            # xor filter
            ts_load = time.time()
            if len(goods) > 1:
                x_or = xor_preprocess(goods)
            get_and_add(time_profile, 'xor', time.time() - ts_load)

            # near_hot_pixel2
            ts_load = time.time()
            goods, bads = near_hot_pixel2(goods)
            get_and_add(time_profile, 'near_hot_pixel2', time.time() - ts_load)

            get_and_add(drop_counts, 'drop_near_hot_pixel2', len(bads))

            # end, counting goods
            leave_good += len(goods)

    print('\nCount of cut off by filters:')
    for f, v in drop_counts.items():
        print('%s: %d' % (f, v))
    print('Goods: %d' % leave_good)


def main():
    # config data source, please uncomment and use one from both

    ts_load = time.time()
    # choice 1: download from website
    working_sets = [download_working_set(WORKING_SET)]  # download our working set from our hosting

    # choice 2: load from files
    # file_names = ['working_set.json']
    # working_sets = [load_json(fn, progress_and_process_image) for fn in file_names]
    get_and_add(time_profile, 'load', time.time() - ts_load)

    for all_detections, count in working_sets:
        start_analyze(all_detections)

    print('\nTime count:')
    for ts, tv in time_profile.items():
        print('time: %03d - %s' % (int(tv), ts))


if __name__ == '__main__':
    main()
