"""
This is script for prepare RAW cosmic-ray set from CREDO Database to our future working.

The script:

 1. Load JSON's from INPUT_DIR
 2. Exclude some cosmic-ray records.
 3. Store as JSON in OUTPUT_DIR with the same file name.

Exclusion rules:

 * non image
 * non 60x60 size
 * too_often(10, 60000)
 * near_hot_pixel2(3, 5)
 * # too_bright(70, 70)  # currently not used

"""
import glob
import json
import os
import sys
import threading
import time
from io import BytesIO
from multiprocessing import Pool
from typing import Optional, List

from PIL import Image

from credo_cf import group_by_device_id, group_by_resolution, too_often, near_hot_pixel2, load_json, CLASSIFIED, CLASS_ARTIFACT, ID, FRAME_CONTENT, \
    ARTIFACT_NEAR_HOT_PIXEL2, ARTIFACT_TOO_OFTEN
from credo_cf.io.io_utils import decode_base64

INPUT_DIR = '/tmp/credo/source'
PASSED_DIR = '/tmp/credo/passed'
OUTPUT_DIR = '/tmp/credo/destination'
PARTS_DIR = '/tmp/credo/parts'


def write_detections(detections: List[dict], fn: str, ident=False):
    with open(fn, 'w') as json_file:
        if ident:
            json.dump({'detections': detections}, json_file, indent=2)
        else:
            json.dump({'detections': detections}, json_file)


def start_analyze(all_detections, log_prefix):
    print('%s  group by devices...' % log_prefix)
    by_devices = group_by_device_id(all_detections)
    print('%s  ... done' % log_prefix)

    dev_no = 0
    dev_count = len(by_devices.keys())

    for device_id, device_detections in by_devices.items():
        by_resolution = group_by_resolution(device_detections)
        for resolution, detections in by_resolution.items():
            dev_no += 1
            print('%s    start device %d of %d, device id: %s, resolution: %dx%d, detections count: %d' % (log_prefix, dev_no, dev_count, str(device_id), resolution[0], resolution[1], len(detections)))

            goods = detections

            # too_often
            goods, bads = too_often(goods)
            print('%s    ... dropped by too_often: %d' % (log_prefix, len(bads)))

            # too_bright
            # goods, bads = too_bright(goods, 70, 70)

            # near_hot_pixel2
            goods, bads = near_hot_pixel2(goods)
            print('%s    ... dropped by near_hot_pixel2: %d' % (log_prefix, len(bads)))

            # end, counting goods
            print('%s    ... goods: %d' % (log_prefix, len(goods)))


def load_parser(obj: dict, count: int, ret: List[dict]) -> Optional[bool]:
    log_prefix = '%s: ' % str(threading.get_ident())

    skip = count - len(ret) - 1
    if count % 10000 == 0:
        print('%s  ... just parsed %d and skip %d objects.' % (log_prefix, count, skip))

    if not obj.get(FRAME_CONTENT):
        return False

    try:
        from credo_cf.image.image_utils import load_image, image_basic_metrics
        frame_decoded = decode_base64(obj.get(FRAME_CONTENT))
        pil = Image.open(BytesIO(frame_decoded))
        if pil.size == (60, 60):
            return True

    except Exception as e:
        print('%sFail of load image in object with ID: %d, error: %s' % (log_prefix, obj.get(ID), str(e)))
    return False


def run_file(fn):
    log_prefix = '%s: ' % str(threading.get_ident())

    fn_name = fn[len(INPUT_DIR) + 1:]
    print('%sStart file: %s' % (log_prefix, fn_name))
    fn_load = time.time()

    # load and analyse
    detections, count = load_json(fn, load_parser)
    print('%s  ... droped by non image: %d' % (log_prefix, count - len(detections)))
    start_analyze(detections, log_prefix)

    # found IDs of goods
    leave_good = []
    for d in detections:
        if d.get(CLASSIFIED) != CLASS_ARTIFACT:
            if CLASSIFIED in d.keys():
                del d[CLASSIFIED]
            if ARTIFACT_NEAR_HOT_PIXEL2 in d.keys():
                del d[ARTIFACT_NEAR_HOT_PIXEL2]
            if ARTIFACT_TOO_OFTEN in d.keys():
                del d[ARTIFACT_TOO_OFTEN]
            leave_good.append(d)

    # load again and save as
    fn_out = '%s/%s' % (OUTPUT_DIR, fn_name)
    write_detections(leave_good, fn_out)

    print('%s  file %s done, since start: %03ds, hits with images: %d, dropped: %d, leaved: %d' % (log_prefix, fn_name, time.time() - fn_load, count, count - len(leave_good), len(leave_good)))
    os.rename(fn, '%s/%s' % (PASSED_DIR, fn_name))
    return len(leave_good)


part = []  # safe because is out of the multi-thread part
part_no = 0


def write_part_and_clean():
    global part
    global part_no

    part_no += 1
    write_detections(part, '%s/%d.json' % (PARTS_DIR, part_no))
    print('Writen part no %d with %d hits' % (part_no, len(part)))
    part = []


def part_write(d: dict, c: int, r: List[dict]) -> Optional[bool]:
    global part

    part.append(d)
    if len(part) == 100000:
        write_part_and_clean()
    return False


def div_per_parts():
    files = glob.glob('%s/*.json' % OUTPUT_DIR)
    files = sorted(files)

    for fn in files:
        load_json(fn, part_write)

    if len(part) > 0:
        write_part_and_clean()


def main():
    # list all files in INPUT_DIR
    files = glob.glob('%s/*.json' % INPUT_DIR)

    with Pool(4) as pool:
        # each file parsed separately
        pool.map(run_file, files)

    # divide by 100000 parts
    div_per_parts()


if __name__ == '__main__':
    main()
    sys.exit(0)  # not always close
