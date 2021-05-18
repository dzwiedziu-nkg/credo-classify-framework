"""
This is script for prepare RAW cosmic-ray set from CREDO Database to our future working.

The script:

 1. Load JSON's from INPUT_DIR
 2. Exclude some cosmic-ray records.
 3. Store as JSON in OUTPUT_DIR with the same file name.

Exclusion rules:

 * non image
 * non 60x60 nor 64x64 nor 128x128 size
 * non X and Y in metadata
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
    ARTIFACT_NEAR_HOT_PIXEL2, ARTIFACT_TOO_OFTEN, X, Y, DEVICE_ID, group_by_timestamp_division, IMAGE, decode_base64, store_png, encode_base64, CROP_SIZE, \
    CROP_WIDTH, CROP_HEIGHT

INPUT_DIR = '/tmp/credo/source'
PASSED_DIR = '/tmp/credo/passed'
OUTPUT_DIR = '/tmp/credo/destination'
PARTS_DIR = '/tmp/credo/parts'
ERROR_DIR = '/tmp/credo/error'
DEBUG = True
DEBUG_DIR = '/tmp/credo/debug'


def write_detections(detections: List[dict], fn: str, ident=True):
    with open(fn, 'w') as json_file:
        if ident:
            json.dump({'detections': detections}, json_file, indent=2)
        else:
            json.dump({'detections': detections}, json_file)


def store_png_for_debug(detections: List[dict], subdirs: List[str]):
    if DEBUG:
        for d in detections:
            store_png(DEBUG_DIR, subdirs, str(d.get(ID)), d.get(FRAME_CONTENT))
            store_png(DEBUG_DIR, [*subdirs, d.get(DEVICE_ID)], str(d.get(ID)), d.get(FRAME_CONTENT))


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
            bads, goods = too_often(goods)
            print('%s    ... dropped by too_often: %d' % (log_prefix, len(bads)))
            store_png_for_debug(bads, ['too_often'])

            # too_bright
            # bads, goods = too_bright(goods, 70, 70)

            # near_hot_pixel2
            bads, goods = near_hot_pixel2(goods)
            print('%s    ... dropped by near_hot_pixel2: %d' % (log_prefix, len(bads)))
            store_png_for_debug(bads, ['near_hot_pixel2'])

            # end, counting goods
            print('%s    ... goods: %d' % (log_prefix, len(goods)))
            store_png_for_debug(goods, ['goods'])

            # try to merge hits on the same frame
            by_frame = group_by_timestamp_division(device_detections)
            reconstructed = 0
            for timestmp, in_frame in by_frame.items():
                if len(in_frame) <= 1:
                    continue

                image = None

                for d in reversed(in_frame):
                    if d.get(CROP_SIZE) == (60, 60):
                        if image is None:
                            Image.new('RGBA', (resolution[0], resolution[1]), (0, 0, 0))

                        cx = d.get(X) - 30
                        cy = d.get(Y) - 30
                        w, h = (60, 60)

                        if DEBUG:
                            store_png(DEBUG_DIR, ['reconstruct', str(device_id), str(timestmp), 'before'], str(d.get(ID)), d.get(FRAME_CONTENT))

                        image.paste(d.get(IMAGE), (cx, cy, cx + w, cy + h))

                        # fix bug in early CREDO Detector App: black filled boundary 1px too large
                        image.paste(image.crop((cx + w - 1, cy, cx + w, cy + h)), (cx + w, cy, cx + w + 1, cy + h))
                        image.paste(image.crop((cx, cy + h - 1, cx + w, cy + h)), (cx, cy + h, cx + w, cy + h + 1))
                        image.paste(image.crop((cx + w - 1, cy + h - 1, cx + w, cy + h)), (cx + w, cy + h, cx + w + 1, cy + h + 1))

                for d in in_frame:
                    if d.get(CROP_SIZE) == (60, 60):
                        cx = d.get(X) - 30
                        cy = d.get(Y) - 30
                        w, h = (60, 60)

                        hit_img = image.crop((cx, cy, cx + w, cy + h))
                        with BytesIO() as output:
                            hit_img.save(output, format="png")
                            d[FRAME_CONTENT] = encode_base64(output.getvalue())
                        if DEBUG:
                            store_png(DEBUG_DIR, ['reconstruct', str(device_id), str(timestmp), 'after'], str(d.get(ID)), d.get(FRAME_CONTENT))
                reconstructed += 1
                if DEBUG and image is not None:
                    store_png(DEBUG_DIR, ['reconstruct', str(device_id)], str(timestmp), image)
            print('%s    ... reconstructed frames: %d' % (log_prefix, reconstructed))


def load_parser(obj: dict, count: int, ret: List[dict]) -> Optional[bool]:
    log_prefix = '%s: ' % str(threading.get_ident())

    skip = count - len(ret) - 1
    if count % 10000 == 0:
        print('%s  ... just parsed %d and skip %d objects.' % (log_prefix, count, skip))

    if not obj.get(FRAME_CONTENT) or not obj.get(X) or not obj.get(Y):
        return False

    try:
        from credo_cf.image.image_utils import load_image, image_basic_metrics
        frame_decoded = decode_base64(obj.get(FRAME_CONTENT))
        pil = Image.open(BytesIO(frame_decoded))
        if pil.size == (60, 60) or pil.size == (64, 64) or pil.size == (128, 128):
            obj[IMAGE] = pil
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
    detections, count, errors = load_json(fn, load_parser)
    print('%s  ... droped by non image: %d' % (log_prefix, count - len(detections)))
    if len(errors):
        print('%s   ... errors in: %s' % (log_prefix, fn))
        lp = 0
        for error in errors:
            lp += 1
            with open('%s/%s-%06d.txt' % (ERROR_DIR, fn_name, lp), 'w') as f:
                f.write(error)

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
            if IMAGE in d.keys():
                d[CROP_WIDTH] = d[IMAGE].size[0]
                d[CROP_HEIGHT] = d[IMAGE].size[1]
                del d[IMAGE]
            leave_good.append(d)

    # load again and save as
    fn_out = '%s/%s' % (OUTPUT_DIR, fn_name)
    write_detections(leave_good, fn_out)

    print('%s  file %s done, since start: %03ds, hits with images: %d, dropped: %d, leaved: %d' % (log_prefix, fn_name, time.time() - fn_load, count, count - len(leave_good), len(leave_good)))
    if not DEBUG:
        os.rename(fn, '%s/%s' % (PASSED_DIR, fn_name))
    return len(leave_good)


part = []  # safe because is out of the multi-thread part
part_no = 0


def write_part_and_clean():
    global part
    global part_no

    part_no += 1
    write_detections(part, '%s/%03d.json' % (PARTS_DIR, part_no))
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

    if DEBUG:
        for fn in files:
            run_file(fn)
    else:
        with Pool(4) as pool:
            # each file parsed separately
            pool.map(run_file, files)

    # divide by 100000 parts
    div_per_parts()


if __name__ == '__main__':
    main()
    sys.exit(0)  # not always close
