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
import time

from credo_cf import progress_and_process_image, group_by_device_id, group_by_resolution, too_often, near_hot_pixel2, load_json, CLASSIFIED, CLASS_ARTIFACT, ID, CROP_SIZE

from credo_cf.commons.utils import get_and_add

INPUT_DIR = '/tmp/credo/source'
OUTPUT_DIR = '/tmp/credo/destination'


def start_analyze(all_detections):
    print('  group by devices...')
    by_devices = group_by_device_id(all_detections)
    print('  ... done')

    dev_no = 0
    dev_count = len(by_devices.keys())

    for device_id, device_detections in by_devices.items():
        by_resolution = group_by_resolution(device_detections)
        for resolution, detections in by_resolution.items():
            dev_no += 1
            print('    start device %d of %d, device id: %s, resolution: %dx%d, detections count: %d' % (dev_no, dev_count, str(device_id), resolution[0], resolution[1], len(detections)))

            goods = detections

            # too_often
            goods, bads = too_often(goods)
            print('    ... dropped by too_often: %d' % len(bads))

            # too_bright
            # goods, bads = too_bright(goods, 70, 70)

            # near_hot_pixel2
            goods, bads = near_hot_pixel2(goods)
            print('    ... dropped by near_hot_pixel2: %d' % len(bads))

            # end, counting goods
            dropped = 0
            for d in goods:
                if d.get(CROP_SIZE) != (60, 60):
                    d[CLASSIFIED] = CLASS_ARTIFACT
                    dropped += 1
            print('    ... dropped by non 60x60 size: %d' % dropped)
            print('    ... goods: %d' % (len(goods) - dropped))


def main():
    ts_load = time.time()

    # list all files in INPUT_DIR
    files = glob.glob('%s/*.json' % INPUT_DIR)

    # each file parsed separately
    for fn in files:
        fn_name = fn[len(INPUT_DIR) + 1:]
        print('Start file: %s' % fn_name)
        fn_load = time.time()

        # load and analyse
        detections, count = load_json(fn, progress_and_process_image)
        print('  ... droped by non image: %d' % (count - len(detections)))
        start_analyze(detections)

        # found IDs of goods
        leave_good = set()
        for d in detections:
            if d.get(CLASSIFIED) != CLASS_ARTIFACT:
                leave_good.add(d.get(ID))

        # load again and save as
        to_save, count = load_json(fn, lambda d, c, r: d.get(ID) in leave_good)
        fn_out = '%s/%s' % (OUTPUT_DIR, fn_name)
        with open(fn_out, 'w') as json_file:
            json.dump(to_save, json_file)

        print('  file %s done, time: %03ds, since start: %03ds, hits with images: %d, dropped: %d, leaved: %d' % (fn_name, time.time() - fn_load, time.time() - ts_load, count, count - len(to_save), len(to_save)))


if __name__ == '__main__':
    main()
