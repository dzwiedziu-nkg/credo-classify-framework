import bz2
import urllib.request
import io
from typing import List

from credo_cf import load_json_from_stream, progress_and_process_image, group_by_device_id, group_by_resolution, convert_to_gray, too_often, near_hot_pixel2

WORKING_SET = 'http://mars.iti.pk.edu.pl/~nkg/credo/working_set.json.bz2'


def download_working_set(url: str) -> List[dict]:
    print('Download working set...')
    data = urllib.request.urlopen(url).read()

    print('Decompress...')
    json_content = bz2.decompress(data).decode("utf-8")

    print('Prase JSON...')
    objs, count = load_json_from_stream(io.StringIO(json_content), progress_and_process_image)
    print('Parsed %d, skipped %d' % (count, count - len(objs)))
    return objs


all_detections = download_working_set(WORKING_SET)

# print('Make custom grayscale conversion...')
# for d in all_detections:
#     convert_to_gray(d)

print('Group by devices...')
by_devices = group_by_device_id(all_detections)

drop_too_often = 0
drop_near_hot_pixel2 = 0
leave_good = 0

print('Run experiment...')
dev_no = 0
dev_count = len(by_devices.keys())
for device_id, device_detections in by_devices.items():
    by_resolution = group_by_resolution(device_detections)
    for resolution, detections in by_resolution.items():
        dev_no += 1
        print('Start device %d of %d, detectons count: %d' % (dev_no, dev_count, len(detections)))

        # TODO: loop for execute base_xor_filter

        # too_often
        goods = detections
        goods, bads = too_often(goods)
        drop_too_often += len(bads)

        # TODO: too_bright

        # near_hot_pixel2
        goods, bads = near_hot_pixel2(goods)
        drop_near_hot_pixel2 += len(bads)

        # end, counting goods
        leave_good += len(goods)

print('Drop too_often: %d' % drop_too_often)
print('Drop near_hot_pixel2: %d' % drop_near_hot_pixel2)
print('Goods: %d' % leave_good)
