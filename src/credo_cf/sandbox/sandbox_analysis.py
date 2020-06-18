from typing import List

import numpy as np
from PIL import Image

from credo_cf.classification.artifact.too_large_bright_area import too_large_bright_area_classify
from credo_cf.commons.config import Config
from credo_cf.commons.consts import FRAME_DECODED, FRAME_CONTENT, DEVICE_ID, ID, ARTIFACT_HOT_PIXEL, ARTIFACT_TOO_LARGE_BRIGHT_AREA, ARTIFACT_TOO_DARK, \
    ARTIFACT_TOO_OFTEN, ARTIFACT_NEAR_HOT_PIXEL2, EDGE, IMAGE, X, WIDTH, Y, HEIGHT, BRIGHTEST, DARKNESS, BRIGHTER_COUNT_USED, CLASSIFIED, CLASS_ARTIFACT
from credo_cf.commons.grouping import group_by_resolution, group_by_device_id
from credo_cf.commons.utils import get_resolution_key, join_tuple, get_and_set
from credo_cf.image.image_utils import count_of_brightest_pixels, get_brightest_channel
from credo_cf.io.io_utils import decode_base64


def get_th(d: dict):
    return (d.get(BRIGHTEST) - d.get(DARKNESS))*1/4 + d.get(DARKNESS)


def sandbox_analysis(detections: List[dict], config: Config) -> None:
    samples = {}

    by_device_id = group_by_device_id(detections)
    for device_id, dd in by_device_id.items():
        by_resolution = group_by_resolution(dd, None)
        for res, cd in by_resolution.items():
            h = np.zeros((res[1], res[0]))
            name = '%s_%s' % (str(device_id), join_tuple(res))
            counting = {}

            for d in cd:
                ok = d.get(CLASSIFIED, '') != CLASS_ARTIFACT
                if ok:
                    continue

                # if d.get(ARTIFACT_NEAR_HOT_PIXEL2) < 2:
                #     continue

                _id = d.get(ID)
                # if _id != 17468625:
                #     continue
                kd = d.get(DEVICE_ID)
                x = d.get(X)
                y = d.get(Y)
                img = d.get(FRAME_DECODED) or decode_base64(d.get(FRAME_CONTENT))

                rk = get_resolution_key(d)
                rk_str = join_tuple(rk)

                hit_img = d.get(IMAGE)
                width = hit_img.size[0]
                height = hit_img.size[1]

                if width != height:
                    d[EDGE] = True
                else:
                    fx = width // 2
                    fy = height // 2
                    if fx > x or d.get(WIDTH) < fx + x or fy > y or d.get(HEIGHT) < fy + y:
                        d[EDGE] = True

                if d.get(EDGE):
                    continue

                # if d.get(ARTIFACT_TOO_DARK) < config.too_dark_spread:
                #     continue

                th = get_th(d)
                count_of_brightest_pixels(d, th)
                too_large_bright_area_classify(d, th, 0)

                rel_x = x - width // 2
                rel_y = y - height // 2

                # if d.get(BRIGHTER_COUNT_USED) > 2:
                #     continue

                for cy in range(height):
                    for cx in range(width):
                        if get_brightest_channel(hit_img.getpixel((cx, cy))) >= th:
                            h[rel_y + cy][rel_x + cx] += 1

                get_and_set(counting, (x, y), [])
                counting[(x, y)].append(d)

                # config.store_png(['hot_pixels_histogram', name], _id, img)

            if len(counting.keys()) == 0:
                continue

            prefix = '%04d_%04d' % (len(counting.keys()), h.max())
            fn = '%s_%s' % (prefix, name)
            h = np.log10(h + 1)
            h_max = h.max()
            h_normalized = (h * 127.0 / h_max)
            shift_128 = (h_normalized > 0).astype(int) * 128
            h_normalized = (h_normalized + shift_128).astype('uint8')
            im = Image.fromarray(h_normalized)
            config.store_png(['hot_pixels_histogram'], fn, im)
            with open('%s/hot_pixels_histogram/%s.csv' % (config.out_dir, fn), 'wt') as f:
                for key, hits in counting.items():
                    f.write('%s\t%d\n' % (join_tuple(key), len(hits)))
                    for d in hits:
                        _id = d.get(ID)
                        idn = '%s_%d' % (join_tuple(key), _id)
                        img = d.get(FRAME_DECODED) or decode_base64(d.get(FRAME_CONTENT))
                        config.store_png(['hot_pixels_histogram', fn], idn, img)
