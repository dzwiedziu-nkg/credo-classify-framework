"""
This is script for prepare RAW cosmic-ray set from CREDO Database to our future working.

The script:

 1. Load JSON's from INPUT_DIR
 2. Exclude some cosmic-ray records.
 3. Store as JSON in OUTPUT_DIR with the same file name.

Exclusion rules:

 * non image
 * non 60x60 size
 * non X and Y in metadata
 * too_often(10, 60000)
 * near_hot_pixel2(3, 5)
 * # too_bright(70, 70)  # currently not used

"""
import json
import math
import sys
import threading
from io import BytesIO
from typing import Optional, List
import numpy as np

from PIL import Image

from credo_cf import group_by_device_id, group_by_resolution, load_json, ID, FRAME_CONTENT, X, Y, DEVICE_ID, IMAGE, decode_base64, store_png, GRAY, \
    progress_and_process_image, ASTROPY_FOUND, ASTROPY_ELLIPTICITY, ASTROPY_SOLIDITY, nkg_mark_hit_area, nkg_make_track, NKG_PATH
from credo_cf.classification.preprocess.astropy_measurs import astropy_measures
from credo_cf.classification.preprocess.nkg_processings import analyse_path


def measure_angle(fn: str):
    hits, count, errors = load_json('../../data/%s' % fn, progress_and_process_image)
    for h in hits:
        nkg_mark_hit_area(h)
        nkg_make_track(h, scale=1, downscale=False, skeleton_method='zhang')
        angle = 0
        if h.get(NKG_PATH):
            angles = analyse_path(h.get(NKG_PATH), cut_first=1, cut_latest=1)
            if len(angles):
                angle = angles[0]
        if math.isnan(angle):
            angle = 0
        store_png('/tmp/credo', [fn], '%03d_%s' % (abs(angle), str(h.get(ID))), h.get(IMAGE))


def main():
    measure_angle('hits_votes_4_class_2.json')
    measure_angle('hits_votes_4_class_3.json')


if __name__ == '__main__':
    main()
    sys.exit(0)  # not always close
