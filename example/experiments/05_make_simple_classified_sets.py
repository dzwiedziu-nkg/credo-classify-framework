import json
import os
from typing import Optional, List

from credo_cf import load_json, progress_load_filter, load_image, GRAY, ID, store_png, \
    ASTROPY_FOUND, ASTROPY_ELLIPTICITY, ASTROPY_SOLIDITY, \
    IMAGE, FRAME_DECODED, CROP_SIZE
from credo_cf.classification.preprocess.astropy_measurs import astropy_measures

OUTPUT_DIR = '/tmp/credo'
WORKING_SET = '/tmp/trusted-fixed.json'


def main():
    def load_parser(obj: dict, count: int, ret: List[dict]) -> Optional[bool]:
        progress_load_filter(obj, count, ret)
        load_image(obj, False)
        return True

    objs, count, errors = load_json(WORKING_SET, load_parser)
    # config data source, please uncomment and use one from both

    spots = []
    tracks = []
    worms = []
    others = []
    all = []
    multi = []

    for d in objs:
        astropy_measures(d)
        d.pop(GRAY)
        d.pop(FRAME_DECODED)

        if d[ASTROPY_FOUND] == 1:
            all.append(d)

            if d[ASTROPY_ELLIPTICITY][0] < 0.1 and d[ASTROPY_SOLIDITY][0] > 0.8:
                spots.append(d)
            elif d[ASTROPY_ELLIPTICITY][0] > 0.8 and d[ASTROPY_SOLIDITY][0] > 0.8:
                tracks.append(d)
            elif d[ASTROPY_SOLIDITY][0] < 0.8:
                worms.append(d)
            else:
                others.append(d)
        elif d[ASTROPY_FOUND] > 1:
            multi.append(d)

    def store_pngs(arr, subdir):
        for a in arr:
            store_png(OUTPUT_DIR, [subdir, 'by_solidity'], '%.3f_%.3f_%d' % (a[ASTROPY_SOLIDITY][0], a[ASTROPY_ELLIPTICITY][0], a[ID]), a[IMAGE])
            store_png(OUTPUT_DIR, [subdir, 'by_ellipticity'], '%.3f_%.3f_%d' % (a[ASTROPY_ELLIPTICITY][0], a[ASTROPY_SOLIDITY][0], a[ID]), a[IMAGE])

            s = int(a[ASTROPY_SOLIDITY][0] * 5) * 2
            e = int(a[ASTROPY_ELLIPTICITY][0] * 5) * 2
            store_png(OUTPUT_DIR, [subdir, 'by_matrix_solidity_per_ellipticity', '%02d-%02d_%02d-%02d' % (s, s + 2, e, e + 2)], '%d' % a[ID], a[IMAGE])

            # a.pop(IMAGE)
        # with open(os.path.join(OUTPUT_DIR, '%s.json' % subdir), 'w') as json_file:
        #     json.dump({'detections': arr}, json_file)

    store_pngs(spots, 'spots')
    store_pngs(tracks, 'tracks')
    store_pngs(worms, 'worms')
    store_pngs(others, 'others')
    store_pngs(all, 'all')
    store_pngs(multi, 'multi')


if __name__ == '__main__':
    main()
