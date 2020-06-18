from io import BytesIO
from typing import List, Dict

from PIL import Image

from credo_cf.commons.config import Config
from credo_cf.commons.consts import IMAGE, CROP_X, CROP_Y, CROP_SIZE, FRAME_DECODED, CLASSIFIED, CLASS_ARTIFACT, ORIG_IMAGE


def append_to_frame(image: Image, detection: dict):
    hit_img = detection.get(IMAGE)

    cx = detection[CROP_X]
    cy = detection[CROP_Y]
    w, h = detection[CROP_SIZE]

    image.paste(hit_img, (cx, cy, cx + w, cy + h))

    # fix bug in early CREDO Detector App: black filled boundary 1px too large
    image.paste(image.crop((cx + w - 1, cy, cx + w, cy + h)), (cx + w, cy, cx + w + 1, cy + h))
    image.paste(image.crop((cx, cy + h - 1, cx + w, cy + h)), (cx, cy + h, cx + w, cy + h + 1))
    image.paste(image.crop((cx + w - 1, cy + h - 1, cx + w, cy + h)), (cx + w, cy + h, cx + w + 1, cy + h + 1))


def replace_from_frame(image: Image, detection: dict):
    cx = detection.get(CROP_X)
    cy = detection.get(CROP_Y)
    w, h = detection.get(CROP_SIZE)
    hit_img = image.crop((cx, cy, cx + w, cy + h))
    detection[ORIG_IMAGE] = detection[IMAGE]
    detection[IMAGE] = hit_img
    with BytesIO() as output:
        hit_img.save(output, format="png")
        # hit_img.save('/tmp/%d.png' % detection.get('id'))
        detection[FRAME_DECODED] = output.getvalue()


def do_reconstruct(detections: List[dict], config: Config) -> None:
    """
    Reconstruction the fill by black cropped frame in CREDO Detector app v2.

    The detection[x]['frame_decoded'] will be replaced by new value, old value will be stored in detection[x]['frame_decoded_orig'].

    No any changes when count of detections is less or equal 1

    :param detections: should be sorted by detection_id
    :param config: config object
    """
    if len(detections) <= 1:
        return

    sp = [str(detections[0].get('device_id')), str(detections[0].get('timestamp'))]

    image = Image.new('RGBA', (detections[0].get('width'), detections[0].get('height')), (0, 0, 0))
    edge = 'no_edge'
    for d in detections:
        if d.get('edge'):
            edge = 'edge'
    for d in reversed(detections):
        append_to_frame(image, d)
        config.store_png(['recostruct', edge, *sp, 'orig'], d.get('id'), d.get(IMAGE))
    for d in detections:
        replace_from_frame(image, d)
        config.store_png(['recostruct', edge, *sp], d.get('id'), d.get(IMAGE))
    if config.out_dir:
        image.save('%s/recostruct/%s/%s/frame.png' % (config.out_dir, edge, "/".join(sp)))


def check_all_artifacts(detections: List[dict]) -> bool:
    """
    Check if all detections is just classified as artifacts
    :param detections: list of detections to check
    :return: True - all detections is artifacts
    """
    for d in detections:
        if d.get(CLASSIFIED) != CLASS_ARTIFACT:
            return False
    return True


def filter_unclassified(by_timestamp: Dict[int, List[dict]]) -> List[int]:
    """
    Filter detections with one or more unclassified as artifact.
    :param by_timestamp: detections grouped by timestamp
    :return: list of filtered timestamp keys
    """
    ret = []
    for timestamp, detections in by_timestamp.items():
        if not check_all_artifacts(detections):
            ret.append(timestamp)
    return ret
