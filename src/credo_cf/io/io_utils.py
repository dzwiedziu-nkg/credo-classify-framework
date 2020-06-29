import sys
from base64 import decodebytes
from pathlib import Path
from typing import List, Optional

from PIL import Image

from credo_cf.commons.consts import FRAME_CONTENT, ID, FRAME_DECODED


def decode_base64(frame_content: str) -> bytes:
    """
    Convert bytes encoded in base64 to array of bytes.
    :param frame_content: bytes encoded in base64
    :return: byte array
    """
    return decodebytes(str.encode(frame_content))


def store_png(root: str, path: List[str or int], name: str or int, image: bytes or Image) -> None:
    """
    Save image in PNG file.
    :param root: root directory for PNG files storage
    :param path: subdirectories, will be created when not exists
    :param name: file name without extensions
    :param image: instance of PIL.Image or array of bytes
    """
    dirs = '/'.join(map(lambda x: str(x), path))
    p = "%s/%s" % (root, dirs)
    fn = '%s/%s.png' % (p, str(name))
    Path(p).mkdir(parents=True, exist_ok=True)
    if isinstance(image, bytes):
        with open(fn, 'wb') as f:
            f.write(image)
    else:
        image.save(fn)


def progress_load_filter(obj: dict, count: int, ret: List[dict]) -> Optional[bool]:
    """
    Notify progress to ``stdout`` after each 10000 parsed objects.

    Note: it is simplest sample implementation of ``_filter`` arg for ``load_json_from_stream()``.
    """
    skip = count - len(ret)
    if count % 10000 == 0:
        print('... just parsed %d and skip %d objects.' % (count, skip), file=sys.stderr)

    return True


def progress_and_process_image(obj: dict, count: int, ret: List[dict]) -> Optional[bool]:
    """
    Notify progress to ``stdout`` after each 10000 parsed objects and load images from ``frame_content``.
    Objects without image will be ignored.

    See ``progress_load_filter()`` for progress notification
    and ``load_image()`` for more info about new keys added to object.

    After load the ``frame_content`` key was be removed from object for memory free.

    Note: it is more complex sample implementation of ``_filter`` arg for ``load_json_from_stream()``.
    """
    progress_load_filter(obj, count, ret)

    if not obj.get(FRAME_CONTENT):
        return False

    try:
        from credo_cf.image.image_utils import load_image
        load_image(obj)
    except Exception as e:
        print('Fail of load image in object with ID: %d, error: %s' % (obj.get(ID), str(e)), file=sys.stderr)
        return False

    obj.pop(FRAME_CONTENT)
    obj.pop(FRAME_DECODED)

    return True
