import sys
from base64 import decodebytes, b64encode
from pathlib import Path
from typing import List, Optional
import numpy as np

from PIL import Image

from credo_cf.commons.consts import FRAME_CONTENT, ID


def decode_base64(frame_content: str) -> bytes:
    """
    Convert bytes encoded in base64 to array of bytes.
    :param frame_content: bytes encoded in base64
    :return: byte array
    """
    return decodebytes(str.encode(frame_content))


def encode_base64(data: bytes) -> str:
    """
    Convert bytes to str with data of bytes encoded by base64
    :param data: binary data
    :return: utf-8 string with encoded data
    """
    return b64encode(data).decode("UTF-8")


def store_png(root: str, path: List[str or int], name: str or int, image: bytes or Image or str) -> None:
    """
    Save image in PNG file.
    :param root: root directory for PNG files storage
    :param path: subdirectories, will be created when not exists
    :param name: file name without extensions
    :param image: instance of PIL.Image or array of bytes or string in base64
    """
    dirs = '/'.join(map(lambda x: str(x), path))
    p = "%s/%s" % (root, dirs)
    fn = '%s/%s.png' % (p, str(name))
    Path(p).mkdir(parents=True, exist_ok=True)

    _image = image
    if isinstance(_image, str):
        _image = decode_base64(_image)

    if isinstance(_image, np.ndarray):
        if len(_image.shape) == 3:
            _image = Image.fromarray(_image, 'RGB')
        else:
            _image = Image.fromarray(_image, 'L')

    if isinstance(_image, bytes):
        with open(fn, 'wb') as f:
            f.write(_image)
    else:
        _image.save(fn)


def progress_load_filter(obj: dict, count: int, ret: List[dict]) -> Optional[bool]:
    """
    Notify progress to ``stdout`` after each 10000 parsed objects.

    Note: it is simplest sample implementation of ``_filter`` arg for ``load_json_from_stream()``.
    """
    skip = count - len(ret) - 1
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
        from credo_cf.image.image_utils import load_image, image_basic_metrics
        load_image(obj, True)
        image_basic_metrics(obj)

    except Exception as e:
        print('Fail of load image in object with ID: %d, error: %s' % (obj.get(ID), str(e)), file=sys.stderr)
        return False

    return True
