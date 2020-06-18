from io import BytesIO
from typing import Tuple, Callable

from PIL import Image

from credo_cf.commons.consts import IMAGE, FRAME_DECODED, DARKNESS, BRIGHTEST, BRIGHTER_COUNT, FRAME_CONTENT, CROP_SIZE, EDGE, X, \
    WIDTH, Y, HEIGHT, CROP_X, CROP_Y
from credo_cf.io.io_utils import decode_base64


def get_brightest_channel(pixel: Tuple[int, int, int, int]) -> int:
    """
    Get brightest channel for pixel
    :param pixel: RGBA pixel
    :return: brightest pixel from [R, G, B] list
    """
    r, g, b, a = pixel
    return max([r, g, b])


def load_image(detection: dict) -> Image:
    """
    Load image from ``frame_content`` to object's key with some calculated metrics.

    Required keys:
      * ``frame_encoded`` (byte array) or ``frame_content`` (base64-encoded string)

    Keys will be add:
      * ``frame_encoded``: when no ``frame_encoded`` then ``frame_content`` will be decoded and stored in this key
      * ``image``: object of ``PIP`` library with loaded image
      * ``crop_size``: tuple of (width, height) of loaded image
      * ``edge``: ``True`` when image is near edge of original image frame (half of max of width and height of loaded image)
      * ``crop_x`` and ``crop_y``: coordinates of left-top corner of loaded image in original image frame

    The ``crop_x`` and ``crop_y`` may be used to reconstruction original image frame from loaded images.

    :param detection: detection object with frame_encoded or frame_content
    :return: image object
    """
    if detection.get(FRAME_DECODED) is None:
        detection[FRAME_DECODED] = decode_base64(detection.get(FRAME_CONTENT))

    frame_decoded = detection.get(FRAME_DECODED)
    img = Image.open(BytesIO(frame_decoded)).convert('RGBA')
    detection[IMAGE] = img

    # extract basic image parameters
    detection[CROP_SIZE] = img.size
    w, h = img.size

    # center of crop position
    x = detection.get(X)
    y = detection.get(Y)

    # CMOS/CCD resolution
    width = detection.get(WIDTH)
    height = detection.get(HEIGHT)

    # check if detection is at the edge of sensor frame
    detection[EDGE] = False

    fx = w // 2
    fy = h // 2
    fm = max(fx, fy)
    left_edge = fm > x
    top_edge = fm > y
    right_edge = width < fm + x
    bottom_edge = height < fm + y

    if w != h:
        detection[EDGE] = True
    else:
        if left_edge or top_edge or right_edge or bottom_edge:
            detection[EDGE] = True

    # calc left top position on sensor frame of image crop
    if detection[EDGE]:
        if left_edge and top_edge:
            detection[CROP_X] = 0
            detection[CROP_Y] = 0
        elif top_edge and right_edge:
            detection[CROP_X] = width - w
            detection[CROP_Y] = 0
        elif right_edge and bottom_edge:
            detection[CROP_X] = width - w
            detection[CROP_Y] = height - h
        elif bottom_edge and left_edge:
            detection[CROP_X] = 0
            detection[CROP_Y] = height - h
        elif left_edge:
            detection[CROP_X] = 0
            detection[CROP_Y] = y - fy
        elif top_edge:
            detection[CROP_X] = x - fx
            detection[CROP_Y] = 0
        elif right_edge:
            detection[CROP_X] = width - w
            detection[CROP_Y] = y - fy
        elif bottom_edge:
            detection[CROP_X] = x - fx
            detection[CROP_Y] = y - fy
    else:
        detection[CROP_X] = x - fx
        detection[CROP_Y] = y - fy

    return img


def measure_darkness_brightest(detection: dict, pixel_parser: Callable[[Tuple[int, int, int, int]], int] = get_brightest_channel) -> Tuple[int, int]:
    """
    Measure brightest and darkness pixel excluding #000 pixels and using pixel_parser for get pixel value.
    Set values to 'image_darkness' and 'image_brightest' fields and return.
    :param detection: detection with 'image' field
    :param pixel_parser: get one value from RGBA channels
    :return: tuple of darkness and brightest
    """
    assert detection.get(IMAGE) is not None

    hit_img = detection.get(IMAGE)
    width, height = hit_img.size

    darkness = 255
    brightest = 0
    for cy in range(height):
        for cx in range(width):
            g = pixel_parser(hit_img.getpixel((cx, cy)))
            if g != 0:
                brightest = max(brightest, g)
                darkness = min(darkness, g)
    detection[DARKNESS] = darkness
    detection[BRIGHTEST] = brightest
    return darkness, brightest


def count_of_brightest_pixels(detection: dict, threshold: int, pixel_parser: Callable[[Tuple[int, int, int, int]], int] = get_brightest_channel) -> int:
    """
    Count pixels brighter than threshold param. Using pixel_parser for get pixel value.
    Set values to 'image_brighter_count_{threshold}' fields and return.
    :param detection: detection with 'image' field
    :param threshold: greater of equal bright of pixel will be counted
    :param pixel_parser: get one value from RGBA channels
    :return: count of pixel brighter or equal than threshold
    """
    assert detection.get(IMAGE) is not None

    hit_img = detection.get(IMAGE)
    width, height = hit_img.size

    bright_count = 0
    for cy in range(height):
        for cx in range(width):
            g = pixel_parser(hit_img.getpixel((cx, cy)))
            if g >= threshold:
                bright_count += 1
    detection[BRIGHTER_COUNT % threshold] = bright_count
    return bright_count


def detection_load_parser(detection: dict):
    if not detection.get(FRAME_CONTENT):
        return False
    load_image(detection)
    return True
