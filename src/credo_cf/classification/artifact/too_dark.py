from credo_cf.commons.consts import DARKNESS, BRIGHTEST, CLASSIFIED, CLASS_ARTIFACT, ARTIFACT_TOO_DARK, IMAGE
from credo_cf.image.image_utils import measure_darkness_brightest


def too_dark_classify(detection: dict, spread: int = 50) -> None:
    """
    Classify detections as artifact when subtraction of brightest and darkness is less than spread param.
    :param detection: detection with 'image_darkness' and 'image_brightest' field
    :param spread: threshold for classify as artifact
    """
    assert detection.get(DARKNESS) is not None
    assert detection.get(BRIGHTEST) is not None

    darkness = detection.get(DARKNESS)
    brightest = detection.get(BRIGHTEST)

    diff = brightest - darkness

    detection[ARTIFACT_TOO_DARK] = diff
    if diff < spread:
        detection[CLASSIFIED] = CLASS_ARTIFACT


def too_dark(detection: dict, spread: int = 50) -> None:
    """
    Classify detections as artifact when subtraction of brightest and darkness is less than spread param.
    :param detection: detection with 'image'
    :param spread: threshold for classify as artifact
    """
    assert detection.get(IMAGE) is not None

    measure_darkness_brightest(detection)
    too_dark_classify(detection, spread)
