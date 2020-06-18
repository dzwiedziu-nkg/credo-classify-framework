from time import time
from typing import Optional, List, Any

from PIL import Image

from credo_cf.classification.artifact.too_large_bright_area import by_darkness_brightest_threshold
from credo_cf.io.io_utils import store_png
from credo_cf.commons.utils import print_log


class Config:
    """
    Provide methods for printing of logs of progress and storage image files for debug and store default values.
    """

    hot_pixel_often = 3
    near_hot_pixel_often = 3
    near_hot_pixel_distance = 5
    too_often = 4
    too_often_time_division = 60000

    simple_classify_ignore_artifacts = False  # set to True to: no image processing classified as artifact by *hot_pizel and too_often
    too_dark_spread = 50
    too_large_bright_area_bac = 30
    too_large_bright_area_threshold = by_darkness_brightest_threshold

    count_of_brightest_pixels = False  # set to True cause significantly increase time of analysis
    count_of_brightest_pixels_from = 0
    count_of_brightest_pixels_to = 256

    log_indent = 0

    def __init__(self, out_dir: str = None, log: bool = True) -> None:
        """
        Initialize config.
        :param out_dir: root dir for storage image files for debug
        :param log: do print progress logs
        """
        self.out_dir = out_dir
        self.log = log

    def print_log(self, message: str, timesatmp: Optional[float] = None):
        """
        Print log with execution time measuring and using self.log_indent.
        :param message: log message
        :param timesatmp: previous timestamp for measure execution time
        :return: current timestamp
        """
        if self.log:
            return print_log(" " * self.log_indent + message, timesatmp)
        return time()

    def store_png(self, path: List[str], name: Any, image: [bytes, Image]) -> None:
        """
        Store image file for debug as PNG file.
        :param path: subdirectories, will be created when not exists
        :param name: file name without extensions
        :param image: instance of PIL.Image or array of bytes
        """
        if self.out_dir:
            store_png(self.out_dir, path, name, image)

    def change_log_indent(self, value: int) -> None:
        """
        Relative change of current log indent.
        :param value: please use positive value for increase od negative for decrease log_indent.
        """
        self.log_indent += value
