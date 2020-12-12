# original fields of detection from JSON
ID = 'id'
DEVICE_ID = 'device_id'
TIMESTAMP = 'timestamp'

WIDTH = 'width'
HEIGHT = 'height'

X = 'x'
Y = 'y'

FRAME_CONTENT = 'frame_content'  # excluded in CSV


# added fields by processing functions (all excluded in CSV)
FRAME_DECODED = 'frame_decoded'
IMAGE = 'image'
GRAY = 'gray'
ORIG_IMAGE = 'orig_image'

# image basic stats
DARKNESS = 'image_darkness'  # bright of darkness pixel (0-255)
BRIGHTEST = 'image_brightest'  # bright of brightest pixel (0-255)
BRIGHTER_COUNT = 'image_brighter_count_%03d'  # count of pixels brighter or equal than %03d
BRIGHTER_COUNT_USED = 'image_brighter_count_used'  # count of pixels brighter or equal than BRIGHTER_COUNT_THRESHOLD
BRIGHTER_COUNT_THRESHOLD = 'image_brighter_count_threshold'  # used threshold for BRIGHTER_COUNT_USED

# reconstruction dark filled area
EDGE = 'edge'
CROP_X = 'crop_x'
CROP_Y = 'crop_y'
CROP_SIZE = 'crop_size'

# classification
CLASSIFIED = 'classified'  # result of classification
CLASS_ARTIFACT = 'artifact'

# artifact filter values
ARTIFACT_TOO_OFTEN = 'artifact_too_often'  # count of occurrences for too_often filter
ARTIFACT_HOT_PIXEL = 'artifact_hot_pixel'  # count of occurrences for hot_pixel filter
ARTIFACT_NEAR_HOT_PIXEL = 'artifact_near_hot_pixel'  # count of occurrences for near_hot_pixel filter
ARTIFACT_NEAR_HOT_PIXEL_REFXY = 'artifact_near_hot_pixel_refxy'  # reference XY for near_hot_pixel, all hot pixels in radius to this XY are classified as near hot pixel, default radius: 5px
ARTIFACT_TOO_DARK = 'artifact_too_dark'  # BRIGHTEST - DARKNESS
ARTIFACT_TOO_LARGE_BRIGHT_AREA = 'artifact_too_large_bright_area'  # BRIGHTER_COUNT * 1000 / (WIDTH * HEIGHT) for used brighter threshold, default threshold: (BRIGHTEST - DARKNESS) / 4 + DARKNESS

ARTIFACT_NEAR_HOT_PIXEL2 = 'artifact_near_hot_pixel2'  # count of occurrences for near_hot_pixel filter
#ARTIFACT_NEAR_HOT_PIXEL2_COUNT = 'artifact_near_hot_pixel2_count'  # count of neighborhood pixels

BRIGHT_PIXELS = 'bright_pixels'  # number of bright pixels (too_bright.py)
GOOD_BRIGHT = 'good_bright'       # Image isn't too bright or too dark? True / False (too_bright.py)

NOISE_THRESHOLD = 'noise_threshold'  # upper noise threshold
CLEARLY = 'clearly'
WHOLE_AREA = 'whole_area'
FLOOD_AREA = 'flood_area'
WHOLE_DIAGONAL = 'whole_diagonal'
FLOOD_DIAGONAL = 'flood_diagonal'

ASTROPY_FOUND = 'astropy_found'
ASTROPY_ELLIPTICITY = 'astropy_ellipticity'
ASTROPY_ELONGATION = 'astropy_elongation'
ASTROPY_CONVEX_AREA = 'astropy_convex_area'
ASTROPY_SOLIDITY = 'astropy_solidity'
ASTROPY_ORIENTATION = 'astropy_orientation'
ASTROPY_MAJOR_AXIS_LENGTH = 'astropy_major_axis_length'
ASTROPY_MINOR_AXIS_LENGTH = 'astropy_minor_axis_length'

NKG_MASK = 'nkg_mask'
NKG_VALUES = 'nkg_values'
NKG_CORES = 'nkg_cores'
NKG_THRESHOLD = 'nkg_threshold'
NKG_PATH = 'nkg_path'
NKG_DIRECTION = 'nkg_direction'
NKG_DERIVATIVE = 'nkg_derivative'
NKG_UPSCALE = 'nkg_upscale'
NKG_UPSCALE_MASK = 'nkg_upscale_mask'
NKG_SKELETON = 'nkg_skeleton'
