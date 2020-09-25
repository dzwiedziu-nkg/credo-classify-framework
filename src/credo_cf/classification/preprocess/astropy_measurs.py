import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from photutils import detect_sources, deblend_sources, source_properties, SegmentationImage
from skimage.measure import regionprops, label

from credo_cf import GRAY, ASTROPY_FOUND, append_to_array, ASTROPY_ELLIPTICITY, ASTROPY_ELONGATION, ASTROPY_CONVEX_AREA, ASTROPY_SOLIDITY, \
    ASTROPY_ORIENTATION, ASTROPY_MAJOR_AXIS_LENGTH, ASTROPY_MINOR_AXIS_LENGTH


def astropy_measures(detection: dict, filter_kernel=None, detect_npixels=5, deblend_npixels=200, deblend_nlevels=32, deblend_contrast=0.001) -> dict:
    # load PNG and convert to grayscale
    data = detection[GRAY]
    data = np.flipud(data)  # flip by Y axis (like in screen)

    # find darkness pixel
    brightest = np.max(data)
    darkest = brightest

    rows = data.shape[0]
    cols = data.shape[1]
    for x in range(0, cols):
        for y in range(0, rows):
            v = data[y, x]
            if v:  # skip pixels with 0 bright
                darkest = min(darkest, v)

    # set threshold as 1/8 in line segment from darkness pixel to brightness
    threshold = np.ones(data.shape) * ((brightest - darkest) / 8 + darkest)

    # used parametrs from tutorial for detect_sources
    kernel = filter_kernel
    if kernel is None:
        sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
    segm = detect_sources(data, threshold, npixels=detect_npixels, filter_kernel=kernel)

    # break when not found any hit
    if segm is None:
        detection[ASTROPY_FOUND] = 0
        print(detection['id'])
        return detection
    detection[ASTROPY_FOUND] = segm.nlabels
    if segm.nlabels == 0:
        return detection

    # average bright of background
    sum_background = 0
    count_background = 0
    for x in range(0, cols):
        for y in range(0, rows):
            c = segm.data[y, x]
            v = data[y, x]
            if v and not c:  # skip pixels with 0 bright
                sum_background += v
                count_background += 1
    background_bright = sum_background / count_background if count_background else 0

    # rozdzielenie znalezionych obiektów jednych od drugich, parametry jak w tutorialu poza npixels=200,
    # który trzeba było zwiększyć bo niepotrzebnie dzielił na kawałki długie hity, dzięki temu nie dzieli i nie odbiło
    # się to niegatywnie na wyraźnie osobnych hitach na jednym PNG
    # segm_deblend = deblend_sources(data, segm, npixels=deblend_npixels, filter_kernel=kernel, nlevels=deblend_nlevels, contrast=deblend_contrast)

    # analiza znalezionych obiektów (powierzchnia, eliptyczność itd.)
    cat = source_properties(data, segm)

    # zapis wyników na dysku
    nth = 0
    #
    # image = np.arange(16.).reshape(4, 4)
    # segm = SegmentationImage([[1, 1, 0, 0],
    #                           [1, 0, 0, 2],
    #                           [0, 0, 2, 2],
    #                           [0, 2, 2, 0]])
    # cat = source_properties(image, segm)
    # obj = cat[0]
    #
    # cat = source_properties(data, segm_deblend)
    # obj = cat[0]

    for obj in cat:
        nth += 1
        brightest_obj = 0
        brigh_obj_sum = 0
        brightest_obj_sum = 0
        brightest_obj_count = 0

        for x in range(0, int(obj.xmax.value - obj.xmin.value + 1)):
            for y in range(0, int(obj.ymax.value - obj.ymin.value + 1)):
                c = obj.data_cutout[y, x]
                v = data[int(obj.ymin.value + y), int(obj.xmin.value + x)]
                if c:
                    brightest_obj = max(v, brightest_obj)
                    brightest_obj_sum += v
                    brightest_obj_count += 1
                    brigh_obj_sum += v - background_bright

        brightest_obj_avg = brightest_obj_sum / brightest_obj_count

        try:
            s = segm[0]
            another_props = regionprops(label(s.data))[0]
        except:
            continue

        append_to_array(detection, ASTROPY_ELLIPTICITY, float(obj.ellipticity.value))
        append_to_array(detection, ASTROPY_ELONGATION, float(obj.elongation.value))
        append_to_array(detection, ASTROPY_CONVEX_AREA, int(another_props.convex_area))
        append_to_array(detection, ASTROPY_SOLIDITY, float(another_props.solidity))
        append_to_array(detection, ASTROPY_ORIENTATION, float(obj.orientation.value))
        append_to_array(detection, ASTROPY_MAJOR_AXIS_LENGTH, float(another_props.major_axis_length))
        append_to_array(detection, ASTROPY_MINOR_AXIS_LENGTH, float(another_props.minor_axis_length))
        #
        # # zapis pliku z wynikiem zamarkowania detect_sources
        # pngfn = join(pngdir, fn.replace('.png', '-segm.png'))
        # plt.imsave(pngfn, segm_deblend, origin='lower', cmap=segm_deblend.cmap(random_state=12345))
        #
        # # append to CSV file
        # csv = join(csvdir, '%s-segm.csv' % str(device_id))
        # with open(csv, "a") as fcsv:
        #     fcsv.write('%s\n' % '\t'.join(nvalues))
    return detection