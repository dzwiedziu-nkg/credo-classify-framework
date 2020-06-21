from credo_cf import X, Y, hot_pixel, ARTIFACT_HOT_PIXEL, CLASSIFIED, CLASS_ARTIFACT


def test_hot_pixel():

    # 1. Have 3 hits in the same XY and another 2 in other XY
    detections = [
        {X: 200, Y: 300},
        {X: 200, Y: 300},
        {X: 200, Y: 300},
        {X: 201, Y: 300},
        {X: 200, Y: 301},
    ]

    # Check with threshold - 3, so 3 should me classified as artifact and anothers no
    yes, no = hot_pixel(detections, 3)
    assert len(yes) == 3
    assert len(no) == 2
    for y in yes:
        assert y[X] == 200
        assert y[Y] == 300
        assert y[ARTIFACT_HOT_PIXEL] == 3
        assert y[CLASSIFIED] == CLASS_ARTIFACT
    for n in no:
        assert n[X] != 200 or n[Y] != 300
        assert n[ARTIFACT_HOT_PIXEL] == 1
        assert n.get(CLASSIFIED) != CLASS_ARTIFACT

    # Check with threshold - 4, so any shouldn't classified as artifact
    yes, no = hot_pixel(detections, 4)
    assert len(yes) == 0
    assert len(no) == 5

    # 2. Have only one hit
    detections = [
        {X: 200, Y: 300}
    ]

    yes, no = hot_pixel(detections, 3)
    assert len(yes) == 0
    assert len(no) == 1
    assert no[0][ARTIFACT_HOT_PIXEL] == 1

    # 3. Empty array, should not be crashed
    yes, no = hot_pixel([], 3)
    assert len(yes) == 0
    assert len(no) == 0
