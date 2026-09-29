from nuc2d._geometry import BBox, bbox_around, is_empty_bbox


def test_the_size_is_the_distance_between_the_corners():
    bbox = BBox(-2, 3, 8, 7)

    assert (bbox.width, bbox.height) == (10, 4)


def test_a_single_point_encloses_an_empty_area_but_is_not_an_empty_box():
    bbox = BBox(3, 4, 3, 4)

    assert not is_empty_bbox(bbox)
    assert (bbox.width, bbox.height) == (0.0, 0.0)


def test_the_box_around_boxes_encloses_them_all():
    boxes = [BBox(0, 0, 1, 1), BBox(5, 5, 6, 6), BBox(-3, 2, -1, 4)]

    assert bbox_around(boxes) == BBox(-3, 0, 6, 6)
    assert bbox_around(reversed(boxes)) == BBox(-3, 0, 6, 6)


def test_the_box_around_one_box_is_that_box():
    assert bbox_around([BBox(1, 2, 3, 4)]) == BBox(1, 2, 3, 4)


def test_the_box_around_nothing_is_empty_and_has_no_size():
    around = bbox_around([])

    assert is_empty_bbox(around)
    assert (around.width, around.height) == (0.0, 0.0)


def test_an_empty_box_is_passed_over():
    empty = bbox_around([])

    assert bbox_around([empty, BBox(0, 0, 10, 10), empty]) == BBox(0, 0, 10, 10)
    assert bbox_around([BBox(5, 5, 1, 1), BBox(0, 0, 2, 2)]) == BBox(0, 0, 2, 2)
    assert is_empty_bbox(bbox_around([empty, empty]))
