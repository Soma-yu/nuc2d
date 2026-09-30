from nuc2d._geometry import bbox_around, is_empty_bbox, make_bbox



def test_the_size_is_the_distance_between_the_corners():
    bbox = make_bbox(-2, 3, 8, 7)

    assert (bbox.width, bbox.height) == (10, 4)


def test_a_single_point_encloses_an_empty_area_but_is_not_an_empty_box():
    bbox = make_bbox(3, 4, 3, 4)

    assert not is_empty_bbox(bbox)
    assert (bbox.width, bbox.height) == (0.0, 0.0)


def test_the_box_around_boxes_encloses_them_all():
    boxes = [make_bbox(0, 0, 1, 1), make_bbox(5, 5, 6, 6), make_bbox(-3, 2, -1, 4)]

    assert bbox_around(boxes) == make_bbox(-3, 0, 6, 6)
    assert bbox_around(reversed(boxes)) == make_bbox(-3, 0, 6, 6)


def test_the_box_around_one_box_is_that_box():
    assert bbox_around([make_bbox(1, 2, 3, 4)]) == make_bbox(1, 2, 3, 4)


def test_the_box_around_nothing_is_empty_and_has_no_size():
    around = bbox_around([])

    assert is_empty_bbox(around)
    assert (around.width, around.height) == (0.0, 0.0)


def test_an_empty_box_is_passed_over():
    empty = bbox_around([])

    assert bbox_around([empty, make_bbox(0, 0, 10, 10), empty]) == make_bbox(0, 0, 10, 10)
    assert bbox_around([make_bbox(5, 5, 1, 1), make_bbox(0, 0, 2, 2)]) == make_bbox(0, 0, 2, 2)
    assert is_empty_bbox(bbox_around([empty, empty]))


def test_a_box_is_equal_to_one_with_the_same_corners():
    assert make_bbox(0, 0, 1, 2) == make_bbox(0, 0, 1, 2)
    assert make_bbox(0, 0, 1, 2) != make_bbox(0, 0, 2, 1)
    assert len({make_bbox(0, 0, 1, 2), make_bbox(0, 0, 1, 2)}) == 1


def test_a_box_is_not_equal_to_its_corners():
    """A tuple is not a box, whatever it holds."""
    assert make_bbox(0, 0, 1, 2) != (0, 0, 1, 2)


def test_a_box_shows_its_corners():
    assert repr(make_bbox(0.0, 1.0, 2.0, 3.0)) == (
        "<BBox xmin=0.0 ymin=1.0 xmax=2.0 ymax=3.0>"
    )
