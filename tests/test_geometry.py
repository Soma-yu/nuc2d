import pytest

from nuc2d._geometry import BBox


def test_a_single_point_encloses_an_empty_area_but_is_not_the_empty_box():
    bbox = BBox(3, 4, 3, 4)

    assert not bbox.is_empty
    assert (bbox.width, bbox.height) == (0.0, 0.0)


def test_union():
    a = BBox(0, 0, 10, 10)
    b = BBox(5, -5, 20, 5)

    assert a.union(b) == BBox(0, -5, 20, 10)
    assert a.union(b) == b.union(a)


def test_empty_is_the_identity_of_union():
    a = BBox(0, 0, 10, 10)

    assert BBox.empty().union(a) == a
    assert a.union(BBox.empty()) == a
    assert BBox.empty().union(BBox.empty()).is_empty


def test_union_reduces_over_a_sequence():
    from functools import reduce

    boxes = [BBox(0, 0, 1, 1), BBox(5, 5, 6, 6), BBox(-3, 2, -1, 4)]

    assert reduce(BBox.union, boxes, BBox.empty()) == BBox(-3, 0, 6, 6)


def test_center_x_and_center_y_are_the_midpoints_of_the_corners():
    assert (BBox(0, 0, 10, 20).center_x, BBox(0, 0, 10, 20).center_y) == (5.0, 10.0)
    assert (BBox(-6, -8, -2, -2).center_x, BBox(-6, -8, -2, -2).center_y) == (-4.0, -5.0)

    point = BBox(3, 4, 3, 4)

    assert (point.center_x, point.center_y) == (3.0, 4.0)


def test_an_empty_box_has_no_center():
    empty = BBox.empty()

    with pytest.raises(ValueError):
        empty.center_x
    with pytest.raises(ValueError):
        empty.center_y
