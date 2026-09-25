import math

import pytest

from nuc2d.geometry import BBox


def test_a_single_point_encloses_an_empty_area_but_is_not_the_empty_box():
    box = BBox(3, 4, 3, 4)

    assert not box.is_empty
    assert (box.width, box.height) == (0.0, 0.0)


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


def test_expanded():
    box = BBox(0, 0, 10, 20)

    assert box.expanded(5) == BBox(-5, -5, 15, 25)
    assert box.expanded(1, 2) == BBox(-1, -2, 11, 22)


def test_expanded_past_itself_is_empty():
    assert BBox(0, 0, 10, 10).expanded(-100).is_empty


def test_translated():
    box = BBox(0, 0, 10, 20).translated(3, -4)

    assert box == BBox(3, -4, 13, 16)


def test_scaled_is_about_the_origin():
    box = BBox(2, 2, 4, 4).scaled(2)

    assert box == BBox(4, 4, 8, 8)


def test_scaled_by_a_negative_factor_keeps_corners_ordered():
    box = BBox(1, 1, 3, 3).scaled(-1)

    assert box == BBox(-3, -3, -1, -1)
    assert not box.is_empty


def test_empty_box_is_unchanged_by_transforms():
    empty = BBox.empty()

    assert empty.expanded(10).is_empty
    assert empty.translated(5, 5).is_empty
    assert empty.scaled(3).is_empty


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


def test_to_viewbox():
    assert BBox(-2, -3, 8, 7).to_viewbox() == (-2, -3, 10, 10)


@pytest.mark.parametrize(
    "method, args",
    [
        ("expanded", (5,)),
        ("translated", (1, 1)),
        ("scaled", (2,)),
    ],
)
def test_transforms_return_a_new_box(method, args):
    box = BBox(0, 0, 10, 10)

    assert getattr(box, method)(*args) is not box
    assert box == BBox(0, 0, 10, 10)
