import dataclasses
import xml.etree.ElementTree as ET
from fractions import Fraction

import matplotlib as mpl
import numpy as np
import pytest

from nuc2d import (
    Component,
    DrawingStyle,
    Placement,
    Scene,
    draw_colorbar,
    draw_structure,
)
from nuc2d._component import _ANCHOR_FRACTIONS, _anchor_point, fit, make_component
from nuc2d._geometry import bbox_around, make_bbox


CLOVERLEAF = "(((((((..((((........)))).(((((.......+))))).....(((((.......))))))))))))...."
PROBS = np.eye(9) * 0.4 + 0.3


def structure(dot_bracket="(((...)))", **kwargs):
    return draw_structure(dot_bracket, **kwargs)


def count(svg_string, tag):
    return sum(
        1 for element in ET.fromstring(svg_string).iter()
        if element.tag.endswith(tag)
    )


# ---------------------------------------------------------------- anchors


@pytest.mark.parametrize("anchor", [*_ANCHOR_FRACTIONS, (0.25, 0.75), (1.3, -0.2)])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_the_anchor_point_lands_on_x_y(anchor, scale):
    placement = Placement(
        component=structure(CLOVERLEAF), x=100.0, y=50.0, anchor=anchor, scale=scale
    )

    assert _anchor_point(placement.bbox, anchor) == pytest.approx((100.0, 50.0))


def test_the_anchor_stays_put_while_the_component_is_scaled():
    component = structure(CLOVERLEAF)
    points = {
        scale: _anchor_point(
            Placement(component=component, x=7.0, y=9.0, anchor=(0.3, 0.6),
                      scale=scale).bbox,
            (0.3, 0.6),
        )
        for scale in (0.5, 3.0)
    }

    assert points[0.5] == pytest.approx(points[3.0])


def test_the_placed_box_is_the_component_box_scaled():
    component = structure(CLOVERLEAF)
    bbox = Placement(component=component, scale=2.5).bbox

    assert bbox.width == pytest.approx(component.bbox.width * 2.5)
    assert bbox.height == pytest.approx(component.bbox.height * 2.5)


def test_upper_left_is_the_default_anchor():
    """So x and y read the way an SVG rect's do."""
    bbox = Placement(component=structure(CLOVERLEAF), x=10.0, y=20.0).bbox

    assert (bbox.xmin, bbox.ymin) == pytest.approx((10.0, 20.0))


@pytest.mark.parametrize(
    "anchor",
    ["upperleft", "top left", (float("nan"), 0.0), (0.0, float("inf"))],
)
def test_an_anchor_of_the_wrong_value_is_a_value_error(anchor):
    with pytest.raises(ValueError, match="anchor"):
        Placement(component=structure(), anchor=anchor)


@pytest.mark.parametrize(
    "anchor, name",
    [([0.5, 0.5], "anchor"), (None, "anchor"), (1, "anchor"),
     ((0.5,), "anchor"), ((0.5, 0.5, 0.5), "anchor"),
     (("a", 0.0), r"anchor\[0\]"), ((0.0, "a"), r"anchor\[1\]"),
     ((True, False), r"anchor\[0\]")],
)
def test_an_anchor_of_the_wrong_type_is_a_type_error(anchor, name):
    """A tuple of another length is another type, as os.utime has it.

    An item that is not a number is named, as str.join names one.
    """
    with pytest.raises(TypeError, match=name):
        Placement(component=structure(), anchor=anchor)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_scale_must_be_positive_and_finite(scale):
    with pytest.raises(ValueError, match="scale"):
        Placement(component=structure(), scale=scale)


def test_a_placement_keeps_its_numbers_as_floats():
    placement = Placement(
        component=structure(),
        x=np.int64(3),
        y=Fraction(1, 2),
        anchor=(1, np.float32(0.5)),
        scale=2,
    )

    kept = (placement.x, placement.y, *placement.anchor, placement.scale)
    assert kept == (3.0, 0.5, 1.0, 0.5, 2.0)
    assert all(type(number) is float for number in kept)
    assert type(placement.anchor) is tuple


@pytest.mark.parametrize("name", ["center", np.str_("center")])
def test_a_named_anchor_is_kept_as_its_name(name):
    anchor = Placement(component=structure(), anchor=name).anchor

    assert anchor == "center" and type(anchor) is str


def test_placements_are_equal_when_they_place_one_component_alike():
    """The component is the same one; one drawn alike is another."""
    component = structure()

    assert Placement(component=component, x=1) == Placement(component=component, x=1.0)
    assert Placement(component=component) != Placement(component=component, x=1.0)
    assert Placement(component=component) != Placement(component=structure())
    assert len({Placement(component=component), Placement(component=component)}) == 1


def test_an_anchor_is_compared_as_it_is_given():
    component = structure()

    assert Placement(component=component, anchor="upper left") != Placement(
        component=component, anchor=(0.0, 0.0)
    )


def test_placements_are_frozen_and_hashable():
    placement = Placement(component=structure())

    with pytest.raises(dataclasses.FrozenInstanceError):
        placement.x = 1.0
    hash(placement)


@pytest.mark.parametrize("component", ["(((...)))", None])
def test_what_is_placed_must_be_a_component(component):
    with pytest.raises(TypeError, match="component must be a Component"):
        Placement(component=component)


def test_a_placement_is_not_placed_again_without_making_a_component_of_it():
    with pytest.raises(TypeError, match="got Placement"):
        Placement(component=Placement(component=structure()))


@pytest.mark.parametrize("name", ["x", "y"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_x_and_y_must_be_finite(name, value):
    with pytest.raises(ValueError, match=name):
        Placement(component=structure(), **{name: value})


@pytest.mark.parametrize("name", ["x", "y", "scale"])
@pytest.mark.parametrize("value", ["1", True, None])
def test_x_y_and_scale_must_be_numbers(name, value):
    """True is refused too: it is a mistake, not a coordinate of 1."""
    with pytest.raises(TypeError, match=name):
        Placement(component=structure(), **{name: value})


def test_numbers_of_any_real_type_are_accepted():
    placement = Placement(
        component=structure(), x=np.float32(1.5), y=2, scale=np.int64(3)
    )

    assert (placement.bbox.xmin, placement.bbox.ymin) == pytest.approx((1.5, 2.0))


# ---------------------------------------------------------------- components


@pytest.mark.parametrize(
    "kwargs", [{}, {"bbox": make_bbox(0.0, 0.0, 10.0, 10.0)}]
)
def test_a_component_is_not_made_by_calling_the_class(kwargs):
    """What a component holds is private, so there is nothing to make one of."""
    with pytest.raises(TypeError, match="draw_structure"):
        Component(**kwargs)


def test_a_component_is_equal_only_to_itself():
    """Two drawn alike are two components, as two of anything drawn are."""
    placement = Placement(component=structure())
    a, b = structure(), structure()
    c, d = (Component.from_placements([placement]) for _ in range(2))

    assert a == a and c == c
    assert a != b and c != d
    assert len({a, b, c, d}) == 4


def test_a_component_cannot_be_changed():
    component = structure()
    bbox = component.bbox

    with pytest.raises(AttributeError):
        component.bbox = make_bbox(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(AttributeError):
        component.label = "tRNA"

    assert component.bbox == bbox


def test_a_component_shows_its_box():
    component = structure()

    assert repr(component) == f"<Component bbox={component.bbox!r}>"


# ---------------------------------------------------------------- from_placements


def test_from_placements_encloses_everything_it_places():
    a, b = structure(CLOVERLEAF), structure()
    pa = Placement(component=a)
    pb = Placement(component=b, x=pa.bbox.xmax + 20.0, y=0.0)

    panel = Component.from_placements([pa, pb])

    assert panel.bbox == bbox_around([pa.bbox, pb.bbox])


def test_from_placements_of_nothing_is_refused():
    """Its box would enclose nothing, and no scene can be framed on it."""
    with pytest.raises(ValueError, match="placements is empty"):
        Component.from_placements([])


def test_from_placements_refuses_a_component_that_was_not_placed():
    with pytest.raises(TypeError, match="Placement"):
        Component.from_placements([structure()])


@pytest.mark.parametrize(
    "make",
    [
        lambda p: p,                    # one placement, not in a list
        lambda p: (p,),
        lambda p: (q for q in [p]),     # read once, and it used to come out empty
        lambda p: {p},                  # no order to draw in
    ],
)
def test_from_placements_takes_a_list(make):
    placement = Placement(component=structure())

    with pytest.raises(TypeError, match="placements must be a list"):
        Component.from_placements(make(placement))


def test_from_placements_is_not_changed_by_changing_the_list_it_was_given():
    placements = [Placement(component=structure())]
    component = Component.from_placements(placements)
    bbox, svg = component.bbox, Scene(component).to_svg()

    placements.append(Placement(component=structure(CLOVERLEAF), x=500.0))

    assert component.bbox == bbox
    assert Scene(component).to_svg() == svg


def first_tag(svg_string, tags):
    """Return which of ``tags`` appears first in the document."""
    for element in ET.fromstring(svg_string).iter():
        for tag in tags:
            if element.tag.endswith(tag):
                return tag
    return None


def test_components_are_drawn_in_the_order_given():
    """A later one covers an earlier one, as SVG draws in document order."""
    colorbar = Placement(component=draw_colorbar())  # draws a rect
    plain = Placement(component=structure())  # draws circles, no rect

    def drawn_first(placements):
        svg = Scene(Component.from_placements(placements)).to_svg()
        return first_tag(svg, ["rect", "circle"])

    assert drawn_first([colorbar, plain]) == "rect"
    assert drawn_first([plain, colorbar]) == "circle"


def test_a_definition_is_written_once_however_many_components_use_it():
    a, b = draw_colorbar(), draw_colorbar()  # each refers to one gradient
    pa = Placement(component=a)
    panel = Component.from_placements([pa, Placement(component=b, x=pa.bbox.xmax)])

    svg = Scene(panel).to_svg()

    assert count(svg, "linearGradient") == 1


def test_different_definitions_are_all_written():
    turbo = draw_colorbar()
    viridis = draw_colorbar(style=DrawingStyle(colormap=mpl.colormaps["viridis"]))
    pt = Placement(component=turbo)

    svg = Scene(
        Component.from_placements([pt, Placement(component=viridis, x=pt.bbox.xmax)])
    ).to_svg()

    assert count(svg, "linearGradient") == 2


def test_a_structure_refers_to_no_definitions():
    """Its arrowheads are shapes of their own, not markers defined once."""
    svg = Scene(structure(CLOVERLEAF)).to_svg()

    assert count(svg, "marker") == 0
    assert count(svg, "linearGradient") == 0


def test_a_composed_component_can_be_placed_again():
    inner = Component.from_placements([Placement(component=structure(CLOVERLEAF))])
    outer = Component.from_placements([Placement(component=inner, x=5.0, y=5.0, scale=0.5)])

    assert (outer.bbox.xmin, outer.bbox.ymin) == pytest.approx((5.0, 5.0))
    assert outer.bbox.width == pytest.approx(inner.bbox.width * 0.5)


# ---------------------------------------------------------------- fit


@pytest.mark.parametrize("anchor", ["center", "upper left", "center right", (0.2, 0.8)])
def test_fit_occupies_exactly_the_slot(anchor):
    slot = make_bbox(-500.0, 0.0, 0.0, 500.0)

    placement = fit(structure(CLOVERLEAF), slot, anchor=anchor)

    for got, want in zip(
        (placement.bbox.xmin, placement.bbox.ymin, placement.bbox.xmax, placement.bbox.ymax),
        (slot.xmin, slot.ymin, slot.xmax, slot.ymax),
    ):
        assert got == pytest.approx(want)


def test_fit_uses_the_largest_scale_that_fits():
    component = structure(CLOVERLEAF)
    slot = make_bbox(0.0, 0.0, 300.0, 1000.0)

    placement = fit(component, slot, anchor="center")

    assert placement.scale == pytest.approx(
        min(300.0 / component.bbox.width, 1000.0 / component.bbox.height)
    )


def test_fit_aligns_the_component_by_its_anchor():
    """The drawn part sits against the side the anchor names."""
    component = structure(CLOVERLEAF)
    slot = make_bbox(0.0, 0.0, 2000.0, 500.0)  # much wider than the component

    placement = fit(component, slot, anchor="center right")

    # The padded box is placed by its upper left corner, so the component's
    # own box lands at that corner plus its offset inside the padding.
    padded, s = placement.component.bbox, placement.scale
    bbox = component.bbox
    right = placement.x + s * (bbox.xmax - padded.xmin)
    top = placement.y + s * (bbox.ymin - padded.ymin)
    bottom = placement.y + s * (bbox.ymax - padded.ymin)

    assert right == pytest.approx(slot.xmax)
    assert (top, bottom) == pytest.approx((slot.ymin, slot.ymax))


def test_fit_refuses_what_has_no_area():
    with pytest.raises(ValueError):
        fit(
            make_component(make_bbox(0.0, 0.0, 0.0, 1.0)),
            make_bbox(0.0, 0.0, 1.0, 1.0),
            anchor="center",
        )
    with pytest.raises(ValueError):
        fit(structure(), make_bbox(0.0, 0.0, 0.0, 1.0), anchor="center")
