import dataclasses

import numpy as np
import pytest

from nuc2d import StructureStyle, draw_svg
from nuc2d._style import _FIELD_CHECKS


COLOR_FIELDS = ["node_color", "backbone_color", "basepair_color"]
DASHARRAY_FIELDS = ["backbone_dasharray", "basepair_dasharray"]

GOOD_COLORS = [
    "crimson",
    "#A1b2C3",
    "none",
]
BAD_COLORS = [
    "notacolor",
    "#333",
    "rgb(10, 20, 30)",
    "rgb(10%,20.5%, 30%)",
    "tab:blue",       # matplotlib's spelling, not SVG's
    "C0",
    "rebeccapurple",  # CSS added it after SVG 1.1
    "#12",
    "#12345",
    "rgb(1, 2)",
    "",
    " crimson",
]
GOOD_DASHARRAYS = ["none", "3", "1,0", "0.5,1.5", "4,2,1", "6,2,1,2"]
BAD_DASHARRAYS = [
    "abc", "-1,2", "4,,2", "", "5px,3px", "4,2,",
    "4 2", "1, 1", ".5,1", "1.,1",
]


def test_the_defaults_are_valid():
    StructureStyle()


@pytest.mark.parametrize("field", COLOR_FIELDS)
@pytest.mark.parametrize("value", GOOD_COLORS)
def test_each_way_of_writing_a_color_is_accepted_and_drawn(field, value):
    style = StructureStyle(**{field: value})

    # The renderer checks the value again as it writes it, so accepting one
    # that it would refuse would only move the error to the drawing.
    draw_svg("((...))", style=style).to_svg()


@pytest.mark.parametrize("field", COLOR_FIELDS)
@pytest.mark.parametrize("value", BAD_COLORS)
def test_anything_else_is_refused_where_it_is_written(field, value):
    with pytest.raises(ValueError, match=field):
        StructureStyle(**{field: value})


@pytest.mark.parametrize("field", DASHARRAY_FIELDS)
@pytest.mark.parametrize("value", GOOD_DASHARRAYS)
def test_dash_patterns_are_accepted_and_drawn(field, value):
    draw_svg("((...))", style=StructureStyle(**{field: value})).to_svg()


@pytest.mark.parametrize("field", DASHARRAY_FIELDS)
@pytest.mark.parametrize("value", BAD_DASHARRAYS)
def test_malformed_dash_patterns_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        StructureStyle(**{field: value})


@pytest.mark.parametrize("field", COLOR_FIELDS + DASHARRAY_FIELDS)
def test_a_value_that_is_not_a_string_is_a_type_error(field):
    with pytest.raises(TypeError, match=field):
        StructureStyle(**{field: (1.0, 0.0, 0.0)})


def test_an_assignment_is_checked_too_and_leaves_the_style_as_it_was():
    style = StructureStyle(node_color="crimson")

    with pytest.raises(ValueError, match="node_color"):
        style.node_color = "notacolor"

    assert style.node_color == "crimson"


def test_other_fields_are_assigned_as_before():
    style = StructureStyle()
    style.node_radius = 6.0

    assert style.node_radius == 6.0


SIZE_FIELDS = [
    "backbone_width", "three_prime_arrow_length", "basepair_width",
    "node_radius", "node_font_size",
]


@pytest.mark.parametrize("field", SIZE_FIELDS)
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_a_size_that_is_negative_or_not_finite_is_refused(field, value):
    with pytest.raises(ValueError, match=field):
        StructureStyle(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [("node_color", "red"), ("basepair_dasharray", "2,1")],
)
def test_a_string_is_kept_as_a_str(field, value):
    """Rather than as the subclass of str it was given as."""
    kept = getattr(StructureStyle(**{field: np.str_(value)}), field)

    assert kept == value and type(kept) is str


@pytest.mark.parametrize("value", [10**400, -(10**400)], ids=["positive", "negative"])
def test_a_size_too_large_for_a_float_is_refused_as_not_finite(value):
    """float() raises OverflowError for it, which is not what is promised."""
    with pytest.raises(ValueError, match="node_radius must be a finite number"):
        StructureStyle(node_radius=value)


@pytest.mark.parametrize("field", SIZE_FIELDS)
@pytest.mark.parametrize("value", ["2", None, True])
def test_a_size_that_is_not_a_number_is_a_type_error(field, value):
    with pytest.raises(TypeError, match=field):
        StructureStyle(**{field: value})


@pytest.mark.parametrize("field", SIZE_FIELDS)
def test_a_size_of_zero_is_drawn(field):
    """Zero draws nothing of that part, which is a choice and not a mistake."""
    style = StructureStyle(**{field: 0})

    draw_svg("(((...)))", sequences=["GCGAAACGC"], style=style).to_svg()


def test_a_misspelled_attribute_is_refused_and_the_right_one_named():
    """Assigned, it would otherwise make an attribute that nothing reads."""
    style = StructureStyle()

    with pytest.raises(AttributeError, match="Did you mean: 'node_color'"):
        style.node_colour = "crimson"

    assert not hasattr(style, "node_colour")
    assert style.node_color == "black"


def test_every_field_is_checked_when_it_is_assigned():
    """A field left out of the checks would be refused as a misspelling."""
    assert set(_FIELD_CHECKS) == {f.name for f in dataclasses.fields(StructureStyle)}


def test_styles_are_equal_when_their_fields_are():
    assert StructureStyle() == StructureStyle()
    assert StructureStyle(node_radius=5) == StructureStyle(node_radius=5.0)
    assert StructureStyle() != StructureStyle(node_radius=5.0)


def test_a_color_is_compared_as_it_is_written():
    assert StructureStyle() != StructureStyle(node_color="#000000")


def test_a_style_can_still_be_copied_with_a_field_changed():
    style = StructureStyle(node_color="crimson")

    changed = dataclasses.replace(style, node_radius=6.0)

    assert (changed.node_color, changed.node_radius) == ("crimson", 6.0)
