import dataclasses

import matplotlib as mpl
import numpy as np
import pytest

from nuc2d import DrawingStyle, draw_svg
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
    DrawingStyle()


@pytest.mark.parametrize("field", COLOR_FIELDS)
@pytest.mark.parametrize("value", GOOD_COLORS)
def test_each_way_of_writing_a_color_is_accepted_and_drawn(field, value):
    style = DrawingStyle(**{field: value})

    # The renderer checks the value again as it writes it, so accepting one
    # that it would refuse would only move the error to the drawing.
    draw_svg("((...))", style=style).to_svg()


@pytest.mark.parametrize("field", COLOR_FIELDS)
@pytest.mark.parametrize("value", BAD_COLORS)
def test_anything_else_is_refused_where_it_is_written(field, value):
    with pytest.raises(ValueError, match=field):
        DrawingStyle(**{field: value})


@pytest.mark.parametrize("field", DASHARRAY_FIELDS)
@pytest.mark.parametrize("value", GOOD_DASHARRAYS)
def test_dash_patterns_are_accepted_and_drawn(field, value):
    draw_svg("((...))", style=DrawingStyle(**{field: value})).to_svg()


@pytest.mark.parametrize("field", DASHARRAY_FIELDS)
@pytest.mark.parametrize("value", BAD_DASHARRAYS)
def test_malformed_dash_patterns_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        DrawingStyle(**{field: value})


@pytest.mark.parametrize("field", COLOR_FIELDS + DASHARRAY_FIELDS)
def test_a_value_that_is_not_a_string_is_a_type_error(field):
    with pytest.raises(TypeError, match=field):
        DrawingStyle(**{field: (1.0, 0.0, 0.0)})


def test_an_assignment_is_checked_too_and_leaves_the_style_as_it_was():
    style = DrawingStyle(node_color="crimson")

    with pytest.raises(ValueError, match="node_color"):
        style.node_color = "notacolor"

    assert style.node_color == "crimson"


def test_other_fields_are_assigned_as_before():
    style = DrawingStyle()
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
        DrawingStyle(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [("node_color", "red"), ("basepair_dasharray", "2,1"), ("font_family", "Arial")],
)
def test_a_string_is_kept_as_a_str(field, value):
    """Rather than as the subclass of str it was given as."""
    kept = getattr(DrawingStyle(**{field: np.str_(value)}), field)

    assert kept == value and type(kept) is str


@pytest.mark.parametrize("value", [10**400, -(10**400)], ids=["positive", "negative"])
def test_a_size_too_large_for_a_float_is_refused_as_not_finite(value):
    """float() raises OverflowError for it, which is not what is promised."""
    with pytest.raises(ValueError, match="node_radius must be a finite number"):
        DrawingStyle(node_radius=value)


@pytest.mark.parametrize("field", SIZE_FIELDS)
@pytest.mark.parametrize("value", ["2", None, True])
def test_a_size_that_is_not_a_number_is_a_type_error(field, value):
    with pytest.raises(TypeError, match=field):
        DrawingStyle(**{field: value})


@pytest.mark.parametrize("field", SIZE_FIELDS)
def test_a_size_of_zero_is_drawn(field):
    """Zero draws nothing of that part, which is a choice and not a mistake."""
    style = DrawingStyle(**{field: 0})

    draw_svg("(((...)))", sequences=["GCGAAACGC"], style=style).to_svg()


def test_a_colormap_given_by_name_is_answered_with_how_to_give_it():
    """matplotlib's own functions take a name, so this is the likely slip."""
    with pytest.raises(TypeError, match=r"mpl\.colormaps\['magma'\]"):
        DrawingStyle(colormap="magma")


@pytest.mark.parametrize("value", [None, ["white", "red"]])
def test_a_colormap_that_is_not_one_is_a_type_error(value):
    with pytest.raises(TypeError, match=r"colormap.*mpl\.colormaps\['turbo'\]"):
        DrawingStyle(colormap=value)


def test_any_matplotlib_colormap_is_accepted():
    colormap = mpl.colors.LinearSegmentedColormap.from_list("mine", ["white", "red"])

    assert DrawingStyle(colormap=colormap).colormap is colormap


@pytest.mark.parametrize(
    "value, error",
    [(3, TypeError), (None, TypeError), ("", ValueError), ("  ", ValueError),
     ("Arial, sans-serif", ValueError), ("Arial\n", ValueError)],
)
def test_the_font_family_is_one_family_name(value, error):
    with pytest.raises(error, match="font_family"):
        DrawingStyle(font_family=value)


def test_a_list_of_families_is_answered_with_the_first_of_them():
    with pytest.raises(ValueError, match="'Arial'"):
        DrawingStyle(font_family="Arial, sans-serif")


def test_a_misspelt_attribute_is_refused_and_the_right_one_named():
    """Assigned, it would otherwise make an attribute that nothing reads."""
    style = DrawingStyle()

    with pytest.raises(AttributeError, match="Did you mean: 'node_color'"):
        style.node_colour = "crimson"

    assert not hasattr(style, "node_colour")
    assert style.node_color == "black"


def test_every_field_is_checked_when_it_is_assigned():
    """A field left out of the checks would be refused as a misspelling."""
    assert set(_FIELD_CHECKS) == {f.name for f in dataclasses.fields(DrawingStyle)}


def test_styles_are_equal_when_their_fields_are():
    assert DrawingStyle() == DrawingStyle()
    assert DrawingStyle(node_radius=5) == DrawingStyle(node_radius=5.0)
    assert DrawingStyle() != DrawingStyle(node_radius=5.0)
    assert DrawingStyle() != DrawingStyle(colormap=mpl.colormaps["viridis"])


def test_a_color_is_compared_as_it_is_written():
    assert DrawingStyle() != DrawingStyle(node_color="#000000")


def test_a_style_can_still_be_copied_with_a_field_changed():
    style = DrawingStyle(node_color="crimson")

    changed = dataclasses.replace(style, node_radius=6.0)

    assert (changed.node_color, changed.node_radius) == ("crimson", 6.0)
