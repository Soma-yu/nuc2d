import pytest

from nuc2d import DrawingStyle, draw_svg


COLOR_FIELDS = ["node_color", "backbone_color", "basepair_color"]
DASHARRAY_FIELDS = ["backbone_dasharray", "basepair_dasharray"]

GOOD_COLORS = [
    "crimson",
    "#333",
    "#A1b2C3",
    "rgb(10, 20, 30)",
    "rgb(10%,20.5%, 30%)",
    "none",
]
BAD_COLORS = [
    "notacolor",
    "tab:blue",       # matplotlib's spelling, not SVG's
    "C0",
    "rebeccapurple",  # CSS added it after SVG 1.1
    "#12",
    "#12345",
    "rgb(1, 2)",
    "",
    " crimson",
]
GOOD_DASHARRAYS = ["none", "1,0", "4 2", "0.5,1.5", ".5", "3"]
BAD_DASHARRAYS = ["abc", "-1,2", "4,,2", "", "5px,3px", "4,2,"]


def test_the_defaults_are_valid():
    DrawingStyle()


@pytest.mark.parametrize("field", COLOR_FIELDS)
@pytest.mark.parametrize("value", GOOD_COLORS)
def test_every_svg_way_of_writing_a_color_is_accepted_and_drawn(field, value):
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
