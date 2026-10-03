"""What the drawing functions accept, and what they refuse where it is given."""

import xml.etree.ElementTree as ET

import matplotlib as mpl
import numpy as np
import pytest

from nuc2d import (
    RadialLayoutEngine,
    StructureStyle,
    draw_svg,
    draw_svg_as_component,
)


HAIRPIN = "(((...)))"


@pytest.mark.parametrize("structure", [None, 3, list(HAIRPIN)])
def test_the_structure_must_be_a_string(structure):
    with pytest.raises(TypeError, match="structure must be a string"):
        draw_svg_as_component(structure)


def test_one_sequence_not_in_a_list_is_refused():
    with pytest.raises(TypeError, match="sequences must be a list"):
        draw_svg(HAIRPIN, sequences="GCGAAACGC")


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"style": {"node_color": "crimson"}}, "style"),
        ({"style": RadialLayoutEngine()}, "style"),
        ({"layout_engine": StructureStyle()}, "layout_engine"),
        ({"title": 3}, "title"),
        ({"colorbar_label": 3}, "colorbar_label"),
    ],
)
def test_an_argument_of_another_type_is_refused(kwargs, name):
    with pytest.raises(TypeError, match=name):
        draw_svg_as_component(HAIRPIN, **kwargs)


@pytest.mark.parametrize("name", ["title", "colorbar_label"])
@pytest.mark.parametrize("text", ["", "two\nlines", "a\tb"])
def test_a_title_or_a_colorbar_label_is_one_line_or_none(name, text):
    """None, not an empty string, is how either is left out."""
    with pytest.raises(ValueError, match=name):
        draw_svg_as_component(HAIRPIN, **{name: text})


def test_a_probability_outside_zero_to_one_takes_the_color_of_the_end():
    """Whatever colors the colormap itself keeps for values out of range."""
    colormap = mpl.colormaps["turbo"].with_extremes(over="magenta", under="cyan")
    # A pair, and the three unpaired nucleotides of the hairpin loop.
    inside, outside = np.zeros((9, 9)), np.zeros((9, 9))
    inside[0, 8], outside[0, 8] = 1.0, 2.0
    inside[3, 3], outside[3, 3] = 0.0, -0.5
    inside[4, 4], outside[4, 4] = 1.0, 1.5
    inside[5, 5], outside[5, 5] = 1.0, np.inf

    assert (
        draw_svg(HAIRPIN, basepair_probabilities=outside, colormap=colormap).to_svg()
        == draw_svg(HAIRPIN, basepair_probabilities=inside, colormap=colormap).to_svg()
    )


def test_the_colormap_colors_the_nucleotides_and_the_colorbar():
    colormap = mpl.colors.LinearSegmentedColormap.from_list("mine", ["white", "red"])

    svg = draw_svg(
        HAIRPIN, basepair_probabilities=np.eye(9) * 0.5, colormap=colormap
    ).to_svg()

    root = ET.fromstring(svg)
    fills = {e.attrib["fill"] for e in root.iter() if e.tag.endswith("circle")}
    stops = [e.attrib["stop-color"] for e in root.iter() if e.tag.endswith("stop")]
    # The paired nucleotides are colored by 0, the unpaired ones by 0.5.
    assert fills == {mpl.colors.to_hex(colormap(0.0)), mpl.colors.to_hex(colormap(0.5))}
    assert (stops[0], stops[-1]) == ("#ffffff", "#ff0000")


def test_a_colormap_without_probabilities_changes_nothing():
    viridis = mpl.colormaps["viridis"]

    assert draw_svg(HAIRPIN, colormap=viridis).to_svg() == draw_svg(HAIRPIN).to_svg()


@pytest.mark.parametrize("value", ["magma", 3, ["white", "red"]])
def test_a_colormap_that_is_not_one_is_a_type_error(value):
    """A name too, though matplotlib's own functions take one."""
    with pytest.raises(TypeError, match=r"colormap.*mpl\.colormaps\['turbo'\]"):
        draw_svg_as_component(HAIRPIN, colormap=value)


def test_a_nan_probability_is_refused():
    probs = np.eye(9)
    probs[4, 4] = np.nan

    with pytest.raises(ValueError, match=r"basepair_probabilities\[4\]\[4\]"):
        draw_svg(HAIRPIN, basepair_probabilities=probs)


@pytest.mark.parametrize("name", ["title", "colorbar_label"])
@pytest.mark.parametrize("text", [" ", "   ", "\u3000", "\u00a0"])
def test_a_blank_title_or_colorbar_label_is_refused(name, text):
    """It would show nothing, and still take the room of a line."""
    with pytest.raises(ValueError, match=f"{name} is blank; pass None"):
        draw_svg_as_component(HAIRPIN, **{name: text})


@pytest.mark.parametrize("name", ["title", "colorbar_label"])
def test_an_empty_title_or_colorbar_label_is_answered_with_none(name):
    with pytest.raises(ValueError, match="pass None"):
        draw_svg_as_component(HAIRPIN, **{name: ""})


@pytest.mark.parametrize(
    "as_given",
    [lambda m: m, lambda m: m.tolist(), lambda m: tuple(map(tuple, m))],
)
def test_probabilities_are_anything_numpy_makes_an_array_of(as_given):
    """An array, a list of lists or a tuple of tuples draws the same."""
    probs = np.eye(9) * 0.4 + 0.3

    assert (
        draw_svg(HAIRPIN, basepair_probabilities=as_given(probs)).to_svg()
        == draw_svg(HAIRPIN, basepair_probabilities=probs).to_svg()
    )
