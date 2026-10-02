"""What the drawing functions accept, and what they refuse where it is given."""

import matplotlib as mpl
import numpy as np
import pytest

from nuc2d import (
    RadialLayoutEngine,
    StructureStyle,
    draw_colorbar,
    draw_structure,
    draw_svg,
)


HAIRPIN = "(((...)))"


@pytest.mark.parametrize("structure", [None, 3, list(HAIRPIN)])
def test_the_structure_must_be_a_string(structure):
    with pytest.raises(TypeError, match="structure must be a string"):
        draw_structure(structure)


def test_one_sequence_not_in_a_list_is_refused():
    with pytest.raises(TypeError, match="sequences must be a list"):
        draw_svg(HAIRPIN, sequences="GCGAAACGC")


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"style": {"node_color": "crimson"}}, "style"),
        ({"style": RadialLayoutEngine()}, "style"),
        ({"layout_engine": StructureStyle()}, "layout_engine"),
        ({"add_colorbar": "no"}, "add_colorbar"),
        ({"add_colorbar": None}, "add_colorbar"),
        ({"colorbar_label": 3}, "colorbar_label"),
    ],
)
def test_an_argument_of_another_type_is_refused(kwargs, name):
    with pytest.raises(TypeError, match=name):
        draw_structure(HAIRPIN, **kwargs)


@pytest.mark.parametrize("label", ["", "two\nlines", "a\tb"])
def test_a_colorbar_label_is_one_line_or_none(label):
    """None, not an empty string, is how a colorbar is left without one."""
    with pytest.raises(ValueError, match="colorbar_label"):
        draw_structure(HAIRPIN, colorbar_label=label)
    with pytest.raises(ValueError, match="label"):
        draw_colorbar(label=label)


def test_the_colorbar_takes_a_style_and_nothing_else():
    with pytest.raises(TypeError, match="style"):
        draw_colorbar(style="turbo")


def test_a_probability_outside_zero_to_one_takes_the_color_of_the_end():
    """Whatever colors the colormap itself keeps for values out of range."""
    style = StructureStyle(
        colormap=mpl.colormaps["turbo"].with_extremes(over="magenta", under="cyan")
    )
    # A pair, and the three unpaired nucleotides of the hairpin loop.
    inside, outside = np.zeros((9, 9)), np.zeros((9, 9))
    inside[0, 8], outside[0, 8] = 1.0, 2.0
    inside[3, 3], outside[3, 3] = 0.0, -0.5
    inside[4, 4], outside[4, 4] = 1.0, 1.5
    inside[5, 5], outside[5, 5] = 1.0, np.inf

    assert (
        draw_svg(HAIRPIN, basepair_probabilities=outside, style=style).to_svg()
        == draw_svg(HAIRPIN, basepair_probabilities=inside, style=style).to_svg()
    )


def test_a_nan_probability_is_refused():
    probs = np.eye(9)
    probs[4, 4] = np.nan

    with pytest.raises(ValueError, match=r"basepair_probabilities\[4\]\[4\]"):
        draw_svg(HAIRPIN, basepair_probabilities=probs)


def test_an_empty_colorbar_label_is_answered_with_none():
    with pytest.raises(ValueError, match="pass None"):
        draw_colorbar(label="")


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
