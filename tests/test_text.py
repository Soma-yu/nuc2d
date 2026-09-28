import xml.etree.ElementTree as ET

import pytest

from nuc2d import Component, Placement, Scene, draw_structure, draw_text
from nuc2d._font import find_font_path, vertical_extent


def texts(svg_string):
    return [
        element for element in ET.fromstring(svg_string).iter()
        if element.tag.endswith("text")
    ]


def test_the_text_is_written_as_given():
    [element] = texts(Scene(draw_text("tRNA cloverleaf")).to_svg())

    assert element.text == "tRNA cloverleaf"
    assert element.attrib["font-family"] == "Arial"
    assert float(element.attrib["font-size"]) == 12.0


def test_markup_in_the_text_is_written_as_text():
    [element] = texts(Scene(draw_text("a < b & c")).to_svg())

    assert element.text == "a < b & c"


def test_the_box_runs_from_the_ascender_to_the_descender():
    above, below = vertical_extent(find_font_path("Arial"), 20.0)

    bbox = draw_text("Wg", font_size=20.0).bbox

    assert (bbox.xmin, bbox.ymin) == (0.0, 0.0)
    assert bbox.height == pytest.approx(above + below)


def test_the_box_grows_with_the_text_and_the_size():
    short = draw_text("tRNA").bbox
    longer = draw_text("tRNA cloverleaf").bbox
    larger = draw_text("tRNA", font_size=24.0).bbox

    assert longer.width > short.width
    assert larger.width == pytest.approx(short.width * 2)
    assert larger.height == pytest.approx(short.height * 2)


def test_a_title_can_be_centred_over_a_structure():
    title = Placement(component=draw_text("tRNA", font_size=15.0), anchor="upper center")
    structure = Placement(
        component=draw_structure("(((...)))"),
        y=title.bbox.ymax + 10.0,
        anchor="upper center",
    )

    panel = Component.from_placements([title, structure])

    assert title.bbox.center_x == pytest.approx(structure.bbox.center_x)
    assert panel.bbox.height == pytest.approx(
        title.bbox.height + 10.0 + structure.bbox.height
    )


@pytest.mark.parametrize("text", ["", "two\nlines", "two\rlines"])
def test_text_that_is_not_one_line_is_refused(text):
    with pytest.raises(ValueError):
        draw_text(text)


@pytest.mark.parametrize("font_size", [0.0, -1.0, float("inf"), float("nan")])
def test_the_font_size_must_be_positive_and_finite(font_size):
    with pytest.raises(ValueError, match="font_size"):
        draw_text("tRNA", font_size=font_size)


@pytest.mark.parametrize("kwargs", [{"text": 3}, {"text": "tRNA", "font_family": None}])
def test_text_and_family_must_be_strings(kwargs):
    text = kwargs.pop("text")

    with pytest.raises(TypeError):
        draw_text(text, **kwargs)
