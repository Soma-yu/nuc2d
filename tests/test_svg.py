import re
import xml.etree.ElementTree as ET
from collections import Counter

import matplotlib as mpl
import numpy as np
import pytest
import svgwrite

from nuc2d import draw_component, draw_svg
from nuc2d.style import DrawingStyle


def collect_ids(svg_string):
    """Return every id attribute in the document, in order."""
    return [
        element.attrib["id"]
        for element in ET.fromstring(svg_string).iter()
        if "id" in element.attrib
    ]


def collect_references(svg_string):
    """Return every id referenced through url(#...)."""
    return re.findall(r"url\(#([^)]+)\)", svg_string)


PROBS = np.eye(9) * 0.4 + 0.3


def test_ids_are_unique_within_one_drawing():
    svg = draw_svg("(((...)))", probs=PROBS).tostring()

    ids = collect_ids(svg)

    assert ids, "expected the drawing to define at least one element"
    assert [i for i, n in Counter(ids).items() if n > 1] == []


def test_ids_stay_unique_across_several_groups_in_one_drawing():
    drawing = svgwrite.Drawing()

    for _ in range(3):
        component = draw_component(drawing, "(((...)))", probs=PROBS)
        drawing.add(component.group)

    ids = collect_ids(drawing.tostring())

    assert [i for i, n in Counter(ids).items() if n > 1] == []


def test_differing_styles_get_their_own_definitions():
    drawing = svgwrite.Drawing()

    for style in [
        DrawingStyle(edge_color="black", cmap=mpl.colormaps["turbo"]),
        DrawingStyle(edge_color="red", cmap=mpl.colormaps["viridis"]),
    ]:
        component = draw_component(drawing, "(((...)))", probs=PROBS, style=style)
        drawing.add(component.group)

    svg = drawing.tostring()
    ids = collect_ids(svg)

    assert [i for i, n in Counter(ids).items() if n > 1] == []
    assert len([i for i in ids if i.startswith("arrowhead-")]) == 2
    assert len([i for i in ids if i.startswith("colorbar-gradient-")]) == 2


def test_identical_styles_share_one_definition():
    drawing = svgwrite.Drawing()

    for _ in range(3):
        component = draw_component(drawing, "(((...)))", probs=PROBS)
        drawing.add(component.group)

    ids = collect_ids(drawing.tostring())

    assert len([i for i in ids if i.startswith("arrowhead-")]) == 1
    assert len([i for i in ids if i.startswith("colorbar-gradient-")]) == 1


def test_every_reference_resolves():
    drawing = svgwrite.Drawing()

    for style in [
        DrawingStyle(edge_color="black"),
        DrawingStyle(edge_color="red"),
    ]:
        component = draw_component(drawing, "(((...)))", probs=PROBS, style=style)
        drawing.add(component.group)

    svg = drawing.tostring()

    assert set(collect_references(svg)) <= set(collect_ids(svg))


@pytest.mark.parametrize(
    "dpp_string",
    ["(((...)))", "(((..+...)))", ".....", "((..((..))..))"],
)
def test_output_is_well_formed_xml(dpp_string):
    ET.fromstring(draw_svg(dpp_string).tostring())


def test_viewbox_frames_exactly_the_component():
    drawing = svgwrite.Drawing()
    component = draw_component(drawing, "..(((...)))..", probs=None)

    svg = draw_svg("..(((...)))..").tostring()
    viewbox = ET.fromstring(svg).attrib["viewBox"]

    assert [float(v) for v in viewbox.replace(",", " ").split()] == list(
        component.bbox.to_viewbox()
    )


def test_colorbar_sits_beside_the_structure():
    drawing = svgwrite.Drawing()

    without = draw_component(drawing, "(((...)))")
    with_bar = draw_component(svgwrite.Drawing(), "(((...)))", probs=PROBS)

    assert with_bar.bbox.width > without.bbox.width
    assert with_bar.bbox.height == pytest.approx(without.bbox.height)


def test_layout_engine_is_configurable():
    from nuc2d.layout import RadialLayoutEngine

    default = draw_component(svgwrite.Drawing(), "(((...)))")
    wider = draw_component(
        svgwrite.Drawing(),
        "(((...)))",
        layout_engine=RadialLayoutEngine(backbone_spacing=30),
    )

    assert wider.bbox.height > default.bbox.height


def test_output_is_reproducible():
    first = draw_svg("(((...)))", probs=PROBS).tostring()
    second = draw_svg("(((...)))", probs=PROBS).tostring()

    assert first == second
