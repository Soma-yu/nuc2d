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
    ["(((...)))", "(((..+...)))", ".....", "((..((...))..))"],
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


def collect_texts(svg_string):
    """Return the text content of every text element, in order."""
    return [
        element.text
        for element in ET.fromstring(svg_string).iter()
        if element.tag.endswith("text")
    ]


def test_colorbar_carries_a_default_label():
    svg = draw_svg("(((...)))", probs=PROBS).tostring()

    assert "Base-pair probability" in collect_texts(svg)


def test_colorbar_label_is_configurable():
    svg = draw_svg(
        "(((...)))", probs=PROBS, colorbar_label="Unpaired probability"
    ).tostring()

    texts = collect_texts(svg)

    assert "Unpaired probability" in texts
    assert "Base-pair probability" not in texts


def test_colorbar_label_is_ignored_without_probabilities():
    with_label = draw_svg("(((...)))", colorbar_label="Unpaired probability")

    assert with_label.tostring() == draw_svg("(((...)))").tostring()


def test_the_colorbar_can_be_left_out():
    """probs colors the nucleotides; the colorbar beside them is optional."""
    drawing = svgwrite.Drawing()

    with_bar = draw_component(drawing, "(((...)))", probs=PROBS)
    without_bar = draw_component(svgwrite.Drawing(), "(((...)))", probs=PROBS,
                                 add_colorbar=False)
    plain = draw_component(svgwrite.Drawing(), "(((...)))")

    # The structure itself is unchanged; only the colorbar beside it is gone.
    assert without_bar.bbox.width == pytest.approx(plain.bbox.width)
    assert without_bar.bbox.width < with_bar.bbox.width
    assert "Base-pair probability" not in collect_texts(
        draw_svg("(((...)))", probs=PROBS, add_colorbar=False).tostring()
    )


def test_a_colorbar_can_be_placed_at_a_size_of_its_own():
    """The point of leaving it out: size it against something else."""
    from nuc2d import Placement, compose, render_colorbar

    drawing = svgwrite.Drawing()
    structure = draw_component(drawing, "(((...)))", probs=PROBS, add_colorbar=False)
    colorbar = render_colorbar(drawing)

    panel = compose(drawing.g(), [
        Placement(component=structure, x=0.0, y=0.0, scale=1.0),
        Placement(component=colorbar, x=structure.bbox.xmax, y=0.0, scale=0.5),
    ])

    assert panel.bbox.width == pytest.approx(
        structure.bbox.width + colorbar.bbox.width * 0.5
    )


def test_a_size_is_read_from_a_bounding_box():
    """There is one way to ask how big something is, and it is the box.

    A component and a placement each report where they are; how wide and
    how tall follows from that. Repeating the two on the objects gave the
    same numbers a second spelling, which callers then mixed.
    """
    from nuc2d import Placement

    component = draw_component(svgwrite.Drawing(), "(((...)))")
    placement = Placement(component=component, x=3.0, y=4.0, scale=2.0)

    for obj in (component, placement):
        assert not hasattr(obj, "width")
        assert not hasattr(obj, "height")

    assert placement.bbox.width == pytest.approx(component.bbox.width * 2.0)
    assert placement.bbox.height == pytest.approx(component.bbox.height * 2.0)


def test_layout_engine_is_configurable():
    from nuc2d.layout import RadialLayoutEngine

    default = draw_component(svgwrite.Drawing(), "(((...)))")
    wider = draw_component(
        svgwrite.Drawing(),
        "(((...)))",
        layout_engine=RadialLayoutEngine(backbone_spacing=30),
    )

    assert wider.bbox.height > default.bbox.height


def test_arrowhead_marker_carries_its_own_coordinate_system():
    """Pin the marker's viewBox, which decides the arrowhead's size.

    A marker without a viewBox leaves renderers to guess how its content
    maps into the marker viewport: browsers draw it unscaled, while
    cairosvg stretches it to fill the viewport. Declaring a viewBox as
    large as the viewport settles it at a scale of one.
    """
    svg = draw_svg("(((...)))").tostring()

    marker = ET.fromstring(svg).find(".//{*}marker")

    assert marker is not None, "expected an arrowhead marker definition"
    assert "viewBox" in marker.attrib, "the marker must declare a viewBox"

    *_, vb_width, vb_height = [
        float(value) for value in marker.attrib["viewBox"].replace(",", " ").split()
    ]

    assert float(marker.attrib["markerWidth"]) == vb_width
    assert float(marker.attrib["markerHeight"]) == vb_height


def test_output_is_reproducible():
    first = draw_svg("(((...)))", probs=PROBS).tostring()
    second = draw_svg("(((...)))", probs=PROBS).tostring()

    assert first == second
