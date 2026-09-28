import math
import re
import xml.etree.ElementTree as ET
from collections import Counter

import matplotlib as mpl
import numpy as np
import pytest
import svgwrite

from nuc2d import (
    DrawingStyle,
    Placement,
    RadialLayoutEngine,
    Scene,
    compose,
    draw_colorbar,
    draw_structure,
    draw_svg,
)


def side_by_side(components):
    """Place components in a row, each to the right of the last."""
    placements, cursor_x = [], 0.0
    for component in components:
        placement = Placement(component=component, x=cursor_x)
        placements.append(placement)
        cursor_x = placement.bbox.xmax
    return Scene(compose(placements))


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
    svg = draw_svg("(((...)))", probs=PROBS).to_svg()

    ids = collect_ids(svg)

    assert ids, "expected the drawing to define at least one element"
    assert [i for i, n in Counter(ids).items() if n > 1] == []


def test_ids_stay_unique_across_several_components_in_one_scene():
    scene = side_by_side([draw_structure("(((...)))", probs=PROBS) for _ in range(3)])

    ids = collect_ids(scene.to_svg())

    assert [i for i, n in Counter(ids).items() if n > 1] == []


def test_differing_styles_get_their_own_definitions():
    scene = side_by_side([
        draw_structure("(((...)))", probs=PROBS, style=style)
        for style in [
            DrawingStyle(backbone_color="black", cmap=mpl.colormaps["turbo"]),
            DrawingStyle(backbone_color="red", cmap=mpl.colormaps["viridis"]),
        ]
    ])

    ids = collect_ids(scene.to_svg())

    assert [i for i, n in Counter(ids).items() if n > 1] == []
    assert len([i for i in ids if i.startswith("arrowhead-")]) == 2
    assert len([i for i in ids if i.startswith("colorbar-gradient-")]) == 2


def test_identical_styles_share_one_definition():
    scene = side_by_side([draw_structure("(((...)))", probs=PROBS) for _ in range(3)])

    ids = collect_ids(scene.to_svg())

    assert len([i for i in ids if i.startswith("arrowhead-")]) == 1
    assert len([i for i in ids if i.startswith("colorbar-gradient-")]) == 1


def test_every_reference_resolves():
    scene = side_by_side([
        draw_structure("(((...)))", probs=PROBS, style=DrawingStyle(backbone_color=color))
        for color in ["black", "red"]
    ])

    svg = scene.to_svg()

    assert set(collect_references(svg)) <= set(collect_ids(svg))


@pytest.mark.parametrize(
    "dot_bracket",
    ["(((...)))", "(((..+...)))", ".....", "((..((...))..))"],
)
def test_output_is_well_formed_xml(dot_bracket):
    ET.fromstring(draw_svg(dot_bracket).to_svg())


def test_viewbox_frames_exactly_the_component():
    component = draw_structure("..(((...)))..")

    svg = draw_svg("..(((...)))..").to_svg()
    viewbox = ET.fromstring(svg).attrib["viewBox"]

    assert [float(v) for v in viewbox.replace(",", " ").split()] == list(
        component.bbox.to_viewbox()
    )


def test_the_colorbar_sits_beside_the_structure_at_its_own_size():
    """The structure is fitted to the colorbar, not the colorbar to it.

    A colorbar scaled to the structure shrank to a sliver beside a wide
    one. Fitted into a square as tall as the colorbar, a structure of any
    shape leaves the colorbar its own size.
    """
    colorbar = draw_colorbar().bbox

    for dot_bracket in ["(((...)))", "." * 40, "((((....))))" * 3]:
        with_bar = draw_structure(
            dot_bracket, probs=np.eye(len(dot_bracket)) * 0.5
        ).bbox

        assert with_bar.height == pytest.approx(colorbar.height)
        assert with_bar.width == pytest.approx(colorbar.height + colorbar.width)


def collect_texts(svg_string):
    """Return the text content of every text element, in order."""
    return [
        element.text
        for element in ET.fromstring(svg_string).iter()
        if element.tag.endswith("text")
    ]


def test_colorbar_carries_a_default_label():
    svg = draw_svg("(((...)))", probs=PROBS).to_svg()

    assert "Equilibrium probability" in collect_texts(svg)


def test_colorbar_label_is_configurable():
    svg = draw_svg(
        "(((...)))", probs=PROBS, colorbar_label="Unpaired probability"
    ).to_svg()

    texts = collect_texts(svg)

    assert "Unpaired probability" in texts
    assert "Equilibrium probability" not in texts


def test_colorbar_label_none_leaves_the_label_out():
    labelled = draw_svg("(((...)))", probs=PROBS).to_svg()
    unlabelled = draw_svg("(((...)))", probs=PROBS, colorbar_label=None).to_svg()

    assert "Equilibrium probability" not in collect_texts(unlabelled)
    assert len(collect_texts(unlabelled)) == len(collect_texts(labelled)) - 1


def test_colorbar_keeps_its_box_without_a_label():
    """So colorbars with and without a label line up when placed."""
    assert draw_colorbar(label=None).bbox == draw_colorbar().bbox


def test_colorbar_label_is_ignored_without_probabilities():
    with_label = draw_svg("(((...)))", colorbar_label="Unpaired probability")

    assert with_label.to_svg() == draw_svg("(((...)))").to_svg()


def test_the_colorbar_can_be_left_out():
    """probs colors the nucleotides; the colorbar beside them is optional."""
    with_bar = draw_structure("(((...)))", probs=PROBS)
    without_bar = draw_structure("(((...)))", probs=PROBS, add_colorbar=False)
    plain = draw_structure("(((...)))")

    # The structure itself is unchanged; only the colorbar beside it is gone.
    assert without_bar.bbox == plain.bbox
    assert without_bar.bbox.width < with_bar.bbox.width
    assert "Equilibrium probability" not in collect_texts(
        draw_svg("(((...)))", probs=PROBS, add_colorbar=False).to_svg()
    )


def test_a_colorbar_can_be_placed_at_a_size_of_its_own():
    """The point of leaving it out: size it against something else."""
    structure = draw_structure("(((...)))", probs=PROBS, add_colorbar=False)
    colorbar = draw_colorbar()

    placed = Placement(component=structure)
    panel = compose([
        placed,
        Placement(
            component=colorbar,
            x=placed.bbox.xmax,
            y=placed.bbox.center_y,
            anchor="center left",
            scale=0.5,
        ),
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
    component = draw_structure("(((...)))")
    placement = Placement(component=component, x=3.0, y=4.0, scale=2.0)
    scene = Scene(component)

    for obj in (component, placement, scene):
        assert not hasattr(obj, "width")
        assert not hasattr(obj, "height")

    assert placement.bbox.width == pytest.approx(component.bbox.width * 2.0)
    assert placement.bbox.height == pytest.approx(component.bbox.height * 2.0)


def test_the_sequence_reaches_the_drawing():
    """Drawing the bases is what sequences= is for, end to end.

    Each base is written twice, once outlined and once filled, so that the
    letter stays readable over any node color.
    """
    svg = draw_svg("(((..+...)))", sequences=["AUGCA", "UGCCAU"]).to_svg()

    letters = [text for text in collect_texts(svg) if text in set("ACGU")]

    assert letters == [base for base in "AUGCAUGCCAU" for _ in range(2)]

    drawn = [
        element
        for element in ET.fromstring(svg).iter()
        if element.tag.endswith("text") and element.text == "A"
    ]
    assert {element.attrib["fill"] for element in drawn} == {"black", "white"}
    assert {element.attrib["font-family"] for element in drawn} == {"Arial"}


def test_layout_engine_is_configurable():
    default = draw_structure("(((...)))")
    wider = draw_structure(
        "(((...)))",
        layout_engine=RadialLayoutEngine(backbone_spacing=30),
    )

    assert wider.bbox.height > default.bbox.height


def collect_arrow_lengths(svg_string):
    """Return the length of every line carrying an arrowhead, in order."""
    return [
        math.hypot(
            float(element.attrib["x2"]) - float(element.attrib["x1"]),
            float(element.attrib["y2"]) - float(element.attrib["y1"]),
        )
        for element in ET.fromstring(svg_string).iter()
        if element.tag.endswith("line") and "marker-end" in element.attrib
    ]


def test_the_three_prime_arrow_length_is_a_style_setting():
    """How far the 3' arrow reaches is drawn, not laid out.

    The layout says where a terminus is and which way the strand runs.
    How long an arrow to draw from there is a matter of appearance, so it
    lives with the other style settings rather than on the decoration.
    """
    assert collect_arrow_lengths(
        draw_svg("(((..+...)))").to_svg()
    ) == pytest.approx([7.0, 7.0])

    assert collect_arrow_lengths(
        draw_svg(
            "(((..+...)))", style=DrawingStyle(three_prime_arrow_length=15.0)
        ).to_svg()
    ) == pytest.approx([15.0, 15.0])


def test_something_the_renderer_cannot_draw_says_so():
    """Not knowing how to draw a thing is reported, not passed over.

    A drawing that quietly leaves a piece out looks finished, so a
    caller extending the layout has nothing to go on. Both dispatches
    answer the same way.
    """
    from dataclasses import dataclass

    from nuc2d._layout import Decoration, Edge, EdgeType
    from nuc2d._parser import parse
    from nuc2d._svg import SVGRenderer

    renderer = SVGRenderer()
    drawing = svgwrite.Drawing()
    node = RadialLayoutEngine().layout(parse("(((...)))")).nodes[0]

    @dataclass(kw_only=True)
    class Squiggle(Decoration):
        pass

    with pytest.raises(TypeError, match="Unsupported decoration type"):
        renderer._draw_decoration(drawing, Squiggle(node=node))

    with pytest.raises(TypeError, match="Unsupported edge type"):
        renderer._draw_edge(
            drawing, Edge(start=node, end=node, edge_type=EdgeType.BACKBONE)
        )


def test_the_dash_pattern_reaches_the_whole_backbone():
    """A backbone runs as straight segments in stems and as arcs in loops.

    Both are backbone edges, so both take backbone_dasharray. Leaving the
    arcs out left most of the backbone solid while the stems went dashed.
    """
    root = ET.fromstring(
        draw_svg(
            "((..((...))..))",
            style=DrawingStyle(backbone_dasharray="6,3", basepair_dasharray="1,4"),
        ).to_svg()
    )

    dashes = Counter()
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        if tag not in ("line", "path") or "stroke" not in element.attrib:
            continue
        # The 3' arrow is a terminus marker rather than a stretch of
        # backbone, so it stays solid whatever the backbone is dashed with.
        kind = (
            "arrow" if "marker-end" in element.attrib
            else "arc" if tag == "path"
            else "line"
        )
        dashes[(kind, element.attrib.get("stroke-dasharray"))] += 1

    assert dashes[("arc", "6,3")] > 0, "the loop arcs must follow the backbone"
    assert dashes[("line", "6,3")] > 0
    assert dashes[("line", "1,4")] > 0
    assert [key for key in dashes if key[0] != "arrow" and key[1] is None] == []


def test_the_backbone_and_the_base_pairs_take_their_own_colors():
    """Color joins the width and the dash pattern in being per edge type.

    The arrow at a 3' terminus continues the backbone, so it and its
    arrowhead follow the backbone color rather than the base-pair one.
    """
    root = ET.fromstring(
        draw_svg(
            "(((...)))",
            style=DrawingStyle(backbone_color="crimson", basepair_color="steelblue"),
        ).to_svg()
    )

    strokes = Counter(
        element.attrib["stroke"] for element in root.iter() if "stroke" in element.attrib
    )

    assert set(strokes) == {"crimson", "steelblue"}
    assert collect_arrow_lengths(
        ET.tostring(root, encoding="unicode")
    ), "expected an arrow to be drawn"

    arrow = next(
        element
        for element in root.iter()
        if element.tag.endswith("line") and "marker-end" in element.attrib
    )
    assert arrow.attrib["stroke"] == "crimson"

    marker = root.find(".//{*}marker")
    assert marker.find("{*}path").attrib["fill"] == "crimson"


def test_arrowhead_marker_carries_its_own_coordinate_system():
    """Pin the marker's viewBox, which decides the arrowhead's size.

    A marker without a viewBox leaves renderers to guess how its content
    maps into the marker viewport: browsers draw it unscaled, while
    cairosvg stretches it to fill the viewport. Declaring a viewBox as
    large as the viewport settles it at a scale of one.
    """
    svg = draw_svg("(((...)))").to_svg()

    marker = ET.fromstring(svg).find(".//{*}marker")

    assert marker is not None, "expected an arrowhead marker definition"
    assert "viewBox" in marker.attrib, "the marker must declare a viewBox"

    *_, vb_width, vb_height = [
        float(value) for value in marker.attrib["viewBox"].replace(",", " ").split()
    ]

    assert float(marker.attrib["markerWidth"]) == vb_width
    assert float(marker.attrib["markerHeight"]) == vb_height


def test_output_is_reproducible():
    first = draw_svg("(((...)))", probs=PROBS).to_svg()
    second = draw_svg("(((...)))", probs=PROBS).to_svg()

    assert first == second
