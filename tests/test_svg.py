import math
import re
import xml.etree.ElementTree as ET
from collections import Counter

import matplotlib as mpl
import numpy as np
import pytest
import svgwrite

from nuc2d import (
    Component,
    Placement,
    RadialLayoutEngine,
    Scene,
    StructureStyle,
    draw_svg,
    draw_svg_as_component,
)
from nuc2d._draw import _TITLE_FONT_SIZE, _TITLE_GAP
from nuc2d._svg import render_colorbar, render_text


def side_by_side(components):
    """Place components in a row, each to the right of the last."""
    placements, cursor_x = [], 0.0
    for component in components:
        placement = Placement(component=component, x=cursor_x)
        placements.append(placement)
        cursor_x = placement.bbox.xmax
    return Scene(Component.from_placements(placements))


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
    svg = draw_svg("(((...)))", basepair_probabilities=PROBS).to_svg()

    ids = collect_ids(svg)

    assert ids, "expected the drawing to define at least one element"
    assert [i for i, n in Counter(ids).items() if n > 1] == []


def test_ids_stay_unique_across_several_components_in_one_scene():
    scene = side_by_side([draw_svg_as_component("(((...)))", basepair_probabilities=PROBS) for _ in range(3)])

    ids = collect_ids(scene.to_svg())

    assert [i for i, n in Counter(ids).items() if n > 1] == []


def test_differing_colormaps_get_their_own_definitions():
    scene = side_by_side([
        draw_svg_as_component(
            "(((...)))",
            basepair_probabilities=PROBS,
            colormap=colormap,
            style=StructureStyle(backbone_color=color),
        )
        for color, colormap in [
            ("black", mpl.colormaps["turbo"]),
            ("red", mpl.colormaps["viridis"]),
        ]
    ])

    ids = collect_ids(scene.to_svg())

    assert [i for i, n in Counter(ids).items() if n > 1] == []
    assert len([i for i in ids if i.startswith("colorbar-gradient-")]) == 2


def test_identical_styles_share_one_definition():
    scene = side_by_side([draw_svg_as_component("(((...)))", basepair_probabilities=PROBS) for _ in range(3)])

    ids = collect_ids(scene.to_svg())

    assert len([i for i in ids if i.startswith("colorbar-gradient-")]) == 1


def test_every_reference_resolves():
    scene = side_by_side([
        draw_svg_as_component("(((...)))", basepair_probabilities=PROBS, style=StructureStyle(backbone_color=color))
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


def node_extent(dot_bracket):
    """Return the box around the centers of a structure's nodes."""
    from nuc2d._layout import layout
    from nuc2d._parse import parse

    nodes = layout(parse(dot_bracket), RadialLayoutEngine()).nodes
    xs = [node.pos.x for node in nodes]
    ys = [node.pos.y for node in nodes]
    return min(xs), min(ys), max(xs), max(ys)


def test_the_box_runs_twenty_units_past_the_outermost_nodes():
    xmin, ymin, xmax, ymax = node_extent("(((...)))")

    bbox = draw_svg_as_component("(((...)))").bbox

    assert (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax) == pytest.approx(
        (xmin - 20.0, ymin - 20.0, xmax + 20.0, ymax + 20.0)
    )


def test_the_box_is_not_widened_by_the_style():
    """The box is not measured from what is drawn, so a style leaves it be."""
    default = draw_svg_as_component("(((...)))").bbox

    for style in [
        StructureStyle(node_radius=30.0),
        StructureStyle(node_font_size=50.0),
        StructureStyle(three_prime_arrow_length=40.0),
    ]:
        assert draw_svg_as_component("(((...)))", style=style).bbox == default


def test_viewbox_frames_exactly_the_component():
    component = draw_svg_as_component("..(((...)))..")

    svg = draw_svg("..(((...)))..").to_svg()
    viewbox = ET.fromstring(svg).attrib["viewBox"]

    bbox = component.bbox
    assert [float(v) for v in viewbox.replace(",", " ").split()] == [
        bbox.xmin, bbox.ymin, bbox.width, bbox.height
    ]


def test_the_colorbar_sits_beside_the_structure_at_its_own_size():
    """The structure is fitted to the colorbar, not the colorbar to it.

    A colorbar scaled to the structure shrank to a sliver beside a wide
    one. Fitted into a square as tall as the colorbar, a structure of any
    shape leaves the colorbar its own size.
    """
    colorbar = render_colorbar().bbox

    for dot_bracket in ["(((...)))", "." * 40, "((((....))))" * 3]:
        with_bar = draw_svg_as_component(
            dot_bracket, basepair_probabilities=np.eye(len(dot_bracket)) * 0.5
        ).bbox

        assert with_bar.height == pytest.approx(colorbar.height)
        assert with_bar.width == pytest.approx(colorbar.height + colorbar.width)


def test_the_structure_is_centered_in_its_square_beside_the_colorbar():
    """A structure wider than it is tall sits midway up the colorbar."""
    colorbar = render_colorbar().bbox
    square_center = (
        colorbar.xmin - colorbar.height / 2, (colorbar.ymin + colorbar.ymax) / 2
    )

    for dot_bracket in ["." * 40, "((((....))))" * 3]:
        alone = draw_svg_as_component(dot_bracket).bbox
        assert alone.width > alone.height
        svg = Scene(
            draw_svg_as_component(dot_bracket, basepair_probabilities=np.eye(len(dot_bracket)) * 0.5)
        ).to_svg()

        # The structure is placed first, so its transform is the first one.
        transform = next(
            element.attrib["transform"]
            for element in ET.fromstring(svg).iter()
            if "transform" in element.attrib
        )
        dx, dy, scale, _ = map(float, re.findall(r"-?[\d.]+(?:e-?\d+)?", transform))

        center_x = (alone.xmin + alone.xmax) / 2
        center_y = (alone.ymin + alone.ymax) / 2
        assert (dx + scale * center_x, dy + scale * center_y) == (
            pytest.approx(square_center)
        )


def test_under_a_title_a_structure_is_as_large_as_beside_a_colorbar():
    """So that the title, at one size, reads the same over any structure."""
    colorbar = render_colorbar().bbox

    for dot_bracket in ["(((...)))", "." * 40, "(" * 30 + "..." + ")" * 30]:
        bbox = draw_svg_as_component(dot_bracket, title="t").bbox

        # The structure runs down from y = 0, and is wider than the title.
        assert max(bbox.width, bbox.ymax) == pytest.approx(colorbar.height)


def test_a_title_is_set_above_what_it_is_over():
    title = render_text("tRNA", font_size=_TITLE_FONT_SIZE).bbox

    for kwargs in [{}, {"basepair_probabilities": PROBS}]:
        bbox = draw_svg_as_component("(((...)))", title="tRNA", **kwargs).bbox

        assert bbox.ymin == pytest.approx(-_TITLE_GAP - title.height)


def test_a_title_is_centered_over_the_structure():
    """Over its square beside a colorbar, whose center is the same."""
    wide = "tRNA" * 50
    colorbar = render_colorbar().bbox

    alone = draw_svg_as_component("(((...)))", title="t").bbox
    titled = draw_svg_as_component("(((...)))", title=wide).bbox
    assert titled.width > alone.width
    assert (titled.xmin + titled.xmax) / 2 == pytest.approx(
        (alone.xmin + alone.xmax) / 2
    )

    titled = draw_svg_as_component(
        "(((...)))", basepair_probabilities=PROBS, title=wide
    ).bbox
    assert (titled.xmin + titled.xmax) / 2 == pytest.approx(
        colorbar.xmin - colorbar.height / 2
    )


def test_the_title_is_written_in_the_font_of_every_letter():
    svg = draw_svg("(((...)))", title="tRNA").to_svg()

    [element] = [
        element
        for element in ET.fromstring(svg).iter()
        if element.tag.endswith("text") and element.text == "tRNA"
    ]
    assert element.attrib["font-family"] == "Arial"
    assert float(element.attrib["font-size"]) == _TITLE_FONT_SIZE


def collect_texts(svg_string):
    """Return the text content of every text element, in order."""
    return [
        element.text
        for element in ET.fromstring(svg_string).iter()
        if element.tag.endswith("text")
    ]


def test_colorbar_carries_a_default_label():
    svg = draw_svg("(((...)))", basepair_probabilities=PROBS).to_svg()

    assert "Equilibrium probability" in collect_texts(svg)


def test_colorbar_label_is_configurable():
    svg = draw_svg(
        "(((...)))", basepair_probabilities=PROBS, colorbar_label="Unpaired probability"
    ).to_svg()

    texts = collect_texts(svg)

    assert "Unpaired probability" in texts
    assert "Equilibrium probability" not in texts


def test_colorbar_label_none_leaves_the_label_out():
    labeled = draw_svg("(((...)))", basepair_probabilities=PROBS).to_svg()
    unlabeled = draw_svg("(((...)))", basepair_probabilities=PROBS, colorbar_label=None).to_svg()

    assert "Equilibrium probability" not in collect_texts(unlabeled)
    assert len(collect_texts(unlabeled)) == len(collect_texts(labeled)) - 1


def test_colorbar_keeps_its_box_without_a_label():
    """So structures with and without a label line up when placed."""
    unlabeled = draw_svg_as_component(
        "(((...)))", basepair_probabilities=PROBS, colorbar_label=None
    )

    assert (
        unlabeled.bbox
        == draw_svg_as_component("(((...)))", basepair_probabilities=PROBS).bbox
    )


def test_colorbar_label_is_ignored_without_probabilities():
    with_label = draw_svg("(((...)))", colorbar_label="Unpaired probability")

    assert with_label.to_svg() == draw_svg("(((...)))").to_svg()


def test_a_size_is_read_from_a_bounding_box():
    """There is one way to ask how big something is, and it is the box.

    A component and a placement each report where they are; how wide and
    how tall follows from that. Repeating the two on the objects gave the
    same numbers a second spelling, which callers then mixed.
    """
    component = draw_svg_as_component("(((...)))")
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
    default = draw_svg_as_component("(((...)))")
    wider = draw_svg_as_component(
        "(((...)))",
        layout_engine=RadialLayoutEngine(stem_spacing=30),
    )

    assert wider.bbox.height > default.bbox.height


def tag_of(element):
    return element.tag.split("}")[-1]


def arrows(root):
    """Return the (line, head) of each 3' arrow, in order.

    An arrow is a group holding its line and the polygon of its head.
    """
    found = []
    for group in root.iter():
        kids = list(group)
        if [tag_of(kid) for kid in kids] == ["line", "polygon"]:
            found.append((kids[0], kids[1]))
    return found


def length_of(line):
    return math.hypot(
        float(line.attrib["x2"]) - float(line.attrib["x1"]),
        float(line.attrib["y2"]) - float(line.attrib["y1"]),
    )


def collect_arrow_lengths(svg_string):
    """Return the length of every 3' arrow's line, in order."""
    return [length_of(line) for line, _ in arrows(ET.fromstring(svg_string))]


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
            "(((..+...)))", style=StructureStyle(three_prime_arrow_length=15.0)
        ).to_svg()
    ) == pytest.approx([15.0, 15.0])


def test_something_the_renderer_cannot_draw_says_so():
    """Not knowing how to draw a thing is reported, not passed over.

    A drawing that quietly leaves a piece out looks finished, so a
    caller extending the layout has nothing to go on. Both dispatches
    answer the same way.
    """
    from dataclasses import dataclass

    from nuc2d._layout import Decoration, Edge, EdgeType, layout
    from nuc2d._parse import parse
    from nuc2d._svg import _Renderer

    renderer = _Renderer()
    drawing = svgwrite.Drawing()
    node = layout(parse("(((...)))"), RadialLayoutEngine()).nodes[0]

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
            style=StructureStyle(backbone_dasharray="6,3", basepair_dasharray="1,4"),
        ).to_svg()
    )

    arrow_lines = {id(line) for line, _ in arrows(root)}
    dashes = Counter()
    for element in root.iter():
        tag = tag_of(element)
        if tag not in ("line", "path") or "stroke" not in element.attrib:
            continue
        # The 3' arrow marks a terminus rather than being a stretch of
        # backbone, so it stays solid whatever the backbone is dashed with.
        kind = (
            "arrow" if id(element) in arrow_lines
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
            style=StructureStyle(backbone_color="crimson", basepair_color="steelblue"),
        ).to_svg()
    )

    strokes = Counter(
        element.attrib["stroke"] for element in root.iter() if "stroke" in element.attrib
    )

    assert set(strokes) == {"crimson", "steelblue"}
    assert collect_arrow_lengths(
        ET.tostring(root, encoding="unicode")
    ), "expected an arrow to be drawn"

    [(line, head)] = arrows(root)
    assert line.attrib["stroke"] == "crimson"
    assert head.attrib["fill"] == "crimson"


def test_the_arrowhead_is_a_shape_not_a_marker():
    """Adobe Illustrator mishandles SVG markers.

    With a marker, it dropped the line carrying it, and the backbone and
    base-pair lines beside it vanished once the drawing was scaled a few
    times. The head is drawn as a shape of its own, so no marker is left.
    """
    svg = draw_svg("(((..+...)))").to_svg()
    root = ET.fromstring(svg)

    assert [tag_of(e) for e in root.iter() if tag_of(e) == "marker"] == []
    assert "marker-end" not in svg
    assert len(arrows(root)) == 2


@pytest.mark.parametrize("width", [2.0, 5.0])
def test_the_arrowhead_is_measured_in_stroke_widths(width):
    """The head runs from one stroke width behind the end of its line to
    two past it, and is three wide, as the marker it replaced was."""
    root = ET.fromstring(
        draw_svg("(((...)))", style=StructureStyle(backbone_width=width)).to_svg()
    )
    [(line, head)] = arrows(root)
    x1, y1, x2, y2 = (float(line.attrib[k]) for k in ("x1", "y1", "x2", "y2"))
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    corners = [
        tuple(map(float, pair.split(",")))
        for pair in head.attrib["points"].split()
    ]
    along = [(x - x2) * ux + (y - y2) * uy for x, y in corners]
    across = [-(x - x2) * uy + (y - y2) * ux for x, y in corners]

    assert (min(along), max(along)) == pytest.approx((-width, 2 * width))
    assert max(across) - min(across) == pytest.approx(3 * width)


def test_the_backbone_is_solid_by_default():
    """"none" is SVG's own value for a solid line.

    A solid line used to be written as the dash pattern "1,0", dashes
    with no gaps between them, which draws the same but asks every
    renderer to work out dashes that are not there.
    """
    root = ET.fromstring(draw_svg("((..((...))..))").to_svg())
    arrow_lines = {id(line) for line, _ in arrows(root)}

    patterns = Counter(
        (e.attrib["stroke-width"], e.attrib.get("stroke-dasharray"))
        for e in root.iter()
        if tag_of(e) in ("line", "path")
        and "stroke" in e.attrib
        and id(e) not in arrow_lines
    )

    assert {pattern for width, pattern in patterns if width == "2.0"} == {"none"}
    assert {pattern for width, pattern in patterns if width == "1.5"} == {"1,1"}


def test_the_outline_under_each_letter_has_round_corners():
    """A mitred corner juts out at the sharp apex of an A."""
    root = ET.fromstring(draw_svg("(((...)))", sequences=["AAAAAAAAA"]).to_svg())
    outlines = [
        e for e in root.iter() if tag_of(e) == "text" and "stroke" in e.attrib
    ]

    assert outlines
    assert all(e.attrib["stroke-linejoin"] == "round" for e in outlines)


def test_output_is_reproducible():
    first = draw_svg("(((...)))", basepair_probabilities=PROBS).to_svg()
    second = draw_svg("(((...)))", basepair_probabilities=PROBS).to_svg()

    assert first == second
