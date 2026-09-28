import ast
import dataclasses
import importlib
import inspect
import pkgutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import nuc2d
from nuc2d import (
    BBox,
    Placement,
    Scene,
    compose,
    draw_colorbar,
    draw_structure,
    draw_svg,
)
from nuc2d._compose import _ANCHOR_FRACTIONS, _anchor_point, fit


CLOVERLEAF = "(((((((..((((........)))).(((((.......+))))).....(((((.......))))))))))))...."
PROBS = np.eye(9) * 0.4 + 0.3


def structure(dot_bracket="(((...)))", **kwargs):
    return draw_structure(dot_bracket, **kwargs)


def count(svg_string, tag):
    return sum(
        1 for element in ET.fromstring(svg_string).iter()
        if element.tag.endswith(tag)
    )


# ---------------------------------------------------------------- anchors


@pytest.mark.parametrize("anchor", [*_ANCHOR_FRACTIONS, (0.25, 0.75), (1.3, -0.2)])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_the_anchor_point_lands_on_x_y(anchor, scale):
    placement = Placement(
        component=structure(CLOVERLEAF), x=100.0, y=50.0, anchor=anchor, scale=scale
    )

    assert _anchor_point(placement.bbox, anchor) == pytest.approx((100.0, 50.0))


def test_the_anchor_stays_put_while_the_component_is_scaled():
    component = structure(CLOVERLEAF)
    points = {
        scale: _anchor_point(
            Placement(component=component, x=7.0, y=9.0, anchor=(0.3, 0.6),
                      scale=scale).bbox,
            (0.3, 0.6),
        )
        for scale in (0.5, 3.0)
    }

    assert points[0.5] == pytest.approx(points[3.0])


def test_the_placed_box_is_the_component_box_scaled():
    component = structure(CLOVERLEAF)
    box = Placement(component=component, scale=2.5).bbox

    assert box.width == pytest.approx(component.bbox.width * 2.5)
    assert box.height == pytest.approx(component.bbox.height * 2.5)


def test_upper_left_is_the_default_anchor():
    """So x and y read the way an SVG rect's do."""
    box = Placement(component=structure(CLOVERLEAF), x=10.0, y=20.0).bbox

    assert (box.xmin, box.ymin) == pytest.approx((10.0, 20.0))


@pytest.mark.parametrize(
    "anchor", ["upperleft", "top left", (0.5,), (0.5, 0.5, 0.5), ("a", 0.0),
               (float("nan"), 0.0), (True, False), [0.5, 0.5]],
)
def test_a_bad_anchor_is_refused_where_it_is_written(anchor):
    with pytest.raises(ValueError, match="anchor"):
        Placement(component=structure(), anchor=anchor)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_scale_must_be_positive_and_finite(scale):
    with pytest.raises(ValueError, match="scale"):
        Placement(component=structure(), scale=scale)


def test_placements_are_frozen_and_hashable():
    placement = Placement(component=structure())

    with pytest.raises(dataclasses.FrozenInstanceError):
        placement.x = 1.0
    hash(placement)


# ---------------------------------------------------------------- compose


def test_compose_encloses_everything_it_places():
    a, b = structure(CLOVERLEAF), structure()
    pa = Placement(component=a)
    pb = Placement(component=b, x=pa.bbox.xmax + 20.0, y=0.0)

    panel = compose([pa, pb])

    assert panel.bbox == pa.bbox.union(pb.bbox)


def test_compose_of_nothing_is_empty():
    assert compose([]).bbox.is_empty


def test_compose_refuses_a_component_that_was_not_placed():
    with pytest.raises(TypeError, match="Placement"):
        compose([structure()])


def first_tag(svg_string, tags):
    """Return which of ``tags`` appears first in the document."""
    for element in ET.fromstring(svg_string).iter():
        for tag in tags:
            if element.tag.endswith(tag):
                return tag
    return None


def test_compose_draws_in_order_of_z_index():
    colorbar = draw_colorbar()  # draws a rect
    plain = structure()  # draws circles, no rect

    def drawn_first(placements):
        return first_tag(Scene(compose(placements)).to_svg(), ["rect", "circle"])

    assert drawn_first([Placement(component=colorbar), Placement(component=plain)]) == "rect"
    assert drawn_first(
        [Placement(component=colorbar, z_index=1), Placement(component=plain)]
    ) == "circle"


def test_a_definition_is_written_once_however_many_components_use_it():
    a, b = structure(CLOVERLEAF), structure(CLOVERLEAF)
    pa = Placement(component=a)
    panel = compose([pa, Placement(component=b, x=pa.bbox.xmax)])

    svg = Scene(panel).to_svg()

    assert count(svg, "marker") == 1


def test_different_definitions_are_all_written():
    colored = structure(probs=PROBS)  # structure with colorbar: a gradient
    plain = structure()
    pc = Placement(component=colored)

    svg = Scene(compose([pc, Placement(component=plain, x=pc.bbox.xmax)])).to_svg()

    assert count(svg, "marker") == 1
    assert count(svg, "linearGradient") == 1


def test_a_composed_component_can_be_placed_again():
    inner = compose([Placement(component=structure(CLOVERLEAF))])
    outer = compose([Placement(component=inner, x=5.0, y=5.0, scale=0.5)])

    assert (outer.bbox.xmin, outer.bbox.ymin) == pytest.approx((5.0, 5.0))
    assert outer.bbox.width == pytest.approx(inner.bbox.width * 0.5)


# ---------------------------------------------------------------- fit


@pytest.mark.parametrize("anchor", ["center", "upper left", "center right", (0.2, 0.8)])
def test_fit_occupies_exactly_the_slot(anchor):
    slot = BBox(-500.0, 0.0, 0.0, 500.0)

    placement = fit(structure(CLOVERLEAF), slot, anchor=anchor)

    for got, want in zip(
        (placement.bbox.xmin, placement.bbox.ymin, placement.bbox.xmax, placement.bbox.ymax),
        (slot.xmin, slot.ymin, slot.xmax, slot.ymax),
    ):
        assert got == pytest.approx(want)


def test_fit_uses_the_largest_scale_that_fits():
    component = structure(CLOVERLEAF)
    slot = BBox(0.0, 0.0, 300.0, 1000.0)

    placement = fit(component, slot)

    assert placement.scale == pytest.approx(
        min(300.0 / component.bbox.width, 1000.0 / component.bbox.height)
    )


def test_fit_aligns_the_component_by_its_anchor():
    """The drawn part sits against the side the anchor names."""
    component = structure(CLOVERLEAF)
    slot = BBox(0.0, 0.0, 2000.0, 500.0)  # much wider than the component

    placement = fit(component, slot, anchor="center right")

    # The padded box is placed by its upper left corner, so the component's
    # own box lands at that corner plus its offset inside the padding.
    padded, s = placement.component.bbox, placement.scale
    box = component.bbox
    right = placement.x + s * (box.xmax - padded.xmin)
    top = placement.y + s * (box.ymin - padded.ymin)
    bottom = placement.y + s * (box.ymax - padded.ymin)

    assert right == pytest.approx(slot.xmax)
    assert (top, bottom) == pytest.approx((slot.ymin, slot.ymax))


def test_fit_refuses_what_has_no_area():
    with pytest.raises(ValueError):
        fit(compose([]), BBox(0.0, 0.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        fit(structure(), BBox(0.0, 0.0, 0.0, 1.0))


# ---------------------------------------------------------------- Scene


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"probs": PROBS},
        {"probs": PROBS, "colorbar_label": None},
        {"probs": PROBS, "add_colorbar": False},
        {"sequences": ["AUGCAUGCA"]},
    ],
)
@pytest.mark.parametrize("size", [{}, {"width_px": 300.0}, {"height_px": 120.0}])
def test_draw_svg_is_a_scene_of_draw_structure(kwargs, size):
    """draw_svg(...) is Scene(draw_structure(...)), byte for byte."""
    expected = Scene(draw_structure("(((...)))", **kwargs), **size).to_svg()

    assert draw_svg("(((...)))", **kwargs, **size).to_svg() == expected


def test_a_scene_is_framed_on_the_component():
    component = structure(CLOVERLEAF)
    root = ET.fromstring(Scene(component).to_svg())

    assert [float(v) for v in root.attrib["viewBox"].split(",")] == pytest.approx(
        list(component.bbox.to_viewbox())
    )


def test_the_size_follows_the_component_proportions():
    component = structure(CLOVERLEAF)
    aspect = component.bbox.width / component.bbox.height

    default = ET.fromstring(Scene(component).to_svg()).attrib
    by_width = ET.fromstring(Scene(component, width_px=300.0).to_svg()).attrib
    both = ET.fromstring(
        Scene(component, width_px=300.0, height_px=100.0).to_svg()
    ).attrib

    assert default["height"] == "500.0px"
    assert float(default["width"][:-2]) == pytest.approx(500.0 * aspect)
    assert float(by_width["height"][:-2]) == pytest.approx(300.0 / aspect)
    assert (both["width"], both["height"]) == ("300.0px", "100.0px")


def test_save_svg_writes_the_same_document(tmp_path):
    scene = Scene(structure(CLOVERLEAF))
    path = tmp_path / "structure.svg"

    scene.save_svg(path)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("<?xml")
    assert text.endswith(scene.to_svg())


def test_jupyter_displays_the_scene():
    scene = Scene(structure())

    assert scene._repr_svg_() == scene.to_svg()


@pytest.mark.parametrize("old, new", [("tostring", "to_svg"), ("saveas", "save_svg")])
def test_the_names_1x_used_say_what_replaced_them(old, new):
    scene = Scene(structure())

    with pytest.raises(AttributeError, match=new):
        getattr(scene, old)
    assert not hasattr(scene, old)


def test_any_other_missing_attribute_is_a_plain_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'add'$"):
        Scene(structure()).add


def test_an_empty_component_cannot_be_drawn():
    with pytest.raises(ValueError, match="empty"):
        Scene(compose([]))


@pytest.mark.parametrize("size", [0.0, -10.0, float("inf")])
def test_a_size_must_be_positive_and_finite(size):
    with pytest.raises(ValueError, match="width_px"):
        Scene(structure(), width_px=size)
    with pytest.raises(ValueError, match="height_px"):
        Scene(structure(), height_px=size)


def test_a_scene_can_be_written_more_than_once():
    """Writing does not consume the component."""
    scene = Scene(structure(CLOVERLEAF, probs=np.eye(76) * 0.5))

    assert scene.to_svg() == scene.to_svg()


def test_the_colorbar_travels_with_its_gradient():
    colorbar = draw_colorbar()

    assert count(Scene(colorbar).to_svg(), "linearGradient") == 1


def imports_of(module):
    """Return the top-level names of every package a module imports."""
    tree = ast.parse(inspect.getsource(module))
    return {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
    }


def test_only_the_svg_module_knows_how_svg_is_written():
    """Writing SVG another way changes nuc2d._svg and no other module."""
    importers = {
        info.name
        for info in pkgutil.iter_modules(nuc2d.__path__)
        if "svgwrite" in imports_of(importlib.import_module(f"nuc2d.{info.name}"))
    }

    assert importers == {"_svg"}
