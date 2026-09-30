import xml.etree.ElementTree as ET

import numpy as np
import pytest

from nuc2d import Component, Placement, Scene, draw_colorbar, draw_structure, draw_svg


CLOVERLEAF = "(((((((..((((........)))).(((((.......+))))).....(((((.......))))))))))))...."
PROBS = np.eye(9) * 0.4 + 0.3


def structure(dot_bracket="(((...)))", **kwargs):
    return draw_structure(dot_bracket, **kwargs)


def count(svg_string, tag):
    return sum(
        1 for element in ET.fromstring(svg_string).iter()
        if element.tag.endswith(tag)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"probabilities": PROBS},
        {"probabilities": PROBS, "colorbar_label": None},
        {"probabilities": PROBS, "add_colorbar": False},
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

    bbox = component.bbox
    assert [float(v) for v in root.attrib["viewBox"].split(",")] == pytest.approx(
        [bbox.xmin, bbox.ymin, bbox.width, bbox.height]
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
        Scene(Component.from_placements([]))


@pytest.mark.parametrize("size", [0.0, -10.0, float("inf")])
def test_a_size_must_be_positive_and_finite(size):
    with pytest.raises(ValueError, match="width_px"):
        Scene(structure(), width_px=size)
    with pytest.raises(ValueError, match="height_px"):
        Scene(structure(), height_px=size)


def test_a_scene_can_be_written_more_than_once():
    """Writing does not consume the component."""
    scene = Scene(structure(CLOVERLEAF, probabilities=np.eye(76) * 0.5))

    assert scene.to_svg() == scene.to_svg()


def test_the_colorbar_travels_with_its_gradient():
    colorbar = draw_colorbar()

    assert count(Scene(colorbar).to_svg(), "linearGradient") == 1


@pytest.mark.parametrize("component", ["(((...)))", None])
def test_a_scene_shows_a_component(component):
    with pytest.raises(TypeError, match="Scene takes a Component"):
        Scene(component)


def test_a_placement_given_to_a_scene_is_answered_with_what_to_pass():
    placement = Placement(component=structure())

    with pytest.raises(TypeError, match=r"placement\.component"):
        Scene(placement)


@pytest.mark.parametrize("name", ["width_px", "height_px"])
@pytest.mark.parametrize("size", ["300", True])
def test_a_size_must_be_a_number(name, size):
    with pytest.raises(TypeError, match=name):
        Scene(structure(), **{name: size})


def test_a_scene_takes_no_attribute_it_does_not_have():
    """An attribute assigned by mistake raises.

    The size is given when the scene is made, so width_px assigned later
    would change nothing.
    """
    scene = Scene(structure())

    with pytest.raises(AttributeError, match="width_px"):
        scene.width_px = 800.0
