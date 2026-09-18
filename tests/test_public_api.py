import inspect

import nuc2d
from nuc2d.layout import ArcEdge, ArrowMarker, Edge, LayoutResult, LineEdge, Marker, Node
from nuc2d.svg import render_colorbar, render_structure


# The names the package promises. Changing this set changes the promise, so
# the edit belongs in the same commit as the change that caused it.
PUBLIC_NAMES = {
    "BBox",
    "DrawingStyle",
    "LayoutEngine",
    "ParseError",
    "Placement",
    "RadialLayoutEngine",
    "SVGComponent",
    "Vec2",
    "__version__",
    "compose",
    "draw_component",
    "draw_svg",
    "render_colorbar",
}


def test_all_lists_exactly_the_public_names():
    assert set(nuc2d.__all__) == PUBLIC_NAMES


def test_star_import_provides_every_listed_name():
    namespace = {}
    exec("from nuc2d import *", namespace)

    assert PUBLIC_NAMES <= set(namespace)


# How many leading arguments each callable takes positionally. Everything
# after them is keyword-only, so a later release can insert an argument
# where it belongs instead of appending it to keep the order intact.
POSITIONAL_COUNT = {
    nuc2d.draw_svg: 1,  # dpp_string
    nuc2d.draw_component: 2,  # drawing, dpp_string
    nuc2d.DrawingStyle: 0,
    nuc2d.Placement: 0,
    nuc2d.RadialLayoutEngine: 0,
    render_colorbar: 1,  # drawing
    render_structure: 2,  # drawing, layout_result
    # Not in __all__, but a caller writing a LayoutEngine of its own has to
    # build these to return a LayoutResult, so they are promised too.
    Node: 0,
    Edge: 0,
    LineEdge: 0,
    ArcEdge: 0,
    Marker: 0,
    ArrowMarker: 0,
    LayoutResult: 0,
}


def test_optional_arguments_are_keyword_only():
    """Adding an argument must not be a breaking change.

    The required arguments stay positional, since they are what the call is
    about. Everything optional is keyword-only.
    """
    for target, positional in POSITIONAL_COUNT.items():
        kinds = [p.kind for p in inspect.signature(target).parameters.values()]

        assert kinds[:positional] == [inspect.Parameter.POSITIONAL_OR_KEYWORD] * (
            positional
        ), f"{target.__name__} should take {positional} positional argument(s)"
        assert all(
            kind is inspect.Parameter.KEYWORD_ONLY for kind in kinds[positional:]
        ), f"{target.__name__} takes an argument that is not keyword-only"


def test_coordinates_are_still_positional():
    """The exception: a pair or a box reads better written out in order."""
    for target in (nuc2d.Vec2, nuc2d.BBox, nuc2d.SVGComponent):
        kinds = [p.kind for p in inspect.signature(target).parameters.values()]

        assert all(
            kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for kind in kinds
        ), f"{target.__name__} should stay positional"
