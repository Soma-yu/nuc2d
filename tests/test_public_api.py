import nuc2d


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
}


def test_all_lists_exactly_the_public_names():
    assert set(nuc2d.__all__) == PUBLIC_NAMES


def test_star_import_provides_every_listed_name():
    namespace = {}
    exec("from nuc2d import *", namespace)

    assert PUBLIC_NAMES <= set(namespace)
