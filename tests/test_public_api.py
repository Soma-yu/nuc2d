import dataclasses
import inspect
import pkgutil

import nuc2d


# The names the package promises. Changing this set changes the promise, so
# the edit belongs in the same commit as the change that caused it.
PUBLIC_NAMES = {
    "BBox",
    "DrawingStyle",
    "ParseError",
    "Placement",
    "RadialLayoutEngine",
    "SVGComponent",
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


def test_every_module_inside_the_package_is_private():
    """A module without an underscore reads as public, whatever __all__ says.

    Everything public is imported from nuc2d itself, so each module inside
    it is marked as the place the code lives rather than an address to
    import from. A new module added without the underscore fails here.
    """
    names = [module.name for module in pkgutil.iter_modules(nuc2d.__path__)]

    assert names, "no modules found inside the package"
    assert all(name.startswith("_") for name in names), (
        f"public modules: {[name for name in names if not name.startswith('_')]}"
    )


def test_every_public_name_reports_nuc2d_as_its_module():
    """A caller sees nuc2d.ParseError in a traceback, not nuc2d._parser."""
    for name in PUBLIC_NAMES - {"__version__"}:
        assert getattr(nuc2d, name).__module__ == "nuc2d", name


# How many leading arguments each callable takes positionally. Everything
# after them is keyword-only, so a later release can insert an argument
# where it belongs instead of appending it to keep the order intact.
POSITIONAL_COUNT = {
    nuc2d.draw_svg: 1,  # dot_bracket
    nuc2d.draw_component: 2,  # drawing, dot_bracket
    nuc2d.render_colorbar: 1,  # drawing
    nuc2d.RadialLayoutEngine: 0,
    nuc2d.DrawingStyle: 0,
    nuc2d.Placement: 0,
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


def test_the_values_a_caller_holds_cannot_be_changed_in_place():
    """These three describe a drawing; they are not part of one.

    A box, a component and a placement are each read by whatever
    consumes them and never read again, so an assignment after the fact
    reaches nothing. Frozen, it raises where it is written instead of
    leaving the value and the drawing disagreeing.
    """
    for target in (nuc2d.BBox, nuc2d.SVGComponent, nuc2d.Placement):
        assert dataclasses.is_dataclass(target), f"{target.__name__} is not a dataclass"
        assert target.__dataclass_params__.frozen, (
            f"{target.__name__} should be frozen"
        )


def test_the_box_is_built_by_position():
    """Every other type in the package is built by keyword; the box is not.

    Its field set is closed — a box has four numbers — and their names are
    the ones every geometry library uses, so neither an insertion nor a
    rename is coming. Callers write them out in order, which is the promise
    this pins.
    """
    kinds = {p.kind for p in inspect.signature(nuc2d.BBox).parameters.values()}

    assert kinds == {inspect.Parameter.POSITIONAL_OR_KEYWORD}
