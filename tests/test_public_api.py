import dataclasses
import inspect
import pkgutil

import pytest

import nuc2d


# The names the package promises. Changing this set changes the promise, so
# the edit belongs in the same commit as the change that caused it.
PUBLIC_NAMES = {
    "BBox",
    "Component",
    "DrawingStyle",
    "ParseError",
    "Placement",
    "RadialLayoutEngine",
    "Scene",
    "__version__",
    "draw_colorbar",
    "draw_structure",
    "draw_svg",
    "draw_text",
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
    """A caller sees nuc2d.ParseError in a traceback, not nuc2d._parse."""
    for name in PUBLIC_NAMES - {"__version__"}:
        assert getattr(nuc2d, name).__module__ == "nuc2d", name


# How many leading arguments each callable takes positionally. Everything
# after them is keyword-only, so a later release can insert an argument
# where it belongs instead of appending it to keep the order intact.
POSITIONAL_COUNT = {
    nuc2d.draw_svg: 1,  # dot_bracket
    nuc2d.draw_structure: 1,  # dot_bracket
    nuc2d.draw_colorbar: 0,
    nuc2d.draw_text: 1,  # text
    nuc2d.Component.from_placements: 1,  # placements
    nuc2d.Scene: 1,  # component
    nuc2d.Placement: 0,
    nuc2d.RadialLayoutEngine: 0,
    nuc2d.DrawingStyle: 0,
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
    for target in (nuc2d.BBox, nuc2d.Component, nuc2d.Placement):
        assert dataclasses.is_dataclass(target), f"{target.__name__} is not a dataclass"
        assert target.__dataclass_params__.frozen, (
            f"{target.__name__} should be frozen"
        )


# What a caller can reach on each of these, beyond the constructor. A member
# is as much a promise as a name in __all__, so this is edited in the same
# commit as a change to it, like PUBLIC_NAMES.
PUBLIC_MEMBERS = {
    "BBox": {
        "xmin", "ymin", "xmax", "ymax",
        "width", "height", "center_x", "center_y",
        "is_empty", "empty", "union",
    },
    "Component": {"bbox", "from_placements"},
    "Placement": {"component", "x", "y", "anchor", "scale", "z_index", "bbox"},
    "Scene": {"to_svg", "save_svg"},
}


@pytest.mark.parametrize("name, members", sorted(PUBLIC_MEMBERS.items()))
def test_each_value_type_offers_only_what_it_promises(name, members):
    target = getattr(nuc2d, name)
    fields = (
        {f.name for f in dataclasses.fields(target)}
        if dataclasses.is_dataclass(target)
        else set()
    )
    offered = {m for m in fields | set(dir(target)) if not m.startswith("_")}

    assert offered == members


def test_the_box_is_built_by_position():
    """Every other type in the package is built by keyword; the box is not.

    Its field set is closed — a box has four numbers — and their names are
    the ones every geometry library uses, so neither an insertion nor a
    rename is coming. Callers write them out in order, which is the promise
    this pins.
    """
    kinds = {p.kind for p in inspect.signature(nuc2d.BBox).parameters.values()}

    assert kinds == {inspect.Parameter.POSITIONAL_OR_KEYWORD}


def test_a_malformed_structure_is_a_value_error():
    """One except ValueError catches every malformed input.

    A sequence that does not match its strand and a probability matrix of
    the wrong shape raise ValueError, so a structure that does not parse
    does too.
    """
    assert issubclass(nuc2d.ParseError, ValueError)

    with pytest.raises(ValueError):
        nuc2d.draw_svg("(((")


def public_callables():
    for name in sorted(PUBLIC_NAMES - {"__version__"}):
        target = getattr(nuc2d, name)
        if inspect.isclass(target):
            yield name, target.__init__
            for attr in vars(target):
                # Through getattr, so that a classmethod comes back bound
                # and callable, as a caller sees it.
                member = getattr(target, attr)
                if callable(member) and not attr.startswith("_"):
                    yield f"{name}.{attr}", member
        elif callable(target):
            yield name, target


def test_no_public_signature_mentions_svgwrite():
    """svgwrite is how nuc2d writes SVG today, not part of what it promises.

    Nothing a caller passes or receives is an svgwrite object, so writing
    SVG another way is not a breaking change.
    """
    for name, target in public_callables():
        signature = inspect.signature(target)
        annotations = [p.annotation for p in signature.parameters.values()]
        annotations.append(signature.return_annotation)

        assert not any("svgwrite" in repr(a) for a in annotations), name
