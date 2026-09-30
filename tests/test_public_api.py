import dataclasses
import enum
import inspect
import pkgutil
from fractions import Fraction

import numpy as np
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


def test_no_other_name_is_reachable_without_an_underscore():
    """What the package imports for itself stays out of sight.

    A name such as nuc2d.version would show up in completion as if it were
    offered, though it is only importlib's function.
    """
    names = {name for name in dir(nuc2d) if not name.startswith("_")}

    assert names == PUBLIC_NAMES - {"__version__"}


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
    nuc2d.Placement: 1,  # component
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
    reaches nothing. It raises where it is written instead of leaving the
    value and the drawing disagreeing, and so does an assignment to a
    name that is not there at all.
    """
    component = nuc2d.draw_text("tRNA")
    placement = nuc2d.Placement(component=component)

    for value, name in [(component.bbox, "BBox"), (component, "Component"),
                        (placement, "Placement")]:
        for attribute in [*sorted(PUBLIC_MEMBERS[name]), "not_an_attribute"]:
            before = getattr(value, attribute, None)

            with pytest.raises(AttributeError):
                setattr(value, attribute, 1.0)

            assert getattr(value, attribute, None) == before, f"{name}.{attribute}"


# What a caller can reach on each of these, beyond the constructor. A member
# is as much a promise as a name in __all__, so this is edited in the same
# commit as a change to it, like PUBLIC_NAMES.
PUBLIC_MEMBERS = {
    "BBox": {"xmin", "ymin", "xmax", "ymax", "width", "height"},
    "Component": {"bbox", "from_placements"},
    "DrawingStyle": {
        "backbone_width", "basepair_width",
        "backbone_dasharray", "basepair_dasharray",
        "backbone_color", "basepair_color", "node_color",
        "node_radius", "node_font_size", "font_family",
        "three_prime_arrow_length", "colormap",
    },
    "Placement": {"component", "x", "y", "anchor", "scale", "bbox"},
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


def test_a_layout_engine_offers_only_its_settings():
    """An engine is made to be passed to a drawing function.

    Its settings can be read back, as floats. How it lays a structure
    out works on types that are not public, so it is not public either.
    """
    engine = nuc2d.RadialLayoutEngine()
    offered = {m for m in dir(engine) if not m.startswith("_")}

    assert offered == {"stem_spacing", "loop_spacing", "coaxial_stack_deflection"}


@pytest.mark.parametrize(
    "args, kwargs",
    [((0.0, 0.0, 1.0, 1.0), {}),
     ((), {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0})],
)
def test_a_box_is_read_rather_than_built(args, kwargs):
    """Nothing public takes a box, so one built by hand would reach nothing."""
    with pytest.raises(TypeError, match="Component.bbox"):
        nuc2d.BBox(*args, **kwargs)


@pytest.mark.parametrize("make", [nuc2d.DrawingStyle, nuc2d.RadialLayoutEngine])
def test_what_can_be_changed_has_no_hash(make):
    """A style and an engine can be changed, and compare by their values.

    So neither has a hash, as a list has none: one changed while it was a
    key of a dict would no longer be found under its hash.
    """
    with pytest.raises(TypeError, match="unhashable"):
        hash(make())


# Every argument that takes a number, drawn with the number made by the
# type given. The structure has a coaxial stack, so the deflection shows,
# and sequences, so the letters do.
def _structure(**kwargs):
    return nuc2d.draw_structure(
        "((+((...))))", sequences=["GG", "GGAAACCCC"], **kwargs
    )


def _style(**kwargs):
    return nuc2d.Scene(_structure(style=nuc2d.DrawingStyle(**kwargs)))


def _engine(**kwargs):
    return nuc2d.Scene(
        _structure(layout_engine=nuc2d.RadialLayoutEngine(**kwargs))
    )


def _placed(**kwargs):
    placement = nuc2d.Placement(component=_structure(), **kwargs)
    return nuc2d.Scene(nuc2d.Component.from_placements([placement]))


NUMBER_ARGUMENTS = {
    "backbone_width": lambda n: _style(backbone_width=n(3)),
    "three_prime_arrow_length": lambda n: _style(three_prime_arrow_length=n(9)),
    "basepair_width": lambda n: _style(basepair_width=n(2)),
    "node_radius": lambda n: _style(node_radius=n(5)),
    "node_font_size": lambda n: _style(node_font_size=n(8)),
    "stem_spacing": lambda n: _engine(stem_spacing=n(18)),
    "loop_spacing": lambda n: _engine(loop_spacing=n(24)),
    "coaxial_stack_deflection": lambda n: _engine(coaxial_stack_deflection=n(25)),
    "x": lambda n: _placed(x=n(3)),
    "y": lambda n: _placed(y=n(3)),
    "anchor": lambda n: _placed(anchor=(n(1), n(0))),
    "scale": lambda n: _placed(scale=n(2)),
    "width_px": lambda n: nuc2d.Scene(_structure(), width_px=n(300)),
    "height_px": lambda n: nuc2d.Scene(_structure(), height_px=n(300)),
    "font_size": lambda n: nuc2d.Scene(nuc2d.draw_text("tRNA", font_size=n(14))),
}


@pytest.mark.parametrize("number", [int, np.int64, np.float32, Fraction])
@pytest.mark.parametrize(
    "draw", NUMBER_ARGUMENTS.values(), ids=list(NUMBER_ARGUMENTS)
)
def test_a_number_of_any_type_is_drawn_as_the_float_it_equals(draw, number):
    """A NumPy scalar or a Fraction used to reach the SVG writer as it was.

    The writer refused most of them, when the drawing was made rather
    than where the number was given, and a NumPy float32 laid a structure
    out at its own precision.
    """
    assert draw(number).to_svg() == draw(float).to_svg()


# Every argument that takes a string, drawn with the string made by the
# kind given.
STRING_ARGUMENTS = {
    "dot_bracket": lambda s: nuc2d.draw_svg(s("((+((...))))")),
    "sequences": lambda s: nuc2d.draw_svg(
        "((+((...))))", sequences=[s("GG"), s("GGAAACCCC")]
    ),
    "colorbar_label": lambda s: nuc2d.draw_svg(
        "(((...)))", probabilities=np.eye(9) * 0.5, colorbar_label=s("Unpaired")
    ),
    "label": lambda s: nuc2d.Scene(nuc2d.draw_colorbar(label=s("Unpaired"))),
    "text": lambda s: nuc2d.Scene(nuc2d.draw_text(s("tRNA"))),
    "font_family": lambda s: nuc2d.Scene(
        nuc2d.draw_text("tRNA", font_family=s("Arial"))
    ),
    "backbone_color": lambda s: _style(backbone_color=s("red")),
    "backbone_dasharray": lambda s: _style(backbone_dasharray=s("2,1")),
    "basepair_color": lambda s: _style(basepair_color=s("red")),
    "basepair_dasharray": lambda s: _style(basepair_dasharray=s("2,1")),
    "node_color": lambda s: _style(node_color=s("red")),
    "style font_family": lambda s: _style(font_family=s("Arial")),
    "anchor": lambda s: _placed(anchor=s("center")),
}


def _enum_member(text):
    """Return a member of an Enum with str mixed in, whose value is text."""
    return enum.Enum("Named", {"MEMBER": text}, type=str).MEMBER


@pytest.mark.parametrize(
    "string", [np.str_, _enum_member], ids=["np.str_", "str Enum"]
)
@pytest.mark.parametrize(
    "draw", STRING_ARGUMENTS.values(), ids=list(STRING_ARGUMENTS)
)
def test_a_string_of_any_kind_is_drawn_as_the_text_it_holds(draw, string):
    """A subclass of str used to reach the SVG writer as it was.

    The writer asks for its text with str(), and an Enum with str mixed
    in answers with the member's name: its colour was refused when the
    drawing was made, and its text was written as Named.MEMBER.
    """
    assert draw(string).to_svg() == draw(str).to_svg()


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
