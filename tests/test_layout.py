import math
from fractions import Fraction

import numpy as np
import pytest

from nuc2d._parse import parse
from nuc2d._layout import (
    RadialLayoutEngine,
    EdgeType,
    layout,
)


def get_edge_counts(layout_result):
    n_backbone = sum(
        edge.edge_type == EdgeType.BACKBONE
        for edge in layout_result.edges
    )

    n_basepair = sum(
        edge.edge_type == EdgeType.BASEPAIR
        for edge in layout_result.edges
    )

    return n_backbone, n_basepair


def test_layout_unpaired():
    root = parse(".....")

    result = layout(root, RadialLayoutEngine())

    n_backbone, n_basepair = get_edge_counts(result)

    assert len(result.nodes) == 5
    assert n_backbone == 4
    assert n_basepair == 0
    assert len(result.decorations) == 1


def test_a_short_strand_with_no_base_pair_is_drawn_straight():
    engine = RadialLayoutEngine()

    result = layout(parse("....."), engine)

    xs = [node.pos.x for node in result.nodes]
    ys = [node.pos.y for node in result.nodes]
    assert ys == pytest.approx([ys[0]] * 5)
    assert np.diff(xs) == pytest.approx([engine.stem_spacing] * 4)


@pytest.mark.parametrize("n", [6, 7, 12, 40])
def test_a_longer_strand_with_no_base_pair_is_drawn_on_a_circle(n):
    """As the outermost loop of a structure with base pairs is."""
    engine = RadialLayoutEngine()

    result = layout(parse("." * n), engine)

    points = np.array([(node.pos.x, node.pos.y) for node in result.nodes])
    center = points.mean(axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert radii == pytest.approx(np.full(n, radii[0]))
    assert chords == pytest.approx(np.full(n - 1, engine.loop_spacing))

    # The two ends side by side at the top, where y is smallest, 5' on
    # the right; the backbone does not join them.
    (x5, y5), (x3, y3) = points[0], points[-1]
    assert y5 == pytest.approx(y3) and y5 < center[1]
    assert x5 > x3
    assert get_edge_counts(result) == (n - 1, 0)
    assert len(result.decorations) == 1


def test_layout_hairpin():
    root = parse("(((...)))")

    result = layout(root, RadialLayoutEngine())

    n_backbone, n_basepair = get_edge_counts(result)
    print(len(result.nodes))

    assert len(result.nodes) == 9
    assert n_backbone == 8
    assert n_basepair == 3
    assert len(result.decorations) == 1


def test_layout_duplex():
    root = parse("(((((+)))))")

    result = layout(root, RadialLayoutEngine())

    n_backbone, n_basepair = get_edge_counts(result)

    assert len(result.nodes) == 10
    assert n_backbone == 8
    assert n_basepair == 5
    assert len(result.decorations) == 2


def test_layout_hinge():
    root = parse("((((((...)))+)))(((...)))")

    result = layout(root, RadialLayoutEngine())

    n_backbone, n_basepair = get_edge_counts(result)

    assert len(result.nodes) == 24
    assert n_backbone == 22
    assert n_basepair == 9
    assert len(result.decorations) == 2


def test_layout_nested():
    root = parse("((..((...))..))")

    result = layout(root, RadialLayoutEngine())

    n_backbone, n_basepair = get_edge_counts(result)

    assert len(result.nodes) == 15
    assert n_backbone == 14
    assert n_basepair == 4
    assert len(result.decorations) == 1


def positions(dot_bracket, engine):
    return [node.pos for node in layout(parse(dot_bracket), engine).nodes]


def test_a_setting_assigned_after_the_engine_is_made_is_used():
    """The settings are attributes a caller may assign, not only read."""
    engine = RadialLayoutEngine()
    engine.stem_spacing = 30

    assert positions("(((...)))", engine) == positions(
        "(((...)))", RadialLayoutEngine(stem_spacing=30)
    )
    assert positions("(((...)))", engine) != positions(
        "(((...)))", RadialLayoutEngine()
    )


@pytest.mark.parametrize("degrees", [0.0, 10.0, 30.0])
def test_stems_stacked_across_a_break_bend_by_the_angle_in_degrees(degrees):
    """The second stem turns from the first by coaxial_stack_deflection."""
    engine = RadialLayoutEngine(coaxial_stack_deflection=degrees)
    nodes = layout(parse("((+((...))))"), engine).nodes
    pos = {node.nucleotide.index: node.pos for node in nodes}
    first, second = pos[1] - pos[0], pos[3] - pos[2]

    bend = math.degrees(math.atan2(
        first.x * second.y - first.y * second.x,
        first.x * second.x + first.y * second.y,
    ))

    assert bend == pytest.approx(degrees)


SETTINGS = ["stem_spacing", "loop_spacing", "coaxial_stack_deflection"]
SPACINGS = ["stem_spacing", "loop_spacing"]


def test_the_defaults_are_written_as_the_floats_they_are():
    engine = RadialLayoutEngine()

    assert [type(getattr(engine, name)) for name in SETTINGS] == [float] * 3


@pytest.mark.parametrize("name", SPACINGS)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_a_spacing_must_be_positive_and_finite(name, value):
    with pytest.raises(ValueError, match=name):
        RadialLayoutEngine(**{name: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_the_deflection_must_be_finite(value):
    with pytest.raises(ValueError, match="coaxial_stack_deflection"):
        RadialLayoutEngine(coaxial_stack_deflection=value)


@pytest.mark.parametrize("degrees", [-10.0, 0.0])
def test_the_deflection_may_be_zero_or_bend_the_other_way(degrees):
    RadialLayoutEngine(coaxial_stack_deflection=degrees)


@pytest.mark.parametrize("name", SETTINGS)
@pytest.mark.parametrize("value", ["15", None, True])
def test_a_setting_that_is_not_a_number_is_a_type_error(name, value):
    with pytest.raises(TypeError, match=name):
        RadialLayoutEngine(**{name: value})


def test_an_assignment_is_checked_too_and_leaves_the_engine_as_it_was():
    engine = RadialLayoutEngine(stem_spacing=18.0)

    with pytest.raises(ValueError, match="stem_spacing"):
        engine.stem_spacing = -1.0

    assert engine.stem_spacing == 18.0


@pytest.mark.parametrize(
    "value", [18, np.int64(18), np.float32(18.0), Fraction(18)]
)
def test_a_setting_is_kept_as_a_float(value):
    """What lays a structure out then computes in floats, whatever was given."""
    engine = RadialLayoutEngine(stem_spacing=value)

    assert engine.stem_spacing == 18.0 and type(engine.stem_spacing) is float


def test_engines_are_equal_when_their_settings_are():
    assert RadialLayoutEngine() == RadialLayoutEngine(stem_spacing=15)
    assert RadialLayoutEngine() != RadialLayoutEngine(stem_spacing=18.0)


def test_a_misspelled_setting_is_refused_and_the_right_one_named():
    engine = RadialLayoutEngine()

    with pytest.raises(AttributeError, match="Did you mean: 'stem_spacing'"):
        engine.stem_spcing = 30.0

    assert not hasattr(engine, "stem_spcing")
