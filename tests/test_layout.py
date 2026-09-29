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
        edge.edge_type == EdgeType.BASE_PAIR
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
