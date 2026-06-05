from nuc2d.parser import parse
from nuc2d.layout import (
    RadialLayoutEngine,
    EdgeType,
)


def get_edge_counts(layout_result):
    n_backbone = sum(
        edge.type == EdgeType.BACKBONE
        for edge in layout_result.edges
    )

    n_basepair = sum(
        edge.type == EdgeType.BASE_PAIR
        for edge in layout_result.edges
    )

    return n_backbone, n_basepair


def test_layout_unpaired():
    root = parse(".....")

    result = RadialLayoutEngine().layout(root)

    n_backbone, n_basepair = get_edge_counts(result)

    assert len(result.nodes) == 5
    assert n_backbone == 4
    assert n_basepair == 0
    assert len(result.markers) == 1


def test_layout_hairpin():
    root = parse("(((...)))")

    result = RadialLayoutEngine().layout(root)

    n_backbone, n_basepair = get_edge_counts(result)
    print(len(result.nodes))

    assert len(result.nodes) == 9
    assert n_backbone == 8
    assert n_basepair == 3
    assert len(result.markers) == 1


def test_layout_duplex():
    root = parse("(((((+)))))")

    result = RadialLayoutEngine().layout(root)

    n_backbone, n_basepair = get_edge_counts(result)

    assert len(result.nodes) == 10
    assert n_backbone == 8
    assert n_basepair == 5
    assert len(result.markers) == 2


def test_layout_hinge():
    root = parse("((((((...)))+)))(((...)))")

    result = RadialLayoutEngine().layout(root)

    n_backbone, n_basepair = get_edge_counts(result)

    assert len(result.nodes) == 24
    assert n_backbone == 22
    assert n_basepair == 9
    assert len(result.markers) == 2


def test_layout_nested():
    root = parse("((..((...))..))")

    result = RadialLayoutEngine().layout(root)

    n_backbone, n_basepair = get_edge_counts(result)

    assert len(result.nodes) == 15
    assert n_backbone == 14
    assert n_basepair == 4
    assert len(result.markers) == 1
