from nuc2d.parser import parse
from nuc2d.structure import StemRegion, LoopRegion

def collect_boundary_nucleotide_locations(root_loop):
    boundary_nucleotide = []

    def _collect_stem(current_stem):
        assert isinstance(current_stem, StemRegion)
        first_nt = current_stem.nucleotides[0]
        boundary_nucleotide.append(
            (first_nt.strand_index, first_nt.index_in_strand, first_nt.index)
        )
        _collect_loop(current_stem.child_loop)
        end_nt = current_stem.nucleotides[-1]
        boundary_nucleotide.append(
            (end_nt.strand_index, end_nt.index_in_strand, end_nt.index)
        )
    
    def _collect_loop(current_loop):
        assert isinstance(current_loop, LoopRegion)
        first_nt = current_loop.nucleotides[0]
        boundary_nucleotide.append(
            (first_nt.strand_index, first_nt.index_in_strand, first_nt.index)
        )
        for stem in current_loop.child_stems:
            _collect_stem(stem)
        end_nt = current_loop.nucleotides[-1]
        boundary_nucleotide.append(
            (end_nt.strand_index, end_nt.index_in_strand, end_nt.index)
        )
    
    _collect_loop(root_loop)
    if len(boundary_nucleotide) >= 4:
        if boundary_nucleotide[0] == boundary_nucleotide[1]:
            boundary_nucleotide.pop(0)
        if boundary_nucleotide[-2] == boundary_nucleotide[-1]:
            boundary_nucleotide.pop()
    return boundary_nucleotide


def test_parse_unpaired():
    root = parse(".....")

    assert collect_boundary_nucleotide_locations(root) == [
        (0, 0, 0),
        (0, 4, 4),
    ]


def test_parse_hairpin():
    root = parse("(((...)))")

    assert collect_boundary_nucleotide_locations(root) == [
        (0, 0, 0),
        (0, 2, 2),
        (0, 6, 6),
        (0, 8, 8),
    ]


def test_parse_duplex():
    root = parse("(((((+)))))")

    assert collect_boundary_nucleotide_locations(root) == [
        (0, 0, 0),
        (0, 4, 4),
        (1, 0, 5),
        (1, 4, 9),
    ]


def test_parse_hinge():
    root = parse("((((((...)))+)))(((...)))")

    assert collect_boundary_nucleotide_locations(root) == [
        (0, 0, 0),
        (0, 2, 2),
        (0, 3, 3),
        (0, 5, 5),
        (0, 9, 9),
        (0, 11, 11),
        (1, 0, 12),
        (1, 2, 14),
        (1, 3, 15),
        (1, 5, 17),
        (1, 9, 21),
        (1, 11, 23),
    ]


def test_parse_nested():
    root = parse("((..((..))..))")

    assert collect_boundary_nucleotide_locations(root) == [
        (0, 0, 0),
        (0, 1, 1),
        (0, 4, 4),
        (0, 5, 5),
        (0, 8, 8),
        (0, 9, 9),
        (0, 12, 12),
        (0, 13, 13),
    ]
