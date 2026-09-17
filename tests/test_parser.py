import pytest

from nuc2d.parser import parse, ParseError, _StrandGroups
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


@pytest.mark.parametrize(
    "dpp_string",
    [
        "",             # empty input
        ")))",          # closing parens with nothing to close
        "+",            # strand break with no nucleotide before it
        "(((",          # unclosed stem
        "...(((",       # unclosed stem after an unpaired region
        "(((...)))x",   # unsupported character
        "abc",          # no valid character at all
    ],
)
def test_parse_invalid_raises_parse_error(dpp_string):
    with pytest.raises(ParseError):
        parse(dpp_string)


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


@pytest.mark.parametrize(
    "dpp_string",
    [
        "...+...",          # two strands, no base pairs at all
        "((...))+...",      # a hairpin and a loose strand
        "((...))+((...))",  # two hairpins that never meet
        "((+))+((+))",      # two duplexes that never meet
        "((+..+))",         # a loose strand sitting inside a loop
        "((+(+)+))",        # an independent duplex inside a loop
        "((.+.((+)).+.))",  # the same, with unpaired nucleotides around it
    ],
)
def test_disconnected_strands_are_rejected(dpp_string):
    """A secondary structure describes one complex, so the strands must hold together."""
    with pytest.raises(ParseError):
        parse(dpp_string)


@pytest.mark.parametrize(
    "dpp_string",
    [
        "(((..+...)))",       # two strands held by one stem
        "(((+)))",            # the same, with no unpaired nucleotides
        "((..((..+..))..))",  # the pair that joins them is nested
        "(((+)))(((+)))",     # the outer strands meet only through the middle one
    ],
)
def test_strands_joined_by_base_pairs_are_accepted(dpp_string):
    """A base pair anywhere is enough, however deeply nested it is."""
    assert parse(dpp_string) is not None


def test_the_groups_are_named_in_the_error():
    with pytest.raises(ParseError, match=r"\(0, 3\) and \(1, 2\)"):
        parse("((+(+)+))")


def test_strand_groups_merges_through_a_third_strand():
    """Joining 0-1 and 1-2 leaves one group, not two."""
    groups = _StrandGroups(4)
    groups.join(0, 1)
    groups.join(1, 2)

    assert sorted(len(group) for group in groups.groups()) == [1, 3]
    assert groups.root_of(0) == groups.root_of(2)
    assert groups.root_of(3) != groups.root_of(0)
