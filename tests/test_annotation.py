import numpy as np
import pytest

from nuc2d.annotation import (
    attach_sequences,
    attach_basepair_probabilities,
)
from nuc2d.parser import parse
from nuc2d.structure import iter_nucleotides


def collect_nucleotides(root_loop):
    nucleotides = []

    def _collect_stem(current_stem):
        nucleotides.extend(current_stem.nucleotides[1:-1])
        _collect_loop(current_stem.child_loop)

    def _collect_loop(current_loop):
        if current_loop.is_root:
            nucleotides.extend(current_loop.nucleotides)
        else:
            nucleotides.extend(current_loop.nucleotides[1:-1])

        for stem in current_loop.child_stems:
            _collect_stem(stem)

    _collect_loop(root_loop)

    return sorted(
        nucleotides,
        key=lambda nt: nt.index,
    )


def test_attach_sequences_hairpin():
    root = parse("(((...)))")

    attach_sequences(
        root,
        ["ACGUACGUA"],
    )

    nts = collect_nucleotides(root)

    assert [nt.base for nt in nts] == list("ACGUACGUA")


def test_attach_sequences_duplex():
    root = parse("(((((+)))))")

    attach_sequences(
        root,
        [
            "AAAAA",
            "UUUUU",
        ],
    )

    nts = collect_nucleotides(root)

    assert [nt.base for nt in nts] == [
        "A", "A", "A", "A", "A",
        "U", "U", "U", "U", "U",
    ]


def test_attach_basepair_probabilities_hairpin():
    root = parse("(((...)))")

    probs = np.zeros((9, 9))

    probs[0, 8] = 0.9
    probs[8, 0] = 0.9

    probs[1, 7] = 0.8
    probs[7, 1] = 0.8

    probs[2, 6] = 0.7
    probs[6, 2] = 0.7

    probs[3, 3] = 0.1
    probs[4, 4] = 0.2
    probs[5, 5] = 0.3

    attach_basepair_probabilities(
        root,
        probs,
    )

    nts = collect_nucleotides(root)

    assert nts[0].basepair_probability == 0.9
    assert nts[1].basepair_probability == 0.8
    assert nts[2].basepair_probability == 0.7

    assert nts[3].basepair_probability == 0.1
    assert nts[4].basepair_probability == 0.2
    assert nts[5].basepair_probability == 0.3

    assert nts[6].basepair_probability == 0.7
    assert nts[7].basepair_probability == 0.8
    assert nts[8].basepair_probability == 0.9


@pytest.mark.parametrize(
    "dpp_string, expected",
    [
        ("(((...)))", [9]),
        (".....", [5]),
        ("(((..+...)))", [5, 6]),
        ("((((((...)))+)))(((...)))", [12, 12]),
    ],
)
def test_iter_nucleotides_yields_each_once(dpp_string, expected):
    root = parse(dpp_string)

    indices = sorted(nt.index for nt in iter_nucleotides(root))

    assert indices == list(range(sum(expected)))


@pytest.mark.parametrize(
    "dpp_string, expected",
    [
        ("(((...)))", [9]),
        (".....", [5]),
        ("(((..+...)))", [5, 6]),
        ("((((((...)))+)))(((...)))", [12, 12]),
    ],
)
def test_positions_within_each_strand_are_consecutive(dpp_string, expected):
    """_strand_lengths counts nucleotides, which is only a length because
    every position from zero is present exactly once."""
    root = parse(dpp_string)

    positions: dict[int, list[int]] = {}
    for nt in iter_nucleotides(root):
        positions.setdefault(nt.strand_index, []).append(nt.index_in_strand)

    assert [sorted(v) for _, v in sorted(positions.items())] == [
        list(range(length)) for length in expected
    ]


def test_attach_sequences_rejects_a_wrong_strand_count():
    root = parse("(((..+...)))")

    with pytest.raises(ValueError, match="2 strand"):
        attach_sequences(root, ["AUGCAUGCAUG"])


def test_attach_sequences_rejects_a_wrong_sequence_length():
    root = parse("(((..+...)))")

    with pytest.raises(ValueError, match="Strand 0"):
        attach_sequences(root, ["AU", "UGCCAU"])


def test_attach_sequences_accepts_matching_sequences():
    root = parse("(((..+...)))")

    attach_sequences(root, ["AUGCA", "UGCCAU"])

    assert [nt.base for nt in sorted(iter_nucleotides(root), key=lambda n: n.index)] == list(
        "AUGCA" "UGCCAU"
    )


@pytest.mark.parametrize("size", [3, 8, 10])
def test_attach_probabilities_rejects_a_wrong_shape(size):
    root = parse("(((...)))")

    with pytest.raises(ValueError, match=r"shape \(9, 9\)"):
        attach_basepair_probabilities(root, np.eye(size))


def test_attach_probabilities_rejects_a_non_square_matrix():
    root = parse("(((...)))")

    with pytest.raises(ValueError, match="shape"):
        attach_basepair_probabilities(root, np.zeros((9, 4)))


def test_attach_probabilities_accepts_a_nested_list():
    root = parse("(((...)))")

    attach_basepair_probabilities(root, np.eye(9).tolist())

    assert all(
        nt.basepair_probability is not None for nt in iter_nucleotides(root)
    )
