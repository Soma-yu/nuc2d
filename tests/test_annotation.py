from nuc2d.parser import parse
from nuc2d.annotation import (
    attach_sequences,
    attach_basepair_probabilities,
)

import numpy as np


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
