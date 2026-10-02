"""Utilities for attaching annotations to secondary structures.

This module provides functions for adding biological or visualization-
related annotations to parsed secondary structure objects. Examples
include nucleotide sequences, equilibrium probabilities, and other
metadata associated with nucleotides or structural elements.

Annotations are applied after parsing and before layout or rendering,
allowing structural topology and auxiliary information to remain
separated.
"""

import math
from collections import Counter

import numpy as np
import numpy.typing as npt

from ._validation import check_single_line_text
from ._structure import (
    LoopRegion,
    Nucleotide,
    iter_nucleotides,
    iter_stems,
)


def _strand_lengths(root_loop: LoopRegion) -> list[int]:
    """Return the number of nucleotides in each strand of a structure.

    Parameters
    ----------
    root_loop : LoopRegion
        Root loop region of the secondary structure tree.

    Returns
    -------
    list[int]
        Length of each strand, indexed by strand.

    Notes
    -----
    Counting is enough because :func:`~nuc2d._structure.iter_nucleotides`
    yields each nucleotide exactly once, and the parser numbers both the
    strands and the positions within them consecutively from zero. Both
    invariants are pinned by the tests.
    """
    counts = Counter(nt.strand_index for nt in iter_nucleotides(root_loop))
    return [counts[index] for index in range(len(counts))]


def attach_sequences(root_loop: LoopRegion, sequences: list[str]) -> None:
    """Attach nucleotide sequences to a secondary structure.

    Parameters
    ----------
    root_loop : LoopRegion
        Root loop of the secondary structure.
    sequences : list[str]
        Nucleotide sequences for all strands, in the order the strands
        appear in the structure.

    Raises
    ------
    TypeError
        If ``sequences`` is not a list, or a sequence in it is not a
        string.
    ValueError
        If a sequence is empty or holds a line break, a tab or another
        control character, the number of sequences does not match the
        number of strands, or a sequence is not as long as the strand it
        describes.
    """
    if not isinstance(sequences, list):
        raise TypeError(
            "sequences must be a list of strings, one per strand; "
            f"got {type(sequences).__name__}."
        )
    # Each letter is drawn in its node, so a sequence can hold what a
    # single line of text can, and nothing else.
    sequences = [
        check_single_line_text(f"sequences[{index}]", sequence)
        for index, sequence in enumerate(sequences)
    ]

    strand_lengths = _strand_lengths(root_loop)

    if len(sequences) != len(strand_lengths):
        raise ValueError(
            f"The structure has {len(strand_lengths)} strand(s), "
            f"but {len(sequences)} sequence(s) were given."
        )

    for index, (sequence, length) in enumerate(zip(sequences, strand_lengths)):
        if len(sequence) != length:
            raise ValueError(
                f"Strand {index} of the structure has {length} nucleotide(s), "
                f"but the sequence given for it has {len(sequence)}."
            )

    for nt in iter_nucleotides(root_loop):
        nt.base = sequences[nt.strand_index][nt.index_in_strand]


def attach_basepair_probabilities(
    root_loop: LoopRegion,
    basepair_probabilities: npt.ArrayLike,
) -> None:
    """Attach an equilibrium probability to every nucleotide.

    Each nucleotide is given the probability of the state the structure
    puts it in: of pairing with its partner if it is paired, and of
    being unpaired if it is not.

    Parameters
    ----------
    root_loop : LoopRegion
        Root loop of the secondary structure.
    basepair_probabilities : array_like
        Base-pair probability matrix. Element (i, j) is how likely
        nucleotides i and j are to be paired with each other, and element
        (i, i) how likely nucleotide i is to be left unpaired. Only the
        elements on and above the diagonal are read, so the matrix may be
        symmetric or have only its upper triangle filled in. A value below
        0 or above 1 is attached as 0 or 1.

    Raises
    ------
    TypeError
        If ``basepair_probabilities`` does not hold numbers.
    ValueError
        If ``basepair_probabilities`` is not a square matrix whose size
        matches the number of nucleotides in the structure, or a
        probability attached is NaN.
    """
    probs = np.asarray(basepair_probabilities)
    # bool, signed and unsigned integers, and floating point.
    if probs.dtype.kind not in "biuf":
        raise TypeError(
            "basepair_probabilities must be a matrix of numbers, such as a "
            f"NumPy array of floats; got one of dtype {probs.dtype}."
        )
    size = sum(_strand_lengths(root_loop))

    if probs.shape != (size, size):
        raise ValueError(
            f"The structure has {size} nucleotide(s), so "
            f"basepair_probabilities must have shape ({size}, {size}), but "
            f"its shape is {probs.shape}."
        )

    def _probability(i: int, j: int) -> float:
        # Only the elements attached are checked, so that a matrix whose
        # other half is left as NaN, as one drawn as a heatmap often is,
        # can be passed as it is.
        value = float(probs[i][j])
        if math.isnan(value):
            raise ValueError(
                f"basepair_probabilities[{i}][{j}] is NaN; a nucleotide "
                "cannot be colored by a probability that is not a number."
            )
        # A tool writing 1 - (sum of the pairing probabilities) can land
        # just outside [0, 1], and the color of the end it is near is the
        # right one. It is clipped here rather than left to the colormap,
        # whose own colors for values out of range can be anything.
        return min(max(value, 0.0), 1.0)

    # The partner of each paired nucleotide. A stem lists its nucleotides
    # in the order of the structure, from one outer end in to the loop it
    # encloses and out again, so its first nucleotide is paired with its
    # last, its second with the one before that, and so on.
    partner_of: dict[Nucleotide, Nucleotide] = {}
    for stem in iter_stems(root_loop):
        nucleotides = stem.nucleotides
        for one, other in zip(nucleotides, reversed(nucleotides)):
            partner_of[one] = other

    # Each nucleotide reads one element and no other: (i, i) if it is
    # unpaired, and the element of its pair above the diagonal, (i, j)
    # with i < j, if it is paired.
    for nt in iter_nucleotides(root_loop):
        partner = partner_of.get(nt)
        if partner is None:
            nt.probability = _probability(nt.index, nt.index)
        else:
            i, j = sorted((nt.index, partner.index))
            nt.probability = _probability(i, j)
