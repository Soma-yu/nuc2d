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

from ._structure import (
    LoopRegion,
    StemRegion,
    iter_nucleotides,
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
        If the number of sequences does not match the number of strands,
        or if a sequence is not as long as the strand it describes.
    """
    if not isinstance(sequences, list):
        raise TypeError(
            "sequences must be a list of strings, one per strand; "
            f"got {type(sequences).__name__}."
        )
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, str):
            raise TypeError(
                f"Each sequence must be a string; sequence {index} is "
                f"{type(sequence).__name__}."
            )

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


def attach_probabilities(
    root_loop: LoopRegion,
    probabilities: npt.ArrayLike,
) -> None:
    """Attach an equilibrium probability to every nucleotide.

    Each nucleotide is given the probability of the state the structure
    puts it in: of pairing with its partner if it is paired, and of
    being unpaired if it is not.

    Parameters
    ----------
    root_loop : LoopRegion
        Root loop of the secondary structure.
    probabilities : array_like
        Base-pair probability matrix. Element (i, j) is how likely
        nucleotides i and j are to be paired with each other, and element
        (i, i) how likely nucleotide i is to be left unpaired. A value
        below 0 or above 1 is attached as 0 or 1.

    Raises
    ------
    TypeError
        If ``probabilities`` does not hold numbers.
    ValueError
        If ``probabilities`` is not a square matrix whose size matches the
        number of nucleotides in the structure, or a probability attached
        is NaN.
    """
    probs = np.asarray(probabilities)
    # bool, signed and unsigned integers, and floating point.
    if probs.dtype.kind not in "biuf":
        raise TypeError(
            "probabilities must be a matrix of numbers, such as a NumPy "
            f"array of floats; got one of dtype {probs.dtype}."
        )
    size = sum(_strand_lengths(root_loop))

    if probs.shape != (size, size):
        raise ValueError(
            f"The structure has {size} nucleotide(s), so probabilities must have "
            f"shape ({size}, {size}), but its shape is {probs.shape}."
        )

    def _probability(i: int, j: int) -> float:
        # Only the elements attached are checked, so that a matrix whose
        # other half is left as NaN, as one drawn as a heatmap often is,
        # can be passed as it is.
        value = float(probs[i][j])
        if math.isnan(value):
            raise ValueError(
                f"probabilities[{i}][{j}] is NaN; a nucleotide cannot be "
                "colored by a probability that is not a number."
            )
        # A tool writing 1 - (sum of the pairing probabilities) can land
        # just outside [0, 1], and the color of the end it is near is the
        # right one. It is clipped here rather than left to the colormap,
        # whose own colors for values out of range can be anything.
        return min(max(value, 0.0), 1.0)

    def _attach_stem_probs(current_stem: StemRegion) -> None:
        nucleotides = current_stem.nucleotides
        for idx in range(len(nucleotides)//2):
            nt1 = nucleotides[idx]
            nt2 = nucleotides[-(idx+1)]
            prob = _probability(nt1.index, nt2.index)
            nt1.probability = nt2.probability = prob
        _attach_loop_probs(current_stem.child_loop)

    def _attach_loop_probs(current_loop: LoopRegion) -> None:
        if current_loop.is_root:
            nucleotides = current_loop.nucleotides
        else:
            nucleotides = current_loop.nucleotides[1:-1]
        for nt in nucleotides:
            nt.probability = _probability(nt.index, nt.index)
        for stem in current_loop.child_stems:
            _attach_stem_probs(stem)

    _attach_loop_probs(root_loop)
