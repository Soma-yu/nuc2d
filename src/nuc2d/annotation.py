"""Utilities for attaching annotations to secondary structures.

This module provides functions for adding biological or visualization-
related annotations to parsed secondary structure objects. Examples
include nucleotide sequences, base-pair probabilities, and other
metadata associated with nucleotides or structural elements.

Annotations are applied after parsing and before layout or rendering,
allowing structural topology and auxiliary information to remain
separated.
"""

from collections import Counter

import numpy as np

from .structure import (
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
    Counting is enough because :func:`~nuc2d.structure.iter_nucleotides`
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
    ValueError
        If the number of sequences does not match the number of strands,
        or if a sequence is not as long as the strand it describes.
    """
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
    probs: np.ndarray,
) -> None:
    """Attach base-pair probabilities to a secondary structure.

    Parameters
    ----------
    root_loop : LoopRegion
        Root loop of the secondary structure.
    probs : ndarray
        Base-pair probability matrix. Element (i, j) gives the probability
        that nucleotide i pairs with nucleotide j. Diagonal elements give
        the probabilities that nucleotides remain unpaired.

    Raises
    ------
    ValueError
        If ``probs`` is not a square matrix whose size matches the number
        of nucleotides in the structure.
    """
    probs = np.asarray(probs)
    size = sum(_strand_lengths(root_loop))

    if probs.shape != (size, size):
        raise ValueError(
            f"The structure has {size} nucleotide(s), so probs must have "
            f"shape ({size}, {size}), but its shape is {probs.shape}."
        )

    def _attach_stem_probs(current_stem: StemRegion) -> None:
        nucleotides = current_stem.nucleotides
        for idx in range(len(nucleotides)//2):
            nt1 = nucleotides[idx]
            nt2 = nucleotides[-(idx+1)]
            prob = probs[nt1.index][nt2.index]
            nt1.basepair_probability = nt2.basepair_probability = prob
        _attach_loop_probs(current_stem.child_loop)

    def _attach_loop_probs(current_loop: LoopRegion) -> None:
        if current_loop.is_root:
            nucleotides = current_loop.nucleotides
        else:
            nucleotides = current_loop.nucleotides[1:-1]
        for nt in nucleotides:
            nt.basepair_probability = probs[nt.index][nt.index]
        for stem in current_loop.child_stems:
            _attach_stem_probs(stem)

    _attach_loop_probs(root_loop)
