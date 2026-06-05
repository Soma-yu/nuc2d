"""Utilities for attaching annotations to RNA secondary structures.

This module provides functions for adding biological or visualization-
related annotations to parsed secondary structure objects. Examples
include nucleotide sequences, base-pair probabilities, and other
metadata associated with nucleotides or structural elements.

Annotations are applied after parsing and before layout or rendering,
allowing structural topology and auxiliary information to remain
separated.
"""

import numpy as np

from .structure import (
    StemRegion,
    LoopRegion,
)

def attach_sequences(root_loop: LoopRegion, sequences: list[str]):
    """Attach nucleotide sequences to a secondary structure.

    Parameters
    ----------
    root_loop : LoopRegion
        Root loop of the secondary structure.
    sequences : list[str]
        Nucleotide sequences for all strands.
    """
    def _attach_stem_sequences(current_stem: StemRegion):
        for nt in current_stem.nucleotides[1:-1]:
            nt.base = sequences[nt.strand_index][nt.index_in_strand]
        _attach_loop_sequences(current_stem.child_loop)
    
    def _attach_loop_sequences(current_loop: LoopRegion):
        if current_loop.is_root:
            nucleotides = current_loop.nucleotides
        else:
            nucleotides = current_loop.nucleotides[1:-1]
        for nt in nucleotides:
            nt.base = sequences[nt.strand_index][nt.index_in_strand]
        for stem in current_loop.child_stems:
            _attach_stem_sequences(stem)
    
    _attach_loop_sequences(root_loop)

def attach_basepair_probabilities(root_loop: LoopRegion, probs: np.ndarray):
    """Attach base-pair probabilities to a secondary structure.

    Parameters
    ----------
    root_loop : LoopRegion
        Root loop of the secondary structure.
    probs : ndarray
        Base-pair probability matrix. Element (i, j) gives the probability
        that nucleotide i pairs with nucleotide j. Diagonal elements give
        the probabilities that nucleotides remain unpaired.
    """
    def _attach_stem_probs(current_stem: StemRegion):
        nucleotides = current_stem.nucleotides
        for idx in range(len(nucleotides)//2):
            nt1 = nucleotides[idx]
            nt2 = nucleotides[-(idx+1)]
            prob = probs[nt1.index][nt2.index]
            nt1.basepair_probability = nt2.basepair_probability = prob
        _attach_loop_probs(current_stem.child_loop)
    
    def _attach_loop_probs(current_loop: LoopRegion):
        if current_loop.is_root:
            nucleotides = current_loop.nucleotides
        else:
            nucleotides = current_loop.nucleotides[1:-1]
        for nt in nucleotides:
            nt.basepair_probability = probs[nt.index][nt.index]
        for stem in current_loop.child_stems:
            _attach_stem_probs(stem)
    
    _attach_loop_probs(root_loop)
