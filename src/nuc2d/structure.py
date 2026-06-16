"""Data structures for representing secondary structures.

This module provides classes for representing secondary structures,
including nucleotides and structural regions such as stems and loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(eq=False)
class Nucleotide:
    """A class representing a nucleotide.
    
    Parameters
    ----------
    strand_index : int
        Index of the strand this nucleotide belongs to among all strands (0-based).
    index_in_strand : int
        Index within the strand this nucleotide belongs to (0-based).
    index : int
        Index of this nucleotide among all nucleotides in the secondary structure (0-based).

    Attributes
    ----------
    base : str or None
        The nucleotide base.
    basepair_probability : float or None
        If the nucleotide is paired, this is the probability of pairing with its partner.
        If unpaired, this is the probability of remaining unpaired.
    is_three_prime : bool, default=False
        Whether this nucleotide corresponds to the 3' terminus.
    """
    strand_index: int
    index_in_strand: int
    index: int

    base: Optional[str] = None
    basepair_probability: Optional[float] = None
    is_three_prime: bool = False

    @property
    def is_five_prime(self) -> bool:
        """Return whether this nucleotide corresponds to the 5' terminus."""
        return self.index_in_strand == 0

@dataclass
class Region:
    """Base class for regions in a secondary structure.
    
    Attributes
    ----------
    nucleotides : list[Nucleotide]
        List of nucleotides that belong to this region.
    """
    nucleotides: list[Nucleotide] = field(default_factory=list)

@dataclass
class StemRegion(Region):
    """Class representing a stem region in a secondary structure.
    
    Attributes
    ----------
    child_loop : LoopRegion | None
        Child region in the secondary structure tree.
        For a stem region, this is assumed to be a single LoopRegion instance
        corresponding to the loop connected to this stem.
        None if no such loop exists.
    """
    child_loop: Optional[LoopRegion] = None

@dataclass
class LoopRegion(Region):
    """Class representing a loop region (non-stem region) in a secondary structure.
    
    Attributes
    ----------
    child_stems : list[StemRegion] | None
        Child regions in the secondary structure tree.
        For a loop region, this is assumed to be a list of one or more
        StemRegion instances corresponding to stems connected to this loop.
        None if no such stems exist.
    is_loot : bool
        Whether this loop region is the root of the secondary structure tree.
    """
    child_stems: list[StemRegion] = field(default_factory=list)
    is_root: bool = False

    @property
    def is_hinge(self) -> bool:
        """Return whether this loop region forms a hinge-like structure between stem regions."""
        frag1 = len(self.nucleotides) == 4 and len(self.child_stems) == 2 and self.is_root
        frag2 = len(self.nucleotides) == 4 and len(self.child_stems) == 1 and not self.is_root
        return frag1 or frag2
