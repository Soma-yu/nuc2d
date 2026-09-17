"""Data structures for representing secondary structures.

This module provides classes for representing secondary structures,
including nucleotides and structural regions such as stems and loops.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


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

    base: str | None = None
    basepair_probability: float | None = None
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
        Loop region enclosed by this stem. None only while the parser is
        still building the stem; every stem in a parsed structure has one.
    """
    child_loop: LoopRegion | None = None

@dataclass
class LoopRegion(Region):
    """Class representing a loop region (non-stem region) in a secondary structure.

    Attributes
    ----------
    child_stems : list[StemRegion]
        Stem regions connected to this loop. Empty when the loop closes no
        stems, as in a structure with no base pairs.
    is_root : bool
        Whether this loop region is the root of the secondary structure tree.
    """
    child_stems: list[StemRegion] = field(default_factory=list)
    is_root: bool = False

    @property
    def is_stacked(self) -> bool:
        """Return whether two stem regions stack directly across this loop.

        A loop region of four nucleotides with no unpaired nucleotide of its
        own carries no loop of its own shape: the closing base pair of one
        stem and the opening base pair of the next meet across it, so the two
        stems continue as a single helix.
        """
        frag1 = len(self.nucleotides) == 4 and len(self.child_stems) == 2 and self.is_root
        frag2 = len(self.nucleotides) == 4 and len(self.child_stems) == 1 and not self.is_root
        return frag1 or frag2


def iter_nucleotides(root_loop: LoopRegion) -> Iterator[Nucleotide]:
    """Yield every nucleotide of a secondary structure exactly once.

    Parameters
    ----------
    root_loop : LoopRegion
        Root loop region of the secondary structure tree.

    Yields
    ------
    Nucleotide
        Each nucleotide in the structure. The order follows the structure
        tree rather than the sequence.

    Notes
    -----
    A region shares its first and last nucleotide with the region it is
    nested in, so every region except the root contributes only the
    nucleotides between them. That is what keeps each nucleotide from
    being yielded twice.
    """
    def _from_stem(current_stem: StemRegion) -> Iterator[Nucleotide]:
        yield from current_stem.nucleotides[1:-1]
        yield from _from_loop(current_stem.child_loop)

    def _from_loop(current_loop: LoopRegion) -> Iterator[Nucleotide]:
        if current_loop.is_root:
            yield from current_loop.nucleotides
        else:
            yield from current_loop.nucleotides[1:-1]
        for stem in current_loop.child_stems:
            yield from _from_stem(stem)

    yield from _from_loop(root_loop)
