"""Utilities for parsing nucleic acid secondary structure strings.
"""

from __future__ import annotations

from .structure import (
    Nucleotide, StemRegion, LoopRegion, iter_nucleotides, iter_stems
)

# A hairpin loop is one backbone turning back on itself, which it cannot do
# in fewer nucleotides than this. Structure prediction tools impose the same
# minimum, so a tighter loop comes from a string written by hand.
MINIMUM_HAIRPIN_LOOP_SIZE = 3


class ParseError(Exception):
    """Exception raised when an error occurs during parsing of a string."""
    pass

class _Parser:
    def __init__(self, dpp_string: str):
        self.dpp_string: str = dpp_string
        self.char_index: int = 0
        self.strand_index: int = 0
        self.nt_index = 0
        self.index_in_strand: int = 0

    def is_eof(self) -> bool:
        """Return whether the entire input string has been consumed."""
        return self.char_index == len(self.dpp_string)

    def peek(self) -> str:
        """Return the next character in the input string without consuming it.

        Raises
        ------
        ParseError
            If the end of the input string is reached.
        """
        if self.is_eof():
            raise ParseError("Unexpected end of input string.")
        return self.dpp_string[self.char_index]

    def consume(self, char: str) -> None:
        """Consume a single expected character from the input string.

        Parameters
        ----------
        char : str
            Expected character to consume.

        Raises
        ------
        ParseError
            If the next character in the input string does not match the expected one.
        """
        actual = self.peek()
        if actual != char:
            raise ParseError(
                f"Unexpected '{actual}' at position {self.char_index}. Expected '{char}'."
            )
        self.char_index += 1

    def consume_run(self, char: str, max_count: int | None = None) -> int:
        """Consume a run of one or more occurrences of a character.

        Parameters
        ----------
        char : str
            Character to consume.
        max_count : int | None
            Maximum number of characters to consume. If None, the run is
            consumed until a different character or the end of the input
            string is reached.

        Returns
        -------
        int
            Number of characters consumed.

        Raises
        ------
        ParseError
            If the next character in the input string is not the expected one.
        """
        if max_count is None:
            max_count = len(self.dpp_string) - self.char_index

        self.consume(char)
        count = 1

        while count < max_count:
            if self.is_eof():
                break
            if self.dpp_string[self.char_index] != char:
                break
            self.consume(char)
            count += 1

        return count

    def finish(self) -> None:
        """Ensure that the entire input string has been consumed.

        Raises
        ------
        ParseError
            If unconsumed characters remain in the input string.
        """
        if not self.is_eof():
            raise ParseError(
                f"Unexpected '{self.dpp_string[self.char_index]}' at position {self.char_index}."
            )
        return None

    def advance_strand(self) -> None:
        """Advance the parser to the next strand in the input structure."""
        self.strand_index += 1
        self.index_in_strand = 0
        return None

    def create_nucleotide(self) -> Nucleotide:
        """Create a new Nucleotide at the current parser position."""
        nt = Nucleotide(
            strand_index=self.strand_index, index=self.nt_index, index_in_strand=self.index_in_strand
        )
        self.nt_index += 1
        self.index_in_strand += 1
        return nt

    def parse_stem(self, current_stem: StemRegion) -> None:
        """Parse a stem region.

        Parameters
        ----------
        current_stem : StemRegion
            Object holding information about the stem region currently being parsed.
        """
        # Parse consecutive "(" characters
        pairing_stack = []
        for _ in range(self.consume_run("(")):
            pairing_stack.append(self.create_nucleotide())
        # Parse child loop region in the second structure tree
        child_loop = LoopRegion(nucleotides=[pairing_stack[-1]])
        self.parse_loop(child_loop)
        # Parse the same number of ")" as previously parsed "("
        while pairing_stack:
            # Parse consecutive ")" characters
            n_close_parens = self.consume_run(")", len(pairing_stack))
            for idx in range(n_close_parens):
                nt = self.create_nucleotide()
                pairing_stack.append(nt)
                if idx == 0:
                    child_loop.nucleotides.append(nt)
            paired_stack  = pairing_stack[-n_close_parens*2:]
            pairing_stack = pairing_stack[:-n_close_parens*2]
            if pairing_stack:
                # Stem region formed by a subset of previously parsed "("
                child_stem = StemRegion(nucleotides=paired_stack, child_loop=child_loop)
                # Parse child loop region of the stem region above
                child_loop = LoopRegion(
                    nucleotides=[pairing_stack[-1], paired_stack[0], paired_stack[-1]],
                    child_stems=[child_stem],
                )
                self.parse_loop(child_loop)
            else:
                current_stem.nucleotides = paired_stack
                current_stem.child_loop  = child_loop

    def parse_loop(self, current_loop: LoopRegion) -> None:
        """Parse a loop region.

        Parameters
        ----------
        current_loop : LoopRegion
            Object holding information about the loop region currently being parsed.

        Raises
        ------
        ParseError
            If the input string is not a well-formed secondary structure.
        """
        while not self.is_eof():
            if self.peek() == "+":
                if not current_loop.nucleotides:
                    raise ParseError(
                        f"Unexpected '+' at position {self.char_index}. "
                        "A strand break must be preceded by at least one nucleotide."
                    )
                # Mark as the 3' terminus
                current_loop.nucleotides[-1].is_three_prime = True
                # Move to the next strand
                self.consume("+")
                self.advance_strand()

            if self.peek() == "(":
                # Parse child stem region
                child_stem = StemRegion()
                current_loop.child_stems.append(child_stem)
                self.parse_stem(child_stem)
                current_loop.nucleotides.extend(
                    [child_stem.nucleotides[0], child_stem.nucleotides[-1]]
                )
            elif self.peek() == ")":
                # End of this loop region.
                # A loop closing no stem of its own is a hairpin: the backbone
                # turns back here. A loop the strands break inside is two
                # backbones meeting instead, so the strand must not have
                # changed since the loop opened.
                is_hairpin = (
                    not current_loop.is_root
                    and not current_loop.child_stems
                    and self.strand_index == current_loop.nucleotides[0].strand_index
                )
                # Only the opening nucleotide and the unpaired ones are in the
                # list so far; the closing one is added by parse_stem.
                loop_size = len(current_loop.nucleotides) - 1
                if is_hairpin and loop_size < MINIMUM_HAIRPIN_LOOP_SIZE:
                    raise ParseError(
                        f"The hairpin loop closing at position "
                        f"{self.char_index} holds {loop_size} nucleotide(s). "
                        f"A hairpin loop needs at least "
                        f"{MINIMUM_HAIRPIN_LOOP_SIZE}, because a backbone "
                        "cannot turn back on itself in fewer."
                    )
                break
            else:
                # Parse unpaired region
                if self.peek() != ".":
                    raise ParseError(
                        f"Unexpected {self.peek()!r} at position {self.char_index}. "
                        "Expected one of '.', '(', ')' or '+'."
                    )
                for _ in range(self.consume_run(".")):
                    current_loop.nucleotides.append(self.create_nucleotide())

    def parse(self) -> LoopRegion:
        """Parse a string representing a secondary structure.

        Returns
        -------
        LoopRegion
            The root loop region of the secondary structure tree.

        Raises
        ------
        ParseError
            If the input string is not a well-formed secondary structure.
        """
        root_loop = LoopRegion(is_root=True)
        self.parse_loop(root_loop)
        # Report leftover input before inspecting the parsed nucleotides,
        # so that an unbalanced ")" is named at its own position.
        self.finish()
        if not root_loop.nucleotides:
            raise ParseError("Empty secondary structure.")
        _check_strands_are_connected(root_loop)
        # The last nucleotide of the last strand is a 3' terminus.
        root_loop.nucleotides[-1].is_three_prime = True
        return root_loop

class _StrandGroups:
    """Strands grouped by the base pairs that join them.

    Each strand starts in a group of its own, and :meth:`join` merges the
    groups of two strands that a base pair holds together.

    Notes
    -----
    This is the disjoint-set forest usually called union-find. A group is
    a tree of strands, each pointing at its parent, with the strand at the
    root pointing at itself; that root stands for the whole group, so two
    strands share a group exactly when they have the same root. Merging
    two groups is one assignment: point the root of one at the root of
    the other.

    Two habits keep the trees flat, which is what makes lookups cheap.
    :meth:`root_of` points every strand it passes at its grandparent,
    halving the path it just walked, and :meth:`join` hangs the smaller
    group under the larger one.

    The operations are named ``find`` and ``union`` in the literature.
    They are named for what they return and what they do here, because
    ``find`` alone says nothing about what is found.
    """

    def __init__(self, strand_count: int) -> None:
        # Every strand is the root of its own group to begin with.
        self.parent_of = list(range(strand_count))
        self.group_size_at = [1] * strand_count

    def root_of(self, strand: int) -> int:
        """Return the strand at the root of this strand's group."""
        while self.parent_of[strand] != strand:
            self.parent_of[strand] = self.parent_of[self.parent_of[strand]]
            strand = self.parent_of[strand]
        return strand

    def join(self, one: int, other: int) -> None:
        """Merge the groups of two strands."""
        one_root, other_root = self.root_of(one), self.root_of(other)
        if one_root == other_root:
            return
        if self.group_size_at[one_root] < self.group_size_at[other_root]:
            one_root, other_root = other_root, one_root
        self.parent_of[other_root] = one_root
        self.group_size_at[one_root] += self.group_size_at[other_root]

    def groups(self) -> list[list[int]]:
        """Return the strands of each group, in the order the strands appear."""
        grouped: dict[int, list[int]] = {}
        for strand in range(len(self.parent_of)):
            grouped.setdefault(self.root_of(strand), []).append(strand)
        return list(grouped.values())


def _check_strands_are_connected(root_loop: LoopRegion) -> None:
    """Raise unless base pairs join every strand into one complex.

    Parameters
    ----------
    root_loop : LoopRegion
        Root loop region of the parsed structure.

    Raises
    ------
    ParseError
        If the strands fall into more than one group, where two strands
        are in the same group when base pairs join them, directly or
        through other strands.

    Notes
    -----
    A secondary structure describes a single complex, and a complex is
    held together by its base pairs. Strands that no base pair reaches
    are separate molecules that happen to share a string.
    """
    strand_count = 1 + max(
        nt.strand_index for nt in iter_nucleotides(root_loop)
    )
    strand_groups = _StrandGroups(strand_count)

    for stem in iter_stems(root_loop):
        nucleotides = stem.nucleotides
        for one, other in zip(nucleotides, reversed(nucleotides)):
            strand_groups.join(one.strand_index, other.strand_index)

    groups = strand_groups.groups()
    if len(groups) > 1:
        listed = " and ".join(
            "(" + ", ".join(str(strand) for strand in group) + ")"
            for group in groups
        )
        raise ParseError(
            f"The {strand_count} strands fall into {len(groups)} groups that "
            f"no base pair joins: {listed}. A secondary structure describes "
            "one complex, so every strand must be reachable from every other "
            "through base pairs."
        )


def parse(dpp_string: str):
    return _Parser(dpp_string).parse()
