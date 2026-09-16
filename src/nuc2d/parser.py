"""Utilities for parsing nucleic acid secondary structure strings.
"""

from __future__ import annotations

from typing import Optional

from .structure import (
    Nucleotide, StemRegion, LoopRegion
)

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

    def consume_run(self, char: str, max_count: Optional[int] = None) -> int:
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
                # End of this loop region
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
        # The last nucleotide of the last strand is a 3' terminus.
        root_loop.nucleotides[-1].is_three_prime = True
        return root_loop

def parse(dpp_string: str):
    return _Parser(dpp_string).parse()
