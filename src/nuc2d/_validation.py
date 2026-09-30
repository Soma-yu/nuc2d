"""Validation of what a caller passes in.

Each check is made where a value is given, so that a mistake is reported
at the line that made it rather than wherever the value is first used. The
checks that more than one module makes are here, so that a rule is worded
the same way wherever it applies.

This module imports nothing from the package, so that any module can use
it.
"""

import difflib
import math
import numbers
import unicodedata
from collections.abc import Collection


def _real(name: str, value: object) -> float:
    """Return ``value`` as a float, or raise unless it is a real number.

    A bool is refused, although Python counts it as one: True given for a
    size is a mistake, not a size of 1.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a number; got {type(value).__name__}."
        )
    return float(value)


def check_finite(name: str, value: object) -> None:
    """Raise unless ``value`` is a finite real number."""
    if not math.isfinite(_real(name, value)):
        raise ValueError(f"{name} must be a finite number; got {value!r}.")


def check_finite_positive(name: str, value: object) -> None:
    """Raise unless ``value`` is a finite real number greater than 0."""
    number = _real(name, value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(
            f"{name} must be a finite number greater than 0; got {value!r}."
        )


def check_finite_non_negative(name: str, value: object) -> None:
    """Raise unless ``value`` is a finite real number of at least 0."""
    number = _real(name, value)
    if not (math.isfinite(number) and number >= 0):
        raise ValueError(
            f"{name} must be a finite number of at least 0; got {value!r}."
        )


# The Unicode categories of the characters a single line of text cannot
# hold: control characters, among them the tab, "\n" and "\r"; the line
# and paragraph separators; and surrogates. SVG shows a tab as a space
# while the text is measured with the tab, XML cannot hold the other
# control characters at all, and UTF-8 cannot encode a surrogate.
_NOT_IN_A_SINGLE_LINE = frozenset({"Cc", "Zl", "Zp", "Cs"})


def _string(name: str, value: object) -> str:
    """Return ``value``, or raise unless it is a string."""
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string; got {type(value).__name__}."
        )
    return value


def _check_single_line(name: str, text: str) -> None:
    """Raise if ``text`` is empty or holds a character a line cannot."""
    if not text:
        raise ValueError(f"{name} is empty.")
    if any(unicodedata.category(c) in _NOT_IN_A_SINGLE_LINE for c in text):
        raise ValueError(
            f"{name} must be a single line of text, without line breaks, "
            f"tabs or other control characters; got {text!r}."
        )


def check_single_line_text(name: str, value: object) -> None:
    """Raise unless ``value`` is text that can be drawn on a single line.

    That is a non-empty string without line breaks, tabs or other control
    characters. Any other character, in any script, is drawn as written.
    """
    _check_single_line(name, _string(name, value))


def check_font_family(name: str, value: object) -> None:
    """Raise unless ``value`` is the name of one font family.

    It is single-line text, as :func:`check_single_line_text` checks, and
    more than blank. The font is looked up by this name to measure the
    text, so a CSS list such as ``"Arial, sans-serif"`` is refused too: it
    would be measured with whatever font the lookup falls back to, while
    the SVG asked for another.
    """
    family = _string(name, value)
    _check_single_line(name, family)
    if not family.strip():
        raise ValueError(
            f"{name} is blank; give a font family, such as 'Arial'."
        )
    if "," in family:
        first = family.split(",")[0].strip()
        raise ValueError(
            f"{name} must be one font family, not a list; got {family!r}. "
            f"Give one name, such as {first!r}."
        )


def check_attribute(owner: object, name: str, names: Collection[str]) -> None:
    """Raise unless ``name`` is one of the attributes ``owner`` has.

    Assigning to a misspelt attribute would otherwise make a new one that
    nothing reads, and the drawing would silently stay as it was.
    """
    if name in names:
        return
    message = f"{type(owner).__name__!r} object has no attribute {name!r}"
    close = difflib.get_close_matches(name, names, n=1)
    if close:
        message += f". Did you mean: {close[0]!r}?"
    raise AttributeError(message)
