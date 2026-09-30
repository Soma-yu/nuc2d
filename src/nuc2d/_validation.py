"""Validation of what a caller passes in.

Each check is made where a value is given, so that a mistake is reported
at the line that made it rather than wherever the value is first used. The
checks that more than one module makes are here, so that a rule is worded
the same way wherever it applies.

Each check of a value returns it, a number as a float and a string as a
str, and what is kept is what the check returns. A NumPy scalar or a
Fraction would otherwise reach the SVG writer as it is, and the writer
refuses it. A subclass of str would reach it too, and the writer asks it
for its text with str(), which an Enum with str mixed in answers with
the member's name instead of the text that was checked.

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
    size is a mistake, not a size of 1. A number too large for a float,
    such as ``10**400``, is returned as an infinity of its sign, so that
    it is refused as a number that is not finite.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a number; got {type(value).__name__}."
        )
    try:
        return float(value)
    except OverflowError:
        return -math.inf if value < 0 else math.inf


def check_finite(name: str, value: object) -> float:
    """Raise unless ``value`` is a finite real number.

    Return it as a float.
    """
    number = _real(name, value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number; got {value!r}.")
    return number


def check_finite_positive(name: str, value: object) -> float:
    """Raise unless ``value`` is a finite real number greater than 0.

    Return it as a float.
    """
    number = _real(name, value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(
            f"{name} must be a finite number greater than 0; got {value!r}."
        )
    return number


def check_finite_non_negative(name: str, value: object) -> float:
    """Raise unless ``value`` is a finite real number of at least 0.

    Return it as a float.
    """
    number = _real(name, value)
    if not (math.isfinite(number) and number >= 0):
        raise ValueError(
            f"{name} must be a finite number of at least 0; got {value!r}."
        )
    return number


# The Unicode categories of the characters a single line of text cannot
# hold: control characters, among them the tab, "\n" and "\r"; the line
# and paragraph separators; and surrogates. SVG shows a tab as a space
# while the text is measured with the tab, XML cannot hold the other
# control characters at all, and UTF-8 cannot encode a surrogate.
_NOT_IN_A_SINGLE_LINE = frozenset({"Cc", "Zl", "Zp", "Cs"})

# XML cannot hold U+FFFE or U+FFFF either. They are noncharacters, which
# Unicode leaves unassigned for good, so none of those categories has them.
_NONCHARACTERS_NOT_IN_XML = frozenset({"\ufffe", "\uffff"})


def exact_str(value: str) -> str:
    """Return ``value`` as an exact str: of the type str, not a subclass.

    Its characters are kept as they are. str() would not do: it asks a
    subclass for its own __str__, and an Enum with str mixed in answers
    with the member's name. str.__str__ returns the characters the string
    holds.
    """
    return str.__str__(value)


def check_single_line_text(name: str, value: object) -> str:
    """Raise unless ``value`` is text that can be drawn on a single line.

    That is a non-empty string without line breaks, tabs or other control
    characters. Any other character, in any script, is drawn as written.
    The text is returned, as a str.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string; got {type(value).__name__}."
        )
    text = exact_str(value)
    if not text:
        raise ValueError(f"{name} is empty.")
    if any(
        unicodedata.category(c) in _NOT_IN_A_SINGLE_LINE
        or c in _NONCHARACTERS_NOT_IN_XML
        for c in text
    ):
        raise ValueError(
            f"{name} must be a single line of text, without line breaks, "
            f"tabs or other control characters; got {value!r}."
        )
    return text


def check_font_family(name: str, value: object) -> str:
    """Raise unless ``value`` is the name of one font family.

    It is single-line text, as :func:`check_single_line_text` checks, and
    more than blank. The font is looked up by this name to measure the
    text, so a CSS list such as ``"Arial, sans-serif"`` is refused too: it
    would be measured with whatever font the lookup falls back to, while
    the SVG asked for another. The name is returned.
    """
    family = check_single_line_text(name, value)
    if not family.strip():
        raise ValueError(
            f"{name} is blank; give a font family, such as 'Arial'."
        )
    if "," in family:
        first = family.split(",")[0].strip()
        raise ValueError(
            f"{name} must be one font family, not a list; got {value!r}. "
            f"Give one name, such as {first!r}."
        )
    return family


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
