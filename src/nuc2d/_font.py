"""Font utilities for text rendering.

This module provides utilities for locating font files and calculating
text positioning parameters from font metrics. The utilities are
renderer-independent and can be used by different drawing backends.

Both resolving a font family and reading a font file are expensive
compared to emitting a single SVG element, and a drawing resolves the
same font once per nucleotide. Results are therefore cached; the cached
values are plain numbers and paths, so no font object is kept alive.
"""

from functools import cache

from matplotlib import font_manager
from fontTools.ttLib import TTFont


@cache
def find_font_path(font_family: str) -> str:
    """Find the font file corresponding to a font family.

    Parameters
    ----------
    font_family : str
        Font family name to search for.

    Returns
    -------
    str
        Path to the font file selected by Matplotlib. If the requested
        font is not available, Matplotlib's default font fallback is used.

    Notes
    -----
    Results are cached per font family.
    """
    return str(font_manager.findfont(font_family))


def _open(font_path: str) -> TTFont:
    """Open a font file, taking the first font of a collection.

    A font collection (``.ttc``, as macOS ships many system fonts) holds
    several fonts, and fontTools refuses to open one without being told
    which. The first is the one to measure: Matplotlib, which chose the
    file in :func:`find_font_path`, reads only the first font of a
    collection, so that is the font whose family name matched. A file
    holding a single font ignores the number.
    """
    return TTFont(font_path, fontNumber=0)


@cache
def _vertical_center_ratio(font_path: str) -> float:
    """Return the vertical center offset as a fraction of the font size.

    Parameters
    ----------
    font_path : str
        Path to the font file used for rendering the text.

    Returns
    -------
    float
        Offset from the baseline to the vertical center, expressed in em
        units.

    Notes
    -----
    The ratio depends only on the font, not on the font size, so it is
    cached per font file and scaled by the caller. Caching on the ratio
    rather than on the finished offset means a drawing that mixes font
    sizes still reads each font file only once.
    """
    font = _open(font_path)

    units_per_em = font["head"].unitsPerEm
    ascender = font["hhea"].ascent
    descender = font["hhea"].descent

    center = (ascender + descender) / 2

    return float(center / units_per_em)


def vertical_center_offset(
    font_path: str,
    font_size: float,
) -> float:
    """Calculate the vertical center offset from the baseline.

    Parameters
    ----------
    font_path : str
        Path to the font file used for rendering the text.
    font_size : float
        Font size of the text.

    Returns
    -------
    float
        Vertical offset to apply to the text baseline so that the text
        is vertically centered according to the font's ascender and
        descender metrics.
    """
    return _vertical_center_ratio(font_path) * font_size


@cache
def _vertical_extent_ratios(font_path: str) -> tuple[float, float]:
    """Return how far a line of text reaches above and below its baseline.

    Both are fractions of the font size, taken from the same ascender and
    descender :func:`vertical_center_offset` centres on, and both are
    positive.
    """
    font = _open(font_path)
    units_per_em = font["head"].unitsPerEm
    return (
        float(font["hhea"].ascent / units_per_em),
        float(-font["hhea"].descent / units_per_em),
    )


def vertical_extent(font_path: str, font_size: float) -> tuple[float, float]:
    """Return how far a line of text reaches above and below its baseline.

    Parameters
    ----------
    font_path : str
        Path to the font file used for rendering the text.
    font_size : float
        Font size of the text.

    Returns
    -------
    tuple of float
        ``(above, below)``, both positive, in the units of ``font_size``.
    """
    above, below = _vertical_extent_ratios(font_path)
    return above * font_size, below * font_size


@cache
def _advance_ratios(font_path: str) -> tuple[dict[int, float], float]:
    """Return each character's advance width, as a fraction of the size.

    Returns
    -------
    tuple
        A mapping from code point to advance width, and the advance width
        of the glyph a font draws for a character it does not have.
    """
    font = _open(font_path)
    units_per_em = font["head"].unitsPerEm
    metrics = font["hmtx"]
    advances = {
        code_point: float(metrics[glyph][0] / units_per_em)
        for code_point, glyph in font.getBestCmap().items()
    }
    missing = float(metrics[font.getGlyphOrder()[0]][0] / units_per_em)
    return advances, missing


def text_width(font_path: str, text: str, font_size: float) -> float:
    """Return how wide a line of text is set in a font.

    Parameters
    ----------
    font_path : str
        Path to the font file used for rendering the text.
    text : str
        The text, on one line.
    font_size : float
        Font size of the text.

    Returns
    -------
    float
        The sum of the characters' advance widths, in the units of
        ``font_size``.

    Notes
    -----
    Kerning is not applied, so a pair such as "AV", which a renderer sets
    slightly closer, is measured a little wide.
    """
    advances, missing = _advance_ratios(font_path)
    return sum(advances.get(ord(char), missing) for char in text) * font_size
