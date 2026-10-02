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


Font = tuple[str, int]
"""A font as Matplotlib finds one: the path of its file, and which font
of the file it is, counted from 0. A font collection (``.ttc``, as macOS
ships many system fonts) holds several; any other file holds one."""


@cache
def find_font(font_family: str) -> Font:
    """Find the font Matplotlib selects for a font family.

    Parameters
    ----------
    font_family : str
        Font family name to search for.

    Returns
    -------
    Font
        The font Matplotlib selects. If the requested font is not
        available, Matplotlib's default font fallback is used.

    Notes
    -----
    The family is given to Matplotlib as a list, which it reads as names.
    A string alone it reads as a fontconfig pattern, in which
    ``"Arial:bold"`` is Arial in bold, so the text would be measured in a
    font that the SVG, which names the family as it is given, does not
    ask for.

    From Matplotlib 3.11, what it finds carries which font of a collection
    matched. Up to 3.10 it is the path alone, and Matplotlib reads only
    the first font of a collection, so that is the one that matched.

    Results are cached per font family.
    """
    properties = font_manager.FontProperties(family=[font_family])
    found = font_manager.findfont(properties)
    return str(found), getattr(found, "face_index", 0)


def _open(font: Font) -> TTFont:
    """Open a font: the one of its file that it names.

    fontTools refuses to open a collection without being told which of its
    fonts to read. A file holding a single font ignores the number.
    """
    path, index = font
    return TTFont(path, fontNumber=index)


@cache
def _vertical_center_ratio(font: Font) -> float:
    """Return the vertical center offset as a fraction of the font size.

    Parameters
    ----------
    font : Font
        Font used for rendering the text, as :func:`find_font` finds it.

    Returns
    -------
    float
        Offset from the baseline to the vertical center, expressed in em
        units.

    Notes
    -----
    The ratio depends only on the font, not on the font size, so it is
    cached per font and scaled by the caller. Caching on the ratio rather
    than on the finished offset means a drawing that mixes font sizes
    still reads each font only once.
    """
    ttf = _open(font)

    units_per_em = ttf["head"].unitsPerEm
    ascender = ttf["hhea"].ascent
    descender = ttf["hhea"].descent

    center = (ascender + descender) / 2

    return float(center / units_per_em)


def vertical_center_offset(
    font: Font,
    font_size: float,
) -> float:
    """Calculate the vertical center offset from the baseline.

    Parameters
    ----------
    font : Font
        Font used for rendering the text, as :func:`find_font` finds it.
    font_size : float
        Font size of the text.

    Returns
    -------
    float
        Vertical offset to apply to the text baseline so that the text
        is vertically centered according to the font's ascender and
        descender metrics.
    """
    return _vertical_center_ratio(font) * font_size


@cache
def _vertical_extent_ratios(font: Font) -> tuple[float, float]:
    """Return how far a line of text reaches above and below its baseline.

    Both are fractions of the font size, taken from the same ascender and
    descender :func:`vertical_center_offset` centers on, and both are
    positive.
    """
    ttf = _open(font)
    units_per_em = ttf["head"].unitsPerEm
    return (
        float(ttf["hhea"].ascent / units_per_em),
        float(-ttf["hhea"].descent / units_per_em),
    )


def vertical_extent(font: Font, font_size: float) -> tuple[float, float]:
    """Return how far a line of text reaches above and below its baseline.

    Parameters
    ----------
    font : Font
        Font used for rendering the text, as :func:`find_font` finds it.
    font_size : float
        Font size of the text.

    Returns
    -------
    tuple of float
        ``(above, below)``, both positive, in the units of ``font_size``.
    """
    above, below = _vertical_extent_ratios(font)
    return above * font_size, below * font_size


@cache
def _advance_ratios(font: Font) -> tuple[dict[int, float], float]:
    """Return each character's advance width, as a fraction of the size.

    Returns
    -------
    tuple
        A mapping from code point to advance width, and the advance width
        of the glyph a font draws for a character it does not have.
    """
    ttf = _open(font)
    units_per_em = ttf["head"].unitsPerEm
    metrics = ttf["hmtx"]
    advances = {
        code_point: float(metrics[glyph][0] / units_per_em)
        for code_point, glyph in ttf.getBestCmap().items()
    }
    missing = float(metrics[ttf.getGlyphOrder()[0]][0] / units_per_em)
    return advances, missing


def text_width(font: Font, text: str, font_size: float) -> float:
    """Return how wide a line of text is set in a font.

    Parameters
    ----------
    font : Font
        Font used for rendering the text, as :func:`find_font` finds it.
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
    advances, missing = _advance_ratios(font)
    return sum(advances.get(ord(char), missing) for char in text) * font_size
