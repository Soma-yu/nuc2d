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
    font = TTFont(font_path)

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
