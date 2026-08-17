"""Font utilities for text rendering.

This module provides utilities for locating font files and calculating
text positioning parameters from font metrics. The utilities are
renderer-independent and can be used by different drawing backends.
"""

from matplotlib import font_manager
from fontTools.ttLib import TTFont


def find_font(font_family: str) -> str:
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
    """
    return font_manager.findfont(font_family)


def get_vertical_center_offset(
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
    font = TTFont(font_path)

    units_per_em = font["head"].unitsPerEm
    ascender = font["hhea"].ascent
    descender = font["hhea"].descent

    center = (ascender + descender) / 2

    return center / units_per_em * font_size
