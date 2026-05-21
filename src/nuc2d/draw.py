from typing import Optional

import svgwrite

from .parser import parse_dotbracket
from .layout import layout_structure
from .style import DrawingStyle
from .svg import render_svg


def draw_svg(
    dotbracket: str,
    sequence: Optional[str] = None,
    style: Optional[DrawingStyle] = None,
) -> svgwrite.Drawing:
    """Generate an SVG drawing from a secondary structure string.

    Parameters
    ----------
    dotbracket : str
        RNA secondary structure written in dot-bracket notation.
    sequence : str or None
        RNA sequence corresponding to the structure.
    style : DrawingStyle or None
        Drawing style configuration.

    Returns
    -------
    svgwrite.Drawing
        Generated SVG drawing.
    """

    parsed = parse_dotbracket(
        dotbracket,
        sequence=sequence,
    )

    layout_result = layout_structure(parsed)

    return render_svg(
        layout_result,
        style=style,
        sequences=[sequence] if sequence is not None else None,
    )
