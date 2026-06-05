"""High-level drawing interface for nucleic acid secondary structures.

This module provides convenience functions for generating SVG drawings
directly from secondary structure strings. Parsing, annotation,
layout generation, and rendering are performed automatically.
"""

from typing import Optional

import svgwrite

import numpy as np

from .parser import parse
from .annotation import attach_sequences, attach_basepair_probabilities
from .layout import RadialLayoutEngine
from .style import DrawingStyle
from .svg import render as svg_render


def draw_svg(
    dpp_string: str,
    sequences: Optional[list[str]] = None,
    probs: Optional[np.ndarray] = None,
    style: Optional[DrawingStyle] = None,
) -> svgwrite.Drawing:
    """Generate an SVG drawing from a secondary structure string.

    Parameters
    ----------
    dpp_string : str
        A secondary structure written in dot-parens-plus notation.
    sequences : list[str], optional
        A list of sequences corresponding to the structure.
    probs : ndarray, optional
        Base-pair probability matrix.
    style : DrawingStyle, optional
        Drawing style configuration.

    Returns
    -------
    svgwrite.Drawing
        Generated SVG drawing.
    """
    # Parse the secondary structure string.
    root_loop = parse(dpp_string)

    # Attach sequence and probability annotations.
    if sequences is not None:
        attach_sequences(root_loop, sequences)
    if probs is not None:
        attach_basepair_probabilities(root_loop, probs)

    # Compute nucleotide positions and drawing geometry.
    layout_result = RadialLayoutEngine().layout(root_loop)

    # Convert the layout into an SVG drawing.
    return svg_render(
        layout_result,
        style=style,
    )
