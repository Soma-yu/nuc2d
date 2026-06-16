"""High-level drawing interface for nucleic acid secondary structures.

This module provides convenience functions for generating SVG drawings
directly from secondary structure strings. Parsing, annotation,
layout generation, and rendering are performed automatically.
"""

from typing import Optional

import numpy as np
import svgwrite

from .parser import parse
from .annotation import (
    attach_sequences,
    attach_basepair_probabilities,
)
from .layout import RadialLayoutEngine
from .style import DrawingStyle
from .svg import (
    PlacedComponent,
    render_structure,
    render_colorbar,
    compose,
)


def draw_svg(
    dpp_string: str,
    sequences: Optional[list[str]] = None,
    probs: Optional[np.ndarray] = None,
    style: Optional[DrawingStyle] = None,
    target_height: float = 500.0,
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
    target_height : float, default=500.0
        Target height of the rendered secondary structure in the SVG
        coordinate system.

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

    # Create an empty SVG drawing that will hold all rendered components.
    svg_drawing = svgwrite.Drawing()

    # Render the RNA secondary structure as an independent SVG component.
    structure = render_structure(
        svg_drawing,
        layout_result,
        style,
    )
    placed_components = [
        PlacedComponent(
            component=structure,
            x=0.0,
            y=0.0,
            scale=target_height / structure.height,
        )
    ]

    # Add a colorbar when base-pair probabilities are visualized.
    if probs is not None:
        colorbar = render_colorbar(
            svg_drawing,
            style=style,
        )
        placed_components.append(
            PlacedComponent(
                component=colorbar,
                x=placed_components[0].width,
                y=0.0,
                scale=target_height / colorbar.height,
            )
        )

    # Compose all positioned components into the final SVG drawing.
    compose(
        svg_drawing,
        placed_components,
    )
    return svg_drawing
