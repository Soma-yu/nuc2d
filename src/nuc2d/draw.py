"""High-level drawing interface for nucleic acid secondary structures.

This module provides convenience functions for generating SVG drawings
directly from secondary structure strings. Parsing, annotation,
layout generation, and rendering are performed automatically.
"""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box representing the spatial dimensions of an SVG element."""

    xmin: float
    ymin: float
    width: float
    height: float


def draw_group(
    drawing: svgwrite.Drawing,
    dpp_string: str,
    sequences: Optional[list[str]] = None,
    probs: Optional[np.ndarray] = None,
    style: Optional[DrawingStyle] = None,
) -> tuple[svgwrite.container.Group, BoundingBox]:
    """Generate an SVG group and its bounding box from a secondary structure string.

    Parameters
    ----------
    drawing : svgwrite.Drawing
        The target SVG drawing instance used for element factory and defs registration.
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
    tuple[svgwrite.container.Group, BoundingBox]
        A tuple containing the generated SVG group and its bounding box.
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

    # Render the secondary structure as an independent SVG component.
    structure = render_structure(
        drawing,
        layout_result,
        style,
    )
    placed_components = [
        PlacedComponent(
            component=structure,
            x=0.0,
            y=0.0,
            scale=1.0,
        )
    ]

    total_width = structure.width
    total_height = structure.height

    # Add a colorbar when base-pair probabilities are visualized.
    if probs is not None:
        colorbar = render_colorbar(
            drawing,
            style=style,
        )
        # Match colorbar height to the structure height at base scale.
        colorbar_scale = structure.height / colorbar.height
        placed_components.append(
            PlacedComponent(
                component=colorbar,
                x=structure.width,
                y=0.0,
                scale=colorbar_scale,
            )
        )
        total_width += colorbar.width * colorbar_scale

    # Create the bounding box representing unscaled component bounds.
    bbox = BoundingBox(
        xmin=0.0,
        ymin=0.0,
        width=total_width,
        height=total_height,
    )

    # Compose all positioned components into the SVG group.
    group = drawing.g()
    compose(
        group,
        placed_components,
    )

    return group, bbox


def draw_svg(
    dpp_string: str,
    sequences: Optional[list[str]] = None,
    probs: Optional[np.ndarray] = None,
    style: Optional[DrawingStyle] = None,
    width_px: Optional[float] = None,
    height_px: Optional[float] = None,
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
    width_px : float, optional
        Width of the final SVG output (in pixels).
        If specified without height_px, height is calculated automatically to maintain aspect ratio.
    height_px : float, optional
        Height of the final SVG output (in pixels).
        If specified without width_px, width is calculated automatically to maintain aspect ratio.
        If both width_px and height_px are None, height_px defaults to 500.0.

    Returns
    -------
    svgwrite.Drawing
        Generated SVG drawing.
    """
    # Create the root SVG drawing container.
    drawing = svgwrite.Drawing()

    # Generate the component group and retrieve its bounding box.
    group, bbox = draw_group(
        drawing=drawing,
        dpp_string=dpp_string,
        sequences=sequences,
        probs=probs,
        style=style,
    )

    # Add the composed group to the main drawing.
    drawing.add(group)

    # Set the viewBox based on the unscaled bounding box.
    drawing.viewbox(bbox.xmin, bbox.ymin, bbox.width, bbox.height)

    # Calculate missing dimension to maintain aspect ratio
    aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 1.0

    # Both dimensions are None -> Fallback to default height (500.0px)
    if width_px is None and height_px is None:
        height_px = 500.0

    if width_px is not None and height_px is None:
        height_px = width_px / aspect_ratio
    elif width_px is None and height_px is not None:
        width_px = height_px * aspect_ratio

    drawing["width"] = f"{width_px}px"
    drawing["height"] = f"{height_px}px"

    return drawing
