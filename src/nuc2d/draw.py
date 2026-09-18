"""High-level drawing interface for nucleic acid secondary structures.

This module provides convenience functions for generating SVG drawings
directly from secondary structure strings. Parsing, annotation,
layout generation, and rendering are performed automatically.
"""

import numpy as np
import svgwrite

from .parser import parse
from .annotation import (
    attach_sequences,
    attach_equilibrium_probabilities,
)
from .layout import LayoutEngine, RadialLayoutEngine
from .style import DrawingStyle
from .svg import (
    Placement,
    SVGComponent,
    render_structure,
    render_colorbar,
    compose,
)


def draw_component(
    drawing: svgwrite.Drawing,
    dpp_string: str,
    *,
    sequences: list[str] | None = None,
    probs: np.ndarray | None = None,
    style: DrawingStyle | None = None,
    layout_engine: LayoutEngine | None = None,
    colorbar_label: str | None = None,
    add_colorbar: bool = True,
) -> SVGComponent:
    """Generate an SVG component from a secondary structure string.

    Parameters
    ----------
    drawing : svgwrite.Drawing
        The target SVG drawing instance used for element factory and defs
        registration. The component is not added to the drawing; the
        caller decides where it goes.
    dpp_string : str
        A secondary structure written in dot-parens-plus notation.
    sequences : list[str], optional
        A list of sequences corresponding to the structure.
    probs : ndarray, optional
        Base-pairing probability matrix. ``probs[i][j]`` is the equilibrium
        probability that bases ``i`` and ``j`` pair, and the diagonal
        ``probs[i][i]`` the equilibrium probability that base ``i`` is
        unpaired. When given, a colorbar is placed
        beside the structure.
    style : DrawingStyle, optional
        Drawing style configuration.
    layout_engine : LayoutEngine, optional
        Engine computing nucleotide positions. Defaults to a
        :class:`~nuc2d.layout.RadialLayoutEngine` with its own defaults.
    colorbar_label : str, optional
        Text written alongside the colorbar. Defaults to
        ``"Equilibrium probability"``. Has no effect unless ``probs`` is
        given, since the colorbar is drawn only then.
    add_colorbar : bool, default=True
        Whether to place a colorbar beside the structure. Passing False
        colors the nucleotides from ``probs`` but leaves the colorbar out,
        for a caller placing one of its own with
        :func:`~nuc2d.svg.render_colorbar`. The colorbar placed here is as
        tall as the structure, which is a poor fit for a structure much
        wider than it is tall.

    Returns
    -------
    SVGComponent
        The generated SVG group together with the bounding box it
        occupies, in the coordinate system the component was drawn in.

    Raises
    ------
    ParseError
        If ``dpp_string`` is not a well-formed secondary structure.
    """
    # Parse the secondary structure string.
    root_loop = parse(dpp_string)

    # Attach sequence and probability annotations.
    if sequences is not None:
        attach_sequences(root_loop, sequences)
    if probs is not None:
        attach_equilibrium_probabilities(root_loop, probs)

    # Compute nucleotide positions and drawing geometry.
    engine = layout_engine if layout_engine is not None else RadialLayoutEngine()
    layout_result = engine.layout(root_loop)

    # Render the secondary structure as an independent SVG component.
    structure = render_structure(
        drawing,
        layout_result,
        style=style,
    )
    placements = [
        Placement(
            component=structure,
            x=0.0,
            y=0.0,
            scale=1.0,
        )
    ]

    # Add a colorbar when base-pair probabilities are visualized.
    if probs is not None and add_colorbar:
        colorbar = render_colorbar(
            drawing,
            label=colorbar_label,
            style=style,
        )
        # Match colorbar height to the structure height, and set it beside
        # the structure's right edge.
        placements.append(
            Placement(
                component=colorbar,
                x=structure.bbox.xmax,
                y=structure.bbox.ymin,
                scale=structure.bbox.height / colorbar.bbox.height,
            )
        )

    # Compose all positioned components into a single SVG group.
    return compose(drawing.g(), placements)


def draw_svg(
    dpp_string: str,
    *,
    sequences: list[str] | None = None,
    probs: np.ndarray | None = None,
    style: DrawingStyle | None = None,
    layout_engine: LayoutEngine | None = None,
    colorbar_label: str | None = None,
    add_colorbar: bool = True,
    width_px: float | None = None,
    height_px: float | None = None,
) -> svgwrite.Drawing:
    """Generate an SVG drawing from a secondary structure string.

    Parameters
    ----------
    dpp_string : str
        A secondary structure written in dot-parens-plus notation.
    sequences : list[str], optional
        A list of sequences corresponding to the structure.
    probs : ndarray, optional
        Base-pairing probability matrix. ``probs[i][j]`` is the equilibrium
        probability that bases ``i`` and ``j`` pair, and the diagonal
        ``probs[i][i]`` the equilibrium probability that base ``i`` is
        unpaired.
    style : DrawingStyle, optional
        Drawing style configuration.
    layout_engine : LayoutEngine, optional
        Engine computing nucleotide positions. Defaults to a
        :class:`~nuc2d.layout.RadialLayoutEngine` with its own defaults.
    colorbar_label : str, optional
        Text written alongside the colorbar. Defaults to
        ``"Equilibrium probability"``. Has no effect unless ``probs`` is
        given, since the colorbar is drawn only then.
    add_colorbar : bool, default=True
        Whether to place a colorbar beside the structure.
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

    Raises
    ------
    ParseError
        If ``dpp_string`` is not a well-formed secondary structure.
    """
    # Create the root SVG drawing container.
    drawing = svgwrite.Drawing()

    # Generate the component and add it to the drawing.
    component = draw_component(
        drawing=drawing,
        dpp_string=dpp_string,
        sequences=sequences,
        probs=probs,
        style=style,
        layout_engine=layout_engine,
        colorbar_label=colorbar_label,
        add_colorbar=add_colorbar,
    )
    drawing.add(component.group)

    # Frame the drawing on exactly the area the component occupies.
    bbox = component.bbox
    drawing.viewbox(*bbox.to_viewbox())

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
