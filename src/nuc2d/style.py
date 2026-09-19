"""Drawing style definitions for secondary structure rendering.

This module provides classes for configuring the visual appearance of
rendered secondary structure diagrams. Style parameters control
the appearance of graphical elements such as nucleotide nodes,
backbone and base-pair edges, labels, margins, and color mappings.

The main class, DrawingStyle, stores rendering parameters used by
SVGRenderer and other rendering backends.
"""

from dataclasses import dataclass, field

import matplotlib as mpl


@dataclass(kw_only=True)
class DrawingStyle:
    """Container for drawing style parameters.

    Attributes
    ----------
    backbone_width : float
        Stroke width used for backbone edges.
    basepair_width : float
        Stroke width used for base-pair edges.
    backbone_dasharray : str
        Dash pattern used for backbone edges, specified as an SVG
        ``stroke-dasharray`` value.
    basepair_dasharray : str
        Dash pattern used for base-pair edges, specified as an SVG
        ``stroke-dasharray`` value.
    three_prime_arrow_length : float
        Length of the arrow drawn at each 3' terminus. The arrowhead is
        measured in stroke widths, so its size follows
        ``backbone_width`` rather than this.
    node_radius : float
        Radius of nucleotide nodes.
    x_margin : float
        Horizontal margin added around the drawing area.
    y_margin : float
        Vertical margin added around the drawing area.
    node_color : str
        Default node color.
    backbone_color : str
        Color of backbone edges. The arrow at each 3' terminus continues
        the backbone, so it is drawn in this color too.
    basepair_color : str
        Color of base-pair edges.
    font_family : str
        Font family used for nucleotide labels.
        Arial is recommended for consistent rendering in PowerPoint.
    node_font_size : float
        Font size of the base letter drawn inside a node.
    cmap : mpl.colors.Colormap, default=mpl.colormaps["turbo"]
        Colormap used for probability visualization.

    colorbar_aspect_ratio : float
        Aspect ratio of the colorbar's bar: its width divided by its
        height.
    colorbar_tick_length : float
        Length of colorbar tick marks.
    colorbar_tick_font_size : float
        Font size used for colorbar tick labels.
    colorbar_label_font_size : float
        Font size used for the colorbar label.
    """
    backbone_width: float = 2.0
    basepair_width: float = 1.5
    backbone_dasharray: str = "1,0"
    basepair_dasharray: str = "1,1"

    three_prime_arrow_length: float = 7.0

    node_radius: float = 4.2

    x_margin: float = 20.0
    y_margin: float = 20.0

    node_color: str = "black"
    backbone_color: str = "black"
    basepair_color: str = "black"

    font_family: str = "Arial"
    node_font_size: float = 6.5

    cmap: mpl.colors.Colormap = field(
        default_factory=lambda: mpl.colormaps["turbo"]
    )

    colorbar_aspect_ratio: float = 1 / 30

    colorbar_tick_length: float = 5.0
    colorbar_tick_font_size: float = 12.0
    colorbar_label_font_size: float = 15.0
