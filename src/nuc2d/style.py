"""Drawing style definitions for RNA secondary structure rendering.

This module provides classes for configuring the visual appearance of
rendered RNA secondary structure diagrams. Style parameters control
the appearance of graphical elements such as nucleotide nodes,
backbone and base-pair edges, labels, margins, and color mappings.

The main class, DrawingStyle, stores rendering parameters used by
SVGRenderer and other rendering backends.
"""

from dataclasses import dataclass

import matplotlib as mpl


@dataclass
class DrawingStyle:
    """Container for SVG drawing style parameters.

    Attributes
    ----------
    backbone_width : float
        Stroke width used for backbone edges.
    basepair_width : float
        Stroke width used for base-pair edges.
    node_radius : float
        Radius of nucleotide nodes.
    x_margin : float
        Horizontal margin added around the drawing area.
    y_margin : float
        Vertical margin added around the drawing area.
    node_fill : str
        Default node fill color.
    edge_color : str
        Default edge color.
    font_size : float
        Font size for nucleotide labels.
    cmap : mpl.colors.Colormap, default=mpl.colormaps["turbo"]
        Colormap used for probability visualization.
    """
    backbone_width: float = 2.0
    basepair_width: float = 1.0

    node_radius: float = 4.5

    x_margin: float = 20.0
    y_margin: float = 20.0

    node_fill: str = "black"
    edge_color: str = "black"

    font_size: float = 6.5

    cmap: mpl.colors.Colormap = mpl.colormaps["turbo"]
