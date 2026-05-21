from dataclasses import dataclass
from typing import Optional

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
    margin : float
        Margin added around the drawing area.
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
