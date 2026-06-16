"""SVG rendering utilities for RNA secondary structure visualization.

This module provides functions for rendering RNA secondary structures
and related graphical elements as reusable SVG components. Individual
components can be positioned, scaled, and combined into a complete SVG
drawing using the composition utilities defined in this module.
"""

from dataclasses import dataclass
from typing import Optional
from collections.abc import Sequence

import numpy as np
import matplotlib as mpl
import svgwrite

from .layout import (
    EdgeType,
    Node,
    Edge,
    LineEdge,
    ArcEdge,
    Marker,
    ArrowMarker,
    LayoutResult,
)
from .style import DrawingStyle
from .vec2 import Vec2


@dataclass(frozen=True)
class SVGComponent:
    """SVG component defined in its own local coordinate system.

    Parameters
    ----------
    group : svgwrite.container.Group
        SVG group containing the graphical elements of the component.
        The group is assumed to be defined in a local coordinate system
        whose origin is typically ``(0, 0)``.
    width : float
        Width of the component in its local coordinate system.
    height : float
        Height of the component in its local coordinate system.

    Notes
    -----
    The component itself does not store placement information.
    Positioning and scaling are handled by :class:`PlacedComponent` and
    applied during composition.
    """

    group: svgwrite.container.Group
    width: float
    height: float


@dataclass
class PlacedComponent:
    """An SVG component together with its placement information.

    Parameters
    ----------
    component : SVGComponent
        SVG component to be placed in the composed drawing.
    x : float
        X-coordinate of the component origin in the composed drawing.
    y : float
        Y-coordinate of the component origin in the composed drawing.
    scale : float
        Uniform scaling factor applied to the component.
    z_index : int, default=0
        Drawing order of the component. Components with smaller values
        are rendered first.
    layer : str, default="main"
        Logical layer identifier reserved for future use.

    Attributes
    ----------
    width : float
        Width of the placed component after scaling.
    height : float
        Height of the placed component after scaling.

    Notes
    -----
    The ``width`` and ``height`` attributes are computed dynamically
    from the original component size and the scaling factor:

    - ``width = component.width * scale``
    - ``height = component.height * scale``
    """

    component: SVGComponent
    x: float
    y: float
    scale: float
    z_index: int = 0
    layer: str = "main"

    @property
    def width(self) -> float:
        """Width of the component after scaling."""
        return self.component.width * self.scale

    @property
    def height(self) -> float:
        """Height of the component after scaling."""
        return self.component.height * self.scale


class SVGRenderer:
    """Renderer converting LayoutResult objects into SVG drawings."""

    def __init__(
        self,
        style: Optional[DrawingStyle] = None,
    ) -> None:
        self.style = style or DrawingStyle()
        self._color_norm = mpl.colors.Normalize(vmin=0, vmax=1)

    def _draw_node(
        self,
        drawing: svgwrite.Drawing,
        node: Node,
        shift_vec: Vec2,
    ):
        """Draw a nucleotide node."""

        pos = node.pos + shift_vec
        nt = node.nucleotide

        group = drawing.g()

        if nt.basepair_probability is None:
            fill = self.style.node_fill
        else:
            fill = mpl.colors.to_hex(
                self.style.cmap(
                    self._color_norm(
                        nt.basepair_probability
                    )
                )
            )
        
        group.add(
            drawing.circle(
                center=pos.to_tuple(),
                r=self.style.node_radius,
                fill=fill,
            )
        )

        if nt.base is not None:
            group.add(
                drawing.text(
                    nt.base,
                    insert=pos.to_tuple(),
                    text_anchor="middle",
                    dominant_baseline="middle",
                    font_size=self.style.font_size,
                    fill="white",
                    stroke="black",
                    stroke_width=1,
                    style="paint-order: stroke fill;",
                )
            )

        return group

    def _draw_edge(
        self,
        drawing: svgwrite.Drawing,
        edge: Edge,
        shift_vec: Vec2,
    ):
        """Draw an edge."""

        if isinstance(edge, LineEdge):
            return self._draw_line_edge(
                drawing,
                edge,
                shift_vec,
            )

        if isinstance(edge, ArcEdge):
            return self._draw_arc_edge(
                drawing,
                edge,
                shift_vec,
            )

        raise TypeError(f"Unsupported edge type: {type(edge)}")

    def _draw_line_edge(
        self,
        drawing: svgwrite.Drawing,
        edge: LineEdge,
        shift_vec: Vec2,
    ):
        """Draw a straight line edge."""

        start = edge.start.pos + shift_vec
        end = edge.end.pos + shift_vec

        if edge.type == EdgeType.BACKBONE:
            width = self.style.backbone_width
        elif edge.type == EdgeType.BASE_PAIR:
            width = self.style.basepair_width

        return drawing.line(
            start=start.to_tuple(),
            end=end.to_tuple(),
            stroke=self.style.edge_color,
            stroke_width=width,
        )

    def _draw_arc_edge(
        self,
        drawing: svgwrite.Drawing,
        edge: ArcEdge,
        shift_vec: Vec2,
    ):
        """Draw an SVG arc edge."""

        start = edge.start.pos + shift_vec
        end = edge.end.pos + shift_vec

        large_arc = int(edge.large_arc)
        sweep = int(edge.sweep)

        path = (
            f"M {start.x} {start.y} "
            f"A {edge.rx} {edge.ry} "
            f"{edge.x_axis_rotation} "
            f"{large_arc} {sweep} "
            f"{end.x} {end.y}"
        )

        return drawing.path(
            d=path,
            stroke=self.style.edge_color,
            fill="none",
            stroke_width=self.style.backbone_width,
        )

    def _draw_marker(
        self,
        drawing: svgwrite.Drawing,
        marker: Marker,
        shift_vec: Vec2,
    ):
        """Draw a marker."""

        if isinstance(marker, ArrowMarker):
            return self._draw_arrow_marker(
                drawing,
                marker,
                shift_vec,
            )

        return None

    def _draw_arrow_marker(
        self,
        drawing: svgwrite.Drawing,
        marker: ArrowMarker,
        shift_vec: Vec2,
    ):
        """Draw an arrow marker."""

        if marker.is_start:
            start = marker.node.pos + shift_vec
            end = start + marker.direction * marker.length
        else:
            end = marker.node.pos + shift_vec
            start = end - marker.direction * marker.length

        line = drawing.line(
            start=start.to_tuple(),
            end=end.to_tuple(),
            stroke=self.style.edge_color,
            stroke_width=self.style.backbone_width,
        )

        line["marker-end"] = "url(#arrow)"

        return line

    def _add_arrowhead_def(
        self,
        drawing: svgwrite.Drawing,
    ) -> None:
        """Add SVG arrow marker definitions."""

        arrow = drawing.marker(
            id="arrow",
            insert=(1, 1.5),
            size=(10, 10),
            orient="auto",
        )

        arrow.add(
            drawing.path(
                d="M 0,0 L 0.7,1.5 L 0,3 L 3,1.5 Z",
                fill=self.style.edge_color,
            )
        )

        drawing.defs.add(arrow)
    
    def _render_structure(
        self,
        svg_drawing: svgwrite.Drawing,
        layout_result: LayoutResult,
    ) -> SVGComponent:
        """Render an RNA secondary structure as an SVG component."""

        svg_group = svg_drawing.g()

        xs = [node.pos.x for node in layout_result.nodes]
        ys = [node.pos.y for node in layout_result.nodes]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        width = x_max - x_min + 2 * self.style.x_margin
        height = y_max - y_min + 2 * self.style.y_margin

        # Shift the layout so that its bounding box starts at
        # the configured drawing margin.
        shift_vec = Vec2(
            -x_min + self.style.x_margin,
            -y_min + self.style.y_margin,
        )

        for edge in layout_result.edges:
            svg_group.add(
                self._draw_edge(
                    svg_drawing,
                    edge,
                    shift_vec,
                )
            )

        self._add_arrowhead_def(svg_drawing)
        for marker in layout_result.markers:
            marker_element = self._draw_marker(
                svg_drawing,
                marker,
                shift_vec,
            )
            if marker_element is not None:
                svg_group.add(marker_element)

        for node in layout_result.nodes:
            svg_group.add(
                self._draw_node(
                    svg_drawing,
                    node,
                    shift_vec,
                )
            )

        return SVGComponent(
            group=svg_group,
            width=width,
            height=height,
        )
    
    def _render_colorbar(
        self,
        svg_drawing: svgwrite.Drawing,
        label: str | None = None,
    ) -> SVGComponent:
        """Render a colorbar as an SVG component."""

        if label is None:
            label = "Base-pair probability"

        svg_group = svg_drawing.g()

        vb_width = 155
        vb_height = 500

        bar_width = 15
        bar_height = 450
        bar_x = 30
        bar_y = (vb_height - bar_height) / 2

        # Define the vertical color gradient.
        gradient = svg_drawing.linearGradient(
            start=(0, 1),
            end=(0, 0),
            id="colorbar_grad",
        )

        for value in np.linspace(0.0, 1.0, 101):
            color = mpl.colors.to_hex(
                self.style.cmap(
                    self._color_norm(value)
                )
            )
            gradient.add_stop_color(
                offset=value,
                color=color,
            )

        svg_drawing.defs.add(gradient)

        svg_group.add(
            svg_drawing.rect(
                insert=(bar_x, bar_y),
                size=(bar_width, bar_height),
                fill="url(#colorbar_grad)",
            )
        )

        # Draw tick marks and labels.
        for value in np.linspace(0.0, 1.0, 11):
            y = bar_y + (1.0 - value) * bar_height

            svg_group.add(
                svg_drawing.line(
                    start=(bar_x + bar_width, y),
                    end=(bar_x + bar_width + 5, y),
                    stroke="black",
                )
            )

            svg_group.add(
                svg_drawing.text(
                    f"{value:.1f}",
                    insert=(bar_x + bar_width + 10, y + 4),
                    font_size=12,
                    fill="black",
                )
            )

        svg_group.add(
            svg_drawing.text(
                label,
                insert=(120, vb_height / 2),
                text_anchor="middle",
                dominant_baseline="middle",
                font_size=15,
                fill="black",
                transform=f"rotate(90, 120, {vb_height / 2})",
            )
        )

        return SVGComponent(
            group=svg_group,
            width=vb_width,
            height=vb_height,
        )


def render_structure(
    svg_drawing: svgwrite.Drawing,
    layout_result: LayoutResult,
    style: Optional[DrawingStyle] = None,
) -> SVGComponent:
    """Render an RNA secondary structure as an SVG component.

    Parameters
    ----------
    svg_drawing : svgwrite.Drawing
        Drawing object used to create SVG elements and definitions.
    layout_result : LayoutResult
        Layout result describing the geometry of the RNA secondary
        structure.
    style : DrawingStyle, optional
        Drawing style controlling colors, sizes, and line widths.

    Returns
    -------
    SVGComponent
        SVG component containing the rendered secondary structure.
    """
    renderer = SVGRenderer(style)
    return renderer._render_structure(
        svg_drawing, layout_result,
    )

def render_colorbar(
    svg_drawing: svgwrite.Drawing,
    label: str | None = None,
    style: Optional[DrawingStyle] = None,
) -> SVGComponent:
    """Render a colorbar as an SVG component.

    Parameters
    ----------
    svg_drawing : svgwrite.Drawing
        Drawing object used to create SVG elements and definitions.
    label : str, optional
        Label displayed alongside the colorbar. If ``None``, a default
        label is used.
    style : DrawingStyle, optional
        Drawing style providing the colormap used for rendering.

    Returns
    -------
    SVGComponent
        SVG component containing the rendered colorbar.
    """
    renderer = SVGRenderer(style)
    return renderer._render_colorbar(
        svg_drawing, label,
    )


class Composer:
    """Compose multiple SVG components into a single SVG drawing.

    The composer applies placement and scaling information stored in
    :class:`PlacedComponent` objects and inserts the resulting groups into
    an existing ``svgwrite.Drawing``.

    The SVG viewBox is automatically computed from the bounding boxes
    of all placed components. No additional padding or layout
    adjustment is performed.
    """

    def _compose(
        self,
        svg_drawing: svgwrite.Drawing,
        groups: Sequence[PlacedComponent],
    ) -> None:
        """Compose multiple positioned components into a single SVG.

        Parameters
        ----------
        groups
            Positioned SVG components.

        Returns
        -------
        svgwrite.Drawing
            The composed SVG drawing.
        """
        if not groups:
            svg_drawing.viewbox(0, 0, 0, 0)
            return None

        groups = sorted(groups, key=lambda g: g.z_index)

        xmin = min(g.x for g in groups)
        ymin = min(g.y for g in groups)
        xmax = max(g.x + g.width for g in groups)
        ymax = max(g.y + g.height for g in groups)

        svg_drawing.viewbox(
            xmin,
            ymin,
            xmax - xmin,
            ymax - ymin,
        )
        svg_drawing["width"] = f"{xmax - xmin}px"
        svg_drawing["height"] = f"{ymax - ymin}px"

        for placed in groups:
            wrapper = svg_drawing.g(
                transform=(
                    f"translate({placed.x},{placed.y}) "
                    f"scale({placed.scale},{placed.scale})"
                )
            )
            wrapper.add(placed.component.group)
            svg_drawing.add(wrapper)

        return None

def compose(
    svg_drawing: svgwrite.Drawing,
    groups: Sequence[PlacedComponent],
) -> None:
    """Compose positioned SVG components into an SVG drawing.

    Parameters
    ----------
    svg_drawing : svgwrite.Drawing
        Drawing object that receives the composed SVG elements.
    groups : Sequence[PlacedComponent]
        Sequence of placed components to insert into the drawing.

    Notes
    -----
    Components are rendered in ascending order of ``z_index``.
    Their associated transformations are applied using SVG
    ``translate`` and ``scale`` operations, and the drawing viewBox
    is adjusted to enclose all components.
    """
    return Composer()._compose(svg_drawing, groups)

