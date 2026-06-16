"""SVG rendering utilities for RNA secondary structure layouts.

This module provides classes and functions for rendering layout results
as SVG graphics using the svgwrite library. It converts geometric layout
objects, such as nodes, edges, and markers, into SVG drawing elements
while applying configurable drawing styles.

The main renderer class, SVGRenderer, generates SVG representations
from LayoutResult objects produced by layout engines.
"""

from typing import Optional

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
from .overlay import Overlay, Colorbar, Title


class SVGRenderer:
    """Renderer converting LayoutResult objects into SVG drawings."""

    def __init__(
        self,
        style: Optional[DrawingStyle] = None,
        overlays: Optional[list[Overlay]] = None,
    ) -> None:
        self.style = style or DrawingStyle()
        self.overlays = overlays or []
        self._color_norm = mpl.colors.Normalize(vmin=0, vmax=1)

    def _render(
        self,
        layout_result: LayoutResult,
    ) -> svgwrite.Drawing:
        """Render a layout result as an SVG drawing."""

        drawing = svgwrite.Drawing()

        xs = [node.pos.x for node in layout_result.nodes]
        ys = [node.pos.y for node in layout_result.nodes]
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)

        width = x_max - x_min + 2 * self.style.x_margin
        height = y_max - y_min + 2 * self.style.y_margin

        shift_vec = Vec2(
            -x_min + self.style.x_margin,
            -y_min + self.style.y_margin,
        )

        group = drawing.g()

        self._add_arrowhead_def(drawing)

        for edge in layout_result.edges:
            group.add(
                self._draw_edge(
                    drawing,
                    edge,
                    shift_vec,
                )
            )
        
        for marker in layout_result.markers:
            element = self._draw_marker(
                drawing,
                marker,
                shift_vec,
            )

            if element is not None:
                group.add(element)

        for node in layout_result.nodes:
            group.add(
                self._draw_node(
                    drawing,
                    node,
                    shift_vec,
                )
            )

        extra_width = 0

        extra_width = 0

        for overlay in self.overlays:
            if isinstance(overlay, Colorbar):

                g_colorbar, vb_width, vb_height = self._draw_colorbar(
                    drawing,
                    self.style.cmap,
                    "Base-pair probability"
                )

                colorbar_scale = height / vb_height
                vb_width *= colorbar_scale

                g_colorbar['transform'] = f'translate({width}, 0), scale({colorbar_scale}, {colorbar_scale})'
                width += vb_width

                drawing.add(g_colorbar)

        drawing.viewbox(
            0,
            0,
            width,
            height,
        )

        drawing.add(group)
        return drawing

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
    
    def _draw_colorbar(self, svg_drawing, cmap, label):
        # 仮想座標領域
        svg_group = svg_drawing.g()
        vb_width = 155
        vb_height = 500

        # カラーバー用グラデーション
        norm = self._color_norm
        grad = svg_drawing.linearGradient(start=(0, 1), end=(0, 0), id='colorbar_grad')
        n_steps = 100
        for i in range(n_steps + 1):
            val = i / n_steps
            color = mpl.colors.to_hex(cmap(norm(val)))
            grad.add_stop_color(offset=val, color=color)
        svg_drawing.defs.add(grad)

        # カラーバー矩形
        bar_width = 15
        bar_height = 450
        bar_x = 30
        bar_y = (vb_height - bar_height) / 2
        svg_group.add(svg_drawing.rect(
            insert=(bar_x, bar_y),
            size=(bar_width, bar_height),
            fill='url(#colorbar_grad)'))

        # 目盛り
        for t in np.linspace(0, 1, 11):
            y = bar_y + (1 - t) * bar_height
            svg_group.add(svg_drawing.line(
                start=(bar_x + bar_width, y),
                end=(bar_x + bar_width + 5, y),
                stroke='black'))
            svg_group.add(svg_drawing.text(
                f'{t:.1f}',
                insert=(bar_x + bar_width + 10, y + 4),
                font_size=12,
                fill='black'))

        # ラベル
        svg_group.add(svg_drawing.text(
            label,
            insert=(120, vb_height / 2),
            text_anchor='middle',
            dominant_baseline='middle',
            font_size=15,
            fill='black',
            transform=f'rotate(90, 120, {vb_height / 2})'))

        return svg_group, vb_width, vb_height


def render(
    layout_result: LayoutResult,
    style: Optional[DrawingStyle] = None,
    overlays: Optional[list[Overlay]] = None,
) -> svgwrite.Drawing:
    """Render a layout result as an SVG drawing.

    Parameters
    ----------
    layout_result : LayoutResult
        Layout result to render.
    style : DrawingStyle, optional
        Drawing style used for rendering.
    overlays : list[Overlay] or None
        Optional list of overlay elements (e.g., colorbar, annotations)
        to be rendered on top of the structure. Overlays may affect the
        final canvas size (e.g., require additional width/height in the
        viewbox).

    Returns
    -------
    svgwrite.Drawing
        SVG drawing representing the layout.
    """

    renderer = SVGRenderer(style, overlays)

    return renderer._render(
        layout_result,
    )
