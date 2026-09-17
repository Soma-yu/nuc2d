"""SVG rendering utilities for RNA secondary structure visualization.

This module provides functions for rendering RNA secondary structures
and related graphical elements as reusable SVG components. Individual
components can be positioned, scaled, and combined into a complete SVG
drawing using the composition utilities defined in this module.
"""

import hashlib
from dataclasses import dataclass
from functools import reduce
from collections.abc import Callable, Sequence

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
from .geometry import BBox, Vec2
from .font import find_font_path, vertical_center_offset


def _def_id(prefix: str, *parts: object) -> str:
    """Return a stable id for a shared SVG definition.

    Parameters
    ----------
    prefix : str
        Readable prefix identifying the kind of definition.
    *parts : object
        Every value that determines the definition's content.

    Returns
    -------
    str
        An id of the form ``"<prefix>-<digest>"``.

    Notes
    -----
    Deriving the id from the content rather than from a counter keeps the
    output of a given drawing reproducible, lets two components that need
    the same definition share one, and guarantees that two components
    needing *different* definitions never collide on an id.
    """
    digest = hashlib.blake2s(
        "|".join(map(str, parts)).encode(), digest_size=4
    ).hexdigest()
    return f"{prefix}-{digest}"


def _ensure_def(
    drawing: svgwrite.Drawing,
    element_id: str,
    build: Callable[[], object],
) -> str:
    """Add a definition to a drawing's ``<defs>`` at most once.

    Parameters
    ----------
    drawing : svgwrite.Drawing
        Drawing whose definitions are being populated.
    element_id : str
        Id the definition will be referenced by.
    build : callable
        Builds the definition element. It is called only when no
        definition with this id is present yet.

    Returns
    -------
    str
        ``element_id``, so that callers can reference it inline.
    """
    for element in drawing.defs.elements:
        if element.attribs.get("id") == element_id:
            return element_id
    drawing.defs.add(build())
    return element_id


@dataclass(frozen=True)
class SVGComponent:
    """SVG component defined in its own local coordinate system.

    Parameters
    ----------
    group : svgwrite.container.Group
        SVG group containing the graphical elements of the component.
    bbox : BBox
        Extent of the component in its own local coordinate system.

    Attributes
    ----------
    width : float
        Width of the component, derived from ``bbox``.
    height : float
        Height of the component, derived from ``bbox``.

    Notes
    -----
    The component itself does not store placement information.
    Positioning and scaling are handled by :class:`Placement` and
    applied during composition.
    """

    group: svgwrite.container.Group
    bbox: BBox

    @property
    def width(self) -> float:
        """Width of the component in its local coordinate system."""
        return self.bbox.width

    @property
    def height(self) -> float:
        """Height of the component in its local coordinate system."""
        return self.bbox.height


@dataclass
class Placement:
    """Where a component goes in a composed drawing.

    Parameters
    ----------
    component : SVGComponent
        Component being placed.
    x : float
        X-coordinate of the component origin in the composed drawing.
    y : float
        Y-coordinate of the component origin in the composed drawing.
    scale : float
        Uniform scaling factor applied to the component.
    z_index : int, default=0
        Drawing order of the component. Components with smaller values
        are rendered first.

    Attributes
    ----------
    bbox : BBox
        Extent of the component in the composed drawing, after placement.
    width : float
        Width of the component after scaling.
    height : float
        Height of the component after scaling.

    Notes
    -----
    The placement corresponds to the SVG transform
    ``translate(x, y) scale(scale)``, which scales about the origin of the
    component's own coordinate system and then moves the result, so
    :attr:`bbox` applies the two in that order.
    """

    component: SVGComponent
    x: float
    y: float
    scale: float
    z_index: int = 0

    @property
    def bbox(self) -> BBox:
        """Extent of the component in the composed drawing."""
        return self.component.bbox.scaled(self.scale).translated(Vec2(self.x, self.y))

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
        style: DrawingStyle | None = None,
    ) -> None:
        self.style = style or DrawingStyle()
        self._color_norm = mpl.colors.Normalize(vmin=0, vmax=1)

    def _draw_node(
        self,
        drawing: svgwrite.Drawing,
        node: Node,
    ):
        """Draw a nucleotide node."""

        pos = node.pos
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
            font_path = find_font_path(self.style.font_family)
            baseline_offset = vertical_center_offset(
                font_path,
                self.style.font_size,
            )
            text_pos = pos + Vec2(0, baseline_offset)
            group.add(
                drawing.text(
                    nt.base,
                    insert=text_pos.to_tuple(),
                    text_anchor="middle",
                    font_family=self.style.font_family,
                    font_size=self.style.font_size,
                    fill="black",
                    stroke="black",
                    stroke_width=1,
                )
            )
            group.add(
                drawing.text(
                    nt.base,
                    insert=text_pos.to_tuple(),
                    text_anchor="middle",
                    font_family=self.style.font_family,
                    font_size=self.style.font_size,
                    fill="white",
                )
            )

        return group

    def _draw_edge(
        self,
        drawing: svgwrite.Drawing,
        edge: Edge,
    ):
        """Draw an edge."""

        if isinstance(edge, LineEdge):
            return self._draw_line_edge(
                drawing,
                edge,
            )

        if isinstance(edge, ArcEdge):
            return self._draw_arc_edge(
                drawing,
                edge,
            )

        raise TypeError(f"Unsupported edge type: {type(edge)}")

    def _draw_line_edge(
        self,
        drawing: svgwrite.Drawing,
        edge: LineEdge,
    ):
        """Draw a straight line edge."""

        start = edge.start.pos
        end = edge.end.pos

        if edge.type == EdgeType.BACKBONE:
            width = self.style.backbone_width
            dasharray = self.style.backbone_dasharray
        elif edge.type == EdgeType.BASE_PAIR:
            width = self.style.basepair_width
            dasharray = self.style.basepair_dasharray
        else:
            raise ValueError(f"Unsupported edge type: {edge.type}")

        return drawing.line(
            start=start.to_tuple(),
            end=end.to_tuple(),
            stroke=self.style.edge_color,
            stroke_width=width,
            stroke_dasharray=dasharray,
        )

    def _draw_arc_edge(
        self,
        drawing: svgwrite.Drawing,
        edge: ArcEdge,
    ):
        """Draw an SVG arc edge."""

        start = edge.start.pos
        end = edge.end.pos

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
    ):
        """Draw a marker."""

        if isinstance(marker, ArrowMarker):
            return self._draw_arrow_marker(
                drawing,
                marker,
            )

        return None

    def _draw_arrow_marker(
        self,
        drawing: svgwrite.Drawing,
        marker: ArrowMarker,
    ):
        """Draw an arrow marker."""

        if marker.node_at_start:
            start = marker.node.pos
            end = start + marker.direction * marker.length
        else:
            end = marker.node.pos
            start = end - marker.direction * marker.length

        line = drawing.line(
            start=start.to_tuple(),
            end=end.to_tuple(),
            stroke=self.style.edge_color,
            stroke_width=self.style.backbone_width,
        )

        line["marker-end"] = f"url(#{self._arrowhead_id()})"

        return line

    def _arrowhead_id(self) -> str:
        """Return the id of the arrowhead marker for the current style.

        The id is derived from the style values the marker depends on, so
        renderers using the same style share one definition while
        renderers using different ones never collide.
        """
        return _def_id("arrowhead", self.style.edge_color)

    def _add_arrowhead_def(
        self,
        drawing: svgwrite.Drawing,
    ) -> str:
        """Ensure the arrowhead marker is defined and return its id."""

        def build() -> svgwrite.container.Marker:
            # The viewBox matches the path's own extent, so the arrowhead is
            # drawn at the size the path describes. Without one, a renderer
            # has to guess how the path maps into the marker viewport: a
            # browser draws it unscaled, while cairosvg stretches it to fill
            # the viewport, which made PNG exports show an oversized arrow.
            arrow = drawing.marker(
                id=self._arrowhead_id(),
                insert=(1, 1.5),
                size=(3, 3),
                orient="auto",
                markerUnits="strokeWidth",
                viewBox="0 0 3 3",
            )
            arrow.add(
                drawing.path(
                    d="M 0,0 L 0.7,1.5 L 0,3 L 3,1.5 Z",
                    fill=self.style.edge_color,
                )
            )
            return arrow

        return _ensure_def(drawing, self._arrowhead_id(), build)

    def render_structure(
        self,
        drawing: svgwrite.Drawing,
        layout_result: LayoutResult,
    ) -> SVGComponent:
        """Render an RNA secondary structure as an SVG component.

        The elements are drawn at the coordinates the layout produced,
        without being moved to the origin first. The component's bounding
        box records where they actually are, and composition places the
        component from there.
        """

        group = drawing.g()

        bbox = BBox.from_points(
            node.pos for node in layout_result.nodes
        ).expanded(self.style.x_margin, self.style.y_margin)

        for edge in layout_result.edges:
            group.add(
                self._draw_edge(
                    drawing,
                    edge,
                )
            )

        self._add_arrowhead_def(drawing)
        for marker in layout_result.markers:
            marker_element = self._draw_marker(
                drawing,
                marker,
            )
            if marker_element is not None:
                group.add(marker_element)

        for node in layout_result.nodes:
            group.add(
                self._draw_node(
                    drawing,
                    node,
                )
            )

        return SVGComponent(group=group, bbox=bbox)

    def render_colorbar(
        self,
        drawing: svgwrite.Drawing,
        label: str | None = None,
    ) -> SVGComponent:
        """Render a colorbar as an SVG component."""

        if label is None:
            label = "Base-pair probability"

        group = drawing.g()

        vb_width = 150
        vb_height = 500

        bar_height = 450
        bar_width = bar_height * self.style.colorbar_width_ratio
        bar_x = 30
        bar_y = (vb_height - bar_height) / 2

        # Define the vertical color gradient.
        stops = [
            (
                value,
                mpl.colors.to_hex(
                    self.style.cmap(
                        self._color_norm(value)
                    )
                ),
            )
            for value in np.linspace(0.0, 1.0, 101)
        ]

        gradient_id = _def_id(
            "colorbar-gradient", *(color for _, color in stops)
        )

        def build_gradient() -> svgwrite.gradients.LinearGradient:
            gradient = drawing.linearGradient(
                start=(0, 1),
                end=(0, 0),
                id=gradient_id,
            )
            for offset, color in stops:
                gradient.add_stop_color(
                    offset=offset,
                    color=color,
                )
            return gradient

        _ensure_def(drawing, gradient_id, build_gradient)

        group.add(
            drawing.rect(
                insert=(bar_x, bar_y),
                size=(bar_width, bar_height),
                fill=f"url(#{gradient_id})",
            )
        )

        # Draw tick marks and labels.
        for value in np.linspace(0.0, 1.0, 11):
            y = bar_y + (1.0 - value) * bar_height

            group.add(
                drawing.line(
                    start=(bar_x + bar_width, y),
                    end=(bar_x + bar_width + self.style.colorbar_tick_length, y),
                    stroke="black",
                )
            )

            group.add(
                drawing.text(
                    f"{value:.1f}",
                    insert=(bar_x + bar_width + 10, y + 4),
                    font_family = self.style.font_family,
                    font_size=self.style.colorbar_tick_font_size,
                    fill="black",
                )
            )

        group.add(
            drawing.text(
                label,
                insert=(100, vb_height/2),
                text_anchor="middle",
                font_family = self.style.font_family,
                font_size=self.style.colorbar_label_font_size,
                fill="black",
                transform=f"rotate(90, 100, {vb_height / 2})",
            )
        )

        return SVGComponent(
            group=group,
            bbox=BBox(0.0, 0.0, vb_width, vb_height),
        )


def render_structure(
    drawing: svgwrite.Drawing,
    layout_result: LayoutResult,
    style: DrawingStyle | None = None,
) -> SVGComponent:
    """Render an RNA secondary structure as an SVG component.

    Parameters
    ----------
    drawing : svgwrite.Drawing
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
    return renderer.render_structure(
        drawing, layout_result,
    )


def render_colorbar(
    drawing: svgwrite.Drawing,
    label: str | None = None,
    style: DrawingStyle | None = None,
) -> SVGComponent:
    """Render a colorbar as an SVG component.

    Parameters
    ----------
    drawing : svgwrite.Drawing
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
    return renderer.render_colorbar(
        drawing, label,
    )


def compose(
    container: svgwrite.container.Group,
    placements: Sequence[Placement],
) -> SVGComponent:
    """Compose positioned SVG components into a single group.

    Parameters
    ----------
    container : svgwrite.container.Group
        Group that receives the composed SVG elements.
    placements : Sequence[Placement]
        Components to insert, each with the placement to apply to it.

    Returns
    -------
    SVGComponent
        The container together with the extent enclosing every component.
        An empty sequence yields a component with an empty bounding box.

    Notes
    -----
    Components are inserted in ascending order of ``z_index``. Each one is
    wrapped in a group carrying its SVG ``translate`` and ``scale``. No
    padding or layout adjustment is applied.
    """
    for placement in sorted(placements, key=lambda p: p.z_index):
        wrapper = svgwrite.container.Group(
            transform=(
                f"translate({placement.x},{placement.y}) "
                f"scale({placement.scale},{placement.scale})"
            )
        )
        wrapper.add(placement.component.group)
        container.add(wrapper)

    bbox = reduce(BBox.union, (p.bbox for p in placements), BBox.empty())

    return SVGComponent(group=container, bbox=bbox)
