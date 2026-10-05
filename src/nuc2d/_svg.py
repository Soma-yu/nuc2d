"""Everything nuc2d writes as SVG.

This module draws a laid-out structure, a colorbar and a line of text as
SVG, each as a component, and writes a component and everything placed
in it as a finished document. It is the only module that knows how SVG
is written, so that writing it another way changes this module and no
other.

Each part is drawn into a drawing of its own, and handed back together
with the definitions it registered there, such as the gradient of a
colorbar, so that the document can write each of them once however many
parts refer to it.

What other modules use is the three ``render_`` functions,
:func:`document_string` and :func:`save_document`. Everything else here,
the renderer class included, is private to this module.
"""

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import matplotlib as mpl
import svgwrite

from ._layout import (
    EdgeType,
    Node,
    Edge,
    LineEdge,
    ArcEdge,
    Decoration,
    ArrowDecoration,
    LayoutResult,
)
from ._style import StructureStyle
from ._component import (
    Component,
    child_placements_of,
    graphics_of,
    make_component,
    transform_of,
)
from ._geometry import BBox, Vec2, bbox_around, make_bbox
from ._font import find_font, text_width, vertical_center_offset, vertical_extent


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
) -> None:
    """Add a definition to a drawing's ``<defs>`` at most once.

    Parameters
    ----------
    drawing : svgwrite.Drawing
        Drawing whose definitions are being populated.
    element_id : str
        Id the definition will be referenced by. The caller already
        holds it, so nothing is handed back.
    build : callable
        Builds the definition element. It is called only when no
        definition with this id is present yet.
    """
    for element in drawing.defs.elements:
        if element.attribs.get("id") == element_id:
            return
    drawing.defs.add(build())


# The box around a structure runs this far past the centers of its
# outermost nodes, which leaves some room around what is drawn there at
# the default sizes. It is not measured from what is drawn, so a node, a
# letter or a 3' arrow drawn larger than this reaches past the box.
_STRUCTURE_MARGIN = 20.0

# Every letter is set in this font: the bases, the colorbar's numbers and
# label, and a title. It is the family the SVG asks for, and the one looked
# up on this machine to measure the letters with.
_FONT_FAMILY = "Arial"

# The colorbar's own proportions and lettering. They are not style
# settings: a style says how a structure looks, and the colorbar is only
# the key to the colors its probabilities are shown in.
_COLORBAR_ASPECT_RATIO = 1 / 30  # the bar's width over its height
# The height of the colorbar's box, which the structure beside it is
# fitted to as well.
COLORBAR_HEIGHT = 500.0
_COLORBAR_TICK_LENGTH = 5.0
_COLORBAR_TICK_FONT_SIZE = 12.0
_COLORBAR_LABEL_FONT_SIZE = 16.0


class _Renderer:
    """Draws a laid-out structure or a colorbar into the drawing it is handed.

    Only this module's ``render_*`` functions use it: they hand it a
    drawing of their own, and hand back what it drew together with the
    definitions it registered there. The class is what lets the steps of
    drawing one part share a colormap and a style.

    Parameters
    ----------
    colormap : matplotlib.colors.Colormap, optional
        Colormap probabilities are shown in. Defaults to
        ``mpl.colormaps["turbo"]``.
    style : StructureStyle, optional
        Appearance settings. Defaults to ``StructureStyle()``.

    Notes
    -----
    A renderer holds only its settings, nothing about a drawing in
    progress, so one can be reused for any number of drawings.
    """

    def __init__(
        self,
        *,
        colormap: mpl.colors.Colormap | None = None,
        style: StructureStyle | None = None,
    ) -> None:
        self.colormap = (
            colormap if colormap is not None else mpl.colormaps["turbo"]
        )
        self.style = style if style is not None else StructureStyle()
        self._color_norm = mpl.colors.Normalize(vmin=0, vmax=1)

    def _draw_node(
        self,
        drawing: svgwrite.Drawing,
        node: Node,
    ) -> svgwrite.container.Group:
        """Draw a nucleotide node."""

        pos = node.pos
        nt = node.nucleotide

        group = drawing.g()

        if nt.probability is None:
            fill = self.style.nucleotide_color
        else:
            fill = mpl.colors.to_hex(
                self.colormap(
                    self._color_norm(
                        nt.probability
                    )
                )
            )

        group.add(
            drawing.circle(
                center=pos.to_tuple(),
                r=self.style.nucleotide_radius,
                fill=fill,
            )
        )

        if nt.base is not None:
            font = find_font(_FONT_FAMILY)
            baseline_offset = vertical_center_offset(
                font,
                self.style.nucleotide_font_size,
            )
            text_pos = pos + Vec2(0, baseline_offset)
            group.add(
                drawing.text(
                    nt.base,
                    insert=text_pos.to_tuple(),
                    text_anchor="middle",
                    font_family=_FONT_FAMILY,
                    font_size=self.style.nucleotide_font_size,
                    fill="black",
                    stroke="black",
                    stroke_width=1,
                    # A mitred corner juts out at the sharp apex of an A.
                    stroke_linejoin="round",
                )
            )
            group.add(
                drawing.text(
                    nt.base,
                    insert=text_pos.to_tuple(),
                    text_anchor="middle",
                    font_family=_FONT_FAMILY,
                    font_size=self.style.nucleotide_font_size,
                    fill="white",
                )
            )

        return group

    def _draw_edge(
        self,
        drawing: svgwrite.Drawing,
        edge: Edge,
    ) -> svgwrite.shapes.Line | svgwrite.path.Path:
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
    ) -> svgwrite.shapes.Line:
        """Draw a straight line edge."""

        start = edge.start.pos
        end = edge.end.pos

        if edge.edge_type == EdgeType.BACKBONE:
            color = self.style.backbone_color
            width = self.style.backbone_width
            dasharray = self.style.backbone_dasharray
        elif edge.edge_type == EdgeType.BASEPAIR:
            color = self.style.basepair_color
            width = self.style.basepair_width
            dasharray = self.style.basepair_dasharray
        else:
            raise ValueError(f"Unsupported edge type: {edge.edge_type}")

        return drawing.line(
            start=start.to_tuple(),
            end=end.to_tuple(),
            stroke=color,
            stroke_width=width,
            stroke_dasharray=dasharray,
        )

    def _draw_arc_edge(
        self,
        drawing: svgwrite.Drawing,
        edge: ArcEdge,
    ) -> svgwrite.path.Path:
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
            stroke=self.style.backbone_color,
            fill="none",
            stroke_width=self.style.backbone_width,
            stroke_dasharray=self.style.backbone_dasharray,
        )

    def _draw_decoration(
        self,
        drawing: svgwrite.Drawing,
        decoration: Decoration,
    ) -> svgwrite.container.Group:
        """Draw a decoration."""

        if isinstance(decoration, ArrowDecoration):
            return self._draw_arrow_decoration(
                drawing,
                decoration,
            )

        raise TypeError(f"Unsupported decoration type: {type(decoration)}")

    def _draw_arrow_decoration(
        self,
        drawing: svgwrite.Drawing,
        decoration: ArrowDecoration,
    ) -> svgwrite.container.Group:
        """Draw the arrow that marks a 3' terminus.

        The arrowhead is a shape of its own at the end of the line, not a
        marker on it. Adobe Illustrator mishandles SVG markers: it drops
        the line that carries one, and the backbone and base-pair lines
        drawn beside it vanish once the drawing is scaled a few times. A
        plain shape is drawn the same everywhere.
        """

        start = decoration.node.pos
        end = start + decoration.direction * self.style.three_prime_arrow_length

        arrow = drawing.g()
        arrow.add(
            drawing.line(
                start=start.to_tuple(),
                end=end.to_tuple(),
                stroke=self.style.backbone_color,
                stroke_width=self.style.backbone_width,
            )
        )
        arrow.add(
            drawing.polygon(
                points=self._arrowhead_points(end, decoration.direction),
                fill=self.style.backbone_color,
            )
        )
        return arrow

    def _arrowhead_points(
        self, end: Vec2, direction: Vec2
    ) -> list[tuple[float, float]]:
        """Return the corners of the arrowhead at ``end``, pointing along
        ``direction``.

        The head is measured in stroke widths, so that it keeps to the
        width of the line it ends: from one stroke width behind the end
        of the line to two past it, three wide, with its back notched.
        The notch sits a little behind the end of the line, so that the
        line runs into the head rather than meeting it at a point.
        """
        width = self.style.backbone_width
        along = direction.normalized()
        across = Vec2(-along.y, along.x)
        # (u, v) in stroke widths from the end of the line: u along the
        # arrow, v across it.
        corners = [(-1.0, -1.5), (-0.3, 0.0), (-1.0, 1.5), (2.0, 0.0)]
        return [
            (end + along * (u * width) + across * (v * width)).to_tuple()
            for u, v in corners
        ]

    def render_structure(
        self,
        drawing: svgwrite.Drawing,
        layout_result: LayoutResult,
    ) -> tuple[Any, BBox]:
        """Render a secondary structure as an SVG group and its extent.

        The elements are drawn at the coordinates the layout produced,
        without being moved to the origin first. The bounding box records
        where they actually are, and placement works from there.
        """

        group = drawing.g()

        # Each node contributes the point it sits at. What the node draws
        # around that point is not enclosed yet; the margin covers it.
        points = bbox_around(
            make_bbox(node.pos.x, node.pos.y, node.pos.x, node.pos.y)
            for node in layout_result.nodes
        )
        bbox = make_bbox(
            points.xmin - _STRUCTURE_MARGIN,
            points.ymin - _STRUCTURE_MARGIN,
            points.xmax + _STRUCTURE_MARGIN,
            points.ymax + _STRUCTURE_MARGIN,
        )

        for edge in layout_result.edges:
            group.add(
                self._draw_edge(
                    drawing,
                    edge,
                )
            )

        for decoration in layout_result.decorations:
            group.add(
                self._draw_decoration(
                    drawing,
                    decoration,
                )
            )

        for node in layout_result.nodes:
            group.add(
                self._draw_node(
                    drawing,
                    node,
                )
            )

        return group, bbox

    def render_colorbar(
        self,
        drawing: svgwrite.Drawing,
        *,
        label: str | None = None,
    ) -> tuple[Any, BBox]:
        """Render a colorbar as an SVG group and its extent.

        With ``label=None`` the label is left out. The box keeps the space
        it would have taken, so colorbars with and without one line up.
        """

        group = drawing.g()

        box_width = 150
        box_height = COLORBAR_HEIGHT

        bar_height = 450
        bar_width = bar_height * _COLORBAR_ASPECT_RATIO
        bar_x = 30
        bar_y = (box_height - bar_height) / 2

        # Define the vertical color gradient. The colormap is asked for
        # every stop in one call, because its cost is per call rather
        # than per value: 101 separate calls took longer than the rest
        # of the colorbar put together.
        offsets = np.linspace(0.0, 1.0, 101)
        stops = list(
            zip(
                offsets,
                map(
                    mpl.colors.to_hex,
                    self.colormap(self._color_norm(offsets)),
                ),
            )
        )

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
                    end=(bar_x + bar_width + _COLORBAR_TICK_LENGTH, y),
                    stroke="black",
                )
            )

            group.add(
                drawing.text(
                    f"{value:.1f}",
                    insert=(bar_x + bar_width + 10, y + 4),
                    font_family=_FONT_FAMILY,
                    font_size=_COLORBAR_TICK_FONT_SIZE,
                    fill="black",
                )
            )

        if label is not None:
            group.add(
                drawing.text(
                    label,
                    insert=(100, box_height/2),
                    text_anchor="middle",
                    font_family=_FONT_FAMILY,
                    font_size=_COLORBAR_LABEL_FONT_SIZE,
                    fill="black",
                    transform=f"rotate(90, 100, {box_height / 2})",
                )
            )

        return group, make_bbox(0.0, 0.0, box_width, box_height)


@dataclass(frozen=True, kw_only=True)
class _Graphics:
    """What a component drawn here holds: its group and its definitions.

    Attributes
    ----------
    group : svgwrite.container.Group
        The ``<g>`` that draws the part.
    definitions : tuple
        The definitions the group refers to, such as a colorbar's
        gradient, which the document showing it has to contain.
    """

    group: svgwrite.container.Group
    definitions: tuple[Any, ...]


def render_structure(
    layout_result: LayoutResult,
    *,
    colormap: mpl.colors.Colormap | None = None,
    style: StructureStyle | None = None,
) -> Component:
    """Render a laid-out secondary structure.

    Parameters
    ----------
    layout_result : LayoutResult
        Geometry of the structure.
    colormap : matplotlib.colors.Colormap, optional
        Colormap the probabilities are shown in. Defaults to
        ``mpl.colormaps["turbo"]``.
    style : StructureStyle, optional
        Appearance settings. Defaults to ``StructureStyle()``.

    Returns
    -------
    Component
        The drawn structure.
    """
    drawing = svgwrite.Drawing()
    group, bbox = _Renderer(colormap=colormap, style=style).render_structure(
        drawing, layout_result
    )
    graphics = _Graphics(group=group, definitions=_definitions_in(drawing))
    return make_component(bbox, graphics=graphics)


def render_colorbar(
    *,
    colormap: mpl.colors.Colormap | None = None,
    label: str | None = None,
) -> Component:
    """Render a colorbar.

    Parameters
    ----------
    colormap : matplotlib.colors.Colormap, optional
        Colormap the bar shows. Defaults to ``mpl.colormaps["turbo"]``.
    label : str, optional
        Label written alongside the colorbar. The colorbar occupies the
        same box with or without one.

    Returns
    -------
    Component
        The drawn colorbar.
    """
    drawing = svgwrite.Drawing()
    group, bbox = _Renderer(colormap=colormap).render_colorbar(
        drawing, label=label
    )
    graphics = _Graphics(group=group, definitions=_definitions_in(drawing))
    return make_component(bbox, graphics=graphics)


def render_text(text: str, *, font_size: float) -> Component:
    """Render one line of text, in the font every letter is set in.

    The text is set on a baseline as far below the top of its box as the
    font's ascender reaches, so that the box runs from the ascender to the
    descender, and from the start of the first character to the end of
    the last. It is anchored at the middle of the box rather than at its
    left edge, so that shown in a font of other widths than the one it
    was measured with, it runs past the box as far on either side, and
    stays centered on it.

    Parameters
    ----------
    text : str
        The text, on one line.
    font_size : float
        Font size of the text.

    Returns
    -------
    Component
        The drawn text.
    """
    font = find_font(_FONT_FAMILY)
    above, below = vertical_extent(font, font_size)
    width = text_width(font, text, font_size)

    drawing = svgwrite.Drawing()
    group = drawing.g()
    group.add(
        drawing.text(
            text,
            insert=(width / 2, above),
            text_anchor="middle",
            font_family=_FONT_FAMILY,
            font_size=font_size,
            fill="black",
        )
    )
    bbox = make_bbox(0.0, 0.0, width, above + below)
    graphics = _Graphics(group=group, definitions=_definitions_in(drawing))
    return make_component(bbox, graphics=graphics)


def _definitions_in(drawing: svgwrite.Drawing) -> tuple[Any, ...]:
    """Return the definitions a renderer registered in ``drawing``.

    A renderer draws into a drawing of its own, and what it put in that
    drawing's ``<defs>`` has to travel with what it drew.
    """
    return tuple(drawing.defs.elements)


# Writing a component. One made of others holds only the placements of its
# children, so the groups and transforms that put each child where it goes
# are assembled here, each time a document is written.


def _assemble(component: Component) -> _Graphics:
    """Assemble the graphics of ``component`` and everything placed in it.

    A drawn component's graphics are its own. Those of a component made of
    others are its children's, each in a group that moves and scales it
    to where it is placed, with every definition they refer to kept once.
    An id is derived from the definition's content, so two definitions
    with one id are the same definition, and either can stand for both.
    """
    graphics = graphics_of(component)
    if graphics is not None:
        # Drawn: its graphics are its own.
        return cast(_Graphics, graphics)
    # Made of others: put its children's graphics together.
    group = svgwrite.container.Group()
    definitions: dict[str, Any] = {}
    for placement in child_placements_of(component):
        child = _assemble(placement.component)
        dx, dy, scale = transform_of(placement)
        placed = svgwrite.container.Group(
            transform=f"translate({dx},{dy}) scale({scale},{scale})"
        )
        placed.add(child.group)
        group.add(placed)
        for element in child.definitions:
            definitions.setdefault(element.attribs["id"], element)
    return _Graphics(group=group, definitions=tuple(definitions.values()))


def _document(
    component: Component, *, width_px: float, height_px: float
) -> svgwrite.Drawing:
    graphics = _assemble(component)
    drawing = svgwrite.Drawing()
    for element in graphics.definitions:
        drawing.defs.add(element)
    drawing.add(graphics.group)
    bbox = component.bbox
    drawing.viewbox(bbox.xmin, bbox.ymin, bbox.width, bbox.height)
    drawing["width"] = f"{width_px}px"
    drawing["height"] = f"{height_px}px"
    return drawing


def document_string(
    component: Component, *, width_px: float, height_px: float
) -> str:
    """Return a complete SVG document showing ``component``.

    The document is framed on the component's box and sized ``width_px``
    by ``height_px``.
    """
    return str(
        _document(component, width_px=width_px, height_px=height_px).tostring()
    )


def save_document(
    path: str | os.PathLike[str],
    component: Component,
    *,
    width_px: float,
    height_px: float,
) -> None:
    """Write a complete SVG document showing ``component`` to ``path``.

    The document is the one :func:`document_string` returns.
    """
    _document(component, width_px=width_px, height_px=height_px).saveas(path)
