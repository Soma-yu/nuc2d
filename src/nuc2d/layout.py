"""Geometry layout generation for secondary structure visualization.

This module converts secondary structure representations into drawable
geometric layouts. The generated layouts define spatial relationships
between nucleotides, stems, loops, and their connections, independently
from rendering.

The layout result typically consists of layout nodes and edges annotated
with geometric information such as positions, orientations, and edge
shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import math

from .structure import Nucleotide, LoopRegion, StemRegion
from .geometry import Vec2


class EdgeType(Enum):
    """Enumeration of edge types used in the drawing graph."""
    BACKBONE = auto()
    BASE_PAIR = auto()


@dataclass(kw_only=True)
class Node:
    """Node representing a nucleotide and its drawing position.

    Attributes
    ----------
    nucleotide : Nucleotide
        Nucleotide associated with this node.
    pos : Vec2
        Position of the node in the drawing coordinate system.
    """
    nucleotide: Nucleotide
    pos: Vec2


@dataclass(kw_only=True)
class Edge:
    """Base class representing a connection between two nodes.

    Attributes
    ----------
    start : Node
        Start node of the edge.
    end : Node
        End node of the edge.
    edge_type : EdgeType
        Type of the edge.
    """
    start: Node
    end: Node
    edge_type: EdgeType


@dataclass(kw_only=True)
class LineEdge(Edge):
    """Edge represented as a straight line segment."""
    pass


@dataclass(kw_only=True)
class ArcEdge(Edge):
    """Edge represented as an SVG elliptical arc.

    Attributes
    ----------
    rx : float
        Radius of the ellipse along the x-axis.
    ry : float
        Radius of the ellipse along the y-axis.
    x_axis_rotation : float
        Rotation angle of the ellipse x-axis in degrees.
    large_arc : bool
        Whether to use the larger arc between the endpoints.
    sweep : bool
        Direction of the arc sweep.
    """
    rx: float
    ry: float
    x_axis_rotation: float
    large_arc: bool
    sweep: bool

@dataclass(kw_only=True)
class Decoration():
    """Base class for something drawn at a single node.

    Attributes
    ----------
    node : Node
        Node the decoration is attached to.
    """
    node: Node

@dataclass(kw_only=True)
class ArrowDecoration(Decoration):
    """Arrow drawn alongside a node to indicate strand direction.

    Attributes
    ----------
    direction : Vec2
        Unit vector the arrow points along.
    length : float, default=7.0
        Length of the arrow segment.
    node_at_start : bool, default=True
        Whether the node sits at the start of the arrow segment, so that
        the arrow extends away from it. When False, the segment ends at
        the node and the arrow points into it.
    """
    direction: Vec2
    length: float = 7.0
    node_at_start: bool = True

@dataclass(kw_only=True)
class LayoutResult():
    """Container for the generated layout information.

    Attributes
    ----------
    nodes : list[Node]
        Nodes with computed layout positions.
    edges : list[Edge]
        Edges connecting the laid out nodes.
    decorations : list[Decoration]
        Anything drawn at a node rather than between nodes, such as the
        arrow at a 3' terminus.
    """
    nodes: list[Node]
    edges: list[Edge]
    decorations: list[Decoration]


@dataclass(kw_only=True)
class _LayoutState:
    """Mutable state belonging to a single layout run.

    Attributes
    ----------
    nodes : list[Node]
        Nodes generated so far.
    edges : list[Edge]
        Edges generated so far.
    decorations : list[Decoration]
        Decorations generated so far.
    pos : Vec2
        Current position of the layout walk.
    vec : Vec2
        Current unit direction vector of the layout walk.

    Notes
    -----
    This is deliberately separate from the layout engine. The engine holds
    configuration, which is the same for every run; this holds the work in
    progress, which is not. Keeping the two apart makes an engine instance
    reusable, and keeps a returned :class:`LayoutResult` from aliasing state
    that a later run would overwrite.
    """
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    decorations: list[Decoration] = field(default_factory=list)
    pos: Vec2 = Vec2(0, 0)
    vec: Vec2 = Vec2(1, 0)


class LayoutEngine(ABC):
    """Abstract base class for secondary structure layout engines."""

    @abstractmethod
    def layout(self, root_loop: LoopRegion) -> LayoutResult:
        """Compute a layout for the given secondary structure."""
        pass

class RadialLayoutEngine(LayoutEngine):
    """Layout engine for generating a radial representation of a secondary structure.

    This layout engine places nucleotides and structural elements using a
    radial geometry, built from the spacing along the backbone, the spacing
    around a loop, and the angle by which stacked stems are deflected.

    Parameters
    ----------
    backbone_spacing : float, default=15
        Distance between adjacent nucleotides wherever the backbone runs
        straight: inside a stem, along an unpaired strand, and through a
        loop whose stems stack coaxially on one another.
    loop_spacing : float, default=20
        Distance between adjacent nucleotides around a loop that is not
        stacked, measured as the chord of the loop circle. This is the same
        backbone curved rather than straight, and it is what sets the
        radius the loop is drawn on.

        It sets the width of every stem as well: the base pair closing a
        loop joins two nucleotides that are neighbours on that loop's
        circle, so the pair spans one chord, and the two strands of the
        stem then run parallel at that separation.
    stack_deflection : float, default=math.pi/18
        Angle in radians by which the backbone is deflected where two stems
        stack directly on one another.

    Notes
    -----
    Every parameter is configuration that :meth:`layout` does not modify,
    so one engine can lay out any number of structures.
    """

    def __init__(
        self,
        *,
        backbone_spacing: float = 15,
        loop_spacing: float = 20,
        stack_deflection: float = math.pi/18,
    ) -> None:
        self.backbone_spacing = backbone_spacing
        self.loop_spacing = loop_spacing
        self.stack_deflection = stack_deflection

    def _add_backbone_line(self, state: _LayoutState) -> None:
        """Join the last two nodes with a straight backbone edge.

        Nothing is added when the earlier node is a 3' terminus, because the
        two nodes then belong to different strands.
        """
        if not state.nodes[-2].nucleotide.is_three_prime:
            state.edges.append(
                LineEdge(
                    start=state.nodes[-2],
                    end=state.nodes[-1],
                    edge_type=EdgeType.BACKBONE,
                )
            )

    def _add_backbone_arc(self, state: _LayoutState, radius: float) -> None:
        """Join the last two nodes with an arched backbone edge.

        Nothing is added when the earlier node is a 3' terminus, because the
        two nodes then belong to different strands.
        """
        if not state.nodes[-2].nucleotide.is_three_prime:
            state.edges.append(
                ArcEdge(
                    start=state.nodes[-2],
                    end=state.nodes[-1],
                    edge_type=EdgeType.BACKBONE,
                    rx=radius,
                    ry=radius,
                    x_axis_rotation=0.0,
                    large_arc=False,
                    sweep=True,
                )
            )

    def _layout_stem(
        self,
        state: _LayoutState,
        current_stem: StemRegion,
    ) -> None:
        """Generate layout information for a stem region.

        Parameters
        ----------
        state : _LayoutState
            State of the layout run in progress.
        current_stem : StemRegion
            Stem region to layout.
        """
        state.vec = state.vec.normalized()
        stem_length = len(current_stem.nucleotides)//2
        # Index of the node the stem starts from. Nodes are only ever
        # appended, so this stays valid while the stem and everything nested
        # inside it is laid out.
        base_idx = len(state.nodes) - 1
        for start_idx in [0, stem_length]:
            nucleotides = current_stem.nucleotides[start_idx+1:start_idx+stem_length]
            for nt in nucleotides:
                # Generate nodes
                state.pos += state.vec * self.backbone_spacing
                state.nodes.append(Node(nucleotide=nt, pos=state.pos))
                # Generate backbones
                self._add_backbone_line(state)
            # Generate decorations for 3' termini
            if nucleotides and nucleotides[-1].is_three_prime:
                state.decorations.append(ArrowDecoration(node=state.nodes[-1], direction=state.vec))
            # Layout child loop region
            if start_idx == 0:
                child_loop = current_stem.child_loop
                if not child_loop.is_stacked:
                    state.vec = state.vec.rotated(-math.pi/2)
                self._layout_loop(state, child_loop)
                if not child_loop.is_stacked:
                    state.vec = state.vec.rotated(-math.pi/2)
        # Generate base pairs
        for idx in range(stem_length):
            state.edges.append(
                LineEdge(
                    start=state.nodes[base_idx+idx],
                    end=state.nodes[-(idx+1)],
                    edge_type=EdgeType.BASE_PAIR,
                )
            )
        state.vec = state.vec.normalized()
        return None

    def _layout_loop(
        self,
        state: _LayoutState,
        current_loop: LoopRegion,
    ) -> None:
        """Generate layout information for a loop region.

        Parameters
        ----------
        state : _LayoutState
            State of the layout run in progress.
        current_loop : LoopRegion
            Loop region to layout.
        """
        state.vec = state.vec.normalized()
        nucleotides = current_loop.nucleotides
        child_stems = current_loop.child_stems
        if (current_loop.is_root
                and child_stems
                and child_stems[0].nucleotides[0] is nucleotides[0]):
            state.vec = state.vec.rotated(-math.pi/2)
            self._layout_stem(state, child_stems[0])
            if not current_loop.is_stacked:
                state.vec = state.vec.rotated(-math.pi/2)
            nucleotides = nucleotides[1:]
            child_stems = child_stems[1:]
        if current_loop.is_stacked:
            defl_angle = (
                self.stack_deflection
                if nucleotides[0].is_three_prime
                else -self.stack_deflection
            )
            intermediate_vec = state.vec.rotated(defl_angle/2)
            delta = self.loop_spacing * math.sin(defl_angle/2)
            # Layout the second nucleotide in this loop region
            state.pos += intermediate_vec * (self.backbone_spacing + delta)
            state.vec = state.vec.rotated(defl_angle)
            state.nodes.append(Node(nucleotide=nucleotides[1], pos=state.pos))
            self._add_backbone_line(state)
            # Layout child stem region
            self._layout_stem(state, child_stems[0])
            # Layout the 4th nucleotide in this loop region
            if not current_loop.is_root:
                state.pos -= intermediate_vec * (self.backbone_spacing - delta)
                state.vec = state.vec.rotated(-defl_angle)
                state.nodes.append(Node(nucleotide=nucleotides[3], pos=state.pos))
                self._add_backbone_line(state)
        else:
            delta_angle = 2*math.pi / len(current_loop.nucleotides)
            radius = self.loop_spacing/2 / math.sin(delta_angle/2)
            state.vec = state.vec.rotated(delta_angle)
            stem_map = {stem.nucleotides[0]: stem for stem in child_stems}
            # Layout nucleotides except the first and stem merge nucleotides
            for nt in [curr for prev, curr in zip(nucleotides, nucleotides[1:]) if prev not in stem_map]:
                state.pos += state.vec * self.loop_spacing
                state.vec = state.vec.rotated(delta_angle)
                state.nodes.append(Node(nucleotide=nt, pos=state.pos))
                self._add_backbone_arc(state, radius)
                if nt.is_three_prime:
                    direction = state.vec.rotated(-delta_angle/2)
                    state.decorations.append(ArrowDecoration(node=state.nodes[-1], direction=direction))
                if (stem := stem_map.pop(nt, None)) is not None:
                    state.vec = state.vec.rotated(-math.pi/2)
                    self._layout_stem(state, stem)
                    state.vec = state.vec.rotated(-math.pi/2+delta_angle)
        state.vec = state.vec.normalized()
        return None

    def layout(self, root_loop: LoopRegion) -> LayoutResult:
        """Generate a complete layout starting from the root loop region.

        Parameters
        ----------
        root_loop : LoopRegion
            Root loop region of the secondary structure tree.

        Returns
        -------
        LayoutResult
            Nodes, edges and decorations describing the geometry of the
            structure. The result owns its lists; a later call to this
            method does not modify it.
        """
        state = _LayoutState()
        nucleotides = root_loop.nucleotides
        delta_angle = 2*math.pi / len(nucleotides)
        if root_loop.child_stems:
            # Turn the starting direction so that the first stem region
            # extends upward. Only the direction matters: the walk's starting
            # position just translates the whole drawing, which the bounding
            # box absorbs.
            offset = nucleotides.index(root_loop.child_stems[0].nucleotides[0])
            for _ in range(offset):
                state.vec = state.vec.rotated(-delta_angle)
            if offset != 0:
                state.vec = state.vec.rotated(-delta_angle)
            state.nodes.append(Node(nucleotide=nucleotides[0], pos=state.pos))
            self._layout_loop(state, root_loop)
        else:
            # Layout for secondary structures without base pairs
            state.nodes.append(Node(nucleotide=nucleotides[0], pos=state.pos))
            for nt in nucleotides[1:]:
                state.pos += self.backbone_spacing * state.vec
                state.nodes.append(Node(nucleotide=nt, pos=state.pos))
                state.edges.append(
                    LineEdge(
                        start=state.nodes[-2],
                        end=state.nodes[-1],
                        edge_type=EdgeType.BACKBONE,
                    )
                )
            state.decorations.append(ArrowDecoration(node=state.nodes[-1], direction=state.vec))
        return LayoutResult(
            nodes=state.nodes,
            edges=state.edges,
            decorations=state.decorations,
        )
