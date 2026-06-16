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

from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
import math

from .structure import Nucleotide, LoopRegion, StemRegion
from .vec2 import Vec2

class EdgeType(Enum):
    """Enumeration of edge types used in the drawing graph."""
    BACKBONE = auto()
    BASE_PAIR = auto()


@dataclass
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


@dataclass
class Edge:
    """Base class representing a connection between two nodes.

    Attributes
    ----------
    start : Node
        Start node of the edge.
    end : Node
        End node of the edge.
    type : EdgeType
        Type of the edge.
    """
    start: Node
    end: Node
    type: EdgeType


@dataclass
class LineEdge(Edge):
    """Edge represented as a straight line segment."""
    pass


@dataclass
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

@dataclass
class Marker():
    node: Node

@dataclass
class ArrowMarker(Marker):
    direction: Vec2
    length: float = 7.0
    is_start: bool = True

@dataclass
class LayoutResult():
    """Container for the generated layout information.

    Attributes
    ----------
    nodes : list[Node]
        Nodes with computed layout positions.
    edges : list[Edge]
        Edges connecting the laid out nodes.
    markers : list[Marker]
        Markers associated with the layout, such as directional annotations.
    """
    nodes: list[Node]
    edges: list[Edge]
    markers: list[Marker]

class LayoutEngine(ABC):
    """Abstract base class for secondary structure layout engines."""

    @abstractmethod
    def layout(self, root_loop: LoopRegion):
        """Compute a layout for the given secondary structure."""
        pass

class RadialLayoutEngine(LayoutEngine):
    """Layout engine for generating a radial representation of a secondary structure.
    
    This layout engine places nucleotides and structural elements using a
    radial geometry based on backbone lengths, base-pair lengths, and
    deflection angles between connected regions.
    
    Parameters
    ----------
    backbone_length : float, default=15
        Length assigned to backbone connections between adjacent nucleotides.
    basepair_length : float, default=20
        Length assigned to base-pair connections in stem regions.
    deflection_angle : float, default=math.pi/18
        Angular deflection applied when traversing connected regions.
    
    Attributes
    ----------
    backbone_length : float
        Length assigned to backbone connections.
    basepair_length : float
        Length assigned to base-pair connections.
    deflection_angle : float
        Angular deflection between connected regions.
    nodes : list[Node]
        Layout nodes generated during layout computation.
    edges : list[Edge]
        Layout edges generated during layout computation.
    current_pos : Vec2
        Current position used during recursive layout generation.
    current_vec : Vec2
        Current unit direction vector used during recursive layout generation.
    """

    def __init__(
        self,
        backbone_length: float = 15,
        basepair_length: float = 20,
        deflection_angle: float = math.pi/18,
    ) -> None:
        self.backbone_length = backbone_length
        self.basepair_length = basepair_length
        self.deflection_angle = deflection_angle
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.markers: list[Marker] = []
        self.current_pos: Vec2 = Vec2(0, 0)
        self.current_vec: Vec2 = Vec2(1, 0)
    
    def add_last_stem_backbone(self):
        if not self.nodes[-2].nucleotide.is_three_prime:
            self.edges.append(LineEdge(self.nodes[-2], self.nodes[-1], EdgeType.BACKBONE))
    
    def add_last_loop_backbone(self, radius: float):
        if not self.nodes[-2].nucleotide.is_three_prime:
            self.edges.append(ArcEdge(self.nodes[-2], self.nodes[-1], EdgeType.BACKBONE, radius, radius, 0, 0, 1))

    def layout_stem(
        self,
        current_stem: StemRegion,
    ) -> None:
        """Generate layout information for a stem region.

        Parameters
        ----------
        current_stem : StemRegion
            Stem region to layout.
        """
        self.current_vec = self.current_vec.normalized()
        stem_length = len(current_stem.nucleotides)//2
        start_node = self.nodes[-1]
        for start_idx in [0, stem_length]:
            nucleotides = current_stem.nucleotides[start_idx+1:start_idx+stem_length]
            for nt in nucleotides:
                # Generate nodes
                self.current_pos += self.current_vec * self.backbone_length
                self.nodes.append(Node(nt, self.current_pos))
                # Generate backbones
                self.add_last_stem_backbone()
            # Generate markers for 3' termini
            if nucleotides and nucleotides[-1].is_three_prime:
                self.markers.append(ArrowMarker(self.nodes[-1], self.current_vec))
            # Layout child loop region
            if start_idx == 0:
                child_loop = current_stem.child_loop
                if not child_loop.is_hinge:
                    self.current_vec = self.current_vec.rotated(-math.pi/2)
                self.layout_loop(child_loop)
                if not child_loop.is_hinge:
                    self.current_vec = self.current_vec.rotated(-math.pi/2)
        # Generate base pairs
        base_idx = self.nodes.index(start_node)
        for idx in range(stem_length):
            self.edges.append(LineEdge(self.nodes[base_idx+idx], self.nodes[-(idx+1)], EdgeType.BASE_PAIR))
        self.current_vec = self.current_vec.normalized()
        return None

    def layout_loop(
        self,
        current_loop: LoopRegion,
    ) -> None:
        """Generate layout information for a loop region.

        Parameters
        ----------
        current_loop : LoopRegion
            Loop region to layout.
        """
        self.current_vec = self.current_vec.normalized()
        nucleotides = current_loop.nucleotides
        child_stems = current_loop.child_stems
        if (current_loop.is_root
                and child_stems
                and child_stems[0].nucleotides[0] is nucleotides[0]):
            self.current_vec = self.current_vec.rotated(-math.pi/2)
            self.layout_stem(child_stems[0])
            if not current_loop.is_hinge:
                self.current_vec = self.current_vec.rotated(-math.pi/2)
            nucleotides = nucleotides[1:]
            child_stems = child_stems[1:]
        if current_loop.is_hinge:
            defl_angle = (
                self.deflection_angle 
                if nucleotides[0].is_three_prime
                else -self.deflection_angle
            )
            intermediate_vec = self.current_vec.rotated(defl_angle/2)
            delta = self.basepair_length * math.sin(defl_angle/2)
            # Layout the second nucleotide in this loop region
            self.current_pos += intermediate_vec * (self.backbone_length + delta)
            self.current_vec = self.current_vec.rotated(defl_angle)
            self.nodes.append(Node(nucleotides[1], self.current_pos))
            self.add_last_stem_backbone()
            # Layout child stem region
            self.layout_stem(child_stems[0])
            # Layout the 4th nucleotide in this loop region
            if not current_loop.is_root:
                self.current_pos -= intermediate_vec * (self.backbone_length - delta)
                self.current_vec = self.current_vec.rotated(-defl_angle)
                self.nodes.append(Node(nucleotides[3], self.current_pos))
                self.add_last_stem_backbone()
        else:
            delta_angle = 2*math.pi / len(current_loop.nucleotides)
            radius = self.basepair_length/2 / math.sin(delta_angle/2)
            self.current_vec = self.current_vec.rotated(delta_angle)
            stem_map = {stem.nucleotides[0]: stem for stem in child_stems}
            # Layout nucleotides except the first and stem merge nucleotides
            for nt in [curr for prev, curr in zip(nucleotides, nucleotides[1:]) if prev not in stem_map]:
                self.current_pos += self.current_vec * self.basepair_length
                self.current_vec = self.current_vec.rotated(delta_angle)
                self.nodes.append(Node(nt, self.current_pos))
                self.add_last_loop_backbone(radius)
                if nt.is_three_prime:
                    direction = self.current_vec.rotated(-delta_angle/2)
                    self.markers.append(ArrowMarker(self.nodes[-1], direction=direction))
                if (stem := stem_map.pop(nt, None)) is not None:
                    self.current_vec = self.current_vec.rotated(-math.pi/2)
                    self.layout_stem(stem)
                    self.current_vec = self.current_vec.rotated(-math.pi/2+delta_angle)
        self.current_vec = self.current_vec.normalized()
        return None

    def layout(self, root_loop: LoopRegion) -> None:
        """Generate a complete layout starting from the root loop region.
        
        Parameters
        ----------
        root_loop : LoopRegion
            Root loop region of the secondary structure tree.
        """
        self.current_pos = Vec2(0, 0)
        self.current_vec = Vec2(1, 0)
        nucleotides = root_loop.nucleotides
        delta_angle = 2*math.pi / len(nucleotides)
        if root_loop.child_stems:
            # Adjust the layout so that the first stem region extends upward
            offset = nucleotides.index(root_loop.child_stems[0].nucleotides[0])
            for _ in range(offset):
                self.current_vec = self.current_vec.rotated(-delta_angle)
                self.current_pos -= self.backbone_length * self.current_vec
            if offset != 0:
                self.current_vec = self.current_vec.rotated(-delta_angle)
            self.nodes.append(Node(nucleotides[0], self.current_pos))
            self.layout_loop(root_loop)
        else:
            # Layout for secondary structures without base pairs
            self.nodes.append(Node(nucleotides[0], self.current_pos))
            for nt in nucleotides[1:]:
                self.current_pos += self.backbone_length * self.current_vec
                self.nodes.append(Node(nt, self.current_pos))
                self.edges.append(LineEdge(self.nodes[-2], self.nodes[-1], EdgeType.BACKBONE))
            self.markers.append(ArrowMarker(self.nodes[-1], self.current_vec))
        return LayoutResult(self.nodes, self.edges, self.markers)

def layout(root_loop):
    return RadialLayoutEngine().layout(root_loop)
