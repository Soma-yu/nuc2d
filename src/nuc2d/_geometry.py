"""Geometric value types shared by layout and rendering.

This module holds the small immutable types that the rest of the package
computes with: :class:`Vec2` for points and directions, and :class:`BBox`
for axis-aligned extents. They carry no knowledge of secondary structures
or of SVG, so every other module is free to depend on this one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Vec2:
    """A 2D vector.

    Parameters
    ----------
    x : float
        The x component.
    y : float
        The y component.
    """

    x: float
    y: float

    def __neg__(self) -> Vec2:
        """Return the vector with both components negated."""
        return Vec2(-self.x, -self.y)

    def __add__(self, other: Vec2) -> Vec2:
        """Add another vector."""
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        """Subtract another vector."""
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        """Multiply the vector by a scalar (vector * scalar)."""
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vec2:
        """Multiply the vector by a scalar (scalar * vector)."""
        return self * scalar

    def norm(self) -> float:
        """Return the Euclidean norm of the vector."""
        return math.hypot(self.x, self.y)

    def normalized(self) -> Vec2:
        """Return a unit vector in the same direction.

        Returns
        -------
        Vec2
            The normalized vector.

        Raises
        ------
        ValueError
            If the vector has zero length.
        """
        n = self.norm()
        if n == 0.0:
            raise ValueError("Cannot normalize a zero-length vector.")
        return Vec2(self.x / n, self.y / n)

    def dot(self, other: Vec2) -> float:
        """Compute the dot product with another vector."""
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: Vec2) -> float:
        """Return the Euclidean distance to another vector.

        Parameters
        ----------
        other : Vec2
            The other vector.

        Returns
        -------
        float
            The distance between the two vectors.
        """
        return (self - other).norm()

    def rotated(self, theta: float) -> Vec2:
        """Return the vector rotated by ``theta`` radians.

        The rotation is counterclockwise in the usual mathematical sense,
        from the x-axis towards the y-axis. The y-axis points down, as in
        SVG, so the rotation looks clockwise when drawn.

        Parameters
        ----------
        theta : float
            The rotation angle in radians.

        Returns
        -------
        Vec2
            The rotated vector.
        """
        c = math.cos(theta)
        s = math.sin(theta)
        return Vec2(
            c * self.x - s * self.y,
            s * self.x + c * self.y,
        )

    def to_tuple(self) -> tuple[float, float]:
        """Return the vector as a tuple."""
        return (float(self.x), float(self.y))


@dataclass(frozen=True)
class BBox:
    """An axis-aligned bounding box.

    Parameters
    ----------
    xmin, ymin : float
        Corner with the smallest coordinates. The y-axis points down, as
        in SVG, so this is the upper left corner.
    xmax, ymax : float
        Corner with the largest coordinates, the lower right.

    Attributes
    ----------
    width, height : float
        Extent of the box along each axis.

    Notes
    -----
    The box of a component with nothing in it is empty, and its width and
    height are zero. A box around a single point is not empty, although
    its width and height are zero too.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        """Extent of the box along the x-axis, or 0 when it is empty."""
        return 0.0 if is_empty_bbox(self) else self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Extent of the box along the y-axis, or 0 when it is empty."""
        return 0.0 if is_empty_bbox(self) else self.ymax - self.ymin


# What the package does with boxes, as against what a caller reads from
# one. These are functions of this module rather than methods of BBox, so
# that BBox offers a caller its corners and its size and nothing more.

# The box enclosing nothing. Its minimum exceeds its maximum along both
# axes, so it is where bbox_around starts from.
_EMPTY_BBOX = BBox(math.inf, math.inf, -math.inf, -math.inf)


def is_empty_bbox(bbox: BBox) -> bool:
    """Return whether ``bbox`` encloses nothing."""
    return bbox.xmin > bbox.xmax or bbox.ymin > bbox.ymax


def bbox_around(boxes: Iterable[BBox]) -> BBox:
    """Return the smallest box enclosing every box in ``boxes``.

    An empty box encloses nothing and is passed over, so no boxes at all,
    or empty ones only, give an empty box.
    """
    around = _EMPTY_BBOX
    for bbox in boxes:
        if is_empty_bbox(bbox):
            continue
        around = BBox(
            min(around.xmin, bbox.xmin),
            min(around.ymin, bbox.ymin),
            max(around.xmax, bbox.xmax),
            max(around.ymax, bbox.ymax),
        )
    return around
