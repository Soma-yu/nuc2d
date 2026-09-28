"""Geometric value types shared by layout and rendering.

This module holds the small immutable types that the rest of the package
computes with: :class:`Vec2` for points and directions, and :class:`BBox`
for axis-aligned extents. They carry no knowledge of secondary structures
or of SVG, so every other module is free to depend on this one.
"""

from __future__ import annotations

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
        """Return the vector rotated counterclockwise.

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

    Notes
    -----
    A box is stored as its two corners, and ``width``, ``height``,
    ``center_x`` and ``center_y`` are derived from them. A box whose
    minimum exceeds its maximum along either axis is empty: :meth:`empty`
    returns one, its width and height are zero, its centre raises, and it
    is the identity element of :meth:`union`. A box around a single point
    is not empty, although its width and height are zero.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def empty(cls) -> BBox:
        """Return the empty box, the identity element of :meth:`union`."""
        return cls(math.inf, math.inf, -math.inf, -math.inf)

    @property
    def is_empty(self) -> bool:
        """Return whether the box encloses nothing."""
        return self.xmin > self.xmax or self.ymin > self.ymax

    @property
    def width(self) -> float:
        """Extent of the box along the x-axis, or 0 when it is empty."""
        return 0.0 if self.is_empty else self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Extent of the box along the y-axis, or 0 when it is empty."""
        return 0.0 if self.is_empty else self.ymax - self.ymin

    @property
    def center_x(self) -> float:
        """Midpoint of the box along the x-axis.

        Raises
        ------
        ValueError
            If the box is empty. ``width`` and ``height`` are zero for such
            a box, that being the extent of nothing, but a midpoint has no
            answer of the same kind.
        """
        if self.is_empty:
            raise ValueError("An empty box has no centre.")
        return (self.xmin + self.xmax) / 2

    @property
    def center_y(self) -> float:
        """Midpoint of the box along the y-axis.

        Raises
        ------
        ValueError
            If the box is empty. See :attr:`center_x`.
        """
        if self.is_empty:
            raise ValueError("An empty box has no centre.")
        return (self.ymin + self.ymax) / 2

    def union(self, other: BBox) -> BBox:
        """Return the smallest box containing both boxes.

        Parameters
        ----------
        other : BBox
            Box to combine with this one.

        Returns
        -------
        BBox
            The combined box. An empty operand is ignored.
        """
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        return BBox(
            min(self.xmin, other.xmin),
            min(self.ymin, other.ymin),
            max(self.xmax, other.xmax),
            max(self.ymax, other.ymax),
        )
