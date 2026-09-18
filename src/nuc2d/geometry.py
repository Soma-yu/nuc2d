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
        Lower corner of the box.
    xmax, ymax : float
        Upper corner of the box.

    Notes
    -----
    The box is stored as its two corners rather than as an origin and a
    size, because corners are what :meth:`union` combines; ``width`` and
    ``height`` are derived. A box whose lower corner exceeds its upper
    corner is empty, which gives :meth:`union` an identity element and
    lets a box be built up from nothing without special cases.

    A box is built from its corners and moved by two numbers, so nothing
    here depends on :class:`Vec2`. Enclosing several things is a
    :func:`~functools.reduce` of :meth:`union` over the boxes they occupy.
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

    def __or__(self, other: BBox) -> BBox:
        """Return ``self.union(other)``."""
        return self.union(other)

    def expanded(self, dx: float, dy: float | None = None) -> BBox:
        """Return the box grown outwards on every side.

        Parameters
        ----------
        dx : float
            Amount to grow by along the x-axis, on each side.
        dy : float, optional
            Amount to grow by along the y-axis, on each side. Defaults to
            ``dx``.

        Returns
        -------
        BBox
            The grown box. Negative amounts shrink it, and shrinking a box
            past itself yields an empty box.
        """
        if self.is_empty:
            return self
        dy = dx if dy is None else dy
        return BBox(self.xmin - dx, self.ymin - dy, self.xmax + dx, self.ymax + dy)

    def translated(self, dx: float, dy: float) -> BBox:
        """Return the box moved by the given offset.

        Parameters
        ----------
        dx : float
            Distance to move along the x-axis.
        dy : float
            Distance to move along the y-axis.

        Returns
        -------
        BBox
            The translated box.
        """
        if self.is_empty:
            return self
        return BBox(
            self.xmin + dx,
            self.ymin + dy,
            self.xmax + dx,
            self.ymax + dy,
        )

    def scaled(self, scale: float) -> BBox:
        """Return the box scaled about the origin.

        Parameters
        ----------
        scale : float
            Uniform scaling factor. This scales about the coordinate
            origin, not about the centre of the box, to match the order of
            an SVG ``translate`` followed by ``scale``.

        Returns
        -------
        BBox
            The scaled box.
        """
        if self.is_empty:
            return self
        xs = (self.xmin * scale, self.xmax * scale)
        ys = (self.ymin * scale, self.ymax * scale)
        return BBox(min(xs), min(ys), max(xs), max(ys))

    def to_viewbox(self) -> tuple[float, float, float, float]:
        """Return the box as an SVG ``viewBox`` tuple.

        Returns
        -------
        tuple of float
            ``(min-x, min-y, width, height)``.
        """
        return (self.xmin, self.ymin, self.width, self.height)
