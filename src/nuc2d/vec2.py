"""
Utilities for 2D vector operations.

This module provides:
- Vec2: A class representing a 2D vector.
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
